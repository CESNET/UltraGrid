/**
 * @file   video_compress/j2k.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2018 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "utils/misc.h"
#include "utils/video_frame_pool.h"
#include "video_compress.h"
#include "video.h"

#include <cmpto_j2k_enc.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "[J2K] %s: %s\n", \
                                err_msg, cmpto_j2k_enc_get_last_error()); \
                action_fail;\
        } \
} while(0)

#define NOOP ((void) 0)
#define DEFAULT_POOL_SIZE 4
#define DEFAULT_TILE_LIMIT 1
#define DEFAULT_MEM_LIMIT 1000000000llu

using namespace std;

struct state_video_compress_j2k {
        state_video_compress_j2k(long long int bitrate, unsigned int pool_size)
                : rate{bitrate}, pool{pool_size}, max_in_frames{pool_size} {}
        struct module module_data{};

        struct cmpto_j2k_enc_ctx *context{};
        struct cmpto_j2k_enc_cfg *enc_settings{};
        long long int rate;
        video_frame_pool<default_data_allocator> pool;
        unsigned int max_in_frames;
        unsigned int in_frames{};
        mutex lock;
        condition_variable frame_popped;
        video_desc saved_desc{};
};

static void j2k_compressed_frame_dispose(struct video_frame *frame);
static void j2k_compress_done(struct module *mod);

#define HANDLE_ERROR_COMPRESS_POP do { cmpto_j2k_enc_img_destroy(img); goto start; } while (0)
static std::shared_ptr<video_frame> j2k_compress_pop(struct module *state)
{
start:
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) state;

        struct cmpto_j2k_enc_img *img;
        int status;
        CHECK_OK(cmpto_j2k_enc_ctx_get_encoded_img(
                                s->context,
                                1,
                                &img /* Set to NULL if encoder stopped */,
                                &status), "Encode image", HANDLE_ERROR_COMPRESS_POP);
        {
                unique_lock<mutex> lk(s->lock);
                s->in_frames--;
                s->frame_popped.notify_one();
        }
        if (status != CMPTO_J2K_ENC_IMG_OK) {
                const char * encoding_error = "";
                CHECK_OK(cmpto_j2k_enc_img_get_error(img, &encoding_error), "get error status",
                                encoding_error = "(failed)");
                log_msg(LOG_LEVEL_ERROR, "Image encoding failed: %s\n", encoding_error);
                goto start;
        }

        if (!img) {
                // pass poison pill
                return {};
        }
        struct video_desc *desc;
        size_t len;
        CHECK_OK(cmpto_j2k_enc_img_get_custom_data(img, (void **) &desc, &len),
                        "get custom data", HANDLE_ERROR_COMPRESS_POP);
        size_t size;
        void * ptr;
        CHECK_OK(cmpto_j2k_enc_img_get_cstream(img, &ptr, &size),
                        "get cstream", HANDLE_ERROR_COMPRESS_POP);

        struct video_frame *out = vf_alloc_desc(*desc);
        out->tiles[0].data_len = size;
        out->tiles[0].data = (char *) malloc(size);
        memcpy(out->tiles[0].data, ptr, size);
        out->color_spec = codec_is_a_rgb(desc->color_spec) ? J2KR : J2K;
        CHECK_OK(cmpto_j2k_enc_img_destroy(img), "Destroy image", NOOP);
        out->dispose = j2k_compressed_frame_dispose;
        return shared_ptr<video_frame>(out, out->dispose);
}

static void usage() {
        printf("J2K compress usage:\n");
        printf("\t-c j2k[:rate=<bitrate>][:quality=<q>][:mcu][:mem_limit=<m>][:tile_limit=<t>][:pool_size=<p>] [--cuda-device <c_index>]\n");
        printf("\twhere:\n");
        printf("\t\t<bitrate> - target bitrate\n");
        printf("\t\t<q> - quality\n");
        printf("\t\t<m> - CUDA device memory limit (in bytes), default %llu\n", DEFAULT_MEM_LIMIT);
        printf("\t\t<t> - number of tiles encoded at moment (less to reduce latency, more to increase performance, 0 means infinity), default %d\n", DEFAULT_TILE_LIMIT);
        printf("\t\t<p> - total number of tiles encoder can hold at moment (same meaning as above), default %d, should be greater than <t>\n", DEFAULT_POOL_SIZE);
        printf("\t\tmcu - use MCU\n");
        printf("\t\t<c_index> - CUDA device(s) to use (comma separated)\n");
}

static struct module * j2k_compress_init(struct module *parent, const char *c_cfg)
{
        struct state_video_compress_j2k *s;
        double quality = 0.7;
        bool mct = false;
        long long int bitrate = 0;
        long long int mem_limit = DEFAULT_MEM_LIMIT;
        unsigned int tile_limit = DEFAULT_TILE_LIMIT;
        unsigned int pool_size = DEFAULT_POOL_SIZE;

        char *tmp = (char *) alloca(strlen(c_cfg) + 1);
        strcpy(tmp, c_cfg);
        char *save_ptr, *item;
        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                tmp = NULL;
                if (strncasecmp("rate=", item, strlen("rate=")) == 0) {
                        bitrate = unit_evaluate(item + strlen("rate="));
                        if (bitrate <= 0) {
                                log_msg(LOG_LEVEL_ERROR, "[J2K] Wrong bitrate!\n");
                                return NULL;
                        }
                } else if (strncasecmp("quality=", item, strlen("quality=")) == 0) {
                        quality = atof(item + strlen("quality="));
                } else if (strcasecmp("mct", item) == 0) {
                        mct = true;
                } else if (strncasecmp("mem_limit=", item, strlen("mem_limit=")) == 0) {
                        mem_limit = unit_evaluate(item + strlen("mem_limit="));
                } else if (strncasecmp("tile_limit=", item, strlen("tile_limit=")) == 0) {
                        tile_limit = atoi(item + strlen("tile_limit="));
                } else if (strncasecmp("pool_size=", item, strlen("pool_size=")) == 0) {
                        pool_size = atoi(item + strlen("pool_size="));
                } else if (strcasecmp("help", item) == 0) {
                        usage();
                        return &compress_init_noerr;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[J2K] Wrong option: %s\n", item);
                        return NULL;
                }

        }

        s = new state_video_compress_j2k(bitrate, pool_size);

        struct cmpto_j2k_enc_ctx_cfg *ctx_cfg;
        CHECK_OK(cmpto_j2k_enc_ctx_cfg_create(&ctx_cfg), "Context configuration create",
                        goto error);
        for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                CHECK_OK(cmpto_j2k_enc_ctx_cfg_add_cuda_device(ctx_cfg, cuda_devices[i], mem_limit, tile_limit),
                                "Setting CUDA device", goto error);
        }

        CHECK_OK(cmpto_j2k_enc_ctx_create(ctx_cfg, &s->context), "Context create",
                        goto error);
        CHECK_OK(cmpto_j2k_enc_ctx_cfg_destroy(ctx_cfg), "Context configuration destroy",
                        NOOP);

        CHECK_OK(cmpto_j2k_enc_cfg_create(
                                s->context,
                                &s->enc_settings),
                        "Creating context configuration:",
                        goto error);
        CHECK_OK(cmpto_j2k_enc_cfg_set_quantization(
                                s->enc_settings,
                                quality /* 0.0 = poor quality, 1.0 = full quality */
                                ),
                        "Setting quantization",
                        NOOP);

        if (mct) {
                CHECK_OK(cmpto_j2k_enc_cfg_set_mct(s->enc_settings, 1), // only for RGB
                                "Setting MCT",
                                NOOP);
        }

        CHECK_OK(cmpto_j2k_enc_cfg_set_resolutions( s->enc_settings, 6),
                        "Setting DWT levels",
                        NOOP);

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = j2k_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;

error:
        if (s) {
                delete s;
        }
        return NULL;
}

static void j2k_compressed_frame_dispose(struct video_frame *frame)
{
        free(frame->tiles[0].data);
        vf_free(frame);
}

static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size)
{
        (void) codestream; (void) custom_data_size; (void) codestream_size;
        ((shared_ptr<video_frame> *) ((char *) custom_data + sizeof(struct video_desc)))->~shared_ptr<video_frame>();
}

#define HANDLE_ERROR_COMPRESS_PUSH if (img) cmpto_j2k_enc_img_destroy(img); return
static void j2k_compress_push(struct module *state, std::shared_ptr<video_frame> tx)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) state;
        struct cmpto_j2k_enc_img *img = NULL;
        struct video_desc desc;
        void *udata;
        shared_ptr<video_frame> *ref;

        if (tx == NULL) {
                CHECK_OK(cmpto_j2k_enc_ctx_stop(s->context), "stop", NOOP);
                return;
        }

        desc = video_desc_from_frame(tx.get());
        if (!video_desc_eq(s->saved_desc, desc)) {
                s->pool.reconfigure(desc, vc_get_linesize(desc.width, desc.color_spec)
                                * desc.height);
                s->saved_desc = desc;
        }

        assert(tx->tile_count == 1); // TODO

        enum cmpto_sample_format_type cmpto_sf;
        switch (tx->color_spec) {
                case UYVY:
                        cmpto_sf = CMPTO_422_U8_P1020;
                        break;
                case v210:
                        cmpto_sf = CMPTO_422_U10_V210;
                        break;
                case RGB:
                        cmpto_sf = CMPTO_444_U8_P012;
                        break;
                case R10k:
                        cmpto_sf = CMPTO_444_U10U10U10_MSB32BE_P210;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "[J2K] Unsupported codec!\n");
                        abort();
        }
        CHECK_OK(cmpto_j2k_enc_cfg_set_samples_format_type(s->enc_settings, cmpto_sf),
                        "Setting sample format", return);
        CHECK_OK(cmpto_j2k_enc_cfg_set_size(s->enc_settings, tx->tiles[0].width, tx->tiles[0].height),
                        "Setting image size", return);
        if (s->rate) {
                CHECK_OK(cmpto_j2k_enc_cfg_set_rate_limit(s->enc_settings,
                                        CMPTO_J2K_ENC_COMP_MASK_ALL,
                                        CMPTO_J2K_ENC_RES_MASK_ALL, s->rate / 8 / tx->fps),
                                "Setting rate limit",
                                NOOP);
                //CMPTO_J2K_Enc_Settings_Enable(s->enc_settings, CMPTO_J2K_Rate_Control);
        }
        CHECK_OK(cmpto_j2k_enc_img_create(s->context, &img),
                        "Image create", return);

        CHECK_OK(cmpto_j2k_enc_img_allocate_custom_data(
                                img,
                                sizeof(struct video_desc) + sizeof(shared_ptr<video_frame>),
                                &udata),
                        "Allocate custom image data",
                        HANDLE_ERROR_COMPRESS_PUSH);
        memcpy(udata, &desc, sizeof(desc));

        ref = (shared_ptr<video_frame> *)((char *) udata + sizeof(struct video_desc));
        new (ref) shared_ptr<video_frame>(s->pool.get_frame());

        memcpy(ref->get()->tiles[0].data, tx->tiles[0].data, tx->tiles[0].data_len);
        CHECK_OK(cmpto_j2k_enc_img_set_samples(img, ref->get()->tiles[0].data, tx->tiles[0].data_len, release_cstream),
                        "Setting image samples", HANDLE_ERROR_COMPRESS_PUSH);

        unique_lock<mutex> lk(s->lock);
        s->frame_popped.wait(lk, [s]{return s->in_frames < s->max_in_frames;});
        lk.unlock();
        CHECK_OK(cmpto_j2k_enc_img_encode(img, s->enc_settings),
                        "Encode image", return);
        lk.lock();
        s->in_frames++;
        lk.unlock();

}

static void j2k_compress_done(struct module *mod)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;

        cmpto_j2k_enc_cfg_destroy(s->enc_settings);
        cmpto_j2k_enc_ctx_destroy(s->context);

        delete s;
}

static struct video_compress_info j2k_compress_info = {
        "cmpto_j2k",
        j2k_compress_init,
        NULL,
        NULL,
        j2k_compress_push,
        j2k_compress_pop,
        [] { return list<compress_preset>{}; }
};

REGISTER_MODULE(cmpto_j2k, &j2k_compress_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);


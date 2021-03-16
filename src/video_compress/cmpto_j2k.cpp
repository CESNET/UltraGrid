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
/**
 * @file
 * Main idea behind the code below is to control how many frames the encoder
 * holds. The codec itself doesn't have a limit, thus without that it is
 * possible to run out of memory. This is possible even in the case when
 * the GPU is powerful enough due to the fact that CUDA registers the new
 * buffers which is very slow and because of that the frames cumulate before
 * the GPU encoder.
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

constexpr const char *MOD_NAME = "[Cmpto J2K enc.] ";

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "[J2K enc.] %s: %s\n", \
                                err_msg, cmpto_j2k_enc_get_last_error()); \
                action_fail;\
        } \
} while(0)

#define NOOP ((void) 0)
/// default max size of state_video_compress_j2k::pool and also value
/// for state_video_compress_j2k::max_in_frames
#define DEFAULT_POOL_SIZE 4
/// number of frames that encoder encodes at moment
#define DEFAULT_TILE_LIMIT 1
#define DEFAULT_MEM_LIMIT 1000000000llu

using namespace std;

struct state_video_compress_j2k {
        state_video_compress_j2k(long long int bitrate, unsigned int pool_size, int mct)
                : rate{bitrate}, mct(mct), pool{pool_size}, max_in_frames{pool_size} {}
        struct module module_data{};

        struct cmpto_j2k_enc_ctx *context{};
        struct cmpto_j2k_enc_cfg *enc_settings{};
        long long int rate; ///< bitrate in bits per second
        int mct; // force use of mct - -1 means default
        video_frame_pool pool; ///< pool for frames allocated by us but not yet consumed by encoder
        unsigned int max_in_frames; ///< max number of frames between push and pop
        unsigned int in_frames{};   ///< number of currently encoding frames
        mutex lock;
        condition_variable frame_popped;
        video_desc saved_desc{}; ///< for pool reconfiguration
        video_desc precompress_desc{};
        video_desc compressed_desc{};
        void (*convertFunc)(video_frame *dst, video_frame *src){nullptr};
};

static void j2k_compressed_frame_dispose(struct video_frame *frame);
static void j2k_compress_done(struct module *mod);

static void R12L_to_RG48(video_frame *dst, video_frame *src){
        int src_pitch = vc_get_linesize(src->tiles[0].width, src->color_spec);
        int dst_pitch = vc_get_linesize(dst->tiles[0].width, dst->color_spec);

        unsigned char *s = (unsigned char *) src->tiles[0].data;
        unsigned char *d = (unsigned char *) dst->tiles[0].data;

        for(unsigned i = 0; i < src->tiles[0].height; i++){
                vc_copylineR12LtoRG48(d, s, dst_pitch, 0, 0, 0);
                s += src_pitch;
                d += dst_pitch;
        }
}

static struct {
        codec_t ug_codec;
        enum cmpto_sample_format_type cmpto_sf;
        codec_t convert_codec;
        void (*convertFunc)(video_frame *dst, video_frame *src);
} codecs[] = {
        {UYVY, CMPTO_422_U8_P1020, VIDEO_CODEC_NONE, nullptr},
        {v210, CMPTO_422_U10_V210, VIDEO_CODEC_NONE, nullptr},
        {RGB, CMPTO_444_U8_P012, VIDEO_CODEC_NONE, nullptr},
        {RGBA, CMPTO_444_U8_P012Z, VIDEO_CODEC_NONE, nullptr},
        {R10k, CMPTO_444_U10U10U10_MSB32BE_P210, VIDEO_CODEC_NONE, nullptr},
        {R12L, CMPTO_444_U12_MSB16LE_P012, RG48, R12L_to_RG48},
};

static bool configure_with(struct state_video_compress_j2k *s, struct video_desc desc){
        enum cmpto_sample_format_type sample_format;
        bool found = false;

        for(const auto &codec : codecs){
                if(codec.ug_codec == desc.color_spec){
                        sample_format = codec.cmpto_sf;
                        s->convertFunc = codec.convertFunc;
                        s->precompress_desc = desc;
                        if(codec.convert_codec != VIDEO_CODEC_NONE){
                                s->precompress_desc.color_spec = codec.convert_codec;
                        }
                        found = true;
                        break;
                }
        }

        if(!found){
                log_msg(LOG_LEVEL_ERROR, "[J2K] Failed to find suitable pixel format\n");
                return false;
        }

        CHECK_OK(cmpto_j2k_enc_cfg_set_samples_format_type(s->enc_settings, sample_format),
                        "Setting sample format", return false);
        CHECK_OK(cmpto_j2k_enc_cfg_set_size(s->enc_settings, desc.width, desc.height),
                        "Setting image size", return false);
        if (s->rate) {
                CHECK_OK(cmpto_j2k_enc_cfg_set_rate_limit(s->enc_settings,
                                        CMPTO_J2K_ENC_COMP_MASK_ALL,
                                        CMPTO_J2K_ENC_RES_MASK_ALL, s->rate / 8 / desc.fps),
                                "Setting rate limit",
                                NOOP);
        }

        int mct = s->mct;
        if (mct == -1) {
                mct = codec_is_a_rgb(desc.color_spec) ? 1 : 0;
        }
        CHECK_OK(cmpto_j2k_enc_cfg_set_mct(s->enc_settings, mct),
                        "Setting MCT",
                        NOOP);

        s->compressed_desc = desc;
        s->compressed_desc.color_spec = codec_is_a_rgb(desc.color_spec) ? J2KR : J2K;
        s->compressed_desc.tile_count = 1;

        s->saved_desc = desc;

        return true;
}

static shared_ptr<video_frame> get_copy(struct state_video_compress_j2k *s, video_frame *frame){
        std::shared_ptr<video_frame> ret = s->pool.get_frame();

        if (s->convertFunc) {
                s->convertFunc(ret.get(), frame);
        } else {
                memcpy(ret->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
        }

        return ret;
}

/**
 * @fn j2k_compress_pop
 * @note
 * Do not return empty frame in case of error - that would be interpreted
 * as a poison pill (see below) and would stop the further processing
 * pipeline. Because of that goto + start label is used.
 */
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
        if (!img) {
                // this happens cmpto_j2k_enc_ctx_stop() is called
                // pass poison pill further
                return {};
        }
        if (status != CMPTO_J2K_ENC_IMG_OK) {
                const char * encoding_error = "";
                CHECK_OK(cmpto_j2k_enc_img_get_error(img, &encoding_error), "get error status",
                                encoding_error = "(failed)");
                log_msg(LOG_LEVEL_ERROR, "Image encoding failed: %s\n", encoding_error);
                goto start;
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
        CHECK_OK(cmpto_j2k_enc_img_destroy(img), "Destroy image", NOOP);
        out->callbacks.dispose = j2k_compressed_frame_dispose;
        return shared_ptr<video_frame>(out, out->callbacks.dispose);
}

static void usage() {
        printf("J2K compress usage:\n");
        printf("\t-c cmpto_j2k[:rate=<bitrate>][:quality=<q>][:mct][:mem_limit=<m>][:tile_limit=<t>][:pool_size=<p>] [--cuda-device <c_index>]\n");
        printf("\twhere:\n");
        printf("\t\t<bitrate> - target bitrate\n");
        printf("\t\t<q> - quality\n");
        printf("\t\t<m> - CUDA device memory limit (in bytes), default %llu\n", DEFAULT_MEM_LIMIT);
        printf("\t\t<t> - number of tiles encoded at moment (less to reduce latency, more to increase performance, 0 means infinity), default %d\n", DEFAULT_TILE_LIMIT);
        printf("\t\t<p> - total number of tiles encoder can hold at moment (same meaning as above), default %d, should be greater than <t>\n", DEFAULT_POOL_SIZE);
        printf("\t\tmct - use MCT\n");
        printf("\t\t<c_index> - CUDA device(s) to use (comma separated)\n");
}

static struct module * j2k_compress_init(struct module *parent, const char *c_cfg)
{
        struct state_video_compress_j2k *s;
        double quality = 0.7;
        int mct = -1;
        long long int bitrate = 0;
        long long int mem_limit = DEFAULT_MEM_LIMIT;
        unsigned int tile_limit = DEFAULT_TILE_LIMIT;
        unsigned int pool_size = DEFAULT_POOL_SIZE;

        const auto *version = cmpto_j2k_enc_get_version();
        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Using codec version: " << (version == nullptr ? "(unknown)" : version->name) << "\n";

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
                } else if (strcasecmp("mct", item) == 0 || strcasecmp("nomct", item) == 0) {
                        mct = strcasecmp("mct", item) ? 1 : 0;
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

        s = new state_video_compress_j2k(bitrate, pool_size, mct);

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
        delete s;
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

        if (tx == NULL) { // pass poison pill through encoder
                CHECK_OK(cmpto_j2k_enc_ctx_stop(s->context), "stop", NOOP);
                return;
        }

        desc = video_desc_from_frame(tx.get());
        if (!video_desc_eq(s->saved_desc, desc)) {
                int ret = configure_with(s, desc);
                if (!ret) {
                        return;
                }
                s->pool.reconfigure(s->precompress_desc, vc_get_linesize(s->precompress_desc.width, s->precompress_desc.color_spec)
                                * s->precompress_desc.height);
        }

        assert(tx->tile_count == 1); // TODO

        CHECK_OK(cmpto_j2k_enc_img_create(s->context, &img),
                        "Image create", return);

        /*
         * Copy video desc to udata (to be able to reconstruct in j2k_compress_pop().
         * Further make a place for a shared pointer of allocated data, deleter
         * returns frame to pool in call of release_cstream() callback (called when
         * encoder no longer needs the input data).
         */
        CHECK_OK(cmpto_j2k_enc_img_allocate_custom_data(
                                img,
                                sizeof(struct video_desc) + sizeof(shared_ptr<video_frame>),
                                &udata),
                        "Allocate custom image data",
                        HANDLE_ERROR_COMPRESS_PUSH);
        memcpy(udata, &s->compressed_desc, sizeof(s->compressed_desc));

        ref = (shared_ptr<video_frame> *)((char *) udata + sizeof(struct video_desc));
        new (ref) shared_ptr<video_frame>(get_copy(s, tx.get()));

        CHECK_OK(cmpto_j2k_enc_img_set_samples(img, ref->get()->tiles[0].data, ref->get()->tiles[0].data_len, release_cstream),
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
        NULL,
        NULL,
        j2k_compress_push,
        j2k_compress_pop,
        [] { return list<compress_preset>{}; },
        NULL
};

REGISTER_MODULE(cmpto_j2k, &j2k_compress_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);


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
#include "video_compress.h"
#include "video.h"

#include <cmpto_j2k_enc.h>

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

using namespace std;

struct encoded_image {
        char *data;
        int len;
        struct video_desc *desc;
};

struct state_video_compress_j2k {
        struct module module_data;

        struct cmpto_j2k_enc_ctx *context;
        struct cmpto_j2k_enc_cfg *enc_settings;

        pthread_cond_t frame_ready;
        pthread_mutex_t lock;
        queue<struct encoded_image *> *encoded_images;

        pthread_t thread_id;
};

static void j2k_compress_done(struct module *mod);
static void *j2k_compress_worker(void *args);

static void *j2k_compress_worker(void *args)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) args;

        while (true) {
                struct cmpto_j2k_enc_img *img;
                int status;
                CHECK_OK(cmpto_j2k_enc_ctx_get_encoded_img(
                                        s->context,
                                        1,
                                        &img /* Set to NULL if encoder stopped */,
                                        &status), "Encode image", continue);
                if (status != CMPTO_J2K_ENC_IMG_OK) {
                        const char * encoding_error;
                        CHECK_OK(cmpto_j2k_enc_img_get_error(img, &encoding_error), "get error status", NOOP);
                        log_msg(LOG_LEVEL_ERROR, "Image encoding failed: %s\n", encoding_error);
                        // some better solution?
                        continue;
                }

                if (img == NULL) {
                        break;
                }
                struct video_desc *desc;
                size_t len;
                CHECK_OK(cmpto_j2k_enc_img_get_custom_data(img, (void **) &desc, &len),
                                "get custom data", continue);
                size_t size;
                void * ptr;
                CHECK_OK(cmpto_j2k_enc_img_get_cstream(img, &ptr, &size),
                                "get cstream", continue);
                struct encoded_image *encoded = (struct encoded_image *)
                        malloc(sizeof(struct encoded_image));
                encoded->data = (char *) malloc(size);
                memcpy(encoded->data, ptr, size);
                encoded->len = size;
                encoded->desc = (struct video_desc *) malloc(sizeof(struct video_frame));
                memcpy(encoded->desc, desc, sizeof(struct video_frame));
                encoded->desc->color_spec = codec_is_a_rgb(desc->color_spec) ? J2KR : J2K;
                CHECK_OK(cmpto_j2k_enc_img_destroy(img), "Destroy image", NOOP);

                pthread_mutex_lock(&s->lock);
                s->encoded_images->push(encoded);
                pthread_cond_signal(&s->frame_ready);
                pthread_mutex_unlock(&s->lock);
        }

        return NULL;
}

static struct module * j2k_compress_init(struct module *parent, const char *c_cfg)
{
        struct state_video_compress_j2k *s;
        int j2k_error;
        int rate = 1100000;
        double quality = 0.7;
        bool mct = false;

        s = (struct state_video_compress_j2k *) calloc(1, sizeof(struct state_video_compress_j2k));

        char *cfg = strdup(c_cfg);
        char *save_ptr, *item, *tmp;
        tmp = cfg;
        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                tmp = NULL;
                if (strncasecmp("rate=", item, strlen("rate=")) == 0) {
                        rate = atoi(item + strlen("rate="));
                } else if (strncasecmp("quality=", item, strlen("quality=")) == 0) {
                        quality = atof(item + strlen("quality="));
                } else if (strcasecmp("mct", item) == 0) {
                        mct = true;
                }
        }
        free(cfg);

        struct cmpto_j2k_enc_ctx_cfg *ctx_cfg;
        cmpto_j2k_enc_ctx_cfg_create(&ctx_cfg);
        cmpto_j2k_enc_ctx_cfg_add_cuda_device(ctx_cfg, cuda_devices[0], 0, 0);

        j2k_error = cmpto_j2k_enc_ctx_create(ctx_cfg,
                        &s->context);
        cmpto_j2k_enc_ctx_cfg_destroy(ctx_cfg);
        if (j2k_error != CMPTO_OK) {
                fprintf(stderr, "enc_ctx_create: %s\n", cmpto_j2k_enc_get_last_error());
                goto error;
        }

        j2k_error = cmpto_j2k_enc_cfg_create(
                        s->context,
                        &s->enc_settings);
        if (j2k_error != CMPTO_OK) {
                fprintf(stderr, "enc_cfg_create: %s\n", cmpto_j2k_enc_get_last_error());
                goto error;
        }
        cmpto_j2k_enc_cfg_set_quantization(
                        s->enc_settings,
                        quality /* 0.0 = poor quality, 1.0 = full quality */
                        );

        cmpto_j2k_enc_cfg_set_rate_limit(s->enc_settings, CMPTO_J2K_ENC_COMP_MASK_ALL, CMPTO_J2K_ENC_RES_MASK_ALL, rate);
        //CMPTO_J2K_Enc_Settings_Enable(s->enc_settings, CMPTO_J2K_Rate_Control);
        if (mct) {
                cmpto_j2k_enc_cfg_set_mct(s->enc_settings, 1); // only for RGB
        }

        j2k_error = cmpto_j2k_enc_cfg_set_resolutions( s->enc_settings, 6);
        if (j2k_error != CMPTO_OK) {
                goto error;
        }
        assert(pthread_cond_init(&s->frame_ready, NULL) == 0);
        assert(pthread_mutex_init(&s->lock, NULL) == 0);

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = j2k_compress_done;
        module_register(&s->module_data, parent);

        s->encoded_images = new queue<struct encoded_image *>();

        assert(pthread_create(&s->thread_id, NULL, j2k_compress_worker,
                                (void *) s) == 0);

        return &s->module_data;

error:
        if (s) {
                free(s);
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
        delete *(shared_ptr<video_frame> **) ((char *) custom_data + sizeof(struct video_desc));
}

static shared_ptr<video_frame> j2k_compress(struct module *mod, shared_ptr<video_frame> tx)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;
        struct cmpto_j2k_enc_img *img;
        int j2k_error;
        struct video_desc desc;
        void *udata;
        shared_ptr<video_frame> **ref;

        if (tx == NULL)
                goto get_frame_from_queue;

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
        cmpto_j2k_enc_cfg_set_samples_format_type(s->enc_settings, cmpto_sf);
        cmpto_j2k_enc_cfg_set_size(s->enc_settings, tx->tiles[0].width, tx->tiles[0].height);

        j2k_error = cmpto_j2k_enc_img_create(s->context, &img);
        if (j2k_error != CMPTO_OK) {
                return NULL;
        }

        j2k_error = cmpto_j2k_enc_img_set_samples(img, tx->tiles[0].data, tx->tiles[0].data_len, release_cstream);

        if (j2k_error != CMPTO_OK) {
                return NULL;
        }

        desc = video_desc_from_frame(tx.get());

        j2k_error = cmpto_j2k_enc_img_allocate_custom_data(
                        img,
                        sizeof(struct video_desc) + sizeof(shared_ptr<video_frame> *),
                        &udata);
        if (j2k_error != CMPTO_OK) {
                return NULL;
        }
        memcpy(udata, &desc, sizeof(desc));
        ref = (shared_ptr<video_frame> **)((char *) udata + sizeof(struct video_desc));
        *ref = new shared_ptr<video_frame>(tx);

        j2k_error = cmpto_j2k_enc_img_encode(img, s->enc_settings);
        if (j2k_error != CMPTO_OK) {
                return NULL;
        }

get_frame_from_queue:
        pthread_mutex_lock(&s->lock);
        struct encoded_image *encoded_img = NULL;
        if (s->encoded_images->size() > 0) {
                encoded_img = s->encoded_images->front();
                s->encoded_images->pop();
        }
        pthread_mutex_unlock(&s->lock);

        if (encoded_img != NULL) {
                struct video_frame *out = vf_alloc_desc(*(encoded_img->desc));

                free(encoded_img->desc);
                out->tiles[0].data = encoded_img->data;
                out->tiles[0].data_len =
                        encoded_img->len;
                out->dispose = j2k_compressed_frame_dispose;
                free(encoded_img);
                assert (out->tiles[0].data_len != 0);
                return shared_ptr<video_frame>(out, out->dispose);
        } else {
                return {};
        }
}


static void j2k_compress_done(struct module *mod)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;

        cmpto_j2k_enc_cfg_destroy(s->enc_settings);
        cmpto_j2k_enc_ctx_destroy(s->context);
        pthread_cond_destroy(&s->frame_ready);
        pthread_mutex_destroy(&s->lock);
        delete s->encoded_images;

        free(s);
}

struct video_compress_info j2k_info = {
        "j2k",
        j2k_compress_init,
        j2k_compress,
        NULL,
        NULL,
        NULL,
        [] { return list<compress_preset>{}; }
};

REGISTER_MODULE(j2k, &j2k_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);


/**
 * @file   video_decompress/j2k.cpp
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
#include "utils/misc.h"
#include "video.h"
#include "video_decompress.h"

#include <cmpto_j2k_dec.h>

#include <queue>

using std::queue;

struct state_decompress_j2k {
        cmpto_j2k_dec_ctx *decoder;
        cmpto_j2k_dec_cfg *settings;

        struct video_desc desc;
        codec_t out_codec;

        pthread_mutex_t lock;
        queue<char *> *decompressed_frames;
        pthread_t thread_id;
};

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "[J2K] %s: %s\n", \
                                err_msg, cmpto_j2k_dec_get_last_error()); \
                action_fail;\
        } \
} while(0)

#define NOOP ((void) 0)

static void *decompress_j2k_worker(void *args)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) args;

        while (true) {
                struct cmpto_j2k_dec_img *img;
                int decoded_img_status;
                CHECK_OK(cmpto_j2k_dec_ctx_get_decoded_img(s->decoder, 1, &img, &decoded_img_status),
				"Decode image", continue);
                if (img == NULL) {
                        /// @todo what about reconfiguration
                        break;
                }

                if (decoded_img_status != CMPTO_J2K_DEC_IMG_OK) {
			const char * decoding_error = "";
			CHECK_OK(cmpto_j2k_dec_img_get_error(img, &decoding_error), "get error status",
					decoding_error = "(failed)");
			log_msg(LOG_LEVEL_ERROR, "Image decoding failed: %s\n", decoding_error);
                        continue;
                }

                void *dec_data;
                size_t len;
                CHECK_OK(cmpto_j2k_dec_img_get_samples(img, &dec_data, &len),
                                "Error getting samples", cmpto_j2k_dec_img_destroy(img); continue);

                char *buffer = (char *) malloc(len);
                memcpy(buffer, dec_data, len);

                pthread_mutex_lock(&s->lock);
                s->decompressed_frames->push(buffer);
                pthread_mutex_unlock(&s->lock);
                CHECK_OK(cmpto_j2k_dec_img_destroy(img),
                                "Unable to to return processed image", NOOP);
        }

        return NULL;
}

ADD_TO_PARAM(j2k_dec_mem_limit, "j2k-dec-mem-limit", "* j2k-dec-mem-limit=<limit>\n"
                                "  J2K max memory usage in bytes.\n");
ADD_TO_PARAM(j2k_dec_tile_limit, "j2k-dec-tile-limit", "* j2k-dec-tile-limit=<limit>\n"
                                "  number of tiles decoded at moment (less to reduce latency, more to increase performance)\n");
static void * j2k_decompress_init(void)
{
        struct state_decompress_j2k *s = NULL;
        long long int mem_limit = 0;
        unsigned int tile_limit = 0u;

        if (get_commandline_param("j2k-dec-mem-limit")) {
                mem_limit = unit_evaluate(get_commandline_param("j2k-dec-mem-limit"));
        }

        if (get_commandline_param("j2k-dec-tile-limit")) {
                tile_limit = atoi(get_commandline_param("j2k-dec-tile-limit"));
        }

        s = (struct state_decompress_j2k *)
                calloc(1, sizeof(struct state_decompress_j2k));
        assert(pthread_mutex_init(&s->lock, NULL) == 0);

        struct cmpto_j2k_dec_ctx_cfg *ctx_cfg;
        CHECK_OK(cmpto_j2k_dec_ctx_cfg_create(&ctx_cfg), "Error creating dec cfg", goto error);
        for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                CHECK_OK(cmpto_j2k_dec_ctx_cfg_add_cuda_device(ctx_cfg, cuda_devices[i], mem_limit, tile_limit),
                                "Error setting CUDA device", goto error);
        }

        CHECK_OK(cmpto_j2k_dec_ctx_create(ctx_cfg, &s->decoder), "Error initializing context",
                        goto error);

        CHECK_OK(cmpto_j2k_dec_ctx_cfg_destroy(ctx_cfg), "Destroy cfg", NOOP);

        CHECK_OK(cmpto_j2k_dec_cfg_create(s->decoder, &s->settings), "Error creating configuration",
                        goto error);

        s->decompressed_frames = new queue<char *>();

        assert(pthread_create(&s->thread_id, NULL, decompress_j2k_worker,
                                (void *) s) == 0);

        return s;

error:
        delete s->decompressed_frames;
        if (s->settings) {
                cmpto_j2k_dec_cfg_destroy(s->settings);
        }
        if (s->decoder) {
                cmpto_j2k_dec_ctx_destroy(s->decoder);
        }
        if (s) {
                pthread_mutex_destroy(&s->lock);
                free(s);
        }
        return NULL;
}

static int j2k_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_j2k *s = (struct state_decompress_j2k *) state;

        assert((rshift == 0 && gshift == 8 && bshift == 16) ||
                        (rshift == 16 && gshift == 8 && bshift == 0));
        assert(pitch == vc_get_linesize(desc.width, out_codec));

        enum cmpto_sample_format_type cmpto_sf;
        switch (out_codec) {
                case UYVY:
                        cmpto_sf = CMPTO_422_U8_P1020;
                        break;
                case v210:
                        cmpto_sf = CMPTO_422_U10_V210;
                        break;
                case RGB:
                        cmpto_sf = (rshift == 0 ?  CMPTO_444_U8_P012 : CMPTO_444_U8_P210 /*BGR*/);
                        break;
                case R10k:
                        cmpto_sf = CMPTO_444_U10U10U10_MSB32BE_P210;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "[J2K] Unsupported output codec: %s\n",
                                        get_codec_name(out_codec));
                        abort();
        }
        CHECK_OK(cmpto_j2k_dec_cfg_set_samples_format_type(s->settings, cmpto_sf),
                        "Error setting sample format type", return FALSE);

        s->desc = desc;
        s->out_codec = out_codec;

        return TRUE;
}

static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size)
{
        (void) custom_data; (void) custom_data_size; (void) codestream_size;
        free(const_cast<void *>(codestream));
}

static decompress_status j2k_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int /* frame_seq */)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) state;
        struct cmpto_j2k_dec_img *img;
        char *decoded;
        void *tmp;

        CHECK_OK(cmpto_j2k_dec_img_create(s->decoder, &img),
                        "Could not create frame", goto return_previous);

        tmp = malloc(src_len);
        memcpy(tmp, buffer, src_len);
        CHECK_OK(cmpto_j2k_dec_img_set_cstream(img, tmp, src_len, &release_cstream),
                        "Error setting cstream", cmpto_j2k_dec_img_destroy(img); goto return_previous);

        CHECK_OK(cmpto_j2k_dec_img_decode(img, s->settings), "Decode image",
                        cmpto_j2k_dec_img_destroy(img); goto return_previous);

return_previous:
        pthread_mutex_lock(&s->lock);
        if (s->decompressed_frames->size() == 0) {
                pthread_mutex_unlock(&s->lock);
                return DECODER_NO_FRAME;
        }
        decoded = s->decompressed_frames->front();
        s->decompressed_frames->pop();
        pthread_mutex_unlock(&s->lock);

        memcpy(dst, decoded, s->desc.height *
                        vc_get_linesize(s->desc.width, s->out_codec));

        free(decoded);

        return DECODER_GOT_FRAME;
}

static int j2k_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = FALSE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static void j2k_decompress_done(void *state)
{
        struct state_decompress_j2k *s = (struct state_decompress_j2k *) state;

        cmpto_j2k_dec_ctx_stop(s->decoder);
        pthread_join(s->thread_id, NULL);
        log_msg(LOG_LEVEL_VERBOSE, "[J2K dec.] Decoder stopped.\n");

        cmpto_j2k_dec_cfg_destroy(s->settings);
        cmpto_j2k_dec_ctx_destroy(s->decoder);

        pthread_mutex_destroy(&s->lock);

        while (s->decompressed_frames->size() > 0) {
                char *decoded = s->decompressed_frames->front();
                s->decompressed_frames->pop();
                free(decoded);
        }
        delete s->decompressed_frames;

        free(s);
}

static const struct decode_from_to *j2k_decompress_get_decoders() {

        static const struct decode_from_to ret[] = {
                { J2K, UYVY, 300 },
                { J2K, v210, 200 }, // prefer decoding to 10-bit
                { J2KR, RGB, 300 },
                { J2KR, R10k, 200 }, // ditto
                { VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 0 }
        };
        return ret;
}

static const struct video_decompress_info j2k_decompress_info = {
        j2k_decompress_init,
        j2k_decompress_reconfigure,
        j2k_decompress,
        j2k_decompress_get_property,
        j2k_decompress_done,
        j2k_decompress_get_decoders,
};

REGISTER_MODULE(j2k, &j2k_decompress_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);


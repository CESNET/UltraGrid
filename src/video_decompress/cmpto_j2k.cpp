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
/**
 * Some of the concepts are similar to encoder (eg. keeping limited number of
 * frames in decoder) so please refer to that file.
 *
 * Problematic part of following code is that UltraGrid decompress API is
 * synchronous only while the CMPTO J2K decoder is inherently asynchronous.
 * Threrefore the integration works in following fashion:
 * - there is a thread that waits for completed (decompressed) frames,
 *   if there is any, it put it in queue (or drop if full)
 * - when a new frame arives, j2k_decompress() passes it to decoder
 *   (which is asynchronous, thus non-blocking)
 * - then queue (filled by thread in first point) is checked - if it is
 *   non-empty, frame is copied to framebufffer. If not false is returned.
 *
 * @todo
 * Reconfiguration isn't entirely correct - on reconfigure, all frames
 * should be dropped and not copied to framebuffer. However this is usually
 * not an issue because dynamic video change is rare (except switching to
 * another stream, which, however, creates a new decoder).
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

#include <mutex>
#include <queue>
#include <utility>

#define DEFAULT_TILE_LIMIT 1
/// maximal size of queue for decompressed frames
#define DEFAULT_MAX_QUEUE_SIZE 2
/// maximal number of concurrently decompressed frames
#define DEFAULT_MAX_IN_FRAMES 4
#define DEFAULT_MEM_LIMIT 1000000000ll

using namespace std;

struct state_decompress_j2k {
        state_decompress_j2k(unsigned int mqs, unsigned int mif)
                : max_queue_size(mqs), max_in_frames(mif) {}
        cmpto_j2k_dec_ctx *decoder{};
        cmpto_j2k_dec_cfg *settings{};

        struct video_desc desc{};
        codec_t out_codec{};

        mutex lock;
        queue<pair<char *, size_t>> decompressed_frames; ///< buffer, length
        pthread_t thread_id{};
        unsigned int max_queue_size; ///< maximal length of @ref decompressed_frames
        unsigned int max_in_frames; ///< maximal frames that can be "in progress"
        unsigned int in_frames{}; ///< actual number of decompressed frames

        unsigned long long int dropped{}; ///< number of dropped frames because queue was full

        void (*convert)(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                unsigned int width, unsigned int height);
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

static void rg48_to_r12l(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                unsigned int width, unsigned int height)
{
        int src_pitch = vc_get_linesize(width, RG48);
        int dst_len = vc_get_linesize(width, R12L);

        for(unsigned i = 0; i < height; i++){
                vc_copylineRG48toR12L(dst_buffer, src_buffer, dst_len);
                src_buffer += src_pitch;
                dst_buffer += dst_len;
        }
}

/**
 * This function just runs in thread and gets decompressed images from decoder
 * putting them to queue (or dropping if full).
 */
static void *decompress_j2k_worker(void *args)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) args;

        while (true) {
                struct cmpto_j2k_dec_img *img;
                int decoded_img_status;
                CHECK_OK(cmpto_j2k_dec_ctx_get_decoded_img(s->decoder, 1, &img, &decoded_img_status),
				"Decode image", continue);

                {
                        lock_guard<mutex> lk(s->lock);
                        if (s->in_frames) s->in_frames--;
                }

                if (decoded_img_status != CMPTO_J2K_DEC_IMG_OK) {
			const char * decoding_error = "";
			CHECK_OK(cmpto_j2k_dec_img_get_error(img, &decoding_error), "get error status",
					decoding_error = "(failed)");
			log_msg(LOG_LEVEL_ERROR, "Image decoding failed: %s\n", decoding_error);
                        continue;
                }

                if (img == NULL) { // decoder stopped (poison pill)
                        break;
                }

                void *dec_data;
                size_t len;
                CHECK_OK(cmpto_j2k_dec_img_get_samples(img, &dec_data, &len),
                                "Error getting samples", cmpto_j2k_dec_img_destroy(img); continue);

                char *buffer = (char *) malloc(len);
                if (s->convert) {
                        s->convert((unsigned char*) buffer, (unsigned char*) dec_data, s->desc.width, s->desc.height);
                        len = vc_get_linesize(s->desc.width, s->out_codec) * s->desc.height;
                } else {
                        memcpy(buffer, dec_data, len);
                }

                CHECK_OK(cmpto_j2k_dec_img_destroy(img),
                                "Unable to to return processed image", NOOP);
                lock_guard<mutex> lk(s->lock);
                while (s->decompressed_frames.size() >= s->max_queue_size) {
                        if (s->dropped++ % 10 == 0) {
                                log_msg(LOG_LEVEL_WARNING, "[J2K dec] Some frames (%llu) dropped.\n", s->dropped);

                        }
                        auto decoded = s->decompressed_frames.front();
                        s->decompressed_frames.pop();
                        free(decoded.first);
                }
                s->decompressed_frames.push({buffer,len});
        }

        return NULL;
}

ADD_TO_PARAM(j2k_dec_mem_limit, "j2k-dec-mem-limit", "* j2k-dec-mem-limit=<limit>\n"
                                "  J2K max memory usage in bytes.\n");
ADD_TO_PARAM(j2k_dec_tile_limit, "j2k-dec-tile-limit", "* j2k-dec-tile-limit=<limit>\n"
                                "  number of tiles decoded at moment (less to reduce latency, more to increase performance, 0 unlimited)\n");
ADD_TO_PARAM(j2k_dec_queue_len, "j2k-dec-queue-len", "* j2k-queue-len=<len>\n"
                                "  max queue len\n");
ADD_TO_PARAM(j2k_dec_encoder_queue, "j2k-dec-encoder-queue", "* j2k-encoder-queue=<len>\n"
                                "  max number of frames hold by encoder\n");
static void * j2k_decompress_init(void)
{
        struct state_decompress_j2k *s = NULL;
        long long int mem_limit = DEFAULT_MEM_LIMIT;
        unsigned int tile_limit = DEFAULT_TILE_LIMIT;
        unsigned int queue_len = DEFAULT_MAX_QUEUE_SIZE;
        unsigned int encoder_in_frames = DEFAULT_MAX_IN_FRAMES;

        if (get_commandline_param("j2k-dec-mem-limit")) {
                mem_limit = unit_evaluate(get_commandline_param("j2k-dec-mem-limit"));
        }

        if (get_commandline_param("j2k-dec-tile-limit")) {
                tile_limit = atoi(get_commandline_param("j2k-dec-tile-limit"));
        }

        if (get_commandline_param("j2k-dec-queue-len")) {
                queue_len = atoi(get_commandline_param("j2k-dec-queue-len"));
        }

        if (get_commandline_param("j2k-dec-encoder-queue")) {
                encoder_in_frames = atoi(get_commandline_param("j2k-dec-encoder-queue"));
        }


        s = new state_decompress_j2k(queue_len, encoder_in_frames);

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

        assert(pthread_create(&s->thread_id, NULL, decompress_j2k_worker,
                                (void *) s) == 0);

        return s;

error:
        if (s->settings) {
                cmpto_j2k_dec_cfg_destroy(s->settings);
        }
        if (s->decoder) {
                cmpto_j2k_dec_ctx_destroy(s->decoder);
        }
        if (s) {
                delete s;
        }
        return NULL;
}

static struct {
        codec_t ug_codec;
        enum cmpto_sample_format_type cmpto_sf;
        void (*convert)(unsigned char *dst_buffer, unsigned char *src_buffer, unsigned int width, unsigned int height);
} codecs[] = {
        {UYVY, CMPTO_422_U8_P1020, nullptr},
        {v210, CMPTO_422_U10_V210, nullptr},
        {RGB, CMPTO_444_U8_P012, nullptr},
        {R10k, CMPTO_444_U10U10U10_MSB32BE_P210, nullptr},
        {R12L, CMPTO_444_U12_MSB16LE_P012, rg48_to_r12l},
};

static int j2k_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_j2k *s = (struct state_decompress_j2k *) state;

        assert((rshift == 0 && gshift == 8 && bshift == 16) ||
                        (rshift == 16 && gshift == 8 && bshift == 0));
        assert(pitch == vc_get_linesize(desc.width, out_codec));

        enum cmpto_sample_format_type cmpto_sf;
        bool found = false;

        for(const auto &codec : codecs){
                if(codec.ug_codec == out_codec){
                        switch (out_codec) {
                                case RGB:
                                        cmpto_sf = (rshift == 0 ?  CMPTO_444_U8_P012 : CMPTO_444_U8_P210 /*BGR*/);
                                        break;
                                case R12L:
                                        log_msg(LOG_LEVEL_NOTICE, "[J2K] Decoding to 12-bit RGB.\n"); /* fall through */
                                default:
                                        cmpto_sf = codec.cmpto_sf;
                        }
                        s->convert = codec.convert;
                        found = true;
                        break;
                }
        }

        if(!found){
                log_msg(LOG_LEVEL_ERROR, "[J2K] Unsupported output codec: %s\n",
                                get_codec_name(out_codec));
                abort();
        }
        CHECK_OK(cmpto_j2k_dec_cfg_set_samples_format_type(s->settings, cmpto_sf),
                        "Error setting sample format type", return false);

        s->desc = desc;
        s->out_codec = out_codec;

        return true;
}

/**
 * Callback called by the codec when codestream is no longer required.
 */
static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size)
{
        (void) custom_data; (void) custom_data_size; (void) codestream_size;
        free(const_cast<void *>(codestream));
}

/**
 * Main decompress function - passes frame to the codec and checks if there are
 * some decoded frames. If so, copies that to framebuffer. In the opposite case
 * it just returns false.
 */
static decompress_status j2k_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int /* frame_seq */, struct video_frame_callbacks * /* callbacks */)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) state;
        struct cmpto_j2k_dec_img *img;
        pair<char *, size_t> decoded;
        void *tmp;

        if (s->in_frames >= s->max_in_frames + 1) {
                if (s->dropped++ % 10 == 0) {
                        log_msg(LOG_LEVEL_WARNING, "[J2K dec] Some frames (%llu) dropped.\n", s->dropped);

                }
                goto return_previous;
        }

        CHECK_OK(cmpto_j2k_dec_img_create(s->decoder, &img),
                        "Could not create frame", goto return_previous);

        tmp = malloc(src_len);
        memcpy(tmp, buffer, src_len);
        CHECK_OK(cmpto_j2k_dec_img_set_cstream(img, tmp, src_len, &release_cstream),
                        "Error setting cstream", cmpto_j2k_dec_img_destroy(img); goto return_previous);

        CHECK_OK(cmpto_j2k_dec_img_decode(img, s->settings), "Decode image",
                        cmpto_j2k_dec_img_destroy(img); goto return_previous);
        {
                lock_guard<mutex> lk(s->lock);
                s->in_frames++;
        }

return_previous:
        unique_lock<mutex> lk(s->lock);
        if (s->decompressed_frames.size() == 0) {
                return DECODER_NO_FRAME;
        }
        decoded = s->decompressed_frames.front();
        s->decompressed_frames.pop();
        lk.unlock();

        memcpy(dst, decoded.first, max<size_t>(s->desc.height *
                        vc_get_linesize(s->desc.width, s->out_codec), decoded.second));

        free(decoded.first);

        return DECODER_GOT_FRAME;
}

static int j2k_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        int ret = false;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = false;
                                *len = sizeof(int);
                                ret = true;
                        }
                        break;
                default:
                        ret = false;
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

        while (s->decompressed_frames.size() > 0) {
                auto decoded = s->decompressed_frames.front();
                s->decompressed_frames.pop();
                free(decoded.first);
        }

        delete s;
}

static const struct decode_from_to *j2k_decompress_get_decoders() {

        static const struct decode_from_to ret[] = {
                { J2K, UYVY, 300 },
                { J2K, v210, 200 }, // prefer decoding to 10-bit
                { J2KR, RGB, 300 },
                { J2KR, R10k, 200 },
                { J2KR, R12L, 100 }, // prefer RGB decoding to 12-bit
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


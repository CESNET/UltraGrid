/**
 * @file   video_compress/cineform.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 */
/* Copyright (c) 2019-2025 CESNET
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#include <cstddef>             // CFHD hdrs use size_t w/o include
#include <CFHDEncoder.h>
#include <CFHDError.h>         // for CFHD_Error
#include <CFHDTypes.h>

#include <cassert>             // for assert
#include <condition_variable>
#include <cstdint>             // for uint32_t
#include <cstdio>              // for printf
#include <cstdlib>             // for atoi, free
#include <cstring>             // for strlen, NULL, memset, size_t, strdup
#include <memory>
#include <mutex>
#include <queue>
#include <string>              // for char_traits, basic_string
#include <tuple>               // for get, tuple
#include <utility>             // for move
#include <vector>

#ifdef MEASUREMENT
#include <time.h>
#endif

#include "compat/strings.h"    // for strncasecmp
#include "debug.h"             // for log_msg, LOG_LEVEL_ERROR, LOG, LOG_LEV...
#include "host.h"              // for INIT_NOERR
#include "lib_common.h"        // for REGISTER_MODULE, library_class
#include "pixfmt_conv.h"       // for get_best_decoder_from, decoder_t
#include "tv.h"                // for get_time_in_ns
#include "types.h"             // for video_desc, tile, video_frame, UYVY, RGB
#include "utils/macros.h"      // for TOSTRING, to_fourcc
#include "utils/color_out.h"   // for color_printf
#include "video_codec.h"       // for vc_get_linesize, get_codec_name
#include "video_compress.h"    // for codec, module_option, encoder, compres...
#include "video_frame.h"       // for vf_free, vf_alloc_desc_data, vf_copy_m...

#define DEFAULT_POOL_SIZE 16
#define DEFAULT_THREAD_COUNT 8
#define MOD_NAME "[cineform] "

struct state_video_compress_cineform{
        std::mutex mutex;

        struct video_desc saved_desc;
        struct video_desc precompress_desc;
        struct video_desc compressed_desc;

        decoder_t dec;

        CFHD_EncodingQuality requested_quality;
        int requested_threads;
        int requested_pool_size;

        CFHD_EncoderPoolRef encoderPoolRef;
        CFHD_MetadataRef metadataRef;

        uint32_t frame_seq_in;
        uint32_t frame_seq_out;

        std::queue<std::unique_ptr<video_frame, decltype(&vf_free)>> frame_queue;

        bool started;
        bool stop;
        std::condition_variable cv;

#ifdef MEASUREMENT
        std::map<uint32_t, struct timespec> times_map;
#endif
};

static void cineform_compress_done(void *state) {
        auto *s = (struct state_video_compress_cineform *) state;

        s->mutex.lock();

        CFHD_StopEncoderPool(s->encoderPoolRef);
        CFHD_ReleaseEncoderPool(s->encoderPoolRef);
        CFHD_MetadataClose(s->metadataRef);

        s->mutex.unlock();
        delete s;
}

struct {
        const char *label;
        const char *key;
        const char *description;
        const char *opt_str;
        const char *placeholder;
} usage_opts[] = {
        {"Quality", "quality", "specifies encode quality, range 1-6 (default: 4)", ":quality=", "4"},
        {"Threads", "num_threads", "specifies number of threads for encoding (default: " TOSTRING(DEFAULT_THREAD_COUNT) ")", ":num_threads=", ""},
        {"Pool size", "pool_size", "specifies the size of encoding pool (default: " TOSTRING(DEFAULT_POOL_SIZE) ")", ":pool_size=", ""},
};

static void usage() {
        color_printf(TBOLD("Cineform") " encoder usage:\n");
        color_printf("\t" TBOLD(
            TRED("-c cineform") "[:quality=<quality>][:threads=<num_threads>][:"
                                "pool_size=<pool_size>]") "\n");
        color_printf("\t" TBOLD("-c cineform:help") "\n");

        color_printf("\nOptions:\n");

        for(const auto& opt : usage_opts){
                color_printf("\t" TBOLD("<%s>") " %s\n", opt.key, opt.description);
        }
        printf("\n");
}

static int parse_fmt(struct state_video_compress_cineform *s, char *fmt) {
        char *item, *save_ptr = NULL;
        if(fmt) {
                while((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                        if(strncasecmp("help", item, strlen("help")) == 0) {
                                usage();
                                return 1;
                        } else if(strncasecmp("quality=", item, strlen("quality=")) == 0) {
                                char *quality = item + strlen("quality=");
                                int qual = atoi(quality);
                                if(qual >= 1 && qual <= 6){
                                        s->requested_quality = static_cast<CFHD_EncodingQuality>(qual);
                                } else {
                                        log_msg(LOG_LEVEL_ERROR, "[cineform] Error: Quality must be in range 1-6.\n");
                                        return -1;
                                }
                        } else if(strncasecmp("threads=", item, strlen("threads=")) == 0) {
                                char *threads = item + strlen("threads=");
                                s->requested_threads = atoi(threads);
                        } else if(strncasecmp("pool_size=", item, strlen("pool_size=")) == 0) {
                                char *pool_size = item + strlen("pool_size=");
                                s->requested_pool_size = atoi(pool_size);
                        } else {
                                log_msg(LOG_LEVEL_ERROR, "[cineform] Error: unknown option %s.\n",
                                                item);
                                return -1;
                        }
                        fmt = NULL;
                }
        }

        return 0;
}

static void * cineform_compress_init(struct module *parent, const char *opts)
{
        (void) parent;
        struct state_video_compress_cineform *s;

        s = new state_video_compress_cineform();

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));
        s->requested_quality = CFHD_ENCODING_QUALITY_DEFAULT;
        s->requested_threads = DEFAULT_THREAD_COUNT;
        s->requested_pool_size = DEFAULT_POOL_SIZE;

        char *fmt = strdup(opts);
        int ret = parse_fmt(s, fmt);
        free(fmt);
        if(ret != 0) {
                delete s;
                return ret > 0 ? INIT_NOERR : nullptr;
        }

        log_msg(LOG_LEVEL_NOTICE, "[cineform] : Threads: %d.\n", s->requested_threads);
        CFHD_Error status = CFHD_ERROR_OKAY;
        status = CFHD_CreateEncoderPool(&s->encoderPoolRef,
                        s->requested_threads,
                        s->requested_pool_size,
                        nullptr);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to create encoder pool\n");
                delete s;
                return nullptr;
        }
        status = CFHD_MetadataOpen(&s->metadataRef);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to create metadataRef\n");
                CFHD_ReleaseEncoderPool(s->encoderPoolRef);
                delete s;
                return nullptr;
        }

        s->frame_seq_in = 0;
        s->frame_seq_out = 0;
        s->started = false;
        s->stop = false;

        return s;
}

static struct {
        codec_t ug_codec;
        CFHD_PixelFormat cfhd_pixel_format;
        CFHD_EncodedFormat cfhd_encoded_format;
} codecs[] = {
        {UYVY, CFHD_PIXEL_FORMAT_2VUY, CFHD_ENCODED_FORMAT_YUV_422},
        {RGB, CFHD_PIXEL_FORMAT_RG24, CFHD_ENCODED_FORMAT_RGB_444},
        {RGBA, CFHD_PIXEL_FORMAT_BGRa, CFHD_ENCODED_FORMAT_RGBA_4444},
        {v210, CFHD_PIXEL_FORMAT_V210, CFHD_ENCODED_FORMAT_YUV_422},
        {R10k, CFHD_PIXEL_FORMAT_DPX0, CFHD_ENCODED_FORMAT_RGB_444},
        {RG48, CFHD_PIXEL_FORMAT_RG48, CFHD_ENCODED_FORMAT_RGB_444},
};

static bool configure_with(struct state_video_compress_cineform *s, struct video_desc desc)
{
        CFHD_Error status = CFHD_ERROR_OKAY;

        if(s->started){
                status = CFHD_StopEncoderPool(s->encoderPoolRef);
                if(status != CFHD_ERROR_OKAY){
                        log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to stop encoder pool\n");
                        return false;
                }
        }

        codec_t to_convs[VIDEO_CODEC_COUNT] = { VIDEO_CODEC_NONE };
        int i = 0;
        for(const auto &codec : codecs) {
                to_convs[i++] = codec.ug_codec;
        }
        s->precompress_desc = desc;
        s->dec = get_best_decoder_from(desc.color_spec, to_convs, &s->precompress_desc.color_spec);
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Using " << get_codec_name(s->precompress_desc.color_spec) << " as intermediate.\n";

        CFHD_PixelFormat pix_fmt = CFHD_PIXEL_FORMAT_UNKNOWN;
        CFHD_EncodedFormat enc_fmt = CFHD_ENCODED_FORMAT_UNKNOWN;
        for(const auto &codec : codecs){
                if (codec.ug_codec == s->precompress_desc.color_spec){
                        pix_fmt = codec.cfhd_pixel_format;
                        enc_fmt = codec.cfhd_encoded_format;
                        break;
                }
        }

        if(pix_fmt == CFHD_PIXEL_FORMAT_UNKNOWN){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to find suitable pixel format\n");
                return false;
        }

        status = CFHD_PrepareEncoderPool(s->encoderPoolRef,
                        desc.width,
                        desc.height,
                        pix_fmt,
                        enc_fmt,
                        CFHD_ENCODING_FLAGS_NONE,
                        s->requested_quality);

        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to prepare to encode\n");
                return false;
        }

        s->compressed_desc = desc;
        s->compressed_desc.color_spec = CFHD;
        s->compressed_desc.tile_count = 1;

        s->saved_desc = desc;

        status = CFHD_AttachEncoderPoolMetadata(s->encoderPoolRef, s->metadataRef);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to attach metadata to encoder pool\n");
        }

        uint32_t fcc_tag = to_fourcc('U', 'G', 'P', 'F');
        uint32_t val = desc.color_spec;
        status = CFHD_MetadataAdd(s->metadataRef,
                                  fcc_tag,
                                  METADATATYPE_UINT32,
                                  sizeof(uint32_t),
                                  &val,
                                  false);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to add metadata %u\n", status);
        }

        status = CFHD_AttachEncoderPoolMetadata(s->encoderPoolRef, s->metadataRef);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to attach metadata to encoder pool\n");
        }

        log_msg(LOG_LEVEL_INFO, "[cineform] start encoder pool\n");
        status = CFHD_StartEncoderPool(s->encoderPoolRef);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to start encoder pool\n");
                return false;
        }

        s->started = true;
        s->cv.notify_all();

        return true;
}

static video_frame *get_copy(struct state_video_compress_cineform *s, video_frame *frame){
        video_frame *ret = vf_alloc_desc_data(s->precompress_desc);
        vf_copy_metadata(ret, frame); // seq and compress_start
        int src_linesize = vc_get_linesize(frame->tiles[0].width, frame->color_spec);
        int dst_linesize = vc_get_linesize(frame->tiles[0].width, ret->color_spec);
        auto *dst = (unsigned char *) ret->tiles[0].data;
        int dst_sign = 1;
        if (s->precompress_desc.color_spec == RGB) { // upside down
                dst += static_cast<size_t>(dst_linesize) *
                       (frame->tiles[0].height - 1);
                dst_sign = -1;
        }
        for (long int i = 0; i < (long int) frame->tiles[0].height; ++i) {
                s->dec(dst + dst_sign * i * dst_linesize, (unsigned char *) frame->tiles[0].data + i * src_linesize, dst_linesize, 16, 8, 0);
        }

        return ret;
}

static void cineform_compress_push(void *state, std::shared_ptr<video_frame> tx)
{
        auto *s = (struct state_video_compress_cineform *) state;

        CFHD_Error status = CFHD_ERROR_OKAY;
        std::unique_lock<std::mutex> lock(s->mutex);

        if(s->stop)
                return;

        if(!tx){
                if (!s->started) {
                        video_desc desc{1920, 1080, UYVY, 25.0, PROGRESSIVE, 1};
                        configure_with(s, desc);
                }
                std::unique_ptr<video_frame, decltype(&vf_free)> dummy(vf_alloc_desc_data(s->precompress_desc), vf_free);
                video_frame *dummy_ptr = dummy.get();
                vf_clear(dummy_ptr);

                s->stop = true;
                s->frame_queue.push(std::move(dummy));
                lock.unlock();

                status = CFHD_EncodeAsyncSample(s->encoderPoolRef,
                                s->frame_seq_in++,
                                dummy_ptr->tiles[0].data,
                                vc_get_linesize(s->precompress_desc.width, s->precompress_desc.color_spec),
                                nullptr);
                if(status != CFHD_ERROR_OKAY){
                        log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to push null %i pool\n", status);
                }
                return;
        }

        assert(tx->tile_count == 1); // TODO

        if(!video_desc_eq_excl_param(video_desc_from_frame(tx.get()),
                                s->saved_desc, PARAM_TILE_COUNT))
        {
                //cleanup(s);
                int ret = configure_with(s, video_desc_from_frame(tx.get()));
                if(!ret) {
                        return;
                }
        }

        std::unique_ptr<video_frame, decltype(&vf_free)> frame_copy(get_copy(s, tx.get()), vf_free);
        video_frame *frame_ptr = frame_copy.get();

        s->frame_queue.push(std::move(frame_copy));

#ifdef MEASUREMENT
        struct timespec t_0;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_0);

        s->times_map[s->frame_seq_in] = t_0;
#endif
        lock.unlock();

        status = CFHD_EncodeAsyncSample(s->encoderPoolRef,
                        s->frame_seq_in++,
                        frame_ptr->tiles[0].data,
                        vc_get_linesize(s->precompress_desc.width, s->precompress_desc.color_spec),
                        nullptr);

        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to push sample to encode pool\n");
                return;
        }
}

#ifdef MEASUREMENT
static void timespec_diff(struct timespec *start, struct timespec *stop,
                struct timespec *result)
{
        if ((stop->tv_nsec - start->tv_nsec) < 0) {
                result->tv_sec = stop->tv_sec - start->tv_sec - 1;
                result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
        } else {
                result->tv_sec = stop->tv_sec - start->tv_sec;
                result->tv_nsec = stop->tv_nsec - start->tv_nsec;
        }
        return;
}
#endif

static std::shared_ptr<video_frame> cineform_compress_pop(void *state)
{
        auto *s = (struct state_video_compress_cineform *) state;

        std::unique_lock<std::mutex> lock(s->mutex);

        if(s->stop){
                CFHD_SampleBufferRef buf;
                uint32_t frame_num;
                while(CFHD_TestForSample(s->encoderPoolRef, &frame_num, &buf) == CFHD_ERROR_OKAY){
                        CFHD_ReleaseSampleBuffer(s->encoderPoolRef, buf);
                }
                return {};
        }

        const auto& started = s->started;
        while(!started)
                s->cv.wait(lock, [&started](){return started;});

        lock.unlock();

        CFHD_Error status = CFHD_ERROR_OKAY;
        uint32_t frame_num;
        CFHD_SampleBufferRef buf;
        status = CFHD_WaitForSample(s->encoderPoolRef,
                        &frame_num,
                        &buf);
#if MEASUREMENT
        struct timespec t_0, t_1, t_res;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_1);
        t_0 = s->times_map[frame_num];

        timespec_diff(&t_0, &t_1, &t_res);

        printf("[cineform] Encoding %u frame took %f milliseconds.\n", frame_num, t_res.tv_nsec / 1000000.0);
#endif
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to wait for sample %d\n", status);
                return {};
        }

        static auto dispose = [](struct video_frame *frame) {
                std::tuple<CFHD_EncoderPoolRef, CFHD_SampleBufferRef> *t = 
                        static_cast<std::tuple<CFHD_EncoderPoolRef, CFHD_SampleBufferRef> *>(frame->callbacks.dispose_udata);
                if(t)
                        CFHD_ReleaseSampleBuffer(std::get<0>(*t), std::get<1>(*t));
                vf_free(frame);
        };

        std::shared_ptr<video_frame> out = std::shared_ptr<video_frame>(vf_alloc_desc(s->compressed_desc), dispose);
        size_t encoded_len;
        status = CFHD_GetEncodedSample(buf,
                        (void **) &out->tiles[0].data,
                        &encoded_len);
        out->tiles[0].data_len = encoded_len;
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to get sample data\n");
                return {};
        }
        out->callbacks.dispose_udata = new std::tuple<CFHD_EncoderPoolRef, CFHD_SampleBufferRef>(s->encoderPoolRef, buf);
        out->seq = frame_num;

        lock.lock();
        if(s->frame_queue.empty()){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to pop\n");
        } else {
                auto &src = s->frame_queue.front();
                vf_copy_metadata(out.get(), src.get());
        }
        s->frame_queue.pop();

        out->compress_end = get_time_in_ns();

        return out;
}

static compress_module_info get_cineform_module_info(){
        compress_module_info module_info;
        module_info.name = "cineform";

        for(const auto& opt : usage_opts){
                module_info.opts.emplace_back(module_option{opt.label,
                                opt.description, opt.placeholder, opt.key, opt.opt_str, false});
        }

        codec codec_info;
        codec_info.name = "Cineform";
        codec_info.priority = 400;
        codec_info.encoders.emplace_back(encoder{"default", ""});

        module_info.codecs.emplace_back(std::move(codec_info));

        return module_info;
}

const struct video_compress_info cineform_info = {
        cineform_compress_init,
        cineform_compress_done,
        NULL,
        NULL,
        NULL,
        NULL,
        cineform_compress_push,
        cineform_compress_pop,
        get_cineform_module_info,
};

REGISTER_MODULE(cineform, &cineform_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

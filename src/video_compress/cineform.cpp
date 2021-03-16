/**
 * @file   video_compress/cineform.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 */
/* Copyright (c) 2019 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "compat/platform_time.h"
#include "debug.h"
#include "host.h"
#include "video_compress.h"
#include "module.h"
#include "lib_common.h"

#include "CFHDTypes.h"
#include "CFHDEncoder.h"

#include "video.h"
#include "video_codec.h"
#include <memory>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>

#ifdef MEASUREMENT
#include <time.h>
#endif
#include "utils/misc.h" // to_fourcc

#define DEFAULT_POOL_SIZE 16
#define DEFAULT_THREAD_COUNT 8

struct state_video_compress_cineform{
        struct module module_data;

        std::mutex mutex;

        struct video_desc saved_desc;
        struct video_desc precompress_desc;
        struct video_desc compressed_desc;

        void (*convertFunc)(video_frame *dst, video_frame *src);

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

static void cineform_compress_done(struct module *mod){
        struct state_video_compress_cineform *s = (struct state_video_compress_cineform *) mod->priv_data;

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
} usage_opts[] = {
        {"Quality", "quality", "specifies encode quality, range 1-6 (default: 4)", ":quality="},
        {"Threads", "num_threads", "specifies number of threads for encoding", ":num_threads="},
        {"Pool size", "pool_size", "specifies the size of encoding pool", ":pool_size="},
};

static void usage() {
        printf("Cineform encoder usage:\n");
        printf("\t-c cineform[:quality=<quality>][:threads=<num_threads>][:pool_size=<pool_size>]\n");

        printf("\t\t<quality> specifies encode quality, range 1-6 (default: 4)\n");
        printf("\t\t<num_threads> specifies number of threads for encoding \n");
        printf("\t\t<pool_size> specifies the size of encoding pool \n");

        for(const auto& opt : usage_opts){
                printf("\t\t<%s> %s\n", opt.key, opt.description);
        }
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

static struct module * cineform_compress_init(struct module *parent, const char *opts)
{
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
                if(ret > 0)
                        return &compress_init_noerr;
                else
                        return nullptr;
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

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = cineform_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;
}

static void I420toUYVY(video_frame *dst, video_frame *src)
{
        size_t y_len = src->tiles[0].width * src->tiles[0].height;
        size_t chroma_len = ((src->tiles[0].width + 1) / 2) * ((src->tiles[0].height + 1) / 2);
        char *y = src->tiles[0].data;
        char *out = dst->tiles[0].data;
        size_t chroma_pitch = ((dst->tiles[0].width + 1) & ~1) / 2; // handle also odd line widths
        for (size_t i = 0; i < dst->tiles[0].height; i += 1) {
                char *u = src->tiles[0].data + y_len + i / 2 * chroma_pitch;
                char *v = src->tiles[0].data + y_len + chroma_len + i / 2 * chroma_pitch;
                size_t j;
                for (j = 0; j <= dst->tiles[0].width - 2; j += 2) {
                        *out++ = *u++;
                        *out++ = *y++;
                        *out++ = *v++;
                        *out++ = *y++;
                }
                if (j < dst->tiles[0].width) { // last pixel on line with odd width
                        *out++ = *u++;
                        *out++ = *y++;
                        *out++ = *v++;
                        *out++ = 0;
                }
        }
}

static void RGBtoBGR_invert(video_frame *dst, video_frame *src){
        int pitch = vc_get_linesize(src->tiles[0].width, src->color_spec);

        unsigned char *s = (unsigned char *) src->tiles[0].data + (src->tiles[0].height - 1) * pitch;
        unsigned char *d = (unsigned char *) dst->tiles[0].data;

        for(unsigned i = 0; i < src->tiles[0].height; i++){
                vc_copylineBGRtoRGB(d, s, pitch, 0, 8, 16);
                s -= pitch;
                d += pitch;
        }
}

static void RGBAtoBGRA(video_frame *dst, video_frame *src){
        int pitch = vc_get_linesize(src->tiles[0].width, src->color_spec);

        unsigned char *s = (unsigned char *) src->tiles[0].data;
        unsigned char *d = (unsigned char *) dst->tiles[0].data;

        for(unsigned i = 0; i < src->tiles[0].height; i++){
                vc_copylineRGBA(d, s, pitch, 16, 8, 0);
                s += pitch;
                d += pitch;
        }
}

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
        CFHD_PixelFormat cfhd_pixel_format;
        CFHD_EncodedFormat cfhd_encoded_format;
        codec_t convert_codec;
        void (*convertFunc)(video_frame *dst, video_frame *src);
} codecs[] = {
        {I420, CFHD_PIXEL_FORMAT_2VUY, CFHD_ENCODED_FORMAT_YUV_422, UYVY, I420toUYVY},
        {UYVY, CFHD_PIXEL_FORMAT_2VUY, CFHD_ENCODED_FORMAT_YUV_422, VIDEO_CODEC_NONE, nullptr},
        {RGB, CFHD_PIXEL_FORMAT_RG24, CFHD_ENCODED_FORMAT_RGB_444, VIDEO_CODEC_NONE, RGBtoBGR_invert},
        {RGBA, CFHD_PIXEL_FORMAT_BGRa, CFHD_ENCODED_FORMAT_RGBA_4444, VIDEO_CODEC_NONE, RGBAtoBGRA},
        {v210, CFHD_PIXEL_FORMAT_V210, CFHD_ENCODED_FORMAT_YUV_422, VIDEO_CODEC_NONE, nullptr},
        {R10k, CFHD_PIXEL_FORMAT_DPX0, CFHD_ENCODED_FORMAT_RGB_444, VIDEO_CODEC_NONE, nullptr},
        {R12L, CFHD_PIXEL_FORMAT_RG48, CFHD_ENCODED_FORMAT_RGB_444, RG48, R12L_to_RG48},
};

static bool configure_with(struct state_video_compress_cineform *s, struct video_desc desc)
{
        std::lock_guard<std::mutex> lock(s->mutex);

        CFHD_Error status = CFHD_ERROR_OKAY;

        if(s->started){
                status = CFHD_StopEncoderPool(s->encoderPoolRef);
                if(status != CFHD_ERROR_OKAY){
                        log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to stop encoder pool\n");
                        return false;
                }
        }

        CFHD_PixelFormat pix_fmt = CFHD_PIXEL_FORMAT_UNKNOWN;
        CFHD_EncodedFormat enc_fmt = CFHD_ENCODED_FORMAT_UNKNOWN;
        for(const auto &codec : codecs){
                if(codec.ug_codec == desc.color_spec){
                        pix_fmt = codec.cfhd_pixel_format;
                        enc_fmt = codec.cfhd_encoded_format;
                        s->convertFunc = codec.convertFunc;
                        s->precompress_desc = desc;
                        if(codec.convert_codec != VIDEO_CODEC_NONE){
                                s->precompress_desc.color_spec = codec.convert_codec;
                        }
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
        if(!s->convertFunc){
                video_frame *ret = vf_get_copy(frame);
                vf_copy_metadata(ret, frame); // seq and compress_start
                return ret;
        }

        video_frame *ret = vf_alloc_desc_data(s->precompress_desc);
        s->convertFunc(ret, frame);
        vf_copy_metadata(ret, frame); // seq and compress_start

        return ret;
}

static void cineform_compress_push(struct module *state, std::shared_ptr<video_frame> tx)
{
        struct state_video_compress_cineform *s = (struct state_video_compress_cineform *) state->priv_data;

        CFHD_Error status = CFHD_ERROR_OKAY;
        std::unique_lock<std::mutex> lock(s->mutex, std::defer_lock);

        if(s->stop)
                return;

        if(!tx){
                if (!s->started) {
                        video_desc desc{1920, 1080, UYVY, 25.0, PROGRESSIVE, 1};
                        configure_with(s, desc);
                }
                std::unique_ptr<video_frame, decltype(&vf_free)> dummy(vf_alloc_desc_data(s->precompress_desc), vf_free);
                video_frame *dummy_ptr = dummy.get();

                lock.lock();
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
                                s->saved_desc, PARAM_TILE_COUNT)) {
                //cleanup(s);
                int ret = configure_with(s, video_desc_from_frame(tx.get()));
                if(!ret) {
                        return;
                }
        }

        std::unique_ptr<video_frame, decltype(&vf_free)> frame_copy(get_copy(s, tx.get()), vf_free);

        video_frame *frame_ptr = frame_copy.get();

        lock.lock();
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

static std::shared_ptr<video_frame> cineform_compress_pop(struct module *state)
{
        struct state_video_compress_cineform *s = (struct state_video_compress_cineform *) state->priv_data;

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

        printf("[cineform] Encoding %u frame took %f miliseconds.\n", frame_num, t_res.tv_nsec / 1000000.0);
#endif
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to wait for sample %d\n", status);
                return {};
        }

        auto dispose = [](struct video_frame *frame) {
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

        out->compress_end = time_since_epoch_in_ms();

        return out;
}


static std::list<compress_preset> get_cineform_presets() {
        std::list<compress_preset> ret;
        return ret;
}

static compress_module_info get_cineform_module_info(){
        compress_module_info module_info;
        module_info.name = "cineform";

        for(const auto& opt : usage_opts){
                module_info.opts.emplace_back(module_option{opt.label,
                                opt.description, opt.key, opt.opt_str, false});
        }

        codec codec_info;
        codec_info.name = "Cineform";
        codec_info.priority = 400;
        codec_info.encoders.emplace_back(encoder{"default", ""});

        module_info.codecs.emplace_back(std::move(codec_info));

        return module_info;
}

const struct video_compress_info cineform_info = {
        "cineform",
        cineform_compress_init,
        NULL,
        NULL,
        NULL,
        NULL,
        cineform_compress_push,
        cineform_compress_pop,
        get_cineform_presets,
        get_cineform_module_info,
};

REGISTER_MODULE(cineform, &cineform_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

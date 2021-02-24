/**
 * @file   video_capture/gpustitch.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 *         Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief Gpustitch 360 video stitcher
 */
/*
 * Copyright (c) 2021 CESNET z.s.p.o.
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
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <limits>
#include <thread>
#include <condition_variable>
#include <libgpustitch/stitcher.hpp>
#include <libgpustitch/config_utils.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/cuda_pix_conv.h"
#include "utils/profile_timer.hpp"

#define PI 3.14159265

const static char *log_str = "[gpustitch] ";

/* prototypes of functions defined in this module */
static void show_help(void);

static double toRadian(double deg){
        return (deg / 180) * 3.14159265;
}

static void show_help()
{
        printf("Vrworks 360 video stitcher\n");
        printf("Usage\n");
        printf("\t-t gpustitch -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in stitching\n");

}

struct grab_worker_state;

struct vidcap_gpustitch_state {
        int                 devices_cnt;

        struct video_frame      **captured_frames;
        struct video_frame       *frame; 
        int frames;
        struct       timeval t, t0;

        int          audio_source_index;


        gpustitch::Stitcher stitcher;
        gpustitch::Stitcher_params stitch_params;
        std::vector<gpustitch::Cam_params> cam_properties;

        std::string spec_path;
        double fps;
        codec_t out_fmt;
        bool tiled_capture = false;
        bool output_cuda_buf = false;

        unsigned char *conv_tmp_frame = nullptr;
        void (*conv_func)(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);

        std::mutex stitched_mut;
        unsigned stitched_count = 0;
        bool done = false;
        std::condition_variable stitched_cv;

        std::vector<grab_worker_state> capture_workers;
};

struct grab_worker_state {
        std::thread thread;
        std::mutex grabbed_mut;
        unsigned grabbed_count = 0;
        double grabbed_fps = -1;
        std::condition_variable grabbed_cv;

        vidcap_gpustitch_state *s;

        bool tiled = false;

        unsigned width = 0;
        unsigned height = 0;
        unsigned char *tmp_in_frame;
        unsigned char *tmp_rgba_frame;
        unsigned tmp_rgba_frame_pitch;
        void (*conv_func)(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);
        codec_t in_codec = VIDEO_CODEC_NONE;

        cudaStream_t tmp_in_frame_stream;
};

static struct vidcap_type *
vidcap_gpustitch_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        *deleter = free;
        struct vidcap_type* vt;
    
        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name        = "gpustitch";
                vt->description = "gpustitch video capture";
        }
        return vt;
}

static bool check_in_format(grab_worker_state *gs, video_frame *in, int i){
        if(in->tile_count != 1){
                std::cerr << log_str << "Only frames with tile_count == 1 are supported" << std::endl;
                return false;
        }

        unsigned int expected_w = gs->width;
        unsigned int expected_h = gs->height;

        if(in->tiles[0].width != expected_w
                        || in->tiles[0].height != expected_h)
        {
                std::cerr << log_str << "Wrong resolution for input " << i << "!"
                       << " Expected " << expected_w << "x" << expected_h
                       << ", but got " << in->tiles[0].width << "x" << in->tiles[0].height << std::endl;
                return false;
        }

        if(gs->in_codec != in->color_spec){
                printf("New codec detected! RECONF\n");
                assert( in->color_spec == RGB
                                || in->color_spec == RGBA
                                || in->color_spec == UYVY
                      );
                gs->in_codec = in->color_spec;
                cudaFree(gs->tmp_in_frame);
                gs->tmp_in_frame = nullptr;

                int in_line_size = vc_get_linesize(expected_w, in->color_spec);
                cudaMalloc(&gs->tmp_in_frame, in_line_size * expected_h);
                switch(in->color_spec){
                        case RGB:
                                gs->conv_func = cuda_RGB_to_RGBA;
                                break;
                        case UYVY:
                                gs->conv_func = cuda_UYVY_to_RGBA;
                                break;
                        default:
                                gs->conv_func = nullptr;
                }
        }

        return true;
}

static bool upload_to_cuda_buf(grab_worker_state *gs, video_frame *in_frame,
                size_t x_offset, size_t y_offset, size_t w, size_t h,
                unsigned char *dst,
                unsigned dst_pitch,
                cudaStream_t stream)
{
        unsigned int width = in_frame->tiles[0].width;
        unsigned int height = in_frame->tiles[0].height;
        int in_line_size = vc_get_linesize(width, in_frame->color_spec);
        int roi_line_size = vc_get_linesize(w, in_frame->color_spec);

        size_t offset_bytes = y_offset * in_line_size
                + vc_get_linesize(x_offset, in_frame->color_spec); 

        w = (w / get_pf_block_size(in_frame->color_spec)) * get_pf_block_size(in_frame->color_spec);

        if(gs->conv_func){
                if (cudaMemcpy2DAsync(gs->tmp_in_frame + offset_bytes,
                                        in_line_size,
                                        (unsigned char *)(in_frame->tiles[0].data) + offset_bytes,
                                        in_line_size,
                                        roi_line_size, h,
                                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
                {
                        std::cerr << "Error copying input image bitmap to CUDA buffer" << std::endl;
                        return false;
                }

                gs->conv_func(dst + y_offset * dst_pitch + vc_get_linesize(x_offset, RGBA),
                                dst_pitch,
                                gs->tmp_in_frame + offset_bytes, in_line_size,
                                w, h, stream);
        } else {
                if (cudaMemcpy2D(dst + y_offset * dst_pitch + vc_get_linesize(x_offset, RGBA), dst_pitch,
                                        (unsigned char*)(in_frame->tiles[0].data) + offset_bytes,
                                        in_line_size,
                                        w, h,
                                        cudaMemcpyHostToDevice) != cudaSuccess)
                {
                        std::cerr << "Error copying RGBA image bitmap to CUDA buffer" << std::endl;
                        return false;

                }
        }

        return true;
}

static bool upload_frame(grab_worker_state *gs, video_frame *in_frame, int i){
        cudaStream_t stream;
        gs->s->stitcher.get_input_stream(i, &stream);

        if(!stream){
                std::cerr << std::endl << "Failed to get input stream." << std::endl;
                return false;
        }

        upload_to_cuda_buf(gs,
                        in_frame,
                        0, 0, gs->width, gs->height,
                        gs->tmp_rgba_frame,
                        gs->tmp_rgba_frame_pitch,
                        stream);

        gs->s->stitcher.submit_input_image_async(i, gs->tmp_rgba_frame,
                        gs->width, gs->height,
                        gs->tmp_rgba_frame_pitch,
                        gpustitch::Src_mem_kind::Device);

        return true;
}

static void upload_tiles(grab_worker_state *gs, video_frame *frame){
        const int order[] = {3, 1, 2, 0};

        for(int i = 0; i < 4; i++){
                cudaStream_t stream;

                gs->s->stitcher.get_input_stream(i, &stream);

                unsigned tile_height = gs->height / 2;
                unsigned tile_width = gs->width / 2;
                unsigned src_pitch = gs->tmp_rgba_frame_pitch;
                unsigned tile_line_len = gs->tmp_rgba_frame_pitch / 2;
                unsigned x_offset = (order[i] % 2) * tile_width;
                unsigned y_offset = (order[i] / 2) * tile_height;

                upload_to_cuda_buf(gs,
                                frame,
                                x_offset, y_offset, tile_width, tile_height,
                                gs->tmp_rgba_frame,
                                gs->tmp_rgba_frame_pitch,
                                stream
                                );

                unsigned char *src = static_cast<unsigned char *>(gs->tmp_rgba_frame);
                src += (order[i] % 2) * tile_line_len;
                src += (order[i] / 2) * src_pitch * tile_height;

                gs->s->stitcher.submit_input_image_async(i, src,
                                gs->width / 2, gs->height / 2,
                                gs->tmp_rgba_frame_pitch,
                                gpustitch::Src_mem_kind::Device);

        }
}

static void grab_worker(grab_worker_state *gs, vidcap_params *param, int id){
        PROFILE_FUNC;
        struct audio_frame *audio_frame = NULL;
        struct video_frame *frame = NULL;
        std::unique_lock<std::mutex> stitch_lk(gs->s->stitched_mut, std::defer_lock);
        std::unique_lock<std::mutex> grab_lk(gs->grabbed_mut, std::defer_lock);

        gs->width = gs->s->cam_properties[id].width;
        gs->height = gs->s->cam_properties[id].height;
        if(gs->tiled){
                gs->width *= 2;
                gs->height *= 2;
        }

        int in_line_size = vc_get_linesize(gs->width, RGBA);
        gs->tmp_rgba_frame_pitch = in_line_size;
        cudaMalloc(&gs->tmp_rgba_frame, in_line_size * gs->height);

        vidcap *device = nullptr;

        int ret = initialize_video_capture(NULL, param, &device);
        if(ret != 0) {
                fprintf(stderr, "[gpustitch] Unable to initialize device %d (%s:%s).\n",
                                id, vidcap_params_get_driver(param),
                                vidcap_params_get_fmt(param));
                return;
        }

        cudaStreamCreate(&gs->tmp_in_frame_stream);

        while(true){
                stitch_lk.lock();
                PROFILE_DETAIL("Wait for cv");
                gs->s->stitched_cv.wait(stitch_lk,
                                [gs]{return gs->s->stitched_count == gs->grabbed_count || gs->s->done;});
                if(gs->s->done){
                        VIDEO_FRAME_DISPOSE(frame);
                        vidcap_done(device);
                        cudaStreamDestroy(gs->tmp_in_frame_stream);
                        return;
                }
                stitch_lk.unlock();

                VIDEO_FRAME_DISPOSE(frame);

                frame = nullptr;
                PROFILE_DETAIL("Grab frame");
                while(!frame){
                        frame = vidcap_grab(device, &audio_frame);
                }

                if (check_in_format(gs, frame, id)){
                        PROFILE_DETAIL("Upload");
                        if(gs->tiled){
                                upload_tiles(gs, frame);

                        } else {
                                upload_frame(gs, frame, id);
                        }
                }

                grab_lk.lock();
                gs->grabbed_count += 1;
                if(frame)
                        gs->grabbed_fps = frame->fps;
                grab_lk.unlock();
                gs->grabbed_cv.notify_one();
        }
}

static bool init_stitcher(struct vidcap_gpustitch_state *s){
        switch(s->out_fmt){
                case RGBA:
                        s->conv_func = nullptr;
                        break;
                case RGB:
                        s->conv_func = cuda_RGBA_to_RGB;
                        break;
                case UYVY:
                        s->conv_func = cuda_RGBA_to_UYVY;
                        break;
                default:
                        assert(false && "Unsupported output pixel format!");
        }

        int num_gpus;
        cudaGetDeviceCount(&num_gpus);

        int selected_gpu;

        for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, gpu);

                // Require minimum compute 5.2
                if (prop.major > 5 || (prop.major == 5 && prop.minor >= 2))
                {
                        selected_gpu = gpu;
                        break;
                }
        }

        gpustitch::read_params(s->spec_path, s->stitch_params, s->cam_properties);

        if(s->cam_properties.size() != s->devices_cnt && !(s->tiled_capture && s->devices_cnt == 1)){
                log_msg(LOG_LEVEL_ERROR, "Number of capture devices is different from specified number of cameras!\n");
                if(s->cam_properties.size() < s->devices_cnt)
                        return false;
        }

        try{
                s->stitcher = gpustitch::Stitcher(s->stitch_params, s->cam_properties);
        } catch(char const *exc){
                log_msg(LOG_LEVEL_ERROR, "Exception while initializing stitcher: %s\n", exc);
                return false;
        }

        return true;
}

static void parse_fmt(vidcap_gpustitch_state *s, const char * const fmt){
        s->fps = -1;
        s->out_fmt = RGBA;

        s->stitch_params.width = 3840;
        s->stitch_params.height = s->stitch_params.width / 2;
        s->stitch_params.blend_algorithm = gpustitch::Blend_algorithm::Multiband;
        s->stitch_params.feather_width = 30;
        s->stitch_params.multiband_levels = 4;


        if(!fmt)
                return;

        char *tmp = strdup(fmt);
        char *init_fmt = tmp;
        char *save_ptr = NULL;
        char *item;
#define FMT_CMP(param) (strncmp(item, (param), strlen((param))) == 0)
        while((item = strtok_r(init_fmt, ":", &save_ptr))) {
                if (FMT_CMP("width=")) {
                        s->stitch_params.width = atoi(strchr(item, '=') + 1);
                        s->stitch_params.height = s->stitch_params.width / 2;
                } else if(FMT_CMP("blend_algo=")){
                        struct {
                                const char *name;
                                gpustitch::Blend_algorithm algo;
                        } blend_algos[] = {
                                {"multiband", gpustitch::Blend_algorithm::Multiband},
                                {"feather", gpustitch::Blend_algorithm::Feather},
                        };

                        const char *req = strchr(item, '=') + 1;

                        bool selected = false;
                        for(const auto& i : blend_algos){
                                if(strcmp(req, i.name) == 0){
                                        s->stitch_params.blend_algorithm = i.algo;
                                        selected = true;
                                        break;
                                }
                        }

                        if(!selected){
                                log_msg(LOG_LEVEL_WARNING, "Invalid blend algorithm, falling back to default\n");
                        }
                } else if(FMT_CMP("feather_width=")){
                        s->stitch_params.feather_width = atoi(strchr(item, '=') + 1);
                } else if(FMT_CMP("multiband_levels=")){
                        s->stitch_params.multiband_levels = atoi(strchr(item, '=') + 1);
                } else if(FMT_CMP("rig_spec=")){
                        s->spec_path = strchr(item, '=') + 1;
                } else if(FMT_CMP("fps=")){
                        s->fps = strtod(strchr(item, '=') + 1, nullptr);
                } else if(FMT_CMP("fmt=")){
                        s->out_fmt = get_codec_from_name(strchr(item, '=') + 1);
                } else if(FMT_CMP("tiled")){
                        s->tiled_capture = true;
                } else if(FMT_CMP("cudabuf")){
                        s->output_cuda_buf = true;
                }
                init_fmt = NULL;
        }
        free(tmp);
}

static void stop_grab_workers(vidcap_gpustitch_state *s){
        std::unique_lock<std::mutex> lk(s->stitched_mut);
        s->done = true;
        lk.unlock();

        s->stitched_cv.notify_all();

        for(auto& worker : s->capture_workers){
                if(worker.thread.joinable())
                        worker.thread.join();
        }
}

static int
vidcap_gpustitch_init(struct vidcap_params *params, void **state)
{
        struct vidcap_gpustitch_state *s;

        printf("vidcap_gpustitch_init\n");


        s = new vidcap_gpustitch_state();
        if(s == NULL) {
                printf("Unable to allocate gpustitch capture state\n");
                return VIDCAP_INIT_FAIL;
        }

        s->audio_source_index = -1;
        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "help") == 0) {
               show_help(); 
               free(s);
               return VIDCAP_INIT_NOERR;
        }

        parse_fmt(s, vidcap_params_get_fmt(params));

        s->devices_cnt = 0;
        struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
                if (vidcap_params_get_driver(tmp) != NULL)
                        s->devices_cnt++;
                else
                        break;
        }

        if(s->tiled_capture && s->devices_cnt != 1){
                log_msg(LOG_LEVEL_ERROR, "Number of capture devices must be exactly 1 when using tiled capture!\n");
                return VIDCAP_INIT_FAIL;
        }

        s->captured_frames = (struct video_frame **) calloc(s->devices_cnt, sizeof(struct video_frame *));

        s->frame = nullptr;

        if(!init_stitcher(s)){
                goto error;
        }

        s->capture_workers = std::vector<grab_worker_state>(s->devices_cnt);
        tmp = params;
        for (int i = 0; i < s->devices_cnt; ++i) {
                tmp = vidcap_params_get_next(tmp);

                s->capture_workers[i].s = s;
                s->capture_workers[i].tiled = s->tiled_capture;
                s->capture_workers[i].thread = 
                        std::thread(grab_worker, &s->capture_workers[i], tmp, i);
        }



        *state = s;
        return VIDCAP_INIT_OK;

error:
        stop_grab_workers(s);
        free(s);
        return VIDCAP_INIT_FAIL;
}

static void
vidcap_gpustitch_done(void *state)
{
        struct vidcap_gpustitch_state *s = (struct vidcap_gpustitch_state *) state;

        if(!s) return;

        stop_grab_workers(s);

        free(s->captured_frames);
        vf_free(s->frame);
        cudaFree(s->conv_tmp_frame);

        delete s;
}

static void result_data_delete(struct video_frame *buf){
        if(!buf)
                return;

        cudaFreeHost(buf->tiles[0].data);
}

static bool allocate_result_frame(vidcap_gpustitch_state *s, unsigned width, unsigned height){
        video_desc desc{};
        desc.width = width;
        desc.height = height;
        desc.color_spec = s->out_fmt;
        desc.tile_count = 1;
        desc.fps = s->fps;

        s->frame = vf_alloc_desc(desc);
        s->frame->tiles[0].data_len = vc_get_linesize(desc.width,
                        desc.color_spec) * desc.height;

        s->frame->callbacks.data_deleter = NULL;
        s->frame->callbacks.recycle = NULL;

        if(!s->output_cuda_buf){
                if(cudaMallocHost(&s->frame->tiles[0].data, s->frame->tiles[0].data_len) != cudaSuccess){
                        std::cerr << log_str << "Failed to allocate result frame" << std::endl;
                        return false;
                }
                s->frame->callbacks.data_deleter = result_data_delete;
        }

        if(s->conv_func){
                size_t size = vc_get_linesize(desc.width, s->out_fmt) * desc.height;

                if(cudaMalloc(&s->conv_tmp_frame, size) != cudaSuccess){
                        std::cerr << log_str << "Failed to allocate tmp conv frame" << std::endl;
                        return false;
                }
        }

        return true;
}

static bool download_stitched(vidcap_gpustitch_state *s, cudaStream_t out_stream){
        PROFILE_FUNC;
        gpustitch::Image_cuda *output_image;

        output_image = s->stitcher.get_output_image();
        if(!output_image){
                std::cerr << log_str << "Failed to get output buffer" << std::endl;
                return false;
        }

        size_t w = output_image->get_width();
        size_t h = output_image->get_height();

        if(!s->frame){
                if(!allocate_result_frame(s, w, h)){
                        return false;
                }
        }

        void *src = output_image->data();
        size_t src_pitch = output_image->get_pitch();
        size_t row_bytes = output_image->get_row_bytes();
        if(s->conv_func){
                s->conv_func(s->conv_tmp_frame, vc_get_linesize(w, s->out_fmt),
                                (unsigned char *) src, src_pitch,
                                w, h,
                                out_stream);
                src = s->conv_tmp_frame;
                src_pitch = vc_get_linesize(w, s->out_fmt);
                row_bytes = src_pitch;
        }

        if(s->output_cuda_buf){
                s->frame->tiles[0].data = (char *) src;
        } else {
                if (cudaMemcpy2DAsync(s->frame->tiles[0].data, row_bytes,
                                        src, src_pitch,
                                        row_bytes, h,
                                        cudaMemcpyDeviceToHost, out_stream) != cudaSuccess)
                {
                        std::cerr << log_str << "Error copying output panorama from CUDA buffer" << std::endl;
                        return false;
                }
        }

        return true;
}

static void report_stats(vidcap_gpustitch_state *s){
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[gpustitch cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }  
}

static double wait_for_frames(struct vidcap_gpustitch_state *s){
        PROFILE_FUNC;
        double fps = std::numeric_limits<double>::max();
        for (auto& worker : s->capture_workers) {
                std::unique_lock<std::mutex> lk(worker.grabbed_mut);
                worker.grabbed_cv.wait(lk, [&]{return worker.grabbed_count > s->stitched_count;});
                if(worker.grabbed_fps != -1 && worker.grabbed_fps < fps)
                        fps = worker.grabbed_fps;
        }

        return fps;
}

static struct video_frame *stitch(struct vidcap_gpustitch_state *s){
        PROFILE_FUNC;
        s->stitcher.stitch();
        if(false){
                std::cout << std::endl << "Failed to stitch." << std::endl;
                return NULL;
        }

        cudaStream_t out_stream;
        s->stitcher.get_output_stream(&out_stream);

        if(!out_stream){
                std::cerr << log_str << "Failed to get output stream" << std::endl;
                return NULL;
        }

        std::unique_lock<std::mutex> stitch_lk(s->stitched_mut);
        s->stitched_count += 1;
        stitch_lk.unlock();
        s->stitched_cv.notify_all();

        if(!download_stitched(s, out_stream)){
                return NULL;
        }

        PROFILE_DETAIL("Synchronize 2");
        if (cudaStreamSynchronize(out_stream) != cudaSuccess)
        {
                std::cerr << "Error synchronizing with the output CUDA stream" << std::endl;
                return NULL;
        }

        return s->frame;
}

static struct video_frame *
vidcap_gpustitch_grab(void *state, struct audio_frame **audio)
{
        PROFILE_FUNC;
        struct vidcap_gpustitch_state *s = (struct vidcap_gpustitch_state *) state;

        video_frame *f = nullptr;
        double fps = wait_for_frames(s);

        f = stitch(s);
        if(!f){
                return nullptr;
        }

        //If s->fps == -1 it means it was not set by user so we use the fps
        //of the slowest grabber
        if(s->fps <= 0){
                f->fps = fps;
        } else {
                f->fps = s->fps;
        }

        s->frames++;
        report_stats(s);

        return f;
}

static const struct video_capture_info vidcap_gpustitch_info = {
        vidcap_gpustitch_probe,
        vidcap_gpustitch_init,
        vidcap_gpustitch_done,
        vidcap_gpustitch_grab,
};

REGISTER_MODULE(gpustitch, &vidcap_gpustitch_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


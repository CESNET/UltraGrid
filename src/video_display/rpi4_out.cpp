/**
 * @file   video_display/rpi4_out.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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
#include "lib_common.h"
#include "rang.hpp"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
#include "hwaccel_rpi4.h"
#include "utils/misc.h"
#include "utils/color_out.h"

#include <memory>
#include <queue>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <type_traits>
#include <chrono>
#include <charconv>

#include <bcm_host.h>
#include <interface/mmal/mmal.h>
#include <interface/mmal/mmal_component.h>
#include <interface/mmal/util/mmal_default_components.h>
#include <interface/mmal/util/mmal_util_params.h>

extern "C" { //needed for rpi_zc.h and rpi_sand_fns.h
#include <libavcodec/avcodec.h> //needed for rpi_zc.h
#include <libavcodec/rpi_zc.h>
#include <libavutil/rpi_sand_fns.h>
}

#define MAX_BUFFER_SIZE 4

namespace{

struct frame_deleter{
        void operator()(struct video_frame *f){ vf_free(f); }
};

using unique_frame = std::unique_ptr<struct video_frame, frame_deleter>;

struct mmal_component_deleter{
        void operator()(MMAL_COMPONENT_T *c){ mmal_component_destroy(c); }
};

using mmal_component_unique = std::unique_ptr<MMAL_COMPONENT_T, mmal_component_deleter>;

struct mmal_pool_deleter{
        void operator()(MMAL_POOL_T *p){ mmal_pool_destroy(p); }
};

using mmal_pool_unique = std::unique_ptr<MMAL_POOL_T, mmal_pool_deleter>;

struct av_zc_deleter{
        void operator()(AVZcEnvPtr env) { av_rpi_zc_int_env_freep(&env); }
};

using av_zc_env_unique = std::unique_ptr<std::remove_pointer_t<AVZcEnvPtr>, av_zc_deleter>;

struct av_zc_frame_deleter{
        void operator()(AVRpiZcRefPtr f){ av_rpi_zc_unref(f); }
};

using av_zero_copy_frame_unique = std::unique_ptr<std::remove_pointer_t<AVRpiZcRefPtr>, av_zc_frame_deleter>;

struct mmal_buf_header_deleter{
        void operator()(MMAL_BUFFER_HEADER_T *buf){ mmal_buffer_header_release(buf); }
};

using mmal_buf_header_unique = std::unique_ptr<MMAL_BUFFER_HEADER_T, mmal_buf_header_deleter>;

class Rpi4_video_out{
public:
        Rpi4_video_out() = default;
        Rpi4_video_out(int x, int y, int width, int height, bool fs, int layer);

        void display(AVFrame *f);

        void resize(int width, int height);
        void move(){
                const int max_x = 1920 - out_width;
                const int max_y = 1080 - out_height;

                out_pos_x += x_dir;
                out_pos_y += y_dir;

                if(out_pos_x >= max_x){
                        out_pos_x = max_x;
                        x_dir = -x_dir;
                }

                if(out_pos_x <= 0){
                        out_pos_x = 0;
                        x_dir = -x_dir;
                }

                if(out_pos_y >= max_y){
                        out_pos_y = max_y;
                        y_dir = -y_dir;
                }

                if(out_pos_y <= 0){
                        out_pos_y = 0;
                        y_dir = -y_dir;
                }

                set_output_params();
        }
private:
        void set_output_params();
        void stream_fmt_from_frame(MMAL_ES_FORMAT_T *fmt, const AVFrame *f, const AVRpiZcRefPtr zc_frame);
        void set_output_format(MMAL_ES_FORMAT_T *fmt);

        int out_pos_x;
        int out_pos_y;
        int out_width;
        int out_height;
        int x_dir = 2;
        int y_dir = 2;
        bool fullscreen;
        int layer;

        MMAL_ES_FORMAT_T curr_stream_format = {};

        mmal_component_unique renderer_component;
        mmal_pool_unique pool;
        av_zc_env_unique zero_copy_env;
};

void Rpi4_video_out::resize(int width, int height){
        if(width == out_width && height == out_height)
                return;

        out_width = width;
        out_height = height;

        set_output_params();
}

void Rpi4_video_out::set_output_params(){
        MMAL_DISPLAYREGION_T region = {};
        region.hdr = {MMAL_PARAMETER_DISPLAYREGION, sizeof(region)};
        region.set = MMAL_DISPLAY_SET_DEST_RECT
                | MMAL_DISPLAY_SET_FULLSCREEN
                | MMAL_DISPLAY_SET_LAYER
                | MMAL_DISPLAY_SET_ALPHA;
        region.dest_rect = {out_pos_x, out_pos_y, out_width, out_height};
        region.fullscreen = fullscreen;
        region.layer = layer;
        region.alpha = 0xff;

        mmal_port_parameter_set(renderer_component->input[0], &region.hdr);
}

static void release_buf_cb(MMAL_PORT_T *, MMAL_BUFFER_HEADER_T *buf){
        mmal_buffer_header_release(buf);
}

Rpi4_video_out::Rpi4_video_out(int x, int y, int width, int height, bool fs, int layer):
        out_pos_x(x),
        out_pos_y(y),
        out_width(width),
        out_height(height),
        fullscreen(fs),
        layer(layer)
{
        bcm_host_init();

        MMAL_COMPONENT_T *c = nullptr;
        if(mmal_component_create(MMAL_COMPONENT_DEFAULT_VIDEO_RENDERER, &c) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to create renderer component");
        }
        renderer_component.reset(c);

        set_output_params();


        if(mmal_component_enable(renderer_component.get()) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to enable renderer component");
        }

        if(mmal_port_enable(renderer_component->control, release_buf_cb) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to enable control port");
        }


        pool.reset(mmal_pool_create(MAX_BUFFER_SIZE, 0));
        if(!pool){
                throw std::runtime_error("Failed to create pool");
        }

        zero_copy_env.reset(av_rpi_zc_int_env_alloc(nullptr));
}

void Rpi4_video_out::stream_fmt_from_frame(MMAL_ES_FORMAT_T *stream_fmt, const AVFrame *f, const AVRpiZcRefPtr zc_frame){
        MMAL_VIDEO_FORMAT_T *v_fmt = &stream_fmt->es->video;
        const AVRpiZcFrameGeometry *geo = av_rpi_zc_geometry(zc_frame);

        stream_fmt->flags = 0;

        if(av_rpi_is_sand_format(geo->format)){
                v_fmt->width = geo->height_y;
                v_fmt->height = geo->height_y;

                if(geo->stripe_is_yc)
                        v_fmt->width += geo->height_c;

                stream_fmt->flags |= MMAL_ES_FORMAT_FLAG_COL_FMTS_WIDTH_IS_COL_STRIDE;
        } else {
                v_fmt->width = geo->stride_y / geo->bytes_per_pel;
                v_fmt->height = geo->height_y;
        }

        stream_fmt->type = MMAL_ES_TYPE_VIDEO;
        assert(geo->format == AV_PIX_FMT_RPI4_8
                        || geo->format == AV_PIX_FMT_SAND128);
        stream_fmt->encoding = MMAL_ENCODING_YUVUV128;

        v_fmt->crop.x = f->crop_left;
        v_fmt->crop.y = f->crop_top;
        v_fmt->crop.width = av_frame_cropped_width(f);
        v_fmt->crop.height = av_frame_cropped_height(f);

        v_fmt->frame_rate.den = 30;
        v_fmt->frame_rate.num = 1;

        v_fmt->par.den = f->sample_aspect_ratio.den;
        v_fmt->par.num = f->sample_aspect_ratio.num;
}

void Rpi4_video_out::set_output_format(MMAL_ES_FORMAT_T *fmt){
        if(mmal_format_compare(fmt, &curr_stream_format) == 0)
                return;

        mmal_format_copy(renderer_component->input[0]->format, fmt);

        if(mmal_port_format_commit(renderer_component->input[0]) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to commit port format");
        }
}

static MMAL_BOOL_T buf_pre_release_cb(MMAL_BUFFER_HEADER_T * buf, void *){
        if(buf->user_data){
                av_zc_frame_deleter()(static_cast<AVRpiZcRefPtr>(buf->user_data));
                buf->user_data = nullptr;
        }

        return MMAL_FALSE;
}

void Rpi4_video_out::display(AVFrame *f){
        auto zc_frame = av_zero_copy_frame_unique(
                        av_rpi_zc_ref(nullptr,
                                zero_copy_env.get(),
                                f,
                                static_cast<AVPixelFormat>(f->format),
                                true)
                        );
        if(!zc_frame)
                return;


        auto buf = mmal_buf_header_unique(mmal_queue_get(pool->queue));
        if(!buf)
                return;

        MMAL_ES_SPECIFIC_FORMAT_T sfmt = {};
        MMAL_ES_FORMAT_T stream_fmt = {};

        stream_fmt.es = &sfmt;

        stream_fmt_from_frame(&stream_fmt, f, zc_frame.get());
        set_output_format(&stream_fmt);

        renderer_component->input[0]->buffer_num = MAX_BUFFER_SIZE;
        renderer_component->input[0]->buffer_size = av_rpi_zc_numbytes(zc_frame.get());

        if(!renderer_component->input[0]->is_enabled){
                mmal_port_parameter_set_boolean(renderer_component->input[0],
                                MMAL_PARAMETER_ZERO_COPY, MMAL_TRUE);

                if(mmal_port_enable(renderer_component->input[0], release_buf_cb) != MMAL_SUCCESS){
                        throw std::runtime_error("Failed to enable port");
                }
        }

        mmal_buffer_header_reset(buf.get());

        buf->data = reinterpret_cast<uint8_t *>(av_rpi_zc_vc_handle(zc_frame.get()));
        buf->length = av_rpi_zc_length(zc_frame.get());
        buf->offset = av_rpi_zc_offset(zc_frame.get());
        buf->alloc_size = av_rpi_zc_numbytes(zc_frame.get());

        mmal_buffer_header_pre_release_cb_set(buf.get(), buf_pre_release_cb, nullptr);
        buf->user_data = zc_frame.get();
        if(mmal_port_send_buffer(renderer_component->input[0], buf.get()) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to send buffer");
        }

        //Frame was successfully submitted, we don't own the data anymore
        zc_frame.release();
        buf.release();
}

} //anonymous namespace

struct rpi4_display_state{
        struct video_desc current_desc;

        std::mutex frame_queue_mut;
        std::condition_variable new_frame_ready_cv;
        std::condition_variable frame_consumed_cv;
        std::queue<unique_frame> frame_queue;

        std::mutex free_frames_mut;
        std::stack<unique_frame> free_frames;

        int requested_pos_x = 0;
        int requested_pos_y = 0;
        int force_w = 0;
        int force_h = 0;
        bool fullscreen = false;

        Rpi4_video_out video_out;
};

static void print_rpi4_out_help(){
        std::cout << "usage:\n";
        std::cout << rang::style::bold << rang::fg::red << "\t-d rpi4" << rang::fg::reset << "[:force-size=<w>x<h>|:position=<x>x<y>|:fs]* | help\n\n" << rang::style::reset;
        std::cout << "options:\n";
        std::cout << BOLD("\tfs")          << "\t\tfullscreen\n";
        std::cout << BOLD("\tforce-size")  << "\t\tspecifies desired size of output\n";
        std::cout << BOLD("\tposition")    << "\t\tspecifies the desired position of output (coordinates of top left corner)\n";
}

static void *display_rpi4_init(struct module *parent, const char *cfg, unsigned int flags)
{
        auto s = std::make_unique<rpi4_display_state>();

        std::string_view conf(cfg);
        while(!conf.empty()){
                auto token = tokenize(conf, ':');

                auto key = tokenize(token, '=');
                if(key == "help"){
                        print_rpi4_out_help();
                        return nullptr;
                } else if(key == "force-size"){
                        auto val = tokenize(token, '=');

                        auto width = tokenize(val, 'x');
                        auto height = tokenize(val, 'x');

                        if(width.empty() || height.empty())
                                return nullptr;

                        if(std::from_chars(width.data(), width.data() + width.size(), s->force_w).ec != std::errc()
                                        || std::from_chars(height.data(), height.data() + height.size(), s->force_h).ec != std::errc())
                        {
                                return nullptr;
                        }
                } else if(key == "fs"){
                        s->fullscreen = true;
                } else if(key == "position"){
                        auto val = tokenize(token, '=');

                        auto x_str = tokenize(val, 'x');
                        auto y_str = tokenize(val, 'x');

                        if(x_str.empty() || y_str.empty())
                                return nullptr;

                        if(std::from_chars(x_str.data(), x_str.data() + x_str.size(), s->requested_pos_x).ec != std::errc()
                                        || std::from_chars(y_str.data(), y_str.data() + y_str.size(), s->requested_pos_y).ec != std::errc())
                        {
                                return nullptr;
                        }
                }
        }

        int width = s->force_w ? s->force_w : 640;
        int height = s->force_h ? s->force_h : 480;

        s->video_out = Rpi4_video_out(s->requested_pos_x, s->requested_pos_y,
                        width, height,
                        s->fullscreen, 2);

        return s.release();
}

static void display_rpi4_done(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        delete s;
}

static void frame_data_deleter(struct video_frame *buf){
        auto wrapper = reinterpret_cast<av_frame_wrapper *>(buf->tiles[0].data);

        av_frame_free(&wrapper->av_frame);

        delete wrapper;
}

static inline void av_frame_wrapper_recycle(struct video_frame *f){
        for(unsigned i = 0; i < f->tile_count; i++){
                av_frame_wrapper *wrapper = (av_frame_wrapper *)(void *) f->tiles[i].data;

                av_frame_unref(wrapper->av_frame);
        }
}

static inline void av_frame_wrapper_copy(struct video_frame *f){
        for(unsigned i = 0; i < f->tile_count; i++){
                av_frame_wrapper *wrapper = (av_frame_wrapper *)(void *) f->tiles[i].data;

                wrapper->av_frame = av_frame_clone(wrapper->av_frame);
        }
}

static struct video_frame *alloc_new_frame(rpi4_display_state *s){
        auto new_frame = vf_alloc_desc(s->current_desc);

        assert(new_frame->tile_count == 1);

        auto wrapper = new av_frame_wrapper();

        wrapper->av_frame = av_frame_alloc();

        new_frame->tiles[0].data_len = sizeof(av_frame_wrapper);
        new_frame->tiles[0].data = reinterpret_cast<char *>(wrapper);
        new_frame->callbacks.recycle = av_frame_wrapper_recycle;
        new_frame->callbacks.copy = av_frame_wrapper_copy;
        new_frame->callbacks.data_deleter = frame_data_deleter;

        return new_frame;
}

static struct video_frame *display_rpi4_getf(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        {//lock
                std::lock_guard lk(s->free_frames_mut);

                while(!s->free_frames.empty()){
                        unique_frame frame = std::move(s->free_frames.top());
                        s->free_frames.pop();
                        if(video_desc_eq(video_desc_from_frame(frame.get()), s->current_desc))
                        {
                                return frame.release();
                        }
                }
        }

        auto new_frame = alloc_new_frame(s);
        return new_frame;
}

static int display_rpi4_putf(void *state, struct video_frame *frame, int flags)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        if(!frame){
                std::unique_lock lk(s->frame_queue_mut);
                s->frame_queue.emplace(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return 0;
        }

        if (flags == PUTF_DISCARD) {
                vf_recycle(frame);
                std::lock_guard(s->free_frames_mut);
                s->free_frames.emplace(frame);
                return 0;
        }

        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && flags == PUTF_NONBLOCK) {
                log_msg(LOG_LEVEL_VERBOSE, "nonblock putf drop\n");
                vf_recycle(frame);
                std::lock_guard(s->free_frames_mut);
                s->free_frames.emplace(frame);
                return 1;
        }

        std::unique_lock lk(s->frame_queue_mut);
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.emplace(frame);
        lk.unlock();
        s->new_frame_ready_cv.notify_one();

        return 0;
}

static void display_rpi4_run(void *state)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        bool run = true;
        while(run){
                std::unique_lock lk(s->frame_queue_mut);
                s->new_frame_ready_cv.wait(lk, [s] {return s->frame_queue.size() > 0;});

                if (s->frame_queue.size() == 0) {
                        continue;
                }

                unique_frame frame = std::move(s->frame_queue.front());
                s->frame_queue.pop();
                lk.unlock();
                s->frame_consumed_cv.notify_one();

                if(!frame){
                        run = false;
                        break;
                }

                auto av_wrap = reinterpret_cast<struct av_frame_wrapper *>(
                                reinterpret_cast<void *>(frame->tiles[0].data));

                s->video_out.display(av_wrap->av_frame);

                vf_recycle(frame.get());
                std::lock_guard(s->free_frames_mut);
                s->free_frames.push(std::move(frame));
        }
}

static int display_rpi4_reconfigure(void *state, struct video_desc desc)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        assert(desc.color_spec == RPI4_8);
        s->current_desc = desc;

        if(s->force_w == 0 && s->force_h == 0)
                s->video_out.resize(desc.width, desc.height);

        return TRUE;
}

static auto display_rpi4_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {
                RPI4_8,
        };
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static void display_rpi4_put_audio_frame(void *, const struct audio_frame *)
{
}

static int display_rpi4_reconfigure_audio(void *, int, int, int)
{
        return false;
}

static void display_rpi4_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
};

static const struct video_display_info display_rpi4_info = {
        display_rpi4_probe,
        display_rpi4_init,
        display_rpi4_run,
        display_rpi4_done,
        display_rpi4_getf,
        display_rpi4_putf,
        display_rpi4_reconfigure,
        display_rpi4_get_property,
        display_rpi4_put_audio_frame,
        display_rpi4_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
        false,
};

REGISTER_MODULE(rpi4, &display_rpi4_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

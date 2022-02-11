/**
 * @file   video_display/conference.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET, z. s. p. o.
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
#include "video.h"
#include "video_display.h"
#include "video_codec.h"
#include "utils/misc.h"

#include <cinttypes>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <string_view>
#include <charconv>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#ifdef __SSSE3__
#include "tmmintrin.h"
#endif

#include "utils/profile_timer.hpp"

#define MOD_NAME "[conference] "

namespace{
using clock = std::chrono::steady_clock;

struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;

struct Participant{
        void to_cv_frame();
        void frame_recieved(unique_frame &&f);
        void set_pos_keep_aspect(int x, int y, int w, int h);

        unique_frame frame;
        clock::time_point last_time_recieved;
        unsigned src_w = 0;
        unsigned src_h = 0;

        unsigned x = 0;
        unsigned y = 0;
        unsigned width = 0;
        unsigned height = 0;

        cv::Mat luma;
        cv::Mat chroma;
};

void Participant::frame_recieved(unique_frame &&f){
        frame = std::move(f);
        last_time_recieved = clock::now();

        src_w = frame->tiles[0].width;
        src_h = frame->tiles[0].height;
}

void Participant::set_pos_keep_aspect(int x, int y, int w, int h){
        double fx = static_cast<double>(w) / src_w;
        double fy = static_cast<double>(h) / src_h;

        if(fx > fy){
                height = h;
                width = src_w * fy;
                x += (w - width) / 2;
        } else {
                height = src_h * fx;
                width = w;
                y += (h - height) / 2;
        }

        this->x = x;
        this->y = y;
}

void Participant::to_cv_frame(){
        if(!frame)
                return;

        assert(frame->color_spec == UYVY);
        assert(frame->tile_count == 1);

        PROFILE_FUNC;

        auto& frame_tile = frame->tiles[0];
        luma.create(cv::Size(frame_tile.width, frame_tile.height), CV_8UC1);
        chroma.create(cv::Size(frame_tile.width / 2, frame_tile.height), CV_8UC2);

        assert(luma.isContinuous() && chroma.isContinuous());
        unsigned char *src = reinterpret_cast<unsigned char *>(frame_tile.data);
        unsigned char *luma_dst = luma.ptr(0);
        unsigned char *chroma_dst = chroma.ptr(0);

        unsigned src_len = frame_tile.data_len;

#ifdef __SSSE3__
        __m128i uv_shuff = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 14, 12, 10, 8, 6, 4, 2, 0);
        __m128i y_shuff = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 15, 13, 11, 9, 7, 5, 3, 1);
        while(src_len >= 32){
                __m128i uyvy = _mm_lddqu_si128((__m128i *)(void *) src);
                src += 16;

                __m128i y_low = _mm_shuffle_epi8(uyvy, y_shuff);
                __m128i uv_low = _mm_shuffle_epi8(uyvy, uv_shuff);
                
                uyvy = _mm_lddqu_si128((__m128i *)(void *) src);
                src += 16;

                __m128i y_high = _mm_shuffle_epi8(uyvy, y_shuff);
                __m128i uv_high = _mm_shuffle_epi8(uyvy, uv_shuff);

                _mm_store_si128((__m128i *)(void *) luma_dst, _mm_or_si128(y_low, _mm_bslli_si128(y_high, 8)));
                luma_dst += 16;
                _mm_store_si128((__m128i *)(void *) chroma_dst, _mm_or_si128(uv_low, _mm_bslli_si128(uv_high, 8)));
                chroma_dst += 16;

                src_len -= 32;
        }
#endif

        while(src_len >= 4){
                *chroma_dst++ = *src++;
                *luma_dst++ = *src++;
                *chroma_dst++ = *src++;
                *luma_dst++ = *src++;

                src_len -= 4;
        }
}

class Video_mixer{
public:
        enum class Layout{ Tiled, One_big };

        Video_mixer(int width, int height, codec_t color_space, Layout l = Layout::Tiled);

        void process_frame(unique_frame&& f);
        void get_mixed(video_frame *result);

private:
        void recompute_layout();
        void tiled_layout();
        void one_big_layout();

        unsigned width;
        unsigned height;
        codec_t color_space;
        Layout layout = Layout::Tiled;

        uint32_t primary_ssrc = 0;

        cv::Mat mixed_luma;
        cv::Mat mixed_chroma;

        std::map<uint32_t, Participant> participants;
};

Video_mixer::Video_mixer(int width, int height, codec_t color_space, Layout layout):
        width(width),
        height(height),
        color_space(color_space),
        layout(layout)
{
        mixed_luma.create(cv::Size(width, height), CV_8UC1);
        mixed_chroma.create(cv::Size(width / 2, height), CV_8UC2);
}

void Video_mixer::tiled_layout(){
        const unsigned rows = (unsigned) ceil(sqrt(participants.size()));
        const unsigned tile_width = width / rows;
        const unsigned tile_height = height / rows;

        int pos = 0;
        for(auto& [ssrc, t]: participants){
                t.set_pos_keep_aspect((pos % rows) * tile_width,
                                (pos / rows) * tile_height,
                                tile_width, tile_height);

                pos++;
        }
}

void Video_mixer::one_big_layout(){
        const unsigned small_height = height / 4;
        const unsigned big_height = small_height * 3;
        const unsigned tile_width = width / 4;

        int pos = 0;
        for(auto& [ssrc, t]: participants){
                if(ssrc == primary_ssrc){
                        t.set_pos_keep_aspect(0, 0, width, big_height);
                        continue;
                }

                if(pos * tile_width > width)
                        continue;

                t.set_pos_keep_aspect(pos * tile_width, big_height, tile_width, small_height);
                pos++;
        }
}

void Video_mixer::recompute_layout(){
        mixed_luma.setTo(16);
        mixed_chroma.setTo(128);

        if(!primary_ssrc && !participants.empty()){
                primary_ssrc = participants.begin()->first;
        }

        if(participants.size() == 1){
                participants.begin()->second.set_pos_keep_aspect(0, 0, width, height);
                return;
        }

        switch(layout){
        case Layout::Tiled:
                tiled_layout();
                break;
        case Layout::One_big:
                one_big_layout();
                break;
        }
}

void Video_mixer::process_frame(unique_frame&& f){
        auto iter = participants.find(f->ssrc);
        auto& p = participants[f->ssrc];
        p.frame_recieved(std::move(f));

        if(iter == participants.end()){
                recompute_layout();
        }
}

void Video_mixer::get_mixed(video_frame *result){
        PROFILE_FUNC;

        auto now = clock::now();

        bool recompute = false;
        for(auto it = participants.begin(); it != participants.end();){
                auto& p = it->second;
                if(now - p.last_time_recieved > std::chrono::seconds(2)){
                        if(it->first == primary_ssrc)
                                primary_ssrc = 0;

                        it = participants.erase(it);
                        recompute = true;
                        continue;
                }
                it++;
        }
        if(recompute)
                recompute_layout();

        for(auto&& [ssrc, p] : participants){
                p.to_cv_frame();

                PROFILE_DETAIL("resize participant");
                cv::Size l_size(p.width, p.height);
                cv::Size c_size(p.width / 2, p.height);
                cv::resize(p.luma, mixed_luma(cv::Rect(p.x, p.y, p.width, p.height)), l_size, 0, 0);
                cv::resize(p.chroma, mixed_chroma(cv::Rect(p.x / 2, p.y, p.width / 2, p.height)), c_size, 0, 0);
                PROFILE_DETAIL("");
        }

        unsigned char *dst = reinterpret_cast<unsigned char *>(result->tiles[0].data);
        unsigned char *chroma_src = mixed_chroma.ptr(0);
        unsigned char *luma_src = mixed_luma.ptr(0);
        assert(mixed_luma.isContinuous() && mixed_chroma.isContinuous());
        PROFILE_DETAIL("Convert to ug frame");
        unsigned dst_len = result->tiles[0].data_len;

#ifdef __SSSE3__
        __m128i y_shuff = _mm_set_epi8(7, 0xFF, 6, 0xFF, 5, 0xFF, 4, 0xFF, 3, 0xFF, 2, 0xFF, 1, 0xFF, 0, 0xFF);
        __m128i uv_shuff = _mm_set_epi8(0xFF, 7, 0xFF, 6, 0xFF, 5, 0xFF, 4, 0xFF, 3, 0xFF, 2, 0xFF, 1, 0xFF, 0);
        while(dst_len >= 32){
               __m128i luma = _mm_load_si128((__m128i const*)(const void *) luma_src); 
               luma_src += 16;
               __m128i chroma = _mm_load_si128((__m128i const*)(const void *) chroma_src); 
               chroma_src += 16;

               __m128i res = _mm_or_si128(_mm_shuffle_epi8(luma, y_shuff), _mm_shuffle_epi8(chroma, uv_shuff));
               _mm_storeu_si128((__m128i *)(void *) dst, res);
               dst += 16;

               luma = _mm_bsrli_si128(luma, 8);
               chroma = _mm_bsrli_si128(chroma, 8);

               res = _mm_or_si128(_mm_shuffle_epi8(luma, y_shuff), _mm_shuffle_epi8(chroma, uv_shuff));
               _mm_storeu_si128((__m128i *)(void *) dst, res);
               dst += 16;

               dst_len -= 32;
        }
#endif

        while(dst_len >= 4){
                *dst++ = *chroma_src++;
                *dst++ = *luma_src++;
                *dst++ = *chroma_src++;
                *dst++ = *luma_src++;

                dst_len -= 4;
        }
}

struct display_deleter{ void operator()(display *d){ display_done(d); } };
using unique_disp = std::unique_ptr<display, display_deleter>;
}//anon namespace

static constexpr std::chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_conference_common{
        struct module *parent = nullptr;

        struct video_desc desc = {};
        Video_mixer::Layout layout = Video_mixer::Layout::Tiled;

        unique_disp real_display;
        struct video_desc display_desc = {};

        std::mutex incoming_frames_lock;
        std::condition_variable incoming_frame_consumed;
        std::condition_variable new_incoming_frame_cv;
        std::queue<unique_frame> incoming_frames;
};

struct state_conference {
        std::shared_ptr<state_conference_common> common;
        struct video_desc desc;
};

static struct display *display_conference_fork(void *state)
{
        auto s = ((struct state_conference *)state)->common;
        struct display *out;
        char fmt[2 + sizeof(void *) * 2 + 1] = "";
        snprintf(fmt, sizeof fmt, "%p", state);

        int rc = initialize_video_display(s->parent,
                        "conference", fmt, 0, NULL, &out);
        if (rc == 0) return out; else return NULL;

        return out;
}

static void show_help(){
        printf("Conference display\n");
        printf("Usage:\n");
        printf("\t-d conference:<display_config>#<width>:<height>:[fps]:[layout]\n");
}

static void *display_conference_init(struct module *parent, const char *fmt, unsigned int flags)
{
        auto s = std::make_unique<state_conference>();
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "init fmt: %s\n", fmt);

        if(!fmt){
                show_help();
                return &display_init_noerr;
        }

        if (isdigit(fmt[0])){ // fork
                struct state_conference *orig;
                sscanf(fmt, "%p", &orig);
                s->common = orig->common;
                return s.release();
        }

        std::string requested_display = "gl";
        std::string disp_conf;

        s->common = std::make_shared<state_conference_common>();
        s->common->parent = parent;

        auto& desc = s->common->desc;
        desc.color_spec = UYVY;
        desc.interlacing = PROGRESSIVE;
        desc.tile_count = 1;
        desc.fps = 0;

        std::string_view param = fmt;
        auto disp_cfg = tokenize(param, '#');
        auto conf_cfg = tokenize(param, '#');

        auto parseNum = [](std::string_view sv, auto& res){
                if(sv.empty())
                        return false;
                return std::from_chars(sv.begin(), sv.end(), res).ec == std::errc();
        };

#define FAIL_IF(x, msg) \
        do {\
                if(x){\
                        log_msg(LOG_LEVEL_ERROR, msg);\
                        show_help();\
                        return &display_init_noerr;\
                }\
        } while(0)\

        requested_display = tokenize(disp_cfg, ':');
        FAIL_IF(requested_display.empty(), "Requested display cannot be empty\n");
        disp_conf = tokenize(disp_cfg, ':');
        FAIL_IF(!parseNum(tokenize(conf_cfg, ':'), desc.width), "Failed to parse width\n");
        FAIL_IF(!parseNum(tokenize(conf_cfg, ':'), desc.height), "Failed to parse height\n");

        int fps;
        FAIL_IF(!parseNum(tokenize(conf_cfg, ':'), fps), "Failed to parse fps\n");
        desc.fps = fps;

        auto tok = tokenize(conf_cfg, ':');
        if(tok == "one_big"){
                s->common->layout = Video_mixer::Layout::One_big;
        }

        struct display *d_ptr;
        int ret = initialize_video_display(parent, requested_display.c_str(), disp_conf.c_str(),
                        flags, nullptr, &d_ptr);

        if(ret != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to init real display\n");
                return nullptr;
        }

        s->common->real_display.reset(d_ptr);

        return s.release();
}

static void check_reconf(struct state_conference_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "reconfiguring real display\n");
                display_reconfigure(s->real_display.get(), s->display_desc, VIDEO_NORMAL);
        }
}

static void display_conference_worker(std::shared_ptr<state_conference_common> s){
        PROFILE_FUNC;
        Video_mixer mixer(s->desc.width, s->desc.height, UYVY, s->layout);

        auto next_frame_time = clock::now() + std::chrono::hours(1); //workaround for gcc bug 58931
        auto last_frame_time = clock::time_point::min();
        for(;;){
                unique_frame frame;

                auto wait_p = [s]{ return !s->incoming_frames.empty(); };
                std::unique_lock<std::mutex> lock(s->incoming_frames_lock);
                bool have_frame = s->new_incoming_frame_cv.wait_until(lock, next_frame_time, wait_p);
                if(have_frame){
                        frame = std::move(s->incoming_frames.front());
                        s->incoming_frames.pop();
                }
                lock.unlock();

                if(have_frame){
                        s->incoming_frame_consumed.notify_one();
                        if(!frame){
                                display_put_frame(s->real_display.get(), nullptr, PUTF_BLOCKING);
                                break;
                        }
                        mixer.process_frame(std::move(frame));
                }

                auto now = clock::now();
                if(next_frame_time <= now){
                        check_reconf(s.get(), s->desc);
                        auto disp_frame = display_get_frame(s->real_display.get());

                        mixer.get_mixed(disp_frame);

                        display_put_frame(s->real_display.get(), disp_frame, PUTF_BLOCKING);

                        last_frame_time = next_frame_time;
                        next_frame_time = now + std::chrono::hours(1);
                } else {
                        if(!have_frame)
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Woke up without new frame and before render time. This should not happen.\n");
                        using namespace std::chrono_literals;
                        next_frame_time = last_frame_time + std::chrono::duration_cast<clock::duration>(1s / s->desc.fps);
                        if(next_frame_time <= now){
                                next_frame_time = now;
                        }
                }
        }

        return;
}

static void display_conference_run(void *state)
{
        PROFILE_FUNC;
        auto s = static_cast<state_conference *>(state)->common;

        std::thread worker = std::thread(display_conference_worker, s);

        display_run_this_thread(s->real_display.get());

        worker.join();
}

static int display_conference_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = ((struct state_conference *)state)->common;
        if (property == DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES) {
                ((struct multi_sources_supp_info *) val)->val = true;
                ((struct multi_sources_supp_info *) val)->fork_display = display_conference_fork;
                ((struct multi_sources_supp_info *) val)->state = state;
                *len = sizeof(struct multi_sources_supp_info);
                return TRUE;

        } else if(property == DISPLAY_PROPERTY_CODECS) {
                codec_t codecs[] = {UYVY};

                memcpy(val, codecs, sizeof(codecs));

                *len = sizeof(codecs);

                return TRUE;
        }
        
        return display_ctl_property(s->real_display.get(), property, val, len);
}

static int display_conference_reconfigure(void *state, struct video_desc desc)
{
        struct state_conference *s = (struct state_conference *) state;

        s->desc = desc;

        return 1;
}

static void display_conference_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_conference_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static void display_conference_done(void *state)
{
        auto s = static_cast<state_conference *>(state);

        delete s;
}

static struct video_frame *display_conference_getf(void *state)
{
        PROFILE_FUNC;
        auto s = static_cast<state_conference *>(state);

        return vf_alloc_desc_data(s->desc);
}

static int display_conference_putf(void *state, struct video_frame *frame, int flags)
{
        auto s = static_cast<state_conference *>(state)->common;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                std::unique_lock<std::mutex> lg(s->incoming_frames_lock);
                if (s->incoming_frames.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        log_msg(LOG_LEVEL_WARNING, "[conference] queue full!\n");
                        if(flags == PUTF_NONBLOCK){
                                vf_free(frame);
                                return 1;
                        }
                }
                s->incoming_frame_consumed.wait(lg,
                                [s]{
                                return s->incoming_frames.size() < IN_QUEUE_MAX_BUFFER_LEN;
                                });
                s->incoming_frames.emplace(frame);
                lg.unlock();    
                s->new_incoming_frame_cv.notify_one();
        }

        return 0;
}

static auto display_conference_needs_mainloop(void *state)
{
        auto s = static_cast<struct state_conference *>(state)->common;
        return display_needs_mainloop(s->real_display.get());
}

static const struct video_display_info display_conference_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_conference_init,
        display_conference_run,
        display_conference_done,
        display_conference_getf,
        display_conference_putf,
        display_conference_reconfigure,
        display_conference_get_property,
        display_conference_put_audio_frame,
        display_conference_reconfigure_audio,
        display_conference_needs_mainloop
};

REGISTER_HIDDEN_MODULE(conference, &display_conference_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


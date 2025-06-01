/**
 * @file   video_capture/pipewire.cpp
 * @author Matej Hrica       <492778@mail.muni.cz>
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET z.s.p.o.
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

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <pipewire/pipewire.h>
#include <pipewire/version.h>
#include <spa/utils/result.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/props.h>
#include <spa/debug/types.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/dbus_portal.hpp"
#include "utils/synchronized_queue.h"
#include "utils/profile_timer.hpp"
#include "video_frame.h"
#include "video_codec.h"
#include "video_capture.h"
#include "pipewire_common.hpp"
#include "pixfmt_conv.h"

#define MOD_NAME "[PW vcap] "

static constexpr int DEFAULT_BUFFERS_PW = 2;
static constexpr int MIN_BUFFERS_PW = 2;
static constexpr int MAX_BUFFERS_PW = 10;
static constexpr int QUEUE_SIZE = 3;
static constexpr int DEFAULT_EXPECTING_FPS = 30;

struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;

struct vcap_pw_state { 
        pipewire_state_common pw;

        // used exclusively by ultragrid thread
        unique_frame in_flight_frame;

        std::mutex mut;
        std::vector<unique_frame> blank_frames;
        synchronized_queue<unique_frame, QUEUE_SIZE> sending_frames;

        video_desc desc = {};

#ifdef HAVE_DBUS_SCREENCAST
        ScreenCastPortal portal;
#endif
        int fd = -1;
        uint32_t node = PW_ID_ANY;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;
        struct spa_video_info format = {};

        enum class Mode {
                Generic,
                Screen_capture
        } mode = Mode::Generic;

        struct {
                bool show_cursor = false;
                std::string restore_file = "";
                uint32_t fps = 0;
                bool crop = true;
                std::string target = "";
        } user_options;

};

static void on_stream_state_changed(void * /*state*/, enum pw_stream_state old, enum pw_stream_state state, const char *error) {
        LOG(LOG_LEVEL_INFO) << MOD_NAME "stream state changed \"" << pw_stream_state_as_string(old) 
                                                << "\" -> \""<<pw_stream_state_as_string(state)<<"\"\n";
        
        if (error != nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "stream error: '"<< error << "'\n";
        }
}


static void on_stream_param_changed(void *state, uint32_t id, const struct spa_pod *param) {
        auto s = static_cast<vcap_pw_state *>(state);
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "param changed:\n";

        if (param == nullptr || id != SPA_PARAM_Format)
                return;

        int parse_format_ret = spa_format_parse(param, &s->format.media_type, &s->format.media_subtype);
        assert(parse_format_ret > 0);

        if(s->format.media_type != SPA_MEDIA_TYPE_video
                        || s->format.media_subtype != SPA_MEDIA_SUBTYPE_raw)
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Format not video/raw!\n");
                return;
        }

        auto& raw_format = s->format.info.raw;
        spa_format_video_raw_parse(param, &raw_format);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Got format: %s\n", spa_debug_type_find_name(spa_type_video_format, raw_format.format));

        s->desc.width = raw_format.size.width;
        s->desc.height = raw_format.size.height;
        if(raw_format.framerate.num != 0){
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Got framerate: %d / %d\n", raw_format.framerate.num, raw_format.framerate.denom);
                s->desc.fps = static_cast<double>(raw_format.framerate.num) / raw_format.framerate.denom;
        } else {
                //Variable framerate
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Got variable framerate: %d / %d\n", raw_format.max_framerate.num, raw_format.max_framerate.denom);
                if(raw_format.max_framerate.num != 0){
                        s->desc.fps = static_cast<double>(raw_format.max_framerate.num) / raw_format.max_framerate.denom;
                } else {
                        s->desc.fps = 60;
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid max framerate, using %f instead\n", s->desc.fps);
                }
        }
        s->desc.color_spec = uv_codec_from_pw_fmt(raw_format.format);
        s->desc.interlacing = PROGRESSIVE;
        s->desc.tile_count = 1;

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "size: %dx%d\n", s->desc.width, s->desc.height);

        int linesize = vc_get_linesize(s->desc.width, s->desc.color_spec);
        int32_t size = linesize * s->desc.height;

        uint8_t params_buffer[1024];

        struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));
        const struct spa_pod *params[3] = {};
        int n_params = 0;

        params[n_params++] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                SPA_PARAM_BUFFERS_buffers,
                SPA_POD_CHOICE_RANGE_Int(DEFAULT_BUFFERS_PW, MIN_BUFFERS_PW, MAX_BUFFERS_PW),
                SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
                SPA_PARAM_BUFFERS_size, SPA_POD_Int(size),
                SPA_PARAM_BUFFERS_stride, SPA_POD_Int(linesize),
                SPA_PARAM_BUFFERS_dataType,
                SPA_POD_CHOICE_FLAGS_Int((1 << SPA_DATA_MemPtr)))
        );
        
        if(s->user_options.crop) {
                params[n_params++] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                        SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                        SPA_PARAM_META_type, SPA_POD_Id(SPA_META_VideoCrop),
                        SPA_PARAM_META_size,
                        SPA_POD_Int(sizeof(struct spa_meta_region)))
                );
        }
        
        pw_stream_update_params(s->stream.get(), params, n_params);
}

static void pw_frame_to_uv_frame_memcpy(video_frame *dst, spa_buffer *src, spa_video_format fmt, spa_rectangle size, const spa_region *crop){
        auto offset = src->datas[0].chunk->offset;
        auto chunk_size = src->datas[0].chunk->size;
        auto stride = src->datas[0].chunk->stride;

        auto width = size.width;
        auto height = size.height;

        unsigned start_x = 0;
        unsigned start_y = 0;
        if(crop){
                width = crop->size.width;
                height = crop->size.height;
                start_x = crop->position.x;
                start_y = crop->position.y;
        }

        if(stride == 0)
                stride = chunk_size / size.height;

        auto linesize = vc_get_linesize(width, dst->color_spec);
        auto skip = vc_get_linesize(start_x, dst->color_spec);
        bool swap_red_blue = fmt == SPA_VIDEO_FORMAT_BGRA || fmt == SPA_VIDEO_FORMAT_BGRx; //TODO
        for(unsigned i = 0; i < height; i++){
                auto src_p = static_cast<unsigned char *>(src->datas[0].data) + offset + skip + stride * (i + start_y);
                auto dst_p = reinterpret_cast<unsigned char *>(dst->tiles[0].data) + linesize * i;
                /* It would probably be better to have separate functions for
                 * pipweire to uv frame conversions like lavd has. For now,
                 * let's handle BGRA to RGBA like this.
                 */
                if(swap_red_blue)
                        vc_copylineRGBA(dst_p, src_p, linesize, 16, 8, 0);
                else
                        memcpy(dst_p, src_p, linesize);
        }

        dst->tiles[0].width = width;
        dst->tiles[0].height = height;
        dst->tiles[0].data_len = linesize * height;
}

static void on_process(void *state) {
        PROFILE_FUNC;

        auto s= static_cast<vcap_pw_state *>(state);
        pw_buffer *buffer;
        [[maybe_unused]] int n_buffers_from_pw = 0;
        while((buffer = pw_stream_dequeue_buffer(s->stream.get())) != nullptr){
                ++n_buffers_from_pw;

                unique_frame next_frame;
                
                assert(buffer->buffer != nullptr);
                assert(buffer->buffer->datas != nullptr);
                assert(buffer->buffer->n_datas == 1);
                assert(buffer->buffer->datas[0].data != nullptr);

                if(buffer->buffer->datas[0].chunk == nullptr || buffer->buffer->datas[0].chunk->size == 0) {
                        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "dropping - empty pw frame " << "\n";
                        pw_stream_queue_buffer(s->stream.get(), buffer);
                        continue;
                }

                {
                        std::lock_guard<std::mutex> lock(s->mut);
                        if(!s->blank_frames.empty()){
                                next_frame = std::move(s->blank_frames.back());
                                s->blank_frames.pop_back();
                        }
                }

                if(!next_frame) {
                        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "dropping frame (no blank frames)\n";
                        pw_stream_queue_buffer(s->stream.get(), buffer);
                        continue;
                }

                spa_region *crop_region = nullptr;
                if (s->user_options.crop) {
                        spa_meta_region *meta_crop_region = static_cast<spa_meta_region*>(spa_buffer_find_meta_data(buffer->buffer, SPA_META_VideoCrop, sizeof(*meta_crop_region)));
                        if (meta_crop_region != nullptr && spa_meta_region_is_valid(meta_crop_region))
                           crop_region = &meta_crop_region->region;
                }

                if(crop_region){
                        //Update desc so that we don't reallocate on each frame
                        //TODO: Figure what to do when we can't actually crop (MJPEG)
                        s->desc.width = crop_region->size.width;
                        s->desc.height = crop_region->size.height;
                }

                if(!next_frame || !video_desc_eq(video_desc_from_frame(next_frame.get()), s->desc)){
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Desc changed, allocating new video_frame\n");
                        next_frame.reset(vf_alloc_desc_data(s->desc));
                }


                auto& raw_format = s->format.info.raw;

                //copy_frame(s->pw.video_format(), buffer->buffer, next_frame.get(), s->desc.width, s->desc.height, crop_region);
                pw_frame_to_uv_frame_memcpy(next_frame.get(), buffer->buffer, raw_format.format, raw_format.size, crop_region);

                s->sending_frames.push(std::move(next_frame));
                pw_stream_queue_buffer(s->stream.get(), buffer);
        }
        
        //LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: from pw: "<< n_buffers_from_pw << "\t sending: "<<s->sending_frames.size_approx() << "\t blank: " << s->blank_frames.size_approx() << "\n";
        
}

static void on_drained(void*)
{
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "pipewire: drained\n";
}

static void on_add_buffer(void * /*state*/, struct pw_buffer *)
{
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "pipewire: add_buffer\n";
}

static void on_remove_buffer(void * /*state*/, struct pw_buffer *)
{
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "pipewire: remove_buffer\n";
}

static const struct pw_stream_events stream_events = {
                .version = PW_VERSION_STREAM_EVENTS,
                .destroy = nullptr,
                .state_changed = on_stream_state_changed,
                .control_info = nullptr,
                .io_changed = nullptr,
                .param_changed = on_stream_param_changed,
                .add_buffer = on_add_buffer,
                .remove_buffer = on_remove_buffer,
                .process = on_process,
                .drained = on_drained,
#if PW_MAJOR > 0 || PW_MINOR > 3 || (PW_MINOR == 3 && PW_MICRO > 39)
                .command = nullptr,
                .trigger_done = nullptr,
#endif
};

static int start_pipewire(vcap_pw_state *s)
{    
        const struct spa_pod *params[2] = {};
        uint8_t params_buffer[1024];
        struct spa_pod_builder pod_builder = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));

        initialize_pw_common(s->pw, s->fd);

        std::string node_name = "ultragrid_in_";
        {
                char buf[32];
                snprintf(buf, sizeof(buf), "%ld", (long) getpid());
                node_name += buf;
        }

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Video",
                        PW_KEY_MEDIA_CATEGORY, "Capture",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, node_name.c_str(),
                        PW_KEY_NODE_DESCRIPTION, "UltraGrid capture",
                        nullptr);

        if(!s->user_options.target.empty()){
                pw_properties_set(props, STREAM_TARGET_PROPERTY_KEY, s->user_options.target.c_str());
        }

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        s->stream.reset(pw_stream_new(s->pw.pipewire_core.get(),
                        s->mode == vcap_pw_state::Mode::Screen_capture ? "ug_screencapture" : "ug_videocap",
                        props));

        assert(s->stream != nullptr);

        pw_stream_add_listener(
                        s->stream.get(),
                        &s->stream_listener.get(),
                        &stream_events,
                        s);

        auto size_rect_def = SPA_RECTANGLE(1920, 1080);
        auto size_rect_min = SPA_RECTANGLE(1, 1);
        auto size_rect_max = SPA_RECTANGLE(3840, 2160);

        auto framerate_def = SPA_FRACTION(s->user_options.fps > 0 ? s->user_options.fps : DEFAULT_EXPECTING_FPS, 1);
        auto framerate_min = SPA_FRACTION(0, 1);
        auto framerate_max = SPA_FRACTION(600, 1);

        const int n_params = 1;
        params[0] = static_cast<spa_pod *> (spa_pod_builder_add_object(
                        &pod_builder, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
                        SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
                        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
                        SPA_FORMAT_VIDEO_format,
                        SPA_POD_CHOICE_ENUM_Id(8, 
                                SPA_VIDEO_FORMAT_UYVY,
                                SPA_VIDEO_FORMAT_UYVY,
                                SPA_VIDEO_FORMAT_RGB,
                                SPA_VIDEO_FORMAT_RGBA,
                                SPA_VIDEO_FORMAT_RGBx,
                                SPA_VIDEO_FORMAT_YUY2,
                                SPA_VIDEO_FORMAT_BGRA,
                                SPA_VIDEO_FORMAT_BGRx
                                ),
                        SPA_FORMAT_VIDEO_size,
                        SPA_POD_CHOICE_RANGE_Rectangle(
                                        &size_rect_def,
                                        &size_rect_min,
                                        &size_rect_max),
                        SPA_FORMAT_VIDEO_framerate,
                        SPA_POD_CHOICE_RANGE_Fraction(
                                        &framerate_def,
                                        &framerate_min,
                                        &framerate_max)
        ));

        auto flags = PW_STREAM_FLAG_MAP_BUFFERS |
                PW_STREAM_FLAG_DONT_RECONNECT;

        if(!s->user_options.target.empty()){
                flags |= PW_STREAM_FLAG_AUTOCONNECT;
        }

        int res = pw_stream_connect(s->stream.get(),
                                        PW_DIRECTION_INPUT,
                                        s->node,
                                        static_cast<pw_stream_flags>(flags),
                                        params, n_params);
        if (res < 0) {
                fprintf(stderr, MOD_NAME "can't connect: %s\n", spa_strerror(res));
                return -1;
        }

        pw_stream_set_active(s->stream.get(), true);
        return 0;
}

static void vidcap_screen_pw_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_devices = (struct device_info *) calloc(*count, sizeof(struct device_info));
        // (*available_devices)[0].dev can be "" since screen cap. doesn't require parameters
        snprintf((*available_devices)[0].name, sizeof (*available_devices)[0].name, "Screen capture PipeWire");
}

static void vidcap_pw_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_devices = (struct device_info *) calloc(*count, sizeof(struct device_info));
        // (*available_devices)[0].dev can be "" since screen cap. doesn't require parameters
        snprintf((*available_devices)[0].name, sizeof (*available_devices)[0].name, "Generic PipeWire video capture");
}

static void show_screen_help() {
        auto param = [](const char* name) -> std::ostream& {
                col() << "  " << SBOLD(name) << " - ";
                return std::cout;
        };

        std::cout << "Screen capture using PipeWire and ScreenCast freedesktop portal API\n";
        std::cout << "Usage: -t screen_pw[:cursor|:nocrop|:fps=<fps>|:restore=<token_file>]]\n";
        param("cursor") << "make the cursor visible (default hidden)\n";
        param("nocrop") << "when capturing a window do not crop out the empty background\n";
        param("<fps>") << "preferred FPS passed to PipeWire (PipeWire may ignore it)\n";
        param("<token_file>") << "restore the selected window/display from a file.\n\t\tIf not possible, display the selection dialog and save the token to the file specified.\n";
}

static void show_generic_help(){
        color_printf("Pipewire video capture.\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-t pipewire" TERM_FG_RESET "[:fps=<fps>][:target=<device>]\n" TERM_RESET);
        color_printf("\n");

        color_printf("Devices:\n");
        print_devices("Video/Source");
}


static int parse_params(struct vidcap_params *params, vcap_pw_state *s) {
        if(const char *fmt = vidcap_params_get_fmt(params)) {        
                std::istringstream params_stream(fmt);
                
                std::string param;
                while (std::getline(params_stream, param, ':')) {
                        if (param == "help") {
                                if(s->mode == vcap_pw_state::Mode::Screen_capture)
                                        show_screen_help();
                                else
                                        show_generic_help();
                                return VIDCAP_INIT_NOERR;
                        } else if (param == "cursor") {
                                s->user_options.show_cursor = true;
                        } else if (param == "nocrop") {
                                s->user_options.crop = false;
                        } else {
                                auto split_index = param.find('=');
                                if(split_index != std::string::npos && split_index != 0){
                                        std::string name = param.substr(0, split_index);
                                        std::string value = param.substr(split_index + 1);

                                        if (name == "fps" || name == "FPS"){
                                                std::istringstream is(value);
                                                is >> s->user_options.fps;
                                                continue;
                                        }else if(name == "restore"){
                                                s->user_options.restore_file = std::move(value);
                                                continue;
                                        } else if(name =="target"){
                                                s->user_options.target = std::move(value);
                                                continue;
                                        }
                                }

                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "invalid option: \"" << param << "\"\n";
                                return VIDCAP_INIT_FAIL;
                        }
                }
        }
        return VIDCAP_INIT_OK;
}

#ifdef HAVE_DBUS_SCREENCAST
static int vidcap_screen_pw_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

        auto s = std::make_unique<vcap_pw_state>();

        s->mode = vcap_pw_state::Mode::Screen_capture;

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "init\n";
        
        int params_ok = parse_params(params, s.get());
        if(params_ok != VIDCAP_INIT_OK)
                return params_ok;

        for(int i = 0; i < QUEUE_SIZE; i++)
                s->blank_frames.emplace_back(vf_alloc(1));

        auto portalResult = s->portal.run(s->user_options.restore_file, s->user_options.show_cursor);
        
        if (portalResult.pipewire_fd == -1) {
                return VIDCAP_INIT_FAIL;
        }

        s->fd = portalResult.pipewire_fd;
        /* TODO: The node target_id param when calling stream_connect should be
         * always set to PW_ID_ANY as using object ids is now deprecated.
         * However, the dbus ScreenCast portal doesn't yet expose the object
         * serial which should be used instead.
         */
        s->node = portalResult.pipewire_node;

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "init ok\n";
        start_pipewire(s.get());
        
        *state = s.release();

        return VIDCAP_INIT_OK;
}
#endif

static int vidcap_pw_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

        auto s = std::make_unique<vcap_pw_state>();

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "init\n";
        
        int params_ok = parse_params(params, s.get());
        if(params_ok != VIDCAP_INIT_OK)
                return params_ok;

        for(int i = 0; i < QUEUE_SIZE; i++)
                s->blank_frames.emplace_back(vf_alloc(1));

        s->fd = -1;

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "init ok\n";
        start_pipewire(s.get());

        *state = s.release();

        return VIDCAP_INIT_OK;
}

static void vidcap_screen_pw_done(void *state)
{
        auto s = static_cast<vcap_pw_state *>(state);

        {
                pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());
                pw_stream_disconnect(s->stream.get());
        }

        pw_thread_loop_stop(s->pw.pipewire_loop.get());
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "done\n";   
        delete s;
}

static struct video_frame *vidcap_screen_pw_grab(void *state, struct audio_frame **audio)
{    
        PROFILE_FUNC;

        assert(state != nullptr);
        auto s = static_cast<vcap_pw_state *>(state);
        *audio = nullptr;
   
        if(s->in_flight_frame.get() != nullptr){
                s->blank_frames.push_back(std::move(s->in_flight_frame));
        }

        using namespace std::chrono_literals;
        s->sending_frames.timed_pop(s->in_flight_frame, 500ms);
        return s->in_flight_frame.get();
}

#ifdef HAVE_DBUS_SCREENCAST

static const struct video_capture_info vidcap_screen_pw_info = {
        vidcap_screen_pw_probe,
        vidcap_screen_pw_init,
        vidcap_screen_pw_done,
        vidcap_screen_pw_grab,
        MOD_NAME,
};

REGISTER_MODULE(screen_pw, &vidcap_screen_pw_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

#endif

static const struct video_capture_info vidcap_pw_info = {
        vidcap_pw_probe,
        vidcap_pw_init,
        vidcap_screen_pw_done,
        vidcap_screen_pw_grab,
        MOD_NAME,
};

REGISTER_MODULE(pipewire, &vidcap_pw_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

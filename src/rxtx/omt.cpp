/**
 * @file   rxtx/omt.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právických osob
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

#include <cassert>
#include <config.h>
#include <libomt.h>

#include "omt_common.hpp"
#include "debug.h"
#include "lib_common.h"
#include "rxtx.h"
#include "video_codec.h"
#include "video_display.h"
#include "video_frame.h"
#include "utils/misc.h"
#include "utils/string_view_utils.hpp"

#define MOD_NAME "[OMT] "

namespace{
struct omt_rxtx_state{
        module *parent = nullptr;
        omt_receive_uniq omt_recv_handle;
        omt_send_uniq omt_send_handle;

        OMTMediaFrame send_video_frame{};

        std::atomic<bool> should_exit = false;

        video_desc send_desc{};
        video_desc recv_desc{};
        display *display_device = nullptr;

        OMTQuality quality = OMTQuality_Default;
        std::string sender_name = "UltraGrid";
};

void omt_should_exit_callback(void *state){
        auto s = static_cast<omt_rxtx_state *>(state);
        s->should_exit = true;
}

void print_help(){
        color_printf("Open Media Transport\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-x omt" TERM_FG_RESET "[:quality=<qual>][:name=<name>]\n" TERM_RESET);
        color_printf("\n");
}

bool parse_params(omt_rxtx_state *s, std::string_view cfg){
        while(!cfg.empty()){
                auto tok = tokenize(cfg, ':', '"');

                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "help"){
                        print_help();
                        return false;
                } else if(key == "quality"){
                        if(val == "low"){
                                s->quality = OMTQuality_Low;
                        } else if(val == "medium"){
                                s->quality = OMTQuality_Medium;
                        } else if(val == "high"){
                                s->quality = OMTQuality_High;
                        } else{
                                log_msg(LOG_LEVEL_FATAL, MOD_NAME "Invalid quality \"%s\"\n", std::string(val).c_str());
                                return false;
                        }
                } else if(key == "name"){
                        s->sender_name = val;
                }
        }

        return true;
}

void init_recv(const rxtx_params *params, omt_rxtx_state *s){
        s->display_device = params->display_device;
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Create omt receive with address %s\n", params->receiver);
        s->omt_recv_handle.reset(omt_receive_create(params->receiver, static_cast<OMTFrameType>(OMTFrameType_Audio | OMTFrameType_Video),
                OMTPreferredVideoFormat_UYVY, OMTReceiveFlags_None));
}

void init_send(omt_rxtx_state *s){
        s->omt_send_handle.reset(omt_send_create(s->sender_name.c_str(), s->quality));
        set_omt_sender_info(s->omt_send_handle.get());
        s->send_video_frame.Type = OMTFrameType_Video;
        s->send_video_frame.Timestamp = -1;
}

void *omt_rxtx_create(rxtx_params *params){
        const struct rxtx_medium_params *params_video =
            &params->medium[TX_MEDIA_VIDEO];
        auto s    = std::make_unique<omt_rxtx_state>();
        s->parent = params->parent;

        if(!parse_params(s.get(), params->protocol_opts)){
                return strcmp(params->protocol_opts, "help") == 0 ? INIT_NOERR
                                                                  : nullptr;
        }

        ug_register_omt_log_callback();

        if(params_video->rxtx_mode & MODE_RECEIVER)
            init_recv(params, s.get());

        if(params_video->rxtx_mode & MODE_SENDER)
            init_send(s.get());

        return s.release();
}

void omt_rxtx_done(void *state){
        auto s = static_cast<omt_rxtx_state *>(state);
        delete s;
}

bool send_reconfigure(omt_rxtx_state *s, const video_desc& frame_desc){
        bool ret = omt_frame_init_from_desc(s->send_video_frame, frame_desc);
        if(ret)
                s->send_desc = frame_desc;
        return ret;
}

void omt_rxtx_send_frame(void *state, std::shared_ptr<video_frame> f){
        auto s = static_cast<omt_rxtx_state *>(state);
        auto frame_desc = video_desc_from_frame(f.get());
        if(!video_desc_eq(s->send_desc, frame_desc)){
                if (!send_reconfigure(s, frame_desc))
                        return;
        }

        omt_frame_set_data(s->send_video_frame, *f);

        omt_send(s->omt_send_handle.get(), &s->send_video_frame);
}

void *omt_rxtx_recv_worker(void *state){
        auto s = static_cast<omt_rxtx_state *>(state);
        register_should_exit_callback(s->parent, omt_should_exit_callback, s);

        while(!s->should_exit){
                auto omt_frame = omt_receive(s->omt_recv_handle.get(), OMTFrameType_Video, 1000);
                if(!omt_frame){
                        continue;
                }

                video_desc incoming_desc = video_desc_from_omt_frame(omt_frame);

                if(!video_desc_eq(incoming_desc, s->recv_desc)){
                        display_reconfigure(s->display_device, incoming_desc);
                        s->recv_desc = incoming_desc;
                }

                auto ug_frame = display_get_frame(s->display_device);
                assert(omt_frame->Stride == vc_get_linesize(ug_frame->tiles[0].width, ug_frame->color_spec));
                memcpy(ug_frame->tiles[0].data, omt_frame->Data, omt_frame->DataLength);

                display_put_frame(s->display_device, ug_frame, PUTF_BLOCKING);
        }

        display_put_frame(s->display_device, nullptr, PUTF_BLOCKING);
        unregister_should_exit_callback(s->parent, omt_should_exit_callback, s);

        return nullptr;
}
}

constexpr rxtx_info omt_video_rxtx_info = {
        .long_name          = "Open media transport",
        .create             = omt_rxtx_create,
        .done               = omt_rxtx_done,
        .ctl_property       = nullptr,
        .send_audio_frame   = nullptr,
        .recv_audio_frame   = nullptr,
        .send_video_frame   = omt_rxtx_send_frame,
        .send_video_frame_c = nullptr,
        .video_recv_routine = omt_rxtx_recv_worker,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(omt, &omt_video_rxtx_info, LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION);

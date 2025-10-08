/**
 * @file   video_rxtx.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET, zájmové sdružení právnických osob
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

#include <sstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

#include "debug.h"
#include "export.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/rtp.h"
#include "rtp/video_decoders.h"
#include "rtp/pbuf.h"
#include "tfrc.h"
#include "transmit.h"
#include "tv.h"
#include "utils/thread.h"
#include "utils/vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_rxtx.hpp"

#define MOD_NAME "[vrxtx] "

using std::map;
using std::shared_ptr;
using std::ostringstream;
using std::string;

video_rxtx::video_rxtx(map<string, param_u> const &params): m_port_id("default"),
                m_rxtx_mode(params.at("rxtx_mode").i),
                m_frames_sent(0ull),
                m_common(*static_cast<struct common_opts const *>(params.at("common").cptr)),
                m_thread_id(), m_poisoned(false), m_joined(true) {

        module_init_default(&m_sender_mod);
        m_sender_mod.cls = MODULE_CLASS_SENDER;
        module_register(&m_sender_mod, m_common.parent);

        module_init_default(&m_receiver_mod);
        m_receiver_mod.cls = MODULE_CLASS_RECEIVER;
        module_register(&m_receiver_mod, m_common.parent);

        try {
                int ret = compress_init(&m_sender_mod, static_cast<const char *>(params.at("compression").ptr),
                                &m_compression);
                if(ret != 0) {
                        if(ret < 0) {
                                throw string("Error initializing compression.");
                        }
                        if(ret > 0) {
                                throw EXIT_SUCCESS;
                        }
                }

                pthread_mutex_init(&m_lock, NULL);

        } catch (...) {
                if (m_compression) {
                        compress_done(m_compression);
                }

                module_done(&m_receiver_mod);
                module_done(&m_sender_mod);

                throw;
        }
}

void video_rxtx::should_exit(void *state) {
        video_rxtx *s = (video_rxtx *) state;
        s->m_should_exit = true;
}

video_rxtx::~video_rxtx() {
        join();
        if (!m_poisoned && m_compression) {
                send(NULL);
                compress_pop(m_compression);
        }
        compress_done(m_compression);
        module_done(&m_receiver_mod);
        module_done(&m_sender_mod);
}

void video_rxtx::start() {
        register_should_exit_callback(m_common.parent, video_rxtx::should_exit,
                                      this);
        if (pthread_create(&m_thread_id, NULL, video_rxtx::sender_thread,
                           (void *) this) != 0) {
                throw string("Unable to create sender thread!\n");
        }
        m_joined = false;
}

void video_rxtx::join() {
        if (m_joined) {
                return;
        }
        send(NULL); // pass poisoned pill
        pthread_join(m_thread_id, NULL);
        unregister_should_exit_callback(m_common.parent, video_rxtx::should_exit, this);
        m_joined = true;
}

const char *video_rxtx::get_long_name(string const & short_name) {
        auto vri = static_cast<const video_rxtx_info *>(load_library(short_name.c_str(), LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION));
        if (vri) {
                return vri->long_name;
        } else {
                return "";
        }
}

void video_rxtx::send(shared_ptr<video_frame> frame) {
        if (!frame && m_poisoned) {
                return;
        }
        if (!frame) {
                m_poisoned = true;
        } else {
                m_input_codec = frame->color_spec;
        }
        compress_frame(m_compression, std::move(frame));
}

void *video_rxtx::sender_thread(void *args) {
        return static_cast<video_rxtx *>(args)->sender_loop();
}

void video_rxtx::check_sender_messages() {
        // process external messages
        struct message *msg_external = nullptr;
        while((msg_external = check_message(&m_sender_mod))) {
                struct response *r = nullptr;
                auto *msg = (struct msg_sender *) msg_external;
                if (msg->type == SENDER_MSG_QUERY_VIDEO_MODE) {
                        if (!m_video_desc) {
                                r = new_response(RESPONSE_NO_CONTENT, nullptr);
                        } else {
                                ostringstream oss;
                                oss << m_video_desc
                                    << " (input " << get_codec_name(m_input_codec)
                                    << ")";
                                r = new_response(RESPONSE_OK,
                                                 oss.str().c_str());
                        }
                } else { // delegate to implementations
                        r = process_sender_message(msg);
                }

                free_message(msg_external, r);
        }
}

void *video_rxtx::sender_loop() {
        set_thread_name(__func__);
        struct video_desc saved_vid_desc;

        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        while(1) {
                check_sender_messages();

                shared_ptr<video_frame> tx_frame;

                tx_frame = compress_pop(m_compression);
                if (!tx_frame) {
                        break;
                }

                m_video_desc = video_desc_from_frame(tx_frame.get());
                export_video(m_common.exporter, tx_frame.get());

                send_frame(std::move(tx_frame));
                m_frames_sent += 1;
        }

        check_sender_messages();
        return NULL;
}

video_rxtx *video_rxtx::create(string const & proto, std::map<std::string, param_u> const &params)
{
        auto vri = static_cast<const video_rxtx_info *>(load_library(proto.c_str(), LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION));
        if (!vri) {
                return nullptr;
        }
        auto ret = vri->create(params);
        if (!ret) {
                return nullptr;
        }
        ret->start();
        return ret;
}

void video_rxtx::list(bool full)
{
        printf("Available TX protocols:\n");
        list_modules(LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION, full);
}

void
video_rxtx::set_audio_spec(const struct audio_desc * /* desc */,
                           int /* audio_rx_port */, int /* audio_tx_port */,
                           bool /* ipv6 */)
{
        MSG(INFO, "video RXTX not h264_rtp, not setting audio...\n");
}

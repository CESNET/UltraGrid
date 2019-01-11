/**
 * @file   video_rxtx.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2017 CESNET z.s.p.o.
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

#include <sstream>
#include <stdexcept>
#include <string>

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
#include "utils/vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_rxtx.h"

using namespace std;

video_rxtx::video_rxtx(map<string, param_u> const &params): m_port_id("default"), m_paused(params.at("paused").b),
                m_report_paused_play(false), m_rxtx_mode(params.at("rxtx_mode").i),
                m_parent(static_cast<struct module *>(params.at("parent").ptr)),
                m_frames_sent(0ull), m_compression(nullptr),
                m_exporter(static_cast<struct exporter *>(params.at("exporter").ptr)),
                m_thread_id(), m_poisoned(false), m_joined(true) {

        module_init_default(&m_sender_mod);
        m_sender_mod.cls = MODULE_CLASS_SENDER;
        module_register(&m_sender_mod, static_cast<struct module *>(params.at("parent").ptr));

        module_init_default(&m_receiver_mod);
        m_receiver_mod.cls = MODULE_CLASS_RECEIVER;
        module_register(&m_receiver_mod, static_cast<struct module *>(params.at("parent").ptr));

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
                        module_done(CAST_MODULE(m_compression));
                }

                module_done(&m_receiver_mod);
                module_done(&m_sender_mod);

                throw;
        }
}

video_rxtx::~video_rxtx() {
        join();
        if (!m_poisoned && m_compression) {
                send(NULL);
                compress_pop(m_compression);
        }
        module_done(CAST_MODULE(m_compression));
        module_done(&m_receiver_mod);
        module_done(&m_sender_mod);
}

void video_rxtx::start() {
        if (pthread_create
                        (&m_thread_id, NULL, video_rxtx::sender_thread,
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
        compress_frame(m_compression, frame);
        if (!frame) {
                m_poisoned = true;
        }
}

void *video_rxtx::sender_thread(void *args) {
        return static_cast<video_rxtx *>(args)->sender_loop();
}

int video_rxtx::check_sender_messages() {
        int ret = 0;
        // process external messages
        struct message *msg_external;
        while((msg_external = check_message(&m_sender_mod))) {
                int status;
                struct response *r = process_sender_message((struct msg_sender *) msg_external, &status);
                if (status == STREAM_PAUSED_PLAY) {
                        ret = STREAM_PAUSED_PLAY;
                }
                free_message(msg_external, r);
        }

        return ret;
}

void *video_rxtx::sender_loop() {
        struct video_desc saved_vid_desc;

        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        while(1) {
                int ret = check_sender_messages();

                shared_ptr<video_frame> tx_frame;

                tx_frame = compress_pop(m_compression);
                if (!tx_frame)
                        goto exit;

                export_video(m_exporter, tx_frame.get());

                tx_frame->paused_play = ret == STREAM_PAUSED_PLAY;

                send_frame(tx_frame);
                m_frames_sent += 1;
        }

exit:
        return NULL;
}

video_rxtx *video_rxtx::create(string const & proto, std::map<std::string, param_u> const &params)
{
        if (proto == "help") {
                printf("Available TX protocols:\n");
                list_modules(LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);
        }
        auto vri = static_cast<const video_rxtx_info *>(load_library(proto.c_str(), LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION));
        if (vri) {
                auto ret = vri->create(params);
                ret->start();
                return ret;
        } else {
                return nullptr;
        }
}


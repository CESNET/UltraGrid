/**
 * @file   video_rxtx.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET z.s.p.o.
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
#include <string>
#include <stdexcept>

#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/rtp.h"
#include "rtp/video_decoders.h"
#include "rtp/pbuf.h"
#include "tfrc.h"
#include "stats.h"
#include "transmit.h"
#include "tv.h"
#include "utils/vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_export.h"
#include "video_rxtx.h"
#include "video_rxtx/h264_rtp.h"
#include "video_rxtx/ihdtv.h"
#include "video_rxtx/rtp.h"
#include "video_rxtx/sage.h"
#include "video_rxtx/ultragrid_rtp.h"

using namespace std;

map<enum rxtx_protocol, struct video_rxtx_info> *registred_video_rxtx = nullptr;

/**
 * The purpose of this initializor instead of ordinary static initialization is that register_video_rxtx()
 * may be called before static members are initialized (it is __attribute__((constructor)))
 */
struct init_registred_video_rxtx {
        init_registred_video_rxtx()
        {
                if (registred_video_rxtx == nullptr) {
                        registred_video_rxtx = new map<enum rxtx_protocol, struct video_rxtx_info>;

                        *registred_video_rxtx = {{ULTRAGRID_RTP, {"UltraGrid RTP", create_video_rxtx_ultragrid_rtp}},
                                {IHDTV, {"iHDTV", create_video_rxtx_ihdtv}},
                                {SAGE, {"SAGE", create_video_rxtx_sage}}
                                //{H264_STD, {"H264 standard", create_video_rxtx_h264_std}},
                        };
                }
        }
};

static init_registred_video_rxtx loader;

video_rxtx_loader::video_rxtx_loader() {
#ifdef BUILD_LIBRARIES
        char name[128];
        snprintf(name, sizeof(name), "video_rxtx_*.so.%d", VIDEO_RXTX_ABI_VERSION);
        open_all(name);
#endif
}

void register_video_rxtx(enum rxtx_protocol proto, struct video_rxtx_info info)
{
        init_registred_video_rxtx loader;
        (*registred_video_rxtx)[proto] = info;
}

video_rxtx::video_rxtx(map<string, param_u> const &params): m_paused(false),
                m_rxtx_mode(params.at("rxtx_mode").i), m_compression(nullptr),
                m_video_exporter(static_cast<struct video_export *>(params.at("exporter").ptr)) {

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

                if (pthread_create
                                (&m_thread_id, NULL, video_rxtx::sender_thread,
                                 (void *) this) != 0) {
                        throw string("Unable to create sender thread!\n");
                }
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
        module_done(&m_receiver_mod);
        module_done(&m_sender_mod);
}

void video_rxtx::join() {
        send(NULL); // pass poisoned pill
        pthread_join(m_thread_id, NULL);
}

const char *video_rxtx::get_name(enum rxtx_protocol proto) {
        if (registred_video_rxtx->find(proto) != registred_video_rxtx->end()) {
                return registred_video_rxtx->at(proto).name;
        } else {
                return nullptr;
        }
}

void video_rxtx::send(shared_ptr<video_frame> frame) {
        compress_frame(m_compression, frame);
}

void *video_rxtx::sender_thread(void *args) {
        return static_cast<video_rxtx *>(args)->sender_loop();
}

void video_rxtx::check_sender_messages() {
        // process external messages
        struct message *msg_external;
        while((msg_external = check_message(&m_sender_mod))) {
                process_message((struct msg_sender *) msg_external);
                free_message(msg_external);
        }
}

void *video_rxtx::sender_loop() {
        struct video_desc saved_vid_desc;

        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        struct module *control_mod = get_module(get_root_module(&m_sender_mod), "control");
        struct stats *stat_data_sent = stats_new_statistics((struct control_state *)
                        control_mod, "data");

        while(1) {
                check_sender_messages();

                shared_ptr<video_frame> tx_frame;

                tx_frame = compress_pop(m_compression);
                if (!tx_frame)
                        goto exit;

                video_export(m_video_exporter, tx_frame.get());

                if (!m_paused) {
                        send_frame(tx_frame);

                        rtp_video_rxtx *rtp_rxtx = dynamic_cast<rtp_video_rxtx *>(this);
                        if (rtp_rxtx) {
                                stats_update_int(stat_data_sent,
                                                rtp_get_bytes_sent(rtp_rxtx->m_network_devices[0]));
                        }
                }
        }

exit:
        module_done(CAST_MODULE(m_compression));
        m_compression = nullptr;

        stats_destroy(stat_data_sent);

        return NULL;
}

video_rxtx *video_rxtx::create(enum rxtx_protocol proto, std::map<std::string, param_u> const &params)
{
        if (registred_video_rxtx->find(proto) != registred_video_rxtx->end()) {
                return registred_video_rxtx->at(proto).create(params);
        } else {
                return nullptr;
        }
}


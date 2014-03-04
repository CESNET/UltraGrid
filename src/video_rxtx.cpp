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
#include "video_rxtx/ihdtv.h"
#include "video_rxtx/rtp.h"
#include "video_rxtx/ultragrid_rtp.h"

using namespace std;

static void sage_rxtx_send(void *state, struct video_frame *tx_frame);
static void sage_rxtx_done(void *state);
static void h264_rtp_send(void *state, struct video_frame *tx_frame);
static void h264_rtp_done(void *state);

struct rx_tx sage_rxtx = {
        SAGE,
        "SAGE",
        sage_rxtx_send,
        sage_rxtx_done,
        NULL
};

struct rx_tx h264_rtp = {
        H264_STD,
        "H264 standard",
        h264_rtp_send,
        h264_rtp_done,
        NULL //TODO: h264_rtp_receiver_thread
};

video_rxtx::video_rxtx(struct module *parent, struct video_export *video_exporter,
                const char *requested_compression): m_paused(false), m_compression(NULL),
                m_video_exporter(video_exporter) {

        module_init_default(&m_sender_mod);
        m_sender_mod.cls = MODULE_CLASS_SENDER;
        module_register(&m_sender_mod, parent);

        module_init_default(&m_receiver_mod);
        m_receiver_mod.cls = MODULE_CLASS_RECEIVER;
        module_register(&m_receiver_mod, parent);

        int ret = compress_init(&m_sender_mod, requested_compression, &m_compression);
        if(ret != 0) {
                if(ret < 0) {
                        throw string("Error initializing compression.");
                }
                if(ret > 0) {
                        throw string("Error initializing compression.");
                }
        }

        pthread_mutex_init(&m_lock, NULL);

        if (pthread_create
                        (&m_thread_id, NULL, video_rxtx::sender_thread,
                         (void *) this) != 0) {
                throw string("Unable to create sender thread!\n");
        }
}

video_rxtx::~video_rxtx() {
        send(NULL); // pass poisoned pill
        pthread_join(m_thread_id, NULL);

        module_done(CAST_MODULE(m_compression));

        module_done(&m_sender_mod);
}

const char *video_rxtx::get_name(enum rxtx_protocol proto) {
        switch (proto) {
        case ULTRAGRID_RTP:
                return "UltraGrid RTP";
        case IHDTV:
                return "iHDTV";
        default:
                return NULL;
        }
}

static void sage_rxtx_send(void *state, struct video_frame *tx_frame)
{
        struct sage_rxtx_state *data = (struct sage_rxtx_state *) state;

        if(!video_desc_eq(data->saved_vid_desc,
                                video_desc_from_frame(tx_frame))) {
                display_reconfigure(data->sage_tx_device,
                                video_desc_from_frame(tx_frame));
                data->saved_vid_desc = video_desc_from_frame(tx_frame);
        }
        struct video_frame *frame =
                display_get_frame(data->sage_tx_device);
        memcpy(frame->tiles[0].data, tx_frame->tiles[0].data,
                        tx_frame->tiles[0].data_len);
        display_put_frame(data->sage_tx_device, frame, PUTF_NONBLOCK);

        VIDEO_FRAME_DISPOSE(tx_frame);
}

static void sage_rxtx_done(void *state)
{
        struct sage_rxtx_state *data = (struct sage_rxtx_state *) state;

        // poisoned pill to exit thread
        display_put_frame(data->sage_tx_device, NULL, PUTF_NONBLOCK);
        pthread_join(data->thread_id, NULL);

        display_done(data->sage_tx_device);
}

static void h264_rtp_send(void *state, struct video_frame *tx_frame)
{
        struct h264_rtp_state *data = (struct h264_rtp_state *) state;

        if(data->connections_count == 1) { /* normal/default case - only one connection */
            tx_send_h264(data->tx, tx_frame, data->network_devices[0]);
        } else {
            //TODO to be tested, the idea is to reply per destiny
                for (int i = 0; i < data->connections_count; ++i) {
                    tx_send_h264(data->tx, tx_frame,
                                        data->network_devices[i]);
                }
        }

        VIDEO_FRAME_DISPOSE(tx_frame);
}

static void h264_rtp_done(void *state)
{
        struct h264_rtp_state *data = (struct h264_rtp_state *) state;

        if (data->tx) {
                module_done(CAST_MODULE(data->tx));
        }
}

void video_rxtx::send(struct video_frame *frame) {
        compress_frame(m_compression, frame);
}

void *video_rxtx::sender_thread(void *args) {
        return static_cast<video_rxtx *>(args)->sender_loop();
}

void *video_rxtx::sender_loop() {
        struct video_desc saved_vid_desc;

        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        struct module *control_mod = get_module(get_root_module(&m_sender_mod), "control");
        struct stats *stat_data_sent = stats_new_statistics((struct control_state *)
                        control_mod, "data");

        while(1) {
                // process external messages
                struct message *msg_external;
                while((msg_external = check_message(&m_sender_mod))) {
                        process_message((struct msg_sender *) msg_external);
                        free_message(msg_external);
                }

                struct video_frame *tx_frame = NULL;

                tx_frame = compress_pop(m_compression);
                if (!tx_frame)
                        goto exit;

                video_export(m_video_exporter, tx_frame);

                if (!m_paused) {
                        send_frame(tx_frame);
                }

                VIDEO_FRAME_DISPOSE(tx_frame);

                if (dynamic_cast<rtp_video_rxtx *>(this)) {
                        rtp_video_rxtx *rtp_rxtx = dynamic_cast<rtp_video_rxtx *>(this);
                        stats_update_int(stat_data_sent,
                                        rtp_get_bytes_sent(rtp_rxtx->m_network_devices[0]));
                }

        }

exit:
        stats_destroy(stat_data_sent);

        return NULL;
}


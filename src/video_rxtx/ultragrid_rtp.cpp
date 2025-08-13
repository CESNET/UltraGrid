/**
 * @file   video_rxtx/ultragrid_rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET
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

#include "control_socket.h"
#include "export.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/ldgm.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
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
#include "video_rxtx/ultragrid_rtp.hpp"
#include "ug_runtime_error.hpp"
#include "utils/worker.h"

#include <chrono>
#include <sstream>
#include <utility>

using namespace std;

ultragrid_rtp_video_rxtx::ultragrid_rtp_video_rxtx(const map<string, param_u> &params) :
        rtp_video_rxtx(params), m_send_bytes_total(0)
{
        m_decoder_mode = (enum video_mode) params.at("decoder_mode").l;
        m_display_device = (struct display *) params.at("display_device").ptr;
        m_async_sending = false;

        if (get_commandline_param("decoder-use-codec") != nullptr && "help"s == get_commandline_param("decoder-use-codec")) {
                destroy_video_decoder(new_video_decoder(m_display_device));
                throw ug_no_error();
        }

        m_control = get_control_state(m_common.parent);
}

ultragrid_rtp_video_rxtx::~ultragrid_rtp_video_rxtx()
{
        for (auto d : m_display_copies) {
                display_done(d);
        }
}

void ultragrid_rtp_video_rxtx::join()
{
        video_rxtx::join();
        unique_lock<mutex> lk(m_async_sending_lock);
        m_async_sending_cv.wait(lk, [this]{return !m_async_sending;});
}

void *ultragrid_rtp_video_rxtx::receiver_thread(void *arg) {
        ultragrid_rtp_video_rxtx *s = static_cast<ultragrid_rtp_video_rxtx *>(arg);
        return s->receiver_loop();
}

void *(*ultragrid_rtp_video_rxtx::get_receiver_thread() noexcept)(void *arg)
{
        return receiver_thread;
}

void
ultragrid_rtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        m_video_desc = video_desc_from_frame(tx_frame.get());
        if (m_fec_state) {
                tx_frame = m_fec_state->encode(tx_frame);
        }

        auto data = new pair<ultragrid_rtp_video_rxtx *, shared_ptr<video_frame>>(this, tx_frame);

        unique_lock<mutex> lk(m_async_sending_lock);
        m_async_sending_cv.wait(lk, [this]{return !m_async_sending;});
        m_async_sending = true;
        task_run_async_detached(ultragrid_rtp_video_rxtx::send_frame_async_callback,
                        (void *) data);
}

void *ultragrid_rtp_video_rxtx::send_frame_async_callback(void *arg) {
        auto data = (pair<ultragrid_rtp_video_rxtx *, shared_ptr<video_frame>> *) arg;

        data->first->send_frame_async(data->second);
        delete data;

        return NULL;
}


void ultragrid_rtp_video_rxtx::send_frame_async(shared_ptr<video_frame> tx_frame)
{
        lock_guard<mutex> lock(m_network_devices_lock);

        tx_send(m_tx, tx_frame.get(), m_network_device);

        if ((m_rxtx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - m_common.start_time) / 100'000 * 9; // at 90000 Hz
                rtp_update(m_network_device, curr_time);
                rtp_send_ctrl(m_network_device, ts, nullptr, curr_time);

                // receive RTCP
                bool ret = true;
                do {
                        struct timeval timeout { 0, 0 };
                        ret = rtcp_recv_r(m_network_device, &timeout, ts);
                } while (!m_should_exit && ret);
        }

        m_async_sending_lock.lock();
        m_async_sending = false;
        m_async_sending_lock.unlock();
        m_async_sending_cv.notify_all();
}

void ultragrid_rtp_video_rxtx::receiver_process_messages()
{
        struct msg_receiver *msg;
        while ((msg = (struct msg_receiver *) check_message(&m_receiver_mod))) {
                lock_guard<mutex> lock(m_network_devices_lock);
                struct response *r = NULL;

                switch (msg->type) {
                case RECEIVER_MSG_CHANGE_RX_PORT:
                        {
                                assert(m_rxtx_mode == MODE_RECEIVER); // receiver only
                                auto *old_device = m_network_device;
                                auto old_port = m_recv_port_number;
                                m_recv_port_number = msg->new_rx_port;
                                m_network_device = initialize_network(m_requested_receiver.c_str(),
                                                m_recv_port_number,
                                                m_send_port_number, m_participants,
                                                m_common.force_ip_version,
                                                m_common.mcast_if,
                                                m_common.ttl);
                                if (m_network_device == nullptr) {
                                        log_msg(LOG_LEVEL_ERROR, "[control] Failed to change RX port to %d\n", msg->new_rx_port);
                                        r = new_response(RESPONSE_INT_SERV_ERR, "Changing RX port failed!");
                                        m_network_device = old_device;
                                        m_recv_port_number = old_port;
                                } else {
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Changed RX port to %d\n", msg->new_rx_port);
                                        destroy_rtp_device(old_device);
                                }
                                break;
                        }
                case RECEIVER_MSG_VIDEO_PROP_CHANGED:
                        {
                                pdb_iter_t it;
                                /// @todo should be set only to relevant participant, not all
                                struct pdb_e *cp = pdb_iter_init(m_participants, &it);
                                while (cp) {
                                        pbuf_set_playout_delay(cp->playout_buffer,
                                                        1.0 / msg->new_desc.fps);

                                        cp = pdb_iter_next(&it);
                                }
                        }
                        break;
                default:
                        assert(0 && "Wrong message passed to ultragrid_rtp_video_rxtx::receiver_process_messages()");
                }

                free_message((struct message *) msg, r ? r : new_response(RESPONSE_OK, NULL));
        }
}

/**
 * Removes display from decoders and effectively kills them. They cannot be used
 * until new display assigned.
 */
void ultragrid_rtp_video_rxtx::remove_display_from_decoders() {
        if (m_participants != NULL) {
                pdb_iter_t it;
                struct pdb_e *cp = pdb_iter_init(m_participants, &it);
                while (cp != NULL) {
                        if(cp->decoder_state)
                                video_decoder_remove_display(
                                                ((struct vcodec_state*) cp->decoder_state)->decoder);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
}

void ultragrid_rtp_video_rxtx::destroy_video_decoder(void *state) {
        struct vcodec_state *video_decoder_state = (struct vcodec_state *) state;

        if(!video_decoder_state) {
                return;
        }

        video_decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

struct vcodec_state *ultragrid_rtp_video_rxtx::new_video_decoder(struct display *d) {
        struct vcodec_state *state = (struct vcodec_state *) calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->decoder = video_decoder_init(&m_receiver_mod, m_decoder_mode,
                                d, m_common.encryption);

                if(!state->decoder) {
                        fprintf(stderr, "Error initializing decoder (incorrect '-M' or '-p' option?).\n");
                        free(state);
                        exit_uv(1);
                        return NULL;
                } else {
                        //decoder_register_display(state->decoder, uv->display_device);
                }
        }

        return state;
}

void *ultragrid_rtp_video_rxtx::receiver_loop()
{
        set_thread_name(__func__);
        struct pdb_e *cp;
        int fr;
        int last_buf_size = rtp_get_recv_buf(m_network_device);

#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_video_decoder(m_display_device);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER

        fr = 1;

        time_ns_t last_not_timeout = 0;

        while (!m_should_exit) {
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (m_common.start_time - curr_time) / 100'000 * 9; // at 90000 Hz

                rtp_update(m_network_device, curr_time);
                rtp_send_ctrl(m_network_device, ts, nullptr, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        curr_time = get_time_in_ns();
                        receiver_process_messages();
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                // use longer timeout when we are not receivng any data
                if ((curr_time - last_not_timeout) > NS_IN_SEC) {
                        timeout.tv_usec = 100000;
                } else {
                        timeout.tv_usec = 1000;
                }
                const bool ret = rtp_recv_r(m_network_device, &timeout, ts);

                // timeout
                if (!ret) {
                        // processing is needed here in case we are not receiving any data
                        receiver_process_messages();
                        //printf("Failed to receive data\n");
                } else {
                        last_not_timeout = curr_time;
                }

                /* Decode and render for each participant in the conference... */
                pdb_iter_t it;
                cp = pdb_iter_init(m_participants, &it);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               curr_time));
                        }

                        if(cp->decoder_state == NULL &&
                                        !pbuf_is_empty(cp->playout_buffer)) { // the second check is needed because we want to assign display to participant that really sends data
#ifdef SHARED_DECODER
                                cp->decoder_state = shared_decoder;
#else
                                // we are assigning our display so we make sure it is removed from other display

                                struct multi_sources_supp_info supp_for_mult_sources;
                                size_t len = sizeof(multi_sources_supp_info);
                                int ret = display_ctl_property(m_display_device,
                                                DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES, &supp_for_mult_sources, &len);
                                if (!ret) {
                                        supp_for_mult_sources.val = false;
                                }

                                struct display *d;
                                if (supp_for_mult_sources.val == false) {
                                        remove_display_from_decoders(); // must be called before creating new decoder state
                                        d = m_display_device;
                                } else {
                                        d = supp_for_mult_sources.fork_display(supp_for_mult_sources.state);
                                        assert(d != NULL);
                                        m_display_copies.push_back(d);
                                }

                                cp->decoder_state = new_video_decoder(d);
                                cp->decoder_state_deleter = destroy_video_decoder;

                                if (cp->decoder_state == NULL) {
                                        log_msg(LOG_LEVEL_FATAL, "Fatal: unable to create decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
#endif // SHARED_DECODER
                        }

                        struct vcodec_state *vdecoder_state = (struct vcodec_state *) cp->decoder_state;

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_video_frame, vdecoder_state)) {
                                fr = 1;
                        }

                        if(vdecoder_state && vdecoder_state->decoded % 100 == 99) {
                                int new_size = vdecoder_state->max_frame_size * 110ull / 100;
                                if(new_size > last_buf_size) {
                                        if (rtp_set_recv_buf(m_network_device, new_size)) {
                                                debug_msg("Recv buffer adjusted to %d\n", new_size);
                                        } else {
                                                display_buf_increase_warning(new_size);
                                        }
                                        last_buf_size = new_size;
                                }
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }

#ifdef SHARED_DECODER
        destroy_video_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders();
#endif //  SHARED_DECODER

        // pass poisoned pill to display
        display_put_frame(m_display_device, NULL, PUTF_BLOCKING);

        return 0;
}

uint32_t ultragrid_rtp_video_rxtx::get_ssrc()
{
        return rtp_my_ssrc(m_network_device);
}

static video_rxtx *create_video_rxtx_ultragrid_rtp(std::map<std::string, param_u> const &params)
{
        return new ultragrid_rtp_video_rxtx(params);
}

static const struct video_rxtx_info ultragrid_rtp_video_rxtx_info = {
        "UltraGrid RTP",
        create_video_rxtx_ultragrid_rtp
};

REGISTER_MODULE(ultragrid_rtp, &ultragrid_rtp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);


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

#include "compat/platform_time.h"
#include "control_socket.h"
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
#include "utils/vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_export.h"
#include "video_rxtx.h"
#include "video_rxtx/ultragrid_rtp.h"
#include "utils/worker.h"

#include <chrono>
#include <sstream>
#include <utility>

using namespace std;

ultragrid_rtp_video_rxtx::ultragrid_rtp_video_rxtx(const map<string, param_u> &params) :
        rtp_video_rxtx(params), m_send_bytes_total(0)
{
        if ((params.at("postprocess").ptr != NULL &&
                                strstr((const char *) params.at("postprocess").ptr, "help") != NULL)) {
                struct state_video_decoder *dec = video_decoder_init(m_parent, VIDEO_NORMAL,
                                (const char *) params.at("postprocess").ptr, NULL, NULL);
                video_decoder_destroy(dec);
                throw EXIT_SUCCESS;
        }

        m_decoder_mode = (enum video_mode) params.at("decoder_mode").l;
        auto postprocess_c = (const char *) params.at("postprocess").ptr;
        m_postprocess = postprocess_c ? postprocess_c : string();
        m_display_device = (struct display *) params.at("display_device").ptr;
        m_requested_encryption = (const char *) params.at("encryption").ptr;
        m_async_sending = false;

        m_control = (struct control_state *) get_module(get_root_module(static_cast<struct module *>(params.at("parent").ptr)), "control");
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

void *(*ultragrid_rtp_video_rxtx::get_receiver_thread())(void *arg) {
        return receiver_thread;
}

void ultragrid_rtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame)
{
        auto new_desc = video_desc_from_frame(tx_frame.get());
        if (new_desc != m_video_desc) {
                control_report_event(m_control, string("captured video changed - ") +
                                (string) new_desc);
                m_video_desc = new_desc;

        }
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
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        lock_guard<mutex> lock(m_network_devices_lock);

        int buffer_id = tx_get_buffer_id(m_tx);

        if (m_paused) {
                goto after_send;
        }

        if (m_connections_count == 1) { /* normal case - only one connection */
                tx_send(m_tx, tx_frame.get(),
                                m_network_devices[0]);
        } else { /* split */
                struct video_frame *split_frames = vf_alloc(m_connections_count);

                //assert(frame_count == 1);
                vf_split_horizontal(split_frames, tx_frame.get(),
                                m_connections_count);
                for (int i = 0; i < m_connections_count; ++i) {
                        tx_send_tile(m_tx, split_frames, i,
                                        m_network_devices[i]);
                }

                vf_free(split_frames);
        }

        if ((m_rxtx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                struct timeval curr_time;
                uint32_t ts;
                gettimeofday(&curr_time, NULL);
                ts = std::chrono::duration_cast<std::chrono::duration<double>>(m_start_time - std::chrono::steady_clock::now()).count() * 90000;
                rtp_update(m_network_devices[0], curr_time);
                rtp_send_ctrl(m_network_devices[0], ts, 0, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(m_network_devices[0], &timeout, ts);
        }

after_send:
        m_async_sending_lock.lock();
        m_async_sending = false;
        m_async_sending_lock.unlock();
        m_async_sending_cv.notify_all();

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        int dropped_frames = 0; /// @todo
        auto nano_actual = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        long long int nano_expected = 1000l * 1000 * 1000 / tx_frame->fps;
        int send_bytes = tx_frame->tiles[0].data_len;
        auto now = time_since_epoch_in_ms();
        auto compress_millis = tx_frame->compress_end - tx_frame->compress_start;

        ostringstream oss;
        if (m_port_id != -1) {
                oss << "-" << m_port_id << " ";
        }
        oss << "bufferId " << buffer_id <<
                " droppedFrames " << dropped_frames <<
                " nanoPerFrameActual " << (m_nano_per_frame_actual_cumul += nano_actual) <<
                " nanoPerFrameExpected " << (m_nano_per_frame_expected_cumul += nano_expected) <<
                " sendBytesTotal " << (m_send_bytes_total += send_bytes) <<
                " timestamp " << now <<
                " compressMillis " << (m_compress_millis_cumul += compress_millis);
        control_report_stats(m_control, oss.str());
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
                                auto old_devices = m_network_devices;
                                auto old_port = m_recv_port_number;
                                m_recv_port_number = msg->new_rx_port;
                                m_network_devices = initialize_network(m_requested_receiver.c_str(),
                                                m_recv_port_number,
                                                m_send_port_number, m_participants, m_ipv6,
                                                m_requested_mcast_if);
                                if (!m_network_devices) {
                                        log_msg(LOG_LEVEL_ERROR, "[control] Failed to change RX port to %d\n", msg->new_rx_port);
                                        r = new_response(RESPONSE_INT_SERV_ERR, "Changing RX port failed!");
                                        m_network_devices = old_devices;
                                        m_recv_port_number = old_port;
                                } else {
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Changed RX port to %d\n", msg->new_rx_port);
                                        destroy_rtp_devices(old_devices);
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
                case RECEIVER_MSG_POSTPROCESS:
                        if (strcmp("flush", msg->postprocess_cfg) == 0) {
                                m_postprocess = {};
                        } else {
                                m_postprocess = msg->postprocess_cfg;
                        }
                        {
                                pdb_iter_t it;
                                struct pdb_e *cp = pdb_iter_init(m_participants, &it);
                                while (cp != NULL) {
                                        if (cp->decoder_state) {
                                                video_decoder_remove_display(
                                                                ((struct vcodec_state*) cp->decoder_state)->decoder);
                                                video_decoder_destroy(
                                                                ((struct vcodec_state*) cp->decoder_state)->decoder);
                                                cp->decoder_state = NULL;
                                        }
                                        cp = pdb_iter_next(&it);
                                }
                                pdb_iter_done(&it);

                        }

                        break;
                case RECEIVER_MSG_INCREASE_VOLUME:
                case RECEIVER_MSG_DECREASE_VOLUME:
                case RECEIVER_MSG_MUTE:
                        abort();
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
                                m_postprocess.c_str(), d,
                                m_requested_encryption);

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
        uint32_t ts;
        struct pdb_e *cp;
        struct timeval curr_time;
        int fr;
        int ret;
        int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_decoder(uv);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER

        fr = 1;

        while (!should_exit) {
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                gettimeofday(&curr_time, NULL);
                auto curr_time_hr = std::chrono::high_resolution_clock::now();
                ts = std::chrono::duration_cast<std::chrono::duration<double>>(m_start_time - std::chrono::steady_clock::now()).count() * 90000;

                rtp_update(m_network_devices[0], curr_time);
                rtp_send_ctrl(m_network_devices[0], ts, 0, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&curr_time, NULL);
                        receiver_process_messages();
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                timeout.tv_usec = 1000;
                ret = rtp_recv_r(m_network_devices[0], &timeout, ts);

                // timeout
                if (ret == FALSE) {
                        // processing is needed here in case we are not receiving any data
                        receiver_process_messages();
                        //printf("Failed to receive data\n");
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
                                // we are assigning our display so we make sure it is removed from other dispaly

                                struct multi_sources_supp_info supp_for_mult_sources;
                                size_t len = sizeof(multi_sources_supp_info);
                                int ret = display_get_property(m_display_device,
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
                            (cp->playout_buffer, curr_time_hr, decode_video_frame, vdecoder_state)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == m_connections_count)
                                {
                                        tiles_post = 0;
                                        gettimeofday(&curr_time, NULL);
                                        fr = 1;
#if 0
                                        display_put_frame(uv->display_device,
                                                          cp->video_decoder_state->frame_buffer);
                                        cp->video_decoder_state->frame_buffer =
                                            display_get_frame(uv->display_device);
#endif
                                }
                                last_tile_received = curr_time;
                        }

                        /* dual-link TIMEOUT - we won't wait for next tiles */
                        if(tiles_post > 1 && tv_diff(curr_time, last_tile_received) >
                                        999999 / 59.94 / m_connections_count) {
                                tiles_post = 0;
                                gettimeofday(&curr_time, NULL);
                                fr = 1;
#if 0
                                display_put_frame(uv->display_device,
                                                cp->video_decoder_state->frame_buffer);
                                cp->video_decoder_state->frame_buffer =
                                        display_get_frame(uv->display_device);
#endif
                                last_tile_received = curr_time;
                        }

                        if(vdecoder_state && vdecoder_state->decoded % 100 == 99) {
                                int new_size = vdecoder_state->max_frame_size * 110ull / 100;
                                if(new_size > last_buf_size) {
                                        struct rtp **device = m_network_devices;
                                        while(*device) {
                                                int ret = rtp_set_recv_buf(*device, new_size);
                                                if(!ret) {
                                                        display_buf_increase_warning(new_size);
                                                }
                                                debug_msg("Recv buffer adjusted to %d\n", new_size);
                                                device++;
                                        }
                                        last_buf_size = new_size;
                                }
                        }

                        pbuf_remove(cp->playout_buffer, curr_time_hr);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }

#ifdef SHARED_DECODER
        destroy_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders();
#endif //  SHARED_DECODER

        // pass posioned pill to display
        display_put_frame(m_display_device, NULL, PUTF_BLOCKING);

        return 0;
}

uint32_t ultragrid_rtp_video_rxtx::get_ssrc()
{
        return rtp_my_ssrc(m_network_devices[0]);
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


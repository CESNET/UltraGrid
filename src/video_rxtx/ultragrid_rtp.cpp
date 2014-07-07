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
#include "ihdtv.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/ldgm.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
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
#include "video_rxtx/ultragrid_rtp.h"
#include "utils/worker.h"

#include <utility>

using namespace std;

ultragrid_rtp_video_rxtx::ultragrid_rtp_video_rxtx(struct module *parent, struct video_export *video_exporter,
                const char *requested_compression, const char *requested_encryption,
                const char *receiver, int rx_port, int tx_port,
                bool use_ipv6, const char *mcast_if, const char *requested_video_fec, int mtu,
                long packet_rate, enum video_mode decoder_mode, const char *postprocess,
                struct display *display_device) :
        rtp_video_rxtx(parent, video_exporter, requested_compression, requested_encryption,
                        receiver, rx_port, tx_port,
                        use_ipv6, mcast_if, requested_video_fec, mtu, packet_rate)
{
        if((postprocess && strstr(postprocess, "help") != NULL)) {
                struct state_video_decoder *dec = video_decoder_init(NULL, VIDEO_NORMAL, postprocess, NULL, NULL);
                video_decoder_destroy(dec);
                throw EXIT_SUCCESS;
        }

        gettimeofday(&m_start_time, NULL);
        m_decoder_mode = decoder_mode;
        m_postprocess = postprocess;
        m_display_device = display_device;
        m_requested_encryption = requested_encryption;
        m_async_sending = false;
}

ultragrid_rtp_video_rxtx::~ultragrid_rtp_video_rxtx()
{
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

void ultragrid_rtp_video_rxtx::send_frame(struct video_frame *tx_frame)
{
        if (m_fec_state) {
                struct video_frame *old_frame = tx_frame;
                tx_frame = m_fec_state->encode(tx_frame);
                VIDEO_FRAME_DISPOSE(old_frame);
        }

        auto data = new pair<ultragrid_rtp_video_rxtx *, struct video_frame *>(this, tx_frame);

        unique_lock<mutex> lk(m_async_sending_lock);
        m_async_sending_cv.wait(lk, [this]{return !m_async_sending;});
        m_async_sending = true;
        task_run_async_detached(ultragrid_rtp_video_rxtx::send_frame_async_callback,
                        (void *) data);
}

void *ultragrid_rtp_video_rxtx::send_frame_async_callback(void *arg) {
        auto data = (pair<ultragrid_rtp_video_rxtx *, struct video_frame *> *) arg;

        data->first->send_frame_async(data->second);
        delete data;

        return NULL;
}


void ultragrid_rtp_video_rxtx::send_frame_async(struct video_frame *tx_frame)
{
        lock_guard<mutex> lock(m_network_devices_lock);

        if (m_connections_count == 1) { /* normal case - only one connection */
                tx_send(m_tx, tx_frame,
                                m_network_devices[0]);
        } else { /* split */
                struct video_frame *split_frames = vf_alloc(m_connections_count);

                //assert(frame_count == 1);
                vf_split_horizontal(split_frames, tx_frame,
                                m_connections_count);
                for (int i = 0; i < m_connections_count; ++i) {
                        tx_send_tile(m_tx, split_frames, i,
                                        m_network_devices[i]);
                }

                vf_free(split_frames);
        }

        VIDEO_FRAME_DISPOSE(tx_frame);

        m_async_sending_lock.lock();
        m_async_sending = false;
        m_async_sending_cv.notify_all();
        m_async_sending_lock.unlock();
}

void ultragrid_rtp_video_rxtx::receiver_process_messages()
{
        struct msg_receiver *msg;
        while ((msg = (struct msg_receiver *) check_message(&m_receiver_mod))) {
                lock_guard<mutex> lock(m_network_devices_lock);

                switch (msg->type) {
                case RECEIVER_MSG_CHANGE_RX_PORT:
                        assert(rxtx_mode == MODE_RECEIVER); // receiver only
                        destroy_rtp_devices(m_network_devices);
                        m_recv_port_number = msg->new_rx_port;
                        m_network_devices = initialize_network(m_requested_receiver, m_recv_port_number,
                                        m_send_port_number, m_participants, m_ipv6,
                                        m_requested_mcast_if);
                        if (!m_network_devices) {
                                throw runtime_error("Changing RX port failed!");
                        }
                        break;
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
                }

                free_message((struct message *) msg);
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

struct vcodec_state *ultragrid_rtp_video_rxtx::new_video_decoder() {
        struct vcodec_state *state = (struct vcodec_state *) calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->decoder = video_decoder_init(&m_receiver_mod, m_decoder_mode,
                                m_postprocess, m_display_device,
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

        initialize_video_decompress();

        fr = 1;

        struct module *control_mod = get_module(get_root_module(&m_sender_mod), "control");
        struct stats *stat_loss = stats_new_statistics(
                        (struct control_state *) control_mod,
                        "loss");
        struct stats *stat_received = stats_new_statistics(
                        (struct control_state *) control_mod,
                        "received");
        uint64_t total_received = 0ull;

        while (!should_exit_receiver) {
                bool decoded = false;
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                gettimeofday(&curr_time, NULL);
                ts = tv_diff(curr_time, m_start_time) * 90000;
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
                timeout.tv_usec = 10000;
                ret = rtp_recv_r(m_network_devices[0], &timeout, ts);

                // timeout
                if (ret == FALSE) {
                        // processing is needed here in case we are not receiving any data
                        receiver_process_messages();
                        //printf("Failed to receive data\n");
                }
                total_received += ret;
                stats_update_int(stat_received, total_received);

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
                                remove_display_from_decoders();
                                cp->decoder_state = new_video_decoder();
                                cp->decoder_state_deleter = destroy_video_decoder;
#endif // SHARED_DECODER
                                if (cp->decoder_state == NULL) {
                                        fprintf(stderr, "Fatal: unable to create decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
                                ((struct vcodec_state*) cp->decoder_state)->display = m_display_device;
                        }

                        struct vcodec_state *vdecoder_state = (struct vcodec_state *) cp->decoder_state;

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_video_frame, vdecoder_state)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == m_connections_count)
                                {
                                        tiles_post = 0;
                                        gettimeofday(&curr_time, NULL);
                                        fr = 1;
                                        decoded = true;
#if 0
                                        display_put_frame(uv->display_device,
                                                          cp->video_decoder_state->frame_buffer);
                                        cp->video_decoder_state->frame_buffer =
                                            display_get_frame(uv->display_device);
#endif
                                }
                                last_tile_received = curr_time;
                                uint32_t sender_ssrc = cp->ssrc;
                                stats_update_int(stat_loss,
                                                rtp_compute_fract_lost(m_network_devices[0],
                                                        sender_ssrc));
                        }

                        /* dual-link TIMEOUT - we won't wait for next tiles */
                        if(tiles_post > 1 && tv_diff(curr_time, last_tile_received) >
                                        999999 / 59.94 / m_connections_count) {
                                tiles_post = 0;
                                gettimeofday(&curr_time, NULL);
                                fr = 1;
                                decoded = true;
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
                                }
                                last_buf_size = new_size;
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
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

        stats_destroy(stat_loss);
        stats_destroy(stat_received);

        return 0;
}


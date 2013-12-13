/*
 * FILE:    sender.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          David Cassany    <david.cassany@i2cat.net>
 *          Ignacio Contreras <ignacio.contreras@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "host.h"
#include "ihdtv.h"
#include "messaging.h"
#include "module.h"
#include "rtp/rtp.h"
#include "sender.h"
#include "stats.h"
#include "transmit.h"
#include "vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_display.h"
#include "video_export.h"

static void *sender_thread(void *arg);
static void ultragrid_rtp_send(void *state, struct video_frame *tx_frame);
static void ultragrid_rtp_done(void *state);
static void sage_rxtx_send(void *state, struct video_frame *tx_frame);
static void sage_rxtx_done(void *state);
static void h264_rtp_send(void *state, struct video_frame *tx_frame);
static void h264_rtp_done(void *state);

struct sender_priv_data {
        struct module mod;

        pthread_mutex_t lock;

        pthread_t thread_id;
        bool paused;
};

struct rx_tx ultragrid_rtp = {
        ULTRAGRID_RTP,
        "UltraGrid RTP",
        ultragrid_rtp_send,
        ultragrid_rtp_done,
        ultragrid_rtp_receiver_thread
};

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

static void sender_process_external_message(struct sender_data *data, struct msg_sender *msg)
{
        int ret;
        switch(msg->type) {
                case SENDER_MSG_CHANGE_RECEIVER:
                        assert(data->rxtx_protocol == ULTRAGRID_RTP || data->rxtx_protocol == H264_STD);
                        assert(((struct ultragrid_rtp_state *) data->tx_module_state)->connections_count == 1);

                        if(data->rxtx_protocol == ULTRAGRID_RTP){
                            printf("\n[SENDER.C  ULTRAGRID_RTP] processing external message --> SENDER MSG CHANGE RECEIVER: %s", msg->receiver);
                            ret = rtp_change_dest(((struct ultragrid_rtp_state *)
                                                                            data->tx_module_state)->network_devices[0],
                                                                    msg->receiver);
                        }else if(data->rxtx_protocol == H264_STD){
                            printf("\n[SENDER.C  H264_STD] processing external message --> SENDER MSG CHANGE RECEIVER: %s", msg->receiver);
                            ret = rtp_change_dest(
                                    ((struct h264_rtp_state *) data->tx_module_state)->network_devices[0],
                                    msg->receiver);
                        }

                        if(ret == FALSE) {
                                fprintf(stderr, "Changing receiver to: %s failed!\n",
                                                msg->receiver);
                        }
                        break;
                case SENDER_MSG_CHANGE_PORT:

                        printf("\n[SENDER.C] processing external message --> SENDER_MSG_CHANGE_PORT: %d", msg->port);

                        ((struct ultragrid_rtp_state *)
                         data->tx_module_state)->network_devices
                                = change_tx_port(data->uv, msg->port);
                        break;
                case SENDER_MSG_PAUSE:
                        data->priv->paused = true;
                        break;
                case SENDER_MSG_PLAY:
                        data->priv->paused = false;
                        break;
        }
}

bool sender_init(struct sender_data *data) {
        data->priv = calloc(1, sizeof(struct sender_priv_data));
        pthread_mutex_init(&data->priv->lock, NULL);

        // we lock and thread unlocks after initialized
        pthread_mutex_lock(&data->priv->lock);

        if (pthread_create
                        (&data->priv->thread_id, NULL, sender_thread,
                         (void *) data) != 0) {
                perror("Unable to create sender thread!\n");
                return false;
        }

        // this construct forces waiting for thread inititialization
        pthread_mutex_lock(&data->priv->lock);
        pthread_mutex_unlock(&data->priv->lock);

        return true;
}

void sender_done(struct sender_data *data) {
        pthread_join(data->priv->thread_id, NULL);

        free(data->priv);
}

static void ultragrid_rtp_send(void *state, struct video_frame *tx_frame)
{
        struct ultragrid_rtp_state *data = (struct ultragrid_rtp_state *) state;

        if(data->connections_count == 1) { /* normal case - only one connection */
                tx_send(data->tx, tx_frame,
                                data->network_devices[0]);
        } else { /* split */
                struct video_frame *split_frames = vf_alloc(data->connections_count);

                //assert(frame_count == 1);
                vf_split_horizontal(split_frames, tx_frame,
                                data->connections_count);
                for (int i = 0; i < data->connections_count; ++i) {
                        tx_send_tile(data->tx, split_frames, i,
                                        data->network_devices[i]);
                }

                vf_free(split_frames);
        }
}

static void ultragrid_rtp_done(void *state)
{
        struct ultragrid_rtp_state *data = (struct ultragrid_rtp_state *) state;

        if (data->tx) {
                module_done(CAST_MODULE(data->tx));
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
}

static void h264_rtp_done(void *state)
{
        struct h264_rtp_state *data = (struct h264_rtp_state *) state;

        if (data->tx) {
                module_done(CAST_MODULE(data->tx));
        }
        if (data->network_devices) {
                destroy_rtp_devices(data->network_devices);
        }
}

static void *sender_thread(void *arg) {
        struct sender_data *data = (struct sender_data *)arg;
        struct video_desc saved_vid_desc;

        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        module_init_default(&data->priv->mod);
        data->priv->mod.cls = MODULE_CLASS_SENDER;
        data->priv->mod.priv_data = data;
        module_register(&data->priv->mod, data->parent);

        pthread_mutex_unlock(&data->priv->lock);

        struct module *control_mod = get_module(get_root_module(&data->priv->mod), "control");
        struct stats *stat_data_sent = stats_new_statistics((struct control_state *)
                        control_mod, "data");
        unlock_module(control_mod);

        while(1) {
                // process external messages
                struct message *msg_external;
                while((msg_external = check_message(&data->priv->mod))) {
                        sender_process_external_message(data, (struct msg_sender *) msg_external);
                        free_message(msg_external);
                }

                struct video_frame *tx_frame = NULL;

                tx_frame = compress_pop(data->compression);
                if (!tx_frame)
                        goto exit;

                video_export(data->video_exporter, tx_frame);

                if (!data->priv->paused) {
                        data->send_frame(data->tx_module_state, tx_frame);
                }

                if (tx_frame->dispose) {
                        tx_frame->dispose(tx_frame);
                }

                if (data->rxtx_protocol == ULTRAGRID_RTP || data->rxtx_protocol == H264_STD) {
                        struct ultragrid_rtp_state *rtp_state = data->tx_module_state;
                        stats_update_int(stat_data_sent,
                                        rtp_get_bytes_sent(rtp_state->network_devices[0]));
                }

        }

exit:
        module_done(&data->priv->mod);
        stats_destroy(stat_data_sent);

        return NULL;
}


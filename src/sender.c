/*
 * FILE:    main.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
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

#include "messaging.h"
#include "rtp/rtp.h"
#include "sender.h"
#include "transmit.h"
#include "video.h"
#include "video_display.h"

static bool sender_change_receiver(struct sender_data *data, const char *new_receiver);

void sender_finish(struct sender_data *data) {
        pthread_mutex_lock(&data->lock);

        while(data->msg) {
                pthread_cond_wait(&data->msg_processed_cv, &data->lock);
        }

        data->msg = malloc(sizeof(struct sender_msg));
        data->msg->type = QUIT;
        data->msg->deleter = (void (*) (struct sender_msg *)) free;

        pthread_cond_signal(&data->msg_ready_cv);

        pthread_mutex_unlock(&data->lock);
}

static bool sender_change_receiver(struct sender_data *data, const char *new_receiver)
{
        struct sender_change_receiver_data msg_data;
        pthread_mutex_lock(&data->lock);

        while(data->msg) {
                pthread_cond_wait(&data->msg_processed_cv, &data->lock);
        }

        data->msg = malloc(sizeof(struct sender_msg));
        data->msg->type = CHANGE_RECEIVER;
        data->msg->deleter = (void (*) (struct sender_msg *)) free;
        msg_data.new_receiver = new_receiver;
        data->msg->data = (void *) &msg_data;

        pthread_cond_signal(&data->msg_ready_cv);

        // wait until it is processed
        while(data->msg) {
                pthread_cond_wait(&data->msg_processed_cv, &data->lock);
        }

        pthread_mutex_unlock(&data->lock);

        return msg_data.exit_status;
}

struct response *sender_change_receiver_callback(struct received_message *msg, void *udata)
{
        struct sender_data *s = (struct sender_data *) udata;
        const char *receiver = (const char *) msg->data;

        struct response *response = malloc(sizeof(struct response));
        response->deleter = (void (*) (struct response *)) free;

        if(sender_change_receiver(s, receiver)) {
                response->status = 200;
        } else {
                response->status = 400;
        }

        msg->deleter(msg);

        return response;
}

void sender_data_init(struct sender_data *data) {
        pthread_mutex_init(&data->lock, NULL);
        pthread_cond_init(&data->msg_ready_cv, NULL);
        pthread_cond_init(&data->msg_processed_cv, NULL);
        data->msg = NULL;
}

void sender_data_done(struct sender_data *data) {
        pthread_mutex_destroy(&data->lock);
        pthread_cond_destroy(&data->msg_ready_cv);
        pthread_cond_destroy(&data->msg_processed_cv);
}

void *sender_thread(void *arg) {
        struct sender_data *data = (struct sender_data *)arg;
        struct video_frame *splitted_frames = NULL;
        int tile_y_count;
        struct video_desc saved_vid_desc;

        tile_y_count = data->connections_count;
        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));

        /* we have more than one connection */
        if(tile_y_count > 1) {
                /* it is simply stripping frame */
                splitted_frames = vf_alloc(tile_y_count);
        }

        subscribe_messages(messaging_instance(), MSG_CHANGE_RECEIVER_ADDRESS,
                        sender_change_receiver_callback,
                        data);

        pthread_mutex_unlock(&data->lock);

        while(1) {
                pthread_mutex_lock(&data->lock);
                struct video_frame *tx_frame;

                while(!data->msg) {
                        pthread_cond_wait(&data->msg_ready_cv, &data->lock);
                }
                struct sender_msg *msg = data->msg;
                switch (msg->type) {

                        case QUIT:
                                data->msg = NULL;
                                msg->deleter(msg);
                                pthread_cond_signal(&data->msg_processed_cv);
                                pthread_mutex_unlock(&data->lock);
                                goto exit;
                        case CHANGE_RECEIVER:
                                {
                                        struct sender_change_receiver_data *msg_data = msg->data;
                                        assert(data->tx_protocol == ULTRAGRID_RTP);
                                        assert(data->connections_count == 1);
                                        msg_data->exit_status =
                                                rtp_change_dest(data->network_devices[0],
                                                                msg_data->new_receiver);
                                        msg->deleter(msg);
                                }
                        case FRAME:
                                tx_frame = msg->data;
                                break;
                        default:
                                abort();
                }

                pthread_mutex_unlock(&data->lock);

                if(data->tx_protocol == ULTRAGRID_RTP) {
                        if(data->connections_count == 1) { /* normal case - only one connection */
                                tx_send(data->tx, tx_frame,
                                                data->network_devices[0]);
                        } else { /* split */
                                int i;

                                //assert(frame_count == 1);
                                vf_split_horizontal(splitted_frames, tx_frame,
                                                tile_y_count);
                                for (i = 0; i < tile_y_count; ++i) {
                                        tx_send_tile(data->tx, splitted_frames, i,
                                                        data->network_devices[i]);
                                }
                        }
                } else { // SAGE
                        if(!video_desc_eq(saved_vid_desc,
                                                video_desc_from_frame(tx_frame))) {
                                display_reconfigure(data->sage_tx_device,
                                                video_desc_from_frame(tx_frame));
                                saved_vid_desc = video_desc_from_frame(tx_frame);
                        }
                        struct video_frame *frame =
                                display_get_frame(data->sage_tx_device);
                        memcpy(frame->tiles[0].data, tx_frame->tiles[0].data,
                                        tx_frame->tiles[0].data_len);
                        display_put_frame(data->sage_tx_device, frame, 0);
                }

                pthread_mutex_lock(&data->lock);

                data->msg = NULL;
                msg->deleter(msg);

                pthread_cond_signal(&data->msg_processed_cv);
                pthread_mutex_unlock(&data->lock);
        }

exit:
        vf_free(splitted_frames);

        return NULL;
}


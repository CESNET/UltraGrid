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
static struct sender_msg *new_frame_msg(struct video_frame *frame);
static void *sender_thread(void *arg);
static void sender_finish(struct sender_data *data);

struct sender_priv_data {
        pthread_mutex_t lock;
        pthread_cond_t msg_ready_cv;
        pthread_cond_t msg_processed_cv;
        struct sender_msg *msg;

        pthread_t thread_id;
};

struct sender_change_receiver_data {
        const char *new_receiver;
        int exit_status;
};

enum sender_msg_type {
        FRAME,
        CHANGE_RECEIVER,
        QUIT
};

struct sender_msg {
        enum sender_msg_type type;
        void *data;
        void (*deleter)(struct sender_msg *);
};

static struct sender_msg *new_frame_msg(struct video_frame *frame)
{
        struct sender_msg *msg = malloc(sizeof(struct sender_msg));
        msg->type = FRAME;
        msg->deleter = (void (*) (struct sender_msg *)) free;
        msg->data = frame;

        return msg;
}

void sender_post_new_frame(struct sender_data *data, struct video_frame *frame, bool nonblock) {
        pthread_mutex_lock(&data->priv->lock);
        while(data->priv->msg) {
                pthread_cond_wait(&data->priv->msg_processed_cv, &data->priv->lock);
        }

        data->priv->msg = new_frame_msg(frame);
        pthread_cond_signal(&data->priv->msg_ready_cv);

        if(!nonblock) {
                // we wait until frame is completed
                while(data->priv->msg) {
                        pthread_cond_wait(&data->priv->msg_processed_cv, &data->priv->lock);
                }
        }
        pthread_mutex_unlock(&data->priv->lock);
}

static void sender_finish(struct sender_data *data) {
        pthread_mutex_lock(&data->priv->lock);

        while(data->priv->msg) {
                pthread_cond_wait(&data->priv->msg_processed_cv, &data->priv->lock);
        }

        data->priv->msg = malloc(sizeof(struct sender_msg));
        data->priv->msg->type = QUIT;
        data->priv->msg->deleter = (void (*) (struct sender_msg *)) free;

        pthread_cond_signal(&data->priv->msg_ready_cv);

        pthread_mutex_unlock(&data->priv->lock);
}

static bool sender_change_receiver(struct sender_data *data, const char *new_receiver)
{
        struct sender_change_receiver_data msg_data;
        pthread_mutex_lock(&data->priv->lock);

        while(data->priv->msg) {
                pthread_cond_wait(&data->priv->msg_processed_cv, &data->priv->lock);
        }

        struct sender_msg *msg = malloc(sizeof(struct sender_msg));
        msg->type = CHANGE_RECEIVER;
        msg->deleter = (void (*) (struct sender_msg *)) free;
        msg_data.new_receiver = strdup(new_receiver);
        msg_data.exit_status = -1;
        msg->data = (void *) &msg_data;

        data->priv->msg = msg;

        pthread_cond_signal(&data->priv->msg_ready_cv);

        // wait until it is processed
        while(msg_data.exit_status == -1) {
                pthread_cond_wait(&data->priv->msg_processed_cv, &data->priv->lock);
        }

        pthread_mutex_unlock(&data->priv->lock);

        return msg_data.exit_status;
}

struct response *sender_change_receiver_callback(struct received_message *msg, void *udata)
{
        struct sender_data *s = (struct sender_data *) udata;
        const char *receiver = (const char *) msg->data;

        struct response *response;

        if(sender_change_receiver(s, receiver)) {
                response = new_response(RESPONSE_OK);
        } else {
                response = new_response(RESPONSE_NOT_FOUND);
        }

        msg->deleter(msg);

        return response;
}

bool sender_init(struct sender_data *data) {
        data->priv = malloc(sizeof(struct sender_priv_data));
        pthread_mutex_init(&data->priv->lock, NULL);
        pthread_cond_init(&data->priv->msg_ready_cv, NULL);
        pthread_cond_init(&data->priv->msg_processed_cv, NULL);
        data->priv->msg = NULL;

        // we lock and thred unlocks after initialized
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
        sender_finish(data);
        pthread_join(data->priv->thread_id, NULL);

        pthread_mutex_destroy(&data->priv->lock);
        pthread_cond_destroy(&data->priv->msg_ready_cv);
        pthread_cond_destroy(&data->priv->msg_processed_cv);
        free(data->priv);
}

static void *sender_thread(void *arg) {
        struct sender_data *data = (struct sender_data *)arg;
        struct video_frame *splitted_frames = NULL;
        int tile_y_count;
        struct video_desc saved_vid_desc;
        void *messaging_subscribtion = NULL;

        tile_y_count = data->connections_count;
        memset(&saved_vid_desc, 0, sizeof(saved_vid_desc));


        /* we have more than one connection */
        if(tile_y_count > 1) {
                /* it is simply stripping frame */
                splitted_frames = vf_alloc(tile_y_count);
        }

        messaging_subscribtion =
                subscribe_messages(messaging_instance(), MSG_CHANGE_RECEIVER_ADDRESS,
                        sender_change_receiver_callback,
                        data);

        pthread_mutex_unlock(&data->priv->lock);

        while(1) {
                pthread_mutex_lock(&data->priv->lock);
                struct video_frame *tx_frame = NULL;

                while(!data->priv->msg) {
                        pthread_cond_wait(&data->priv->msg_ready_cv, &data->priv->lock);
                }
                struct sender_msg *msg = data->priv->msg;
                switch (msg->type) {

                        case QUIT:
                                data->priv->msg = NULL;
                                msg->deleter(msg);
                                pthread_cond_broadcast(&data->priv->msg_processed_cv);
                                pthread_mutex_unlock(&data->priv->lock);
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
                                break;
                        case FRAME:
                                tx_frame = msg->data;
                                break;
                        default:
                                abort();
                }

                if(!tx_frame) {
                        data->priv->msg = NULL;
                        pthread_cond_broadcast(&data->priv->msg_processed_cv);
                        pthread_mutex_unlock(&data->priv->lock);
                        continue;
                }

                pthread_mutex_unlock(&data->priv->lock);

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

                pthread_mutex_lock(&data->priv->lock);

                data->priv->msg = NULL;
                msg->deleter(msg);

                pthread_cond_broadcast(&data->priv->msg_processed_cv);
                pthread_mutex_unlock(&data->priv->lock);
        }

exit:
        vf_free(splitted_frames);
        unsubscribe_messages(messaging_instance(), messaging_subscribtion);

        return NULL;
}


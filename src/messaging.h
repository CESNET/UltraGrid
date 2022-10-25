/**
 * @file   messaging.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2021 CESNET, z. s. p. o.
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

#ifndef _MESSAGING_H
#define _MESSAGING_H

#ifdef __cplusplus
#include <cstring>
#endif

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct messaging;
struct module;
struct response;

#define RESPONSE_OK           200
#define RESPONSE_ACCEPTED     202
#define RESPONSE_NO_CONTENT   204
#define RESPONSE_BAD_REQUEST  400
#define RESPONSE_NOT_FOUND    404
#define RESPONSE_REQ_TIMEOUT  408
#define RESPONSE_INT_SERV_ERR 500
#define RESPONSE_NOT_IMPL     501

#define RESPONSE_SUCCESSFUL(code) ((code) >= 200 && (code) < 300)

struct message;

struct message {
        /**
         * Individual message types may have defined custom data deleters.
         * Please note that the deleter must not delete the struct itself.
         */
        void (*data_deleter)(struct message *);

        // following members are used internally and should not be touched anyhow
        // except from messaging.cpp
        void (*send_response)(void *priv_data, struct response *);
        void *priv_data;
};

enum msg_root_type {
        ROOT_MSG_REGISTER_SHOULD_EXIT,
};

struct msg_root {
        struct message m;
        enum msg_root_type type;
        union {
                struct {
                        void (*should_exit_callback)(void *udata);
                        void *udata;
                };
        };
};

enum msg_sender_type {
        SENDER_MSG_CHANGE_RECEIVER,
        SENDER_MSG_CHANGE_PORT,
        SENDER_MSG_GET_STATUS,
        SENDER_MSG_MUTE,
        SENDER_MSG_PLAY,
        SENDER_MSG_PAUSE,
        SENDER_MSG_CHANGE_FEC,
        SENDER_MSG_QUERY_VIDEO_MODE,
        SENDER_MSG_RESET_SSRC,
};

struct msg_sender {
        struct message m;
        enum msg_sender_type type;
        union {
                struct {
                        int rx_port;
                        int tx_port;
                };
                char receiver[128];
                char fec_cfg[1024];
        };
};

enum msg_receiver_type {
        RECEIVER_MSG_CHANGE_RX_PORT,
        RECEIVER_MSG_VIDEO_PROP_CHANGED,
        RECEIVER_MSG_GET_AUDIO_STATUS,
        RECEIVER_MSG_INCREASE_VOLUME,
        RECEIVER_MSG_DECREASE_VOLUME,
        RECEIVER_MSG_MUTE,
};
struct msg_receiver {
        struct message m;
        enum msg_receiver_type type;
        union {
                uint16_t new_rx_port;
                struct video_desc new_desc;
                char postprocess_cfg[1024];
        };
};

enum compress_change_type {
        CHANGE_COMPRESS,
        CHANGE_PARAMS
};

struct msg_change_compress_data {
        struct message m;
        enum compress_change_type what;
        char config_string[128];
};

struct msg_stats {
        struct message m;
        char what[128];
        int value;
};

/**
 * It is suggested to use a tag at the beginning of
 * msg_universal::text to identify the receiving module.
 */
struct msg_universal {
#ifdef __cplusplus
        msg_universal(const char *contents) {
                memset(&m, 0, sizeof m);
                strncpy(text, contents, sizeof text - 1);
        }
#endif
        struct message m;
        char text[8192];
};

#define MSG_UNIVERSAL_TAG_TX "TX_msg "

struct response *new_response(int status, const char *optional_message);
void free_response(struct response *r);
int response_get_status(struct response *r);
const char *response_get_text(struct response *r);

void module_check_undelivered_messages(struct module *);
struct response *send_message(struct module *, const char *path, struct message *msg) __attribute__ ((warn_unused_result));
#define SEND_MESSAGE_FLAG_QUIET    (1<<0) ///< do not print error messages on console (recv not found/queue full)
#define SEND_MESSAGE_FLAG_NO_STORE (1<<1) ///< if receiver doesn't exist, doesn't store it and retur 404 instead
struct response *send_message_sync(struct module *, const char *path, struct message *msg, int timeout_ms, int flags) __attribute__ ((warn_unused_result));
struct response *send_message_to_receiver(struct module *, struct message *msg) __attribute__ ((warn_unused_result));
struct message *new_message(size_t length) __attribute__ ((warn_unused_result));
void free_message(struct message *m, struct response *r);
const char *response_status_to_text(int status);

struct message *check_message(struct module *);

void free_message_for_child(void *m, struct response *r);

#ifdef __cplusplus
}
#endif

#endif// _MESSAGING_H


#ifndef _MESSAGING_H
#define _MESSAGING_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <transmit.h>

#ifdef __cplusplus
extern "C" {
#endif

struct messaging;
struct module;

struct response {
        int status;
        void (*deleter)(struct response*);
        char *text;
};

#define RESPONSE_OK           200
#define RESPONSE_BAD_REQUEST  400
#define RESPONSE_NOT_FOUND    404
#define RESPONSE_INT_SERV_ERR 500
#define RESPONSE_NOT_IMPL     501

struct msg_change_receiver_address {
        char *receiver;
};

struct msg_change_fec_data {
        enum tx_media_type media_type;
        const char *fec;
};

struct msg_change_compress_data {
        enum {
                CHANGE_COMPRESS,
                CHANGE_PARAMS
        } what;
        const char *config_string;
};

struct msg_stats {
        const char *what;
        int value;
};

struct response *new_response(int status, char *optional_message);

typedef struct response *(*msg_callback_t)(void *msg, struct module *mod);

struct response *send_message(struct module *, const char *path, void *data);
struct response *send_message_to_receiver(struct module *, void *data);
const char *response_status_to_text(int status);


#ifdef __cplusplus
}
#endif

#endif// _MESSAGING_H


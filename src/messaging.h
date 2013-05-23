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

enum msg_class {
        MSG_CHANGE_RECEIVER_ADDRESS,
        MSG_CHANGE_FEC,
        MSG_CHANGE_COMPRESS
};

struct received_message {
        enum msg_class message_type;
        void *data;
        void (*deleter)(struct received_message*);
};

struct response {
        int status;
        void (*deleter)(struct response*);
};

#define RESPONSE_OK           200
#define RESPONSE_BAD_REQUEST  400
#define RESPONSE_NOT_FOUND    404
#define RESPONSE_INT_SERV_ERR 500

struct msg_change_fec_data {
        enum tx_media_type media_type;
        const char *fec;
};

struct msg_change_compress_data {
        enum {
                CHANGE_COMPRESS,
                CHANGE_PARAMS
        } what;
        const char *module;
        const char *params;
};

struct response *new_response(int status);

typedef struct response *(*msg_callback_t)(struct received_message *msg, void *udata);

struct messaging *messaging_instance(void);
/**
 * Registers callback for subscripition of events.
 *
 * @warning
 * The calls of messaging objects from within the callbacks should be restricted to the particular
 * thread. Calls from different threads will be blocked until the end of the lifetime of of the callback
 * and thus must not be waited for from the callback (deadlock)!
 *
 * @returns handle to the subscribtion (to be passed to unsubscribe function)
 */
void *subscribe_messages(struct messaging *, enum msg_class cls, msg_callback_t callback, void *udata);
void unsubscribe_messages(struct messaging *, void *handle);
struct response *send_message(struct messaging *, enum msg_class cls, void *data);
const char *response_status_to_text(int status);

#ifdef __cplusplus
}
#endif

#endif// _MESSAGING_H


#ifndef _MESSAGING_H
#define _MESSAGING_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct messaging;

enum msg_class {
        MSG_CHANGE_RECEIVER_ADDRESS
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

typedef struct response *(*msg_callback_t)(struct received_message *msg, void *udata);

struct messaging *messaging_instance(void);
void subscribe_messages(struct messaging *, enum msg_class cls, msg_callback_t callback, void *udata);

#ifdef __cplusplus
}
#endif

#endif// _MESSAGING_H


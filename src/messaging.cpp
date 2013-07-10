#include "messaging.h"

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "debug.h"
#include "module.h"
#include "utils/list.h"
#include "utils/lock_guard.h"

struct response *send_message(struct module *root, const char *const_path, struct message *msg)
{
        char buf[1024];
        struct module *receiver = get_module(root, const_path);
        /**
         * @invariant
         * either receiver is NULL or receiver->lock is locked (exactly once)
         */

        if(receiver == NULL) {
                fprintf(stderr, "%s not found:\n", const_path);
                dump_tree(root, 0);
                snprintf(buf, sizeof(buf), "(path: %s)", const_path);
                return new_response(RESPONSE_NOT_FOUND, strdup(buf));
        }

        lock_guard guard(receiver->lock, lock_guard_retain_ownership_t());

        if(receiver->msg_callback == NULL) {
                simple_linked_list_append(receiver->msg_queue, msg);
                return new_response(RESPONSE_ACCEPTED, NULL);
        }

        struct response *resp = receiver->msg_callback(receiver, msg);

        if(resp) {
                return resp;
        } else {
                return new_response(RESPONSE_INT_SERV_ERR, strdup("(empty response)"));
        }
}

struct response *send_message_to_receiver(struct module *receiver, struct message *msg)
{
        lock_guard guard(receiver->lock);
        if(receiver->msg_callback) {
                return receiver->msg_callback(receiver, msg);
        } else {
                simple_linked_list_append(receiver->msg_queue, msg);
                return new_response(RESPONSE_ACCEPTED, NULL);
        }
}

struct message *new_message(size_t len)
{
        assert(len >= sizeof(struct message));

        struct message *ret = (struct message *)
                calloc(1, len);

        return ret;
}

void free_message(struct message *msg)
{
        if(msg && msg->data_deleter) {
                msg->data_deleter(msg);
        }
        if(msg) {
                free(msg);
        }
}

static void response_deleter(struct response *response)
{
        free(response->text);
        free(response);
}

/**
 * Creates new response
 *
 * @param status status
 * @param text   optional text contained in message, will be freeed after send (with free())
 */
struct response *new_response(int status, char *text)
{
        struct response *resp = (struct response *) malloc(sizeof(struct response));
        resp->status = status;
        resp->text = text;
        resp->deleter = response_deleter;
        return resp;
}

const char *response_status_to_text(int status)
{
        struct {
                int status;
                const char *text;
        } mapping[] = {
                { 200, "OK" },
                { 202, "Accepted" },
                { 400, "Bad Request" },
                { 404, "Not Found" },
                { 500, "Internal Server Error" },
                { 501, "Not Implemented" },
                { 0, NULL },
        };

        for(int i = 0; mapping[i].status != 0; ++i) {
                if(status == mapping[i].status)
                        return mapping[i].text;
        }

        return NULL;
}

struct message *check_message(struct module *mod)
{
        lock_guard guard(mod->lock);

        if(simple_linked_list_size(mod->msg_queue) > 0) {
                return (struct message *) simple_linked_list_pop(mod->msg_queue);
        } else {
                return NULL;
        }
}


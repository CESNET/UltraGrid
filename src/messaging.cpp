#include "messaging.h"

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "debug.h"
#include "module.h"
#include "utils/list.h"
#include "utils/lock_guard.h"

static struct module *find_child(struct module *node, const char *node_name, int index)
{
        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = (struct module *) simple_linked_list_it_next(&it);
                if(strcasecmp(module_class_name(child->cls), node_name) == 0) {
                        if(index-- == 0) {
                                return child;
                        }
                }
        }
        return NULL;
}

static void get_receiver_index(char *node_str, int *index) {
        *index = 0;
        if(strchr(node_str, '[')) {
                *index = atoi(strchr(node_str, '[') + 1);
                *strchr(node_str, '[') = '\0';
        }
}

struct response *send_message(struct module *root, const char *const_path, struct message *msg)
{
        struct module *receiver = root;
        char *path, *tmp;
        char *item, *save_ptr;
        char buf[1024];

        assert(root != NULL);

        pthread_mutex_lock(&receiver->lock);

        tmp = path = strdup(const_path);
        while ((item = strtok_r(path, ".", &save_ptr))) {
                struct module *old_receiver = receiver;
                int index;
                get_receiver_index(item, &index);
                receiver = find_child(receiver, item, index);
                if(!receiver) {
                        pthread_mutex_unlock(&old_receiver->lock);
                        break;
                }
                pthread_mutex_lock(&receiver->lock);
                pthread_mutex_unlock(&old_receiver->lock);

                path = NULL;

        }
        free(tmp);

        /**
         * @invariant
         * either receiver is NULL or receiver->lock is locked (exactly once)
         */

        if(receiver == NULL) {
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
        if(receiver->msg_callback) {
                lock_guard guard(receiver->lock);
                return receiver->msg_callback(receiver, msg);
        } else {
                return new_response(RESPONSE_NOT_IMPL, NULL);
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


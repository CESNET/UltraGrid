#include "messaging.h"

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "debug.h"
#include "module.h"
#include "utils/list.h"
#include "utils/lock_guard.h"

static struct module *find_child(struct module *node, const char *node_name)
{
        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = (struct module *) simple_linked_list_it_next(&it);
                if(strcasecmp(module_class_name(child->cls), node_name) == 0) {
                        return child;
                }
        }
        return NULL;
}

struct response *send_message(struct module *root, const char *const_path, void *data)
{
        struct module *receiver = root;
        char *path, *tmp;
        char *item, *save_ptr;

        tmp = path = strdup(const_path);
        while ((item = strtok_r(path, ".", &save_ptr))) {
                struct module *old_receiver = receiver;
                pthread_mutex_lock(&old_receiver->lock);
                receiver = find_child(receiver, item);
                pthread_mutex_lock(&receiver->lock);
                pthread_mutex_unlock(&old_receiver->lock);

                path = NULL;

        }
        free(tmp);

        if(receiver == NULL) {
                return new_response(RESPONSE_NOT_FOUND, NULL);
        }

        lock_guard guard(receiver->lock, lock_guard_retain_ownership_t());

        if(receiver->msg_callback == NULL) {
                return new_response(RESPONSE_NOT_IMPL, NULL);
        }

        return receiver->msg_callback(data, receiver);
}

struct response *send_message_to_receiver(struct module *receiver, void *data)
{
        if(receiver->msg_callback) {
                lock_guard guard(receiver->lock);
                return receiver->msg_callback(data, receiver);
        } else {
                return new_response(RESPONSE_NOT_IMPL, NULL);
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
 * @param text   optional text contained in message, will be freeed after send
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


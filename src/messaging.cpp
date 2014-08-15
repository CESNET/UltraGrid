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
        /**
         * @invariant
         * either receiver is NULL or receiver->lock is locked (exactly once)
         */
        char *path, *tmp;
        char *item, *save_ptr;
        tmp = path = strdup(const_path);
        struct module *receiver = root;

        pthread_mutex_lock(&receiver->lock);

        while ((item = strtok_r(path, ".", &save_ptr))) {
                struct module *old_receiver = receiver;

                receiver = get_matching_child(receiver, item);

                if (!receiver) {
                        printf("Receiver %s not yet exists. Message will be delivered "
                                        "when it's created.\n", const_path);
                        //dump_tree(root, 0);
                        
                        struct pair_msg_path *saved_message = (struct pair_msg_path *)
                                malloc(sizeof(struct pair_msg_path));
                        saved_message->msg = msg;
                        memset(saved_message->path, 0, sizeof(saved_message->path));
                        strncpy(saved_message->path, const_path + (item - tmp), sizeof(saved_message->path));

                        simple_linked_list_append(old_receiver->msg_queue_childs, saved_message);
                        pthread_mutex_unlock(&old_receiver->lock);

                        return new_response(RESPONSE_ACCEPTED, NULL);
                }
                pthread_mutex_lock(&receiver->lock);
                pthread_mutex_unlock(&old_receiver->lock);

                path = NULL;
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

void module_check_undelivered_messages(struct module *node)
{
        lock_guard guard(node->lock);

        for(void *it = simple_linked_list_it_init(node->msg_queue_childs); it != NULL; ) {
                struct pair_msg_path *msg = (struct pair_msg_path *) simple_linked_list_it_next(&it);
                struct module *receiver = get_matching_child(node, msg->path);
                if (receiver) {
                        send_message_to_receiver(receiver, msg->msg);
                        simple_linked_list_remove(node->msg_queue_childs, msg);
                        free(msg);
                        // reinit iterator
                        it = simple_linked_list_it_init(node->msg_queue_childs);
                }
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


#include "messaging.h"

#include <condition_variable>
#include <iostream>
#include <list>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "debug.h"
#include "module.h"
#include "utils/list.h"
#include "utils/lock_guard.h"

#define MAX_MESSAGES 100
#define MAX_MESSAGES_FOR_NOT_EXISTING_RECV 10

using namespace std;
using namespace ultragrid;

struct response {
        int status;
        char text[];
};


namespace {
struct responder {
        responder() : received_response(nullptr) {}
        static void receive_response(void *s, struct response *r) {
                (*((shared_ptr<responder> *) s))->receive_response_real(r);
        }
        void receive_response_real(struct response *r) {
                unique_lock<mutex> lk(lock);
                received_response = r;
                lk.unlock();
                cv.notify_one();
        }

        struct response *received_response;
        condition_variable cv;
        mutex lock;
};

struct pair_msg_path {
        struct message *msg;
        char path[];
};
}

void free_message_for_child(void *m, struct response *r) {
        struct pair_msg_path *mp = (struct pair_msg_path *) m;
        free_message(mp->msg, r);
        free(mp);

}

static struct response *send_message_common(struct module *root, const char *const_path, struct message *msg, bool sync, int timeout_ms, int flags)
{
        /**
         * @invariant
         * either receiver is NULL or receiver->lock is locked (exactly once)
         */
        char *path, *tmp;
        char *item, *save_ptr;
        tmp = path = strdup(const_path);
        struct module *receiver = root;
        shared_ptr<struct responder> responder;

        if (sync) {
                msg->send_response = responder::receive_response;
                responder = shared_ptr<struct responder>(new struct responder());
                msg->priv_data = new shared_ptr<struct responder>(responder);
        }

        pthread_mutex_lock(&receiver->lock);

        while ((item = strtok_r(path, ".", &save_ptr))) {
                path = NULL;
                struct module *old_receiver = receiver;

                if (strcmp(item, "root") == 0)
                        continue;

                receiver = get_matching_child(receiver, item);

                if (!receiver) {
                        if (!(flags & SEND_MESSAGE_FLAG_NO_STORE)) {
                                if (!(flags & SEND_MESSAGE_FLAG_QUIET))
                                        printf("Receiver %s does not exist.\n", const_path);
                                //dump_tree(root, 0);
                                if (simple_linked_list_size(old_receiver->msg_queue_childs) > MAX_MESSAGES_FOR_NOT_EXISTING_RECV) {
                                        if (!(flags & SEND_MESSAGE_FLAG_QUIET))
                                                printf("Dropping some old messages for %s (queue full).\n", const_path);
                                        free_message_for_child(simple_linked_list_pop(old_receiver->msg_queue_childs),
                                                        new_response(RESPONSE_NOT_FOUND, "Receiver not found"));
                                }

                                struct pair_msg_path *saved_message = (struct pair_msg_path *)
                                        malloc(sizeof(struct pair_msg_path) + strlen(const_path + (item - tmp)) + 1);
                                saved_message->msg = msg;
                                strcpy(saved_message->path, const_path + (item - tmp));

                                simple_linked_list_append(old_receiver->msg_queue_childs, saved_message);
                                pthread_mutex_unlock(&old_receiver->lock);

                                free(tmp);
                                if (!sync) {
                                        return new_response(RESPONSE_ACCEPTED, "(receiver not yet exists)");
                                } else {
                                        unique_lock<mutex> lk(responder->lock);
                                        responder->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [responder]{return responder->received_response != NULL;});
                                        if (responder->received_response) {
                                                return responder->received_response;
                                        } else {
                                                return new_response(RESPONSE_ACCEPTED, NULL);
                                        }
                                }
                        } else {
                                pthread_mutex_unlock(&old_receiver->lock);
                                free_message(msg, NULL);
                                free(tmp);
                                return new_response(RESPONSE_NOT_FOUND, NULL);
                        }
                }
                pthread_mutex_lock(&receiver->lock);
                pthread_mutex_unlock(&old_receiver->lock);
        }

        free(tmp);

        //pthread_mutex_guard guard(receiver->lock, lock_guard_retain_ownership_t());

        if (simple_linked_list_size(receiver->msg_queue) >= MAX_MESSAGES) {
                struct message *m = (struct message *) simple_linked_list_pop(receiver->msg_queue);
                free_message(m, new_response(RESPONSE_INT_SERV_ERR, "Too much unprocessed messages"));
                printf("Dropping some messages for %s - queue full.\n", const_path);
        }

        simple_linked_list_append(receiver->msg_queue, msg);

        pthread_mutex_unlock(&receiver->lock);

        if (!sync) {
                return new_response(RESPONSE_ACCEPTED, NULL);
        } else {
                unique_lock<mutex> lk(responder->lock);
                responder->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [responder]{return responder->received_response != NULL;});
                if (responder->received_response) {
                        return responder->received_response;
                } else {
                        return new_response(RESPONSE_ACCEPTED, NULL);
                }
        }
}

struct response *send_message(struct module *root, const char *const_path, struct message *msg)
{
        return send_message_common(root, const_path, msg, false, 0, 0);
}

struct response *send_message_sync(struct module *root, const char *const_path, struct message *msg, int timeout_ms, int flags)
{
        return send_message_common(root, const_path, msg, true, timeout_ms, flags);
}

void module_check_undelivered_messages(struct module *node)
{
        pthread_mutex_guard guard(node->lock);

        for(void *it = simple_linked_list_it_init(node->msg_queue_childs); it != NULL; ) {
                struct pair_msg_path *msg = (struct pair_msg_path *) simple_linked_list_it_next(&it);
                struct module *receiver = get_matching_child(node, msg->path);
                if (receiver) {
                        struct response *resp = send_message_to_receiver(receiver, msg->msg);
                        free_response(resp);
                        simple_linked_list_remove(node->msg_queue_childs, msg);
                        free(msg);
                        // reinit iterator
                        it = simple_linked_list_it_init(node->msg_queue_childs);
                }
        }
}

struct response *send_message_to_receiver(struct module *receiver, struct message *msg)
{
        pthread_mutex_guard guard(receiver->lock);
        simple_linked_list_append(receiver->msg_queue, msg);
        return new_response(RESPONSE_ACCEPTED, NULL);
}

struct message *new_message(size_t len)
{
        assert(len >= sizeof(struct message));

        struct message *ret = (struct message *)
                calloc(1, len);

        return ret;
}

/**
 * Frees message
 *
 * Additionally, it enforces user to pass response to be sent to sender.
 */
void free_message(struct message *msg, struct response *r)
{
        if (!msg) {
                return;
        }

        if (r) {
                if (msg->send_response) {
                        msg->send_response(msg->priv_data, r);
                } else {
                        free_response(r);
                }
        }

        if (msg->data_deleter) {
                msg->data_deleter(msg);
        }

        if (msg->priv_data) {
                delete (shared_ptr<struct responder> *) msg->priv_data;
        }

        free(msg);
}

/**
 * Creates new response
 *
 * @param status status
 * @param text   optional text contained in message
 */
struct response *new_response(int status, const char *text)
{
        struct response *resp = (struct response *) malloc(sizeof(struct response) + (text ? strlen(text) : 0) + 1);
        resp->status = status;
        if (text) {
                strcpy(resp->text, text);
        } else {
                resp->text[0] = '\0';
        }
        return resp;
}

void free_response(struct response *r) {
        free(r);
}

int response_get_status(struct response *r) {
        return r->status;
}

const char *response_get_text(struct response *r) {
        return r->text;
}

const char *response_status_to_text(int status)
{
        const static unordered_map<int, const char *> mapping = {
                { RESPONSE_OK, "OK" },
                { RESPONSE_ACCEPTED, "Accepted" },
                { RESPONSE_NO_CONTENT, "No Content" },
                { RESPONSE_BAD_REQUEST, "Bad Request" },
                { RESPONSE_NOT_FOUND, "Not Found" },
                { RESPONSE_REQ_TIMEOUT, "Request Timeout" },
                { RESPONSE_INT_SERV_ERR, "Internal Server Error" },
                { RESPONSE_NOT_IMPL, "Not Implemented" },
        };

        auto it = mapping.find(status);
        if (it != mapping.end()) {
                return it->second;
        }

        return NULL;
}

struct message *check_message(struct module *mod)
{
        pthread_mutex_guard guard(mod->lock);

        if(simple_linked_list_size(mod->msg_queue) > 0) {
                return (struct message *) simple_linked_list_pop(mod->msg_queue);
        } else {
                return NULL;
        }
}


/**
 * @file   messaging.cpp
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * Communication infrastructure for passing messages between individual
 * UltraGrid modules.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "messaging.h"

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "debug.h"
#include "module.h"
#include "utils/list.h"
#include "utils/lock_guard.h"

#define MAX_MESSAGES 100
#define MAX_MESSAGES_FOR_NOT_EXISTING_RECV 10
#define MOD_NAME "[messaging] "

using namespace std;
using namespace ultragrid;

struct response {
        int status;
        char text[];
};


namespace {
struct responder {
        responder() : received_response(nullptr) {}
        ~responder() {
                free_response(received_response);
        }
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

/**
 * Stores message to module message box. If new_message callback is present it is called. Otherwise it
 * may take a long time until module reads the message therefore setting timeout_ms is strongly
 * recommended to prevent freeze (unless knowing that module implements synchronnous message processing
 * via module::new_message callback).
 *
 * If not set otherwise, when receiver doesn't exist, message is stored by nearest existing parent until
 * it starts (this may be disabled with flag SEND_MESSAGE_FLAG_NO_STORE).
 *
 * @param            sync wait for response timeout_ms milliseconds, if false, 202 is immediately returned
 *                   (or 404 if NO_STORE flag is set and receiver doesn't exist)
 * @param timeout_ms if sync==true number of ms to wait for response (-1 means infinitely),
 *                   ignored if sync is false,
 * @params flags     bit mask of SEND_MESSAGE_FLAG_NO_STORE and SEND_MESSAGE_FLAG_QUIET
 */
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
                                        if (timeout_ms == -1) {
                                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: infinite wait for "
                                                                "non-existent recv. Please report!\n");
                                                responder->cv.wait(lk, [responder]{return responder->received_response != NULL;});
                                        } else {
                                                responder->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [responder]{return responder->received_response != NULL;});
                                        }
                                        if (responder->received_response) {
                                                struct response *resp = responder->received_response;
                                                responder->received_response = NULL;
                                                return resp;
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

        pthread_mutex_lock(&receiver->msg_queue_lock);
        int size = simple_linked_list_size(receiver->msg_queue);
        pthread_mutex_unlock(&receiver->msg_queue_lock);
        if (size >= MAX_MESSAGES) {
                struct message *m = (struct message *) simple_linked_list_pop(receiver->msg_queue);
                free_message(m, new_response(RESPONSE_INT_SERV_ERR, "Too many unprocessed messages"));
                printf("Dropping some messages for %s - queue full.\n", const_path);
        }
        pthread_mutex_lock(&receiver->msg_queue_lock);
        simple_linked_list_append(receiver->msg_queue, msg);
        pthread_mutex_unlock(&receiver->msg_queue_lock);

        if (receiver->new_message) {
                receiver->new_message(receiver);
        }

        pthread_mutex_unlock(&receiver->lock);

        if (!sync) {
                return new_response(RESPONSE_ACCEPTED, NULL);
        } else {
                unique_lock<mutex> lk(responder->lock);
                if (timeout_ms == -1) {
                        if (receiver->new_message == NULL) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: infinite wait for "
                                                "module without msg notifier. Please report!\n");
                        }
                        responder->cv.wait(lk, [responder]{return responder->received_response != NULL;});
                } else {
                        responder->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [responder]{return responder->received_response != NULL;});
                }
                if (responder->received_response) {
                        struct response *resp = responder->received_response;
                        responder->received_response = NULL;
                        return resp;
                } else {
                        return new_response(RESPONSE_ACCEPTED, NULL);
                }
        }
}

/** @brief Sends message without waiting for response
 * @copydetails send_message_common
 */
struct response *send_message(struct module *root, const char *const_path, struct message *msg)
{
        return send_message_common(root, const_path, msg, false, 0, 0);
}

/** @brief Sends message without waiting for response
 * @copydetails send_message_common
 */
struct response *send_message_sync(struct module *root, const char *const_path, struct message *msg, int timeout_ms, int flags)
{
        return send_message_common(root, const_path, msg, true, timeout_ms, flags);
}

/** @brief Sends message with waiting for response
 * @copydetails send_message_common
 */
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

void module_store_message(struct module *node, struct message *m)
{
        pthread_mutex_guard guard(node->msg_queue_lock);
        simple_linked_list_append(node->msg_queue, m);
}

struct response *send_message_to_receiver(struct module *receiver, struct message *msg)
{
        pthread_mutex_lock(&receiver->msg_queue_lock);
        simple_linked_list_append(receiver->msg_queue, msg);
        pthread_mutex_unlock(&receiver->msg_queue_lock);

        pthread_mutex_guard guard(receiver->lock);
        if (receiver->new_message) {
                receiver->new_message(receiver);
        }

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
        pthread_mutex_guard guard(mod->msg_queue_lock);

        if(simple_linked_list_size(mod->msg_queue) > 0) {
                return (struct message *) simple_linked_list_pop(mod->msg_queue);
        } else {
                return NULL;
        }
}


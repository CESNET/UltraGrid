#include "messaging.h"

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "utils/lock_guard.h"

using namespace std;

struct responder {
        msg_callback_t callback;
        void *udata;
};

class message_manager {
        public:
                message_manager() {
                        pthread_mutex_init(&lock, NULL);
                }

                virtual ~message_manager() {
                        pthread_mutex_destroy(&lock);
                }

                void register_responder(enum msg_class cls, msg_callback_t callback, void *udata) {
                        lock_guard guard(lock);
                        struct responder *r = new responder;
                        r->callback = callback;
                        r->udata = udata;
                        if(responders.find(cls) == responders.end()) {
                                responders[cls] = list<responder *>();
                        }
                        responders[cls].push_back(r);
                }

                struct response *send(enum msg_class cls, void *data) {
                        lock_guard guard(lock);
                        if(responders.find(cls) == responders.end()) {
                                cerr << "Warning: cannot send message, no receiver registered to "
                                        "the given class!" << endl;
                                return new_response(RESPONSE_INT_SERV_ERR);
                        } else {
                                for(list<responder *>::iterator it = responders[cls].begin();
                                                it != responders[cls].end(); ++it) {
                                        struct received_message *msg = (struct received_message *)
                                                malloc(sizeof(struct received_message));
                                        msg->message_type = cls;
                                        msg->data = data;
                                        msg->deleter = (void (*)(struct received_message *)) free;
                                        struct response *response = (*it)->callback(msg,
                                                        (*it)->udata);
                                        if(response)
                                                return response;
                                }
                                cerr << "Warning: cannot send message, no receiver is responding "
                                        "the given class!" << endl;
                                return new_response(RESPONSE_INT_SERV_ERR);
                        }
                }

        private:
                pthread_mutex_t lock;
                map<enum msg_class, list<responder *> > responders;
};

class message_manager instance;

struct messaging *messaging_instance(void) {
        return (struct messaging*) &instance;
}

void subscribe_messages(struct messaging *state, enum msg_class cls, msg_callback_t callback, void *udata)
{
        class message_manager *s = (class message_manager *) state;
        s->register_responder(cls, callback, udata);
}

struct response *send_message(struct messaging *state, enum msg_class cls, void *data)
{
        class message_manager *s = (class message_manager *) state;
        return s->send(cls, data);
}

struct response *new_response(int status)
{
        struct response *resp = (struct response *) malloc(sizeof(struct response));
        resp->status = status;
        resp->deleter = (void (*)(struct response *)) free;
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
                { 0, NULL },
        };

        for(int i = 0; mapping[i].status != 0; ++i) {
                if(status == mapping[i].status)
                        return mapping[i].text;
        }

        return NULL;
}


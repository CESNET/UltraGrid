#include "messaging.h"

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "debug.h"

#include "utils/lock_guard.h"
#include "utils/resource_manager.h"

static void *init_instance(void *data);
static void delete_instance(void *instance);

using namespace std;

struct responder {
        msg_callback_t callback;
        void *udata;
        enum msg_class cls;
};

class message_manager {
        public:
                message_manager() : m_traversing(0), m_dirty(false)
                {
                        pthread_mutexattr_t attr;
                        assert(pthread_mutexattr_init(&attr) == 0);
                        assert(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) == 0);
                        assert(pthread_mutex_init(&m_lock, &attr) == 0);
                        pthread_mutexattr_destroy(&attr);
                }

                virtual ~message_manager() {
                        pthread_mutex_destroy(&m_lock);
                }

                struct responder *register_responder(enum msg_class cls, msg_callback_t callback, void *udata) {
                        lock_guard guard(m_lock);
                        struct responder *r = new responder;
                        r->callback = callback;
                        r->udata = udata;
                        r->cls = cls;
                        if(responders.find(cls) == responders.end()) {
                                responders[cls] = list<responder *>();
                        }
                        responders[cls].push_back(r);
                        if(m_traversing)
                                m_dirty = true;

                        return r;
                }

                void unregister_responder(struct responder *r) {
                        lock_guard guard(m_lock);
                        responders[r->cls].remove(r);
                        delete r;
                        if(m_traversing)
                                m_dirty = true;
                }

                struct response *send(enum msg_class cls, void *data) {
                        lock_guard guard(m_lock);
                        if(responders.find(cls) == responders.end()) {
                                cerr << "Warning: cannot send message, no receiver registered to "
                                        "the given class!" << endl;
                                return new_response(RESPONSE_INT_SERV_ERR);
                        } else {
                                bool was_dirty = false;
again:
                                for(list<responder *>::iterator it = responders[cls].begin();
                                                it != responders[cls].end(); ++it) {
                                        struct received_message *msg = (struct received_message *)
                                                malloc(sizeof(struct received_message));
                                        msg->message_type = cls;
                                        msg->data = data;
                                        msg->deleter = (void (*)(struct received_message *)) free;

                                        // incrementing only here since this is the only
                                        // non-open call
                                        m_traversing += 1;
                                        struct response *response = (*it)->callback(msg,
                                                        (*it)->udata);
                                        m_traversing -= 1;

                                        if(response)
                                                return response;
                                        if(m_dirty) {
                                                // we need to reset the traversal
                                                m_dirty = false;
                                                was_dirty = true;
                                                goto again;
                                        }
                                }
                                if(m_traversing && was_dirty) {
                                        m_dirty = true;
                                }
                                cerr << "Warning: cannot send message, no receiver is responding "
                                        "the given class!" << endl;
                                return new_response(RESPONSE_INT_SERV_ERR);
                        }
                }

        private:
                int m_traversing;
                bool m_dirty;
                pthread_mutex_t m_lock;
                map<enum msg_class, list<responder *> > responders;
};

static void *init_instance(void *data) {
        UNUSED(data);
        return new message_manager;
}

static void delete_instance(void *instance) {
        delete (message_manager *) instance;
}

struct messaging *messaging_instance(void) {
        return (struct messaging *) rm_singleton("MESSAGING", init_instance, NULL,
                                delete_instance);
}

void *subscribe_messages(struct messaging *state, enum msg_class cls, msg_callback_t callback, void *udata)
{
        class message_manager *s = (class message_manager *) state;
        return s->register_responder(cls, callback, udata);
}

void unsubscribe_messages(struct messaging *state, void *handle)
{
        if(!handle) {
                return;
        }
        class message_manager *s = (class message_manager *) state;
        s->unregister_responder(static_cast<struct responder *>(handle));
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


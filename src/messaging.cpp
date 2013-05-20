#include "messaging.h"

#include <list>
#include <map>
#include <string>

using namespace std;

class message {
        public:
                virtual ~message() {
                }
};

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
                        pthread_mutex_lock(&lock);
                        struct responder *r = new responder;
                        r->callback = callback;
                        r->udata = udata;
                        responders[cls] = r;
                        pthread_mutex_unlock(&lock);
                }

        private:
                pthread_mutex_t lock;
                map<enum msg_class, responder *> responders;
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


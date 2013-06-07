#include "stats.h"

#include <map>
#include <string>

#include "control.h"
#include "compat/platform_spin.h"
#include "debug.h"
#include "messaging.h"
#include "module.h"
#include "utils/lock_guard.h"
#include "utils/resource_manager.h"

#define MAX_ITEMS 100

using namespace std;

struct stats {
        public:
                stats(string name, struct control_state *control)
                : m_control(control), m_name(name), m_val(0) {
                        platform_spin_init(&m_spin);
                        control_add_stats(control, this);
                }

                ~stats() {
                        control_remove_stats(m_control, this);
                        platform_spin_destroy(&m_spin);
                }

                void update_int(int64_t val) {
                        platform_spin_lock(&m_spin);
                        m_val = val;
                        platform_spin_unlock(&m_spin);
                }

                void get_stat(char *buffer, int buffer_len) {
                        platform_spin_lock(&m_spin);
                        snprintf(buffer, buffer_len, "%s %lu", m_name.c_str(), m_val);
                        platform_spin_unlock(&m_spin);
                }

        private:
                string m_name;
                int64_t m_val;
                void *m_messaging_subscribtion;
                struct control_state *m_control;
                platform_spin_t m_spin;
};

struct stats *stats_new_statistics(struct control_state *control, char * name)
{
        return new stats(string(name), control);
}

void stats_update_int(struct stats *s, int64_t val)
{
        return s->update_int(val);
}

void stats_format(struct stats *s, char *buffer, int buffer_len)
{
        s->get_stat(buffer, buffer_len);
}

void stats_destroy(struct stats *s)
{
        delete s;
}


#ifndef stat_h_
#define stat_h_

#include <atomic>
#include <string>
#include <sstream>

#include "control_socket.h"
#include "module.h"

struct stats_reportable {
        virtual std::string get_stat() = 0;
};

template <typename T>
struct stats : public stats_reportable {
        public:
                /**
                 * @param mod should be the module that collect the statistics or its closest
                 * ancestor because it should be used to identify branch for which statistics
                 * are reported.
                 */
                stats(struct module *mod, std::string name)
                : m_name(name), m_val() {
                        m_control = (struct control_state *) get_module(get_root_module(mod), "control");
                        uint32_t id;
                        bool ret = get_port_id(mod, &id);
                        control_add_stats(m_control, this, ret ? id : -1);
                }

                ~stats() {
                        control_remove_stats(m_control, this);
                }

                void update(T val) {
                        m_val = val;
                }

                std::string get_stat() {
                        std::ostringstream oss;
                        oss << m_name << " " << m_val.load();
                        return oss.str();
                }


        private:
                struct control_state *m_control;
                std::string m_name;
                std::atomic<T> m_val;
};

#endif // stat_h_


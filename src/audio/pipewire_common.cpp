#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#include "debug.h"
#include "utils/color_out.h"

#include "pipewire_common.hpp"

static void on_core_done(void *data, uint32_t /*id*/, int seq)
{
        auto s = static_cast<pipewire_state_common *>(data);

        s->pw_last_seq = seq;
        pw_thread_loop_signal(s->pipewire_loop.get(), false);
}

static void on_core_error(void * /*data*/, uint32_t id, int seq, int res, const char *message)
{
        log_msg(LOG_LEVEL_ERROR, "Pipewire error: id:%d seq:%x res:%d msg: %s\n", id, seq, res, message);
}

static const struct pw_core_events core_events = {
        PW_VERSION_CORE_EVENTS,
        .info = nullptr,
        .done = on_core_done,
        .ping = nullptr,
        .error = on_core_error,
        .remove_id = nullptr,
        .bound_id = nullptr,
        .add_mem = nullptr,
        .remove_mem = nullptr,
};

bool initialize_pw_common(pipewire_state_common& s){
        s.init_guard.init();
        s.pipewire_loop.reset(pw_thread_loop_new("Playback", nullptr));
        s.pipewire_context.reset(pw_context_new(pw_thread_loop_get_loop(s.pipewire_loop.get()), nullptr, 0)); //TODO check return
        s.pipewire_core.reset(pw_context_connect(s.pipewire_context.get(), nullptr, 0));

        pw_core_add_listener(s.pipewire_core.get(), &s.core_listener.get(), &core_events, &s);

        pw_thread_loop_start(s.pipewire_loop.get());

        return true;
}

static void on_registry_event_global(void *data, uint32_t /*id*/,
                uint32_t /*permissions*/, const char *type, uint32_t /*version*/,
                const struct spa_dict *props)
{
        auto result = static_cast<std::vector<Pipewire_device> *>(data);
        std::string_view type_sv(type);
        if(type_sv != "PipeWire:Interface:Node")
                return;

        Pipewire_device dev;
        for(unsigned i = 0; i < props->n_items; i++){
                std::string_view key(props->items[i].key);
                std::string_view val(props->items[i].value);

                if(key == "node.description")
                        dev.description = val;
                else if(key == "node.nick")
                        dev.nick = val;
                else if(key == "media.class")
                        dev.media_class = val;
                else if(key == "node.name")
                        dev.name = val;
                else if(key == "object.serial")
                        dev.serial = val;
        }

        result->push_back(std::move(dev));
}

std::vector<Pipewire_device> get_pw_device_list(){
        pipewire_state_common s;
        initialize_pw_common(s);

        pw_registry_uniq registry(pw_core_get_registry(s.pipewire_core.get(), PW_VERSION_REGISTRY, 0));

        const static pw_registry_events registry_events = {
                PW_VERSION_REGISTRY_EVENTS,
                .global = on_registry_event_global,
                .global_remove = nullptr
        };

        std::vector<Pipewire_device> result;
        spa_hook_uniq registry_listener;
        pw_registry_add_listener(registry.get(), &registry_listener.get(), &registry_events, &result);

        pipewire_thread_loop_lock_guard lock(s.pipewire_loop.get());

        s.pw_pending_seq = pw_core_sync(s.pipewire_core.get(), PW_ID_CORE, s.pw_pending_seq);
        int wait_seq = s.pw_pending_seq;

        do{
                pw_thread_loop_wait(s.pipewire_loop.get());
        } while(s.pw_last_seq < wait_seq);

        return result;
}

void print_devices(std::string_view media_class){
        auto devices = get_pw_device_list();

        for(const auto& dev : devices){
                if(dev.media_class == media_class){
                        printf("\t%s: %s (%s)\n", dev.serial.c_str(), dev.description.c_str(), dev.name.c_str());
                }
        }
}



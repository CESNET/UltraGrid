#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#include "debug.h"
#include "utils/color_out.h"

#include "pipewire_common.hpp"

static void on_core_done(void *data, uint32_t id, int seq)
{
        auto s = static_cast<pipewire_state_common *>(data);

        s->pw_last_seq = seq;
        pw_thread_loop_signal(s->pipewire_loop.get(), false);
}

static void on_core_error(void *data, uint32_t id, int seq, int res, const char *message)
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



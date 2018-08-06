#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"

#include "debug.h"
#include "lib_common.h"

#include "video.h"
#include "video_codec.h"

#include "shared_mem_frame.hpp"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_preview{
        Shared_mem shared_mem;
};


static int init(struct module *parent, const char *cfg, void **state){
        UNUSED(parent);
        UNUSED(cfg);

        struct state_preview *s = new state_preview();
        s->shared_mem.create();

        *state = s;

        return 0;
}

static void done(void *state){
        delete (state_preview *) state;
}

static struct video_frame *filter(void *state, struct video_frame *in){
        struct state_preview *s = (state_preview *) state;

        s->shared_mem.put_frame(in);
        
        return in;
}


static const struct capture_filter_info capture_filter_preview = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(preview, &capture_filter_preview, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);


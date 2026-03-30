#include "lib_common.h"
#include "video_compress.h"
#include <oapv/oapv.h>

using std::shared_ptr;

namespace {

struct state_video_compress_oapv {
};

static void* oapv_compress_init(module *parent, const char *)
{
        state_video_compress_oapv *s;
        s = new state_video_compress_oapv();
        return s;
}

static void oapv_compress_push(void *state, shared_ptr<video_frame> frame) {
        
}

static shared_ptr<video_frame> oapv_compress_pop(void *state) {
    return {};
}

static void oapv_compress_done(void  *state)
{
        auto *s = (struct state_video_compress_oapv *) state;
        delete s;
}

static compress_module_info get_openapv_module_info() {
        compress_module_info module_info;
        module_info.name = "openapv";

        return module_info;
}

const struct video_compress_info oapv_info = {
        oapv_compress_init,
        oapv_compress_done,
        NULL,
        NULL,
        NULL,
        NULL,
        oapv_compress_push,
        oapv_compress_pop,
        get_openapv_module_info
};

REGISTER_MODULE(openapv, &oapv_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
#include "lib_common.h"
#include "video_compress.h"
#include <oapv/oapv.h>

using std::shared_ptr;

namespace {

struct state_video_compress_oapv {
};

static void* openapv_compress_init(module *parent, const char *)
{
        state_video_compress_oapv *s;
        s = new state_video_compress_oapv();
        return s;
}

static void openapv_compress_push(void *state, shared_ptr<video_frame> frame) {
        
}

static shared_ptr<video_frame> openapv_compress_pop(void *state) {
    return {};
}

static void openapv_compress_done(void  *state)
{
        auto *s = (struct state_video_compress_oapv *) state;
        delete s;
}

static compress_module_info get_openapv_module_info() {
        compress_module_info module_info;
        module_info.name = "openapv";

        return module_info;
}

const struct video_compress_info openapv_info = {
        openapv_compress_init,
        openapv_compress_done,
        NULL,
        NULL,
        NULL,
        NULL,
        openapv_compress_push,
        openapv_compress_pop,
        get_openapv_module_info
};

REGISTER_MODULE(openapv, &openapv_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
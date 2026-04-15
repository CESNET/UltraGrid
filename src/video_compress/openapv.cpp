#include "lib_common.h"
#include "video_compress.h"
#include <oapv/oapv.h>

using std::shared_ptr;

namespace {

struct state_video_compress_oapv {

        state_video_compress_oapv(module *parent, const char *opts);

        oapve_t id;             // OAPV encoder id
        oapvm_t mid;            // OAPV metadata id
        oapve_cdesc_t   cdsc;   // description for encoder creation
        oapv_bitb_t     bitb;   // bitstream buffer (output)
        oapve_stat_t    stat;   // encoding status (output)

        oapv_frms_t input_frms;      // frames for input
        int num_frames;         // number of frames in an access unit
        int preset_id;          // preset of apv (fastest, fast, medium, slow, placebo)
        int qp;                 // quantization parameter
};

state_video_compress_oapv::state_video_compress_oapv(module *parent, const char *opts)
{
        (void) parent;

        int ret = oapve_param_default(cdsc.param);
        if (OAPV_FAILED(ret)) {
                printf("Failed to get default parameters for OAPV encoder: %d\n", ret);
                throw 1;
        }
}

static void* openapv_compress_init(module *parent, const char *opts)
{
        state_video_compress_oapv *s;

        if (opts && strcmp(opts, "help") == 0) {
                printf("help for openapv encoder\n");
                return INIT_NOERR;
        }

        s = new state_video_compress_oapv(parent, opts);
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
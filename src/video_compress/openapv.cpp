#include "lib_common.h"
#include "utils/misc.h"
#include "video_compress.h"
#include <oapv/oapv.h>

#define MAX_BS_BUF   (128 * 1024 * 1024) // bitstream buffer size (128 MiB)
#define MAX_NUM_FRMS (1) // support only primary frame

using std::shared_ptr;

namespace {

struct state_video_compress_oapv {

        state_video_compress_oapv(module *parent, const char *opts);
        ~state_video_compress_oapv();

        oapve_t id = nullptr;       // OAPV encoder handle
        oapvm_t mid = nullptr;      // OAPV metadata handle
        oapve_cdesc_t cdsc{};       // description used for encoder creation (params, threads, …)
        
        oapv_bitb_t bitb{};         // bitstream buffer (output)
        oapve_stat_t stat{};        // encoding status (output)

        oapv_frms_t input_frm{};    // frame for input
        oapv_imgb_t imgb{};         // planar pixel data of input frame
};

state_video_compress_oapv::state_video_compress_oapv(module *parent, const char *opts)
{
        (void) parent;

        int ret = oapve_param_default(cdsc.param);
        if (OAPV_FAILED(ret)) {
                printf("Failed to get default parameters for OAPV encoder: %d\n", ret);
                throw 1;
        }

        // Parse opts
        cdsc.max_bs_buf_size = MAX_BS_BUF;
        cdsc.max_num_frms = MAX_NUM_FRMS;
        cdsc.threads = OAPV_CDESC_THREADS_AUTO;

        unsigned char *bs_buf = (unsigned char *) malloc(cdsc.max_bs_buf_size);
        if (bs_buf == nullptr) {
                printf("Failed to allocate bitstream buffer\n");
                throw 1;
        }
        bitb.addr = bs_buf;
        bitb.bsize = cdsc.max_bs_buf_size;
}

state_video_compress_oapv::~state_video_compress_oapv() {
        for (int i = 0; i < OAPV_MAX_CC; i++) {
                free(imgb.baddr[i]);
        }

        if (id) {
                oapve_delete(id);
        }
        
        if (mid) {
                oapvm_delete(mid);
        }

        free(bitb.addr);
}

static bool configure_with(struct state_video_compress_oapv *s, struct video_desc desc) {
        s->cdsc.param[0].w = desc.width;
        s->cdsc.param[0].h = desc.height;

        s->cdsc.param[0].rc_type = OAPV_RC_ABR;
        s->cdsc.param[0].qp = OAPVE_PARAM_QP_AUTO;

        s->cdsc.param[0].fps_num = get_framerate_n(desc.fps);
        s->cdsc.param[0].fps_den = get_framerate_d(desc.fps);

        return true;
}

static void* openapv_compress_init(module *parent, const char *opts) {
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

static void openapv_compress_done(void  *state) {
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
#include <oapv/oapv.h>
#include <iostream>

#include "lib_common.h"
#include "utils/misc.h"
#include "video_compress.h"
#include "video_frame.h"
#include "video.h"
#include "utils/video_frame_pool.h"
#include "utils/synchronized_queue.h"

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

        struct video_desc saved_desc{}; // last configured video description

        video_frame_pool pool;
        bool configured = false;

        synchronized_queue<shared_ptr<struct video_frame>, 1> out_queue;
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

        memset(&input_frm, 0, sizeof(input_frm));
        input_frm.num_frms         = MAX_NUM_FRMS;
        input_frm.frm[0].pbu_type  = OAPV_PBU_TYPE_PRIMARY_FRAME;
        input_frm.frm[0].group_id  = 1;
        input_frm.frm[0].imgb      = &imgb;
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

static bool input_buffer_setup(const oapve_cdesc_t *cdsc, oapv_imgb_t *imgb) {
        for (int i = 0; i < OAPV_MAX_CC; i++) {
                // baddr and a are same for input buffer, but we set both for clarity.
                free(imgb->baddr[i]);
                imgb->baddr[i] = nullptr;
                imgb->a[i]     = nullptr;
        }
        memset(imgb, 0, sizeof(*imgb));

        imgb->w[0] = cdsc->param[0].w;
        imgb->h[0] = cdsc->param[0].h;

        imgb->w[1] = imgb->w[2] = cdsc->param[0].w / 2;
        imgb->h[1] = imgb->h[2] = cdsc->param[0].h;
        imgb->np = 3;  // three colour components
        imgb->cs = OAPV_CS_YCBCR422_10LE;

        for (int i = 0; i < imgb->np; i++) {
                imgb->aw[i] = imgb->w[i];
                imgb->ah[i] = 1088;

                imgb->s[i] = imgb->aw[i] * OAPV_CS_GET_BYTE_DEPTH(imgb->cs);
                imgb->e[i] = imgb->ah[i];
                imgb->bsize[i] = imgb->s[i] * imgb->e[i];

                imgb->baddr[i] = malloc(imgb->bsize[i]);
                if (!imgb->baddr[i]) {
                        return false;
                }
                imgb->a[i] = imgb->baddr[i];

                memset(imgb->baddr[i], 0, imgb->bsize[i]);
        }

        return true;
}

static bool configure_with(struct state_video_compress_oapv *s, struct video_desc desc) {
        int ret;

        s->cdsc.param[0].w = desc.width;
        s->cdsc.param[0].h = desc.height;

        s->cdsc.param[0].fps_num = get_framerate_n(desc.fps);
        s->cdsc.param[0].fps_den = get_framerate_d(desc.fps);

        s->cdsc.param[0].rc_type = OAPV_RC_ABR;

        s->cdsc.param[0].profile_idc = OAPV_PROFILE_422_10;

        s->id = oapve_create(&s->cdsc, &ret);
        if (OAPV_FAILED(ret)) {
                printf("Failed to create OAPV encoder: %d\n", ret);
                return false;
        }

        s->mid = oapvm_create(&ret);
        if (OAPV_FAILED(ret)) {
                printf("Failed to create OAPV metadata: %d\n", ret);
                oapve_delete(s->id);
                return false;
        }

        if (!input_buffer_setup(&s->cdsc, &s->imgb)) {
                printf("Failed to set up input buffer\n");
                oapve_delete(s->id);
                oapvm_delete(s->mid);
                return false;
        }

        s->saved_desc.width = desc.width;
        s->saved_desc.height = desc.height;
        s->saved_desc.fps = desc.fps;
        s->saved_desc.interlacing = desc.interlacing;
        s->saved_desc.color_spec = desc.color_spec;
        s->saved_desc.tile_count  = 1;

        s->pool.reconfigure(s->saved_desc, s->cdsc.max_bs_buf_size);

        s->configured = true;

        return true;
}

static void uyvy_to_yuv422p(const uint8_t *uyvy, int width, int height, oapv_imgb_t *in_buf) {
        uint16_t *dst_y = (uint16_t *) in_buf->a[0];
        uint16_t *dst_u = (uint16_t *) in_buf->a[1];
        uint16_t *dst_v = (uint16_t *) in_buf->a[2];

        const int y_stride_px = in_buf->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = in_buf->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = in_buf->s[2] / (int) sizeof(uint16_t);

        for (int y = 0; y < height; ++y) {
                const uint8_t *src_line = uyvy + (size_t) y * width * 2;
                uint16_t *dst_y_line = dst_y + (size_t) y * y_stride_px;
                uint16_t *dst_u_line = dst_u + (size_t) y * u_stride_px;
                uint16_t *dst_v_line = dst_v + (size_t) y * v_stride_px;

                for (int x = 0; x < width; x += 2) {
                        int i = x * 2;
                        uint16_t u = (uint16_t) src_line[i + 0] << 2;
                        uint16_t y0 = (uint16_t) src_line[i + 1] << 2;
                        uint16_t v = (uint16_t) src_line[i + 2] << 2;
                        uint16_t y1 = (uint16_t) src_line[i + 3] << 2;

                        dst_y_line[x + 0] = y0;
                        if (x + 1 < width) {
                                dst_y_line[x + 1] = y1;
                        }

                        int chroma_index = x / 2;
                        dst_u_line[chroma_index] = u;
                        dst_v_line[chroma_index] = v;
                }
        }
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
        state_video_compress_oapv *s = static_cast<state_video_compress_oapv *>(state);
        const auto desc = video_desc_from_frame(frame.get());

        if (!video_desc_eq_excl_param(desc, s->saved_desc, PARAM_INTERLACING)) {
                if (!configure_with(s, desc)) {
                        printf("Failed to configure OpenAPV encoder with new video description\n");
                        return;
                }
        }
        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        uyvy_to_yuv422p((const uint8_t *) in_tile->data, desc.width, desc.height, &s->imgb);

        int ret = oapve_encode(s->id, &s->input_frm, s->mid, &s->bitb, &s->stat, NULL);
        if (OAPV_FAILED(ret)) {
                return;
        }

        shared_ptr<video_frame> out = s->pool.get_frame();
        struct tile *out_tile = vf_get_tile(out.get(), 0);
        memcpy(out_tile->data, s->bitb.addr, s->stat.write);
        out_tile->data_len = s->stat.write;

        vf_copy_metadata(out.get(), frame.get());
        out->compress_end = get_time_in_ns();

        s->out_queue.push(out);
}

static shared_ptr<video_frame> openapv_compress_pop(void *state) {
        auto *s = (struct state_video_compress_oapv *) state;
        return s->out_queue.pop();
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
#include <oapv/oapv.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "openapv/openapv_conversions.h"

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "video_compress.h"
#include "video_frame.h"
#include "video.h"
#include "utils/video_frame_pool.h"
#include "utils/synchronized_queue.h"

using std::shared_ptr;

namespace {

#define MOD_NAME "[OpenAPV enc.] "

#define MAX_BS_BUF   (128 * 1024 * 1024)
#define MAX_NUM_FRMS (1) // support only primary frame
#define FRM_INDEX    (0) // index of primary frame in oapv_frms_t

struct state_video_compress_oapv {

        state_video_compress_oapv(module *parent, const char *opts);
        ~state_video_compress_oapv();

        bool parse_fmt(char *fmt);

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

        void (*convert_to_planar)(const uint8_t *src, int width, int height, oapv_imgb_t *dst) = nullptr;
};

state_video_compress_oapv::state_video_compress_oapv(module *parent, const char *opts)
{
        (void) parent;

        int ret = oapve_param_default(cdsc.param);
        if (OAPV_FAILED(ret)) {
                printf("Failed to get default parameters for OAPV encoder: %d\n", ret);
                throw 1;
        }

        cdsc.max_bs_buf_size = MAX_BS_BUF;
        cdsc.max_num_frms = MAX_NUM_FRMS;
        cdsc.threads = OAPV_CDESC_THREADS_AUTO;
        cdsc.param[0].rc_type = OAPV_RC_ABR;
        cdsc.param[0].qp      = OAPVE_PARAM_QP_AUTO;

        unsigned char *bs_buf = (unsigned char *) malloc(cdsc.max_bs_buf_size);
        if (bs_buf == nullptr) {
                printf("Failed to allocate bitstream buffer\n");
                throw 1;
        }
        bitb.addr = bs_buf;
        bitb.bsize = cdsc.max_bs_buf_size;

        mid = oapvm_create(&ret);
        if (OAPV_FAILED(ret)) {
                printf("Failed to create OAPV metadata: %d\n", ret);
                throw 1;
        }

        memset(&input_frm, 0, sizeof(input_frm));
        input_frm.num_frms         = MAX_NUM_FRMS;
        input_frm.frm[FRM_INDEX].pbu_type  = OAPV_PBU_TYPE_PRIMARY_FRAME;
        input_frm.frm[FRM_INDEX].group_id  = 1;
        input_frm.frm[FRM_INDEX].imgb      = &imgb;

        if (opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(fmt)) {
                        free(fmt);
                        throw 1;
                }
                free(fmt);
        }
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

static bool input_buffer_setup(const oapve_cdesc_t *cdsc, oapv_imgb_t *imgb, int cs) {
        for (int i = 0; i < OAPV_MAX_CC; i++) {
                // baddr and a are same for input buffer
                free(imgb->baddr[i]);
                imgb->baddr[i] = nullptr;
                imgb->a[i]     = nullptr;
        }
        memset(imgb, 0, sizeof(*imgb));

        int bd = OAPV_CS_GET_BYTE_DEPTH(cs);

        imgb->w[0] = cdsc->param[0].w;
        imgb->h[0] = cdsc->param[0].h;

        switch(cs) {
                case OAPV_CS_YCBCR4444_10LE:
                case OAPV_CS_YCBCR4444_12LE:
                        imgb->w[1] = imgb->w[2] = imgb->w[3] = cdsc->param[0].w;
                        imgb->h[1] = imgb->h[2] = imgb->h[3] = cdsc->param[0].h;
                        imgb->np = 4;
                        break;
                case OAPV_CS_YCBCR422_10LE:
                        imgb->w[1] = imgb->w[2] = (cdsc->param[0].w + 1) >> 1;
                        imgb->h[1] = imgb->h[2] = cdsc->param[0].h;
                        imgb->np = 3;
                        break;
                case OAPV_CS_YCBCR444_10LE:
                        imgb->w[1] = imgb->w[2] = cdsc->param[0].w;
                        imgb->h[1] = imgb->h[2] = cdsc->param[0].h;
                        imgb->np = 3;
                        break;
                default:
                        printf("Unsupported color format for input buffer: %d\n", OAPV_CS_GET_FORMAT(cs));
                        return false;
                }

        for(int i = 0; i < imgb->np; i++) {
                // align width and height to macroblock size, and calculate buffer size
                imgb->aw[i] = ((imgb->w[i] + OAPV_MB_W - 1) / OAPV_MB_W) * OAPV_MB_W;
                imgb->ah[i] = ((imgb->h[i] + OAPV_MB_H - 1) / OAPV_MB_H) * OAPV_MB_H;
                imgb->s[i] = imgb->aw[i] * bd;
                imgb->e[i] = imgb->ah[i];

                imgb->bsize[i] = imgb->s[i] * imgb->e[i];
                imgb->a[i] = imgb->baddr[i] = malloc(imgb->bsize[i]);
                if (imgb->baddr[i] == nullptr) {
                        printf("Failed to allocate plane %d for input buffer (%zu bytes).\n", i, (size_t) imgb->bsize[i]);
                        for (int j = 0; j < i; j++) {
                                free(imgb->baddr[j]);
                                imgb->baddr[j] = nullptr;
                                imgb->a[j] = nullptr;
                        }
                        return false;
                }
                memset(imgb->a[i], 0, imgb->bsize[i]);
        }
        imgb->cs = cs;

        return imgb;
}

static int map_color_spaces_to_profiles(int cs) {
        switch (cs) {
                case OAPV_CS_YCBCR4444_10LE:
                        return OAPV_PROFILE_4444_10;
                case OAPV_CS_YCBCR422_10LE:
                        return OAPV_PROFILE_422_10;
                case OAPV_CS_YCBCR4444_12LE:
                        return OAPV_PROFILE_4444_12;
                case OAPV_CS_YCBCR444_10LE:
                        return OAPV_PROFILE_444_10;
                default:
                        return -1;
        }
}

static const struct {
        const char *label;
        const char *key;
        const char *help_name;
        const char *description;
        const char *opt_str;
        bool is_boolean;
        const char *placeholder;
} usage_opts[] = {
        {"Threads", "threads", "threads",
                "\t\tNumber of encoder threads. 0 = auto (default).\n",
                ":threads=", false, "0"
        },
        {"Rate control", "rc", "rc",
                "\t\tRate control type:\n"
                "\t\t 0 = CQP (constant quantization parameter, default)\n"
                "\t\t 1 = ABR (average bitrate)\n",
                ":rc=", false, "0"
        },
        {"Quantization parameter", "qp", "qp",
                "\t\tQuantization parameter. Range: 0~63 (10-bit), 0~75 (12-bit).\n"
                "\t\tUsed with CQP rate control. 255 = auto (default).\n",
                ":qp=", false, "255"
        },
        {"Bit rate", "bitrate", "bitrate",
                "\t\tTarget bitrate in kbps (e.g. 50000). Used with ABR rate control.\n"
                "\t\tAlternatively a unit suffix can be used (e.g. 50M = 50000 kbps).\n",
                ":bitrate=", false, "50000"
        },
        {"Preset", "preset", "preset",
                "\t\tEncoder speed/quality trade-off preset:\n"
                "\t\t 0 = fastest, 1 = fast, 2 = medium (default), 3 = slow, 4 = placebo\n",
                ":preset=", false, "2"
        },
        {"Tile width", "tile_w", "tile_w",
                "\t\tWidth of encoding tile in pixels. Must be a multiple of macroblock\n"
                "\t\twidth (" TOSTRING(OAPV_MB_W) "). 0 = auto (default).\n",
                ":tile_w=", false, "0"
        },
        {"Tile height", "tile_h", "tile_h",
                "\t\tHeight of encoding tile in pixels. Must be a multiple of macroblock\n"
                "\t\theight (" TOSTRING(OAPV_MB_H) "). 0 = auto (default).\n",
                ":tile_h=", false, "0"
        },
        {"Use filler", "use_filler", "use_filler",
                "\t\tInsert filler data to achieve tight constant bitrate (ABR only).\n"
                "\t\t0 = disabled (default), 1 = enabled.\n",
                ":use_filler=", false, "0"
        },
        {"Chroma QP offset 1", "qp_offset_c1", "qp_offset_c1",
                "\t\tQP offset for first chroma channel (Cb). Range: -32~31. Default: 0.\n",
                ":qp_offset_c1=", false, "0"
        },
        {"Chroma QP offset 2", "qp_offset_c2", "qp_offset_c2",
                "\t\tQP offset for second chroma channel (Cr). Range: -32~31. Default: 0.\n",
                ":qp_offset_c2=", false, "0"
        },
        {"Chroma QP offset 3", "qp_offset_c3", "qp_offset_c3",
                "\t\tQP offset for third chroma channel (Cg/A). Range: -32~31. Default: 0.\n",
                ":qp_offset_c3=", false, "0"
        },
};

bool state_video_compress_oapv::parse_fmt(char *fmt)
{
        char *tok = nullptr;
        char *save_ptr = nullptr;

        while ((tok = strtok_r(fmt, ":", &save_ptr)) != nullptr) {
                fmt = nullptr;
                const char *val = strchr(tok, '=');
                if (val == nullptr) {
                        MSG(ERROR, "Missing value for option: %s\n", tok);
                        return false;
                }
                val += 1;

                if (IS_KEY_PREFIX(tok, "threads")) {
                        const int threads = atoi(val);
                        if (threads < 0) {
                                MSG(ERROR, "Invalid number of threads '%s' (must be >= 0, 0 = auto).\n", val);
                                return false;
                        }
                        cdsc.threads = threads;
                } else if (IS_KEY_PREFIX(tok, "rc")) {
                        const int rc = atoi(val);
                        if (rc != OAPV_RC_CQP && rc != OAPV_RC_ABR) {
                                MSG(ERROR, "Invalid rc type '%s' (0 = CQP, 1 = ABR).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].rc_type = rc;
                } else if (IS_KEY_PREFIX(tok, "qp")) {
                        const int qp = atoi(val);
                        if ((qp < 0 || qp > 75) && qp != OAPVE_PARAM_QP_AUTO) {
                                MSG(ERROR, "Invalid QP value '%s' (0~75 or 255 for auto).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].qp = (unsigned char) qp;
                } else if (IS_KEY_PREFIX(tok, "bitrate")) {
                        const char *endptr = nullptr;
                        long long br = unit_evaluate(val, &endptr);
                        if (br == LLONG_MIN || *endptr != '\0' || br <= 0) {
                                MSG(ERROR, "Invalid bitrate value '%s'.\n", val);
                                return false;
                        }
                        // unit_evaluate returns bytes/bits – the value is in kbps here
                        // accept plain numbers as kbps, suffixed values convert from bps to kbps
                        cdsc.param[FRM_INDEX].bitrate = (int)(br / 1000);
                        if (cdsc.param[FRM_INDEX].bitrate == 0) {
                                // plain small number was given, treat as kbps directly
                                cdsc.param[FRM_INDEX].bitrate = (int) br;
                        }
                        cdsc.param[FRM_INDEX].rc_type = OAPV_RC_ABR;
                } else if (IS_KEY_PREFIX(tok, "preset")) {
                        const int preset = atoi(val);
                        if (preset < OAPV_PRESET_FASTEST || preset > OAPV_PRESET_PLACEBO) {
                                MSG(ERROR, "Invalid preset '%s' (0=fastest .. 4=placebo).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].preset = preset;
                } else if (IS_KEY_PREFIX(tok, "tile_w")) {
                        const int tw = atoi(val);
                        if (tw < 0 || (tw > 0 && (tw % OAPV_MB_W) != 0)) {
                                MSG(ERROR, "Invalid tile_w '%s' (must be 0 or a multiple of %d).\n", val, OAPV_MB_W);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].tile_w = tw;
                } else if (IS_KEY_PREFIX(tok, "tile_h")) {
                        const int th = atoi(val);
                        if (th < 0 || (th > 0 && (th % OAPV_MB_H) != 0)) {
                                MSG(ERROR, "Invalid tile_h '%s' (must be 0 or a multiple of %d).\n", val, OAPV_MB_H);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].tile_h = th;
                } else if (IS_KEY_PREFIX(tok, "use_filler")) {
                        const int uf = atoi(val);
                        if (uf != 0 && uf != 1) {
                                MSG(ERROR, "Invalid use_filler '%s' (0 or 1).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].use_filler = uf;
                } else if (IS_KEY_PREFIX(tok, "qp_offset_c1")) {
                        const int v = atoi(val);
                        if (v < -32 || v > 31) {
                                MSG(ERROR, "Invalid qp_offset_c1 '%s' (range: -32~31).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].qp_offset_c1 = (signed char) v;
                } else if (IS_KEY_PREFIX(tok, "qp_offset_c2")) {
                        const int v = atoi(val);
                        if (v < -32 || v > 31) {
                                MSG(ERROR, "Invalid qp_offset_c2 '%s' (range: -32~31).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].qp_offset_c2 = (signed char) v;
                } else if (IS_KEY_PREFIX(tok, "qp_offset_c3")) {
                        const int v = atoi(val);
                        if (v < -32 || v > 31) {
                                MSG(ERROR, "Invalid qp_offset_c3 '%s' (range: -32~31).\n", val);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].qp_offset_c3 = (signed char) v;
                } else {
                        MSG(ERROR, "Unknown option or missing value: %s\n", tok);
                        return false;
                }
        }

        return true;
}

static bool configure_with(struct state_video_compress_oapv *s, struct video_desc desc) {
        int ret;

        const struct uv_to_openapv_conversion* conv_struct = get_uv_to_openapv_conversion(desc.color_spec);
        if (!conv_struct || conv_struct->convert == nullptr) {
                printf("unsupported codec");
                return false;
        }
        s->convert_to_planar = conv_struct->convert;

        s->cdsc.param[FRM_INDEX].w = desc.width;
        s->cdsc.param[FRM_INDEX].h = desc.height;

        s->cdsc.param[FRM_INDEX].fps_num = get_framerate_n(desc.fps);
        s->cdsc.param[FRM_INDEX].fps_den = get_framerate_d(desc.fps);

        s->cdsc.param[FRM_INDEX].profile_idc = map_color_spaces_to_profiles(conv_struct->dst_color_format);

        if (s->id != nullptr) {
                oapve_delete(s->id);
                s->id = nullptr;
        }
        s->id = oapve_create(&s->cdsc, &ret);
        if (OAPV_FAILED(ret)) {
                printf("Failed to create OAPV encoder: %d\n", ret);
                return false;
        }

        int au_bs_fmt = OAPV_CFG_VAL_AU_BS_FMT_NONE;
        int au_bs_fmt_size = sizeof(au_bs_fmt);
        ret = oapve_config(s->id, OAPV_CFG_SET_AU_BS_FMT, &au_bs_fmt, &au_bs_fmt_size);
        if (OAPV_FAILED(ret)) {
                printf("Failed to set OAPV AU bitstream format: %d\n", ret);
                oapve_delete(s->id);
                s->id = nullptr;
                return false;
        }

        if (!input_buffer_setup(&s->cdsc, &s->imgb, conv_struct->dst_color_format)) {
                printf("Failed to set up input buffer\n");
                oapve_delete(s->id);
                s->id = nullptr;
                return false;
        }

        struct video_desc compressed_desc = desc;
        compressed_desc.color_spec  = APV;

        s->pool.reconfigure(compressed_desc, s->cdsc.max_bs_buf_size);

        s->configured = true;
        s->saved_desc = desc;

        return true;
}

static void* openapv_compress_init(module *parent, const char *opts) {
        state_video_compress_oapv *s;

        if (opts && strcmp(opts, "help") == 0) {
                color_printf(TBOLD("OpenAPV") " compression usage:\n");
                color_printf("\t" TBOLD(
                        TRED("-c openapv") "[:threads=<n>][:rc=<0|1>][:qp=<n>][:bitrate=<br>]"
                        "[:preset=<0-4>][:tile_w=<n>][:tile_h=<n>]"
                        "[:use_filler=<0|1>][:qp_offset_c1=<n>][:qp_offset_c2=<n>][:qp_offset_c3=<n>]") "\n");
                color_printf("\t" TBOLD(TRED("-c openapv") ":help") "\n");
                color_printf("\nwhere:\n");
                for (const auto &opt : usage_opts) {
                        color_printf("\t" TBOLD("<%s>") "\n%s\n", opt.key, opt.description);
                }
                printf("\n");
                return INIT_NOERR;
        }

        s = new state_video_compress_oapv(parent, opts);
        return s;
}

static void openapv_compress_push(void *state, shared_ptr<video_frame> frame) {
        state_video_compress_oapv *s = static_cast<state_video_compress_oapv *>(state);
        if (!frame) {
                s->out_queue.push({});
                return;
        }
        const auto desc = video_desc_from_frame(frame.get());

        if (!s->configured || !video_desc_eq_excl_param(desc, s->saved_desc, PARAM_INTERLACING)) {
                if (!configure_with(s, desc)) {
                        printf("Failed to configure OpenAPV encoder with new video description\n");
                        return;
                }
        }
        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        s->convert_to_planar((const uint8_t *) in_tile->data, desc.width, desc.height, &s->imgb);

        s->bitb.ssize = 0;
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

        const auto frame = s->out_queue.pop();
        if (!frame) {
                return nullptr;
        }
        return frame;
}

static void openapv_compress_done(void  *state) {
        auto *s = (struct state_video_compress_oapv *) state;
        delete s;
}

static compress_module_info get_openapv_module_info() {
        compress_module_info module_info;
        module_info.name = "openapv";

        for (const auto &opt : usage_opts) {
                module_info.opts.emplace_back(module_option{opt.label,
                        opt.description, opt.placeholder, opt.key, opt.opt_str, opt.is_boolean});
        }

        codec codec_info;
        codec_info.name = "APV";
        codec_info.priority = 400;
        codec_info.encoders.emplace_back(encoder{"default", ""});
        module_info.codecs.emplace_back(std::move(codec_info));

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
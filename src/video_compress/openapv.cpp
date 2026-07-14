/**
 * @file   video_compress/openapv.cpp
 * @author Juraj Zemančík    <550535@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <oapv/oapv.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "video_compress.h"
#include "video_frame.h"
#include "openapv/openapv_common.hpp"
#include "openapv/to_openapv_conversions.h"
#include "utils/string_view_utils.hpp"
#include "utils/video_frame_pool.h"

using std::shared_ptr;

namespace {

#define MOD_NAME "[OpenAPV enc.] "

#define MAX_NUM_FRMS (1) // support only primary frame
#define FRM_INDEX    (0) // index of primary frame in oapv_frms_t

using oapve_uniq = std::unique_ptr<std::remove_pointer_t<oapve_t>, deleter_from_fcn<oapve_delete>>;

struct state_video_compress_oapv {
        state_video_compress_oapv() = default;

        bool parse_fmt(std::string_view cfg);

        oapve_uniq enc_h;           // OAPV encoder handle
        oapve_cdesc_t cdsc{};       // description used for encoder creation (params, threads, …)
        
        oapv_bitb_t bitb{};         // bitstream buffer (output)
        oapve_stat_t stat{};        // encoding status (output)

        Oapv_Frames input_frms;    // frame for input

        video_desc saved_desc{}; // last configured video description

        video_frame_pool pool;
        bool configured = false;

        uv_to_openapv_conversion_f convert_to_planar = nullptr;
};

int map_color_spaces_to_profiles(int cs) {
        switch (cs) {
                case OAPV_CS_YCBCR4444_10LE:
                        return OAPV_PROFILE_4444_10;
                case OAPV_CS_YCBCR422_10LE:
                        return OAPV_PROFILE_422_10;
                case OAPV_CS_YCBCR4444_12LE:
                        return OAPV_PROFILE_4444_12;
                case OAPV_CS_YCBCR444_10LE:
                        return OAPV_PROFILE_444_10;
                case OAPV_CS_YCBCR444_12LE:
                        return OAPV_PROFILE_444_12;
                default:
                        return -1;
        }
}

const struct {
        const char *label;
        const char *key;
        const char *description;
        const char *opt_str;
        bool is_boolean;
        const char *placeholder;
} usage_opts[] = {
        {"Threads", "threads",
                "\t\tNumber of encoder threads. 0 = auto (default).\n",
                ":threads=", false, "0"
        },
        {"Quantization parameter", "qp",
                "\t\tQuantization parameter. Range: 0~63 (10-bit), 0~75 (12-bit).\n"
                "\t\tUsed with CQP rate control. 255 = auto (default).\n",
                ":qp=", false, "255"
        },
        {"Bit rate", "bitrate",
                "\t\tTarget bitrate in bps. Used with ABR rate control.\n"
                "\t\tSI unit suffix can be used (e.g. 50M = 50000 kbps).\n",
                ":bitrate=", false, "50M"
        },
        {"Preset", "preset",
                "\t\tEncoder speed/quality trade-off preset:\n"
                "\t\tfastest, fast, medium (default), slow, placebo\n",
                ":preset=", false, "medium"
        },
        {"Tile width", "tile-w",
                "\t\tWidth of encoding tile in pixels. Must be a multiple of macroblock\n"
                "\t\twidth (" TOSTRING(OAPV_MB_W) "), minimum is 256.\n",
                ":tile-w=", false, "0"
        },
        {"Tile height", "tile-h",
                "\t\tHeight of encoding tile in pixels. Must be a multiple of macroblock\n"
                "\t\theight (" TOSTRING(OAPV_MB_H) "), minimum is 256.\n",
                ":tile-h=", false, "0"
        },
        {"Chroma QP offset 1", "qp-offset-c1",
                "\t\tQP offset for first chroma channel (Cb). Default: 0.\n",
                ":qp-offset-c1=", false, "0"
        },
        {"Chroma QP offset 2", "qp-offset-c2",
                "\t\tQP offset for second chroma channel (Cr). Default: 0.\n",
                ":qp-offset-c2=", false, "0"
        },
        {"Chroma QP offset 3", "qp-offset-c3",
                "\t\tQP offset for third chroma channel (Cg/A). Default: 0.\n",
                ":qp-offset-c3=", false, "0"
        },
};

bool state_video_compress_oapv::parse_fmt(std::string_view cfg)
{
        while(!cfg.empty()){
                auto tok = tokenize(cfg, ':', '"');
                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(val.empty()){
                        MSG(ERROR, "Missing value for option: %s\n", SV_TO_CSTR(key));
                        return false;
                }

                if(key == "threads"){
                        if(!parse_num(val, cdsc.threads)){
                                MSG(ERROR, "Failed to parse %s\n", SV_TO_CSTR(val));
                                return false;
                        }
                        if (cdsc.threads < 0) {
                                MSG(ERROR, "Invalid number of threads '%s' (must be >= 0, 0 = auto).\n", SV_TO_CSTR(val));
                                return false;
                        }
                } else if(key == "qp"){
                        if(oapve_param_parse(&cdsc.param[FRM_INDEX], "qp", SV_TO_CSTR(val)) != 0){
                                MSG(ERROR, "Invalid QP value '%s' (0~75 or 255 for auto).\n", SV_TO_CSTR(val));
                                return false;
                        }
                } else if(key == "bitrate"){
                        std::string valstr(val);
                        const char *endptr = nullptr;
                        long long rate_bps = unit_evaluate(valstr.c_str(), &endptr);
                        if (rate_bps == LLONG_MIN || *endptr != '\0') {
                                MSG(ERROR, "Invalid bitrate value '%s'.\n", valstr.c_str());
                                return false;
                        }
                        if(rate_bps < 1000){
                                MSG(ERROR, "Specified bitrate %lld is less than 1kbps. Refusing to continue\n", rate_bps);
                                return false;
                        }
                        cdsc.param[FRM_INDEX].bitrate = (int)(rate_bps / 1000);
                } else if(key == "preset"){
                        if (oapve_param_parse(&cdsc.param[FRM_INDEX], "preset", SV_TO_CSTR(val)) != 0){
                                MSG(ERROR, "Invalid preset '%s' (fastest, fast, medium, slow, placebo).\n", SV_TO_CSTR(val));
                                return false;
                        }
                } else if(key == "tile-w"){
                        if (oapve_param_parse(&cdsc.param[FRM_INDEX], "tile-w", SV_TO_CSTR(val)) != 0){
                                MSG(ERROR, "Invalid tile-h '%s' (must be at least %d and a multiple of %d).\n",
                                    SV_TO_CSTR(val), OAPV_MIN_TILE_W, OAPV_MB_W);
                                return false;
                        }
                } else if(key == "tile-h"){
                        if (oapve_param_parse(&cdsc.param[FRM_INDEX], "tile-h", SV_TO_CSTR(val)) != 0) {
                                MSG(ERROR, "Invalid tile-h '%s' (must be at least %d and a multiple of %d).\n",
                                    SV_TO_CSTR(val), OAPV_MIN_TILE_W, OAPV_MB_W);
                                return false;
                        }
                } else if (key == "qp-offset-c1"
                           || key == "qp-offset-c2"
                           || key == "qp-offset-c3")
                {
                        if (oapve_param_parse(&cdsc.param[FRM_INDEX], SV_TO_CSTR(key), SV_TO_CSTR(val)) != 0) {
                                MSG(ERROR, "Invalid %s '%s'.\n", SV_TO_CSTR(key), SV_TO_CSTR(val));
                                return false;
                        }
                } else{
                        if (oapve_param_parse(&cdsc.param[FRM_INDEX], SV_TO_CSTR(key), SV_TO_CSTR(val)) != 0) {
                                MSG(ERROR, "Unknown option or invalid value: %s=%s\n", SV_TO_CSTR(key), SV_TO_CSTR(val));
                                return false;
                        }
                }
        }

        return true;
}

bool configure_with(state_video_compress_oapv *s, video_desc desc){
        const uv_to_openapv_conversion* conv_struct = get_uv_to_openapv_conversion(desc.color_spec);
        if (!conv_struct || conv_struct->convert == nullptr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unsupported codec\n");
                return false;
        }
        s->convert_to_planar = conv_struct->convert;

        s->cdsc.param[FRM_INDEX].w = desc.width;
        s->cdsc.param[FRM_INDEX].h = desc.height;

        s->cdsc.param[FRM_INDEX].fps_num = get_framerate_n(desc.fps);
        s->cdsc.param[FRM_INDEX].fps_den = get_framerate_d(desc.fps);

        s->cdsc.param[FRM_INDEX].profile_idc = map_color_spaces_to_profiles(conv_struct->dst_color_format);

        if (!s->input_frms.configure_with(desc.width, desc.height, conv_struct->dst_color_format)){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set up input buffer\n");
                return false;
        }

        int raw_bytes = 0;
        for (int i = 0; i < s->input_frms.get_primary()->imgb->np; i++) {
                raw_bytes += s->input_frms.get_primary()->imgb->bsize[i];
        }
        // allocate bitstream buffer with size based on raw input size * 2 for safe upper bound
        const int new_buf_size = raw_bytes * 2;

        s->bitb.bsize = new_buf_size;
        s->cdsc.max_bs_buf_size = new_buf_size;

        s->enc_h.reset();
        int ret;
        s->enc_h.reset(oapve_create(&s->cdsc, &ret));
        if (OAPV_FAILED(ret)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create OAPV encoder: %d\n", ret);
                return false;
        }

        int au_bs_fmt = OAPV_CFG_VAL_AU_BS_FMT_NONE;
        int au_bs_fmt_size = sizeof(au_bs_fmt);
        ret = oapve_config(s->enc_h.get(), OAPV_CFG_SET_AU_BS_FMT, &au_bs_fmt, &au_bs_fmt_size);
        if (OAPV_FAILED(ret)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set OAPV AU bitstream format: %d\n", ret);
                s->enc_h = nullptr;
                return false;
        }

        video_desc compressed_desc = desc;
        compressed_desc.color_spec  = APV;

        s->pool.reconfigure(compressed_desc, s->cdsc.max_bs_buf_size);

        s->configured = true;
        s->saved_desc = desc;

        return true;
}

shared_ptr<video_frame> openapv_compress_tile(void *state, shared_ptr<video_frame> tile) {
        auto *s = static_cast<state_video_compress_oapv *>(state);

        if (!tile) {
                return {};
        }

        const auto desc = video_desc_from_frame(tile.get());
        if (!s->configured || !video_desc_eq_excl_param(desc, s->saved_desc, PARAM_INTERLACING)) {
                if (!configure_with(s, desc)) {
                        MSG(ERROR, "Failed to configure OpenAPV encoder with new video description\n");
                        return {};
                }
        }

        struct tile *in_tile = vf_get_tile(tile.get(), 0);
        s->convert_to_planar((const uint8_t *) in_tile->data, desc.width, desc.height, s->input_frms.get_primary()->imgb);

        shared_ptr<video_frame> out = s->pool.get_frame();
        s->bitb.ssize = 0;
        s->bitb.addr = out->tiles[0].data;
        int ret = oapve_encode(s->enc_h.get(), s->input_frms.get_frms(), nullptr, &s->bitb, &s->stat, nullptr);
        if (OAPV_FAILED(ret)) {
                MSG(ERROR, "oapve_encode failed: %s (%d) (frame dropped)\n", oapv_err_str(ret), ret);
                return {};
        }
        out->tiles[0].data_len = s->stat.write;

        vf_copy_metadata(out.get(), tile.get());

        return out;
}

void openapv_print_help(){
        color_printf(TBOLD("OpenAPV") " compression usage:\n");
        color_printf("\t" TBOLD(
                TRED("-c openapv") "[:qp=<n>|:bitrate=<br>][:threads=<n>]"
                "[:preset=<preset>][:tile-w=<n>][:tile-h=<n>]"
                "[:qp-offset-c1=<n>][:qp-offset-c2=<n>][:qp-offset-c3=<n>]") "\n");
        color_printf("\t" TBOLD(TRED("-c openapv") ":help") "\n");
        color_printf("\nwhere:\n");
        for (const auto &opt : usage_opts) {
                color_printf("\t" TBOLD("<%s>") "\n%s\n", opt.key, opt.description);
        }

        color_printf_wrapped("Other codec options like "
                TBOLD("color-primaries") ", " TBOLD("color-transfer") ", "
                TBOLD("color-matrix") ", " TBOLD("color-range") " and others "
                "are also allowed and are passed through to the codec. "
                "Example: " TBOLD("-c openapv:color-range=full") ".\n");

        color_printf("\n");
}

void* openapv_compress_init(module */*parent*/, const char *opts) {
        if (opts && strcmp(opts, "help") == 0) {
                openapv_print_help();
                return INIT_NOERR;
        }

        auto s = std::make_unique<state_video_compress_oapv>();

        int ret = oapve_param_default(s->cdsc.param);
        if (OAPV_FAILED(ret)) {
                log_msg(LOG_LEVEL_FATAL, MOD_NAME "Failed to get default parameters for OAPV encoder: %d\n", ret);
                return nullptr;
        }

        s->cdsc.param->qp = OAPVE_PARAM_QP_AUTO;
        s->cdsc.param->rc_type = OAPV_RC_ABR;
        s->cdsc.param->bitrate = 0;

        s->cdsc.max_num_frms = MAX_NUM_FRMS;

        if (opts && opts[0] != '\0') {
                if (!s->parse_fmt(opts)) {
                        return nullptr;
                }
        }

        if(s->cdsc.param->bitrate != 0 && s->cdsc.param->rc_type == OAPV_RC_CQP){
                log_msg(LOG_LEVEL_FATAL, MOD_NAME "QP and bitrate can't be set at the same time!\n");
                return nullptr;
        }

        return s.release();
}

void openapv_compress_done(void  *state) {
        auto s = static_cast<state_video_compress_oapv *>(state);
        delete s;
}

compress_module_info get_openapv_module_info() {
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

constexpr video_compress_info openapv_info = {
        openapv_compress_init,
        openapv_compress_done,
        nullptr,
        openapv_compress_tile,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        get_openapv_module_info
};

REGISTER_MODULE(openapv, &openapv_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}

/**
 * @file   video_decompress/openapv.cpp
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

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "from_planar.h"
#include "openapv/from_openapv_conversions.h"

namespace {

#define MOD_NAME "[OpenAPV dec.] "

#define MAX_NUM_FRMS (1) // support only primary frame
#define FRM_INDEX    (0) // index of primary frame in oapv_frms_t

struct state_video_decompress_openapv {
        state_video_decompress_openapv();
        ~state_video_decompress_openapv();
        
        oapvd_t decoder_handle{};
        oapvd_cdesc_t cdesc{};

        oapv_frms_t decoded_frames{};
        oapv_bitb_t input_buffer{};

        bool configured = false;

        const from_openapv_conversion *convert_from_planar = nullptr;

        struct video_desc desc{};
        int pitch = 0;
        codec_t out_codec = VIDEO_CODEC_NONE;
};

state_video_decompress_openapv::state_video_decompress_openapv() {
        int ret = -1;

        cdesc.threads = OAPV_CDESC_THREADS_AUTO;
        decoder_handle = oapvd_create(&cdesc, &ret);
        if (OAPV_FAILED(ret)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "oapvd_create failed (ret=%d).\n", ret);
                throw 1;
        }

        decoded_frames.num_frms = MAX_NUM_FRMS;
        decoded_frames.frm[FRM_INDEX].imgb = new oapv_imgb_t();
        memset(decoded_frames.frm[FRM_INDEX].imgb, 0, sizeof(oapv_imgb_t));
}

state_video_decompress_openapv::~state_video_decompress_openapv() {
        if (decoder_handle) {
                oapvd_delete(decoder_handle);
        }
        for (int i = 0; i < decoded_frames.num_frms; ++i) {
                if (decoded_frames.frm[i].imgb) {
                        for (int j = 0; j < OAPV_MAX_CC; ++j) {
                                free(decoded_frames.frm[i].imgb->baddr[j]);
                        }
                        delete decoded_frames.frm[i].imgb;
                }
        }
}

static bool output_buffer_setup(oapv_imgb_t *imgb, const oapv_frm_info_t *frm_info) {
        for (int i = 0; i < OAPV_MAX_CC; ++i) {
                free(imgb->baddr[i]);
        }
        memset(imgb, 0, sizeof(*imgb));
        imgb->cs = frm_info->cs;
        imgb->w[0] = frm_info->w;
        imgb->h[0] = frm_info->h;
        int cf = OAPV_CS_GET_FORMAT(frm_info->cs);
        switch(cf) {
                case OAPV_CF_YCBCR422:
                        imgb->w[1] = imgb->w[2] = (frm_info->w + 1) / 2;
                        imgb->h[1] = imgb->h[2] = frm_info->h;
                        imgb->np = 3;
                        break;
                case OAPV_CF_YCBCR444:
                        imgb->w[1] = imgb->w[2] = frm_info->w;
                        imgb->h[1] = imgb->h[2] = frm_info->h;
                        imgb->np = 3;
                        break;
                case OAPV_CF_YCBCR4444:
                        imgb->w[1] = imgb->w[2] = imgb->w[3] = frm_info->w;
                        imgb->h[1] = imgb->h[2] = imgb->h[3] = frm_info->h;
                        imgb->np = 4;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported color format for input buffer: %d\n", cf);
                        return false;
        }

        for (int i = 0; i < imgb->np; ++i) {
                imgb->aw[i] = ((imgb->w[i] + OAPV_MB_W - 1) / OAPV_MB_W) * OAPV_MB_W;
                imgb->ah[i] = ((imgb->h[i] + OAPV_MB_H - 1) / OAPV_MB_H) * OAPV_MB_H;
                imgb->s[i] = imgb->aw[i] * OAPV_CS_GET_BYTE_DEPTH(imgb->cs);
                imgb->e[i] = imgb->ah[i];
                imgb->bsize[i] = imgb->s[i] * imgb->e[i];
                imgb->baddr[i] = malloc(imgb->bsize[i]);
                if (imgb->baddr[i] == nullptr) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate plane %d for frame %d (%zu bytes).\n",
                                i, FRM_INDEX, (size_t) imgb->bsize[i]);
                        for (int j = 0; j < i; ++j) {
                                free(imgb->baddr[j]);
                        }
                        return false;
                }
                memset(imgb->baddr[i], 0, imgb->bsize[i]);
                imgb->a[i] = imgb->baddr[i];
        }
        return true;
}

static bool configure_with(struct state_video_decompress_openapv *s, unsigned char *bitstream_buffer, size_t codestream_size) {
        int ret = -1;
        oapv_au_info_t aui{};

        ret = oapvd_info(bitstream_buffer, codestream_size, &aui);
        if (OAPV_FAILED(ret) || aui.num_frms != 1) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get APV info (ret=%d, num_frms=%d).\n", ret, aui.num_frms);
                return false;
        }

        for (int i = 0; i < OAPV_MAX_CC; ++i) {
                free(s->decoded_frames.frm[FRM_INDEX].imgb->baddr[i]);
        }

        const oapv_frm_info_t &frm_info = aui.frm_info[FRM_INDEX];
        int fmt = OAPV_CS_GET_FORMAT(frm_info.cs);
        if (fmt != OAPV_CF_YCBCR422 && fmt != OAPV_CF_YCBCR444 && fmt != OAPV_CF_YCBCR4444) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported APV color format %d for frame %d.\n", fmt, FRM_INDEX);
                return false;
        }

        oapv_imgb_t *imgb = s->decoded_frames.frm[FRM_INDEX].imgb;
        if (!output_buffer_setup(imgb, &frm_info)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set up output buffer for frame %d.\n", FRM_INDEX);
                return false;
        }

        s->decoded_frames.frm[FRM_INDEX].pbu_type = frm_info.pbu_type;
        s->decoded_frames.frm[FRM_INDEX].group_id = frm_info.group_id;

        const from_openapv_conversion *conv = get_from_openapv_conversion(s->out_codec, frm_info.cs);
        if (conv == nullptr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported out_codec=%s for stream cs=0x%x.\n",
                        get_codec_name(s->out_codec), frm_info.cs);
                return false;
        }

        s->convert_from_planar = conv;
        s->configured = true;

        return true;
}

static void openapv_to_uv_convert(struct state_video_decompress_openapv *s,
                     const oapv_imgb_t *src, uint8_t *dst) {
        struct from_planar_data d = {};
        d.width = src->w[0];
        d.height = src->h[0];
        d.out_data = dst;
        d.out_pitch = s->pitch;
        d.in_data[0] = (const unsigned char *) src->a[0];
        d.in_data[1] = (const unsigned char *) src->a[1];
        d.in_data[2] = (const unsigned char *) src->a[2];
        d.in_data[3] = (const unsigned char *) src->a[3];
        d.in_linesize[0] = src->s[0];
        d.in_linesize[1] = src->s[1];
        d.in_linesize[2] = src->s[2];
        d.in_linesize[3] = src->s[3];
        d.in_depth = OAPV_CS_GET_BIT_DEPTH(src->cs);
        d.log2_chroma_h = 0;
        decode_planar_parallel(s->convert_from_planar->convert, d, FROM_PLANAR_THREADS_AUTO);
}

static void *openapv_decompress_init(void) {
        state_video_decompress_openapv *s = new state_video_decompress_openapv();
        return s;
}

static int openapv_decompress_reconfigure(void *state, struct video_desc desc,
        int rshift, int gshift, int bshift, int pitch, codec_t out_codec) {
        state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;
        (void) rshift; (void) gshift; (void) bshift;

        if (s->out_codec == out_codec &&
                s->pitch == pitch &&
                video_desc_eq_excl_param(s->desc, desc, PARAM_INTERLACING)) {
                return true;
        }

        s->desc = desc;
        s->pitch = pitch;
        s->out_codec = out_codec;
        s->configured = false;
        s->convert_from_planar = nullptr;

        return true;
}

static decompress_status openapv_probe_internal_codec(struct pixfmt_desc *internal_prop,
        unsigned char *buffer, size_t buffer_size) {
        oapv_au_info_t aui{};
        if (OAPV_FAILED(oapvd_info(buffer, buffer_size, &aui)) || aui.num_frms < 1) {
                return DECODER_NO_FRAME;
        }
        int frm_idx = 0;
        for (int i = 0; i < aui.num_frms; ++i) {
                if (aui.frm_info[i].pbu_type == OAPV_PBU_TYPE_PRIMARY_FRAME) {
                        frm_idx = i;
                        break;
                }
        }
        *internal_prop = {};
        internal_prop->depth = aui.frm_info[frm_idx].bit_depth;
        switch (OAPV_CS_GET_FORMAT(aui.frm_info[frm_idx].cs)) {
                case OAPV_CF_YCBCR444:  internal_prop->subsampling = SUBS_444;  break;
                case OAPV_CF_YCBCR4444: internal_prop->subsampling = SUBS_4444; break;
                case OAPV_CF_YCBCR422:
                default:                internal_prop->subsampling = SUBS_422;  break;
        }
        internal_prop->rgb = false;
        return DECODER_GOT_CODEC;
}

static decompress_status openapv_decompress(void *state, unsigned char *dst, unsigned char *buffer,
        unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop) {
        UNUSED(frame_seq);
        UNUSED(callbacks);
        state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;

        if (s->out_codec == VIDEO_CODEC_NONE) {
                return openapv_probe_internal_codec(internal_prop, buffer, src_len);
        }

        if (!s->configured) {
                if (!configure_with(s, buffer, src_len)) {
                        return DECODER_NO_FRAME;
                }
        }

        s->input_buffer.addr = buffer;
        s->input_buffer.ssize = src_len;
        s->input_buffer.bsize = src_len;

        oapvd_stat_t stat{};
        int ret = oapvd_decode(s->decoder_handle, &s->input_buffer, &s->decoded_frames, NULL, &stat);
        if (OAPV_FAILED(ret)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "oapvd_decode failed ret=%d src_len=%u.\n", ret, src_len);
                return DECODER_NO_FRAME;
        }

        oapv_imgb_t *imgb = s->decoded_frames.frm[FRM_INDEX].imgb;
        if (imgb->cs != s->convert_from_planar->required_src_cs) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Decoded colorspace 0x%x does not match required 0x%x for output codec %s.\n",
                        imgb->cs, s->convert_from_planar->required_src_cs, get_codec_name(s->out_codec));
                return DECODER_NO_FRAME;
        }

        openapv_to_uv_convert(s, imgb, dst);

        return DECODER_GOT_FRAME;
}

static int openapv_decompress_get_property(void *state, int property, void *val, size_t *len) {
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = false;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = false;
                                *len = sizeof(int);
                                ret = true;
                        }
                        break;
                default:
                        ret = false;
        }

        return ret;
}

static void openapv_decompress_done(void *state) {
       state_video_decompress_openapv *s = (state_video_decompress_openapv *) state;
       delete s;
}

static int openapv_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        if (compression != APV) {
                return VDEC_PRIO_NA;
        }
        if (ugc == VIDEO_CODEC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }

        // Build the OAPV color space from the probed internal pixfmt so we
        // can look up the dispatch table by (ugc, src_cs) — multiple entries
        // can share the same ugc (e.g. Y416 from 444_10 / 4444_10 / 444_12).
        int oapv_cf = 0;
        switch (internal.subsampling) {
                case SUBS_422:  oapv_cf = OAPV_CF_YCBCR422;  break;
                case SUBS_444:  oapv_cf = OAPV_CF_YCBCR444;  break;
                case SUBS_4444: oapv_cf = OAPV_CF_YCBCR4444; break;
                default:        return VDEC_PRIO_NA;
        }
        const int src_cs = OAPV_CS_SET(oapv_cf, internal.depth, 0);

        return get_from_openapv_conversion(ugc, src_cs) != nullptr
                       ? VDEC_PRIO_PREFERRED
                       : VDEC_PRIO_NA;
}

static const struct video_decompress_info openapv_info = {
        openapv_decompress_init,
        openapv_decompress_reconfigure,
        openapv_decompress,
        openapv_decompress_get_property,
        openapv_decompress_done,
        openapv_decompress_get_priority,
};

REGISTER_MODULE(openapv, &openapv_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

}

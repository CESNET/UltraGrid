/**
 * @file   vo_postprocess/double-framerate.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file contains multiple temporal deinterlacers (doubling frame-rate):
 * 1. double-framerate
 * 2. bob
 * 2. linear
 */
/*
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <chrono>
#include <stdlib.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/text.h" // indent_paragraph
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

#define MOD_NAME "[double_framerate] "
#define TIMEOUT "20ms"
#define DFR_DEINTERLACE_IMPOSSIBLE_MSG_ID 0x27ff0a78

enum algo { DF, BOB, LINEAR };

struct state_df {
        enum algo algo;
        struct video_frame *in;
        char *buffers[2];
        int buffer_current;
        bool deinterlace;
        bool nodelay;

        std::chrono::steady_clock::time_point frame_received;
};

static void df_usage()
{
        char desc[] = TBOLD("double-framerate") " is an interleaver that "
                "creates creates a progressive stream by consecutive "
                "interleaving fields, eg. f1f2f3f4f5f6 -> \"f1-- f1f2 f3f2 "
                "f3f4 f5f4 f5f6 --f6\". So saw-like artifacts will still occur "
                "and blending can be used.\n\n";
        color_printf(indent_paragraph(desc));
        color_printf("Usage:\n");
        color_printf("\t" TBOLD(TRED("-p double_framerate") "[:d][:nodelay]") "\n");
        color_printf("\nwhere:\n");
        color_printf("\t" TBOLD("d      ") " - blend the output\n");
        color_printf("\t" TBOLD("nodelay") " - do not delay the other frame to keep timing. Both frames are output in burst. May not work correctly (depends on display).\n");
}

static void * init_common(enum algo algo, const char *config) {
        bool deinterlace = false;
        bool nodelay = false;

        if (strcmp(config, "d") == 0) {
                deinterlace = true;
        } else if (strcmp(config, "nodelay") == 0) {
                nodelay = true;
        } else if (strlen(config) > 0) {
                log_msg(LOG_LEVEL_ERROR, "Unknown config: %s\n", config);
                return NULL;
        }

        struct state_df *s = new state_df{};
        assert(s != NULL);
        s->algo = algo;

        s->in = vf_alloc(1);
        s->buffers[0] = s->buffers[1] = NULL;
        s->buffer_current = 0;
        s->deinterlace = deinterlace;
        s->nodelay = nodelay;

        if (s->nodelay && commandline_params.find("decoder-drop-policy") == commandline_params.end()) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "nodelay option used, setting drop policy to %s timeout.\n", TIMEOUT);
                commandline_params["decoder-drop-policy"] = TIMEOUT;
        }

        return s;
}

static void * df_init(const char *config) {
        if (strcmp(config, "help") == 0) {
                df_usage();
                return NULL;
        }
        return init_common(DF, config);
}

static void * bob_init(const char *config) {
        if (strcmp(config, "help") == 0) {
                color_printf("Deinterlacer " TBOLD("bob") " is a simple doubler. It doubles "
                                "every field line to achieve full resolution.\n\n");
                color_printf("Usage:\n");
                color_printf("\t" TBOLD(TRED("-p deinterlace_bob")) "\n");
                return NULL;
        }
        return init_common(BOB, config);
}

static void * linear_init(const char *config) {
        if (strcmp(config, "help") == 0) {
                color_printf(TBOLD("Linear") " deinterlacer doubles every field to "
                                "full resolution by interpolating missing lines.\n\n");
                color_printf("Usage:\n");
                color_printf("\t" TBOLD(TRED("-p deinterlace_linear")) "\n");
                return NULL;
        }
        return init_common(LINEAR, config);
}

static bool common_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

static int common_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_df *s = (struct state_df *) state;
        struct tile *in_tile = vf_get_tile(s->in, 0);

        free(s->buffers[0]);
        free(s->buffers[1]);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;
        if(desc.interlacing != INTERLACED_MERGED) {
                log_msg(LOG_LEVEL_ERROR, "[Double Framerate] Warning: %s video detected. This filter is intended "
                               "mainly for interlaced merged video. The result might be incorrect.\n",
                               get_interlacing_description(desc.interlacing)); 
        }

        in_tile->width = desc.width;
        in_tile->height = desc.height;

        in_tile->data_len = vc_get_linesize(desc.width, desc.color_spec) *
                desc.height;

        s->buffers[0] = (char *) malloc(in_tile->data_len);
        s->buffers[1] = (char *) malloc(in_tile->data_len);
        in_tile->data = s->buffers[s->buffer_current];
        
        return TRUE;
}

static struct video_frame * common_getf(void *state)
{
        struct state_df *s = (struct state_df *) state;

        s->buffer_current = (s->buffer_current + 1) % 2;
        s->in->tiles[0].data = s->buffers[s->buffer_current];

        return s->in;
}

/// @todo perhaps suboptimal
static void perform_df(struct state_df *s, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        if(in != NULL) {
                char *src = s->buffers[(s->buffer_current + 1) % 2] + vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                char *dst = out->tiles[0].data + req_pitch;
                for (unsigned y = 0; y < out->tiles[0].height; y += 2) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += 2 * req_pitch;
                        src += 2 * vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
                src = s->buffers[s->buffer_current];
                dst = out->tiles[0].data;
                for (unsigned y = 1; y < out->tiles[0].height; y += 2) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += 2 * req_pitch;
                        src += 2 * vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
        } else {
                char *src = s->buffers[s->buffer_current];
                char *dst = out->tiles[0].data;
                for (unsigned y = 0; y < out->tiles[0].height; ++y) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += req_pitch;
                        src += vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
        }

        if (s->deinterlace) {
                if (!vc_deinterlace_ex(out->color_spec,
                                (unsigned char *) out->tiles[0].data, vc_get_linesize(out->tiles[0].width, out->color_spec),
                                (unsigned char *) out->tiles[0].data, vc_get_linesize(out->tiles[0].width, out->color_spec),
                                out->tiles[0].height)) {
                         log_msg_once(LOG_LEVEL_ERROR, DFR_DEINTERLACE_IMPOSSIBLE_MSG_ID, MOD_NAME "Cannot deinterlace, unsupported pixel format '%s'!\n", get_codec_name(out->color_spec));
                }
        }
}

static void perform_bob(struct state_df *s, struct video_frame *in, struct video_frame *out, int pitch)
{
        int linesize = vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
        char *dst = out->tiles[0].data;
        char *src = s->buffers[s->buffer_current] + (in ? 0 : linesize);
        if (in == NULL) {
                memcpy(dst, src, linesize); // copy first line up
                dst += pitch; // then copy every even line to subsequent odd line
        }
        unsigned y = in ? 0 : 1;
        for ( ; y < out->tiles[0].height - 1; y += 2) {
                memcpy(dst, src, linesize);
                memcpy(dst + pitch, src, linesize);
                dst += 2 * pitch;
                src += 2 * linesize;
        }
        if (y < out->tiles[0].height) {
                memcpy(dst, dst - pitch, linesize);
        }
}

/// copied from vc_deinterlace_ex
///
/// consider merging with vc_deinterlace_ex but perhaps not needed (that func
/// has slightly different structure - always averages 2 adjacent lines and
/// writes the result twice to those. This version also uses GCC generic
/// vectorization support instead of SSE (performs fast if vector_size==16)
static bool avg_lines(codec_t codec, size_t linesize, char *src1, char *src2, char *dst)
{
        char *s1 = (char *) src1;
        char *s2 = (char *) src2;
        char *d = (char *) dst;
        if (is_codec_opaque(codec) && codec_is_planar(codec)) {
                return false;
        }
        int bpp = get_bits_per_component(codec);
        if (bpp == 8 || bpp == 16) {
                size_t x = 0;
                if (bpp == 8) {
                        typedef unsigned char v16uc __attribute__ ((vector_size (16)));
                        for ( ; x < linesize / 16; ++x) {
                                v16uc i1, i2;
                                memcpy(&i1, s1, sizeof i1);
                                memcpy(&i2, s2, sizeof i2);
                                v16uc res = ((i1 / 2) + (i2 / 2) + (i1 % 2 + i1 % 2) / 2);
                                memcpy(d, &res, sizeof res);
                                s1 += 16;
                                s2 += 16;
                                d += 16;
                        }
                } else {
                        typedef unsigned short v16us __attribute__ ((vector_size (16)));
                        for ( ; x < linesize / 16; ++x) {
                                v16us i1, i2;
                                memcpy(&i1, s1, sizeof i1);
                                memcpy(&i2, s2, sizeof i2);
                                v16us res = ((i1 / 2) + (i2 / 2) + (i1 % 2 + i1 % 2) / 2);
                                memcpy(d, &res, sizeof res);
                                s1 += 16;
                                s2 += 16;
                                d += 16;
                        }
                }
                x *= 16;
                if (bpp  == 8) {
                        for ( ; x < linesize; ++x) {
                                *d++ = (*s1++ + *s2++ + 1) >> 1;
                        }
                } else {
                        uint16_t *d16 = (uint16_t *) d;
                        uint16_t *s16_1 = (uint16_t *) s1;
                        uint16_t *s16_2 = (uint16_t *) s2;
                        for ( ; x < linesize / 2; ++x) {
                                *d16++ = (*s16_1++ + *s16_2++ + 1) >> 1;
                        }
                }
        } else if (codec == v210) {
                uint32_t *s32_1 = (uint32_t *) s1;
                uint32_t *s32_2 = (uint32_t *) s2;
                uint32_t *d32 = (uint32_t *) d;
                for (size_t x = 0; x < linesize / 16; ++x) {
                        #pragma GCC unroll 4
                        for (int y = 0; y < 4; ++y) {
                                uint32_t v1 = *s32_1++;
                                uint32_t v2 = *s32_2++;
                                *d32++ =
                                        (((v1 >> 20        ) + (v2 >> 20        ) + 1) / 2) << 20 |
                                        (((v1 >> 10 & 0x3ff) + (v2 >> 10 & 0x3ff) + 1) / 2) << 10 |
                                        (((v1       & 0x3ff) + (v2       & 0x3ff) + 1) / 2);
                        }
                }
        } else if (codec == R10k) {
                uint32_t *s32_1 = (uint32_t *) s1;
                uint32_t *s32_2 = (uint32_t *) s2;
                uint32_t *d32 = (uint32_t *) d;
                for (size_t x = 0; x < linesize / 4; ++x) {
                        #pragma GCC unroll 4
                        for (int y = 0; y < 4; ++y) {
                                uint32_t v1 = ntohl(*s32_1++);
                                uint32_t v2 = ntohl(*s32_2++);
                                *d32++ =
                                        (((v1 >> 22        ) + (v2 >> 22        ) + 1) / 2) << 22 |
                                        (((v1 >> 12 & 0x3ff) + (v2 >> 12 & 0x3ff) + 1) / 2) << 12 |
                                        (((v1 >>  2 & 0x3ff) + (v2 >>  2 & 0x3ff) + 1) / 2) << 2;
                        }
                }
        } else if (codec == R12L) {
                uint32_t *s32_1 = (uint32_t *) s1;
                uint32_t *s32_2 = (uint32_t *) s2;
                uint32_t *d32 = (uint32_t *) d;
                int shift = 0;
                uint32_t remain1 = 0;
                uint32_t remain2 = 0;
                uint32_t out = 0;
                for (size_t x = 0; x < linesize / 16; ++x) {
                        #pragma GCC unroll 8
                        for (int y = 0; y < 4; ++y) {
                                uint32_t in1 = *s32_1++;
                                uint32_t in2 = *s32_2++;
                                if (shift > 0) {
                                        remain1 = remain1 | (in1 & ((1<<((shift + 12) % 32)) - 1)) << (32-shift);
                                        remain2 = remain2 | (in2 & ((1<<((shift + 12) % 32)) - 1)) << (32-shift);
                                        uint32_t ret = (remain1 + remain2 + 1) / 2;
                                        out |= ret << shift;
                                        *d32++ = out;
                                        out = ret >> (32-shift);
                                        shift = (shift + 12) % 32;
                                        in1 >>= shift;
                                        in2 >>= shift;
                                }
                                while (shift <= 32 - 12) {
                                        out |= ((((in1 & 0xfff) + (in2 & 0xfff)) + 1) / 2) << shift;
                                        in1 >>= 12;
                                        in2 >>= 12;
                                        shift += 12;
                                }
                                if (shift == 32) {
                                        *d32++ = out;
                                        out = 0;
                                        shift = 0;
                                } else {
                                        remain1 = in1;
                                        remain2 = in2;
                                }
                        }
                }
        } else {
                return false;
        }
        return true;
}

static void perform_linear(struct state_df *s, struct video_frame *in, struct video_frame *out, int pitch)
{
        int linesize = vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
        char *src = s->buffers[s->buffer_current] + (in ? 0 : linesize);
        char *dst = out->tiles[0].data;
        if (in == NULL) {
                memcpy(dst, src, linesize); // copy first line up
                dst += pitch; // then copy every even line to subsequent odd line
        }
        unsigned y = in ? 0 : 1;
        for ( ; y < out->tiles[0].height - 2; y += 2) {
                memcpy(dst, src, linesize);
                if (!avg_lines(out->color_spec, linesize, src, src + 2 * linesize, dst + pitch)) {
                        memcpy(dst + pitch, src, linesize); // fallback bob
                }
                dst += 2 * pitch;
                src += 2 * linesize;
        }
        for ( ; y < out->tiles[0].height; y++) { // last line(s) if needed
                memcpy(dst, src, linesize);
                dst += pitch;
        }
}

/// @param in  may be NULL
static bool common_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_df *s = (struct state_df *) state;

        switch (s->algo) {
                case DF:
                        perform_df(s, in, out, req_pitch);
                        break;
                case BOB:
                        perform_bob(s, in, out, req_pitch);
                        break;
                case LINEAR:
                        perform_linear(s, in, out, req_pitch);
                        break;
        }

        if (!s->nodelay) {
                // In following code we fix timing in order not to pass both frames
                // in bulk but rather we busy-wait half of the frame time.
                if (in) {
                        s->frame_received = std::chrono::steady_clock::now();
                } else {
                        decltype(s->frame_received) t;
                        do {
                                t = std::chrono::steady_clock::now();
                        } while (std::chrono::duration_cast<std::chrono::duration<double>>(t - s->frame_received).count() <= 0.5 / out->fps);
                }
        }

        return true;
}

static void common_done(void *state)
{
        struct state_df *s = (struct state_df *) state;
        
        free(s->buffers[0]);
        free(s->buffers[1]);
        vf_free(s->in);
        delete s;
}

static void common_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_df *s = (struct state_df *) state;

        out->width = vf_get_tile(s->in, 0)->width;
        out->height = vf_get_tile(s->in, 0)->height;
        out->color_spec = s->in->color_spec;
        out->interlacing = PROGRESSIVE;
        out->fps = s->in->fps * 2.0;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 2;
}

static const struct vo_postprocess_info vo_pp_df_info = {
        df_init,
        common_postprocess_reconfigure,
        common_getf,
        common_get_out_desc,
        common_get_property,
        common_postprocess,
        common_done,
};

static const struct vo_postprocess_info vo_pp_bob_info = {
        bob_init,
        common_postprocess_reconfigure,
        common_getf,
        common_get_out_desc,
        common_get_property,
        common_postprocess,
        common_done,
};

static const struct vo_postprocess_info vo_pp_linear_info = {
        linear_init,
        common_postprocess_reconfigure,
        common_getf,
        common_get_out_desc,
        common_get_property,
        common_postprocess,
        common_done,
};

REGISTER_MODULE(double_framerate, &vo_pp_df_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
REGISTER_MODULE(deinterlace_bob, &vo_pp_bob_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
REGISTER_MODULE(deinterlace_linear, &vo_pp_linear_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);


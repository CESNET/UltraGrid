/**
 * @file   video_compress/dxt_glsl.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2023 CESNET, z. s. p. o.
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

#include <cassert>                     // for assert
#include <cstdio>                      // for printf, fprintf, stderr
#include <cstring>                     // for strcmp
#include <cstdlib>                     // for NULL, abort, calloc, EXIT_FAILURE
#include <memory>                      // for shared_ptr

#include "../dxt_compress/dxt_common.h"   // for dxt_format, dxt_type
#include "../dxt_compress/dxt_encoder.h"
#include "../dxt_compress/dxt_util.h"
#include "compat/strings.h"            // for strcasecmp
#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "pixfmt_conv.h"               // for get_decoder_from_to, decoder_t
#include "types.h"                     // for tile, video_frame, video_desc
#include "utils/video_frame_pool.h"
#include "video_codec.h"               // for codec_is_a_rgb, get_bits_per_c...
#include "video_compress.h"
#include "video_frame.h"               // for vf_get_tile, video_desc_from_f...

using namespace std;

namespace {

struct state_video_compress_rtdxt {
        struct module module_data;

        struct dxt_encoder **encoder;
        int encoder_count;

        decoder_t decoder;
        unique_ptr<char []> decoded;
        unsigned int configured:1;
        unsigned int interlaced_input:1;
        codec_t color_spec;

        int encoder_input_linesize;

        struct gl_context gl_context;

        video_frame_pool pool;
};

static int configure_with(struct state_video_compress_rtdxt *s, struct video_frame *frame);
static void dxt_glsl_compress_done(struct module *mod);

static int configure_with(struct state_video_compress_rtdxt *s, struct video_frame *frame)
{
        unsigned int x;
        enum dxt_format format{};

        for (x = 0; x < frame->tile_count; ++x) {
                if (vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width ||
                                vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width) {

                        fprintf(stderr,"[RTDXT] Requested to compress tiles of different size!");
                        exit_uv(EXIT_FAILURE);
                        return false;
                }
        }

        if (get_bits_per_component(frame->color_spec) > 8) {
                LOG(LOG_LEVEL_NOTICE) << "[RTDXT] Converting from " << get_bits_per_component(frame->color_spec) <<
                        " to 8 bits. You may directly capture 8-bit signal to improve performance.\n";
        }

        s->decoder = NULL;
        codec_t codec_try[3] = {};
        if (s->color_spec == DXT1_YUV) {
                codec_try[0] = UYVY; // RTDXT doesn't convert RGB to (DXT1_)YUV
                if (codec_is_a_rgb(frame->color_spec)) {
                        log_msg(LOG_LEVEL_WARNING, "[RTDXT] Compression of a RGB to DXT1_YUV will use CPU conversion!\n");
                }
        } else if (codec_is_a_rgb(frame->color_spec)) {
                codec_try[0] = RGB;
                codec_try[1] = RGBA;
                codec_try[2] = UYVY;
        } else {
                codec_try[0] = UYVY;
                codec_try[1] = RGB;
                codec_try[2] = RGBA;
        }
        for (size_t i = 0; i < sizeof codec_try / sizeof codec_try[0]; ++i) {
                if ((s->decoder = get_decoder_from_to(frame->color_spec, codec_try[i])) != NULL) {
                        switch (codec_try[i]) {
                                case RGB: format = DXT_FORMAT_RGB; break;
                                case RGBA: format = DXT_FORMAT_RGBA; break;
                                case UYVY: format = DXT_FORMAT_YUV422; break;
                                case VIDEO_CODEC_NONE: break;
                                default: assert(0 && "Shouldn't get other codecs than the above.");
                        }
                        break;
                }
        }
        if (!s->decoder) {
                fprintf(stderr, "[RTDXT] Unsupported codec: %d\n", frame->color_spec);
                return false;
        }

        int data_len = 0;

        s->encoder = (struct dxt_encoder **) calloc(frame->tile_count, sizeof(struct dxt_encoder *));
        if(s->color_spec == DXT1) {
                for(int i = 0; i < (int) frame->tile_count; ++i) {
                        s->encoder[i] =
                                dxt_encoder_create(DXT_TYPE_DXT1, frame->tiles[0].width, frame->tiles[0].height, format,
                                                s->gl_context.legacy);
                }
                data_len = dxt_get_size(frame->tiles[0].width, frame->tiles[0].height, DXT_TYPE_DXT1);
        } else if(s->color_spec == DXT1_YUV) {
                for(int i = 0; i < (int) frame->tile_count; ++i) {
                        s->encoder[i] =
                                dxt_encoder_create(DXT_TYPE_DXT1_YUV, frame->tiles[0].width, frame->tiles[0].height, format,
                                                s->gl_context.legacy);
                }
                data_len =
                    dxt_get_size(frame->tiles[0].width, frame->tiles[0].height,
                                 DXT_TYPE_DXT1_YUV);
        } else if(s->color_spec == DXT5){
                for(int i = 0; i < (int) frame->tile_count; ++i) {
                        s->encoder[i] =
                                dxt_encoder_create(DXT_TYPE_DXT5_YCOCG, frame->tiles[0].width, frame->tiles[0].height, format,
                                                s->gl_context.legacy);
                }
                data_len = dxt_get_size(frame->tiles[0].width, frame->tiles[0].height, DXT_TYPE_DXT5_YCOCG);
        }
        s->encoder_count = frame->tile_count;

        for(int i = 0; i < (int) frame->tile_count; ++i) {
                if(s->encoder[i] == NULL) {
                        fprintf(stderr, "[RTDXT] Unable to create decoder.\n");
                        exit_uv(EXIT_FAILURE);
                        return false;
                }
        }

        s->encoder_input_linesize = frame->tiles[0].width;
        switch(format) {
                case DXT_FORMAT_RGBA:
                        s->encoder_input_linesize *= 4;
                        break;
                case DXT_FORMAT_RGB:
                        s->encoder_input_linesize *= 3;
                        break;
                case DXT_FORMAT_YUV422:
                        s->encoder_input_linesize *= 2;
                        break;
                case DXT_FORMAT_YUV:
                        /* not used - just not compilator to complain */
                        abort();
                        break;
        }

        assert(data_len > 0);
        assert(s->encoder_input_linesize > 0);

        struct video_desc compressed_desc;
        compressed_desc = video_desc_from_frame(frame);
        compressed_desc.color_spec = s->color_spec;
        /* We will deinterlace the output frame */
        if(frame->interlacing  == INTERLACED_MERGED) {
                compressed_desc.interlacing = PROGRESSIVE;
                s->interlaced_input = true;
                fprintf(stderr, "[DXT compress] Enabling automatic deinterlacing.\n");
        } else {
                s->interlaced_input = false;
        }
        s->pool.reconfigure(compressed_desc, data_len);

        s->decoded = unique_ptr<char []>(new char[4 * compressed_desc.width * compressed_desc.height]);

        s->configured = true;
        return true;
}

struct module *dxt_glsl_compress_init(struct module *parent, const char *opts)
{
        struct state_video_compress_rtdxt *s;

        if(strcmp(opts, "help") == 0) {
                printf("DXT GLSL comperssion usage:\n");
                printf("\t-c RTDXT:DXT1\n");
                printf("\t\tcompress with DXT1\n");
                printf("\t-c RTDXT:DXT1_YUV\n");
                printf("\t\tcompress with DXT1_YUV\n");
                printf("\t-c RTDXT:DXT5\n");
                printf("\t\tcompress with DXT5 YCoCg\n");
                return static_cast<module*>(INIT_NOERR);
        }

        s = new state_video_compress_rtdxt();

        if (strcasecmp(opts, "DXT5") == 0) {
                s->color_spec = DXT5;
        } else if (strcasecmp(opts, "DXT1") == 0) {
                s->color_spec = DXT1;
        } else if (strcasecmp(opts, "DXT1_YUV") == 0) {
                s->color_spec = DXT1_YUV;
        } else if (opts[0] == '\0') {
                s->color_spec = DXT1;
        } else {
                fprintf(stderr, "Unknown compression: %s\n", opts);
                delete s;
                return NULL;
        }

        if(!init_gl_context(&s->gl_context, GL_CONTEXT_ANY)) {
                fprintf(stderr, "[RTDXT] Error initializing GL context");
                delete s;
                return NULL;
        }

        gl_context_make_current(NULL);

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = dxt_glsl_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;
}

shared_ptr<video_frame> dxt_glsl_compress(struct module *mod, shared_ptr<video_frame> tx)
{
        if (!tx) {
                return {};
        }

        struct state_video_compress_rtdxt *s = (struct state_video_compress_rtdxt *) mod->priv_data;
        int i;
        unsigned char *line1, *line2;

        unsigned int x;

        gl_context_make_current(&s->gl_context);

        if(!s->configured) {
                int ret;
                ret = configure_with(s, tx.get());
                if(!ret)
                        return NULL;
        }

        shared_ptr<video_frame> out_frame = s->pool.get_frame();

        for (x = 0; x < tx->tile_count; ++x) {
                struct tile *in_tile = vf_get_tile(tx.get(), x);
                struct tile *out_tile = vf_get_tile(out_frame.get(), x);

                line1 = (unsigned char *) in_tile->data;
                line2 = (unsigned char *) s->decoded.get();

                for (i = 0; i < (int) in_tile->height; ++i) {
                        s->decoder(line2, line1, s->encoder_input_linesize,
                                        0, 8, 16);
                        line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                        line2 += s->encoder_input_linesize;
                }

                if(s->interlaced_input)
                        vc_deinterlace((unsigned char *) s->decoded.get(), s->encoder_input_linesize,
                                        in_tile->height);

                dxt_encoder_compress(s->encoder[x],
                                (unsigned char *) s->decoded.get(),
                                (unsigned char *) out_tile->data);
        }

        gl_context_make_current(NULL);

        return out_frame;
}

static void dxt_glsl_compress_done(struct module *mod)
{
        struct state_video_compress_rtdxt *s = (struct state_video_compress_rtdxt *) mod->priv_data;

        if(s->encoder) {
                for(int i = 0; i < s->encoder_count; ++i) {
                        if(s->encoder[i])
                                dxt_encoder_destroy(s->encoder[i]);
                }
        }

        destroy_gl_context(&s->gl_context);
        delete s;
}

const struct video_compress_info rtdxt_info = {
        "RTDXT",
        dxt_glsl_compress_init,
        dxt_glsl_compress,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};

REGISTER_MODULE(rtdxt, &rtdxt_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace


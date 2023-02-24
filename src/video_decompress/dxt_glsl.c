/**
 * @file   video_decompress/dxt_glsl.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2019 CESNET z.s.p.o.
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
#include "debug.h"
#include "host.h"
#include "video_decompress.h"

#include "dxt_compress/dxt_decoder.h"
#include "dxt_compress/dxt_util.h"
//#include "compat/platform_semaphore.h"
#include "video.h"
#include <pthread.h>
#include <stdlib.h>
#include "lib_common.h"

#include "gl_context.h"

struct state_decompress_rtdxt {
        struct dxt_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;
        unsigned int configured:1;
        
        struct gl_context context;
};

static int configure_with(struct state_decompress_rtdxt *decompressor, struct video_desc desc)
{
        enum dxt_type type;

        if(!init_gl_context(&decompressor->context, GL_CONTEXT_ANY)) {
                fprintf(stderr, "[RTDXT decompress] Failed to create GL context.\n");
                exit_uv(EXIT_FAILURE);
                decompressor->compressed_len = 0;
                return FALSE;
        }

        if(desc.color_spec == DXT5) {
                type = DXT_TYPE_DXT5_YCOCG;
        } else if(desc.color_spec == DXT1) {
                type = DXT_TYPE_DXT1;
        } else if(desc.color_spec == DXT1_YUV) {
                type = DXT_TYPE_DXT1_YUV;
        } else {
                fprintf(stderr, "Wrong compressiong to decompress.\n");
                return FALSE;
        }
        
        decompressor->desc = desc;

        decompressor->decoder = dxt_decoder_create(type, desc.width,
                        desc.height, decompressor->out_codec == RGBA ?
                        DXT_FORMAT_RGBA : DXT_FORMAT_YUV422,
                        decompressor->context.legacy);

        if (decompressor->decoder == NULL) {
                fprintf(stderr, "[RTDXT decompress] State initialization failed.\n");
                return FALSE;
        }
        
        decompressor->compressed_len = dxt_get_size(desc.width, desc.height, type);
        decompressor->configured = TRUE;

        return TRUE;
}

static void * dxt_glsl_decompress_init(void)
{
        struct state_decompress_rtdxt *s;
        
        s = (struct state_decompress_rtdxt *) malloc(sizeof(struct state_decompress_rtdxt));
        s->configured = FALSE;

        return s;
}

static int dxt_glsl_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_rtdxt *s = (struct state_decompress_rtdxt *) state;
        int ret = TRUE;
        
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->out_codec = out_codec;
        if(!s->configured) {
                ret = configure_with(s, desc);
        } else {
                gl_context_make_current(&s->context);
                dxt_decoder_destroy(s->decoder);
                destroy_gl_context(&s->context);
                ret = configure_with(s, desc);
        }

        gl_context_make_current(NULL);

        return ret;
}

static decompress_status dxt_glsl_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop)
{
        struct state_decompress_rtdxt *s = (struct state_decompress_rtdxt *) state;
        UNUSED(src_len);
        UNUSED(frame_seq);
        UNUSED(callbacks);
        UNUSED(internal_prop);

        if (!s->configured) {
                fprintf(stderr, "DXT decoder not configured!\n");
                return DECODER_NO_FRAME;
        }

        gl_context_make_current(&s->context);
        
        if(s->pitch == vc_get_linesize(s->desc.width, s->out_codec)) {
                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) dst);
        } else {
                int i;
                int linesize;
                unsigned char *line_src, *line_dst;
                
                unsigned char *tmp;
                
                if(s->out_codec == UYVY)
                        linesize = s->desc.width * 2;
                else
                        linesize = s->desc.width * 4;
                tmp = malloc(linesize * s->desc.height);

                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) tmp);
                line_dst = dst;
                line_src = tmp;
                for(i = 0; i < (int) s->desc.height; i++) {
                        if(s->out_codec == RGBA) {
                                vc_copylineRGBA(line_dst, line_src, linesize,
                                                s->rshift, s->gshift, s->bshift);
                        } else { /* UYVY */
                                memcpy(line_dst, line_src, linesize);
                        }
                        line_dst += s->pitch;
                        line_src += linesize;
                        
                }
                free(tmp);
        }

        gl_context_make_current(NULL);
        return DECODER_GOT_FRAME;
}

static int dxt_glsl_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress_rtdxt *s = (struct state_decompress_rtdxt *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = TRUE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static void dxt_glsl_decompress_done(void *state)
{
        struct state_decompress_rtdxt *s = (struct state_decompress_rtdxt *) state;
        
        if(s->configured) {
                gl_context_make_current(&s->context);
                dxt_decoder_destroy(s->decoder);
                gl_context_make_current(NULL);
                destroy_gl_context(&s->context);
        }
        free(s);
}

static int dxt_glsl_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        UNUSED(internal);
        if (compression != DXT1 && compression != DXT1_YUV && compression != DXT5) {
                return -1;
        }
        if (ugc != RGBA && ugc != UYVY) {
                return -1;
        }
        return 500;
}

static const struct video_decompress_info dxt_glsl_info = {
        dxt_glsl_decompress_init,
        dxt_glsl_decompress_reconfigure,
        dxt_glsl_decompress,
        dxt_glsl_decompress_get_property,
        dxt_glsl_decompress_done,
        dxt_glsl_decompress_get_priority,
};

REGISTER_MODULE(dxt_glsl, &dxt_glsl_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);


/**
 * @file   vo_postprocess/scale.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET z.s.p.o.
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

#include <assert.h>          // for assert
#include <stdbool.h>         // for bool, true, false
#include <stdio.h>           // for fprintf, printf, stderr
#include <stdlib.h>          // for free, NULL, atoi, malloc, calloc, size_t
#include <string.h>          // for memcpy, strtok_r, strcmp, strdup

#include "debug.h"
#include "lib_common.h"
#include "gl_context.h"      // for GL_TEXTURE_2D, glTexParameteri, glBindTe...
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_scale {
        struct video_frame *in;
        struct gl_context context;

        int scaled_width, scaled_height;
        GLuint tex_input;
        GLuint tex_output;
        GLuint fbo;
};

static bool scale_get_property(void *state, int property, void *val, size_t *len)
{
        bool ret = false;
        codec_t supported[] = {UYVY, RGBA};

        UNUSED(state);

        switch(property) {
                case VO_PP_PROPERTY_CODECS:
                        if(*len < (int) sizeof(supported)) {
                                fprintf(stderr, "Scale postprocessor query little space.\n");
                                *len = 0; 
                        } else {
                                memcpy(val, &supported, sizeof(supported));
                                *len = sizeof(supported);
                        }
                        ret = true;
                        break;
        }

        return ret;
}

static void usage()
{
        printf("Scale postprocessor settings:\n");
        printf("\t-p scale:width:height\n");
}

static void * scale_init(const char *config) {
        char *save_ptr = NULL;
        char *ptr;

        if (strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }

        char *tmp = strdup(config);
        assert(tmp != NULL);

        struct state_scale *s = (struct state_scale *)
                calloc(1, sizeof(struct state_scale));
        assert(s != NULL);

        ptr = strtok_r(tmp, ":", &save_ptr);
        if (ptr != NULL) {
                s->scaled_width = atoi(ptr);
        }
        ptr = strtok_r(NULL, ":", &save_ptr);
        if (ptr != NULL) {
                s->scaled_height = atoi(ptr);
        }
        if (s->scaled_width <= 0 || s->scaled_height <= 0) {
                fprintf(stderr, "Scale postprocessor incorrect usage.\n");
                usage();
                free(s);
                free(tmp);
                return NULL;
        }

        free(tmp);

        s->in = NULL;

        init_gl_context(&s->context, GL_CONTEXT_LEGACY);
        gl_context_make_current(&s->context);

        glEnable(GL_TEXTURE_2D);

        glGenTextures(1, &s->tex_input);
        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &s->tex_output);
        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &s->fbo);

        return s;
}

static bool
scale_reconfigure(void *state, struct video_desc desc)
{
        struct state_scale *s = (struct state_scale *) state;
        struct tile *in_tile;
        int i;
        int width, height;

        if(s->in) {
                int i;
                for(i = 0; i < (int) s->in->tile_count; ++i) {
                        free(s->in->tiles[i].data);
                }
                vf_free(s->in);
        }

        s->in = vf_alloc(desc.tile_count);


        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;

        for(i = 0; i < (int) desc.tile_count; ++i) {
                in_tile = vf_get_tile(s->in, i);
                in_tile->width = desc.width;
                in_tile->height = desc.height;

                in_tile->data_len =
                        vc_get_linesize(desc.width, desc.color_spec) * desc.height;
                in_tile->data = malloc(in_tile->data_len);
        }

        assert(desc.tile_count >= 1);
        in_tile = vf_get_tile(s->in, 0);

        gl_context_make_current(&s->context);

        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        width = in_tile->width;
        height = in_tile->height;
        if(s->in->color_spec == UYVY) {
                width /= 2;
        }
        if(s->in->interlacing == INTERLACED_MERGED) {
                width *= 2;
                height /= 2;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);



        glBindTexture(GL_TEXTURE_2D, s->tex_output);
        width = s->scaled_width;
        height = s->scaled_height;
        if(s->in->color_spec == UYVY) {
                width /= 2;
        }
        if(s->in->interlacing == INTERLACED_MERGED) {
                width *= 2;
                height /= 2;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


        return true;
}

static struct video_frame * scale_getf(void *state)
{
        struct state_scale *s = (struct state_scale *) state;

        return s->in;
}

static bool scale_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_scale *s = (struct state_scale *) state;
        int i;
        int width, height;

        int src_linesize = vc_get_linesize(out->tiles[0].width, out->color_spec);

        char *tmp_data = NULL;

        if(req_pitch != src_linesize) {
                tmp_data = malloc(src_linesize *
                                out->tiles[0].height);
        }

        gl_context_make_current(&s->context);

        for(i = 0; i < (int) in->tile_count; ++i) {
                struct tile *in_tile = vf_get_tile(s->in, i);

                glBindTexture(GL_TEXTURE_2D, s->tex_input);
                width = in_tile->width;
                height = in_tile->height;
                if(s->in->color_spec == UYVY) {
                        width /= 2;
                }
                if(s->in->interlacing == INTERLACED_MERGED) {
                        width *= 2;
                        height /= 2;
                }
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                                GL_RGBA, GL_UNSIGNED_BYTE, in_tile->data); 

                glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_output, 0);

                width = s->scaled_width;
                height = s->scaled_height;
                if(s->in->color_spec == UYVY) {
                        width /= 2;
                }
                if(s->in->interlacing == INTERLACED_MERGED) {
                        width *= 2;
                        height /= 2;
                }
                glViewport(0, 0, width, height);
                glBindTexture(GL_TEXTURE_2D, s->tex_input);

                glClearColor(1,0,0,1);
                glClear(GL_COLOR_BUFFER_BIT);

                glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                glEnd();

                glBindTexture(GL_TEXTURE_2D, s->tex_output);
                if(tmp_data) { /* we need to change pitch */
                        int y;
                        glReadPixels(0, 0, width , height, GL_RGBA, GL_UNSIGNED_BYTE, tmp_data);
                        char *src = tmp_data;
                        char *dst = out->tiles[i].data;
                        for (y = 0; y < (int) out->tiles[i].height; y += 1) {
                                memcpy(dst, src, src_linesize);
                                dst += req_pitch;
                                src += src_linesize;
                        }
                } else {
                        glReadPixels(0, 0, width , height, GL_RGBA, GL_UNSIGNED_BYTE, out->tiles[i].data);
                }
        }

        free(tmp_data);

        return true;
}

static void scale_done(void *state)
{
        struct state_scale *s = (struct state_scale *) state;

        glDeleteTextures(1, &s->tex_input);
        glDeleteTextures(1, &s->tex_output);
        glDeleteFramebuffers(1, &s->fbo);

        free(s->in->tiles[0].data);

        vf_free(s->in);

        destroy_gl_context(&s->context);

        free(state);
}

static void scale_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_scale *s = (struct state_scale *) state;

        out->width = s->scaled_width;
        out->height = s->scaled_height;
        out->color_spec = s->in->color_spec;
        out->interlacing = s->in->interlacing;
        out->fps = s->in->fps;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

static const struct vo_postprocess_info vo_pp_scale_info = {
        scale_init,
        scale_reconfigure,
        scale_getf,
        scale_get_out_desc,
        scale_get_property,
        scale_postprocess,
        scale_done,
};

REGISTER_MODULE(scale, &vo_pp_scale_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);


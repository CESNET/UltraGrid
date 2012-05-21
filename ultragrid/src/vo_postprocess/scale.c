/*
 * FILE:    vo_postprocess/double-framerate.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"

#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "vo_postprocess/scale.h"
#include "video_display.h"

#include <GL/glew.h>
#include "gl_context.h"

struct state_scale {
        struct video_frame *in;
        struct gl_context context;

        int scaled_width, scaled_height;
        codec_t out_codec;
        GLuint tex_input;
        GLuint tex_output;
        GLuint fbo;

        char *decoded;


        decoder_t decoder;
};

void scale_get_supported_codecs(codec_t ** supported_codecs, int *count)
{
        codec_t supported[] = {UYVY, RGBA};

        *supported_codecs = supported;
        *count = sizeof(supported) / sizeof(codec_t);
}

static void usage()
{
        printf("Scale postprocessor settings:\n");
        printf("\t-p scale:width:height\n");
}

void * scale_init(char *config) {
        struct state_scale *s;
        char *save_ptr = NULL;
        char *ptr;

        if(!config) {
                fprintf(stderr, "Scale postprocessor incorrect usage.\n");
        }

        if(!config || strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }

        s = (struct state_scale *) 
                        malloc(sizeof(struct state_scale));


        ptr = strtok_r(config, ":", &save_ptr);
        assert(ptr != NULL);
        s->scaled_width = atoi(ptr);
        assert(ptr != NULL);
        ptr = strtok_r(NULL, ":", &save_ptr);
        s->scaled_height = atoi(ptr);

        assert(s != NULL);
        s->in = vf_alloc(1);

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

        s->decoded = NULL;

        
        return s;
}

int scale_reconfigure(void *state, struct video_desc desc)
{
        struct state_scale *s = (struct state_scale *) state;
        struct tile *in_tile = vf_get_tile(s->in, 0);

        free(s->decoded);

        switch(desc.color_spec) {
                case RGB:
                        s->decoder = (decoder_t) vc_copylineRGBtoRGBA;
                        s->out_codec = RGBA;
                case RGBA:
                        s->decoder = (decoder_t) memcpy;
                        s->out_codec = RGBA;
                        break;
                case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        s->out_codec = RGBA;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        s->decoder = (decoder_t) memcpy;
                        s->out_codec = UYVY;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        s->out_codec = UYVY;
                        break;
                case DVS10:
                        s->decoder = (decoder_t) vc_copylineDVS10;
                        s->out_codec = UYVY;
                        break;
                case DPX10:        
                        s->decoder = (decoder_t) vc_copylineDPX10toRGB;
                        s->out_codec = UYVY;
                        break;
                default:
                        fprintf(stderr, "[scale] Unknown codec: %d\n", desc.color_spec);
                        exit_uv(128);
                        return FALSE;

        }

        s->decoded = (char *) malloc(vc_get_linesize(desc.width, s->out_codec) * desc.height);

        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;

        in_tile->width = desc.width;
        in_tile->height = desc.height;

        in_tile->linesize = vc_get_linesize(desc.width, desc.color_spec);
        in_tile->data_len = in_tile->linesize * desc.height;
        in_tile->data = malloc(in_tile->data_len);

        gl_context_make_current(&s->context);

        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        if(s->out_codec == RGBA) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, in_tile->width, in_tile->height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        } else {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, in_tile->width / 2, in_tile->height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        }

        glBindTexture(GL_TEXTURE_2D, s->tex_output);
        if(s->out_codec == RGBA) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, s->scaled_width, s->scaled_height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        } else {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, s->scaled_width / 2, s->scaled_height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        }


        return TRUE;
}

struct video_frame * scale_getf(void *state)
{
        struct state_scale *s = (struct state_scale *) state;

        return s->in;
}

void scale_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_scale *s = (struct state_scale *) state;
        int y;
        unsigned char *line1, *line2;

        gl_context_make_current(&s->context);

        struct tile *in_tile = vf_get_tile(in, 0);

        line1 = in_tile->data;
        line2 = s->decoded;

        for(y = 0; y < in_tile->height; ++y) {
                int out_linesize = vc_get_linesize(in_tile->width, s->out_codec);

                s->decoder(line2, line1, out_linesize, 0, 8, 16);
                line1 += vc_get_linesize(in_tile->width, in->color_spec);
                line2 += out_linesize;
        }

        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, in->tiles[0].width / (s->out_codec == UYVY ? 2 : 1), in->tiles[0].height,
                        GL_RGBA, GL_UNSIGNED_BYTE, s->decoded); 

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_output, 0);

        glViewport(0, 0, s->scaled_width / (s->out_codec == UYVY ? 2 : 1), s->scaled_height);
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
        glReadPixels(0, 0, s->scaled_width / (s->out_codec == UYVY ? 2 : 1), s->scaled_height, GL_RGBA, GL_UNSIGNED_BYTE, out->tiles[0].data);
}

void scale_done(void *state)
{
        struct state_scale *s = (struct state_scale *) state;

        if(s->in->tiles[0].data) 
                free(s->in->tiles[0].data);

        vf_free(s->in);

        destroy_gl_context(&s->context);

        free(state);
}

void scale_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_scale *s = (struct state_scale *) state;

        out->width = s->scaled_width;
        out->height = s->scaled_height;
        out->color_spec = s->out_codec;
        out->interlacing = s->in->interlacing;
        out->fps = s->in->fps;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}


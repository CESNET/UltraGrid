/*
 * FILE:    dxt_glsl_compress.c
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
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "host.h"
#include "video_compress/uyvy.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>

#include <GL/glew.h>

#ifndef HAVE_MACOSX
#include "x11_common.h"
#endif

#include "gl_context.h"

static const char fp_display_rgba_to_yuv422_legacy[] = 
"#define GL_legacy 1\n"
    "#if GL_legacy\n"
    "#define TEXCOORD gl_TexCoord[0]\n"
    "#else\n"
    "#define TEXCOORD TEX0\n"
    "#define texture2D texture\n"
    "#endif\n"
    "\n"
    "#if GL_legacy\n"
    "#define colorOut gl_FragColor\n"
    "#else\n"
    "out vec4 colorOut;\n"
    "#endif\n"
    "\n"
    "#if ! GL_legacy\n"
    "in vec4 TEX0;\n"
    "#endif\n"
    "\n"
    "uniform sampler2D image;\n"
    "uniform float imageWidth; // is original image width, it means twice as wide as ours\n"
    "\n"
    "void main()\n"
    "{\n"
    "        vec4 rgba1, rgba2;\n"
    "        vec4 yuv1, yuv2;\n"
    "        vec2 coor1, coor2;\n"
    "        float U, V;\n"
    "\n"
    "        coor1 = TEXCOORD.xy - vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "        coor2 = TEXCOORD.xy + vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "\n"
    "        rgba1  = texture2D(image, coor1);\n"
    "        rgba2  = texture2D(image, coor2);\n"
    "        \n"
    "        yuv1.x = 1.0/16.0 + (rgba1.r * 0.2126 + rgba1.g * 0.7152 + rgba1.b * 0.0722) * 0.8588; // Y\n"
    "        yuv1.y = 0.5 + (-rgba1.r * 0.1145 - rgba1.g * 0.3854 + rgba1.b * 0.5) * 0.8784;\n"
    "        yuv1.z = 0.5 + (rgba1.r * 0.5 - rgba1.g * 0.4541 - rgba1.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        yuv2.x = 1.0/16.0 + (rgba2.r * 0.2126 + rgba2.g * 0.7152 + rgba2.b * 0.0722) * 0.8588; // Y\n"
    "        yuv2.y = 0.5 + (-rgba2.r * 0.1145 - rgba2.g * 0.3854 + rgba2.b * 0.5) * 0.8784;\n"
    "        yuv2.z = 0.5 + (rgba2.r * 0.5 - rgba2.g * 0.4541 - rgba2.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        U = mix(yuv1.y, yuv2.y, 0.5);\n"
    "        V = mix(yuv1.z, yuv2.z, 0.5);\n"
    "        \n"
    "        colorOut = vec4(U,yuv1.x, V, yuv2.x);\n"
    "}\n"
;


struct uyvy_video_compress {
        struct video_frame *out[2];
        unsigned int configured:1;
        struct video_desc saved_desc;

        struct gl_context context;

        GLuint program_rgba_to_yuv422;

        GLuint texture_rgba;
        GLuint fbo;
        GLuint texture;

        int gl_format;
};

int uyvy_configure_with(struct uyvy_video_compress *s, struct video_frame *tx);

void * uyvy_compress_init(char * fmt)
{
        UNUSED(fmt);
        struct uyvy_video_compress *s;
        
        s = (struct uyvy_video_compress *) malloc(sizeof(struct uyvy_video_compress));
        s->out[0] = s->out[1] = NULL;

        if(!init_gl_context(&s->context, GL_CONTEXT_LEGACY))
                abort();
        glewInit();

        glEnable(GL_TEXTURE_2D);

        const GLchar  *FProgram;
        char          *log;
        GLuint         FSHandle,PHandle;

        int len;
        GLsizei gllen;

        PHandle = glCreateProgram();
        
        FProgram = (const GLchar *) fp_display_rgba_to_yuv422_legacy;
        /* Set up program objects. */
        s->program_rgba_to_yuv422 = glCreateProgram();
        FSHandle=glCreateShader(GL_FRAGMENT_SHADER);
        
        /* Compile Shader */
        len = strlen(FProgram);
        glShaderSource(FSHandle, 1, &FProgram, &len);
        glCompileShader(FSHandle);
        
        /* Print compile log */
        log = calloc(32768,sizeof(char));
        glGetShaderInfoLog(FSHandle, 32768, &gllen, log);
        printf("Compile Log: %s\n", log);
#if 0
        glShaderSource(VSHandle,1, &VProgram,NULL);
        glCompileShaderARB(VSHandle);
        memset(log, 0, 32768);
        glGetInfoLogARB(VSHandle,32768, &gllen,log);
        printf("Compile Log: %s\n", log);

        /* Attach and link our program */
        glAttachObjectARB(PHandle,VSHandle);
#endif
        glAttachShader(PHandle, FSHandle);
        glLinkProgram(PHandle);
        
        /* Print link log. */
        memset(log, 0, 32768);
        glGetInfoLogARB(PHandle,32768,NULL,log);
        printf("Link Log: %s\n", log);
        free(log);

        s->program_rgba_to_yuv422 = PHandle;

        glUseProgram(PHandle);
        glUniform1i(glGetUniformLocationARB(PHandle,"image"),0);
        glUseProgram(0);

        s->configured = FALSE;

        gl_context_make_current(NULL);

        return s;
}

int uyvy_configure_with(struct uyvy_video_compress *s, struct video_frame *tx)
{
        switch (tx->color_spec) {
                case RGB:
                        s->gl_format = GL_RGB;
                        break;
                case RGBA:
                        s->gl_format = GL_RGBA;
                        break;
                default:
                        fprintf(stderr, "[UYVY compress] We can transform only RGB or RGBA to UYVY.\n");
                        return FALSE;
        }

        for(int i = 0; i < 2; ++i) {
                s->out[i] = vf_alloc(1);
                s->out[i]->color_spec = UYVY;
                s->out[i]->interlacing = tx->interlacing;
                s->out[i]->fps = tx->fps;

                struct tile *tile = &s->out[i]->tiles[0];
                tile->width = tx->tiles[0].width;
                tile->height = tx->tiles[0].height;
                tile->data_len = 2 * tile->width * tile->height; /* UYVY */
                tile->data = malloc(tile->data_len);
        }

        glUseProgram(s->program_rgba_to_yuv422);
        glUniform1f(glGetUniformLocation(s->program_rgba_to_yuv422, "imageWidth"),
                                        (GLfloat) tx->tiles[0].width);
        glUseProgram(0);

        glGenFramebuffers(1, &s->fbo);

        glGenTextures(1, &s->texture_rgba); 
        glBindTexture(GL_TEXTURE_2D, s->texture_rgba); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , s->gl_format, tx->tiles[0].width, tx->tiles[0].height, 0, s->gl_format, GL_UNSIGNED_BYTE, 0); 

        glGenTextures(1, &s->texture); 
        glBindTexture(GL_TEXTURE_2D, s->texture); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, tx->tiles[0].width / 2, tx->tiles[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); 

        glBindTexture(GL_TEXTURE_2D, 0);

        glClearColor(1.0,1.0,0,1.0);

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity( );
        glOrtho(-1,1,-1,1,10,-10);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity( );

        glViewport( 0, 0, tx->tiles[0].width / 2, tx->tiles[0].height);

        s->configured = TRUE;
        s->saved_desc = video_desc_from_frame(tx);

        return true;
}

struct video_frame * uyvy_compress(void *arg, struct video_frame * tx, int buffer)
{
        struct uyvy_video_compress *s = (struct uyvy_video_compress *) arg;
        assert (buffer == 0 || buffer == 1);

        gl_context_make_current(&s->context);

        if(!s->configured) {
                int ret;
                ret = uyvy_configure_with(s, tx);
                if(!ret)
                        return NULL;
        }

        assert(video_desc_eq(video_desc_from_frame(tx), s->saved_desc));

        struct tile *tile = &s->out[buffer]->tiles[0];

        glBindTexture(GL_TEXTURE_2D, s->texture_rgba);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        tx->tiles[0].width,
                        tx->tiles[0].height,
                        s->gl_format,
                        GL_UNSIGNED_BYTE,
                        tx->tiles[0].data);

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture, 0);

        glBindTexture(GL_TEXTURE_2D, s->texture_rgba); /* to texturing unit 0 */

        glUseProgram(s->program_rgba_to_yuv422);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();

        // Read back
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, tile->width / 2, tile->height, GL_RGBA, GL_UNSIGNED_BYTE, tile->data);

        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        gl_context_make_current(NULL);

        return s->out[buffer];
}

void uyvy_compress_done(void *arg)
{
        struct uyvy_video_compress *s = (struct uyvy_video_compress *) arg;

        for (int i = 0; i < 2; ++i) {
                vf_free_data(s->out[i]);
        }
        glDeleteFramebuffers(1, &s->fbo);
        glDeleteTextures(1, &s->texture_rgba);
        glDeleteTextures(1, &s->texture);
        destroy_gl_context(&s->context);
}


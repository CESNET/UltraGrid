/**
 * @file   video_compress/uyvy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "utils/video_frame_pool.h"
#include "video_compress.h"
#include "compat/platform_semaphore.h"
#include "video.h"
#include <pthread.h>
#include <stdlib.h>

#include "gl_context.h"

using namespace std;

namespace {

static const char fp_display_rgba_to_yuv422_legacy[] =
"#define LEGACY 1\n"
    "#if LEGACY\n"
    "#define TEXCOORD gl_TexCoord[0]\n"
    "#else\n"
    "#define TEXCOORD TEX0\n"
    "#define texture2D texture\n"
    "#endif\n"
    "\n"
    "#if LEGACY\n"
    "#define colorOut gl_FragColor\n"
    "#else\n"
    "out vec4 colorOut;\n"
    "#endif\n"
    "\n"
    "#if ! LEGACY\n"
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


struct state_video_compress_uyvy {
        struct module module_data;

        unsigned int configured:1;
        struct video_desc saved_desc;

        struct gl_context context;

        GLuint program_rgba_to_yuv422;

        GLuint texture_rgba;
        GLuint fbo;
        GLuint texture;

        int gl_format;

        video_frame_pool *pool;
};

int uyvy_configure_with(struct state_video_compress_uyvy *s, struct video_frame *tx);
static void uyvy_compress_done(struct module *mod);

struct module * uyvy_compress_init(struct module *parent, const char *)
{
        struct state_video_compress_uyvy *s;

        s = (struct state_video_compress_uyvy *) malloc(sizeof(struct state_video_compress_uyvy));

        if(!init_gl_context(&s->context, GL_CONTEXT_LEGACY))
                abort();

        glEnable(GL_TEXTURE_2D);

        s->program_rgba_to_yuv422 = glsl_compile_link(NULL, fp_display_rgba_to_yuv422_legacy);
        if (!s->program_rgba_to_yuv422) {
                log_msg(LOG_LEVEL_ERROR, "UYVY: Unable to create shader!\n");
                free(s);
                return NULL;
        }

        glUseProgram(s->program_rgba_to_yuv422);
        glUniform1i(glGetUniformLocation(s->program_rgba_to_yuv422,"image"),0);
        glUseProgram(0);

        s->configured = FALSE;

        gl_context_make_current(NULL);

        s->pool = new video_frame_pool();

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = uyvy_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;
}

int uyvy_configure_with(struct state_video_compress_uyvy *s, struct video_frame *tx)
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

        struct video_desc compressed_desc;
        compressed_desc = video_desc_from_frame(tx);
        compressed_desc.color_spec = UYVY;
        s->pool->reconfigure(compressed_desc, 2 * compressed_desc.width *
                        compressed_desc.height /* UYVY */);

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

shared_ptr<video_frame> uyvy_compress(struct module *mod, shared_ptr<video_frame> tx)
{
        struct state_video_compress_uyvy *s = (struct state_video_compress_uyvy *) mod->priv_data;

        gl_context_make_current(&s->context);

        if(!s->configured) {
                int ret;
                ret = uyvy_configure_with(s, tx.get());
                if(!ret)
                        return NULL;
        }

        assert(video_desc_eq(video_desc_from_frame(tx.get()), s->saved_desc));

        shared_ptr<video_frame> out = s->pool->get_frame();
        struct tile *tile = &out->tiles[0];

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

        return out;
}

static void uyvy_compress_done(struct module *mod)
{
        struct state_video_compress_uyvy *s = (struct state_video_compress_uyvy *) mod->priv_data;

        glDeleteFramebuffers(1, &s->fbo);
        glDeleteTextures(1, &s->texture_rgba);
        glDeleteTextures(1, &s->texture);
        destroy_gl_context(&s->context);

        delete s->pool;

        free(s);
}

const struct video_compress_info uyvy_info = {
        "UYVY",
        uyvy_compress_init,
        uyvy_compress,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        [] {return list<compress_preset>{}; },
        NULL
};

REGISTER_MODULE(uyvy, &uyvy_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace


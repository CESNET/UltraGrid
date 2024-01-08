/**
 * @file   video_display/opengl_conversions.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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

#include "color.h"
#include "debug.h"
#include "host.h"
#include "opengl_conversions.hpp"

#define MOD_NAME "[GL conversions] "

static const char *vert_src = R"END(
#version 330 core
layout(location = 0) in vec2 vert_pos;
layout(location = 1) in vec2 vert_uv;

out vec2 UV;

uniform vec2 scale_vec;

void main(){
        gl_Position = vec4(vert_pos, 0.0f, 1.0f);
        UV = vert_uv;
}
)END";

static const char *yuv_conv_frag_src = R"END(
#version 330 core
layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

uniform float width;

uniform float luma_scale = 1.1643f;
uniform float r_cr = 1.7926f;
uniform float g_cb = -0.2132f;
uniform float g_cr = -0.5328f;
uniform float b_cb = 2.1124f;

void main(){
        //The width could be odd, but the width of texture is always even
        float textureWidth = float((int(width) + 1) / 2 * 2);
        vec4 yuv;
        yuv.rgba  = texture(tex, vec2(UV.x / textureWidth * width, UV.y)).grba;
        if(UV.x * width / 2.0 - floor(UV.x * width / 2.0) > 0.5)
                yuv.r = yuv.a;

        yuv.r = 1.1643 * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;

        color.r = yuv.r + r_cr * yuv.b;
        color.g = yuv.r + g_cb * yuv.g + g_cr * yuv.b;
        color.b = yuv.r + b_cb * yuv.g;
        color.a = 1.0;
}
)END";


class Yuv_convertor : public Frame_convertor{
public:
        Yuv_convertor();

        /**
         * Renders the video frame containing YUV image data to attached texture
         *
         * @param f video frame to convert
         * @param pbo_frame true if video frame contains the image data
         * in a PBO buffer
         */
        void put_frame(video_frame *f, bool pbo_frame = false) override;

        /**
         * Attach texture to be used as output
         *
         * @param tex texture to attach
         */
        void attach_texture(const Texture& tex) override {
                fbuf.attach_texture(tex);
        }

private:
        GlProgram program;// = GlProgram(vert_src, yuv_conv_frag_src);
        Model quad;// = Model::get_quad();
        Framebuffer fbuf;
        Texture yuv_tex;
};

Yuv_convertor::Yuv_convertor(): program(vert_src, yuv_conv_frag_src),
        quad(Model::get_quad())
{
        const char *col = get_commandline_param("color");
        if(!col)
                return;

        int color = std::stol(col, nullptr, 16) >> 4; // first nibble
        if (color < 1 || color > 3){
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Wrong chromicities index " << color << "\n";
                return;
        }

        double cs_coeffs[2*4] = { 0, 0, KR_709, KB_709, KR_2020, KB_2020, KR_P3, KB_P3 };
        double kr = cs_coeffs[2 * color];
        double kb = cs_coeffs[2 * color + 1];

        glUseProgram(program.get());
        GLuint loc = glGetUniformLocation(program.get(), "luma_scale");
        glUniform1f(loc, Y_LIMIT_INV);
        loc = glGetUniformLocation(program.get(), "r_cr");
        glUniform1f(loc, R_CR(kr, kb));
        loc = glGetUniformLocation(program.get(), "g_cr");
        glUniform1f(loc, G_CR(kr, kb));
        loc = glGetUniformLocation(program.get(), "g_cb");
        glUniform1f(loc, G_CB(kr, kb));
        loc = glGetUniformLocation(program.get(), "b_cb");
        glUniform1f(loc, B_CB(kr, kb));
}


void Yuv_convertor::put_frame(video_frame *f, bool pbo_frame){
        glUseProgram(program.get());
        glBindFramebuffer(GL_FRAMEBUFFER, fbuf.get());
        glViewport(0, 0, f->tiles[0].width, f->tiles[0].height);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, yuv_tex.get());
        yuv_tex.allocate((f->tiles[0].width + 1) / 2, f->tiles[0].height, GL_RGBA);

        yuv_tex.upload_frame(f, pbo_frame);

        GLuint w_loc = glGetUniformLocation(program.get(), "width");
        glUniform1f(w_loc, f->tiles[0].width);

        quad.render();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
}



std::unique_ptr<Frame_convertor> get_convertor_for_codec(codec_t codec){
        switch(codec){
        case UYVY:
                return std::make_unique<Yuv_convertor>();
        default:
                return nullptr;
        }
}

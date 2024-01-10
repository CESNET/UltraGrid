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
#include "video_codec.h"
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

static void load_yuv_coefficients(GlProgram& program){
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
        load_yuv_coefficients(program);
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

/// with courtesy of https://stackoverflow.com/questions/20317882/how-can-i-correctly-unpack-a-v210-video-frame-using-glsl
/// adapted to GLSL 1.1 with help of https://stackoverflow.com/questions/5879403/opengl-texture-coordinates-in-pixel-space/5879551#5879551
static const char * v210_to_rgb_fp = R"raw(
#version 330

layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

uniform float width;

uniform float luma_scale = 1.1643f;
uniform float r_cr = 1.7926f;
uniform float g_cb = -0.2132f;
uniform float g_cr = -0.5328f;
uniform float b_cb = 2.1124f;

// YUV offset
const vec3 yuvOffset = vec3(-0.0625, -0.5, -0.5);

// RGB coefficients
vec3 Rcoeff = vec3(luma_scale, 0.0, r_cr);
vec3 Gcoeff = vec3(luma_scale, g_cb, g_cr);
vec3 Bcoeff = vec3(luma_scale, b_cb, 0.0);

// U Y V A | Y U Y A | V Y U A | Y V Y A

int GROUP_FOR_INDEX(int i) {
  return i / 4;
}

int SUBINDEX_FOR_INDEX(int i) {
  return i % 4;
}

int _y(int i) {
  return 2 * i + 1;
}

int _u(int i) {
  return 4 * (i/2);
}

int _v(int i) {
  return 4 * (i / 2) + 2;
}

int offset(int i) {
  return i + (i / 3);
}

vec3 ycbcr2rgb(vec3 yuvToConvert) {
  vec3 pix;
  yuvToConvert += yuvOffset;
  pix.r = dot(yuvToConvert, Rcoeff);
  pix.g = dot(yuvToConvert, Gcoeff);
  pix.b = dot(yuvToConvert, Bcoeff);
  return pix;
}

void main(void) {
  float imageWidthRaw; // v210 texture size
  imageWidthRaw = float((int(width) + 47) / 48 * 32); // 720->480

  // interpolate (0,1) texcoords to [0,719]
  int texcoordDenormX;
  texcoordDenormX = int(round(UV.x * width - .5));

  // 0 1 1 2 3 3 4 5 5 6 7 7 etc.
  int yOffset;
  yOffset = offset(_y(texcoordDenormX));
  int sourceColumnIndexY;
  sourceColumnIndexY = GROUP_FOR_INDEX(yOffset);

  // 0 0 1 1 2 2 4 4 5 5 6 6 etc.
  int uOffset;
  uOffset = offset(_u(texcoordDenormX));
  int sourceColumnIndexU;
  sourceColumnIndexU = GROUP_FOR_INDEX(uOffset);

  // 0 0 2 2 3 3 4 4 6 6 7 7 etc.
  int vOffset;
  vOffset = offset(_v(texcoordDenormX));
  int sourceColumnIndexV;
  sourceColumnIndexV = GROUP_FOR_INDEX(vOffset);

  // 1 0 2 1 0 2 1 0 2 etc.
  int compY;
  compY = SUBINDEX_FOR_INDEX(yOffset);

  // 0 0 1 1 2 2 0 0 1 1 2 2 etc.
  int compU;
  compU = SUBINDEX_FOR_INDEX(uOffset);

  // 2 2 0 0 1 1 2 2 0 0 1 1 etc.
  int compV;
  compV = SUBINDEX_FOR_INDEX(vOffset);

  vec4 y;
  vec4 u;
  vec4 v;
  y = texture(tex, vec2((float(sourceColumnIndexY) + .5) / imageWidthRaw, UV.y));
  u = texture(tex, vec2((float(sourceColumnIndexU) + .5) / imageWidthRaw, UV.y));
  v = texture(tex, vec2((float(sourceColumnIndexV) + .5) / imageWidthRaw, UV.y));

  vec3 outColor = ycbcr2rgb(vec3(y[compY], u[compU], v[compV]));

  color = vec4(outColor, 1.0);
}
)raw";

class V210_convertor : public Frame_convertor{
public:
        V210_convertor(): program(vert_src, v210_to_rgb_fp),
        quad(Model::get_quad())
        {
                load_yuv_coefficients(program);
        }

        void put_frame(video_frame *f, bool pbo_frame = false) override{
                glUseProgram(program.get());
                glBindFramebuffer(GL_FRAMEBUFFER, fbuf.get());
                glViewport(0, 0, f->tiles[0].width, f->tiles[0].height);
                glClear(GL_COLOR_BUFFER_BIT);

                //TODO
                int w = vc_get_linesize(f->tiles[0].width, v210) / 4;
                int h = f->tiles[0].height;
                yuv_tex.allocate(w, h, GL_RGB10_A2);
                glBindTexture(GL_TEXTURE_2D, yuv_tex.get());

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2,
                                w, h, 0,
                                GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,
                                f->tiles[0].data);

                GLuint w_loc = glGetUniformLocation(program.get(), "width");
                glUniform1f(w_loc, f->tiles[0].width);

                quad.render();

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glUseProgram(0);
        }

        void attach_texture(const Texture& tex) override {
                fbuf.attach_texture(tex);
        }

private:
        GlProgram program;// = GlProgram(vert_src, yuv_conv_frag_src);
        Model quad;// = Model::get_quad();
        Framebuffer fbuf;
        Texture yuv_tex;
};

static const char * yuva_to_rgb_fp = R"raw(
#version 330

layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

uniform float luma_scale = 1.1643f;
uniform float r_cr = 1.7926f;
uniform float g_cb = -0.2132f;
uniform float g_cr = -0.5328f;
uniform float b_cb = 2.1124f;
void main()
{
        vec4 yuv;
        yuv.rgba  = texture(tex, UV).grba;
        yuv.r = luma_scale * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        color.r = yuv.r + r_cr * yuv.b;
        color.g = yuv.r + g_cb * yuv.g + g_cr * yuv.b;
        color.b = yuv.r + b_cb * yuv.g;
        color.a = yuv.a;
}
)raw";

class Y416_convertor : public Frame_convertor{
public:
        Y416_convertor(): program(vert_src, yuva_to_rgb_fp),
        quad(Model::get_quad())
        {
                load_yuv_coefficients(program);
        }

        void put_frame(video_frame *f, bool pbo_frame = false) override{
                glUseProgram(program.get());
                glBindFramebuffer(GL_FRAMEBUFFER, fbuf.get());
                glViewport(0, 0, f->tiles[0].width, f->tiles[0].height);
                glClear(GL_COLOR_BUFFER_BIT);

                //TODO
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                yuv_tex.allocate(w, h, GL_RGBA);
                glBindTexture(GL_TEXTURE_2D, yuv_tex.get());

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                w, h, 0,
                                GL_RGBA, GL_UNSIGNED_SHORT,
                                f->tiles[0].data);

                quad.render();

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glUseProgram(0);
        }

        void attach_texture(const Texture& tex) override {
                fbuf.attach_texture(tex);
        }

private:
        GlProgram program;// = GlProgram(vert_src, yuv_conv_frag_src);
        Model quad;// = Model::get_quad();
        Framebuffer fbuf;
        Texture yuv_tex;
};



std::unique_ptr<Frame_convertor> get_convertor_for_codec(codec_t codec){
        switch(codec){
        case UYVY:
                return std::make_unique<Yuv_convertor>();
        case v210:
                return std::make_unique<V210_convertor>();
        case Y416:
                return std::make_unique<Y416_convertor>();
        default:
                return nullptr;
        }
}

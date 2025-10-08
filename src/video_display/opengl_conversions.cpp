/**
 * @file   video_display/opengl_conversions.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023-2025 CESNET, z. s. p. o.
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

#include "config.h"                // for HWACC_VDPAU
#include "color_space.h"
#include "debug.h"
#include "host.h"
#include "video_codec.h"
#include "opengl_conversions.hpp"
#include "gl_vdpau.hpp"

#define MOD_NAME "[GL conversions] "

namespace {

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
        const struct color_coeffs cfs = compute_color_coeffs(kr, kb, 8);
        glUniform1f(loc, (double) cfs.y_scale / (1 << COMP_BASE));
        loc = glGetUniformLocation(program.get(), "r_cr");
        glUniform1f(loc, (double) cfs.r_cr / (1 << COMP_BASE));
        loc = glGetUniformLocation(program.get(), "g_cr");
        glUniform1f(loc, (double) cfs.g_cb / (1 << COMP_BASE));
        loc = glGetUniformLocation(program.get(), "g_cb");
        glUniform1f(loc, (double) cfs.g_cr / (1 << COMP_BASE));
        loc = glGetUniformLocation(program.get(), "b_cb");
        glUniform1f(loc, (double) cfs.b_cb / (1 << COMP_BASE));
}

class Rendering_convertor : public Frame_convertor{
public:
        Rendering_convertor(const char* vert_src, const char* frag_src) :
                program(vert_src, frag_src),
                quad(Model::get_quad())
        {
                width_uniform_location = glGetUniformLocation(program.get(), "width");
        }

        virtual ~Rendering_convertor() {  };

        void attach_texture(Texture& tex) override final {
                fbuf.attach_texture(tex);
        }

        void put_frame(video_frame *f, bool pbo_frame) override final{
                glUseProgram(program.get());
                glBindFramebuffer(GL_FRAMEBUFFER, fbuf.get());
                glViewport(0, 0, f->tiles[0].width, f->tiles[0].height);
                glClear(GL_COLOR_BUFFER_BIT);

                prepare_input_tex(f, pbo_frame);

                if(width_uniform_location != -1)
                        glUniform1f(width_uniform_location, f->tiles[0].width);

                quad.render();

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glUseProgram(0);
        }

protected:
        virtual void prepare_input_tex(video_frame *f, bool pbo_frame) = 0;

        GlProgram program;
        Model quad;
        Gl_framebuffer fbuf;
        GLint width_uniform_location = -1;
        Texture input_tex;
};

class Loading_convertor : public Frame_convertor{
public:

        void attach_texture(Texture& tex) override final{
                this->tex = &tex;
        }

protected:
        Texture *tex = nullptr;
};

class Yuv_convertor : public Rendering_convertor{
public:
        Yuv_convertor() : Rendering_convertor(vert_src, yuv_conv_frag_src){
                load_yuv_coefficients(program);
        }

private:
        void prepare_input_tex(video_frame *f, bool pbo_frame) override final{
                int w = (f->tiles[0].width + 1) / 2;
                int h = f->tiles[0].height;

                input_tex.load_frame(w, h, GL_RGBA,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                f, pbo_frame);
        }
};

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

class V210_convertor : public Rendering_convertor{
public:
        V210_convertor(): Rendering_convertor(vert_src, v210_to_rgb_fp)
        {
                load_yuv_coefficients(program);
        }

private:
        void prepare_input_tex(video_frame *f, bool pbo_frame = false) override final{
                int w = vc_get_linesize(f->tiles[0].width, v210) / 4;
                int h = f->tiles[0].height;
                input_tex.load_frame(w, h, GL_RGB10_A2,
                                GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,
                                f, pbo_frame);
        }
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

class Y416_convertor : public Rendering_convertor{
public:
        Y416_convertor(): Rendering_convertor(vert_src, yuva_to_rgb_fp)
        {
                load_yuv_coefficients(program);
        }

private:
        void prepare_input_tex(video_frame *f, bool pbo_frame = false) override final{
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                input_tex.load_frame(w, h, GL_RGBA,
                                GL_RGBA, GL_UNSIGNED_SHORT,
                                f, pbo_frame);
        }
};

class DXT1_convertor : public Loading_convertor{
public:
        void put_frame(video_frame *f, bool) override{
                int w = (f->tiles[0].width + 3) / 4 * 4;
                int h = (f->tiles[0].height + 3) / 4 * 4;
                glBindTexture(GL_TEXTURE_2D, tex->get());
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                w, h, 0,
                                w * h / 2,
                                f->tiles[0].data);
        }
};

static const char fp_display_dxt5ycocg[] = R"raw(
#version 330

layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

void main()
{
        vec4 element = texture(tex, UV);
        float scale = (element.z * ( 255.0 / 8.0 )) + 1.0;
        float Co = (element.x - (0.5 * 256.0 / 255.0)) / scale;
        float Cg = (element.y - (0.5 * 256.0 / 255.0)) / scale;
        float Y = element.w;
        color = vec4(Y + Co - Cg, Y + Cg, Y - Co - Cg, 1.0);
}
)raw";

class DXT5_convertor : public Rendering_convertor{
public:
        DXT5_convertor(): Rendering_convertor(vert_src, fp_display_dxt5ycocg) {  }

private:
        void prepare_input_tex(video_frame *f, bool) override final{
                int w = (f->tiles[0].width + 3) / 4 * 4;
                int h = (f->tiles[0].height + 3) / 4 * 4;
                input_tex.allocate();
                glBindTexture(GL_TEXTURE_2D, input_tex.get());
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                w, h, 0,
                                w * h,
                                f->tiles[0].data);
        }
};

static const char *fp_display_dxt1_yuv = R"raw(
#version 330

layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

void main(void) {
        vec4 col = texture(tex, UV);

        float Y = 1.1643 * (col[0] - 0.0625);
        float U = (col[1] - 0.5);
        float V = (col[2] - 0.5);

        float R = Y + 1.7926 * V;
        float G = Y - 0.2132 * U - 0.5328 * V;
        float B = Y + 2.1124 * U;

        color = vec4(R,G,B,1.0);
}
)raw";

class DXT1_YUV_convertor : public Rendering_convertor{
public:
        DXT1_YUV_convertor(): Rendering_convertor(vert_src, fp_display_dxt1_yuv) {  }

private:
        void prepare_input_tex(video_frame *f, bool) override final{
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                input_tex.allocate();
                glBindTexture(GL_TEXTURE_2D, input_tex.get());
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                w, h, 0,
                                (w * h/16) * 8,
                                f->tiles[0].data);
        }
};

class R10k_convertor : public Loading_convertor{
public:
        void put_frame(video_frame *f, bool = false) override{
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                uint32_t *dst = nullptr;
                int data_len = w * h * 8;
                char *tex_src_data = nullptr;
                if(internal_pbo){
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, internal_pbo->get());
                        glBufferData(GL_PIXEL_UNPACK_BUFFER, data_len, 0, GL_STREAM_DRAW);
                        dst = (uint32_t *) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
                        process_r10k(dst, reinterpret_cast<uint32_t *>(f->tiles[0].data), w, h);
                        tex_src_data = nullptr;
                        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
                } else {
                        scratchpad.resize(w * h * 8);
                        dst = reinterpret_cast<uint32_t *>(scratchpad.data());
                        tex_src_data = scratchpad.data();
                        process_r10k(dst, reinterpret_cast<uint32_t *>(f->tiles[0].data), w, h);
                }

                glBindTexture(GL_TEXTURE_2D, tex->get());
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, tex_src_data);

                if(internal_pbo){
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                }
        }

private:
        void process_r10k(uint32_t * __restrict out, const uint32_t *__restrict in, long width, long height) {
                long line_padding_b = vc_get_linesize(width, R10k) - 4 * width;
                for (long i = 0; i < height; i += 1) {
                        OPTIMIZED_FOR (long j = 0; j < width; j += 1) {
                                uint32_t x = *in++;
                                *out++ = /* output is x2b8g8r8 little-endian */
                                        (x & 0xFFU) << 2U | (x & 0xC0'00U) >> 14U | // R
                                        (x & 0x3F'00U) << 6U | (x & 0xF0'00'00) >> 10U | // G
                                        (x & 0x0F'00'00U) << 10U | (x & 0xFC'00'00'00U) >> 6U; // B
                        }
                        in += line_padding_b / sizeof(uint32_t);
                }
        }
        std::vector<char> scratchpad;
};

class RGB_convertor : public Loading_convertor{
public:
        void put_frame(video_frame *f, bool pbo_frame = false) override{
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                tex->load_frame(w, h, GL_RGB,
                                GL_RGB, GL_UNSIGNED_BYTE,
                                f, pbo_frame);
        }
};

class RGBA_convertor : public Loading_convertor{
public:
        void put_frame(video_frame *f, bool pbo_frame = false) override{
                int w = f->tiles[0].width;
                int h = f->tiles[0].height;
                tex->load_frame(w, h, GL_RGBA,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                f, pbo_frame);
        }
};

static const char fp_passthrough[] = R"raw(
#version 330

layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;

void main()
{
        color = texture(tex, UV);
}
)raw";

#ifdef HWACC_VDPAU
class VDPAU_convertor : public Rendering_convertor{
public:
        VDPAU_convertor(): Rendering_convertor(vert_src, fp_passthrough) {
                vdp.init();
        }

        ~VDPAU_convertor() {
                vdp.uninit();
        }

private:
        void prepare_input_tex(video_frame *f, bool = false) override{
                vdp.loadFrame(reinterpret_cast<hw_vdpau_frame *>(f->tiles[0].data));
        }

        struct state_vdpau vdp;
};
#endif

struct {
        codec_t codec;
        std::unique_ptr<Frame_convertor> (*construct_func)();
} codec_convertor_map[]{
        {RGB, &Frame_convertor::construct_unique<RGB_convertor>},
        {RGBA, &Frame_convertor::construct_unique<RGBA_convertor>},
        {UYVY, &Frame_convertor::construct_unique<Yuv_convertor>},
        {v210, &Frame_convertor::construct_unique<V210_convertor>},
        {Y416, &Frame_convertor::construct_unique<Y416_convertor>},
        {DXT1, &Frame_convertor::construct_unique<DXT1_convertor>},
        {DXT1, &Frame_convertor::construct_unique<DXT1_YUV_convertor>},
        {DXT5, &Frame_convertor::construct_unique<DXT5_convertor>},
        {R10k, &Frame_convertor::construct_unique<R10k_convertor>},
#ifdef HWACC_VDPAU
        {HW_VDPAU, &Frame_convertor::construct_unique<VDPAU_convertor>},
#endif
};

} //anon namespace

std::unique_ptr<Frame_convertor> get_convertor_for_codec(codec_t codec){
        for(const auto& i : codec_convertor_map){
                if(i.codec == codec)
                        return i.construct_func();
        }

        return nullptr;
}

std::vector<codec_t> get_convertor_supported_codecs(){
        std::vector<codec_t> ret;
        ret.reserve(std::size(codec_convertor_map));
        for(const auto& i : codec_convertor_map){
                ret.push_back(i.codec);
        }
        return ret;
}

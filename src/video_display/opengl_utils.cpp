/**
 * @file   video_display/opengl_utils.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2025 CESNET, z. s. p. o.
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
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include "opengl_utils.hpp"
#include "opengl_conversions.hpp"
#include "video_frame.h"
#include "video_codec.h"
#include "debug.h"
#include "host.h"

#include "utils/profile_timer.hpp"

#define MOD_NAME "[Opengl utils] "

static const float PI_F=3.14159265358979f;

static const GLfloat rectangle[] = {
        1.0f,  1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  0.0f,  1.0f,
        -1.0f, -1.0f,  0.0f,  0.0f,

        1.0f,  1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  0.0f,  0.0f,
        1.0f, -1.0f,  1.0f,  0.0f
};

static unsigned char pixels[] = {
        255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
        255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
        255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
        255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255
};

static void compileShader(GLuint shaderId){
        glCompileShader(shaderId);

        GLint ret = GL_FALSE;
        int len;

        glGetShaderiv(shaderId, GL_COMPILE_STATUS, &ret);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &len);
        if (len > 0){
                std::vector<char> errorMsg(len+1);
                glGetShaderInfoLog(shaderId, len, NULL, &errorMsg[0]);
                printf("%s\n", errorMsg.data());
        }
}

static std::vector<float> gen_sphere_vertices(int r, int latitude_n, int longtitude_n);
static std::vector<unsigned> gen_sphere_indices(int latitude_n, int longtitude_n);

bool check_gl_extension_present(const char *ext){
        int count = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &count);

        for(int i = 0; i < count; i++){
                const char *item = (const char *) glGetStringi(GL_EXTENSIONS, i);
                if(strcmp(item, ext) == 0)
                        return true;
        }

        return false;
}

GlProgram::GlProgram(const char *vert_src, const char *frag_src){
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

        glShaderSource(vertexShader, 1, &vert_src, NULL);
        compileShader(vertexShader);
        glShaderSource(fragShader, 1, &frag_src, NULL);
        compileShader(fragShader);

        program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);
        //glUseProgram(program);

        glDetachShader(program, vertexShader);
        glDetachShader(program, fragShader);
        glDeleteShader(vertexShader);
        glDeleteShader(fragShader);
}

GlProgram::~GlProgram() {
        if(program == 0)
                return;

        glUseProgram(0);
        glDeleteProgram(program);
}

Model::~Model(){
        if(vao == 0)
                return;

        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &elem_buf);
        glDeleteVertexArrays(1, &vao);
}

void Model::render(){
        PROFILE_FUNC;

        glBindVertexArray(vao);
        if(elem_buf != 0){
                glDrawElements(GL_TRIANGLES, indices_num, GL_UNSIGNED_INT, (void *) 0);
        } else {
                glDrawArrays(GL_TRIANGLES, 0, indices_num);
        }

        glBindVertexArray(0);
}

Model Model::get_sphere(){
        Model model;
        glGenVertexArrays(1, &model.vao);
        glBindVertexArray(model.vao);

        auto vertices = gen_sphere_vertices(1, 64, 64);
        auto indices = gen_sphere_indices(64, 64);

        glGenBuffers(1, &model.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &model.elem_buf);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model.elem_buf);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glVertexAttribPointer(
                        0,
                        3,
                        GL_FLOAT,
                        GL_FALSE,
                        5 * sizeof(float),
                        (void*)0
                        );
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glVertexAttribPointer(
                        1,
                        2,
                        GL_FLOAT,
                        GL_FALSE,
                        5 * sizeof(float),
                        (void*)(3 * sizeof(float))
                        );

        glBindVertexArray(0);
        model.indices_num = indices.size();

        return model;
}

Model Model::get_quad(){
        Model model;
        glGenVertexArrays(1, &model.vao);
        glBindVertexArray(model.vao);

        glGenBuffers(1, &model.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle), rectangle, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glVertexAttribPointer(
                        0,
                        2,
                        GL_FLOAT,
                        GL_FALSE,
                        4 * sizeof(float),
                        (void*)0
                        );
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
        glVertexAttribPointer(
                        1,
                        2,
                        GL_FLOAT,
                        GL_FALSE,
                        4 * sizeof(float),
                        (void*)(2 * sizeof(float))
                        );

        glBindVertexArray(0);
        model.indices_num = 6;

        return model;
}

Texture::~Texture(){
        if(tex_id == 0)
                return;

        glDeleteTextures(1, &tex_id);
}

void Texture::allocate(int w, int h, GLint internal_fmt) {
        allocate();
        if(w != width || h != height || internal_fmt != internal_format){
                width = w;
                height = h;
                internal_format = internal_fmt;

                glBindTexture(GL_TEXTURE_2D, tex_id);
                /* Only the internalformat parameter is relevant here, the
                 * format and type parameters refer to the source data and we
                 * are passing nullptr here
                 */
                glTexImage2D(GL_TEXTURE_2D, 0, internal_format,
                                width, height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                nullptr);
        }
}

void Texture::load_frame(int w, int h,
                GLint internal_format,
                GLenum src_format,
                GLenum type,
                video_frame *f,
                bool pbo_frame)
{
        char *src_data = f->tiles[0].data;
        allocate(w, h, internal_format);
        glBindTexture(GL_TEXTURE_2D, tex_id);

        if(pbo_frame){
                GlBuffer *pbo = static_cast<GlBuffer *>(f->callbacks.dispose_udata);

                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->get());
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
                f->tiles[0].data = nullptr;
                src_data = nullptr;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, internal_format,
                        w, h, 0,
                        src_format, type,
                        src_data);

        if(pbo_frame){
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }
}

void Texture::allocate(){
        if(tex_id != 0)
                return;

        glGenTextures(1, &tex_id);
        glBindTexture(GL_TEXTURE_2D, tex_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


void Gl_framebuffer::attach_texture(GLuint tex){
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);


        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
                printf("Error %d\n", glGetError());
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameUploader::put_frame(video_frame *f, bool pbo_frame){
        assert(tex);
        tex->allocate(f->tiles[0].width, f->tiles[0].height, GL_RGB);
        glBindTexture(GL_TEXTURE_2D, tex->get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        if(!conv || f->color_spec != configured_codec){
                conv = get_convertor_for_codec(f->color_spec);
                conv->set_pbo(upload_pbo.get());
                configured_codec = f->color_spec;
        }
        assert(conv && "Frame uploader for codec not A;");
        conv->attach_texture(*tex);
        conv->put_frame(f, pbo_frame);
}

std::vector<codec_t> FrameUploader::get_supported_codecs(){
        std::vector<codec_t> ret;

        ret.push_back(RGB);
        ret.push_back(RGBA);

        auto cvt = get_convertor_supported_codecs();
        ret.insert(ret.end(), cvt.begin(), cvt.end());

        return ret;
}

void FlatVideoScene::init(){
        const char *vert_src = R"END(
#version 330 core
layout(location = 0) in vec2 vert_pos;
layout(location = 1) in vec2 vert_uv;

uniform vec2 scale_vec;

out vec2 UV;

void main(){
        gl_Position = vec4(vert_pos * scale_vec, 0.0f, 1.0f);
        UV = vert_uv;
}
)END";

        const char *frag_src = R"END(
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D tex;
uniform bool deinterlace = false;
void main(){
        color = texture(tex, UV).rgb;

        if(deinterlace){
                float lineOff = 1.0f / textureSize(tex, 0).y;
                vec3 pix_down = texture(tex, vec2(UV.x, UV.y + lineOff)).rgb;
                color = (color + pix_down) / 2.0f;
        }
}
)END";

        program = GlProgram(vert_src, frag_src);
        quad = Model::get_quad();
        texture.allocate();
        uploader.attach_dst_texture(&texture);
        uploader.enable_pbo(false);
}

void FlatVideoScene::put_frame(video_frame *f){
        auto frame_desc = video_desc_from_frame(f);
        if(!video_desc_eq(frame_desc, current_desc)){
                current_desc = frame_desc;
                resize(screen_width, screen_height);
        }

        uploader.put_frame(f, false);
}

void FlatVideoScene::render(){
        glUseProgram(program.get());

        glViewport(0, 0, screen_width, screen_height);
        glBindTexture(GL_TEXTURE_2D, texture.get());
        quad.render();
}

void FlatVideoScene::resize(int width, int height){
        screen_width = width;
        screen_height = height;

        double x = 1.0;
        double y = -1.0;

        double screen_aspect = (double) width / height;
        double video_aspect = (double) current_desc.width / current_desc.height;
        if(screen_aspect > video_aspect) {
                x = (double) height * video_aspect / width;
        } else {
                y = (double) -width / (height * video_aspect);
        }

        glUseProgram(program.get());

        GLuint pvLoc;
        pvLoc = glGetUniformLocation(program.get(), "scale_vec");
        glUniform2f(pvLoc, x, y);
}

void FlatVideoScene::enableDeinterlacing(bool enable){
        glUseProgram(program.get());

        GLuint pvLoc;
        pvLoc = glGetUniformLocation(program.get(), "deinterlace");
        glUniform1i(pvLoc, enable);
}

static std::vector<float> gen_sphere_vertices(int r, int latitude_n, int longtitude_n){
        std::vector<float> verts;

        float lat_step = PI_F / latitude_n;
        float long_step = 2 * PI_F / longtitude_n;

        for(int i = 0; i < latitude_n + 1; i++){
                float y = std::cos(i * lat_step) * r;
                float y_slice_r = std::sin(i * lat_step) * r;

                //The first and last vertex on the y slice circle are in the same place
                for(int j = 0; j < longtitude_n + 1; j++){
                        float x = std::sin(j * long_step) * y_slice_r;
                        float z = std::cos(j * long_step) * y_slice_r;
                        verts.push_back(x);
                        verts.push_back(y);
                        verts.push_back(z);

                        float u = 1 - static_cast<float>(j) / longtitude_n;
                        float v = static_cast<float>(i) / latitude_n;
                        verts.push_back(u);
                        verts.push_back(v);
                }
        }

        return verts;
}

//Generate indices for sphere
//Faces facing inwards have counter-clockwise vertex order
static std::vector<unsigned> gen_sphere_indices(int latitude_n, int longtitude_n){
        std::vector<unsigned int> indices;

        for(int i = 0; i < latitude_n; i++){
                int slice_idx = i * (latitude_n + 1);
                int next_slice_idx = (i + 1) * (latitude_n + 1);

                for(int j = 0; j < longtitude_n; j++){
                        //Since the top and bottom slices are circles with radius 0,
                        //we only need one triangle for those
                        if(i != latitude_n - 1){
                                indices.push_back(slice_idx + j + 1);
                                indices.push_back(next_slice_idx + j);
                                indices.push_back(next_slice_idx + j + 1);
                        }

                        if(i != 0){
                                indices.push_back(slice_idx + j + 1);
                                indices.push_back(slice_idx + j);
                                indices.push_back(next_slice_idx + j);
                        }
                }
        }

        return indices;
}


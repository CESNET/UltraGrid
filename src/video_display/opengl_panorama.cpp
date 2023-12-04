/**
 * @file   video_display/opengl_panorama.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include "opengl_utils.hpp"
#include "video_frame.h"
#include "debug.h"

#include "opengl_panorama.hpp"
#include "utils/profile_timer.hpp"

static const char *persp_vert_src = R"END(
#version 330 core
layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec2 vert_uv;

out vec2 UV;

uniform mat4 pv_mat;

void main(){
        gl_Position = pv_mat * vec4(vert_pos, 1.0f);
        UV = vert_uv;
}
)END";

static const char *persp_frag_src = R"END(
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D tex;
void main(){
        color = texture(tex, UV).rgb;
}
)END";


PanoramaScene::PanoramaScene(): PanoramaScene(GlProgram(persp_vert_src, persp_frag_src),
        Model::get_sphere()) { }

PanoramaScene::PanoramaScene(GlProgram program, Model model): program(std::move(program)),
        model(std::move(model)) {  }

void PanoramaScene::render(int width, int height){
        float aspect_ratio = static_cast<float>(width) / height;
        glm::mat4 projMat = glm::perspective(glm::radians(fov),
                        aspect_ratio,
                        0.1f,
                        300.0f);
        glm::mat4 viewMat(1.f);
        viewMat = glm::rotate(viewMat, glm::radians(rot_y), {1.f, 0, 0});
        viewMat = glm::rotate(viewMat, glm::radians(rot_x), {0.f, 1, 0});
        glm::mat4 pvMat = projMat * viewMat;

        render(width, height, pvMat);
}

void PanoramaScene::render(int width, int height, const glm::mat4& pvMat){
        PROFILE_FUNC;

        glUseProgram(program.get());
        glViewport(0, 0, width, height);
        GLuint pvLoc;
        pvLoc = glGetUniformLocation(program.get(), "pv_mat");
        glUniformMatrix4fv(pvLoc, 1, GL_FALSE, glm::value_ptr(pvMat));

        {//lock
                std::lock_guard<std::mutex> lock(tex_mut);
                if(tex.is_new_front_available()){
                        tex.swap_front();
                }
        }

        glBindTexture(GL_TEXTURE_2D, tex.get_front().get());
        model.render();
}

void PanoramaScene::put_frame(video_frame *f, bool pbo_frame){
        PROFILE_FUNC;

        Texture& back_texture = tex.get_back();

        glBindTexture(GL_TEXTURE_2D, back_texture.get());
        back_texture.allocate(f->tiles[0].width, f->tiles[0].height, GL_RGB);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        if(f->color_spec == UYVY){
                if(!conv){
                        conv = std::unique_ptr<Yuv_convertor>(new Yuv_convertor());
                }
                conv->attach_texture(back_texture);
                conv->put_frame(f, pbo_frame);
        } else {
                back_texture.upload_frame(f, pbo_frame);
        }

        glFinish();

        {
                std::lock_guard<std::mutex> lock(tex_mut);
                tex.swap_back();
        }
}

void PanoramaScene::rotate(float dx, float dy){
        rot_x -= dx;
        rot_y -= dy;

        if(rot_y > 90) rot_y = 90;
        if(rot_y < -90) rot_y = -90;
}


/**
 * @file   video_display/opengl_panorama.hpp
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
#ifndef OPENGL_PANORAMA_HPP_248d1fff7721
#define OPENGL_PANORAMA_HPP_248d1fff7721

#include <glm/glm.hpp>
#include "opengl_utils.hpp"

/**
 * Class representing the scene to render. Contains the model and texture to
 * be rendered. Automatically performs pixel format coversions if needed.
 * It is possible to call put_frame() from another thread, but limitations apply
 */
struct PanoramaScene{
        PanoramaScene();
        PanoramaScene(GlProgram program, Model model);

        /**
         * Renders the scene.
         *
         * @param width Width of result used for aspect ration correction
         * @param height Height of result used for aspect ration correction
         */
        void render(int width, int height);

        /**
         * Renders the scene.
         *
         * @param width Width of result used for aspect ration correction
         * @param height Height of result used for aspect ration correction
         * @param pvMat custom perspective and view Matrix to be used
         */
        void render(int width, int height, const glm::mat4& pvMat);

        /**
         * Uploads a video frame to be rendered. Can be called from other
         * thread, but must be always called from the thread that called it the
         * first time.
         *
         * @param f video frame to upload
         * @param pbo_frame true if video frame contains the image data
         * in a PBO buffer
         */
        void put_frame(video_frame *f, bool pbo_frame = false);

        /**
         * Rotate the view. The vertical rotation is constrained to -90 to 90
         * degrees.
         *
         * @param dx horizontal angle to rotate by
         * @param dy vertical angle to rotate by
         */
        void rotate(float dx, float dy);

        std::vector<codec_t> get_codecs() { return uploader.get_supported_codecs(); }

        GlProgram program;// = GlProgram(persp_vert_src, persp_frag_src);
        Model model;// = Model::get_sphere();
        TripleBufferTexture tex;
        std::mutex tex_mut;

        FrameUploader uploader;
        float rot_x = 0;
        float rot_y = 0;
        float fov = 55;
};

#endif //OPENGL_PANORAMA_HPP_248d1fff7721


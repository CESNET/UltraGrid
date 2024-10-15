/**
 * @file   video_display/opengl_utils.hpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2023 CESNET, z. s. p. o.
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
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#ifdef HAVE_CONFIG_H
#       include "config.h"
#endif //HAVE_CONFIG_H

#ifdef __APPLE__
#       include <OpenGL/OpenGL.h> // CGL
#       include <OpenGL/gl3.h>
#       include <OpenGL/glext.h>
#elif defined __linux__
#       include <X11/Xlib.h>
#       include <GL/glew.h>
#       include <GL/glx.h>
#else // _WIN32
#       include <GL/glew.h>
#endif //__APPLE__

#include <SDL2/SDL.h>
//#include <SDL2/SDL_opengl.h>
#include <mutex>
#include <memory>
#include <vector>
#include <cassert>

#include "types.h"

bool check_gl_extension_present(const char *ext);

/**
 * RAII wrapper for OpenGL program
 */
class GlProgram{
public:
        /**
         * Compiles provided shaders and constructs an OpenGL program
         *
         * @param vert_src vertex shader source
         * @param frag_src fragment shader source
         */
        GlProgram() = default;
        GlProgram(const char *vert_src, const char *frag_src);
        ~GlProgram();

        /**
         * Returns the underlying OpenGL program id
         */
        GLuint get() { return program; }

        GlProgram(const GlProgram&) = delete;
        GlProgram(GlProgram&& o) { swap(o); }
        GlProgram& operator=(const GlProgram&) = delete;
        GlProgram& operator=(GlProgram&& o) { swap(o); return *this; }

private:
        void swap(GlProgram& o){
                std::swap(program, o.program);
        }
        GLuint program = 0;
};

/**
 * RAII wrapper for OpenGL VBO and VAO containing a 3D model
 */
class Model{
public:
        Model() = default;
        Model(const Model&) = delete;
        Model(Model&& o) { swap(o); }
        Model& operator=(const Model&) = delete;
        Model& operator=(Model&& o) { swap(o); return *this; }
        ~Model();

        /**
         * Return the underlying OpenGL VAO id
         */
        GLuint get_vao() const { return vao; }

        /**
         * Renders the contained model by calling glDrawElements or glDrawArrays
         */
        void render();

        /**
         * Returns an instance containing a 3D sphere
         */
        static Model get_sphere();

        /**
         * Returns an instance containing a plane formed by 2 triangles
         */
        static Model get_quad();

private:
        void swap(Model& o){
                std::swap(vao, o.vao);
                std::swap(vbo, o.vbo);
                std::swap(elem_buf, o.elem_buf);
                std::swap(indices_num, o.indices_num);
        }
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint elem_buf = 0;
        GLsizei indices_num = 0;
};

/**
 * RAII wrapper for OpenGL texture object
 */
class Texture{
public:
        /**
         * Constructs an instance containing a new OpenGL texture object
         */
        Texture() = default;
        ~Texture();

        /**
         * Returns the underlying OpenGL texture id. The texture needs to be
         * allocated first with the allocate member function.
         */
        GLuint get() const { assert(tex_id); return tex_id; }

        /**
         * Allocates the the storage for the texture according to requested
         * resolution and format
         *
         * @param w width of the image 
         * @param h height of the image 
         * @param internal_fmt same as internalformat parameter of glTexImage2D()
         */
        void allocate(int w, int h, GLint internal_fmt);

        void allocate();

        void load_frame(int w, int h,
                        GLint internal_format,
                        GLenum src_format,
                        GLenum type,
                        video_frame *f,
                        bool pbo_frame);

        Texture(const Texture&) = delete;
        Texture(Texture&& o) { swap(o); }
        Texture& operator=(const Texture&) = delete;
        Texture& operator=(Texture&& o) { swap(o); return *this; }


        /**
         * Swaps the contents of two Texture objects
         */
        void swap(Texture& o){
                std::swap(tex_id, o.tex_id);
                std::swap(width, o.width);
                std::swap(height, o.height);
                std::swap(internal_format, o.internal_format);
        }
private:
        GLuint tex_id = 0;
        int width = 0;
        int height = 0;
        GLint internal_format = 0;
};

/**
 * RAII wrapper for an OpenGL framebuffer object
 */
class Framebuffer{
public:
        /**
         * Constructs a new framebuffer object
         */
        Framebuffer(){
                glGenFramebuffers(1, &fbo);
        }

        ~Framebuffer(){
                glDeleteFramebuffers(1, &fbo);
        }

        Framebuffer(const Framebuffer&) = delete;
        Framebuffer(Framebuffer&& o) { swap(o); }
        Framebuffer& operator=(const Framebuffer&) = delete;
        Framebuffer& operator=(Framebuffer&& o) { swap(o); return *this; }

        /**
         * Returns the underlying OpenGL framebuffer id
         */
        GLuint get() { return fbo; }

        /**
         * Attaches a texture to the contained framebuffer
         *
         * @param tex Id of texture to attach
         */
        void attach_texture(GLuint tex);

        /**
         * Attaches a texture to the contained framebuffer
         *
         * @param tex Texture object containg the texture to attach
         */
        void attach_texture(const Texture& tex){
                attach_texture(tex.get());
        }

private:
        void swap(Framebuffer& o){
                std::swap(fbo, o.fbo);
        }

        GLuint fbo = 0;
};

/**
 * RAII wrapper for OpenGL buffer object
 */
class GlBuffer{
public:
        /**
         * Constructs a new OpenGL buffer
         */
        GlBuffer(){
                glGenBuffers(1, &buf_id);
        }

        ~GlBuffer(){
                glDeleteBuffers(1, &buf_id);
        }

        /**
         * Returns the underlying OpenGL buffer
         */
        GLuint get() {
                return buf_id;
        }

        GlBuffer(const GlBuffer&) = delete;
        GlBuffer(GlBuffer&& o) { swap(o); }
        GlBuffer& operator=(const GlBuffer&) = delete;
        GlBuffer& operator=(GlBuffer&& o) { swap(o); return *this; }

private:
        void swap(GlBuffer& o){
                std::swap(buf_id, o.buf_id);
        }

        GLuint buf_id = 0;
};

class Frame_convertor{
public:
        virtual ~Frame_convertor() {};

        template<class Derived>
        static std::unique_ptr<Frame_convertor> construct_unique() {
                return std::make_unique<Derived>();
        }

        /**
         * Renders the video frame containing YUV image data to attached texture
         *
         * @param f video frame to convert
         * @param pbo_frame true if video frame contains the image data
         * in a PBO buffer
         */
        virtual void put_frame(video_frame *f, bool pbo_frame = false) = 0;

        /**
         * Attach texture to be used as output
         *
         * @param tex texture to attach
         */
        virtual void attach_texture(Texture& tex) = 0;

        void set_pbo(GlBuffer *pbo) { internal_pbo = pbo; }
protected:
        GlBuffer *internal_pbo = nullptr;
};

/**
 * Class which takes an ultragrid video_frame and turns it into opengl texture
 */
class FrameUploader{
public:
        void put_frame(video_frame *f, bool pbo_frame = false);
        void attach_dst_texture(Texture *tex){ this->tex = tex; }

        void enable_pbo(bool enable) {
                if(enable)
                        upload_pbo = std::make_unique<GlBuffer>();
                else
                        upload_pbo.reset();
        }

        std::vector<codec_t> get_supported_codecs();

private:
        std::unique_ptr<Frame_convertor> conv;
        codec_t configured_codec = VIDEO_CODEC_NONE;
        Texture *tex = nullptr;
        std::unique_ptr<GlBuffer> upload_pbo;
};

/**
 * Class containing 3 Texture objects to use as a triple buffer.
 */
class TripleBufferTexture{
public:
        void allocateTextures() {
                back.allocate();
                mid.allocate();
                front.allocate();
        }

        /**
         * Returns a reference to the back texture
         */
        Texture& get_back(){ return back; }

        /**
         * Returns a reference to the front texture
         */
        Texture& get_front(){ return front; }

        /**
         * Should be called after writing to back texture is finished.
         * Swaps the back and mid textures and sets a flag indicating that a new
         * texture is available for reading
         */
        void swap_back(){
                back.swap(mid);
                new_front_available = true;
        }

        /**
         * Swaps front and mid texture. Unsets the flag indicating a new texture.
         */
        void swap_front(){
                front.swap(mid);
                new_front_available = false;
        }

        /**
         * Returns the state of the flag indicating a new texture
         */
        bool is_new_front_available() const { return new_front_available; }

private:
        Texture back;
        Texture front;
        Texture mid;

        bool new_front_available = false;
};


class FlatVideoScene{
public:
        FlatVideoScene() = default;

        void init();
        void put_frame(video_frame *f);
        void render();
        void resize(int width, int height);

        void enableDeinterlacing(bool enable);

        GLuint get_texture() { return texture.get(); }
private:
        GlProgram program;
        Model quad;
        Texture texture;
        FrameUploader uploader;

        video_desc current_desc;
        int screen_width;
        int screen_height;
};

#endif

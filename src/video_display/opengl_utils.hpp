/**
 * @file   video_display/opengl_utils.hpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2021 CESNET, z. s. p. o.
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

#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <mutex>
#include <memory>

#include "types.h"

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
        Model() = default;
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
        Texture();
        ~Texture();

        /**
         * Returns the underlying OpenGL texture id
         */
        GLuint get() const { return tex_id; }

        /**
         * Allocates the the storage for the texture according to requested
         * resolution and format
         *
         * @param w width of the image 
         * @param h height of the image 
         * @param fmt format of the image (same as internalformat parameter of 
         * glTexSubImage2D())
         */
        void allocate(int w, int h, GLenum fmt);

        /**
         * Uploads image data to the texture
         *
         * @param w width of the image 
         * @param h height of the image 
         * @param type format of the image (same as type parameter of 
         * glTexSubImage2D())
         * @param fmt format of the image (same as internalformat parameter of 
         * glTexSubImage2D())
         * @param data pointer to image data to upload
         * @param data_len length of image data to upload
         */
        void upload(size_t w, size_t h,
                        GLenum fmt, GLenum type,
                        const void *data, size_t data_len);

        /**
         * Uploads image data to the texture
         *
         * @param w width of the image 
         * @param h height of the image 
         * @param type format of the image (same as type parameter of 
         * glTexSubImage2D())
         * @param fmt format of the image (same as internalformat parameter of 
         * glTexSubImage2D())
         * @param data pointer to image data to upload
         * @param data_len length of image data to upload
         */
        void upload_internal_pbo(size_t w, size_t h,
                        GLenum fmt, GLenum type,
                        const void *data, size_t data_len);

        /**
         * Uploads video frame to the texture
         *
         * @param f video frame to upload
         * @pbo_frame true if the video frame contains the image data
         * in a PBO buffer
         */
        void upload_frame(video_frame *f, bool pbo_frame);

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
                std::swap(format, o.format);
                std::swap(pbo, o.pbo);
        }
private:
        GLuint tex_id = 0;
        int width = 0;
        int height = 0;
        GLenum format = 0;

        GLuint pbo;
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

/**
 * Class used to convert YUV textures to RGB textures
 */
class Yuv_convertor{
public:
        Yuv_convertor();

        /**
         * Renders the video frame containing YUV image data to attached texture
         *
         * @param f video frame to convert
         * @param pbo_frame true if video frame contains the image data
         * in a PBO buffer
         */
        void put_frame(video_frame *f, bool pbo_frame = false);

        /**
         * Attach texture to be used as output
         *
         * @param tex texture to attach
         */
        void attach_texture(const Texture& tex){
                fbuf.attach_texture(tex);
        }

private:
        GlProgram program;// = GlProgram(vert_src, yuv_conv_frag_src);
        Model quad;// = Model::get_quad();
        Framebuffer fbuf;
        Texture yuv_tex;
};

/**
 * Class containing 3 Texture objects to use as a triple buffer.
 */
class TripleBufferTexture{
public:
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

/**
 * Class representing the scene to render. Contains the model and texture to
 * be rendered. Automatically performs pixel format coversions if needed.
 * It is possible to call put_frame() from another thread, but limitations apply
 */
struct Scene{
        Scene();

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

        GlProgram program;// = GlProgram(persp_vert_src, persp_frag_src);
        Model model;// = Model::get_sphere();
        TripleBufferTexture tex;
        std::mutex tex_mut;

        Framebuffer framebuffer;
        std::unique_ptr<Yuv_convertor> conv;
        float rot_x = 0;
        float rot_y = 0;
        float fov = 55;

        int width, height;
};

/**
 * RAII wrapper for Sdl window
 */
struct Sdl_window{
        /**
         * Creates a window using default parameters.
         *
         * @param double_buffer Controls the double buffering. If true, the default
         * is used, if false it is disabled
         */
        Sdl_window(bool double_buffer = false);

        /**
         * Creates a window.
         *
         * @param title window title
         * @param x x position of the window, for centering
         * SDL_WINDOWPOS_CENTERED can be passed
         * @param y y position of the window, for centering
         * SDL_WINDOWPOS_CENTERED can be passed
         * @param w width of window
         * @param h height of window
         * @param flags Sdl flags for window creation
         * @param double_buffer Controls the double buffering. If true, the default
         * is used, if false it is disabled
         */
        Sdl_window(const char *title,
                        int x, int y, int w, int h,
                        Uint32 flags, bool double_buffer = false);

        ~Sdl_window();

        Sdl_window(const Sdl_window&) = delete;
        Sdl_window(Sdl_window&& o) { swap(o); }
        Sdl_window& operator=(const Sdl_window&) = delete;
        Sdl_window& operator=(Sdl_window&& o) { swap(o); return *this; }

        /**
         * Used to obtain Xlib window handles
         */
        void getXlibHandles(Display  **xDisplay,
                        GLXContext *glxContext,
                        GLXDrawable *glxDrawable);

        /**
         * Sets window title
         *
         * @param title Title to use
         */
        void set_title(const std::string& title){
                SDL_SetWindowTitle(sdl_window, title.c_str());
        }

        /**
         * If using multiple OpenGL contexts for multithreading this function
         * makes the render context current
         */
        void make_render_context_current();

        /**
         * If using multiple OpenGL contexts for multithreading this function
         * makes the worker context current
         */
        void make_worker_context_current();

        /**
         * Returns the underlying worker context object. If it doesn'ŧ yet
         * exists it gets created. Should be called at least once from the
         * rendering thread before the worker thread starts.
         */
        SDL_GLContext get_worker_context();

        void swap(Sdl_window& o);

        SDL_Window *sdl_window;
        SDL_GLContext sdl_gl_context;
        SDL_GLContext sdl_gl_worker_context = nullptr;
        int width;
        int height;
};

#endif

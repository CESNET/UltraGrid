/**
 * @file   video_display/sdL_window.hpp
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
#ifndef SDL_WINDOW_HPP
#define SDL_WINDOW_HPP

#ifdef HAVE_CONFIG_H
#       include "config.h"
#endif //HAVE_CONFIG_H

#ifdef HAVE_MACOSX
#       include <OpenGL/OpenGL.h> // CGL
#       include <OpenGL/gl3.h>
#       include <OpenGL/glext.h>
#elif defined HAVE_LINUX
#       include <X11/Xlib.h>
#       include <GL/glew.h>
#       include <GL/glx.h>
#else // WIN32
#       include <GL/glew.h>
#       include <GL/glut.h>
#endif //HAVE_MACOSX



#include <string>
#include <SDL2/SDL.h>

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

#ifdef HAVE_LINUX
        /**
         * Used to obtain Xlib window handles
         */
        void getXlibHandles(Display  **xDisplay,
                        GLXContext *glxContext,
                        GLXDrawable *glxDrawable);
#endif //HAVE_LINUX

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

#endif //SDL_WINDOW_HPP

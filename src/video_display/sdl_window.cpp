/**
 * @file   video_display/sdL_window.cpp
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

#include <stdexcept>
#include "sdl_window.hpp"

Sdl_window::Sdl_window(bool double_buffer) : Sdl_window("UltraGrid",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                640,
                480,
                SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE, double_buffer) {  }

Sdl_window::Sdl_window(const char *title,
                int x, int y,
                int w, int h,
                Uint32 flags, bool double_buffer) : width(w), height(h)
{
        SDL_InitSubSystem(SDL_INIT_VIDEO);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

        if(!double_buffer)
                SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);

        sdl_window = SDL_CreateWindow(title,
                        x,
                        y,
                        w,
                        h,
                        flags);

        if(!sdl_window){
                throw std::runtime_error("Failed to create SDL window!");
        }

        SDL_SetWindowMinimumSize(sdl_window, 200, 200);

        sdl_gl_context = SDL_GL_CreateContext(sdl_window);
        if(!sdl_gl_context){
                throw std::runtime_error("Failed to create gl context!");
        }

#ifndef HAVE_MACOSX
        glewExperimental = GL_TRUE;
        GLenum glewError = glewInit();
        if(glewError != GLEW_OK){
                throw std::runtime_error("Failed to initialize gl context!");
        }
#endif //HAVE_MACOSX

        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);
        SDL_GL_SwapWindow(sdl_window);
}

Sdl_window::~Sdl_window(){
        SDL_GL_DeleteContext(sdl_gl_context);
        SDL_GL_DeleteContext(sdl_gl_worker_context);
        SDL_DestroyWindow(sdl_window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

#ifdef HAVE_LINUX
void Sdl_window::getXlibHandles(Display  **xDisplay,
                GLXContext *glxContext,
                GLXDrawable *glxDrawable)
{
        SDL_GL_MakeCurrent(sdl_window, sdl_gl_context);
        *xDisplay = XOpenDisplay(NULL);
        *glxContext = glXGetCurrentContext();
        *glxDrawable = glXGetCurrentDrawable();
}
#endif //HAVE_LINUX

void Sdl_window::make_render_context_current(){
        SDL_GL_MakeCurrent(sdl_window, sdl_gl_context);
}

void Sdl_window::make_worker_context_current(){
        SDL_GLContext worker_context = get_worker_context();
        SDL_GL_MakeCurrent(sdl_window, worker_context);
}

SDL_GLContext Sdl_window::get_worker_context(){
        if(!sdl_gl_worker_context){
                SDL_GL_MakeCurrent(sdl_window, sdl_gl_context);
                SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
                sdl_gl_worker_context = SDL_GL_CreateContext(sdl_window);
                SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
                SDL_GL_MakeCurrent(sdl_window, sdl_gl_context);
        }

        return sdl_gl_worker_context;
}

void Sdl_window::swap(Sdl_window& o){
        std::swap(sdl_window, o.sdl_window);
        std::swap(sdl_gl_context, o.sdl_gl_context);
        std::swap(width, o.width);
        std::swap(height, o.height);
}

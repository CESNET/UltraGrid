/**
 * @file   win32_gl_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This context has to be run in a separate thread to ensure
 * that DC is stil valid because it is automatically released
 * when thread, where it is created, exits.
 */
/*
 * Copyright (c) 2013-2014 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <assert.h>
#include <GL/glew.h>
#include <GL/wglew.h>
#include <wingdi.h>
#include <winuser.h>

#include "debug.h"
#include "utils/windows.h"
#include "win32_gl_common.h"

#define VERSION_GET_MAJOR(version) (version >> 8u)
#define VERSION_GET_MINOR(version) (version & 0xFF)
#define VERSION_IS_UNSPECIFIED(version) (!version)
#define VERSION_IS_LEGACY(version) (version == OPENGL_VERSION_LEGACY)

#define WNDCLASSNAME "GLClass"

pthread_once_t register_windows_class_once = PTHREAD_ONCE_INIT;

enum status {
        UNKNOWN,
        SUCCESS,
        FAILED
};

struct state_win32_gl_context
{
        HWND win;
        HDC hdc;
        HGLRC hglrc;
        win32_opengl_version_t version;
        pthread_mutex_t lock;
        pthread_cond_t status_cv;
        pthread_t thread_id;
        enum status status;
        bool should_exit;
        pthread_cond_t loop_cv;
};

static HWND CreateWnd (int width, int height);
static BOOL SetGLFormat(HDC hdc);
static void PrintError(DWORD err);
static void create_and_register_windows_class(void);
static void win32_context_cleanup_common(struct state_win32_gl_context
                *context);

static void *owner_thread(void *arg) {
        struct state_win32_gl_context *context =
                (struct state_win32_gl_context *) arg;

        context->win = CreateWnd(512, 512);
        assert(context->win != NULL);

        if ((context->hdc = GetDC(context->win)) == NULL) // get device context
        {
                fprintf(stderr, "Failed to Get the Window Device Context\n");
                goto failed;
        }

        if(!SetGLFormat(context->hdc)) {
                fprintf(stderr, "Unable to set pixel format.\n");
                goto failed;
        }

        if ((context->hglrc = wglCreateContext(context->hdc)) == NULL) // create the rendering context
        {
                fprintf(stderr, "Failed to Create the OpenGL Rendering Context.\n");
                goto release_context;
        }

        if(wglMakeCurrent(context->hdc, context->hglrc) == FALSE) {
                fprintf(stderr, "Unable to make context current.\n");
                PrintError(GetLastError());
                goto release_context;
        }

        int err = glewInit();
        if(err != GLEW_OK) {
                fprintf(stderr, "Error intializing GLEW.\n");
                goto release_context;
        }

        if(!VERSION_IS_UNSPECIFIED(context->version) &&
                        !VERSION_IS_LEGACY(context->version)) {
                if(glewIsSupported("GL_ARB_create_context_profile") != GLEW_OK) {
                        fprintf(stderr, "OpenGL implementation is not capable "
                                        "of selecting profile.\n");
                        goto release_context;
                } else {
                        // first destroy current context
                        wglMakeCurrent(context->hdc, NULL);
                        wglDeleteContext(context->hglrc);

                        const int attrib_list[] = {
                                WGL_CONTEXT_MAJOR_VERSION_ARB, VERSION_GET_MAJOR(context->version),
                                WGL_CONTEXT_MINOR_VERSION_ARB, VERSION_GET_MINOR(context->version),
                                0
                        };

                        if ((context->hglrc = wglCreateContextAttribsARB(context->hdc, NULL, attrib_list)) == NULL)
                        {
                                fprintf(stderr, "Failed to Create the OpenGL Rendering Context %d,%d.\n",
                                                VERSION_GET_MAJOR(context->version),
                                                VERSION_GET_MINOR(context->version));
                                goto release_context;
                        }
                }
        }


        //ShowWindow(context->win, SW_SHOW);
        //UpdateWindow(context->win);
        wglMakeCurrent(context->hdc, NULL);

        pthread_mutex_lock(&context->lock);
        context->status = SUCCESS;
        pthread_cond_signal(&context->status_cv);
        pthread_mutex_unlock(&context->lock);

        pthread_mutex_lock(&context->lock);
        while (!context->should_exit)
                pthread_cond_wait(&context->loop_cv, &context->lock);
        pthread_mutex_unlock(&context->lock);

        wglMakeCurrent(context->hdc, NULL);
        wglDeleteContext(context->hglrc);

        PostQuitMessage(0);

        return NULL;

release_context:
        wglMakeCurrent(context->hdc, NULL);
        wglDeleteContext(context->hglrc);

failed:
        PostQuitMessage(0);
        pthread_mutex_lock(&context->lock);
        context->status = FAILED;
        pthread_cond_signal(&context->status_cv);
        pthread_mutex_unlock(&context->lock);
        
        return NULL;
}

static void PrintError(DWORD err)
{
    // Display the error message and exit the process
    log_msg(LOG_LEVEL_FATAL, "win32_gl_common %ld: %s\n", err, get_win_error(err));
}

static BOOL SetGLFormat(HDC hdc)
{
       // number of available formats
       int indexPixelFormat = 0;

       PIXELFORMATDESCRIPTOR pfd =
       {
               sizeof(PIXELFORMATDESCRIPTOR),
               1,
               PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
               PFD_TYPE_RGBA,
               32,
               0,0,0,0,0,0,0,0,0,0,0,0,0, // useles parameters
               16,
               0,0,0,0,0,0,0
       };

       // Choose the closest pixel format available
       indexPixelFormat = ChoosePixelFormat(hdc, &pfd);

       // Set the pixel format for the provided window DC
       return SetPixelFormat(hdc, indexPixelFormat, &pfd);// number of available formats
}

static void create_and_register_windows_class(void)
{
        WNDCLASSEX ex;

        ex.cbSize = sizeof(WNDCLASSEX);
        ex.style = CS_HREDRAW|CS_VREDRAW|CS_OWNDC;
        ex.lpfnWndProc = DefWindowProc;
        ex.cbClsExtra = 0;
        ex.cbWndExtra = 0;
        ex.hInstance = NULL;
        ex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        ex.hCursor = LoadCursor(NULL, IDC_ARROW);
        ex.hbrBackground = NULL;
        ex.lpszMenuName = NULL;
        ex.lpszClassName = WNDCLASSNAME;
        ex.hIconSm = NULL;

        ATOM res = RegisterClassEx(&ex);
        if (res == 0) {
                PrintError(GetLastError());
                abort();
        }
}

static HWND CreateWnd (int width, int height)
{
       HWND win;
        // center position of the window
        int posx = (GetSystemMetrics(SM_CXSCREEN) / 2) - (width / 2);
        int posy = (GetSystemMetrics(SM_CYSCREEN) / 2) - (height / 2);

        // set up the window for a windowed application by default
        long wndStyle = WS_OVERLAPPEDWINDOW;

        // create the window
        win = CreateWindowEx(0,
                        WNDCLASSNAME,
                        "OpenGL Context Window",
                        wndStyle|WS_CLIPCHILDREN|WS_CLIPSIBLINGS,
                        posx, posy,
                        width, height,
                        NULL,
                        NULL,
                        NULL,
                        NULL);

       if(!win) {
               PrintError(GetLastError());
       }

        // at this point WM_CREATE message is sent/received
        // the WM_CREATE branch inside WinProc function will execute here
       return win;
}

void *win32_context_init(win32_opengl_version_t version)
{
        struct state_win32_gl_context *context;

        context = calloc(1, sizeof(struct state_win32_gl_context));
        if(!context) {
                fprintf(stderr, "Unable to allocate memory.\n");
                return NULL;
        }
        context->version = version;
        context->status = UNKNOWN;

        pthread_once(&register_windows_class_once,
                        create_and_register_windows_class);

        pthread_mutex_init(&context->lock, NULL);
        pthread_cond_init(&context->status_cv, NULL);
        pthread_cond_init(&context->loop_cv, NULL);
        pthread_create(&context->thread_id, NULL, owner_thread, context);

        pthread_mutex_lock(&context->lock);
        while (context->status == UNKNOWN) {
                pthread_cond_wait(&context->status_cv, &context->lock);
        }
        pthread_mutex_unlock(&context->lock);

        if (context->status == FAILED) {
                pthread_join(context->thread_id, NULL);
                win32_context_cleanup_common(context);
                return NULL;
        }

        wglMakeCurrent(context->hdc, context->hglrc);

        return context;
}

void win32_context_make_current(void *arg)
{
        struct state_win32_gl_context *context = arg;
        BOOL ret;

        if(!arg) {
                ret = wglMakeCurrent(NULL, NULL);
        } else {
                ret = wglMakeCurrent(context->hdc, context->hglrc);
        }
        if(ret == FALSE) {
                fprintf(stderr, "Warning: wglMakeCurrent did not succeeded!\n");
                PrintError(GetLastError());
        }
}

static void win32_context_cleanup_common(struct state_win32_gl_context
                *context)
{
        pthread_mutex_destroy(&context->lock);
        pthread_cond_destroy(&context->status_cv);
        pthread_cond_destroy(&context->loop_cv);

        free(context);
}

void win32_context_free(void *arg)
{
        if(!arg)
                return;

        struct state_win32_gl_context *context = arg;

        pthread_mutex_lock(&context->lock);
        context->should_exit = true;
        pthread_cond_signal(&context->loop_cv);
        pthread_mutex_unlock(&context->lock);

        pthread_join(context->thread_id, NULL);
        win32_context_cleanup_common(context);
}


/**
 * Copyright (c) 2011, CESNET z.s.p.o
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "view.h"
#include "image.h"
#include "util.h"

/** Documented at declaration */
struct view
{
    int width;
    int height;
    int window_width;
    int window_height;
    view_callback_on_init_t on_init;
    void* on_init_param;
    view_callback_on_render_t on_render;
    void* on_render_param;
    unsigned int texture_id;
    
    // GLX
    Display* glx_display;
    GLXContext glx_context;
    Window glx_window;
};

/** Documented at declaration */
struct view*
view_create(int width, int height, int window_width, int window_height)
{
    if ( window_width == 0 )
        window_width = width;
    if ( window_width == 0 )
        window_height = height;
        
    struct view* view = (struct view*)malloc(sizeof(struct view));
    if ( view == NULL )
        return NULL;
    view->width = width;
    view->height = height;
    view->window_width = window_width;
    view->window_height = window_height;
    view->on_init = NULL;
    view->on_init_param = NULL;
    view->on_render = NULL;
    view->on_render_param = NULL;
    view->texture_id = 0;
    
    return view;
}

/** Documented at declaration */
void
view_set_on_init(struct view* view, view_callback_on_init_t on_init, void* param)
{
    view->on_init = on_init;
    view->on_init_param = param;
}

/** Documented at declaration */
void
view_set_on_render(struct view* view, view_callback_on_render_t on_render, void* param)
{
    view->on_render = on_render;
    view->on_render_param = param;
}

/** Documented at declaration */
void
view_set_texture(struct view* view, int texture_id)
{
    view->texture_id = texture_id;
}

/** Documented at declaration */
void
view_destroy(struct view* view)
{
    free(view);
}

void
view_init(struct view* view)
{    
    view->texture_id = 0;
    
    glEnable(GL_TEXTURE_2D);

    if ( view->on_init != NULL )
        view->on_init(view->on_init_param);
}

void
view_render(struct view* view)
{
    glViewport(0, 0, view->window_width, view->window_height);
    glClear(GL_COLOR_BUFFER_BIT);
    
    if ( view->on_render != NULL )
        view->on_render(view->on_render_param);
    
    if ( view->texture_id != 0 ) {
        glBindTexture(GL_TEXTURE_2D, view->texture_id);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    glFlush();
    glFinish();
}

/** View for glut */
struct view* g_glut_view = NULL;

void
view_glut_render()
{
    struct view* view = g_glut_view;
    if ( view == NULL )
        return;
    
    TIMER_INIT();
    TIMER_START();
    
    view_render(view);
    
    glutSwapBuffers();
    
    TIMER_STOP_PRINT("View: GlutRender");
    
    usleep(1000);
    
    glutPostRedisplay();
}

void
view_glut_keyboard(unsigned char key, int x, int y)
{
    switch ( key ) {
        case 27:
            g_glut_view = NULL;
            break;
        default:
            printf("Key pressed: %c (%d)\n", key, (int)key);
            break;
    }
}

/** Documented at declaration */
int
view_glut(struct view* view)
{
    int argc = 0;
    glutInit(&argc, NULL);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
    glutCreateWindow("OpenGL and CUDA");
    glutReshapeWindow(view->window_width, view->window_height);
    glutDisplayFunc(view_glut_render);
    glutKeyboardFunc(view_glut_keyboard);
    glutIdleFunc(view_glut_render);

    view_init(view);
    
    g_glut_view = view;
    
    glutPostRedisplay();
    while( g_glut_view != NULL ) {
        glutMainLoopEvent();
    }
    glutHideWindow();
    
    return 0;
}

void
view_glx_render(struct view* view)
{
    TIMER_INIT();
    TIMER_START();
    
    view_render(view);
        
    glXSwapBuffers(view->glx_display, view->glx_window);
    
    TIMER_STOP_PRINT("View: GlxRender");
    
    usleep(1000);
}

int
view_glx_keyboard(int keycode, const char* key)
{
    switch ( keycode ) {
        case 9:
            return 1;
        default:
            printf("Key pressed: %s (%d)\n", key, keycode);
            break;
    }
    return 0;
}

/** Documented at declaration */
int
view_glx(struct view* view)
{    
    // Open display
    view->glx_display = XOpenDisplay(0);
    if ( view->glx_display == NULL ) {
        fprintf(stderr, "Failed to open X display!\n");
        pthread_exit(0);
    }

    // Choose visual
    static int attributes[] = {
        GLX_RGBA,
        GLX_DOUBLEBUFFER,
        GLX_RED_SIZE,   1,
        GLX_GREEN_SIZE, 1,
        GLX_BLUE_SIZE,  1,
        None
    };
    XVisualInfo* visual = glXChooseVisual(view->glx_display, DefaultScreen(view->glx_display), attributes);
    if ( visual == NULL ) {
        fprintf(stderr, "Failed to choose visual!\n");
        pthread_exit(0);
    }

    // Create OpenGL context
    view->glx_context = glXCreateContext(view->glx_display, visual, 0, GL_TRUE);
    if ( view->glx_context == NULL ) {
        fprintf(stderr, "Failed to create OpenGL context!\n");
        pthread_exit(0);
    }

    // Create window
    Colormap colormap = XCreateColormap(view->glx_display, RootWindow(view->glx_display, visual->screen), visual->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = colormap;
    swa.border_pixel = 0;
    swa.event_mask = KeyPressMask;
    view->glx_window = XCreateWindow(
        view->glx_display, 
        RootWindow(view->glx_display, visual->screen), 
        0, 0, view->window_width, view->window_height,
        0, visual->depth, InputOutput, visual->visual,
        CWBorderPixel | CWColormap | CWEventMask, 
        &swa
    );
    XStoreName(view->glx_display, view->glx_window, "OpenGL and CUDA interoperability");
    XMapWindow(view->glx_display, view->glx_window);
    
    view_opengl_attach(view);
    view_init(view);
    
    while ( 1 ) {
        view_glx_render(view);
        
        XEvent event;
        if ( XCheckWindowEvent(view->glx_display, view->glx_window, KeyPressMask, &event)) {
            int keycode = (int)event.xkey.keycode;
            char* key = XKeysymToString(XKeycodeToKeysym(view->glx_display, keycode, 0));
            if ( view_glx_keyboard(keycode, key) != 0 )
                break;
        }
    }
    
    view_opengl_detach(view);
    
    // Cleanup
    glXDestroyContext(view->glx_display, view->glx_context);
    XDestroyWindow(view->glx_display, view->glx_window);
    XCloseDisplay(view->glx_display);

    return 0;
}

/** Documented at declaration */
void
view_opengl_attach(struct view* view)
{
    glXMakeCurrent(view->glx_display, view->glx_window, view->glx_context);
}

/** Documented at declaration */
void
view_opengl_detach(struct view* view)
{
    glXMakeCurrent(view->glx_display, None, NULL);
}

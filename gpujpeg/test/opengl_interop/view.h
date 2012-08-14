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

#ifndef TEST_OPENGL_INTEROP_VIEW_H
#define TEST_OPENGL_INTEROP_VIEW_H

/**
 * View structure
 */
struct view;

/**
 * View on init callback called
 */
typedef void (*view_callback_on_init_t)(void* param);

/**
 * View on render callback called
 */
typedef void (*view_callback_on_render_t)(void* param);

/**
 * Create view
 * 
 * @param width  Data width
 * @param height  Data height
 * @param window_width  Window width
 * @param window_height  Window height
 * @return view structure if succeeds, otherwise NULL
 */
struct view*
view_create(int width, int height, int window_width, int window_height);

/**
 * Set on init callback to view
 * 
 * @param view
 * @param on_init
 * @param param
 */
void
view_set_on_init(struct view* view, view_callback_on_init_t on_init, void* param);

/**
 * Set on render callback to view
 * 
 * @param view
 * @param on_render
 * @param param
 */
void
view_set_on_render(struct view* view, view_callback_on_render_t on_render, void* param);

/**
 * Set texture to show
 * 
 * @param view
 * @param texture_id
 * @return void
 */
void
view_set_texture(struct view* view, int texture_id);

/**
 * Destroy view
 * 
 * @param view
 * @return void
 */
void
view_destroy(struct view* view);

/**
 * Run view by GLX
 * 
 * @param view
 * @return void
 */
int
view_glx(struct view* view);

/**
 * Attach OpenGL context
 * 
 * @param view
 * @return void
 */
void
view_opengl_attach(struct view* view);

/**
 * Detach OpenGL context
 * 
 * @param view
 * @return void
 */
void
view_opengl_detach(struct view* view);

#endif // TEST_OPENGL_INTEROP_VIEW_H

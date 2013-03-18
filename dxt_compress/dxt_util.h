/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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
 
#ifndef DXT_UTIL_H
#define DXT_UTIL_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "dxt_common.h"

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#elif defined WIN32
#include <GL/glew.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#include <string.h>

#ifdef HAVE_GLUT
#ifdef HAVE_MACOSX
#include <GLUT/glut.h>
extern void glutCheckLoop(void);
#else
#include <GL/glut.h>
extern void glutMainLoopEvent(void);
#endif /* HAVE_MACOSX */
#endif /* HAVE_GLUT */

static inline int dxt_get_size(int width, int height, enum dxt_type format)
{
    assert( format == DXT_TYPE_DXT5_YCOCG || format == DXT_TYPE_DXT1 ||  format == DXT_TYPE_DXT1_YUV );

    if ( format == DXT_TYPE_DXT5_YCOCG )
        return ((width + 3) / 4 * 4) * ((height + 3) / 4 * 4);
    else
        return ((width + 3) / 4 * 4) * ((height + 3) / 4 * 4) / 2;
}

/**
 * Create shader from source
 * 
 * @param source  Shade source 
 * @param type  Shader type
 * @return shader handle if succeeds, otherwise zero
 */
GLuint
dxt_shader_create_from_source(const char* source, GLenum type);

/**
 * Create shader from file
 * 
 * @param filename  Shade source filename
 * @param type  Shader type
 * @return shader handle if succeeds, otherwise zero
 */
GLhandleARB
dxt_shader_create_from_file(const char* filename, GLenum type);



#endif // DXT_UTIL_H

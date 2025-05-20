/**
 * @file   gl_context.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
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

#include "gl_context.h"

#include <stdio.h>         // for NULL, fprintf, stderr
#include <stdlib.h>        // for free, atoi
#include <string.h>        // for strtok_r

#ifdef __APPLE__
#include "mac_gl_common.h"
#elif defined __linux__
#include "glx_common.h"
#else // _WIN32
#include "win32_gl_common.h"
#endif

#include "debug.h"
#include "utils/macros.h"

/**
 * @brief initializes specified OpenGL context
 *
 * @note
 * After this call, context is current for calling thread.
 *
 * @param[out] context newly created context
 * @param      which   OpenGL version specifier
 * @return     info if context creation succeeded or failed
 */
bool init_gl_context(struct gl_context *context, int which) {
        context->context = NULL;
#ifdef __linux__
        if(which == GL_CONTEXT_ANY) {
                debug_msg("Trying OpenGL 3.1 first.\n");
                context->context = glx_init(MK_OPENGL_VERSION(3,1));
                context->legacy = false;
        }
        if(!context->context) {
                if(which != GL_CONTEXT_LEGACY) {
                        debug_msg("OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
                }
                context->context = glx_init(OPENGL_VERSION_UNSPECIFIED);
                context->legacy = true;
        }
        if(context->context) {
                glx_validate(context->context);
        }
#elif defined __APPLE__
        if(which == GL_CONTEXT_ANY) {
                if(get_mac_kernel_version_major() >= 11) {
                        debug_msg("Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                        context->context = mac_gl_init(MAC_GL_PROFILE_3_2);
                        if(!context->context) {
                                debug_msg("OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
                        } else {
                                context->legacy = false;
                        }
                }
        }

        if(!context->context) {
                context->context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
                context->legacy = true;
        }
#else // _WIN32
        if(which == GL_CONTEXT_ANY) {
                context->context = win32_context_init(OPENGL_VERSION_UNSPECIFIED);
        } else if(which == GL_CONTEXT_LEGACY) {
                context->context = win32_context_init(OPENGL_VERSION_LEGACY);
        }

        context->legacy = true;
#endif

        {
                char *save_ptr;
                const char *gl_ver = (const char *) glGetString(GL_VERSION);
                if (!gl_ver) {
                        fprintf(stderr, "Unable to determine OpenGL version!\n");
                        return false;
                }

                char *tmp = strdup(gl_ver);
                char *item = strtok_r(tmp, ".", &save_ptr);
                if (!item) {
                        fprintf(stderr, "Unable to determine OpenGL version!\n");
                        free(tmp);
                        return false;
                }
                context->gl_major = atoi(item);
                item = strtok_r(NULL, ".", &save_ptr);
                if (!item) {
                        fprintf(stderr, "Unable to determine OpenGL version!\n");
                        free(tmp);
                        return false;
                }
                context->gl_minor = atoi(item);
                free(tmp);
        }

        if(context->context) {
                return true;
        } else {
                return false;
        }
}

GLuint glsl_compile_link(const char *vprogram, const char *fprogram)
{
        char log[32768];
        GLuint vhandle = 0;
        GLuint fhandle = 0;
        GLint status;

        log_msg(LOG_LEVEL_DEBUG, "Compiling program:\nVertex:%s\nShader:\n%s\n", IF_NOT_NULL_ELSE(vprogram, "(none)"), IF_NOT_NULL_ELSE(fprogram, "(none)"));

        /* compile */
        /* fragmemt */
        if (fprogram) {
                fhandle = glCreateShader(GL_FRAGMENT_SHADER);
                glShaderSource(fhandle, 1, &fprogram, NULL);
                glCompileShader(fhandle);
                glGetShaderiv(fhandle, GL_COMPILE_STATUS, &status);
                /* Print compile log */
                glGetShaderInfoLog(fhandle, sizeof log, NULL, log);
                if (status == GL_FALSE || (strlen(log) > 0 && log_level >= LOG_LEVEL_VERBOSE)) {
                        log_msg(status == GL_FALSE ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "Fragment compile log: %s\n", log);
                }
                if (status != GL_TRUE) {
                        glDeleteShader(fhandle);
                        return 0;
                }
        }
        /* vertex */
        if (vprogram) {
                vhandle = glCreateShader(GL_VERTEX_SHADER);
                glShaderSource(vhandle, 1, &vprogram, NULL);
                glCompileShader(vhandle);
                glGetShaderiv(vhandle, GL_COMPILE_STATUS, &status);
                /* Print compile log */
                glGetShaderInfoLog(vhandle, sizeof log, NULL, log);
                if (status == GL_FALSE || (strlen(log) > 0 && log_level >= LOG_LEVEL_VERBOSE)) {
                        log_msg(status == GL_FALSE ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "Vertex compile log: %s\n", log);
                }
                if (status != GL_TRUE) {
                        glDeleteShader(fhandle);
                        glDeleteShader(vhandle);
                        return 0;
                }
        }

        GLuint phandle = glCreateProgram();

        /* attach and link */
        if (vprogram) {
                glAttachShader(phandle, vhandle);
        }
        if (fprogram) {
                glAttachShader(phandle, fhandle);
        }
        glLinkProgram(phandle);

        // mark shaders for deletion when program is deleted
        glDeleteShader(vhandle);
        glDeleteShader(fhandle);

        glGetProgramiv(phandle, GL_LINK_STATUS, &status);
        glGetProgramInfoLog(phandle, sizeof log, NULL, (GLchar*) log);
        if (status == GL_FALSE || (strlen(log) > 0 && log_level >= LOG_LEVEL_VERBOSE)) {
                log_msg(status == GL_FALSE ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "Link log: %s\n", log);
        }
        if (status != GL_TRUE) {
                glDeleteProgram(phandle);
                return 0;
        }

        // check GL errors
        gl_check_error();

        return phandle;
}


void destroy_gl_context(struct gl_context *context) {
        if (context == NULL || context->context == NULL) {
                return;
        }
#ifdef __APPLE__
        mac_gl_free(context->context);
#elif defined __linux__
        glx_free(context->context);
#else
        win32_context_free(context->context);
#endif
}

void gl_context_make_current(struct gl_context *context)
{
	void *context_state = NULL;
        if(context) {
		context_state = context->context;
	}
#ifdef __linux__
	glx_make_current(context_state);
#elif defined __APPLE__
	mac_gl_make_current(context_state);
#else
        win32_context_make_current(context_state);
#endif
}


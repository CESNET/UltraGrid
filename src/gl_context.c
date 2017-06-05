#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#if defined HAVE_MACOSX || (defined HAVE_LINUX && defined HAVE_LIBGL) || defined WIN32

#include <stdio.h>

#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#elif defined HAVE_LINUX
#include "x11_common.h"
#include "glx_common.h"
#else // WIN32
#include "win32_gl_common.h"
#endif

#include "debug.h"
#include "gl_context.h"

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
#ifdef HAVE_LINUX
        if(which == GL_CONTEXT_ANY) {
                debug_msg("Trying OpenGL 3.1 first.\n");
                context->context = glx_init(MK_OPENGL_VERSION(3,1));
                context->legacy = FALSE;
        }
        if(!context->context) {
                if(which != GL_CONTEXT_LEGACY) {
                        debug_msg("OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
                }
                context->context = glx_init(OPENGL_VERSION_UNSPECIFIED);
                context->legacy = TRUE;
        }
        if(context->context) {
                glx_validate(context->context);
        }
#elif defined HAVE_MACOSX
        if(which == GL_CONTEXT_ANY) {
                if(get_mac_kernel_version_major() >= 11) {
                        debug_msg("Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                        context->context = mac_gl_init(MAC_GL_PROFILE_3_2);
                        if(!context->context) {
                                debug_msg("OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
                        } else {
                                context->legacy = FALSE;
                        }
                }
        }

        if(!context->context) {
                context->context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
                context->legacy = TRUE;
        }
#else // WIN32
        if(which == GL_CONTEXT_ANY) {
                context->context = win32_context_init(OPENGL_VERSION_UNSPECIFIED);
        } else if(which == GL_CONTEXT_LEGACY) {
                context->context = win32_context_init(OPENGL_VERSION_LEGACY);
        }

        context->legacy = TRUE;
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
        GLuint vhandle, fhandle;
        GLuint phandle;

        phandle = glCreateProgram();
        vhandle = glCreateShader(GL_VERTEX_SHADER);
        fhandle = glCreateShader(GL_FRAGMENT_SHADER);

        /* compile */
        /* fragmemt */
        if (fprogram) {
                glShaderSource(fhandle, 1, &fprogram, NULL);
                glCompileShader(fhandle);
                /* Print compile log */
                glGetShaderInfoLog(fhandle, sizeof log, NULL, log);
                if (strlen(log) > 0) {
                        log_msg(LOG_LEVEL_INFO, "Fragment compile log: %s\n", log);
                }
        }
        /* vertex */
        if (vprogram) {
                glShaderSource(vhandle, 1, &vprogram, NULL);
                glCompileShader(vhandle);
                /* Print compile log */
                glGetShaderInfoLog(vhandle, sizeof log, NULL, log);
                if (strlen(log) > 0) {
                        log_msg(LOG_LEVEL_INFO, "Vertex compile log: %s\n", log);
                }
        }

        /* attach and link */
        if (vprogram) {
                glAttachShader(phandle, vhandle);
        }
        if (fprogram) {
                glAttachShader(phandle, fhandle);
        }
        glLinkProgram(phandle);
        glGetProgramInfoLog(phandle, sizeof log, NULL, (GLchar*) log);
        if (strlen(log) > 0) {
                log_msg(LOG_LEVEL_INFO, "Link Log: %s\n", log);
        }

        // check GL errors
        gl_check_error();

        // mark shaders for deletion when program is deleted
        glDeleteShader(vhandle);
        glDeleteShader(fhandle);

        return phandle;
}


void destroy_gl_context(struct gl_context *context) {
#ifdef HAVE_MACOSX
        mac_gl_free(context->context);
#elif defined HAVE_LINUX
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
#ifdef HAVE_LINUX
	glx_make_current(context_state);
#elif defined HAVE_MACOSX
	mac_gl_make_current(context_state);
#else
        win32_context_make_current(context_state);
#endif
}

#endif /* defined HAVE_MACOSX || (defined HAVE_LINUX && defined HAVE_LIBGLEW) */

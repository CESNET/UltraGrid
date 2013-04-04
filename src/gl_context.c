#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#if defined HAVE_MACOSX || (defined HAVE_LINUX && defined HAVE_LIBGL) || defined WIN32

#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#elif defined HAVE_LINUX
#include "x11_common.h"
#include "glx_common.h"
#else // WIN32
#include "win32_gl_common.h"
#endif

#include "gl_context.h"

bool init_gl_context(struct gl_context *context, int which) {
        context->context = NULL;
#ifdef HAVE_LINUX
        x11_enter_thread();
        if(which == GL_CONTEXT_ANY) {
                printf("Trying OpenGL 3.1 first.\n");
                context->context = glx_init(MK_OPENGL_VERSION(3,1));
                context->legacy = FALSE;
        }
        if(!context->context) {
                if(which != GL_CONTEXT_LEGACY) {
                        fprintf(stderr, "[RTDXT] OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
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
                        printf("[RTDXT] Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                        context->context = mac_gl_init(MAC_GL_PROFILE_3_2);
                        if(!context->context) {
                                fprintf(stderr, "[RTDXT] OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
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
        glShaderSource(fhandle, 1, &fprogram, NULL);
        glCompileShader(fhandle);
        /* Print compile log */
        glGetShaderInfoLog(fhandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);
        /* vertex */
        glShaderSource(vhandle, 1, &vprogram, NULL);
        glCompileShader(vhandle);
        /* Print compile log */
        glGetShaderInfoLog(vhandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        /* attach and link */
        glAttachShader(phandle, vhandle);
        glAttachShader(phandle, fhandle);
        glLinkProgram(phandle);

        printf("Program compilation/link status: ");
        gl_check_error();

        glGetProgramInfoLog(phandle, 32768, NULL, (GLchar*)log);
        if ( strlen(log) > 0 )
                printf("Link Log: %s\n", log);

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

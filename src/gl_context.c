#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#if defined HAVE_MACOSX || (defined HAVE_LINUX && defined HAVE_LIBGL)

#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#else
#include "x11_common.h"
#include "glx_common.h"
#endif

#include "gl_context.h"

bool init_gl_context(struct gl_context *context, int which) {
#ifndef HAVE_MACOSX
        x11_enter_thread();
        context->context = NULL;
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
#else
        context->context = NULL;
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
#endif

        if(context->context) {
                return true;
        } else {
                return false;
        }
}


void destroy_gl_context(struct gl_context *context) {
#ifdef HAVE_MACOSX
        mac_gl_free(context->context);
#else
        glx_free(context->context);
#endif
}

void gl_context_make_current(struct gl_context *context)
{
#ifdef HAVE_MACOSX
        // TODO
#else
        if(context) {
                glx_make_current(context->context);
        } else {
                glx_make_current(NULL);
        }
#endif
}

#endif /* defined HAVE_MACOSX || (defined HAVE_LINUX && defined HAVE_LIBGLEW) */

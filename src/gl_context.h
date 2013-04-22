#ifndef _GL_CONTEXT_H_
#define _GL_CONTEXT_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#endif // HAVE_CONFIG_H

#if defined HAVE_LINUX || defined WIN32
#include <GL/glew.h>
#else
#include <OpenGL/GL.h>
#endif /*  HAVE_LINUX */

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined HAVE_MACOSX && OS_VERSION_MAJOR < 11
#define glGenFramebuffers glGenFramebuffersEXT
#define glBindFramebuffer glBindFramebufferEXT
#define GL_FRAMEBUFFER GL_FRAMEBUFFER_EXT
#define glFramebufferTexture2D glFramebufferTexture2DEXT
#define glDeleteFramebuffers glDeleteFramebuffersEXT
#define GL_FRAMEBUFFER_COMPLETE GL_FRAMEBUFFER_COMPLETE_EXT
#define glCheckFramebufferStatus glCheckFramebufferStatusEXT
#endif

struct gl_context {
        int legacy:1;
        void *context;
};

#define GL_CONTEXT_ANY 0x00
#define GL_CONTEXT_LEGACY 0x01

/**
 * @param which GL_CONTEXT_ANY or GL_CONTEXT_LEGACY
 */
bool init_gl_context(struct gl_context *context, int which);
void gl_context_make_current(struct gl_context *context);
void destroy_gl_context(struct gl_context *context);
static void gl_check_error() __attribute__((unused));

static void gl_check_error()
{
        GLenum msg;
        msg=glGetError();
        int flag = 1;
        switch(msg){
                case GL_NO_ERROR:
                        flag = 0;
                        //printf("No error\n");
                        break;
                case GL_INVALID_ENUM:
                        fprintf(stderr, "GL_INVALID_ENUM\n");
                        break;
                case GL_INVALID_VALUE:
                        fprintf(stderr, "GL_INVALID_VALUE\n");
                        break;
                case GL_INVALID_OPERATION:
                        fprintf(stderr, "GL_INVALID_OPERATION\n");
                        break;
                case GL_STACK_OVERFLOW:
                        fprintf(stderr, "GL_STACK_OVERFLOW\n");
                        break;
                case GL_STACK_UNDERFLOW:
                        fprintf(stderr, "GL_STACK_UNDERFLOW\n");
                        break;
                case GL_OUT_OF_MEMORY:
                        fprintf(stderr, "GL_OUT_OF_MEMORY\n");
                        break;
                case 1286:
                        fprintf(stderr, "INVALID_FRAMEBUFFER_OPERATION_EXT\n");
                        break;
                default:
                        fprintf(stderr, "wft mate? Unknown GL ERROR: %d\n", msg);
                        break;
        }
        if(flag)
                abort();
}

GLuint glsl_compile_link(const char *vprogram, const char *fprogram);


#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* _GL_CONTEXT_H_ */


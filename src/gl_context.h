#ifndef _GL_CONTEXT_H_
#define _GL_CONTEXT_H_

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

#endif /* _GL_CONTEXT_H_ */


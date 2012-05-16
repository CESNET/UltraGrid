#ifndef _GL_CONTEXT_H_
#define _GL_CONTEXT_H_
struct gl_context {
        int legacy:1;
        void *context;
};

void init_gl_context(struct gl_context *context);
void gl_context_make_current(struct gl_context *context);
void destroy_gl_context(struct gl_context *context);

#endif /* _GL_CONTEXT_H_ */


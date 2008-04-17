#define DISPLAY_GL_ID  0xba370a2a

display_type_t          *display_gl_probe(void);
void                    *display_gl_init(void);
void                     display_gl_done(void *state);
char                    *display_gl_getf(void *state);
int                      display_gl_putf(void *state, char *frame);
display_colour_t         display_gl_colour(void *state);

int                      display_gl_handle_events(void *arg);

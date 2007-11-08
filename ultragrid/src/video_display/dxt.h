#define DISPLAY_DXT_ID  0xba370d2a

display_type_t          *display_dxt_probe(void);
void                    *display_dxt_init(void);
void                     display_dxt_done(void *state);
unsigned char                    *display_dxt_getf(void *state);
int                      display_dxt_putf(void *state, unsigned char *frame);
display_colour_t         display_dxt_colour(void *state);

int                      display_dxt_handle_events(void *arg);

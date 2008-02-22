#define DISPLAY_KONA_ID    0xba370f2f

display_type_t          *display_kona_probe(void);
void                    *display_kona_init(void);
void                     display_kona_done(void *state);
char                    *display_kona_getf(void *state);
int                      display_kona_putf(void *state, char *frame);
display_colour_t         display_kona_colour(void *state);


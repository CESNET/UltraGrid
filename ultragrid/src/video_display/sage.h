
#define DISPLAY_SAGE_ID	0xba370a2f

display_type_t		*display_sage_probe(void);
void 			*display_sage_init(void);
void 			 display_sage_done(void *state);
char 			*display_sage_getf(void *state);
int  			 display_sage_putf(void *state, char *frame);
display_colour_t	 display_sage_colour(void *state);

int			 display_sage_handle_events(void);


#define VIDCAP_QUAD_ID	0x10203044

struct vidcap_type	*vidcap_quad_probe(void);
void			*vidcap_quad_init(int fps);
void			 vidcap_quad_done(void *state);
struct video_frame	*vidcap_quad_grab(void *state);

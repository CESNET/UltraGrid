#define VIDCAP_DECKLINK_ID	0x10203041 // TOCHANGE borek //

#ifdef __cplusplus
extern "C" {
#endif

struct vidcap_type	*vidcap_decklink_probe(void);
void			*vidcap_decklink_init(struct vidcap_fmt *fmt);
void			 vidcap_decklink_done(void *state);
struct video_frame	*vidcap_decklink_grab(void *state);

#ifdef __cplusplus
} // END extern "C"
#endif

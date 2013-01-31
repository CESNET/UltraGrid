// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the DIRECTSHOWGRABBER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// DIRECTSHOWGRABBER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef DIRECTSHOWGRABBER_EXPORTS
#define DIRECTSHOWGRABBER_API __declspec(dllexport)
#else
#define DIRECTSHOWGRABBER_API __declspec(dllimport)
#endif

#undef DIRECTSHOWGRABBER_API
#define DIRECTSHOWGRABBER_API

#define VIDCAP_DSHOW_ID 0xD5CAD5CA

#ifdef __cplusplus
extern "C" {
#endif
	DIRECTSHOWGRABBER_API struct vidcap_type * vidcap_dshow_probe(void);
	DIRECTSHOWGRABBER_API void * vidcap_dshow_init(char *init_fmt, unsigned int flags);
	DIRECTSHOWGRABBER_API void vidcap_dshow_finish(void *state);
	DIRECTSHOWGRABBER_API void vidcap_dshow_done(void *state);
	DIRECTSHOWGRABBER_API struct video_frame * vidcap_dshow_grab(void *state, struct audio_frame **audio);
#ifdef __cplusplus
}
#endif

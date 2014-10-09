#ifndef _RTSP_TYPES_HH
#define _RTSP_TYPES_HH

typedef enum {
    none,
    av,
    video,
    audio,
    NUM_RTSP_FORMATS
}rtps_types_t;

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void rtps_server_usage(void);
EXTERNC int get_rtsp_server_port(char *config);

#endif

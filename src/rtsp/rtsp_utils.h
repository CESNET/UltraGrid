#ifndef _RTSP_TYPES_HH
#define _RTSP_TYPES_HH

typedef enum {
        rtsp_type_none    = 0,
        rtsp_type_audio   = 1 << 0,
        rtsp_type_video   = 1 << 1,
        rtsp_av_type_both = rtsp_type_audio | rtsp_type_video,
} rtsp_types_t;

#endif

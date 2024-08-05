#ifndef _RTSP_TYPES_HH
#define _RTSP_TYPES_HH

typedef enum {
        none              = 0,
        audio             = 1 << 0,
        video             = 1 << 1,
        rtsp_av_type_both = audio | video,
} rtsp_types_t;

#endif

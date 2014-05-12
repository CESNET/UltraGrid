#ifndef _RTSP_TYPES_HH
#define _RTSP_TYPES_HH

typedef enum {
    none,
    avStd,
    avStdDyn,
    avUG,
    videoH264,
    videoUG,
    audioPCMUstd,
    audioPCMUdyn,
    NUM_RTSP_FORMATS
}rtps_types_t;


#endif

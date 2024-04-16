#ifndef HWACCEL_DRM_H_d8e44cdd67bf
#define HWACCEL_DRM_H_d8e44cdd67bf

#include <stdint.h>
#include <stdint.h>

struct AVFrame;

struct drm_prime_frame{
        uint32_t drm_format;

        int fd_count;
        int dmabuf_fds[4];

        int planes;
        uint32_t fd_indices[4]; //index into dmabuf_fds
        uint32_t pitches[4];
        uint32_t offsets[4];
        uint64_t modifiers[4];

        struct AVFrame *av_frame;
};

#endif

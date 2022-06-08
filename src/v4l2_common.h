/**
 * @file    v4l2_common.h
 * @author  Milos Liska      <xliska@fi.muni.cz>
 * @author  Martin Piatka    <piatka@cesnet.cz>
 * @author  Martin Pulec     <martin.pulec@cesnet.cz>
 */
 /*
 * Copyright (c) 2012-2022 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef V4L2_COMMON_4B01337E_4F0D_48EC_91A7_D220AD919D3B
#define V4L2_COMMON_4B01337E_4F0D_48EC_91A7_D220AD919D3B

#include <linux/videodev2.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "types.h"

#define V4L2_PROBE_MAX 64

static struct {
        enum v4l2_field v4l_f;
        enum interlacing_t ug_f;
} v4l2_field_map [] = {
        { V4L2_FIELD_NONE, PROGRESSIVE },
        { V4L2_FIELD_SEQ_TB, UPPER_FIELD_FIRST },
        { V4L2_FIELD_SEQ_BT, LOWER_FIELD_FIRST },
        { V4L2_FIELD_INTERLACED, INTERLACED_MERGED },
};

/**
 * In theory, direct mapping (get_codec_from_fcc/get_fourcc) of FourCC obtained by VIDIOC_ENUM_FMT,
 * could be used. But UG uses different FCC for some codecs than V4l2 (RGB, I420).
 */
static struct {
        uint32_t v4l2_fcc;
        codec_t ug_codec;
} v4l2_ug_map[] = {
        {V4L2_PIX_FMT_YUYV, YUYV},
        {V4L2_PIX_FMT_UYVY, UYVY},
        {V4L2_PIX_FMT_YUV420, I420},
        {V4L2_PIX_FMT_RGB24, RGB},
        {V4L2_PIX_FMT_RGB32, RGBA},
#ifdef V4L2_PIX_FMT_RGBX32
        {V4L2_PIX_FMT_RGBX32, RGBA},
#endif
        {V4L2_PIX_FMT_MJPEG, MJPG},
        {V4L2_PIX_FMT_JPEG, MJPG},
        {V4L2_PIX_FMT_H264, H264},
        //{V4L2_PIX_FMT_H264_NO_SC, H264}, ///< H.264 without tart codes, @todo implement adding start codes (capture)
#ifdef V4L2_PIX_FMT_HEVC
        {V4L2_PIX_FMT_HEVC, H265},
#endif
#ifdef V4L2_PIX_FMT_HEVC
        {V4L2_PIX_FMT_VP9, VP9},
#endif
};

struct v4l2_buffer_data {
        void *start;
        size_t length;
};

static _Bool set_v4l2_buffers(int fd, struct v4l2_requestbuffers *reqbuf, struct v4l2_buffer_data *buffers) {
        if (ioctl (fd, VIDIOC_REQBUFS, reqbuf) != 0) {
                if (errno == EINVAL)
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Video capturing or mmap-streaming is not supported\n");
                else
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "VIDIOC_REQBUFS");
                return 0;

        }

        if (reqbuf->count < 2) {
                /* You may need to free the buffers here. */
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Not enough buffer memory\n");
                return 0;
        }

        for (unsigned int i = 0; i < reqbuf->count; i++) {
                struct v4l2_buffer buf;
                memset(&buf, 0, sizeof(buf));
                buf.type = reqbuf->type;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;

                if (-1 == ioctl (fd, VIDIOC_QUERYBUF, &buf)) {
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "VIDIOC_QUERYBUF");
                        return 0;
                }

                buffers[i].length = buf.length; /* remember for munmap() */

                buffers[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE, /* recommended */
                                MAP_SHARED,             /* recommended */
                                fd, buf.m.offset);

                if (MAP_FAILED == buffers[i].start) {
                        /* If you do not exit here you should unmap() and free()
                           the buffers mapped so far. */
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "mmap");
                        return 0;
                }

                buf.flags = 0;

                if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to enqueue buffer");
                        return 0;
                }
        }
        return 1;
}

static int try_open_v4l2_device(int log_level, const char *dev_name, int cap) {
        int fd = open(dev_name, O_RDWR);
        if (fd == -1) {
                char errbuf[1024];
                log_msg(log_level, MOD_NAME "Unable to open input device %s: %s\n",
                                dev_name, strerror_r(errno, errbuf, sizeof errbuf));
                return -1;
        }

        struct v4l2_capability capability;
        memset(&capability, 0, sizeof(capability));
        if (ioctl(fd, VIDIOC_QUERYCAP, &capability) != 0) {
                log_perror(log_level, MOD_NAME "ioctl VIDIOC_QUERYCAP");
                close(fd);
                return -1;
        }

        if (!(capability.device_caps & cap)) {
                const char *cap_str = cap == V4L2_CAP_VIDEO_CAPTURE ? "capture" : "playback";
                log_msg(log_level, MOD_NAME "%s, %s can't %s\n",capability.card,capability.bus_info, cap_str);
                close(fd);
                return -1;
        }

        if (!(capability.device_caps & V4L2_CAP_STREAMING)) {
                log_msg(log_level, MOD_NAME "Streaming capability not present.\n");
                close(fd);
                return -1;
        }

        int index = 0;

        if (ioctl(fd, VIDIOC_S_INPUT, &index) != 0) {
                log_perror(log_level, MOD_NAME "Could not enable input (VIDIOC_S_INPUT)");
                close(fd);
                return -1;
        }

        return fd;
}

#endif // defined V4L2_COMMON_4B01337E_4F0D_48EC_91A7_D220AD919D3B

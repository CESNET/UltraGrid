/**
 * @file    video_capture/v4l2.c
 * @author  Milos Liska      <xliska@fi.muni.cz>
 * @author  Martin Piatka    <piatka@cesnet.cz>
 * @author  Martin Pulec     <martin.pulec@cesnet.cz>
 */
 /*
 * Copyright (c) 2012-2019 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "video_capture.h"

#include <arpa/inet.h> // ntohl
#include <libv4l2.h>
#include <libv4lconvert.h>
#include <linux/videodev2.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/list.h"
#include "video.h"

#ifndef _GNU_SOURCE
#error "GNU variant of strerror_r is used below!"
#endif

/* prototypes of functions defined in this module */
static void show_help(void);
static void print_fps(int fd, struct v4l2_frmivalenum *param);

#define DEFAULT_DEVICE "/dev/video0"

#define DEFAULT_BUF_COUNT 2
#define MAX_BUF_COUNT 30
#define MOD_NAME "[V4L cap.] "

struct vidcap_v4l2_state {
        struct video_desc desc;

        int fd;
        struct {
                void *start;
                size_t length;
        } buffers[MAX_BUF_COUNT];

        bool conversion_needed;
        struct v4lconvert_data *convert;
        struct v4l2_format src_fmt, dst_fmt;

        struct timeval t0;
        int frames;

        int buffer_count;

        struct simple_linked_list *buffers_to_enqueue;
        int dequeued_buffers;
        pthread_mutex_t lock;
        pthread_cond_t cv;
};

struct v4l2_dispose_deq_buffer_data {
        struct vidcap_v4l2_state *s;
        struct v4l2_buffer buf;
};

static struct {
        uint32_t v4l2_fcc;
        codec_t ug_codec;
} v4l2_ug_map[] = {
        {V4L2_PIX_FMT_YUYV, YUYV},
        {V4L2_PIX_FMT_UYVY, UYVY},
        {V4L2_PIX_FMT_RGB24, RGB},
        {V4L2_PIX_FMT_RGB32, RGBA},
        {V4L2_PIX_FMT_MJPEG, MJPG},
        {V4L2_PIX_FMT_H264, H264},
};

static void enqueue_all_finished_frames(struct vidcap_v4l2_state *s) {
        struct v4l2_dispose_deq_buffer_data *dequeue_data;
        while ((dequeue_data = simple_linked_list_pop(s->buffers_to_enqueue)) != NULL) {
                s->dequeued_buffers -= 1;
                if (ioctl(s->fd, VIDIOC_QBUF, &dequeue_data->buf) != 0) {
                        log_perror(LOG_LEVEL_ERROR, "Unable to enqueue buffer");
                }
                free(dequeue_data);
        }
}

static void vidcap_v4l2_common_cleanup(struct vidcap_v4l2_state *s) {
        if (!s) {
                return;
        }

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if(ioctl(s->fd, VIDIOC_STREAMOFF, &type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Stream stopping error");
        };

        pthread_mutex_lock(&s->lock);
        enqueue_all_finished_frames(s);
        while (s->dequeued_buffers != 0) {
                pthread_cond_wait(&s->cv, &s->lock);
                enqueue_all_finished_frames(s);
        }
        pthread_mutex_unlock(&s->lock);

        pthread_cond_destroy(&s->cv);
        pthread_mutex_destroy(&s->lock);
        simple_linked_list_destroy(s->buffers_to_enqueue);

        if (s->fd != -1)
                close(s->fd);

        if (s->convert) {
                v4lconvert_destroy(s->convert);
        }

        free(s);
}

static void print_fps(int fd, struct v4l2_frmivalenum *param) {
        int res = ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, param);

        if(res == -1) {
                log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to get FPS");
                return;
        }

        switch (param->type) {
                case V4L2_FRMIVAL_TYPE_DISCRETE:
                        while(ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, param) == 0) {
                                printf("%u/%u ", param->discrete.numerator,
                                                param->discrete.denominator);
                                param->index++;
                        }
                        break;
                case V4L2_FRMIVAL_TYPE_CONTINUOUS:
                        printf("(any FPS)");
                        break;
                case V4L2_FRMIVAL_TYPE_STEPWISE:
                        printf("%u/%u - %u/%u with step %u/%u",
                                        param->stepwise.min.numerator,
                                        param->stepwise.min.denominator,
                                        param->stepwise.max.numerator,
                                        param->stepwise.max.denominator,
                                        param->stepwise.step.numerator,
                                        param->stepwise.step.denominator);
                        break;
        }
}

static void write_fcc(char *out, int pixelformat){
        out[4] = '\0';

        for(int i = 0; i < 4; i++){
                out[i] = pixelformat & 0xff;
                pixelformat >>= 8;
        }
}

static void show_help()
{
        printf("V4L2 capture\n");
        printf("Usage\n");
        printf("\t-t v4l2[:device=<dev>][:codec=<pixel_fmt>][:size=<width>x<height>][:tpf=<tpf>|:fps=<fps>][:buffers=<bufcnt>][:convert=<conv>]\n");
        printf("\t\tuse device <dev> for grab (default: %s)\n", DEFAULT_DEVICE);
        printf("\t\t<tpf> - time per frame in format <numerator>/<denominator>\n");
        printf("\t\t<bufcnt> - number of capture buffers to be used (default: %d)\n", DEFAULT_BUF_COUNT);
        printf("\t\t<tpf> or <fps> should be given as a single integer or a fraction\n");
        printf("\t\t<conv> - forces conversion, eg. to RGB (may be useful eg. to convert captured MJPG from USB 2.0 webcam to HEVC)\n");
        printf("\t\t         mapping available for codecs:");
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                printf(" %s", get_codec_name(v4l2_ug_map[i].ug_codec));
        }
        printf("\n\n");

        for (int i = 0; i < 64; ++i) {
                char name[32];

                snprintf(name, 32, "/dev/video%d", i);
                int fd = open(name, O_RDWR);
                if (fd == -1) {
                        if (errno != ENOENT) {
                                char errbuf[1024];
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to open input device %s: %s\n",
                                                name, strerror_r(errno, errbuf, sizeof errbuf));
                        }
                        continue;
                }

                struct v4l2_capability capab;
                memset(&capab, 0, sizeof capab);
                if (ioctl(fd, VIDIOC_QUERYCAP, &capab) != 0) {
                        log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to query device capabilities");
                }

                log_msg(LOG_LEVEL_VERBOSE, "Device %s capabilities: %#x (CAP_VIDEO_CAPTURE = %#x)\n", name, capab.device_caps, V4L2_CAP_VIDEO_CAPTURE);
                if (!(capab.device_caps & V4L2_CAP_VIDEO_CAPTURE)){
                        goto next_device;
                }

                printf("\t%sDevice %s (%s, %s):\n",
                                (i == 0 ? "(*) " : "    "),
                                name, capab.card, capab.bus_info);


                struct v4l2_fmtdesc format;
                memset(&format, 0, sizeof(format));
                format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                format.index = 0;

                struct v4l2_format fmt;
                memset(&fmt, 0, sizeof(fmt));
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (ioctl(fd, VIDIOC_G_FMT, &fmt) != 0) {
                        log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to get video format");
                        goto next_device;
                }

                while(ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0) {
                        printf("\t\t");
                        if(fmt.fmt.pix.pixelformat == format.pixelformat) {
                                printf("(*) ");
                        } else {
                                printf("    ");
                        }
                        char codec[5];
                        write_fcc(codec, format.pixelformat);
                        printf("Pixel format %4s (%s). Available frame sizes:\n",
                                        codec, format.description);

                        struct v4l2_frmsizeenum size;
                        memset(&size, 0, sizeof(size));
                        size.pixel_format = format.pixelformat;

                        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == -1) {
                                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get frame size iterator");
                                goto next_device;
                        }

                        struct v4l2_frmivalenum frame_int;
                        memset(&frame_int, 0, sizeof(frame_int));
                        frame_int.index = 0;
                        frame_int.pixel_format = format.pixelformat;

                        switch (size.type) {
                                case V4L2_FRMSIZE_TYPE_DISCRETE:
                                        while(ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0) {
                                                printf("\t\t\t");
                                                if(fmt.fmt.pix.width == size.discrete.width &&
                                                                fmt.fmt.pix.height == size.discrete.height) {
                                                        printf("(*) ");
                                                } else {
                                                        printf("    ");
                                                }
                                                printf("%ux%u\t",
                                                                size.discrete.width, size.discrete.height);
                                                frame_int.width = size.discrete.width;
                                                frame_int.height = size.discrete.height;
                                                frame_int.index = 0;
                                                print_fps(fd, &frame_int);
                                                printf("\n");
                                                size.index++;
                                        }
                                        break;
                                case V4L2_FRMSIZE_TYPE_CONTINUOUS:
                                        printf("\t\t\t%u-%ux%u-%u with steps %u vertically and %u horizontally\n",
                                                        size.stepwise.min_width, size.stepwise.max_width,
                                                        size.stepwise.min_height, size.stepwise.max_height,
                                                        size.stepwise.step_width, size.stepwise.step_height);
                                        break;
                                case V4L2_FRMSIZE_TYPE_STEPWISE:
                                        printf("\t\t\tany\n");
                                        break;
                        }

                        format.index++;
                }

                printf("\n");

next_device:
                close(fd);
        }
}

static void write_mode(struct mode *m,
                int width, int height,
                unsigned tpf_num, unsigned tpf_denom,
                int pixelformat)
{
        double fps = (double) tpf_denom / tpf_num;
        char codec[5];
        write_fcc(codec, pixelformat);
        snprintf(m->name, sizeof(m->name), "%dx%d %.2f fps %4s",
                        width, height, fps, codec);

        snprintf(m->id, sizeof(m->id), "{"
                        "\"codec\":\"%.4s\", "
                        "\"size\":\"%dx%d\", "
                        "\"tpf\":\"%u/%u\" }",
                        codec,
                        width, height,
                        tpf_num, tpf_denom);
}

static struct vidcap_type * vidcap_v4l2_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type*		vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name        = "v4l2";
        vt->description = "V4L2 capture";

        vt->card_count = 0;
        vt->cards = 0;

        if (!verbose) {
                return vt;
        }

        for (int i = 0; i < 64; ++i) {
                char name[32];

                snprintf(name, 32, "/dev/video%d", i);
                int fd = open(name, O_RDWR);
                if(fd == -1) continue;

                struct v4l2_capability capab;
                memset(&capab, 0, sizeof capab);
                if (ioctl(fd, VIDIOC_QUERYCAP, &capab) != 0) {
                        log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to query device capabilities");
                }

                if (!(capab.device_caps & V4L2_CAP_VIDEO_CAPTURE)){
                        goto next_device;
                }

                vt->card_count += 1;
                vt->cards = realloc(vt->cards, vt->card_count * sizeof(struct device_info));
                memset(&vt->cards[vt->card_count - 1], 0, sizeof(struct device_info));
                snprintf(vt->cards[vt->card_count - 1].dev, sizeof vt->cards[vt->card_count - 1].dev, ":device=%s", name);
                snprintf(vt->cards[vt->card_count - 1].name, sizeof vt->cards[vt->card_count - 1].name, "V4L2 %s", capab.card);

                struct v4l2_fmtdesc format;
                memset(&format, 0, sizeof(format));
                format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                format.index = 0;

                int fmt_idx = 0;
                while(ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0) {
                        struct v4l2_frmsizeenum size;
                        memset(&size, 0, sizeof(size));
                        size.pixel_format = format.pixelformat;

                        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == -1) {
                                log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to get frame size iterator");
                                goto next_device;
                        }

                        struct v4l2_frmivalenum frame_int;
                        memset(&frame_int, 0, sizeof(frame_int));
                        frame_int.index = 0;
                        frame_int.pixel_format = format.pixelformat;

                        switch (size.type) {
                                case V4L2_FRMSIZE_TYPE_DISCRETE:
                                        while(ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0) {
                                                frame_int.width = size.discrete.width;
                                                frame_int.height = size.discrete.height;
                                                frame_int.index = 0;

                                                if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frame_int) == -1) {
                                                        log_perror(LOG_LEVEL_WARNING, MOD_NAME "Unable to get FPS");
                                                        goto next_device;
                                                }

                                                switch (frame_int.type) {
                                                        case V4L2_FRMIVAL_TYPE_DISCRETE:
                                                                while(ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frame_int) == 0) {
                                                                        write_mode(&vt->cards[vt->card_count - 1].modes[fmt_idx++],
                                                                                        size.discrete.width, size.discrete.height,
                                                                                        frame_int.discrete.numerator, frame_int.discrete.denominator,
                                                                                        format.pixelformat);
                                                                        frame_int.index++;
                                                                }
                                                                break;
                                                        default:
                                                                break;
                                                }
                                                size.index++;
                                        }
                                        break;
                                default:
                                        break;
                        }

                        format.index++;
                }
next_device:
                close(fd);
        }

        return vt;
}

static uint32_t get_ug_to_v4l2(codec_t ug_codec) {
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                if (v4l2_ug_map[i].ug_codec == ug_codec) {
                        return v4l2_ug_map[i].v4l2_fcc;
                }
        }
        return 0;
}

static codec_t get_v4l2_to_ug(uint32_t fcc) {
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                if (v4l2_ug_map[i].v4l2_fcc == fcc) {
                        return v4l2_ug_map[i].ug_codec;
                }
        }
        return VIDEO_CODEC_NONE;
}

static int vidcap_v4l2_init(struct vidcap_params *params, void **state)
{
        struct vidcap_v4l2_state *s;
        const char *dev_name = DEFAULT_DEVICE;
        uint32_t pixelformat = 0;
        uint32_t width = 0,
                 height = 0;
        uint32_t numerator = 0,
                 denominator = 0;
        codec_t v4l2_convert_to = VIDEO_CODEC_NONE;

        printf("vidcap_v4l2_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        if(vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "help") == 0) {
               show_help(); 
               return VIDCAP_INIT_NOERR;
        }


        s = (struct vidcap_v4l2_state *) calloc(1, sizeof(struct vidcap_v4l2_state));
        if(s == NULL) {
                printf("Unable to allocate v4l2 capture state\n");
                return VIDCAP_INIT_FAIL;
        }
        s->buffer_count = DEFAULT_BUF_COUNT;
        s->fd = -1;
        s->buffers_to_enqueue = simple_linked_list_init();
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);

        char *tmp = NULL;

        if(vidcap_params_get_fmt(params)) {
                tmp = strdup(vidcap_params_get_fmt(params));
                assert(tmp != NULL);
                char *init_fmt = tmp;
                char *save_ptr = NULL;
                char *item;
                while((item = strtok_r(init_fmt, ":", &save_ptr))) {
                        if (strncmp(item, "dev=", strlen("dev=")) == 0
                                        || strncmp(item, "device=", strlen("device=")) == 0) {
                                dev_name = strchr(item, '=') + 1;
                        } else if (strncmp(item, "fmt=", strlen("fmt=")) == 0
                         || strncmp(item, "codec=", strlen("codec=")) == 0) {
                                                char *fmt = strchr(item, '=') + 1;
                                                union {
                                                        uint32_t fourcc;
                                                        char str[4];
                                                } str_to_uint;
                                                int len = 4;
                                                if(strlen(fmt) < 4) len = strlen(fmt);
                                                memset(str_to_uint.str, 0, 4);
                                                memcpy(str_to_uint.str, fmt, len);
                                                pixelformat = str_to_uint.fourcc;
                        } else if (strncmp(item, "size=",
                                        strlen("size=")) == 0) {
                                if(strchr(item, 'x')) {
                                        width = atoi(item + strlen("size="));
                                        height = atoi(strchr(item, 'x') + 1);
                                }
                        } else if (strncmp(item, "tpf=", strlen("tpf=")) == 0) {
                                numerator = atoi(item + strlen("tpf="));
                                if (strchr(item, '/')) {
                                        denominator = atoi(strchr(item, '/') + 1);
                                } else {
                                        denominator = 1;
                                }
                        } else if (strncmp(item, "fps=", strlen("fps=")) == 0) {
                                denominator = atoi(item + strlen("fps="));
                                if(strchr(item, '/')) {
                                        numerator = atoi(strchr(item, '/') + 1);
                                } else {
                                        numerator = 1;
                                }
                        } else if (strncmp(item, "buffers=",
                                        strlen("buffers=")) == 0) {
                                s->buffer_count = atoi(item + strlen("buffers="));
                                assert (s->buffer_count <= MAX_BUF_COUNT);
                        } else if (strcasecmp(item, "RGB") == 0) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Deprecated, use: ':convert=RGB' instead");
                                v4l2_convert_to = RGB;
                        } else if (strstr(item, "convert=") == item) {
                                const char *codec = item + strlen("convert=");
                                v4l2_convert_to = get_codec_from_name(codec);
                                if (v4l2_convert_to == VIDEO_CODEC_NONE) {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown codec: %s\n", codec);
                                        goto error;
                                }
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Invalid configuration argument: %s\n",
                                                item);
                                goto error;
                        }
                        init_fmt = NULL;
                }
        }

        s->fd = open(dev_name, O_RDWR);

        if(s->fd == -1) {
                char errbuf[1024];
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to open input device %s: %s\n",
                                dev_name, strerror_r(errno, errbuf, sizeof errbuf));
                goto error;
        }

        struct v4l2_capability   capability;
        memset(&capability, 0, sizeof(capability));
        if (ioctl(s->fd,VIDIOC_QUERYCAP, &capability) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "ioctl VIDIOC_QUERYCAP");
                goto error;
        }

        if (!(capability.device_caps & V4L2_CAP_VIDEO_CAPTURE)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s, %s can't capture\n",capability.card,capability.bus_info);
                goto error;
        }

        if (!(capability.device_caps & V4L2_CAP_STREAMING)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Streaming capability not present.\n");
                goto error;
        }

        int index = 0;

        if (ioctl(s->fd, VIDIOC_S_INPUT, &index) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Could not enable input (VIDIOC_S_INPUT)");
                goto error;
        }


        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(s->fd, VIDIOC_G_FMT, &fmt) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get video format");
                goto error;
        }

        struct v4l2_streamparm stream_params;
        memset(&stream_params, 0, sizeof(stream_params));
        stream_params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get stream params");
                goto error;
        }

        if (pixelformat) {
                fmt.fmt.pix.pixelformat = pixelformat;
        }

        if(width != 0 && height != 0) {
                fmt.fmt.pix.width = width;
                fmt.fmt.pix.height = height;
        }

        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        fmt.fmt.pix.bytesperline = 0;

        if (ioctl(s->fd, VIDIOC_S_FMT, &fmt) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to set video format");
                goto error;
        } else {
                char fourcc[5];
                memcpy(fourcc, &fmt.fmt.pix.pixelformat, sizeof(uint32_t));
                fourcc[4] = '\0';
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Capturing %dx%d %s\n", fmt.fmt.pix.width, fmt.fmt.pix.height, fourcc);
        }

        if(numerator != 0 && denominator != 0) {
                stream_params.parm.capture.timeperframe.numerator = numerator;
                stream_params.parm.capture.timeperframe.denominator = denominator;

                if(ioctl(s->fd, VIDIOC_S_PARM, &stream_params) != 0) {
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to set stream params");
                        goto error;
                }
        }

        assert(fmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE);
        memcpy(&s->src_fmt, &fmt, sizeof(fmt));
        memcpy(&s->dst_fmt, &fmt, sizeof(fmt));
        s->dst_fmt.fmt.pix.bytesperline = 0;
        s->dst_fmt.fmt.pix.colorspace = V4L2_COLORSPACE_DEFAULT;

        if(ioctl(s->fd, VIDIOC_G_FMT, &fmt) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get video format");
                goto error;
        }

        if(ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get stream params");
                goto error;
        }

        s->desc.tile_count = 1;

        if (v4l2_convert_to == VIDEO_CODEC_NONE) {
                s->desc.color_spec = get_v4l2_to_ug(fmt.fmt.pix.pixelformat);
                if (s->desc.color_spec == VIDEO_CODEC_NONE) {
                        char fcc[5];
                        memcpy(fcc, &fmt.fmt.pix.pixelformat, 4);
                        fcc[4] = '\0';
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "No mapping for FCC '%s', converting to RGB!\n", fcc);
                        v4l2_convert_to = s->dst_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
                }
        }
        if (v4l2_convert_to != VIDEO_CODEC_NONE) {
                s->dst_fmt.fmt.pix.pixelformat = get_ug_to_v4l2(v4l2_convert_to);
                if (s->dst_fmt.fmt.pix.pixelformat == 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot find %s to V4L2 mapping!\n", get_codec_name(v4l2_convert_to));
                        goto error;
                }
                s->desc.color_spec = v4l2_convert_to;
        }

        switch(fmt.fmt.pix.field) {
                case V4L2_FIELD_NONE:
                        s->desc.interlacing = PROGRESSIVE;
                        break;
                case V4L2_FIELD_SEQ_TB:
                        s->desc.interlacing = UPPER_FIELD_FIRST;
                        break;
                case V4L2_FIELD_SEQ_BT:
                        s->desc.interlacing = LOWER_FIELD_FIRST;
                        break;
                case V4L2_FIELD_INTERLACED:
                        s->desc.interlacing = INTERLACED_MERGED;
                        break;
                case V4L2_FIELD_TOP:
                case V4L2_FIELD_BOTTOM:
                case V4L2_FIELD_ALTERNATE:
                case V4L2_FIELD_INTERLACED_TB:
                case V4L2_FIELD_INTERLACED_BT:
                default:
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported interlacing format reported from driver (%d).\n", fmt.fmt.pix.field);
                        goto error;
        }
        s->desc.fps = (double) stream_params.parm.capture.timeperframe.denominator /
                stream_params.parm.capture.timeperframe.numerator;
        s->desc.width = fmt.fmt.pix.width;
        s->desc.height = fmt.fmt.pix.height;

        if (v4l2_convert_to != VIDEO_CODEC_NONE) {
                s->convert = v4lconvert_create(s->fd);
        } else {
                s->convert = NULL;
        }

        struct v4l2_requestbuffers reqbuf;

        memset(&reqbuf, 0, sizeof(reqbuf));
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbuf.memory = V4L2_MEMORY_MMAP;
        reqbuf.count = s->buffer_count;

        if (ioctl (s->fd, VIDIOC_REQBUFS, &reqbuf) != 0) {
                if (errno == EINVAL)
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Video capturing or mmap-streaming is not supported\n");
                else
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "VIDIOC_REQBUFS");
                goto error;

        }

        if (reqbuf.count < 2) {
                /* You may need to free the buffers here. */
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Not enough buffer memory\n");
                goto error;
        }

        for (unsigned int i = 0; i < reqbuf.count; i++) {
                struct v4l2_buffer buf;
                memset(&buf, 0, sizeof(buf));
                buf.type = reqbuf.type;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;

                if (-1 == ioctl (s->fd, VIDIOC_QUERYBUF, &buf)) {
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "VIDIOC_QUERYBUF");
                        goto error;
                }

                s->buffers[i].length = buf.length; /* remember for munmap() */

                s->buffers[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE, /* recommended */
                                MAP_SHARED,             /* recommended */
                                s->fd, buf.m.offset);

                if (MAP_FAILED == s->buffers[i].start) {
                        /* If you do not exit here you should unmap() and free()
                           the buffers mapped so far. */
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "mmap");
                        goto error;
                }

                buf.flags = 0;

                if(ioctl(s->fd, VIDIOC_QBUF, &buf) != 0) {
                        log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to enqueue buffer");
                        goto error;
                }
        }

        if(ioctl(s->fd, VIDIOC_STREAMON, &reqbuf.type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to start stream");
                goto error;
        };

        gettimeofday(&s->t0, NULL);
        s->frames = 0;

        free(tmp);

        printf("Enable video input: %dx%d %f fps %s, codec %s\n", s->desc.width, s->desc.height, s->desc.fps, get_interlacing_description(s->desc.interlacing), get_codec_name(s->desc.color_spec));

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(tmp);

        vidcap_v4l2_common_cleanup(s);

        return VIDCAP_INIT_FAIL;
}

static void vidcap_v4l2_done(void *state)
{
        struct vidcap_v4l2_state *s = (struct vidcap_v4l2_state *) state;

        vidcap_v4l2_common_cleanup(s);
}

static void vidcap_v4l2_dispose_video_frame(struct video_frame *frame) {
        struct v4l2_dispose_deq_buffer_data *data =
                (struct v4l2_dispose_deq_buffer_data *) frame->callbacks.dispose_udata;

        if (data) {
                pthread_mutex_lock(&data->s->lock);
                simple_linked_list_append(data->s->buffers_to_enqueue, data);
                pthread_mutex_unlock(&data->s->lock);
                pthread_cond_signal(&data->s->cv);
        } else {
                free(frame->tiles[0].data);
        }

        vf_free(frame);
}

static struct video_frame * vidcap_v4l2_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_v4l2_state *s = (struct vidcap_v4l2_state *) state;
        struct video_frame *out;

        pthread_mutex_lock(&s->lock);
        enqueue_all_finished_frames(s);
        while (s->dequeued_buffers == s->buffer_count) { // we cannot dequeue any buffer
                pthread_cond_wait(&s->cv, &s->lock);
                enqueue_all_finished_frames(s);
        }
        pthread_mutex_unlock(&s->lock);

        *audio = NULL;

        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if(ioctl(s->fd, VIDIOC_DQBUF, &buf) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to dequeue buffer");
                return NULL;
        };

        s->dequeued_buffers += 1;

        out = vf_alloc_desc(s->desc);
        out->callbacks.dispose = vidcap_v4l2_dispose_video_frame;

        if (!s->convert) {
                struct v4l2_dispose_deq_buffer_data *frame_data =
                        malloc(sizeof(struct v4l2_dispose_deq_buffer_data));
                frame_data->s = s;
                memcpy(&frame_data->buf, &buf, sizeof(buf));
                out->tiles[0].data = s->buffers[frame_data->buf.index].start;
                out->tiles[0].data_len = frame_data->buf.bytesused;
                out->callbacks.dispose_udata = frame_data;
        } else {
                out->callbacks.dispose_udata = NULL;
                out->tiles[0].data = (char *) malloc(out->tiles[0].data_len);
                int ret = v4lconvert_convert(s->convert,
                                &s->src_fmt,  /*  in */
                                &s->dst_fmt, /*  in */
                                s->buffers[buf.index].start,
                                buf.bytesused,
                                (unsigned char *) out->tiles[0].data,
                                out->tiles[0].data_len);

                // we do not need the driver buffer any more
                if (ioctl(s->fd, VIDIOC_QBUF, &buf) != 0) {
                        log_perror(LOG_LEVEL_ERROR, "[V4L2 capture] Unable to enqueue buffer");
                } else {
                        s->dequeued_buffers -= 1;
                }

                if(ret == -1) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error converting video: %s\n", v4lconvert_get_error_message(s->convert));
                        VIDEO_FRAME_DISPOSE(out);
                        return NULL;
                }

                out->tiles[0].data_len = ret;
        }

        s->frames++;

        struct timeval t;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, s->t0);
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[V4L2 capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = t;
                s->frames = 0;
        }


        return out;
}

static const struct video_capture_info vidcap_v4l2_info = {
        vidcap_v4l2_probe,
        vidcap_v4l2_init,
        vidcap_v4l2_done,
        vidcap_v4l2_grab,
        false
};

REGISTER_MODULE(v4l2, &vidcap_v4l2_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


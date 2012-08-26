/*
 * FILE:    v4l2.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */


#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "video_capture/v4l2.h"

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
#include "tv.h"
#include "video.h"
#include "video_capture.h"
#include "video_codec.h"


/* prototypes of functions defined in this module */
static void show_help(void);
void print_fps(int fd, struct v4l2_frmivalenum *param);

#define DEFAULT_DEVICE "/dev/video0"

struct vidcap_v4l2_state {
        struct video_frame *frame;
        struct tile *tile;

        int fd;
        struct v4l2_buffer buffer[2];
        struct {
                void *start;
                size_t length;
        } buffers[2];

        unsigned int buffer_network;

        bool conversion_needed;
        struct v4lconvert_data *convert;
        struct v4l2_format src_fmt, dst_fmt;

        struct timeval t0;
        int frames;
};


void print_fps(int fd, struct v4l2_frmivalenum *param) {
        int res = ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, param);

        if(res == -1) {
                fprintf(stderr, "[V4L2] Unable to get FPS.\n");
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

static void show_help()
{
        printf("V4L2 capture\n");
        printf("Usage\n");
        printf("\t-t v4l2[:<dev>[:<pixel_fmt>:[<width>:<height>[:<tpf>]]]]\n");
        printf("\t\tuse device <dev> for grab (default: %s)\n", DEFAULT_DEVICE);
        printf("\t\t<tpf> - time per frame in format <numerator>/<denominator>\n");

        for (int i = 0; i < 64; ++i) {
                char name[32];
                int res;

                snprintf(name, 32, "/dev/video%d", i);
                int fd = open(name, O_RDWR);
                if(fd == -1) continue;

                printf("\t%sDevice %s:\n", 
                                (i == 0 ? "(*) " : "    "),
                                name);


                struct v4l2_fmtdesc format;
                memset(&format, 0, sizeof(format));
                format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                format.index = 0;

                struct v4l2_format fmt;
                memset(&fmt, 0, sizeof(fmt));
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if(ioctl(fd, VIDIOC_G_FMT, &fmt) != 0) {
                        perror("[V4L2] Unable to get video formant");
                        continue;
                }

                while(ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0) {
                        printf("\t\t");
                        if(fmt.fmt.pix.pixelformat == format.pixelformat) {
                                printf("(*) ");
                        } else {
                                printf("    ");
                        }
                        printf("Pixel format %4s (%s). Available frame sizes:\n",
                                        (char *) &format.pixelformat, format.description);

                        struct v4l2_frmsizeenum size;
                        memset(&size, 0, sizeof(size));
                        size.pixel_format = format.pixelformat;

                        res = ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size);

                        if(res == -1) {
                                close(fd);
                                fprintf(stderr, "[V4L2] Unable to get frame size iterator.\n");
                                continue;
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
                                        printf("\t\t\t%u-%ux%u-%u with steps %u vertically and %u horizontally",
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

                close(fd);
        }
}

struct vidcap_type * vidcap_v4l2_probe(void)
{
        struct vidcap_type*		vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_V4L2_ID;
                vt->name        = "v4l2";
                vt->description = "V4L2 capture";
        }
        return vt;
}

void * vidcap_v4l2_init(char *init_fmt, unsigned int flags)
{
        struct vidcap_v4l2_state *s;
        char *dev_name = DEFAULT_DEVICE;
        uint32_t pixelformat = 0;
        uint32_t width = 0,
                 height = 0;
        uint32_t numerator = 0,
                 denominator = 0;

        UNUSED(flags);

        printf("vidcap_v4l2_init\n");

        if(init_fmt && strcmp(init_fmt, "help") == 0) {
               show_help(); 
               return NULL;
        }


        s = (struct vidcap_v4l2_state *) malloc(sizeof(struct vidcap_v4l2_state));
        if(s == NULL) {
                printf("Unable to allocate v4l2 capture state\n");
                return NULL;
        }

        if(init_fmt) {
                char *save_ptr = NULL;
                char *item;
                int i = 0;
                while((item = strtok_r(init_fmt, ":", &save_ptr))) {
                        int len;
                        switch (i) {
                                case 0:
                                        dev_name = item;
                                        break;
                                case 1:
                                        {
                                                union {
                                                        uint32_t fourcc;
                                                        char str[4];
                                                } str_to_uint;
                                                len = 4;
                                                if(strlen(item) < 4) len = strlen(item);
                                                memset(str_to_uint.str, 0, 4);
                                                memcpy(str_to_uint.str, item, len);
                                                pixelformat = str_to_uint.fourcc;
                                        }
                                        break;
                                case 2:
                                        width = atoi(item);
                                        break;
                                case 3:
                                        height = atoi(item);
                                        break;
                                case 4:
                                        numerator = atoi(item);
                                        break;
                                case 5:
                                        denominator = atoi(item);
                                        break;

                        }
                        init_fmt = NULL;
                        ++i;
                }
        }

        s->frame = NULL;
        s->fd = open(dev_name, O_RDWR);

        if(s->fd == -1) {
                perror("[V4L2] Unable to open input device");
                goto error_fd;
        }

        int index = 0;

        if (ioctl(s->fd, VIDIOC_S_INPUT, &index) != 0) {
                perror ("Could not enable input (VIDIOC_S_INPUT)");
                goto error_fd;
        }

        struct v4l2_capability   capability;
        memset(&capability, 0, sizeof(capability));
        if (ioctl(s->fd,VIDIOC_QUERYCAP, &capability) != 0) {
                perror("V4L2: ioctl VIDIOC_QUERYCAP");
                goto error_fd;
        }

        if (!(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                fprintf(stderr, "%s, %s can't capture\n",capability.card,capability.bus_info);
                goto error_fd;
        }

        if (!(capability.capabilities & V4L2_CAP_STREAMING)) {
                fprintf(stderr, "[V4L2] Streaming capability not present.\n");
                goto error_fd;
        }


        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if(ioctl(s->fd, VIDIOC_G_FMT, &fmt) != 0) {
                perror("[V4L2] Unable to get video formant");

                goto error_fd;
        }

        struct v4l2_streamparm stream_params;
        memset(&stream_params, 0, sizeof(stream_params));
        stream_params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if(ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                perror("[V4L2] Unable to get stream params");

                goto error_fd;
        }

        if(pixelformat) {
                fmt.fmt.pix.pixelformat = pixelformat;

                if(width != 0 && height != 0) {
                        fmt.fmt.pix.width = width;
                        fmt.fmt.pix.height = height;
                }
                fmt.fmt.pix.field = V4L2_FIELD_ANY;
                fmt.fmt.pix.bytesperline = 0;

                if(ioctl(s->fd, VIDIOC_S_FMT, &fmt) != 0) {
                        perror("[V4L2] Unable to set video formant");
                        goto error_fd;
                }

                if(numerator != 0 && denominator != 0) {
                        stream_params.parm.capture.timeperframe.numerator = numerator;
                        stream_params.parm.capture.timeperframe.denominator = denominator;

                        if(ioctl(s->fd, VIDIOC_S_PARM, &stream_params) != 0) {
                                perror("[V4L2] Unable to set stream params");

                                goto error_fd;
                        }
                }
        }

        memcpy(&s->src_fmt, &fmt, sizeof(fmt));
        memcpy(&s->dst_fmt, &fmt, sizeof(fmt));

        if(ioctl(s->fd, VIDIOC_G_FMT, &fmt) != 0) {
                perror("[V4L2] Unable to get video formant");

                goto error_fd;
        }

        if(ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                perror("[V4L2] Unable to get stream params");

                goto error_fd;
        }

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

        s->conversion_needed = false;

        switch(fmt.fmt.pix.pixelformat) {
                case V4L2_PIX_FMT_YUYV:
                        s->frame->color_spec = YUYV;
                        break;
                case V4L2_PIX_FMT_UYVY:
                        s->frame->color_spec = UYVY;
                        break;
                case V4L2_PIX_FMT_RGB24:
                        s->frame->color_spec = RGB;
                        break;
                case V4L2_PIX_FMT_RGB32:
                        s->frame->color_spec = RGBA;
                        break;
                case V4L2_PIX_FMT_MJPEG:
                        s->frame->color_spec = JPEG;
                        break;
                default:
                        s->conversion_needed = true;
                        s->dst_fmt.fmt.pix.pixelformat =  V4L2_PIX_FMT_RGB24;
                        s->frame->color_spec = RGB;
                        break;
        }

        switch(fmt.fmt.pix.field) {
                case V4L2_FIELD_NONE:
                        s->frame->interlacing = PROGRESSIVE;
                        break;
                case V4L2_FIELD_TOP:
                        s->frame->interlacing = UPPER_FIELD_FIRST;
                        break;
                case V4L2_FIELD_BOTTOM:
                        s->frame->interlacing = LOWER_FIELD_FIRST;
                        break;
                case V4L2_FIELD_INTERLACED:
                        s->frame->interlacing = INTERLACED_MERGED;
                        break;
                case V4L2_FIELD_SEQ_TB:
                case V4L2_FIELD_SEQ_BT:
                case V4L2_FIELD_ALTERNATE:
                case V4L2_FIELD_INTERLACED_TB:
                case V4L2_FIELD_INTERLACED_BT:
                default:
                        fprintf(stderr, "[V4L2] Unsupported interlacing format reported from driver.\n");
                        goto free_frame;
        }
        s->frame->fps = (double) denominator / numerator;
        s->tile->width = fmt.fmt.pix.width;
        s->tile->height = fmt.fmt.pix.height;

        if(s->conversion_needed) {
                s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) *
                        s->tile->height;
                s->tile->data = malloc(s->tile->data_len);
                s->convert = v4lconvert_create(s->fd);
        } else {
                s->convert = NULL;
        }

        struct v4l2_requestbuffers reqbuf;

        memset(&reqbuf, 0, sizeof(reqbuf));
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbuf.memory = V4L2_MEMORY_MMAP;
        reqbuf.count = 2;

        if (ioctl (s->fd, VIDIOC_REQBUFS, &reqbuf) != 0) {
                if (errno == EINVAL)
                        printf("Video capturing or mmap-streaming is not supported\n");
                else
                        perror("VIDIOC_REQBUFS");
                goto free_frame;

        }

        if (reqbuf.count < 2) {
                /* You may need to free the buffers here. */
                printf("Not enough buffer memory\n");
                goto free_frame;
        }

        for (unsigned int i = 0; i < reqbuf.count; i++) {
                memset(&s->buffer[i], 0, sizeof(s->buffer[i]));
                s->buffer[i].type = reqbuf.type;
                s->buffer[i].memory = V4L2_MEMORY_MMAP;
                s->buffer[i].index = i;

                if (-1 == ioctl (s->fd, VIDIOC_QUERYBUF, &s->buffer[i])) {
                        perror("VIDIOC_QUERYBUF");
                        goto free_frame;
                }

                s->buffers[i].length = s->buffer[i].length; /* remember for munmap() */

                s->buffers[i].start = mmap(NULL, s->buffer[i].length,
                                PROT_READ | PROT_WRITE, /* recommended */
                                MAP_SHARED,             /* recommended */
                                s->fd, s->buffer[i].m.offset);

                if (MAP_FAILED == s->buffers[i].start) {
                        /* If you do not exit here you should unmap() and free()
                           the buffers mapped so far. */
                        perror("mmap");
                        goto free_frame;
                }

                s->buffer[i].flags = 0;
        }

        if(ioctl(s->fd, VIDIOC_QBUF, &s->buffer[0]) != 0) {
                perror("Unable to enqueue buffer");
                goto free_frame;
        };

        if(ioctl(s->fd, VIDIOC_STREAMON, &reqbuf.type) != 0) {
                perror("Unable to start stream");
                goto free_frame;
        };

        s->buffer_network = 1;

        gettimeofday(&s->t0, NULL);
        s->frames = 0;

        return s;

free_frame:
        vf_free(s->frame);
error_fd:
        close(s->fd);
        free(s);
        return NULL;
}

void vidcap_v4l2_finish(void *state)
{
        UNUSED(state);
}

void vidcap_v4l2_done(void *state)
{
        struct vidcap_v4l2_state *s = (struct vidcap_v4l2_state *) state;

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if(ioctl(s->fd, VIDIOC_STREAMOFF, &type) != 0) {
                fprintf(stderr, "Stream stopping error.\n");
        };

        close(s->fd);

        if(!s->conversion_needed) {
                vf_free(s->frame);
        } else {
                vf_free_data(s->frame);
                v4lconvert_destroy(s->convert);
        }

        free(s);
}

struct video_frame * vidcap_v4l2_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_v4l2_state *s = (struct vidcap_v4l2_state *) state;

        *audio = NULL;

        if(ioctl(s->fd, VIDIOC_QBUF, &s->buffer[s->buffer_network]) != 0) {
                perror("Unable to enqueue buffer");
        };

        s->buffer_network = (s->buffer_network + 1) % 2;

        if(ioctl(s->fd, VIDIOC_DQBUF, &s->buffer[s->buffer_network]) != 0) {
                perror("Unable to dequeue buffer");
        };

        if(!s->conversion_needed) {
                s->tile->data = s->buffers[s->buffer_network].start;
                s->tile->data_len = s->buffers[s->buffer_network].length;
        } else {
                int ret = v4lconvert_convert(s->convert,
                                &s->src_fmt,  /*  in */
                                &s->dst_fmt, /*  in */
                                s->buffers[s->buffer_network].start, 
                                s->buffers[s->buffer_network].length,
                                (unsigned char *) s->tile->data, 
                                s->tile->data_len);
                if(ret == -1) {
                        fprintf(stderr, "Error converting video.\n");
                }

                s->tile->data_len = ret;
        }

        s->frames++;

        struct timeval t;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, s->t0);
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                fprintf(stderr, "[V4L2 capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = t;
                s->frames = 0;
        }


        return s->frame;
}


/**
 * @file    video_capture/v4l2.c
 * @author  Milos Liska      <xliska@fi.muni.cz>
 * @author  Martin Piatka    <piatka@cesnet.cz>
 * @author  Martin Pulec     <martin.pulec@cesnet.cz>
 */
 /*
 * Copyright (c) 2012-2025 CESNET
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
#include "config.h"               // for HAVE_LIBV4LCONVERT
#endif /* HAVE_CONFIG_H */

#ifdef HAVE_LIBV4LCONVERT
#include <libv4lconvert.h>
#endif
#ifdef HAVE_LINUX_VERSION_H
#include <linux/version.h>
#else
#define KERNEL_VERSION(x,y,z) -1
#endif

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>               // for bool, false, true
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>              // for gettimeofday, timeval
#include <unistd.h>

enum {
        DEFAULT_BUF_COUNT = 3,
        MAX_BUF_COUNT     = 30,
};
#define MOD_NAME "[V4L cap.] "

#include "debug.h"
#include "lib_common.h"
#include "tv.h"
#include "types.h"                 // for device_info, video_desc, tile, vid...
#include "utils/color_out.h"
#include "utils/list.h"
#include "utils/macros.h"
#include "utils/misc.h" // ug_strerror
#include "v4l2_common.h"
#include "video_capture.h"
#include "video_capture_params.h"  // for vidcap_params_get_fmt, vidcap_para...
#include "video_codec.h"           // for codec_is_planar, get_codec_name
#include "video_frame.h"           // for get_interlacing_suffix, vf_alloc_desc

struct audio_frame;
struct vidcap_params;

/* prototypes of functions defined in this module */
static void print_fps(int fd, struct v4l2_frmivalenum *param);

struct vidcap_v4l2_state {
        struct video_desc desc;

        int fd;
        struct v4l2_buffer_data buffers[MAX_BUF_COUNT];

        _Bool permissive; ///< do not fail if parameters (size, FPS...) not set exactly
#ifdef HAVE_LIBV4LCONVERT
        struct v4lconvert_data *convert;
#endif
        struct v4l2_format src_fmt; ///< captured format
        struct v4l2_format dst_fmt; ///< converted format if v4lconvert is used

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
        if (s->fd != -1 && ioctl(s->fd, VIDIOC_STREAMOFF, &type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Stream stopping error");
        };

        pthread_mutex_lock(&s->lock);
        enqueue_all_finished_frames(s);
        while (s->dequeued_buffers != 0) {
                pthread_cond_wait(&s->cv, &s->lock);
                enqueue_all_finished_frames(s);
        }
        pthread_mutex_unlock(&s->lock);

        for (int i = 0; i < s->buffer_count; ++i) {
                if (s->buffers[i].start) {
                        if (-1 == munmap(s->buffers[i].start, s->buffers[i].length)) {
                                log_perror(LOG_LEVEL_ERROR, MOD_NAME "munmap");
                        }
                }
        }

        pthread_cond_destroy(&s->cv);
        pthread_mutex_destroy(&s->lock);
        simple_linked_list_destroy(s->buffers_to_enqueue);

        if (s->fd != -1)
                close(s->fd);

#ifdef HAVE_LIBV4LCONVERT
        if (s->convert) {
                v4lconvert_destroy(s->convert);
        }
#endif

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

static void show_help(_Bool full)
{
        printf("V4L2 capture\n");
        printf("Usage\n");
        color_printf(TBOLD(
            TRED("\t-t v4l2[:device=<dev>]")
                "[:codec=<pixel_fmt>][:size=<width>x<height>][:tpf=<tpf>|:fps=<"
                "fps>][:buffers=<bufcnt>][:convert=<conv>][:permissive]")
            "\n");
        color_printf("\t" TBOLD("-t v4l2:[short]help") "\n");
        printf("where\n");
        color_printf(TERM_BOLD "<dev> -" TERM_RESET "\tuse device to grab from (default: first usable)\n");
        color_printf(TERM_BOLD "\t<tpf>" TERM_RESET " - time per frame in format <numerator>/<denominator>\n");
        color_printf(TERM_BOLD "\t<bufcnt>" TERM_RESET " - number of capture buffers to be used (default: %d)\n", DEFAULT_BUF_COUNT);
        color_printf(TERM_BOLD "\t<tpf>" TERM_RESET " or " TERM_BOLD "<fps>" TERM_RESET " should be given as a single integer or a fraction\n");
        color_printf(TERM_BOLD "\t<conv>" TERM_RESET " - SW conversion, eg. to RGB (useful eg. to convert captured MJPG from USB 2.0 webcam to uncompressed),\n"
               "\t         codecs available to convert to:");
#ifdef HAVE_LIBV4LCONVERT
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                if (v4lconvert_supported_dst_format(v4l2_ug_map[i].v4l2_fcc)) {
                        color_printf(TERM_BOLD " %s" TERM_RESET, get_codec_name(v4l2_ug_map[i].ug_codec));
                }
        }
#else
        color_printf(TERM_FG_RED " v4lconvert support not compiled in!" TERM_RESET);
#endif
        printf("\n");
        printf("\t\tpermissive - do not fail if configuration values (size, FPS) are adjusted by driver and not set exactly\n");
        printf("\n");

        printf("Available devices:\n");
        for (int i = 0; i < V4L2_PROBE_MAX; ++i) {
                char name[32];

                snprintf(name, 32, "/dev/video%d", i);
                int fd = open(name, O_RDWR);
                if (fd == -1) {
                        if (errno != ENOENT) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to open input device %s: %s\n",
                                                name, ug_strerror(errno));
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

                color_printf("\t%sDevice " TERM_BOLD "%s " TERM_RESET"%s (%s)%s\n", (i == 0 ? "(*) " : "    "), name, capab.card, capab.bus_info, full ? ":" : "");

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

                while (full && ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0) {
                        printf("\t\t");
                        if(fmt.fmt.pix.pixelformat == format.pixelformat) {
                                printf("(*) ");
                        } else {
                                printf("    ");
                        }
                        printf("Pixel format ");
                        color_printf(TERM_BOLD "%4.4s" TERM_RESET " (%s). Available frame sizes:\n", (const char *) &format.pixelformat, format.description);

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

        if (full) {
                printf("(use \"shorthelp\" to display more terse output)\n");
        }
}

#ifdef HAVE_LIBV4LCONVERT
static uint32_t get_ug_to_v4l2(codec_t ug_codec) {
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                if (v4l2_ug_map[i].ug_codec == ug_codec) {
                        return v4l2_ug_map[i].v4l2_fcc;
                }
        }
        return 0;
}
#endif

static codec_t get_v4l2_to_ug(uint32_t fcc) {
        for (unsigned int i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; ++i) {
                if (v4l2_ug_map[i].v4l2_fcc == fcc) {
                        return v4l2_ug_map[i].ug_codec;
                }
        }
        return VIDEO_CODEC_NONE;
}

static void write_mode(struct mode *m,
                int width, int height,
                unsigned tpf_num, unsigned tpf_denom,
                int pixelformat)
{
        double fps = (double) tpf_denom / tpf_num;
        char codec[5];
        write_fcc(codec, pixelformat);

        codec_t ug_codec = get_v4l2_to_ug(pixelformat);
        bool force_rgb = is_codec_opaque(ug_codec) || codec_is_planar(ug_codec) || ug_codec == VIDEO_CODEC_NONE;

        snprintf(m->name, sizeof(m->name), "%dx%d %.2f fps %4s",
                        width, height, fps, codec);

        snprintf(m->id, sizeof(m->id), "{"
                        "\"codec\":\"%.4s\", "
                        "\"size\":\"%dx%d\", "
                        "\"tpf\":\"%u/%u\", "
                        "\"force_rgb\":\"%c\"}",
                        codec,
                        width, height,
                        tpf_num, tpf_denom, force_rgb ? 't' : 'f');
}

static void vidcap_v4l2_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;

        int card_count = 0;
        struct device_info *cards = NULL;

        for (int i = 0; i < V4L2_PROBE_MAX; ++i) {
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

                card_count += 1;
                cards = realloc(cards, card_count * sizeof(struct device_info));
                memset(&cards[card_count - 1], 0, sizeof(struct device_info));
                snprintf(cards[card_count - 1].dev, sizeof cards[card_count - 1].dev, ":device=%s", name);
                snprintf(cards[card_count - 1].name, sizeof cards[card_count - 1].name, "V4L2 %s", capab.card);

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
                                                                        write_mode(&cards[card_count - 1].modes[fmt_idx++],
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
        *available_cards = cards;
        *count = card_count;
}

static _Bool v4l2_cap_verify_params(_Bool permissive, const struct v4l2_format *req_format, const struct v4l2_format *actual_format, struct v4l2_streamparm *req_stream_params, struct v4l2_streamparm *actual_stream_params)
{
        int level = permissive ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR;
        if (req_format->fmt.pix.pixelformat != actual_format->fmt.pix.pixelformat) {
                log_msg(level, MOD_NAME "Unable to set requested format \"%.4s\", got \"%.4s\".\n",
                                (const char *) &req_format->fmt.pix.pixelformat, (const char *) &actual_format->fmt.pix.pixelformat);
                return 0;
        }

        if (req_format->fmt.pix.width != actual_format->fmt.pix.width ||
                        req_format->fmt.pix.height != actual_format->fmt.pix.height) {
                log_msg(level, MOD_NAME "Unable to set requested size %" PRIu32 "x%" PRIu32 ", got %" PRIu32 "x%" PRIu32 ".\n",
                                req_format->fmt.pix.width, req_format->fmt.pix.height,
                                actual_format->fmt.pix.width, actual_format->fmt.pix.height);
                return 0;
        }

        if (req_stream_params->parm.capture.timeperframe.numerator != actual_stream_params->parm.capture.timeperframe.numerator ||
                        req_stream_params->parm.capture.timeperframe.denominator != actual_stream_params->parm.capture.timeperframe.denominator) {
                log_msg(level, MOD_NAME "Unable to set requested TPF %" PRIu32 "/%" PRIu32 ", got %" PRIu32 "/%" PRIu32 ".\n", req_stream_params->parm.capture.timeperframe.numerator, req_stream_params->parm.capture.timeperframe.denominator, actual_stream_params->parm.capture.timeperframe.numerator, actual_stream_params->parm.capture.timeperframe.denominator);
                return 0;
        }

        return 1;
}

struct parsed_opts {
        uint32_t pixelformat;
        char *dev_name;
        uint32_t width;
        uint32_t height;
        uint32_t numerator;
        uint32_t denominator;
        int buffer_count;
        bool permissive;
        codec_t v4l2_convert_to;
};

static bool
parse_fmt(char *fmt, struct parsed_opts *opts)
{
        char *save_ptr = NULL;
        char *item = NULL;;
        while ((item = strtok_r(fmt, ":", &save_ptr))) {
                fmt = NULL;
                if (IS_KEY_PREFIX(item, "device")) {
                        opts->dev_name = strchr(item, '=') + 1;
                } else if (IS_KEY_PREFIX(item, "fmt") ||
                           IS_KEY_PREFIX(item, "codec")) {
                        char *fmt = strchr(item, '=') + 1;
                        union {
                                uint32_t fourcc;
                                char     str[4];
                        } str_to_uint = { .fourcc = 0 };
                        memcpy(str_to_uint.str, fmt, MIN(strlen(fmt), 4));
                        opts->pixelformat = str_to_uint.fourcc;
                } else if (IS_KEY_PREFIX(item, "size") &&
                           strchr(item, 'x') != NULL) {
                        opts->width  = atoi(strchr(item, '=') + 1);
                        opts->height = atoi(strchr(item, 'x') + 1);
                } else if (IS_KEY_PREFIX(item, "tpf")) {
                        opts->numerator   = atoi(strchr(item, '=') + 1);
                        opts->denominator = strchr(item, '/') == NULL
                                                ? 1
                                                : atoi(strchr(item, '/') + 1);
                } else if (IS_KEY_PREFIX(item, "fps")) {
                        opts->denominator = atoi(strchr(item, '=') + 1);
                        opts->numerator   = strchr(item, '/') == NULL
                                                ? 1
                                                : atoi(strchr(item, '/') + 1);
                } else if (IS_KEY_PREFIX(item, "buffers")) {
                        opts->buffer_count = atoi(strchr(item, '=') + 1);
                        assert(opts->buffer_count <= MAX_BUF_COUNT);
                } else if (IS_KEY_PREFIX(item, "convert")) {
#ifdef HAVE_LIBV4LCONVERT
                        const char *codec     = strchr(item, '=') + 1;
                        opts->v4l2_convert_to = get_codec_from_name(codec);
                        if (opts->v4l2_convert_to == VIDEO_CODEC_NONE) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "Unknown codec: %s\n",
                                        codec);
                                return false;
                        }
#else
                        MSG(ERROR, "v4lconvert support not compiled in!");
                        return false;
#endif
                } else if (IS_PREFIX(item, "permissive")) {
                        opts->permissive = 1;
                } else {
                        MSG(ERROR, "Invalid configuration argument: %s\n",
                            item);
                        return false;
                }
        }
        return true;
}

static int vidcap_v4l2_init(struct vidcap_params *params, void **state)
{
        struct parsed_opts opts = { .buffer_count = DEFAULT_BUF_COUNT };

        printf("vidcap_v4l2_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

        const char *cfg = vidcap_params_get_fmt(params);
        if (cfg && strstr(cfg, "help") != NULL) {
               show_help(strcmp(cfg, "shorthelp") != 0);
               return VIDCAP_INIT_NOERR;
        }

        struct vidcap_v4l2_state *s = calloc(1, sizeof(struct vidcap_v4l2_state));
        if(s == NULL) {
                printf("Unable to allocate v4l2 capture state\n");
                return VIDCAP_INIT_FAIL;
        }
        s->fd = -1;
        s->buffers_to_enqueue = simple_linked_list_init();
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);

        char *tmp = NULL;

        if(vidcap_params_get_fmt(params)) {
                tmp = strdup(vidcap_params_get_fmt(params));
                assert(tmp != NULL);
                if (!parse_fmt(tmp, &opts)) {
                        goto error;
                }
        }

        s->buffer_count = opts.buffer_count;
        s->permissive= opts.permissive;

        static_assert(V4L2_PROBE_MAX < 100, "Pattern below has place only for 2 digits");
        char dev_name_try[] = "/dev/videoXX";
        if (opts.dev_name != NULL) {
                s->fd = try_open_v4l2_device(LOG_LEVEL_ERROR, opts.dev_name, V4L2_CAP_VIDEO_CAPTURE);
        } else {
                for (int i = 0; i < V4L2_PROBE_MAX; ++i) {
                        snprintf(dev_name_try, sizeof dev_name_try, "/dev/video%d", i);
                        s->fd = try_open_v4l2_device(LOG_LEVEL_WARNING, dev_name_try, V4L2_CAP_VIDEO_CAPTURE);
                        if (s->fd != -1) {
                                opts.dev_name = dev_name_try;
                                break;
                        }
                }
        }
        if (s->fd == -1) {
                goto error;
        }

        struct v4l2_format fmt = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
        if (ioctl(s->fd, VIDIOC_G_FMT, &fmt) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get video format");
                goto error;
        }

        struct v4l2_streamparm stream_params = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
        if (ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get stream params");
                goto error;
        }

        if (opts.pixelformat) {
                fmt.fmt.pix.pixelformat = opts.pixelformat;
        }

        if (opts.width != 0 && opts.height != 0) {
                fmt.fmt.pix.width  = opts.width;
                fmt.fmt.pix.height = opts.height;
        }

        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        fmt.fmt.pix.bytesperline = 0;

        struct v4l2_format req_fmt = fmt;
        if (ioctl(s->fd, VIDIOC_S_FMT, &fmt) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to set video format");
                goto error;
        }
        if (opts.numerator != 0 && opts.denominator != 0) {
                stream_params.parm.capture.timeperframe.numerator =
                    opts.numerator;
                stream_params.parm.capture.timeperframe.denominator =
                    opts.denominator;
        }
        struct v4l2_streamparm req_stream_params = stream_params;
        if (opts.numerator != 0 && opts.denominator != 0 &&
            ioctl(s->fd, VIDIOC_S_PARM, &stream_params) != 0) {
                log_perror(LOG_LEVEL_ERROR,
                           MOD_NAME "Unable to set stream params");
                goto error;
        }
        if (!v4l2_cap_verify_params(s->permissive, &req_fmt, &fmt, &req_stream_params, &stream_params) && !s->permissive) {
                goto error;
        }

        assert(fmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE);
        memcpy(&s->src_fmt, &fmt, sizeof(fmt));
        memcpy(&s->dst_fmt, &fmt, sizeof(fmt));
        s->dst_fmt.fmt.pix.bytesperline = 0;
#if LINUX_VERSION_CODE > KERNEL_VERSION(3,10,0)
        s->dst_fmt.fmt.pix.colorspace = V4L2_COLORSPACE_DEFAULT;
#else // CentOS 7
        s->dst_fmt.fmt.pix.colorspace = V4L2_COLORSPACE_REC709;
#endif

        if(ioctl(s->fd, VIDIOC_G_PARM, &stream_params) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to get stream params");
                goto error;
        }

        s->desc.tile_count = 1;

        if (opts.v4l2_convert_to == VC_NONE) {
                s->desc.color_spec = get_v4l2_to_ug(fmt.fmt.pix.pixelformat);
                if (s->desc.color_spec == VIDEO_CODEC_NONE) {
                        char fcc[5];
                        memcpy(fcc, &fmt.fmt.pix.pixelformat, 4);
                        fcc[4] = '\0';
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "No mapping for FCC '%s', converting to RGB!\n", fcc);
                        opts.v4l2_convert_to = RGB;
                }
        }
        if (opts.v4l2_convert_to != VC_NONE) {
#ifdef HAVE_LIBV4LCONVERT
                s->dst_fmt.fmt.pix.pixelformat =
                    get_ug_to_v4l2(opts.v4l2_convert_to);
                if (!v4lconvert_supported_dst_format(s->dst_fmt.fmt.pix.pixelformat)) {
                        MSG(WARNING,
                            "Conversion to %s doesn't seem to be supported by "
                            "v4lconvert but proceeding as requested...\n",
                            get_codec_name(opts.v4l2_convert_to));
                }

                if (s->dst_fmt.fmt.pix.pixelformat == 0) {
                        MSG(ERROR, "Cannot find %s to V4L2 mapping!\n",
                            get_codec_name(opts.v4l2_convert_to));
                        goto error;
                }
                s->desc.color_spec = opts.v4l2_convert_to;
#endif
        }

        unsigned i = 0;
        for ( ; i < sizeof v4l2_field_map / sizeof v4l2_field_map[0]; ++i) {
                if (v4l2_field_map[i].v4l_f == fmt.fmt.pix.field) {
                        s->desc.interlacing = v4l2_field_map[i].ug_f;
                        break;
                }
        }
        if (i == sizeof v4l2_field_map / sizeof v4l2_field_map[0]) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported interlacing format reported from driver (%d).\n", fmt.fmt.pix.field);
                goto error;
        }
        s->desc.fps = (double) stream_params.parm.capture.timeperframe.denominator /
                stream_params.parm.capture.timeperframe.numerator;
        s->desc.width = fmt.fmt.pix.width;
        s->desc.height = fmt.fmt.pix.height;

#ifdef HAVE_LIBV4LCONVERT
        s->convert = NULL;
        if (opts.v4l2_convert_to != VC_NONE) {
                s->convert = v4lconvert_create(s->fd);
        }
#endif

        struct v4l2_requestbuffers reqbuf;

        memset(&reqbuf, 0, sizeof(reqbuf));
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbuf.memory = V4L2_MEMORY_MMAP;
        reqbuf.count = s->buffer_count;

        if (!set_v4l2_buffers(s->fd, &reqbuf, s->buffers)) {
                goto error;
        }
        s->buffer_count = reqbuf.count;

        if(ioctl(s->fd, VIDIOC_STREAMON, &reqbuf.type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to start stream");
                goto error;
        };

        gettimeofday(&s->t0, NULL);
        s->frames = 0;

        free(tmp);

        MSG(NOTICE, "Capturing %dx%d @%.2f%s %s from %s\n", s->desc.width,
            s->desc.height, s->desc.fps,
            get_interlacing_suffix(s->desc.interlacing),
            get_codec_name(s->desc.color_spec), opts.dev_name);

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

#ifdef HAVE_LIBV4LCONVERT
        if (s->convert) {
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
#else
        if (0) {
#endif // HAVE_LIBV4LCONVERT
        } else {
                struct v4l2_dispose_deq_buffer_data *frame_data =
                        malloc(sizeof(struct v4l2_dispose_deq_buffer_data));
                frame_data->s = s;
                memcpy(&frame_data->buf, &buf, sizeof(buf));
                out->tiles[0].data = s->buffers[frame_data->buf.index].start;
                out->tiles[0].data_len = frame_data->buf.bytesused;
                out->callbacks.dispose_udata = frame_data;
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
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(v4l2, &vidcap_v4l2_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
// vi: set et sw=8:

/**
 * @file   video_display/v4l2.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET z.s.p.o.
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
/**
 * @todo
 * * consider using v4l2_ family of functions (such as v4l2_open) - but
 *   v4l2_open currently fails witn v4l2loopback on first open (perhaps because
 *   it is non-compliant implementation). We perhaps don't need conversions, so
 *   the advantage would be (in display) perhaps only better verbosity
 *   (v4l2_log_file) and error checking (?).
 * * do we want v4lconvert? (perhaps not)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // defined HAVE_CONFIG_H
#include "config_unix.h"
#include "config_win32.h"

#define BUFFERS 2
#define MOD_NAME "[v4l2 disp.] "

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"
#include "v4l2_common.h"

struct display_v4l2_state {
        int fd;
        _Bool stream_started;
        struct v4l2_buffer_data buffers[BUFFERS];

        struct video_frame *f;
        struct v4l2_buffer buf;
};

static void deinit_device(struct display_v4l2_state *s);

static void display_v4l2_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;

        *available_cards = NULL;
        *count = 0;

        for (int i = 0; i < V4L2_PROBE_MAX; ++i) {
                char name[32];

                snprintf(name, 32, "/dev/video%d", i);
                int fd = open(name, O_RDWR);
                if(fd == -1) continue;

                struct v4l2_capability capab;
                memset(&capab, 0, sizeof capab);
                if (ioctl(fd, VIDIOC_QUERYCAP, &capab) != 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to query device capabilities for %s: %s",
                                        name, ug_strerror(errno));
                }

                if (!(capab.device_caps & V4L2_CAP_VIDEO_OUTPUT)){
                        goto next_device;
                }

                *count += 1;
                *available_cards = realloc(*available_cards, *count * sizeof **available_cards);
                memset(*available_cards + *count - 1, 0, sizeof **available_cards);
                snprintf((*available_cards)[*count - 1].dev, sizeof (*available_cards)[*count - 1].dev, ":device=%s", name);
                snprintf((*available_cards)[*count - 1].name, sizeof (*available_cards)[*count - 1].name, "V4L2 %s", capab.card);
next_device:
                close(fd);
        }
}

static void usage() {
        printf("Usage:\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-d v4l2" TERM_FG_RESET "[:device=<path>] | -d v4l2:help\n" TERM_RESET);
        printf("\n");
        printf("Available devices:\n");
        int count = 0;
        struct device_info *devices = NULL;
        void (*deleter)(void *) = free;
        display_v4l2_probe(&devices, &count, &deleter);
        if (count == 0) {
                color_printf(TERM_FG_RED "\tNo V4L2 output devices available!\n" TERM_FG_RESET);
        }
        for (int i = 0; i < count; ++i) {
                const char *dev_path = strchr(devices[i].dev, '=');
                if (strchr(dev_path, '=') != 0) {
                        dev_path = strchr(dev_path, '=') + 1;
                } // else bogus, ":device=" should be stripped
                color_printf("\t%s: " TERM_BOLD "%s\n" TERM_RESET, dev_path, devices[i].name);
        }
        deleter(devices);
}

static void *display_v4l2_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(parent);
        const char *dev_name = NULL;
        char *tok = NULL;
        char *save_ptr = NULL;
        char *fmt_c = strdupa(fmt);
        while ((tok = strtok_r(fmt_c, ":", &save_ptr)) != NULL) {
                fmt_c = NULL;
                if (strstr(tok, "device=") == tok) {
                        dev_name = strchr(fmt, '=') + 1;
                } else {
                        usage();
                        return strstr(fmt, "help") != 0 ? INIT_NOERR : NULL;
                }
        }

        struct display_v4l2_state *s = calloc(1, sizeof(struct display_v4l2_state));

        static_assert(V4L2_PROBE_MAX < 100, "Pattern below has place only for 2 digits");
        char dev_name_try[] = "/dev/videoXX";
        if (dev_name != NULL) {
                s->fd = try_open_v4l2_device(LOG_LEVEL_ERROR, dev_name, V4L2_CAP_VIDEO_OUTPUT);
        } else {
                for (int i = 0; i < V4L2_PROBE_MAX; ++i) {
                        snprintf(dev_name_try, sizeof dev_name_try, "/dev/video%d", i);
                        s->fd = try_open_v4l2_device(LOG_LEVEL_WARNING, dev_name_try, V4L2_CAP_VIDEO_OUTPUT);
                        if (s->fd != -1) {
                                dev_name = dev_name_try;
                                break;
                        }
                }
        }
        if (s->fd == -1) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to open output device %s: %s\n",
                                dev_name ? dev_name : "(any)", ug_strerror(errno));
                free(s);
                return NULL;
        }
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using display: %s\n", dev_name);

        s->buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        s->buf.memory = V4L2_MEMORY_MMAP;

        return s;
}

static void deinit_device(struct display_v4l2_state *s) {
        if (!s->stream_started) {
                return;
        }
        int type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        if (ioctl(s->fd, VIDIOC_STREAMOFF, &type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Stream stopping error");
        }

        for (int i = 0; i < BUFFERS; ++i) {
                if (s->buffers[i].start) {
                        if (-1 == munmap(s->buffers[i].start, s->buffers[i].length)) {
                                log_perror(LOG_LEVEL_ERROR, MOD_NAME "munmap");
                        }
                }
        }
        s->stream_started = 0;
}

static void display_v4l2_done(void *state)
{
        struct display_v4l2_state *s = state;

        deinit_device(s);

        close(s->fd);
        free(s);
}

static struct video_frame *display_v4l2_getf(void *state)
{
        struct display_v4l2_state *s = state;
        int ret = ioctl(s->fd, VIDIOC_DQBUF, &s->buf);
        if (ret != 0) {
                log_perror(LOG_LEVEL_WARNING, MOD_NAME "QBUF");
                return NULL;
        }

        s->f->tiles[0].data = s->buffers[s->buf.index].start;

        return s->f;
}

static int display_v4l2_putf(void *state, struct video_frame *frame, long long nonblock)
{
        UNUSED(nonblock);
        struct display_v4l2_state *s = state;

        if (frame == NULL) {
                return 0;
        }
        int ret = ioctl(s->fd, VIDIOC_QBUF, &s->buf);
        if (ret != 0) {
                log_perror(LOG_LEVEL_WARNING, MOD_NAME "QBUF");
        }
        return 0;
}

static int display_v4l2_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        enum interlacing_t supported_il_modes[sizeof v4l2_field_map / sizeof v4l2_field_map[0]];
        for (unsigned int i = 0; i < sizeof v4l2_field_map / sizeof v4l2_field_map[0]; ++i) {
                supported_il_modes[i] = v4l2_field_map[i].ug_f;
        }
        codec_t codecs[sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]] = { };
        for (unsigned i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; i++) {
                codecs[i] = v4l2_ug_map[i].ug_codec;
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof codecs > *len) {
                                return FALSE;
                        }
                        memcpy(val, codecs, sizeof codecs);
                        *len = sizeof codecs;
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if (sizeof(supported_il_modes) > *len) {
                                return FALSE;
                        }
                        memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        *len = sizeof(supported_il_modes);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}


static int display_v4l2_reconfigure(void *state, struct video_desc desc)
{
#define CHECK(cmd) if (cmd != 0) { log_perror(LOG_LEVEL_ERROR, #cmd); return FALSE; }
        struct display_v4l2_state *s = state;

        deinit_device(s);

        struct v4l2_format fmt = { .type = V4L2_BUF_TYPE_VIDEO_OUTPUT };
        CHECK(ioctl(s->fd, VIDIOC_G_FMT, &fmt));
        fmt.fmt.pix.width = desc.width;
        fmt.fmt.pix.height = desc.height;
        for (unsigned i = 0; i < sizeof v4l2_ug_map / sizeof v4l2_ug_map[0]; i++) {
                if (v4l2_ug_map[i].ug_codec == desc.color_spec) {
                        fmt.fmt.pix.pixelformat = v4l2_ug_map[i].v4l2_fcc;
                        break;
                }
        }
        for (unsigned int i = 0; i < sizeof v4l2_field_map / sizeof v4l2_field_map[0]; ++i) {
                if (desc.interlacing == v4l2_field_map[i].ug_f) {
                        fmt.fmt.pix.field = v4l2_field_map[i].v4l_f;
                        break;
                }
        }
        CHECK(ioctl(s->fd, VIDIOC_S_FMT, &fmt));

        struct v4l2_streamparm parm = { .type = V4L2_BUF_TYPE_VIDEO_OUTPUT };
        CHECK(ioctl(s->fd, VIDIOC_G_PARM, &parm));
        parm.parm.output.capability = V4L2_CAP_TIMEPERFRAME;
        parm.parm.output.timeperframe.numerator = get_framerate_d(desc.fps);
        parm.parm.output.timeperframe.denominator = get_framerate_n(desc.fps);
        CHECK(ioctl(s->fd, VIDIOC_S_PARM, &parm));

        struct v4l2_requestbuffers reqbuf = { 0 };
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        reqbuf.memory = V4L2_MEMORY_MMAP;
        reqbuf.count = BUFFERS;
        if (!set_v4l2_buffers(s->fd, &reqbuf, s->buffers)) {
                return FALSE;
        }

        if (ioctl(s->fd, VIDIOC_STREAMON, &reqbuf.type) != 0) {
                log_perror(LOG_LEVEL_ERROR, MOD_NAME "Unable to start stream");
                return FALSE;
        };

        s->f = vf_alloc_desc(desc);

        s->stream_started = 1;

        return TRUE;
#undef CHECK
}

static const struct video_display_info display_v4l2_info = {
        display_v4l2_probe,
        display_v4l2_init,
        NULL,
        display_v4l2_done,
        display_v4l2_getf,
        display_v4l2_putf,
        display_v4l2_reconfigure,
        display_v4l2_get_property,
        NULL,
        NULL,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(v4l2, &display_v4l2_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


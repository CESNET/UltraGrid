/**
 * @file   video_display/ndi.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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
 * @note
 * Although the NDI SDK provides a C interface, there are constructors for the
 * NDI structs when invoked from C++. So it is needed at least to zero-initialize
 * the structs and check the original constructors.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <Processing.NDI.Lib.h>

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"

#define MOD_NAME "[NDI disp.] "

struct display_ndi {
        NDIlib_send_instance_t pNDI_send;
        struct video_desc desc;
};

static void display_ndi_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_cards = malloc(sizeof(struct device_info));
        strncat((*available_cards)[0].id, "ndi", sizeof (*available_cards)[0].id - 1);
        strncat((*available_cards)[0].name, "NDI", sizeof (*available_cards)[0].name - 1);
        (*available_cards)[0].repeatable = true;
}

static void display_ndi_run(void *arg)
{
        UNUSED(arg);
}


static int display_ndi_reconfigure(void *state, struct video_desc desc)
{
        struct display_ndi *s = (struct display_ndi *) state;

        s->desc = desc;

        return TRUE;
}

static void usage()
{
        printf("Usage:\n");
        color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-d ndi");
        color_out(COLOR_OUT_BOLD, "[:help][:name=<n>]\n");
        printf("\twhere\n");
        color_out(COLOR_OUT_BOLD, "\t\tname\n");
        printf("\t\t\tname of the server\n");
}

static void *display_ndi_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);

        const char *ndi_name = NULL;
        char *fmt_copy = alloca(strlen(fmt) + 1);
        strcpy(fmt_copy, fmt);

        char *tmp = fmt_copy, *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (strcmp(item, "help") == 0) {
                        usage();
                        return &display_init_noerr;
                }
                if (strstr(item, "name=") != NULL) {
                        ndi_name = item + strlen("name=");
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s!\n", item);
                        return NULL;
                }
                tmp = NULL;
        }

        if (!NDIlib_initialize()) {
                return NULL;
        }

        struct display_ndi *s = calloc(1, sizeof(struct display_ndi));
        NDIlib_send_create_t create_settings = { 0 };
        create_settings.clock_video = false;
        create_settings.clock_audio = false;
        create_settings.p_ndi_name = ndi_name;
        s->pNDI_send = NDIlib_send_create(&create_settings);
        if (s->pNDI_send == NULL) {
                free(s);
                return NULL;
        }

        return s;
}

static void display_ndi_done(void *state)
{
        struct display_ndi *s = (struct display_ndi *) state;

        NDIlib_send_destroy(s->pNDI_send);
        free(s);
        NDIlib_destroy();
}

static struct video_frame *display_ndi_getf(void *state)
{
        struct display_ndi *s = (struct display_ndi *) state;

        return vf_alloc_desc_data(s->desc);
}

static int display_ndi_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct display_ndi *s = (struct display_ndi *) state;

        if (frame == NULL) {
                return TRUE;
        }

        NDIlib_video_frame_v2_t NDI_video_frame = { 0 };
        NDI_video_frame.xres = s->desc.width;
        NDI_video_frame.yres = s->desc.height;
        NDI_video_frame.FourCC = s->desc.color_spec == RGBA ? NDIlib_FourCC_type_RGBA : NDIlib_FourCC_type_UYVY;
        NDI_video_frame.p_data = frame->tiles[0].data;
        NDI_video_frame.frame_rate_N = get_framerate_n(frame->fps);
        NDI_video_frame.frame_rate_D = get_framerate_d(frame->fps);
        NDI_video_frame.frame_format_type = frame->interlacing = PROGRESSIVE ? NDIlib_frame_format_type_progressive : NDIlib_frame_format_type_interleaved;
        NDI_video_frame.timecode = NDIlib_send_timecode_synthesize;

        NDIlib_send_send_video_v2(s->pNDI_send, &NDI_video_frame);
        vf_free(frame);

        return TRUE;
}

static int display_ndi_get_property(void *state, int property, void *val, size_t *len)
{
        struct display_ndi *s = (struct display_ndi *) state;
        codec_t codecs[] = {RGBA, UYVY};
        int rgb_shift[] = {0, 8, 16};
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED};
        int count = 0;

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof codecs <= *len) {
                                memcpy(val, codecs, sizeof codecs);
                                *len = sizeof codecs;
                        } else {
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if (sizeof rgb_shift > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof rgb_shift);
                        *len = sizeof rgb_shift;
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if (sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
#if 0
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                if (desc->ch_count <= 2) {
                                        desc->ch_count = 2;
                                } else if (desc->ch_count > 2 && desc->ch_count <= 8) {
                                        desc->ch_count = 8;
                                } else {
                                        desc->ch_count = 16;
                                }
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 4;
                        }
                        break;
#endif
                default:
                        return FALSE;
        }
        return TRUE;
}

static void display_ndi_put_audio_frame(void *state, struct audio_frame *frame)
{
}

static int display_ndi_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
}

static const struct video_display_info display_ndi_info = {
        display_ndi_probe,
        display_ndi_init,
        display_ndi_run,
        display_ndi_done,
        display_ndi_getf,
        display_ndi_putf,
        display_ndi_reconfigure,
        display_ndi_get_property,
        display_ndi_put_audio_frame,
        display_ndi_reconfigure_audio
};

REGISTER_MODULE(ndi, &display_ndi_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

/* vim: set expandtab sw=8 cino=N-8: */

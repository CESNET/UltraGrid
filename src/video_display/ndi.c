/**
 * @file   video_display/ndi.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2022 CESNET, z. s. p. o.
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
 * @file
 * @note
 * Although the NDI SDK provides a C interface, there are constructors for the
 * NDI structs when invoked from C++. So it is needed at least to zero-initialize
 * the structs and check the original constructors.
 *
 * There is also async NDI API - it may improve the throughput if needed
 * (NDIlib_send_send_video_async_v2).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "ndi_common.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"

#define DEFAULT_AUDIO_LEVEL 0
#define MOD_NAME "[NDI disp.] "

struct display_ndi {
        LIB_HANDLE lib;
        const NDIlib_t *NDIlib;
        int audio_level; ///< audio reference level - usually 0 or 20
        NDIlib_send_instance_t pNDI_send;
        struct video_desc desc;
        struct audio_desc audio_desc;

        char *convert_buffer; ///< for codecs that need conversion (eg. Y216->P216)
};

static void display_ndi_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *count = 1;
        *available_cards = calloc(1, sizeof(struct device_info));
        strncpy((*available_cards)[0].dev, "", sizeof (*available_cards)[0].dev - 1);
        strncpy((*available_cards)[0].name, "NDI", sizeof (*available_cards)[0].name - 1);
        strncpy((*available_cards)[0].extra, "\"embeddedAudioAvailable\":\"t\"", sizeof (*available_cards)[0].extra - 1);
        (*available_cards)[0].repeatable = true;
        *deleter = free;
}

static void display_ndi_run(void *arg)
{
        UNUSED(arg);
}

static int display_ndi_reconfigure(void *state, struct video_desc desc)
{
        struct display_ndi *s = (struct display_ndi *) state;

        s->desc = desc;
        free(s->convert_buffer);
        s->convert_buffer = malloc(MAX_BPS * desc.width * desc.height + MAX_PADDING);

        return TRUE;
}

static void usage()
{
        printf("Usage:\n");
        color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-d ndi");
        color_out(COLOR_OUT_BOLD, "[:help][:name=<n>][:audio_level=<x>]\n");
        printf("\twhere\n");
        color_out(COLOR_OUT_BOLD, "\t\tname\n");
        printf("\t\t\tthe name of the server\n");
        color_out(COLOR_OUT_BOLD, "\t\taudio_level\n");
        printf("\t\t\taudio headroom above reference level (in dB, or mic/line, default %d)\n", DEFAULT_AUDIO_LEVEL);
}

#define BEGIN_TRY int ret = 0; do
#define END_TRY while(0);
#define THROW(x) ret = (x); break;
#define COMMON
#define CATCH(x) if (ret == (x))
#define FAIL (-1)
#define HELP_SHOWN 1
static void *display_ndi_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(parent);
        NDI_PRINT_COPYRIGHT();

        char *fmt_copy = NULL;
        struct display_ndi *s = calloc(1, sizeof(struct display_ndi));
        s->audio_level = DEFAULT_AUDIO_LEVEL;

        BEGIN_TRY {
                fmt_copy = strdup(fmt);
                assert(fmt_copy != NULL);

                const char *ndi_name = NULL;
                char *tmp = fmt_copy, *item, *save_ptr;
                while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                        if (strcmp(item, "help") == 0) {
                                usage();
                                THROW(HELP_SHOWN);
                        }
                        if (strstr(item, "audio_level=") != NULL) {
                                char *val = item + strlen("audio_level=");
                                if (strcasecmp(val, "mic") == 0) {
                                        s->audio_level = 0;
                                } else if (strcasecmp(val, "line") == 0) {
                                        s->audio_level = 20; // NOLINT
                                } else {
                                        char *endptr = NULL;
                                        long val_num = strtol(val, &endptr, 0);
                                        if (val_num < 0 || val_num >= INT_MAX || *val == '\0' || *endptr != '\0') {
                                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong value: %s!\n", val);
                                                THROW(FAIL);
                                        }
                                        s->audio_level = val_num; // NOLINT
                                }
                        } else if (strstr(item, "name=") != NULL) {
                                ndi_name = item + strlen("name=");
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s!\n", item);
                                THROW(FAIL);
                        }
                        tmp = NULL;
                }

                s->NDIlib = NDIlib_load(&s->lib);
                if (s->NDIlib == NULL) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot open NDI library!\n");
                        THROW(FAIL);
                }

                if (!s->NDIlib->initialize()) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot initialize NDI library!\n");
                        THROW(FAIL);
                }

                NDIlib_send_create_t NDI_send_create_desc = { 0 };
                NDI_send_create_desc.clock_video = false;
                NDI_send_create_desc.clock_audio = false;
                NDI_send_create_desc.p_ndi_name = ndi_name;
                s->pNDI_send = s->NDIlib->send_create(&NDI_send_create_desc);
                if (s->pNDI_send == NULL) {
                        THROW(FAIL);
                }
        } END_TRY
        COMMON {
                free(fmt_copy);
        }
        CATCH(HELP_SHOWN) {
                free(s);
                return &display_init_noerr;
        }
        CATCH(FAIL) {
                free(s);
                return NULL;
        }

        return s;
}

static void display_ndi_done(void *state)
{
        struct display_ndi *s = (struct display_ndi *) state;

        s->NDIlib->send_destroy(s->pNDI_send);
        free(s->convert_buffer);
        s->NDIlib->destroy();
        close_ndi_library(s->lib);
        free(s);
}

static struct video_frame *display_ndi_getf(void *state)
{
        struct display_ndi *s = (struct display_ndi *) state;

        return vf_alloc_desc_data(s->desc);
}

typedef void ndi_disp_convert_t(const struct video_frame *f, char *out);

static void ndi_disp_convert_Y216_to_P216(const struct video_frame *f, char *out)
{
        assert((uintptr_t) out % 2 == 0);

        uint16_t *in = (uint16_t *)(void *) f->tiles[0].data;
        uint16_t *out_y = (uint16_t *)(void *) out;
        uint16_t *out_cb_cr = (uint16_t *)(void *) out + f->tiles[0].width * f->tiles[0].height;

        for (unsigned int i = 0; i < f->tiles[0].height; ++i) {
                for (unsigned int j = 0; j < f->tiles[0].width; j += 2) {
                        *out_y++ = *in++;
                        *out_cb_cr++ = *in++;
                        *out_y++ = *in++;
                        *out_cb_cr++ = *in++;
                }
        }
}

static const struct {
        codec_t ug_codec;
        NDIlib_FourCC_video_type_e ndi_fourcc;
        ndi_disp_convert_t *convert;
} codec_mapping[] = {
        { RGBA, NDIlib_FourCC_type_RGBA, NULL },
        { UYVY, NDIlib_FourCC_type_UYVY, NULL },
        { I420, NDIlib_FourCC_video_type_I420, NULL },
        { Y216, NDIlib_FourCC_type_P216, ndi_disp_convert_Y216_to_P216 },
};

/**
 * flag = PUTF_NONBLOCK is not implemented
 */
static int display_ndi_putf(void *state, struct video_frame *frame, int flag)
{
        struct display_ndi *s = (struct display_ndi *) state;

        if (frame == NULL) {
                return TRUE;
        }

        if (flag == PUTF_DISCARD) {
                vf_free(frame);
                return TRUE;
        }

        ndi_disp_convert_t *convert = NULL;

        NDIlib_video_frame_v2_t NDI_video_frame = { 0 };
        NDI_video_frame.xres = s->desc.width;
        NDI_video_frame.yres = s->desc.height;
        for (size_t i = 0; i < sizeof codec_mapping / sizeof codec_mapping[0]; ++i) {
                if (codec_mapping[i].ug_codec == frame->color_spec) {
                        NDI_video_frame.FourCC = codec_mapping[i].ndi_fourcc;
                        convert = codec_mapping[i].convert;
                }
        }
        assert(NDI_video_frame.FourCC != 0);
        NDI_video_frame.frame_rate_N = get_framerate_n(frame->fps);
        NDI_video_frame.frame_rate_D = get_framerate_d(frame->fps);
        NDI_video_frame.frame_format_type = frame->interlacing == PROGRESSIVE ? NDIlib_frame_format_type_progressive : NDIlib_frame_format_type_interleaved;
        NDI_video_frame.timecode = NDIlib_send_timecode_synthesize;

        if (convert != NULL) {
                convert(frame, s->convert_buffer);
                NDI_video_frame.p_data = (uint8_t *) s->convert_buffer;
        } else {
                NDI_video_frame.p_data = (uint8_t *) frame->tiles[0].data;
        }

        s->NDIlib->send_send_video_v2(s->pNDI_send, &NDI_video_frame);
        vf_free(frame);

        return TRUE;
}

static int display_ndi_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[sizeof codec_mapping / sizeof codec_mapping[0]];
        int rgb_shift[] = {0, 8, 16};
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED};

        for (size_t i = 0; i < sizeof codec_mapping / sizeof codec_mapping[0]; ++i) {
                codecs[i] = codec_mapping[i].ug_codec;
        }

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
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if (sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->bps = desc->bps <= 2 ? 2 : 4;
                                desc->codec = AC_PCM;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

#define NDI_SEND_AUDIO(frame, bit_depth) do { \
                assert((frame)->bps * 8 == (bit_depth)); \
                NDIlib_audio_frame_interleaved_ ## bit_depth ## s_t NDI_audio_frame = { 0 }; \
                NDI_audio_frame.sample_rate = (frame)->sample_rate; \
                NDI_audio_frame.no_channels = (frame)->ch_count; \
                NDI_audio_frame.reference_level = s->audio_level; \
                NDI_audio_frame.timecode = NDIlib_send_timecode_synthesize; \
                NDI_audio_frame.p_data = (int ## bit_depth ## _t *)(void *) (frame)->data; \
                NDI_audio_frame.no_samples = (frame)->data_len / (frame)->ch_count / ((bit_depth) / 8); \
                \
                s->NDIlib->util_send_send_audio_interleaved_ ## bit_depth ## s(s->pNDI_send, &NDI_audio_frame); \
        } while(0)

static void display_ndi_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct display_ndi *s = (struct display_ndi *) state;
        switch (frame->bps * 8) {
#define HANDLE_CASE(b) case b: NDI_SEND_AUDIO(frame, b); break;
        HANDLE_CASE(16)
        HANDLE_CASE(32)
        default:
                abort();
        }
}

static int display_ndi_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct display_ndi *s = (struct display_ndi *) state;
        s->audio_desc.bps = quant_samples / 8;
        s->audio_desc.ch_count = channels;
        s->audio_desc.sample_rate = sample_rate;

        return TRUE;
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
        display_ndi_reconfigure_audio,
#ifdef __APPLE__
        DISPLAY_NEEDS_MAINLOOP,
#else
        DISPLAY_DOESNT_NEED_MAINLOOP,
#endif
};

REGISTER_MODULE(ndi, &display_ndi_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

/* vim: set expandtab sw=8: */

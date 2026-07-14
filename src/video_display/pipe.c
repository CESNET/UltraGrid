/**
 * @file   video_display/pipe.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2026 CESNET, zájmové sdružení právnických osob
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

#include <assert.h>
#include <pthread.h>
#include <stdio.h>            // for fprintf, sscanf, stderr
#include <stdlib.h>           // for calloc
#include <string.h>

#include "audio/types.h"
#include "audio/utils.h"
#include "compat/c23.h"       // IWYU pragma: keep
#include "debug.h"
#include "export.h"
#include "lib_common.h"
#include "utils/color_out.h"  // for color_printf
#include "utils/list.h"
#include "utils/pthread.h"
#include "video_codec.h"
#include "video_display.h"
#include "video_display/pipe.h"
#include "video_frame.h"

#define MOD_NAME "[pipe] "

struct state_pipe {
        struct module                  *parent;
        struct pipe_frame_recv_delegate delegate;
        codec_t                         decode_to;
        struct video_desc               desc;
        struct simple_linked_list      *audio_frames;
        pthread_mutex_t                 audio_lock;
};

static void
display_pipe_usage(bool show_help)
{
        color_printf(TBOLD("pipe")
            " dummy video driver. For internal usage - please do not use.\n\n");
        if (!show_help) {
                return;
        }
        color_printf("Usage:\n");
        color_printf("\t" TBOLD(TRED("-d pipe") ":<dlg_ptr>[:codec=<c>]")
                     "\n");
        color_printf("\n");

        color_printf("Options\n");
        color_printf("\t" TBOLD("<dlg_ptr>")
                     " - pointer to struct pipe_frame_recv_delegate\n");
        color_printf("\t" TBOLD("<c>")
                     " - codec to enforce decode to the received stream\n");
        color_printf("\n");

        color_printf(TBOLD("Note:")
            " the delegate struct will be copied by init (can be temporary)\n");
        color_printf("\n");
}

/**
 * @note
 * Audio is always received regardless if enabled in flags.
 */
static void *display_pipe_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        codec_t decode_to = VIDEO_CODEC_NONE;
        const struct pipe_frame_recv_delegate *delegate = nullptr;

        if (strlen(fmt) == 0 || strcmp(fmt, "help") == 0) {
                display_pipe_usage(strcmp(fmt, "help") == 0);
                return nullptr;
        }

        sscanf(fmt, "%p", &delegate);
        if (strchr(fmt, ':') != nullptr) {
                fmt = strchr(fmt, ':') + 1;
                if (strstr(fmt, "codec=") == fmt) {
                        const char *codec_name = fmt + strlen("codec=");
                        decode_to = get_codec_from_name(codec_name);
                        if (decode_to == VIDEO_CODEC_NONE) {
                                MSG(ERROR, "Wrong codec name: %s\n", codec_name);
                                return nullptr;
                        }
                } else {
                        display_pipe_usage(true);
                        return nullptr;
                }
        }

        assert(delegate->frame_arrived != nullptr);
        struct state_pipe *s = calloc(1, sizeof *s);
        s->parent            = parent;
        s->delegate          = *delegate;
        s->decode_to         = decode_to;
        s->audio_frames      = simple_linked_list_init();
        ug_pthread_mutex_init(&s->audio_lock);

        return s;
}

static void display_pipe_done(void *state)
{
        struct state_pipe *s = (struct state_pipe *)state;

        struct audio_frame *a = nullptr;
        while ((a = simple_linked_list_pop(s->audio_frames)) != nullptr) {
                free(a->data);
                free(a);
        }
        CHK_PTHR(pthread_mutex_destroy(&s->audio_lock));
        free(s);
}

static struct video_frame *display_pipe_getf(void *state)
{
        struct state_pipe *s = (struct state_pipe *)state;

        struct video_frame *out = vf_alloc_desc_data(s->desc);
        // explicit dispose is needed because we do not process the frame
        // by ourselves but it is passed to further processing
        out->callbacks.dispose = vf_free;
        return out;
}

static void display_pipe_dispose_audio(struct audio_frame *f) {
        free(f->data);
        free(f);
}

static struct audio_frame * display_pipe_get_audio(struct state_pipe *s)
{
        CHK_PTHR(pthread_mutex_lock(&s->audio_lock));

        size_t len = 0;
        struct audio_frame *a = simple_linked_list_first(s->audio_frames);
        if (a == nullptr) {
                CHK_PTHR(pthread_mutex_unlock(&s->audio_lock));
                return nullptr;
        }
        struct audio_desc desc = audio_desc_from_frame(a);
        for (list_it it = simple_linked_list_it_init(s->audio_frames);
             it != LIST_IT_END;) {
                a = simple_linked_list_it_next(&it);
                if (!audio_desc_eq(desc, audio_desc_from_frame(a))) {
                        MSG(WARNING, "Discarding audio - incompatible format!\n");
                        simple_linked_list_remove(s->audio_frames, a);
                        free(a->data);
                        free(a);
                        // start over
                        it = simple_linked_list_it_init(s->audio_frames);
                        continue;
                }
                len += a->data_len;
        }
        if (len == 0) {
                return nullptr;
        }
        struct audio_frame *out = calloc(1, sizeof(struct audio_frame));
        audio_frame_write_desc(out, desc);
        out->max_size = len;
        out->data = (char *) malloc(len);
        while ((a = simple_linked_list_pop(s->audio_frames)) != nullptr) {
                append_audio_frame(out, a->data, a->data_len);
                free(a->data);
                free(a);
        }
        out->dispose = display_pipe_dispose_audio;
        CHK_PTHR(pthread_mutex_unlock(&s->audio_lock));
        return out;
}

static bool display_pipe_putf(void *state, struct video_frame *frame, long long flags)
{
        struct state_pipe *s = (struct state_pipe *) state;

        if (flags == PUTF_DISCARD) {
                VIDEO_FRAME_DISPOSE(frame);
                return true;
        }

        struct audio_frame *af = display_pipe_get_audio(s);
        s->delegate.frame_arrived(s->delegate.state, frame, af);

        return true;
}

static bool
get_codecs(struct state_pipe *s, void *val, size_t *len)
{
        if (s->decode_to != VIDEO_CODEC_NONE) {
                if (sizeof(codec_t) > *len) {
                        MSG(ERROR, "Insufficient prop length %zu B!\n", *len);
                        return false;
                }
                memcpy(val, &s->decode_to, sizeof(s->decode_to));
                *len = sizeof s->decode_to;
                return true;
        }
        codec_t    *out = val;
        const void *end = (char *) val + *len;
        for (int i = VC_FIRST; i < VIDEO_CODEC_COUNT; ++i) {
                const codec_t c = i;
                if (is_codec_opaque(c)) {
                        continue;
                }
                if ((void *) (out + 1) > end) {
                        MSG(ERROR, "Insufficient prop length %zu B!\n", *len);
                        return false;
                }
                memcpy(out++, &c, sizeof c);
        }
        *len = (char *) out - (char *) val;
        return true;
}

static bool display_pipe_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_pipe *s = state;
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        return get_codecs(s, val, len);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return false;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return false;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert (*len >= sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->codec = AC_PCM; // decompress
                        }
                        break;
                default:
                        return false;
        }
        return true;
}

static bool display_pipe_reconfigure(void *state, struct video_desc desc)
{
        struct state_pipe *s = (struct state_pipe *) state;

        s->desc = desc;

        return true;
}

static void display_pipe_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_pipe *s = state;
        CHK_PTHR(pthread_mutex_lock(&s->audio_lock));
        simple_linked_list_append(s->audio_frames,
                                  audio_frame_copy(frame, false));
        CHK_PTHR(pthread_mutex_unlock(&s->audio_lock));
}

static bool display_pipe_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return true;
}

static void display_pipe_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
}

static const struct video_display_info display_pipe_info = {
        display_pipe_probe,
        display_pipe_init,
        nullptr, // _run
        display_pipe_done,
        display_pipe_getf,
        display_pipe_putf,
        display_pipe_reconfigure,
        display_pipe_get_property,
        display_pipe_put_audio_frame,
        display_pipe_reconfigure_audio,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_HIDDEN_MODULE(pipe, &display_pipe_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


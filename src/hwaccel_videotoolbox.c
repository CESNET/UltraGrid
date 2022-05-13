/**
 * @file   hwaccel_videotoolbox.c
 * @author Martin Piatka <piatka@cesnet.cz>
 * @author Martin Pulec <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2021-2022 CESNET z.s.p.o.
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

#include "compat/qsort_s.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "hwaccel_videotoolbox.h"

#include <libavutil/pixdesc.h>

#define MOD_NAME "[videotoolbox dec.] "

#ifdef QSORT_S_COMP_FIRST
static int compare(void *ctx, const void *a, const void *b) {
#else
static int compare(const void *a, const void *b, void *ctx) {
#endif
        int bits_per_comp = ((const AVPixFmtDescriptor *) ctx)->comp[0].depth;
        int deptha = av_pix_fmt_desc_get(*(const enum AVPixelFormat*) a)->comp[0].depth;
        int depthb = av_pix_fmt_desc_get(*(const enum AVPixelFormat*) b)->comp[0].depth;
        if (deptha != depthb) {
                // either a or b is lower than bits_per_comp - sort higher bit depth first
                if (deptha < bits_per_comp || depthb < bits_per_comp) {
                        return depthb - deptha;
                }
                // both are equal or higher - sort lower bit depth first
                return deptha - depthb;
        }
        /// @todo more fine-grained comparison when there will be more than 2 alternative codecs
        return 0;
}

static void sort_codecs(enum AVPixelFormat sw_format, int count, enum AVPixelFormat *formats) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
        AVPixFmtDescriptor *sw_fmt_desc = (AVPixFmtDescriptor *) av_pix_fmt_desc_get(sw_format);
#pragma clang diagnostic pop
        qsort_s(formats, count, sizeof(enum AVPixelFormat), compare, sw_fmt_desc);
}

int videotoolbox_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        (void) out_codec;
        AVBufferRef *device_ref = NULL;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VIDEOTOOLBOX, &device_ref);
        if(ret < 0)
                return ret;

        enum AVPixelFormat probe_formats[] = { s->sw_pix_fmt, AV_PIX_FMT_UYVY422,
#ifdef HAVE_P210
                AV_PIX_FMT_P210LE,
#endif // defined HAVE_P210
        };
        sort_codecs(s->sw_pix_fmt, sizeof probe_formats / sizeof probe_formats[0] - 1, probe_formats + 1);

        AVBufferRef *hw_frames_ctx = NULL;
        for (unsigned i = 0; i < sizeof probe_formats / sizeof probe_formats[0]; ++i) {
                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Trying SW pixel format: %s\n", av_get_pix_fmt_name(probe_formats[i]));
                ret = create_hw_frame_ctx(device_ref,
                                s->coded_width,
                                s->coded_height,
                                AV_PIX_FMT_VIDEOTOOLBOX,
                                probe_formats[i],
                                0, //has to be 0, ffmpeg can't allocate frames by itself
                                &hw_frames_ctx);
                if (ret >= 0) {
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Successfully initialized with SW pixel format: %s\n",
                                        av_get_pix_fmt_name(probe_formats[i]));
                        break;
                }
        }

        if(ret < 0)
                goto fail;

        AVFrame *frame = av_frame_alloc();
        if(!frame){
                ret = -1;
                goto fail;
        }

        state->type = HWACCEL_VIDEOTOOLBOX;
        state->copy = true;
        state->tmp_frame = frame;

        s->hw_frames_ctx = hw_frames_ctx;
        s->hw_device_ctx = device_ref;

        return 0;

fail:
        av_frame_free(&state->tmp_frame);
        av_buffer_unref(&device_ref);
        av_buffer_unref(&hw_frames_ctx);
        return ret;
}

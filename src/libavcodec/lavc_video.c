/**
 * @file   libavcodec/lavc_video.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2021 CESNET, z. s. p. o.
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
 * References:
 * 1. [v210](https://wiki.multimedia.cx/index.php/V210)
 *
 * @todo
 * Some conversions to RGBA ignore RGB-shifts - either fix that or deprecate RGB-shifts
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "lib_common.h"
#include "libavcodec/lavc_common.h"
#include "libavcodec/lavc_video.h"

#ifdef HAVE_SWSCALE
#include <libswscale/swscale.h>

/**
* Simplified version of this: https://cpp.hotexamples.com/examples/-/-/av_opt_set_int/cpp-av_opt_set_int-function-examples.html
*
* @todo
* Use more fine-grained color-space characteristics that swscale support (eg. used in the original code).
*/
struct SwsContext *getSwsContext(unsigned int SrcW, unsigned int SrcH, enum AVPixelFormat SrcFormat, unsigned int DstW, unsigned int DstH, enum AVPixelFormat DstFormat, int64_t Flags) {
    struct SwsContext *Context = sws_alloc_context();
    if (!Context) {
            return 0;
    }

    const struct AVPixFmtDescriptor *SrcFormatDesc = av_pix_fmt_desc_get(SrcFormat);
    const struct AVPixFmtDescriptor *DstFormatDesc = av_pix_fmt_desc_get(DstFormat);

    // 0 = limited range, 1 = full range
    int SrcRange = SrcFormatDesc != NULL && (SrcFormatDesc->flags & AV_PIX_FMT_FLAG_RGB) != 0 ? 1 : 0;
    int DstRange = DstFormatDesc != NULL && (DstFormatDesc->flags & AV_PIX_FMT_FLAG_RGB) != 0 ? 1 : 0;

    int rc = 0;
    av_opt_set_int(Context, "sws_flags",  Flags, 0);
    av_opt_set_int(Context, "srcw",       SrcW, 0);
    av_opt_set_int(Context, "srch",       SrcH, 0);
    av_opt_set_int(Context, "dstw",       DstW, 0);
    av_opt_set_int(Context, "dsth",       DstH, 0);
    av_opt_set_int(Context, "src_range",  SrcRange, 0);
    av_opt_set_int(Context, "dst_range",  DstRange, 0);
    av_opt_set_int(Context, "src_format", SrcFormat, 0);
    av_opt_set_int(Context, "dst_format", DstFormat, 0);
    if ((rc = av_opt_set(Context, "threads", "auto", 0)) != 0) {
            print_libav_error(LOG_LEVEL_VERBOSE, "Swscale - cannot set thread mode", rc);
    }

    if (sws_init_context(Context, 0, 0) < 0) {
        sws_freeContext(Context);
        return 0;
    }

    return Context;
}
#endif // defined HAVE_SWSCALE

/**
 * Serializes (prints) AVFrame f to output file out
 *
 * @param f   actual (AVFrame *) cast to (void *)
 * @param out file stream to be written to
 */
void serialize_video_avframe(const void *f, FILE *out) {
        const AVFrame *frame = f;
        const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(frame->format);
        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (frame->data[comp] == NULL) {
                        break;
                }
                for (int y = 0; y < frame->height >> (comp == 0 ? 0 : fmt_desc->log2_chroma_h); ++y) {
                        if (fwrite(frame->data[comp] + y * frame->linesize[comp], frame->linesize[comp], 1, out) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "%s fwrite error\n", __func__);
                        }
                }
        }
}

/* vi: set expandtab sw=8: */

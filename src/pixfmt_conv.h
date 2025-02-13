/**
 * @file   pixfmt_conv.h
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * This file contains conversions between UltraGrid pixel formats (uncompressed
 * codec_t).
 * @sa video_codec.h
 * @sa from_lavc_vid_conv.h to_lavc_vid_conv.h
 * @sa utils/parallel_conv.h
 */
/* Copyright (c) 2005-2023 CESNET z.s.p.o.
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
 */

#ifndef PIXFMT_CONV_H_8C5FFAF7_CE36_4885_943F_527153D90865
#define PIXFMT_CONV_H_8C5FFAF7_CE36_4885_943F_527153D90865

#include "types.h" // codec_t

#ifdef _MSC_VER
#define __attribute__(a)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_R_SHIFT  0
#define DEFAULT_G_SHIFT  8
#define DEFAULT_B_SHIFT 16
#define DEFAULT_RGB_SHIFT_INIT { DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT }

/**
 * @brief Defines type for pixelformat conversions
 *
 * dst and src must not overlap.
 *
 * src should have allocated MAX_PADDING bytes more to accomodate some pixel
 * block conversions requirements (Already done by vf_alloc_desc_data() and by
 * RTP stack.)
 *
 * @param[out] dst     destination buffer
 * @param[in]  src     source buffer
 * @param[in]  dst_len expected number of bytes to be written
 * @param[in]  rshift  offset of red field inside a word (in bits)
 * @param[in]  gshift  offset of green field inside a word (in bits)
 * @param[in]  bshift  offset of blue field inside a word (in bits)
 *
 * @note
 * {r,g,b}shift are usually applicable only for output RGBA. If decoder
 * doesn't output RGBA, values are ignored.
 */
typedef void decoder_func_t(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift);
typedef decoder_func_t *decoder_t;

decoder_t        get_decoder_from_to(codec_t in, codec_t out) __attribute__((const));
decoder_t        get_best_decoder_from(codec_t in, const codec_t *out_candidates, codec_t *out);

decoder_func_t vc_copylineRGBA;
decoder_func_t vc_copylineToRGBA_inplace;
decoder_func_t vc_copylineABGRtoRGB;
decoder_func_t vc_copylineBGRAtoRGB;
decoder_func_t vc_copylineRGBtoRGBA;
decoder_func_t vc_copylineRGBtoUYVY_SSE;
decoder_func_t vc_copylineRGBtoGrayscale_SSE;
decoder_func_t vc_copylineUYVYtoRGB_SSE;
decoder_func_t vc_copylineUYVYtoGrayscale;
/// dummy conversion - ptr to it returned if no conversion needed
decoder_func_t vc_memcpy;

void v210_to_p010le(char *__restrict *__restrict out_data,
                    const int *__restrict out_linesize,
                    const char *__restrict in_data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // defined PIXFMT_CONV_H_8C5FFAF7_CE36_4885_943F_527153D90865


/**
 * @file   video_codec.h
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 */
/* Copyright (c) 2005-2021 CESNET z.s.p.o.
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
#ifndef __video_codec_h
#define __video_codec_h

#include "types.h" // codec_t

#ifndef __cplusplus
#include <stdbool.h>
#endif // !defined __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_R_SHIFT  0
#define DEFAULT_G_SHIFT  8
#define DEFAULT_B_SHIFT 16
#define DEFAULT_RGB_SHIFT_INIT { DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT }
#define MAX_BPS 8 /* for Y416 */  ///< maximal (average) number of pixels per know pixel formats (up-round if needed)
#define MAX_PADDING 64 ///< maximal padding in bytes that may be needed to align to pixfmt block size for Y416->R12L:
                       ///< 64 = vc_linesize(8 /*maximal block pix count (R12L)*/, Y416 /*codec with maximal lenght of 1st arg-sized block*/)
#define PIX_BLOCK_LCM 24 ///< least common multiple of all pixfmt block sizes in pixels (codec_info_t::block_size/codec_info_t::bpp). "contributors:" 8 R12L, 6 v210

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

/// Prints list of suppored codecs for video module
void             show_codec_help(const char *module, const codec_t *codecs8, const codec_t *codecs10, const codec_t *codecs_ge12);
/// @returns number of bits per color component
int              get_bits_per_component(codec_t codec) ATTRIBUTE(const);
int              get_subsampling(codec_t codec) ATTRIBUTE(const);
/// @returns number of bytes per pixel
double           get_bpp(codec_t codec) ATTRIBUTE(const);
uint32_t         get_fourcc(codec_t codec) ATTRIBUTE(const);
const char      *get_codec_name(codec_t codec) ATTRIBUTE(const);
const char      *get_codec_name_long(codec_t codec) ATTRIBUTE(const);
bool             is_codec_opaque(codec_t codec) ATTRIBUTE(const);
bool             is_codec_interframe(codec_t codec) ATTRIBUTE(const);
codec_t          get_codec_from_fcc(uint32_t fourcc) ATTRIBUTE(const);
codec_t          get_codec_from_name(const char *name) ATTRIBUTE(const);
const char      *get_codec_file_extension(codec_t codec) ATTRIBUTE(const);
decoder_t        get_decoder_from_to(codec_t in, codec_t out, bool slow) ATTRIBUTE(const);
decoder_t        get_best_decoder_from(codec_t in, const codec_t *out_candidates, codec_t *out, bool include_slow) ATTRIBUTE(const);

int get_pf_block_bytes(codec_t codec) ATTRIBUTE(const);
int get_pf_block_pixels(codec_t codec) ATTRIBUTE(const);
int vc_get_linesize(unsigned int width, codec_t codec) ATTRIBUTE(const);
size_t vc_get_datalen(unsigned int width, unsigned int height, codec_t codec) ATTRIBUTE(const);
void codec_get_planes_subsampling(codec_t pix_fmt, int *sub);
bool codec_is_420(codec_t pix_fmt);
bool codec_is_a_rgb(codec_t codec) ATTRIBUTE(const);
bool codec_is_in_set(codec_t codec, codec_t *set) ATTRIBUTE(const);
bool codec_is_const_size(codec_t codec) ATTRIBUTE(const);
bool codec_is_hw_accelerated(codec_t codec) ATTRIBUTE(const);
bool codec_is_planar(codec_t codec) ATTRIBUTE(const);

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
void vc_deinterlace_ex(unsigned char *src, size_t src_linesize, unsigned char *dst, size_t dst_pitch, size_t lines);

decoder_func_t vc_copyliner10k;
decoder_func_t vc_copylineRGBA;
decoder_func_t vc_copylineABGRtoRGB;
decoder_func_t vc_copylineRGBtoRGBA;
decoder_func_t vc_copylineRGBtoUYVY;
decoder_func_t vc_copylineRGBtoUYVY_SSE;
decoder_func_t vc_copylineRGBtoGrayscale_SSE;
decoder_func_t vc_copylineR12LtoRG48;
decoder_func_t vc_copylineRG48toR12L;
decoder_func_t vc_copylineRG48toRGBA;
decoder_func_t vc_copylineUYVYtoRGB_SSE;
decoder_func_t vc_copylineUYVYtoGrayscale;
decoder_func_t vc_copylineRGBAtoUYVY;
decoder_func_t vc_copylineBGRtoRGB;
decoder_func_t vc_copylineRGB;
decoder_func_t vc_memcpy;

void vc_copylineToRGBA_inplace(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift);

bool clear_video_buffer(unsigned char *data, size_t linesize, size_t pitch, size_t height, codec_t color_spec);

#ifdef __cplusplus
}
#endif

#endif


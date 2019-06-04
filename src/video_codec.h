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
/* Copyright (c) 2005-2017 CESNET z.s.p.o.
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

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BPS 5 /* 4.5 for R12L */
#define MAX_PADDING 16

/**
 * @brief Defines type for pixelformat conversions
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
typedef void decoder_func_t(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift);
typedef decoder_func_t *decoder_t;

/// Prints list of suppored codecs for video module
void             show_codec_help(const char *module, const codec_t *codecs8, const codec_t *codecs10, const codec_t *codecs12);
/// @returns number of bits per color component
int              get_bits_per_component(codec_t codec) ATTRIBUTE(pure);
/// @returns number of bytes per pixel
double           get_bpp(codec_t codec) ATTRIBUTE(pure);
uint32_t         get_fourcc(codec_t codec) ATTRIBUTE(pure);
int              get_halign(codec_t codec) ATTRIBUTE(pure);
const char      *get_codec_name(codec_t codec) ATTRIBUTE(pure);
const char      *get_codec_name_long(codec_t codec) ATTRIBUTE(pure);
int              is_codec_opaque(codec_t codec) ATTRIBUTE(pure);
int              is_codec_interframe(codec_t codec) ATTRIBUTE(pure);
codec_t          get_codec_from_fcc(uint32_t fourcc) ATTRIBUTE(pure);
codec_t          get_codec_from_name(const char *name) ATTRIBUTE(pure);
const char      *get_codec_file_extension(codec_t codec) ATTRIBUTE(pure);
decoder_t        get_decoder_from_to(codec_t in, codec_t out, bool slow) ATTRIBUTE(pure);

int get_aligned_length(int width, codec_t codec) ATTRIBUTE(pure);
int get_pf_block_size(codec_t codec) ATTRIBUTE(pure);
int vc_get_linesize(unsigned int width, codec_t codec) ATTRIBUTE(pure);
int codec_is_a_rgb(codec_t codec) ATTRIBUTE(pure);
bool codec_is_in_set(codec_t codec, codec_t *set) ATTRIBUTE(pure);
int codec_is_const_size(codec_t codec) ATTRIBUTE(pure);
bool codec_is_hw_accelerated(codec_t codec) ATTRIBUTE(pure);

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
void vc_deinterlace_ex(unsigned char *src, size_t src_linesize, unsigned char *dst, size_t dst_pitch, size_t lines);

decoder_func_t vc_copylineDVS10;
decoder_func_t vc_copylinev210;
decoder_func_t vc_copylineYUYV;
decoder_func_t vc_copyliner10k;
decoder_func_t vc_copylineR12L;
decoder_func_t vc_copylineRGBA;
decoder_func_t vc_copylineToRGBA;
decoder_func_t vc_copylineDVS10toV210;
decoder_func_t vc_copylineRGBAtoRGB;
decoder_func_t vc_copylineABGRtoRGB;
decoder_func_t vc_copylineRGBAtoRGBwithShift;
decoder_func_t vc_copylineRGBtoRGBA;
decoder_func_t vc_copylineRGBtoUYVY;
decoder_func_t vc_copylineRGBtoUYVY_SSE;
decoder_func_t vc_copylineRGBtoGrayscale_SSE;
decoder_func_t vc_copylineRGBtoR12L;
decoder_func_t vc_copylineR12LtoRG48;
decoder_func_t vc_copylineRG48toR12L;
decoder_func_t vc_copylineRG48toRGBA;
decoder_func_t vc_copylineUYVYtoRGB;
decoder_func_t vc_copylineUYVYtoRGB_SSE;
decoder_func_t vc_copylineUYVYtoGrayscale;
decoder_func_t vc_copylineYUYVtoRGB;
decoder_func_t vc_copylineBGRtoUYVY;
decoder_func_t vc_copylineRGBAtoUYVY;
decoder_func_t vc_copylineBGRtoRGB;
decoder_func_t vc_copylineDPX10toRGBA;
decoder_func_t vc_copylineDPX10toRGB;
decoder_func_t vc_copylineRGB;

bool clear_video_buffer(unsigned char *data, size_t linesize, size_t pitch, size_t height, codec_t color_spec);

#ifdef __cplusplus
}
#endif

#endif


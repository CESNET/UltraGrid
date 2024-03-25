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
 *
 */
#ifndef __video_codec_h
#define __video_codec_h

#include "types.h" // codec_t

#ifndef __cplusplus
#include <stdbool.h>
#endif // !defined __cplusplus

#include "pixfmt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BPS 8 /* for Y416 */  ///< maximal (average) number of pixels per know pixel formats (up-round if needed)
#define MAX_PADDING 64 ///< maximal padding in bytes that may be needed to align to pixfmt block size for Y416->R12L:
                       ///< 64 = vc_linesize(8 /*maximal block pix count (R12L)*/, Y416 /*codec with maximal lenght of 1st arg-sized block*/)
#define MAX_PFB_SIZE   8 ///< maximal pixfmt block size (R12L - 8 B)
#define PIX_BLOCK_LCM 24 ///< least common multiple of all pixfmt block sizes in pixels (codec_info_t::block_size/codec_info_t::bpp). "contributors:" 8 R12L, 6 v210

/// Prints list of suppored codecs for video module
void             show_codec_help(const char *module, const codec_t *codecs8, const codec_t *codecs10, const codec_t *codecs_ge12);
/// @returns number of bits per color component
int              get_bits_per_component(codec_t codec) __attribute__((const));
int              get_subsampling(codec_t codec) __attribute__((const));
/// @returns number of bytes per pixel
double           get_bpp(codec_t codec) __attribute__((const));
uint32_t         get_fourcc(codec_t codec) __attribute__((const));
const char      *get_codec_name(codec_t codec) __attribute__((const));
const char      *get_codec_name_long(codec_t codec) __attribute__((const));
bool             is_codec_opaque(codec_t codec) __attribute__((const));
bool             is_codec_interframe(codec_t codec) __attribute__((const));
codec_t          get_codec_from_fcc(uint32_t fourcc) __attribute__((const));
codec_t          get_codec_from_name(const char *name) __attribute__((const));
const char      *get_codec_file_extension(codec_t codec) __attribute__((const));
codec_t          get_codec_from_file_extension(const char *ext) __attribute__((const));

struct pixfmt_desc get_pixfmt_desc(codec_t pixfmt);
int              compare_pixdesc(const struct pixfmt_desc *desc_a, const struct pixfmt_desc *desc_b, const struct pixfmt_desc *src_desc);
const char      *get_pixdesc_desc(struct pixfmt_desc);
void             watch_pixfmt_degrade(const char *mod_name, struct pixfmt_desc desc_src, struct pixfmt_desc desc_dst);
bool             pixdesc_equals(struct pixfmt_desc desc_a, struct pixfmt_desc desc_b);

int get_pf_block_bytes(codec_t codec) __attribute__((const));
int get_pf_block_pixels(codec_t codec) __attribute__((const));
int vc_get_linesize(unsigned int width, codec_t codec) __attribute__((const));
int vc_get_size(unsigned int width, codec_t codec) __attribute__((const));
size_t vc_get_datalen(unsigned int width, unsigned int height, codec_t codec) __attribute__((const));
void codec_get_planes_subsampling(codec_t pix_fmt, int *sub);
bool codec_is_420(codec_t pix_fmt);
bool codec_is_a_rgb(codec_t codec) __attribute__((const));
bool codec_is_in_set(codec_t codec, const codec_t *set) __attribute__((pure));
bool codec_is_const_size(codec_t codec) __attribute__((const));
bool codec_is_hw_accelerated(codec_t codec) __attribute__((const));
bool codec_is_planar(codec_t codec) __attribute__((const));

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
bool vc_deinterlace_ex(codec_t codec, unsigned char *src, size_t src_linesize, unsigned char *dst, size_t dst_pitch, size_t lines);

bool clear_video_buffer(unsigned char *data, size_t linesize, size_t pitch, size_t height, codec_t color_spec);

// conversions from/to planar formats
void uyvy_to_i422(int width, int height, const unsigned char *in,
                  unsigned char *out);
void y416_to_i444(int width, int height, const unsigned char *in,
                  unsigned char *out, int depth);
void i444_16_to_y416(int width, int height, const unsigned char *in,
                     unsigned char *out, int in_depth);
void i422_16_to_y416(int width, int height, const unsigned char *in,
                     unsigned char *out, int in_depth);
void i420_16_to_y416(int width, int height, const unsigned char *in,
                     unsigned char *out, int in_depth);
void i420_8_to_uyvy(int width, int height, const unsigned char *in,
                    unsigned char *out);
void i422_8_to_uyvy(int width, int height, const unsigned char *in,
                    unsigned char *out);
void i444_8_to_uyvy(int width, int height, const unsigned char *in,
                    unsigned char *out);

#ifdef __cplusplus
}
#endif

#endif


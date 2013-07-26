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
/* Copyright (c) 2005-2013 CESNET z.s.p.o.
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

/**
 * @brief Defines type for pixelformat conversions
 * @param[out] dst     destination buffer
 * @param[in]  src     source buffer
 * @param[in]  dst_len expected number of bytes to be written
 * @param[in]  rshift  offset of red field inside a word (in bits)
 * @param[in]  gshift  offset of green field inside a word (in bits)
 * @param[in]  bshift  offset of blue field inside a word (in bits)
 */
typedef void (*decoder_t)(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift);

/**
 * Defines codec metadata
 * @note
 * Members that are not relevant for specified codec (eg. bpp, rgb for opaque
 * and interframe for not opaque) should be zero.
 * @todo This should be perhaps private in .c file and properties should be
 * queried by functions.
 */
struct codec_info_t {
        codec_t codec;                ///< codec descriptor
        const char *name;             ///< displayed name
        uint32_t fcc;                 ///< FourCC
        int h_align;                  ///< Number of pixels each line is aligned to
        double bpp;                   ///< Number of bytes per pixel
        unsigned rgb:1;               ///< Whether pixelformat is RGB
        unsigned opaque:1;            ///< If codec is opaque (= compressed)
        unsigned interframe:1;        ///< Indicates if compression is interframe
        const char *file_extension;   ///< Extension that should be added to name if frame is saved to file.
};

/** Defines decoder from one pixel format to another */
struct line_decode_from_to {
        codec_t from;           ///< source pixel format
        codec_t to;             ///< destination pixel format
        decoder_t line_decoder; ///< decoding function
};

/** @brief 0-terminated list of UltraGrid supported codecs' metadata */
extern const struct codec_info_t codec_info[];           /* defined int .c */
/** @brief 0-terminated list of available supported pixelformat decoders */
extern const struct line_decode_from_to line_decoders[]; /* defined int .c */

/** Prints list of suppored codecs for video module
 * @deprecated Individual modules should print list of supported codecs by itself.
 */
void             show_codec_help(char *module);
double           get_bpp(codec_t codec);
uint32_t         get_fourcc(codec_t codec);
const char      *get_codec_name(codec_t codec);
int              is_codec_opaque(codec_t codec);
int              is_codec_interframe(codec_t codec);
codec_t          get_codec_from_fcc(uint32_t fourcc);
const char      *get_codec_file_extension(codec_t codec);

uint32_t get_fcc_from_codec(codec_t codec);
int get_aligned_length(int width, codec_t codec);
int vc_get_linesize(unsigned int width, codec_t codec);

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
void vc_copylineDVS10(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylinev210(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineYUYV(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copyliner10k(unsigned char *dst, const unsigned char *src, int len,
                int rshift, int gshift, int bshift);
void vc_copylineRGBA(unsigned char *dst, const unsigned char *src, int len,
                int rshift, int gshift, int bshift);
void vc_copylineDVS10toV210(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineRGBAtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineABGRtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineRGBAtoRGBwithShift(unsigned char *dst, const unsigned char *src, int len,
                int rshift, int gshift, int bshift);
void vc_copylineRGBtoRGBA(unsigned char *dst, const unsigned char *src, int len,
                int rshift, int gshift, int bshift);
void vc_copylineRGBtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineUYVYtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineBGRtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineRGBAtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineBGRtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineDPX10toRGBA(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift);
void vc_copylineDPX10toRGB(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineRGB(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift);

int codec_is_a_rgb(codec_t codec);

#ifdef __cplusplus
}
#endif

#endif


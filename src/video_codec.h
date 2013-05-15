/*
 * FILE:    video_codec.h
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef  void (*decoder_t)(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift);

struct codec_info_t {
        codec_t codec;
        const char *name;
        uint32_t fcc;
        int h_align;
        double bpp;
        unsigned rgb:1;
        unsigned opaque:1;
        unsigned interframe:1;
        const char *file_extension;
};

struct line_decode_from_to {
        codec_t from;
        codec_t to;
        decoder_t line_decoder;
};

extern const struct codec_info_t codec_info[];           /* defined int .c */
extern const struct line_decode_from_to line_decoders[]; /* defined int .c */

void             show_codec_help(char *mode);
double           get_bpp(codec_t codec);
uint32_t         get_fourcc(codec_t codec);
const char      *get_codec_name(codec_t codec);
int              is_codec_opaque(codec_t codec);
int              is_codec_interframe(codec_t codec);
codec_t          get_codec_from_fcc(uint32_t fourcc);
const char      *get_codec_file_extension(codec_t codec);

/**
 * Returns FCC for codec
 *
 * @param  codec        input codec
 * @return              respective FourCC
 *                      0 if not found in database
 */
uint32_t get_fcc_from_codec(codec_t codec);
int get_haligned(int width_pixels, codec_t codec);

int vc_get_linesize(unsigned int width, codec_t codec);

void vc_deinterlace(unsigned char *src, long src_linesize, int lines);
void vc_copylineDVS10(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylinev210(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineYUYV(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copyliner10k(unsigned char *dst, const unsigned char *src, int len, int rshift, int gshift, int bshift);
void vc_copylineRGBA(unsigned char *dst, const unsigned char *src, int len, int rshift, int gshift, int bshift);
void vc_copylineDVS10toV210(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineRGBAtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineABGRtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineRGBAtoRGBwithShift(unsigned char *dst, const unsigned char *src, int len, int rshift, int gshift, int bshift);
void vc_copylineRGBtoRGBA(unsigned char *dst, const unsigned char *src, int len, int rshift, int gshift, int bshift);
void vc_copylineRGBtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineUYVYtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineBGRtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineRGBAtoUYVY(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineBGRtoRGB(unsigned char *dst, const unsigned char *src, int len);
void vc_copylineDPX10toRGBA(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift);
void vc_copylineDPX10toRGB(unsigned char *dst, const unsigned char *src, int dst_len);
void vc_copylineRGB(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift);
/*
 * @return TRUE or FALSE
 */
int codec_is_a_rgb(codec_t codec);

#ifdef __cplusplus
}
#endif

#endif


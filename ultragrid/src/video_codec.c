/*
 * FILE:    video_codec.c
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "video_codec.h"

static int get_halign(codec_t codec);

const struct codec_info_t codec_info[] = {
        {RGBA, "RGBA", 0, 1, 4.0, 1},
        {UYVY, "UYVY", 846624121, 1, 2, 0},
        {Vuy2, "2vuy", '2Vuy', 1, 2, 0},
        {DVS8, "DVS8", 0, 1, 2, 0},
        {R10k, "R10k", 1378955371, 1, 4, 1},
        {v210, "v210", 1983000880, 48, 8.0 / 3.0, 0},
        {DVS10, "DVS10", 0, 48, 8.0 / 3.0, 0},
        {DXT1, "DXT1", 'DXT1', 1, 0.5, 1},
        {DXT1_YUV, "DXT1 YUV", 'DXTY', 1, 0.5, 0}, /* packet YCbCr inside DXT1 channels */
        {DXT5, "DXT5", 'DXT5', 1, 1.0, 1},/* DXT5 YCoCg */
        {RGB, "RGB", 0x32424752, 1, 3.0, 1},
        {DPX10, "DPX10", 0, 1, 4.0, 1},
        {JPEG, "JPEG", 'JPEG', 0, 0.0, 0},
        {0, NULL, 0, 0, 0.0, 0}
};

/* take care that UYVY is alias for both 2vuy and dvs8, do not use
 * the further two and refer only to UYVY!! */
 
/* Also note that this is a priority list - is choosen first one that
 * matches input codec and one of the supported output codec, so eg.
 * list 10b->10b earlier to 10b->8b etc. */
const struct line_decode_from_to line_decoders[] = {
        { RGBA, RGBA, vc_copylineRGBA},
        { RGB, RGB, vc_copylineRGB},
        { DVS10, v210, vc_copylineDVS10toV210},
        { DVS10, UYVY, vc_copylineDVS10},
        { R10k, RGBA, vc_copyliner10k},
        { v210, UYVY, vc_copylinev210},
        { RGBA, RGB, vc_copylineRGBAtoRGB},
        { RGB, RGBA, vc_copylineRGBtoRGBA},
        { DPX10, RGBA, vc_copylineDPX10toRGBA},
        { 0, 0, NULL }
};

void show_codec_help(char *module)
{
        printf("\tSupported codecs (%s):\n", module);

        printf("\t\t8bits\n");

        printf("\t\t\t'RGBA' - Red Green Blue Alpha 32bit\n");
        printf("\t\t\t'RGB' - Red Green Blue 24bit\n");
        printf("\t\t\t'UYVY' - YUV 4:2:2\n");
	printf("\t\t\t'2vuy' - YUV 4:2:2\n");
        printf("\t\t\t'DVS8' - Centaurus 8bit YUV 4:2:2\n");

        printf("\t\t10bits\n");
	if (strcmp(module, "dvs") != 0) {
		printf("\t\t\t'R10k' - RGB 4:4:4\n");
		printf("\t\t\t'v210' - YUV 4:2:2\n");
	} 
        printf("\t\t\t'DVS10' - Centaurus 10bit YUV 4:2:2\n");
}

double get_bpp(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].bpp;
                i++;
        }
        return 0;
}

static int get_halign(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].h_align;
                i++;
        }
        return 0;
}

int get_haligned(int width_pixels, codec_t codec)
{
        int h_align = get_halign(codec);
        return ((width_pixels + h_align - 1) / h_align) * h_align;
}

int vc_get_linesize(unsigned int width, codec_t codec)
{
        if (codec_info[codec].h_align) {
                width =
                    ((width + codec_info[codec].h_align -
                      1) / codec_info[codec].h_align) *
                    codec_info[codec].h_align;
        }
        return width * codec_info[codec].bpp;
}

/* linear blend deinterlace */
void vc_deinterlace(unsigned char *src, long src_linesize, int lines)
{
        int i, j;
        long pitch = src_linesize;
        register long pitch2 = pitch * 2;
        unsigned char *bline1, *bline2, *bline3;
        register unsigned char *line1, *line2, *line3;

        bline1 = src;
        bline2 = src + pitch;
        bline3 = src + 3 * pitch;
        for (i = 0; i < src_linesize; i += 16) {
                /* preload first two lines */
                asm volatile ("movdqa (%0), %%xmm0\n"
                              "movdqa (%1), %%xmm1\n"::"r" ((unsigned long *)
                                                            bline1),
                              "r"((unsigned long *)bline2));
                line1 = bline2;
                line2 = bline2 + pitch;
                line3 = bline3;
                for (j = 0; j < lines - 4; j += 2) {
                        asm volatile ("movdqa (%1), %%xmm2\n"
                                      "pavgb %%xmm2, %%xmm0\n"
                                      "pavgb %%xmm1, %%xmm0\n"
                                      "movdqa (%2), %%xmm1\n"
                                      "movdqa %%xmm0, (%0)\n"
                                      "pavgb %%xmm1, %%xmm0\n"
                                      "pavgb %%xmm2, %%xmm0\n"
                                      "movdqa %%xmm0, (%1)\n"::"r" ((unsigned
                                                                     long *)
                                                                    line1),
                                      "r"((unsigned long *)line2),
                                      "r"((unsigned long *)line3)
                            );
                        line1 += pitch2;
                        line2 += pitch2;
                        line3 += pitch2;
                }
                bline1 += 16;
                bline2 += 16;
                bline3 += 16;
        }
}

void vc_copylinev210(unsigned char *dst, unsigned char *src, int dst_len)
{
        struct {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } *s;
        register uint32_t *d;
        register uint32_t tmp;

        d = (uint32_t *) dst;
        s = (void *)src;

        while (dst_len >= 12) {
                tmp = (s->a >> 2) | (s->b >> 2) << 8 | (((s)->c >> 2) << 16);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 24);
                tmp = (s->b >> 2) | (((s)->c >> 2) << 8);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 16) | ((s->b >> 2) << 24);
                tmp = (s->c >> 2);
                s++;
                *(d++) =
                    tmp | ((s->a >> 2) << 8) | ((s->b >> 2) << 16) |
                    ((s->c >> 2) << 24);
                s++;

                dst_len -= 12;
        }
        if (dst_len >= 4) {
                tmp = (s->a >> 2) | (s->b >> 2) << 8 | (((s)->c >> 2) << 16);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 24);
        }
        if (dst_len >= 8) {
                tmp = (s->b >> 2) | (((s)->c >> 2) << 8);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 16) | ((s->b >> 2) << 24);
        }
}

void
vc_copyliner10k(unsigned char *dst, unsigned char *src, int len, int rshift,
                int gshift, int bshift)
{
        struct {
                unsigned r:8;

                unsigned gh:6;
                unsigned p1:2;

                unsigned bh:4;
                unsigned p2:2;
                unsigned gl:2;

                unsigned p3:2;
                unsigned p4:2;
                unsigned bl:4;
        } *s;
        register uint32_t *d;
        register uint32_t tmp;

        d = (uint32_t *) dst;
        s = (void *)src;

        while (len > 0) {
                tmp =
                    (s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                len -= 16;
        }
}

void
vc_copylineRGBA(unsigned char *dst, unsigned char *src, int len, int rshift,
                int gshift, int bshift)
{
        register uint32_t *d = (uint32_t *) dst;
        register uint32_t *s = (uint32_t *) src;
        register uint32_t tmp;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, len);
        } else {
                while (len > 0) {
                        register unsigned int r, g, b;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        len -= 16;
                }
        }
}

void vc_copylineDVS10toV210(unsigned char *dst, unsigned char *src, int dst_len)
{
        unsigned int *d, *s1;
        register unsigned int a,b;
        d = dst;
        s1 = src;

        while(dst_len > 0) {
                a = b = *s1++;
                b = ((b >> 24) * 0x00010101) & 0x00300c03;
                a <<= 2;
                b |= a & (0xff<<2);
                a <<= 2;
                b |= a & (0xff00<<4);
                a <<= 2;
                b |= a & (0xff0000<<6);
                *d++ = b;
                dst_len -= 4;
        }
}

/* convert 10bits Cb Y Cr A Y Cb Y A to 8bits Cb Y Cr Y Cb Y */

/* TODO: undo it - currently this decoder is broken */
#if 0 /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

void vc_copylineDVS10(unsigned char *dst, unsigned char *src, int src_len)
{
        register unsigned char *_d = dst, *_s = src;

        while (src_len > 0) {

 asm("movd %0, %%xmm4\n": :"r"(0xffffff));

                asm volatile ("movdqa (%0), %%xmm0\n"
                              "movdqa 16(%0), %%xmm5\n"
                              "movdqa %%xmm0, %%xmm1\n"
                              "movdqa %%xmm0, %%xmm2\n"
                              "movdqa %%xmm0, %%xmm3\n"
                              "pand  %%xmm4, %%xmm0\n"
                              "movdqa %%xmm5, %%xmm6\n"
                              "movdqa %%xmm5, %%xmm7\n"
                              "movdqa %%xmm5, %%xmm8\n"
                              "pand  %%xmm4, %%xmm5\n"
                              "pslldq $4, %%xmm4\n"
                              "pand  %%xmm4, %%xmm1\n"
                              "pand  %%xmm4, %%xmm6\n"
                              "pslldq $4, %%xmm4\n"
                              "psrldq $1, %%xmm1\n"
                              "psrldq $1, %%xmm6\n"
                              "pand  %%xmm4, %%xmm2\n"
                              "pand  %%xmm4, %%xmm7\n"
                              "pslldq $4, %%xmm4\n"
                              "psrldq $2, %%xmm2\n"
                              "psrldq $2, %%xmm7\n"
                              "pand  %%xmm4, %%xmm3\n"
                              "pand  %%xmm4, %%xmm8\n"
                              "por %%xmm1, %%xmm0\n"
                              "psrldq $3, %%xmm3\n"
                              "psrldq $3, %%xmm8\n"
                              "por %%xmm2, %%xmm0\n"
                              "por %%xmm6, %%xmm5\n"
                              "por %%xmm3, %%xmm0\n"
                              "por %%xmm7, %%xmm5\n"
                              "movdq2q %%xmm0, %%mm0\n"
                              "por %%xmm8, %%xmm5\n"
                              "movdqa %%xmm5, %%xmm1\n"
                              "pslldq $12, %%xmm5\n"
                              "psrldq $4, %%xmm1\n"
                              "por %%xmm5, %%xmm0\n"
                              "psrldq $8, %%xmm0\n"
                              "movq %%mm0, (%1)\n"
                              "movdq2q %%xmm0, %%mm1\n"
                              "movdq2q %%xmm1, %%mm2\n"
                              "movq %%mm1, 8(%1)\n"
                              "movq %%mm2, 16(%1)\n"::"r" (_s), "r"(_d));
                _s += 32;
                _d += 24;
                src_len -= 32;
        }
}

#else

void vc_copylineDVS10(unsigned char *dst, unsigned char *src, int dst_len)
{
        int src_len = dst_len / 1.5; /* right units */
        register uint64_t *d, *s;

        register uint64_t a1, a2, a3, a4;

        d = (uint64_t *) dst;
        s = (uint64_t *) src;

        while (src_len > 0) {
                a1 = *(s++);
                a2 = *(s++);
                a3 = *(s++);
                a4 = *(s++);

                a1 = (a1 & 0xffffff) | ((a1 >> 8) & 0xffffff000000LL);
                a2 = (a2 & 0xffffff) | ((a2 >> 8) & 0xffffff000000LL);
                a3 = (a3 & 0xffffff) | ((a3 >> 8) & 0xffffff000000LL);
                a4 = (a4 & 0xffffff) | ((a4 >> 8) & 0xffffff000000LL);

                *(d++) = a1 | (a2 << 48);       /* 0xa2|a2|a1|a1|a1|a1|a1|a1 */
                *(d++) = (a2 >> 16) | (a3 << 32);       /* 0xa3|a3|a3|a3|a2|a2|a2|a2 */
                *(d++) = (a3 >> 32) | (a4 << 16);       /* 0xa4|a4|a4|a4|a4|a4|a3|a3 */

                src_len -= 16;
        }
}

#endif                          /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

void vc_copylineRGB(unsigned char *dst, unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        register unsigned int r, g, b;
        union {
                unsigned int out;
                unsigned char c[4];
        } u;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, dst_len);
        } else {
                while(dst_len > 0) {
                        r = *src++;
                        g = *src++;
                        b = *src++;
                        u.out = (r << rshift) | (g << gshift) | (b << bshift);
                        *dst++ = u.c[0];
                        *dst++ = u.c[1];
                        *dst++ = u.c[2];
                        dst_len -= 3;
                }
        }
}

void vc_copylineRGBAtoRGB(unsigned char *dst, unsigned char *src, int dst_len)
{
        while(dst_len > 0) {
                *dst++ = *src++;
                *dst++ = *src++;
                *dst++ = *src++;
                src++;
                dst_len -= 3;
        }
}

void vc_copylineRGBtoRGBA(unsigned char *dst, unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        register uint32_t tmp;
        register unsigned int r, g, b;
        register uint32_t *d = (uint32_t *) dst;
        
        while(dst_len > 0) {
                r = *src++;
                g = *src++;
                b = *src++;
                
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                dst_len -= 4;
        }
}

void
vc_copylineDPX10toRGBA(unsigned char *dst, unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        
        register unsigned int *in = src;
        register unsigned int *out = dst;
        register int r,g,b;
       
        while(dst_len > 0) {
                register unsigned int val = *in;
                r = val >> 24;
                g = 0xff & (val >> 14);
                b = 0xff & (val >> 4);
                
                *out++ = (r << rshift) | (g << gshift) | (b << bshift);
                ++in;
                dst_len -= 4;
        }
}

void
vc_copylineDPX10toRGB(unsigned char *dst, unsigned char *src, int dst_len)
{
        
        register unsigned int *in = src;
        register unsigned int *out = dst;
        register int r1,g1,b1,r2,g2,b2;
       
        while(dst_len > 0) {
                register unsigned int val;
                
                val = *in++;
                r1 = val >> 24;
                g1 = 0xff & (val >> 14);
                b1 = 0xff & (val >> 4);
                
                val = *in++;
                r2 = val >> 24;
                g2 = 0xff & (val >> 14);
                b2 = 0xff & (val >> 4);
                
                *out++ = r1 | g1 << 8 | b1 << 16 | r2 << 24;
                
                val = *in++;
                r1 = val >> 24;
                g1 = 0xff & (val >> 14);
                b1 = 0xff & (val >> 4);
                
                *out++ = g2 | b2 << 8 | r1 << 16 | g1 << 24;
                
                val = *in++;
                r2 = val >> 24;
                g2 = 0xff & (val >> 14);
                b2 = 0xff & (val >> 4);
                
                *out++ = b1 | r2 << 8 | g2 << 16 | b2 << 24;
                
                dst_len -= 12;
        }
}


int codec_is_a_rgb(codec_t codec)
{
        int i;
        
        for (i = 0; codec_info[i].name != NULL; i++) {
		if (codec == codec_info[i].codec) {
			return codec_info[i].rgb;
		}
	}
        error_with_code_msg(128, "Unknown codec !");
}

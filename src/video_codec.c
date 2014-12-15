/**
 * @file   video_codec.c
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @brief This file contains video codec-related functions.
 *
 * This file contains video codecs' metadata and helper
 * functions as well as pixelformat converting functions.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"

#include <stdio.h>
#include <string.h>
#include "video_codec.h"

/**
 * @brief Creates FourCC word
 *
 * The main idea of FourCC is that it can be anytime read by human (by hexa editor, gdb, tcpdump).
 * Therefore, this is stored as a big endian even on little-endian architectures - first byte
 * of FourCC is in the memory on the lowest address.
 */
#ifdef WORDS_BIGENDIAN
#define to_fourcc(a,b,c,d)     (((uint32_t)(d)) | ((uint32_t)(c)<<8) | ((uint32_t)(b)<<16) | ((uint32_t)(a)<<24))
#else
#define to_fourcc(a,b,c,d)     (((uint32_t)(a)) | ((uint32_t)(b)<<8) | ((uint32_t)(c)<<16) | ((uint32_t)(d)<<24))
#endif

#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))

static void vc_deinterlace_aligned(unsigned char *src, long src_linesize, int lines);
static void vc_deinterlace_unaligned(unsigned char *src, long src_linesize, int lines);
static void vc_copylineToUYVY709(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) __attribute__((unused));
static void vc_copylineToUYVY601(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) __attribute__((unused));

/**
 * Defines codec metadata
 * @note
 * Members that are not relevant for specified codec (eg. bpp, rgb for opaque
 * and interframe for not opaque) should be zero.
 */
struct codec_info_t {
        codec_t codec;                ///< codec descriptor
        const char *name;             ///< displayed name
        uint32_t fcc;                 ///< FourCC
        int h_align;                  ///< Number of pixels each line is aligned to
        double bpp;                   ///< Number of bytes per pixel
        int block_size;               ///< Bytes per pixel block (pixelformats only)
        unsigned rgb:1;               ///< Whether pixelformat is RGB
        unsigned opaque:1;            ///< If codec is opaque (= compressed)
        unsigned interframe:1;        ///< Indicates if compression is interframe
        const char *file_extension;   ///< Extension that should be added to name if frame is saved to file.
};

static const struct codec_info_t codec_info[] = {
        [VIDEO_CODEC_NONE] = {VIDEO_CODEC_NONE, "(none)", 0, 0, 0.0, 0, FALSE, FALSE, FALSE, NULL},
        [RGBA] = {RGBA, "RGBA", to_fourcc('R','G','B','A'), 1, 4.0, 4, TRUE, FALSE, FALSE, "rgba"},
        [UYVY] = {UYVY, "UYVY", to_fourcc('U','Y','V','Y'), 2, 2, 4, FALSE, FALSE, FALSE, "yuv"},
        [YUYV] = {YUYV, "YUYV", to_fourcc('Y','U','Y','V'), 2, 2, 4, FALSE, FALSE, FALSE, "yuv"},
        [R10k] = {R10k, "R10k", to_fourcc('R','1','0','k'), 64, 4, 4, TRUE, FALSE, FALSE, "r10k"},
        [v210] = {v210, "v210", to_fourcc('v','2','1','0'), 48, 8.0 / 3.0, 16, FALSE, FALSE, FALSE, "v210"},
        [DVS10] = {DVS10, "DVS10", to_fourcc('D','S','1','0'), 48, 8.0 / 3.0, 4, FALSE, FALSE, FALSE, "dvs10"},
        [DXT1] = {DXT1, "DXT1", to_fourcc('D','X','T','1'), 0, 0.5, 0, TRUE, TRUE, FALSE, "dxt1"},
        [DXT1_YUV] = {DXT1_YUV, "DXT1 YUV", to_fourcc('D','X','T','Y'), 0, 0.5, 0, FALSE, TRUE, FALSE, "dxt1y"}, /* packet YCbCr inside DXT1 channels */
        [DXT5] = {DXT5, "DXT5", to_fourcc('D','X','T','5'), 0, 1.0, 0, FALSE, TRUE, FALSE, "yog"},/* DXT5 YCoCg */
        [RGB] = {RGB, "RGB", to_fourcc('R','G','B','2'), 1, 3.0, 3, TRUE, FALSE, FALSE, "rgb"},
        [DPX10] = {DPX10, "DPX10", to_fourcc('D','P','1','0'), 1, 4.0, 4, TRUE, FALSE, FALSE, "dpx"},
        [JPEG] = {JPEG, "JPEG", to_fourcc('J','P','E','G'), 0, 0.0, 0, FALSE, TRUE, FALSE, "jpg"},
        [RAW] = {RAW, "raw", to_fourcc('r','a','w','s'), 0, 1.0, 0, FALSE, TRUE, FALSE, "raw"}, /* raw SDI */
        [H264] = {H264, "H.264", to_fourcc('A','V','C','1'), 0, 1.0, 0, FALSE, TRUE, TRUE, "h264"},
        [MJPG] = {MJPG, "MJPEG", to_fourcc('M','J','P','G'), 0, 1.0, 0, FALSE, TRUE, FALSE, "jpg"},
        [VP8] = {VP8, "VP8", to_fourcc('V','P','8','0'), 0, 1.0, 0, FALSE, TRUE, TRUE, "vp8"},
        [BGR] = {BGR, "BGR", to_fourcc('B','G','R','2'), 1, 3.0, 0, TRUE, FALSE, FALSE, "bgr"},
        [J2K] = {J2K, "J2K", to_fourcc('M','J','2','C'), 0, 0.0, 0, FALSE, TRUE, FALSE, "j2k"},
        {(codec_t) 0, NULL, 0, 0, 0.0, 0, FALSE, FALSE, FALSE, NULL}
};

/* Also note that this is a priority list - is choosen first one that
 * matches input codec and one of the supported output codec, so eg.
 * list 10b->10b earlier to 10b->8b etc. */
const struct line_decode_from_to line_decoders[] = {
        { RGBA, RGBA, vc_copylineRGBA},
        { RGB, RGB, vc_copylineRGB},
        { DVS10, v210, (decoder_t) vc_copylineDVS10toV210},
        { DVS10, UYVY, (decoder_t) vc_copylineDVS10},
        { R10k, RGBA, vc_copyliner10k},
        { v210, UYVY, (decoder_t) vc_copylinev210},
        { YUYV, UYVY, (decoder_t) vc_copylineYUYV},
        { RGBA, RGB, (decoder_t) vc_copylineRGBAtoRGB},
        { RGB, RGBA, vc_copylineRGBtoRGBA},
        { DPX10, RGBA, vc_copylineDPX10toRGBA},
        { DPX10, RGB, (decoder_t) vc_copylineDPX10toRGB},
        { RGB, UYVY, (decoder_t) vc_copylineRGBtoUYVY},
        { BGR, RGB, (decoder_t) vc_copylineBGRtoRGB},
        { (codec_t) 0, (codec_t) 0, NULL }
};

/**
 * This struct specifies alias FourCC used for another FourCC
 */
struct alternative_fourcc {
        uint32_t alias;
        uint32_t primary_fcc;
};

/**
 * This array contains FourCC aliases mapping
 */
const struct alternative_fourcc fourcc_aliases[] = {
        // the following two are here because it was sent with wrong endiannes in past
        {to_fourcc('A', 'B', 'G', 'R'), to_fourcc('R', 'G', 'B', 'A')},
        {to_fourcc('2', 'B', 'G', 'R'), to_fourcc('R', 'G', 'B', '2')},
        // following ones are rather for further compatibility (proposed codecs rename)
        {to_fourcc('M', 'J', 'P', 'G'), to_fourcc('J', 'P', 'E', 'G')},

        {to_fourcc('2', 'V', 'u', 'y'), to_fourcc('U', 'Y', 'V', 'Y')},
        {to_fourcc('2', 'v', 'u', 'y'), to_fourcc('U', 'Y', 'V', 'Y')},
        {to_fourcc('d', 'v', 's', '8'), to_fourcc('U', 'Y', 'V', 'Y')},
        {to_fourcc('D', 'V', 'S', '8'), to_fourcc('U', 'Y', 'V', 'Y')},
        {to_fourcc('y', 'u', 'v', '2'), to_fourcc('U', 'Y', 'V', 'Y')},
        {to_fourcc('y', 'u', 'V', '2'), to_fourcc('U', 'Y', 'V', 'Y')},
};

struct alternative_codec_name {
        const char *alias;
        const char *primary_name;
};

const struct alternative_codec_name codec_name_aliases[] = {
        {"2vuy", "UYVY"},
};

void show_codec_help(char *module)
{
        printf("\tSupported codecs (%s):\n", module);

        printf("\t\t8bits\n");

        printf("\t\t\t'RGBA' - Red Green Blue Alpha 32bit\n");
        printf("\t\t\t'RGB' - Red Green Blue 24bit\n");
        printf("\t\t\t'UYVY' - YUV 4:2:2\n");

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

uint32_t get_fourcc(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].fcc;
                i++;
        }
        return 0;
}

const char * get_codec_name(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].name;
                i++;
        }
        return 0;
}

/** @brief Returns FourCC for specified codec. */
uint32_t get_fcc_from_codec(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].fcc;
                i++;
        }

        return 0;
}

codec_t get_codec_from_fcc(uint32_t fourcc)
{
        int i = 0;
        while (codec_info[i].name != NULL) {
                if (fourcc == codec_info[i].fcc)
                        return codec_info[i].codec;
                i++;
        }

        // try to look through aliases
        for (size_t i = 0; i < sizeof(fourcc_aliases) / sizeof(struct alternative_fourcc); ++i) {
                if (fourcc == fourcc_aliases[i].alias) {
                        int j = 0;
                        while (codec_info[j].name != NULL) {
                                if (fourcc_aliases[i].primary_fcc == codec_info[j].fcc)
                                        return codec_info[j].codec;
                                j++;
                        }
                }
        }
        return VIDEO_CODEC_NONE;
}

/**
 * Helper codec finding function
 *
 * Iterates through codec list and finds appropriate codec.
 *
 * @returns codec
 */
static codec_t get_codec_from_name_wo_alias(const char *name)
{
        for (int i = 0; codec_info[i].name != NULL; i++) {
                if (strcmp(codec_info[i].name, name) == 0) {
                        return codec_info[i].codec;
                }
        }

        return VIDEO_CODEC_NONE;
}

codec_t get_codec_from_name(const char *name)
{
        codec_t ret = get_codec_from_name_wo_alias(name);
        if (ret != VIDEO_CODEC_NONE) {
                return ret;
        }

        // try to find if this is not an alias
        for (size_t i = 0; i < sizeof(codec_name_aliases) / sizeof(struct alternative_codec_name); ++i) {
                if (strcmp(name, codec_name_aliases[i].alias) == 0) {
                        ret = get_codec_from_name_wo_alias(name);
                        if (ret != VIDEO_CODEC_NONE) {
                                return ret;
                        }
                }
        }
        return VIDEO_CODEC_NONE;
}

const char *get_codec_file_extension(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].file_extension;
                i++;
        }

        return 0;
}

/**
 * @retval TRUE if codec is compressed
 * @retval FALSE if codec is pixelformat
 */
int is_codec_opaque(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].opaque;
                i++;
        }
        return 0;
}

/**
 * Returns whether specified codec is an interframe compression.
 * Not defined for pixelformats
 * @retval TRUE if compression is interframe
 * @retval FALSE if compression is not interframe
 */
int is_codec_interframe(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].interframe;
                i++;
        }
        return 0;
}

int get_halign(codec_t codec)
{
        int i = 0;

        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].h_align;
                i++;
        }
       return 0;
}

/** @brief Returns aligned linesize according to pixelformat specification (in pixels) */
int get_aligned_length(int width_pixels, codec_t codec)
{
        int h_align = get_halign(codec);
        assert(h_align > 0);
        return ((width_pixels + h_align - 1) / h_align) * h_align;
}

/** @brief Returns aligned linesize according to pixelformat specification (in bytes) */
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

/// @brief returns @ref codec_info_t::block_size
int get_pf_block_size(codec_t codec)
{
        int i = 0;
        while (codec_info[i].name != NULL) {
                if (codec == codec_info[i].codec)
                        return codec_info[i].block_size;
                i++;
        }
       return 0;
}

/** @brief Deinterlaces framebuffer.
 *
 * vc_deinterlace performs linear blend deinterlace on a framebuffer.
 * @param[in,out] src          framebuffer to be deinterlaced
 * @param[in]     src_linesize length of a line (bytes)
 * @param[in]     lines        number of lines
 * @see vc_deinterlace_aligned
 * @see vc_deinterlace_unaligned
 */
void vc_deinterlace(unsigned char *src, long src_linesize, int lines)
{
        if(((long int) src & 0x0F) == 0 && src_linesize % 16 == 0) {
                vc_deinterlace_aligned(src, src_linesize, lines);
        } else {
                vc_deinterlace_unaligned(src, src_linesize, lines);
        }
}

/**
 * Aligned version of deinterlace filter
 *
 * @param src 16-byte aligned buffer
 * @see vc_deinterlace
 */
static void vc_deinterlace_aligned(unsigned char *src, long src_linesize, int lines)
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
                              "movdqa (%1), %%xmm1\n"::"r" ((unsigned long *)(void *)
                                                            bline1),
                              "r"((unsigned long *)(void *) bline2));
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
                                                      long *) (void *) line1),
                                      "r"((unsigned long *) (void *) line2),
                                      "r"((unsigned long *) (void *) line3)
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
/**
 * Unaligned version of deinterlace filter
 *
 * @param src 4-byte aligned buffer
 * @see vc_deinterlace
 */
static void vc_deinterlace_unaligned(unsigned char *src, long src_linesize, int lines)
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
                asm volatile ("movdqu (%0), %%xmm0\n"
                              "movdqu (%1), %%xmm1\n"::"r" (bline1),
                              "r" (bline2));
                line1 = bline2;
                line2 = bline2 + pitch;
                line3 = bline3;
                for (j = 0; j < lines - 4; j += 2) {
                        asm volatile ("movdqu (%1), %%xmm2\n"
                                      "pavgb %%xmm2, %%xmm0\n"
                                      "pavgb %%xmm1, %%xmm0\n"
                                      "movdqu (%2), %%xmm1\n"
                                      "movdqu %%xmm0, (%0)\n"
                                      "pavgb %%xmm1, %%xmm0\n"
                                      "pavgb %%xmm2, %%xmm0\n"
                                      "movdqu %%xmm0, (%1)\n"::"r" (line1),
                                      "r" (line2),
                                      "r" (line3)
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

/**
 * @brief Converts v210 to UYVY
 * @param[out] dst     4-byte aligned output buffer where UYVY will be stored
 * @param[in]  src     4-byte aligned input buffer containing v210 (by definition of v210
 *                     should be even aligned to 16B boundary)
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
 */
void vc_copylinev210(unsigned char *dst, const unsigned char *src, int dst_len)
{
        struct {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } const *s;
        register uint32_t *d;
        register uint32_t tmp;

        d = (uint32_t *)(void *) dst;
        s = (const void *)src;

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

/**
 * @brief Converts from YUYV to UYVY.
 * @copydetails vc_copylinev210
 */
void vc_copylineYUYV(unsigned char *dst, const unsigned char *src, int dst_len)
{
#if WORD_LEN == 64
        register uint32_t *d;
        register const uint32_t *s;
        const uint32_t * const end = (uint32_t *)(void *) dst + dst_len / 4;

        uint32_t mask[4] = {
                0xff00ff00ul,
                0xff00ff00ul,
                0xff00ff00ul,
                0xff00ff00ul};

        d = (uint32_t *)(void *) dst;
        s = (const uint32_t *)(const void *) src;

        assert(dst_len % 4 == 0);

        if((dst_len % 16 == 0)) {
                asm("movdqa (%0), %%xmm4\n"
                                "movdqa %%xmm4, %%xmm5\n"
                                "psrldq $1, %%xmm5\n"
                                : :"r"(mask));
                while(d < end) {
                        asm volatile ("movdqu (%0), %%xmm0\n"
                                        "movdqu %%xmm0, %%xmm1\n"
                                        "pand %%xmm4, %%xmm0\n"
                                        "psrldq $1, %%xmm0\n"
                                        "pand %%xmm5, %%xmm1\n"
                                        "pslldq $1, %%xmm1\n"
                                        "por %%xmm0, %%xmm1\n"
                                        "movdqu %%xmm1, (%1)\n"::"r" (s), "r"(d));
                        s += 4;
                        d += 4;
                }
        } else {
                while(d < end) {
                        register uint32_t tmp = *s;
                        *d = ((tmp & 0x00ff0000) << 8) | ((tmp & 0xff000000) >> 8) |
                                ((tmp & 0x000000ff) << 8) | ((tmp & 0x0000ff00) >> 8);
                        s++;
                        d++;

                }
        }
#else
	char u, y1, v, y2;
	while (dst_len > 0) {
		y1 = *src++;
		u = *src++;
		y2 = *src++;
		v = *src++;
		*dst++ = u;
		*dst++ = y1;
		*dst++ = v;
		*dst++ = y2;
		dst_len -= 4;
	}
#endif
}

/**
 * @brief Converts from R10k to RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     4B-aligned buffer containing pixels in R10k
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
 */
void
vc_copyliner10k(unsigned char *dst, const unsigned char *src, int len, int rshift,
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
        } const *s;
        register uint32_t *d;
        register uint32_t tmp;

        d = (uint32_t *)(void *) dst;
        s = (const void *)(const void *) src;

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

/**
 * @brief Changes color channels' order in RGBA
 * @copydetails vc_copyliner10k
 */
void
vc_copylineRGBA(unsigned char *dst, const unsigned char *src, int len, int rshift,
                int gshift, int bshift)
{
        register uint32_t *d = (uint32_t *)(void *) dst;
        register const uint32_t *s = (const uint32_t *)(const void *) src;
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

/**
 * @brief Converts from DVS10 to v210
 * @copydetails vc_copylinev210
 */
void vc_copylineDVS10toV210(unsigned char *dst, const unsigned char *src, int dst_len)
{
        unsigned int *d;
        const unsigned int *s1;
        register unsigned int a,b;
        d = (unsigned int *)(void *) dst;
        s1 = (const unsigned int *)(const void *) src;

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

/**
 * @brief Converts from DVS10 to UYVY
 * @copydetails vc_copylinev210
 */
void vc_copylineDVS10(unsigned char *dst, const unsigned char *src, int dst_len)
{
        int src_len = dst_len / 1.5; /* right units */
        register const uint64_t *s;
        register uint64_t *d;

        register uint64_t a1, a2, a3, a4;

        d = (uint64_t *)(void *) dst;
        s = (const uint64_t *)(const void *) src;

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

/**
 * @brief Changes color order of an RGB
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGB(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
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

/**
 * @brief Converts from RGBA to RGB
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBAtoRGB(unsigned char *dst2, const unsigned char *src2, int dst_len, int rshift, int gshift, int bshift)
{
        assert(rshift == 0 && gshift == 8 && bshift == 16);

	register const uint32_t * src = (const uint32_t *)(const void *) src2;
	register uint32_t * dst = (uint32_t *)(void *) dst2;
        while(dst_len > 0) {
		register uint32_t in1 = *src++;
		register uint32_t in2 = *src++;
		register uint32_t in3 = *src++;
		register uint32_t in4 = *src++;
		*dst++ = ((in2 & 0xff) << 24) | (in1 & 0xffffff);
		*dst++ = ((in3 & 0xffff) << 16) |  ((in2 & 0xffff00) >> 8);
		*dst++ = ((in4 & 0xffffff) << 8) | ((in3 & 0xff0000) >> 16) ;

                dst_len -= 12;
        }
}

/**
 * @brief Converts from RGBA to RGB. Channels in RGBA can be differently ordered.
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGBAtoRGBwithShift(unsigned char *dst2, const unsigned char *src2, int dst_len, int rshift, int gshift, int bshift)
{
	register const uint32_t * src = (const uint32_t *)(const void *) src2;
	register uint32_t * dst = (uint32_t *)(void *) dst2;
        while(dst_len > 0) {
		register uint32_t in1 = *src++;
		register uint32_t in2 = *src++;
		register uint32_t in3 = *src++;
		register uint32_t in4 = *src++;

                *dst++ = ((in2 >> rshift)) << 24 |
                        ((in1 >> bshift) & 0xff) << 16 |
                        ((in1 >> gshift) & 0xff) << 8 |
                        ((in1 >> rshift) & 0xff);
                *dst++ = ((in3 >> gshift)) << 24 |
                        ((in3 >> rshift) & 0xff) << 16 |
                        ((in2 >> bshift) & 0xff) << 8 |
                        ((in2 >> gshift) & 0xff);
                *dst++  = ((in4 >> bshift)) << 24 |
                        ((in4 >> gshift) & 0xff) << 16 |
                        ((in4 >> rshift) & 0xff) << 8 |
                        ((in3 >> bshift) & 0xff);

                dst_len -= 12;
        }
}

/**
 * @brief Converts from AGBR to RGB
 * @copydetails vc_copylinev210
 * @see vc_copylineRGBAtoRGBwithShift
 * @see vc_copylineRGBAtoRGB
 */
void vc_copylineABGRtoRGB(unsigned char *dst2, const unsigned char *src2, int dst_len, int rshift, int gshift, int bshift)
{
        assert(rshift == 0 && gshift == 8 && bshift == 16);

	register const uint32_t * src = (const uint32_t *)(const void *) src2;
	register uint32_t * dst = (uint32_t *)(void *) dst2;
        while(dst_len > 0) {
		register uint32_t in1 = *src++;
		register uint32_t in2 = *src++;
		register uint32_t in3 = *src++;
		register uint32_t in4 = *src++;

                *dst++ = (in2 & 0xff0000) << 8 |
                        (in1 & 0xff) << 16 |
                        (in1 & 0xff00) |
                        (in1 & 0xff0000) >> 16;
                *dst++ = (in3 & 0xff00) << 16 |
                        (in3 & 0xff0000) |
                        (in2 & 0xff) << 8 |
                        (in2 & 0xff00) >> 8;
                *dst++  = (in4 & 0xff) << 24 |
                        (in4 & 0xff00) << 8 |
                        (in4 & 0xff0000) >> 8 |
                        (in3 & 0xff);

                dst_len -= 12;
        }
}

/**
 * @brief Converts RGBA with different shifts to RGBA
 */
void vc_copylineToRGBA(unsigned char *dst, const unsigned char *src, int dst_len,
                int src_rshift, int src_gshift, int src_bshift)
{
	register const uint32_t * in = (const uint32_t *)(const void *) src;
	register uint32_t * out = (uint32_t *)(void *) dst;
        while(dst_len > 0) {
		register uint32_t in_val = *in++;

                *out++ = ((in_val >> src_rshift) & 0xff) |
                        ((in_val >> src_gshift) & 0xff) << 8 |
                        ((in_val >> src_bshift) & 0xff) << 16;

                dst_len -= 4;
        }
}

/**
 * @brief Converts RGB to RGBA
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGBtoRGBA(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        register unsigned int r, g, b;
        register uint32_t *d = (uint32_t *)(void *) dst;
        
        while(dst_len > 0) {
                r = *src++;
                g = *src++;
                b = *src++;
                
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                dst_len -= 4;
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copyliner10k
 * @param[in] source pixel size (3 for RGB, 4 for RGBA)
 */
static void vc_copylineToUYVY601(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) {
        register int r, g, b;
        register int y1, y2, u ,v;
        register uint32_t *d = (uint32_t *)(void *) dst;

        while(dst_len > 0) {
                r = *(src + rshift);
                g = *(src + gshift);
                b = *(src + bshift);
                src += pix_size;
                y1 = 19595 * r + 38469 * g + 7471 * b;
                u  = -9642 * r -18931 * g + 28573 * b;
                v  = 40304 * r - 33750 * g - 6554 * b;
                r = *(src + rshift);
                g = *(src + gshift);
                b = *(src + bshift);
                src += pix_size;
                y2 = 19595 * r + 38469 * g + 7471 * b;
                u += -9642 * r -18931 * g + 28573 * b;
                v += 40304 * r - 33750 * g - 6554 * b;
                u = u / 2 + (1<<23);
                v = v / 2 + (1<<23);

                *d++ = (min(max(y2, 0), (1<<24)-1) >> 16) << 24 |
                        (min(max(v, 0), (1<<24)-1) >> 16) << 16 |
                        (min(max(y1, 0), (1<<24)-1) >> 16) << 8 |
                        (min(max(u, 0), (1<<24)-1) >> 16);
                dst_len -= 4;
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * Uses Rec. 709 with standard SDI ceiling and floor
 * @copydetails vc_copyliner10k
 * @param[in] source pixel size (3 for RGB, 4 for RGBA)
 */
static void vc_copylineToUYVY709(unsigned char *dst, const unsigned char *src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) {
        register int r, g, b;
        register int y1, y2, u ,v;
        register uint32_t *d = (uint32_t *)(void *) dst;

        while(dst_len > 0) {
                r = *(src + rshift);
                g = *(src + gshift);
                b = *(src + bshift);
                src += pix_size;
                y1 = 11993 * r + 40239 * g + 4063 * b + (1<<20);
                u  = -6619 * r -22151 * g + 28770 * b;
                v  = 28770 * r - 26149 * g - 2621 * b;
                r = *(src + rshift);
                g = *(src + gshift);
                b = *(src + bshift);
                src += pix_size;
                y2 = 11993 * r + 40239 * g + 4063 * b + (1<<20);
                u += -6619 * r -22151 * g + 28770 * b;
                v += 28770 * r - 26149 * g - 2621 * b;
                u = u / 2 + (1<<23);
                v = v / 2 + (1<<23);

                *d++ = (min(max(y2, 0), (1<<24)-1) >> 16) << 24 |
                        (min(max(v, 0), (1<<24)-1) >> 16) << 16 |
                        (min(max(y1, 0), (1<<24)-1) >> 16) << 8 |
                        (min(max(u, 0), (1<<24)-1) >> 16);
                dst_len -= 4;
        }
}

/**
 * @brief Converts UYVY to RGB.
 * Uses Rec. 709 with standard SDI ceiling and floor
 * @copydetails vc_copylinev210
 * @todo make it faster if needed
 */
void vc_copylineUYVYtoRGB(unsigned char *dst, const unsigned char *src, int dst_len) {
        while(dst_len > 0) {
                register int y1, y2, u ,v;
                u = *src++;
                y1 = *src++;
                v = *src++;
                y2 = *src++;
                *dst++ = min(max(1.164*(y1 - 16) + 1.793*(v - 128), 0), 255);
                *dst++ = min(max(1.164*(y1 - 16) - 0.534*(v - 128) - 0.213*(u - 128), 0), 255);
                *dst++ = min(max(1.164*(y1 - 16) + 2.115*(u - 128), 0), 255);
                *dst++ = min(max(1.164*(y2 - 16) + 1.793*(v - 128), 0), 255);
                *dst++ = min(max(1.164*(y2 - 16) - 0.534*(v - 128) - 0.213*(u - 128), 0), 255);
                *dst++ = min(max(1.164*(y2 - 16) + 2.115*(u - 128), 0), 255);

                dst_len -= 6;
        }
}

/**
 * @brief Converts RGB to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoUYVY(unsigned char *dst, const unsigned char *src, int dst_len)
{
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 3);
}

/**
 * @brief Converts BGR to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineBGRtoUYVY(unsigned char *dst, const unsigned char *src, int dst_len)
{
        vc_copylineToUYVY709(dst, src, dst_len, 2, 1, 0, 3);
}

/**
 * @brief Converts RGBA to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBAtoUYVY(unsigned char *dst, const unsigned char *src, int dst_len)
{
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 4);
}

/**
 * Converts BGR to RGB.
 * @copydetails vc_copylinev210
 */
void vc_copylineBGRtoRGB(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        register int r, g, b;

        assert((rshift == 0 && gshift == 8 && bshift == 16) ||
                        (rshift == 16 && gshift == 8 && bshift == 0));

        if (rshift == 16 && gshift == 8 && bshift == 0) {
                memcpy(dst, src, dst_len);
        } else {
                while(dst_len > 0) {
                        b = *src++;
                        g = *src++;
                        r = *src++;
                        *dst++ = r;
                        *dst++ = g;
                        *dst++ = b;
                        dst_len -= 3;
                }
        }
}

/**
 * @brief Converts DPX10 to RGBA
 * @copydetails vc_copyliner10k
 */
void
vc_copylineDPX10toRGBA(unsigned char *dst, const unsigned char *src, int dst_len, int rshift, int gshift, int bshift)
{
        
        register const unsigned int *in = (const unsigned int *)(const void *) src;
        register unsigned int *out = (unsigned int *)(void *) dst;
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

/**
 * @brief Converts DPX10 to RGB.
 * @copydetails vc_copylinev210
 */
void
vc_copylineDPX10toRGB(unsigned char *dst, const unsigned char *src, int dst_len)
{
        
        register const unsigned int *in = (const unsigned int *)(const void *) src;
        register unsigned int *out = (unsigned int *)(void *) dst;
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

/**
 * Returns line decoder for specifiedn input and output codec.
 */
decoder_t get_decoder_from_to(codec_t in, codec_t out, bool slow)
{
        struct item {
                decoder_t decoder;
                codec_t in;
                codec_t out;
                bool slow;
        };

        struct item decoders[] = {
                { (decoder_t) vc_copylineDVS10,       DVS10, UYVY, false },
                { (decoder_t) vc_copylinev210,        v210,  UYVY, false },
                { (decoder_t) vc_copylineYUYV,        YUYV,  UYVY, false },
                { (decoder_t) vc_copyliner10k,        R10k,  RGBA, false },
                { vc_copylineRGBA,        RGBA,  RGBA, false },
                { (decoder_t) vc_copylineDVS10toV210, DVS10, v210, false },
                { (decoder_t) vc_copylineRGBAtoRGB,   RGBA,  RGB, false },
                { (decoder_t) vc_copylineRGBtoRGBA,   RGB,   RGBA, false },
                { (decoder_t) vc_copylineRGBtoUYVY,   RGB,   UYVY, true },
                { (decoder_t) vc_copylineUYVYtoRGB,   UYVY,  RGB, true },
                { (decoder_t) vc_copylineBGRtoUYVY,   BGR,   UYVY, true },
                { (decoder_t) vc_copylineRGBAtoUYVY,  RGBA,  UYVY, true },
                { (decoder_t) vc_copylineBGRtoRGB,    BGR,   RGB, false },
                { (decoder_t) vc_copylineDPX10toRGBA, DPX10, RGBA, false },
                { (decoder_t) vc_copylineDPX10toRGB,  DPX10, RGB, false },
                { vc_copylineRGB,         RGB,   RGB, false },
        };

        for (unsigned int i = 0; i < sizeof(decoders)/sizeof(struct item); ++i) {
                if (decoders[i].in == in && decoders[i].out == out &&
                                (decoders[i].slow == false || slow == true)) {
                        return decoders[i].decoder;
                }
        }

        if (in == out)
                return (decoder_t) memcpy;

        return NULL;
}

/** @brief Returns TRUE if specified pixelformat is some form of RGB (not YUV).
 *
 * Unspecified for compressed codecs.
 * @retval TRUE  if pixelformat is RGB
 * @retval FALSE if pixelformat is not a RGB */
int codec_is_a_rgb(codec_t codec)
{
        int i;

        for (i = 0; codec_info[i].name != NULL; i++) {
		if (codec == codec_info[i].codec) {
			return codec_info[i].rgb;
		}
	}
        return 0;
}

/**
 * Tries to find specified codec in set of video codecs.
 * The set must by ended by VIDEO_CODEC_NONE.
 */
bool codec_is_in_set(codec_t codec, codec_t *set)
{
        assert (codec != VIDEO_CODEC_NONE);
        assert (set != NULL);
        while (*set != VIDEO_CODEC_NONE) {
                if (*(set++) == codec)
                        return true;
        }
        return false;
}

bool clear_video_buffer(unsigned char *data, size_t linesize, size_t pitch, size_t height, codec_t color_spec)
{
        uint32_t pattern[4];

        switch (color_spec) {
                case BGR:
                case RGB:
                case RGBA:
                        memset(pattern, 0, sizeof(pattern));
                        break;
                case UYVY:
                        for (int i = 0; i < 4; i++) {
                                pattern[i] = 0x00800080;
                        }
                        break;
                case v210:
                        pattern[0] = 0x20000200;
                        pattern[1] = 0x00080000;
                        pattern[2] = 0x20000200;
                        pattern[3] = 0x00080000;
                        break;
                default:
                        return false;
        }

        for (size_t y = 0; y < height; ++y) {
                uintptr_t i;
                for( i = 0; i < (linesize & (~15)); i+=16)
                {
                        memcpy(data + i, pattern, 16);
                }
                for( ; i < linesize; i++ )
                {
                        ((char*)data)[i] = ((char*)pattern)[i&15];
                }

                data += pitch;
        }

        return true;
}


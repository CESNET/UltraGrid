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
/* Copyright (c) 2005-2019 CESNET z.s.p.o.
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


#include <stdio.h>
#include <string.h>

#include "debug.h"
#include "hwaccel_vdpau.h"
#include "utils/misc.h" // to_fourcc
#include "video_codec.h"

#ifdef __SSSE3__
#include "tmmintrin.h"
// compat with older Clang compiler
#ifndef _mm_bslli_si128
#define _mm_bslli_si128 _mm_slli_si128
#endif
#ifndef _mm_bsrli_si128
#define _mm_bsrli_si128 _mm_srli_si128
#endif
#endif

#ifdef __cplusplus
#include <algorithm>
using std::max;
using std::min;
#else
#undef max
#undef min
#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))
#endif

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#ifdef __SSE2__
static void vc_deinterlace_aligned(unsigned char *src, long src_linesize, int lines);
static void vc_deinterlace_unaligned(unsigned char *src, long src_linesize, int lines);
#endif
static void vc_copylineToUYVY601(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) __attribute__((unused));

/**
 * Defines codec metadata
 * @note
 * Members that are not relevant for specified codec (eg. bpp, rgb for opaque
 * and interframe for not opaque) should be zero.
 */
struct codec_info_t {
        const char *name;                ///< displayed name
        const char *name_long;           ///< more descriptive name
        uint32_t fcc;                    ///< FourCC
        int h_align;                     ///< Number of pixels each line is aligned to
        double bpp;                      ///< Number of bytes per pixel
                                         ///< @note
                                         ///< Should be nonzero even for compressed codecs
                                         ///< when display gets requestes compressed codec
                                         ///< (otherwise division by zero occurs).
        int bits_per_channel;            ///< Number of bits per color channel
        int block_size;                  ///< Bytes per pixel block (pixelformats only)
        unsigned rgb:1;                  ///< Whether pixelformat is RGB
        unsigned opaque:1;               ///< If codec is opaque (= compressed)
        unsigned interframe:1;           ///< Indicates if compression is interframe
        unsigned const_size:1;           ///< Indicates if data length is constant for all resolutions (hw surfaces)
        const char *file_extension;      ///< Extension that should be added to name if frame is saved to file.
};

static const struct codec_info_t codec_info[] = {
        [VIDEO_CODEC_NONE] = {"(none)", "Undefined Codec",
                0, 0, 0.0, 0, 0, FALSE, FALSE, FALSE, FALSE, NULL},
        [RGBA] = {"RGBA", "Red Green Blue Alpha 32bit",
                to_fourcc('R','G','B','A'), 1, 4.0, 8, 4, TRUE, FALSE, FALSE, FALSE, "rgba"},
        [UYVY] = {"UYVY", "YUV 4:2:2",
                to_fourcc('U','Y','V','Y'), 2, 2, 8, 4, FALSE, FALSE, FALSE, FALSE, "yuv"},
        [YUYV] = {"YUYV", "YUV 4:2:2",
                to_fourcc('Y','U','Y','V'), 2, 2, 8, 4, FALSE, FALSE, FALSE, FALSE, "yuv"},
        [R10k] = {"R10k", "RGB 4:4:4",
                to_fourcc('R','1','0','k'), 64, 4, 10, 4, TRUE, FALSE, FALSE, FALSE, "r10k"},
        [R12L] = {"R12L", "12-bit packed RGB 4:4:4 little-endian", // SMPTE 268M DPX v1, Annex C, Method C4
                to_fourcc('R','1','2','l'), 8, 36.0/8.0, 12, 36, TRUE, FALSE, FALSE, FALSE, "r12l"},
        [v210] = {"v210", "YUV 4:2:2",
                to_fourcc('v','2','1','0'), 48, 8.0 / 3.0, 10, 16, FALSE, FALSE, FALSE, FALSE, "v210"},
        [DVS10] = {"DVS10", "Centaurus 10bit YUV 4:2:2",
                to_fourcc('D','S','1','0'), 48, 8.0 / 3.0, 10, 4, FALSE, FALSE, FALSE, FALSE, "dvs10"},
        [DXT1] = {"DXT1", "S3 Compressed Texture DXT1",
                to_fourcc('D','X','T','1'), 0, 0.5, 2, 0, TRUE, TRUE, FALSE, FALSE, "dxt1"},
        [DXT1_YUV] = {"DXT1_YUV", "S3 Compressed Texture DXT1 YUV",
                to_fourcc('D','X','T','Y'), 0, 0.5, 2, 0, FALSE, TRUE, FALSE, FALSE, "dxt1y"}, /* packet YCbCr inside DXT1 channels */
        [DXT5] = {"DXT5", "S3 Compressed Texture DXT5 YCoCg",
                to_fourcc('D','X','T','5'), 0, 1.0, 4, 0, FALSE, TRUE, FALSE, FALSE, "yog"},/* DXT5 YCoCg */
        [RGB] = {"RGB", "Red Green Blue 24bit",
                to_fourcc('R','G','B','2'), 1, 3.0, 8, 3, TRUE, FALSE, FALSE, FALSE, "rgb"},
        [DPX10] = {"DPX10", "DPX10",
                to_fourcc('D','P','1','0'), 1, 4.0, 10, 4, TRUE, FALSE, FALSE, FALSE, "dpx"},
        [JPEG] = {"JPEG",  "JPEG",
                to_fourcc('J','P','E','G'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "jpg"},
        [RAW] = {"raw", "Raw SDI video",
                to_fourcc('r','a','w','s'), 0, 1.0, 0, 0, FALSE, TRUE, FALSE, FALSE, "raw"}, /* raw SDI */
        [H264] = {"H.264", "H.264/AVC",
                to_fourcc('A','V','C','1'), 0, 1.0, 8, 0, FALSE, TRUE, TRUE, FALSE, "h264"},
        [H265] = {"H.265", "H.265/HEVC",
                to_fourcc('H','E','V','C'), 0, 1.0, 8, 0, FALSE, TRUE, TRUE, FALSE, "h265"},
        [MJPG] = {"MJPEG", "MJPEG",
                to_fourcc('M','J','P','G'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "jpg"},
        [VP8] = {"VP8", "Google VP8",
                to_fourcc('V','P','8','0'), 0, 1.0, 8, 0, FALSE, TRUE, TRUE, FALSE, "vp8"},
        [VP9] = {"VP9", "Google VP9",
                to_fourcc('V','P','9','0'), 0, 1.0, 8, 0, FALSE, TRUE, TRUE, FALSE, "vp9"},
        [BGR] = {"BGR", "Blue Green Red 24bit",
                to_fourcc('B','G','R','2'), 1, 3.0, 8, 3, TRUE, FALSE, FALSE, FALSE, "bgr"},
        [J2K] = {"J2K", "JPEG 2000",
                to_fourcc('M','J','2','C'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "j2k"},
        [J2KR] = {"J2KR", "JPEG 2000 RGB",
                to_fourcc('M','J','2','R'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "j2k"},
#ifdef HWACC_VDPAU
        [HW_VDPAU] = {"HW_VDPAU", "VDPAU hardware surface",
                to_fourcc('V', 'D', 'P', 'S'), 0, 1.0, 8, sizeof(hw_vdpau_frame), FALSE, TRUE, FALSE, TRUE, "vdpau"},
#endif
        [HFYU] = {"HFYU", "HuffYUV",
                to_fourcc('H','F','Y','U'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "hfyu"},
        [FFV1] = {"FFV1", "FFV1",
                to_fourcc('F','F','V','1'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "ffv1"},
        [CFHD] = {"CFHD", "Cineform",
                to_fourcc('C','F','H','D'), 0, 1.0, 8, 0, FALSE, TRUE, FALSE, FALSE, "cfhd"},
        [RG48] = {"RG48", "16-bit RGB little-endian",
                to_fourcc('R','G','4','8'), 1, 6.0, 16, 3, TRUE, FALSE, FALSE, FALSE, "rg48"},
        [AV1] =  {"AV1", "AOMedia Video 1",
                to_fourcc('a','v','0','1'), 0, 1.0, 8, 0, FALSE, TRUE, TRUE, FALSE, "av1"},
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
static const struct alternative_fourcc fourcc_aliases[] = {
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

static const struct alternative_codec_name codec_name_aliases[] = {
        {"2vuy", "UYVY"},
        {"AVC", "H.264"},
        {"HEVC", "H.265"},
};

void show_codec_help(const char *module, const codec_t *codecs8, const codec_t *codecs10, const codec_t *codecs12)
{
        printf("\tSupported codecs (%s):\n", module);

        if (codecs8) {
                printf("\t\t8bits\n");

                while (*codecs8 != VIDEO_CODEC_NONE) {
                        printf("\t\t\t'%s' - %s\n", codec_info[*codecs8].name, codec_info[*codecs8].name_long);
                        codecs8++;
                }
        }

        if (codecs10) {
                printf("\t\t10bits\n");
                while (*codecs10 != VIDEO_CODEC_NONE) {
                        printf("\t\t\t'%s' - %s\n", codec_info[*codecs10].name, codec_info[*codecs10].name_long);
                        codecs10++;
                }
        }

        if (codecs12) {
                printf("\t\t12bits\n");
                while (*codecs12 != VIDEO_CODEC_NONE) {
                        printf("\t\t\t'%s' - %s\n", codec_info[*codecs12].name, codec_info[*codecs10].name_long);
                        codecs12++;
                }
        }
}

int get_bits_per_component(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].bits_per_channel;
        } else {
                return 0;
        }
}

double get_bpp(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].bpp;
        } else {
                return 0;
        }
}

uint32_t get_fourcc(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].fcc;
        } else {
                return 0;
        }
}

const char * get_codec_name(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].name;
        } else {
                return 0;
        }
}

const char * get_codec_name_long(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].name_long;
        } else {
                return 0;
        }
}

codec_t get_codec_from_fcc(uint32_t fourcc)
{
        for (unsigned int i = 0; i < sizeof codec_info / sizeof(struct codec_info_t); ++i) {
                if (fourcc == codec_info[i].fcc)
                        return i;
        }

        // try to look through aliases
        for (size_t i = 0; i < sizeof(fourcc_aliases) / sizeof(struct alternative_fourcc); ++i) {
                if (fourcc == fourcc_aliases[i].alias) {
                        for (unsigned int j = 0; j < sizeof codec_info / sizeof(struct codec_info_t); ++j) {
                                if (fourcc_aliases[i].primary_fcc == codec_info[j].fcc)
                                        return j;
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
        for (unsigned int i = 0; i < sizeof codec_info / sizeof(struct codec_info_t); ++i) {
                if (codec_info[i].name && strcasecmp(codec_info[i].name, name) == 0) {
                        return i;
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
                if (strcasecmp(name, codec_name_aliases[i].alias) == 0) {
                        ret = get_codec_from_name_wo_alias(codec_name_aliases[i].primary_name);
                        if (ret != VIDEO_CODEC_NONE) {
                                return ret;
                        }
                }
        }
        return VIDEO_CODEC_NONE;
}

const char *get_codec_file_extension(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].file_extension;
        } else {
                return 0;
        }
}

/**
 * @retval TRUE if codec is compressed
 * @retval FALSE if codec is pixelformat
 */
int is_codec_opaque(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].opaque;
        } else {
                return 0;
        }
}

/**
 * Returns whether specified codec is an interframe compression.
 * Not defined for pixelformats
 * @retval TRUE if compression is interframe
 * @retval FALSE if compression is not interframe
 */
int is_codec_interframe(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].interframe;
        } else {
                return 0;
        }
}

/** @brief Returns TRUE if specified pixelformat is some form of RGB (not YUV).
 *
 * Unspecified for compressed codecs.
 * @retval TRUE  if pixelformat is RGB
 * @retval FALSE if pixelformat is not a RGB */
int codec_is_a_rgb(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].rgb;
        } else {
                return 0;
        }
}

/** @brief Returns TRUE if specified pixelformat has constant size regardles
 * of resolution. If so the block_size value represents the size.
 *
 * Unspecified for compressed codecs.
 * @retval TRUE  if pixelformat is const size
 * @retval FALSE if pixelformat is not const size */
int codec_is_const_size(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].const_size;
        } else {
                return 0;
        }
}

bool codec_is_hw_accelerated(codec_t codec) {
        return codec == HW_VDPAU;
}

int get_halign(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].h_align;
        } else {
                return 0;
        }
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
        if (codec >= sizeof codec_info / sizeof(struct codec_info_t)) {
                return 0;
        }

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
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].block_size;
        } else {
                return 0;
        }
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
#ifdef __SSE2__
        if(((uintptr_t) src & 0x0Fu) == 0u && src_linesize % 16 == 0) {
                vc_deinterlace_aligned(src, src_linesize, lines);
        } else {
                vc_deinterlace_unaligned(src, src_linesize, lines);
        }
#else
        for (int y = 0; y < lines; y += 2) {
                for (int x = 0; x < src_linesize; ++x) {
                        int val = (*src + src[src_linesize] + 1) >> 1;
                        *src = src[src_linesize]  = val;
                        src++;
                }
                src += src_linesize;
        }
#endif
}

#ifdef __SSE2__
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
#endif

/**
 * Extended version of vc_deinterlace(). The former version was in-place only.
 * This allows to output to different buffer.
 */
void vc_deinterlace_ex(unsigned char *src, size_t src_linesize, unsigned char *dst, size_t dst_pitch, size_t lines)
{
        for (size_t y = 0; y < lines; y += 2) {
                for (size_t x = 0; x < src_linesize; ++x) {
                        int val = (*src + src[src_linesize] + 1) >> 1;
                        *dst = dst[dst_pitch] = val;
                        src++;
                        dst++;
                }
                src += src_linesize;
                dst += dst_pitch;
        }
}

/**
 * @brief Converts v210 to UYVY
 * @param[out] dst     4-byte aligned output buffer where UYVY will be stored
 * @param[in]  src     4-byte aligned input buffer containing v210 (by definition of v210
 *                     should be even aligned to 16B boundary)
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
 */
void vc_copylinev210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
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
void vc_copylineYUYV(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
#if defined __SSE2__
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
                asm("movdqu (%0), %%xmm4\n"
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
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
		y1 = *src++;
		u = *src++;
		y2 = *src++;
		v = *src++;
		*dst++ = u;
		*dst++ = y1;
		*dst++ = v;
		*dst++ = y2;
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
vc_copyliner10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int len, int rshift,
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

        while (len >= 16) {
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
        while (len >= 4) {
                tmp =
                    (s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                len -= 4;
        }
}

/**
 * @brief Converts from R12L to RGB
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     buffer containing pixels in R12L
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
 * @param[in]  rshift  ignored
 * @param[in]  gshift  ignored
 * @param[in]  bshift  ignored
 */
void
vc_copylineR12LtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 24; x += 24) {
                uint8_t tmp;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // r0
                *dst++ = src[BYTE_SWAP(2)]; // g0
                tmp = src[BYTE_SWAP(3)]>> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // b0
                *dst++ = src[BYTE_SWAP(1)]; // r1
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // g1
                *dst++ = src[BYTE_SWAP(0)]; // b1
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // r2
                *dst++ = src[BYTE_SWAP(3)]; // g2
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // b2
                *dst++ = src[BYTE_SWAP(2)]; // r3
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // g3
                *dst++ = src[BYTE_SWAP(1)]; // b3
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // r4
                *dst++ = src[BYTE_SWAP(0)]; // g4
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // b4
                *dst++ = src[BYTE_SWAP(3)]; // r5
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // g5
                *dst++ = src[BYTE_SWAP(2)]; // b5
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // r6
                *dst++ = src[BYTE_SWAP(1)]; // g6
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // b6
                *dst++ = src[BYTE_SWAP(0)]; // r7
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // g7
                *dst++ = src[BYTE_SWAP(3)]; // b7
                src += 4;
        }
}

/**
 * @brief Converts from R12L to RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     buffer containing pixels in R12L
 * @param[in]  dstlen  length of data that should be writen to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
 */
        void
vc_copylineR12L(unsigned char *dst, const unsigned char *src, int dstlen, int rshift,
                int gshift, int bshift)
{
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        uint32_t *d = (uint32_t *)(void *) dst;

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 32; x += 32) {
                uint8_t tmp;
                uint8_t r, g, b;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                r = tmp; // r0
                g = src[BYTE_SWAP(2)]; // g0
                tmp = src[BYTE_SWAP(3)]>> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                b = tmp; // b0
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(1)]; // r1
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                g = tmp; // g1
                b = src[BYTE_SWAP(0)]; // b1
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                r = tmp; // r2
                g = src[BYTE_SWAP(3)]; // g2
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                b = tmp; // b2
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(2)]; // r3
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                g = tmp; // g3
                b = src[BYTE_SWAP(1)]; // b3
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                r = tmp; // r4
                g = src[BYTE_SWAP(0)]; // g4
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                b = tmp; // b4
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(3)]; // r5
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                g = tmp; // g5
                b = src[BYTE_SWAP(2)]; // b5
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                r = tmp; // r6
                g = src[BYTE_SWAP(1)]; // g6
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                b = tmp; // b6
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(0)]; // r7
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                g = tmp; // g7
                b = src[BYTE_SWAP(3)]; // b7
                src += 4;
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
        }
}

/**
 * @brief Changes color channels' order in RGBA
 * @copydetails vc_copyliner10k
 */
void
vc_copylineRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int len, int rshift,
                int gshift, int bshift)
{
        register uint32_t *d = (uint32_t *)(void *) dst;
        register const uint32_t *s = (const uint32_t *)(const void *) src;
        register uint32_t tmp;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, len);
        } else {
                while (len >= 16) {
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
                while (len >= 4) {
                        register unsigned int r, g, b;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        len -= 4;
                }
        }
}

/**
 * @brief Converts from DVS10 to v210
 * @copydetails vc_copylinev210
 */
void vc_copylineDVS10toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        unsigned int *d;
        const unsigned int *s1;
        register unsigned int a,b;
        d = (unsigned int *)(void *) dst;
        s1 = (const unsigned int *)(const void *) src;

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                a = b = *s1++;
                b = ((b >> 24) * 0x00010101) & 0x00300c03;
                a <<= 2;
                b |= a & (0xff<<2);
                a <<= 2;
                b |= a & (0xff00<<4);
                a <<= 2;
                b |= a & (0xff0000<<6);
                *d++ = b;
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
void vc_copylineDVS10(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        int src_len = dst_len / 1.5; /* right units */
        register const uint64_t *s;
        register uint64_t *d;

        register uint64_t a1, a2, a3, a4;

        d = (uint64_t *)(void *) dst;
        s = (const uint64_t *)(const void *) src;

        OPTIMIZED_FOR (int x = 0; x <= src_len - 16; x += 16) {
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
        }
}

#endif                          /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

/**
 * @brief Changes color order of an RGB
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        register unsigned int r, g, b;
        union {
                unsigned int out;
                unsigned char c[4];
        } u;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, dst_len);
        } else {
                OPTIMIZED_FOR (int x = 0; x <= dst_len - 3; x += 3) {
                        r = *src++;
                        g = *src++;
                        b = *src++;
                        u.out = (r << rshift) | (g << gshift) | (b << bshift);
                        *dst++ = u.c[0];
                        *dst++ = u.c[1];
                        *dst++ = u.c[2];
                }
        }
}

/**
 * @brief Converts from RGBA to RGB
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBAtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineRGBAtoRGBwithShift(dst, src, dst_len, 0, 8, 16);
}

/**
 * @brief Converts from RGBA to RGB. Channels in RGBA can be differently ordered.
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGBAtoRGBwithShift(unsigned char * __restrict dst2, const unsigned char * __restrict src2, int dst_len, int rshift, int gshift, int bshift)
{
	register const uint32_t * src = (const uint32_t *)(const void *) src2;
	register uint32_t * dst = (uint32_t *)(void *) dst2;
        int x;
        OPTIMIZED_FOR (x = 0; x <= dst_len - 12; x += 12) {
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
        }

        uint8_t *dst_c = (uint8_t *) dst;
        for (; x <= dst_len - 3; x += 3) {
		register uint32_t in = *src++;
                *dst_c++ = (in >> rshift) & 0xff;
                *dst_c++ = (in >> gshift) & 0xff;
                *dst_c++ = (in >> bshift) & 0xff;
        }
}

/**
 * @brief Converts from AGBR to RGB
 * @copydetails vc_copylinev210
 * @see vc_copylineRGBAtoRGBwithShift
 * @see vc_copylineRGBAtoRGB
 */
void vc_copylineABGRtoRGB(unsigned char * __restrict dst2, const unsigned char * __restrict src2, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        vc_copylineRGBAtoRGBwithShift(dst2, src2, dst_len, 16, 8, 0);
}

/**
 * @brief Converts RGBA with different shifts to RGBA
 *
 * dst and src may overlap
 */
void vc_copylineToRGBA_inplace(unsigned char *dst, const unsigned char *src, int dst_len,
                int src_rshift, int src_gshift, int src_bshift)
{
	register const uint32_t * in = (const uint32_t *)(const void *) src;
	register uint32_t * out = (uint32_t *)(void *) dst;
        while (dst_len >= 4) {
		register uint32_t in_val = *in++;

                *out++ = ((in_val >> src_rshift) & 0xff) |
                        ((in_val >> src_gshift) & 0xff) << 8 |
                        ((in_val >> src_bshift) & 0xff) << 16;

                dst_len -= 4;
        }
}

/**
 * @brief Converts UYVY to grayscale.
 * @todo is this correct??
 */
void vc_copylineUYVYtoGrayscale(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 2; x += 2) {
                src++; // U
                *dst++ = *src++; // Y
                src++; // V
                *dst++ = *src++; // Y
        }
}

/**
 * @brief Converts RGB to RGBA
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGBtoRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        register unsigned int r, g, b;
        register uint32_t *d = (uint32_t *)(void *) dst;
        
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                r = *src++;
                g = *src++;
                b = *src++;
                
                *d++ = (r << rshift) | (g << gshift) | (b << bshift);
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copyliner10k
 * @param[in] source pixel size (3 for RGB, 4 for RGBA)
 */
static void vc_copylineToUYVY601(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift, int pix_size) {
        register int r, g, b;
        register int y1, y2, u ,v;
        register uint32_t *d = (uint32_t *)(void *) dst;

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
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
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * Uses Rec. 709 with standard SDI ceiling and floor
 * @copydetails vc_copyliner10k
 * @param[in] roff     red offset in bytes (0 for RGB)
 * @param[in] goff     green offset in bytes (1 for RGB)
 * @param[in] boff     blue offset in bytes (2 for RGB)
 * @param[in] pix_size source pixel size (3 for RGB, 4 for RGBA)
 */
#define vc_copylineToUYVY709(dst, src, dst_len, roff, goff, boff, pix_size) {\
        register uint32_t *d = (uint32_t *)(void *) dst;\
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += pix_size) {\
                register int r, g, b;\
                register int y1, y2, u ,v;\
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y1 = 11993 * r + 40239 * g + 4063 * b + (1<<20);\
                u  = -6619 * r -22151 * g + 28770 * b;\
                v  = 28770 * r - 26149 * g - 2621 * b;\
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y2 = 11993 * r + 40239 * g + 4063 * b + (1<<20);\
                u += -6619 * r -22151 * g + 28770 * b;\
                v += 28770 * r - 26149 * g - 2621 * b;\
                u = u / 2 + (1<<23);\
                v = v / 2 + (1<<23);\
\
                *d++ = (min(max(y2, 0), (1<<24)-1) >> 16) << 24 |\
                        (min(max(v, 0), (1<<24)-1) >> 16) << 16 |\
                        (min(max(y1, 0), (1<<24)-1) >> 16) << 8 |\
                        (min(max(u, 0), (1<<24)-1) >> 16);\
        }\
}

/**
 * @brief Converts 8-bit YCbCr (packed 4:2:2 in 32-bit) word to RGB.
 *
 * Converts 8-bit YCbCr (packed 4:2:2 in 32-bit word to RGB. Offset of YCbCr
 * components can be given by parameters (in bytes). This macro is used by
 * vc_copylineUYVYtoRGB() and vc_copylineYUYVtoRGB().
 *
 * Uses Rec. 709 with standard SDI ceiling and floor
 *
 * @todo make it faster if needed
 */
#define copylineYUVtoRGB(dst, src, dst_len, y1_off, y2_off, u_off, v_off) {\
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 6; x += 6) {\
                register int y1, y2, u ,v;\
                y1 = src[y1_off];\
                y2 = src[y2_off];\
                u = src[u_off];\
                v = src[v_off];\
                src += 4;\
                *dst++ = min(max(1.164*(y1 - 16) + 1.793*(v - 128), 0), 255);\
                *dst++ = min(max(1.164*(y1 - 16) - 0.534*(v - 128) - 0.213*(u - 128), 0), 255);\
                *dst++ = min(max(1.164*(y1 - 16) + 2.115*(u - 128), 0), 255);\
                *dst++ = min(max(1.164*(y2 - 16) + 1.793*(v - 128), 0), 255);\
                *dst++ = min(max(1.164*(y2 - 16) - 0.534*(v - 128) - 0.213*(u - 128), 0), 255);\
                *dst++ = min(max(1.164*(y2 - 16) + 2.115*(u - 128), 0), 255);\
        }\
}

/**
 * @brief Converts UYVY to RGB.
 * @see copylineYUVtoRGB
 * @param[out] dst     output buffer for RGB
 * @param[in]  src     input buffer with UYVY
 */
void vc_copylineUYVYtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        copylineYUVtoRGB(dst, src, dst_len, 1, 3, 0, 2);
}

/**
 * @brief Converts YUYV to RGB.
 * @see copylineYUVtoRGB
 * @param[out] dst     output buffer for RGB
 * @param[in]  src     input buffer with YUYV
 */
void vc_copylineYUYVtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        copylineYUVtoRGB(dst, src, dst_len, 0, 2, 1, 3);
}

/**
 * @brief Converts UYVY to RGB using SSE.
 * Uses Rec. 709 with standard SDI ceiling and floor
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 * @todo make it faster if needed
 */
void vc_copylineUYVYtoRGB_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
#ifdef __SSSE3__
        __m128i yfactor = _mm_set1_epi16(74); //1.164 << 6
        __m128i rvfactor = _mm_set1_epi16(115); //1.793 << 6
        __m128i gvfactor = _mm_set1_epi16(34); //0.534 << 6
        __m128i gufactor = _mm_set1_epi16(14); //0.213 << 6
        __m128i bufactor = _mm_set1_epi16(135); //2.115 << 6

        __m128i ysub = _mm_set1_epi16(16);
        __m128i uvsub = _mm_set1_epi16(128);

        __m128i zero128 = _mm_set1_epi32(0);
        __m128i max = _mm_set1_epi16(255);
        //YYVVYYUU
        __m128i ymask = _mm_set1_epi32(0x00FF00FF);
        __m128i umask = _mm_set1_epi32(0x000000FF);

        __m128i rgbshuffle = _mm_setr_epi8(0, 2, 1, 4, 6, 5, 8, 10, 9, 12, 14, 13, 15, 11, 7, 3);

        __m128i yuv;
        __m128i rgb;

        __m128i y;
        __m128i u;
        __m128i v;
        __m128i r;
        __m128i g;
        __m128i b;

        while(dst_len >= 28){
                yuv = _mm_lddqu_si128((__m128i const*) src);
                src += 16;

                u = _mm_and_si128(yuv, umask);
                u = _mm_or_si128(u, _mm_bslli_si128(u, 2));

                yuv = _mm_bsrli_si128(yuv, 1);
                y = _mm_and_si128(yuv, ymask);

                yuv = _mm_bsrli_si128(yuv, 1);
                v = _mm_and_si128(yuv, umask);
                v = _mm_or_si128(v, _mm_bslli_si128(v, 2));

                y = _mm_subs_epi16(y, ysub);
                y = _mm_mullo_epi16(y, yfactor);

                u = _mm_subs_epi16(u, uvsub);
                v = _mm_subs_epi16(v, uvsub);

                r = _mm_adds_epi16(y, _mm_mullo_epi16(v, rvfactor));
                g = _mm_subs_epi16(y, _mm_mullo_epi16(v, gvfactor));
                g = _mm_subs_epi16(g, _mm_mullo_epi16(u, gufactor));
                b = _mm_adds_epi16(y, _mm_mullo_epi16(u, bufactor));

                //Make sure that the result is in the interval 0..255
                r = _mm_max_epi16(zero128, r);
                g = _mm_max_epi16(zero128, g);
                b = _mm_max_epi16(zero128, b);

                r = _mm_srli_epi16(r, 6);
                g = _mm_srli_epi16(g, 6);
                b = _mm_srli_epi16(b, 6);

                r = _mm_min_epi16(max, r);
                g = _mm_min_epi16(max, g);
                b = _mm_min_epi16(max, b);

                rgb = _mm_or_si128(_mm_bslli_si128(g, 1), r);
                rgb = _mm_unpacklo_epi8(rgb, b);
                rgb = _mm_shuffle_epi8(rgb, rgbshuffle);
                _mm_storeu_si128((__m128i *) dst, rgb);
                dst += 12;

                rgb = _mm_or_si128(_mm_bslli_si128(g, 1), r);
                rgb = _mm_unpackhi_epi8(rgb, b);
                rgb = _mm_shuffle_epi8(rgb, rgbshuffle);
                _mm_storeu_si128((__m128i *) dst, rgb);
                dst += 12;

                dst_len -= 24;
        }
#endif
        //copy last few pixels
        vc_copylineUYVYtoRGB(dst, src, dst_len, rshift, gshift, bshift);
}

/**
 * Converts 8-bit RGB to 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing).
 */
void vc_copylineRGBtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 36; x += 36) {
                unsigned char r = *src++;
                unsigned char g = *src++;
                unsigned char b = *src++;
                dst[BYTE_SWAP(0)] = r << 4;
                dst[BYTE_SWAP(1)] = r >> 4;
                dst[BYTE_SWAP(2)] = g;
                dst[BYTE_SWAP(3)] = b << 4;
                dst[4 + BYTE_SWAP(0)] = b >> 4;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[4 + BYTE_SWAP(1)] = r;
                dst[4 + BYTE_SWAP(2)] = g << 4;
                dst[4 + BYTE_SWAP(3)] = g >> 4;
                dst[8 + BYTE_SWAP(0)] = b;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[8 + BYTE_SWAP(1)] = r << 4;
                dst[8 + BYTE_SWAP(2)] = r >> 4;
                dst[8 + BYTE_SWAP(3)] = g;
                dst[12 + BYTE_SWAP(0)] = b << 4;
                dst[12 + BYTE_SWAP(1)] = b >> 4;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[12 + BYTE_SWAP(2)] = r;
                dst[12 + BYTE_SWAP(3)] = g << 4;
                dst[16 + BYTE_SWAP(0)] = g >> 4;
                dst[16 + BYTE_SWAP(1)] = b;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[16 + BYTE_SWAP(2)] = r << 4;
                dst[16 + BYTE_SWAP(3)] = r >> 4;
                dst[20 + BYTE_SWAP(0)] = g;
                dst[20 + BYTE_SWAP(1)] = b << 4;
                dst[20 + BYTE_SWAP(2)] = b >> 4;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[20 + BYTE_SWAP(3)] = r;
                dst[24 + BYTE_SWAP(0)] = g << 4;
                dst[24 + BYTE_SWAP(1)] = g >> 4;
                dst[24 + BYTE_SWAP(2)] = b;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[24 + BYTE_SWAP(3)] = r << 4;
                dst[28 + BYTE_SWAP(0)] = r >> 4;
                dst[28 + BYTE_SWAP(1)] = g;
                dst[28 + BYTE_SWAP(2)] = b << 4;
                dst[28 + BYTE_SWAP(3)] = b >> 4;
                r = *src++;
                g = *src++;
                b = *src++;
                dst[32 + BYTE_SWAP(0)] = r;
                dst[32 + BYTE_SWAP(1)] = g << 4;
                dst[32 + BYTE_SWAP(2)] = g >> 4;
                dst[32 + BYTE_SWAP(3)] = b;
                dst += 36;
        }
}

/**
 * @brief Converts R12L to RG48.
 * Converts 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing) to 16-bit RGB
 * @copydetails vc_copylinev210
 */
void vc_copylineR12LtoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 48; x += 48) {
                //0
                //R
                *dst++ = src[BYTE_SWAP(0)] << 4;
                *dst++ = (src[BYTE_SWAP(1)] << 4) | (src[BYTE_SWAP(0)] >> 4);
                //G
                *dst++ = src[BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[BYTE_SWAP(2)];
                //B
                *dst++ = src[BYTE_SWAP(3)] << 4;
                *dst++ = (src[4 + BYTE_SWAP(0)] << 4) | (src[BYTE_SWAP(3)] >> 4);

                //1
                *dst++ = src[4 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[4 + BYTE_SWAP(1)];

                *dst++ = src[4 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[4 + BYTE_SWAP(3)] << 4) | (src[4 + BYTE_SWAP(2)] >> 4);

                *dst++ = src[4 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[8 + BYTE_SWAP(0)];

                //2
                *dst++ = src[8 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[8 + BYTE_SWAP(2)] << 4) | (src[8 + BYTE_SWAP(1)] >> 4);

                *dst++ = src[8 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[8 + BYTE_SWAP(3)];

                *dst++ = src[12 + BYTE_SWAP(0)] << 4;
                *dst++ = (src[12 + BYTE_SWAP(1)] << 4) | (src[12 + BYTE_SWAP(0)] >> 4);

                //3
                *dst++ = src[12 + BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[12 + BYTE_SWAP(2)];

                *dst++ = src[12 + BYTE_SWAP(3)] << 4;
                *dst++ = (src[16 + BYTE_SWAP(0)] << 4) | (src[12 + BYTE_SWAP(3)] >> 4);

                *dst++ = src[16 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[16 + BYTE_SWAP(1)];

                //4
                *dst++ = src[16 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[16 + BYTE_SWAP(3)] << 4) | (src[16 + BYTE_SWAP(2)] >> 4);

                *dst++ = src[16 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[20 + BYTE_SWAP(0)];

                *dst++ = src[20 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[20 + BYTE_SWAP(2)] << 4) | (src[20 + BYTE_SWAP(1)] >> 4);

                //5
                *dst++ = src[20 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[20 + BYTE_SWAP(3)];

                *dst++ = src[24 + BYTE_SWAP(0)] << 4;
                *dst++ = (src[24 + BYTE_SWAP(1)] << 4) | (src[24 + BYTE_SWAP(0)] >> 4);

                *dst++ = src[24 + BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[24 + BYTE_SWAP(2)];

                //6
                *dst++ = src[24 + BYTE_SWAP(3)] << 4;
                *dst++ = (src[28 + BYTE_SWAP(0)] << 4) | (src[24 + BYTE_SWAP(3)] >> 4);

                *dst++ = src[28 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[28 + BYTE_SWAP(1)];

                *dst++ = src[28 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[28 + BYTE_SWAP(3)] << 4) | (src[28 + BYTE_SWAP(2)] >> 4);

                //7
                *dst++ = src[28 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[32 + BYTE_SWAP(0)];

                *dst++ = src[32 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[32 + BYTE_SWAP(2)] << 4) | (src[32 + BYTE_SWAP(1)] >> 4);

                *dst++ = src[32 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[32 + BYTE_SWAP(3)];

                src += 36;
        }
}

/**
 * @brief Converts RG48 to R12L.
 * Converts 16-bit RGB to 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing)
 * @copydetails vc_copylinev210
 */
void vc_copylineRG48toR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 36; x += 36) {
                //0
                dst[BYTE_SWAP(0)] = src[0] >> 4;
                dst[BYTE_SWAP(0)] |= src[1] << 4;
                dst[BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                dst[BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[BYTE_SWAP(2)] = src[1];
                src += 2;

                dst[BYTE_SWAP(3)] = src[0] >> 4;
                dst[BYTE_SWAP(3)] |= src[1] << 4;
                dst[4 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                //1
                dst[4 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[4 + BYTE_SWAP(1)] = src[1];
                src += 2;

                dst[4 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[4 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[4 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                dst[4 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[8 + BYTE_SWAP(0)] = src[1];
                src += 2;

                //2
                dst[8 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[8 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[8 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                dst[8 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[8 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst[12 + BYTE_SWAP(0)] = src[0] >> 4;
                dst[12 + BYTE_SWAP(0)] |= src[1] << 4;
                dst[12 + BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                //3
                dst[12 + BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[12 + BYTE_SWAP(2)] = src[1];
                src += 2;

                dst[12 + BYTE_SWAP(3)] = src[0] >> 4;
                dst[12 + BYTE_SWAP(3)] |= src[1] << 4;
                dst[16 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                dst[16 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[16 + BYTE_SWAP(1)] = src[1];
                src += 2;

                //4
                dst[16 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[16 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[16 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                dst[16 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[20 + BYTE_SWAP(0)] = src[1];
                src += 2;

                dst[20 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[20 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[20 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                //5
                dst[20 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[20 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst[24 + BYTE_SWAP(0)] = src[0] >> 4;
                dst[24 + BYTE_SWAP(0)] |= src[1] << 4;
                dst[24 + BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                dst[24 + BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[24 + BYTE_SWAP(2)] = src[1];
                src += 2;

                //6
                dst[24 + BYTE_SWAP(3)] = src[0] >> 4;
                dst[24 + BYTE_SWAP(3)] |= src[1] << 4;
                dst[28 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                dst[28 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[28 + BYTE_SWAP(1)] = src[1];
                src += 2;

                dst[28 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[28 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[28 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                //7
                dst[28 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[32 + BYTE_SWAP(0)] = src[1];
                src += 2;

                dst[32 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[32 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[32 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                dst[32 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[32 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst += 36;
        }
}

void vc_copylineRG48toRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        uint32_t *dst32 = (uint32_t *)(void *) dst;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                *dst32++ = src[1] << rshift | src[3] << gshift | src[5] << bshift;
                src += 6;
        }
}

/**
 * @brief Converts RGB to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 3);
}

/**
 * @brief Converts RGB to UYVY using SSE.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoUYVY_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
#ifdef __SSSE3__
        __m128i rgb;
        __m128i yuv;

        __m128i r;
        __m128i g;
        __m128i b;

        __m128i y;
        __m128i u;
        __m128i v;

        //BB BB BB BB GR GR GR GR
        __m128i shuffle = _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 2, 2, 5, 5, 8, 8, 11, 11);
        __m128i mask = _mm_setr_epi16(0x00FF, 0x00FF, 0x00FF, 0x00FF, 0, 0, 0, 0);
        __m128i lowmask = _mm_setr_epi16(0, 0, 0, 0, 0x00FF, 0x00FF, 0x00FF, 0x00FF);
        __m128i zero = _mm_set1_epi16(0);

        __m128i yrf = _mm_set1_epi16(23);
        __m128i ygf = _mm_set1_epi16(79);
        __m128i ybf = _mm_set1_epi16(8);

        __m128i urf = _mm_set1_epi16(13);
        __m128i ugf = _mm_set1_epi16(43);
        __m128i ubf = _mm_set1_epi16(56);

        __m128i vrf = _mm_set1_epi16(56);
        __m128i vgf = _mm_set1_epi16(51);
        __m128i vbf = _mm_set1_epi16(5);

        __m128i yadd = _mm_set1_epi16(2048);
        __m128i uvadd = _mm_set1_epi16(16384);

        while(dst_len >= 16){
                //Load first 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*) src);

                src += 12;
                rgb = _mm_shuffle_epi8(rgb, shuffle);

                r = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 1);
                //0B BB BB BB BG RG RG RG
                g = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 7);
                //00 00 00 00 BB BB BB BB
                b = _mm_and_si128(rgb, mask);

                //Load next 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*) src);
                src += 12;
                rgb = _mm_shuffle_epi8(rgb, shuffle);
                b = _mm_or_si128(b, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 7);
                g = _mm_or_si128(g, _mm_and_si128(rgb, lowmask));
                //0B BB BB BB BG RG RG RG
                rgb = _mm_bslli_si128(rgb, 1);
                r = _mm_or_si128(r, _mm_and_si128(rgb, lowmask));
                //00 00 00 00 BB BB BB BB

                //Compute YUV values
                y = _mm_adds_epi16(yadd, _mm_mullo_epi16(r, yrf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(g, ygf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(b, ybf));

                u = _mm_subs_epi16(uvadd, _mm_mullo_epi16(r, urf));
                u = _mm_subs_epi16(u, _mm_mullo_epi16(g, ugf));
                u = _mm_adds_epi16(u, _mm_mullo_epi16(b, ubf));

                v = _mm_adds_epi16(uvadd, _mm_mullo_epi16(r, vrf));
                v = _mm_subs_epi16(v, _mm_mullo_epi16(g, vgf));
                v = _mm_subs_epi16(v, _mm_mullo_epi16(b, vbf));

                y = _mm_srli_epi16(y, 7);
                u = _mm_srli_epi16(u, 7);
                v = _mm_srli_epi16(v, 7);

                u = _mm_hadd_epi16(u, v);
                u = _mm_srli_epi16(u, 1);

                v = _mm_unpackhi_epi16(zero, u);
                u = _mm_unpacklo_epi16(u, zero);

                y = _mm_bslli_si128(y, 1);
                yuv = _mm_or_si128(y, u);
                yuv = _mm_or_si128(yuv, v);

                _mm_storeu_si128((__m128i *) dst, yuv);
                dst += 16;
                dst_len -= 16;
        }
#endif
        //copy last few pixels
        vc_copylineRGBtoUYVY(dst, src, dst_len, rshift, gshift, bshift);
}

/**
 * @brief Converts RGB to Grayscale using SSE.
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoGrayscale_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
#ifdef __SSSE3__
        __m128i rgb;

        __m128i r;
        __m128i g;
        __m128i b;

        __m128i y;

        //BB BB BB BB GR GR GR GR
        __m128i inshuffle = _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 2, 2, 5, 5, 8, 8, 11, 11);
        __m128i outshuffle = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 1, 1, 1, 1, 1, 1, 1);
        __m128i mask = _mm_setr_epi16(0x00FF, 0x00FF, 0x00FF, 0x00FF, 0, 0, 0, 0);
        __m128i lowmask = _mm_setr_epi16(0, 0, 0, 0, 0x00FF, 0x00FF, 0x00FF, 0x00FF);

        __m128i yrf = _mm_set1_epi16(23);
        __m128i ygf = _mm_set1_epi16(79);
        __m128i ybf = _mm_set1_epi16(8);

        __m128i yadd = _mm_set1_epi16(2048);

        while(dst_len >= 16){
                //Load first 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*) src);

                src += 12;
                rgb = _mm_shuffle_epi8(rgb, inshuffle);

                //BB BB BB BB GR GR GR GR
                r = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 1);
                //0B BB BB BB BG RG RG RG
                g = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 7);
                //00 00 00 00 BB BB BB BB
                b = _mm_and_si128(rgb, mask);

                //Load next 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*) src);
                src += 12;
                rgb = _mm_shuffle_epi8(rgb, inshuffle);
				//BB BB BB BB GR GR GR GR
                b = _mm_or_si128(b, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 7);
				//BG RG RG RG R0 00 00 00
                g = _mm_or_si128(g, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 1);
                //GR GR GR GR 00 00 00 00
                r = _mm_or_si128(r, _mm_and_si128(rgb, lowmask));

                //Compute Y values
                y = _mm_adds_epi16(yadd, _mm_mullo_epi16(r, yrf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(g, ygf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(b, ybf));

                y = _mm_srli_epi16(y, 7);

                y = _mm_shuffle_epi8(y, outshuffle);

                _mm_storeu_si128((__m128i *) dst, y);
                dst += 8;
                dst_len -= 8;
        }
#endif
        //copy last few pixels
        register int ri, gi, bi;
        register int y1, y2;
        register uint16_t *d = (uint16_t *)(void *) dst;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 2; x += 2) {
                ri = *(src++);
                gi = *(src++);
                bi = *(src++);
                y1 = 11993 * ri + 40239 * gi + 4063 * bi + (1<<20);
                ri = *(src++);
                gi = *(src++);
                bi = *(src++);
                y2 = 11993 * ri + 40239 * gi + 4063 * bi + (1<<20);

                *d++ = (min(max(y2, 0), (1<<24)-1) >> 16) << 8 |
                       (min(max(y1, 0), (1<<24)-1) >> 16);
        }
}

/**
 * @brief Converts BGR to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineBGRtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY709(dst, src, dst_len, 2, 1, 0, 3);
}

/**
 * @brief Converts RGBA to UYVY.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBAtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 4);
}

/**
 * Converts BGR to RGB.
 * @copydetails vc_copylinev210
 */
void vc_copylineBGRtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        vc_copylineRGB(dst, src, dst_len, 16, 8, 0);
}

void vc_memcpy(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        memcpy(dst, src, dst_len);
}

/**
 * @brief Converts DPX10 to RGBA
 * @copydetails vc_copyliner10k
 */
void
vc_copylineDPX10toRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        
        register const unsigned int *in = (const unsigned int *)(const void *) src;
        register unsigned int *out = (unsigned int *)(void *) dst;
        register int r,g,b;

        while (dst_len >= 4) {
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
vc_copylineDPX10toRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        register const unsigned int *in = (const unsigned int *)(const void *) src;
        register unsigned int *out = (unsigned int *)(void *) dst;
        register int r1,g1,b1,r2,g2,b2;
       
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 12; x += 12) {
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
        }
        if (dst_len != 0) {
                /// @todo
                log_msg(LOG_LEVEL_WARNING, "TODO: incomplete %s implementation!\n", __func__);
        }
}

struct decoder_item {
        decoder_t decoder;
        codec_t in;
        codec_t out;
        bool slow;
};

static const struct decoder_item decoders[] = {
        { (decoder_t) vc_copylineDVS10,       DVS10, UYVY, false },
        { (decoder_t) vc_copylinev210,        v210,  UYVY, false },
        { (decoder_t) vc_copylineYUYV,        YUYV,  UYVY, false },
        { (decoder_t) vc_copyliner10k,        R10k,  RGBA, false },
        { (decoder_t) vc_copylineR12L,        R12L,  RGBA, false },
        { (decoder_t) vc_copylineR12LtoRGB,   R12L,  RGB, false },
        { vc_copylineRGBA,        RGBA,  RGBA, false },
        { (decoder_t) vc_copylineDVS10toV210, DVS10, v210, false },
        { (decoder_t) vc_copylineRGBAtoRGB,   RGBA,  RGB, false },
        { (decoder_t) vc_copylineRGBtoRGBA,   RGB,   RGBA, false },
        { (decoder_t) vc_copylineRGBtoUYVY,   RGB,   UYVY, true },
        { (decoder_t) vc_copylineUYVYtoRGB,   UYVY,  RGB, true },
        { (decoder_t) vc_copylineYUYVtoRGB,   YUYV,  RGB, true },
        { (decoder_t) vc_copylineBGRtoUYVY,   BGR,   UYVY, true },
        { (decoder_t) vc_copylineRGBAtoUYVY,  RGBA,  UYVY, true },
        { (decoder_t) vc_copylineBGRtoRGB,    BGR,   RGB, false },
        { (decoder_t) vc_copylineDPX10toRGBA, DPX10, RGBA, false },
        { (decoder_t) vc_copylineDPX10toRGB,  DPX10, RGB, false },
        { vc_copylineRGB,         RGB,   RGB, false },
};

/**
 * Returns line decoder for specifiedn input and output codec.
 *
 * @param[in] slow  include also slow decoders
 */
decoder_t get_decoder_from_to(codec_t in, codec_t out, bool slow)
{
        for (unsigned int i = 0; i < sizeof(decoders)/sizeof(struct decoder_item); ++i) {
                if (decoders[i].in == in && decoders[i].out == out &&
                                (decoders[i].slow == false || slow == true)) {
                        return decoders[i].decoder;
                }
        }

        if (in == out)
                return vc_memcpy;

        return NULL;
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
#ifdef HWACC_VDPAU
                case HW_VDPAU:
                        memset(data, 0,sizeof(hw_vdpau_frame));
                        return true;
#endif
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

/* vim: set expandtab sw=8: */

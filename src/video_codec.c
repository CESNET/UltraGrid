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
 * functions.
 */
/* Copyright (c) 2005-2025 CESNET
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

#define __STDC_WANT_LIB_EXT1__ 1

#ifdef HAVE_CONFIG_H
#include "config.h"            // for HWACC_VDPAU
#endif // HAVE_CONFIG_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "color.h"
#include "compat/endian.h"       // for be32toh
#include "compat/qsort_s.h"
#include "compat/strings.h"      // for strcasecmp
#include "debug.h"
#include "host.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_drm.h"
#include "utils/debug.h"         // for DEBUG_TIMER_*
#include "utils/macros.h" // to_fourcc, OPTIMEZED_FOR
#include "video_codec.h"

#ifdef __SSSE3__
#include "tmmintrin.h"
#endif

char pixfmt_conv_pref[] = "dsc"; ///< bitdepth, subsampling, color space

#ifdef __SSE2__
static void vc_deinterlace_aligned(unsigned char *src, long src_linesize, int lines);
static void vc_deinterlace_unaligned(unsigned char *src, long src_linesize, int lines);
#endif

enum {
        VC_OPAQUE = 0, ///< codec is not a raw pixelformat
};

enum video_codec_flag {
        VCF_NONE       = 0,      ///< none of the flags below
        VCF_RGB        = 1 << 0, ///< Whether pixelformat is RGB [pf only]
        VCF_INTERFRAME = 1 << 1, ///< Indicates if compression is interframe
        VCF_CONST_SIZE = 1 << 2, ///< Indicates if data length is constant for all resolutions (hw surfaces)
};

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
        int block_size_bytes;            ///< Bytes per pixel block (packed pixelformats only, otherwise set to 1)
        int block_size_pixels;           ///< Pixels per pixel block (ditto), _bytes/_pixels yield BPP
        int h_align;                     ///< Number of pixels each line is aligned to
        int bits_per_channel;            ///< Number of bits per color channel
        unsigned flags;                  ///< bitwise OR of flags in @ref video_codec_flag
        int subsampling;                 ///< @ref enum_subsampling "enum subsampling" for PF, @ref VC_OPAQUE otherwise
        const char *file_extension;      ///< Extension that should be added to name if frame is saved to file.
};

#ifdef HWACC_VDPAU
#define HW_VDPAU_FRAME_SZ sizeof(hw_vdpau_frame)
#else
#define HW_VDPAU_FRAME_SZ 0
#endif

static const struct codec_info_t codec_info[] = {
        [VIDEO_CODEC_NONE] = {"(none)", "Undefined Codec",
                0, 0, 0, 0, 0, VCF_NONE, VC_OPAQUE, NULL},
        [RGBA] = {"RGBA", "Red Green Blue Alpha 32bit",
                to_fourcc('R','G','B','A'), 4, 1, 1, 8, VCF_RGB, SUBS_4444, "rgba"},
        [UYVY] = {"UYVY", "YUV 4:2:2",
                to_fourcc('U','Y','V','Y'), 4, 2, 2, 8, VCF_NONE, SUBS_422, "yuv"},
        [YUYV] = {"YUYV", "YUV 4:2:2",
                to_fourcc('Y','U','Y','V'), 4, 2, 2, 8, VCF_NONE, SUBS_422, "yuv"},
        [VUYA] = {"VUYA", "VUYA 4:4:4:4",
                to_fourcc('V','U','Y','A'), 4, 1, 1, 8, VCF_NONE, SUBS_4444, "vuya"},
        [R10k] = {"R10k", "10-bit RGB 4:4:4", // called 'R10b' in BMD SDK
                to_fourcc('R','1','0','k'), 4, 1, 64, 10, VCF_RGB, SUBS_444, "r10k"},
        [R12L] = {"R12L", "12-bit packed RGB 4:4:4 little-endian", // SMPTE 268M DPX v1, Annex C, Method C4
                to_fourcc('R','1','2','l'), 36, 8, 8, 12, VCF_RGB, SUBS_444, "r12l"},
        [v210] = {"v210", "10-bit YUV 4:2:2",
                to_fourcc('v','2','1','0'), 16, 6, 48, 10, VCF_NONE, SUBS_422, "v210"},
        [DVS10] = {"DVS10", "Centaurus 10bit YUV 4:2:2",
                to_fourcc('D','S','1','0'), 16, 6, 48, 10, VCF_NONE, SUBS_422, "dvs10"},
        [DXT1] = {"DXT1", "S3 Compressed Texture DXT1",
                to_fourcc('D','X','T','1'), 1, 2, 0, 2, VCF_RGB, VC_OPAQUE, "dxt1"},
        /// packed YCbCr inside DXT1 channels
        [DXT1_YUV] = {"DXT1_YUV", "S3 Compressed Texture DXT1 YUV",
                to_fourcc('D','X','T','Y'), 1, 2, 0, 2, VCF_NONE, VC_OPAQUE, "dxt1y"},
        [DXT5] = {"DXT5", "S3 Compressed Texture DXT5 YCoCg",
                to_fourcc('D','X','T','5'), 1, 1, 0, 4, VCF_NONE, VC_OPAQUE, "yog"},/* DXT5 YCoCg */
        [RGB] = {"RGB", "Red Green Blue 24bit",
                to_fourcc('R','G','B','2'), 3, 1, 1, 8, VCF_RGB, SUBS_444, "rgb"},
        [JPEG] = {"JPEG",  "JPEG",
                to_fourcc('J','P','E','G'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "jpg"},
        [RAW] = {"raw", "Raw SDI video",
                to_fourcc('r','a','w','s'), 1, 1, 0, 0, VCF_NONE, VC_OPAQUE, "raw"}, /* raw SDI */
        [H264] = {"H.264", "H.264/AVC",
                to_fourcc('A','V','C','1'), 1, 1, 0, 8, VCF_INTERFRAME, VC_OPAQUE, "h264"},
        [H265] = {"H.265", "H.265/HEVC",
                to_fourcc('H','E','V','C'), 1, 1, 0, 8, VCF_INTERFRAME, VC_OPAQUE, "h265"},
        [VP8] = {"VP8", "Google VP8",
                to_fourcc('V','P','8','0'), 1, 1, 0, 8, VCF_INTERFRAME, VC_OPAQUE, "vp8"},
        [VP9] = {"VP9", "Google VP9",
                to_fourcc('V','P','9','0'), 1, 1, 0, 8, VCF_INTERFRAME, VC_OPAQUE, "vp9"},
        [BGR] = {"BGR", "Blue Green Red 24bit",
                to_fourcc('B','G','R','2'), 3, 1, 1, 8, VCF_RGB, SUBS_444, "bgr"},
        [J2K] = {"J2K", "JPEG 2000",
                to_fourcc('M','J','2','C'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "j2k"},
        [J2KR] = {"J2KR", "JPEG 2000 RGB",
                to_fourcc('M','J','2','R'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "j2k"},
        [HW_VDPAU] = {"HW_VDPAU", "VDPAU hardware surface",
                to_fourcc('V', 'D', 'P', 'S'), HW_VDPAU_FRAME_SZ, 1, 0, 8, VCF_CONST_SIZE, VC_OPAQUE, "vdpau"},
        [HFYU] = {"HFYU", "HuffYUV",
                to_fourcc('H','F','Y','U'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "hfyu"},
        [FFV1] = {"FFV1", "FFV1",
                to_fourcc('F','F','V','1'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "ffv1"},
        [CFHD] = {"CFHD", "Cineform",
                to_fourcc('C','F','H','D'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "cfhd"},
        [RG48] = {"RG48", "16-bit RGB little-endian",
                to_fourcc('R','G','4','8'), 6, 1, 1, 16, VCF_RGB, SUBS_444, "rg48"},
        [AV1] =  {"AV1", "AOMedia Video 1",
                to_fourcc('a','v','0','1'), 1, 1, 0, 8, VCF_RGB, VC_OPAQUE, "av1"},
        [I420] =  {"I420", "planar YUV 4:2:0",
                to_fourcc('I','4','2','0'), 3, 2, 2, 8, VCF_NONE, SUBS_420, "yuv"},
        [Y216] =  {"Y216", "Packed 16-bit YUV 4:2:2 little-endian",
                to_fourcc('Y','2','1','6'), 8, 2, 2, 16, VCF_NONE, SUBS_422, "y216"},
        [Y416] =  {"Y416", "Packed 16-bit YUV 4:4:4:4 little-endian",
                to_fourcc('Y','4','1','6'), 8, 1, 1, 16, VCF_NONE, SUBS_4444, "y416"},
        [PRORES] =  {"PRORES", "Apple ProRes",
                0, 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "pror"},
        [PRORES_4444] =  {"PRORES_4444", "Apple ProRes 4444",
                to_fourcc('a','p','4','h'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "ap4h"},
        [PRORES_4444_XQ] =  {"PRORES_4444_XQ", "Apple ProRes 4444 (XQ)",
                to_fourcc('a','p','4','x'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "ap4x"},
        [PRORES_422_HQ] =  {"PRORES_422_HQ", "Apple ProRes 422 (HQ)",
                to_fourcc('a','p','c','h'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "apch"},
        [PRORES_422] =  {"PRORES_422", "Apple ProRes 422",
                to_fourcc('a','p','c','n'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "apcn"},
        [PRORES_422_PROXY] =  {"PRORES_422_PROXY", "Apple ProRes 422 (Proxy)",
                to_fourcc('a','p','c','o'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "apco"},
        [PRORES_422_LT] =  {"PRORES_422_LT", "Apple ProRes 422 (LT)",
                to_fourcc('a','p','c','s'), 1, 1, 0, 8, VCF_NONE, VC_OPAQUE, "apcs"},
        [DRM_PRIME] = {"DRM_PRIME", "DRM Prime buffer",
                to_fourcc('D', 'R', 'M', 'P'), sizeof(struct drm_prime_frame), 1, 0, 8, VCF_CONST_SIZE, VC_OPAQUE, "drm_prime"},
};

/// for planar pixel formats
struct pixfmt_plane_info_t {
        int plane_info[8];              ///< [1st comp H subsamp, 1st comp V subs., 2nd comp H....]
};

static const struct pixfmt_plane_info_t pixfmt_plane_info[] = {
        [I420] = {{1, 1, 2, 2, 2, 2, 0, 0}},
        [VIDEO_CODEC_END] = {{0}}, // end must be present to all codecs to have the metadata defined
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
        // the following two are here because it was sent with wrong endianness in past
        {to_fourcc('A', 'B', 'G', 'R'), to_fourcc('R', 'G', 'B', 'A')},
        {to_fourcc('2', 'B', 'G', 'R'), to_fourcc('R', 'G', 'B', '2')},

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
        {"H264", "H.264"},
        {"H265", "H.265"},
        {"HEVC", "H.265"},
        {"MJPG", "JPEG"},
};

void show_codec_help(const char *module, const codec_t *codecs8, const codec_t *codecs10, const codec_t *codecs_ge12)
{
        printf("Supported codecs (%s):\n", module);

        if (codecs8) {
                printf("\t8bits\n");

                while (*codecs8 != VIDEO_CODEC_NONE) {
                        printf("\t\t'%s' - %s\n", codec_info[*codecs8].name, codec_info[*codecs8].name_long);
                        codecs8++;
                }
        }

        if (codecs10) {
                printf("\t10bits\n");
                while (*codecs10 != VIDEO_CODEC_NONE) {
                        printf("\t\t'%s' - %s\n", codec_info[*codecs10].name, codec_info[*codecs10].name_long);
                        codecs10++;
                }
        }

        if (codecs_ge12) {
                printf("\t12+ bits\n");
                while (*codecs_ge12 != VIDEO_CODEC_NONE) {
                        printf("\t\t'%s' - %s\n", codec_info[*codecs_ge12].name, codec_info[*codecs_ge12].name_long);
                        codecs_ge12++;
                }
        }
}

int get_bits_per_component(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].bits_per_channel;
        }
        assert(0);
}

/// @returns subsampling in format (int) JabA (A is alpha), eg 4440
int get_subsampling(codec_t codec)
{
        int subsampling = 0;
        if (codec < sizeof codec_info / sizeof(struct codec_info_t)) {
                subsampling = codec_info[codec].subsampling;
        }
        assert(subsampling != 0);
        return subsampling;
}

double get_bpp(codec_t codec)
{
        unsigned int i = (unsigned int) codec;
        double bpp = 0;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                bpp = (double) codec_info[i].block_size_bytes /
                      codec_info[i].block_size_pixels;
        }
        assert(bpp != 0);
        return bpp;
}

uint32_t get_fourcc(codec_t codec)
{
        unsigned int i = (unsigned int) codec;
        uint32_t fourcc = 0;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                fourcc = codec_info[i].fcc;
        }
        assert(fourcc != 0);
        return fourcc;
}

const char * get_codec_name(codec_t codec)
{
        unsigned int i = (unsigned int) codec;
        const char *name = NULL;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                name = codec_info[i].name;
        }
        assert(name != NULL);
        return name;
}

const char * get_codec_name_long(codec_t codec)
{
        unsigned int i = (unsigned int) codec;
        const char *name_long = NULL;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                name_long = codec_info[i].name_long;
        }
        assert(name_long != NULL);
        return name_long;
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

codec_t get_codec_from_file_extension(const char *ext)
{
        for (unsigned int i = 0; i < sizeof codec_info / sizeof(struct codec_info_t); ++i) {
                if (codec_info[i].file_extension && strcasecmp(codec_info[i].file_extension, ext) == 0) {
                        return i;
                }
        }

        return VIDEO_CODEC_NONE;
}

/**
 * @retval TRUE if codec is compressed
 * @retval FALSE if codec is pixelformat
 */
bool is_codec_opaque(codec_t codec)
{
        assert(codec != VC_NONE);
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return codec_info[i].subsampling == VC_OPAQUE;
        }
        return false;
}

/**
 * Returns whether specified codec is an interframe compression.
 * Not defined for pixelformats
 * @retval TRUE if compression is interframe
 * @retval FALSE if compression is not interframe
 */
bool is_codec_interframe(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return (codec_info[i].flags & VCF_INTERFRAME) != 0;
        }
        return false;
}

/** @brief Returns TRUE if specified pixelformat is some form of RGB (not YUV).
 *
 * Unspecified for compressed codecs.
 * @retval TRUE  if pixelformat is RGB
 * @retval FALSE if pixelformat is not a RGB */
bool codec_is_a_rgb(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return (codec_info[i].flags & VCF_RGB) != 0;
        }
        return false;
}

/** @brief Returns TRUE if specified pixelformat has constant size regardless
 * of resolution. If so the block_size value represents the size.
 *
 * Unspecified for compressed codecs.
 * @retval TRUE  if pixelformat is const size
 * @retval FALSE if pixelformat is not const size */
bool codec_is_const_size(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i < sizeof codec_info / sizeof(struct codec_info_t)) {
                return (codec_info[i].flags & VCF_CONST_SIZE) != 0;
        }
        return false;
}

bool codec_is_hw_accelerated(codec_t codec) {
        return codec == HW_VDPAU;
}

/**
 * @returns aligned linesize according to pixelformat specification (in bytes)
 *
 * @sa vc_get_size that should be used eg. as decoder_t linesize parameter instead
 * */
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
        int pixs = codec_info[codec].block_size_pixels;
        return (width + pixs - 1) / pixs * codec_info[codec].block_size_bytes;
}

/**
 * @returns size of "width" pixels in codec _excluding_ line padding
 *
 * This differs from vc_get_linesize for v210, eg. for width=1 that function
 * returns 128, while this function 16. Also for R10k, lines are aligned to
 * 256 B (64 pixels), while single pixel is only 4 B.
 */
int vc_get_size(unsigned int width, codec_t codec)
{
        if (codec >= sizeof codec_info / sizeof(struct codec_info_t)) {
                return 0;
        }

        int pixs = codec_info[codec].block_size_pixels;
        return (width + pixs - 1) / pixs * codec_info[codec].block_size_bytes;
}

/**
 * Returns storage requirements for given parameters
 */
size_t vc_get_datalen(unsigned int width, unsigned int height, codec_t codec)
{
        if (!codec_is_planar(codec)) {
                return vc_get_linesize(width, codec) * height;
        }

        assert(get_bits_per_component(codec) == 8);
        size_t ret = 0;
        int sub[8];
        codec_get_planes_subsampling(codec, sub);
        for (int i = 0; i < 4; ++i) {
                if (sub[i * 2] == 0) { // less than 4 planes
                        break;
                }
                ret += ((width + sub[i * 2] - 1) / sub[i * 2])
                        * ((height + sub[i * 2 + 1] - 1) / sub[i * 2 + 1]);
        }

        return ret;
}

/// @brief returns @ref codec_info_t::block_size_bytes
int get_pf_block_bytes(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i >= sizeof codec_info / sizeof(struct codec_info_t)) {
                abort();
        }
        assert(codec_info[i].block_size_bytes > 0);
        return codec_info[i].block_size_bytes;
}

/// @brief returns @ref codec_info_t::block_size_pixels
int get_pf_block_pixels(codec_t codec)
{
        unsigned int i = (unsigned int) codec;

        if (i >= sizeof codec_info / sizeof(struct codec_info_t)) {
                abort();
        }
        assert(codec_info[i].block_size_pixels > 0);
        return codec_info[i].block_size_pixels;
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
 * This allows to output to a different buffer while it can still be used in-place.
 *
 * @returns false on unsupported codecs
 */
// Sibling of this function is in double-framerate.cpp:avg_lines so consider
// porting changes made here there.
bool vc_deinterlace_ex(codec_t codec, unsigned char *src, size_t src_linesize, unsigned char *dst, size_t dst_pitch, size_t lines)
{
        if (is_codec_opaque(codec) && codec_is_planar(codec)) {
                return false;
        }
        if (lines == 1) {
                memcpy(dst, src, src_linesize);
                return true;
        }
        DEBUG_TIMER_START(vc_deinterlace_ex);
        int bpp = get_bits_per_component(codec);
        for (size_t y = 0; y < lines - 1; y += 1) {
                unsigned char *s = src + y * src_linesize;
                unsigned char *d = dst + y * dst_pitch;
                if (bpp == 8 || bpp == 16) {
                        size_t x = 0;
#ifdef __SSSE3__
                        if (bpp == 8) {
                                for ( ; x < src_linesize / 16; ++x) {
                                        __m128i i1 = _mm_lddqu_si128((__m128i const*)(const void *) s);
                                        __m128i i2 = _mm_lddqu_si128((__m128i const*)(const void *) (s + src_linesize));
                                        __m128i res = _mm_avg_epu8(i1, i2);
                                        _mm_storeu_si128((__m128i *)(void *) d, res);
                                        s += 16;
                                        d += 16;
                                }
                        } else {
                                for ( ; x < src_linesize / 16; ++x) {
                                        __m128i i1 = _mm_lddqu_si128((__m128i const*)(const void *) s);
                                        __m128i i2 = _mm_lddqu_si128((__m128i const*)(const void *) (s + src_linesize));
                                        __m128i res = _mm_avg_epu16(i1, i2);
                                        _mm_storeu_si128((__m128i *)(void *) d, res);
                                        s += 16;
                                        d += 16;
                                }
                        }
#endif
                        x *= 16;
                        if (bpp  == 8) {
                                for ( ; x < src_linesize; ++x) {
                                        int val = (*s + s[src_linesize] + 1) >> 1;
                                        *d++ = val;
                                        s++;
                                }
                        } else {
                                uint16_t *d16 = (void *) d;
                                uint16_t *s16 = (void *) s;
                                for ( ; x < src_linesize / 2; ++x) {
                                        int val = (*s16 + s16[src_linesize / 2] + 1) >> 1;
                                        *d16++ = val;
                                        s16++;
                                }
                        }
                } else if (codec == v210) {
                        uint32_t *s32 = (void *) s;
                        uint32_t *d32 = (void *) d;
                        for (size_t x = 0; x < src_linesize / 16; ++x) {
                                #pragma GCC unroll 4
                                for (size_t y = 0; y < 4; ++y) {
                                        uint32_t v1 = *s32;
                                        uint32_t v2 = s32[src_linesize / 4];
                                        uint32_t out =
                                                (((v1 >> 20        ) + (v2 >> 20        ) + 1) / 2) << 20 |
                                                (((v1 >> 10 & 0x3ff) + (v2 >> 10 & 0x3ff) + 1) / 2) << 10 |
                                                (((v1       & 0x3ff) + (v2       & 0x3ff) + 1) / 2);
                                        *d32++ = out;
                                        s32++;
                                }
                        }
                } else if (codec == R10k) {
                        uint32_t *s32 = (void *) s;
                        uint32_t *d32 = (void *) d;
                        for (size_t x = 0; x < src_linesize / 16; ++x) {
                                #pragma GCC unroll 4
                                for (size_t y = 0; y < 4; ++y) {
                                        uint32_t v1 = be32toh(*s32);
                                        uint32_t v2 = be32toh(s32[src_linesize / 4]);
                                        uint32_t out =
                                                (((v1 >> 22        ) + (v2 >> 22        ) + 1) / 2) << 22 |
                                                (((v1 >> 12 & 0x3ff) + (v2 >> 12 & 0x3ff) + 1) / 2) << 12 |
                                                (((v1 >>  2 & 0x3ff) + (v2 >>  2 & 0x3ff) + 1) / 2) << 2;
                                        *d32++ = htobe32(out);
                                        s32++;
                                }
                        }
                } else if (codec == R12L) {
                        uint32_t *s32 = (void *) s;
                        uint32_t *d32 = (void *) d;
                        int shift = 0;
                        uint32_t remain1 = 0;
                        uint32_t remain2 = 0;
                        uint32_t out = 0;
                        for (size_t x = 0; x < src_linesize / 36; ++x) {
                                #pragma GCC unroll 8
                                for (size_t y = 0; y < 8; ++y) {
                                        uint32_t in1 = *s32;
                                        uint32_t in2 = s32[src_linesize / 4];
                                        if (shift > 0) {
                                                remain1 = remain1 | (in1 & ((1<<((shift + 12) % 32)) - 1)) << (32-shift);
                                                remain2 = remain2 | (in2 & ((1<<((shift + 12) % 32)) - 1)) << (32-shift);
                                                uint32_t ret = (remain1 + remain2 + 1) / 2;
                                                out |= ret << shift;
                                                *d32++ = out;
                                                out = ret >> (32-shift);
                                                shift = (shift + 12) % 32;
                                                in1 >>= shift;
                                                in2 >>= shift;
                                        }
                                        while (shift <= 32 - 12) {
                                                out |= ((((in1 & 0xfff) + (in2 & 0xfff)) + 1) / 2) << shift;
                                                in1 >>= 12;
                                                in2 >>= 12;
                                                shift += 12;
                                        }
                                        if (shift == 32) {
                                                *d32++ = out;
                                                out = 0;
                                                shift = 0;
                                        } else {
                                                remain1 = in1;
                                                remain2 = in2;
                                        }
                                        s32++;
                                }
                        }
                } else {
                        return false;
                }
        }
        memcpy(dst + (lines - 1) * dst_pitch, dst + (lines - 2) * dst_pitch, src_linesize); // last line
        DEBUG_TIMER_STOP(vc_deinterlace_ex);
        return true;
}

/**
 * Tries to find specified codec in set of video codecs.
 * The set must by ended by VIDEO_CODEC_NONE.
 */
bool codec_is_in_set(codec_t codec, const codec_t *set)
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

/**
 * @returns true if codec is a pixel format and is planar
 */
bool codec_is_planar(codec_t codec) {
        return pixfmt_plane_info[codec].plane_info[0] != 0;
}

/**
 * Returns subsampling of individual planes of planar pixel format
 *
 * Undefined if pix_fmt is not a planar pixel format
 *
 * @param[out] sub   subsampling, allocated array must be able to hold
 *                   8 members
 */
void codec_get_planes_subsampling(codec_t pix_fmt, int *sub) {
        for (size_t i = 0; i < 8; ++i) {
                *sub++ = pixfmt_plane_info[pix_fmt].plane_info[i];
        }
}

bool codec_is_420(codec_t pix_fmt)
{
        return pixfmt_plane_info[pix_fmt].plane_info[0] == 1 &&
                pixfmt_plane_info[pix_fmt].plane_info[1] == 1 &&
                pixfmt_plane_info[pix_fmt].plane_info[2] == 2 &&
                pixfmt_plane_info[pix_fmt].plane_info[3] == 2 &&
                pixfmt_plane_info[pix_fmt].plane_info[4] == 2 &&
                pixfmt_plane_info[pix_fmt].plane_info[5] == 2;
}

void
uyvy_to_i422(int width, int height, const unsigned char *in, unsigned char *out)
{
        unsigned char *out_y  = out;
        unsigned char *out_cb = out + width * height;
        unsigned char *out_cr =
            out + width * height + ((width + 1) / 2) * height;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width / 2; ++x) {
                        *out_cb++ = *in++;
                        *out_y++  = *in++;
                        *out_cr++ = *in++;
                        *out_y++  = *in++;
                }
                if (width % 2 == 1) {
                        *out_cb++ = *in++;
                        *out_y++  = *in++;
                        *out_cr++ = *in++;
                }
        }
}

void
y416_to_i444(int width, int height, const unsigned char *in, unsigned char *out,
             int depth)
{
        const uint16_t *inp    = (const uint16_t *) (const void *) in;
        uint16_t       *out_y  = (uint16_t *) (void *) out;
        uint16_t       *out_cb = (uint16_t *) (void *) out + width * height;
        uint16_t       *out_cr = (uint16_t *) (void *) out + 2 * width * height;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        *out_cb++ = *inp++ >> (16 - depth);
                        *out_y++  = *inp++ >> (16 - depth);
                        *out_cr++ = *inp++ >> (16 - depth);
                        inp++; // alpha
                }
        }
}

void
i444_16_to_y416(int width, int height, const unsigned char *in,
                unsigned char *out, int in_depth)
{
        const uint16_t *in_y = (const uint16_t *)(const void *) in;
        const uint16_t *in_cb = (const uint16_t *)(const void *) in + width * height;
        const uint16_t *in_cr = (const uint16_t *)(const void *) in + 2 * width * height;
        uint16_t *outp = (uint16_t *)(void *) out;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        *outp++ = *in_cb++ << (16 - in_depth);
                        *outp++ = *in_y++ << (16 - in_depth);
                        *outp++ = *in_cr++ << (16 - in_depth);
                        *outp++ = 0xFFFFU; // alpha
                }
        }
}

void
i422_16_to_y416(int width, int height, const unsigned char *in,
                unsigned char *out, int in_depth)
{
        enum { ALPHA16_OPAQUE = 0xFFFF };
        const uint16_t *in_y  = (const uint16_t *) (const void *) in;
        const uint16_t *in_cb = in_y + (ptrdiff_t) width * height;
        const uint16_t *in_cr = in_cb + (ptrdiff_t) ((width + 1) / 2) * height;
        uint16_t       *outp  = (void *) out;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < (width + 1) / 2; ++x) {
                        *outp++ = *in_cb << (DEPTH16 - in_depth);
                        *outp++ = *in_y++ << (DEPTH16 - in_depth);
                        *outp++ = *in_cr << (DEPTH16 - in_depth);
                        *outp++ = ALPHA16_OPAQUE;
                        *outp++ = *in_cb++ << (DEPTH16 - in_depth);
                        *outp++ = *in_y++ << (DEPTH16 - in_depth);
                        *outp++ = *in_cr++ << (DEPTH16 - in_depth);
                        *outp++ = ALPHA16_OPAQUE;
                }
        }
}

void
i420_16_to_y416(int width, int height, const unsigned char *in,
                unsigned char *out, int in_depth)
{
        enum { ALPHA16_OPAQUE = 0xFFFF };
        const uint16_t *in_y1 = (const void *) in;
        const uint16_t *in_y2 = in_y1 + width;
        const uint16_t *in_cb =
            (const uint16_t *) (const void *) in + (ptrdiff_t) width * height;
        const uint16_t *in_cr =
            in_cb + (ptrdiff_t) ((width + 1) / 2) * ((height + 1) / 2);
        const size_t out_line_len = vc_get_linesize(width, Y416) / 2;
        uint16_t    *outp1        = (void *) out;
        uint16_t    *outp2        = outp1 + out_line_len;
        for (int y = 0; y < (height + 1) / 2; ++y) {
                for (int x = 0; x < (width + 1) / 2; ++x) {
                        *outp1++ = *in_cb << (DEPTH16 - in_depth);
                        *outp1++ = *in_y1++ << (DEPTH16 - in_depth);
                        *outp1++ = *in_cr << (DEPTH16 - in_depth);
                        *outp1++ = ALPHA16_OPAQUE;

                        *outp2++ = *in_cb << (DEPTH16 - in_depth);
                        *outp2++ = *in_y2++ << (DEPTH16 - in_depth);
                        *outp2++ = *in_cr << (DEPTH16 - in_depth);
                        *outp2++ = ALPHA16_OPAQUE;

                        *outp1++ = *in_cb << (DEPTH16 - in_depth);
                        *outp1++ = *in_y1++ << (DEPTH16 - in_depth);
                        *outp1++ = *in_cr << (DEPTH16 - in_depth);
                        *outp1++ = ALPHA16_OPAQUE;

                        *outp2++ = *in_cb++ << (DEPTH16 - in_depth);
                        *outp2++ = *in_y2++ << (DEPTH16 - in_depth);
                        *outp2++ = *in_cr++ << (DEPTH16 - in_depth);
                        *outp2++ = ALPHA16_OPAQUE;
                }
                outp1 += out_line_len;
                outp2 += out_line_len;
                in_y1 += width;
                in_y2 += width;
        }
}
void
i420_8_to_uyvy(int width, int height, const unsigned char *in, unsigned char *out)
{
        const int                  uyvy_linesize = vc_get_linesize(width, UYVY);
        const unsigned char *const cb = in + (ptrdiff_t) width * height;
        const unsigned char *const cr =
            cb + (ptrdiff_t) (width / 2) * (height / 2);

        for (ptrdiff_t y = 0; y < height; ++y) {
                const unsigned char *src_y  = in + width * y;
                const unsigned char *src_cb = cb + (width / 2) * (y / 2);
                const unsigned char *src_cr = cr + (width / 2) * (y / 2);
                unsigned char       *dst    = out + y * uyvy_linesize;

                OPTIMIZED_FOR(int x = 0; x < width / 2; ++x)
                {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

void
i422_8_to_uyvy(int width, int height, const unsigned char *in,
               unsigned char *out)
{
        const unsigned char *in_y  = in;
        const unsigned char *in_cb = in + width * height;
        const unsigned char *in_cr = in_cb + (((width + 1) / 2) * height);
        unsigned char       *outp  = out;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < (width + 1) / 2; ++x) {
                        *outp++ = *in_cb++;
                        *outp++ = *in_y++;
                        *outp++ = *in_cr++;
                        *outp++ = *in_y++;
                }
        }
}

void
i444_8_to_uyvy(int width, int height, const unsigned char *in,
               unsigned char *out)
{
        const unsigned char *in_y = in;
        const unsigned char *in_cb = in + width * height;
        const unsigned char *in_cr = in_cb + width * height;
        unsigned char *outp = out;
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < (width + 1) / 2; ++x) {
                        *outp++ = *in_cb;
                        *outp++ = *in_y++;
                        *outp++ = *in_cr;
                        *outp++ = *in_y++;
                        in_cb += 2;
                        in_cr += 2;
                }
        }
}

struct pixfmt_desc get_pixfmt_desc(codec_t pixfmt)
{
        assert(pixfmt >= VIDEO_CODEC_FIRST);
        assert(pixfmt <= VIDEO_CODEC_END);
        struct pixfmt_desc ret = { 0 };
        ret.depth = codec_info[pixfmt].bits_per_channel;
        ret.subsampling = codec_info[pixfmt].subsampling;
        ret.rgb = (codec_info[pixfmt].flags & VCF_RGB) != 0;
        return ret;
}

/**
 * qsort(_s)-compatible comparator
 */
int compare_pixdesc(const struct pixfmt_desc *desc_a, const struct pixfmt_desc *desc_b, const struct pixfmt_desc *src_desc)
{
        for (char *feature = pixfmt_conv_pref; *feature != '\0'; ++feature) {
                switch (*feature) {
                        case 'd':
                                if (desc_a->depth != desc_b->depth &&
                                                (desc_a->depth < src_desc->depth || desc_b->depth < src_desc->depth)) { // either a or b is lower than orig - sort higher bit depth first
                                        return desc_b->depth - desc_a->depth;
                                }
                                break;
                        case 's':
                                if (desc_a->subsampling != desc_b->subsampling &&
                                                (desc_a->subsampling < src_desc->subsampling || desc_b->subsampling < src_desc->subsampling)) {
                                        return desc_b->subsampling - desc_a->subsampling; // return better subs
                                }
                                break;
                        case 'c':
                                if (desc_a->rgb != desc_b->rgb) {
                                        return desc_a->rgb == src_desc->rgb ? -1 : 1;
                                }
                                break;
                }
        }

        // if both A and B are either undistinguishable or better than src, return closer ("worse") pixfmt
        for (char *feature = pixfmt_conv_pref; *feature != '\0'; ++feature) {
                switch (*feature) {
                        case 'd':
                                if (desc_a->depth != desc_b->depth) {
                                        return desc_a->depth - desc_b->depth;
                                }
                                break;
                        case 's':
                                if (desc_a->subsampling != desc_b->subsampling) {
                                        return desc_a->subsampling - desc_b->subsampling;
                                }
                                break;
                        case 'c':
                                // will be a tie here
                                break;
                }
        }

        return 0;
}

void watch_pixfmt_degrade(const char *mod_name, struct pixfmt_desc desc_src, struct pixfmt_desc desc_dst)
{
        char message[1024];
        message[0] = '\0';
        if (desc_dst.depth < desc_src.depth) {
                snprintf(message, sizeof message, "conversion is reducing bit depth from %d to %d", desc_src.depth, desc_dst.depth);
        }
        if (desc_dst.subsampling < desc_src.subsampling) {
                if (strlen(message) > 0) {
                        snprintf(message + strlen(message), sizeof message - strlen(message), " and subsampling from %d to %d", desc_src.subsampling, desc_dst.subsampling);
                } else {
                        snprintf(message, sizeof message, "conversion is reducing subsampling from %d to %d", desc_src.subsampling, desc_dst.subsampling);
                }
        }
        if (strlen(message) > 0) {
                log_msg(LOG_LEVEL_WARNING, "%s%s\n", mod_name, message);
        }
}

const char *get_pixdesc_desc(struct pixfmt_desc desc)
{
        if (desc.depth == 0) {
                return "(undefined)";
        }
        _Thread_local static char buf[128];
        snprintf(buf, sizeof buf, "%s %d:%d:%d", desc.rgb ? "RGB" : "YCbCr", desc.subsampling / 1000, desc.subsampling % 1000 / 100, desc.subsampling % 100 / 10);
        if (desc.subsampling % 10 != 0) {
                snprintf(buf + strlen(buf), sizeof buf - strlen(buf), ":%d", desc.subsampling % 10);
        }
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf), " %d bit", desc.depth);
        return buf;
}

bool pixdesc_equals(struct pixfmt_desc desc_a, struct pixfmt_desc desc_b) {
        return desc_a.depth == desc_b.depth &&
                desc_a.subsampling == desc_b.subsampling &&
                desc_a.rgb == desc_b.rgb;
}

/* vim: set expandtab sw=8: */

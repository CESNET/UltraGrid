/*
 * FILE:    video.h
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
#ifndef __video_h
#define __video_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Video codec identificator
 * @see video_frame::color_spec
 */
/**
 * @var codec_t::MJPG
 * May include features not compatible with GPUJPEG.
 */
typedef enum {
        RGBA,     ///< RGBA 8-bit
        UYVY,     ///< YCbCr 422 8-bit - Cb Y0 Cr Y1
        YUYV,     ///< YCbCr 422 8-bit - Y0 Cb Y1 Cr
        Vuy2,     ///< YCbCr 422 8-bit (same as UYVY)
        DVS8,     ///< YCbCr 422 8-bit (same as UYVY)
        R10k,     ///< RGB 10-bit (also know as r210)
        v210,     ///< YCbCr 422 10-bit
        DVS10,    ///< DVS 10-bit format
        DXT1,     ///< S3 Texture Compression DXT1
        DXT1_YUV, ///< Structure same as DXT1, instead of RGB, YCbCr values are stored
        DXT5,     ///< S3 Texture Compression DXT5
        RGB,      ///< RGBA 8-bit (packed into 24-bit word)
        DPX10,    ///< 10-bit DPX raw data
        JPEG,     ///< JPEG image, restart intervals may be used. Compatible with GPUJPEG
        RAW,      ///< RAW HD-SDI frame
        H264,     ///< H.264 frame
        MJPG,     ///< JPEG image, without restart intervals.
        VP8,      ///< VP8 frame
        BGR       ///< 8-bit BGR
} codec_t;

/**
 * Specifies interlacing mode of the frame
 * @see video_frame::interlacing
 */
/**
 * @var interlacing_t::SEGMENTED_FRAME
 * @note
 * This is included to allow describing video mode, not content
 * of a frame.
 */
enum interlacing_t {
        PROGRESSIVE       = 0, ///< progressive frame
        UPPER_FIELD_FIRST = 1, ///< First stored field is top, followed by bottom
        LOWER_FIELD_FIRST = 2, ///< First stored field is bottom, followed by top
        INTERLACED_MERGED = 3, ///< Columngs of both fields are interlaced together
        SEGMENTED_FRAME   = 4  ///< Segmented frame. Contains the same data as progressive frame.
};

/** @defgroup video_mode Video Mode */
/**@{*/
#define VIDEO_NORMAL                    0u ///< normal video (one tile)
#define VIDEO_DUAL                      1u ///< 1x2 video grid (is this ever used?)
#define VIDEO_STEREO                    2u ///< stereoscopic 3D video (full left and right eye)
#define VIDEO_4K                        3u ///< tiled 4K video
/**@}*/

/** @defgroup video_param Video Parameter */
/**@{*/
#define PARAM_WIDTH                     (1<<0u)
#define PARAM_HEIGHT                    (1<<2u)
#define PARAM_CODEC                     (1<<3u)
#define PARAM_INTERLACING               (1<<4u)
#define PARAM_FPS                       (1<<5u)
#define PARAM_TILE_COUNT                (1<<6u)
/**@}*/

/* please note that tiles have also its own widths and heights */
struct video_desc {
        /* in case of tiled video - width and height represent widht and height
         * of each tile, eg. for tiled superHD 1920x1080 */
        unsigned int         width;
        unsigned int         height;

        codec_t              color_spec;
        double               fps;
        enum interlacing_t   interlacing;
        unsigned int         tile_count;
};

struct video_frame {
        codec_t              color_spec;
        enum interlacing_t   interlacing;
        double               fps;
        struct tile         *tiles;

        unsigned int         tile_count;

        // Fragment stuff
        /// Indicates that the tile is fragmented. Normally not used (only for Bluefish444).
        unsigned int         fragment:1;
        /// Used only if (fragment == 0). Indicates this is the last fragment.
        unsigned int         last_fragment:1;
        /// Used only if (fragment == 0). ID of the frame. Fragments of same frame must have the same ID
        unsigned int         frame_fragment_id:14;
};

struct tile {
        unsigned int         width;
        unsigned int         height;

        /**
         * Raw tile data.
         * Pointer must be at least 4B aligned.
         */
        char                *data;
        unsigned int         data_len;
        unsigned int         linesize;

        /// @brief Fragment offset from tile beginning (in bytes). Used only if frame is fragmented.
        /// @see video_frame::fragment
        unsigned int         offset;
};

/**
 * @brief Allocates blank video frame
 * @param count number of allocated tiles
 * @return allocated video frame
 */
struct video_frame * vf_alloc(int count);
/**
 * @brief Allocates video frame accordig given desc
 * @param desc Description of video frame to be created
 *             At least desc.tile_count must be specified!
 * @return allocated video frame
 *         NULL if insufficient memory
 */
struct video_frame * vf_alloc_desc(struct video_desc desc);
/**
 * Same as vf_alloc_desc plus initializes data members of tiles
 *
 * @see vf_alloc_desc
 *
 * @param desc Description of video frame to be created
 * @return allocated video frame
 *         NULL if insufficient memory
 */
struct video_frame * vf_alloc_desc_data(struct video_desc desc);
/**
 * @brief Frees video frame excluding video data
 * @note
 * Video data is not freed
 */
void vf_free(struct video_frame *buf);
/**
 * @brief Frees video frame including video data
 * @see vf_free
 */
void vf_free_data(struct video_frame *buf);
/**
 * @brief Allocates blank tile
 * @return allocated tile
 */
struct tile * tile_alloc(void);
/**
 * @brief Allocates blank tile and sets width and height
 * @param desc video description
 * @return allocated tile
 */
struct tile *tile_alloc_desc(struct video_desc desc);
/**
 * @brief Frees video tile excluding video data
 * @note
 * Only tile allocated with tile_alloc should be freed
 */
void tile_free(struct tile*);
/**
 * @brief Frees video tile including video data
 * @see tile_free
 * @note
 * Only tile allocated with tile_alloc should be freed
 */
void tile_free_data(struct tile*);

/**
 * @brief Returns n-th tile from video frame
 */
struct tile * vf_get_tile(struct video_frame *buf, int pos);
/**
 * @brief Makes deep copy of the video frame
 */
struct video_frame * vf_get_copy(struct video_frame *frame);
/**
 * @brief Compares two video descriptions.
 *
 * This is equivalent to
 * @ref video_desc_eq_excl_param
 * when no parameter is excluded.
 * @retval 0 if different
 * @retval !=0 structs are the same
 */
int video_desc_eq(struct video_desc, struct video_desc);
/**
 * @brief Compares two video descriptions
 * @see video_desc_eq
 * @param excluded_params
 * bitwise OR of @ref video_param
 * @retval 0 if different
 * @retval !=0 structs are the same
 */
int video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params);
/**
 * @brief Returns struct video_desc from video frame
 */
struct video_desc video_desc_from_frame(struct video_frame *frame);
/**
 * @brief Returns vertical count of tiles
 * @param video_mode
 * One of @ref video_mode
 * @returns vertical count of tiles
 */
int get_video_mode_tiles_x(int video_mode);
/**
 * @brief Returns horizontal count of tiles
 * @param video_mode
 * One of @ref video_mode
 * @returns horizontal count of tiles
 */
int get_video_mode_tiles_y(int video_mode);
/**
 * @brief Returns description of interlacing
 * Eg. "progressive"
 */
const char *get_interlacing_description(enum interlacing_t interlacing);
/**
 * @brief Returns suffix describing interlacing
 * Eg. p, i or psf
 */
const char *get_interlacing_suffix(enum interlacing_t interlacing);
/**
 * @brief Returns description of video mode
 * Eg. "tiled 4K"
 * @param video_mode
 * One of @ref video_mode
 */
const char *get_video_mode_description(int video_mode);

/* these functions transcode one interlacing format to another */
/**
 * @brief Converts upper-field-first to interlaced merged.
 */
void il_upper_to_merged(char *dst, char *src, int linesize, int height);
/**
 * @brief Converts interlaced merged to upper-field-first.
 */
void il_merged_to_upper(char *dst, char *src, int linesize, int height);

/**
 * @brief Computes FPS as a double from packet fields.
 *
 * Individual field semantics can be found in paper referenced from
 * @ref av_pkt_description
 */
double compute_fps(int fps, int fpsd, int fd, int fi);

/** @defgroup video_flags Video Flags */
/**@{*/
#define AUX_INTERLACED  (1<<0)
#define AUX_PROGRESSIVE (1<<1)
#define AUX_SF          (1<<2)
#define AUX_RGB         (1<<3) /* if device supports both, set both */
#define AUX_YUV         (1<<4)
#define AUX_10Bit       (1<<5)
/**@}*/

#ifdef __cplusplus
}
#endif

#endif


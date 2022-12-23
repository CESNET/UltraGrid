/**
 * @file    types.h
 * @author  Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief   This file contains some common data types.
 *
 * This file should be directly included only from header files,
 * not implementation files.
 */
/*
 * Copyright (c) 2013-2021 CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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
 */

#ifndef TYPES_H_
#define TYPES_H_

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <string>
extern "C" {
#endif

/**
 * @enum codec_t
 * Video codec identificator
 * @see video_frame::color_spec
 */
/** @var codec_t::MJPG
 * @note May include features not compatible with GPUJPEG.
 */
typedef enum {
        VIDEO_CODEC_NONE = 0, ///< dummy color spec
        VIDEO_CODEC_FIRST,
        RGBA = VIDEO_CODEC_FIRST, ///< RGBA 8-bit, big-endian
        UYVY,     ///< YCbCr 422 8-bit - Cb Y0 Cr Y1
        YUYV,     ///< YCbCr 422 8-bit - Y0 Cb Y1 Cr
        R10k,     ///< RGB 10-bit packed - RGBX, big-endian (2 bit padding)
        R12L,     ///< RGB 12-bit packed, little-endian
        v210,     ///< YCbCr 422 10-bit - xVY0U|xY2UY1|xUY3V|xY5VY4, little-endian (2 bit padding /'x'/), line aligned to 128 B
        DVS10,    ///< DVS 10-bit format
        DXT1,     ///< S3 Texture Compression DXT1
        DXT1_YUV, ///< Structure same as DXT1, instead of RGB, YCbCr values are stored
        DXT5,     ///< S3 Texture Compression DXT5
        RGB,      ///< RGB 8-bit (packed into 24-bit word)
        DPX10,    ///< 10-bit DPX raw data
        JPEG,     ///< JPEG image, restart intervals may be used. Compatible with GPUJPEG
        RAW,      ///< RAW HD-SDI frame
        H264,     ///< H.264 frame
        H265,     ///< H.264 frame
        MJPG,     ///< JPEG image, without restart intervals.
        VP8,      ///< VP8 frame
        VP9,      ///< VP9 frame
        BGR,      ///< 8-bit BGR
        J2K,      ///< JPEG 2000
        J2KR,     ///< JPEG 2000 RGB
        HW_VDPAU, ///< VDPAU hardware surface
        RPI4_8,   ///< Raspberry pi 4 hw decoded (SAND)
        HFYU,     ///< HuffYUV
        FFV1,     ///< FFV1
        CFHD,     ///< Cineform
        RG48,     ///< 16-bit RGB little-endian (RlRhGl...)
        AV1,      ///< AOMedia Video 1
        I420,     ///< planar YCbCr 4:2:0
        Y216,     ///< YCbCr 422 16-bit - Y0 Cb Y1 Cr
        Y416,     ///< interleaved little-endian YCbCr 4444 16-bit - UYVA
        PRORES,           ///< abstract Apple ProRes, must not be used in transmit
        PRORES_4444,      ///< Apple ProRes 4444
        PRORES_4444_XQ,   ///< Apple ProRes 4444 (XQ)
        PRORES_422_HQ,    ///< Apple ProRes 422 (HQ)
        PRORES_422,       ///< Apple ProRes 422
        PRORES_422_PROXY, ///< Apple ProRes 422 (Proxy)
        PRORES_422_LT,    ///< Apple ProRes 422 (LT)
        VIDEO_CODEC_COUNT, ///< count of known video codecs (including VIDEO_CODEC_NONE)
        VIDEO_CODEC_END = VIDEO_CODEC_COUNT
} codec_t;

/**
 * @enum interlacing_t
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
        SEGMENTED_FRAME   = 4,  ///< Segmented frame. Contains the same data as progressive frame.
        INTERLACING_MAX   = SEGMENTED_FRAME,
};

/**
 * video_desc represents video description
 * @note
 * in case of tiled video - width and height represent widht and height
 * of each tile, eg. for tiled superHD 1920x1080
 */
struct video_desc {
        unsigned int         width;  ///< width of every tile (!)
        unsigned int         height; ///< height of every tile (!)

        codec_t              color_spec;
        double               fps;
        enum interlacing_t   interlacing;
        unsigned int         tile_count;
#ifdef __cplusplus
        bool operator==(video_desc const &) const;
        bool operator!=(video_desc const &) const;
        bool operator!() const;
        operator std::string() const;
#endif
};

typedef enum frame_type {
    INTRA = 0,
    BFRAME,
    OTHER
} frame_type_t;

enum fec_type {
        FEC_NONE = 0,
        FEC_MULT = 1,
        FEC_LDGM = 2,
        FEC_RS   = 3,
};

struct fec_desc {
        enum fec_type type;
        unsigned int k, m, c;
        unsigned int seed;
        unsigned int symbol_size;
        unsigned int mult;
#ifdef __cplusplus
        fec_desc() = default;
        inline fec_desc(enum fec_type type_, unsigned int k_ = 0, unsigned int m_ = 0,
                        unsigned int c_ = 0,
                        unsigned int seed_ = 0,
                        unsigned int ss_ = 0,
                        unsigned int mult_ = 0) : type(type_), k(k_), m(m_), c(c_), seed(seed_), symbol_size(ss_), mult(mult_) {}
#endif
};

struct video_frame;
/**
 * @brief Struct containing callbacks of a @ref video_frame
 */
struct video_frame_callbacks {
        /** @note
         * Can be changed only by frame originator.
         * @deprecated
         * This is currently used only by video_capture and capture_filter modules
         * and should not be used elsewhere!
         * @{
         */
        /**
         * This function (if defined) is called when frame is no longer needed
         * by processing queue.
         * @note
         * Currently, this is only used in sending workflow, not the receiving one!
         * Can be called from arbitrary thread.
         */
        void               (*dispose)(struct video_frame *);
        /**
         * Additional data needed to dispose the frame
         */
        void                *dispose_udata;
        /// @}

        /**
         * This function (if defined) is called by vf_free() to destruct video data
         * (@ref tile::data members).
         */
        void               (*data_deleter)(struct video_frame *);

        /**
         * This function is used to free extra data held by the frame and should
         * be called before returning the frame to frame pool. 
         * This function is currently used to free references to hw surfaces.
         */
        void               (*recycle)(struct video_frame *);

        /**
         * This function is used to copy extra data held by the frame and is
         * automatically called by the vf_get_copy function.
         * This function is currently used to make new references to hw surfaces
         * when creating copies of hw frames.
         */
        void               (*copy)(struct video_frame *);
};

/**
 * @brief Struct tile is an area of video_frame.
 * @note
 * Currently all tiles of a frame should have the same width and height.
 */
struct tile {
        unsigned int         width;
        unsigned int         height;

        /**
         * @brief Raw tile data.
         * Pointer must be at least 4B aligned.
         */
        char                *data;
        unsigned int         data_len; ///< length of the data

        /// @brief Fragment offset from tile beginning (in bytes). Used only if frame is fragmented.
        /// @see video_frame::fragment
        unsigned int         offset;
};

#define FLEXIBLE_ARRAY_MEMBER 0

enum mem_location_t {
        CPU_MEM = 0,
        CUDA_MEM
};

/**
 * @brief Struct video_frame represents a video frame and contains video description.
 */
struct video_frame {
        codec_t              color_spec;
        enum interlacing_t   interlacing;
        double               fps;
        frame_type_t         frame_type;
        enum mem_location_t  mem_location;

        /** @name Fragment Stuff 
         * @{ */
        /// Indicates that the tile is fragmented. Normally not used (only for Bluefish444).
        unsigned int         fragment:1;
        /// Used only if (fragment == 1). Indicates this is the last fragment.
        unsigned int         last_fragment:1;
        /// Used only if (fragment == 1). ID of the frame. Fragments of same frame must have the same ID
        unsigned int         frame_fragment_id:14;
        /// @}

        unsigned int         decoder_overrides_data_len:1;

        struct video_frame_callbacks callbacks;

        // metadata follow
#define VF_METADATA_START fec_params
        struct fec_desc fec_params;
        uint32_t ssrc;

        uint32_t seq; ///< sequential number, used internally by JPEG encoder
        uint32_t timecode; ///< BCD timecode (hours, minutes, seconds, frame number)
        uint64_t compress_start; ///< in ms from epoch
        uint64_t compress_end; ///< in ms from epoch
        unsigned int paused_play:1;
#define VF_METADATA_END tile_count

        /// tiles contain actual video frame data. A frame usually contains exactly one
        /// tile but in some cases it can contain more tiles (eg. 4 for tiled 4K).
        unsigned int         tile_count;
        struct tile          tiles[FLEXIBLE_ARRAY_MEMBER];
};

#define VF_METADATA_SIZE (offsetof(struct video_frame, VF_METADATA_END) - offsetof(struct video_frame, VF_METADATA_START))

/** Video Mode
 *
 * Video mode metadata are stored in file video.c in @ref video_mode_info.
 */
enum video_mode {
        VIDEO_UNKNOWN, ///< unspecified video mode
        VIDEO_NORMAL,  ///< normal video (one tile)
        VIDEO_DUAL,    ///< 1x2 video grid (is this ever used?)
        VIDEO_STEREO,  ///< stereoscopic 3D video (full left and right eye)
        VIDEO_4K,      ///< tiled 4K video
        VIDEO_3X1,     ///< 3x1 video
};

enum tx_media_type {
        TX_MEDIA_AUDIO,
        TX_MEDIA_VIDEO
};

struct device_info {
        char dev[1024];  ///< device options to be passed to UltraGrid to initialize
                         ///<  (eg.r ":device=0" for DeckLink). May be empty ("").
        char name[1024]; ///< human readable name of the device
        bool repeatable; ///< Whether can be card used multiple times (eg. GL) or it
                         ///< can output simoultaneously only one output (DeckLink).
                         ///< Used for video display only.
        char extra[1024];///< Json object containing extra data reported with capabilities
                         ///< currently used to report embedded audio availability
        struct mode {    ///< optional zero-terminated array of available modes
                char id[1024];   ///< options to be passed to UltraGrid to initialize the device
                                 ///< with the appropriate mode (eg. "mode=Hi50" for DeckLink 0).
                                 ///< Must not be empty (!)
                char name[1024]; ///< human readable name of the mode
        } modes[512];
};

struct vidcap_params;

#ifdef __cplusplus
}
#endif

#endif // TYPES_H_


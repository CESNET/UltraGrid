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
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#ifdef __cplusplus
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
        RGBA,     ///< RGBA 8-bit
        UYVY,     ///< YCbCr 422 8-bit - Cb Y0 Cr Y1
        YUYV,     ///< YCbCr 422 8-bit - Y0 Cb Y1 Cr
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
        BGR,      ///< 8-bit BGR
        J2K,      ///< JPEG 2000
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
};

typedef enum h264_frame_type {
    INTRA,
    BFRAME,
    OTHER
} h264_frame_type_t;

enum fec_type {
        FEC_NONE = 0,
        FEC_MULT = 1,
        FEC_LDGM = 2,
};

struct fec_desc {
        enum fec_type type;
        unsigned int k, m, c;
        unsigned int seed;
        unsigned int symbol_size;
#ifdef __cplusplus
        fec_desc() = default;
        inline fec_desc(enum fec_type type_, unsigned int k_ = 0, unsigned int m_ = 0,
                        unsigned int c_ = 0,
                        unsigned int seed_ = 0,
                        unsigned int ss_ = 0) : type(type_), k(k_), m(m_), c(c_), seed(seed_), symbol_size(ss_) {}
#endif
};

/**
 * @brief Struct video_frame represents a video frame and contains video description.
 */
struct video_frame {
        codec_t              color_spec;
        enum interlacing_t   interlacing;
        double               fps;

        struct tile         *tiles;
        unsigned int         tile_count;

        /** @name Fragment Stuff 
         * @{ */
        /// Indicates that the tile is fragmented. Normally not used (only for Bluefish444).
        unsigned int         fragment:1;
        /// Used only if (fragment == 1). Indicates this is the last fragment.
        unsigned int         last_fragment:1;
        /// Used only if (fragment == 1). ID of the frame. Fragments of same frame must have the same ID
        unsigned int         frame_fragment_id:14;
        /// @}

        /// @{
        /**
         * This function (if defined) is called when frame is no longer needed
         * by processing queue.
         * @note
         * Currently, this is only used in sending workflow, not the receiving one!
         * Can be called from arbitrary thread.
         * @note
         * Can be changed only by frame originator.
         * @deprecated
         * Should not be longer used. Suggested replacement is with shared pointers with custom deleter.
         */
        void               (*dispose)(struct video_frame *);
        /**
         * Additional data needed to dispose the frame
         *
         * @note
         * Can be changed only by frame originator.
         * @deprecated
         * Should not be longer used. Suggested replacement is with shared pointers with custom deleter.
        */
        void                *dispose_udata;
        /**
         * This function (if defined) is called by vf_free() to destruct video data
         * (@ref tile::data members).
         */
        void               (*data_deleter)(struct video_frame *);
        /// @}

        //standard transport
        uint8_t isStd;
        //H264 Standard transport
            // Config
        unsigned int h264_width;
        unsigned int h264_height;
        codec_t codec;
            // Data
        unsigned char *h264_buffer;
        unsigned int h264_buffer_len;
        unsigned int h264_offset_len;
        unsigned char *h264_offset_buffer;
            // Stats
        unsigned int h264_media_time;
        unsigned int h264_seqno;
            // Control
        h264_frame_type_t h264_frame_type;

        struct fec_desc fec_params;

        uint32_t ssrc;
};

/**
 * @brief Struct tile is an area of video_frame.
 * @note
 * Currently all tiles of a frame have to have same width ahd height.
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

struct vidcap_params;

#ifdef __cplusplus
}
#endif

#endif // TYPES_H_


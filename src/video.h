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
#ifndef __video_h

#define __video_h
#include "tile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
        RGBA,
        UYVY,
        YUYV,
        Vuy2,
        DVS8,
        R10k,
        v210,
        DVS10,
        DXT1,
        DXT1_YUV,
        DXT5,
        RGB,
        DPX10,
        JPEG,
        RAW,
        H264,
        MJPG,
        VP8,
        BGR
} codec_t;

enum interlacing_t {
        PROGRESSIVE = 0,
        UPPER_FIELD_FIRST = 1,
        LOWER_FIELD_FIRST = 2,
        INTERLACED_MERGED = 3,
        SEGMENTED_FRAME = 4
};

#define VIDEO_NORMAL                    0u
#define VIDEO_DUAL                      1u
#define VIDEO_STEREO                    2u
#define VIDEO_4K                        3u

#define PARAM_WIDTH                     (1<<0u)
#define PARAM_HEIGHT                    (1<<2u)
#define PARAM_CODEC                     (1<<3u)
#define PARAM_INTERLACING               (1<<4u)
#define PARAM_FPS                       (1<<5u)
#define PARAM_TILE_COUNT                (1<<6u)


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

/* contains full information both about video and about tiles.
 */
struct video_desc_ti {
        struct video_desc desc;
        struct tile_info ti;
};

struct video_frame 
{
        codec_t              color_spec;
        enum interlacing_t   interlacing;
        double               fps;
        struct tile         *tiles;
        
        unsigned int         tile_count;

        // Fragment stuff
        unsigned int         fragment:1;        // indicates that the tile is fragmented
        unsigned int         last_fragment:1;   // this is last fragment
        unsigned int         frame_fragment_id:14; // ID of the frame. Fragments of same frame must
                                                // have same ID.
};

struct tile {
        unsigned int         width;
        unsigned int         height;
        
        /*
         * data must be at least 4B aligned
         */
        char                *data; /* this is not beginning of the frame buffer actually but beginning of displayed data,
                                     * it is the case display is centered in larger window, 
                                     * i.e., data = pixmap start + x_start + y_start*linesize
                                     */
        unsigned int         data_len; /* relative to data pos, not framebuffer size! */      
        unsigned int         linesize;

        // Fragment stuff
        unsigned int         offset;            // Offset of the fragment (bytes)
};

struct tile * tile_alloc(void);

struct video_frame * vf_alloc(int count);
/**
 * Allocates video frame accordig given desc
 *
 * @param desc Description of video frame to be created
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
void vf_free(struct video_frame *buf);
/**
 * Same as vf_free plus removing (free) data fields
 */
void vf_free_data(struct video_frame *buf);

struct tile *tile_alloc();
struct tile *tile_alloc_desc(struct video_desc);
void tile_free(struct tile*);
void tile_free_data(struct tile*);

void vf_write_desc(struct video_frame *frame, struct video_desc desc);

struct tile * vf_get_tile(struct video_frame *buf, int pos);
struct video_frame * vf_get_copy(struct video_frame *frame);
int video_desc_eq(struct video_desc, struct video_desc);
int video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params);
struct video_desc video_desc_from_frame(struct video_frame *frame);
int get_video_mode_tiles_x(int video_mode);
int get_video_mode_tiles_y(int video_mode);
const char *get_interlacing_description(enum interlacing_t);
const char *get_interlacing_suffix(enum interlacing_t);
const char *get_video_mode_description(int video_mode);


/* these functions transcode one interlacing format to another */
void il_upper_to_merged(char *dst, char *src, int linesize, int height);
void il_merged_to_upper(char *dst, char *src, int linesize, int height);

double compute_fps(int fps, int fpsd, int fd, int fi);

#define AUX_INTERLACED  (1<<0)
#define AUX_PROGRESSIVE (1<<1)
#define AUX_SF          (1<<2)
#define AUX_RGB         (1<<3) /* if device supports both, set both */
#define AUX_YUV         (1<<4) 
#define AUX_10Bit       (1<<5)

#ifdef __cplusplus
}
#endif

#endif


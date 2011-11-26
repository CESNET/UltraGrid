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

typedef enum {
        RGBA,
        UYVY,
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
        JPEG
} codec_t;

/* please note that tiles have also its own widths and heights */
struct video_desc {
        unsigned int         width;
        unsigned int         height;
        codec_t              color_spec;
        int                  aux;
        double               fps;
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
        int                  aux;
        double               fps;
        struct tile         *tiles;
        
        unsigned int         grid_width; /* tiles */
        unsigned int         grid_height;
};

struct tile {
        unsigned int         width;
        unsigned int         height;
        
        char                *data; /* this is not beginning of the frame buffer actually but beginning of displayed data,
                                     * it is the case display is centered in larger window, 
                                     * i.e., data = pixmap start + x_start + y_start*linesize
                                     */
        unsigned int         data_len; /* relative to data pos, not framebuffer size! */      
        unsigned int         linesize;
        
        struct tile_info     tile_info;
};

struct video_frame * vf_alloc(int grid_width, int grid_height);
void vf_free(struct video_frame *buf);
struct tile * tile_get(struct video_frame *buf, int grid_x_pos, int grid_y_pos);
int video_desc_eq(struct video_desc, struct video_desc);

/*
 * Currently used (pre 1.0) is only AUX_{INTERLACED, PROGRESSIVE, SF, TILED}
 */
#define AUX_INTERLACED  (1<<0)
#define AUX_PROGRESSIVE (1<<1)
#define AUX_SF          (1<<2)
#define AUX_RGB         (1<<3) /* if device supports both, set both */
#define AUX_YUV         (1<<4) 
#define AUX_10Bit       (1<<5)
#define AUX_TILED       (1<<6)

#endif


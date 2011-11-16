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
#include "video.h"

struct video_frame * vf_alloc(int grid_width, int grid_height)
{
        struct video_frame *buf;
        
        buf = (struct video_frame *) malloc(sizeof(struct video_frame));
        
        buf->tiles = (struct tiles *) 
                        calloc(1, sizeof(struct tile) * grid_width *
                        grid_height);
        buf->grid_width = grid_width;
        buf->grid_height = grid_height;

        int x, y;
        for(x = 0; x < grid_width; ++x) {
                for(y = 0; y < grid_height; ++y) {
                        buf->tiles[x + y * grid_width].tile_info.x_count = 
                                grid_width;
                        buf->tiles[x + y * grid_width].tile_info.y_count = 
                                grid_height;
                        buf->tiles[x + y * grid_width].tile_info.pos_x = 
                                x;
                        buf->tiles[x + y * grid_width].tile_info.pos_y = 
                                y;
                }
        }
        
        return buf;
}

void vf_free(struct video_frame *buf)
{
        if(!buf)
                return;
        free(buf->tiles);
        free(buf);
}

struct tile * tile_get(struct video_frame *buf, int grid_x_pos, int grid_y_pos)
{
        assert (grid_x_pos < buf->grid_width && grid_y_pos < buf->grid_height);

        return &buf->tiles[grid_x_pos + grid_y_pos * buf->grid_width];
}

int video_desc_eq(struct video_desc a, struct video_desc b)
{
        return a.width == b.width &&
               a.height == b.height &&
               a.color_spec == b.color_spec &&
               fabs(a.fps - b.fps) < 0.01 &&
               // TODO: remove these obsolete constants
               (a.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit)) == (b.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit));
}

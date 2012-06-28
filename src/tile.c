/*
 * FILE:    tile.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <stdio.h>
#include <string.h>
#include "video_codec.h"
#include "tile.h"

#define MAGIC_H 0xFF
#define MAGIC_T 0xFE

void vf_split(struct video_frame *out, struct video_frame *src,
              unsigned int x_count, unsigned int y_count, int preallocate)
{
        unsigned int               tile_idx, line_idx;
        struct tile        *cur_tiles;
        unsigned int               tile_line = 0;

        out->color_spec = src->color_spec;
        out->fps = src->fps;
        //out->aux = src->aux | AUX_TILED;
        
        assert(vf_get_tile(src, 0)->width % x_count == 0u && vf_get_tile(src, 0)->height % y_count == 0u);

        for(tile_idx = 0u; tile_idx < x_count * y_count; ++tile_idx) {
                out->tiles[tile_idx].width = vf_get_tile(src, 0)->width / x_count;
                out->tiles[tile_idx].height = vf_get_tile(src, 0)->height / y_count;
                
                out->tiles[tile_idx].linesize = vc_get_linesize(out->tiles[tile_idx].width,
                                src->color_spec);
                out->tiles[tile_idx].data_len = out->tiles[tile_idx].linesize * out->tiles[tile_idx].height;
        }

        cur_tiles = &out->tiles[0];
        for(line_idx = 0u; line_idx < vf_get_tile(src, 0)->height; ++line_idx, ++tile_line) {
                unsigned int cur_tile_idx;
                unsigned int byte = 0u;

                if (line_idx % (vf_get_tile(src, 0)->height / y_count) == 0u) /* next tiles*/
                {
                        tile_line = 0u;
                        if (line_idx != 0u)
                                cur_tiles += x_count;
                        if (preallocate) {
                                for (cur_tile_idx = 0u; cur_tile_idx < x_count;
                                               ++cur_tile_idx) {
                                        cur_tiles[cur_tile_idx].data =
                                                malloc(cur_tiles[cur_tile_idx].
                                                                data_len);
                                }
                        }
                }

                for(cur_tile_idx = 0u; cur_tile_idx < x_count; ++cur_tile_idx) {
                        memcpy((void *) &cur_tiles[cur_tile_idx].data[
                                        tile_line *
                                        cur_tiles[cur_tile_idx].linesize],
                                        (void *) &src->tiles[0].data[line_idx *
                                        src->tiles[0].linesize + byte],
                                        cur_tiles[cur_tile_idx].width *
                                        get_bpp(src->color_spec));
                        byte += cur_tiles[cur_tile_idx].width * get_bpp(src->color_spec);
                }
        }
}

void vf_split_horizontal(struct video_frame *out, struct video_frame *src,
              unsigned int y_count)
{
        unsigned int i;

        for(i = 0u; i < y_count; ++i) {
                //out->aux = src->aux | AUX_TILED;
                out->fps = src->fps;
                out->color_spec = src->color_spec;
                out->tiles[i].width = src->tiles[0].width;
                out->tiles[i].height = src->tiles[0].height / y_count;
                
                out->tiles[i].linesize = vc_get_linesize(out->tiles[i].width,
                                out->color_spec);
                out->tiles[i].data_len = out->tiles[i].linesize * 
                        out->tiles[i].height;
                out->tiles[i].data = src->tiles[0].data + i * out->tiles[i].height 
                        * out->tiles[i].linesize;
        }
}

int tileinfo_eq_count(struct tile_info t1, struct tile_info t2)
{
        return t1.x_count == t2.x_count && t1.y_count == t2.y_count;
}


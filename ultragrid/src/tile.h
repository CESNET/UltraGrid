/*
 * FILE:    tile.h
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
#ifndef __tile_h

#define __tile_h
#include "config.h"
#include "config_unix.h"

struct video_frame;

struct tile_info {
        unsigned int h_reserved:8;
        unsigned int pos_x:4;
        unsigned int pos_y:4;
        unsigned int x_count:4;
        unsigned int y_count:4;
        unsigned int t_reserved:8;
} __attribute__((__packed__));

/**
 * vf_split splits the frame into multiple tiles.
 * Caller is responsible for allocating memory for all of these: out (to hold
 * all elements), out elements and theirs data member to hold tile data.
 *
 * width must be divisible by x_count && heigth by y_count (!)
 *
 * @param out          output video frames array
 *                     the resulting matrix will be stored row-dominant
 * @param src          source video frame
 * @param x_count      number of columns
 * @param y_count      number of rows
 * @param preallocate  used for preallocating buffers because determining right
 *                     size can be cumbersome. Anyway only .data are allocated.
 */
void vf_split(struct video_frame *out, struct video_frame *src,
              unsigned int x_count, unsigned int y_count, int preallocate);

void vf_split_horizontal(struct video_frame *out, struct video_frame *src,
              unsigned int y_count);

/**
 * tileinfo_eq:
 * compares count of tiles
 *
 * @param   t1  first structure
 * @param   t2  second structure
 * @return  0   if different
 *          !0  if equal
 */
int tileinfo_eq_count(struct tile_info t1, struct tile_info t2);

#endif

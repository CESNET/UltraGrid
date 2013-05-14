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
#ifndef __video_compress_h

#define __video_compress_h
#include "video_codec.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

struct compress_state;

// this is placeholder state, refer to inidividual compressions for example
extern int compress_init_noerr;

/**
 * Initializes compression
 * 
 * @param cfg command-line argument
 * @return intern state
 */
typedef  void *(*compress_init_t)(char *cfg);

/**
 * Compresses video frame
 * 
 * @param state compressor state
 * @param frame uncompressed frame
 * @return compressed frame
 */
typedef  struct video_frame * (*compress_frame_t)(void *state, struct video_frame *frame,
                int buffer_index);

/**
 * Compresses tile of a video frame
 * 
 * @param state compressor state
 * @param[in]     tile          uncompressed tile
 * @param[in,out] desc          input and then output video desc
 * @param[in]     buffer_index
 * @return compressed frame
 */
typedef  struct tile * (*compress_tile_t)(void *state, struct tile *tile,
                struct video_desc *desc, int buffer_index);


/**
 * Cleanup function
 */
typedef  void (*compress_done_t)(void *);

void show_compress_help(void);
int compress_init(char *config_string, struct compress_state **);
const char *get_compress_name(struct compress_state *);

int is_compress_none(struct compress_state *);

struct video_frame *compress_frame(struct compress_state *, struct video_frame*, int buffer_index);
void compress_done(struct compress_state *);

#ifdef __cplusplus
}
#endif

#endif /* __video_compress_h */

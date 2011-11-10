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
#ifndef __vo_postprocess_h

#define __vo_postprocess_h
#include "video_codec.h"

struct vo_postprocess_state;

typedef  void *(*vo_postprocess_init_t)(char *cfg);

/**
 * Reconfigures postprocessor for frame
 * and returns resulting frame properties (they can be different)
 * 
 * @return frame to be written to
 */
typedef  struct video_frame * (*vo_postprocess_reconfigure_t)(void *state, struct video_desc desc, struct tile_info);
typedef void (*vo_postprocess_get_out_desc_t)(struct vo_postprocess_state *, struct video_desc_ti *out);

/**
 * Postprocesses video frame
 * 
 * @param state postprocess state
 * @param input frame
 * @return output frame
 */
typedef  void (*vo_postprocess_t)(void *state, struct video_frame *in, struct video_frame *out, int req_out_pitch);

/**
 * Cleanup function
 */
typedef  void (*vo_postprocess_done_t)(void *);

struct vo_postprocess_state *vo_postprocess_init(char *config_string);
struct video_frame * vo_postprocess_reconfigure(struct vo_postprocess_state *, struct video_desc, struct tile_info);
void vo_postprocess_get_out_desc(struct vo_postprocess_state *, struct video_desc_ti *out);
void vo_postprocess(struct vo_postprocess_state *, struct video_frame*, struct video_frame*, int req_pitch);
void compress_done(struct vo_postprocess_state *);

void show_vo_postprocess_help(void);

#endif /* __vo_postprocess_h */

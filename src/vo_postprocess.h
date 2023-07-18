/**
 * @file   vo_postprocess.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2023 CESNET, z. s. p. o.
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
#ifndef __vo_postprocess_h

#define __vo_postprocess_h

#include "types.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/*          property                               type                   default          */
#define VO_PP_PROPERTY_CODECS                0 /*  codec_t[]          all uncompressed     */
#define VO_PP_DOES_CHANGE_TILING_MODE        1 /*  bool                    false           */

#define VO_PP_ABI_VERSION 7

struct vo_postprocess_state;

/**
 * @param init optional configuration, cannot be null
 */
typedef  void *(*vo_postprocess_init_t)(const char *cfg);

/**
 * Reconfigures postprocessor for frame
 * and returns resulting frame properties (they can be different)
 *
 * @param state postprocessor state
 * @param desc  video description to be configured to
 */
typedef  bool (*vo_postprocess_reconfigure_t)(void *state, struct video_desc desc);
typedef  struct video_frame * (*vo_postprocess_getf_t)(void *state);
/*
 * Returns various information about postprocessor format not only output (legacy name).
 *
 * @param s                        postprocessor state
 * @param out                      output video description according to input parameters
 * @param in_display_mode          some postprocessors change tiling mode (this is queryied by get_property
 *                                 function). If postprocessor does not report that it changes tiling mode,
 *                                 this parameter should be ignored.
 * @param out_frame_count          Because the postprocess function is called synchronously, in case that 
 *                                 from one input frame is generated more, this sets how many can be generated
 *                                 at maximum.
 */
typedef void (*vo_postprocess_get_out_desc_t)(void *s, struct video_desc *out, int *in_tile_mode, int *out_frame_count);

/**
 * Returns supported codecs
 *
 * @param state    postprocessor state
 * @param property property to be queried
 * @param val      returned value,
 *                      0 indicates that the property is understood, but there was a problem
 *                        processing the query (true is returned)
 * @param len      argument length
 *                 IN - length of allocated block
 *                 OUT - returned size (must be equal or less than allocated space)
 * @return         true  if the flag is understood
 *                 false if not, in that case, default values are assumed (if possible)
 */
typedef bool (*vo_postprocess_get_property_t)(void *state, int property, void *val, size_t *len);

/**
 * Postprocesses video frame
 * 
 * @param state postprocessor state
 * @param input frame
 *
 * @return flag If output video frame is filled with valid data.
 *
 */
typedef bool (*vo_postprocess_t)(void *state, struct video_frame *in, struct video_frame *out, int req_out_pitch);

/**
 * Cleanup function
 */
typedef  void (*vo_postprocess_done_t)(void *);


struct vo_postprocess_info {
        vo_postprocess_init_t init;
        vo_postprocess_reconfigure_t reconfigure;
        vo_postprocess_getf_t getf;
        vo_postprocess_get_out_desc_t get_out_desc;
        vo_postprocess_get_property_t get_property;
        vo_postprocess_t vo_postprocess;
        vo_postprocess_done_t done;
};

struct vo_postprocess_state *vo_postprocess_init(const char *config_string);

bool vo_postprocess_reconfigure(struct vo_postprocess_state *, struct video_desc);
struct video_frame * vo_postprocess_getf(struct vo_postprocess_state *);
void vo_postprocess_get_out_desc(struct vo_postprocess_state *, struct video_desc *out, int *display_mode, int *out_frames_count);
bool vo_postprocess_get_property(struct vo_postprocess_state *, int property, void *val, size_t *len);

bool vo_postprocess(struct vo_postprocess_state *, struct video_frame*, struct video_frame*, int req_pitch);
void vo_postprocess_done(struct vo_postprocess_state *s);

void show_vo_postprocess_help(bool full);

#ifdef __cplusplus
}
#endif

#endif /* __vo_postprocess_h */

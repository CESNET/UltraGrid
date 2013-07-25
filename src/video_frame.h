/**
 * @file   video_frame.h
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 */
/* Copyright (c) 2005-2013 CESNET z.s.p.o.
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
#ifndef video_frame_h_
#define video_frame_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

/** @anchor video_eq_param
 * @{ */
#define PARAM_WIDTH                     (1<<0u)
#define PARAM_HEIGHT                    (1<<2u)
#define PARAM_CODEC                     (1<<3u)
#define PARAM_INTERLACING               (1<<4u)
#define PARAM_FPS                       (1<<5u)
#define PARAM_TILE_COUNT                (1<<6u)
/**@}*/

/**
 * @brief Allocates blank video frame
 * @param count number of allocated tiles
 * @return allocated video frame
 */
struct video_frame * vf_alloc(int count);
/**
 * @brief Allocates video frame accordig given desc
 * @param desc Description of video frame to be created
 *             At least desc.tile_count must be specified!
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
/**
 * @brief Frees video frame excluding video data
 * @note
 * Video data is not freed
 */
void vf_free(struct video_frame *buf);
/**
 * @brief Frees video frame including video data
 * @see vf_free
 */
void vf_free_data(struct video_frame *buf);
/**
 * @brief Allocates blank tile
 * @return allocated tile
 */
struct tile * tile_alloc(void);
/**
 * @brief Allocates blank tile and sets width and height
 * @param desc video description
 * @return allocated tile
 */
struct tile *tile_alloc_desc(struct video_desc desc);
/**
 * @brief Frees video tile excluding video data
 * @note
 * Only tile allocated with tile_alloc should be freed
 */
void tile_free(struct tile*);
/**
 * @brief Frees video tile including video data
 * @see tile_free
 * @note
 * Only tile allocated with tile_alloc should be freed
 */
void tile_free_data(struct tile*);

/**
 * @brief Returns n-th tile from video frame
 */
struct tile * vf_get_tile(struct video_frame *buf, int pos);
/**
 * @brief Makes deep copy of the video frame
 */
struct video_frame * vf_get_copy(struct video_frame *frame);
/**
 * @brief Compares two video descriptions.
 *
 * This is equivalent to
 * @ref video_desc_eq_excl_param
 * when no parameter is excluded.
 * @retval 0 if different
 * @retval !=0 structs are the same
 */
int video_desc_eq(struct video_desc, struct video_desc);
/**
 * @brief Compares two video descriptions
 * @see video_desc_eq
 * @param excluded_params
 * bitwise OR of @ref video_eq_param
 * @retval 0 if different
 * @retval !=0 structs are the same
 */
int video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params);
/**
 * @brief Returns struct video_desc from video frame
 */
struct video_desc video_desc_from_frame(struct video_frame *frame);
/**
 * @brief Returns vertical count of tiles
 * @param mode requestd video mode
 * @returns vertical count of tiles
 */
int get_video_mode_tiles_x(enum video_mode mode);
/**
 * @brief Returns horizontal count of tiles
 * @param mode requestd video mode
 * @returns horizontal count of tiles
 */
int get_video_mode_tiles_y(enum video_mode mode);
/**
 * @brief Returns description of interlacing
 * Eg. "progressive"
 */
const char *get_interlacing_description(enum interlacing_t interlacing);
/**
 * @brief Returns suffix describing interlacing
 * Eg. p, i or psf
 */
const char *get_interlacing_suffix(enum interlacing_t interlacing);
/**
 * @brief Returns description of video mode
 * Eg. "tiled 4K"
 * @param mode requestd video mode
 */
const char *get_video_mode_description(enum video_mode mode);

/* these functions transcode one interlacing format to another */
/**
 * @brief Converts upper-field-first to interlaced merged.
 */
void il_upper_to_merged(char *dst, char *src, int linesize, int height);
/**
 * @brief Converts interlaced merged to upper-field-first.
 */
void il_merged_to_upper(char *dst, char *src, int linesize, int height);

/**
 * @brief Computes FPS as a double from packet fields.
 *
 * Individual field semantics can be found in paper referenced from
 * @ref av_pkt_description
 */
double compute_fps(int fps, int fpsd, int fd, int fi);

/** @name Video Flags
 * @deprecated use rather video_frame or video_desc members
 * @{ */
#define AUX_INTERLACED  (1<<0)
#define AUX_PROGRESSIVE (1<<1)
#define AUX_SF          (1<<2)
#define AUX_RGB         (1<<3) /* if device supports both, set both */
#define AUX_YUV         (1<<4)
#define AUX_10Bit       (1<<5)
/** @} */

#ifdef __cplusplus
}
#endif

#endif


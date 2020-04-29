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
 * Calls video_frame::dispose if defined.
 *
 * @see video_frame::dispose
 */
#define VIDEO_FRAME_DISPOSE(frame) if ((frame) && (frame)->callbacks.dispose) \
        (frame)->callbacks.dispose(frame)

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
 * according to video description.
 *
 * The allocated data members are deleted automatically by vf_free()
 *
 * @see vf_alloc_desc
 *
 * @param desc Description of video frame to be created
 * @return allocated video frame
 *         NULL if insufficient memory
 */
struct video_frame * vf_alloc_desc_data(struct video_desc desc);

/**
 * @brief Frees video_frame structure
 *
 * Calls video_frame::data_deleter if defined (suplied by user
 * or fiiled by vf_alloc_desc_data()).
 */
void vf_free(struct video_frame *buf);

/**
 * @brief This function should be called before returning a frame back to
 * a frame pool. Currently this is used to free hw surfaces when decoding
 * using hw acceleration
 *
 * Calls video_frame.callbacks.recycle if defined.
 */
void vf_recycle(struct video_frame *buf);

/**
 * Deletes video data members with a free() call.
 * @see video_frame::data_deleter
 */
void vf_data_deleter(struct video_frame *buf);

/**
 * @brief Returns pointer to n-th tile from video frame
 * Equivalent to &video_frame::tiles[pos]
 */
struct tile * vf_get_tile(struct video_frame *buf, int pos);
/**
 * @brief Makes deep copy of the video frame
 *
 * Copied data are automatically freeed by vf_free()
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
 * @brief Returns description of interlacing
 * Eg. "progressive"
 */
const char *get_interlacing_description(enum interlacing_t interlacing);
/**
 * @brief Returns suffix describing interlacing
 * Eg. p, i or psf
 */
const char *get_interlacing_suffix(enum interlacing_t interlacing);
enum interlacing_t get_interlacing_from_suffix(const char *suffix);

void il_lower_to_merged(char *dst, char *src, int linesize, int height, void **stored_state);
/* these functions transcode one interlacing format to another */
/**
 * @brief Converts upper-field-first to interlaced merged.
 */
void il_upper_to_merged(char *dst, char *src, int linesize, int height, void **stored_state);
/**
 * @brief Converts interlaced merged to upper-field-first.
 */
void il_merged_to_upper(char *dst, char *src, int linesize, int height, void **stored_state);

/**
 * @brief Computes FPS as a double from packet fields.
 *
 * Individual field semantics can be found in paper referenced from
 * @ref av_pkt_description
 */
double compute_fps(int fps, int fpsd, int fd, int fi);

bool save_video_frame_as_pnm(struct video_frame *frame, const char *name);

void vf_copy_metadata(struct video_frame *desc, const struct video_frame *src);
void vf_store_metadata(struct video_frame *f, void *);
void vf_restore_metadata(struct video_frame *f, void *);

/**
 * Returns sum of lengths of all tiles.
 */
unsigned int vf_get_data_len(struct video_frame *f);

/**
 * Fills @param planes with pointers to data accordint to provided
 * spec.
 *
 * Works with planar pixel formats only.
 */
void buf_get_planes(int width, int height, codec_t color_spec, char *data, char **planes);

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


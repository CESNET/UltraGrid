/**
 * @file   video.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This is an umbrella header for video functions.
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#ifndef VIDEO_H_
#define VIDEO_H_

#include "video_codec.h"
#include "video_frame.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

enum video_mode get_video_mode_from_str(const char *requested_mode);
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
 * @brief Returns description of video mode
 * Eg. "tiled 4K"
 * @param mode requestd video mode
 */
const char *get_video_mode_description(enum video_mode mode);

/**
 * @brief Tries to guess video mode from number of substreams.
 *
 * @note
 * The guessed video mode may not be correct (some modes may have the same
 * number of substreams).
 *
 * @param   num_substreams number of received substreams
 * @returns                guessed video mode
 * @retval VIDEO_UNKNOWN   if the mode was not guessed.
 */
enum video_mode guess_video_mode(int num_substreams);

const char *video_desc_to_string(struct video_desc d);
struct video_desc get_video_desc_from_string(const char *);

#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef __cplusplus
#include <istream>
#include <ostream>
std::istream& operator>>(std::istream& is, video_desc& desc);
std::ostream& operator<<(std::ostream& os, const video_desc& desc);
std::ostream& operator<<(std::ostream& os, const codec_t& color_spec);
#endif

#endif // VIDEO_H_


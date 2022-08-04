/**
 * @file   utils/video_pattern_generator.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2021 CESNET, z. s. p. o.
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

#ifndef VIDEO_PATTERN_GENERATOR_H_411E8141_A7AE_4FCD_8464_41CE032CF81B
#define VIDEO_PATTERN_GENERATOR_H_411E8141_A7AE_4FCD_8464_41CE032CF81B

#include <memory>
#include <string>

#include "video.h"

struct video_pattern_generator;

typedef struct video_pattern_generator *video_pattern_generator_t;

/// @param offset   offset between still image frames (in bytes)
video_pattern_generator_t video_pattern_generator_create(std::string const & config, int width, int height, codec_t color_spec, int offset);
char *video_pattern_generator_next_frame(video_pattern_generator_t);
void video_pattern_generator_fill_data(video_pattern_generator_t, const char *data);
void video_pattern_generator_destroy(video_pattern_generator_t);

#endif // defined VIDEO_PATTERN_GENERATOR_H_411E8141_A7AE_4FCD_8464_41CE032CF81B

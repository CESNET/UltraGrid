/**
 * @file   utils/vs_split.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2013 CESNET z.s.p.o.
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

#ifndef VF_SPLIT_H_
#define VF_SPLIT_H_

struct video_frame;

#ifdef __cplusplus
extern "C" {
#endif

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
 *
 * @deprecated this function should not be used
 */
void vf_split(struct video_frame *out, struct video_frame *src,
              unsigned int x_count, unsigned int y_count, int preallocate);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <memory>
#include <vector>

std::vector<std::shared_ptr<video_frame>> vf_separate_tiles(std::shared_ptr<video_frame> frame);
std::shared_ptr<video_frame> vf_merge_tiles(std::vector<std::shared_ptr<video_frame>> const & tiles);

#endif // __cplusplus

#endif // VF_SPLIT_H_


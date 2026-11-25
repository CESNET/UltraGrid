/**
 * @file   utils/overlay_config.h
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Parser for the overlay postprocessor's configuration string.
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#ifndef UTILS_OVERLAY_CONFIG_H_8F2A1B6D_3E5C_4A7F_B9D2_5C8A1E4F7B3D
#define UTILS_OVERLAY_CONFIG_H_8F2A1B6D_3E5C_4A7F_B9D2_5C8A1E4F7B3D

#include <stdbool.h>

#include "utils/overlay_layout.h"
#include "utils/fs.h"
#include "utils/overlay_scale.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parsed form of the postprocess option string. Owns no allocations: file
 * is a fixed-size buffer so the caller can embed by value.
 */
struct overlay_config {
        char file[MAX_PATH_SIZE];
        enum overlay_position position;
        int  custom_x;
        int  custom_y;
        int  soft_edge;
        int  scale_w;        ///< 0 = no scaling (or scale_to_frame is set)
        int  scale_h;
        bool scale_to_frame; ///< scale=frame: re-scale to the current
                             ///< frame dims on every resolution change
        enum overlay_scale_filter scale_filter;  ///< default BICUBIC
        int  blend_threads;  ///< 0 = auto (min(ncpu, 8)); 1 = single-threaded; >1 = pthread workers
        bool help;
        bool perf;           ///< periodic per-frame timing log
};

/*
 * Parse a "key=value:key=value:..." option string into *out. Returns true on
 * success, false on malformed input (with a message logged). The single
 * keyword "help" is also accepted as a request for usage and produces help=1
 * with no other fields validated.
 *
 * Recognised keys:
 *   file=<path>       — path to the PAM overlay (required, except with help)
 *   position=<name>   — center | top_left | top_right | bottom_left
 *                       | bottom_right | custom (default: center)
 *   custom_x=<int>    — absolute or negative-from-edge x offset; forces
 *   custom_y=<int>      position=custom
 *   soft_edge=<int>   — N-pixel linear alpha fade (0 = disabled, default 0)
 *   scale=<W>x<H>     — resize the overlay to W x H (0x0 = no scaling,
 *                       default)
 *   scale=frame       — re-scale the overlay to the current frame
 *                       dimensions; tracks resolution renegotiations
 *                       on the decode path. Mutually exclusive with
 *                       scale=<W>x<H> (last one wins).
 *   scale_filter=<f>  — resampling filter: nearest, fast_bilinear,
 *                       bilinear, bicubic, lanczos (default: bicubic)
 *   blend_threads=<N> — pthread workers for the per-row alpha-blend
 *                       loop. Default is min(ncpu, 8); pass 1 for
 *                       explicit single-threaded blend.
 *   perf              — bare token; turns on periodic per-frame timing
 *                       log (off by default)
 *
 * Unknown keys are rejected.
 */
bool overlay_config_parse(const char *opts, struct overlay_config *out);

#ifdef __cplusplus
}
#endif

#endif

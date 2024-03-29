/*
 * FILE:    capture_filter/resize_utils.h
 * AUTHORS: Gerard Castillo     <gerard.castillo@i2cat.net>
 *          Marc Palau          <marc.palau@i2cat.net>
 *          Martin Pulec        <martin.pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2015-2023 CESNET, z. s. p. o.
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
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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

#ifndef RESIZE_UTILS_H_
#define RESIZE_UTILS_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RESIZE_SUPPORTED_PIXFMT_INIT RGB, RGBA, I420, UYVY, YUYV, RG48

#define RESIZE_ALGO_DFL        (-1)
#define RESIZE_ALGO_UNKN       (-2)
#define RESIZE_ALGO_HELP_SHOWN (-3)
int resize_algo_from_string(const char *str);

struct resize_param {
        enum resize_mode {
                NONE,
                USE_FRACTION,
                USE_DIMENSIONS,
        } mode;
        union {
                double factor;
                struct {
                        int target_width;
                        int target_height;
                };
        };
        int algo;
};
void resize_frame(char *indata, codec_t in_color, char *outdata, int width,
                  int height, struct resize_param *resize_spec);

#ifdef __cplusplus
}
#endif

#endif// RESIZE_UTILS_H_

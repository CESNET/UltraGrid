/**
 * @file   to_planar.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains various conversions from planar pixel formats as used
 * by libavcodec or jpegxs to packed pixel formats.
 */
/*
 * Copyright (c) 2019-2026 CESNET z.s.p.o.
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

#ifndef TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6
#define TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6

#ifdef __cplusplus
extern "C" {
#endif

#define TO_PLANAR_MAX_COMP 4

struct to_planar_data {
        int width;
        int height;
        unsigned char *__restrict out_data[TO_PLANAR_MAX_COMP];
        unsigned out_linesize[TO_PLANAR_MAX_COMP];
        const unsigned char *__restrict in_data;
};

/// functions to decode whole buffer of packed data to planar or packed
typedef void
decode_buffer_func_t(struct to_planar_data d);

decode_buffer_func_t v210_to_p010le;
decode_buffer_func_t y216_to_p010le;
decode_buffer_func_t uyvy_to_nv12;
decode_buffer_func_t rgba_to_bgra;
// other packed->planar convs are histaorically in video_codec.[ch]
decode_buffer_func_t uyvy_to_i420;
decode_buffer_func_t r12l_to_gbrp12le;
decode_buffer_func_t r12l_to_gbrp16le;
decode_buffer_func_t r12l_to_rgbp12le;

#ifdef __cplusplus
}
#endif

#endif // defined TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6

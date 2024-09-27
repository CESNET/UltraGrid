/**
 * @file   color.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2024 CESNET
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

#include "color.h"

#include <stdio.h>   // for fprintf, stderr
#include <stdlib.h>  // for abort

#include "types.h"   // for depth

#define KG_709 KG(KR_709,KB_709)
#define D (2.*(KR_709+KG_709))
#define E (2.*(1.-KR_709))

#define Y_R(out_depth) \
        ((comp_type_t) (((KR_709 * Y_LIMIT(out_depth)) * (1 << COMP_BASE)) + \
                        C_EPS))
#define Y_G(out_depth) \
        ((comp_type_t) (((KG_709 * Y_LIMIT(out_depth)) * (1 << COMP_BASE)) + \
                        C_EPS))
#define Y_B(out_depth) \
        ((comp_type_t) (((KB_709 * Y_LIMIT(out_depth)) * (1 << COMP_BASE)) + \
                        C_EPS))
#define CB_R(out_depth) \
        ((comp_type_t) (((-KR_709 / D * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) - C_EPS))
#define CB_G(out_depth) \
        ((comp_type_t) (((-KG_709 / D * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) - C_EPS))
#define CB_B(out_depth) \
        ((comp_type_t) ((((1 - KB_709) / D * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) + C_EPS))
#define CR_R(out_depth) \
        ((comp_type_t) ((((1 - KR_709) / E * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) - C_EPS))
#define CR_G(out_depth) \
        ((comp_type_t) (((-KG_709 / E * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) - C_EPS))
#define CR_B(out_depth) \
        ((comp_type_t) (((-KB_709 / E * CBCR_LIMIT(out_depth)) * \
                        (1 << COMP_BASE)) + C_EPS))

#define COEFFS(depth) \
        { \
                Y_R(depth),  Y_G(depth),  Y_B(depth),\
                CB_R(depth), CB_G(depth), CB_B(depth),\
                CR_R(depth), CR_G(depth), CR_B(depth),\
\
                SCALED(R_CR(depth, KR_709, KB_709)), \
                SCALED(G_CB(depth, KR_709, KB_709)), \
                SCALED(G_CR(depth, KR_709, KB_709)), \
                SCALED(B_CB(depth, KR_709, KB_709)), \
        }

/**
 * @brief returns color coefficient for RGB<-YCbCr conversion
 *
 * @note
 * It is suggested to copy the result to a struct (not using the returned ptr
 * directly) when passing to RGB_TO_*() or YCBCR_TO_*(). The compiler may not
 * be sure that the struct itself doesn't alias with other pointers and may
 * produce less optimal code (currently with GCC 12 is not a problem for
 * pixfmt_conv conversions but to_lavc_vid_convs seem to be affected).
 */
const struct color_coeffs *
get_color_coeffs(int ycbcr_bit_depth)
{
        static const struct color_coeffs col_cfs_8  = COEFFS(DEPTH8);
        static const struct color_coeffs col_cfs_10 = COEFFS(DEPTH10);
        static const struct color_coeffs col_cfs_12 = COEFFS(DEPTH12);
        static const struct color_coeffs col_cfs_16 = COEFFS(DEPTH16);
        switch ((enum depth) ycbcr_bit_depth) {
        case DEPTH8:
                return &col_cfs_8;
        case DEPTH10:
                return &col_cfs_10;
        case DEPTH12:
                return &col_cfs_12;
        case DEPTH16:
                return &col_cfs_16;
        }
        fprintf(stderr, "%s: Wrong depth %d!\n", __func__,
                ycbcr_bit_depth);
        abort();
}

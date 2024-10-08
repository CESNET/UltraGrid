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

#include "host.h"    // for ADD_TO_PARAM
#include "types.h"   // for depth

#define D(kr, kb) (2. * ((kr) + KG(kr, kb)))
#define E(kr)     (2. * (1. - (kr)))

#define Y_R(out_depth, kr, kb) \
        ((comp_type_t) (((kr * Y_LIMIT(out_depth)) * (1 << COMP_BASE)) + C_EPS))
#define Y_G(out_depth, kr, kb) \
        ((comp_type_t) (((KG(kr, kb) * Y_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) + \
                        C_EPS))
#define Y_B(out_depth, kr, kb) \
        ((comp_type_t) ((((kb) * Y_LIMIT(out_depth)) * (1 << COMP_BASE)) + \
                        C_EPS))
#define CB_R(out_depth, kr, kb) \
        ((comp_type_t) (((-(kr) / D(kr, kb) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) - \
                        C_EPS))
#define CB_G(out_depth, kr, kb) \
        ((comp_type_t) (((-KG(kr, kb) / D(kr, kb) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) - \
                        C_EPS))
#define CB_B(out_depth, kr, kb) \
        ((comp_type_t) ((((1 - (kb)) / D(kr, kb) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) + \
                        C_EPS))
#define CR_R(out_depth, kr, kb) \
        ((comp_type_t) ((((1 - (kr)) / E(kr) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) - \
                        C_EPS))
#define CR_G(out_depth, kr, kb) \
        ((comp_type_t) (((-KG(kr, kb) / E(kr) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) - \
                        C_EPS))
#define CR_B(out_depth, kr, kb) \
        ((comp_type_t) (((-(kb) / E(kr) * CBCR_LIMIT(out_depth)) * \
                         (1 << COMP_BASE)) + \
                        C_EPS))

#define COEFFS(depth, kr, kb) \
        { \
                Y_R(depth, kr, kb),  Y_G(depth, kr, kb),  Y_B(depth, kr, kb),\
                CB_R(depth, kr, kb), CB_G(depth, kr, kb), CB_B(depth, kr, kb),\
                CR_R(depth, kr, kb), CR_G(depth, kr, kb), CR_B(depth, kr, kb),\
\
                Y_SCALE(depth), \
                SCALED(R_CR(depth, kr, kb)), \
                SCALED(G_CB(depth, kr, kb)), \
                SCALED(G_CR(depth, kr, kb)), \
                SCALED(B_CB(depth, kr, kb)), \
        }

#define COEFFS_601(depth) COEFFS(depth, KR_601, KB_601)
#define COEFFS_709(depth) COEFFS(depth, KR_709, KB_709)

ADD_TO_PARAM("color-601", "* color-601\n"
                "  Use BT.601 color primaries.\n");
/**
 * @brief returns color coefficient for RGB<-YCbCr conversion
 *
 * Using BT.709 by default.
 *
 *
 * @note
 * It is suggested to copy the result to a struct (not using the returned ptr
 * directly) when passing to RGB_TO_*() or YCBCR_TO_*(). The compiler may not
 * be sure that the struct itself doesn't alias with other pointers and may
 * produce less optimal code (currently with GCC 12 is not a problem for
 * pixfmt_conv conversions but to_lavc_vid_convs seem to be affected).
 */
const struct color_coeffs *
get_color_coeffs(enum colorspace cs, int ycbcr_bit_depth)
{
        static _Atomic enum colorspace dfl_cs = CS_DFL;
        if (dfl_cs == CS_DFL) {
                dfl_cs = get_default_cs();
        }

        const int cs_idx = cs != (CS_DFL ? cs : dfl_cs) - 1;

        static const struct {
                struct color_coeffs col_cfs[2];
        } coeffs[] = {
                { { COEFFS_601(DEPTH8),  COEFFS_709(DEPTH8)  } },
                { { COEFFS_601(DEPTH10), COEFFS_709(DEPTH10) } },
                { { COEFFS_601(DEPTH12), COEFFS_709(DEPTH12) } },
                { { COEFFS_601(DEPTH16), COEFFS_709(DEPTH16) } }
        };
        switch ((enum depth) ycbcr_bit_depth) {
        case DEPTH8:
                return &coeffs[0].col_cfs[cs_idx];
        case DEPTH10:
                return &coeffs[1].col_cfs[cs_idx];
        case DEPTH12:
                return &coeffs[2].col_cfs[cs_idx];
        case DEPTH16:
                return &coeffs[3].col_cfs[cs_idx];
        }

        fprintf(stderr, "%s: Wrong depth %d!\n", __func__, ycbcr_bit_depth);
        abort();
}

enum colorspace
get_default_cs()
{
        return get_commandline_param("color-601") != NULL ? CS_601_LIM
                                                          : CS_709_LIM;
}

struct color_coeffs
compute_color_coeffs(double kr, double kb, int ycbcr_bit_depth)
{
        return (struct color_coeffs) COEFFS(ycbcr_bit_depth, kr, kb);
}

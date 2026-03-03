/**
 * @file   aja_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2026 CESNET, zájmové sdružení právnických osob
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_msvc.h" // coompat - __attribute__ etc.

#include <cinttypes>                 // for PRIx64
#include <cstdint>                   // for uint64_t
#include <cstdio>                    // for printf
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <ntv2card.h>                // for CNTV2Card

#include "utils/color_out.h"         // for color_printf, TUNDERLINE
#include "aja_common.hpp" // include after color_out to override its stuff for MSVC

#include "utils/macros.h" // OPTIMIZED_FOR

#ifndef UNUSED
# define UNUSED(x) ((void) x)
#endif

void
vc_copylineR12LtoR12A(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 36; x += 36) {
                *dst++ = src[1] << 4 | src[0] >> 4; // r0
                *dst++ = src[0] << 4 | src[2] >> 4; // r0+g0
                *dst++ = src[2] << 4 | src[1] >> 4; // g0
                *dst++ = src[4] << 4 | src[3] >> 4; // b0
                *dst++ = src[3] << 4 | src[5] >> 4; // b0+r1
                *dst++ = src[5] << 4 | src[4] >> 4; // r1
                *dst++ = src[7] << 4 | src[6] >> 4; // g1
                *dst++ = src[6] << 4 | src[8] >> 4; // g1+b1
                *dst++ = src[8] << 4 | src[7] >> 4; // b1
                *dst++ = src[10] << 4 | src[9] >> 4; // r2
                *dst++ = src[9] << 4 | src[11] >> 4; // r2+g2
                *dst++ = src[11] << 4 | src[10] >> 4; // g2
                *dst++ = src[13] << 4 | src[12] >> 4; // b2
                *dst++ = src[12] << 4 | src[14] >> 4; // b2+r3
                *dst++ = src[14] << 4 | src[13] >> 4; // r3
                *dst++ = src[16] << 4 | src[15] >> 4; // g3
                *dst++ = src[15] << 4 | src[17] >> 4; // g3+b3
                *dst++ = src[17] << 4 | src[16] >> 4; // b3
                *dst++ = src[19] << 4 | src[18] >> 4; // r4
                *dst++ = src[18] << 4 | src[20] >> 4; // r4+g4
                *dst++ = src[20] << 4 | src[19] >> 4; // g4
                *dst++ = src[22] << 4 | src[21] >> 4; // b4
                *dst++ = src[20] >> 4 | src[23] >> 4; // b4+r5
                *dst++ = src[23] << 4 | src[22] >> 4; // r5
                *dst++ = src[25] << 4 | src[24] >> 4; // g5
                *dst++ = src[24] << 4 | src[26] >> 4; // g5+b5
                *dst++ = src[26] << 4 | src[25] >> 4; // b5
                *dst++ = src[28] << 4 | src[27] >> 4; // r6
                *dst++ = src[27] << 4 | src[29] >> 4; // r6+g6
                *dst++ = src[29] << 4 | src[28] >> 4; // g6
                *dst++ = src[31] << 4 | src[30] >> 4; // b6
                *dst++ = src[30] << 4 | src[32] >> 4; // b6+r7
                *dst++ = src[32] << 4 | src[31] >> 4; // r7
                *dst++ = src[34] << 4 | src[33] >> 4; // g7
                *dst++ = src[33] << 4 | src[35] >> 4; // g7+b7
                *dst++ = src[35] << 4 | src[34] >> 4; // b7
                src += 36;
        }
}

/**
 * Converts AJA NTV2_FBF_12BIT_RGB_PACKED to R12L
 */
void
vc_copylineR12AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 36; x += 36) {
                *dst++ = src[0] << 4 |  src[1] >> 4; // r0
                *dst++ = src[2] << 4 |  src[0] >> 4; // r0+g0
                *dst++ = src[1] << 4 | src[2] >> 4; // g0
                *dst++ = src[3] << 4 | src[4] >> 4; // b0
                *dst++ = src[5] << 4 | src[3] >> 4; // b0+r1
                *dst++ = src[4] << 4 | src[5] >> 4; // r1
                *dst++ = src[6] << 4 | src[7] >> 4; // g1
                *dst++ = src[8] << 4 | src[6] >> 4; // g1+b1
                *dst++ = src[7] << 4 | src[8] >> 4;
                *dst++ = src[9] << 4 | src[10] >> 4;
                *dst++ = src[11] << 4 | src[9] >> 4;
                *dst++ = src[10] << 4 | src[11] >> 4;
                *dst++ = src[12] << 4 | src[13] >> 4;
                *dst++ = src[14] << 4 | src[12] >> 4;
                *dst++ = src[13] << 4 | src[14] >> 4;
                *dst++ = src[15] << 4 | src[16] >> 4;
                *dst++ = src[17] << 4 | src[15] >> 4;
                *dst++ = src[16] << 4 | src[17] >> 4;
                *dst++ = src[18] << 4 | src[19] >> 4;
                *dst++ = src[20] << 4 | src[18] >> 4;
                *dst++ = src[19] << 4 | src[20] >> 4;
                *dst++ = src[21] << 4 | src[22] >> 4;
                *dst++ = src[23] << 4 | src[21] >> 4;
                *dst++ = src[22] << 4 | src[23] >> 4; // 23 - R5u - @todo problem here
                *dst++ = src[24] << 4 | src[25] >> 4;
                *dst++ = src[26] << 4 | src[24] >> 4;
                *dst++ = src[25] << 4 | src[26] >> 4;
                *dst++ = src[27] << 4 | src[28] >> 4;
                *dst++ = src[29] << 4 | src[27] >> 4;
                *dst++ = src[28] << 4 | src[29] >> 4;
                *dst++ = src[30] << 4 | src[31] >> 4;
                *dst++ = src[32] << 4 | src[30] >> 4;
                *dst++ = src[31] << 4 | src[32] >> 4;
                *dst++ = src[33] << 4 | src[34] >> 4;
                *dst++ = src[35] << 4 | src[33] >> 4;
                *dst++ = src[34] << 4 | src[35] >> 4;
                src += 36;
        }
}

void
print_aja_device_details(CNTV2Card *device)
{
        printf("\t%s\n", device->GetDescription().c_str());

        color_printf("\t" TUNDERLINE("Device ID:") " 0x%08x\n",
                     device->GetBaseDeviceID());

        ULWord dev_pci_id = 0;
        device->GetPCIDeviceID(dev_pci_id);
        color_printf("\t" TUNDERLINE("Device PCI ID:") " 0x%08x\n", dev_pci_id);

        uint64_t serial_nr = device->GetSerialNumber();
        color_printf("\t" TUNDERLINE("Serial Number:") " 0x%" PRIx64 "\n",
                     serial_nr);

        color_printf("\t" TUNDERLINE("Video Inputs:") " %hu\n",
                     device->features().GetNumVideoInputs());
        printf("\t" TUNDERLINE("Video Outputs:") " %hu\n",
               device->features().GetNumVideoOutputs());
}

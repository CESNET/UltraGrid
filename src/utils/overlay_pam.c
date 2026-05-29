/**
 * @file   utils/overlay_pam.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  PAM image loader normalised to 16-bit RGBA
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>           // for free, malloc

#include "utils/overlay_pam.h"
#include "utils/pam.h"

bool overlay_load_pam_rgba16(const char *path, uint16_t **out_data,
                             int *out_width, int *out_height)
{
        struct pam_metadata info;
        unsigned char *raw = NULL;
        if (!pam_read(path, &info, &raw, malloc)) {
                return false;
        }
        if (info.ch_count != 3 && info.ch_count != 4) {
                free(raw);
                return false;
        }
        /* Only accept exact 8-bit (255) and 16-bit (65535) maxvals. Other
         * values would need scaling to map their full range onto 0xFFFF;
         * silently treating them as 16-bit produces darker overlays than the
         * file authored. Reject rather than misinterpret. */
        if (info.maxval != 255 && info.maxval != 65535) {
                free(raw);
                return false;
        }

        const size_t pixels = (size_t)info.width * info.height;
        uint16_t *out = malloc(pixels * 4 * sizeof(uint16_t));
        if (!out) {
                free(raw);
                return false;
        }

        const bool deep = info.maxval == 65535;
        const size_t bytes_per_pixel = info.ch_count * (deep ? 2u : 1u);

        for (size_t i = 0; i < pixels; i++) {
                const unsigned char *p = raw + i * bytes_per_pixel;
                uint16_t *o = out + i * 4;
                /* Default alpha to opaque; overwritten when ch_count == 4. */
                o[3] = 0xFFFFu;
                if (deep) {
                        /* PAM 16-bit samples are big-endian. */
                        for (int c = 0; c < info.ch_count; c++) {
                                o[c] = (uint16_t)(((unsigned)p[c*2] << 8)
                                                  | p[c*2 + 1]);
                        }
                } else {
                        for (int c = 0; c < info.ch_count; c++) {
                                /* Bit-replicate 8-bit to 16-bit. */
                                o[c] = (uint16_t)(((unsigned)p[c] << 8) | p[c]);
                        }
                }
        }

        free(raw);
        *out_data = out;
        *out_width = info.width;
        *out_height = info.height;
        return true;
}

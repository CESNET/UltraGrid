/**
 * @file   utils/pam.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Very simple library to read and write PAM/PPM files. Only binary formats
 * (P5, P6) for PNM are processed, P4 is also not used.
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2013-2025, CESNET
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
#define PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// metadata read from file
struct pam_metadata {
        int width;    ///< image width
        int height;   ///< image height
        int ch_count; ///< number of channels
        int maxval;   ///< sample maximal value (typically but not necessarily
                      ///< 255)
        bool bitmap_pbm; ///< bitmap data is stored in PBM format (1 bit per
                         ///< pixel, line aligned to whole byte, 1 is black
                         ///< /"ink on"/), otherwise 1 byte per pixel, 1 is
                         ///< white "light on"); if .depth != 1 || .maxval != 1,
                         ///< this value is undefined
};

/**
 * read PAM/PNM file
 *
 * @param      filename   file name
 * @param[out] info       pointer to metadata struct
 * @param[out] data       pointer to byte array, can be 0, in which case no data
 *                        are written (only metadata read )
 * @param[out] allocaltor allocator to alloc @ref data; if 0, no data are
 *                        read/allocated, only @ref info set
 */
bool pam_read(const char *filename, struct pam_metadata *info, unsigned char **data, void *(*allocator)(size_t));

enum {
        PAM_PITCH_CONTINUOUS = 0,
};

/**
 * write PAM or PNM file
 *
 * @param filename file name to be written to
 * @param width    image width
 * @param pitch    input line pitch in bytes; PAM_PITCH_CONTINUOUS can be used
 *                 if input pitch == width * ch_count * (maxval <= 255 ? 1 : 2)
 * @param height   image height
 * @param ch_count image channel count (1-4 for output PAM, 1 or 3 for PNM, see
 *                 @ref pnm)
 * @param maxval   maximal sample value, typically 255 for 8-bit
 * @param data     bytes to be written
 * @param pnm      use PNM file (instead of PAM)
 */
bool pam_write(const char *filename, unsigned int width, unsigned int pitch,
               unsigned int height, int ch_count, int maxval,
               const unsigned char *data, bool pnm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // defined PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F

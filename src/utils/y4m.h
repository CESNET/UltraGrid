/**
 * @file   utils/y4m.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2022, CESNET z.s.p.o.
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

#ifndef Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B
#define Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum y4m_subsampling {
        Y4M_SUBS_MONO = 400,
        Y4M_SUBS_420 = 420,
        Y4M_SUBS_422 = 422,
        Y4M_SUBS_444 = 444,
        Y4M_SUBS_YUVA = 4444,
};

struct y4m_metadata {
        int width;
        int height;
        int bitdepth;
        int subsampling;
        bool limited;
};

/**
 * @retval 0 on error
 * @returns number of raw image data
 */
size_t y4m_read(const char *filename, struct y4m_metadata *info, unsigned char **data, void *(*allocator)(size_t));
bool y4m_write(const char *filename, int width, int height, enum y4m_subsampling subsampling, int depth, bool limited, const unsigned char *data);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // defined Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B

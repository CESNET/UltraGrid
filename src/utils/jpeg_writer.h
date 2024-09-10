/**
 * @file   utils/jpeg_writer.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @sa jpeg_writer.c
 */
/*
 * Copyright (c) 2024 CESNET
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

#ifndef UTILS_JPEG_WRITER_H_1FB9D4E0_CAED_4C8E_97BA_2ECDAF41F006
#define UTILS_JPEG_WRITER_H_1FB9D4E0_CAED_4C8E_97BA_2ECDAF41F006

#include <stdint.h>  // for uint8_t

struct jpeg_writer_data {
        /// values map to RTP type
        enum gpujpeg_writer_subsasmpling {
                JPEG_WRITER_SUBS_422 = 0,
                JPEG_WRITER_SUBS_420 = 1,
        } subsampling;
        unsigned width;
        unsigned height;
        unsigned restart_interval;

        char *dqt_marker_start; // set by jpeg_writer_write_headers()
};

enum {
        JPEG_QUANT_SIZE = 64, ///< items, bytes (for precision=0)
};

void jpeg_writer_write_headers(char **buffer, struct jpeg_writer_data *info);
void jpeg_writer_fill_dqt(char   *buffer,
                          uint8_t quant_table[2][JPEG_QUANT_SIZE]);
void jpeg_writer_write_eoi(char **buffer);

#endif // defined UTILS_JPEG_WRITER_H_1FB9D4E0_CAED_4C8E_97BA_2ECDAF41F006

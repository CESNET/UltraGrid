/**
 * @file   audio/wav_writer.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET z.s.p.o.
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

#ifndef WAV_WRITER_H_722362F0_155A_11EC_88ED_B7401531ECA5
#define WAV_WRITER_H_722362F0_155A_11EC_88ED_B7401531ECA5

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdbool.h>
#include <stdio.h>
#endif

#include "audio/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct wav_writer_file;

/**
 * @retval pointer to output file to which the caller writes output data.
 *         When done, wav_finalize() needs to be called to finish and close
 *         the file.
 */
struct wav_writer_file *wav_writer_create(const char *filename, struct audio_desc fmt);

/**
 * @retval      0 on success
 * @retval -errno on failure
 */
int wav_writer_write(struct wav_writer_file *wav, long long sample_count, const char *data);

/**
 * @param wav file returned by wav_write_header, will be closed by this call
 *            and must not be used after
 * @param total_samples total samples that were written
 */
bool wav_writer_close(struct wav_writer_file *wav);

#ifdef __cplusplus
}
#endif

#endif // defined WAV_WRITER_H_722362F0_155A_11EC_88ED_B7401531ECA5

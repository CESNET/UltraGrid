/**
 * @file   audio/export.h
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

#ifndef _AUDIO_EXPORT_H_
#define _AUDIO_EXPORT_H_

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct audio_export;
struct audio_frame;

struct audio_export * audio_export_init(const char *filename);
void audio_export_destroy(struct audio_export *state);
void audio_export(struct audio_export *state, struct audio_frame *frame);

/**
 * Configure audio export for exporting samples directly from a sample buffer.
 * Cannot be called when export is already configured by a previous call to
 * this function, or by audio_export().
 */
bool audio_export_configure_raw(struct audio_export *s,
        int bps, int sample_rate, int ch_count);
/**
 * For exporting audio directly from a sample buffer. Before using this,
 * the audio_export needs to be configured with audio_export_configure_raw()
 *
 * @param len    size of data in bytes
 */
void audio_export_raw(struct audio_export *s, void *data, unsigned len);
/**
 * Export audio from separate sample buffers for each channel.
 *
 * @param channels_data    array of pointers to sample buffers
 * @param sample_count     number of samples to export
 */
void audio_export_raw_ch(struct audio_export *s,
                const void **channels_data, unsigned sample_count);

#ifdef __cplusplus
}
#endif


#endif /* _AUDIO_EXPORT_H_ */

/**
 * @file   audio_filter.h
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#ifndef AUDIO_FILTER_H_
#define AUDIO_FILTER_H_

#define AUDIO_FILTER_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct audio_frame;

enum af_result_code{
        AF_MISCONFIGURED = -2,
        AF_FAILURE = -1,
        AF_OK = 0,
        AF_CONFIGURED_CLOSEST = 1,
        AF_HELP_SHOWN = 2,
};

struct audio_filter_info{
        const char *name;

        /// @brief Initializes filter
        /// @param      cfg    configuration string from cmdline
        /// @param[out] state  output state
        /// @retval     0      if initialized successfully
        /// @retval     <0     if error
        /// @retval     >0     no error but state was not returned, eg. showing help
        af_result_code (*init)(const char *cfg, void **state);
        void (*done)(void *state);

        af_result_code (*configure)(void *state,
                        int bps, int ch_count, int sample_rate);

        void (*get_configured_in)(void *state,
                        int *bps, int *ch_count, int *sample_rate);

        void (*get_configured_out)(void *state,
                        int *bps, int *ch_count, int *sample_rate);

        af_result_code (*filter)(void *state, struct audio_frame **f);
};

struct audio_filter{
        const struct audio_filter_info *info;
        void *state;
};

af_result_code audio_filter_init(const char *name, const char *cfg,
                struct audio_filter *filter);

void audio_filter_destroy(struct audio_filter *state);

af_result_code audio_filter_configure(struct audio_filter *state,
                int bps, int ch_count, int sample_rate);

void audio_filter_get_configured_out(struct audio_filter *state,
                int *bps, int *ch_count, int *sample_rate);

void audio_filter_get_configured_in(struct audio_filter *state,
                int *bps, int *ch_count, int *sample_rate);

af_result_code audio_filter(struct audio_filter *state, struct audio_frame **frame);

//void register_audio_filter(struct audio_filter_info *filter);

#ifdef __cplusplus
}
#endif

#endif //AUDIO_FILTER_H_

/**
 * @file   video_capture_param.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @ingroup vidcap
 */
/**
 * Copyright (c) 2013-2017 CESNET, z. s. p. o
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

#ifndef VIDEO_CAPTURE_PARAM_H
#define VIDEO_CAPTURE_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/** @anchor vidcap_flags
 * @name Initialization Flags
 * @{ */
#define VIDCAP_FLAG_AUDIO_EMBEDDED (1<<1) ///< HD-SDI embedded audio
#define VIDCAP_FLAG_AUDIO_AESEBU (1<<2)   ///< AES/EBU audio
#define VIDCAP_FLAG_AUDIO_ANALOG (1<<3)   ///< (balanced) analog audio

#define VIDCAP_FLAG_AUDIO_ANY (VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG)
/** @} */

/**
 * @name Vidcap Parameters Handling Functions
 * @{ */
struct vidcap_params *vidcap_params_allocate(void);
struct vidcap_params *vidcap_params_allocate_next(struct vidcap_params *params);
struct vidcap_params *vidcap_params_copy(const struct vidcap_params *params);
void                  vidcap_params_free_struct(struct vidcap_params *params);
struct vidcap_params *vidcap_params_get_next(const struct vidcap_params *params);
struct vidcap_params *vidcap_params_get_nth(struct vidcap_params *params, int index);
const char           *vidcap_params_get_driver(const struct vidcap_params *params);
unsigned int          vidcap_params_get_flags(const struct vidcap_params *params);
const char           *vidcap_params_get_fmt(const struct vidcap_params *params);
const char           *vidcap_params_get_name(const struct vidcap_params *params);
struct module        *vidcap_params_get_parent(const struct vidcap_params *params);
void                  vidcap_params_set_device(struct vidcap_params *params, const char *config);
void                  vidcap_params_set_capture_filter(struct vidcap_params *params,
                const char *req_capture_filter);
void                  vidcap_params_set_flags(struct vidcap_params *params, unsigned int flags);
/// @}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // VIDEO_CAPTURE_PARAMS_H


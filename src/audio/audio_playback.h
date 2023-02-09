/**
 * @file   audio/audio_playback.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
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

#include "../types.h"

struct audio_desc;
struct audio_frame;

#ifdef __cplusplus
extern "C" {
#endif

extern int audio_init_state_ok;

#define AUDIO_PLAYBACK_ABI_VERSION 7

/** @anchor audio_playback_ctl_reqs
 * @name Audio playback control requests
 * @{ */
/**
 * Queries a format that corresponses the proposed audio format
 * (from network) most closely. It can be the same format or
 * a different one (depending on device capabilities). Returned
 * format will thereafter be used with audio_playback_reconfigure().
 *
 * Support for this ctl is mandatory!
 * @param[in,out] audio_desc
 * - IN: suggested (received) audio format
 * - OUT: corresponding supported (nearest) audio format that playback device is able to use
 */
#define AUDIO_PLAYBACK_CTL_QUERY_FORMAT     1
/**
 * Tells whether audio device can (and is willing to) process multiple incoming streams.
 * Typically, mixer can do that. Otherwise, only most current stream will be passed to
 * the audio device.
 * @param[out] bool
 */
#define AUDIO_PLAYBACK_CTL_MULTIPLE_STREAMS 2
/**
 * Passes network device used for receiving to the playback
 * @param[in] struct rtp *
 */
#define AUDIO_PLAYBACK_PUT_NETWORK_DEVICE   3
/// @}

struct audio_playback_info {
        device_probe_func probe;
        void (*help)(const char *driver_name);
        void *(*init)(const char *cfg); ///< @param cfg is not NULL
        void (*write)(void *state, const struct audio_frame *frame);
        /** Returns device supported format that matches best with propsed audio desc */
        bool (*ctl)(void *state, int request, void *data, size_t *len);
        int (*reconfigure)(void *state, struct audio_desc);
        void (*done)(void *state);
};

struct state_audio_playback;

void                            audio_playback_help(bool full);
void                            audio_playback_init_devices(void);
/**
 * @see display_init
 */
int                             audio_playback_init(const char *device, const char *cfg,
                struct state_audio_playback **);
struct state_audio_playback    *audio_playback_init_null_device(void);

/**
 * @param[in]     s        audio state
 * @param[in]     request  one of @ref audio_playback_ctl_reqs
 * @param[in,out] data	   data to be passed/returned
 * @param[in,out] len	   input/output length of data
 * @return                 status of the request
 */
bool audio_playback_ctl(struct state_audio_playback *s, int request, void *data, size_t *len);

/**
 * Reconfigures audio playback to specified values. Those values
 * must have been exactly the ones obtained from
 * audio_playback_query_supported_format().
 *
 * @param[in] state         audio state
 * @param[in] quant_samples number of quantization bits
 * @param[in] channels      number of channels to be played back
 * @param[in] sample_rate   sample rate
 * @retval TRUE             if reconfiguration succeeded
 * @retval FALSE            if reconfiguration failed
 */
int                             audio_playback_reconfigure(struct state_audio_playback *state,
                int quant_samples, int channels,
                int sample_rate);
void                            audio_playback_put_frame(struct state_audio_playback *state, const struct audio_frame *frame);
void                            audio_playback_finish(struct state_audio_playback *state);
void                            audio_playback_done(struct state_audio_playback *state);

unsigned int                    audio_playback_get_display_flags(struct state_audio_playback *s);

/**
 * @returns directly state of audio capture device. Little bit silly, but it is needed for
 * SDI (embedded sound).
 */
void                       *audio_playback_get_state_pointer(struct state_audio_playback *s);

#ifdef __cplusplus
}
#endif

/* vim: set expandtab sw=8: */


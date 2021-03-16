/**
 * @file   audio/audio_capture.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2019 CESNET, z. s. p. o.
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

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_CAPTURE_ABI_VERSION 3

struct audio_capture_info {
        void (*probe)(struct device_info **available_devices, int *count);
        void (*help)(const char *driver_name);
        void *(*init)(const char *cfg);
        struct audio_frame *(*read)(void *state);
        void (*done)(void *state);
};

struct state_audio_capture;
struct audio_frame;


void                        audio_capture_init_devices(void);
void                        audio_capture_print_help(bool);

/**
 * @see display_init
 */
int                         audio_capture_init(const char *driver, char *cfg,
                struct state_audio_capture **);
struct state_audio_capture *audio_capture_init_null_device(void);
struct audio_frame         *audio_capture_read(struct state_audio_capture * state);
void                        audio_capture_done(struct state_audio_capture * state);

unsigned int                audio_capture_get_vidcap_flags(const char *device_name);
unsigned int                audio_capture_get_vidcap_index(const char *device_name);
const char                 *audio_capture_get_driver_name(struct state_audio_capture * state);
/**
 * returns directly state of audio capture device. Little bit silly, but it is needed for
 * SDI (embedded sound).
 */
void                       *audio_capture_get_state_pointer(struct state_audio_capture *s);

#ifdef __cplusplus
}
#endif

/* vim: set expandtab sw=4: */


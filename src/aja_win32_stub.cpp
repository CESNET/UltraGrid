/**
 * @file aja_win32_stub.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This is a stub file importing actual dll. It sets some variables that
 * would be otherwise referenced directly (audio_capture_channels).
 */
/*
 * Copyright (c) 2017-2023 CESNET z.s.p.o.
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
#include "host.h"
#include "lib_common.h"
#include "video_capture.h"
#include "video_display.h"

extern "C" {
__declspec(dllimport) int vidcap_aja_init(struct vidcap_params *params, void **state);
__declspec(dllimport) void vidcap_aja_done(void *state);
__declspec(dllimport) struct video_frame *vidcap_aja_grab(void *state, struct audio_frame **audio);
__declspec(dllimport) void vidcap_aja_probe(struct device_info **available_cards, int *count, void (**deleter)(void *));
__declspec(dllimport) unsigned int *aja_audio_capture_channels;
}

static int vidcap_aja_init_proxy(struct vidcap_params *params, void **state) {
        aja_audio_capture_channels = &audio_capture_channels;
        return vidcap_aja_init(params, state);
}

static const struct video_capture_info vidcap_aja_info = {
        vidcap_aja_probe,
        vidcap_aja_init_proxy,
        vidcap_aja_done,
        vidcap_aja_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(aja, &vidcap_aja_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

extern "C" {
__declspec(dllimport) bool display_aja_get_property(void *state, int property, void *val, size_t *len);
__declspec(dllimport) void display_aja_probe(struct device_info **available_cards, int *count, void (**deleter)(void *));
__declspec(dllimport) bool display_aja_reconfigure(void *state, struct video_desc desc);
__declspec(dllimport) void *display_aja_init(struct module * /* parent */, const char *fmt, unsigned int flags);
__declspec(dllimport) void display_aja_done(void *state);
__declspec(dllimport) struct video_frame *display_aja_getf(void *state);
__declspec(dllimport) bool display_aja_putf(void *state, struct video_frame *frame, long long nonblock);
__declspec(dllimport) void display_aja_put_audio_frame(void *state, const struct audio_frame *frame);
__declspec(dllimport) bool display_aja_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);
}

static const struct video_display_info display_aja_info = {
        display_aja_probe,
        display_aja_init,
        NULL, // _run
        display_aja_done,
        display_aja_getf,
        display_aja_putf,
        display_aja_reconfigure,
        display_aja_get_property,
        display_aja_put_audio_frame,
        display_aja_reconfigure_audio,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(aja, &display_aja_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


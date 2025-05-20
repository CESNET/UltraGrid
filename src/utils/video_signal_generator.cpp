/**
 * @file   utils/video_signal_generator.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "debug.h"
#include "utils/video_signal_generator.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/testcard.h"

struct video_signal_generator {
	void *testcard_state;
	struct video_frame *out;
};

struct video_signal_generator *video_signal_generator_create(void)
{
	return (struct video_signal_generator *) calloc(1, sizeof(struct video_signal_generator));
}

void video_signal_generator_done(struct video_signal_generator *state)
{
        if (state) {
                if (state->testcard_state) {
                        vidcap_testcard_done(state->testcard_state);
                }

                free(state);
        }
}

void video_signal_generator_reconfigure(struct video_signal_generator *state, struct video_desc desc)
{
        if (state->testcard_state) {
                vidcap_testcard_done(state->testcard_state);
                state->testcard_state = NULL;
        }

        char config_string[100];
        snprintf(config_string, sizeof config_string, "testcard:%d:%d:%.2f%s:%s",
                        desc.width, desc.height, desc.fps *
                                (desc.interlacing == INTERLACED_MERGED ? 2 : 1),
                        desc.interlacing == INTERLACED_MERGED ? "i" : "",
                        get_codec_name(desc.color_spec));

        struct vidcap_params *params = vidcap_params_allocate();
        vidcap_params_set_device(params, config_string);

        if (vidcap_testcard_init(params, &state->testcard_state) != 0) {
                abort();
        }

        vidcap_params_free(params);
}

struct video_frame *video_signal_generator_grab(struct video_signal_generator *state)
{
        struct audio_frame *audio;

        return vidcap_testcard_get_next_frame(state->testcard_state, &audio);
}


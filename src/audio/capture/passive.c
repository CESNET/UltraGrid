/**
 * @file   audio/capture/passive.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file provides a pseudo-capture that emit empty frames.
 * The only objective of this device is to subscribe to the UltraGrid audio
 * mixer (`-r mixer`) as a passive (listen-only) participant.
 */
/*
 * Copyright (c) 2024 CESNET
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

#include <stdint.h>               // for uint32_t
#include <stdlib.h>               // for free, NULL, malloc
#include <stdio.h>                // for printf
#include <string.h>               // for strcmp

#include "audio/audio_capture.h"  // for AUDIO_CAPTURE_ABI_VERSION, audio_ca...
#include "audio/types.h"          // for audio_frame
#include "compat/usleep.h"        // for usleep
#include "host.h"                 // for INIT_NOERR
#include "lib_common.h"           // for REGISTER_MODULE, library_class
#include "tv.h"                   // for get_time_in_ns, time_ns_t, NS_TO_US
#include "types.h"                // for kHz48, device_info (ptr only)
#include "utils/macros.h"         // for to_fourcc
#include "utils/text.h"           // for wrap_paragraph

struct module;

struct acap_state_passive {
        uint32_t  magic;
        time_ns_t last_frame_time_ns;
};

#define AUDIO_CAPTURE_NONE_MAGIC to_fourcc('a', 'c', 'p', 's')

static void
audio_cap_passive_probe(struct device_info **available_devices, int *count,
                        void (**deleter)(void *))
{
        *deleter           = free;
        *available_devices = NULL;
        *count             = 0;
}

static void *
audio_cap_passive_init(struct module *parent, const char *cfg)
{
        (void) parent;

        if (strcmp(cfg, "help") == 0) {
                char desc[] = "This pseudo-device is intended to be used "
                              "together with audio mixer to subbscribe passive "
                              "playback of audio data (listen-only).\n";
                printf("%s\n", wrap_paragraph(desc));
                return INIT_NOERR;
        }

        struct acap_state_passive *s = calloc(1, sizeof *s);
        s->magic = AUDIO_CAPTURE_NONE_MAGIC;

        return s;
}

static struct audio_frame *
audio_cap_passive_read(void *state)
{
        enum {
                FRAME_SPAC_SEC = 1,
        };
        struct acap_state_passive *s = state;

        const time_ns_t            wait_ns =
            s->last_frame_time_ns + SEC_TO_NS(FRAME_SPAC_SEC) - get_time_in_ns();
        if (wait_ns > 0) {
                usleep(NS_TO_US(wait_ns));
        }
        s->last_frame_time_ns = get_time_in_ns();

        static struct audio_frame f = {
                .bps         = 2,
                .sample_rate = kHz48,
                .ch_count    = 1,
        };
        return (struct audio_frame *) &f;
}

static void
audio_cap_passive_done(void *state)
{
        free(state);
}

static const struct audio_capture_info acap_passive_info = {
        audio_cap_passive_probe, audio_cap_passive_init, audio_cap_passive_read,
        audio_cap_passive_done
};

REGISTER_MODULE(passive, &acap_passive_info, LIBRARY_CLASS_AUDIO_CAPTURE,
                AUDIO_CAPTURE_ABI_VERSION);

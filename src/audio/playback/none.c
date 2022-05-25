/**
 * @file   audio/playback/none.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2021 CESNET z.s.p.o.
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

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include <stdlib.h>
#include <string.h>

#define AUDIO_PLAYBACK_NONE_MAGIC 0x3bcf376au

struct state_audio_playback_none
{
        uint32_t magic;
};

static void audio_play_none_help(const char *driver_name)
{
        UNUSED(driver_name);
}

static void audio_play_none_probe(struct device_info **available_devices, int *count)
{
        *available_devices = NULL;
        *count = 0;
}

static void * audio_play_none_init(const char *cfg)
{
        UNUSED(cfg);
        struct state_audio_playback_none *s;

        s = (struct state_audio_playback_none *)
                malloc(sizeof(struct state_audio_playback_none));
        assert(s != NULL);
        s->magic = AUDIO_PLAYBACK_NONE_MAGIC;
                                
        return s;
}

static void audio_play_none_put_frame(void *state, const struct audio_frame *frame)
{
        UNUSED(frame);
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);
}

static void audio_play_none_done(void *state)
{
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);
        free(s);
}

static bool audio_play_none_ctl(void *state, int request, void *data, size_t *len)
{
        UNUSED(state);
        UNUSED(request);
        UNUSED(data);
        UNUSED(len);
        return false;
}

static int audio_play_none_reconfigure(void *state, struct audio_desc desc)
{
        UNUSED(desc);
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);

        return TRUE;
}

static const struct audio_playback_info aplay_none_info = {
        audio_play_none_probe,
        audio_play_none_help,
        audio_play_none_init,
        audio_play_none_put_frame,
        audio_play_none_ctl,
        audio_play_none_reconfigure,
        audio_play_none_done
};

REGISTER_MODULE(none, &aplay_none_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


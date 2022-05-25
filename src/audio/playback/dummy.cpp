/**
 * @file   audio/playback/dummy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2021 CESNET, z. s. p. o.
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
#include "lib_common.h"

static int state;

static void audio_play_dummy_probe(struct device_info **available_devices, int *count)
{
        *available_devices = NULL;
        *count = 0;
}

static void audio_play_dummy_help(const char *)
{
        printf("\tdummy: dummy audio playback\n");
}

static void * audio_play_dummy_init(const char *)
{
        return &state;
}

static void audio_play_dummy_put_frame(void *, const struct audio_frame *)
{
}

static void audio_play_dummy_done(void *)
{
}

static bool audio_play_dummy_ctl(void *state [[gnu::unused]], int request, void *data [[gnu::unused]], size_t *len [[gnu::unused]])
{
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return true; // we accept any format
        default:
                return false;
        }
}

static int audio_play_dummy_reconfigure(void *, struct audio_desc)
{
        return TRUE;
}

static const struct audio_playback_info aplay_dummy_info = {
        audio_play_dummy_probe,
        audio_play_dummy_help,
        audio_play_dummy_init,
        audio_play_dummy_put_frame,
        audio_play_dummy_ctl,
        audio_play_dummy_reconfigure,
        audio_play_dummy_done
};

REGISTER_MODULE(dummy, &aplay_dummy_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


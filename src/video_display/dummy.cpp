/**
 * @file   video_display/dummy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This is an umbrella header for video functions.
 */
/*
 * Copyright (c) 2015 CESNET z.s.p.o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video.h"
#include "video_display.h"
#include "video_display/dummy.h"

struct dummy_display_state {
        struct video_desc desc;
};

void *display_dummy_init(struct module *, const char *, unsigned int)
{
        return new dummy_display_state();
}

void display_dummy_run(void *)
{
}

void display_dummy_done(void *state)
{
        delete (dummy_display_state *) state;
}

struct video_frame *display_dummy_getf(void *state)
{
        return vf_alloc_desc_data(((dummy_display_state *) state)->desc);
}

int display_dummy_putf(void *, struct video_frame *frame, int)
{
        vf_free(frame);
        return 0;
}

int display_dummy_get_property(void *, int property, void *val, size_t *len)
{
        codec_t codecs[] = {UYVY, YUYV, v210, RGBA, RGB, BGR};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

int display_dummy_reconfigure(void *state, struct video_desc desc)
{
        ((dummy_display_state *) state)->desc = desc;

        return TRUE;
}

void display_dummy_put_audio_frame(void *, struct audio_frame *)
{
}

int display_dummy_reconfigure_audio(void *, int, int, int)
{
        return FALSE;
}

const struct video_display_info display_dummy_info = {
        display_dummy_init,
        display_dummy_run,
        display_dummy_done,
        display_dummy_getf,
        display_dummy_putf,
        display_dummy_reconfigure,
        display_dummy_get_property,
        display_dummy_put_audio_frame,
        display_dummy_reconfigure_audio,
};


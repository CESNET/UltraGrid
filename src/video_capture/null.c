/*
 * FILE:   video_capture/null.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
 * A fake video capture device, used for systems that either have no capture
 * hardware or do not wish to transmit. This fits the interface of the other
 * capture devices, but never produces any video.
 *
 * Copyright (c) 2005-2025 CESNET
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 2003 University of Southern California
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 *
 * $Revision: 1.3.2.1 $
 * $Date: 2010/01/30 20:07:35 $
 *
 */

#include <assert.h>                // for assert
#include <stdlib.h>                // for NULL, free

#include "lib_common.h"            // for REGISTER_HIDDEN_MODULE, library_class
#include "video_capture.h"         // for VIDCAP_INIT_AUDIO_NOT_SUPPORTED
#include "video_capture_params.h"  // for vidcap_params_get_flags, VIDCAP_FL...

struct audio_frame;
struct device_info;
struct vidcap_params;

static int capture_state = 0;

static int vidcap_null_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }
        capture_state = 0;
        *state = &capture_state;
        return VIDCAP_INIT_OK;
}

static void vidcap_null_done(void *state)
{
        assert(state == &capture_state);
}

static struct video_frame *vidcap_null_grab(void *state, struct audio_frame **audio)
{
        assert(state == &capture_state);
        *audio = NULL;
        return NULL;
}

static void vidcap_null_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_cards = NULL;
        *count = 0;
}

static const struct video_capture_info vidcap_null_info = {
        vidcap_null_probe,
        vidcap_null_init,
        vidcap_null_done,
        vidcap_null_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_HIDDEN_MODULE(none, &vidcap_null_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


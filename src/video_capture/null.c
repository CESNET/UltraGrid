/*
 * FILE:   vidcap_null.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
 * A fake video capture device, used for systems that either have no capture
 * hardware or do not wish to transmit. This fits the interface of the other
 * capture devices, but never produces any video.
 *
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/null.h"

static int capture_state = 0;

void *vidcap_null_init(char *fmt, unsigned int flags)
{
        UNUSED(fmt);
        UNUSED(flags);
        capture_state = 0;
        return &capture_state;
}

void vidcap_null_finish(void *state)
{
        assert(state == &capture_state);
}

void vidcap_null_done(void *state)
{
        assert(state == &capture_state);
}

struct video_frame *vidcap_null_grab(void *state, struct audio_frame **audio)
{
        assert(state == &capture_state);
        *audio = NULL;
        return NULL;
}

struct vidcap_type *vidcap_null_probe(void)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = VIDCAP_NULL_ID;
                vt->name = "none";
                vt->description = "No video capture device";
        }
        return vt;
}

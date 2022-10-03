/*
 * FILE:    audio/capture/none.h
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "audio/audio_capture.h"
#include "debug.h"
#include "lib_common.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define AUDIO_CAPTURE_NONE_MAGIC 0x43fb99ccu

struct state_audio_capture_none {
        uint32_t magic;
};

static void audio_cap_none_probe(struct device_info **available_devices, int *count)
{
        *available_devices = NULL;
        *count = 0;
}

static void audio_cap_none_help(const char *driver_name)
{
        UNUSED(driver_name);
}

static void * audio_cap_none_init(struct module *parent, const char *cfg)
{
        UNUSED(parent);
        struct state_audio_capture_none *s;

        s = (struct state_audio_capture_none *) malloc(sizeof(struct state_audio_capture_none));
        s->magic = AUDIO_CAPTURE_NONE_MAGIC;
        assert(s != 0);
        UNUSED(cfg);
        return s;
}

static struct audio_frame *audio_cap_none_read(void *state)
{
        UNUSED(state);
        return NULL;
}

static void audio_cap_none_done(void *state)
{
        struct state_audio_capture_none *s = (struct state_audio_capture_none *) state;

        assert(s->magic == AUDIO_CAPTURE_NONE_MAGIC);
        free(s);
}

static const struct audio_capture_info acap_none_info = {
        audio_cap_none_probe,
        audio_cap_none_help,
        audio_cap_none_init,
        audio_cap_none_read,
        audio_cap_none_done
};

REGISTER_MODULE(none, &acap_none_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


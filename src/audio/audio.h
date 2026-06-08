/*
 * FILE:    audio/audio.h
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
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

#ifndef _AUDIO_H_
#define _AUDIO_H_

#define DEFAULT_AUDIO_FEC       "none"
#define DEFAULT_AUDIO_SCALE     "mixauto"

#define PORT_AUDIO              5006

#include "audio/types.h"
#include "host.h" // common_opt
#include "module.h"

struct state_audio;

struct audio_options {
        struct module     *parent;
        struct exporter   *exporter;
        const char        *recv_cfg;
        const char        *send_cfg;
        const char        *channel_map;
        const char        *scale;
        bool               echo_cancellation;
        const char        *codec_cfg;
        const char        *filter_cfg;
        struct video_rxtx *vrxtx;
        struct display    *display;
};

#define AUDIO_OPTIONS_INIT \
        { \
                .parent            = nullptr, \
                .exporter          = nullptr, \
                .recv_cfg          = "none", \
                .send_cfg          = "none", \
                .channel_map       = nullptr, \
                .scale             = DEFAULT_AUDIO_SCALE, \
                .echo_cancellation = false, \
                .codec_cfg         = "PCM", \
                .filter_cfg        = "", \
                .vrxtx             = nullptr, \
                .display           = nullptr, \
        }

#ifdef __cplusplus
extern "C" {
#endif

int audio_init(struct state_audio **state,
               const struct audio_options *opt);

void audio_start(struct state_audio *s);
void audio_done(struct state_audio *s);
void audio_join(struct state_audio *s);

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame);
struct audio_frame * sdi_get_frame(void *state);
void sdi_put_frame(void *state, struct audio_frame *frame);

struct audio_frame * audio_get_frame(struct state_audio *s);

unsigned int audio_get_display_flags(const char *playback_dev);

void sdp_send_change_address_message(struct module           *root,
                                     const enum module_class *path,
                                     const char              *address);

#ifdef __cplusplus
}
#endif


#endif

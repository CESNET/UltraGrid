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
 * Copyright (c) 2005-2024 CESNET
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

#define AUDIO_PROTOCOLS "JACK, rtsp, sdp or ultragrid_rtp" // available audio protocols
#define PORT_AUDIO              5006

#include "audio/types.h"
#include "host.h" // common_opt
#include "module.h"
#include "tv.h"

struct state_audio;

#ifdef __cplusplus
#include <chrono>

struct audio_options {
        const char *host = NULL;
        int recv_port = 0;
        int send_port = 0;
        const char *recv_cfg = "none";
        const char *send_cfg = "none";
        const char *proto = "ultragrid_rtp";
        const char *proto_cfg = "";
        const char *fec_cfg = DEFAULT_AUDIO_FEC;
        char *channel_map = nullptr;
        const char *scale = DEFAULT_AUDIO_SCALE;
        bool echo_cancellation = false;
        const char *codec_cfg = "PCM";
        const char *filter_cfg = "";
};

int audio_init(struct state_audio **state, struct module *parent,
               const struct audio_options *opt,
               const struct common_opts   *common);
#endif

#ifdef __cplusplus
extern "C" {
#endif

void audio_start(struct state_audio *s);
void audio_done(struct state_audio *s);
void audio_join(struct state_audio *s);

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame);
struct audio_frame * sdi_get_frame(void *state);
void sdi_put_frame(void *state, struct audio_frame *frame);

struct display;
struct video_rxtx;
struct additional_audio_data {
        struct {
                void *udata;
                void (*putf)(struct display *, const struct audio_frame *);
                bool (*reconfigure)(struct display *, int, int, int);
                bool (*get_property)(struct display *, int, void *, size_t *);
        } display_callbacks;
        struct video_rxtx *vrxtx;
};
void audio_register_aux_data(struct state_audio          *s,
                             struct additional_audio_data data);

struct audio_frame * audio_get_frame(struct state_audio *s);

unsigned int audio_get_display_flags(struct state_audio *s);

void sdp_send_change_address_message(struct module           *root,
                                     const enum module_class *path,
                                     const char              *address);

#ifdef __cplusplus
}
#endif


#endif

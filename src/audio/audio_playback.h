/*
 * FILE:    audio/audio_playback.h
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


struct state_audio_playback;

void                            audio_playback_help(void);
void                            audio_playback_init_devices(void);
/**
 * @see display_init
 */
int                             audio_playback_init(char *device, char *cfg,
                struct state_audio_playback **);
struct state_audio_playback    *audio_playback_init_null_device(void);
int                             audio_playback_reconfigure(struct state_audio_playback *state,
                int quant_samples, int channels,
                int sample_rate);
void                            audio_playback_put_frame(struct state_audio_playback *state, struct audio_frame *frame);
void                            audio_playback_finish(struct state_audio_playback *state);
void                            audio_playback_done(struct state_audio_playback *state);

unsigned int                    audio_playback_get_display_flags(struct state_audio_playback *s);
void audio_register_put_callback(struct state_audio *s, void (*callback)(void *, struct audio_frame *),
                void *udata);
void audio_register_get_callback(struct state_audio *s, struct audio_frame * (*callback)(void *),
                void *udata);
void audio_register_reconfigure_callback(struct state_audio *s, int (*callback)(void *, int, int, int),
                void *udata);

/**
 * @returns directly state of audio capture device. Little bit silly, but it is needed for
 * SDI (embedded sound).
 */
void                       *audio_playback_get_state_pointer(struct state_audio_playback *s);

/* vim: set expandtab: sw=8 */


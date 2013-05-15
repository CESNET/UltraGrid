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

#ifndef _AUDIO_H_
#define _AUDIO_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#define PORT_AUDIO              5006

#define MAX_AUDIO_CHANNELS      8

extern int audio_init_state_ok;

typedef enum {
        AC_NONE,
        AC_PCM,
        AC_ALAW,
        AC_MULAW,
        AC_ADPCM_IMA_WAV,
        AC_SPEEX,
        AC_OPUS,
        AC_G722,
        AC_G726,
} audio_codec_t;

struct state_audio;

struct audio_desc {
        int bps;                /* bytes per sample */
        int sample_rate;
        int ch_count;		/* count of channels */
        audio_codec_t codec;
};

/**
 * @deprecated use audio_frame2 instead
 */
typedef struct audio_frame
{
        int bps;                /* bytes per sample */
        int sample_rate;
        char *data; /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        int ch_count;		/* count of channels */
        unsigned int max_size;  /* maximal size of data in buffer */
}
audio_frame;

typedef struct
{
        int bps;                /* bytes per sample */
        int sample_rate;
        char *data[MAX_AUDIO_CHANNELS]; /* data should be at least 4B aligned */
        int data_len[MAX_AUDIO_CHANNELS];           /* size of useful data in buffer */
        int ch_count;		/* count of channels */
        unsigned int max_size;  /* maximal size of data in buffer */
        audio_codec_t codec;
} audio_frame2;

typedef struct
{
        int bps;                /* bytes per sample */
        int sample_rate;
        char *data; /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        audio_codec_t codec;
} audio_channel;

struct state_audio * audio_cfg_init(char *addrs, int recv_port, int send_port,
                const char *send_cfg, const char *recv_cfg,
                char *jack_cfg, char *fec_cfg, char *audio_channel_map, const char *audio_scale,
                bool echo_cancellation, bool use_ipv6, char *mcast_iface, audio_codec_t audio_codec,
                int resample_to);
void audio_finish(struct state_audio *s);
void audio_done(struct state_audio *s);
void audio_join(struct state_audio *s);

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame);
struct audio_frame * sdi_get_frame(void *state);
void sdi_put_frame(void *state, struct audio_frame *frame);
void audio_register_put_callback(struct state_audio *s, void (*callback)(void *, struct audio_frame *),
                void *udata);
void audio_register_reconfigure_callback(struct state_audio *s, int (*callback)(void *, int, int, int),
                void *udata);

struct audio_frame * audio_get_frame(struct state_audio *s);
int audio_reconfigure(struct state_audio *s, int quant_samples, int channels,
                int sample_rate);

unsigned int audio_get_vidcap_flags(struct state_audio *s);
unsigned int audio_get_display_flags(struct state_audio *s);

/**
 * Changes bps for everey sample.
 * 
 * The memory areas shouldn't (supposedly) overlap.
 */
void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */);

/**
 * Makes n copies of first channel (interleaved).
 */
void audio_frame_multiply_channel(struct audio_frame *frame, int new_channel_count);

#endif

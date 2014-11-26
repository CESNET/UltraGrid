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

#define PORT_AUDIO              5006

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
        AC_MP3,
} audio_codec_t;

struct state_audio;

struct audio_desc {
        int bps;                /* bytes per sample */
        int sample_rate;
        int ch_count;		/* count of channels */
        audio_codec_t codec;
};

#ifdef __cplusplus
#include <ostream>
std::ostream& operator<<(std::ostream& os, const audio_desc& desc);
#endif

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
        const char *data; /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        audio_codec_t codec;
        double duration;
} audio_channel;

struct module;

#ifdef __cplusplus
#include <memory>
#include <utility>
#include <vector>

class audio_frame2
{
public:
        audio_frame2();
        audio_frame2(audio_frame2&& other) = default;
        audio_frame2(struct audio_frame *);
        audio_frame2& operator=(audio_frame2&& other) = default;
        void init(int nr_channels, audio_codec_t codec, int bps, int sample_rate);
        void append(audio_frame2 const &frame);
        void append(int channel, const char *data, size_t length);
        void replace(int channel, size_t offset, const char *data, size_t length);
        void resize(int channel, size_t len);
        void reset();
        int get_bps() const;
        audio_codec_t get_codec() const;
        const char *get_data(int channel) const;
        size_t get_data_len(int channel) const;
        double get_duration() const;
        int get_channel_count() const;
        int get_sample_count() const;
        int get_sample_rate() const;
        bool has_same_prop_as(audio_frame2 const &frame) const;
        void set_duration(double duration);
private:
        int bps;                /* bytes per sample */
        int sample_rate;
        std::vector<std::pair<std::unique_ptr<char []>, size_t> > channels; /* data should be at least 4B aligned */
        audio_codec_t codec;
        double duration;
};
#endif


#ifdef __cplusplus
extern "C" {
#endif

struct state_audio * audio_cfg_init(struct module *parent, const char *addrs, int recv_port, int send_port,
                const char *send_cfg, const char *recv_cfg,
                char *jack_cfg, const char *fec_cfg, const char *encryption,
                char *audio_channel_map, const char *audio_scale,
                bool echo_cancellation, bool use_ipv6, const char *mcast_iface, const char *audio_codec_cfg,
                bool isStd, long packet_rate);
void audio_finish(void);
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

unsigned int audio_get_display_flags(struct state_audio *s);

#ifdef __cplusplus
}
#endif


#endif

/**
 * @file   audio/capture/testcard.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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

#define MODULE_NAME "[Audio testcard] "

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "audio/capture/testcard.h"

#include "audio/audio.h"

#include "audio/utils.h"
#include "audio/wav_reader.h"
#include "debug.h"
#include "host.h"
#include "tv.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define AUDIO_CAPTURE_TESTCARD_MAGIC 0xf4b3c9c9u

#define AUDIO_SAMPLE_RATE 48000ll
#define AUDIO_BPS 2

#define CHUNK (AUDIO_SAMPLE_RATE/25) // 1 video frame time @25 fps
                   // has to be divisor of AUDIO_SAMLE_RATE

#define FREQUENCY 1000
#define DEFAULT_VOLUME -18.0

struct state_audio_capture_testcard {
        uint32_t magic;

        struct audio_frame audio;

        double audio_remained,
               seconds_tone_played;
        char *audio_samples;

        struct timeval next_audio_time;

        int samples_played;

        int total_samples;
};

void audio_cap_testcard_help(const char *driver_name)
{
        UNUSED(driver_name);
        printf("\ttestcard : Testing sound signal\n");
}

/**
 * Generates line-up EBU tone according to https://tech.ebu.ch/docs/tech/tech3304.pdf
 */
static char *get_ebu_signal(int sample_rate, int bps, int channels, int frequency, double volume, int *total_samples) {
                *total_samples = (3 + channels + 1 + 3) * sample_rate;
                char *ret = (char *) calloc(1, *total_samples * channels * bps);
                double scale = pow(10.0, volume/20.0) * sqrt(2.0);

                char* data = ret;
                for (int i=0; i < (int) sample_rate * 3; i += 1)
                {
                        for (int channel = 0; channel < channels; ++channel) {
                                int64_t val = sin( ((double)(i)/((double)sample_rate / frequency)) * M_PI * 2. ) * ((1ll<<(bps*8)) / 2 - 1) * scale;
                                format_to_out_bps(data + i * bps * channels + bps * channel,
                                                bps, val);
                        }
                }
                data += sample_rate * 3 * bps * channels;

                for (int channel = 0; channel < channels; ++channel) {
                        memset(data, 0, sample_rate * bps * channels);
                        data += sample_rate * bps * channels / 2;
                        for (int i=0; i < (int) sample_rate / 2; i += 1)
                        {
                                int64_t val = sin( ((double)(i)/((double)sample_rate / frequency)) * M_PI * 2. ) * ((1ll<<(bps*8)) / 2 - 1) * scale;
                                format_to_out_bps(data + i * bps * channels + bps * channel,
                                                bps, val);
                        }
                        data += sample_rate * bps * channels / 2;
                }

                memset(data, 0, sample_rate * bps * channels);
                data += sample_rate * bps * channels;

                memcpy(data, ret, sample_rate * 3 * bps * channels);

                return ret;
}

void * audio_cap_testcard_init(char *cfg)
{
        struct state_audio_capture_testcard *s;
        char *wav_file = NULL;
        char *item, *save_ptr;

        double volume = DEFAULT_VOLUME;

        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available testcard capture:\n");
                audio_cap_testcard_help(NULL);
                printf("\toptions\n\t\ttestcard[:volume=<vol>][:file=<wav>]\n");
                printf("\t\t\t<vol> is a volume in dBFS (default %.2f dBFS)\n", DEFAULT_VOLUME);
                printf("\t\t\t<wav> is a wav file to be played\n");
                return &audio_init_state_ok;
        }

        if(cfg) {
                while((item = strtok_r(cfg, ":", &save_ptr))) {
                        if(strncasecmp(item, "vol=", strlen("vol=")) == 0) {
                                volume = atof(item + strlen("vol="));
                        } else if(strncasecmp(item, "file=", strlen("file=")) == 0) {
                                wav_file = item + strlen("file=");
                        }

                        cfg = NULL;
                }
        }

        s = (struct state_audio_capture_testcard *) malloc(sizeof(struct state_audio_capture_testcard));
        assert(s != 0);
        s->magic = AUDIO_CAPTURE_TESTCARD_MAGIC;

        if(!wav_file) {
                printf(MODULE_NAME "Generating %d Hz (%.2f RMS dBFS) EBU tone ", FREQUENCY,
                                volume);
                printf("(channels: %u; bps: %d; sample rate: %lld; frames per packet: %lld).\n", audio_capture_channels,
                                AUDIO_BPS, AUDIO_SAMPLE_RATE, CHUNK);

                s->audio_samples = get_ebu_signal(AUDIO_SAMPLE_RATE, AUDIO_BPS, audio_capture_channels,
                                FREQUENCY, volume, &s->total_samples);

                s->audio_samples = realloc(s->audio_samples, (s->total_samples *
                                audio_capture_channels * AUDIO_BPS) + CHUNK - 1);
                memcpy(s->audio_samples + s->total_samples * AUDIO_BPS * audio_capture_channels,
                                s->audio_samples, CHUNK - 1);

                s->audio.bps = AUDIO_BPS;
                s->audio.ch_count = audio_capture_channels;
                s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        } else {
                FILE *wav = fopen(wav_file, "r");
                if(!wav) {
                        fprintf(stderr, "Unable to open WAV.\n");
                        free(s);
                        return NULL;
                }
                struct wav_metadata metadata;
                int ret = read_wav_header(wav, &metadata);
                if(ret != WAV_HDR_PARSE_OK) {
                        print_wav_error(ret);
                        fclose(wav);
                        free(s);
                        return NULL;
                }
                s->audio.bps = metadata.bits_per_sample / 8;
                s->audio.ch_count = metadata.ch_count;
                s->audio.sample_rate = metadata.sample_rate;
                s->audio.max_size = metadata.data_size + (CHUNK - 1) * metadata.ch_count *
                        (metadata.bits_per_sample / 8);

                s->total_samples = metadata.data_size /  metadata.ch_count / metadata.bits_per_sample / 8;

                s->audio_samples = calloc(1, s->audio.max_size);
                int bytes = fread(s->audio_samples, 1, s->audio.max_size, wav);
                if(bytes != (int) s->audio.max_size) {
                        s->audio.max_size = bytes;
                        fprintf(stderr, "Warning: premature end of WAV file!\n");
                }
                memcpy(s->audio_samples + metadata.data_size, s->audio_samples, s->audio.max_size -
                                metadata.data_size);
                fclose(wav);
        }

        s->audio.data_len = CHUNK * s->audio.bps * s->audio.ch_count;
        s->audio.data = (char *) calloc(1, s->audio.data_len);

        s->samples_played = 0;

        gettimeofday(&s->next_audio_time, NULL);

        return s;
}

struct audio_frame *audio_cap_testcard_read(void *state)
{
        struct state_audio_capture_testcard *s;
        s = (struct state_audio_capture_testcard *) state;
        struct timeval curr_time;

        gettimeofday(&curr_time, NULL);

        if(tv_gt(s->next_audio_time, curr_time)) {
                usleep(tv_diff_usec(s->next_audio_time, curr_time));
        } else {
                // we missed more than 2 "frame times", in that case, just drop the packages
                if (tv_diff_usec(curr_time, s->next_audio_time) > 2 * (1000 * 1000 * CHUNK / AUDIO_SAMPLE_RATE)) {
                        s->next_audio_time = curr_time;
                        fprintf(stderr, MODULE_NAME "Warning: skipping some samples (late grab call).\n");
                }
        }

        tv_add_usec(&s->next_audio_time, 1000 * 1000 * CHUNK / AUDIO_SAMPLE_RATE);

        size_t samples = CHUNK;
        if (s->samples_played + CHUNK  > s->total_samples) {
                samples = s->total_samples - s->samples_played;
        }
        size_t len = samples * AUDIO_BPS * audio_capture_channels;
        memcpy(s->audio.data, s->audio_samples + AUDIO_BPS * s->samples_played * audio_capture_channels, len);
        if (samples < CHUNK) {
                memcpy(s->audio.data + len, s->audio_samples, CHUNK * AUDIO_BPS * audio_capture_channels - len);
        }

        s->samples_played = ((s->samples_played + CHUNK) % s->total_samples);

        return &s->audio;
}

void audio_cap_testcard_finish(void *state)
{
        UNUSED(state);
}

void audio_cap_testcard_done(void *state)
{
        struct state_audio_capture_testcard *s = (struct state_audio_capture_testcard *) state;

        assert(s->magic == AUDIO_CAPTURE_TESTCARD_MAGIC);

        free(s->audio_samples);
        free(s->audio.data);

        free(s);
}


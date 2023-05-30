/**
 * @file   audio/capture/testcard.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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

#define MOD_NAME "[Audio testcard] "

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "audio/wav_reader.h"
#include "compat/misc.h"
#include "compat/usleep.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"

#include <assert.h>
#include <string.h>

#define AUDIO_CAPTURE_TESTCARD_MAGIC 0xf4b3c9c9u

#define DEFAULT_AUDIO_BPS 2
#define DEFAULT_AUDIO_SAMPLE_RATE 48000

#define CHUNKS_PER_SEC 25 // 1 video frame time @25 fps
                   // has to be divisor of AUDIO_SAMLE_RATE

#define DEFAULT_FREQUENCY 1000
#define DEFAULT_VOLUME -18.0

struct state_audio_capture_testcard {
        uint32_t magic;

        unsigned long long int chunk_size;
        struct audio_frame audio;

        double audio_remained,
               seconds_tone_played;
        char *audio_samples;

        time_ns_t next_audio_time;

        unsigned int samples_played;

        unsigned int total_samples;

        int crescendo_speed;
};

static void audio_cap_testcard_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = calloc(1, sizeof(struct device_info));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Testing 1 kHZ signal");
        *count = 1;
}

/// @param if crescendo_spd >= 1, amplitude of sound is increasing
static char *get_sine_signal(int sample_rate, int bps, int channels, int frequency, double volume, int crescendo_spd) {
        char *data = (char *) calloc(1, sample_rate * channels * bps);
        double scale = pow(10.0, volume / 20.0) * sqrt(2.0);
        bool dither = sample_rate % frequency != 0 && get_commandline_param("no-dither") == NULL;

        for (int i = 0; i < (int) sample_rate; i += 1) {
                for (int channel = 0; channel < channels; ++channel) {
                        double sine = sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. );
                        if (crescendo_spd != 0) {
                                double multiplier = ((double) ((i * crescendo_spd) % sample_rate) / sample_rate);
                                sine *= 2 * multiplier; // up to 2x amplitude
                        }
                        int32_t val = CLAMP(sine * INT32_MAX * scale, (double) INT32_MIN, (double) INT32_MAX);
                        change_bps2(data + ((ptrdiff_t) i) * bps * channels + ((ptrdiff_t) bps) * channel,
                                        bps, (char *) &val, sizeof val, 1 * sizeof val, dither);
                }
        }

        return data;
}

/**
 * Generates line-up EBU tone according to https://tech.ebu.ch/docs/tech/tech3304.pdf
 */
static char *get_ebu_signal(int sample_rate, int bps, int channels, int frequency, double volume, unsigned int *total_samples) {
        *total_samples = (3 + channels + 1 + 3) * sample_rate;
        char *ret = (char *) calloc(1, *total_samples * channels * bps);
        double scale = pow(10.0, volume / 20.0) * sqrt(2.0);

        char* data = ret;
        for (int i = 0; i < (int) sample_rate * 3; i += 1)
        {
                for (int channel = 0; channel < channels; ++channel) {
                        int64_t val = round(sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. ) * ((1U << (bps * 8U - 1)) - 1) * scale);
                        //fprintf(stderr, "%ld ", val);
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
                        int64_t val = sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. ) * ((1U << (bps * 8U - 1)) - 1) * scale;
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

static bool audio_testcard_read_wav(const char *wav_filename, struct audio_frame *audio_frame,
                char **audio_samples, unsigned int *total_samples,
                unsigned long long int *chunk_size) {
        FILE *wav = fopen(wav_filename, "rb");
        if(!wav) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to open WAV file: %s.\n", wav_filename);
                return false;
        }
        struct wav_metadata metadata;
        int ret = read_wav_header(wav, &metadata);
        if(ret != WAV_HDR_PARSE_OK) {
                log_msg(LOG_LEVEL_ERROR, "%s\n", get_wav_error(ret));
                fclose(wav);
                return false;
        }
        audio_frame->bps = audio_capture_bps ? audio_capture_bps : metadata.bits_per_sample / 8;
        audio_frame->ch_count = metadata.ch_count;
        audio_frame->sample_rate = metadata.sample_rate;
        if (*chunk_size == 0) {
                *chunk_size = audio_frame->sample_rate / CHUNKS_PER_SEC;
        }

        *total_samples = metadata.data_size  * 8ULL /  metadata.ch_count / metadata.bits_per_sample;
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "%u samples read from file %s\n", *total_samples, wav_filename);

        const int headroom = (*chunk_size - 1) * audio_frame->ch_count * audio_frame->bps;
        const int samples_data_size = *total_samples * audio_frame->ch_count * audio_frame->bps;

        const int audio_samples_size = samples_data_size + headroom;
        *audio_samples = (char *) calloc(1, audio_samples_size);

        char *read_to;
        char *tmp = NULL;
        if ((unsigned) audio_frame->bps == metadata.bits_per_sample / 8) {
                read_to = *audio_samples;
        } else {
                tmp = (char *) calloc(1, metadata.data_size);
                read_to = tmp;
        }
        unsigned int samples = wav_read(read_to, *total_samples, wav, &metadata);
        int bytes = samples * (metadata.bits_per_sample / 8) * metadata.ch_count;
        if (samples != *total_samples) {
                log_msg(samples > 0 ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR, MOD_NAME "Warning: premature end of WAV file (%d read, %lld expected)!\n", bytes, metadata.data_size);
                if (samples == 0) {
                        fclose(wav);
                        return false;
                }
                *total_samples = samples;
        }
        fclose(wav);

        if ((unsigned) audio_frame->bps != metadata.bits_per_sample / 8){
                change_bps(*audio_samples, audio_frame->bps, tmp, metadata.bits_per_sample / 8, bytes);
        }
        free(tmp);

        memcpy(*audio_samples + samples_data_size, *audio_samples, headroom);

        return true;
}


static void * audio_cap_testcard_init(struct module *parent, const char *cfg)
{
        UNUSED(parent);
        const char *wav_file = NULL;
        char *item, *save_ptr;

        double volume = DEFAULT_VOLUME;
        int frequency = DEFAULT_FREQUENCY;
        enum {
                SINE,
                EBU,
                WAV,
                SILENCE,
                CRESCENDO,
        } pattern = SINE;

        if(cfg && strcmp(cfg, "help") == 0) {
                struct key_val options[] = {
                        { "volume=<vol>", "a volume in dBFS (default " TOSTRING(DEFAULT_VOLUME) ")" },
                        { "file=<wav>", "a wav file to be played" },
                        { "frames=<nf>", "sets number of audio frames per packet" },
                        { "frequency=<f>", "frequency of sinusoide" },
                        { "ebu", "use EBU sound" },
                        { "silence", "emit silence" },
                        { "crescendo[=<spd>]", "produce amplying sinusoide (optionally accelerated)" },
                        { NULL, NULL }
                };
                print_module_usage("-s testcard", options, NULL, false);
                color_printf("\nYou can also consider using " TBOLD("sdl_mixer") " audio capture card to generate a more complex pattern. "
                                "(It already includes a MIDI that can be played immediately.)\n");
                return INIT_NOERR;
        }

        struct state_audio_capture_testcard *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->magic = AUDIO_CAPTURE_TESTCARD_MAGIC;
        s->crescendo_speed = 1;
        bool failed = false;
        char *tmp = NULL;

        do {
                if (cfg == NULL) {
                        break;
                }
                tmp = strdup(cfg);
                char *fmt = tmp;
                while ((item = strtok_r(fmt, ":", &save_ptr))) {
                        if(strncasecmp(item, "vol=", strlen("vol=")) == 0 ||
                                        strncasecmp(item, "volume=", strlen("volume=")) == 0) {
                                volume = atof(strchr(item, '=') + 1);
                        } else if(strncasecmp(item, "file=", strlen("file=")) == 0) {
                                wav_file = strdupa(item + strlen("file="));
                                pattern = WAV;
                        } else if(strncasecmp(item, "frames=", strlen("frames=")) == 0) {
                                s->chunk_size = atoi(item + strlen("frames="));
                        } else if (strncasecmp(item, "frequency=", strlen("frequency=")) == 0) {
                                frequency = atoi(item + strlen("frequency="));
                        } else if(strstr(item, "crescendo") == item) {
                                pattern = CRESCENDO;
                                char *val = strchr(item, '=');
                                if (val) {
                                        val += 1;
                                        if (strcmp(val, "help") == 0) {
                                                printf("-s testcard:crescendo=<val>\n\t<val> - crescendo duration multiplier (default 1)\n");
                                                failed = true; break;
                                        }
                                        s->crescendo_speed = atoi(val);
                                }
                        } else if(strcasecmp(item, "ebu") == 0) {
                                pattern = EBU;
                        } else if(strcasecmp(item, "silence") == 0) {
                                pattern = SILENCE;
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", item);
                                failed = true; break;
                        }

                        fmt = NULL;
                }
        } while(false);
        free(tmp);
        if (failed) {
                free(s);
                return NULL;
        }

        if (pattern == WAV) {
                if (!audio_testcard_read_wav(wav_file, &s->audio,
                                        &s->audio_samples, &s->total_samples, &s->chunk_size)) {
                        free(s);
                        return NULL;
                }
        } else {
                s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
                s->audio.sample_rate = audio_capture_sample_rate ? audio_capture_sample_rate :
                        DEFAULT_AUDIO_SAMPLE_RATE;
                s->audio.bps = audio_capture_bps ? audio_capture_bps : DEFAULT_AUDIO_BPS;
                if (s->chunk_size == 0) {
                        s->chunk_size = s->audio.sample_rate / CHUNKS_PER_SEC;
                }
                assert(s->chunk_size > 0);
                switch (pattern) {
                case CRESCENDO:
                case SINE:
                        s->audio_samples = get_sine_signal(s->audio.sample_rate, s->audio.bps, s->audio.ch_count,
                                        frequency, volume, pattern == CRESCENDO ? s->crescendo_speed : 0);
                        s->total_samples = s->audio.sample_rate;
                        break;
                case EBU:
                        s->audio_samples = get_ebu_signal(s->audio.sample_rate, s->audio.bps, s->audio.ch_count,
                                        frequency, volume, &s->total_samples);
                        break;
                case SILENCE:
                        s->total_samples = s->audio.sample_rate;
                        s->audio_samples = (char *) calloc(1, (s->total_samples *
                                                s->audio.ch_count * s->audio.bps) + s->chunk_size - 1);
                        break;
                default:
                        abort();
                }

                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Generating %d Hz (%.2f  RMS dBFS) %s (%s, frames per packet: %llu).\n",
                        frequency, volume, (pattern == SINE ? "sine" : pattern == EBU ? "EBU tone" : "silence"), audio_desc_to_cstring(audio_desc_from_frame(&s->audio)), s->chunk_size);

                // add padding if s->total_samples % s->chunk_size != 0
                s->audio_samples = (char *) realloc(s->audio_samples, (s->total_samples *
                                        s->audio.ch_count * s->audio.bps) + s->chunk_size - 1);
                memcpy(s->audio_samples + s->total_samples * s->audio.bps * s->audio.ch_count,
                                s->audio_samples, s->chunk_size - 1);
        }

        s->audio.data_len = s->chunk_size * s->audio.bps * s->audio.ch_count;
        s->audio.max_size = s->audio.data_len;
        s->audio.data = (char *) calloc(1, s->audio.max_size);

        s->samples_played = 0;

        s->next_audio_time = get_time_in_ns();

        return s;
}

static struct audio_frame *audio_cap_testcard_read(void *state)
{
        struct state_audio_capture_testcard *s;
        s = (struct state_audio_capture_testcard *) state;
        time_ns_t curr_time = get_time_in_ns();

        if( s->next_audio_time > curr_time) {
                usleep((s->next_audio_time - curr_time) / US_IN_NS);
        } else {
                // we missed more than 2 "frame times", in that case, just drop the packages
                if ((curr_time - s->next_audio_time) > (long long int) (2 * NS_IN_SEC * s->chunk_size / s->audio.sample_rate)) {
                        s->next_audio_time = curr_time;
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: skipping some samples (late grab call).\n");
                }
        }

        s->next_audio_time += NS_IN_SEC * s->chunk_size / s->audio.sample_rate;

        size_t samples = s->chunk_size;
        if (s->samples_played + s->chunk_size  > s->total_samples) {
                samples = s->total_samples - s->samples_played;
        }
        size_t len = samples * s->audio.bps * s->audio.ch_count;
        memcpy(s->audio.data, s->audio_samples + s->audio.bps * s->samples_played * s->audio.ch_count, len);
        if (samples < s->chunk_size) {
                memcpy(s->audio.data + len, s->audio_samples, s->chunk_size * s->audio.bps * s->audio.ch_count - len);
        }

        s->samples_played = ((s->samples_played + s->chunk_size) % s->total_samples);

        return &s->audio;
}

static void audio_cap_testcard_done(void *state)
{
        struct state_audio_capture_testcard *s = (struct state_audio_capture_testcard *) state;

        assert(s->magic == AUDIO_CAPTURE_TESTCARD_MAGIC);

        free(s->audio_samples);
        free(s->audio.data);

        free(s);
}

static const struct audio_capture_info acap_testcard_info = {
        audio_cap_testcard_probe,
        audio_cap_testcard_init,
        audio_cap_testcard_read,
        audio_cap_testcard_done
};

REGISTER_MODULE(testcard, &acap_testcard_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


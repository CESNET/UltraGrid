/**
 * @file   audio/capture/testcard.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET, z. s. p. o.
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

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "audio/wav_reader.h"
#include "compat/usleep.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace std::chrono;
using std::setprecision;
using std::stoi;
using std::string;

#define AUDIO_CAPTURE_TESTCARD_MAGIC 0xf4b3c9c9u

#define DEFAULT_AUDIO_BPS 2
#define DEFAULT_AUDIO_SAMPLE_RATE 48000

#define CHUNKS_PER_SEC 25 // 1 video frame time @25 fps
                   // has to be divisor of AUDIO_SAMLE_RATE

#define DEFAULT_FREQUENCY 1000
#define DEFAULT_VOLUME -18.0
constexpr const char *MOD_NAME = "[Audio testc.] ";

struct state_audio_capture_testcard {
        uint32_t magic = AUDIO_CAPTURE_TESTCARD_MAGIC;

        unsigned long long int chunk_size;
        struct audio_frame audio;

        double audio_remained,
               seconds_tone_played;
        char *audio_samples;

        steady_clock::time_point next_audio_time;

        unsigned int samples_played;

        unsigned int total_samples;

        int crescendo_speed = 1;
};

static void audio_cap_testcard_probe(struct device_info **available_devices, int *count)
{
        *available_devices = static_cast<struct device_info *>(calloc(1, sizeof(struct device_info)));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Testing 1 kHZ signal");
        *count = 1;
}

static void audio_cap_testcard_help(const char *driver_name)
{
        UNUSED(driver_name);
        printf("\ttestcard : Testing sound signal\n");
}

/// @param if crescendo_spd >= 1, amplitude of sound is increasing
static char *get_sine_signal(int sample_rate, int bps, int channels, int frequency, double volume, int crescendo_spd) {
        char *data = (char *) calloc(1, sample_rate * channels * bps);
        double scale = pow(10.0, volume / 20.0) * sqrt(2.0);
        bool dither = sample_rate % frequency != 0 && commandline_params.find("no-dither") == commandline_params.end();

        for (int i = 0; i < (int) sample_rate; i += 1) {
                for (int channel = 0; channel < channels; ++channel) {
                        double sine = sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. );
                        if (crescendo_spd != 0) {
                                double multiplier = ((double) ((i * crescendo_spd) % sample_rate) / sample_rate);
                                sine *= 2 * multiplier; // up to 2x amplitude
                        }
                        int32_t val = std::clamp(sine * INT32_MAX * scale, (double) INT32_MIN, (double) INT32_MAX);
                        change_bps2(data + static_cast<ptrdiff_t>(i) * bps * channels + static_cast<ptrdiff_t>(bps) * channel,
                                        bps, reinterpret_cast<char *>(&val), sizeof val, 1 * sizeof val, dither);
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
                        int64_t val = round(sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. ) * ((1ll << (bps * 8)) / 2 - 1) * scale);
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
                        int64_t val = sin(((double) i / ((double) sample_rate / frequency)) * M_PI * 2. ) * ((1ll << (bps * 8)) / 2 - 1) * scale;
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

static void * audio_cap_testcard_init(const char *cfg)
{
        string wav_file;
        char *item, *save_ptr;

        double volume = DEFAULT_VOLUME;
        int frequency = DEFAULT_FREQUENCY;
        int chunk_size = 0;
        enum {
                SINE,
                EBU,
                WAV,
                SILENCE,
                CRESCENDO,
        } pattern = SINE;

        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available testcard capture:\n");
                audio_cap_testcard_help(NULL);
                printf("\toptions\n\t\ttestcard[:volume=<vol>][:file=<wav>][:frames=<nf>][:frequency=<f>][:ebu|:silence|:crescendo=<spd>]\n");
                printf("\t\t\t<vol> is a volume in dBFS (default %.2f dBFS)\n", DEFAULT_VOLUME);
                printf("\t\t\t<wav> is a wav file to be played\n");
                printf("\t\t\t<nf> sets number of audio frames per packet\n");
                printf("\t\t\t<f> frequency of sinusoide\n");
                return &audio_init_state_ok;
        }

        auto *s = new state_audio_capture_testcard();
        assert(s != nullptr);
        bool failed = false;
        char *tmp = nullptr;

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
                                wav_file = item + strlen("file=");
                                pattern = WAV;
                        } else if(strncasecmp(item, "frames=", strlen("frames=")) == 0) {
                                chunk_size = atoi(item + strlen("frames="));
                        } else if (strncasecmp(item, "frequency=", strlen("frequency=")) == 0) {
                                frequency = atoi(item + strlen("frequency="));
                        } else if(strstr(item, "crescendo") == item) {
                                pattern = CRESCENDO;
                                if (char *val = strchr(item, '=')) {
                                        val += 1;
                                        if (strcmp(val, "help") == 0) {
                                                printf("-s testcard:crescendo=<val>\n\t<val> - crescendo duration multiplier (default 1)\n");
                                                failed = true; break;
                                        }
                                        s->crescendo_speed = stoi(val);
                                }
                        } else if(strcasecmp(item, "ebu") == 0) {
                                pattern = EBU;
                        } else if(strcasecmp(item, "silence") == 0) {
                                pattern = SILENCE;
                        } else {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unknown option: " << item << "\n";
                                failed = true; break;
                        }

                        fmt = NULL;
                }
        } while(false);
        free(tmp);
        if (failed) {
                delete s;
                return nullptr;
        }

        if (pattern != WAV) {
                s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
                s->audio.sample_rate = audio_capture_sample_rate ? audio_capture_sample_rate :
                        DEFAULT_AUDIO_SAMPLE_RATE;
                s->audio.bps = audio_capture_bps ? audio_capture_bps : DEFAULT_AUDIO_BPS;
                s->chunk_size = chunk_size ? chunk_size : s->audio.sample_rate / CHUNKS_PER_SEC;
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

                LOG(LOG_LEVEL_NOTICE) << MODULE_NAME "Generating " << frequency << " Hz (" << setprecision(2) << volume << " RMS dBFS) "
                        << (pattern == SINE ? "sine" : pattern == EBU ? "EBU tone" : "silence") << " "
                        << "(" << audio_desc_from_frame(&s->audio) << ", frames per packet: " << s->chunk_size << ").\n";

                // add padding if s->total_samples % s->chunk_size != 0
                s->audio_samples = (char *) realloc(s->audio_samples, (s->total_samples *
                                        s->audio.ch_count * s->audio.bps) + s->chunk_size - 1);
                memcpy(s->audio_samples + s->total_samples * s->audio.bps * s->audio.ch_count,
                                s->audio_samples, s->chunk_size - 1);
        } else {
                FILE *wav = fopen(wav_file.c_str(), "r");
                if(!wav) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to open WAV file: " << wav_file << ".\n";
                        delete s;
                        return NULL;
                }
                struct wav_metadata metadata;
                int ret = read_wav_header(wav, &metadata);
                if(ret != WAV_HDR_PARSE_OK) {
                        log_msg(LOG_LEVEL_ERROR, "%s\n", get_wav_error(ret));
                        fclose(wav);
                        delete s;
                        return NULL;
                }
                s->audio.bps = audio_capture_bps ? audio_capture_bps : metadata.bits_per_sample / 8;
                s->audio.ch_count = metadata.ch_count;
                s->audio.sample_rate = metadata.sample_rate;
                s->chunk_size = chunk_size ? chunk_size : s->audio.sample_rate / CHUNKS_PER_SEC;

                s->total_samples = metadata.data_size  * 8ULL /  metadata.ch_count / metadata.bits_per_sample;
                LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << s->total_samples << " samples read from file " << wav_file << "\n";

                const int headroom = (s->chunk_size - 1) * s->audio.ch_count * s->audio.bps;
                const int samples_data_size = s->total_samples * s->audio.ch_count * s->audio.bps;

                const int audio_samples_size = samples_data_size + headroom;
                s->audio_samples = (char *) calloc(1, audio_samples_size);

                char *read_to;
                char *tmp = NULL;
                if ((unsigned) s->audio.bps == metadata.bits_per_sample / 8) {
                    read_to = s->audio_samples;
                } else {
                    tmp = (char *) calloc(1, metadata.data_size);
                    read_to = tmp;
                }
                unsigned int samples = wav_read(read_to, s->total_samples, wav, &metadata);
                int bytes = samples * (metadata.bits_per_sample / 8) * metadata.ch_count;
                if (samples != s->total_samples) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Warning: premature end of WAV file (" << bytes << " read, " << metadata.data_size << " expected)!\n";
                        s->total_samples = samples;
                }
                fclose(wav);

                if ((unsigned) s->audio.bps != metadata.bits_per_sample / 8){
                    change_bps(s->audio_samples, s->audio.bps, tmp, metadata.bits_per_sample / 8, bytes);
                    free(tmp);
                }


                memcpy(s->audio_samples + samples_data_size, s->audio_samples, headroom);
        }

        s->audio.data_len = s->chunk_size * s->audio.bps * s->audio.ch_count;
        s->audio.max_size = s->audio.data_len;
        s->audio.data = (char *) calloc(1, s->audio.max_size);

        s->samples_played = 0;

        s->next_audio_time = steady_clock::now();

        return s;
}

static struct audio_frame *audio_cap_testcard_read(void *state)
{
        struct state_audio_capture_testcard *s;
        s = (struct state_audio_capture_testcard *) state;
        steady_clock::time_point curr_time = steady_clock::now();

        if(s->next_audio_time > curr_time) {
                usleep(duration_cast<microseconds>(s->next_audio_time - curr_time).count());
        } else {
                // we missed more than 2 "frame times", in that case, just drop the packages
                if (duration_cast<microseconds>(curr_time - s->next_audio_time) > microseconds(2 * (1000 * 1000 * s->chunk_size / s->audio.sample_rate))) {
                        s->next_audio_time = curr_time;
                        fprintf(stderr, MODULE_NAME "Warning: skipping some samples (late grab call).\n");
                }
        }

        s->next_audio_time += microseconds(1000 * 1000 * s->chunk_size / s->audio.sample_rate);

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

        delete s;
}

static const struct audio_capture_info acap_testcard_info = {
        audio_cap_testcard_probe,
        audio_cap_testcard_help,
        audio_cap_testcard_init,
        audio_cap_testcard_read,
        audio_cap_testcard_done
};

REGISTER_MODULE(testcard, &acap_testcard_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


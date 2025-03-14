/*
 * FILE:    audio/playback/portaudio.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Martin Pulec     <martin.pulec@cesnet.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <portaudio.h> /* from PortAudio */

#include "audio/audio_playback.h"
#include "audio/portaudio_common.h"
#include "audio/types.h"
#include "compat/misc.h" // strdupa
#include "debug.h"
#include "lib_common.h"
#include "utils/audio_buffer.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "tv.h"

#define MOD_NAME "[Portaudio playback] "
#define BUFFER_LEN_SEC 1

#define NO_DATA_STOP_NSEC (2 * NS_IN_SEC)

struct state_portaudio_playback {
        struct audio_desc desc;
        int samples;
        int device;
        PaStream *stream;
        int max_output_channels;

        struct audio_buffer *data;

        time_ns_t last_audio_read;
        bool quiet;
};

/*
 * For Portaudio threads-related issues see
 * http://www.portaudio.com/trac/wiki/tips/Threading
 */

/* prototyping */
static void      portaudio_close(PaStream *stream);  /* closes and frees all audio resources ( according to valgrind this is not true..  ) */
static int callback( const void *inputBuffer, void *outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData );
static void     cleanup(struct state_portaudio_playback * s);
static bool audio_play_portaudio_reconfigure(void *state, struct audio_desc);

 /*
  * Shared functions
  */
static bool portaudio_start_stream(PaStream *stream)
{
        PaError error = Pa_StartStream(stream);
        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error starting stream:%s\n", Pa_GetErrorText(error));
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                return false;
        }

        return true;
}

static void audio_play_portaudio_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_portaudio_probe(available_devices, count, PORTAUDIO_OUT);
}

static void portaudio_close(PaStream * stream) // closes and frees all audio resources
{
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
}

static _Bool parse_fmt(const char *cfg, int *input_device_idx, const char **device_name) {
        if (isdigit(cfg[0])) {
                *input_device_idx = atoi(cfg);
                cfg = strchr(cfg, ':') ? strchr(cfg, ':') + 1 : cfg + strlen(cfg);
        }
        char *ccfg = strdupa(cfg);
        char *item = NULL;
        char *saveptr = NULL;
        while ((item = strtok_r(ccfg, ":", &saveptr)) != NULL) {
                if (strstr(item, "device=") == item) {
                        const char *dev = strchr(item, '=') + 1;
                        if (isdigit(dev[0])) {
                                *input_device_idx = atoi(dev);
                        } else { // pointer to *input* cfg
                                *device_name = strchr(strstr(cfg, "device="), '=') + 1;
                        }
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s!\n", item);
                        return 0;
                }
                ccfg = NULL;
        }
        return 1;
}

static void audio_play_portaudio_help(void) {
        printf("PortAudio playback usage:\n");
        color_printf("\t" TBOLD(TRED("-r portaudio") "[:<index>]") "\n");
        printf("or\n");
        color_printf("\t" TBOLD(TRED("-r portaudio") "[:device=<dev>]") "\n\n");
        printf("options:\n");
        color_printf("\t" TBOLD("<dev>") "\tdevice name (or a part of it); device index is also accepted here\n");
        printf("\nAvailable PortAudio playback devices:\n");
        portaudio_print_help(PORTAUDIO_OUT);
}

static void * audio_play_portaudio_init(const char *cfg)
{	
        int output_device_idx = -1;
        const char *output_device_name = NULL;

        portaudio_print_version();
        
        if (strcmp(cfg, "help") == 0) {
                audio_play_portaudio_help();
                return INIT_NOERR;
        }
        if (!parse_fmt(cfg, &output_device_idx, &output_device_name)) {
                return NULL;
        }

        PaError error = Pa_Initialize();
        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error initializing portaudio\n");
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                return NULL;
        }

        if (output_device_name != NULL) {
                output_device_idx = portaudio_select_device_by_name(output_device_name);
        }

        struct state_portaudio_playback *s = calloc(1, sizeof *s);
        s->device = output_device_idx;
        s->data = NULL;
        const PaDeviceInfo *device_info = NULL;
        if (output_device_idx >= 0) {
                device_info = Pa_GetDeviceInfo(output_device_idx);
        } else if (output_device_idx == -1) {
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice());
        }
        if (device_info == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Couldn't obtain requested portaudio index %d.\n"
                                MOD_NAME "Follows list of available Portaudio devices.\n", output_device_idx);
                audio_play_portaudio_help();
                Pa_Terminate();
                free(s);
                return NULL;
        }
	s->max_output_channels = device_info->maxOutputChannels;
        Pa_Terminate();

        s->quiet = true;
        
        if (!audio_play_portaudio_reconfigure(s, (struct audio_desc){2, 48000, 2, AC_PCM})) {
                Pa_Terminate();
                free(s);
                return NULL;
        }

	return s;
}

static void audio_play_portaudio_done(void *state)
{
        struct state_portaudio_playback *s = state;
        cleanup(s);
        free(s);
}

static void cleanup(struct state_portaudio_playback * s)
{
        portaudio_close(s->stream);
        s->stream = NULL;
        audio_buffer_destroy(s->data);
        s->data = NULL;
	Pa_Terminate();
}

/**
 * Tries to find format compatible with Portaudio
 *
 * @param      device_idx  index of Portaudio device to be evaluated (-1 for default)
 * @param[in]  ch_count    number of channels to be used
 * @param[in]  sample_rate requested sample rate
 * @param[out] sample_rate sample rate supported by state_portaudio_playback::device
 * @param[in]  bps         requested bytes per sample
 * @param[out] bps         bps supported by state_portaudio_playback::device
 *
 * @returns true if eligible format was found, false otherwise
 */
static bool get_supported_format(int device_idx, int ch_count, int *sample_rate, int *bps) {
        PaSampleFormat sample_formats[] = { paInt8, paInt16, paInt24, paInt32 };
        PaStreamParameters outputParameters = { 0 };
        outputParameters.device = device_idx >= 0 ? device_idx : Pa_GetDefaultOutputDevice();
        outputParameters.channelCount = ch_count;
        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(outputParameters.device);
        outputParameters.suggestedLatency = device_info->defaultLowInputLatency;

        assert(*bps >= 1 && *bps <= 4);
        // @todo sort to select the best rate if not exactly the requested, not
        // the first usable (which will be now the defaultSampleRate)
        int sample_rates[] = { *sample_rate, device_info->defaultSampleRate,
                48000, 44100, 8000, 16000, 32000, 96000, 24000 };
        for (int i = 0; i < (int)(sizeof sample_rates / sizeof sample_rates[0]); ++i) {
                int j = *bps - 1;
                while (true) {
                        outputParameters.sampleFormat = sample_formats[j];
                        int err = Pa_IsFormatSupported(NULL, &outputParameters, sample_rates[i]);
                        if (err == paFormatIsSupported) {
                                *sample_rate = sample_rates[i];
                                *bps = j + 1;
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using %d channel%s, sample rate: %d, bps: %d\n", ch_count, ch_count > 1 ? "s" : "", *sample_rate, *bps * 8);
                                return true;
                        }
                        if (j >= *bps - 1 && j < 3) { // try better sample formats
                                j++;
                                continue;
                        }
                        if (j == 3) { // start trying worse formats
                                j = *bps - 2;
                        } else {
                                j--;
                        }
                        if (j == -1) { // we tested all sample rates
                                break;
                        }
                }
        }

        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to find eligible format!\n");
        return false;
}

static bool audio_play_portaudio_ctl(void *state, int request, void *data, size_t *len)
{
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                if (*len >= sizeof(struct audio_desc)) {
                        struct state_portaudio_playback * s =
                                (struct state_portaudio_playback *) state;
                        struct audio_desc desc;
                        memcpy(&desc, data, sizeof desc);
                        desc.ch_count = MIN(desc.ch_count, s->max_output_channels);
                        desc.codec = AC_PCM;
                        if (!get_supported_format(((struct state_portaudio_playback *) state)->device, desc.ch_count, &desc.sample_rate, &desc.bps)) {
                                return false;
                        }
                        memcpy(data, &desc, sizeof desc);
                        *len = sizeof desc;
                        return true;
                } else {
                        return false;
                }
        default:
                return false;
        }
}

static bool audio_play_portaudio_reconfigure(void *state, struct audio_desc desc)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
        PaError error;
	PaStreamParameters outputParameters;
        
        if(s->stream != NULL) {
                cleanup(s);
        }

        int audio_buf_len_ms = 50;
        if (get_commandline_param("low-latency-audio")) {
                audio_buf_len_ms = 5;
        }
        if (get_commandline_param("audio-buffer-len")) {
                audio_buf_len_ms = atoi(get_commandline_param("audio-buffer-len"));
        }
        s->data = audio_buffer_init(desc.sample_rate, desc.bps, desc.ch_count, audio_buf_len_ms);
        s->desc = desc;

        log_msg(LOG_LEVEL_INFO, MOD_NAME "(Re)initializing portaudio playback.\n");

        error = Pa_Initialize();
        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error initializing portaudio\n");
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                return false;
        }

        log_msg(LOG_LEVEL_INFO, "Using PortAudio version: %s\n", Pa_GetVersionText());

        // default device
        if (s->device == -1) {
                log_msg(LOG_LEVEL_NOTICE, "Using default output audio device: %s\n",
                                portaudio_get_device_name(Pa_GetDefaultOutputDevice()));
                outputParameters.device = Pa_GetDefaultOutputDevice();
        } else {
                assert(s->device >= 0);
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using output audio device: %s\n",
                                portaudio_get_device_name(s->device));
                outputParameters.device = s->device;
        }

        if(desc.ch_count <= s->max_output_channels)
                outputParameters.channelCount = desc.ch_count; // output channels
        else
                outputParameters.channelCount = s->max_output_channels; // output channels
        assert(desc.bps <= 4 && desc.bps != 0);
        switch(desc.bps) {
                case 1:
                        outputParameters.sampleFormat = paInt8;
                        break;
                case 2:
                        outputParameters.sampleFormat = paInt16;
                        break;
                case 3:
                        outputParameters.sampleFormat = paInt24;
                        break;
                case 4:
                        outputParameters.sampleFormat = paInt32;
                        break;
        }
                        
        outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = NULL;

        error = Pa_OpenStream( &s->stream, NULL, &outputParameters, desc.sample_rate, paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
                        paNoFlag,
                        callback,
                        s
                        );

        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, "[Portaudio] Unable to open stream: %s\n",
                                Pa_GetErrorText(error));
                return false;
        }

        if (!portaudio_start_stream(s->stream)) {
                return false;
        }
        
        return true;
}                        

/* This routine will be called by the PortAudio engine when audio is needed.
   It may called at interrupt level on some machines so don't do anything
   that could mess up the system like calling malloc() or free().
   */
static int callback( const void *inputBuffer, void *outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData )
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) userData;
        UNUSED(inputBuffer);
        UNUSED(timeInfo);
        UNUSED(statusFlags);

        ssize_t req_bytes = framesPerBuffer * s->desc.ch_count * s->desc.bps;
        ssize_t bytes_read = audio_buffer_read(s->data, (char *) outputBuffer, req_bytes);

        if (bytes_read < req_bytes) {
                if (!s->quiet)
                        log_msg(LOG_LEVEL_INFO, "[Portaudio] Buffer underflow.\n");
                memset((int8_t *) outputBuffer + bytes_read, 0, req_bytes - bytes_read);
                if (!s->quiet && get_time_in_ns() - s->last_audio_read > NO_DATA_STOP_NSEC) {
                        log_msg(LOG_LEVEL_WARNING, "[Portaudio] No data for %lld seconds!\n", NO_DATA_STOP_NSEC / NS_IN_SEC);
                        s->quiet = true;
                }
        } else {
                if (s->quiet) {
                        log_msg(LOG_LEVEL_NOTICE, "[Portaudio] Starting again.\n");
                }
                s->quiet = false;
                s->last_audio_read = get_time_in_ns();
        }

        return paContinue;
}

static void audio_play_portaudio_put_frame(void *state, const struct audio_frame *buffer)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;

        const int samples_count = buffer->data_len / (buffer->bps * buffer->ch_count);

        /* if we got more channel we can play - skip the additional channels */
        if(s->desc.ch_count > s->max_output_channels) {
                int i;
                for (i = 0; i < samples_count; ++i) {
                        int j;
                        for(j = 0; j < s->max_output_channels; ++j)
                                memcpy(buffer->data + s->desc.bps * ( i * s->max_output_channels + j),
                                        buffer->data + s->desc.bps * ( i * buffer->ch_count + j),
                                        buffer->bps);
                }
        }

        int out_channels = s->desc.ch_count;
        if (out_channels > s->max_output_channels) {
                out_channels = s->max_output_channels;
        }
        
        audio_buffer_write(s->data, buffer->data, samples_count * buffer->bps * out_channels);
}

static const struct audio_playback_info aplay_portaudio_info = {
        audio_play_portaudio_probe,
        audio_play_portaudio_init,
        audio_play_portaudio_put_frame,
        audio_play_portaudio_ctl,
        audio_play_portaudio_reconfigure,
        audio_play_portaudio_done
};

REGISTER_MODULE(portaudio, &aplay_portaudio_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


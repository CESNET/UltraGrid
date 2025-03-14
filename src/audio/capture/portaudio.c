/*
 * FILE:    audio/capture/portaudio.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          Martin Pulec     <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2023 CESNET, z. s. p. o.
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
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <portaudio.h> /* from PortAudio API */

#include "audio/audio_capture.h"
#include "audio/portaudio_common.h"
#include "audio/types.h"
#include "compat/misc.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/ring_buffer.h"

/* default variables for sender */
#define BUF_MS 100

#define MOD_NAME "[Portaudio capture] "

struct state_portaudio_capture {
        struct audio_frame frame;
        PaStream *stream;

        pthread_mutex_t lock;
        pthread_cond_t cv;
        struct ring_buffer *buffer;
};

/*
 * For Portaudio threads-related issues see
 * http://www.portaudio.com/trac/wiki/tips/Threading
 */
static int         portaudio_start_stream(PaStream *stream);
static void        portaudio_close(PaStream *stream); // closes and frees all audio resources ( according to valgrind this is not true..  )
static int         callback( const void *inputBuffer, void *outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData );

static int portaudio_start_stream(PaStream *stream)
{
        PaError error = Pa_StartStream(stream);
        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error starting stream:%s\n", Pa_GetErrorText(error));
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return -1;
	}

	return 0;
}

static void audio_cap_portaudio_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_portaudio_probe(available_devices, count, PORTAUDIO_IN);
}

static void audio_cap_portaudio_help()
{
        portaudio_print_help(PORTAUDIO_IN);
}

static void portaudio_close(PaStream * stream)	// closes and frees all audio resources
{
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
	Pa_Terminate();
}

static _Bool parse_fmt(const char *cfg, PaTime *latency, int *input_device_idx, const char **device_name) {
        if (isdigit(cfg[0])) {
                *input_device_idx = atoi(cfg);
                cfg = strchr(cfg, ':') ? strchr(cfg, ':') + 1 : cfg + strlen(cfg);
        }
        char *ccfg = strdupa(cfg);
        char *item = NULL;
        char *saveptr = NULL;
        while ((item = strtok_r(ccfg, ":", &saveptr)) != NULL) {
                if (strstr(item, "latency=") == item) {
                        *latency = atof(strchr(item, '=') + 1);
                } else if (strstr(item, "device=") == item) {
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

static void usage() {
        printf("PortAudio capture usage:\n");
        color_printf("\t" TBOLD(TRED("-s portaudio") "[:<index>[:latency=<l>]]") "\n");
        printf("or\n");
        color_printf("\t" TBOLD(TRED("-s portaudio") "[:device=<dev>][:latency=<l>]") "\n\n");
        printf("options:\n");
        color_printf("\t" TBOLD(" <l> ") "\tsuggested latency in sec (experimental, use in case of problems)\n");
        color_printf("\t" TBOLD("<dev>") "\tdevice name (or a part of it); device index is also accepted here\n");
        printf("\nAvailable PortAudio capture devices:\n");

        audio_cap_portaudio_help();
}

static void * audio_cap_portaudio_init(struct module *parent, const char *cfg)
{
        UNUSED(parent);
        portaudio_print_version();

        if (strcmp(cfg, "help") == 0) {
                usage();
                return INIT_NOERR;
        }

        int input_device_idx = -1;
        const char *input_device_name = NULL;
	PaError error;
        const	PaDeviceInfo *device_info = NULL;
        PaTime latency = -1.0;

        if (!parse_fmt(cfg, &latency, &input_device_idx, &input_device_name)) {
                return NULL;
        }
        
        struct state_portaudio_capture *s = calloc(1, sizeof *s);

        log_msg(LOG_LEVEL_INFO, "Initializing portaudio capture.\n");

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);
	
	error = Pa_Initialize();
        if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error initializing portaudio\n");
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                free(s);
		return NULL;
	}

	log_msg(LOG_LEVEL_INFO, "Using PortAudio version: %s\n", Pa_GetVersionText());

	PaStreamParameters inputParameters;

	// default device
        if (input_device_name != NULL) {
                input_device_idx = portaudio_select_device_by_name(input_device_name);
        }
        if (input_device_idx == -1) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using default input audio device: %s\n",
                                portaudio_get_device_name(Pa_GetDefaultInputDevice()));
		inputParameters.device = Pa_GetDefaultInputDevice();
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
        } else if (input_device_idx >= 0) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using input audio device: %s\n",
                                portaudio_get_device_name(input_device_idx));
		inputParameters.device = input_device_idx;
                device_info = Pa_GetDeviceInfo(input_device_idx);
	}

        if(device_info == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Couldn't obtain requested portaudio device index %d.\n"
                               MOD_NAME "Follows list of available Portaudio devices.\n", input_device_idx);
                audio_cap_portaudio_help();
                free(s);
                Pa_Terminate();
                return NULL;
        }

        inputParameters.channelCount = s->frame.ch_count = audio_capture_channels > 0 ? (int) audio_capture_channels : MIN(device_info->maxInputChannels, DEFAULT_AUDIO_CAPTURE_CHANNELS);
        if (s->frame.ch_count > device_info->maxInputChannels) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested %d input channels, device offers only %d.\n",
                                s->frame.ch_count,
                                device_info->maxInputChannels);
                free(s);
		return NULL;
        }
        if (audio_capture_bps == 4) {
                inputParameters.sampleFormat = paInt32;
                s->frame.bps = 4;
        } else {
                if (audio_capture_bps != 0 && audio_capture_bps != 2) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Ignoring unsupported Bps %d!\n",
                                        audio_capture_bps);
                }
                inputParameters.sampleFormat = paInt16;
                s->frame.bps = 2;
        }
        inputParameters.suggestedLatency = latency == -1.0 ? Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency : latency;
        inputParameters.hostApiSpecificStreamInfo = NULL;

        s->frame.sample_rate = audio_capture_sample_rate;
        if (s->frame.sample_rate == 0) {
                s->frame.sample_rate = (int) device_info->defaultSampleRate;
                MSG(NOTICE, "Setting device default sample rate %d Hz\n",
                    s->frame.sample_rate);
        }

        error = Pa_OpenStream( &s->stream, &inputParameters, NULL, s->frame.sample_rate,
                        paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
                        paNoFlag,
                        callback,	// callback function; NULL, because we use blocking functions
                        s	// user data - none, because we use blocking functions
                        );
	if (error != paNoError) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error opening audio stream\n");
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return NULL;
	}

        s->frame.max_size = (s->frame.bps * s->frame.ch_count) * s->frame.sample_rate / 1000 * BUF_MS;
        
        s->frame.data = (char*)malloc(s->frame.max_size);

        s->buffer = ring_buffer_init(s->frame.max_size);

        memset(s->frame.data, 0, s->frame.max_size);

        portaudio_start_stream(s->stream);

	return s;
}

static int callback( const void *inputBuffer, void *outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData)
{
        struct state_portaudio_capture * s = 
                (struct state_portaudio_capture *) userData;
        UNUSED(outputBuffer);
        UNUSED(timeInfo);
        UNUSED(statusFlags);

        pthread_mutex_lock(&s->lock);
        ring_buffer_write(s->buffer, inputBuffer, framesPerBuffer * s->frame.ch_count *
                        s->frame.bps);
        pthread_cond_signal(&s->cv);
        pthread_mutex_unlock(&s->lock);

        return paContinue;

}

// read from input device
static struct audio_frame * audio_cap_portaudio_read(void *state)
{
        struct state_portaudio_capture *s = 
                        (struct state_portaudio_capture *) state;

        int ret = 0; 

        pthread_mutex_lock(&s->lock);
        while((ret = ring_buffer_read(s->buffer, s->frame.data, s->frame.max_size)) == 0) {
                pthread_cond_wait(&s->cv, &s->lock);
        }
        pthread_mutex_unlock(&s->lock);

        s->frame.data_len = ret;

	return &s->frame;
}

static void audio_cap_portaudio_done(void *state)
{
        struct state_portaudio_capture *s = (struct state_portaudio_capture *) state;
        portaudio_close(s->stream);
        free(s->frame.data);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        ring_buffer_destroy(s->buffer);
        free(s);
}

static const struct audio_capture_info acap_portaudio_info = {
        audio_cap_portaudio_probe,
        audio_cap_portaudio_init,
        audio_cap_portaudio_read,
        audio_cap_portaudio_done
};

REGISTER_MODULE(portaudio, &acap_portaudio_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


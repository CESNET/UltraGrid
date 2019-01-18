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
 * Copyright (c) 2005-2015 CESNET, z. s. p. o.
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

#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/portaudio_common.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/ring_buffer.h"
#include "utils/color_out.h"

/* default variables for sender */
#define BUF_MS 100

#define MODULE_NAME "[Portaudio capture] "

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
static void audio_cap_portaudio_help(const char *driver_name);

static int portaudio_start_stream(PaStream *stream)
{
	PaError error;

	error = Pa_StartStream(stream);
	if(error != paNoError)
	{
		printf("Error starting stream:%s\n", Pa_GetErrorText(error));
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return -1;
	}

	return 0;
}

static void audio_cap_portaudio_probe(struct device_info **available_devices, int *count)
{
        *available_devices = malloc(sizeof(struct device_info));
        strcpy((*available_devices)[0].id, "portaudio");
        strcpy((*available_devices)[0].name, "Portaudio audio input");
        *count = 1;
}

static void audio_cap_portaudio_help(const char *driver_name)
{
        UNUSED(driver_name);
        portaudio_print_available_devices(PORTAUDIO_IN);
}

static void portaudio_close(PaStream * stream)	// closes and frees all audio resources
{
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
	Pa_Terminate();
}

/*
 * capture funcitons
 */
static void * audio_cap_portaudio_init(const char *cfg)
{
        portaudio_print_version();

        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Portaudio options:\n");
                color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-s poraudio");
                color_out(COLOR_OUT_BOLD, "[:<index>[:latency=<l>]]\n\n");
                color_out(COLOR_OUT_BOLD, "\t<l>");
                printf("\tsuggested latency in sec (experimental, use in case of problems)\n");
                printf("\nAvailable PortAudio capture devices:\n");

                audio_cap_portaudio_help(NULL);
                return &audio_init_state_ok;
        }

        struct state_portaudio_capture *s;
        int input_device;
	PaError error;
        const	PaDeviceInfo *device_info = NULL;
        PaTime latency = -1.0;
        
        s = (struct state_portaudio_capture *) malloc(sizeof(struct state_portaudio_capture));
	/* 
	 * so far we only work with portaudio
	 * might get more complicated later..(jack?)
	 */
        if (cfg) {
                input_device = atoi(cfg);
                if (strchr(cfg, ':')) {
                        const char *option = strchr(cfg, ':') + 1;
                        if (strncmp(option, "latency=", strlen("latency=")) == 0) {
                                latency = atof(option + strlen("latency="));
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MODULE_NAME "Unknown option: %s!\n", option);
                        }
                }
        } else {
                input_device = -1;
        }

	printf("Initializing portaudio capture.\n");

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);
	
	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                free(s);
		return NULL;
	}

	printf("Using PortAudio version: %s\n", Pa_GetVersionText());

	PaStreamParameters inputParameters;

	// default device
	if(input_device == -1)
	{
		printf("Using default input audio device");
		portaudio_print_device_info(Pa_GetDefaultInputDevice());
		printf("\n");
		inputParameters.device = Pa_GetDefaultInputDevice();
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
	} else if(input_device >= 0) {
		printf("Using input audio device:");
		portaudio_print_device_info(input_device);
		printf("\n");
		inputParameters.device = input_device;
                device_info = Pa_GetDeviceInfo(input_device);
	}

        if(device_info == NULL) {
                fprintf(stderr, MODULE_NAME "Couldn't obtain requested portaudio device.\n"
                               MODULE_NAME "Follows list of available Portaudio devices.\n");
                audio_cap_portaudio_help(NULL);
                free(s);
                Pa_Terminate();
                return NULL;
        }

        if((int) audio_capture_channels <= device_info->maxInputChannels) {
                inputParameters.channelCount = audio_capture_channels;
        } else {
                fprintf(stderr, MODULE_NAME "Requested %d input channels, device offers only %d.\n",
                                audio_capture_channels,
                                device_info->maxInputChannels);
                free(s);
		return NULL;
        }
        if (audio_capture_bps == 4) {
                inputParameters.sampleFormat = paInt32;
                s->frame.bps = 4;
        } else {
                if (audio_capture_bps != 0 && audio_capture_bps != 2) {
                        log_msg(LOG_LEVEL_WARNING, "[Portaudio] Ignoring unsupported Bps %d!\n",
                                        audio_capture_bps);
                }
                inputParameters.sampleFormat = paInt16;
                s->frame.bps = 2;
        }
        inputParameters.suggestedLatency = latency == -1.0 ? Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency : latency;
        inputParameters.hostApiSpecificStreamInfo = NULL;

        s->frame.sample_rate = audio_capture_sample_rate ? audio_capture_sample_rate :
                48000;

        error = Pa_OpenStream( &s->stream, &inputParameters, NULL, s->frame.sample_rate,
                        paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
                        paNoFlag,
                        callback,	// callback function; NULL, because we use blocking functions
                        s	// user data - none, because we use blocking functions
                        );
	if(error != paNoError)
	{
		printf("Error opening audio stream\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return NULL;
	}

        s->frame.ch_count = inputParameters.channelCount;
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
        audio_cap_portaudio_help,
        audio_cap_portaudio_init,
        audio_cap_portaudio_read,
        audio_cap_portaudio_done
};

REGISTER_MODULE(portaudio, &acap_portaudio_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


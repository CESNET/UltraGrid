/*
 * FILE:    audio/playback/portaudio.c
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

#include <portaudio.h> /* from PortAudio */

#include "audio/audio.h"
#include "audio/playback/portaudio.h"
#include "debug.h"
#include "utils/ring_buffer.h"

#define MODULE_NAME "[Portaudio playback] "
#define BUFFER_LEN_SEC 2

struct state_portaudio_playback {
        audio_frame frame;
        int samples;
        int device;
        PaStream *stream;
        int max_output_channels;

        struct ring_buffer *data;
        char *tmp_buffer;
};

enum audio_device_kind {
        AUDIO_IN,
        AUDIO_OUT
};

/*
 * For Portaudio threads-related issues see
 * http://www.portaudio.com/trac/wiki/tips/Threading
 */

/* prototyping */
static void      print_device_info(PaDeviceIndex device);
static int       portaudio_start_stream(PaStream *stream);
static void      portaudio_close(PaStream *stream);  /* closes and frees all audio resources ( according to valgrind this is not true..  ) */
static void      portaudio_print_available_devices(enum audio_device_kind);
static int callback( const void *inputBuffer, void *outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData );
static void     cleanup(struct state_portaudio_playback * s);

 /*
  * Shared functions
  */
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


static void print_device_info(PaDeviceIndex device)
{
	if( (device < 0) || (device >= Pa_GetDeviceCount()) )
	{
		printf("Requested info on non-existing device");
		return;
	}
	
	const	PaDeviceInfo *device_info = Pa_GetDeviceInfo(device);
	printf(" %s (output channels: %d; input channels: %d)", device_info->name, device_info->maxOutputChannels, device_info->maxInputChannels);
}

void portaudio_playback_help(const char *driver_name)
{
        UNUSED(driver_name);
        portaudio_print_available_devices(AUDIO_OUT);
}

static void portaudio_print_available_devices(enum audio_device_kind kind)
{
	int numDevices;
        int i;

	PaError error;
	
	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return;
	}

	numDevices = Pa_GetDeviceCount();
	if( numDevices < 0)
	{
		printf("Error getting portaudio devices number\n");
		return;
	}
	if( numDevices == 0)
	{
		printf("There are NO available audio devices!\n");
		return;
	}
        
        printf("\tportaudio : use default Portaudio device (marked with star)\n");
        
	for(i = 0; i < numDevices; i++)
	{
		if((i == Pa_GetDefaultInputDevice() && kind == AUDIO_IN) ||
                                (i == Pa_GetDefaultOutputDevice() && kind == AUDIO_OUT))
			printf("(*) ");
			
		printf("\tportaudio:%d :", i);
		print_device_info(i);
		printf("\n");
	}

	return;
}

static void portaudio_close(PaStream * stream) // closes and frees all audio resources
{
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
	Pa_Terminate();
}

/*
 * Playback functions 
 */
void * portaudio_playback_init(char *cfg)
{	
        struct state_portaudio_playback *s;
        int output_device;
        
        if(cfg) {
                if(strcmp(cfg, "help") == 0) {
                        printf("Available PortAudio playback devices:\n");
                        portaudio_playback_help(NULL);
                        return NULL;
                } else {
                        output_device = atoi(cfg);
                }
        } else {
                output_device = -1;
        }
        Pa_Initialize();
        
        s = calloc(1, sizeof(struct state_portaudio_playback));
        assert(output_device >= -1);
        s->device = output_device;
        s->data = NULL;
        s->tmp_buffer = NULL;
        const	PaDeviceInfo *device_info;
        if(output_device >= 0) {
                device_info = Pa_GetDeviceInfo(output_device);
        } else {
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice());
        }
        if(device_info == NULL) {
                fprintf(stderr, MODULE_NAME "Couldn't obtain requested portaudio device.\n"
                                MODULE_NAME "Follows list of available Portaudio devices.\n");
                portaudio_playback_help(NULL);
                free(s);
                Pa_Terminate();
                return NULL;
        }
	s->max_output_channels = device_info->maxOutputChannels;
        
        portaudio_reconfigure(s, 16, 2, 48000);
         
	return s;
}

void portaudio_close_playback(void *state)
{
        cleanup(state);
        free(state);
}

static void cleanup(struct state_portaudio_playback * s)
{
        free(s->frame.data);
        portaudio_close(s->stream);

        ring_buffer_destroy(s->data);
        free(s->tmp_buffer);
}

int portaudio_reconfigure(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
        PaError error;
	PaStreamParameters outputParameters;
        
        if(s->stream != NULL) {
                cleanup(s);
        }

        int size = BUFFER_LEN_SEC * channels * (quant_samples/8) *
                        sample_rate;
        s->data = ring_buffer_init(size);
        s->tmp_buffer = malloc(size);
        
        s->frame.bps = quant_samples / 8;
        s->frame.ch_count = channels;
        s->frame.sample_rate = sample_rate;
        
        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate; // can hold up to 1 sec
        
        s->frame.data = (char*)malloc(s->frame.max_size);
        assert(s->frame.data != NULL);
        
	printf("(Re)initializing portaudio playback.\n");

	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return FALSE;
	}

	printf("Using PortAudio version: %s\n", Pa_GetVersionText());

	// default device
	if(s->device == -1)
	{
		printf("\nUsing default output audio device:");
		fflush(stdout);
		print_device_info(Pa_GetDefaultOutputDevice());
		printf("\n");
		outputParameters.device = Pa_GetDefaultOutputDevice();
	}
	else if(s->device >= 0)
	{
		printf("\nUsing output audio device:");
		print_device_info(s->device);
		printf("\n");
		outputParameters.device = s->device;
	}
		
                
        if(channels <= s->max_output_channels)
                outputParameters.channelCount = channels; // output channels
        else
                outputParameters.channelCount = s->max_output_channels; // output channels
        assert(quant_samples % 8 == 0 && quant_samples <= 32 && quant_samples != 0);
        switch(quant_samples) {
                case 8:
                        outputParameters.sampleFormat = paInt8;
                        break;
                case 16:
                        outputParameters.sampleFormat = paInt16;
                        break;
                case 24:
                        outputParameters.sampleFormat = paInt24;
                        break;
                case 32:
                        outputParameters.sampleFormat = paInt32;
                        break;
        }
                        
        outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = NULL;

        error = Pa_OpenStream( &s->stream, NULL, &outputParameters, sample_rate, paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
                        paNoFlag,
                        callback,
                        s
                        );
        portaudio_start_stream(s->stream);
        
	if(error != paNoError)
	{
		printf("Error opening audio stream\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
		return FALSE;
	}

        return TRUE;
}                        

// get first empty frame, bud don't consider it as being used
struct audio_frame* portaudio_get_frame(void *state)
{
	return &((struct state_portaudio_playback *) state)->frame;
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

        ring_buffer_read(s->data, outputBuffer, framesPerBuffer * s->frame.ch_count *
                        s->frame.bps);

        return paContinue;
}

void portaudio_put_frame(void *state, struct audio_frame *buffer)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
                
        const int samples_count = buffer->data_len / (buffer->bps * buffer->ch_count);

        /* if we got more channel we can play - skip the additional channels */
        if(s->frame.ch_count > s->max_output_channels) {
                int i;
                for (i = 0; i < samples_count; ++i) {
                        int j;
                        for(j = 0; j < s->max_output_channels; ++j)
                                memcpy(buffer->data + s->frame.bps * ( i * s->max_output_channels + j),
                                        buffer->data + s->frame.bps * ( i * buffer->ch_count + j),
                                        buffer->bps);
                }
        }

        int out_channels = s->frame.ch_count;
        if (out_channels > s->max_output_channels) {
                out_channels = s->max_output_channels;
        }
        
        ring_buffer_write(s->data, buffer->data, samples_count * buffer->bps * out_channels);
}


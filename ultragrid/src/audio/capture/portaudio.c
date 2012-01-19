/*
 * FILE:    audio/capture/portaudio.c
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

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "portaudio/include/portaudio.h" /* from PortAudio API */

#include "audio/audio.h"
#include "audio/capture/portaudio.h"
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "utils/fs_lock.h"


/* default variables for sender */
#define BPS 2 /* paInt16 */
#define SAMPLE_RATE 48000
#define SAMPLES_PER_FRAME 2048
#define CHANNELS 2
#define SECONDS 5

struct state_portaudio_capture {
        struct audio_frame frame;
        PaStream *stream;
        struct fs_lock *portaudio_lock;
};

enum audio_device_kind {
        AUDIO_IN,
        AUDIO_OUT
};


/*
 * For Portaudio threads-related issues see
 * http://www.portaudio.com/trac/wiki/tips/Threading
 */
static void        print_device_info(PaDeviceIndex device);
static int         portaudio_start_stream(PaStream *stream);
static void        portaudio_close(PaStream *stream); // closes and frees all audio resources ( according to valgrind this is not true..  )
static void        portaudio_print_available_devices(enum audio_device_kind);

 /*
  * Shared functions
  */
int portaudio_start_stream(PaStream *stream)
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

void portaudio_capture_help()
{
        portaudio_print_available_devices(AUDIO_IN);
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

void portaudio_close(PaStream * stream)	// closes and frees all audio resources
{
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
	Pa_Terminate();
}

/*
 * capture funcitons
 */
void * portaudio_capture_init(char *cfg)
{
        struct state_portaudio_capture *s;
        int input_device;
	PaError error;
        const	PaDeviceInfo *device_info = NULL;
        
        s = (struct state_portaudio_capture *) malloc(sizeof(struct state_portaudio_capture));
	/* 
	 * so far we only work with portaudio
	 * might get more complicated later..(jack?)
	 */
        s->portaudio_lock = fs_lock_init("portaudio");
         
        if(cfg)
                input_device = atoi(cfg);
        else
                input_device = -1;

        fs_lock_lock(s->portaudio_lock); /* safer with multiple threads */
	printf("Initializing portaudio capture.\n");
	
	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */
		return NULL;
	}

	printf("Using PortAudio version: %s\n", Pa_GetVersionText());

	PaStreamParameters inputParameters;

	// default device
	if(input_device == -1)
	{
		printf("Using default input audio device");
		print_device_info(Pa_GetDefaultInputDevice());
		printf("\n");
		inputParameters.device = Pa_GetDefaultInputDevice();
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
	} else if(input_device >= 0) {
		printf("Using input audio device:");
		print_device_info(input_device);
		printf("\n");
		inputParameters.device = input_device;
                device_info = Pa_GetDeviceInfo(input_device);
	}

        if(CHANNELS <= device_info->maxInputChannels)
                inputParameters.channelCount = CHANNELS;
        else
                inputParameters.channelCount = device_info->maxInputChannels;
        inputParameters.sampleFormat = paInt16;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultHighInputLatency ;
        inputParameters.hostApiSpecificStreamInfo = NULL;


	error = Pa_OpenStream( &s->stream, &inputParameters, NULL, SAMPLE_RATE, paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
									paNoFlag,
									NULL,	// callback function; NULL, because we use blocking functions
									NULL	// user data - none, because we use blocking functions
								);
	if(error != paNoError)
	{
		printf("Error opening audio stream\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */
		return NULL;
	}

        fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */
        
	s->frame.bps = BPS;
        s->frame.ch_count = inputParameters.channelCount;
        s->frame.sample_rate = SAMPLE_RATE;
        s->frame.max_size = SAMPLES_PER_FRAME * s->frame.bps * s->frame.ch_count;
        
        s->frame.data = (char*)malloc(s->frame.max_size);

        memset(s->frame.data, 0, s->frame.max_size);

        portaudio_start_stream(s->stream);

	return s;
}

// read from input device
struct audio_frame * portaudio_read(void *state)
{
        struct state_portaudio_capture *s = 
                        (struct state_portaudio_capture *) state;
	// here we distinguish between interleaved and noninterleved, but non interleaved version hasn't been tested yet
	PaError error;
	
	error = Pa_ReadStream(s->stream, s->frame.data, SAMPLES_PER_FRAME);
        s->frame.data_len = SAMPLES_PER_FRAME * s->frame.bps * s->frame.ch_count;

	if((error != paNoError) && (error != paInputOverflowed))
	{
		printf("Pa read stream error:%s\n", Pa_GetErrorText(error));
		return NULL;
	}
	
	return &s->frame;
}

void portaudio_capture_finish(void *state)
{
        UNUSED(state);
}

void portaudio_capture_done(void *state)
{
        struct state_portaudio_capture *s = (struct state_portaudio_capture *) state;
        fs_lock_lock(s->portaudio_lock);
        portaudio_close(s->stream);
        fs_lock_unlock(s->portaudio_lock);
        free(s->frame.data);
        free(s);
}


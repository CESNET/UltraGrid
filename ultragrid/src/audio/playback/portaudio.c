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

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "portaudio/include/portaudio.h" /* from PortAudio */

#include "audio/audio.h"
#include "audio/playback/portaudio.h"
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "utils/fs_lock.h"

struct state_portaudio_playback {
        audio_frame frame;
        int samples;
        int device;
        PaStream *stream;
        int max_output_channels;
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

/* prototyping */
void portaudio_decode_frame(void *dst, void *src, int data_len, int buffer_len, void *state);
static void      print_device_info(PaDeviceIndex device);
static int       portaudio_start_stream(PaStream *stream);
static void      portaudio_close(PaStream *stream);  /* closes and frees all audio resources ( according to valgrind this is not true..  ) */
static void      portaudio_print_available_devices(enum audio_device_kind);
void             free_audio_frame(audio_frame *buffer);

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

void portaudio_playback_help()
{
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
        
        if(cfg)
                output_device = atoi(cfg);
        else
                output_device = -1;
        Pa_Initialize();
        
        s = calloc(1, sizeof(struct state_portaudio_playback));
        s->portaudio_lock = fs_lock_init("portaudio");
        assert(output_device >= -1);
        s->device = output_device;
        const	PaDeviceInfo *device_info;
        if(output_device >= 0)
                device_info = Pa_GetDeviceInfo(output_device);
        else
                device_info = Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice());
	s->max_output_channels = device_info->maxOutputChannels;
        
        portaudio_reconfigure(s, 16, 2, 48000);
         
	return s;
}

void portaudio_close_playback(void *state)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
                
        free(s->frame.data);
        portaudio_close(s->stream);
}

int portaudio_reconfigure(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
        PaError error;
	PaStreamParameters outputParameters;
        
        if(s->stream != NULL) {
                portaudio_close_playback(s);
        }
        
        s->frame.bps = quant_samples / 8;
        s->frame.ch_count = channels;
        s->frame.sample_rate = sample_rate;
        
        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate; // can hold up to 1 sec
        
        s->frame.data = (char*)malloc(s->frame.max_size);
        assert(s->frame.data != NULL);
        
        fs_lock_lock(s->portaudio_lock); /* safer with multiple threads */
	printf("(Re)initializing portaudio playback.\n");

	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */
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
                        
        outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultHighOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = NULL;

	error = Pa_OpenStream( &s->stream, NULL, &outputParameters, sample_rate, paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
									paNoFlag,
									NULL,	// callback function; NULL, because we use blocking functions
									NULL	// user data - none, because we use blocking functions
								);
        portaudio_start_stream(s->stream);
        
	if(error != paNoError)
	{
		printf("Error opening audio stream\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */
		return FALSE;
	}
        fs_lock_unlock(s->portaudio_lock); /* safer with multiple threads */

        return TRUE;
}                        

// get first empty frame, bud don't consider it as being used
struct audio_frame* portaudio_get_frame(void *state)
{
	return &((struct state_portaudio_playback *) state)->frame;
}

void portaudio_put_frame(void *state, struct audio_frame *buffer)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
                
        PaError error;
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
        
        error = Pa_WriteStream(s->stream, buffer->data, samples_count);

	if(error != paNoError) {
		printf("Pa write stream error: %s\n", Pa_GetErrorText(error));
                while(error == paOutputUnderflowed) { /* put current frame more times to give us time */
                        error = Pa_WriteStream(s->stream, buffer->data, samples_count);
                }
	}
}


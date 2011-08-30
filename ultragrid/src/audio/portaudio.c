/*
 * FILE:    audio/audio.c
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

#include "audio/audio.h"
#include "audio/portaudio.h"

/* default variables for sender */
#define BPS 2 /* paInt16 */
#define SAMPLE_RATE 48000
#define SAMPLES_PER_FRAME 2048
#define CHANNELS 2
#define SECONDS 5

struct state_portaudio_playback {
        audio_frame frame;
        int samples;
        int device;
        PaStream *stream;
};

/*
 * For Portaudio threads-related issues see
 * http://www.portaudio.com/trac/wiki/tips/Threading
 */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* prototyping */
void portaudio_decode_frame(void *dst, void *src, int data_len, int buffer_len, void *state);
void portaudio_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate, int aux);
static void print_device_info(PaDeviceIndex device);

void portaudio_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate, int aux)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
        PaError error;
	PaStreamParameters outputParameters;
        int data_len;
        
        if(s->stream != NULL) {
                portaudio_close_playback(s);
        }
        
        s->frame.bps = quant_samples / 8;
        s->frame.ch_count = channels;
        s->frame.aux = aux;
        s->frame.reconfigure_audio = portaudio_reconfigure_audio;
        s->frame.decoder = portaudio_decode_frame;
        s->frame.sample_rate = sample_rate;
        s->frame.state = s;
        
        data_len = s->frame.bps * s->frame.ch_count * s->frame.sample_rate * SECONDS; // can hold up to 1 sec
        
        s->frame.data = (char*)malloc(data_len);
        assert(s->frame.data != NULL);
        memset(s->frame.data, 0, data_len);
        
        pthread_mutex_lock(&lock); /* safer with multiple threads */
	printf("(Re)initializing portaudio playback.\n");

	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                pthread_mutex_unlock(&lock); /* safer with multiple threads */
		return;
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
		
        outputParameters.channelCount = channels; // output channels
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
                pthread_mutex_unlock(&lock); /* safer with multiple threads */
		return;
	}
        pthread_mutex_unlock(&lock); /* safer with multiple threads */
}                        

void * portaudio_playback_init(int output_device)
{
	
        struct state_portaudio_playback *s;
        
        s = calloc(1, sizeof(struct state_portaudio_playback));
        assert(output_device >= -1);
        s->device = output_device;
        portaudio_reconfigure_audio(s, BPS * 8, CHANNELS, SAMPLE_RATE,
                        0);
         
	return s;
}

PaStream * portaudio_capture_init(int input_device)
{
	PaError error;
        PaStream * ret;
	/* 
	 * so far we only work with portaudio
	 * might get more complicated later..(jack?)
	 */

        pthread_mutex_lock(&lock); /* safer with multiple threads */
	printf("Initializing portaudio capture.\n");
	
	error = Pa_Initialize();
	if(error != paNoError)
	{
		printf("error initializing portaudio\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                pthread_mutex_unlock(&lock); /* safer with multiple threads */
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
	}
	else
	if(input_device >= 0)
	{
		printf("Using input audio device:");
		print_device_info(input_device);
		printf("\n");
		inputParameters.device = input_device;
	}

        inputParameters.channelCount = CHANNELS;
        inputParameters.sampleFormat = paInt16;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultHighInputLatency ;
        inputParameters.hostApiSpecificStreamInfo = NULL;


	error = Pa_OpenStream( &ret, &inputParameters, NULL, SAMPLE_RATE, paFramesPerBufferUnspecified, // frames per buffer // TODO decide on the amount
									paNoFlag,
									NULL,	// callback function; NULL, because we use blocking functions
									NULL	// user data - none, because we use blocking functions
								);
	if(error != paNoError)
	{
		printf("Error opening audio stream\n");
		printf("\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                pthread_mutex_unlock(&lock); /* safer with multiple threads */
		return NULL;
	}
        pthread_mutex_unlock(&lock); /* safer with multiple threads */

	return ret;
}

void portaudio_close(PaStream * stream)	// closes and frees all audio resources
{
        pthread_mutex_lock(&lock); /* safer with multiple threads */
	Pa_StopStream(stream);	// may not be necessary
        Pa_CloseStream(stream);
	Pa_Terminate();
        pthread_mutex_unlock(&lock); /* safer with multiple threads */
}

void portaudio_close_playback(void *state)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
                
        free(s->frame.data);
        portaudio_close(s->stream);
}

// get first empty frame, bud don't consider it as being used
struct audio_frame* portaudio_get_frame(void *state)
{
	return &((struct state_portaudio_playback *) state)->frame;
}

void portaudio_print_available_devices(void)
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

	for(i = 0; i < numDevices; i++)
	{
		if(i == Pa_GetDefaultInputDevice())
			printf("(*i) ");
		if(i == Pa_GetDefaultOutputDevice())
			printf("(*o)");
			
		printf("Device %d:", i);
		print_device_info(i);
		printf("\n");
	}

	return;
}

static void print_device_info(PaDeviceIndex device)
{
	if( (device < 0) || (device >= Pa_GetDeviceCount()) )
	{
		printf("Requested info on non-existing device");
		return;
	}
	
	const	PaDeviceInfo *device_info = Pa_GetDeviceInfo(device);
	printf(" %s   (output channels: %d;   input channels: %d)", device_info->name, device_info->maxOutputChannels, device_info->maxInputChannels);
}

// read from input device
int portaudio_read(PaStream *stream, audio_frame *buffer)
{
	// here we distinguish between interleaved and noninterleved, but non interleaved version hasn't been tested yet
	PaError error;
	
	error = Pa_ReadStream( stream, buffer->data, SAMPLES_PER_FRAME);

	if((error != paNoError) && (error != paInputOverflowed))
	{
		printf("Pa read stream error:%s\n", Pa_GetErrorText(error));
		return 1;
	}
	
	return 0;
}

int portaudio_init_audio_frame(audio_frame *buffer)
{
	buffer->bps = BPS;
        buffer->ch_count = CHANNELS;
        buffer->aux = 0;
        buffer->sample_rate = SAMPLE_RATE;
        buffer->data_len = SAMPLES_PER_FRAME * buffer->bps * buffer->ch_count * SECONDS;
        
        /* we allocate only one block, pointers point to appropriate parts of the block */
        if((buffer->data = (char*)malloc(buffer->data_len)) == NULL)
        {
                printf("Error allocating memory for audio buffers\n");
                return -1;
        }

        memset(buffer->data, 0, buffer->data_len);

	return 0;
}

void free_audio_frame(audio_frame *buffer)
{
        free(buffer->data); /* next channels point to same block */
}

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

void portaudio_put_frame(void *state)
{
        struct state_portaudio_playback * s = 
                (struct state_portaudio_playback *) state;
                
        PaError error;

        error = Pa_WriteStream(s->stream, s->frame.data, s->frame.data_len / (s->frame.bps * s->frame.ch_count));

	if(error != paNoError) {
		printf("Pa write stream error:%s\n", Pa_GetErrorText(error));
	}
}

void portaudio_decode_frame(void *dst, void *src, int data_len, int buffer_len, void *state)
{
        struct state_portaudio_playback * s =
                (struct state_portaudio_playback *) state;
        
        // we can receive 1 sec at most
        assert(buffer_len <= s->frame.bps * s->frame.ch_count * s->frame.sample_rate * SECONDS);
        
        s->frame.data_len = buffer_len;
        memcpy(dst, src, data_len);
}

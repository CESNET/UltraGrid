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
 *
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio.h"
#include "portaudio.h"

const int interleaved = 1;      // whether we use interleaved or non-interleaved input. ideally should get rid of this and use always non-interleaved mode...but there are some troubles with nonInterlevad support in MaxOSX(probably portaudio bug)

PaStream *stream;               // portaudio stream defined globally

void audio_wait_for_reading(void)
{
        unsigned int x;
//      printf("%d..\n", Pa_GetStreamWriteAvailable(stream));
        while ((x =
                Pa_GetStreamReadAvailable(stream)) <
               audio_samples_per_frame * 2)
                printf("%d..", x);

        printf("ok, now:%d\n", x);

}

int audio_init(int output_device, int input_device)
{
        /* 
         * so far we only work with portaudio
         * might get more complicated later..(jack?)
         */

        printf("Initializing portaudio..\n");

        PaError error;

        error = Pa_Initialize();
        if (error != paNoError) {
                printf("error initializing portaudio\n");
                printf("\tPortAudio error: %s\n", Pa_GetErrorText(error));
                return -1;
        }

        printf("Using PortAudio version: %s\n", Pa_GetVersionText());

        PaStreamParameters inputParameters, outputParameters;

        // default device
        if (output_device == -1) {
                printf("\nUsing default output audio device:");
                fflush(stdout);
                print_device_info(Pa_GetDefaultOutputDevice());
                printf("\n");
                outputParameters.device = Pa_GetDefaultOutputDevice();
        } else if (output_device >= 0) {
                printf("\nUsing output audio device:");
                print_device_info(output_device);
                printf("\n");
                outputParameters.device = output_device;
        }
        // default device
        if (input_device == -1) {
                printf("Using default input audio device");
                print_device_info(Pa_GetDefaultInputDevice());
                printf("\n");
                inputParameters.device = Pa_GetDefaultInputDevice();
        } else if (input_device >= 0) {
                printf("Using input audio device:");
                print_device_info(input_device);
                printf("\n");
                inputParameters.device = input_device;
        }

        if (input_device != -2) {
                inputParameters.channelCount = 1;
                inputParameters.sampleFormat = paInt24;
                if (!interleaved)
                        inputParameters.sampleFormat |= paNonInterleaved;

                inputParameters.suggestedLatency =
                    Pa_GetDeviceInfo(inputParameters.device)->
                    defaultHighInputLatency;

                inputParameters.hostApiSpecificStreamInfo = NULL;
        }

        if (output_device != -2) {
                outputParameters.channelCount = 1;      // output channels
                outputParameters.sampleFormat = paInt24;
                if (!interleaved)
                        outputParameters.sampleFormat |= paNonInterleaved;
                outputParameters.suggestedLatency =
                    Pa_GetDeviceInfo(outputParameters.device)->
                    defaultHighOutputLatency;
                outputParameters.hostApiSpecificStreamInfo = NULL;
        }

        error = Pa_OpenStream(&stream, (input_device == -2) ? NULL : &inputParameters, (output_device == -2) ? NULL : &outputParameters, 48000, audio_samples_per_frame,        // frames per buffer // TODO decide on the amount
                              paNoFlag, NULL,   // callback function; NULL, because we use blocking functions
                              NULL      // user data - none, because we use blocking functions
            );
        if (error != paNoError) {
                printf("Error opening audio stream\n");
                printf("\tPortAudio error: %s\n", Pa_GetErrorText(error));
                return -1;
        }

        return 0;
}

int audio_close()               // closes and frees all audio resources
{
        Pa_StopStream(stream);  // may not be necessary
        Pa_Terminate();
}

int audio_buffer_full(const audio_frame_buffer * buffer)
{
        return (buffer->occupation == buffer->total_number_of_frames);
}

int audio_buffer_empty(audio_frame_buffer * buffer)
{
        return (buffer->occupation == 0);
}

int init_audio_frame_buffer(audio_frame_buffer * buffer, int frames_per_buffer)
{
        assert(frames_per_buffer > 0);

        int i;
        if ((buffer->audio_frames =
             (audio_frame **) malloc(frames_per_buffer *
                                     sizeof(audio_frame *))) == NULL) {
                printf("Error allocating memory for audio buffer!\n");
                return 1;
        }

        for (i = 0; i < frames_per_buffer; i++) {
                buffer->audio_frames[i] = (audio_frame *) malloc(sizeof(audio_frame));  // FIXME: all audio frames should be allocated with one malloc call...
                init_audio_frame(buffer->audio_frames[i],
                                 audio_samples_per_frame);
        }

        buffer->total_number_of_frames = frames_per_buffer;
        buffer->start = 0;
        buffer->end = 0;
        buffer->occupation = 0;
        return 0;
}

// write buffer to output device
int audio_write(const audio_frame * buffer)
{
        PaError error;
        if (interleaved) {
                audio_frame_to_interleaved_buffer(buffer->tmp_buffer, buffer,
                                                  1);
                error =
                    Pa_WriteStream(stream, buffer->tmp_buffer,
                                   buffer->samples_per_channel);
        } else
                error =
                    Pa_WriteStream(stream, buffer->channels,
                                   buffer->samples_per_channel);

        if (error != paNoError) {
                printf("Pa write stream error:%s\n", Pa_GetErrorText(error));
                return 1;
        }

        return 0;
}

// get first empty frame, bud don't consider it as being used
audio_frame *audio_buffer_get_empty_frame(audio_frame_buffer * buffer)
{
        // if we are full, we return NULL
        if (audio_buffer_full(buffer))
                return NULL;

        audio_frame *frame = buffer->audio_frames[buffer->end];
        return frame;
}

void print_available_devices(void)
{
        int numDevices;

        PaError error;

        error = Pa_Initialize();
        if (error != paNoError) {
                printf("error initializing portaudio\n");
                printf("\tPortAudio error: %s\n", Pa_GetErrorText(error));
                return;
        }

        numDevices = Pa_GetDeviceCount();
        if (numDevices < 0) {
                printf("Error getting portaudio devices number\n");
                return;
        }
        if (numDevices == 0) {
                printf("There are NO available audio devices!\n");
                return;
        }

        int i;
        printf("Available devices(%d)\n", numDevices);
        for (i = 0; i < numDevices; i++) {
                if (i == Pa_GetDefaultInputDevice())
                        printf("(*i) ");
                if (i == Pa_GetDefaultOutputDevice())
                        printf("(*o)");

                printf("Device %d :", i);
                print_device_info(i);
                printf("\n");
        }

        return;
}

void print_device_info(PaDeviceIndex device)
{
        if ((device < 0) || (device >= Pa_GetDeviceCount())) {
                printf("Requested info on non-existing device");
                return;
        }

        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(device);
        printf(" %s   (output channels: %d;   input channels: %d)",
               device_info->name, device_info->maxOutputChannels,
               device_info->maxInputChannels);
}

// read from input device
int audio_read(audio_frame * buffer)
{
        // here we distinguish between interleaved and noninterleved, but non interleaved version hasn't been tested yet
        PaError error;
        if (interleaved) {
                error =
                    Pa_ReadStream(stream, buffer->tmp_buffer,
                                  buffer->samples_per_channel);
                interleaved_buffer_to_audio_frame(buffer, buffer->tmp_buffer,
                                                  1);
        } else {
                error =
                    Pa_ReadStream(stream, buffer->channels,
                                  buffer->samples_per_channel);
        }

        if ((error != paNoError) && (error != paInputOverflowed)) {
                printf("Pa read stream error:%s\n", Pa_GetErrorText(error));
                return 1;
        }

        return 0;
}

int init_audio_frame(audio_frame * buffer, unsigned int samples_per_channel)
{
        int i;
        for (i = 0; i < 8; i++) {
                if ((buffer->channels[i] =
                     (char *)malloc(samples_per_channel * 3 /* ~24 bit */ )) ==
                    NULL) {
                        printf("Error allocating memory for audio buffers\n");
                        return -1;
                }

                memset(buffer->channels[i], 0, samples_per_channel * 3);
        }
        buffer->samples_per_channel = samples_per_channel;

        // the temporary buffer is large enough to store all channels
        if ((buffer->tmp_buffer =
             (char *)malloc(samples_per_channel * 3 * 8)) == NULL) {
                printf("Error allocating temporary buffer");
                return -1;
        }

        return 0;
}

void free_audio_frame_buffer(audio_frame_buffer * buffer)
{
        int i;
        for (i = 0; i < buffer->total_number_of_frames; i++)
                free(buffer->audio_frames[i]);

        free(buffer->audio_frames);

        buffer->start = buffer->end = buffer->total_number_of_frames = 0;
}

void free_audio_frame(audio_frame * buffer)
{
        int i;
        for (i = 0; i < 8; i++)
                free(buffer->channels[i]);

        free(buffer->tmp_buffer);
}

inline
    void interleaved_buffer_to_audio_frame(audio_frame * pa_buffer,
                                           const char *in_buffer,
                                           int num_channels)
{
        assert(num_channels == 1);      // we only handle mono buffer for now...

        // we suppose the number of samples in in_buffer is equal to the number in pa_buffer (this should be set globally)

        memcpy(pa_buffer->channels[0], in_buffer,
               pa_buffer->samples_per_channel * 3);
}

inline
    void network_buffer_to_audio_frame(audio_frame * pa_buffer,
                                       const char *network_buffer)
{
        unsigned int i;
        const char *tmp_buffer = network_buffer;
        for (i = 0; i < pa_buffer->samples_per_channel * 8; i++) {
                int actual_channel = i % 8;
                int actual_sample = i / 8;

                pa_buffer->channels[actual_channel][actual_sample * 3 + 0] =
                    tmp_buffer[2];
                pa_buffer->channels[actual_channel][actual_sample * 3 + 1] =
                    tmp_buffer[1];
                pa_buffer->channels[actual_channel][actual_sample * 3 + 2] =
                    tmp_buffer[0];

                tmp_buffer += 3;        // we iterate over 3 byte samples..
        }
}

int audio_ready_to_write(void)
{
        return Pa_GetStreamWriteAvailable(stream);
}

int audio_start_stream(void)
{
        PaError error;

        error = Pa_StartStream(stream);
        if (error != paNoError) {
                printf("Error starting stream:%s\n", Pa_GetErrorText(error));
                printf("\tPortAudio error: %s\n", Pa_GetErrorText(error));
                return -1;
        }

        return 0;
}

inline
    void audio_frame_to_interleaved_buffer(char *in_buffer,
                                           const audio_frame * pa_buffer,
                                           int num_channels)
{
        assert(num_channels == 1);

        memcpy(in_buffer, pa_buffer->channels[0],
               pa_buffer->samples_per_channel * 3);
}

inline
    void audio_frame_to_network_buffer(char *network_buffer,
                                       const audio_frame * a_buffer)
{
        assert(!(a_buffer->samples_per_channel % 8));   // we send data in octets

        unsigned int i;
        for (i = 0; i < a_buffer->samples_per_channel * 3; i += 3) {
#ifdef WORDS_BIGENDIAN
                printf
                    ("Error! Big endian machines are not currenty supported by audio transfer protocol!\n");
#endif

                int j;
                for (j = 0; j < 8; j++) {
                        // we need to revers the data to network byte order
                        network_buffer[0] = a_buffer->channels[j][i + 2];
                        network_buffer[1] = a_buffer->channels[j][i + 1];
                        network_buffer[2] = a_buffer->channels[j][i + 0];
                        network_buffer += 3;    // iterate over the output buffer
                }
        }
}

// we return first full frame and think about it as being empty since now
audio_frame *audio_buffer_get_full_frame(audio_frame_buffer * buffer)
{
        if (audio_buffer_empty(buffer))
                return NULL;

        audio_frame *frame = buffer->audio_frames[buffer->start];

        buffer->start++;
        if (buffer->start == buffer->total_number_of_frames)
                buffer->start = 0;

        buffer->occupation--;
        return frame;
}

// this should be called in case last audio_frame returned by audio_buffer_get_empty_frame should be considered full (used)
void audio_buffer_mark_last_frame_full(audio_frame_buffer * buffer)
{
        buffer->end++;
        if (buffer->end == buffer->total_number_of_frames)
                buffer->end = 0;

        buffer->occupation++;
}

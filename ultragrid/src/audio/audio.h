/*
 * FILE:    audio/audio.h
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

#include "config.h"

#ifndef _AUDIO_H_
#define _AUDIO_H_

#ifdef HAVE_AUDIO
#include "portaudio.h"

typedef struct
{
	unsigned int samples_per_channel;	// each sample is 24 bit
	char *channels[8];	// ultragrid uses 8 channels everytime

	char *tmp_buffer;	// just a temporary buffer. has the same size as all channels together
}
audio_frame;

// fifo buffer for incomming audio frames
typedef struct
{
	audio_frame **audio_frames;
	int start, end;
	int total_number_of_frames;
	int occupation;
}
audio_frame_buffer;

static const int audio_samples_per_frame = 32;	// number of samples (3B) each channel has
static const int audio_payload_type = 97;

int audio_init(int playback_device, int capture_device);
int audio_close();	// closes and frees all audio resources ( according to valgrind this is not true..  )

int audio_read(audio_frame *buffer);
int audio_write(const audio_frame *buffer);

void print_available_devices(void);
void print_device_info(PaDeviceIndex device);

int init_audio_frame(audio_frame *frame, unsigned int samples_per_channel);
void free_audio_frame(audio_frame *buffer);	// buffers should be freed after usage

int audio_ready_to_write(void);
int audio_start_stream(void);

audio_frame* audio_buffer_get_full_frame(audio_frame_buffer *buffer);
int audio_buffer_full(const audio_frame_buffer *buffer);
audio_frame* audio_buffer_get_empty_frame(audio_frame_buffer *buffer);
void audio_buffer_mark_last_frame_full(audio_frame_buffer *buffer);
int audio_buffer_empty(audio_frame_buffer *buffer);
int init_audio_frame_buffer(audio_frame_buffer *buffer, int frames_per_buffer);
void free_audio_frame_buffer(audio_frame_buffer *buffer);

inline
void interleaved_buffer_to_audio_frame(audio_frame *pa_buffer, const char *in_buffer, int num_channels);

inline
void audio_frame_to_network_buffer(char *network_buffer, const audio_frame *a_buffer);

inline
void network_buffer_to_audio_frame(audio_frame *pa_buffer, const char *network_buffer);

inline
void audio_frame_to_interleaved_buffer(char *in_buffer, const audio_frame *pa_buffer, int num_channels);


#endif /* HAVE_AUDIO */

#endif

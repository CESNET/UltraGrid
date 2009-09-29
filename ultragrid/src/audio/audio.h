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

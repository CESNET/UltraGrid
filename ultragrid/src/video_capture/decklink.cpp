/*
 * FILE:    decklink.cpp
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
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
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

#ifdef __cplusplus
extern "C" {
#endif

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "tv.h"

#include "debug.h"
#include "video_capture.h"
#include "video_codec.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#ifdef HAVE_DECKLINK		/* From config.h */

#include "video_capture/decklink.h"

#include "DeckLinkAPI.h" /* From DeckLink SDK */ 

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

// static int	device = 0; // use first BlackMagic device
// static int	mode = 5; // for Intensity
// static int	mode = 6; // for Decklink  6) HD 1080i 59.94; 1920 x 1080; 29.97 FPS 7) HD 1080i 60; 1920 x 1080; 30 FPS
static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

int frames = 0;
struct timeval t, t0;

#ifdef __cplusplus
extern "C" {
#endif

extern int		 should_exit;

#ifdef __cplusplus
}
#endif

struct vidcap_decklink_state;

class VideoDelegate : public IDeckLinkInputCallback
{
private:
	int32_t mRefCount;
	double  lastTime;
    
public:
	int	newFrameReady;
	void*	pixelFrame;
	int	first_time;
	struct vidcap_decklink_state	*s;
	
	void set_vidcap_state(vidcap_decklink_state	*state);
	
	VideoDelegate () {
		newFrameReady = 0;
		first_time = 1;
		s = NULL;
	};

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID *ppv) { return E_NOINTERFACE; }
	virtual ULONG STDMETHODCALLTYPE AddRef(void)
	{
		return mRefCount++; 
	};
	virtual ULONG STDMETHODCALLTYPE  Release(void)
	{
		int32_t newRefValue;
        	
		newRefValue = mRefCount--;
		if (newRefValue == 0)
		{
			delete this;
			return 0;
		}        
        	return newRefValue;
	};
	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode*, BMDDetectedVideoInputFormatFlags)
	{
		return S_OK;
	};
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);    
};

struct vidcap_decklink_state {
	IDeckLink*		deckLink;
	IDeckLinkInput*		deckLinkInput;
	int			device;
	int			mode;
	// void*			rtp_buffer;
	VideoDelegate*		delegate;
	unsigned int		next_frame_time; // avarege time between frames
	pthread_mutex_t	 	lock;
	pthread_cond_t	 	boss_cv;
	int		 	boss_waiting;
        struct video_frame      frame;
        const struct codec_info_t *c_info;
};

/* DeckLink SDK objects */

void
print_output_modes (IDeckLink* deckLink);

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *arrivedFrame, IDeckLinkAudioInputPacket *audioPacket)
{

	// Video

	pthread_mutex_lock(&(s->lock));
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	if(arrivedFrame)
	{
		if (arrivedFrame->GetFlags() & bmdFrameHasNoInputSource)
		{
			fprintf(stderr, "Frame received (#%d) - No input signal detected\n", frames);
		}
		else{
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, arrivedFrame->GetRowBytes() * arrivedFrame->GetHeight());
			
		}
		frames++;
	}

	arrivedFrame->GetBytes(&pixelFrame);

	if(first_time){
		first_time = 0;
	}

	newFrameReady = 1; // The new frame is ready to grab
	
	if (s->boss_waiting) {
		pthread_cond_signal(&(s->boss_cv));
	}
	
// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&(s->lock));

	gettimeofday(&t, NULL);
	double seconds = tv_diff(t, t0);	
	if (seconds >= 5) {
		float fps  = frames / seconds;
		fprintf(stderr, "%d frames in %g seconds = %g FPS\n", frames, seconds, fps);
		t0 = t;
		frames = 0;
	}
        
	debug_msg("VideoInputFrameArrived - END\n"); /* TOREMOVE */

	return S_OK;
}

void VideoDelegate::set_vidcap_state(vidcap_decklink_state	*state){
	s = state;
}

/* HELP */

int
decklink_help()
{
	IDeckLinkIterator*		deckLinkIterator;
	IDeckLink*			deckLink;
	int				numDevices = 0;
	HRESULT				result;

	printf("Decklink options:\n");
	printf("\tdevice:mode\n");
	
	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = CreateDeckLinkIteratorInstance();
	if (deckLinkIterator == NULL)
	{
		fprintf(stderr, "A DeckLink iterator could not be created.  The DeckLink drivers may not be installed.\n");
		return 0;
	}
	
	// Enumerate all cards in this system
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		char *		deviceNameString = NULL;
		
		
		// *** Print the model name of the DeckLink card
		result = deckLink->GetModelName((const char **) &deviceNameString);
		if (result == S_OK)
		{
			printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameString);
			free(deviceNameString);
		}
		
		// Increment the total number of DeckLink cards found
		numDevices++;
	
		// ** List the video output display modes supported by the card
		print_output_modes(deckLink);
				
		// Release the IDeckLink instance when we've finished with it to prevent leaks
		deckLink->Release();
	}
	
	deckLinkIterator->Release();

	// If no DeckLink cards were found in the system, inform the user
	if (numDevices == 0)
	{
		printf("\nNo Blackmagic Design devices were found.\n");
		return 0;
	} else {
                printf("Available Colorspaces:\n");
                printf("\t2vuy\n");
                printf("\tv210\n");
                printf("\tRGBA\n");
                printf("\tR10k\n");
        }
	printf("\n");

	return 1;
}

/* SETTINGS */

int
settings_init(void *state, char *fmt)
{
	struct vidcap_decklink_state *s = (struct vidcap_decklink_state *) state;

	if(strcmp(fmt, "help")==0) {
		decklink_help();
		return 0;
	}

	char *tmp;

	// choose device
	tmp = strtok(fmt, ":");
	if(!tmp) {
		fprintf(stderr, "Wrong config %s\n", fmt);
		return 0;
	}
	s->device = atoi(tmp);

	// choose mode
	tmp = strtok(NULL, ":");
	if(!tmp) {
		fprintf(stderr, "Wrong config %s\n", fmt);
		return 0;
	}
	s->mode = atoi(tmp);

	tmp = strtok(NULL, ":");
        s->c_info = 0;
        if(!tmp) {
                int i;
                for(i=0; codec_info[i].name != NULL; i++) {
                    if(codec_info[i].codec == Vuy2) {
                        s->c_info = &codec_info[i];
                        s->frame.color_spec = codec_info[i].codec;
                        break;
                    }
                }
        } else {
                int i;
                for(i=0; codec_info[i].name != NULL; i++) {
                    if(strcmp(codec_info[i].name, tmp) == 0) {
                         s->c_info = &codec_info[i];
                         s->frame.color_spec = codec_info[i].codec;
                         break;
                    }
                }
                if(s->c_info == 0) {
			fprintf(stderr, "Wrong config. Unknown color space %s\n", tmp);
                	return 0;
                }
        }

	return 1;	
}

/* External API ***************************************************************/

struct vidcap_type *
vidcap_decklink_probe(void)
{

	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DECKLINK_ID;
		vt->name        = "decklink";
		vt->description = "Blackmagic DeckLink card";
	}
	return vt;
}

void *
vidcap_decklink_init(char *fmt)
{
	debug_msg("vidcap_decklink_init\n"); /* TOREMOVE */

	struct vidcap_decklink_state *s;

	int dnum, mnum;

	IDeckLinkIterator*	deckLinkIterator;
	IDeckLink*		deckLink;
	HRESULT			result;

	IDeckLinkInput*			deckLinkInput = NULL;
	IDeckLinkDisplayModeIterator*	displayModeIterator = NULL;
	IDeckLinkDisplayMode*		displayMode = NULL;
	IDeckLinkConfiguration*		deckLinkConfiguration = NULL;

	s = (struct vidcap_decklink_state *) calloc(1, sizeof(struct vidcap_decklink_state));
	if (s == NULL) {
		//printf("Unable to allocate DeckLink state\n",fps);
		printf("Unable to allocate DeckLink state\n");
		return NULL;
	}

	// SET UP device and mode
	if(settings_init(s, fmt) == 0) {
		free(s);
		return NULL;
	}

	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = CreateDeckLinkIteratorInstance();
	if (deckLinkIterator == NULL)
	{
		printf("A DeckLink iterator could not be created. The DeckLink drivers may not be installed.\n");
		return NULL;
	}

	dnum = 0;
	deckLink = NULL;
	bool device_found = false;
    
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		if (s->device != dnum) {
			dnum++;

			// Release the IDeckLink instance when we've finished with it to prevent leaks
			deckLink->Release();
			deckLink = NULL;
			continue;	
		}

		device_found = true;
		dnum++;

		s->deckLink = deckLink;

		const char *deviceNameString = NULL;
		
		// Print the model name of the DeckLink card
		result = deckLink->GetModelName(&deviceNameString);
		if (result == S_OK)
		{	
			printf("Using device [%s]\n", deviceNameString);

			// Query the DeckLink for its configuration interface
			result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
			if (result != S_OK)
			{
				printf("Could not obtain the IDeckLinkInput interface - result = %08x\n", result);
				goto error;
			}

			s->deckLinkInput = deckLinkInput;

			// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
			result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
			if (result != S_OK)
			{
				printf("Could not obtain the video input display mode iterator - result = %08x\n", result);
				goto error;
			}

			mnum = 0;
			bool mode_found = false;

			while (displayModeIterator->Next(&displayMode) == S_OK)
			{
				if (s->mode != mnum) {
					mnum++;
					// Release the IDeckLinkDisplayMode object to prevent a leak
					displayMode->Release();
					continue;
				}

				mode_found = true;
				mnum++; 

				printf("The desired display mode is supported: %d\n",s->mode);  
                
				const char *displayModeString = NULL;

				result = displayMode->GetName(&displayModeString);
				if (result == S_OK)
				{
					BMDPixelFormat pf;
					switch(s->c_info->codec) {
                                          case RGBA:
						pf = bmdFormat8BitBGRA;
						break;
					  case Vuy2:
						pf = bmdFormat8BitYUV;
                                                break;
					  case R10k:
						pf = bmdFormat10BitRGB;
						break;
					  case v210:
						pf = bmdFormat10BitYUV;
						break;
					  default:
						printf("Unsupported codec! %s\n", s->c_info->name);
					}
					// get avarage time between frames
					BMDTimeValue	frameRateDuration;
					BMDTimeScale	frameRateScale;
					double				fps;

					// Obtain the display mode's properties
					s->frame.width = displayMode->GetWidth();
					s->frame.height = displayMode->GetHeight();

					displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
					s->frame.fps = (double)frameRateScale / (double)frameRateDuration;
					s->next_frame_time = (int) (1000000 / s->frame.fps); // in microseconds
					debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeString, s->frame.width, s->frame.height, s->frame.fps, s->next_frame_time); /* TOREMOVE */  

					deckLinkInput->StopStreams();

					printf("Enable video input: %s\n", displayModeString);

					result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, 0);
					if (result != S_OK)
					{
						printf("Could not enable video input: %08x\n", result);
						goto error;
					}

					// Query the DeckLink for its configuration interface
					result = deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
					if (result != S_OK)
					{
						printf("Could not obtain the IDeckLinkConfiguration interface: %08x\n", result);
						goto error;
					}

					BMDVideoConnection conn;
					switch (connection) {
					case 0:
						conn = bmdVideoConnectionSDI;
						break;
					case 1:
						conn = bmdVideoConnectionHDMI;
						break;
					case 2:
						conn = bmdVideoConnectionComponent;
						break;
					case 3:
						conn = bmdVideoConnectionComposite;
						break;
					case 4:
						conn = bmdVideoConnectionSVideo;
						break;
					case 5:
						conn = bmdVideoConnectionOpticalSDI;
						break;
					default:
						break;
					}
                    
					if (deckLinkConfiguration->SetVideoInputFormat(conn) == S_OK) {
						printf("Input set to: %d\n", connection);
					}

					// We don't want to process audio
					deckLinkInput->DisableAudioInput();

					// set Callback which returns frames
					s->delegate = new VideoDelegate();
					s->delegate->set_vidcap_state(s);
					deckLinkInput->SetCallback(s->delegate);

					// Start streaming
					printf("Start capture\n", connection);
					result = deckLinkInput->StartStreams();
					if (result != S_OK)
					{
						printf("Could not start stream: %08x\n", result);
						goto error;
					}

				}else{
					printf("Could not : %08x\n", result);
					goto error;
				}

				displayMode->Release();
				displayMode = NULL;
			}

			// check if any mode was found
			if (mode_found == false)
			{
				printf("Mode %d wasn't found.\n", s->mode);
						goto error;
			}

			if (displayModeIterator != NULL){
				displayModeIterator->Release();
				displayModeIterator = NULL;
			}
		}
	}

	// check if any mode was found
	if (device_found == false)
	{
		printf("Device %d wasn't found.\n", s->device);
		goto error;
	}

	// init mutex
	pthread_mutex_init(&(s->lock), NULL);
	pthread_cond_init(&(s->boss_cv), NULL);
	
	s->boss_waiting = FALSE;        	

	if(s->c_info->h_align) {
           s->frame.src_linesize = ((s->frame.width + s->c_info->h_align - 1) / s->c_info->h_align) * 
                s->c_info->h_align;
        } else {
             s->frame.src_linesize = s->frame.width;
        }
        s->frame.src_linesize *= s->c_info->bpp;
        s->frame.data_len = s->frame.src_linesize * s->frame.height;

	printf("DeckLink capture device enabled\n");

	debug_msg("vidcap_decklink_init - END\n"); /* TOREMOVE */

	return s;
error:

	if(displayMode != NULL)
	{
		displayMode->Release();
		displayMode = NULL;
	}

	if(deckLinkInput != NULL)
	{
		deckLinkInput->Release();
		deckLinkInput = NULL;
	}

	if(deckLink != NULL)
	{
		deckLink->Release();
		deckLink = NULL;
	}

	return NULL;
}

void
vidcap_decklink_done(void *state)
{
	debug_msg("vidcap_decklink_done\n"); /* TOREMOVE */

	HRESULT	result;

	struct vidcap_decklink_state *s = (struct vidcap_decklink_state *) state;

	assert (s != NULL);

	if (s!= NULL) {
		result = s->deckLinkInput->StopStreams();
		if (result != S_OK)
		{
			printf("Could not stop stream: %08x\n", result);
		}

		if(s->deckLinkInput != NULL)
		{
			s->deckLinkInput->Release();
			s->deckLinkInput = NULL;
		}

		if(s->deckLink != NULL)
		{
			s->deckLink->Release();
			s->deckLink = NULL;
		}

		free(s);
	}
}

struct video_frame *
vidcap_decklink_grab(void *state, int * count)
{
	debug_msg("vidcap_decklink_grab\n"); /* TO REMOVE */

	struct vidcap_decklink_state 	*s = (struct vidcap_decklink_state *) state;
	struct video_frame		*vf;

	HRESULT	result;
	
	int		rc;
	struct timespec	ts;
	struct timeval	tp;

	int timeout = 0;

	pthread_mutex_lock(&(s->lock));
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	debug_msg("vidcap_decklink_grab - before while\n"); /* TOREMOVE */

	while (!s->delegate->newFrameReady) {
		s->boss_waiting = TRUE;
		debug_msg("vidcap_decklink_grab - pthread_cond_timedwait\n"); /* TOREMOVE */

		// get time for timeout
		rc =  gettimeofday(&tp, NULL);

		/* Convert from timeval to timespec */
		ts.tv_sec  = tp.tv_sec;
		ts.tv_nsec = tp.tv_usec * 1000;
		ts.tv_nsec += 2 * s->next_frame_time * 1000;
		// make it correct
		ts.tv_sec += ts.tv_nsec / 1000000000;
		ts.tv_nsec = ts.tv_nsec % 1000000000;

		debug_msg("vidcap_decklink_grab - current time: %02d:%03d\n",tp.tv_sec, tp.tv_usec/1000); /* TOREMOVE */

		rc = pthread_cond_timedwait(&(s->boss_cv), &(s->lock), &ts);
		s->boss_waiting = FALSE;
		
		debug_msg("vidcap_decklink_grab - AFTER pthread_cond_timedwait - newFrameReady: %d\n",s->delegate->newFrameReady); /* TOREMOVE */

		if (rc != 0) { //(rc == ETIMEDOUT) {
			printf("Waiting for new frame timed out!\n");
			debug_msg("Waiting for new frame timed out!\n");

			// try to restart stream
			/*
			debug_msg("Try to restart DeckLink stream!\n");
			result = s->deckLinkInput->StopStreams();
			if (result != S_OK)
			{
				debug_msg("Could not stop stream: %08x\n", result);
			}
			result = s->deckLinkInput->StartStreams();
			if (result != S_OK)
			{
				debug_msg("Could not start stream: %08x\n", result);
				return NULL; // really end ???
			}
			*/

			if((!s->delegate->first_time) || (should_exit)){
				s->delegate->newFrameReady = 1;
				timeout = 1;
			}else{
				// wait half of timeout
				usleep(s->next_frame_time);
			}
		}
	}

	s->delegate->newFrameReady = 0;

// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&(s->lock));

	if (s->delegate->pixelFrame != NULL) {
                s->frame.data = (char*)s->delegate->pixelFrame;
                if(s->c_info->codec == RGBA) {
                    vc_copylineRGBA((unsigned char*)s->frame.data, (unsigned char*)s->frame.data, s->frame.data_len, 16, 8, 0);
                }
                count = 1;
                return &s->frame;
	}
        count = 0;
	return NULL;
}

/* function from DeckLink SDK sample DeviceList */

void
print_output_modes (IDeckLink* deckLink)
{
	IDeckLinkOutput*			deckLinkOutput = NULL;
	IDeckLinkDisplayModeIterator*		displayModeIterator = NULL;
	IDeckLinkDisplayMode*			displayMode = NULL;
	HRESULT					result;	
	int 					displayModeNumber = 0;
	
	// Query the DeckLink for its configuration interface
	result = deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&deckLinkOutput);
	if (result != S_OK)
	{
		fprintf(stderr, "Could not obtain the IDeckLinkOutput interface - result = %08x\n", result);
		goto bail;
	}
	
	// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on output
	result = deckLinkOutput->GetDisplayModeIterator(&displayModeIterator);
	if (result != S_OK)
	{
		fprintf(stderr, "Could not obtain the video output display mode iterator - result = %08x\n", result);
		goto bail;
	}
	
	// List all supported output display modes
	printf("display modes:\n");
	while (displayModeIterator->Next(&displayMode) == S_OK)
	{
		char *			displayModeString = NULL;
		
		result = displayMode->GetName((const char **) &displayModeString);
		if (result == S_OK)
		{
			char			modeName[64];
			int				modeWidth;
			int				modeHeight;
			BMDTimeValue	frameRateDuration;
			BMDTimeScale	frameRateScale;
			
			
			// Obtain the display mode's properties
			modeWidth = displayMode->GetWidth();
			modeHeight = displayMode->GetHeight();
			displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
			printf("%d.) %-20s \t %d x %d \t %g FPS\n",displayModeNumber, displayModeString, modeWidth, modeHeight, (double)frameRateScale / (double)frameRateDuration);
			free(displayModeString);
		}
		
		// Release the IDeckLinkDisplayMode object to prevent a leak
		displayMode->Release();

		displayModeNumber++;
	}
	
bail:
	// Ensure that the interfaces we obtained are released to prevent a memory leak
	if (displayModeIterator != NULL)
		displayModeIterator->Release();
	
	if (deckLinkOutput != NULL)
		deckLinkOutput->Release();
}

#endif /* HAVE_DECKLINK */

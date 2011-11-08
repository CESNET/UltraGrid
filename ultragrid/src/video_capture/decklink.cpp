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
#include "audio/audio.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#ifdef HAVE_DECKLINK		/* From config.h */

#include "video_capture/decklink.h"

#include "DeckLinkAPI.h" /* From DeckLink SDK */ 

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

#define MAX_DEVICES 4

#ifdef HAVE_MACOSX
#define STRING CFStringRef
#else
#define STRING const char *
#endif

// static int	device = 0; // use first BlackMagic device
// static int	mode = 5; // for Intensity
// static int	mode = 6; // for Decklink  6) HD 1080i 59.94; 1920 x 1080; 29.97 FPS 7) HD 1080i 60; 1920 x 1080; 30 FPS
static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

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
        void*   audioFrame;
        int     audioFrameSamples;
	int	first_time;
	struct  vidcap_decklink_state *s;
        int     i;
	
	void set_device_state(struct vidcap_decklink_state *state, int index);
	
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

struct device_state {
	IDeckLink*		deckLink;
	IDeckLinkInput*		deckLinkInput;
	VideoDelegate*		delegate;
        int                     index;
};

struct vidcap_decklink_state {
        struct device_state     state[MAX_DEVICES];
        int                     devices_cnt;
	int			mode;
	// void*			rtp_buffer;
	unsigned int		next_frame_time; // avarege time between frames
        struct video_frame     *frame;
        struct audio_frame      audio;
        const struct codec_info_t *c_info;
        

	pthread_mutex_t	 	lock;
	pthread_cond_t	 	boss_cv;
	int		 	boss_waiting;
        
        int                     frames;
        unsigned int            grab_audio:1; /* wheather we process audio or not */
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
			fprintf(stderr, "Frame received (#%d) - No input signal detected\n", s->frames);
		}
		else{
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, arrivedFrame->GetRowBytes() * arrivedFrame->GetHeight());
			
		}
	}

	arrivedFrame->GetBytes(&pixelFrame);
        
        if(audioPacket) {
                audioPacket->GetBytes(&audioFrame);
                s->audio.data_len = audioPacket->GetSampleFrameCount() * 2 * 2;
                memcpy(s->audio.data, audioFrame, s->audio.data_len);
        } else {
                audioFrame = NULL;
        }
                

	if(first_time){
		first_time = 0;
	}

	newFrameReady = 1; // The new frame is ready to grab
	
	if (s->boss_waiting) {
		pthread_cond_signal(&(s->boss_cv));
	}
	
// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&(s->lock));

        
	debug_msg("VideoInputFrameArrived - END\n"); /* TOREMOVE */

	return S_OK;
}

void VideoDelegate::set_device_state(struct vidcap_decklink_state *state, int index){
	s = state;
        i = index;
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
	printf("\t-t decklink:<device_index(indices)>:<mode>:<colorspace>\n");
	
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
		STRING		deviceNameString = NULL;
		const char *		deviceNameCString = NULL;
		
		// *** Print the model name of the DeckLink card
		result = deckLink->GetModelName((STRING *) &deviceNameString);
#ifdef HAVE_MACOSX
                deviceNameCString = (char *) malloc(128);
                CFStringGetCString(deviceNameString, (char *)deviceNameCString, 128, kCFStringEncodingMacRoman);
#else
                deviceNameCString = deviceNameString;
#endif
		if (result == S_OK)
		{
			printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
			free((void *)deviceNameCString);
#ifdef HAVE_MACOSX
			CFRelease(deviceNameString);
#endif
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

	if(!fmt || strcmp(fmt, "help") == 0) {
		decklink_help();
		return 0;
	}

	char *tmp;

	// choose device
	tmp = strtok(fmt, ":");
	if(!tmp) {
		fprintf(stderr, "Wrong config %s\n", fmt);
		return 0;
	} else {
                char *devices = strdup(tmp);
                char *ptr;
                char *saveptr;

                s->devices_cnt = 0;
                ptr = strtok_r(devices, ",", &saveptr);
                do {
                        s->state[s->devices_cnt].index = atoi(ptr);
                        ++s->devices_cnt;
                } while (ptr = strtok_r(NULL, ",", &saveptr));
                free (devices);
        }

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
                        break;
                    }
                }
        } else {
                int i;
                for(i=0; codec_info[i].name != NULL; i++) {
                    if(strcmp(codec_info[i].name, tmp) == 0) {
                         s->c_info = &codec_info[i];
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
vidcap_decklink_init(char *fmt, unsigned int flags)
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

        if(flags & VIDCAP_FLAG_ENABLE_AUDIO) {
                s->grab_audio = TRUE;
                s->audio.bps = 2;
                s->audio.sample_rate = 48000;
                s->audio.ch_count = 2;
                s->audio.data = (char *) malloc (48000 * 2 * 2);
        } else {
                s->grab_audio = FALSE;
        }

	// SET UP device and mode
	if(settings_init(s, fmt) == 0) {
		free(s);
		return NULL;
	}

	bool device_found[MAX_DEVICES];
        for(int i = 0; i < s->devices_cnt; ++i)
                device_found[i] = false;
                
        if (s->devices_cnt > 1) {
                double x_cnt = sqrt(s->devices_cnt);
                
                int x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                int y_count = s->devices_cnt / x_cnt;
                s->frame = vf_alloc(x_count, y_count);
                s->frame->aux = AUX_TILED;
        } else {
                s->frame = vf_alloc(1, 1);
                s->frame->aux = 0;
        }
    
        /* TODO: make sure that all devices are have compatible properties */
        for (int i = 0; i < s->devices_cnt; ++i)
        {
                int x_pos = i % s->frame->tiles[0].tile_info.x_count;
                int y_pos = i / s->frame->tiles[0].tile_info.x_count;
                struct tile * tile = tile_get(s->frame, x_pos, y_pos);
                dnum = 0;
                deckLink = NULL;
                // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
                deckLinkIterator = CreateDeckLinkIteratorInstance();
                if (deckLinkIterator == NULL)
                {
                        printf("A DeckLink iterator could not be created. The DeckLink drivers may not be installed.\n");
                        return NULL;
                }
                while (deckLinkIterator->Next(&deckLink) == S_OK)
                {
                        printf("%d\n", dnum);
                        if (s->state[i].index != dnum) {
                                dnum++;

                                // Release the IDeckLink instance when we've finished with it to prevent leaks
                                deckLink->Release();
                                deckLink = NULL;
                                continue;	
                        }

                        device_found[i] = true;
                        dnum++;

                        s->state[i].deckLink = deckLink;

                        STRING deviceNameString = NULL;
                        
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

                                s->state[i].deckLinkInput = deckLinkInput;

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
                        
                                        STRING displayModeString = NULL;

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

                                                tile->width = displayMode->GetWidth();
                                                tile->height = displayMode->GetHeight();
                                                s->frame->color_spec = s->c_info->codec;

                                                displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
                                                s->frame->fps = (double)frameRateScale / (double)frameRateDuration;
                                                s->next_frame_time = (int) (1000000 / s->frame->fps); // in microseconds
                                                switch(displayMode->GetFieldDominance()) {
                                                        case bmdLowerFieldFirst:
                                                        case bmdUpperFieldFirst:
                                                                s->frame->aux |= AUX_INTERLACED;
                                                                break;
                                                        case bmdProgressiveFrame:
                                                                s->frame->aux |= AUX_PROGRESSIVE;
                                                                break;
                                                        case bmdProgressiveSegmentedFrame:
                                                                s->frame->aux |= AUX_SF;
                                                                break;
                                                }

                                                debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeString,
                                                                tile->width, tile->height, s->frame->fps, s->next_frame_time); /* TOREMOVE */  

                                                deckLinkInput->StopStreams();

                                                printf("Enable video input: %s\n", displayModeString);

                                                result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, 0);
                                                if (result != S_OK)
                                                {
                                                        printf("You have required invalid video mode and pixel format combination.\n");
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
                            
                                                /*if (deckLinkConfiguration->SetVideoInputFormat(conn) == S_OK) {
                                                        printf("Input set to: %d\n", connection);
                                                }*/

                                                if(s->grab_audio == FALSE || 
                                                                i != 0)//TODO: figure out output from multiple streams
                                                        deckLinkInput->DisableAudioInput();
                                                else
                                                        deckLinkInput->EnableAudioInput(
                                                                bmdAudioSampleRate48kHz,
                                                                bmdAudioSampleType16bitInteger,
                                                                2);

                                                // set Callback which returns frames
                                                s->state[i].delegate = new VideoDelegate();
                                                s->state[i].delegate->set_device_state(s, i);
                                                deckLinkInput->SetCallback(s->state[i].delegate);

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
		deckLinkIterator->Release();

                tile->linesize = vc_get_linesize(tile->width, s->frame->color_spec);
                tile->data_len = tile->linesize * tile->height;
        }

        // init mutex
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        
        s->boss_waiting = FALSE;        	

	// check if any mode was found
        for (int i = 0; i < s->devices_cnt; ++i)
        {
                if (device_found[i] == false)
                {
                        printf("Device %d wasn't found.\n", s->state[i].index);
                        goto error;
                }
        }


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

        for (int i = 0; i < s->devices_cnt; ++i)
        {
		result = s->state[i].deckLinkInput->StopStreams();
		if (result != S_OK)
		{
			printf("Could not stop stream: %08x\n", result);
		}

		if(s->state[i].deckLinkInput != NULL)
		{
			s->state[i].deckLinkInput->Release();
			s->state[i].deckLinkInput = NULL;
		}

		if(s->state[i].deckLink != NULL)
		{
			s->state[i].deckLink->Release();
			s->state[i].deckLink = NULL;
		}

		free(s);
	}
}

struct video_frame *
vidcap_decklink_grab(void *state, struct audio_frame **audio)
{
	debug_msg("vidcap_decklink_grab\n"); /* TO REMOVE */

	struct vidcap_decklink_state 	*s = (struct vidcap_decklink_state *) state;
	struct video_frame		*vf;
        int                             tiles_total = 0;
        int                             i;

	HRESULT	result;
	
	int		rc;
	struct timespec	ts;
	struct timeval	tp;

	int timeout = 0;

	pthread_mutex_lock(&s->lock);
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	debug_msg("vidcap_decklink_grab - before while\n"); /* TOREMOVE */

        for (i = 0; i < s->devices_cnt; ++i)
                if(s->state[i].delegate->newFrameReady)
                        tiles_total++;

        while(tiles_total != s->devices_cnt) {
	//while (!s->state[0].delegate->newFrameReady) {
                rc = 0;
		debug_msg("vidcap_decklink_grab - pthread_cond_timedwait\n"); /* TOREMOVE */

		// get time for timeout
		gettimeofday(&tp, NULL);

		/* Convert from timeval to timespec */
		ts.tv_sec  = tp.tv_sec;
		ts.tv_nsec = tp.tv_usec * 1000;
		ts.tv_nsec += 2 * s->next_frame_time * 1000;
		// make it correct
		ts.tv_sec += ts.tv_nsec / 1000000000;
		ts.tv_nsec = ts.tv_nsec % 1000000000;

		debug_msg("vidcap_decklink_grab - current time: %02d:%03d\n",tp.tv_sec, tp.tv_usec/1000); /* TOREMOVE */

                while(rc == 0  /*  not timeout AND */
                                && tiles_total != s->devices_cnt) { /* not all tiles */
                        s->boss_waiting = TRUE;
                        rc = pthread_cond_timedwait(&s->boss_cv, &s->lock, &ts);
                        s->boss_waiting = FALSE;
                        // recompute tiles count
                        tiles_total = 0;
                        for (i = 0; i < s->devices_cnt; ++i)
                                if(s->state[i].delegate->newFrameReady) {
                                        tiles_total++;
                                }
                }
                debug_msg("vidcap_decklink_grab - AFTER pthread_cond_timedwait - %d tiles\n", tiles_total); /* TOREMOVE */

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

                        //if((!s->state[i].delegate->first_time) || (should_exit)){
                        if(should_exit){
                                //s->state[i].delegate->newFrameReady = 1;
                                timeout = 1;
                                break;
                        }else{
                                // wait half of timeout
                                usleep(s->next_frame_time);
                        }
                        tiles_total = 0;
                } 
	}

        /*cleanup newframe flag */
        for (i = 0; i < s->devices_cnt; ++i)
                s->state[i].delegate->newFrameReady = 0;

// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	pthread_mutex_unlock(&s->lock);

        /* count returned tiles */
        int count = 0;
        for (i = 0; i < s->devices_cnt; ++i) {
                if (s->state[i].delegate->pixelFrame != NULL) {
                        s->frame->tiles[i].data = (char*)s->state[i].delegate->pixelFrame;
                        if(s->c_info->codec == RGBA) {
                            vc_copylineRGBA((unsigned char*) s->frame->tiles[i].data,
                                        (unsigned char*)s->frame->tiles[i].data,
                                        s->frame->tiles[i].data_len, 16, 8, 0);
                        }
                        ++count;
                } else
                        break;
        }
        if (count == s->devices_cnt) {
                s->frames++;
                
                if(s->state[0].delegate->audioFrame != NULL) {
                        *audio = &s->audio;
                } else {
                        *audio = NULL;
                }

                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, t0);	
                if (seconds >= 5) {
                        float fps  = s->frames / seconds;
                        fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                        t0 = t;
                        s->frames = 0;
                }

                return s->frame;
        }
        
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
		STRING			displayModeString = NULL;
                const char *displayModeCString;
		
		result = displayMode->GetName((STRING *) &displayModeString);
#ifdef HAVE_MACOSX
                displayModeCString = (char *) malloc(128);
                CFStringGetCString(displayModeString, (char *) displayModeCString, 128, kCFStringEncodingMacRoman);
#else
                displayModeCString = displayModeString;
#endif


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
			printf("%d.) %-20s \t %d x %d \t %g FPS\n",displayModeNumber, displayModeCString, modeWidth, modeHeight, (double)frameRateScale / (double)frameRateDuration);
#ifdef HAVE_MACOSX
                        CFRelease(displayModeString);
#endif
			free((void *)displayModeCString);
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

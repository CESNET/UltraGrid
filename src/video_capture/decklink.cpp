/*
 * FILE:    video_capture/decklink.cpp
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

#define MODULE_NAME "[Decklink capture] "

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
#include "audio/utils.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#ifdef WIN32
#include <objbase.h>
#endif

#ifdef HAVE_DECKLINK		/* From config.h */

#include "video_capture/decklink.h"

#ifdef WIN32
#include "DeckLinkAPI_h.h" /* From DeckLink SDK */ 
#else
#include "DeckLinkAPI.h" /* From DeckLink SDK */ 
#endif
#include "DeckLinkAPIVersion.h" /* From DeckLink SDK */ 

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

#define MAX_DEVICES 4

#ifdef HAVE_MACOSX
#define STRING CFStringRef
#elif defined WIN32
#define STRING BSTR
#else
#define STRING const char *
#endif

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

// static int	device = 0; // use first BlackMagic device
// static int	mode = 5; // for Intensity
// static int	mode = 6; // for Decklink  6) HD 1080i 59.94; 1920 x 1080; 29.97 FPS 7) HD 1080i 60; 1920 x 1080; 30 FPS
//static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

static volatile bool should_exit = false;

struct timeval t, t0;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

class VideoDelegate;

struct device_state {
	IDeckLink*		deckLink;
	IDeckLinkInput*		deckLinkInput;
	VideoDelegate*		delegate;
	IDeckLinkConfiguration*		deckLinkConfiguration;
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
        BMDVideoInputFlags flags;
        

	pthread_mutex_t	 	lock;
	pthread_cond_t	 	boss_cv;
	int		 	boss_waiting;
        
        int                     frames;
        unsigned int            grab_audio:1; /* wheather we process audio or not */
        unsigned int            stereo:1; /* for eg. DeckLink HD Extreme, Quad doesn't set this !!! */
        unsigned int            use_timecode:1; /* use timecode when grabbing from multiple inputs */
        unsigned int            autodetect_mode:1;

        BMDVideoConnection      connection;
};

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf);




class VideoDelegate : public IDeckLinkInputCallback
{
private:
	int32_t mRefCount;
	double  lastTime;
    
public:
	int	newFrameReady;
        IDeckLinkVideoFrame *rightEyeFrame;
	void*	pixelFrame;
	void*	pixelFrameRight;
        void*   audioFrame;
        int     audioFrameSamples;
	int	first_time;
	struct  vidcap_decklink_state *s;
        int     i;
        IDeckLinkTimecode      *timecode;
	
	void set_device_state(struct vidcap_decklink_state *state, int index);
	
	VideoDelegate () {
		newFrameReady = 0;
		first_time = 1;
                rightEyeFrame = NULL;
		s = NULL;
	};
        
        ~VideoDelegate () {
		if(rightEyeFrame)
                        rightEyeFrame->Release();
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
	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode* mode, BMDDetectedVideoInputFormatFlags flags)
	{
                codec_t codec;
                BMDPixelFormat pf;
                HRESULT result;

                printf("[DeckLink] Format change detected.\n");

                pthread_mutex_lock(&(s->lock));
                switch(flags) {
                        case bmdDetectedVideoInputYCbCr422:
                                codec = Vuy2;
                                break;
                        case bmdDetectedVideoInputRGB444:
                                codec = RGBA;
                                break;
                }
                int i;
                for(i=0; codec_info[i].name != NULL; i++) {
                    if(codec_info[i].codec == codec) {
                        s->c_info = &codec_info[i];
                        break;
                    }
                }
                IDeckLinkInput *deckLinkInput = s->state[this->i].deckLinkInput;
                deckLinkInput->DisableVideoInput();
                deckLinkInput->StopStreams();
                deckLinkInput->FlushStreams();
                result = set_display_mode_properties(s, vf_get_tile(s->frame, this->i), mode, /* out */ &pf);
                if(result == S_OK) {
                        result = deckLinkInput->EnableVideoInput(mode->GetDisplayMode(), pf, s->flags);
                        if(s->grab_audio == FALSE || 
                                        this->i != 0) { //TODO: figure out output from multiple streams
                                deckLinkInput->DisableAudioInput();
                        } else {
                                deckLinkInput->EnableAudioInput(
                                        bmdAudioSampleRate48kHz,
                                        bmdAudioSampleType16bitInteger,
                                        audio_capture_channels == 1 ? 2 : audio_capture_channels); // BMD isn't able to grab single channel
                        }
                        //deckLinkInput->SetCallback(s->state[i].delegate);
                        deckLinkInput->StartStreams();
                }
                pthread_mutex_unlock(&(s->lock));

                return result;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);    
};


/* DeckLink SDK objects */

void
print_output_modes (IDeckLink* deckLink);
static int blackmagic_api_version_check(STRING *current_version);

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *arrivedFrame, IDeckLinkAudioInputPacket *audioPacket)
{
        bool noSignal = false;
	// Video

	pthread_mutex_lock(&(s->lock));
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	if(arrivedFrame)
	{
		if (arrivedFrame->GetFlags() & bmdFrameHasNoInputSource)
		{
			fprintf(stderr, "Frame received (#%d) - No input signal detected\n", s->frames);
                        noSignal = true;
		}
		else{
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, arrivedFrame->GetRowBytes() * arrivedFrame->GetHeight());
			
		}
	}

	arrivedFrame->GetBytes(&pixelFrame);
        
        if(audioPacket) {
                audioPacket->GetBytes(&audioFrame);
                if(audio_capture_channels == 1) { // ther are actually 2 channels grabbed
                        demux_channel(s->audio.data, (char *) audioFrame, 2, audioPacket->GetSampleFrameCount() * 2 /* channels */ * 2,
                                        2, /* channels (originally( */
                                        0 /* we want first channel */
                                                );
                        s->audio.data_len = audioPacket->GetSampleFrameCount() * 1 * 2;
                } else {
                        s->audio.data_len = audioPacket->GetSampleFrameCount() * audio_capture_channels * 2;
                        memcpy(s->audio.data, audioFrame, s->audio.data_len);
                }
        } else {
                audioFrame = NULL;
        }
        
        if(rightEyeFrame)
                rightEyeFrame->Release();
        pixelFrameRight = NULL;
        rightEyeFrame = NULL;
        
        if(s->stereo) {
                IDeckLinkVideoFrame3DExtensions *rightEye;
                HRESULT result;
                result = arrivedFrame->QueryInterface(IID_IDeckLinkVideoFrame3DExtensions, (void **)&rightEye);
                
                if (result == S_OK) { 
                        result = rightEye->GetFrameForRightEye(&rightEyeFrame);
                                         
                        if(result == S_OK) {
                                if (rightEyeFrame->GetFlags() & bmdFrameHasNoInputSource)
                                {
                                        fprintf(stderr, "Right Eye Frame received (#%d) - No input signal detected\n", s->frames);
                                }
                                rightEyeFrame->GetBytes(&pixelFrameRight);
                        }
                }
                rightEye->Release();
                if(!pixelFrameRight) {
                        fprintf(stderr, "[DeckLink] Sending right eye error.\n");
                }
        }

        timecode = NULL;
        if(s->use_timecode && !noSignal) {
                HRESULT result;
                result = arrivedFrame->GetTimecode(bmdTimecodeRP188Any, &timecode);
                if(result != S_OK) {
                        fprintf(stderr, "Failed to acquire timecode from stream. Disabling sync.\n");
                        s->use_timecode = FALSE;
                }
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

static int blackmagic_api_version_check(STRING *current_version)
{
        int ret = TRUE;
        *current_version = NULL;

        IDeckLinkAPIInformation *APIInformation = NULL;

#ifdef WIN32
	HRESULT result;
	result = CoCreateInstance(CLSID_CDeckLinkAPIInformation, NULL, CLSCTX_ALL,
		IID_IDeckLinkAPIInformation, (void **) &APIInformation);
	if(FAILED(result))
#else
	APIInformation = CreateDeckLinkAPIInformationInstance();
        if(APIInformation == NULL)
#endif
	{
                return FALSE;
        }
        int64_t value;
        HRESULT res;
        res = APIInformation->GetInt(BMDDeckLinkAPIVersion, &value);
        if(res != S_OK) {
                APIInformation->Release();
                return FALSE;
        }

        if(BLACKMAGIC_DECKLINK_API_VERSION > value) { // this is safe comparision, for internal structure please see SDK documentation
                APIInformation->GetString(BMDDeckLinkAPIVersion, current_version);
                ret  = FALSE;
        }


        APIInformation->Release();
        return ret;
}

/* HELP */
int
decklink_help()
{
	IDeckLinkIterator*		deckLinkIterator;
	IDeckLink*			deckLink;
	int				numDevices = 0;
	HRESULT				result;

	printf("\nDecklink options:\n");
	printf("\t-t decklink[:<device_index(indices)>[:<mode>:<colorspace>[:3D][:timecode][:connection=<input>]]\n");
	printf("\t\t(You can ommit device index, mode and color space provided that your cards supports format autodetection.)\n");
	
	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
#ifdef WIN32
	result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
		IID_IDeckLinkIterator, (void **) &deckLinkIterator);
	if (FAILED(result))
#else
	deckLinkIterator = CreateDeckLinkIteratorInstance();
	if (deckLinkIterator == NULL)
#endif
	{
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
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
#elif defined WIN32
                deviceNameCString = (char *) malloc(128);
		wcstombs((char *) deviceNameCString, deviceNameString, 128);
#else
                deviceNameCString = deviceNameString;
#endif
		if (result == S_OK)
		{
			printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
#ifdef HAVE_MACOSX
			CFRelease(deviceNameString);
#endif
                        free((void *)deviceNameCString);
		}
		
		// Increment the total number of DeckLink cards found
		numDevices++;
	
		// ** List the video output display modes supported by the card
		print_output_modes(deckLink);

                IDeckLinkAttributes *deckLinkAttributes;

                result = deckLink->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes);
                if (result != S_OK)
                {
                        printf("Could not query device attributes.\n");
                } else {
                        int64_t connections;
                        if(deckLinkAttributes->GetInt(BMDDeckLinkVideoInputConnections, &connections) != S_OK) {
                                fprintf(stderr, "[DeckLink] Could not get connections.\n");
                        } else {
                                printf("\n");
                                printf("Connection can be one of following (not required):\n");
                                if (connections & bmdVideoConnectionSDI)
                                        printf("\tSDI\n");
                                if (connections & bmdVideoConnectionHDMI)
                                        printf("\tHDMI\n");
                                if (connections & bmdVideoConnectionOpticalSDI)
                                        printf("\tOpticalSDI\n");
                                if (connections & bmdVideoConnectionComponent)
                                        printf("\tComponent\n");
                                if (connections & bmdVideoConnectionComposite)
                                        printf("\tComposite\n");
                                if (connections & bmdVideoConnectionSVideo)
                                        printf("\tSVideo\n");
                        }
                }
				
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
                printf("\n\n");
                printf("Available Colorspaces:\n");
                printf("\t2vuy\n");
                printf("\tv210\n");
                printf("\tRGBA\n");
                printf("\tR10k\n");
        }
	printf("\n");
        
        printf("3D\n");
        printf("\tUse this to capture 3D from supported card (eg. DeckLink HD 3D Extreme).\n");
        printf("\tDo not use it for eg. Quad or Duo. Availability of the mode is indicated\n");
        printf("\tin video format listing above (\"supports 3D\").\n");

	printf("\n");
        printf("timecode\n");
        printf("\tTry to synchronize inputs based on timecode (for multiple inputs, eg. tiled 4K)\n");


	return 1;
}

/* SETTINGS */

int
settings_init(void *state, char *fmt)
{
	struct vidcap_decklink_state *s = (struct vidcap_decklink_state *) state;

        int i;
        for(i=0; codec_info[i].name != NULL; i++) {
            if(codec_info[i].codec == Vuy2) {
                s->c_info = &codec_info[i];
                break;
            }
        }

        if(fmt) {
		char *save_ptr_top = NULL;
                if(strcmp(fmt, "help") == 0) {
                        decklink_help();
                        return 0;
                }

                char *tmp;

                // choose device
                tmp = strtok_r(fmt, ":", &save_ptr_top);
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
                        } while ((ptr = strtok_r(NULL, ",", &saveptr)));
                        free (devices);
                }

                // choose mode
                tmp = strtok_r(NULL, ":", &save_ptr_top);
                if(tmp) {
                        s->mode = atoi(tmp);

                        tmp = strtok_r(NULL, ":", &save_ptr_top);
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
                        while((tmp = strtok_r(NULL, ":", &save_ptr_top))) {
                                if(strcasecmp(tmp, "3D") == 0) {
                                        s->stereo = TRUE;
                                } else if(strcasecmp(tmp, "timecode") == 0) {
                                        s->use_timecode = TRUE;
                                } else if(strncasecmp(tmp, "connection=", strlen("connection=")) == 0) {
                                        char *connection = tmp + strlen("connection=");
                                        if(strcasecmp(connection, "SDI") == 0)
                                                s->connection = bmdVideoConnectionSDI;
                                        else if(strcasecmp(connection, "HDMI") == 0)
                                                s->connection = bmdVideoConnectionHDMI;
                                        else if(strcasecmp(connection, "OpticalSDI") == 0)
                                                s->connection = bmdVideoConnectionOpticalSDI;
                                        else if(strcasecmp(connection, "Component") == 0)
                                                s->connection = bmdVideoConnectionComponent;
                                        else if(strcasecmp(connection, "Composite") == 0)
                                                s->connection = bmdVideoConnectionComposite;
                                        else if(strcasecmp(connection, "SVIdeo") == 0)
                                                s->connection = bmdVideoConnectionSVideo;
                                        else {
                                                fprintf(stderr, "[DeckLink] Unrecognized connection %s.\n", connection);
                                                return 0;
                                        }
                                } else {
                                        fprintf(stderr, "[DeckLink] Warning, unrecognized trailing options in init string: %s", tmp);
                                }
                        }
                } else {
                        s->autodetect_mode = TRUE;
                        printf("[DeckLink] Trying to autodetect format.\n");
                        s->mode = 0;
                }
        } else {
                printf("[DeckLink] Trying to autodetect format.\n");
                s->mode = 0;
                s->autodetect_mode = TRUE;
                s->devices_cnt = 1;
                s->state[s->devices_cnt].index = 0;
                printf("DeckLink] Auto-choosen device 0.\n");
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

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf)
{
        STRING displayModeString = NULL;
        const char *displayModeCString;
        HRESULT result;

        result = displayMode->GetName(&displayModeString);
        if (result == S_OK)
        {
                switch(s->c_info->codec) {
                  case RGBA:
                        *pf = bmdFormat8BitBGRA;
                        break;
                  case Vuy2:
                        *pf = bmdFormat8BitYUV;
                        break;
                  case R10k:
                        *pf = bmdFormat10BitRGB;
                        break;
                  case v210:
                        *pf = bmdFormat10BitYUV;
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
                                s->frame->interlacing = INTERLACED_MERGED;
                                break;
                        case bmdProgressiveFrame:
                                s->frame->interlacing = PROGRESSIVE;
                                break;
                        case bmdProgressiveSegmentedFrame:
                                s->frame->interlacing = SEGMENTED_FRAME;
                                break;
                }

                debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeString,
                                tile->width, tile->height, s->frame->fps, s->next_frame_time); /* TOREMOVE */  
#ifdef HAVE_MACOSX
                displayModeCString = (char *) malloc(128);
                CFStringGetCString(displayModeString, (char *) displayModeCString, 128, kCFStringEncodingMacRoman);
#elif defined WIN32
                displayModeCString = (char *) malloc(128);
		wcstombs((char *) displayModeCString, displayModeString, 128);
#else
                displayModeCString = displayModeString;
#endif
                printf("Enable video input: %s\n", displayModeCString);
#ifdef HAVE_MACOSX
                        CFRelease(displayModeString);
#endif
                        free((void *)displayModeCString);
        }

        tile->linesize = vc_get_linesize(tile->width, s->frame->color_spec);
        tile->data_len = tile->linesize * tile->height;

        if(s->stereo) {
                s->frame->tiles[1].width = s->frame->tiles[0].width;
                s->frame->tiles[1].height = s->frame->tiles[0].height;
                s->frame->tiles[1].linesize = s->frame->tiles[0].linesize;
                s->frame->tiles[1].data_len = s->frame->tiles[0].data_len;
        }

        return result;
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
        BMDAudioConnection              audioConnection;

#ifdef WIN32
	// Initialize COM on this thread
	result = CoInitialize(NULL);
	if(FAILED(result)) {
		fprintf(stderr, "Initialization of COM failed - result = "
				"08x.\n", result);
		return NULL;
	}
#endif


        STRING current_version; 
        if(!blackmagic_api_version_check(&current_version)) {
		fprintf(stderr, "\nThe DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled against DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                fprintf(stderr, "Vendor download page is http://http://www.blackmagic-design.com/support/ \n");
                if(current_version) {
                        const char *currentVersionCString;
#ifdef HAVE_MACOSX
                        currentVersionCString = (char *) malloc(128);
                        CFStringGetCString(current_version, (char *) currentVersionCString, 128, kCFStringEncodingMacRoman);
#elif defined WIN32
                        currentVersionCString = (char *) malloc(128);
			wcstombs((char *) currentVersionCString, current_version, 128);
#else
                        currentVersionCString = current_version;
#endif
                        fprintf(stderr, "Currently installed version is: %s\n", currentVersionCString);
#ifdef HAVE_MACOSX
                        CFRelease(current_version);
#endif
                        free((void *)currentVersionCString);
                } else {
                        fprintf(stderr, "No installed drivers detected\n");
                }
                fprintf(stderr, "\n");
                return NULL;
        }


	s = (struct vidcap_decklink_state *) calloc(1, sizeof(struct vidcap_decklink_state));
	if (s == NULL) {
		//printf("Unable to allocate DeckLink state\n",fps);
		printf("Unable to allocate DeckLink state\n");
		return NULL;
	}

        gettimeofday(&t0, NULL);

        s->stereo = FALSE;
        s->use_timecode = FALSE;
        s->autodetect_mode = FALSE;
        s->connection = (BMDVideoConnection) 0;
        s->flags = 0;

	// SET UP device and mode
	if(settings_init(s, fmt) == 0) {
		free(s);
		return NULL;
	}

        if(flags & (VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG)) {
                s->grab_audio = TRUE;
                switch(flags & (VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG)) {
                        case VIDCAP_FLAG_AUDIO_EMBEDDED:
                                audioConnection = bmdAudioConnectionEmbedded;
                                break;
                        case VIDCAP_FLAG_AUDIO_AESEBU:
                                audioConnection = bmdAudioConnectionAESEBU;
                                break;
                        case VIDCAP_FLAG_AUDIO_ANALOG:
                                audioConnection = bmdAudioConnectionAnalog;
                                break;
                        default:
                                fprintf(stderr, "[Decklink capture] Unexpected audio flag encountered.\n");
                                abort();
                }
                s->audio.bps = 2;
                s->audio.sample_rate = 48000;
                s->audio.ch_count = audio_capture_channels;
                s->audio.data = (char *) malloc (48000 * audio_capture_channels * 2);
        } else {
                s->grab_audio = FALSE;
        }

	bool device_found[MAX_DEVICES];
        for(int i = 0; i < s->devices_cnt; ++i)
                device_found[i] = false;
                
        if(s->stereo) {
                s->frame = vf_alloc(2);
                if (s->devices_cnt > 1) {
                        fprintf(stderr, "[DeckLink] Passed more than one device while setting 3D mode. "
                                        "In this mode, only one device needs to be passed.");
                        free(s);
                        return NULL;
                }
        } else {
                s->frame = vf_alloc(s->devices_cnt);
        }
    
        /* TODO: make sure that all devices are have compatible properties */
        for (int i = 0; i < s->devices_cnt; ++i)
        {
                struct tile * tile = vf_get_tile(s->frame, i);
                dnum = 0;
                deckLink = NULL;
                // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
#ifdef WIN32
		result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
			IID_IDeckLinkIterator, (void **) &deckLinkIterator);
		if (FAILED(result))
#else
		deckLinkIterator = CreateDeckLinkIteratorInstance();
                if (deckLinkIterator == NULL)
#endif
                {
                        fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
                        fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                        BLACKMAGIC_DECKLINK_API_VERSION_STRING);
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
                        const char* deviceNameCString = NULL;
                        
                        // Print the model name of the DeckLink card
                        result = deckLink->GetModelName(&deviceNameString);
                        if (result == S_OK)
                        {	
#ifdef HAVE_MACOSX
                                deviceNameCString = (char *) malloc(128);
                                CFStringGetCString(deviceNameString, (char *) deviceNameCString, 128, kCFStringEncodingMacRoman);
#elif defined WIN32
                                deviceNameCString = (char *) malloc(128);
				wcstombs((char *) deviceNameCString, deviceNameString, 128);
#else
                                deviceNameCString = deviceNameString;
#endif
                                printf("Using device [%s]\n", deviceNameCString);
#ifdef HAVE_MACOSX
                                CFRelease(deviceNameString);
#endif
                                free((void *) deviceNameCString);

                                // Query the DeckLink for its configuration interface
                                result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
                                if (result != S_OK)
                                {
                                        printf("Could not obtain the IDeckLinkInput interface - result = %08x\n", (int) result);
                                        goto error;
                                }

                                s->state[i].deckLinkInput = deckLinkInput;

                                // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
                                result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
                                if (result != S_OK)
                                {
                                        printf("Could not obtain the video input display mode iterator - result = %08x\n", (int) result);
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
                                        break;
                                }

                                printf("The desired display mode is supported: %d\n",s->mode);  
                
                                BMDPixelFormat pf;

                                if(set_display_mode_properties(s, tile, displayMode, &pf) == S_OK) {
                                        IDeckLinkAttributes *deckLinkAttributes;
                                        deckLinkInput->StopStreams();

                                       result = deckLinkInput->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes);
                                        if (result != S_OK)
                                        {
                                                printf("Could not query device attributes.\n");
                                                printf("Could not enable video input: %08x\n", (int) result);
                                                goto error;
                                        }

                                        if(s->autodetect_mode) {
#ifdef WIN32
                                                BOOL autodetection;
#else
                                                bool autodetection;
#endif
                                                if(deckLinkAttributes->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &autodetection) != S_OK) {
                                                        fprintf(stderr, "[DeckLink] Could not verify if device supports autodetection.\n");
                                                        goto error;
                                                }
                                                if(autodetection == false) {
                                                        fprintf(stderr, "[DeckLink] Device doesn't support format autodetection, you must set it manually.\n");
                                                        goto error;
                                                }
                                                s->flags |=  bmdVideoInputEnableFormatDetection;

                                        }

                                        if(s->stereo) {
                                                s->flags |= bmdVideoInputDualStream3D;
                                        }
                                        result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, s->flags);
                                        if (result != S_OK)
                                        {
                                                printf("You have required invalid video mode and pixel format combination.\n");
                                                printf("Could not enable video input: %08x\n", (int) result);
                                                goto error;
                                        }

                                        // Query the DeckLink for its configuration interface
                                        result = deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
                                        if (result != S_OK)
                                        {
                                                printf("Could not obtain the IDeckLinkConfiguration interface: %08x\n", (int) result);
                                                goto error;
                                        }

                                        s->state[i].deckLinkConfiguration = deckLinkConfiguration;

                                        if(s->connection) {
                                                if (deckLinkConfiguration->SetInt(bmdDeckLinkConfigVideoInputConnection,
                                                                        s->connection) == S_OK) {
                                                        printf("Input set to: %d\n", s->connection);
                                                }
                                        }

                                        if(s->grab_audio == FALSE || 
                                                        i != 0) { //TODO: figure out output from multiple streams
                                                deckLinkInput->DisableAudioInput();
                                        } else {
                                                if (deckLinkConfiguration->SetInt(bmdDeckLinkConfigAudioInputConnection,
                                                                        audioConnection) == S_OK) {
                                                        printf("[Decklink capture] Audio input set to: ");
                                                        switch(audioConnection) {
                                                                case bmdAudioConnectionEmbedded:
                                                                        printf("embedded");
                                                                        break;
                                                                case bmdAudioConnectionAESEBU:
                                                                        printf("AES/EBU");
                                                                        break;
                                                                case bmdAudioConnectionAnalog:
                                                                        printf("analog");
                                                                        break;
                                                        }
                                                        printf(".\n");
                                                } else {
                                                        fprintf(stderr, "[Decklink capture] Unable to set audio input!!! Please check if it is OK. Continuing anyway.\n");

                                                }
                                                if(audio_capture_channels != 1 &&
                                                                audio_capture_channels != 2 &&
                                                                audio_capture_channels != 8 &&
                                                                audio_capture_channels != 16) {
                                                        fprintf(stderr, "[DeckLink] Decklink cannot grab %d audio channels. "
                                                                        "Only 1, 2, 8 or 16 are poosible.", audio_capture_channels);
                                                        goto error;
                                                }
                                                deckLinkInput->EnableAudioInput(
                                                        bmdAudioSampleRate48kHz,
                                                        bmdAudioSampleType16bitInteger,
                                                        audio_capture_channels == 1 ? 2 : audio_capture_channels);
                                        }

                                        // set Callback which returns frames
                                        s->state[i].delegate = new VideoDelegate();
                                        s->state[i].delegate->set_device_state(s, i);
                                        deckLinkInput->SetCallback(s->state[i].delegate);

                                        // Start streaming
                                        printf("Start capture\n");
                                        result = deckLinkInput->StartStreams();
                                        if (result != S_OK)
                                        {
                                                printf("Could not start stream: %08x\n", (int) result);
                                                goto error;
                                        }

                                }else{
                                        printf("Could not : %08x\n", (int) result);
                                        goto error;
                                }

                                displayMode->Release();
                                displayMode = NULL;

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
vidcap_decklink_finish(void *state)
{
        UNUSED(state);
        should_exit = true;
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
		if (result != S_OK) {
			fprintf(stderr, MODULE_NAME "Could not stop stream: %08x\n", (int) result);
		}

                if(s->grab_audio && i == 0) {
                        result = s->state[i].deckLinkInput->DisableAudioInput();
                        if (result != S_OK) {
                                fprintf(stderr, MODULE_NAME "Could disable audio input: %08x\n", (int) result);
                        }
                }
		result = s->state[i].deckLinkInput->DisableVideoInput();
                if (result != S_OK) {
                        fprintf(stderr, MODULE_NAME "Could disable video input: %08x\n", (int) result);
                }

		if(s->state[i].deckLinkConfiguration != NULL) {
			s->state[i].deckLinkConfiguration->Release();
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
	}

        vf_free(s->frame);
        free(s);
}

/*  lock needs to be hold during all function call */
int nr_frames(struct vidcap_decklink_state *s) {
        BMDTimecodeBCD max_timecode = 0u;
        int tiles_total = 0;
        int i;

        if(s->use_timecode) {
                for (i = 0; i < s->devices_cnt; ++i) {
                        if(s->state[i].delegate->newFrameReady) {
                                BMDTimecodeBCD timecode;
                                if(s->state[i].delegate->timecode)  {
                                        timecode = s->state[i].delegate->timecode->GetBCD();
                                        if(timecode > max_timecode) {
                                                max_timecode = timecode;
                                        }
                                } else {
                                        fprintf(stderr, "[DeckLink] No timecode found.\n");
                                        break;
                                }
                        }
                }
        }

        for (i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].delegate->newFrameReady) {
                        if(s->use_timecode) {
                                if(s->state[i].delegate->timecode && s->state[i].delegate->timecode->GetBCD () != max_timecode) {
                                        s->state[i].delegate->newFrameReady = FALSE;
                                } else {
                                        tiles_total++;
                                }
                        } else {
                                tiles_total++;
                        }
                }
        }
        return tiles_total;
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

        tiles_total = nr_frames(s);

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
                        tiles_total = nr_frames(s);
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
        if(s->stereo) {
                if (s->state[0].delegate->pixelFrame != NULL &&
                                s->state[0].delegate->pixelFrameRight != NULL) {
                        s->frame->tiles[0].data = (char*)s->state[0].delegate->pixelFrame;
                        if(s->c_info->codec == RGBA) {
                            vc_copylineRGBA((unsigned char*) s->frame->tiles[0].data,
                                        (unsigned char*)s->frame->tiles[0].data,
                                        s->frame->tiles[i].data_len, 16, 8, 0);
                        }
                        s->frame->tiles[1].data = (char*)s->state[0].delegate->pixelFrameRight;
                        if(s->c_info->codec == RGBA) {
                            vc_copylineRGBA((unsigned char*) s->frame->tiles[1].data,
                                        (unsigned char*)s->frame->tiles[1].data,
                                        s->frame->tiles[i].data_len, 16, 8, 0);
                        }
                        ++count;
                } // else count == 0 -> return NULL
        } else {
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
                        fprintf(stderr, "[Decklink capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
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
		fprintf(stderr, "Could not obtain the IDeckLinkOutput interface - result = %08x\n", (int) result);
		goto bail;
	}
	
	// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on output
	result = deckLinkOutput->GetDisplayModeIterator(&displayModeIterator);
	if (result != S_OK)
	{
		fprintf(stderr, "Could not obtain the video output display mode iterator - result = %08x\n", (int) result);
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
#elif defined WIN32
                displayModeCString = (char *) malloc(128);
		wcstombs((char *) displayModeCString, displayModeString, 128);
#else
                displayModeCString = displayModeString;
#endif


		if (result == S_OK)
		{
			char			modeName[64];
			int				modeWidth;
			int				modeHeight;
                        BMDDisplayModeFlags             flags;
			BMDTimeValue	frameRateDuration;
			BMDTimeScale	frameRateScale;
			
			
			// Obtain the display mode's properties
                        flags = displayMode->GetFlags();
			modeWidth = displayMode->GetWidth();
			modeHeight = displayMode->GetHeight();
			displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
			printf("%d.) %-20s \t %d x %d \t %2.2f FPS%s\n",displayModeNumber, displayModeCString,
                                        modeWidth, modeHeight, (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (flags & bmdDisplayModeSupports3D ? "\t (supports 3D)" : ""));
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

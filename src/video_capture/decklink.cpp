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

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "tv.h"

#include "debug.h"
#include "video.h"
#include "video_capture.h"
#include "audio/audio.h"
#include "audio/utils.h"

#ifdef WIN32
#include <objbase.h>
#endif

#ifdef HAVE_DECKLINK		/* From config.h */

#include "blackmagic_common.h"
#include "video_capture/decklink.h"

#ifdef WIN32
#include "DeckLinkAPI_h.h" /* From DeckLink SDK */
#else
#include "DeckLinkAPI.h" /* From DeckLink SDK */
#endif

#include "DeckLinkAPIVersion.h" /* From DeckLink SDK */

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <string>

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

#define MAX_DEVICES 4

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

using namespace std;
using namespace std::chrono;

// static int	device = 0; // use first BlackMagic device
// static int	mode = 5; // for Intensity
// static int	mode = 6; // for Decklink  6) HD 1080i 59.94; 1920 x 1080; 29.97 FPS 7) HD 1080i 60; 1920 x 1080; 30 FPS
//static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

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
	string			mode;
	// void*			rtp_buffer;
	unsigned int		next_frame_time; // avarege time between frames
        struct video_frame     *frame;
        struct audio_frame      audio;
        codec_t                 codec;
        BMDVideoInputFlags flags;

        mutex                   lock;
	condition_variable      boss_cv;

        int                     frames;
        unsigned int            grab_audio:1; /* wheather we process audio or not */
        unsigned int            stereo:1; /* for eg. DeckLink HD Extreme, Quad doesn't set this !!! */
        unsigned int            use_timecode:1; /* use timecode when grabbing from multiple inputs */
        unsigned int            autodetect_mode:1;

        BMDVideoConnection      connection;
        int                     audio_consumer_levels; ///< 0 false, 1 true, -1 default

        struct timeval          t0;
};

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf);
static void cleanup_common(struct vidcap_decklink_state *s);

class VideoDelegate : public IDeckLinkInputCallback
{
private:
	int32_t mRefCount;

public:
        int	newFrameReady;
        IDeckLinkVideoFrame *rightEyeFrame;
	void*	pixelFrame;
	void*	pixelFrameRight;
        void*   audioFrame;
        int     audioFrameSamples;
	struct  vidcap_decklink_state *s;
        int     i;
        IDeckLinkTimecode      *timecode;
	
	void set_device_state(struct vidcap_decklink_state *state, int index);
	
        ~VideoDelegate () {
		if(rightEyeFrame)
                        rightEyeFrame->Release();
	};

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID *) { return E_NOINTERFACE; }
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
                BMDPixelFormat pf;
                HRESULT result;

                printf("[DeckLink] Format change detected.\n");

                unique_lock<mutex> lk(s->lock);
                switch(flags) {
                        case bmdDetectedVideoInputYCbCr422:
                                s->codec = UYVY;
                                break;
                        case bmdDetectedVideoInputRGB444:
                                s->codec = RGBA;
                                break;
                        default:
                                fprintf(stderr, "[Decklink] Unhandled color spec!\n");
                                abort();
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

                return result;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);
};


/* DeckLink SDK objects */

static void print_input_modes (IDeckLink* deckLink);
static int blackmagic_api_version_check(BMD_STR *current_version);

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *videoFrame, IDeckLinkAudioInputPacket *audioPacket)
{
        bool noSignal = false;
	// Video

	unique_lock<mutex> lk(s->lock);
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	if (videoFrame)
	{
		if (videoFrame->GetFlags() & bmdFrameHasNoInputSource)
		{
			fprintf(stderr, "Frame received (#%d) - No input signal detected\n", s->frames);
                        noSignal = true;
		}
		else{
                        newFrameReady = 1; // The new frame is ready to grab
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, videoFrame->GetRowBytes() * videoFrame->GetHeight());
			
		}
	}

        /// @todo
        /// Figure out when there are comming audio packets (if video processing is not fast enough/progressive NTSC with Intensity or 3:2 pulldown)
        /// @todo
        /// All the newFrameReady stuff is a bit ugly...

        if (videoFrame && newFrameReady) {
                /// @todo videoFrame should be actually retained until the data are processed
                videoFrame->GetBytes(&pixelFrame);

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
                        result = videoFrame->QueryInterface(IID_IDeckLinkVideoFrame3DExtensions, (void **)&rightEye);

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
                        result = videoFrame->GetTimecode(bmdTimecodeRP188Any, &timecode);
                        if(result != S_OK) {
                                fprintf(stderr, "Failed to acquire timecode from stream. Disabling sync.\n");
                                s->use_timecode = FALSE;
                        }
                }
        }

        lk.unlock();
        s->boss_cv.notify_one();
	
// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //

	debug_msg("VideoInputFrameArrived - END\n"); /* TOREMOVE */

	return S_OK;
}

void VideoDelegate::set_device_state(struct vidcap_decklink_state *state, int index){
	s = state;
        i = index;
}

static int blackmagic_api_version_check(BMD_STR *current_version)
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
	printf("\t-t decklink[:<device_index(indices)>[:<mode>:<colorspace>[:3D][:timecode][:connection=<input>]][:audioConsumerLevels={true|false}]\\n");
	printf("\t\t(You can omit device index, mode and color space provided that your cards supports format autodetection.)\n");
	
	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = create_decklink_iterator();
	if (deckLinkIterator == NULL) {
		return 0;
	}
	
	// Enumerate all cards in this system
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		BMD_STR                 deviceNameString = NULL;
		const char *		deviceNameCString = NULL;
		
		// *** Print the model name of the DeckLink card
		result = deckLink->GetModelName((BMD_STR *) &deviceNameString);
                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
		if (result == S_OK)
		{
			printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
			release_bmd_api_str(deviceNameString);
                        free((void *)deviceNameCString);
                } else {
			printf("\ndevice: %d.) (unable to get name)\n\n",numDevices);
                }
		
		// Increment the total number of DeckLink cards found
		numDevices++;
	
		// ** List the video input display modes supported by the card
		print_input_modes(deckLink);

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
                printf("\tUYVY\n");
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

        printf("audioConsumerLevels\n");
        printf("\tIf set true the analog audio levels are set to maximum gain on audio input.\n");
        printf("\tIf set false the selected analog input gain levels are used.\n\n");

        printf("Examples:\n");
        printf("\t%s -t decklink # captures autodetected video from first detected decklink\n", uv_argv[0]);
        printf("\t%s -t decklink:0:Hi50:UYVY # captures 1080i50, 8-bit yuv\n", uv_argv[0]);
        printf("\t%s -t decklink:0:10:v210:connection=HDMI # captures 10th format from a card (alternative syntax), 10-bit YUV, from HDMI\n", uv_argv[0]);

	printf("\n");

	return 1;
}

/* SETTINGS */

int
settings_init(void *state, char *fmt)
{
	struct vidcap_decklink_state *s = (struct vidcap_decklink_state *) state;

        s->codec = UYVY; // default

        if(fmt) {
		char *save_ptr_top = NULL;
                if(strcmp(fmt, "help") == 0) {
                        decklink_help();
                        return -1;
                }

                char *tmp;

                // choose device
                tmp = strtok_r(fmt, ":", &save_ptr_top);
                if(!tmp) {
                        s->devices_cnt = 1;
                        s->state[s->devices_cnt].index = 0;
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
                        s->mode = tmp;

                        tmp = strtok_r(NULL, ":", &save_ptr_top);
                        if (tmp) {
                                s->codec = get_codec_from_name(tmp);
                                if(s->codec == VIDEO_CODEC_NONE) {
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
                                } else if(strncasecmp(tmp, "audioConsumerLevels=",
                                                        strlen("audioConsumerLevels=")) == 0) {
                                        char *levels = tmp + strlen("audioConsumerLevels=");
                                        if (strcasecmp(levels, "false") == 0) {
                                                s->audio_consumer_levels = 0;
                                        } else {
                                                s->audio_consumer_levels = 1;
                                        }
                                } else {
                                        fprintf(stderr, "[DeckLink] Warning, unrecognized trailing options in init string: %s", tmp);
                                }
                        }
                } else {
                        s->autodetect_mode = TRUE;
                        printf("[DeckLink] Trying to autodetect format.\n");
                }
        } else {
                printf("[DeckLink] Trying to autodetect format.\n");
                s->autodetect_mode = TRUE;
                s->devices_cnt = 1;
                s->state[s->devices_cnt].index = 0;
                printf("[DeckLink] Auto-choosen device 0.\n");
        }

	return 1;	
}

/* External API ***************************************************************/

struct vidcap_type *
vidcap_decklink_probe(bool verbose)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DECKLINK_ID;
		vt->name        = "decklink";
		vt->description = "Blackmagic DeckLink card";

                IDeckLinkIterator*		deckLinkIterator = nullptr;
                IDeckLink*			deckLink;
                int				numDevices = 0;
                HRESULT				result;

                if (verbose) {
                        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
                        deckLinkIterator = create_decklink_iterator(false);
                        if (deckLinkIterator != NULL) {
                                // Enumerate all cards in this system
                                while (deckLinkIterator->Next(&deckLink) == S_OK)
                                {
                                        vt->card_count = numDevices + 1;
                                        vt->cards = (struct vidcap_card *)
                                                realloc(vt->cards, vt->card_count * sizeof(struct vidcap_card));
                                        memset(&vt->cards[numDevices], 0, sizeof(struct vidcap_card));
                                        snprintf(vt->cards[numDevices].id, sizeof vt->cards[numDevices].id,
                                                        "%d", numDevices);
                                        BMD_STR                 deviceNameString = NULL;
                                        const char *		deviceNameCString = NULL;

                                        // *** Print the model name of the DeckLink card
                                        result = deckLink->GetModelName((BMD_STR *) &deviceNameString);
                                        deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
                                        if (result == S_OK)
                                        {
                                                snprintf(vt->cards[numDevices].name, sizeof vt->cards[numDevices].name,
                                                                "%s", deviceNameCString);
                                                release_bmd_api_str(deviceNameString);
                                                free((void *)deviceNameCString);
                                        }

                                        // Increment the total number of DeckLink cards found
                                        numDevices++;

                                        // Release the IDeckLink instance when we've finished with it to prevent leaks
                                        deckLink->Release();
                                }

                                deckLinkIterator->Release();
                        }
                }
        }
	return vt;
}

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf)
{
        BMD_STR displayModeString = NULL;
        const char *displayModeCString;
        HRESULT result;

        result = displayMode->GetName(&displayModeString);
        if (result == S_OK)
        {
                switch (s->codec) {
                  case RGBA:
                        *pf = bmdFormat8BitBGRA;
                        break;
                  case UYVY:
                        *pf = bmdFormat8BitYUV;
                        break;
                  case R10k:
                        *pf = bmdFormat10BitRGB;
                        break;
                  case v210:
                        *pf = bmdFormat10BitYUV;
                        break;
                  default:
                        printf("Unsupported codec! %s\n", get_codec_name(s->codec));
                }
                // get avarage time between frames
                BMDTimeValue	frameRateDuration;
                BMDTimeScale	frameRateScale;

                tile->width = displayMode->GetWidth();
                tile->height = displayMode->GetHeight();
                s->frame->color_spec = s->codec;

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

                displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeCString,
                                tile->width, tile->height, s->frame->fps, s->next_frame_time); /* TOREMOVE */
                printf("Enable video input: %s\n", displayModeCString);
                        release_bmd_api_str(displayModeString);
                        free((void *)displayModeCString);
        }

        tile->data_len =
                vc_get_linesize(tile->width, s->frame->color_spec) * tile->height;

        if(s->stereo) {
                s->frame->tiles[1].width = s->frame->tiles[0].width;
                s->frame->tiles[1].height = s->frame->tiles[0].height;
                s->frame->tiles[1].data_len = s->frame->tiles[0].data_len;
        }

        return result;
}

void *
vidcap_decklink_init(const struct vidcap_params *params)
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
        BMDAudioConnection              audioConnection = bmdAudioConnectionEmbedded;

#ifdef WIN32
	// Initialize COM on this thread
	result = CoInitialize(NULL);
	if(FAILED(result)) {
                string err_msg = bmd_hresult_to_string(result);
		fprintf(stderr, "Initialization of COM failed: "
				"%s.\n", err_msg.c_str());
		return NULL;
	}
#endif


        BMD_STR current_version;
        if(!blackmagic_api_version_check(&current_version)) {
		fprintf(stderr, "\nThe DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled against DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                fprintf(stderr, "Vendor download page is http://http://www.blackmagic-design.com/support/ \n");
                if(current_version) {
                        const char *currentVersionCString = get_cstr_from_bmd_api_str(current_version);
                        fprintf(stderr, "Currently installed version is: %s\n", currentVersionCString);
                        release_bmd_api_str(current_version);
                        free((void *)currentVersionCString);
                } else {
                        fprintf(stderr, "No installed drivers detected\n");
                }
                fprintf(stderr, "\n");
                return NULL;
        }


	s = new vidcap_decklink_state();
	if (s == NULL) {
		//printf("Unable to allocate DeckLink state\n",fps);
		printf("Unable to allocate DeckLink state\n");
		return NULL;
	}

        gettimeofday(&s->t0, NULL);

        s->stereo = FALSE;
        s->use_timecode = FALSE;
        s->autodetect_mode = FALSE;
        s->connection = (BMDVideoConnection) 0;
        s->flags = 0;
        s->audio_consumer_levels = -1;

	// SET UP device and mode
        char *tmp_fmt = strdup(vidcap_params_get_fmt(params));
        int ret = settings_init(s, tmp_fmt);
        free(tmp_fmt);
	if(ret == 0) {
                delete s;
		return NULL;
	}
	if(ret == -1) {
                delete s;
		return &vidcap_init_noerr;
	}

        if(vidcap_params_get_flags(params) & (VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG)) {
                s->grab_audio = TRUE;
                switch(vidcap_params_get_flags(params) & (VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG)) {
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
                if (s->devices_cnt > 1) {
                        fprintf(stderr, "[DeckLink] Passed more than one device while setting 3D mode. "
                                        "In this mode, only one device needs to be passed.");
                        free(s);
                        return NULL;
                }
                s->frame = vf_alloc(2);
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
                deckLinkIterator = create_decklink_iterator();
                if (deckLinkIterator == NULL) {
                        vf_free(s->frame);
                        free(s);
                        return NULL;
                }
                while (deckLinkIterator->Next(&deckLink) == S_OK)
                {
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

                        BMD_STR deviceNameString = NULL;
                        const char* deviceNameCString = NULL;

                        // Print the model name of the DeckLink card
                        result = deckLink->GetModelName(&deviceNameString);
                        if (result == S_OK)
                        {	
                                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);

                                printf("Using device [%s]\n", deviceNameCString);
                                release_bmd_api_str(deviceNameString);
                                free((void *) deviceNameCString);

                                // Query the DeckLink for its configuration interface
                                result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
                                if (result != S_OK)
                                {
                                        string err_msg = bmd_hresult_to_string(result);
                                        fprintf(stderr, "Could not obtain the IDeckLinkInput interface: %s\n",
                                                        err_msg.c_str());
                                        goto error;
                                }

                                s->state[i].deckLinkInput = deckLinkInput;

                                // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
                                result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
                                if (result != S_OK)
                                {
                                        string err_msg = bmd_hresult_to_string(result);
                                        fprintf(stderr, "Could not obtain the video input display mode iterator: %s\n",
                                                        err_msg.c_str());
                                        goto error;
                                }

                                mnum = 0;
                                bool mode_found = false;

                                while (displayModeIterator->Next(&displayMode) == S_OK)
                                {
					if (s->mode.length() <= 2) {
						if (atoi(s->mode.c_str()) != mnum) {
							mnum++;
							// Release the IDeckLinkDisplayMode object to prevent a leak
							displayMode->Release();
							continue;
						}

						mode_found = true;
						mnum++;
						break;
					} else {
						union {
							uint32_t fourcc;
							char tmp[4];
						};
						memcpy(tmp, s->mode.c_str(), s->mode.length());
						if (s->mode.length() == 3) tmp[3] = ' ';
						fourcc = htonl(fourcc);
						if (displayMode->GetDisplayMode() == fourcc) {
							mode_found = true;
							break;
						}
					}
                                }

                                if(mode_found) {
                                        printf("The desired display mode is supported: %s\n", s->mode.c_str());
                                } else {
                                        fprintf(stderr, "Desired mode index %s is out of bounds.\n",
                                                        s->mode.c_str());
                                        goto error;
                                }

                                BMDPixelFormat pf;

                                if(set_display_mode_properties(s, tile, displayMode, &pf) == S_OK) {
                                        IDeckLinkAttributes *deckLinkAttributes;
                                        deckLinkInput->StopStreams();

                                       result = deckLinkInput->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes);
                                        if (result != S_OK)
                                        {
                                                string err_msg = bmd_hresult_to_string(result);

                                                fprintf(stderr, "Could not query device attributes: %s\n",
                                                                err_msg.c_str());
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
                                                switch (result) {
                                                case E_INVALIDARG:
                                                        fprintf(stderr, "You have required invalid video mode and pixel format combination.\n");
                                                        break;
                                                case E_ACCESSDENIED:
                                                        fprintf(stderr, "Unable to access the hardware or input "
                                                                        "stream currently active (another application using it?).\n");
                                                        break;
                                                }
                                                string err_msg = bmd_hresult_to_string(result);
                                                fprintf(stderr, "Could not enable video input: %s\n",
                                                                err_msg.c_str());
                                                goto error;
                                        }

                                        // Query the DeckLink for its configuration interface
                                        result = deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
                                        if (result != S_OK)
                                        {
                                                string err_msg = bmd_hresult_to_string(result);
                                                fprintf(stderr, "Could not obtain the IDeckLinkConfiguration interface: %s\n", err_msg.c_str());
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
                                                if (s->audio_consumer_levels != -1) {
                                                        result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                                                        s->audio_consumer_levels == 1 ? true : false);
                                                        if(result != S_OK) {
                                                                fprintf(stderr, "[DeckLink capture] Unable set input audio consumer levels.\n");
                                                        }
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
                                                string err_msg = bmd_hresult_to_string(result);
                                                fprintf(stderr, "Could not start stream: %s\n", err_msg.c_str());
                                                goto error;
                                        }

                                }else{
                                        string err_msg = bmd_hresult_to_string(result);
                                        fprintf(stderr, "Could not set display mode properties: %s\n", err_msg.c_str());
                                        goto error;
                                }

                                displayMode->Release();
                                displayMode = NULL;

                                displayModeIterator->Release();
                                displayModeIterator = NULL;
                        }
                }
		deckLinkIterator->Release();
        }

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

        if (displayModeIterator != NULL){
                displayModeIterator->Release();
                displayModeIterator = NULL;
        }

        if (s) {
                cleanup_common(s);
        }

	return NULL;
}

static void cleanup_common(struct vidcap_decklink_state *s) {
        for (int i = 0; i < s->devices_cnt; ++i) {
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

        free(s->audio.data);

        vf_free(s->frame);
        delete s;
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
                        string err_msg = bmd_hresult_to_string(result);
			fprintf(stderr, MODULE_NAME "Could not stop stream: %s\n", err_msg.c_str());
		}

                if(s->grab_audio && i == 0) {
                        result = s->state[i].deckLinkInput->DisableAudioInput();
                        if (result != S_OK) {
                                string err_msg = bmd_hresult_to_string(result);
                                fprintf(stderr, MODULE_NAME "Could disable audio input: %s\n", err_msg.c_str());
                        }
                }
		result = s->state[i].deckLinkInput->DisableVideoInput();
                if (result != S_OK) {
                        string err_msg = bmd_hresult_to_string(result);
                        fprintf(stderr, MODULE_NAME "Could disable video input: %s\n", err_msg.c_str());
                }
        }

        cleanup_common(s);
}

/**
 * This function basically counts frames from all devices, optionally
 * with respect to timecode (if synchronized).
 *
 * Lock needs to be hold during the whole function call.
 *
 * @param s Blackmagic state
 * @return number of captured tiles
 */
int nr_frames(struct vidcap_decklink_state *s) {
        BMDTimecodeBCD max_timecode = 0u;
        int tiles_total = 0;
        int i;

        /* If we use timecode, take maximal timecode value... */
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

        /* count all tiles */
        for (i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].delegate->newFrameReady) {
                        /* if inputs are synchronized, use only up-to-date frames (with same TC)
                         * as the most recent */
                        if(s->use_timecode) {
                                if(s->state[i].delegate->timecode && s->state[i].delegate->timecode->GetBCD () != max_timecode) {
                                        s->state[i].delegate->newFrameReady = FALSE;
                                } else {
                                        tiles_total++;
                                }
                        }
                        /* otherwise, simply add up the count */
                        else {
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
        int                             tiles_total = 0;
        int                             i;
        bool				frame_ready = true;
	
	int timeout = 0;

	unique_lock<mutex> lk(s->lock);
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	debug_msg("vidcap_decklink_grab - before while\n"); /* TOREMOVE */

        tiles_total = nr_frames(s);

        while(tiles_total != s->devices_cnt) {
	//while (!s->state[0].delegate->newFrameReady) {
                cv_status rc = cv_status::no_timeout;
		debug_msg("vidcap_decklink_grab - pthread_cond_timedwait\n"); /* TOREMOVE */
                steady_clock::time_point t0(steady_clock::now());

                while(rc == cv_status::no_timeout
                                && tiles_total != s->devices_cnt /* not all tiles */
                                && !timeout) {
                        rc = s->boss_cv.wait_for(lk, microseconds(2 * s->next_frame_time));
                        // recompute tiles count
                        tiles_total = nr_frames(s);

                        // this is for the case of multiple tiles (eg. when one tile is persistently
                        // missing, eg. 01301301. Therefore, pthread_cond_timewait doesn't timeout but
                        // actual timeout time has reached.
                        steady_clock::time_point t(steady_clock::now());
                        if (duration_cast<microseconds>(t - t0).count() > 2 * s->next_frame_time)
                                timeout = 1;

                }
                debug_msg("vidcap_decklink_grab - AFTER pthread_cond_timedwait - %d tiles\n", tiles_total); /* TOREMOVE */

                if (rc != cv_status::no_timeout || timeout) { //(rc == ETIMEDOUT) {
                        printf("Waiting for new frame timed out!\n");
                        debug_msg("Waiting for new frame timed out!\n");

                        // try to restart stream
                        /*
                        HRESULT result;
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

                        return NULL;
                }
	}

        /* cleanup newframe flag */
        for (i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].delegate->newFrameReady == 0) {
                        frame_ready = false;
                }
                s->state[i].delegate->newFrameReady = 0;
	}

// UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UNLOCK - UN //
	lk.unlock();

	if(!frame_ready)
		return NULL;

        /* count returned tiles */
        int count = 0;
        if(s->stereo) {
                if (s->state[0].delegate->pixelFrame != NULL &&
                                s->state[0].delegate->pixelFrameRight != NULL) {
                        s->frame->tiles[0].data = (char*)s->state[0].delegate->pixelFrame;
                        if (s->codec == RGBA) {
                            vc_copylineRGBA((unsigned char*) s->frame->tiles[0].data,
                                        (unsigned char*)s->frame->tiles[0].data,
                                        s->frame->tiles[i].data_len, 16, 8, 0);
                        }
                        s->frame->tiles[1].data = (char*)s->state[0].delegate->pixelFrameRight;
                        if (s->codec == RGBA) {
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
                                if (s->codec == RGBA) {
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

                struct timeval t;
                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, s->t0);
                if (seconds >= 5) {
                        float fps  = s->frames / seconds;
                        log_msg(LOG_LEVEL_INFO, "[Decklink capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                        s->t0 = t;
                        s->frames = 0;
                }

                return s->frame;
        }

	return NULL;
}

/* function from DeckLink SDK sample DeviceList */

static void print_input_modes (IDeckLink* deckLink)
{
	IDeckLinkInput*			deckLinkInput = NULL;
	IDeckLinkDisplayModeIterator*		displayModeIterator = NULL;
	IDeckLinkDisplayMode*			displayMode = NULL;
	HRESULT					result;	
	int 					displayModeNumber = 0;
	
	// Query the DeckLink for its configuration interface
	result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput);
	if (result != S_OK)
	{
                string err_msg = bmd_hresult_to_string(result);
		fprintf(stderr, "Could not obtain the IDeckLinkInput interface: %s\n", err_msg.c_str());
                if (result == E_NOINTERFACE) {
                        printf("Device doesn't support video capture.\n");
                }
		goto bail;
	}
	
	// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
	result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
	if (result != S_OK)
	{
                string err_msg = bmd_hresult_to_string(result);
		fprintf(stderr, "Could not obtain the video input display mode iterator: %s\n", err_msg.c_str());
		goto bail;
	}
	
	// List all supported output display modes
	printf("capture modes:\n");
	while (displayModeIterator->Next(&displayMode) == S_OK)
	{
		BMD_STR displayModeString = NULL;
                const char *displayModeCString;
		
		result = displayMode->GetName((BMD_STR *) &displayModeString);
                displayModeCString = get_cstr_from_bmd_api_str(displayModeString);

		if (result == S_OK)
		{
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
                        uint32_t mode = ntohl(displayMode->GetDisplayMode());
                        printf("%d (%.4s)) %-20s \t %d x %d \t %2.2f FPS%s\n", displayModeNumber, (char *) &mode, displayModeCString,
                                        modeWidth, modeHeight, (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (flags & bmdDisplayModeSupports3D ? "\t (supports 3D)" : ""));
                        release_bmd_api_str(displayModeString);
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
	
	if (deckLinkInput != NULL)
		deckLinkInput->Release();
}

#endif /* HAVE_DECKLINK */

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
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"
#include "audio/audio.h"
#include "audio/utils.h"

#include "blackmagic_common.h"

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

/**
 * @todo
 * The input conversion doesn't seem to work right now. After fixing, remove this
 * macro (and related ifdefs).
 */
#define IN_CONV_BROKEN 1

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
        string                  device_id; // either numeric value or device name
};

struct vidcap_decklink_state {
        vector <struct device_state>     state;
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
        unsigned int            sync_timecode:1; /* use timecode when grabbing from multiple inputs */
        BMDVideoConnection      connection;
        int                     audio_consumer_levels; ///< 0 false, 1 true, -1 default
        BMDVideoInputConversionMode conversion_mode;

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
        uint32_t timecode;
        void*   audioFrame;
        int     audioFrameSamples;
	struct  vidcap_decklink_state *s;
        int     i;
	
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

                log_msg(LOG_LEVEL_NOTICE, "[DeckLink] Format change detected.\n");

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
                                        s->audio.bps == 2 ? bmdAudioSampleType16bitInteger :
                                                bmdAudioSampleType32bitInteger,
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

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *videoFrame, IDeckLinkAudioInputPacket *audioPacket)
{
	// Video

	unique_lock<mutex> lk(s->lock);
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	if (videoFrame)
	{
		if (videoFrame->GetFlags() & bmdFrameHasNoInputSource)
		{
			log_msg(LOG_LEVEL_INFO, "Frame received (#%d) - No input signal detected\n", s->frames);
		} else {
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

                IDeckLinkTimecode *tc = NULL;
                if (videoFrame->GetTimecode(bmdTimecodeRP188Any, &tc) == S_OK) {
                        timecode = tc->GetBCD();
                } else {
                        timecode = 0;
                        if (s->sync_timecode) {
                                log_msg(LOG_LEVEL_ERROR, "Failed to acquire timecode from stream. Disabling sync.\n");
                                s->sync_timecode = FALSE;
                        }
                }

                if(audioPacket) {
                        audioPacket->GetBytes(&audioFrame);
                        if(audio_capture_channels == 1) { // ther are actually 2 channels grabbed
                                demux_channel(s->audio.data, (char *) audioFrame, s->audio.bps, audioPacket->GetSampleFrameCount() * 2 /* channels */ * s->audio.bps,
                                                2, /* channels (originally( */
                                        0 /* we want first channel */
                                                );
                                s->audio.data_len = audioPacket->GetSampleFrameCount() * 1 * s->audio.bps;
                        } else {
                                s->audio.data_len = audioPacket->GetSampleFrameCount() * audio_capture_channels * s->audio.bps;
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

static map<BMDVideoConnection, string> connection_string_map =  {
        { bmdVideoConnectionSDI, "SDI" },
        { bmdVideoConnectionHDMI, "HDMI"},
        { bmdVideoConnectionOpticalSDI, "OpticalSDI"},
        { bmdVideoConnectionComponent, "Component"},
        { bmdVideoConnectionComposite, "Composite"},
        { bmdVideoConnectionSVideo, "SVideo"}
};

/* HELP */
static int
decklink_help()
{
	IDeckLinkIterator*		deckLinkIterator;
	IDeckLink*			deckLink;
	int				numDevices = 0;
	HRESULT				result;

	printf("\nDecklink options:\n");
	printf("\t-t decklink[:<device_index(indices)>[:<mode>:<colorspace>[:3D][:sync_timecode][:connection=<input>][:audioConsumerLevels={true|false}]"
#ifndef IN_CONV_BROKEN
                        "[:conversion=<conv_mode>]"
#endif
                        "]\n");
        printf("\t\tor\n");
	printf("\t-t decklink{:mode=<mode>|:device=<device_index>|:codec=<colorspace>...<key>=<val>}*\n");
	printf("\t(Mode specification is mandatory if your card does not support format autodetection.)\n");
        printf("\n");

        printf("Available Colorspaces:\n");
        printf("\tUYVY\n");
        printf("\tv210\n");
        printf("\tRGBA\n");
        printf("\tR10k\n");
        printf("\n");

        printf("3D\n");
        printf("\tUse this to capture 3D from supported card (eg. DeckLink HD 3D Extreme).\n");
        printf("\tDo not use it for eg. Quad or Duo. Availability of the mode is indicated\n");
        printf("\tin video format listing above (\"supports 3D\").\n");

	printf("\n");
        printf("sync_timecode\n");
        printf("\tTry to synchronize inputs based on timecode (for multiple inputs, eg. tiled 4K)\n");
	printf("\n");

        printf("audioConsumerLevels\n");
        printf("\tIf set true the analog audio levels are set to maximum gain on audio input.\n");
        printf("\tIf set false the selected analog input gain levels are used.\n");
	printf("\n");

#ifndef IN_CONV_BROKEN
        printf("conversion\n");
        printf("\tnone - No video input conversion\n");
        printf("\t10lb - HD1080 to SD video input down conversion\n");
        printf("\t10am - Anamorphic from HD1080 to SD video input down conversion\n");
        printf("\t72lb - Letter box from HD720 to SD video input down conversion\n");
        printf("\t72ab - Letterbox video input up conversion\n");
        printf("\tamup - Anamorphic video input up conversion\n");
	printf("\n");
#endif

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
		result = deckLink->GetDisplayName((BMD_STR *) &deviceNameString);
                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
		if (result == S_OK)
		{
			printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
			release_bmd_api_str(deviceNameString);
                        free((void *)deviceNameCString);
                } else {
			printf("\ndevice: %d.) (unable to get name)\n\n", numDevices);
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
                                for (auto it : connection_string_map) {
                                        if (connections & it.first)
                                                printf("\t%s\n", it.second.c_str());
                                }
                        }
                }
				
		// Release the IDeckLink instance when we've finished with it to prevent leaks
		deckLink->Release();
	}
	
	deckLinkIterator->Release();

        decklink_uninitialize();

	// If no DeckLink cards were found in the system, inform the user
	if (numDevices == 0)
	{
		log_msg(LOG_LEVEL_ERROR, "No Blackmagic Design devices were found.\n");
        }

        printf("\n");

        printf("Examples:\n");
        printf("\t%s -t decklink # captures autodetected video from first DeckLink in system\n", uv_argv[0]);
        printf("\t%s -t decklink:0:Hi50:UYVY # captures 1080i50, 8-bit yuv\n", uv_argv[0]);
        printf("\t%s -t decklink:0:10:v210:connection=HDMI # captures 10th format from a card (alternative syntax), 10-bit YUV, from HDMI\n", uv_argv[0]);
        printf("\t%s -t decklink:mode=23ps # captures 1080p24, 8-bit yuv from first device\n", uv_argv[0]);
        printf("\t%s -t \"decklink:mode=Hp30:codec=v210:device=DeckLink HD Extreme 3D+\" # captures 1080p30, 10-bit yuv from DeckLink HD Extreme 3D+\n", uv_argv[0]);

	printf("\n");

        print_decklink_version();

        printf("\n");

	return 1;
}

/* SETTINGS */
static void parse_devices(struct vidcap_decklink_state *s, const char *devs)
{
        char *devices = strdup(devs);
        char *ptr;
        char *save_ptr_dev;

        s->devices_cnt = 0;
        ptr = strtok_r(devices, ",", &save_ptr_dev);
        do {
                s->devices_cnt += 1;
                s->state.resize(s->devices_cnt);
                s->state[s->devices_cnt - 1].device_id = ptr;
        } while ((ptr = strtok_r(NULL, ",", &save_ptr_dev)));
        free (devices);
}

/* Parses option in format key=value */
static bool parse_option(struct vidcap_decklink_state *s, const char *opt)
{
        if(strcasecmp(opt, "3D") == 0) {
                s->stereo = TRUE;
        } else if(strcasecmp(opt, "timecode") == 0) {
                s->sync_timecode = TRUE;
        } else if(strncasecmp(opt, "connection=", strlen("connection=")) == 0) {
                const char *connection = opt + strlen("connection=");
                bool found = false;
                for (auto it : connection_string_map) {
                        if (strcasecmp(connection, it.second.c_str()) == 0) {
                                s->connection = it.first;
                                found = true;
                        }
                }
                if (!found) {
                        fprintf(stderr, "[DeckLink] Unrecognized connection %s.\n", connection);
                        return false;
                }
        } else if(strncasecmp(opt, "audioConsumerLevels=",
                                strlen("audioConsumerLevels=")) == 0) {
                const char *levels = opt + strlen("audioConsumerLevels=");
                if (strcasecmp(levels, "false") == 0) {
                        s->audio_consumer_levels = 0;
                } else {
                        s->audio_consumer_levels = 1;
                }
        } else if(strncasecmp(opt, "conversion=",
                                strlen("conversion=")) == 0) {
                const char *conversion_mode = opt + strlen("conversion=");

                union {
                        uint32_t fourcc;
                        char tmp[4];
                };
                memcpy(tmp, conversion_mode, max(strlen(conversion_mode), sizeof(tmp)));
                s->conversion_mode = (BMDVideoInputConversionMode) htonl(fourcc);
        } else if(strncasecmp(opt, "device=",
                                strlen("device=")) == 0) {
                const char *devices = opt + strlen("device=");
                parse_devices(s, devices);
        } else if(strncasecmp(opt, "mode=",
                                strlen("mode=")) == 0) {
                s->mode = opt + strlen("mode=");
        } else if(strncasecmp(opt, "codec=",
                                strlen("codec=")) == 0) {
                const char *codec = opt + strlen("codec=");
                s->codec = get_codec_from_name(codec);
                if(s->codec == VIDEO_CODEC_NONE) {
                        fprintf(stderr, "Wrong config. Unknown color space %s\n", codec);
                        return false;
                }
        } else {
                fprintf(stderr, "[DeckLink] Warning, unrecognized trailing options in init string: %s\n", opt);
                return false;
        }

        return true;

}

static bool settings_init_key_val(struct vidcap_decklink_state *s, char **save_ptr)
{
        char *tmp;

        while((tmp = strtok_r(NULL, ":", save_ptr))) {
                if (!parse_option(s, tmp)) {
                        return false;
                }
        }

        return true;
}

static int settings_init(struct vidcap_decklink_state *s, char *fmt)
{
        // defaults
        s->codec = UYVY;
        s->devices_cnt = 1;
        s->state.resize(s->devices_cnt);
        s->state[0].device_id = "0";

        char *tmp;
        char *save_ptr = NULL;

        if (!fmt || (tmp = strtok_r(fmt, ":", &save_ptr)) == NULL) {
                printf("[DeckLink] Auto-choosen device 0.\n");

                return 1;
        }

        if(strcmp(tmp, "help") == 0) {
                decklink_help();
                return -1;
        }

        // options are in format <device>:<mode>:<codec>[:other_opts]
        if (isdigit(tmp[0])) {
                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Deprecated syntax used, please use options in format \"key=value\"\n";
                // choose device
                parse_devices(s, tmp);

                // choose mode
                tmp = strtok_r(NULL, ":", &save_ptr);
                if(tmp) {
                        s->mode = tmp;

                        tmp = strtok_r(NULL, ":", &save_ptr);
                        if (tmp) {
                                s->codec = get_codec_from_name(tmp);
                                if(s->codec == VIDEO_CODEC_NONE) {
                                        fprintf(stderr, "Wrong config. Unknown color space %s\n", tmp);
                                        return 0;
                                }
                        }
                        if (!settings_init_key_val(s, &save_ptr)) {
                                return 0;
                        }
                }
        } else { // options are in format key=val
                if (!parse_option(s, tmp)) {
                        return 0;
                }
                if (!settings_init_key_val(s, &save_ptr)) {
                        return 0;
                }
        }

        return 1;
}

/* External API ***************************************************************/

static struct vidcap_type *
vidcap_decklink_probe(bool verbose)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
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
                                        IDeckLinkAttributes *deckLinkAttributes;

                                        result = deckLink->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes);
                                        if (result != S_OK) {
                                                continue;
                                        }
                                        int64_t connections;
                                        if(deckLinkAttributes->GetInt(BMDDeckLinkVideoInputConnections, &connections) != S_OK) {
                                                fprintf(stderr, "[DeckLink] Could not get connections.\n");
                                        } else {
                                                for (auto it : connection_string_map) {
                                                        if (connections & it.first) {
                                                                vt->card_count += 1;
                                                                vt->cards = (struct device_info *)
                                                                        realloc(vt->cards, vt->card_count * sizeof(struct device_info));
                                                                memset(&vt->cards[vt->card_count - 1], 0, sizeof(struct device_info));
                                                                snprintf(vt->cards[vt->card_count - 1].id, sizeof vt->cards[vt->card_count - 1].id,
                                                                                "device=%d:connection=%s", numDevices, it.second.c_str());
                                                                BMD_STR deviceNameString = NULL;
                                                                const char *deviceNameCString = NULL;

                                                                // *** Print the model name of the DeckLink card
                                                                result = deckLink->GetModelName((BMD_STR *) &deviceNameString);
                                                                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
                                                                if (result == S_OK)
                                                                {
                                                                        snprintf(vt->cards[vt->card_count - 1].name, sizeof vt->cards[vt->card_count - 1].name,
                                                                                        "%s #%d (%s)", deviceNameCString, numDevices, it.second.c_str());
                                                                        release_bmd_api_str(deviceNameString);
                                                                        free((void *)deviceNameCString);
                                                                }
                                                        }
                                                }
                                        }


                                        // Increment the total number of DeckLink cards found
                                        numDevices++;

                                        // Release the IDeckLink instance when we've finished with it to prevent leaks
                                        deckLink->Release();
                                }

                                deckLinkIterator->Release();
                        }
                decklink_uninitialize();
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

static int
vidcap_decklink_init(const struct vidcap_params *params, void **state)
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

        if (!blackmagic_api_version_check()) {
                return VIDCAP_INIT_FAIL;
        }

	s = new vidcap_decklink_state();
	if (s == NULL) {
		//printf("Unable to allocate DeckLink state\n",fps);
		printf("Unable to allocate DeckLink state\n");
		return VIDCAP_INIT_FAIL;
	}

        gettimeofday(&s->t0, NULL);

        s->stereo = FALSE;
        s->sync_timecode = FALSE;
        s->connection = (BMDVideoConnection) 0;
        s->flags = 0;
        s->audio_consumer_levels = -1;
        s->conversion_mode = bmdNoVideoInputConversion;

	// SET UP device and mode
        char *tmp_fmt = strdup(vidcap_params_get_fmt(params));
        int ret = settings_init(s, tmp_fmt);
        free(tmp_fmt);
	if(ret == 0) {
                delete s;
		return VIDCAP_INIT_FAIL;
	}
	if(ret == -1) {
                delete s;
		return VIDCAP_INIT_NOERR;
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
                if (audio_capture_bps == 2) {
                        s->audio.bps = 2;
                } else {
                        if (audio_capture_bps != 4 && audio_capture_bps != 0) {
                                log_msg(LOG_LEVEL_WARNING, "[Decklink] Ignoring unsupported Bps!\n");
                        }
                        s->audio.bps = 4;
                }
                if (audio_capture_sample_rate != 0 && audio_capture_sample_rate != 48000) {
                                log_msg(LOG_LEVEL_WARNING, "[Decklink] Ignoring unsupported sample rate!\n");
                }
                s->audio.sample_rate = 48000;
                s->audio.ch_count = audio_capture_channels;
                s->audio.data = (char *) malloc (48000 * audio_capture_channels * 2);
        } else {
                s->grab_audio = FALSE;
        }

	vector<bool> device_found(s->devices_cnt);
        for(int i = 0; i < s->devices_cnt; ++i)
                device_found[i] = false;

        if(s->stereo) {
                if (s->devices_cnt > 1) {
                        fprintf(stderr, "[DeckLink] Passed more than one device while setting 3D mode. "
                                        "In this mode, only one device needs to be passed.");
                        delete s;
                        return VIDCAP_INIT_FAIL;
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
                deckLinkIterator = create_decklink_iterator(true, i == 0 ? true : false);
                if (deckLinkIterator == NULL) {
                        vf_free(s->frame);
                        delete s;
                        return VIDCAP_INIT_FAIL;
                }
                while (deckLinkIterator->Next(&deckLink) == S_OK)
                {
                        bool found = false;

                        BMD_STR deviceNameString = NULL;
                        const char* deviceNameCString = NULL;

                        result = deckLink->GetDisplayName(&deviceNameString);
                        if (result == S_OK)
                        {
                                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);

                                if (strcmp(deviceNameCString, s->state[i].device_id.c_str()) == 0) {
                                        found = true;
                                }

                                release_bmd_api_str(deviceNameString);
                                free((void *) deviceNameCString);
                        }

                        if (isdigit(s->state[i].device_id.c_str()[0]) && atoi(s->state[i].device_id.c_str()) == dnum) {
                                found = true;
                        }

                        if (!found) {
                                dnum++;

                                // Release the IDeckLink instance when we've finished with it to prevent leaks
                                deckLink->Release();
                                deckLink = NULL;
                                continue;	
                        }

                        device_found[i] = true;
                        dnum++;

                        s->state[i].deckLink = deckLink;

                        // Print the model name of the DeckLink card
                        result = deckLink->GetDisplayName(&deviceNameString);
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
#define MODE_SPEC_FOURCC -1
                                int mode_idx = MODE_SPEC_FOURCC;

                                if (s->mode.length() <= 2) {
                                        mode_idx = atoi(s->mode.c_str());
                                }

                                while (displayModeIterator->Next(&displayMode) == S_OK)
                                {
					if (mode_idx != MODE_SPEC_FOURCC) {
						if (mode_idx != mnum) {
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
						memcpy(tmp, s->mode.c_str(), min(s->mode.length(), sizeof tmp));
						if (s->mode.length() == 3) tmp[3] = ' ';
						fourcc = htonl(fourcc);
						if (displayMode->GetDisplayMode() == fourcc) {
							mode_found = true;
							break;
						}
					}
                                }

                                if (mode_found) {
                                        log_msg(LOG_LEVEL_INFO, "The desired display mode is supported: %s\n", s->mode.c_str());
                                } else {
                                        if (mode_idx == MODE_SPEC_FOURCC) {
                                                log_msg(LOG_LEVEL_ERROR, "Desired mode \"%s\" is invalid or not supported.\n", s->mode.c_str());
                                        } else {
                                                log_msg(LOG_LEVEL_ERROR, "Desired mode index %s is out of bounds.\n", s->mode.c_str());
                                        }
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

                                        if (s->mode.empty()) {
                                                log_msg(LOG_LEVEL_INFO, "[DeckLink] Trying to autodetect format.\n");
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
                                                result = deckLinkInput->EnableAudioInput(
                                                        bmdAudioSampleRate48kHz,
                                                        s->audio.bps == 2 ? bmdAudioSampleType16bitInteger : bmdAudioSampleType32bitInteger,
                                                        audio_capture_channels == 1 ? 2 : audio_capture_channels);
                                                if (result == S_OK) {
                                                        LOG(LOG_LEVEL_NOTICE) << "Decklink audio capture initialized sucessfully: " << audio_desc_from_frame(&s->audio) << "\n";
                                                }
                                        }


                                        result = deckLinkConfiguration->SetInt(bmdDeckLinkConfigVideoInputConversionMode, s->conversion_mode);
                                        if(result != S_OK) {
                                                log_msg(LOG_LEVEL_ERROR, "[DeckLink capture] Unable to set conversion mode.\n");
                                                goto error;
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
                        LOG(LOG_LEVEL_ERROR) << "Device " << s->state[i].device_id << " was not found.\n";
                        goto error;
                }
        }


	printf("DeckLink capture device enabled\n");

	debug_msg("vidcap_decklink_init - END\n"); /* TOREMOVE */

        *state = s;
	return VIDCAP_INIT_OK;

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

	return VIDCAP_INIT_FAIL;
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

        decklink_uninitialize();
}

static void
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
static int nr_frames(struct vidcap_decklink_state *s) {
        BMDTimecodeBCD max_timecode = 0u;
        int tiles_total = 0;
        int i;

        /* If we use timecode, take maximal timecode value... */
        if (s->sync_timecode) {
                for (i = 0; i < s->devices_cnt; ++i) {
                        if(s->state[i].delegate->newFrameReady) {
                                if (s->state[i].delegate->timecode > max_timecode) {
                                        max_timecode = s->state[i].delegate->timecode;
                                }
                        }
                }
        }

        /* count all tiles */
        for (i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].delegate->newFrameReady) {
                        /* if inputs are synchronized, use only up-to-date frames (with same TC)
                         * as the most recent */
                        if(s->sync_timecode) {
                                if(s->state[i].delegate->timecode && s->state[i].delegate->timecode != max_timecode) {
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

static struct video_frame *
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
                        log_msg(LOG_LEVEL_VERBOSE, "Waiting for new frame timed out!\n");

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

                s->frame->timecode = s->state[0].delegate->timecode;

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
                        printf("%2d (%.4s)) %-20s \t %d x %d \t %2.2f FPS%s\n", displayModeNumber, (char *) &mode, displayModeCString,
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

static const struct video_capture_info vidcap_decklink_info = {
        vidcap_decklink_probe,
        vidcap_decklink_init,
        vidcap_decklink_done,
        vidcap_decklink_grab,
};

REGISTER_MODULE(decklink, &vidcap_decklink_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


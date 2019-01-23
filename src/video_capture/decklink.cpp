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
 * Copyright (c) 2005-2016 CESNET z.s.p.o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <cassert>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <list>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "blackmagic_common.h"
#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rang.hpp"
#include "tv.h"
#include "video.h"
#include "video_capture.h"

#define FRAME_TIMEOUT 60000000 // 30000000 // in nanoseconds
#define MOD_NAME "[DeckLink capture] "

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

using namespace std;
using namespace std::chrono;
using rang::fg;
using rang::style;

// static int	device = 0; // use first BlackMagic device
// static int	mode = 5; // for Intensity
// static int	mode = 6; // for Decklink  6) HD 1080i 59.94; 1920 x 1080; 29.97 FPS 7) HD 1080i 60; 1920 x 1080; 30 FPS
//static int	connection = 0; // the choice of BMDVideoConnection // It should be 0 .... bmdVideoConnectionSDI

// performs command, if failed, displays error and jumps to error label
#define EXIT_IF_FAILED(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                        goto error;\
                }\
        } while (0)

// similar as above, but only displays warning
#define CALL_AND_CHECK_2(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                }\
        } while (0)

#define CALL_AND_CHECK_3(cmd, name, msg) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                } else {\
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << name << ": " << msg << "\n";\
		}\
        } while (0)

#define GET_3TH_ARG(arg1, arg2, arg3, ...) arg3
#define CALL_AND_CHECK_CHOOSER(...) \
    GET_3TH_ARG(__VA_ARGS__, CALL_AND_CHECK_3, CALL_AND_CHECK_2, )

#define CALL_AND_CHECK(cmd, ...) CALL_AND_CHECK_CHOOSER(__VA_ARGS__)(cmd, __VA_ARGS__)

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
        queue<IDeckLinkAudioInputPacket *> audioPackets;
        codec_t                 codec;
        BMDVideoInputFlags flags;

        mutex                   lock;
	condition_variable      boss_cv;

        int                     frames;
        unsigned int            grab_audio:1; /* wheather we process audio or not */
        bool                    stereo; /* for eg. DeckLink HD Extreme, Quad doesn't set this !!! */
        unsigned int            sync_timecode:1; /* use timecode when grabbing from multiple inputs */
        BMDVideoConnection      connection;
        int                     audio_consumer_levels; ///< 0 false, 1 true, -1 default
        BMDVideoInputConversionMode conversion_mode;
        BMDDeckLinkCapturePassthroughMode passthrough; // 0 means don't set

        struct timeval          t0;

        bool                    detect_format;
        int                     requested_bit_depth; // 8, 10 or 12
        bool                    p_not_i;

        uint32_t                duplex;
        uint32_t                link;
};

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf);
static void cleanup_common(struct vidcap_decklink_state *s);
static list<tuple<int, string, string, string>> get_input_modes (IDeckLink* deckLink);
static void print_input_modes (IDeckLink* deckLink);

class VideoDelegate : public IDeckLinkInputCallback {
private:
	int32_t                       mRefCount{};

public:
        int	                      newFrameReady{};
        IDeckLinkVideoFrame          *rightEyeFrame{};
        void                         *pixelFrame{};
        void                         *pixelFrameRight{};
        uint32_t                      timecode{};
        struct vidcap_decklink_state *s;
        int                           i; ///< index of the device
	
        VideoDelegate(struct vidcap_decklink_state *state, int index) : s(state), i(index) {
        }
	
        virtual ~VideoDelegate () {
		if(rightEyeFrame)
                        rightEyeFrame->Release();
	}

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID *) { return E_NOINTERFACE; }
	virtual ULONG STDMETHODCALLTYPE AddRef(void) {
		return mRefCount++;
	}
	virtual ULONG STDMETHODCALLTYPE  Release(void) {
		int32_t newRefValue;
        	
		newRefValue = mRefCount--;
		if (newRefValue == 0)
		{
			delete this;
			return 0;
		}
        	return newRefValue;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(
                        BMDVideoInputFormatChangedEvents notificationEvents,
                        IDeckLinkDisplayMode* mode,
                        BMDDetectedVideoInputFormatFlags flags) override {
                BMDPixelFormat pf;
                HRESULT result;

                string reason{};
                if ((notificationEvents & bmdVideoInputDisplayModeChanged) != 0u) {
                        reason = "display mode";
                }
                if ((notificationEvents & bmdVideoInputFieldDominanceChanged) != 0u) {
                        if (!reason.empty()) {
                                reason += ", ";
                        }
                        reason += "field dominance";
                }
                if ((notificationEvents & bmdVideoInputColorspaceChanged) != 0u) {
                        if (!reason.empty()) {
                                reason += ", ";
                        }
                        reason += "color space";
                }
                if (reason.empty()) {
                        reason = "unknown";
                }
                LOG(LOG_LEVEL_NOTICE) << MODULE_NAME << "Format change detected (" << reason << ").\n";

                unique_lock<mutex> lk(s->lock);
		if ((flags & bmdDetectedVideoInputDualStream3D) != 0u && !s->stereo) {
			LOG(LOG_LEVEL_ERROR) << MODULE_NAME <<  "Stereoscopic 3D detected but not enabled! Please supply a \"3D\" parameter.\n";
			return E_FAIL;
		}
                if ((flags & bmdDetectedVideoInputYCbCr422) != 0u) {
                        unordered_map<int, codec_t> m = {{8, UYVY}, {10, v210}, {12, v210}};
                        if (s->requested_bit_depth == 12) {
                                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Using 10-bit YCbCr.\n";
                        }
                        s->codec = m.at(s->requested_bit_depth);
                } else if ((flags & bmdDetectedVideoInputRGB444) != 0u) {
                        unordered_map<int, codec_t> m = {{8, RGBA}, {10, R10k}, {12, R12L}};
                        s->codec = m.at(s->requested_bit_depth);
                } else {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME <<  "Unhandled flag!\n";
                        abort();
                }
                LOG(LOG_LEVEL_INFO) << MODULE_NAME "Using codec: " << get_codec_name(s->codec) << "\n";
                IDeckLinkInput *deckLinkInput = s->state[this->i].deckLinkInput;
                deckLinkInput->DisableVideoInput();
                deckLinkInput->StopStreams();
                deckLinkInput->FlushStreams();
                result = set_display_mode_properties(s, vf_get_tile(s->frame, this->i), mode, /* out */ &pf);
                if(result == S_OK) {
                        CALL_AND_CHECK(deckLinkInput->EnableVideoInput(mode->GetDisplayMode(), pf, s->flags), "EnableVideoInput");
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
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "set_display_mode_properties: " << bmd_hresult_to_string(result) << "\n";\
                }

                return result;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);
};

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

        if (audioPacket) {
                audioPacket->AddRef();
                s->audioPackets.push(audioPacket);
        }

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
	cout << style::bold << fg::red << "\t-t decklink" << fg::reset << "[:<device_index(indices)>[:<mode>:<colorspace>[:3D][:sync_timecode][:connection=<input>][:audio_level={line|mic}][:detect-format][:conversion=<conv_mode>]]\n" << style::reset;
        printf("\t\tor\n");
	cout << style::bold << fg::red << "\t-t decklink" << fg::reset << "{:mode=<mode>|:device=<device_index>|:codec=<colorspace>...<key>=<val>}*\n" << style::reset;
	printf("\t(Mode specification is mandatory if your card does not support format autodetection.)\n");
        printf("\n");

        printf("Available color spaces:\n");
        for (auto & i : uv_to_bmd_codec_map) {
                cout << "\t" << style::bold << get_codec_name(i.first)
                        << style::reset << "\n";
        }
        printf("\n");

        cout << style::bold << "3D" << style::reset << "\n";
        printf("\tUse this to capture 3D from supported card (eg. DeckLink HD 3D Extreme).\n");
        printf("\tDo not use it for eg. Quad or Duo. Availability of the mode is indicated\n");
        printf("\tin video format listing above (\"supports 3D\").\n");
	printf("\n");

        cout << style::bold << "sync_timecode" << style::reset << "\n";
        printf("\tTry to synchronize inputs based on timecode (for multiple inputs, eg. tiled 4K)\n");
	printf("\n");

        cout << style::bold << "audio_level\n" << style::reset;
        cout << style::bold << "\tline" << style::reset << " - the selected analog input gain levels are used\n";
        cout << style::bold << "\tmic" << style::reset << " - analog audio levels are set to maximum gain on audio input.\n";
	printf("\n");

        cout << style::bold << "conversion\n" << style::reset;
        cout << style::bold << "\tnone" << style::reset << " - No video input conversion\n";
        cout << style::bold << "\t10lb" << style::reset << " - HD1080 to SD video input down conversion\n";
        cout << style::bold << "\t10am" << style::reset << " - Anamorphic from HD1080 to SD video input down conversion\n";
        cout << style::bold << "\t72lb" << style::reset << " - Letter box from HD720 to SD video input down conversion\n";
        cout << style::bold << "\t72ab" << style::reset << " - Letterbox video input up conversion\n";
        cout << style::bold << "\tamup" << style::reset << " - Anamorphic video input up conversion\n";
        printf("\tThen use the set the resulting mode (!) for capture, eg. for 1080p to PAL conversion:\n"
               "\t\t-t decklink:mode=pal:conversion=10lb\n");
	printf("\n");

        cout << style::bold << "detect-format\n" << style::reset;
        printf("\tTry to detect input video format even if the device doesn't support autodetect.\n");
        printf("\tSource interface still has to be given, eg. \"-t decklink:connection=HDMI:detect-format\".\n");
        cout << style::bold << "p_not_i\n" << style::reset;
        printf("\tIncoming signal should be treated as progressive even if detected as interlaced (PsF).\n");
        printf("\n");
        cout << style::bold << "[no]passthrough\n" << style::reset;
        printf("\tEnable/disable capture passthrough.\n");
	printf("\n");
        cout << style::bold << "half-duplex|full-duplex|no-half-duplex\n" << style::reset;
        printf("\tUse half-/full-duplex, no-half-duplex suppresses automatically set half-duplex (for quad-link)\n");
        printf("\n");
        cout << style::bold << "single-/dual-/quad-link\n" << style::reset;
        printf("\tUse single-/dual-/quad-link.\n");
        printf("\n");

	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
	deckLinkIterator = create_decklink_iterator();
	if (deckLinkIterator == NULL) {
		return 0;
	}
	
	// Enumerate all cards in this system
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
		BMD_STR                 deviceNameString = NULL;
		char *		deviceNameCString = NULL;
		
		// *** Print the model name of the DeckLink card
		result = deckLink->GetDisplayName((BMD_STR *) &deviceNameString);
		if (result == S_OK)
		{
                        deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
			cout << "device: " << style::bold << numDevices << style::reset << ") " << style::bold <<  deviceNameCString << style::reset << "\n";
			release_bmd_api_str(deviceNameString);
                        free(deviceNameCString);
                } else {
			printf("device: %d) (unable to get name)\n", numDevices);
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
                                cout << "\tConnection can be one of following:\n";
                                for (auto it : connection_string_map) {
                                        if (connections & it.first) {
                                                cout << style::bold << "\t\t" <<
                                                        it.second << style::reset << "\n";
                                        }
                                }
                        }
                }
				
                printf("\n");

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

        printf("Examples:\n");
        cout << "\t" << style::bold << uv_argv[0] << " -t decklink" << style::reset << " # captures autodetected video from first DeckLink in system\n";
        cout << "\t" << style::bold << uv_argv[0] << " -t decklink:0:Hi50:UYVY" << style::reset << " # captures 1080i50, 8-bit yuv\n";
        cout << "\t" << style::bold << uv_argv[0] << " -t decklink:0:10:v210:connection=HDMI" << style::reset << " # captures 10th format from a card (alternative syntax), 10-bit YUV, from HDMI\n";
        cout << "\t" << style::bold << uv_argv[0] << " -t decklink:mode=23ps" << style::reset << " # captures 1080p24, 8-bit yuv from first device\n";
        cout << "\t" << style::bold << uv_argv[0] << " -t \"decklink:mode=Hp30:codec=v210:device=DeckLink HD Extreme 3D+\"" << style::reset << " # captures 1080p30, 10-bit yuv from DeckLink HD Extreme 3D+\n";

	printf("\n");

        print_decklink_version();

        printf("\n");

	return 1;
}

/* SETTINGS */
static void parse_devices(struct vidcap_decklink_state *s, const char *devs)
{
        assert(devs != NULL && strlen(devs) > 0);
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
                s->stereo = true;
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
        } else if(strncasecmp(opt, "audio_level=",
                                strlen("audio_level=")) == 0) {
                const char *levels = opt + strlen("audio_level=");
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
        } else if (strcasecmp(opt, "detect-format") == 0) {
                s->detect_format = true;
        } else if (strcasecmp(opt, "p_not_i") == 0) {
                s->p_not_i = true;
        } else if (strcasecmp(opt, "passthrough") == 0 || strcasecmp(opt, "nopassthrough") == 0) {
                s->passthrough = opt[0] == 'n' ? bmdDeckLinkCapturePassthroughModeDisabled
                        : bmdDeckLinkCapturePassthroughModeCleanSwitch;
        } else if (strcasecmp(opt, "half-duplex") == 0) {
                s->duplex = bmdDuplexModeHalf;
        } else if (strcasecmp(opt, "full-duplex") == 0) {
                s->duplex = bmdDuplexModeFull;
        } else if (strcasecmp(opt, "single-link") == 0) {
                s->link = bmdLinkConfigurationSingleLink;
        } else if (strcasecmp(opt, "dual-link") == 0) {
                s->link = bmdLinkConfigurationDualLink;
        } else if (strcasecmp(opt, "quad-link") == 0) {
                s->link = bmdLinkConfigurationQuadLink;
        } else {
                log_msg(LOG_LEVEL_WARNING, "[DeckLink] Warning, unrecognized trailing options in init string: %s\n", opt);
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

        // options are in format <device>:<mode>:<codec>[:other_opts]
        if (isdigit(tmp[0]) && strcasecmp(tmp, "3D") != 0) {
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

static struct vidcap_type *vidcap_decklink_probe(bool verbose)
{
        auto vt = static_cast<struct vidcap_type *>(calloc(1, sizeof(struct vidcap_type)));
        if (vt == nullptr) {
                return nullptr;
        }

        vt->name        = "decklink";
        vt->description = "Blackmagic DeckLink card";

        if (!verbose) {
                return vt;
        }

        IDeckLinkIterator*		deckLinkIterator;
        IDeckLink*			deckLink;
        int				numDevices = 0;

        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = create_decklink_iterator(false);
        if (deckLinkIterator == nullptr) {
                return vt;
        }

        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK) {
                HRESULT result;
                IDeckLinkAttributes *deckLinkAttributes;

                result = deckLink->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes);
                if (result != S_OK) {
                        continue;
                }
                int64_t connections_bitmap;
                if(deckLinkAttributes->GetInt(BMDDeckLinkVideoInputConnections, &connections_bitmap) != S_OK) {
                        fprintf(stderr, "[DeckLink] Could not get connections.\n");
                        continue;
                }

                BMD_STR deviceNameBMDString = NULL;
                // *** Print the model name of the DeckLink card
                result = deckLink->GetModelName((BMD_STR *) &deviceNameBMDString);
                if (result != S_OK) {
                        continue;
                }
                string deviceName = get_cstr_from_bmd_api_str(deviceNameBMDString);
                release_bmd_api_str(deviceNameBMDString);

                vt->card_count += 1;
                vt->cards = (struct device_info *)
                        realloc(vt->cards, vt->card_count * sizeof(struct device_info));
                memset(&vt->cards[vt->card_count - 1], 0, sizeof(struct device_info));
                snprintf(vt->cards[vt->card_count - 1].id, sizeof vt->cards[vt->card_count - 1].id,
                                "%d", numDevices);
                snprintf(vt->cards[vt->card_count - 1].name, sizeof vt->cards[vt->card_count - 1].name,
                                "%s #%d", deviceName.c_str(), numDevices);


                list<string> connections;
                for (auto it : connection_string_map) {
                        if (connections_bitmap & it.first) {
                                connections.push_back(it.second);
                        }
                }
                list<tuple<int, string, string, string>> modes = get_input_modes (deckLink);
                int i = 0;
                for (auto &m : modes) {
                        for (auto &c : connections) {
                                if (i >= (int) (sizeof vt->cards[vt->card_count - 1].modes /
                                                sizeof vt->cards[vt->card_count - 1].modes[0])) { // no space
                                        break;
                                }

                                snprintf(vt->cards[vt->card_count - 1].modes[i].id,
                                                sizeof vt->cards[vt->card_count - 1].modes[i].id,
                                                "{\"modeOpt\":\"connection=%s:mode=%s\"}",
                                                c.c_str(), get<1>(m).c_str());
                                snprintf(vt->cards[vt->card_count - 1].modes[i].name,
                                                sizeof vt->cards[vt->card_count - 1].modes[i].name,
                                                "%s (%s)", get<2>(m).c_str(), c.c_str());
                                i++;
                        }
                }

                if (i < (int) (sizeof vt->cards[vt->card_count - 1].modes /
                                        sizeof vt->cards[vt->card_count - 1].modes[0])) {
                        snprintf(vt->cards[vt->card_count - 1].modes[i].id,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].id,
                                        "{\"modeOpt\":\"detect-format\"}");
                        snprintf(vt->cards[vt->card_count - 1].modes[i].name,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].name,
                                        "UltraGrid auto-detect");
                        i++;
                }

                // Increment the total number of DeckLink cards found
                numDevices++;

                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        deckLinkIterator->Release();
        decklink_uninitialize();

        return vt;
}

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf)
{
        BMD_STR displayModeString = NULL;
        char *displayModeCString;
        HRESULT result;

        result = displayMode->GetName(&displayModeString);
        if (result == S_OK)
        {
                auto it = uv_to_bmd_codec_map.find(s->codec);
                if (it == uv_to_bmd_codec_map.end()) {
                        LOG(LOG_LEVEL_ERROR) << "Unsupported codec: " <<  get_codec_name(s->codec) << "!\n";
                        return E_FAIL;
                }
                *pf = it->second;

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
                        case bmdUnknownFieldDominance:
				LOG(LOG_LEVEL_WARNING) << "[DeckLink cap.] Unknown field dominance!\n";
                                s->frame->interlacing = PROGRESSIVE;
                                break;
                }

                if (s->p_not_i) {
                        s->frame->interlacing = PROGRESSIVE;
                }

                displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                debug_msg("%-20s \t %d x %d \t %g FPS \t %d AVAREGE TIME BETWEEN FRAMES\n", displayModeCString,
                                tile->width, tile->height, s->frame->fps, s->next_frame_time); /* TOREMOVE */
                printf("Enable video input: %s\n", displayModeCString);
                release_bmd_api_str(displayModeString);
                free(displayModeCString);
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

/**
 * This function is used when device does not support autodetection and user
 * request explicitly to detect the format (:detect-format)
 */
static bool detect_format(struct vidcap_decklink_state *s, BMDDisplayMode *outDisplayMode, int card_idx)
{
        IDeckLinkDisplayMode *displayMode;
        HRESULT result;
        IDeckLinkDisplayModeIterator*	displayModeIterator = NULL;
        result = s->state[card_idx].deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
        if (result != S_OK) {
                return false;
        }

        vector<BMDPixelFormat> pfs = {bmdFormat8BitYUV, bmdFormat8BitBGRA};

        while (displayModeIterator->Next(&displayMode) == S_OK) {
                for (BMDPixelFormat pf : pfs) {
                        uint32_t mode = ntohl(displayMode->GetDisplayMode());
                        log_msg(LOG_LEVEL_NOTICE, "DeckLink: trying mode %.4s, pixel format %.4s\n", (const char *) &mode, (const char *) &pf);
                        result = s->state[card_idx].deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, 0);
                        if (result == S_OK) {
                                s->state[card_idx].deckLinkInput->StartStreams();
                                unique_lock<mutex> lk(s->lock);
                                s->boss_cv.wait_for(lk, chrono::milliseconds(1200), [s, card_idx]{return s->state[card_idx].delegate->newFrameReady;});
                                lk.unlock();
                                s->state[card_idx].deckLinkInput->StopStreams();
                                s->state[card_idx].deckLinkInput->DisableVideoInput();

                                if (s->state[card_idx].delegate->newFrameReady) {
                                        *outDisplayMode = displayMode->GetDisplayMode();
                                        // set also detected codec (!)
                                        s->codec = pf == bmdFormat8BitYUV ? UYVY : RGBA;
                                        displayMode->Release();
                                        displayModeIterator->Release();
                                        return true;
                                }
                        }
                }

                displayMode->Release();
        }

        displayModeIterator->Release();
        return false;
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

        if(strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                decklink_help();
                return VIDCAP_INIT_NOERR;
        }

        if (!blackmagic_api_version_check()) {
                return VIDCAP_INIT_FAIL;
        }

	s = new vidcap_decklink_state();
	if (s == NULL) {
		LOG(LOG_LEVEL_ERROR) << "Unable to allocate DeckLink state\n";
		return VIDCAP_INIT_FAIL;
	}

        gettimeofday(&s->t0, NULL);

        s->stereo = false;
        s->sync_timecode = FALSE;
        s->connection = (BMDVideoConnection) 0;
        s->flags = 0;
        s->audio_consumer_levels = -1;
        s->conversion_mode = (BMDVideoInputConversionMode) 0;

	// SET UP device and mode
        char *tmp_fmt = strdup(vidcap_params_get_fmt(params));
        int ret = settings_init(s, tmp_fmt);
        free(tmp_fmt);
	if (!ret) {
                delete s;
		return VIDCAP_INIT_FAIL;
	}

	if (s->link == bmdLinkConfigurationQuadLink) {
		if (s->duplex == bmdDuplexModeFull) {
			LOG(LOG_LEVEL_WARNING) << MOD_NAME "Setting quad-link and full-duplex may not be supported!\n";
		}
		if (s->duplex == 0) {
			LOG(LOG_LEVEL_WARNING) << MOD_NAME "Quad-link detected - setting half-duplex automatically, use 'no-half-duplex' to override.\n";
			s->duplex = bmdDuplexModeHalf;
		}
	}

        s->requested_bit_depth = get_bits_per_component(s->codec);
        assert(s->requested_bit_depth == 8 || s->requested_bit_depth == 10
                        || s->requested_bit_depth == 12);

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
                s->audio.max_size = (s->audio.sample_rate / 10) * s->audio.ch_count * s->audio.bps;
                s->audio.data = (char *) malloc(s->audio.max_size);
        } else {
                s->grab_audio = FALSE;
        }

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
        for (int i = 0; i < s->devices_cnt; ++i) {
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
                bool found = false;
                BMD_STR deviceNameString = NULL;
                while (deckLinkIterator->Next(&deckLink) == S_OK) {
                        char* deviceNameCString = NULL;

                        result = deckLink->GetDisplayName(&deviceNameString);
                        if (result == S_OK) {
                                deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);

                                if (strcmp(deviceNameCString, s->state[i].device_id.c_str()) == 0) {
                                        found = true;
                                }

                                release_bmd_api_str(deviceNameString);
                                free(deviceNameCString);
                        }

                        if (isdigit(s->state[i].device_id.c_str()[0]) && atoi(s->state[i].device_id.c_str()) == dnum) {
                                found = true;
                        }

                        if (found) {
                                break;
                        }
                        dnum++;
                        // Release the IDeckLink instance when we've finished with it to prevent leaks
                        deckLink->Release();
                        deckLink = NULL;
                }
                deckLinkIterator->Release();
                deckLinkIterator = NULL;

                if (!found) {
                        LOG(LOG_LEVEL_ERROR) << "Device " << s->state[i].device_id << " was not found.\n";
                        goto error;
                }

                s->state[i].deckLink = deckLink;

                // Print the model name of the DeckLink card
                result = deckLink->GetDisplayName(&deviceNameString);
                if (result == S_OK) {
                        char *deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
                        LOG(LOG_LEVEL_INFO) << "Using device " << deviceNameCString << "\n";
                        release_bmd_api_str(deviceNameString);
                        free(deviceNameCString);
                }

                // Query the DeckLink for its configuration interface
                EXIT_IF_FAILED(deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput), "Could not obtain the IDeckLinkInput interface");
                s->state[i].deckLinkInput = deckLinkInput;

                // Query the DeckLink for its configuration interface
                EXIT_IF_FAILED(deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration), "Could not obtain the IDeckLinkConfiguration interface");
                s->state[i].deckLinkConfiguration = deckLinkConfiguration;

                if(s->connection) {
                        CALL_AND_CHECK(deckLinkConfiguration->SetInt(bmdDeckLinkConfigVideoInputConnection, s->connection),
                                        "bmdDeckLinkConfigVideoInputConnection",
                                        "Input set to: " << s->connection);
                }

                if (s->conversion_mode) {
                        EXIT_IF_FAILED(deckLinkConfiguration->SetInt(bmdDeckLinkConfigVideoInputConversionMode, s->conversion_mode), "Unable to set conversion mode");
                }

                if (s->passthrough) {
                        EXIT_IF_FAILED(deckLinkConfiguration->SetInt(bmdDeckLinkConfigCapturePassThroughMode, s->passthrough), "Unable to set passthrough mode");
                }

                if (s->duplex != 0 && s->duplex != (uint32_t) -1) {
                        CALL_AND_CHECK(deckLinkConfiguration->SetInt(bmdDeckLinkConfigDuplexMode, s->duplex), "Unable set output SDI duplex mode");
                }

                if (s->link == 0) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Setting single link by default.\n";
                        s->link = bmdLinkConfigurationSingleLink;
                }
                CALL_AND_CHECK( deckLinkConfiguration->SetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, s->link), "Unable set output SDI link mode");

                // set Callback which returns frames
                s->state[i].delegate = new VideoDelegate(s, i);
                deckLinkInput->SetCallback(s->state[i].delegate);

                BMDDisplayMode detectedDisplayMode;
                if (s->detect_format) {
                        if (!detect_format(s, &detectedDisplayMode, i)) {
                                LOG(LOG_LEVEL_WARNING) << "Signal could have not been detected!\n";
                                goto error;
                        }
                }

                // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
                EXIT_IF_FAILED(deckLinkInput->GetDisplayModeIterator(&displayModeIterator),
                                "Could not obtain the video input display mode iterator:");

                mnum = 0;
                bool mode_found = false;
#define MODE_SPEC_AUTODETECT -1
#define MODE_SPEC_FOURCC -2
                int mode_idx = MODE_SPEC_AUTODETECT;

                // mode selected manually - either by index or FourCC
                if (s->mode.length() > 0) {
                        if (s->mode.length() <= 2) {
                                mode_idx = atoi(s->mode.c_str());
                        } else {
                                mode_idx = MODE_SPEC_FOURCC;
                        }
                }

                while (displayModeIterator->Next(&displayMode) == S_OK) {
                        if (s->detect_format) { // format already detected manually
                                if (detectedDisplayMode == displayMode->GetDisplayMode()) {
                                        mode_found = true;
                                        break;
                                } else {
                                        displayMode->Release();
                                }
                        } else if (mode_idx == MODE_SPEC_AUTODETECT) { // autodetect, pick first eligible mode and let device autodetect
                                if (s->stereo && (displayMode->GetFlags() & bmdDisplayModeSupports3D) == 0u) {
                                        displayMode->Release();
                                        continue;
                                }
                                auto it = uv_to_bmd_codec_map.find(s->codec);
                                if (it == uv_to_bmd_codec_map.end()) {
                                        LOG(LOG_LEVEL_ERROR) << "Unsupported codec: " <<  get_codec_name(s->codec) << "!\n";
                                        goto error;
                                }
                                BMDPixelFormat pf = it->second;
                                BMDDisplayModeSupport             supported;
                                EXIT_IF_FAILED(deckLinkInput->DoesSupportVideoMode(displayMode->GetDisplayMode(), pf, s->flags, &supported, NULL), "DoesSupportVideoMode");
                                if (supported == bmdDisplayModeSupported) {
                                        break;
                                }
                        } else if (mode_idx != MODE_SPEC_FOURCC) { // manually given idx
                                if (mode_idx != mnum) {
                                        mnum++;
                                        // Release the IDeckLinkDisplayMode object to prevent a leak
                                        displayMode->Release();
                                        continue;
                                }

                                mode_found = true;
                                mnum++;
                                break;
                        } else { // manually given FourCC
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
                                displayMode->Release();
                        }
                }

                if (mode_found) {
                        BMD_STR displayModeString = NULL;
                        result = displayMode->GetName(&displayModeString);
                        if (result == S_OK) {
                                char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                                LOG(LOG_LEVEL_INFO) << "The desired display mode is supported: " << displayModeCString << "\n";
                                release_bmd_api_str(displayModeString);
                                free(displayModeCString);
                        }
                } else {
                        if (mode_idx == MODE_SPEC_FOURCC) {
                                log_msg(LOG_LEVEL_ERROR, "Desired mode \"%s\" is invalid or not supported.\n", s->mode.c_str());
                                goto error;
                        } else if (mode_idx >= 0) {
                                log_msg(LOG_LEVEL_ERROR, "Desired mode index %s is out of bounds.\n", s->mode.c_str());
                                goto error;
                        }
                }

                BMDPixelFormat pf;
                EXIT_IF_FAILED(set_display_mode_properties(s, tile, displayMode, &pf),
                                "Could not set display mode properties");

                IDeckLinkAttributes *deckLinkAttributes;
                deckLinkInput->StopStreams();

                EXIT_IF_FAILED(deckLinkInput->QueryInterface(IID_IDeckLinkAttributes, (void**)&deckLinkAttributes), "Could not query device attributes");

                if (!mode_found) {
                        log_msg(LOG_LEVEL_INFO, "[DeckLink] Trying to autodetect format.\n");
                        BMD_BOOL autodetection;
                        EXIT_IF_FAILED(deckLinkAttributes->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &autodetection), "Could not verify if device supports autodetection");
                        if (autodetection == BMD_FALSE) {
                                log_msg(LOG_LEVEL_ERROR, "[DeckLink] Device doesn't support format autodetection, you must set it manually or try \"-t decklink:detect-format[:connection=<in>]\"\n");
                                goto error;
                        }
                        s->flags |=  bmdVideoInputEnableFormatDetection;
                }

                if (s->stereo) {
                        s->flags |= bmdVideoInputDualStream3D;
                }
                BMDDisplayModeSupport             supported;
                EXIT_IF_FAILED(deckLinkInput->DoesSupportVideoMode(displayMode->GetDisplayMode(), pf, s->flags, &supported, NULL), "DoesSupportVideoMode");

                if (supported == bmdDisplayModeNotSupported) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Requested display mode not supported wit the selected pixel format\n";
                        goto error;
                }

                result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, s->flags);
                if (result != S_OK) {
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

                if (s->grab_audio == FALSE ||
                                i != 0) { //TODO: figure out output from multiple streams
                        deckLinkInput->DisableAudioInput();
                } else {
                        if (deckLinkConfiguration->SetInt(bmdDeckLinkConfigAudioInputConnection,
                                                audioConnection) == S_OK) {
                                const map<BMDAudioConnection, string> mapping = {
                                        { bmdAudioConnectionEmbedded, "embedded" },
                                        { bmdAudioConnectionAESEBU, "AES/EBU" },
                                        { bmdAudioConnectionAnalog, "analog" },
                                        { bmdAudioConnectionAnalogXLR, "analogXLR" },
                                        { bmdAudioConnectionAnalogRCA, "analogRCA" },
                                        { bmdAudioConnectionMicrophone, "microphone" },
                                        { bmdAudioConnectionHeadphones, "headphones" },
                                };
                                printf("[Decklink capture] Audio input set to: %s\n", mapping.find(audioConnection) != mapping.end() ? mapping.at(audioConnection).c_str() : "unknown");
                        } else {
                                fprintf(stderr, "[Decklink capture] Unable to set audio input!!! Please check if it is OK. Continuing anyway.\n");

                        }
                        if (audio_capture_channels != 1 &&
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
                        CALL_AND_CHECK(deckLinkInput->EnableAudioInput(
                                                bmdAudioSampleRate48kHz,
                                                s->audio.bps == 2 ? bmdAudioSampleType16bitInteger : bmdAudioSampleType32bitInteger,
                                                audio_capture_channels == 1 ? 2 : audio_capture_channels),
                                        "EnableAudioInput",
                                        "Decklink audio capture initialized sucessfully: " << audio_desc_from_frame(&s->audio));
                }

                // Start streaming
                printf("Start capture\n");
                EXIT_IF_FAILED(deckLinkInput->StartStreams(), "Could not start stream");

                displayMode->Release();
                displayMode = NULL;

                displayModeIterator->Release();
                displayModeIterator = NULL;
        }

	printf("DeckLink capture device enabled\n");

	debug_msg("vidcap_decklink_init - END\n"); /* TOREMOVE */

        *state = s;
	return VIDCAP_INIT_OK;

error:
	if(displayMode != NULL) {
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

static audio_frame *process_new_audio_packets(struct vidcap_decklink_state *s) {
        if (!s->audioPackets.empty()) {
                s->audio.data_len = 0;
                while (!s->audioPackets.empty()) {
                        auto audioPacket = s->audioPackets.front();
                        s->audioPackets.pop();

                        void *audioFrame;
                        audioPacket->GetBytes(&audioFrame);

                        if(audio_capture_channels == 1) { // ther are actually 2 channels grabbed
                                if (s->audio.data_len + audioPacket->GetSampleFrameCount() * 1u * s->audio.bps <= s->audio.max_size) {
                                        demux_channel(s->audio.data + s->audio.data_len, (char *) audioFrame, s->audio.bps, audioPacket->GetSampleFrameCount() * 2 /* channels */ * s->audio.bps, 2 /* channels (originally) */, 0 /* we want first channel */);
                                        s->audio.data_len += audioPacket->GetSampleFrameCount() * 1 * s->audio.bps;
                                } else {
                                        LOG(LOG_LEVEL_WARNING) << "[DeckLink] Audio frame too small!\n";
                                }
                        } else {
                                if (s->audio.data_len + audioPacket->GetSampleFrameCount() * audio_capture_channels * s->audio.bps <= s->audio.max_size) {
                                        memcpy(s->audio.data + s->audio.data_len, audioFrame, audioPacket->GetSampleFrameCount() * audio_capture_channels * s->audio.bps);
                                        s->audio.data_len += audioPacket->GetSampleFrameCount() * audio_capture_channels * s->audio.bps;
                                } else {
                                        LOG(LOG_LEVEL_WARNING) << "[DeckLink] Audio frame too small!\n";
                                }
                        }
                        audioPacket->Release();
                }
                return &s->audio;
        } else {
                return NULL;
        }
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

                lk.lock();
                *audio = process_new_audio_packets(s);
                lk.unlock();

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
static list<tuple<int, string, string, string>> get_input_modes (IDeckLink* deckLink)
{
        list<tuple<int, string, string, string>> ret;
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
	while (displayModeIterator->Next(&displayMode) == S_OK)
	{
		BMD_STR displayModeString = NULL;

		result = displayMode->GetName((BMD_STR *) &displayModeString);

		if (result == S_OK)
		{
			int				modeWidth;
			int				modeHeight;
                        BMDDisplayModeFlags             flags;
			BMDTimeValue	frameRateDuration;
			BMDTimeScale	frameRateScale;

                        char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
			// Obtain the display mode's properties
                        flags = displayMode->GetFlags();
			modeWidth = displayMode->GetWidth();
			modeHeight = displayMode->GetHeight();
			displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
                        uint32_t mode = ntohl(displayMode->GetDisplayMode());
                        string fcc{(char *) &mode, 4};
                        string name{displayModeCString};
                        char buf[1024];
                        snprintf(buf, sizeof buf, "%d x %d \t %2.2f FPS%s", modeWidth, modeHeight,
                                        (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (flags & bmdDisplayModeSupports3D ? "\t (supports 3D)" : ""));
                        string details{buf};
                        ret.push_back(tuple<int, string, string, string> {displayModeNumber, fcc, name, details});

                        release_bmd_api_str(displayModeString);
			free(displayModeCString);
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

        return ret;
}

static void print_input_modes (IDeckLink* deckLink)
{
        list<tuple<int, string, string, string>> ret = get_input_modes (deckLink);
	printf("\tcapture modes:\n");
        for (auto &i : ret) {
                cout << "\t\t" << right << style::bold << setw(2) << get<0>(i) <<
                        " (" << get<1>(i) << ")" << style::reset  << ") " <<
                        left << setw(20) << get<2>(i) << internal << "  " <<
                        get<3>(i) << "\n";
        }
}

static const struct video_capture_info vidcap_decklink_info = {
        vidcap_decklink_probe,
        vidcap_decklink_init,
        vidcap_decklink_done,
        vidcap_decklink_grab,
};

REGISTER_MODULE(decklink, &vidcap_decklink_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

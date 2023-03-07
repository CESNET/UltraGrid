/*
 * FILE:    video_capture/decklink.cpp
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Martin Pulec     <martin.pulec@cesnet.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "blackmagic_common.hpp"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture.h"

constexpr const int DEFAULT_AUDIO_BPS = 4;
constexpr const size_t MAX_AUDIO_PACKETS = 10;
#define MOD_NAME "[DeckLink capture] "

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

#define RELEASE_IF_NOT_NULL(x) if (x != nullptr) { x->Release(); x = nullptr; }

using namespace std;
using namespace std::chrono;
using std::mutex;

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
        IDeckLink                  *deckLink              = nullptr;
        IDeckLinkInput             *deckLinkInput         = nullptr;
        unique_ptr<VideoDelegate>  delegate;
        IDeckLinkProfileAttributes *deckLinkAttributes    = nullptr;
        IDeckLinkConfiguration     *deckLinkConfiguration = nullptr;
        string                      device_id = "0"; // either numeric value or device name
        bool                        audio                 = false; /* wheather we process audio or not */
        struct tile                *tile                  = nullptr;
        bool init(struct vidcap_decklink_state *s, struct tile *tile, BMDAudioConnection audioConnection);
        void check_attributes(struct vidcap_decklink_state *s);
};

struct vidcap_decklink_state {
        bool                    com_initialized = true;
        vector <struct device_state>     state{vector <struct device_state>(1)};
        int                     devices_cnt = 1;
        string                  mode;
        unsigned int            next_frame_time = 0; // avarege time between frames
        struct video_frame     *frame{nullptr};
        struct audio_frame      audio{};
        queue<IDeckLinkAudioInputPacket *> audioPackets;
        codec_t                 codec{VIDEO_CODEC_NONE};
        BMDVideoInputFlags       enable_flags      = bmdVideoInputFlagDefault;
        BMDSupportedVideoModeFlags supported_flags = bmdSupportedVideoModeDefault;

        mutex                   lock;
	condition_variable      boss_cv;

        int                     frames = 0;
        bool                    stereo{false}; /* for eg. DeckLink HD Extreme, Quad doesn't set this !!! */
        bool                    sync_timecode{false}; /* use timecode when grabbing from multiple inputs */
        static_assert(bmdVideoConnectionUnspecified == BMD_OPT_DEFAULT, "Connection unspecified is not 0!");
        BMDVideoConnection      connection{bmdVideoConnectionUnspecified};
        int                     audio_consumer_levels{-1}; ///< 0 false, 1 true, -1 default
        BMDVideoInputConversionMode conversion_mode{};
        BMDDeckLinkCapturePassthroughMode passthrough{bmdDeckLinkCapturePassthroughModeDisabled};

        bool                    detect_format = false;
        unsigned int            requested_bit_depth = 0; // 0, bmdDetectedVideoInput8BitDepth, bmdDetectedVideoInput10BitDepth or bmdDetectedVideoInput12BitDepth
        bool                    p_not_i = false;
        int                     use1080psf = BMD_OPT_KEEP; // capture PsF instead of progressive

        uint32_t                profile{}; // BMD_OPT_DEFAULT, BMD_OPT_KEEP, bmdDuplexHalf or one of BMDProfileID
        bool                    nosig_send = false; ///< send video even when no signal detected
        bool                    keep_device_defaults = false;

        void set_codec(codec_t c);

        vidcap_decklink_state() {
                if (!decklink_initialize(&com_initialized)) {
                        throw 1;
                }
        }
        ~vidcap_decklink_state() {
                decklink_uninitialize(&com_initialized);
        }
};

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf);
static void cleanup_common(struct vidcap_decklink_state *s);
static list<tuple<int, string, string, string>> get_input_modes (IDeckLink* deckLink);
static void print_input_modes (IDeckLink* deckLink);

class VideoDelegate : public IDeckLinkInputCallback {
private:
	int32_t                       mRefCount{};
        static constexpr BMDDetectedVideoInputFormatFlags csMask{bmdDetectedVideoInputYCbCr422 | bmdDetectedVideoInputRGB444};
        static constexpr BMDDetectedVideoInputFormatFlags bitDepthMask{bmdDetectedVideoInput8BitDepth | bmdDetectedVideoInput10BitDepth | bmdDetectedVideoInput12BitDepth};
        BMDDetectedVideoInputFormatFlags configuredCsBitDepth{};

public:
        int	                      newFrameReady{};
        IDeckLinkVideoFrame          *rightEyeFrame{};
        void                         *pixelFrame{};
        void                         *pixelFrameRight{};
        IDeckLinkVideoInputFrame     *lastFrame{nullptr};
        uint32_t                      timecode{};
        struct vidcap_decklink_state *s;
        struct device_state          &device;
	
        VideoDelegate(struct vidcap_decklink_state *state, struct device_state &device_) : s(state), device(device_) {
        }
	
        virtual ~VideoDelegate () {
                RELEASE_IF_NOT_NULL(rightEyeFrame);
                RELEASE_IF_NOT_NULL(lastFrame);
	}

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID *) override { return E_NOINTERFACE; }
	virtual ULONG STDMETHODCALLTYPE  AddRef(void) override {
		return mRefCount++;
	}
	virtual ULONG STDMETHODCALLTYPE  Release(void) override {
		int32_t newRefValue = mRefCount--;
		if (newRefValue == 0)
		{
			delete this;
			return 0;
		}
        	return newRefValue;
	}
        static auto getNotificationEventsStr(BMDVideoInputFormatChangedEvents notificationEvents, BMDDetectedVideoInputFormatFlags flags) noexcept {
                string status{};
                map<BMDDetectedVideoInputFormatFlags, string> change_map {
                        { bmdVideoInputDisplayModeChanged, "display mode"s },
                        { bmdVideoInputFieldDominanceChanged, "field dominance"s },
                        { bmdVideoInputColorspaceChanged, "color space"s },
                };
                for (auto &i : change_map) {
                        if ((notificationEvents & i.first) != 0U) {
                                if (!status.empty()) {
                                        status += ", "s;
                                }
                                status += i.second;
                        }
                }
                if (status.empty()) {
                        status = "unknown"s;
                }
                status += " - ";
                map<BMDDetectedVideoInputFormatFlags, string> flag_map {
                        { bmdDetectedVideoInputYCbCr422, "YCbCr422"s },
                        { bmdDetectedVideoInputRGB444, "RGB444"s },
                        { bmdDetectedVideoInputDualStream3D, "DualStream3D"s },
                        { bmdDetectedVideoInput12BitDepth, "12bit"s },
                        { bmdDetectedVideoInput10BitDepth, "10bit"s },
                        { bmdDetectedVideoInput8BitDepth, "8bit"s },
                };
                bool first = true;
                for (auto &i : flag_map) {
                        if ((flags & i.first) != 0U) {
                                if (!first) {
                                        status += ", "s;
                                }
                                status += i.second;
                                first = false;
                        }
                }
                return status;
        }

	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(
                        BMDVideoInputFormatChangedEvents notificationEvents,
                        IDeckLinkDisplayMode* mode,
                        BMDDetectedVideoInputFormatFlags flags) noexcept override {
                LOG(LOG_LEVEL_NOTICE) << MODULE_NAME << "Format change detected (" << getNotificationEventsStr(notificationEvents, flags) << ").\n";
                notificationEvents &= ~bmdVideoInputFieldDominanceChanged; // ignore field dominance change
                if (notificationEvents == 0U) {
                        return S_OK;
                }

                bool detected_3d = (flags & bmdDetectedVideoInputDualStream3D) != 0U;
                if ((detected_3d && !s->stereo) || ((!detected_3d && s->stereo))) {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME <<  "Stereoscopic 3D " << (detected_3d ? "" : "not ") << "detected but " << (s->stereo ? "" : "not ") << "enabled!" << (detected_3d ? " Please supply a \"3D\" parameter." : "") << "\n";
                        return E_FAIL;
                }
                BMDDetectedVideoInputFormatFlags csBitDepth = flags & (csMask | bitDepthMask);
                if ((csBitDepth & bitDepthMask) == 0U) { // if no bit depth, assume 8-bit
                        csBitDepth |= bmdDetectedVideoInput8BitDepth;
                }
                if (s->requested_bit_depth != 0) {
                        csBitDepth = (flags & csMask) | s->requested_bit_depth;
                }
                unordered_map<BMDDetectedVideoInputFormatFlags, codec_t> m = {
                        {bmdDetectedVideoInputYCbCr422 | bmdDetectedVideoInput8BitDepth, UYVY},
                        {bmdDetectedVideoInputYCbCr422 | bmdDetectedVideoInput10BitDepth, v210},
                        {bmdDetectedVideoInputYCbCr422 | bmdDetectedVideoInput12BitDepth, v210}, // weird
                        {bmdDetectedVideoInputRGB444 | bmdDetectedVideoInput8BitDepth, RGBA},
                        {bmdDetectedVideoInputRGB444 | bmdDetectedVideoInput10BitDepth, R10k},
                        {bmdDetectedVideoInputRGB444 | bmdDetectedVideoInput12BitDepth, R12L},
                };
                if (notificationEvents == bmdVideoInputColorspaceChanged && csBitDepth == configuredCsBitDepth) { // only CS change which was already performed
                        return S_OK;
                }
                if (s->requested_bit_depth == 0 && (flags & bmdDetectedVideoInput8BitDepth) == 0) {
                        const string & depth = (flags & bmdDetectedVideoInput10BitDepth) != 0U ? "10"s : "12"s;
                        LOG(LOG_LEVEL_WARNING) << MODULE_NAME << "Detected " << depth << "-bit signal, use \":codec=UYVY\" to enforce 8-bit capture (old behavior).\n";
                }

                unique_lock<mutex> lk(s->lock);
                s->set_codec(m.at(csBitDepth));
                configuredCsBitDepth = csBitDepth;

                IDeckLinkInput *deckLinkInput = device.deckLinkInput;
                deckLinkInput->PauseStreams();
                BMDPixelFormat pf{};
                if (HRESULT result = set_display_mode_properties(s, device.tile, mode, /* out */ &pf); FAILED(result)) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "set_display_mode_properties: " << bmd_hresult_to_string(result) << "\n";
                        return result;
                }
                CALL_AND_CHECK(deckLinkInput->EnableVideoInput(mode->GetDisplayMode(), pf, s->enable_flags), "EnableVideoInput");
                if (!device.audio) { //TODO: figure out output from multiple streams
                        deckLinkInput->DisableAudioInput();
                } else {
                        deckLinkInput->EnableAudioInput(
                                bmdAudioSampleRate48kHz,
                                s->audio.bps == 2 ? bmdAudioSampleType16bitInteger :
                                        bmdAudioSampleType32bitInteger,
                                max(s->audio.ch_count, 2)); // BMD isn't able to grab single channel
                }
                //deckLinkInput->SetCallback(s->state[i].delegate);
                deckLinkInput->FlushStreams();
                deckLinkInput->StartStreams();

                return S_OK;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*) override;
};

HRESULT	
VideoDelegate::VideoInputFrameArrived (IDeckLinkVideoInputFrame *videoFrame, IDeckLinkAudioInputPacket *audioPacket)
{
        bool nosig = false;

	unique_lock<mutex> lk(s->lock);
// LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK - LOCK //

	// Video
	if (videoFrame)
	{
                if (videoFrame->GetFlags() & bmdFrameHasNoInputSource) {
                        nosig = true;
			log_msg(LOG_LEVEL_INFO, "Frame received (#%d) - No input signal detected\n", s->frames);
                        if (s->nosig_send) {
                                newFrameReady = 1;
                        }
		} else {
                        newFrameReady = 1; // The new frame is ready to grab
			// printf("Frame received (#%lu) - Valid Frame (Size: %li bytes)\n", framecount, videoFrame->GetRowBytes() * videoFrame->GetHeight());
		}
	}

        if (audioPacket) {
                if (s->audioPackets.size() < MAX_AUDIO_PACKETS) {
                        audioPacket->AddRef();
                        s->audioPackets.push(audioPacket);
                } else {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Dropping audio packet, queue full.\n";
                }
        }

        if (videoFrame && newFrameReady && (!nosig || !lastFrame)) {
                /// @todo videoFrame should be actually retained until the data are processed
                videoFrame->GetBytes(&pixelFrame);

                RELEASE_IF_NOT_NULL(lastFrame);
                lastFrame = videoFrame;
                lastFrame->AddRef();

                IDeckLinkTimecode *tc = NULL;
                if (videoFrame->GetTimecode(bmdTimecodeRP188Any, &tc) == S_OK) {
                        timecode = tc->GetBCD();
                        tc->Release();
                } else {
                        timecode = 0;
                        if (s->sync_timecode) {
                                log_msg(LOG_LEVEL_ERROR, "Failed to acquire timecode from stream. Disabling sync.\n");
                                s->sync_timecode = FALSE;
                        }
                }

                RELEASE_IF_NOT_NULL(rightEyeFrame);
                pixelFrameRight = NULL;

                if(s->stereo) {
                        IDeckLinkVideoFrame3DExtensions *rightEye;
                        HRESULT result;
                        result = videoFrame->QueryInterface(IID_IDeckLinkVideoFrame3DExtensions, (void **)&rightEye);

                        if (result == S_OK) {
                                result = rightEye->GetFrameForRightEye(&rightEyeFrame);

                                if(result == S_OK) {
                                        if (rightEyeFrame->GetFlags() & bmdFrameHasNoInputSource)
                                        {
                                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Right Eye Frame received (#%d) - No input signal detected\n", s->frames);
                                        }
                                        rightEyeFrame->GetBytes(&pixelFrameRight);
                                }
                        }
                        rightEye->Release();
                        if(!pixelFrameRight) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Sending right eye error.\n");
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

static void vidcap_decklink_print_card_info(IDeckLink *deckLink) {

        // ** List the video input display modes supported by the card
        print_input_modes(deckLink);

        IDeckLinkProfileAttributes *deckLinkAttributes = nullptr;

        HRESULT result = deckLink->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&deckLinkAttributes);
        if (result != S_OK) {
                cout << "Could not query device attributes.\n\n";
                return;
        }
        int64_t connections = 0;
        if (deckLinkAttributes->GetInt(BMDDeckLinkVideoInputConnections, &connections) != S_OK) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Could not get connections.\n";
        } else {
                cout << "\n\tConnection can be one of following:\n";
                for (auto const &it : connection_string_map) {
                        if (connections & it.first) {
                                col() << "\t\t" << SBOLD(it.second) << "\n";
                        }
                }
        }
        cout << "\n";

        // Release the IDeckLink instance when we've finished with it to prevent leaks
        RELEASE_IF_NOT_NULL(deckLinkAttributes);
}

/* HELP */
static int
decklink_help(bool full)
{
	col() << "\nDecklink options:\n";
        col() << SBOLD(SRED("\t-t decklink") << "{:mode=<mode>|:device=<device_index>|:codec=<colorspace>...<key>=<val>}* | decklink:[full]help") << "\n";
        col() << "\t\tor\n";
        col() << SBOLD(SRED("\t-t decklink") << "[:<device_index(indices)>[:<mode>:<colorspace>[:3D][:sync_timecode][:connection=<input>][:audio_level={line|mic}][:detect-format][:conversion=<conv_mode>]]") << "\n";
        col() << "\t(mode specification is mandatory if your card does not support format autodetection; syntax on the first line is recommended, the second is obsolescent)\n";
        col() << "\n";

        col() << SBOLD("3D") << "\n";
        printf("\tUse this to capture 3D from supported card (eg. DeckLink HD 3D Extreme).\n");
        printf("\tDo not use it for eg. Quad or Duo. Availability of the mode is indicated\n");
        printf("\tin video format listing above (\"supports 3D\").\n");
	printf("\n");

        col() << SBOLD("audio_level") << "\n";
        col() << SBOLD("\tline") << " - the selected analog input gain levels are used\n";
        col() << SBOLD("\tmic") << "  - analog audio levels are set to maximum gain on audio input.\n";
	printf("\n");

        col() << SBOLD("detect-format") << "\n";
        col() << "\tTry to detect input video format even if the device doesn't support\n"
                "\tautodetect, eg. \"-t decklink:connection=HDMI:detect-format\".\n";
        col() << "\n";

        col() << SBOLD("fullhelp") << "\n";
        col() << "\tPrint description of all available options.\n";
        col() << "\n";

        col() << SBOLD("half-duplex") << "\n";
        col() << "\tSet a profile that allows maximal number of simultaneous IOs.\n";
        col() << "\n";

        if (full) {
                col() << SBOLD("conversion") << "\n";
                col() << SBOLD("\tnone") << " - No video input conversion\n";
                col() << SBOLD("\t10lb") << " - HD1080 to SD video input down conversion\n";
                col() << SBOLD("\t10am") << " - Anamorphic from HD1080 to SD video input down conversion\n";
                col() << SBOLD("\t72lb") << " - Letter box from HD720 to SD video input down conversion\n";
                col() << SBOLD("\t72ab") << " - Letterbox video input up conversion\n";
                col() << SBOLD("\tamup") << " - Anamorphic video input up conversion\n";
                col() << "\tThen use the set the resulting mode (!) for capture, eg. for 1080p to PAL conversion:\n"
                                "\t\t-t decklink:mode=pal:conversion=10lb\n";
                col() << "\n";
                col() << SBOLD("p_not_i") << "\n";
                col() << "\tIncoming signal should be treated as progressive even if detected as interlaced (PsF).\n";
                col() << "\n";

                col() << SBOLD("Use1080PsF[=true|false]") << "\n";
                col() << "\tIncoming signal should be treated as PsF instead of progressive.\n";
                col() << "\n";

                col() << SBOLD("nosig-send") << "\n";
                col() << "\tSend video even if no signal was detected (useful when video interrupts\n"
                        "\tbut the video stream needs to be preserved, eg. to keep sync with audio).\n";
                col() << "\n";

                col() << SBOLD("[no]passthrough[=keep]") << "\n";
                col() << "\tDisables/enables/keeps capture passthrough (default is disable).\n";
                col() << "\n";

                col() << SBOLD("profile=<FourCC>|profile=keep") << "\n";
                col() << "\tUse desired device profile: " << SBOLD("1dfd") << ", "
                        << SBOLD("1dhd") << ", "
                        << SBOLD("2dfd") << ", "
                        << SBOLD("2dhd") << " or "
                        << SBOLD("4dhd") << ". See SDK manual for details. Use "
                        << SBOLD("keep") << " to disable automatic selection.\n";
                col() << "\n";
                col() << SBOLD("sync_timecode") << "\n";
                col() << "\tTry to synchronize inputs based on timecode (for multiple inputs, eg. tiled 4K)\n";
                col() << "\n";
                col() << SBOLD("keep-settings") << "\n\tdo not apply any DeckLink settings by UG than required (keep user-selected defaults)\n";
                col() << "\n";

        } else {
                col() << "(other options available, use \"fullhelp\" to see complete list of options)\n\n";
        }

        col() << "Available color spaces:";
        for (auto & i : uv_to_bmd_codec_map) {
                if (i != *uv_to_bmd_codec_map.begin()) {
                        col() << ",";
                }

                col() << " " << SBOLD(get_codec_name(i.first));
        }
        cout << "\n";
        if (!full) {
                col() << "Possible connections: " << TBOLD("SDI") << ", " << TBOLD("HDMI") << ", " << TBOLD("OpticalSDI") << ", " << TBOLD("Component") << ", " << TBOLD("Composite") << ", " << TBOLD("SVideo") << "\n";
        }
        cout << "\n";

	// Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        bool com_initialized = false;
        IDeckLinkIterator *deckLinkIterator = create_decklink_iterator(&com_initialized);
        if (deckLinkIterator == NULL) {
		return 0;
	}
	
        cout << "Devices:\n";
	// Enumerate all cards in this system
        IDeckLink *deckLink = nullptr;
        int numDevices = 0;
	while (deckLinkIterator->Next(&deckLink) == S_OK)
	{
                string deviceName = bmd_get_device_name(deckLink);
                if (deviceName.empty()) {
                        deviceName = "(unable to get name)";
                }
		
		// *** Print the model name of the DeckLink card
                col() << "\t" << TBOLD(<< numDevices <<) << ") " << TBOLD(TGREEN(<< deviceName <<)) << "\n";
		
		// Increment the total number of DeckLink cards found
		numDevices++;
	
                if (full) {
                        vidcap_decklink_print_card_info(deckLink);
                }
		deckLink->Release();
	}
        if (!full) {
                cout << "(use \"-t decklink:fullhelp\" to see full list of device modes and available connections)\n\n";
        }
	
	deckLinkIterator->Release();

        decklink_uninitialize(&com_initialized);

	// If no DeckLink cards were found in the system, inform the user
	if (numDevices == 0)
	{
		log_msg(LOG_LEVEL_ERROR, "No Blackmagic Design devices were found.\n");
        }

        printf("Examples:\n");
        col() << "\t" << SBOLD(uv_argv[0] << " -t decklink") << " # captures autodetected video from first DeckLink in system\n";
        col() << "\t" << SBOLD(uv_argv[0] << " -t decklink:0:Hi50:UYVY") << " # captures 1080i50, 8-bit yuv\n";
        col() << "\t" << SBOLD(uv_argv[0] << " -t decklink:0:10:v210:connection=HDMI") << " # captures 10th format from a card (alternative syntax), 10-bit YUV, from HDMI\n";
        col() << "\t" << SBOLD(uv_argv[0] << " -t decklink:mode=23ps") << " # captures 1080p24, 8-bit yuv from first device\n";
        col() << "\t" << SBOLD(uv_argv[0] << " -t \"decklink:mode=Hp30:codec=v210:device=DeckLink HD Extreme 3D+\"") << " # captures 1080p30, 10-bit yuv from DeckLink HD Extreme 3D+\n";

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
        char *tmp = devices;

        s->devices_cnt = 0;
        while ((ptr = strtok_r(tmp, ",", &save_ptr_dev))) {
                s->devices_cnt += 1;
                s->state.resize(s->devices_cnt);
                s->state[s->devices_cnt - 1].device_id = ptr;
                tmp = NULL;
        }
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
                for (auto const & it : connection_string_map) {
                        if (strcasecmp(connection, it.second.c_str()) == 0) {
                                s->connection = it.first;
                                found = true;
                        }
                }
                if (!found) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unrecognized connection %s.\n", connection);
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
                s->conversion_mode = (BMDVideoInputConversionMode) bmd_read_fourcc(opt + strlen("conversion="));
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
                s->set_codec(get_codec_from_name(codec));
                if(s->codec == VIDEO_CODEC_NONE) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong config. Unknown color space %s\n", codec);
                        return false;
                }
        } else if (strcasecmp(opt, "detect-format") == 0) {
                s->detect_format = true;
        } else if (strcasecmp(opt, "p_not_i") == 0) {
                s->p_not_i = true;
        } else if (strstr(opt, "Use1080PsF") != nullptr) {
                s->use1080psf = 1;
                if (strcasecmp(opt + strlen("Use1080PsF"), "=false") == 0) {
                        s->use1080psf = 0;
                }
        } else if (strcasecmp(opt, "passthrough") == 0 || strcasecmp(opt, "nopassthrough") == 0) {
                s->passthrough = opt[0] == 'n' ? bmdDeckLinkCapturePassthroughModeDisabled
                        : strstr(opt, "keep") != nullptr ? static_cast<enum _BMDDeckLinkCapturePassthroughMode>(BMD_OPT_KEEP)
                        : bmdDeckLinkCapturePassthroughModeCleanSwitch;
        } else if (strstr(opt, "profile=") == opt) {
                const char *mode = opt + strlen("profile=");
                if (strcmp(mode, "keep") == 0) {
                        s->profile = BMD_OPT_KEEP;
                } else {
                        s->profile = (BMDProfileID) bmd_read_fourcc(mode);
                }
        } else if (strstr(opt, "half-duplex") == opt) {
                s->profile = bmdDuplexHalf;
        } else if (strcasecmp(opt, "nosig-send") == 0) {
                s->nosig_send = true;
        } else if (strstr(opt, "keep-settings") == opt) {
                s->keep_device_defaults = true;
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
                                s->set_codec(get_codec_from_name(tmp));
                                if(s->codec == VIDEO_CODEC_NONE) {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong config. Unknown color space %s\n", tmp);
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

static struct vidcap_type *vidcap_decklink_probe(bool verbose, void (**deleter)(void *))
{
        *deleter = free;
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
        bool com_initialized = false;
        deckLinkIterator = create_decklink_iterator(&com_initialized, false);
        if (deckLinkIterator == nullptr) {
                return vt;
        }

        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK) {
                HRESULT result;
                IDeckLinkProfileAttributes *deckLinkAttributes;

                result = deckLink->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&deckLinkAttributes);
                if (result != S_OK) {
                        continue;
                }
                int64_t connections_bitmap;
                if(deckLinkAttributes->GetInt(BMDDeckLinkVideoInputConnections, &connections_bitmap) != S_OK) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Could not get connections, skipping device.\n");
                        continue;
                }

                string deviceName = bmd_get_device_name(deckLink);
                if (deviceName.empty()) {
                        deviceName = "(unknown)";
                }

                vt->card_count += 1;
                vt->cards = (struct device_info *)
                        realloc(vt->cards, vt->card_count * sizeof(struct device_info));
                memset(&vt->cards[vt->card_count - 1], 0, sizeof(struct device_info));
                snprintf(vt->cards[vt->card_count - 1].dev, sizeof vt->cards[vt->card_count - 1].dev,
                                ":device=%d", numDevices);
                snprintf(vt->cards[vt->card_count - 1].name, sizeof vt->cards[vt->card_count - 1].name,
                                "%s #%d", deviceName.c_str(), numDevices);
                snprintf(vt->cards[vt->card_count - 1].extra, sizeof vt->cards[vt->card_count - 1].extra,
                                "\"embeddedAudioAvailable\":\"t\"");


                list<string> connections;
                for (auto const &it : connection_string_map) {
                        if (connections_bitmap & it.first) {
                                connections.push_back(it.second);
                        }
                }
                list<tuple<int, string, string, string>> modes = get_input_modes (deckLink);
                int i = 0;
                const int mode_count = sizeof vt->cards[vt->card_count - 1].modes /
                                                sizeof vt->cards[vt->card_count - 1].modes[0];
                for (auto &m : modes) {
                        for (auto &c : connections) {
                                if (i >= mode_count) { // no space
                                        break;
                                }

                                snprintf(vt->cards[vt->card_count - 1].modes[i].id,
                                                sizeof vt->cards[vt->card_count - 1].modes[i].id,
                                                R"({"modeOpt":"connection=%s:mode=%s:codec=UYVY"})",
                                                c.c_str(), get<1>(m).c_str());
                                snprintf(vt->cards[vt->card_count - 1].modes[i].name,
                                                sizeof vt->cards[vt->card_count - 1].modes[i].name,
                                                "%s (%s)", get<2>(m).c_str(), c.c_str());
                                i++;
                        }
                }

                for (auto &c : connections) {
                        if (i >= mode_count) {
                                break;
                        }
                        snprintf(vt->cards[vt->card_count - 1].modes[i].id,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].id,
                                        "{\"modeOpt\":\"connection=%s\"}",
                                        c.c_str());
                        snprintf(vt->cards[vt->card_count - 1].modes[i].name,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].name,
                                        "Decklink Auto (%s)", c.c_str());
                        if (++i >= mode_count) {
                                break;
                        }
                        snprintf(vt->cards[vt->card_count - 1].modes[i].id,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].id,
                                        "{\"modeOpt\":\"detect-format:connection=%s\"}",
                                        c.c_str());
                        snprintf(vt->cards[vt->card_count - 1].modes[i].name,
                                        sizeof vt->cards[vt->card_count - 1].modes[i].name,
                                        "UltraGrid auto-detect (%s)", c.c_str());
                        i++;
                }

                // Increment the total number of DeckLink cards found
                numDevices++;

                RELEASE_IF_NOT_NULL(deckLinkAttributes);
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        deckLinkIterator->Release();
        decklink_uninitialize(&com_initialized);

        return vt;
}

static HRESULT set_display_mode_properties(struct vidcap_decklink_state *s, struct tile *tile, IDeckLinkDisplayMode* displayMode, /* out */ BMDPixelFormat *pf)
{
        BMD_STR displayModeBMDString = nullptr;

        if (HRESULT result = displayMode->GetName(&displayModeBMDString); FAILED(result)) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "IDeckLinkDisplayMode::GetName failed: " << bmd_hresult_to_string(result) << "\n";
        }
        string displayModeString = get_str_from_bmd_api_str(displayModeBMDString);
        release_bmd_api_str(displayModeBMDString);

        auto it = std::find_if(uv_to_bmd_codec_map.begin(),
                        uv_to_bmd_codec_map.end(),
                        [&s](const std::pair<codec_t, BMDPixelFormat>& el){ return el.first == s->codec; });
        if (it == uv_to_bmd_codec_map.end()) {
                LOG(LOG_LEVEL_ERROR) << "Unsupported codec: " <<  get_codec_name(s->codec) << "!\n";
                return E_FAIL;
        }
        *pf = it->second;

        // get avarage time between frames
        BMDTimeValue	frameRateDuration = 0;
        BMDTimeScale	frameRateScale = 0;

        tile->width = displayMode->GetWidth();
        tile->height = displayMode->GetHeight();
        s->frame->color_spec = s->codec;

        displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
        s->frame->fps = static_cast<double>(frameRateScale) / frameRateDuration;
        s->next_frame_time = static_cast<int>(std::chrono::microseconds::period::den / s->frame->fps); // in microseconds
        switch(displayMode->GetFieldDominance()) {
                case bmdLowerFieldFirst:
                case bmdUpperFieldFirst:
                        s->frame->interlacing = INTERLACED_MERGED;
                        break;
                case bmdProgressiveFrame:
                case bmdProgressiveSegmentedFrame:
                        s->frame->interlacing = PROGRESSIVE;
                        break;
                case bmdUnknownFieldDominance:
                        LOG(LOG_LEVEL_WARNING) << "[DeckLink cap.] Unknown field dominance!\n";
                        s->frame->interlacing = PROGRESSIVE;
                        break;
        }

        if (s->p_not_i) {
                s->frame->interlacing = PROGRESSIVE;
        }

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << displayModeString << " \t " << tile->width << " x " << tile->height << " \t " <<
                s->frame->fps << " FPS \t " << s->next_frame_time << " AVAREGE TIME BETWEEN FRAMES\n";
        LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Enable video input: " << displayModeString << ((s->enable_flags & bmdVideoInputDualStream3D) != 0U ? " (stereo)" : "") << "\n";

        tile->data_len =
                vc_get_linesize(tile->width, s->frame->color_spec) * tile->height;

        if(s->stereo) {
                s->frame->tiles[1].width = s->frame->tiles[0].width;
                s->frame->tiles[1].height = s->frame->tiles[0].height;
                s->frame->tiles[1].data_len = s->frame->tiles[0].data_len;
        }

        return S_OK;
}

/**
 * This function is used when device does not support autodetection and user
 * request explicitly to detect the format (:detect-format)
 */
static bool detect_format(struct vidcap_decklink_state *s, BMDDisplayMode *outDisplayMode, struct device_state *device)
{
        IDeckLinkDisplayMode *displayMode;
        HRESULT result;
        IDeckLinkDisplayModeIterator*	displayModeIterator = NULL;
        result = device->deckLinkInput->GetDisplayModeIterator(&displayModeIterator);
        if (result != S_OK) {
                return false;
        }

        vector<BMDPixelFormat> pfs = {bmdFormat8BitYUV, bmdFormat8BitBGRA};

        while (displayModeIterator->Next(&displayMode) == S_OK) {
                for (BMDPixelFormat pf : pfs) {
                        uint32_t mode = ntohl(displayMode->GetDisplayMode());
                        log_msg(LOG_LEVEL_NOTICE, "DeckLink: trying mode %.4s, pixel format %.4s\n", (const char *) &mode, (const char *) &pf);
                        result = device->deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, 0);
                        if (result == S_OK) {
                                device->deckLinkInput->StartStreams();
                                unique_lock<mutex> lk(s->lock);
                                s->boss_cv.wait_for(lk, chrono::milliseconds(1200), [device]{return device->delegate->newFrameReady;});
                                lk.unlock();
                                device->deckLinkInput->StopStreams();
                                device->deckLinkInput->DisableVideoInput();

                                if (device->delegate->newFrameReady) {
                                        *outDisplayMode = displayMode->GetDisplayMode();
                                        // set also detected codec (!)
                                        s->set_codec(pf == bmdFormat8BitYUV ? UYVY : RGBA);
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

static bool decklink_cap_configure_audio(struct vidcap_decklink_state *s, unsigned int audio_src_flag, BMDAudioConnection *audioConnection) {
        if (audio_src_flag == 0U) {
                return true;
        }

        s->state[0].audio = true;
        switch (audio_src_flag) {
                case VIDCAP_FLAG_AUDIO_EMBEDDED:
                        *audioConnection = bmdAudioConnectionEmbedded;
                        break;
                case VIDCAP_FLAG_AUDIO_AESEBU:
                        *audioConnection = bmdAudioConnectionAESEBU;
                        break;
                case VIDCAP_FLAG_AUDIO_ANALOG:
                        *audioConnection = bmdAudioConnectionAnalog;
                        break;
                default:
                        LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Unexpected audio flag " << audio_src_flag << " encountered.\n";
                        abort();
        }
        s->audio.bps = audio_capture_bps == 0 ? DEFAULT_AUDIO_BPS : audio_capture_bps;
        if (s->audio.bps != 2 && s->audio.bps != 4) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "[Decklink] Unsupported audio Bps " << audio_capture_bps << "! Supported is 2 or 4 bytes only!\n";
                return false;
        }
        if (audio_capture_sample_rate != 0 && audio_capture_sample_rate != bmdAudioSampleRate48kHz) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unsupported sample rate " << audio_capture_sample_rate << "! Only " << bmdAudioSampleRate48kHz << " is supported.\n";
                return false;
        }
        s->audio.sample_rate = bmdAudioSampleRate48kHz;
        s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
        s->audio.max_size = (s->audio.sample_rate / 10) * s->audio.ch_count * s->audio.bps;
        s->audio.data = (char *) malloc(s->audio.max_size);

        return true;
}

void device_state::check_attributes(struct vidcap_decklink_state *s)
{
        if (s->stereo) {
                bmd_check_stereo_profile(deckLink);
        }
}

#define INIT_ERR() do { RELEASE_IF_NOT_NULL(displayMode); RELEASE_IF_NOT_NULL(displayModeIterator); return false; } while (0)
bool device_state::init(struct vidcap_decklink_state *s, struct tile *t, BMDAudioConnection audioConnection)
{
        IDeckLinkDisplayModeIterator*   displayModeIterator = NULL;
        IDeckLinkDisplayMode*           displayMode = NULL;

        tile = t;

        int dnum = 0;
        bool com_initialized = false;
        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        IDeckLinkIterator *deckLinkIterator = create_decklink_iterator(&com_initialized, true, false);
        if (deckLinkIterator == NULL) {
                return false;
        }
        bool found = false;
        while (deckLinkIterator->Next(&deckLink) == S_OK) {
                string deviceName = bmd_get_device_name(deckLink);
                if (!deviceName.empty() && deviceName == device_id.c_str()) {
                        found = true;
                }

                if (isdigit(device_id.c_str()[0]) && atoi(device_id.c_str()) == dnum) {
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
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Device " << device_id << " was not found.\n";
                INIT_ERR();
        }

        // Print the model name of the DeckLink card
        string deviceName = bmd_get_device_name(deckLink);
        if (!deviceName.empty()) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Using device " << deviceName << "\n";
        }

        if (!s->keep_device_defaults && s->profile != BMD_OPT_DEFAULT && s->profile != BMD_OPT_KEEP) {
                decklink_set_duplex(deckLink, s->profile);
        }

        // Query the DeckLink for its configuration interface
        BMD_CHECK(deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput), "Could not obtain the IDeckLinkInput interface", INIT_ERR());

        // Query the DeckLink for its configuration interface
        BMD_CHECK(deckLinkInput->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration), "Could not obtain the IDeckLinkConfiguration interface", INIT_ERR());

        IDeckLinkProfileAttributes *deckLinkAttributes;
        BMD_CHECK(deckLinkInput->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&deckLinkAttributes), "Could not query device attributes", INIT_ERR());

        BMD_CONFIG_SET(Int, bmdDeckLinkConfigVideoInputConnection, s->connection, INIT_ERR());
        BMD_CONFIG_SET(Int, bmdDeckLinkConfigVideoInputConversionMode, s->conversion_mode, INIT_ERR());
        BMDVideoInputConversionMode supported_conversion_mode = s->conversion_mode ? s->conversion_mode : (BMDVideoInputConversionMode) bmdNoVideoInputConversion;
        BMD_CONFIG_SET(Int, bmdDeckLinkConfigCapturePassThroughMode, s->passthrough, BMD_NO_ACTION);

        if (s->use1080psf != BMD_OPT_KEEP) {
                CALL_AND_CHECK(deckLinkConfiguration->SetFlag(bmdDeckLinkConfigCapture1080pAsPsF, s->use1080psf != 0), "Unable to set output as PsF");
        }

        // set Callback which returns frames
        delegate = make_unique<VideoDelegate>(s, *this);
        deckLinkInput->SetCallback(delegate.get());

        BMDDisplayMode detectedDisplayMode = bmdModeUnknown;
        if (s->detect_format) {
                if (!detect_format(s, &detectedDisplayMode, this)) {
                        LOG(LOG_LEVEL_WARNING) << "Signal could have not been detected!\n";
                        INIT_ERR();
                }
        }

        // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
        BMD_CHECK(deckLinkInput->GetDisplayModeIterator(&displayModeIterator),
                        "Could not obtain the video input display mode iterator:", INIT_ERR());

        check_attributes(s);

        int mnum = 0;
#define MODE_SPEC_AUTODETECT -1
#define MODE_SPEC_FOURCC -2
#define MODE_SPEC_DETECTED -3
        int mode_idx = MODE_SPEC_AUTODETECT;

        // mode selected manually - either by index or FourCC
        if (s->mode.length() > 0) {
                if (s->mode.length() <= 2) {
                        mode_idx = atoi(s->mode.c_str());
                } else {
                        mode_idx = MODE_SPEC_FOURCC;
                }
        }
        if (s->detect_format) { // format already detected manually
                mode_idx = MODE_SPEC_DETECTED;
        }

        while (displayModeIterator->Next(&displayMode) == S_OK) {
                if (mode_idx == MODE_SPEC_DETECTED) { // format already detected manually
                        if (detectedDisplayMode == displayMode->GetDisplayMode()) {
                                break;
                        } else {
                                displayMode->Release();
                        }
                } else if (mode_idx == MODE_SPEC_AUTODETECT) { // autodetect, pick first eligible mode and let device autodetect
                        if (s->stereo && (displayMode->GetFlags() & bmdDisplayModeSupports3D) == 0u) {
                                displayMode->Release();
                                continue;
                        }
                        auto it = std::find_if(uv_to_bmd_codec_map.begin(),
                                        uv_to_bmd_codec_map.end(),
                                        [&s](const std::pair<codec_t, BMDPixelFormat>& el){ return el.first == s->codec; });
                        if (it == uv_to_bmd_codec_map.end()) {
                                LOG(LOG_LEVEL_ERROR) << "Unsupported codec: " <<  get_codec_name(s->codec) << "!\n";
                                INIT_ERR();
                        }
                        BMDPixelFormat pf = it->second;
                        BMD_BOOL supported = 0;
                        BMD_CHECK(deckLinkInput->DoesSupportVideoMode(s->connection, displayMode->GetDisplayMode(), pf, supported_conversion_mode, s->supported_flags, nullptr, &supported), "DoesSupportVideoMode", INIT_ERR());
                        if (supported) {
                                break;
                        }
                } else if (mode_idx != MODE_SPEC_FOURCC) { // manually given idx
                        if (mode_idx != mnum) {
                                mnum++;
                                // Release the IDeckLinkDisplayMode object to prevent a leak
                                displayMode->Release();
                                continue;
                        }

                        mnum++;
                        break;
                } else { // manually given FourCC
                        if (displayMode->GetDisplayMode() == bmd_read_fourcc(s->mode.c_str())) {
                                break;
                        }
                        displayMode->Release();
                }
        }

        if (displayMode) {
                BMD_STR displayModeString = NULL;
                if (HRESULT result = displayMode->GetName(&displayModeString); result == S_OK) {
                        char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                        LOG(LOG_LEVEL_INFO) << "The desired display mode is supported: " << displayModeCString << "\n";
                        release_bmd_api_str(displayModeString);
                        free(displayModeCString);
                }
        } else {
                if (mode_idx == MODE_SPEC_FOURCC) {
                        log_msg(LOG_LEVEL_ERROR, "Desired mode \"%s\" is invalid or not supported.\n", s->mode.c_str());
                } else if (mode_idx >= 0) {
                        log_msg(LOG_LEVEL_ERROR, "Desired mode index %s is out of bounds.\n", s->mode.c_str());
                } else if (mode_idx == MODE_SPEC_AUTODETECT) {
                        log_msg(LOG_LEVEL_ERROR, MODULE_NAME "Cannot set initial format for autodetection - perhaps imposible combinations of parameters were set.\n");
                } else {
                        assert("Invalid mode spec." && 0);
                }
                INIT_ERR();
        }

        BMDPixelFormat pf;
        BMD_CHECK(set_display_mode_properties(s, tile, displayMode, &pf),
                        "Could not set display mode properties", INIT_ERR());

        deckLinkInput->StopStreams();

        if (mode_idx == MODE_SPEC_AUTODETECT) {
                log_msg(LOG_LEVEL_INFO, "[DeckLink] Trying to autodetect format.\n");
                BMD_BOOL autodetection;
                BMD_CHECK(deckLinkAttributes->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &autodetection), "Could not verify if device supports autodetection", INIT_ERR());
                if (autodetection == BMD_FALSE) {
                        log_msg(LOG_LEVEL_ERROR, "[DeckLink] Device doesn't support format autodetection, you must set it manually or try \"-t decklink:detect-format[:connection=<in>]\"\n");
                        INIT_ERR();
                }
                s->enable_flags |=  bmdVideoInputEnableFormatDetection;
        }

        BMD_BOOL supported = 0;
        BMD_CHECK(deckLinkInput->DoesSupportVideoMode(s->connection, displayMode->GetDisplayMode(), pf, supported_conversion_mode, s->supported_flags, nullptr, &supported), "DoesSupportVideoMode", INIT_ERR());

        if (!supported) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Requested display mode not supported with the selected pixel format\n";
                INIT_ERR();
        }

        if (HRESULT result = deckLinkInput->EnableVideoInput(displayMode->GetDisplayMode(), pf, s->enable_flags); result != S_OK) {
                switch (result) {
                        case E_INVALIDARG:
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "You have required invalid video mode and pixel format combination.\n");
                                break;
                        case E_ACCESSDENIED:
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to access the hardware or input "
                                                "stream currently active (another application using it?).\n");
                                break;
                        default:
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Could not enable video input: " << bmd_hresult_to_string(result) << "\n";
                }
                INIT_ERR();
        }

        if (!audio) { //TODO: figure out output from multiple streams
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
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Audio input set to: %s\n", mapping.find(audioConnection) != mapping.end() ? mapping.at(audioConnection).c_str() : "unknown");
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to set audio input!!! Please check if it is OK. Continuing anyway.\n");

                }
                if (s->audio.ch_count != 1 &&
                                s->audio.ch_count != 2 &&
                                s->audio.ch_count != 8 &&
                                s->audio.ch_count != 16) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Decklink cannot grab %d audio channels. "
                                        "Only 1, 2, 8 or 16 are possible.", s->audio.ch_count);
                        INIT_ERR();
                }
                if (s->audio_consumer_levels != -1) {
                        if (HRESULT result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                                s->audio_consumer_levels == 1 ? true : false); result != S_OK) {
                                log_msg(LOG_LEVEL_INFO, MOD_NAME "Unable set input audio consumer levels.\n");
                        }
                }
                CALL_AND_CHECK(deckLinkInput->EnableAudioInput(
                                        bmdAudioSampleRate48kHz,
                                        s->audio.bps == 2 ? bmdAudioSampleType16bitInteger : bmdAudioSampleType32bitInteger,
                                        max(s->audio.ch_count, 2)), // capture at least 2
                                "EnableAudioInput",
                                "Decklink audio capture initialized sucessfully: " << audio_desc_from_frame(&s->audio));
        }

        // Start streaming
        BMD_CHECK(deckLinkInput->StartStreams(), "Could not start stream", INIT_ERR());

        displayMode->Release();
        displayMode = NULL;

        displayModeIterator->Release();
        displayModeIterator = NULL;
        return true;
}

static int
vidcap_decklink_init(struct vidcap_params *params, void **state)
{
        const char *fmt = vidcap_params_get_fmt(params);

        if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
                decklink_help(strcmp(fmt, "fullhelp") == 0);
                return VIDCAP_INIT_NOERR;
        }

        if (!blackmagic_api_version_check()) {
                return VIDCAP_INIT_FAIL;
        }

        struct vidcap_decklink_state *s = nullptr;
        try {
                if ((s = new vidcap_decklink_state()) == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "Unable to allocate DeckLink state\n";
                        return VIDCAP_INIT_FAIL;
                }
        } catch(...) {
                return VIDCAP_INIT_FAIL;
        }

	// SET UP device and mode
        char *tmp_fmt = strdup(fmt);
        int ret = settings_init(s, tmp_fmt);
        free(tmp_fmt);
	if (!ret) {
                delete s;
		return VIDCAP_INIT_FAIL;
	}

        switch (get_bits_per_component(s->codec)) {
        case 0: s->requested_bit_depth = 0; break;
        case 8: s->requested_bit_depth = bmdDetectedVideoInput8BitDepth; break;
        case 10: s->requested_bit_depth = bmdDetectedVideoInput10BitDepth; break;
        case 12: s->requested_bit_depth = bmdDetectedVideoInput12BitDepth; break;
        default: abort();
        }
        if (s->codec == VIDEO_CODEC_NONE) {
                s->set_codec(UYVY); // default one
        }

        BMDAudioConnection audioConnection = bmdAudioConnectionEmbedded;
        if (!decklink_cap_configure_audio(s, vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY, &audioConnection)) {
                delete s;
                return VIDCAP_INIT_FAIL;
        }

        if(s->stereo) {
                s->enable_flags |= bmdVideoInputDualStream3D;
                s->supported_flags = (BMDSupportedVideoModeFlags) (s->supported_flags | bmdSupportedVideoModeDualStream3D);
                if (s->devices_cnt > 1) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Passed more than one device while setting 3D mode. "
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
                if (!s->state[i].init(s, vf_get_tile(s->frame, i), audioConnection)) {
                        cleanup_common(s);
                        return VIDCAP_INIT_FAIL;
                }
        }

        *state = s;
	return VIDCAP_INIT_OK;
}

static void cleanup_common(struct vidcap_decklink_state *s) {
        while (!s->audioPackets.empty()) {
                auto *audioPacket = s->audioPackets.front();
                s->audioPackets.pop();
                audioPacket->Release();
        }

        for (int i = 0; i < s->devices_cnt; ++i) {
                RELEASE_IF_NOT_NULL(s->state[i].deckLinkConfiguration);
                RELEASE_IF_NOT_NULL(s->state[i].deckLinkAttributes);
                RELEASE_IF_NOT_NULL(s->state[i].deckLinkInput);
                RELEASE_IF_NOT_NULL(s->state[i].deckLink);
        }

        free(s->audio.data);

        vf_free(s->frame);
        delete s;
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
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Could not stop stream: " << bmd_hresult_to_string(result) << "\n";
		}

                if (s->state[i].audio) {
                        result = s->state[i].deckLinkInput->DisableAudioInput();
                        if (result != S_OK) {
                                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Could disable audio input: " << bmd_hresult_to_string(result) << "\n";
                        }
                }
		result = s->state[i].deckLinkInput->DisableVideoInput();
                if (result != S_OK) {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Could disable video input: " << bmd_hresult_to_string(result) << "\n";
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
        if (s->audioPackets.empty()) {
                return nullptr;
        }
        s->audio.data_len = 0;
        while (!s->audioPackets.empty()) {
                auto *audioPacket = s->audioPackets.front();
                s->audioPackets.pop();

                void *audioFrame = nullptr;
                audioPacket->GetBytes(&audioFrame);

                if (s->audio.ch_count == 1) { // there are actually 2 channels grabbed
                        if (s->audio.data_len + audioPacket->GetSampleFrameCount() * 1U * s->audio.bps <= static_cast<unsigned>(s->audio.max_size)) {
                                demux_channel(s->audio.data + s->audio.data_len, static_cast<char *>(audioFrame), s->audio.bps, min<int64_t>(audioPacket->GetSampleFrameCount() * 2 /* channels */ * s->audio.bps, INT_MAX), 2 /* channels (originally) */, 0 /* we want first channel */);
                                s->audio.data_len = min<int64_t>(s->audio.data_len + audioPacket->GetSampleFrameCount() * 1 * s->audio.bps, INT_MAX);
                        } else {
                                LOG(LOG_LEVEL_WARNING) << "[DeckLink] Audio frame too small!\n";
                        }
                } else {
                        if (s->audio.data_len + audioPacket->GetSampleFrameCount() * s->audio.ch_count * s->audio.bps <= s->audio.max_size) {
                                memcpy(s->audio.data + s->audio.data_len, audioFrame, audioPacket->GetSampleFrameCount() * s->audio.ch_count * s->audio.bps);
                                s->audio.data_len = min<int64_t>(s->audio.data_len + audioPacket->GetSampleFrameCount() * s->audio.ch_count * s->audio.bps, INT_MAX);
                        } else {
                                LOG(LOG_LEVEL_WARNING) << "[DeckLink] Audio frame too small!\n";
                        }
                }
                audioPacket->Release();
        }
        return &s->audio;
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

                        frame_ready = false;
                        break;
                }
	}

        /* cleanup newframe flag */
        for (i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].delegate->newFrameReady == 0) {
                        frame_ready = false;
                }
                s->state[i].delegate->newFrameReady = 0;
	}

        *audio = process_new_audio_packets(s); // return audio even if there is no video to avoid
                                               //  hoarding and then dropping of audio packets
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
                        s->frame->tiles[1].data = (char*)s->state[0].delegate->pixelFrameRight;
                        ++count;
                } // else count == 0 -> return NULL
        } else {
                for (i = 0; i < s->devices_cnt; ++i) {
                        if (s->state[i].delegate->pixelFrame == NULL) {
                                break;
                        }
                        s->frame->tiles[i].data = (char*)s->state[i].delegate->pixelFrame;
                        ++count;
                }
        }
        if (count < s->devices_cnt) {
                return NULL;
        }
        if (s->codec == RGBA) {
                for (unsigned i = 0; i < s->frame->tile_count; ++i) {
                        vc_copylineToRGBA_inplace((unsigned char*) s->frame->tiles[i].data,
                                        (unsigned char*)s->frame->tiles[i].data,
                                        s->frame->tiles[i].data_len, 16, 8, 0);
                }
        }
        if (s->codec == R10k && get_commandline_param(R10K_FULL_OPT) == nullptr) {
                for (unsigned i = 0; i < s->frame->tile_count; ++i) {
                        r10k_limited_to_full(s->frame->tiles[i].data, s->frame->tiles[i].data,
                                        s->frame->tiles[i].data_len);
                }
        }

        s->frames++;
        s->frame->timecode = s->state[0].delegate->timecode;
        return s->frame;
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
        if ((result = deckLink->QueryInterface(IID_IDeckLinkInput, (void**)&deckLinkInput)) != S_OK) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Could not obtain the IDeckLinkInput interface: " << bmd_hresult_to_string(result) << "\n";
                if (result == E_NOINTERFACE) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Device doesn't support video capture.\n");
                }
		goto bail;
	}

	// Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on input
        if ((result = deckLinkInput->GetDisplayModeIterator(&displayModeIterator)) != S_OK) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Could not obtain the video input display mode iterator: " << bmd_hresult_to_string(result) << "\n";
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
	RELEASE_IF_NOT_NULL(displayModeIterator);
	RELEASE_IF_NOT_NULL(deckLinkInput);

        return ret;
}

static void print_input_modes (IDeckLink* deckLink)
{
        list<tuple<int, string, string, string>> ret = get_input_modes (deckLink);
	printf("\tcapture modes:\n");
        for (auto &i : ret) {
                col() << "\t\t" << right << setw(2) << SBOLD(get<0>(i) << " (" << get<1>(i) << ")") << ") " <<
                        left << setw(20) << get<2>(i) << internal << "  " <<
                        get<3>(i) << "\n";
        }
}

void vidcap_decklink_state::set_codec(codec_t c) {
        codec = c;
        LOG(LOG_LEVEL_INFO) << MODULE_NAME "Using codec: " << get_codec_name(codec) << "\n";
        if (c == R10k && get_commandline_param(R10K_FULL_OPT) == nullptr) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Using limited range R10k as specified by BMD, use '--param "
                                R10K_FULL_OPT "' to override.\n");
        }
}

static const struct video_capture_info vidcap_decklink_info = {
        vidcap_decklink_probe,
        vidcap_decklink_init,
        vidcap_decklink_done,
        vidcap_decklink_grab,
        MOD_NAME,
};

REGISTER_MODULE(decklink, &vidcap_decklink_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vi: set expandtab sw=8: */

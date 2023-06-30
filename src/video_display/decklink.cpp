/**
 * @file   video_display/decklink.cpp
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2023 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#define MOD_NAME "[Decklink display] "

#include "audio/types.h"
#include "blackmagic_common.hpp"
#include "compat/platform_time.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rtp/audio_decoders.h"
#include "tv.h"
#include "ug_runtime_error.hpp"
#include "utils/misc.h"
#include "utils/string.h" // is_prefix_of
#include "video.h"
#include "video_display.h"
#include "video_display/decklink_drift_fix.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "DeckLinkAPIVersion.h"

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

static void print_output_modes(IDeckLink *);
static void display_decklink_done(void *state);

// performs command, if failed, displays error and jumps to error label
#define EXIT_IF_FAILED(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                        return FALSE;\
                }\
        } while (0)

// similar as above, but only displays warning
#define CALL_AND_CHECK(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                }\
        } while (0)

using namespace std;

static int display_decklink_putf(void *state, struct video_frame *frame, long long nonblock);

namespace {
class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
private:
        uint64_t frames_dropped = 0;
        uint64_t frames_flushed = 0;
        uint64_t frames_late = 0;

        friend int ::display_decklink_putf(void *state, struct video_frame *frame, long long nonblock);
public:
        virtual ~PlaybackDelegate() = default;
        // IUnknown needs only a dummy implementation
        virtual HRESULT STDMETHODCALLTYPE        QueryInterface (REFIID , LPVOID *)        { return E_NOINTERFACE;}
        virtual ULONG STDMETHODCALLTYPE          AddRef ()                                                                       {return 1;}
        virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
	{
                if (result == bmdOutputFrameDisplayedLate){
                        frames_late += 1;
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Late frame (total: " << frames_late << ")\n";
                } else if (result == bmdOutputFrameDropped){
                        frames_dropped += 1;
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Dropped frame (total: " << frames_dropped << ")\n";
                } else if (result == bmdOutputFrameFlushed){
                        frames_flushed += 1;
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Flushed frame (total: " << frames_flushed << ")\n";
                }

		if (log_level >= LOG_LEVEL_DEBUG) {
			IDeckLinkTimecode *timecode = NULL;
			if (completedFrame->GetTimecode ((BMDTimecodeFormat) 0, &timecode) == S_OK) {
				BMD_STR timecode_str;
				if (timecode && timecode->GetString(&timecode_str) == S_OK) {
                                        char *timecode_cstr = get_cstr_from_bmd_api_str(timecode_str);
					LOG(LOG_LEVEL_DEBUG) << "Frame " << timecode_cstr << " output at " <<  time_since_epoch_in_ms() / (double) 1e3 << '\n';
                                        release_bmd_api_str(timecode_str);
                                        free(timecode_cstr);
				}
			}
		}


		completedFrame->Release();
		return S_OK;
	}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledPlaybackHasStopped (){
        	return S_OK;
	}
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

class DeckLinkFrame;

struct buffer_pool_t {
        queue<DeckLinkFrame *> frame_queue;
        mutex lock;
};

class DeckLinkTimecode : public IDeckLinkTimecode{
                BMDTimecodeBCD timecode;
        public:
                DeckLinkTimecode() : timecode(0) {}
                virtual ~DeckLinkTimecode() = default;
                /* IDeckLinkTimecode */
                virtual BMDTimecodeBCD STDMETHODCALLTYPE GetBCD (void) { return timecode; }
                virtual HRESULT STDMETHODCALLTYPE GetComponents (/* out */ uint8_t *hours, /* out */ uint8_t *minutes, /* out */ uint8_t *seconds, /* out */ uint8_t *frames) { 
                        *frames =   (timecode & 0xf)              + ((timecode & 0xf0) >> 4) * 10;
                        *seconds = ((timecode & 0xf00) >> 8)      + ((timecode & 0xf000) >> 12) * 10;
                        *minutes = ((timecode & 0xf0000) >> 16)   + ((timecode & 0xf00000) >> 20) * 10;
                        *hours =   ((timecode & 0xf000000) >> 24) + ((timecode & 0xf0000000) >> 28) * 10;
                        return S_OK;
                }
                virtual HRESULT STDMETHODCALLTYPE GetString (/* out */ BMD_STR *timecode) {
                        uint8_t hours, minutes, seconds, frames;
                        GetComponents(&hours, &minutes, &seconds, &frames);
                        char timecode_c[16];
                        assert(hours <= 99 && minutes <= 59 && seconds <= 60 && frames <= 99);
                        snprintf(timecode_c, sizeof timecode_c, "%02" PRIu8 ":%02" PRIu8 ":%02" PRIu8 ":%02" PRIu8, hours, minutes, seconds, frames);
                        *timecode = get_bmd_api_str_from_cstr(timecode_c);
                        return *timecode ? S_OK : E_FAIL;
                }
                virtual BMDTimecodeFlags STDMETHODCALLTYPE GetFlags (void)        { return bmdTimecodeFlagDefault; }
                virtual HRESULT STDMETHODCALLTYPE GetTimecodeUserBits (/* out */ BMDTimecodeUserBits *userBits) { if (!userBits) return E_POINTER; else return S_OK; }

                /* IUnknown */
                virtual HRESULT STDMETHODCALLTYPE QueryInterface (REFIID , LPVOID *)        {return E_NOINTERFACE;}
                virtual ULONG STDMETHODCALLTYPE         AddRef ()                                                                       {return 1;}
                virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}
                
                void STDMETHODCALLTYPE SetBCD(BMDTimecodeBCD timecode) { this->timecode = timecode; }
};

struct ChromaticityCoordinates
{
        double RedX;
        double RedY;
        double GreenX;
        double GreenY;
        double BlueX;
        double BlueY;
        double WhiteX;
        double WhiteY;
};

constexpr ChromaticityCoordinates kDefaultRec2020Colorimetrics = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290 };
constexpr double kDefaultMaxDisplayMasteringLuminance        = 1000.0;
constexpr double kDefaultMinDisplayMasteringLuminance        = 0.0001;
constexpr double kDefaultMaxCLL                              = 1000.0;
constexpr double kDefaultMaxFALL                             = 50.0;
enum class HDR_EOTF { NONE = -1, SDR = 0, HDR = 1, PQ = 2, HLG = 3 };

struct HDRMetadata
{
        int64_t                                 EOTF{static_cast<int64_t>(HDR_EOTF::NONE)};
        ChromaticityCoordinates referencePrimaries{kDefaultRec2020Colorimetrics};
        double                                  maxDisplayMasteringLuminance{kDefaultMaxDisplayMasteringLuminance};
        double                                  minDisplayMasteringLuminance{kDefaultMinDisplayMasteringLuminance};
        double                                  maxCLL{kDefaultMaxCLL};
        double                                  maxFALL{kDefaultMaxFALL};

        void                                    Init(const string & fmt);
};

class DeckLinkFrame : public IDeckLinkMutableVideoFrame, public IDeckLinkVideoFrameMetadataExtensions
{
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                unique_ptr<char []> data;

                IDeckLinkTimecode *timecode;

                long ref;

                buffer_pool_t &buffer_pool;
                struct HDRMetadata m_metadata;
        protected:
                DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & bp, HDRMetadata const & hdr_metadata);

        public:
                virtual ~DeckLinkFrame();
                static DeckLinkFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);

                /* IUnknown */
                HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override;
                ULONG STDMETHODCALLTYPE AddRef() override;
                ULONG STDMETHODCALLTYPE Release() override;
                
                /* IDeckLinkVideoFrame */
                long STDMETHODCALLTYPE GetWidth (void) override;
                long STDMETHODCALLTYPE GetHeight (void) override;
                long STDMETHODCALLTYPE GetRowBytes (void) override;
                BMDPixelFormat STDMETHODCALLTYPE GetPixelFormat (void) override;
                BMDFrameFlags STDMETHODCALLTYPE GetFlags (void) override;
                HRESULT STDMETHODCALLTYPE GetBytes (/* out */ void **buffer) override;
                HRESULT STDMETHODCALLTYPE GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode) override;
                HRESULT STDMETHODCALLTYPE GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary) override;

                /* IDeckLinkMutableVideoFrame */
                HRESULT STDMETHODCALLTYPE SetFlags(BMDFrameFlags) override;
                HRESULT STDMETHODCALLTYPE SetTimecode(BMDTimecodeFormat, IDeckLinkTimecode*) override;
                HRESULT STDMETHODCALLTYPE SetTimecodeFromComponents(BMDTimecodeFormat, uint8_t, uint8_t, uint8_t, uint8_t, BMDTimecodeFlags) override;
                HRESULT STDMETHODCALLTYPE SetAncillaryData(IDeckLinkVideoFrameAncillary*) override;
                HRESULT STDMETHODCALLTYPE SetTimecodeUserBits(BMDTimecodeFormat, BMDTimecodeUserBits) override;

                // IDeckLinkVideoFrameMetadataExtensions interface
                HRESULT STDMETHODCALLTYPE GetInt(BMDDeckLinkFrameMetadataID metadataID, int64_t* value) override;
                HRESULT STDMETHODCALLTYPE GetFloat(BMDDeckLinkFrameMetadataID metadataID, double* value) override;
                HRESULT STDMETHODCALLTYPE GetFlag(BMDDeckLinkFrameMetadataID metadataID, BMD_BOOL* value) override;
                HRESULT STDMETHODCALLTYPE GetString(BMDDeckLinkFrameMetadataID metadataID, BMD_STR * value) override;
                HRESULT STDMETHODCALLTYPE GetBytes(BMDDeckLinkFrameMetadataID metadataID, void* buffer, uint32_t* bufferSize) override;

};

class DeckLink3DFrame : public DeckLinkFrame, public IDeckLinkVideoFrame3DExtensions
{
        private:
                using DeckLinkFrame::DeckLinkFrame;
                DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);
                unique_ptr<DeckLinkFrame> rightEye; // rightEye ref count is always >= 1 therefore deleted by owner (this class)

        public:
                ~DeckLink3DFrame();
                static DeckLink3DFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);
                
                /* IUnknown */
                HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override;
                ULONG STDMETHODCALLTYPE AddRef() override;
                ULONG STDMETHODCALLTYPE Release() override;

                /* IDeckLinkVideoFrame3DExtensions */
                BMDVideo3DPackingFormat STDMETHODCALLTYPE Get3DPackingFormat() override;
                HRESULT STDMETHODCALLTYPE GetFrameForRightEye(IDeckLinkVideoFrame**) override;
};
} // end of unnamed namespace

#define DECKLINK_MAGIC 0x12de326b


struct state_decklink {
        uint32_t            magic = DECKLINK_MAGIC;
        chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();
        bool                com_initialized = false;
        PlaybackDelegate           *delegate;
        IDeckLink                  *deckLink;
        IDeckLinkOutput            *deckLinkOutput;
        IDeckLinkConfiguration     *deckLinkConfiguration;
        IDeckLinkProfileAttributes *deckLinkAttributes;

        BMDTimeValue        frameRateDuration{};
        BMDTimeScale        frameRateScale{};

        DeckLinkTimecode    *timecode{}; ///< @todo Should be actually allocated dynamically and
                                       ///< its lifespan controlled by AddRef()/Release() methods

        struct video_desc   vid_desc{};
        struct audio_desc   aud_desc{};

        unsigned long int   frames            = 0;
        bool                stereo            = false;
        bool                initialized_audio = false;
        bool                initialized_video = false;
        bool                emit_timecode     = false;
        bool                play_audio        = false; ///< the BMD device will be used also for output audio

        BMDPixelFormat      pixelFormat{};

        bmd_option          profile_req;
        bmd_option          quad_square_division_split{true, false};
        map<BMDDeckLinkConfigurationID, bmd_option> device_options = {
                { bmdDeckLinkConfigVideoOutputIdleOperation, bmd_option{(int64_t) bmdIdleVideoOutputLastFrame, false} },
                { bmdDeckLinkConfigOutput1080pAsPsF, bmd_option{false, false}},
                { bmdDeckLinkConfigFieldFlickerRemoval, bmd_option{false, false}}, ///< required for interlaced video in low-latency
                { bmdDeckLinkConfigLowLatencyVideoOutput, bmd_option{true, false}}
        };
        HDRMetadata         requested_hdr_mode{};

        buffer_pool_t       buffer_pool;

        bool                low_latency       = true;

        mutex               reconfiguration_lock; ///< for audio and video reconf to be mutually exclusive
        bool                keep_device_defaults = false;

        AudioDriftFixer audio_drift_fixer{};
 };

static void show_help(bool full);

static void show_help(bool full)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;

        col() << "Decklink display options:\n";
        col() << SBOLD(SRED("\t-d decklink")
                       << "[:d[evice]=<device>][:Level{A|B}][:3D][:audio_level={line|mic}][:half-"
                          "duplex][:HDR[=<t>][:drift_fix]]\n");
        col() << SBOLD(SRED("\t-d decklink") << ":[full]help\n");
        col() << "\nOptions:\n";
        if (!full) {
                col() << SBOLD("\tfullhelp") << "\tdisplay additional options and more details\n";
        }
        col() << SBOLD("\tdevice") << "\t\tindex or name of output device\n";
        col() << SBOLD("\tLevelA/LevelB") << "\tspecifies 3G-SDI output level\n";
        col() << SBOLD("\t3D") << "\t\t3D stream will be received (see also HDMI3DPacking option)\n";
        col() << SBOLD("\taudio_level") << "\tset maximum attenuation for mic\n";
        col() << SBOLD("\thalf-duplex | full-duplex")
                << "\tset a profile that allows maximal number of simultaneous IOs / set device to better compatibility (3D, dual-link)\n";
        col() << SBOLD("\tHDR[=HDR|PQ|HLG|<int>|help]") << " - enable HDR metadata (optionally specifying EOTF, int 0-7 as per CEA 861.), help for extended help\n";
        col() << SBOLD("\tdrift_fix") << "       activates a time drift fix for the Decklink cards with resampler (experimental)\n";
        if (!full) {
                col() << SBOLD("\tconversion") << "\toutput size conversion, use '-d decklink:fullhelp' for list of conversions\n";
                col() << "\n\t(other options available, use \"" << SBOLD("fullhelp") << "\" to see complete list of options)\n";
        } else {
                col() << SBOLD("\tsingle-link/dual-link/quad-link") << "\tspecifies if the video output will be in a single-link (HD/3G/6G/12G), dual-link HD-SDI mode or quad-link HD/3G/6G/12G\n";
                col() << SBOLD("\ttimecode") << "\temit timecode\n";
                col() << SBOLD("\t[no-]quad-square") << " set Quad-link SDI is output in Square Division Quad Split mode\n";
                col() << SBOLD("\t[no-]low-latency") << " do not use low-latency mode (use regular scheduled mode; low-latency is default)\n";
                col() << SBOLD("\tconversion") << "\toutput size conversion, can be:\n" <<
                                SBOLD("\t\tnone") << " - no conversion\n" <<
                                SBOLD("\t\tltbx") << " - down-converted letterbox SD\n" <<
                                SBOLD("\t\tamph") << " - down-converted anamorphic SD\n" <<
                                SBOLD("\t\t720c") << " - HD720 to HD1080 conversion\n" <<
                                SBOLD("\t\tHWlb") << " - simultaneous output of HD and down-converted letterbox SD\n" <<
                                SBOLD("\t\tHWam") << " - simultaneous output of HD and down-converted anamorphic SD\n" <<
                                SBOLD("\t\tHWcc") << " - simultaneous output of HD and center cut SD\n" <<
                                SBOLD("\t\txcap") << " - simultaneous output of 720p and 1080p cross-conversion\n" <<
                                SBOLD("\t\tua7p") << " - simultaneous output of SD and up-converted anamorphic 720p\n" <<
                                SBOLD("\t\tua1i") << " - simultaneous output of SD and up-converted anamorphic 1080i\n" <<
                                SBOLD("\t\tu47p") << " - simultaneous output of SD and up-converted anamorphic widescreen aspect ratio 14:9 to 720p\n" <<
                                SBOLD("\t\tu41i") << " - simultaneous output of SD and up-converted anamorphic widescreen aspect ratio 14:9 to 1080i\n" <<
                                SBOLD("\t\tup7p") << " - simultaneous output of SD and up-converted pollarbox 720p\n" <<
                                SBOLD("\t\tup1i") << " - simultaneous output of SD and up-converted pollarbox 1080i\n";
                col() << SBOLD("\tHDMI3DPacking") << " can be (used in conjunction with \"3D\" option):\n" <<
				SBOLD("\t\tSideBySideHalf, LineByLine, TopAndBottom, FramePacking, LeftOnly, RightOnly\n");
                col() << SBOLD("\tUse1080PsF[=true|false|keep]") << " flag sets use of PsF on output instead of progressive (default is false)\n";
                col() << SBOLD("\tprofile=<P>") << "\tuse desired device profile:\n";
                print_bmd_device_profiles("\t\t");
                col() << SBOLD("\tmaxresample=<N>") << " maximum amount the resample delta can be when scaling is applied. Measured in Hz\n";
                col() << SBOLD("\tminresample=<N>") << " minimum amount the resample delta can be when scaling is applied. Measured in Hz\n";
                col() << SBOLD("\ttargetbuffer=<N>") << " target amount of samples to have in the buffer (per channel)\n";
                col() << SBOLD("\tkeep-settings") << "\tdo not apply any DeckLink settings by UG than required (keep user-selected defaults)\n";
                col() << SBOLD("\t<option_FourCC>=<value>") << "\tarbitrary BMD option (given a FourCC) and corresponding value\n";
        }

        col() << "\nRecognized pixel formats:";
        for_each(uv_to_bmd_codec_map.cbegin(), uv_to_bmd_codec_map.cend(), [](auto const &i) { col() << " " << SBOLD(get_codec_name(i.first)); } );
        cout << "\n";

        col() << "\nDevices:\n";
        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        bool com_initialized = false;
        deckLinkIterator = create_decklink_iterator(&com_initialized, true);
        if (deckLinkIterator == NULL) {
                return;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                string deviceName = bmd_get_device_name(deckLink);
                if (deviceName.empty()) {
                        deviceName = "(unable to get name)";
                }

                // *** Print the model name of the DeckLink card
                col() << "\t" << SBOLD(numDevices) << ") " << SBOLD(deviceName) << "\n";
                if (full) {
                        print_output_modes(deckLink);
                }

                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        if (!full) {
                col() << "(use \"" << SBOLD("fullhelp") << "\" to see device modes)\n";
        }

        deckLinkIterator->Release();

        decklink_uninitialize(&com_initialized);

        // If no DeckLink cards were found in the system, inform the user
        if (numDevices == 0)
        {
                log_msg(LOG_LEVEL_WARNING, "\nNo Blackmagic Design devices were found.\n");
                return;
        } 

        printf("\n");
        if (full) {
                print_decklink_version();
                printf("\n");
        }
}


static struct video_frame *
display_decklink_getf(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        assert(s->magic == DECKLINK_MAGIC);

        if (!s->initialized_video) {
                return nullptr;
        }

        struct video_frame *out = vf_alloc_desc(s->vid_desc);
        static auto dispose = [](struct video_frame *frame) {
                vf_free(frame);
        };
        out->callbacks.dispose = dispose;

        const int linesize = vc_get_linesize(s->vid_desc.width, s->vid_desc.color_spec);
        IDeckLinkMutableVideoFrame *deckLinkFrame = nullptr;
        lock_guard<mutex> lg(s->buffer_pool.lock);

        while (!s->buffer_pool.frame_queue.empty()) {
                auto tmp = s->buffer_pool.frame_queue.front();
                IDeckLinkMutableVideoFrame *frame = s->stereo ? dynamic_cast<DeckLink3DFrame *>(tmp)
                                                              : dynamic_cast<DeckLinkFrame *>(tmp);
                s->buffer_pool.frame_queue.pop();
                if (!frame || // wrong type
                    frame->GetWidth() != (long)s->vid_desc.width ||
                    frame->GetHeight() != (long)s->vid_desc.height ||
                    frame->GetRowBytes() != linesize || frame->GetPixelFormat() != s->pixelFormat) {
                        delete tmp;
                } else {
                        deckLinkFrame = frame;
                        deckLinkFrame->AddRef();
                        break;
                }
        }
        if (!deckLinkFrame) {
                deckLinkFrame = s->stereo
                                    ? DeckLink3DFrame::Create(s->vid_desc.width, s->vid_desc.height,
                                                              linesize, s->pixelFormat,
                                                              s->buffer_pool, s->requested_hdr_mode)
                                    : DeckLinkFrame::Create(s->vid_desc.width, s->vid_desc.height,
                                                            linesize, s->pixelFormat,
                                                            s->buffer_pool, s->requested_hdr_mode);
        }
        out->callbacks.dispose_udata = (void *) deckLinkFrame;

        deckLinkFrame->GetBytes((void **)&out->tiles[0].data);

        if (s->stereo) {
                IDeckLinkVideoFrame *deckLinkFrameRight = nullptr;
                DeckLink3DFrame *frame3D = dynamic_cast<DeckLink3DFrame *>(deckLinkFrame);
                assert(frame3D != nullptr);
                frame3D->GetFrameForRightEye(&deckLinkFrameRight);
                deckLinkFrameRight->GetBytes((void **)&out->tiles[1].data);
                // release immedieatelly (parent still holds the reference)
                deckLinkFrameRight->Release();
        }

        return out;
}

static void update_timecode(DeckLinkTimecode *tc, double fps)
{
        const float epsilon = 0.005;
        uint8_t hours, minutes, seconds, frames;
        BMDTimecodeBCD bcd;
        bool dropFrame = false;

        if(ceil(fps) - fps > epsilon) { /* NTSCi drop framecode  */
                dropFrame = true;
        }

        tc->GetComponents (&hours, &minutes, &seconds, &frames);
        frames++;

        if((double) frames > fps - epsilon) {
                frames = 0;
                seconds++;
                if(seconds >= 60) {
                        seconds = 0;
                        minutes++;
                        if(dropFrame) {
                                if(minutes % 10 != 0)
                                        seconds = 2;
                        }
                        if(minutes >= 60) {
                                minutes = 0;
                                hours++;
                                if(hours >= 24) {
                                        hours = 0;
                                }
                        }
                }
        }

        bcd = (frames % 10) | (frames / 10) << 4 | (seconds % 10) << 8 | (seconds / 10) << 12 | (minutes % 10)  << 16 | (minutes / 10) << 20 |
                (hours % 10) << 24 | (hours / 10) << 28;

        tc->SetBCD(bcd);
}

static int display_decklink_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        struct state_decklink *s = (struct state_decklink *)state;

        if (frame == NULL)
                return FALSE;

        assert(s->magic == DECKLINK_MAGIC);

        uint32_t i;
        s->deckLinkOutput->GetBufferedVideoFrameCount(&i); // writes always 0 in low-latency mode
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "putf - " << i << " frames buffered\n";
        long long max_frames = DIV_ROUNDED_UP(timeout_ns, (long long)(NS_IN_SEC / frame->fps)) + 2;
        if (timeout_ns == PUTF_DISCARD || i > max_frames) {
                if (timeout_ns != PUTF_DISCARD) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Frame dropped!\n");
                }
                IDeckLinkMutableVideoFrame *deckLinkFrame =
                    (IDeckLinkMutableVideoFrame *)frame->callbacks.dispose_udata;
                deckLinkFrame->Release();
                frame->callbacks.dispose(frame);
                return 1;
        }

        if (frame->color_spec == R10k && get_commandline_param(R10K_FULL_OPT) == nullptr) {
                for (unsigned i = 0; i < frame->tile_count; ++i) {
                        r10k_full_to_limited(frame->tiles[i].data, frame->tiles[i].data, frame->tiles[i].data_len);
                }
        }

        IDeckLinkMutableVideoFrame *deckLinkFrame =
            (IDeckLinkMutableVideoFrame *)frame->callbacks.dispose_udata;
        if (s->emit_timecode) {
                deckLinkFrame->SetTimecode(bmdTimecodeRP188Any, s->timecode);
        }

        if (s->low_latency) {
                s->deckLinkOutput->DisplayVideoFrameSync(deckLinkFrame);
                deckLinkFrame->Release();
        } else {
                s->deckLinkOutput->ScheduleVideoFrame(deckLinkFrame,
                                                      s->frames * s->frameRateDuration,
                                                      s->frameRateDuration, s->frameRateScale);
        }
        s->frames++;
        if(s->emit_timecode) {
                update_timecode(s->timecode, s->vid_desc.fps);
        }

        frame->callbacks.dispose(frame);

        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - s->t0).count() > 5) {
                LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << s->delegate->frames_late << " frames late, "
                                << s->delegate->frames_dropped << " dropped, "
                                << s->delegate->frames_flushed << " flushed cumulative\n";
                s->t0 = now;
        }

        return 0;
}

static BMDDisplayMode get_mode(IDeckLinkOutput *deckLinkOutput, struct video_desc desc, BMDTimeValue *frameRateDuration,
		BMDTimeScale        *frameRateScale, bool stereo)
{	IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;
        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fatal: cannot create display mode iterator.\n");
                return bmdModeUnknown;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                BMD_STR modeNameString;
                if (deckLinkDisplayMode->GetName(&modeNameString) == S_OK)
                {
                        char *modeNameCString = get_cstr_from_bmd_api_str(modeNameString);
                        if (deckLinkDisplayMode->GetWidth() == (long) desc.width &&
                                        deckLinkDisplayMode->GetHeight() == (long) desc.height)
                        {
                                double displayFPS;
                                BMDFieldDominance dominance;
                                bool interlaced;

                                dominance = deckLinkDisplayMode->GetFieldDominance();
                                if (dominance == bmdLowerFieldFirst ||
                                                dominance == bmdUpperFieldFirst) {
					if (dominance == bmdLowerFieldFirst) {
						log_msg(LOG_LEVEL_WARNING, MOD_NAME "Lower field first format detected, fields can be switched! If so, please report a bug to " PACKAGE_BUGREPORT "\n");
					}
                                        interlaced = true;
                                } else { // progressive, psf, unknown
                                        interlaced = false;
                                }

                                deckLinkDisplayMode->GetFrameRate(frameRateDuration,
                                                frameRateScale);
                                displayFPS = (double) *frameRateScale / *frameRateDuration;
                                if (fabs(desc.fps - displayFPS) < 0.01 && (desc.interlacing == INTERLACED_MERGED) == interlaced) {
                                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Selected mode: %s%s\n", modeNameCString,
                                                        stereo ? " (3D)" : "");
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        release_bmd_api_str(modeNameString);
                                        free(modeNameCString);
                                        deckLinkDisplayMode->Release();
                                        break;
                                }
                        }
                        release_bmd_api_str(modeNameString);
                        free((void *) modeNameCString);
                }
                deckLinkDisplayMode->Release();
        }
        displayModeIterator->Release();
        
        return displayMode;
}

/**
 * @todo
 * In non-low-latency mode, StopScheduledPlayback should be called. However, since this
 * function is called from different thread than audio-related stuff and these things
 * are not synchronized in any way, it looks like to be more appropriate not to call it,
 * as it doesn't break things up. In low latency mode, this is not an issue.
 */
static int
display_decklink_reconfigure_video(void *state, struct video_desc desc)
{
        struct state_decklink            *s = (struct state_decklink *)state;

        BMDDisplayMode                    displayMode;
        BMD_BOOL                          supported;
        HRESULT                           result;

        unique_lock<mutex> lk(s->reconfiguration_lock);

        assert(s->magic == DECKLINK_MAGIC);
        
        s->initialized_video = false;
        s->vid_desc = desc;

        if (desc.color_spec == R10k && get_commandline_param(R10K_FULL_OPT) == nullptr) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Using limited range R10k as specified by BMD, use '--param "
                                R10K_FULL_OPT "' to override.\n");
        }

        auto it = std::find_if(uv_to_bmd_codec_map.begin(),
                        uv_to_bmd_codec_map.end(),
                        [&desc](const std::pair<codec_t, BMDPixelFormat>& el){ return el.first == desc.color_spec; });
        if (it == uv_to_bmd_codec_map.end()) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported pixel format!\n");
                return FALSE;
        }
        s->pixelFormat = it->second;

        if (s->initialized_video) {
                CALL_AND_CHECK(s->deckLinkOutput->DisableVideoOutput(),
                               "DisableVideoOutput");
                s->initialized_video = false;
        }

        if (desc.tile_count <= 2 && desc.tile_count != (s->stereo ? 2 : 1)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Stereo %s enabled but receiving %u streams. %sabling "
                                "it. This behavior is experimental so please report any problems. "
                                "You can also specify (or not) `3D` option explicitly.\n"
                                , s->stereo ? "" : "not", desc.tile_count, s->stereo ? "dis" : "en");
                s->stereo = !s->stereo;
        }

        if (s->stereo) {
                bmd_check_stereo_profile(s->deckLink);
                if ((int) desc.tile_count != 2) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "In stereo mode exactly "
                                        "2 streams expected, %d received.\n", desc.tile_count);
                        return FALSE;
                }
        } else {
                if ((int) desc.tile_count == 2) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Received 2 streams but stereo mode is not enabled! Did you forget a \"3D\" parameter?\n");
                }
        }

        BMDVideoOutputFlags outputFlags = bmdVideoOutputFlagDefault;
        BMDSupportedVideoModeFlags supportedFlags = bmdSupportedVideoModeDefault;

        displayMode =
            get_mode(s->deckLinkOutput, desc, &s->frameRateDuration, &s->frameRateScale, s->stereo);
        if (displayMode == bmdModeUnknown) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find suitable video mode.\n");
                return FALSE;
        }

        if (s->emit_timecode) {
                outputFlags = (BMDVideoOutputFlags)(outputFlags | bmdVideoOutputRP188);
        }

        if (s->stereo) {
                outputFlags = (BMDVideoOutputFlags)(outputFlags | bmdVideoOutputDualStream3D);
                supportedFlags = (BMDSupportedVideoModeFlags)(supportedFlags |
                                                              bmdSupportedVideoModeDualStream3D);
        }

        const bmd_option subsampling_444(codec_is_a_rgb(desc.color_spec),
                                         false); // we don't have pixfmt for 444 YCbCr
        subsampling_444.device_write(s->deckLinkConfiguration, bmdDeckLinkConfig444SDIVideoOutput,
                                     MOD_NAME);

        if (!s->keep_device_defaults &&
            s->device_options.find(bmdDeckLinkConfigSDIOutputLinkConfiguration) ==
                s->device_options.end()) {
                const int64_t link = desc.width == 7680 ? bmdLinkConfigurationQuadLink
                                                        : bmdLinkConfigurationSingleLink;
                bmd_option(link).device_write(s->deckLinkConfiguration,
                                              bmdDeckLinkConfigSDIOutputLinkConfiguration,
                                              MOD_NAME);
        }

        int64_t link = 0;
        s->deckLinkConfiguration->GetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, &link);
        if (!s->keep_device_defaults && s->profile_req.is_default() &&
            link == bmdLinkConfigurationQuadLink) {
                LOG(LOG_LEVEL_WARNING)
                    << MOD_NAME "Quad-link detected - setting 1-subdevice-1/2-duplex "
                                "profile automatically, use 'profile=keep' to override.\n";
                decklink_set_profile(
                    s->deckLink, bmd_option((int64_t)bmdProfileOneSubDeviceHalfDuplex), s->stereo);
        } else if (link == bmdLinkConfigurationQuadLink &&
                   (!s->profile_req.keep() &&
                    s->profile_req.get_int() != bmdProfileOneSubDeviceHalfDuplex)) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Setting quad-link and an incompatible device "
                                                   "profile may not be supported!\n";
        }

        BMD_BOOL quad_link_supp = BMD_FALSE;
        if (s->deckLinkAttributes != nullptr &&
            s->deckLinkAttributes->GetFlag(BMDDeckLinkSupportsQuadLinkSDI, &quad_link_supp) ==
                S_OK &&
            quad_link_supp == BMD_TRUE) {
                s->quad_square_division_split.device_write(
                    s->deckLinkConfiguration,
                    bmdDeckLinkConfigQuadLinkSDIVideoOutputSquareDivisionSplit, MOD_NAME);
        }

        const BMDVideoOutputConversionMode conversion_mode =
            s->device_options.find(bmdDeckLinkConfigVideoOutputConversionMode) !=
                    s->device_options.end()
                ? (BMDVideoOutputConversionMode)s->device_options
                      .at(bmdDeckLinkConfigVideoOutputConversionMode)
                      .get_int()
                : (BMDVideoOutputConversionMode)bmdNoVideoOutputConversion;
        EXIT_IF_FAILED(s->deckLinkOutput->DoesSupportVideoMode(
                           bmdVideoConnectionUnspecified, displayMode, s->pixelFormat,
                           conversion_mode, supportedFlags, nullptr, &supported),
                       "DoesSupportVideoMode");
        if (!supported) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "Requested parameters "
                                 "combination not supported - %d * %dx%d@%f, timecode %s.\n",
                        desc.tile_count, desc.width, desc.height, desc.fps,
                        (outputFlags & bmdVideoOutputRP188 ? "ON" : "OFF"));
                return FALSE;
        }

        result = s->deckLinkOutput->EnableVideoOutput(displayMode, outputFlags);
        if (FAILED(result)) {
                if (result == E_ACCESSDENIED) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "Unable to access the hardware or output "
                                         "stream currently active (another application "
                                         "using it?).\n");
                } else {
                        LOG(LOG_LEVEL_ERROR)
                            << MOD_NAME << "EnableVideoOutput: " << bmd_hresult_to_string(result)
                            << "\n";
                }
                return FALSE;
        }

        // This workaround is needed (at least) for Decklink Extreme 4K when capturing
        // (possibly from another process) and when playback is in low-latency mode.
        // When video is enabled after audio, audio playback becomes silent without
        // an error.
        if (s->initialized_audio) {
                EXIT_IF_FAILED(s->deckLinkOutput->DisableAudioOutput(), "DisableAudioOutput");
                EXIT_IF_FAILED(s->deckLinkOutput->EnableAudioOutput(
                                   bmdAudioSampleRate48kHz,
                                   s->aud_desc.bps == 2 ? bmdAudioSampleType16bitInteger
                                                        : bmdAudioSampleType32bitInteger,
                                   s->aud_desc.ch_count, bmdAudioOutputStreamContinuous),
                               "EnableAudioOutput");
        }

        if (!s->low_latency) {
                result = s->deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale,
                                                                   (double)s->frameRateDuration);
                if (FAILED(result)) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "StartScheduledPlayback (video): "
                                             << bmd_hresult_to_string(result) << "\n";
                        s->deckLinkOutput->DisableVideoOutput();
                        return FALSE;
                }
        }

        s->initialized_video = true;
        return TRUE;
}

static void display_decklink_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        UNUSED(deleter);
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;

        *count = 0;
        *available_cards = nullptr;

        bool com_initialized = false;
        deckLinkIterator = create_decklink_iterator(&com_initialized, false);
        if (deckLinkIterator == NULL) {
                return;
        }

        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                string deviceName = bmd_get_device_name(deckLink);
		if (deviceName.empty()) {
			deviceName = "(unknown)";
		}

                *count += 1;
                *available_cards = (struct device_info *)
                        realloc(*available_cards, *count * sizeof(struct device_info));
                memset(*available_cards + *count - 1, 0, sizeof(struct device_info));
                snprintf((*available_cards)[*count - 1].dev, sizeof (*available_cards)[*count - 1].dev, ":device=%d", *count - 1);
                snprintf((*available_cards)[*count - 1].extra, sizeof (*available_cards)[*count - 1].extra, "\"embeddedAudioAvailable\":\"t\"");
                (*available_cards)[*count - 1].repeatable = false;

		strncpy((*available_cards)[*count - 1].name, deviceName.c_str(),
				sizeof (*available_cards)[*count - 1].name - 1);

                dev_add_option(&(*available_cards)[*count - 1], "3D", "3D", "3D", ":3D", true);
                dev_add_option(&(*available_cards)[*count - 1], "Profile", "Duplex profile can be one of: 1dhd, 2dhd, 2dfd, 4dhd, keep", "profile", ":profile=", false);

                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        deckLinkIterator->Release();
        decklink_uninitialize(&com_initialized);
}

static bool settings_init(struct state_decklink *s, const char *fmt,
                string &cardId,
                int *audio_consumer_levels) {
        if (strlen(fmt) == 0) {
                return true;
        }

        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);
        char *ptr;
        char *save_ptr = nullptr;

        ptr = strtok_r(tmp, ":", &save_ptr);
        assert(ptr != nullptr);
        int i = 0;
        bool first_option_is_device = true;
        while (ptr[i] != '\0') {
                if (!isdigit(ptr[i]) && ptr[i] != ',') {
                        first_option_is_device = false;
                        break;
                }
                i++;
        }
        if (first_option_is_device) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unnamed device index "
                                "deprecated. Use \"device=%s\" instead.\n", ptr);
                cardId = ptr;
                ptr = strtok_r(nullptr, ":", &save_ptr);
        }

        while (ptr != nullptr)  {
                if (strncasecmp(ptr, "device=", strlen("device=")) == 0 ||
                    strstr(ptr, "d=") == ptr) {
                        cardId = strchr(ptr, '=') + 1;
                } else if (strcasecmp(ptr, "3D") == 0) {
                        s->stereo = true;
                } else if (strcasecmp(ptr, "timecode") == 0) {
                        s->emit_timecode = true;
                } else if (strcasecmp(ptr, "single-link") == 0) {
                        s->device_options[bmdDeckLinkConfigSDIOutputLinkConfiguration].set_int(bmdLinkConfigurationSingleLink);
                } else if (strcasecmp(ptr, "dual-link") == 0) {
                        s->device_options[bmdDeckLinkConfigSDIOutputLinkConfiguration].set_int(bmdLinkConfigurationDualLink);
                } else if (strcasecmp(ptr, "quad-link") == 0) {
                        s->device_options[bmdDeckLinkConfigSDIOutputLinkConfiguration].set_int(bmdLinkConfigurationQuadLink);
                } else if (strstr(ptr, "profile=") == ptr) {
                        s->profile_req.parse(ptr);
                } else if (strcasecmp(ptr, "full-duplex") == 0) {
                        s->profile_req.set_int(bmdProfileOneSubDeviceFullDuplex);
                } else if (strcasecmp(ptr, "half-duplex") == 0) {
                        s->profile_req.set_int(bmdDuplexHalf);
                } else if (strcasecmp(ptr, "LevelA") == 0) {
                        s->device_options[bmdDeckLinkConfigSMPTELevelAOutput].set_flag(true);
                } else if (strcasecmp(ptr, "LevelB") == 0) {
                        s->device_options[bmdDeckLinkConfigSMPTELevelAOutput].set_flag(false);
                } else if (strncasecmp(ptr, "HDMI3DPacking=", strlen("HDMI3DPacking=")) == 0) {
                        char *packing = ptr + strlen("HDMI3DPacking=");
                        if (strcasecmp(packing, "SideBySideHalf") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingSidebySideHalf);
                        } else if (strcasecmp(packing, "LineByLine") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingLinebyLine);
                        } else if (strcasecmp(packing, "TopAndBottom") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingTopAndBottom);
                        } else if (strcasecmp(packing, "FramePacking") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingFramePacking);
                        } else if (strcasecmp(packing, "LeftOnly") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingRightOnly);
                        } else if (strcasecmp(packing, "RightOnly") == 0) {
                                s->device_options[bmdDeckLinkConfigHDMI3DPackingFormat].set_int(bmdVideo3DPackingLeftOnly);
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown HDMI 3D packing %s.\n", packing);
                                return false;
                        }
                } else if (strncasecmp(ptr, "audio_level=", strlen("audio_level=")) == 0) {
                        if (strcasecmp(ptr + strlen("audio_level="), "false") || strcasecmp(ptr + strlen("audio_level="), "mic") == 0) {
                                *audio_consumer_levels = 0;
                        } else {
                                *audio_consumer_levels = 1;
                        }
                } else if (strncasecmp(ptr, "conversion=",
                                        strlen("conversion=")) == 0) {
                        s->device_options[bmdDeckLinkConfigVideoOutputConversionMode].parse(strchr(ptr, '=') + 1);
                } else if (is_prefix_of(ptr, "Use1080pNotPsF") || is_prefix_of(ptr, "Use1080PsF")) {
                        s->device_options[bmdDeckLinkConfigOutput1080pAsPsF].parse(strchr(ptr, '=') + 1);
                        if (strncasecmp(ptr, "Use1080pNotPsF", strlen("Use1080pNotPsF")) == 0) { // compat, inverse
                                s->device_options[bmdDeckLinkConfigOutput1080pAsPsF].set_flag(s->device_options[bmdDeckLinkConfigOutput1080pAsPsF].get_flag());
                        }
                } else if (strcasecmp(ptr, "low-latency") == 0 || strcasecmp(ptr, "no-low-latency") == 0) {
                        s->low_latency = strcasecmp(ptr, "low-latency") == 0;
                } else if (strcasecmp(ptr, "quad-square") == 0 || strcasecmp(ptr, "no-quad-square") == 0) {
                        s->quad_square_division_split.set_flag(strcasecmp(ptr, "quad-square") == 0);
                } else if (strncasecmp(ptr, "hdr", strlen("hdr")) == 0) {
                        s->requested_hdr_mode.EOTF = static_cast<int64_t>(HDR_EOTF::HDR); // default
                        if (strncasecmp(ptr, "hdr=", strlen("hdr=")) == 0) {
                                try {
                                        s->requested_hdr_mode.Init(ptr + strlen("hdr="));
                                } catch (ug_no_error const &e) {
                                        return false;
                                } catch (exception const &e) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "HDR mode init: " << e.what() << "\n";
                                        return false;
                                }
                        }
                } else if (strstr(ptr, "keep-settings") == ptr) {
                        s->keep_device_defaults = true;
                } else if (strstr(ptr, "drift_fix") == ptr) {
                        s->audio_drift_fixer.enable();
                } else if (strncasecmp(ptr, "maxresample=", strlen("maxresample=")) == 0) {
                        s->audio_drift_fixer.set_max_hz(parse_uint32(strchr(ptr, '=') + 1));
                } else if (strncasecmp(ptr, "minresample=", strlen("minresample=")) == 0) {
                        s->audio_drift_fixer.set_min_hz(parse_uint32(strchr(ptr, '=') + 1));
                } else if (strncasecmp(ptr, "targetbuffer=", strlen("targetbuffer=")) == 0) {
                        s->audio_drift_fixer.set_target_buffer(parse_uint32(strchr(ptr, '=') + 1));
                } else if ((strchr(ptr, '=') != nullptr && strchr(ptr, '=') - ptr == 4) || strlen(ptr) == 4) {
                        s->device_options[(BMDDeckLinkConfigurationID) bmd_read_fourcc(ptr)].parse(strchr(ptr, '=') + 1);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "unknown option in config string: %s\n", ptr);
                        return false;
                }
                ptr = strtok_r(nullptr, ":", &save_ptr);
        }

        return true;
}

static void *display_decklink_init(struct module *parent, const char *fmt, unsigned int flags)
{
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        string                                          cardId("0");
        int                                             dnum = 0;
        IDeckLinkConfiguration*         deckLinkConfiguration = NULL;
        // for Decklink Studio which has switchable XLR - analog 3 and 4 or AES/EBU 3,4 and 5,6
        BMDAudioOutputAnalogAESSwitch audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
        int audio_consumer_levels = -1;

        if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
                show_help(strcmp(fmt, "fullhelp") == 0);
                return INIT_NOERR;
        }

        if (!blackmagic_api_version_check()) {
                return NULL;
        }

        auto *s = new state_decklink();
        s->audio_drift_fixer.set_root(get_root_module(parent));

        if (!settings_init(s, fmt, cardId, &audio_consumer_levels)) {
                delete s;
                return NULL;
        }

        // Initialize the DeckLink API
        deckLinkIterator = create_decklink_iterator(&s->com_initialized, true);
        if (!deckLinkIterator)
        {
                delete s;
                return NULL;
        }

        // Connect to the first DeckLink instance
        IDeckLink    *deckLink;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                string deviceName = bmd_get_device_name(deckLink);

                if (!deviceName.empty() && deviceName == cardId) {
                        s->deckLink = deckLink;
                        break;
                }
                if (isdigit(cardId.c_str()[0]) && dnum == atoi(cardId.c_str())) {
                        s->deckLink = deckLink;
                        break;
                }

                if (deckLink != NULL) {
                        deckLink->Release();
                }
                dnum++;
        }
        deckLinkIterator->Release();
        if (s->deckLink == nullptr) {
                LOG(LOG_LEVEL_ERROR) << "No DeckLink PCI card " << cardId << " found\n";
                return FALSE;
        }
        // Print the model name of the DeckLink card
        string deviceName = bmd_get_device_name(s->deckLink);
        if (!deviceName.empty()) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Using device " << deviceName << "\n";
        }

        if(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                s->play_audio = true;
                switch(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                        case DISPLAY_FLAG_AUDIO_EMBEDDED:
                                audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
                                break;
                        case DISPLAY_FLAG_AUDIO_AESEBU:
                                audioConnection = bmdAudioOutputSwitchAESEBU;
                                break;
                        case DISPLAY_FLAG_AUDIO_ANALOG:
                                audioConnection = bmdAudioOutputSwitchAnalog;
                                break;
                        default:
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupporetd audio connection.\n");
                                abort();
                }
        } else {
                s->play_audio = false;
        }
        
        if(s->emit_timecode) {
                s->timecode = new DeckLinkTimecode;
        } else {
                s->timecode = NULL;
        }

        if (!s->keep_device_defaults && !s->profile_req.keep()) {
                decklink_set_profile(s->deckLink, s->profile_req, s->stereo);
        }

        // Get IDeckLinkAttributes object
        IDeckLinkProfileAttributes *deckLinkAttributes = NULL;
        result = s->deckLink->QueryInterface(IID_IDeckLinkProfileAttributes,
                                             reinterpret_cast<void **>(&deckLinkAttributes));
        if (result != S_OK) {
                log_msg(LOG_LEVEL_WARNING, "Could not query device attributes.\n");
        }
        s->deckLinkAttributes = deckLinkAttributes;

        // Obtain the audio/video output interface (IDeckLinkOutput)
        if ((result = s->deckLink->QueryInterface(
                 IID_IDeckLinkOutput, reinterpret_cast<void **>(&s->deckLinkOutput))) != S_OK) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "Could not obtain the IDeckLinkOutput interface: %08x\n",
                        (int)result);
                return FALSE;
        }

        // Query the DeckLink for its configuration interface
        result = s->deckLink->QueryInterface(IID_IDeckLinkConfiguration,
                                             reinterpret_cast<void **>(&deckLinkConfiguration));
        s->deckLinkConfiguration = deckLinkConfiguration;
        if (result != S_OK) {
                log_msg(LOG_LEVEL_ERROR,
                        "Could not obtain the IDeckLinkConfiguration interface: %08x\n",
                        (int)result);
                return FALSE;
        }

        for (const auto &o : s->device_options) {
                if (s->keep_device_defaults && !o.second.is_user_set()) {
                                continue;
                }
                if (!o.second.device_write(deckLinkConfiguration, o.first, MOD_NAME)) {
                                return FALSE;
                }
        }

        if (s->requested_hdr_mode.EOTF != static_cast<int64_t>(HDR_EOTF::NONE)) {
                BMD_BOOL hdr_supp = BMD_FALSE;
                if (s->deckLinkAttributes == nullptr ||
                    s->deckLinkAttributes->GetFlag(BMDDeckLinkSupportsHDRMetadata, &hdr_supp) !=
                        S_OK) {
                                LOG(LOG_LEVEL_WARNING)
                                    << MOD_NAME
                                    << "HDR requested, but unable to validate HDR support. Will "
                                       "try to pass it anyway which may result in blank image if "
                                       "not supported - remove the option if so.\n";
                } else {
                        if (hdr_supp != BMD_TRUE) {
                                LOG(LOG_LEVEL_ERROR)
                                    << MOD_NAME
                                    << "HDR requested, but card doesn't support that.\n";
                                return FALSE;
                        }
                }

                BMD_BOOL rec2020_supp = BMD_FALSE;
                if (s->deckLinkAttributes == nullptr ||
                    s->deckLinkAttributes->GetFlag(BMDDeckLinkSupportsColorspaceMetadata,
                                                   &rec2020_supp) != S_OK) {
                        LOG(LOG_LEVEL_WARNING)
                            << MOD_NAME << "Cannot check Rec. 2020 color space metadata support.\n";
                } else {
                        if (rec2020_supp != BMD_TRUE) {
                                LOG(LOG_LEVEL_WARNING)
                                    << MOD_NAME
                                    << "Rec. 2020 color space metadata not supported.\n";
                                }
                }
        }

        if (s->play_audio) {
                /* Actually no action is required to set audio connection because Blackmagic card
                 * plays audio through all its outputs (AES/SDI/analog) ....
                 */
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Audio output set to: "
                                    << bmd_get_audio_connection_name(audioConnection) << "\n";
                /*
                 * .... one exception is a card that has switchable cables between AES/EBU and
                 * analog. (But this applies only for channels 3 and above.)
                 */
                if (audioConnection != 0) { // we will set switchable AESEBU or analog
                        result = deckLinkConfiguration->SetInt(
                            bmdDeckLinkConfigAudioOutputAESAnalogSwitch, audioConnection);
                        if (result == S_OK) { // has switchable channels
                                log_msg(LOG_LEVEL_INFO,
                                        MOD_NAME "Card with switchable audio channels detected. "
                                                 "Switched to correct format.\n");
                        } else if (result == E_NOTIMPL) {
                                // normal case - without switchable channels
                        } else {
                                log_msg(LOG_LEVEL_WARNING,
                                        MOD_NAME "Unable to switch audio output for channels 3 or "
                                                 "above although \n"
                                                 "card shall support it. Check if it is ok. "
                                                 "Continuing anyway.\n");
                        }
                }

                if (audio_consumer_levels != -1) {
                                result = deckLinkConfiguration->SetFlag(
                                    bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                    audio_consumer_levels == 1 ? true : false);
                                if (result != S_OK) {
                                log_msg(LOG_LEVEL_WARNING,
                                        MOD_NAME "Unable set output audio consumer levels.\n");
                                }
                }
        }

        s->delegate = new PlaybackDelegate();
        // Provide this class as a delegate to the audio and video output interfaces
        if (!s->low_latency) {
                s->deckLinkOutput->SetScheduledFrameCompletionCallback(s->delegate);
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Scheduled playback is obsolescent and may be removed in future. "
                                 "Please let us know if you are using this mode.\n");
        }
        // s->state.at(i).deckLinkOutput->DisableAudioOutput();

        s->frames = 0;
        s->initialized_audio = s->initialized_video = false;

        return (void *)s;
}

#define RELEASE_IF_NOT_NULL(x) if (x != nullptr) x->Release();
static void display_decklink_done(void *state)
{
        debug_msg("display_decklink_done\n"); /* TOREMOVE */
        struct state_decklink *s = (struct state_decklink *)state;

        assert (s != NULL);

        if (s->initialized_video) {
                if (!s->low_latency) {
                                CALL_AND_CHECK(
                                    s->deckLinkOutput->StopScheduledPlayback(0, nullptr, 0),
                                    "StopScheduledPlayback");
                }

                CALL_AND_CHECK(s->deckLinkOutput->DisableVideoOutput(), "DisableVideoOutput");
        }

        if (s->initialized_audio) {
                CALL_AND_CHECK(s->deckLinkOutput->DisableAudioOutput(),
                               "DisableAudiioOutput");
        }

        RELEASE_IF_NOT_NULL(s->deckLinkAttributes);
        RELEASE_IF_NOT_NULL(s->deckLinkConfiguration);
        RELEASE_IF_NOT_NULL(s->deckLinkOutput);
        RELEASE_IF_NOT_NULL(s->deckLink);

        delete s->delegate;

        while (!s->buffer_pool.frame_queue.empty()) {
                auto tmp = s->buffer_pool.frame_queue.front();
                s->buffer_pool.frame_queue.pop();
                delete tmp;
        }

        delete s->timecode;

        decklink_uninitialize(&s->com_initialized);
        delete s;
}

/**
 * This function returns true if any display mode and any output supports the
 * codec. The codec, however, may not be supported with actual video mode.
 *
 * @todo For UltraStudio Pro DoesSupportVideoMode returns E_FAIL on not supported
 * pixel formats instead of setting supported to false.
 */
static bool decklink_display_supports_codec(IDeckLinkOutput *deckLinkOutput, BMDPixelFormat pf) {
        IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;

        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator))) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fatal: cannot create display mode iterator.\n");
                return false;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK) {
                BMD_BOOL supported;
                HRESULT res = deckLinkOutput->DoesSupportVideoMode(bmdVideoConnectionUnspecified, deckLinkDisplayMode->GetDisplayMode(), pf, bmdNoVideoOutputConversion, bmdSupportedVideoModeDefault, nullptr, &supported);
                deckLinkDisplayMode->Release();
                if (res != S_OK) {
                        CALL_AND_CHECK(res, "DoesSupportVideoMode");
                        continue;
                }
                if (supported) {
                        displayModeIterator->Release();
                        return true;
                }
        }
        displayModeIterator->Release();

        return false;
}

static int display_decklink_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decklink *s = (struct state_decklink *)state;
        vector<codec_t> codecs(uv_to_bmd_codec_map.size());
        int rgb_shift[] = {16, 8, 0};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int count = 0;
        for (auto & c : uv_to_bmd_codec_map) {
                if (decklink_display_supports_codec(s->deckLinkOutput, c.second)) {
                        codecs[count++] = c.first;
                }
        }
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codec_t) * count <= *len) {
                                memcpy(val, codecs.data(), sizeof(codec_t) * count);
                                *len = sizeof(codec_t) * count;
                        } else {
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_3D;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                if (desc->ch_count <= 2) {
                                        desc->ch_count = 2;
                                } else if (desc->ch_count > 2 && desc->ch_count <= 8) {
                                        desc->ch_count = 8;
                                } else {
                                        desc->ch_count = 16;
                                }
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 4;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

/*
 * AUDIO
 */
static void display_decklink_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = frame->data_len / (frame->bps * frame->ch_count);

        assert(s->play_audio);

        uint32_t sampleFramesWritten;

        uint32_t buffered = 0;
        s->deckLinkOutput->GetBufferedAudioSampleFrameCount(&buffered);
        if (buffered == 0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "audio buffer underflow!\n";
        }

        if (s->low_latency) {
                HRESULT res = s->deckLinkOutput->WriteAudioSamplesSync(frame->data, sampleFrameCount,
                                &sampleFramesWritten);
                if (FAILED(res)) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WriteAudioSamplesSync failed.\n");
                        return;
                }
        } else {
                s->deckLinkOutput->ScheduleAudioSamples(frame->data, sampleFrameCount, 0,
                                0, &sampleFramesWritten);
        }
        if (sampleFramesWritten != sampleFrameCount) {
                ostringstream details_oss;
                if (log_level >= LOG_LEVEL_VERBOSE) {
                        details_oss
                            << " (" << sampleFramesWritten << " written, "
                            << sampleFrameCount - sampleFramesWritten
                            << " dropped, " << buffered << " buffer size)";
                }
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "audio buffer overflow!"
                                       << details_oss.str() << "\n";
        }
        s->audio_drift_fixer.update(buffered, sampleFrameCount, sampleFramesWritten);
}

static int display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        unique_lock<mutex> lk(s->reconfiguration_lock);

        assert(s->play_audio);

        if (s->initialized_audio) {
                CALL_AND_CHECK(s->deckLinkOutput->DisableAudioOutput(),
                                "DisableAudioOutput");
                s->initialized_audio = false;
        }

        if (channels != 2 && channels != 8 &&
                        channels != 16) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "requested channel count isn't supported: "
                        "%d\n", channels);
                return FALSE;
        }
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        sample_rate != 48000) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "audio format isn't supported: "
                        "samples: %d, sample rate: %d\n",
                        quant_samples, sample_rate);
                return FALSE;
        }
        switch(quant_samples) {
                case 16:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 32:
                        sample_type = bmdAudioSampleType32bitInteger;
                        break;
                default:
                        return FALSE;
        }
                        
        EXIT_IF_FAILED(s->deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        channels,
                        bmdAudioOutputStreamContinuous),
                "EnableAudioOutput");

        if (!s->low_latency) {
                // This will most certainly fail because it is started with in video
                // reconfigure. However, this doesn't seem to bother, anyway.
                CALL_AND_CHECK(s->deckLinkOutput->StartScheduledPlayback(
                                   0, s->frameRateScale, s->frameRateDuration),
                               "StartScheduledPlayback (audio)");
        }

        s->aud_desc = { quant_samples / 8, sample_rate, channels, AC_PCM };

        s->initialized_audio = true;
        return TRUE;
}

#ifndef WIN32
static bool operator==(const REFIID & first, const REFIID & second){
    return (first.byte0 == second.byte0) &&
        (first.byte1 == second.byte1) &&
        (first.byte2 == second.byte2) &&
        (first.byte3 == second.byte3) &&
        (first.byte4 == second.byte4) &&
        (first.byte5 == second.byte5) &&
        (first.byte6 == second.byte6) &&
        (first.byte7 == second.byte7) &&
        (first.byte8 == second.byte8) &&
        (first.byte9 == second.byte9) &&
        (first.byte10 == second.byte10) &&
        (first.byte11 == second.byte11) &&
        (first.byte12 == second.byte12) &&
        (first.byte13 == second.byte13) &&
        (first.byte14 == second.byte14) &&
        (first.byte15 == second.byte15);
}
#endif

HRESULT DeckLinkFrame::QueryInterface(REFIID iid, LPVOID *ppv)
{
#ifdef _WIN32
        IID                     iunknown = IID_IUnknown;
#else
        CFUUIDBytes             iunknown = CFUUIDGetUUIDBytes(IUnknownUUID);
#endif
        HRESULT                 result          = S_OK;

        if (ppv == nullptr) {
                return E_INVALIDARG;
        }

        // Initialise the return result
        *ppv = nullptr;

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "DeckLinkFrame QueryInterface " << iid << "\n";
        if (iid == iunknown) {
                *ppv = this;
                AddRef();
        } else if (iid == IID_IDeckLinkVideoFrame) {
                *ppv = static_cast<IDeckLinkVideoFrame*>(this);
                AddRef();
        } else if (iid == IID_IDeckLinkVideoFrameMetadataExtensions) {
                if (m_metadata.EOTF == static_cast<int64_t>(HDR_EOTF::NONE)) {
                        result = E_NOINTERFACE;
                } else {
                        *ppv = static_cast<IDeckLinkVideoFrameMetadataExtensions*>(this);
                        AddRef();
                }
        } else {
                result = E_NOINTERFACE;
        }

        return result;

        return E_NOINTERFACE;
}

ULONG DeckLinkFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLinkFrame::Release()
{
        if (--ref == 0) {
                lock_guard<mutex> lg(buffer_pool.lock);
                buffer_pool.frame_queue.push(this);
        }
	return ref;
}

DeckLinkFrame::DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & bp, HDRMetadata const & hdr_metadata)
	: width(w), height(h), rawBytes(rb), pixelFormat(pf), data(new char[rb * h]), timecode(NULL), ref(1l),
        buffer_pool(bp)
{
        clear_video_buffer(reinterpret_cast<unsigned char *>(data.get()), rawBytes, rawBytes, height,
                        pf == bmdFormat8BitYUV ? UYVY : (pf == bmdFormat10BitYUV ? v210 : RGBA));
        m_metadata = hdr_metadata;
}

DeckLinkFrame *DeckLinkFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, const HDRMetadata & hdr_metadata)
{
        return new DeckLinkFrame(width, height, rawBytes, pixelFormat, buffer_pool, hdr_metadata);
}

DeckLinkFrame::~DeckLinkFrame() 
{
}

long DeckLinkFrame::GetWidth ()
{
        return width;
}

long DeckLinkFrame::GetHeight ()
{
        return height;
}

long DeckLinkFrame::GetRowBytes ()
{
        return rawBytes;
}

BMDPixelFormat DeckLinkFrame::GetPixelFormat ()
{
        return pixelFormat;
}

BMDFrameFlags DeckLinkFrame::GetFlags ()
{
        return m_metadata.EOTF == static_cast<int64_t>(HDR_EOTF::NONE) ? bmdFrameFlagDefault : bmdFrameContainsHDRMetadata;
}

HRESULT DeckLinkFrame::GetBytes (/* out */ void **buffer)
{
        *buffer = static_cast<void *>(data.get());
        return S_OK;
}

HRESULT DeckLinkFrame::GetTimecode (/* in */ BMDTimecodeFormat, /* out */ IDeckLinkTimecode **timecode)
{
        *timecode = dynamic_cast<IDeckLinkTimecode *>(this->timecode);
        return S_OK;
}

HRESULT DeckLinkFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **)
{
	return S_FALSE;
}

/* IDeckLinkMutableVideoFrame */
HRESULT DeckLinkFrame::SetFlags (/* in */ BMDFrameFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecode (/* in */ BMDTimecodeFormat, /* in */ IDeckLinkTimecode *timecode)
{
        if(this->timecode)
                this->timecode->Release();
        this->timecode = timecode;
        return S_OK;
}

HRESULT DeckLinkFrame::SetTimecodeFromComponents (/* in */ BMDTimecodeFormat, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ BMDTimecodeFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetAncillaryData (/* in */ IDeckLinkVideoFrameAncillary *)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecodeUserBits (/* in */ BMDTimecodeFormat, /* in */ BMDTimecodeUserBits)
{
        return E_FAIL;
}

void HDRMetadata::Init(const string &fmt) {
        auto opts = unique_ptr<char []>(new char [fmt.size() + 1]);
        strcpy(opts.get(), fmt.c_str());
        char *save_ptr = nullptr;
        char *mode_c = strtok_r(opts.get(), ",", &save_ptr);
        assert(mode_c != nullptr);
        string mode = mode_c;
        std::for_each(std::begin(mode), std::end(mode), [](char& c) {
                        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        });
        if (mode == "SDR"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::SDR);
        } else if (mode == "HDR"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::HDR);
        } else if (mode == "PQ"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::PQ);
        } else if (mode == "HLG"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::HLG);
        } else if (mode == "HELP"s) {
                cout << MOD_NAME << "HDR syntax:\n";
                cout << "\tHDR[=<eotf>|int[,{<k>=<v>}*]\n";
                cout << "\t\t<eotf> may be one of SDR, HDR, PQ, HLG or int 0-7\n";
                cout << "\t\tFurther options may be specification of HDR values, accepted keys are (values are floats):\n";
                cout << "\t\t\t- maxDisplayMasteringLuminance\n";
                cout << "\t\t\t- minDisplayMasteringLuminance\n";
                cout << "\t\t\t- maxCLL\n";
                cout << "\t\t\t- maxFALL\n";
                throw ug_no_error{};
        } else {
                EOTF = stoi(mode);
                if (EOTF < 0 || EOTF > 7) {
                        throw out_of_range("Value outside [0..7]");
                }
        }

        char *other_opt = nullptr;
        while ((other_opt = strtok_r(nullptr, ",", &save_ptr)) != nullptr) {
                if (strstr(other_opt, "maxDisplayMasteringLuminance=") != nullptr) {
                        maxDisplayMasteringLuminance = stod(other_opt + strlen("maxDisplayMasteringLuminance="));
                } else if (strstr(other_opt, "minDisplayMasteringLuminance=") != nullptr) {
                        minDisplayMasteringLuminance = stod(other_opt + strlen("minDisplayMasteringLuminance="));
                } else if (strstr(other_opt, "maxCLL=") != nullptr) {
                        maxCLL = stod(other_opt + strlen("maxCLL="));
                } else if (strstr(other_opt, "maxFALL=") != nullptr) {
                        maxFALL = stod(other_opt + strlen("maxFALL="));
                } else {
                        throw invalid_argument("Unrecognized HDR attribute "s + other_opt);
                }
        }
}

static inline void debug_print_metadata_id(const char *fn_name, BMDDeckLinkFrameMetadataID metadataID) {
        if (log_level < LOG_LEVEL_DEBUG2) {
                return;
        }
        array<char, sizeof metadataID + 1> fourcc{};
        copy(reinterpret_cast<char *>(&metadataID), reinterpret_cast<char *>(&metadataID) + sizeof metadataID, fourcc.data());
        LOG(LOG_LEVEL_DEBUG2) << MOD_NAME << "DecklLinkFrame " << fn_name << ": " << fourcc.data() << "\n";
}

// IDeckLinkVideoFrameMetadataExtensions interface
HRESULT DeckLinkFrame::GetInt(BMDDeckLinkFrameMetadataID metadataID, int64_t* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        switch (metadataID)
        {
                case bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc:
                        *value = m_metadata.EOTF;
                        return S_OK;
                case bmdDeckLinkFrameMetadataColorspace:
                        // Colorspace is fixed for this sample
                        *value = bmdColorspaceRec2020;
                        return S_OK;
                default:
                        value = nullptr;
                        return E_INVALIDARG;
        }
}

HRESULT DeckLinkFrame::GetFloat(BMDDeckLinkFrameMetadataID metadataID, double* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        switch (metadataID)
        {
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesRedX:
                        *value = m_metadata.referencePrimaries.RedX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesRedY:
                        *value = m_metadata.referencePrimaries.RedY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesGreenX:
                        *value = m_metadata.referencePrimaries.GreenX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesGreenY:
                        *value = m_metadata.referencePrimaries.GreenY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesBlueX:
                        *value = m_metadata.referencePrimaries.BlueX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesBlueY:
                        *value = m_metadata.referencePrimaries.BlueY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRWhitePointX:
                        *value = m_metadata.referencePrimaries.WhiteX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRWhitePointY:
                        *value = m_metadata.referencePrimaries.WhiteY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaxDisplayMasteringLuminance:
                        *value = m_metadata.maxDisplayMasteringLuminance;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMinDisplayMasteringLuminance:
                        *value = m_metadata.minDisplayMasteringLuminance;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaximumContentLightLevel:
                        *value = m_metadata.maxCLL;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaximumFrameAverageLightLevel:
                        *value = m_metadata.maxFALL;
                        return S_OK;
                default:
                        value = nullptr;
                        return E_INVALIDARG;
        }
}

HRESULT DeckLinkFrame::GetFlag(BMDDeckLinkFrameMetadataID metadataID, BMD_BOOL* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        // Not expecting GetFlag
        *value = BMD_TRUE;
        return E_INVALIDARG;
}

HRESULT DeckLinkFrame::GetString(BMDDeckLinkFrameMetadataID metadataID, BMD_STR* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        // Not expecting GetString
        *value = nullptr;
        return E_INVALIDARG;
}

HRESULT DeckLinkFrame::GetBytes(BMDDeckLinkFrameMetadataID metadataID, void* /* buffer */, uint32_t* bufferSize)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        *bufferSize = 0;
        return E_INVALIDARG;
}

// 3D frame
DeckLink3DFrame::DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata)
        : DeckLinkFrame(w, h, rb, pf, buffer_pool, hdr_metadata), rightEye(DeckLinkFrame::Create(w, h, rb, pf, buffer_pool, hdr_metadata))
{
}

DeckLink3DFrame *DeckLink3DFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata)
{
        DeckLink3DFrame *frame = new DeckLink3DFrame(width, height, rawBytes, pixelFormat, buffer_pool, hdr_metadata);
        return frame;
}

DeckLink3DFrame::~DeckLink3DFrame()
{
}

ULONG DeckLink3DFrame::AddRef()
{
        return DeckLinkFrame::AddRef();
}

ULONG DeckLink3DFrame::Release()
{
        return DeckLinkFrame::Release();
}

HRESULT DeckLink3DFrame::QueryInterface(REFIID id, void **data)
{
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "DecklLink3DFrame QueryInterface " << id << "\n";
        if(id == IID_IDeckLinkVideoFrame3DExtensions)
        {
                this->AddRef();
                *data = dynamic_cast<IDeckLinkVideoFrame3DExtensions *>(this);
                return S_OK;
        }
        return DeckLinkFrame::QueryInterface(id, data);
}

BMDVideo3DPackingFormat DeckLink3DFrame::Get3DPackingFormat()
{
        return bmdVideo3DPackingLeftOnly;
}

HRESULT DeckLink3DFrame::GetFrameForRightEye(IDeckLinkVideoFrame ** frame) 
{
        *frame = rightEye.get();
        rightEye->AddRef();
        return S_OK;
}

/* function from DeckLink SDK sample DeviceList */

static void print_output_modes (IDeckLink* deckLink)
{
        IDeckLinkOutput*                        deckLinkOutput = NULL;
        IDeckLinkDisplayModeIterator*           displayModeIterator = NULL;
        IDeckLinkDisplayMode*                   displayMode = NULL;
        HRESULT                                 result;
        int                                     displayModeNumber = 0;

        // Query the DeckLink for its configuration interface
        result = deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&deckLinkOutput);
        if (result != S_OK)
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not obtain the IDeckLinkOutput interface - result = %08x\n", (int) result);
                if (result == E_NOINTERFACE) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Device doesn't support video playback.\n");
                }
                goto bail;
        }

        // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on output
        result = deckLinkOutput->GetDisplayModeIterator(&displayModeIterator);
        if (result != S_OK)
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not obtain the video output display mode iterator - result = %08x\n", (int) result);
                goto bail;
        }

        // List all supported output display modes
        printf("\tdisplay modes:\n");
        while (displayModeIterator->Next(&displayMode) == S_OK)
        {
                BMD_STR                  displayModeString = NULL;

                result = displayMode->GetName(&displayModeString);

                if (result == S_OK)
                {
                        char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                        BMDTimeValue    frameRateDuration;
                        BMDTimeScale    frameRateScale;

                        // Obtain the display mode's properties
                        string flags_str = bmd_get_flags_str(displayMode->GetFlags());
                        int modeWidth = displayMode->GetWidth();
                        int modeHeight = displayMode->GetHeight();
                        uint32_t field_dominance_n = ntohl(displayMode->GetFieldDominance());
                        displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
                        printf("\t\t%2d) %-20s  %d x %d \t %2.2f FPS %.4s, flags: %s\n",displayModeNumber, displayModeCString,
                                        modeWidth, modeHeight, (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (char *) &field_dominance_n, flags_str.c_str());
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

        if (deckLinkOutput != NULL)
                deckLinkOutput->Release();
}

static const struct video_display_info display_decklink_info = {
        display_decklink_probe,
        display_decklink_init,
        NULL, // _run
        display_decklink_done,
        display_decklink_getf,
        display_decklink_putf,
        display_decklink_reconfigure_video,
        display_decklink_get_property,
        display_decklink_put_audio_frame,
        display_decklink_reconfigure_audio,
        MOD_NAME,
};

REGISTER_MODULE(decklink, &display_decklink_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

/* vim: set expandtab sw=8: */

/**
 * @file   video_capture/aja.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Based on AJA samples ntv2framegrabber, ntv2capture and ntv2llburn (Ping-Pong)
 */
/*
 * Copyright (c) 2015-2020 CESNET, z. s. p. o.
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
/**
 * @file
 * @todo
 * Capture what is on the wire (RGB if RGB, YCbCr otherwise)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_msvc.h"
#include "config_unix.h"
#include "config_win32.h"

#ifdef _MSC_VER
#include <winsock2.h>
#endif

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rang.hpp"
#include "utils/video_frame_pool.h"
#include "video.h"
#include "video_capture.h"

#include "ajatypes.h"
#include "ntv2devicescanner.h"
#include "ntv2democommon.h"
#include "ntv2capture.h"
#if AJA_NTV2_SDK_VERSION_MAJOR >= 13
#include "ajabase/common/types.h"
#include "ajabase/common/videotypes.h"
#include "ajabase/common/circularbuffer.h"
#include "ajabase/system/process.h"
#include "ajabase/system/systemtime.h"
#include "ajabase/system/thread.h"
#else
#include "ajastuff/common/videotypes.h"
#include "ajastuff/common/circularbuffer.h"
#include "ajastuff/system/process.h"
#include "ajastuff/system/systemtime.h"
#include "ajastuff/system/thread.h"
#endif

#include "ntv2utils.h"
#include "ntv2devicefeatures.h"

#define NTV2_AUDIOSIZE_MAX      (401 * 1024)

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "aja_common.h"

#define MOD_NAME "[AJA cap.] "

namespace aja = ultragrid::aja;

#ifdef _MSC_VER
extern "C" __declspec(dllexport) unsigned int *aja_audio_capture_channels = NULL;
extern "C" __declspec(dllexport) volatile bool *aja_should_exit = NULL;
volatile int log_level = 5;
#else
unsigned int *aja_audio_capture_channels = &audio_capture_channels;
volatile bool *aja_should_exit = &should_exit;
#endif

#ifdef _MSC_VER
#define LINK_SPEC extern "C" __declspec(dllexport)
#else
#define LINK_SPEC static
#endif

#define CHECK_OK(cmd, msg, action_failed) do { bool ret = cmd; if (!ret) {\
        LOG(LOG_LEVEL_WARNING) << MOD_NAME << (msg) << "\n";\
        action_failed;\
}\
} while(0)
#define NOOP ((void)0)

#define CHECK_RET_FAIL(cmd) CHECK_OK(cmd, #cmd " failed", return AJA_STATUS_FAIL)
#define CHECK(cmd) CHECK_OK(cmd, #cmd " failed", NOOP)

using namespace std;

struct aligned_data_allocator {
        void *allocate(size_t size) {
                return aligned_malloc(size, AJA_PAGE_SIZE);
        }
        void deallocate(void *ptr) {
                aligned_free(ptr);
        }
};

static const ULWord app = AJA_FOURCC ('U','L','G','R');

class vidcap_state_aja {
        public:
                vidcap_state_aja(unordered_map<string, string> const & parameters, int audioFlags);
                void Init();
                virtual ~vidcap_state_aja();
                struct video_frame *grab(struct audio_frame **audio);
                AJAStatus Run();
                void Quit();
        private:
                unsigned int           mDeviceIndex{0};
                CNTV2Card              mDevice;
                NTV2DeviceID           mDeviceID{};                   ///     My device identifier
                NTV2EveryFrameTaskMode mSavedTaskMode{};              /// Used to restore prior every-frame task mode
                NTV2Channel            mInputChannel{NTV2_CHANNEL_INVALID}; ///     My input channel
                NTV2VideoFormat        mVideoFormat{NTV2_FORMAT_UNKNOWN}; ///     My video format
                NTV2FrameBufferFormat  mPixelFormat{NTV2_FBF_INVALID}; ///     My pixel format
                NTV2VANCMode           mVancMode{};                   ///     VANC enabled?
                bool                   mWideVanc{};                   ///     Wide VANC?
                NTV2InputSource        mInputSource{NTV2_INPUTSOURCE_SDI1};                  ///     The input source I'm using
                NTV2AudioSystem        mAudioSystem{NTV2_AUDIOSYSTEM_1};                  ///     The audio system I'm using
                uint32_t               mVideoBufferSize{};            ///     My video buffer size, in bytes
                uint32_t               mAudioBufferSize{};            ///     My audio buffer size, in bytes
                thread                 mProducerThread;               ///     My producer thread object -- does the frame capturing
                video_frame_pool<aligned_data_allocator> mPool;
                shared_ptr<video_frame> mOutputFrame;
                shared_ptr<uint32_t>   mOutputAudioFrame;
                size_t                 mOutputAudioFrameSize{};
                mutex                  mOutputFrameLock;
                condition_variable     mOutputFrameReady;
                bool                   mProgressive{false};
                chrono::system_clock::time_point mT0{chrono::system_clock::now()};
                int                    mFrames{0};
                struct audio_frame     mAudio{};
                int                    mMaxAudioChannels{0};
                NTV2AudioSource        mAudioSource{};
                NTV2TCSource           mTimeCodeSource{};              ///< @brief     Time code source
                bool                   mbFixedReference{false};
                bool                   mCheckFor4K{false};
                uint32_t               mAudioInLastAddress{};          ///< @brief My record of the location of the last audio sample captured
                bool                   mClearRouting{false};
                bool                   mInputIsRGB;                    ///< @brief SDI Input is in RGB otherwise YCbCr

                AJAStatus SetupHDMI();
                AJAStatus SetupVideo();
                AJAStatus SetupAudio();
                void SetupHostBuffers();
                NTV2VideoFormat GetVideoFormatFromInputSource();
                void EnableInput(NTV2InputSource source);
                bool IsInput3Gb(const NTV2InputSource inputSource);

                /**
                  @brief  Starts my frame consumer thread.
                 **/
                virtual void                    StartProducerThread (void);
                virtual void                    CaptureFrames (void);
                static void     ProducerThreadStatic (vidcap_state_aja * pContext);
};

vidcap_state_aja::vidcap_state_aja(unordered_map<string, string> const & parameters, int audioFlags)
{
        bool colorSpaceSet = false; // explicitly set input color spec

        for (auto it : parameters) {
                if (it.first == "progressive") {
                        mProgressive = true;
                } else if (it.first == "4K" || it.first == "4k") {
                        mCheckFor4K = true;
                } else if (it.first == "clear-routing") {
                        mClearRouting = true;
                } else if (it.first == "device") {
                        mDeviceIndex = stol(it.second, nullptr, 10);
                } else if (it.first == "channel") {
                        mInputChannel = (NTV2Channel) (atoi(it.second.c_str()) - 1);
                } else if (it.first == "codec") {
                        if (get_codec_from_name(it.second.c_str()) == VIDEO_CODEC_NONE) {
                                throw string("Unknown codec " + it.second + "!");
                        }
                        mPixelFormat = NTV2_FBF_INVALID;
                        for (auto c : aja::codec_map) {
                                if (c.second == get_codec_from_name(it.second.c_str())) {
                                        mPixelFormat = c.first;
                                        break;
                                }
                        }
                        if (mPixelFormat == NTV2_FBF_INVALID) {
                                throw string("Cannoc map " + it.second + " to AJA codec!");
                        }
                } else if (it.first == "connection") {
                        NTV2InputSource source = NTV2InputSource();
                        while (source != NTV2_INPUTSOURCE_INVALID) {
                                if (NTV2InputSourceToString(source, true) == it.second) {
                                        mInputSource = source;
                                        break;
                                }
                                // should be this, but GetNTV2InputSourceForIndex knows only SDIs
                                //source = ::GetNTV2InputSourceForIndex(::GetIndexForNTV2InputSource(source) + 1);
                                source = (NTV2InputSource) ((int) source + 1);
                        }
                        if (source == NTV2_INPUTSOURCE_INVALID) {
                                throw string("Unknown source " + it.second + "!");
                        }
                } else if (it.first == "format") {
                        NTV2VideoFormat format = NTV2VideoFormat();
                        while (format != NTV2_MAX_NUM_VIDEO_FORMATS) {
                                if (NTV2VideoFormatToString(format) == it.second) {
                                        mVideoFormat = format;
                                        break;
                                }
                                format = (NTV2VideoFormat) ((int) format + 1);
                        }
                        if (format == NTV2_MAX_NUM_VIDEO_FORMATS) {
                                throw string("Unknown format " + it.second + "!");
                        }
                } else if (it.first == "RGB") {
                        if (colorSpaceSet && mInputIsRGB != true) {
                                throw string("Both RGB and YUV input specified!") + it.first;
                        }
                        colorSpaceSet = true;
                        mInputIsRGB = true;
                } else if (it.first == "YUV") {
                        if (colorSpaceSet && mInputIsRGB != false) {
                                throw string("Both RGB and YUV input specified!") + it.first;
                        }
                        colorSpaceSet = true;
                        mInputIsRGB = false;
                } else {
                        throw string("Unknown option: ") + it.first;
                }
        }

        if (!NTV2_INPUT_SOURCE_IS_SDI(mInputSource) && mInputChannel == NTV2_CHANNEL_INVALID) {
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Non-SDI source detected - we will use "
                        "probably channel 1. Consider passing \"channel\" option (see help).\n";
        }

        if (mPixelFormat == NTV2_FBF_INVALID) {
                if (colorSpaceSet && mInputIsRGB) {
                        mPixelFormat = NTV2_FBF_ABGR;
                } else {
                        mPixelFormat = NTV2_FBF_8BIT_YCBCR;
                }
        }

        if (!colorSpaceSet) {
                mInputIsRGB = IsRGBFormat(mPixelFormat);
        }

        if (audioFlags & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                // this maps according to mInputSource - EMBEDDED (SDI), HDMI or ANALOG
                mAudioSource = NTV2InputSourceToAudioSource(mInputSource);
        } else if (audioFlags & VIDCAP_FLAG_AUDIO_AESEBU) {
                mAudioSource = NTV2_AUDIO_AES;
        } else if (audioFlags & VIDCAP_FLAG_AUDIO_ANALOG) {
                mAudioSource = NTV2_AUDIO_ANALOG;
        } else {
                mAudioSource = NTV2_AUDIO_SOURCE_INVALID;
        }

        Init();
}

#ifndef _MSC_VER
ADD_TO_PARAM(aja_fourcc, "aja-fourcc", "* aja-fourcc\n"
                "  Specifies application FourCC for AJA.\n");
#endif
void vidcap_state_aja::Init()
{
        AJAStatus       status  (AJA_STATUS_SUCCESS);

        //      Open the device...
        CNTV2DeviceScanner      deviceScanner;

        //      Any AJA devices out there?
        if (deviceScanner.GetNumDevices () == 0)
                throw string("No devices found!");

        if (mDeviceIndex >= deviceScanner.GetNumDevices ()) {
                throw string("Device index exceeds number of available devices!");
        }

        //      Using a reference to the discovered device list,
        //      and the index number of the device of interest (mDeviceIndex),
        //      get information about that particular device...
        NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [mDeviceIndex]);
        if (!mDevice.Open (mDeviceIndex))
                throw string("Unable to open device.");

        if (mClearRouting) {
                CHECK_OK(mDevice.ClearRouting(), "ClearRouting", NOOP);
        }

        ULWord fourcc = app;
#ifndef _MSC_VER
        if (get_commandline_param("aja-fourcc")) {
                const char *fcc_req  = get_commandline_param("aja-fourcc");
                char fcc_s[4] = "";
                // trim or pad with spaces
                memcpy(fcc_s, fcc_req, min<int>(strlen(fcc_req), 4));
                fourcc = AJA_FOURCC(fcc_s[0], fcc_s[1], fcc_s[2], fcc_s[3]);
        }

        if (!mDevice.AcquireStreamForApplication (fourcc, static_cast <uint32_t> (getpid())))
                throw string("Cannot aquire stream.");
#endif

        CHECK(mDevice.GetEveryFrameServices (mSavedTaskMode));        //      Save the current state before we change it
        CHECK(mDevice.SetEveryFrameServices (NTV2_OEM_TASKS));

        //      Keep the device ID handy as it will be used frequently...
        mDeviceID = mDevice.GetDeviceID ();

        //      Set up the video and audio...
        status = SetupVideo ();
        if (AJA_FAILURE (status))
                throw string("Cannot setup video. Err = ") + to_string(status);

        status = SetupAudio ();
        if (AJA_FAILURE (status))
                throw string("Cannot setup audio. Err = ") + to_string(status);

        //      Set up the circular buffers, the device signal routing, and both playout and capture Auto Circulate...
        SetupHostBuffers ();

        //      This is for the timecode that we will burn onto the image...
        //NTV2FormatDescriptor fd = GetFormatDescriptor (GetNTV2StandardFromVideoFormat (mVideoFormat), mPixelFormat, mVancEnabled, Is2KFormat (mVideoFormat), mWideVanc);
}

vidcap_state_aja::~vidcap_state_aja() {
        //      Unsubscribe from input vertical event...
        CHECK(mDevice.UnsubscribeInputVerticalEvent (mInputChannel));

        CHECK(mDevice.SetEveryFrameServices (mSavedTaskMode));                               //      Restore previous service level
#ifndef _MSC_VER
        CHECK(mDevice.ReleaseStreamForApplication (app, static_cast <uint32_t> (getpid()))); //      Release the device
#endif

        mOutputFrame = NULL;
        free(mAudio.data);
}

NTV2VideoFormat vidcap_state_aja::GetVideoFormatFromInputSource()
{
        NTV2VideoFormat videoFormat     (NTV2_FORMAT_UNKNOWN);

        switch (mInputSource)
        {
                case NTV2_INPUTSOURCE_SDI1:
                case NTV2_INPUTSOURCE_SDI5:
                {
                        NTV2InputSource source;
                        const ULWord    ndx     (::GetIndexForNTV2InputSource (mInputSource));
                        if (::NTV2DeviceCanDoMultiFormat (mDeviceID))
                                CHECK(mDevice.SetMultiFormatMode (true));
                        source = ::GetNTV2InputSourceForIndex (ndx + 0);
                        EnableInput(source);
                        videoFormat = mDevice.GetInputVideoFormat(source, mProgressive);
                        NTV2Standard    videoStandard   (::GetNTV2StandardFromVideoFormat (videoFormat));
                        if (mCheckFor4K && (videoStandard == NTV2_STANDARD_1080p))
                        {
                                if (::NTV2DeviceCanDoMultiFormat (mDeviceID))
                                        CHECK(mDevice.SetMultiFormatMode (false));
                                int i;
                                bool allSDISameInput = true;
                                NTV2VideoFormat videoFormatNext;
                                for (i = 1; i < 4; i++) {
                                        source = ::GetNTV2InputSourceForIndex (ndx + i);
                                        EnableInput(source);
                                        videoFormatNext = mDevice.GetInputVideoFormat (source, mProgressive);
                                        if (videoFormatNext != videoFormat) {
                                                allSDISameInput = false;
                                                break;
                                        }
                                }
                                if (allSDISameInput)
                                        videoFormat = GetQuadSizedVideoFormat(videoFormat);
                                else
                                        LOG(LOG_LEVEL_WARNING) << "Input " << NTV2InputSourceToString(::GetNTV2InputSourceForIndex(ndx + i), true) << " has input format " << NTV2VideoFormatToString(videoFormatNext) << " which differs from " << NTV2VideoFormatToString(videoFormat) << " on " << NTV2InputSourceToString(::GetNTV2InputSourceForIndex(ndx), true) << "!\n";
                        }
                        if (mCheckFor4K && (videoStandard != NTV2_STANDARD_1080p)) {
                                log_msg(LOG_LEVEL_WARNING, "Video format on first SDI is not 1080p!\n");
                        }
                        break;
                }

                case NTV2_NUM_INPUTSOURCES:
                        break;                  //      indicates no source is currently selected

                default:
                        videoFormat = mDevice.GetInputVideoFormat (mInputSource, mProgressive);
                        break;
        }

        return videoFormat;
}       //      GetVideoFormatFromInputSource

void vidcap_state_aja::EnableInput(NTV2InputSource source)
{
        //      Bi-directional SDI connectors need to be set to capture...
        if (NTV2DeviceHasBiDirectionalSDI (mDeviceID) && NTV2_INPUT_SOURCE_IS_SDI (source))
        {
                NTV2Channel channel = ::NTV2InputSourceToChannel(source);
                CHECK(mDevice.SetSDITransmitEnable (channel, false));       //      Disable transmit mode...
                for (unsigned ndx (0);  ndx < 10;  ndx++)
                        CHECK(mDevice.WaitForInputVerticalInterrupt (channel)); //      ...and give the device some time to lock to a signal
        }

}

/*
 * see NTV2FrameGrabber::SetupInput()
 */
AJAStatus vidcap_state_aja::SetupHDMI()
{
        NTV2LHIHDMIColorSpace	hdmiColor	(NTV2_LHIHDMIColorSpaceRGB);
        CHECK_RET_FAIL(mDevice.GetHDMIInputColor (hdmiColor, mInputChannel));
        if (!mbFixedReference)
                CHECK_RET_FAIL(mDevice.SetReference (::NTV2InputSourceToReferenceSource(mInputSource)));

        // configure hdmi with 2.0 support
        if (NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat) && !mDevice.DeviceCanDoHDMIQuadRasterConversion ()) {
                //	Set two sample interleave
                CHECK_RET_FAIL(mDevice.SetTsiFrameEnable(true, mInputChannel));

                for (unsigned offset = 0; offset < 2; ++offset) {
                        NTV2Channel channel = (NTV2Channel) ((int) mInputChannel + offset);
                        CHECK_RET_FAIL(mDevice.EnableChannel (channel));
                        CHECK_RET_FAIL(mDevice.SetMode (channel, NTV2_MODE_CAPTURE));
                        CHECK_RET_FAIL(mDevice.SetFrameBufferFormat (channel, mPixelFormat));
                }

                for (unsigned offset = 0; offset < 4; ++offset) {
                        if ((hdmiColor == NTV2_LHIHDMIColorSpaceYCbCr && IsRGBFormat(mPixelFormat))
                                        || (hdmiColor == NTV2_LHIHDMIColorSpaceRGB && !IsRGBFormat(mPixelFormat))) {
                                NTV2Channel channel = (NTV2Channel) ((int) mInputChannel + offset);
                                CHECK_RET_FAIL(mDevice.Connect (::GetCSCInputXptFromChannel (channel),
                                                        ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/,
                                                                hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, offset/*hdmiQuadrant*/)));
                                CHECK_RET_FAIL(mDevice.Connect (GetTSIMuxInputXptFromChannel((NTV2Channel) offset, false/* inLinkB */), GetCSCOutputXptFromChannel (channel, false/* inIsKey */, hdmiColor != NTV2_LHIHDMIColorSpaceRGB /* inIsRGB */)));
                        } else {
                                CHECK_RET_FAIL(mDevice.Connect (GetTSIMuxInputXptFromChannel((NTV2Channel) offset, false/* inLinkB */), ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/, hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, offset /*hdmiQuadrant*/)));
                        }
                }

                for (unsigned offset = 0; offset < 4; ++offset) {
                        CHECK_RET_FAIL(mDevice.Connect(GetFrameBufferInputXptFromChannel((NTV2Channel) (offset / 2) /* inChannel */, offset % 2 == 1 /* inIsBInput */),
                                        GetTSIMuxOutputXptFromChannel((NTV2Channel) (offset / 2), offset % 2 == 1 /* inLinkB */)));
                }
        } else if (NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat) && mDevice.DeviceCanDoHDMIQuadRasterConversion ()) {
                CHECK_RET_FAIL(mDevice.SetTsiFrameEnable(false, mInputChannel));
                for (unsigned offset = 0; offset < 4; ++offset) {
                        NTV2Channel channel = (NTV2Channel) ((int) mInputChannel + offset);
                        CHECK_RET_FAIL(mDevice.EnableChannel (channel));
                        CHECK_RET_FAIL(mDevice.SetMode (channel, NTV2_MODE_CAPTURE));
                        CHECK_RET_FAIL(mDevice.SetFrameBufferFormat (channel, mPixelFormat));
                        if ((hdmiColor == NTV2_LHIHDMIColorSpaceYCbCr && IsRGBFormat(mPixelFormat))
                                        || (hdmiColor == NTV2_LHIHDMIColorSpaceRGB && !IsRGBFormat(mPixelFormat))) {
                                CHECK_RET_FAIL(mDevice.Connect (::GetCSCInputXptFromChannel (channel),
                                                ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/, hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, channel /*hdmiQuadrant*/)));
                                CHECK_RET_FAIL(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel),
                                                ::GetCSCOutputXptFromChannel (channel, false/*isKey*/, hdmiColor != NTV2_LHIHDMIColorSpaceRGB /*isRGB*/)));
                        } else {
                                CHECK_RET_FAIL(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel),
                                                ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/, hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, channel /*hdmiQuadrant*/)));
                        }
                }	//	loop once for each channel (4 times for 4K/UHD)
        } else {
                CHECK_RET_FAIL(mDevice.EnableChannel (mInputChannel));
                CHECK_RET_FAIL(mDevice.SetMode (mInputChannel, NTV2_MODE_CAPTURE));
                CHECK_RET_FAIL(mDevice.SetFrameBufferFormat (mInputChannel, mPixelFormat));
                if ((hdmiColor == NTV2_LHIHDMIColorSpaceYCbCr && IsRGBFormat(mPixelFormat))
                                || (hdmiColor == NTV2_LHIHDMIColorSpaceRGB && !IsRGBFormat(mPixelFormat))) {
                        CHECK_RET_FAIL(mDevice.Connect (::GetCSCInputXptFromChannel (mInputChannel),
                                        ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/, hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, 0/*hdmiQuadrant*/)));
                        CHECK_RET_FAIL(mDevice.Connect (::GetFrameBufferInputXptFromChannel (mInputChannel),
                                        ::GetCSCOutputXptFromChannel (mInputChannel, false/*isKey*/, hdmiColor != NTV2_LHIHDMIColorSpaceRGB /*isRGB*/)));
                } else {
                        CHECK_RET_FAIL(mDevice.Connect (::GetFrameBufferInputXptFromChannel (mInputChannel),
                                        ::GetInputSourceOutputXpt (mInputSource, false/*isSDI_DS2*/, hdmiColor == NTV2_LHIHDMIColorSpaceRGB /*isHDMI_RGB*/, 0/*hdmiQuadrant*/)));
                }
        }

        // configure the qrc if present
        if (NTV2DeviceGetHDMIVersion(mDeviceID) == 2) {
                if (NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat)) {
                        CHECK_RET_FAIL(mDevice.SetHDMIV2Mode (NTV2_HDMI_V2_4K_CAPTURE));
                } else {
                        CHECK_RET_FAIL(mDevice.SetHDMIV2Mode (NTV2_HDMI_V2_HDSD_BIDIRECTIONAL));
                }
        }

        return AJA_STATUS_SUCCESS;
}

AJAStatus vidcap_state_aja::SetupVideo()
{
        //      Set the video format to match the incomming video format.
        //      Does the device support the desired input source?
        if (!::NTV2DeviceCanDoInputSource (mDeviceID, mInputSource))
                return AJA_STATUS_BAD_PARAM;    //      Nope

        if (mInputChannel == NTV2_CHANNEL_INVALID) {
                mInputChannel = ::NTV2InputSourceToChannel (mInputSource);
        }

        if (UWord (mInputChannel) >= ::NTV2DeviceGetNumFrameStores (mDeviceID)) {
                ostringstream oss;
                oss    << "## ERROR:  Cannot use channel '" << mInputChannel+1 << "' -- device only supports channel 1"
                        << (::NTV2DeviceGetNumFrameStores (mDeviceID) > 1  ?  string (" thru ") + string (1, uint8_t (::NTV2DeviceGetNumFrameStores (mDeviceID)+'0'))  :  "");
                throw runtime_error(oss.str());
        }

        //      Sometimes other applications disable some or all of the frame buffers, so turn on ours now..
        CHECK_OK(mDevice.EnableChannel (mInputChannel), "Cannot enable channel", NOOP);

        //      Enable and subscribe to the interrupts for the channel to be used...
        CHECK_OK(mDevice.EnableInputInterrupt (mInputChannel), "EnableInputInterrupt failed", NOOP);
        CHECK_OK(mDevice.SubscribeInputVerticalEvent (mInputChannel), "SubscribeInputVerticalEvent failed", NOOP);

        //      Set the video format to match the incomming video format.
        //      Does the device support the desired input source?

        //      If the device supports bi-directional SDI and the
        //      requested input is SDI, ensure the SDI direction
        //      is configured for input...
        if (::NTV2DeviceHasBiDirectionalSDI (mDeviceID) && NTV2_INPUT_SOURCE_IS_SDI (mInputSource))
        {
                CHECK_OK(mDevice.SetSDITransmitEnable (mInputChannel, false), "Cannot disable SDI transmit", NOOP);

                //      Give the input circuit some time (~10 frames) to lock onto the input signal...
#if AJA_NTV2_SDK_VERSION_BEFORE(12,5)
                for (int i = 0; i < 10; i++) {
                        CHECK_OK(mDevice.WaitForInputVerticalInterrupt (mInputChannel), "Cannot wait for VBI", NOOP);
                }
#else
                CHECK_OK(mDevice.WaitForInputVerticalInterrupt (mInputChannel, 10), "Cannot wait for VBI", NOOP);
#endif
        }

        if (NTV2_INPUT_SOURCE_IS_SDI (mInputSource))
#if AJA_NTV2_SDK_VERSION_BEFORE(12,4)
                mTimeCodeSource = ::NTV2ChannelToTimecodeSource (mInputChannel);
#else
                mTimeCodeSource = ::NTV2InputSourceToTimecodeIndex(mInputSource);
#endif
        else if (NTV2_INPUT_SOURCE_IS_ANALOG (mInputSource))
                mTimeCodeSource = NTV2_TCSOURCE_LTC1;
        else
                mTimeCodeSource = NTV2_TCSOURCE_DEFAULT;

        //      Determine the input video signal format...
        if (mVideoFormat == NTV2_FORMAT_UNKNOWN) {
                mVideoFormat =  GetVideoFormatFromInputSource();
                if (mVideoFormat == NTV2_FORMAT_UNKNOWN)
                {
                        LOG(LOG_LEVEL_ERROR) << "## ERROR:  No input signal or unknown format" << endl;
                        return AJA_STATUS_NOINPUT;      //      Sorry, can't handle this format
                }
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Detected input signal: " << NTV2VideoFormatToString(mVideoFormat) << "\n";
        }

        mAudioSystem = ::NTV2ChannelToAudioSystem(mInputChannel);

        //      Set the device video format to whatever we detected at the input...
        CHECK(mDevice.SetVideoFormat (mVideoFormat, false, false, mInputChannel));

        //      Set the frame buffer pixel format for all the channels on the device
        //      (assuming it supports that pixel format -- otherwise default to 8-bit YCbCr)...
        if (!::NTV2DeviceCanDoFrameBufferFormat (mDeviceID, mPixelFormat))
                mPixelFormat = NTV2_FBF_8BIT_YCBCR;

        //      Set the pixel format for both device frame buffers...
        CHECK(mDevice.SetFrameBufferFormat (mInputChannel, mPixelFormat));

        //      Enable and subscribe to the interrupts for the channel to be used...
        CHECK(mDevice.EnableInputInterrupt (mInputChannel));
        CHECK(mDevice.SubscribeInputVerticalEvent (mInputChannel));

        //      Tell the hardware which buffers to use until the main worker thread runs
        CHECK(mDevice.SetInputFrame   (mInputChannel,  0));

        //      Set the Frame Store modes
        CHECK(mDevice.SetMode (mInputChannel,  NTV2_MODE_CAPTURE));

        CHECK(mDevice.SetReference (NTV2_REFERENCE_FREERUN));

        if (NTV2_INPUT_SOURCE_IS_SDI (mInputSource)) {
                for (unsigned offset (0);  offset < 4;  offset++) {
                        NTV2Channel SDIChannel = (NTV2Channel) ((int) NTV2InputSourceToChannel (mInputSource) + offset);
                        NTV2Channel channel = (NTV2Channel) ((int) mInputChannel + offset);

			if (mInputIsRGB) {
				CHECK(mDevice.Connect (::GetDLInInputXptFromChannel(channel, false), ::GetSDIInputOutputXptFromChannel (SDIChannel, false))); // SDIIn ==> DLIn
				CHECK(mDevice.Connect (::GetDLInInputXptFromChannel(channel, true), ::GetSDIInputOutputXptFromChannel (SDIChannel, true)));  // SDIIn ==> DLIn
			}

                        if (IsRGBFormat(mPixelFormat) && !mInputIsRGB) { // convert YUV->RGB
                                CHECK(mDevice.Connect (::GetCSCInputXptFromChannel (channel), ::GetSDIInputOutputXptFromChannel (SDIChannel)));
                                CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel), ::GetCSCOutputXptFromChannel (SDIChannel, false/*isKey*/, true/*isRGB*/)));
			} else if (IsRGBFormat(mPixelFormat) && mInputIsRGB) { // RGB->RGB
                                CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel), ::GetDLInOutputXptFromChannel(SDIChannel))); // DLOut ==> FBRGB
                        } else if (!IsRGBFormat(mPixelFormat) && !mInputIsRGB) { // YUV->YUV
                                CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel), ::GetSDIInputOutputXptFromChannel (SDIChannel)));
                        } else { // RGB->YUV
                                CHECK(mDevice.Connect (::GetCSCInputXptFromChannel (channel), ::GetDLInOutputXptFromChannel(SDIChannel)));
                                CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (channel), ::GetCSCOutputXptFromChannel (SDIChannel, false/*isKey*/, true/*isRGB*/)));
			}

                        CHECK(mDevice.SetFrameBufferFormat (channel, mPixelFormat));
                        CHECK(mDevice.EnableChannel (channel));
                        CHECK(mDevice.SetSDIInLevelBtoLevelAConversion (SDIChannel, IsInput3Gb (mInputSource) && !mInputIsRGB ? true : false));
                        if (!NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat))
                                break;
                        CHECK(mDevice.Set4kSquaresEnable(true, mInputChannel));
                }
        } else if (NTV2_INPUT_SOURCE_IS_ANALOG(mInputSource)) {
                //mDevice.SetTsiFrameEnable(false, NTV2_CHANNEL1);

                if (IsRGBFormat(mPixelFormat)) {
                        CHECK(mDevice.Connect (::GetCSCInputXptFromChannel (mInputChannel), NTV2_XptAnalogIn));
                        CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (mInputChannel), ::GetCSCOutputXptFromChannel (mInputChannel, false/*isKey*/, true/*isRGB*/)));
                } else {
                        CHECK(mDevice.Connect (::GetFrameBufferInputXptFromChannel (NTV2Channel (mInputChannel)), NTV2_XptAnalogIn));
                }
                CHECK(mDevice.SetFrameBufferFormat (mInputChannel, mPixelFormat));
                if (!mbFixedReference)
                        CHECK(mDevice.SetReference (NTV2_REFERENCE_ANALOG_INPUT));
        } else if (NTV2_IS_VALID_VIDEO_FORMAT (mVideoFormat)) {
                SetupHDMI();
        } else {
                LOG(LOG_LEVEL_WARNING) << "## DEBUG:  NTV2FrameGrabber::SetupInput:  Bad mInputSource switch value " << ::NTV2InputSourceToChannelSpec (mInputSource);
        }

        //      Enable and subscribe to the interrupts for the channel to be used...
        CHECK(mDevice.EnableInputInterrupt (mInputChannel));
        CHECK(mDevice.SubscribeInputVerticalEvent (mInputChannel));

        if (aja::codec_map.find(mPixelFormat) == aja::codec_map.end()) {
                cerr << "Cannot find valid mapping from AJA pixel format to UltraGrid" << endl;
                return AJA_STATUS_NOINPUT;
        }

        interlacing_t interlacing;
        if (NTV2_VIDEO_FORMAT_HAS_PROGRESSIVE_PICTURE(mVideoFormat)) {
                if (NTV2_IS_PSF_VIDEO_FORMAT(mVideoFormat)) {
                        interlacing = SEGMENTED_FRAME;
                } else {
                        interlacing = PROGRESSIVE;
                }
        } else {
                interlacing = INTERLACED_MERGED;
        }

        video_desc desc{GetDisplayWidth(mVideoFormat), GetDisplayHeight(mVideoFormat),
                aja::codec_map.at(mPixelFormat),
                GetFramesPerSecond(GetNTV2FrameRateFromVideoFormat(mVideoFormat)),
                interlacing,
                1};
#ifndef _MSC_VER
        cout << MOD_NAME "Detected input video mode: " << desc << endl;
#endif
        mPool.reconfigure(desc, vc_get_linesize(desc.width, desc.color_spec) * desc.height);

        return AJA_STATUS_SUCCESS;
}

AJAStatus vidcap_state_aja::SetupAudio (void)
{
        if (mAudioSource == NTV2_AUDIO_SOURCE_INVALID) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Audio capture disabled.\n";
                return AJA_STATUS_SUCCESS;
        }
        //      Have the audio system capture audio from the designated device input...
#if AJA_NTV2_SDK_VERSION_BEFORE(12,4)
        mDevice.SetAudioSystemInputSource (mAudioSystem, mInputSource);
#else
        CHECK_OK(mDevice.SetAudioSystemInputSource(mAudioSystem, mAudioSource, ::NTV2InputSourceToEmbeddedAudioInput(mInputSource)), string("Cannot set audio input source: ") + NTV2AudioSourceToString(mAudioSource), NOOP);
#endif

        mMaxAudioChannels = ::NTV2DeviceGetMaxAudioChannels (mDeviceID);
        if (mMaxAudioChannels < (int) *aja_audio_capture_channels) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Invalid number of capture channels requested. Requested " <<
                        *aja_audio_capture_channels << ", maximum " << mMaxAudioChannels << endl;
                return AJA_STATUS_FAIL;
        }
        if (!mDevice.SetNumberAudioChannels (mMaxAudioChannels, NTV2InputSourceToAudioSystem(mInputSource))) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unable to set channel count!\n";
                return AJA_STATUS_FAIL;
        }
        CHECK(mDevice.SetAudioRate (NTV2_AUDIO_48K, NTV2InputSourceToAudioSystem(mInputSource)));

        //      How big should the on-device audio buffer be?   1MB? 2MB? 4MB? 8MB?
        //      For this demo, 4MB will work best across all platforms (Windows, Mac & Linux)...
        CHECK(mDevice.SetAudioBufferSize (NTV2_AUDIO_BUFFER_BIG, NTV2InputSourceToAudioSystem(mInputSource)));

        //
        //      Loopback mode plays whatever audio appears in the input signal when it's
        //      connected directly to an output (i.e., "end-to-end" mode). If loopback is
        //      left enabled, the video will lag the audio as video frames get briefly delayed
        //      in our ring buffer. Audio, therefore, needs to come out of the (buffered) fram
        //      data being played, so loopback must be turned off...
        //
        CHECK(mDevice.SetAudioLoopBack (NTV2_AUDIO_LOOPBACK_OFF, mAudioSystem));

        //      Reset both the input and output sides of the audio system so that the buffer
        //      pointers are reset to zero and inhibited from advancing.
        CHECK(mDevice.SetAudioInputReset(mAudioSystem, true));

        //      Ensure that the audio system will capture samples when the reset is removed
        CHECK(mDevice.SetAudioCaptureEnable (mAudioSystem, true));

        mAudio.bps = 4;
        mAudio.sample_rate = 48000;
        mAudio.data = (char *) malloc(NTV2_AUDIOSIZE_MAX);
        mAudio.ch_count = *aja_audio_capture_channels;
        mAudio.max_size = NTV2_AUDIOSIZE_MAX;

#ifndef _MSC_VER
        LOG(LOG_LEVEL_NOTICE) << "AJA audio capture initialized successfully: " << audio_desc_from_frame(&mAudio) << "\n";
#endif

        return AJA_STATUS_SUCCESS;
}       //      SetupAudio

void vidcap_state_aja::SetupHostBuffers (void)
{
        mVancMode = NTV2_VANCMODE_OFF;
        mWideVanc = false;
        CHECK(mDevice.GetVANCMode (mVancMode));
        mVideoBufferSize = GetVideoWriteSize (mVideoFormat, mPixelFormat, mVancMode);
        mAudioBufferSize = NTV2_AUDIOSIZE_MAX;
}       //      SetupHostBuffers

AJAStatus vidcap_state_aja::Run()
{
        // Check that there's a valid input to capture
        if (mDevice.GetInputVideoFormat (mInputSource, mProgressive) == NTV2_FORMAT_UNKNOWN)
                cout << endl << "## WARNING:  No video signal present on the input connector" << endl;

        //      Start the playout and capture threads...
        StartProducerThread ();

        return AJA_STATUS_SUCCESS;
}

void vidcap_state_aja::Quit()
{
        //      Set the global 'quit' flag, and wait for the threads to go inactive...
        //mGlobalQuit = true;

        mProducerThread.join();

        //      Don't leave the audio system active after we exit
        CHECK(mDevice.SetAudioInputReset      (mAudioSystem, true));
}       //      Quit

//////////////////////////////////////////////

//      This is where we start the capture thread
void vidcap_state_aja::StartProducerThread (void)
{
        //      Create and start the capture thread..
        mProducerThread = thread(ProducerThreadStatic, this);
}       //      StartProducerThread


//      The capture thread function
void vidcap_state_aja::ProducerThreadStatic (vidcap_state_aja * pContext)           //      static
{
        pContext->CaptureFrames ();

}       //      ProducerThreadStatic

void vidcap_state_aja::CaptureFrames (void)
{
        uint32_t        currentInFrame                  = mInputChannel * 2;    //      Will ping-pong between 0 and 1

        uint32_t        currentAudioInAddress   = 0;
        uint32_t        audioReadOffset                 = 0;
        uint32_t        audioInWrapAddress              = 0;
        uint32_t        audioOutWrapAddress             = 0;
        uint32_t        audioBytesCaptured              = 0;
        NTV2FieldID     fieldID = NTV2_VIDEO_FORMAT_HAS_PROGRESSIVE_PICTURE(mVideoFormat) ? NTV2_FIELD0 : NTV2_FIELD1;

        CHECK(mDevice.GetAudioReadOffset      (audioReadOffset, mAudioSystem));
        CHECK(mDevice.GetAudioWrapAddress     (audioOutWrapAddress, mAudioSystem));

        //      Wait to make sure the next two SDK calls will be made during the same frame...
        CHECK(mDevice.WaitForInputFieldID (fieldID, mInputChannel));

        currentInFrame  ^= 1;
        CHECK(mDevice.SetInputFrame   (mInputChannel,  currentInFrame));

        //      Wait until the hardware starts filling the new buffers, and then start audio
        //      capture as soon as possible to match the video...
        CHECK(mDevice.WaitForInputFieldID (fieldID, mInputChannel));
        CHECK(mDevice.SetAudioInputReset (mAudioSystem, false));

        mAudioInLastAddress             = audioReadOffset;
        audioInWrapAddress              = audioOutWrapAddress + audioReadOffset;

        currentInFrame  ^= 1;
        CHECK(mDevice.SetInputFrame   (mInputChannel,  currentInFrame));

        while (!*aja_should_exit) {
                uint32_t *pHostAudioBuffer = NULL;
                //      Wait until the input has completed capturing a frame...
                CHECK(mDevice.WaitForInputFieldID (fieldID, mInputChannel));

                if (mAudioSource != NTV2_AUDIO_SOURCE_INVALID) {
                        pHostAudioBuffer = reinterpret_cast <uint32_t *> (aligned_malloc(NTV2_AUDIOSIZE_MAX, AJA_PAGE_SIZE));
                        //      Read the audio position registers as close to the interrupt as possible...
                        CHECK(mDevice.ReadAudioLastIn (currentAudioInAddress, mInputChannel));
                        currentAudioInAddress &= ~0x7f;  //      Force 128 B alignment (originally there was 4 bytes)
                        currentAudioInAddress += audioReadOffset;

                        if (((currentAudioInAddress + (audioInWrapAddress - audioReadOffset) - mAudioInLastAddress)
                                                % (audioInWrapAddress - audioReadOffset)) > NTV2_AUDIOSIZE_MAX) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Discarding audio samples!\n";
                                mAudioInLastAddress = (currentAudioInAddress + audioInWrapAddress - NTV2_AUDIOSIZE_MAX) % audioInWrapAddress;
                        }

                        if (currentAudioInAddress < mAudioInLastAddress) {
                                //      Audio address has wrapped around the end of the buffer.
                                //      Do the calculations and transfer from the last address to the end of the buffer...
                                audioBytesCaptured      = audioInWrapAddress - mAudioInLastAddress;

                                CHECK(mDevice.DMAReadAudio (mAudioSystem, pHostAudioBuffer, mAudioInLastAddress, audioBytesCaptured));

                                //      Transfer the new samples from the start of the buffer to the current address...
                                CHECK(mDevice.DMAReadAudio (mAudioSystem, &pHostAudioBuffer [audioBytesCaptured / sizeof (uint32_t)],
                                                audioReadOffset, currentAudioInAddress - audioReadOffset));

                                audioBytesCaptured += currentAudioInAddress - audioReadOffset;
                        } else {
                                audioBytesCaptured = currentAudioInAddress - mAudioInLastAddress;

                                //      No wrap, so just perform a linear DMA from the buffer...
                                CHECK(mDevice.DMAReadAudio (mAudioSystem, pHostAudioBuffer, mAudioInLastAddress, audioBytesCaptured));
                        }

                        mAudioInLastAddress = currentAudioInAddress;
                }

                //      Flip sense of the buffers again to refer to the buffers that the hardware isn't using (i.e. the off-screen buffers)...
                currentInFrame  ^= 1;

                shared_ptr<video_frame> out = mPool.get_frame();
                //      DMA the new frame to system memory...
                CHECK(mDevice.DMAReadFrame (currentInFrame, reinterpret_cast<uint32_t *>(out->tiles[0].data), mVideoBufferSize));

                if (log_level >= LOG_LEVEL_DEBUG) {
                        RP188_STRUCT    timecodeValue;
                        string 		timeCodeString;
                        //      Use the embedded input time code...
                        CHECK(mDevice.GetRP188Data (mInputChannel, 0, timecodeValue));
                        CRP188  inputRP188Info  (timecodeValue);
                        CHECK(inputRP188Info.GetRP188Str (timeCodeString));
                        LOG(LOG_LEVEL_DEBUG) << "AJA: Captured frame with timecode: " << timeCodeString << '\n';
                }

                //      Check for dropped frames by ensuring the hardware has not started to process
                //      the buffers that were just filled....
                uint32_t readBackIn;
                CHECK(mDevice.GetInputFrame   (mInputChannel,         readBackIn));

                if (readBackIn == currentInFrame) {
                        cerr    << "## WARNING:  Drop detected:  current in " << currentInFrame << ", readback in " << readBackIn << endl;
                }

                //      Tell the hardware which buffers to start using at the beginning of the next frame...
                CHECK(mDevice.SetInputFrame   (mInputChannel,  currentInFrame));

                unique_lock<mutex> lk(mOutputFrameLock);
                mOutputFrame = out;
		mOutputAudioFrame = shared_ptr<uint32_t>(pHostAudioBuffer, aligned_free);
		mOutputAudioFrameSize = audioBytesCaptured;
                lk.unlock();
                mOutputFrameReady.notify_one();
	}       //      loop til quit signaled
}       //      CaptureFrames

struct video_frame *vidcap_state_aja::grab(struct audio_frame **audio)
{
        if (*aja_should_exit) {
                return NULL;
        }

        struct video_frame *ret;

        unique_lock<mutex> lk(mOutputFrameLock);
        if (mOutputFrameReady.wait_for(lk, chrono::milliseconds(100), [this]{return mOutputFrame != NULL;}) == false) {
                return NULL;
        }

        ret = mOutputFrame.get();
        ret->callbacks.dispose_udata = new shared_ptr<video_frame>(mOutputFrame);
        ret->callbacks.dispose = [](video_frame *f) { delete static_cast<shared_ptr<video_frame> *>(f->callbacks.dispose_udata); };

        mOutputFrame = NULL;

        if (mOutputAudioFrame) {
                for (int i = 0; i < mAudio.ch_count; ++i) {
                        remux_channel(mAudio.data, (char *) mOutputAudioFrame.get(), mAudio.bps, mOutputAudioFrameSize, mMaxAudioChannels, mAudio.ch_count, i, i);
                }
                mAudio.data_len = mOutputAudioFrameSize / mMaxAudioChannels * mAudio.ch_count;
                mOutputAudioFrame = NULL;
                *audio = &mAudio;
        } else {
                *audio = NULL;
        }

        lk.unlock();

        mFrames += 1;

        chrono::system_clock::time_point now = chrono::system_clock::now();
        double seconds = chrono::duration_cast<chrono::microseconds>(now - mT0).count() / 1000000.0;

        if (seconds >= 5) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME "" << mFrames << " frames in "
                        << seconds << " seconds = " <<  mFrames / seconds << " FPS\n";
                mT0 = now;
                mFrames = 0;
        }

        return ret;
}

bool vidcap_state_aja::IsInput3Gb(const NTV2InputSource inputSource)
{
        bool    is3Gb   (false);

        CHECK(mDevice.GetSDIInput3GbPresent (is3Gb, ::NTV2InputSourceToChannel (inputSource)));

        return is3Gb;
}

static void show_help() {
        cout << "Usage:\n";
        cout << rang::style::bold << rang::fg::red << "\t-t aja" << rang::fg::reset << "[[:4K][:clear-routing][:channel=<ch>][:codec=<pixfmt>][:connection=<c>][:device=<idx>][:format=<fmt>][:progressive][:RGB|:YUV]|:help] -r [embedded|AESEBU|analog]\n" << rang::style::reset;
        cout << "where\n";

        cout << rang::style::bold << "\t4K\n" << rang::style::reset;
        cout << "\t\tVideo input is 4K.\n";

        cout << rang::style::bold << "\tclear-routing\n" << rang::style::reset <<
                "\t\tremove all existing signal paths for device\n";

        cout << rang::style::bold << "\tchannel\n" << rang::style::reset;
        cout << "\t\tChannel number to use (indexed from 1). Doesn't need to be set for SDI, useful for HDMI (capture and display should have different channel numbers if both used, also other than 1 if SDI1 is in use).\n";

        cout << rang::style::bold << "\tconnection\n" << rang::style::reset <<
                "\t\tConnection can be one of: ";
        NTV2InputSource source = NTV2InputSource();
        while (source != NTV2_INPUTSOURCE_INVALID) {
                if (source > 0) {
                        cout << ", ";
                }
                cout << NTV2InputSourceToString(source, true);
                // should be this, but GetNTV2InputSourceForIndex knows only SDIs
                //source = ::GetNTV2InputSourceForIndex(::GetIndexForNTV2InputSource(source) + 1);
                source = (NTV2InputSource) ((int) source + 1);
        }
        cout << "\n";

        cout << rang::style::bold << "\tprogressive\n" << rang::style::reset;
        cout << "\t\tVideo input is progressive.\n";

        cout << rang::style::bold << "\tRGB|YUV\n" << rang::style::reset;
        cout << "\t\tSet SDI video input to RGB or YCbCr explicitly. If capture pixel format is set, expect the same encoding on input. You can override this setting by specifying the signal color space explicitly by this option (see https://github.com/CESNET/UltraGrid/wiki/Device-Settings#RGB_over_SDI for details).\n";

        cout << "\n";

        printf("Available devices:\n");
        CNTV2DeviceScanner      deviceScanner;
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                cout << "\t"  << rang::style::bold << i << rang::style::reset << ") " << rang::style::bold << info.deviceIdentifier << rang::style::reset << ". " << info;
                NTV2VideoFormatSet fmt_set;
                if (NTV2DeviceGetSupportedVideoFormats(info.deviceID, fmt_set)) {
                        cout << rang::style::underline << "\tAvailable formats:" << rang::style::reset;
                        for (auto fmt : fmt_set) {
                                if (fmt != *fmt_set.begin()) {
                                        cout << ",";
                                }
                                cout << " \"" << NTV2VideoFormatToString(fmt) << "\"";
                        }
                        cout << "\n";
                }
                NTV2FrameBufferFormatSet pix_fmts;
                if (NTV2DeviceGetSupportedPixelFormats(info.deviceID, pix_fmts)) {
                        if (pix_fmts.count(NTV2_FBF_10BIT_YCBCR) == 0 && NTV2DeviceCanDoFrameBufferFormat(info.deviceID, NTV2_FBF_10BIT_YCBCR)) {
                                pix_fmts.insert(NTV2_FBF_10BIT_YCBCR); // workaround NTV2 bug
                        }
                        cout << rang::style::underline << "\tAvailable pixel formats:" << rang::style::reset;
                        for (auto fmt : pix_fmts) {
                                if (fmt != *pix_fmts.begin()) {
                                        cout << ",";
                                }
                                if (aja::codec_map.find(fmt) == aja::codec_map.end()) {
                                        cout << " " << NTV2FrameBufferFormatToString(fmt) << " (unsupported)";
                                } else {
                                        cout << " " << get_codec_name(aja::codec_map.at(fmt));

                                }
                        }
                        cout << "\n";
                }
                cout << rang::style::underline << "\tNumber of frame stores: " << rang::style::reset << NTV2DeviceGetNumFrameStores (info.deviceID) << "\n";
        }
        if (deviceScanner.GetNumDevices() == 0) {
                cout << rang::fg::red << "\tno devices found\n" << rang::fg::reset;
        }
        cout << "\n";
}

LINK_SPEC int vidcap_aja_init(struct vidcap_params *params, void **state)
{
        unordered_map<string, string> parameters_map;
        char *tmp = strdup(vidcap_params_get_fmt(params));
        if (strcmp(tmp, "help") == 0) {
                show_help();
                free(tmp);
                return VIDCAP_INIT_NOERR;
        }
        char *item, *cfg = tmp;
        while ((item = strtok(cfg, ":"))) {
                char *key_cstr = item;
                if (strchr(item, '=')) {
                        char *val_cstr = strchr(item, '=') + 1;
                        *strchr(item, '=') = '\0';
                        parameters_map[key_cstr] = val_cstr;
                } else {
                        parameters_map[key_cstr] = string();
                }
                cfg = NULL;
        }
        free(tmp);

        vidcap_state_aja *ret = nullptr;
        try {
                ret = new vidcap_state_aja(parameters_map, vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY);
                ret->Run();
        } catch (string const & s) {
                LOG(LOG_LEVEL_ERROR) << "[AJA cap.] " << s << "\n";
                delete ret;
                return VIDCAP_INIT_FAIL;
        } catch (...) {
                delete ret;
                return VIDCAP_INIT_FAIL;
        }
        *state = ret;
        return VIDCAP_INIT_OK;
}

LINK_SPEC void vidcap_aja_done(void *state)
{
        auto s = static_cast<vidcap_state_aja *>(state);
        s->Quit();
        delete s;
}

LINK_SPEC struct video_frame *vidcap_aja_grab(void *state, struct audio_frame **audio)
{
        return ((vidcap_state_aja *) state)->grab(audio);
}

LINK_SPEC struct vidcap_type *vidcap_aja_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type *vt;
        *deleter = free;

        vt = (struct vidcap_type *)calloc(1, sizeof(struct vidcap_type));
        if (vt == nullptr) {
                return nullptr;
        }

        vt->name = "aja";
        vt->description = "AJA capture card";

        if (!verbose) {
                return vt;
        }

        CNTV2DeviceScanner      deviceScanner;
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                vt->card_count += 1;
                vt->cards = (struct device_info *)
                        realloc(vt->cards, vt->card_count * sizeof(struct device_info));
                memset(&vt->cards[vt->card_count - 1], 0, sizeof(struct device_info));
                snprintf(vt->cards[vt->card_count - 1].id, sizeof vt->cards[vt->card_count - 1].id,
                                "device=%d", i);
                snprintf(vt->cards[vt->card_count - 1].name, sizeof vt->cards[vt->card_count - 1].name,
                                "AJA %s", info.deviceIdentifier.c_str());
        }

        return vt;
}

static void supersede_compiler_warning_workaround() ATTRIBUTE(unused);
static void supersede_compiler_warning_workaround()
{
        UNUSED(__AJA_trigger_link_error_if_incompatible__);
}

#ifndef _MSC_VER
static const struct video_capture_info vidcap_aja_info = {
        vidcap_aja_probe,
        vidcap_aja_init,
        vidcap_aja_done,
        vidcap_aja_grab,
        false
};

REGISTER_MODULE(aja, &vidcap_aja_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
#endif


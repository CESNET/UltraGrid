/**
 * @file   video_capture/aja.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2016 CESNET, z. s. p. o.
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
 * @todo
 * Audio support is currently broken. Readd it in future.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include "video_capture.h"

#include "ajatypes.h"
#include "ntv2devicescanner.h"
#include "ntv2democommon.h"
#include "ntv2capture.h"
#include "ajastuff/common/videotypes.h"
#include "ajastuff/common/circularbuffer.h"
#include "ajastuff/system/process.h"
#include "ajastuff/system/systemtime.h"
#include "ajastuff/system/thread.h"

#include "ntv2utils.h"
#include "ntv2devicefeatures.h"

#define NTV2_AUDIOSIZE_MAX      (401 * 1024)

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace std;

struct aligned_data_allocator {
        void *allocate(size_t size) {
                return aligned_malloc(size, AJA_PAGE_SIZE);
        }
        void deallocate(void *ptr) {
                aligned_free(ptr);
        }
};

static constexpr ULWord app = AJA_FOURCC ('U','L','G','R');

class vidcap_state_aja {
        public:
                vidcap_state_aja(unordered_map<string, string> const & parameters);
                void Init();
                virtual ~vidcap_state_aja();
                struct video_frame *grab(struct audio_frame **audio);
                AJAStatus Run();
                void Quit();
        private:
                unsigned int           mDeviceIndex;
                CNTV2Card              mDevice;
                NTV2DeviceID           mDeviceID;                     ///     My device identifier
                NTV2EveryFrameTaskMode mSavedTaskMode;                /// Used to restore prior every-frame task mode
                NTV2Channel            mInputChannel;                 ///     My input channel
                NTV2VideoFormat        mVideoFormat;                  ///     My video format
                NTV2FrameBufferFormat  mPixelFormat;                  ///     My pixel format
                bool                   mVancEnabled;                  ///     VANC enabled?
                bool                   mWideVanc;                     ///     Wide VANC?
                NTV2InputSource        mInputSource;                  ///     The input source I'm using
                NTV2AudioSystem        mAudioSystem;                  ///     The audio system I'm using
                uint32_t               mVideoBufferSize;              ///     My video buffer size, in bytes
                uint32_t               mAudioBufferSize;              ///     My audio buffer size, in bytes
                AJAThread             *mProducerThread;               ///     My producer thread object -- does the frame capturing
                video_frame_pool<aligned_data_allocator> mPool;
                shared_ptr<video_frame> mOutputFrame;
                mutex                  mOutputFrameLock;
                condition_variable     mOutputFrameReady;
                bool                   mProgressive;
                chrono::system_clock::time_point mT0;
                int                    mFrames;
                struct audio_frame     mAudio;
                int                    mMaxAudioChannels;
                NTV2TCSource           mTimeCodeSource;                ///< @brief     Time code source
                bool                   mCheckFor4K;

                AJAStatus SetupVideo();
                AJAStatus SetupAudio();
                void SetupHostBuffers();
                NTV2VideoFormat GetVideoFormatFromInputSource();
                void EnableInput(NTV2InputSource source);

                /**
                  @brief  Starts my frame consumer thread.
                 **/
                virtual void                    StartProducerThread (void);
                virtual void                    CaptureFrames (void);
                static void     ProducerThreadStatic (AJAThread * pThread, void * pContext);
};

vidcap_state_aja::vidcap_state_aja(unordered_map<string, string> const & parameters) :
        mDeviceIndex(0), mInputChannel(NTV2_CHANNEL1), mVideoFormat(NTV2_FORMAT_UNKNOWN),
        mPixelFormat(NTV2_FBF_8BIT_YCBCR), mInputSource(NTV2_INPUTSOURCE_SDI1),
        mAudioSystem(NTV2_AUDIOSYSTEM_1),
        mProducerThread(0), mOutputFrame(0), mProgressive(false),
        mT0(chrono::system_clock::now()), mFrames(0), mAudio(audio_frame()), mMaxAudioChannels(0),
        mTimeCodeSource(NTV2_TCSOURCE_DEFAULT), mCheckFor4K(false)
{
        for (auto it : parameters) {
                if (it.first == "progressive") {
                        mProgressive = true;
                } else if (it.first == "4K" || it.first == "4k") {
                        mCheckFor4K = true;
                } else if (it.first == "device") {
                        mDeviceIndex = stol(it.second, nullptr, 10);
                } else if (it.first == "source") {
                        NTV2InputSource source{};
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
                        NTV2VideoFormat format{};
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
                } else {
                        throw string("Unknown option: ") + it.first;
                }
        }
        Init();
}

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

        if (!mDevice.AcquireStreamForApplication (app, static_cast <uint32_t> (AJAProcess::GetPid ())))
                throw string("Cannot aquire stream.");

        mDevice.GetEveryFrameServices (&mSavedTaskMode);        //      Save the current state before we change it
        mDevice.SetEveryFrameServices (NTV2_OEM_TASKS);

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
        delete mProducerThread;
        mProducerThread = NULL;

        //      Unsubscribe from input vertical event...
        mDevice.UnsubscribeInputVerticalEvent (mInputChannel);

        mDevice.SetEveryFrameServices (mSavedTaskMode);                                                                                                 //      Restore previous service level
        mDevice.ReleaseStreamForApplication (app, static_cast <uint32_t> (AJAProcess::GetPid ()));     //      Release the device

        mOutputFrame = {};
        free(mAudio.data);
}

static const unordered_map<NTV2FrameBufferFormat, codec_t, hash<int>> codec_map = {
        { NTV2_FBF_10BIT_YCBCR, v210 },
        { NTV2_FBF_8BIT_YCBCR, UYVY },
        { NTV2_FBF_RGBA, RGBA },
        { NTV2_FBF_10BIT_RGB, R10k },
        { NTV2_FBF_8BIT_YCBCR_YUY2, YUYV },
        { NTV2_FBF_24BIT_RGB, RGB },
        { NTV2_FBF_24BIT_BGR, BGR },
};

static const NTV2InputCrosspointID gCSCVideoInput [] = {
        NTV2_XptCSC1VidInput, NTV2_XptCSC2VidInput, NTV2_XptCSC3VidInput,
        NTV2_XptCSC4VidInput, NTV2_XptCSC5VidInput, NTV2_XptCSC6VidInput,
        NTV2_XptCSC7VidInput, NTV2_XptCSC8VidInput };

static const NTV2OutputCrosspointID gSDIInputOutputs[] = {
        NTV2_XptSDIIn1, NTV2_XptSDIIn2, NTV2_XptSDIIn3, NTV2_XptSDIIn4,
        NTV2_XptSDIIn5, NTV2_XptSDIIn6, NTV2_XptSDIIn7, NTV2_XptSDIIn8 };

static const NTV2InputCrosspointID gFrameBufferInput[] = {
        NTV2_XptFrameBuffer1Input, NTV2_XptFrameBuffer2Input,
        NTV2_XptFrameBuffer3Input, NTV2_XptFrameBuffer4Input,
        NTV2_XptFrameBuffer5Input, NTV2_XptFrameBuffer6Input,
        NTV2_XptFrameBuffer7Input, NTV2_XptFrameBuffer8Input };

static const NTV2OutputCrosspointID gCSCVidRGBOutput[]  __attribute__((unused)) = {
        NTV2_XptCSC1VidRGB, NTV2_XptCSC2VidRGB, NTV2_XptCSC3VidRGB,
        NTV2_XptCSC4VidRGB, NTV2_XptCSC5VidRGB, NTV2_XptCSC6VidRGB,
        NTV2_XptCSC7VidRGB, NTV2_XptCSC8VidRGB };

static const NTV2OutputCrosspointID gCSCVidYUVOutput[] = {
        NTV2_XptCSC1VidYUV, NTV2_XptCSC2VidYUV, NTV2_XptCSC3VidYUV,
        NTV2_XptCSC4VidYUV, NTV2_XptCSC5VidYUV, NTV2_XptCSC6VidYUV,
        NTV2_XptCSC7VidYUV, NTV2_XptCSC8VidYUV };

static const NTV2OutputCrosspointID gHDMIInYUVOutputs[] = {
        NTV2_XptHDMIIn, NTV2_XptHDMIInQ2, NTV2_XptHDMIInQ3, NTV2_XptHDMIInQ4 };

static const NTV2OutputCrosspointID gHDMIInRGBOutputs[] = {
        NTV2_XptHDMIInRGB, NTV2_XptHDMIInQ2RGB, NTV2_XptHDMIInQ3RGB, NTV2_XptHDMIInQ4RGB };

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
                                mDevice.SetMultiFormatMode (true);
                        source = ::GetNTV2InputSourceForIndex (ndx + 0);
                        EnableInput(source);
                        videoFormat = mDevice.GetInputVideoFormat(source, mProgressive);
                        NTV2Standard    videoStandard   (::GetNTV2StandardFromVideoFormat (videoFormat));
                        if (mCheckFor4K && (videoStandard == NTV2_STANDARD_1080p))
                        {
                                if (::NTV2DeviceCanDoMultiFormat (mDeviceID))
                                        mDevice.SetMultiFormatMode (false);
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
                mDevice.SetSDITransmitEnable (channel, false);       //      Disable transmit mode...
                for (unsigned ndx (0);  ndx < 10;  ndx++)
                        mDevice.WaitForInputVerticalInterrupt (channel);     //      ...and give the device some time to lock to a signal
        }

}

AJAStatus vidcap_state_aja::SetupVideo()
{
        //      Set the video format to match the incomming video format.
        //      Does the device support the desired input source?
        if (!::NTV2BoardCanDoInputSource (mDeviceID, mInputSource))
                return AJA_STATUS_BAD_PARAM;    //      Nope

        mInputChannel = ::NTV2InputSourceToChannel (mInputSource);
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
        if (mVideoFormat == NTV2_FORMAT_UNKNOWN)
                mVideoFormat =  GetVideoFormatFromInputSource();
        //mVideoFormat = NTV2_FORMAT_4x1920x1080p_2500;
        //mVideoFormat = NTV2_FORMAT_1080p_5000_A;
        if (mVideoFormat == NTV2_FORMAT_UNKNOWN)
        {
                LOG(LOG_LEVEL_ERROR) << "## ERROR:  No input signal or unknown format" << endl;
                return AJA_STATUS_NOINPUT;      //      Sorry, can't handle this format
        }

        //      Set the device video format to whatever we detected at the input...
        mDevice.SetVideoFormat (mVideoFormat, false, false, mInputChannel);

        //      Set the frame buffer pixel format for all the channels on the device
        //      (assuming it supports that pixel format -- otherwise default to 8-bit YCbCr)...
        if (!::NTV2BoardCanDoFrameBufferFormat (mDeviceID, mPixelFormat))
                mPixelFormat = NTV2_FBF_8BIT_YCBCR;

        //      Set the pixel format for both device frame buffers...
        mDevice.SetFrameBufferFormat (mInputChannel, mPixelFormat);

        //      Enable and subscribe to the interrupts for the channel to be used...
        mDevice.EnableInputInterrupt (mInputChannel);
        mDevice.SubscribeInputVerticalEvent (mInputChannel);

        //      Tell the hardware which buffers to use until the main worker thread runs
        mDevice.SetInputFrame   (mInputChannel,  0);

        //      Set the Frame Store modes
        mDevice.SetMode (mInputChannel,  NTV2_MODE_CAPTURE);

        mDevice.SetReference (NTV2_REFERENCE_FREERUN);

        CNTV2SignalRouter       router;

        if (NTV2_INPUT_SOURCE_IS_SDI (mInputSource)) {
                for (unsigned offset (0);  offset < 4;  offset++) {
                        router.AddConnection (gFrameBufferInput [mInputChannel + offset], gSDIInputOutputs [mInputChannel + offset]);
                        mDevice.SetFrameBufferFormat (NTV2Channel (mInputChannel + offset), mPixelFormat);
                        mDevice.EnableChannel (NTV2Channel (mInputChannel + offset));
                        if (!NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat))
                                break;
                }
        } else if (mInputSource == NTV2_INPUTSOURCE_ANALOG) {
                router.AddConnection (gFrameBufferInput [NTV2_CHANNEL1], NTV2_XptAnalogIn);
                mDevice.SetFrameBufferFormat (NTV2_CHANNEL1, mPixelFormat);
                mDevice.SetReference (NTV2_REFERENCE_ANALOG_INPUT);
        } else if (mInputSource == NTV2_INPUTSOURCE_HDMI) {
                NTV2LHIHDMIColorSpace   hdmiColor       (NTV2_LHIHDMIColorSpaceRGB);
                mDevice.GetHDMIInputColor (hdmiColor);
                mDevice.SetReference (NTV2_REFERENCE_HDMI_INPUT);
                mDevice.SetHDMIV2Mode (NTV2_HDMI_V2_4K_CAPTURE);              //      Allow 4K HDMI capture
                for (unsigned chan (0);  chan < 4;  chan++) {
                        mDevice.EnableChannel (NTV2Channel (chan));
                        mDevice.SetMode (NTV2Channel (chan), NTV2_MODE_CAPTURE);
                        mDevice.SetFrameBufferFormat (NTV2Channel (chan), mPixelFormat);
                        if (hdmiColor == NTV2_LHIHDMIColorSpaceYCbCr) {
                                router.AddConnection (gFrameBufferInput [chan], gHDMIInYUVOutputs [chan]);
                        } else {
                                router.AddConnection (gCSCVideoInput [chan], gHDMIInRGBOutputs [chan]);
                                router.AddConnection (gFrameBufferInput [chan], gCSCVidYUVOutput [chan]);
                        }
                        if (!NTV2_IS_4K_VIDEO_FORMAT (mVideoFormat))
                                break;
                }       //      loop once for single channel, or 4 times for 4K/UHD
        }
        else {
                LOG(LOG_LEVEL_WARNING) << "## DEBUG:  NTV2FrameGrabber::SetupInput:  Bad mInputSource switch value " << ::NTV2InputSourceToChannelSpec (mInputSource);
        }

        mDevice.ApplySignalRoute (router);

        //      Enable and subscribe to the interrupts for the channel to be used...
        mDevice.EnableInputInterrupt (mInputChannel);
        mDevice.SubscribeInputVerticalEvent (mInputChannel);

        if (codec_map.find(mPixelFormat) == codec_map.end()) {
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
                codec_map.at(mPixelFormat),
                GetFramesPerSecond(GetNTV2FrameRateFromVideoFormat(mVideoFormat)),
                interlacing,
                1};
        cout << "[AJA] Detected input video mode: " << desc << endl;
        mPool.reconfigure(desc, vc_get_linesize(desc.width, desc.color_spec) * desc.height);

        return AJA_STATUS_SUCCESS;
}

AJAStatus vidcap_state_aja::SetupAudio (void)
{
        //      Have the audio system capture audio from the designated device input...
#if AJA_NTV2_SDK_VERSION_BEFORE(12,4)
        mDevice.SetAudioSystemInputSource (mAudioSystem, mInputSource);
#else
        NTV2AudioSource audioSource     (NTV2_AUDIO_EMBEDDED);
        if (NTV2_INPUT_SOURCE_IS_HDMI (mInputSource))
                audioSource = NTV2_AUDIO_HDMI;
        else if (NTV2_INPUT_SOURCE_IS_ANALOG (mInputSource))
                audioSource = NTV2_AUDIO_ANALOG;

        mDevice.SetAudioSystemInputSource (mAudioSystem, audioSource, ::NTV2ChannelToEmbeddedAudioInput(mInputChannel));
#endif

        mMaxAudioChannels = ::NTV2BoardGetMaxAudioChannels (mDeviceID);
        mDevice.SetNumberAudioChannels (mMaxAudioChannels, NTV2InputSourceToAudioSystem(mInputSource));
        mDevice.SetAudioRate (NTV2_AUDIO_48K, NTV2InputSourceToAudioSystem(mInputSource));

        //      How big should the on-device audio buffer be?   1MB? 2MB? 4MB? 8MB?
        //      For this demo, 4MB will work best across all platforms (Windows, Mac & Linux)...
        mDevice.SetAudioBufferSize (NTV2_AUDIO_BUFFER_BIG, NTV2InputSourceToAudioSystem(mInputSource));

        if (mMaxAudioChannels < (int) audio_capture_channels)
        {
                cerr << "[AJA] Invalid number of capture channels requested. Requested " <<
                        audio_capture_channels << ", maximum " << mMaxAudioChannels << endl;
        }
        mAudio.bps = 4;
        mAudio.sample_rate = 48000;
        mAudio.data = (char *) malloc(NTV2_AUDIOSIZE_MAX);
        mAudio.ch_count = audio_capture_channels;
        mAudio.max_size = NTV2_AUDIOSIZE_MAX;

        LOG(LOG_LEVEL_NOTICE) << "AJA audio capture initialized sucessfully: " << audio_desc_from_frame(&mAudio) << "\n";

        return AJA_STATUS_SUCCESS;

}       //      SetupAudio

void vidcap_state_aja::SetupHostBuffers (void)
{
        mVancEnabled = false;
        mWideVanc = false;
        mDevice.GetEnableVANCData (&mVancEnabled, &mWideVanc);
        mVideoBufferSize = GetVideoWriteSize (mVideoFormat, mPixelFormat, mVancEnabled, mWideVanc);
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

        if (mProducerThread)
                while (mProducerThread->Active ())
                        AJATime::Sleep (10);

}       //      Quit

//////////////////////////////////////////////

//      This is where we start the capture thread
void vidcap_state_aja::StartProducerThread (void)
{
        //      Create and start the capture thread...
        mProducerThread = new AJAThread ();
        mProducerThread->Attach (ProducerThreadStatic, this);
        mProducerThread->SetPriority (AJA_ThreadPriority_High);
        mProducerThread->Start ();

}       //      StartProducerThread


//      The capture thread function
void vidcap_state_aja::ProducerThreadStatic (AJAThread * pThread, void * pContext)           //      static
{
        (void) pThread;

        //      Grab the NTV2Capture instance pointer from the pContext parameter,
        //      then call its CaptureFrames method...
        auto pApp    (reinterpret_cast <vidcap_state_aja *> (pContext));
        pApp->CaptureFrames ();

}       //      ProducerThreadStatic

void vidcap_state_aja::CaptureFrames (void)
{
        uint32_t        currentInFrame                  = 0;    //      Will ping-pong between 0 and 1

        //      Wait to make sure the next two SDK calls will be made during the same frame...
        mDevice.WaitForInputFieldID (NTV2_FIELD0, mInputChannel);
        currentInFrame  ^= 1;
        mDevice.SetInputFrame   (mInputChannel,  currentInFrame);

        //      Wait until the hardware starts filling the new buffers, and then start audio
        //      capture as soon as possible to match the video...
        mDevice.WaitForInputFieldID (NTV2_FIELD0, mInputChannel);

        currentInFrame  ^= 1;
        mDevice.SetInputFrame   (mInputChannel,  currentInFrame);

	while (!should_exit) {
		//      Wait until the input has completed capturing a frame...
                mDevice.WaitForInputFieldID (NTV2_FIELD0, mInputChannel);
		//      Flip sense of the buffers again to refer to the buffers that the hardware isn't using (i.e. the off-screen buffers)...
                currentInFrame  ^= 1;

                shared_ptr<video_frame> out = mPool.get_frame();
                //      DMA the new frame to system memory...
                mDevice.DMAReadFrame (currentInFrame, reinterpret_cast<uint32_t *>(out->tiles[0].data), mVideoBufferSize);

                //      Check for dropped frames by ensuring the hardware has not started to process
                //      the buffers that were just filled....
                uint32_t readBackIn;
                mDevice.GetInputFrame   (mInputChannel,         readBackIn);

                if (readBackIn == currentInFrame) {
                        cerr    << "## WARNING:  Drop detected:  current in " << currentInFrame << ", readback in " << readBackIn << endl;
                }

                //      Tell the hardware which buffers to start using at the beginning of the next frame...
                mDevice.SetInputFrame   (mInputChannel,  currentInFrame);

                unique_lock<mutex> lk(mOutputFrameLock);
                mOutputFrame = out;
                lk.unlock();
                mOutputFrameReady.notify_one();
	}       //      loop til quit signaled
}       //      CaptureFrames

#define NTV2_AUDIOSIZE_48K (48 * 1024)

struct video_frame *vidcap_state_aja::grab(struct audio_frame **audio)
{
        if (should_exit) {
                return NULL;
        }

        struct video_frame *ret;

        unique_lock<mutex> lk(mOutputFrameLock);
        if (mOutputFrameReady.wait_for(lk, chrono::milliseconds(100), [this]{return mOutputFrame != NULL;}) == false) {
                return NULL;
        }

        ret = mOutputFrame.get();
        ret->dispose_udata = new shared_ptr<video_frame>(mOutputFrame);
        ret->dispose = [](video_frame *f) { delete static_cast<shared_ptr<video_frame> *>(f->dispose_udata); };

        mOutputFrame = NULL;
        lk.unlock();

        *audio = NULL;

        mFrames += 1;

        chrono::system_clock::time_point now = chrono::system_clock::now();
        double seconds = chrono::duration_cast<chrono::microseconds>(now - mT0).count() / 1000000.0;

        if (seconds >= 5) {
                LOG(LOG_LEVEL_INFO) << "[AJA] " << mFrames << " frames in "
                        << seconds << " seconds = " <<  mFrames / seconds << " FPS\n";
                mT0 = now;
                mFrames = 0;
        }

        return ret;
}

static void show_help() {
        printf("Usage:\n");
        printf("\t-t aja[:device=<idx>][:progressive][:4K][:source=<src>][:format=<fmt>]\n");
        printf("\n");

        printf("progressive\n");
        printf("\tVideo input is progressive.\n");
        printf("\n");

        printf("4K\n");
        printf("\tVideo input is 4K.\n");
        printf("\n");

        printf("source\n");
        printf("\tSource can be one of SDIX (replace X with index, starting with 1), HDMI, or Analog.\n");
        printf("\n");


        printf("Available devices:\n");
        CNTV2DeviceScanner      deviceScanner;
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                cout << "\t" << i << ") " << info.deviceIdentifier << ". " << info;
                NTV2VideoFormatSet fmt_set;
                if (NTV2DeviceGetSupportedVideoFormats(info.deviceID, fmt_set)) {
                        cout << "\tAvailable formats:";
                        for (auto fmt : fmt_set) {
                                cout << " \"" << NTV2VideoFormatToString(fmt) << "\"";
                        }
                        cout << "\n";
                }
        }
        cout << "\n";
}

static int vidcap_aja_init(const struct vidcap_params *params, void **state)
{
        unordered_map<string, string> parameters_map;
        char *tmp = strdup(vidcap_params_get_fmt(params));
        if (strcmp(tmp, "help") == 0) {
                show_help();
                free(tmp);
                return VIDCAP_INIT_NOERR;
        }
        char *item, *save_ptr, *cfg = tmp;
        while ((item = strtok_r(cfg, ":", &save_ptr))) {
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
                ret = new vidcap_state_aja(parameters_map);
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

static void vidcap_aja_done(void *state)
{
        auto s = static_cast<vidcap_state_aja *>(state);
        s->Quit();
        delete s;
}

static struct video_frame *vidcap_aja_grab(void *state, struct audio_frame **audio)
{
        return ((vidcap_state_aja *) state)->grab(audio);
}

static struct vidcap_type *vidcap_aja_probe(bool)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "aja";
                vt->description = "AJA capture card";
        }
        return vt;
}

static void supersede_compiler_warning_workaround() __attribute__((unused));
static void supersede_compiler_warning_workaround()
{
        UNUSED(__AJA_trigger_link_error_if_incompatible__);
}

static const struct video_capture_info vidcap_aja_info = {
        vidcap_aja_probe,
        vidcap_aja_init,
        vidcap_aja_done,
        vidcap_aja_grab,
};

REGISTER_MODULE(aja, &vidcap_aja_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


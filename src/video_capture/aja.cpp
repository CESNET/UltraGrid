/**
 * @file   video_capture/aja.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET z.s.p.o.
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
#endif

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/aja.h"

#include "ajatypes.h"
#include "ntv2boardscan.h"
#include "ntv2democommon.h"
#include "ntv2capture.h"
#include "ajastuff/common/videotypes.h"
#include "ajastuff/common/circularbuffer.h"
#include "ajastuff/system/process.h"
#include "ajastuff/system/systemtime.h"
#include "ajastuff/system/thread.h"

#include "ntv2utils.h"
#include "ntv2boardfeatures.h"

#define NTV2_AUDIOSIZE_MAX      (401 * 1024)

#include <chrono>
#include <string>
#include <unordered_map>

using namespace std;

static constexpr ULWord app = AJA_FOURCC ('U','L','G','R');

class vidcap_state_aja {
        public:
                vidcap_state_aja(unordered_map<string, string> const & parameters);
                void Init();
                ~vidcap_state_aja();
                struct video_frame *grab(struct audio_frame **audio);
                AJAStatus Run();
                void Quit();
        private:
                int                    mDeviceIndex;
                CNTV2Card              mDevice;
                NTV2BoardID            mDeviceID;                     ///     My device identifier
                NTV2EveryFrameTaskMode mSavedTaskMode;                /// Used to restore prior every-frame task mode
                const NTV2Channel      mInputChannel;                 ///     My input channel
                NTV2VideoFormat        mVideoFormat;                  ///     My video format
                NTV2FrameBufferFormat  mPixelFormat;                  ///     My pixel format
                bool                   mVancEnabled;                  ///     VANC enabled?
                bool                   mWideVanc;                     ///     Wide VANC?
                NTV2InputSource        mInputSource;                  ///     The input source I'm using
                NTV2AudioSystem        mAudioSystem;                  ///     The audio system I'm using
                AJACircularBuffer <AVDataBuffer *> mAVCircularBuffer; ///     My ring buffer object
                bool                   mGlobalQuit;                   ///     Set "true" to gracefully stop
                uint32_t               mVideoBufferSize;              ///     My video buffer size, in bytes
                uint32_t               mAudioBufferSize;              ///     My audio buffer size, in bytes
                AVDataBuffer           mAVHostBuffer [CIRCULAR_BUFFER_SIZE]; ///     My host buffers
                AUTOCIRCULATE_TRANSFER_STRUCT mInputTransferStruct;   ///     My A/C input transfer info
                AUTOCIRCULATE_TRANSFER_STATUS_STRUCT mInputTransferStatusStruct; ///     My A/C input status
                AJAThread             *mProducerThread;               ///     My producer thread object -- does the frame capturing
                bool                   mFrameCaptured;
                video_frame           *mOutputFrame;
                bool                   mProgressive;
                chrono::system_clock::time_point mT0;
                int                    mFrames;
                struct audio_frame     mAudio;
                int                    mMaxAudioChannels;

                AJAStatus SetupVideo();
                AJAStatus SetupAudio();
                void SetupHostBuffers();
                void RouteInputSignal();
                void SetupInputAutoCirculate();


                /**
                  @brief  Starts my frame consumer thread.
                 **/
                virtual void                    StartProducerThread (void);
                virtual void                    CaptureFrames (void);
                static void     ProducerThreadStatic (AJAThread * pThread, void * pContext);
};

;
vidcap_state_aja::vidcap_state_aja(unordered_map<string, string> const & parameters) :
        mDeviceIndex(0), mInputChannel(NTV2_CHANNEL1), mVideoFormat(NTV2_FORMAT_UNKNOWN),
        mPixelFormat(NTV2_FBF_8BIT_YCBCR), mInputSource(NTV2_INPUTSOURCE_SDI1),
        mAudioSystem(NTV2_AUDIOSYSTEM_1), mGlobalQuit(false),
        mProducerThread(0), mFrameCaptured(false), mOutputFrame(0), mProgressive(false),
        mT0(chrono::system_clock::now()), mFrames(0), mAudio(audio_frame()), mMaxAudioChannels(0)
{
        if (parameters.find("progressive") != parameters.end()) {
                mProgressive = true;
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
                throw 1;

        //      Using a reference to the discovered device list,
        //      and the index number of the device of interest (mDeviceIndex),
        //      get information about that particular device...
        NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [mDeviceIndex]);
        if (!mDevice.Open (info.boardNumber, false, info.boardType))
                throw 1;

        if (!mDevice.AcquireStreamForApplication (app, static_cast <uint32_t> (AJAProcess::GetPid ())))
                throw 1;

        mDevice.GetEveryFrameServices (&mSavedTaskMode);        //      Save the current state before we change it
        mDevice.SetEveryFrameServices (NTV2_OEM_TASKS);

        //      Keep the device ID handy as it will be used frequently...
        mDeviceID = mDevice.GetBoardID ();

        //      Sometimes other applications disable some or all of the frame buffers, so turn on ours no w...
        mDevice.EnableChannel (mInputChannel);

        //      Set up the video and audio...
        status = SetupVideo ();
        if (AJA_FAILURE (status))
                throw status;

        status = SetupAudio ();
        if (AJA_FAILURE (status))
                throw status;

        //      Set up the circular buffers, the device signal routing, and both playout and capture Auto Circulate...
        SetupHostBuffers ();
        RouteInputSignal ();
        SetupInputAutoCirculate ();

        //      This is for the timecode that we will burn onto the image...
        //NTV2FormatDescriptor fd = GetFormatDescriptor (GetNTV2StandardFromVideoFormat (mVideoFormat), mPixelFormat, mVancEnabled, Is2KFormat (mVideoFormat), mWideVanc);
}

vidcap_state_aja::~vidcap_state_aja() {
        delete mProducerThread;
        mProducerThread = NULL;

        //      Unsubscribe from input vertical event...
        mDevice.UnsubscribeInputVerticalEvent (mInputChannel);

        //      Free all my buffers...
        for (unsigned bufferNdx = 0; bufferNdx < CIRCULAR_BUFFER_SIZE; bufferNdx++)
        {
                if (mAVHostBuffer[bufferNdx].fVideoBuffer)
                {
                        delete mAVHostBuffer[bufferNdx].fVideoBuffer;
                        mAVHostBuffer[bufferNdx].fVideoBuffer = NULL;
                }
                if (mAVHostBuffer[bufferNdx].fAudioBuffer)
                {
                        delete mAVHostBuffer[bufferNdx].fAudioBuffer;
                        mAVHostBuffer[bufferNdx].fAudioBuffer = NULL;
                }
        }       //      for each buffer in the ring

        mDevice.SetEveryFrameServices (mSavedTaskMode);                                                                                                 //      Restore previous service level
        mDevice.ReleaseStreamForApplication (app, static_cast <uint32_t> (AJAProcess::GetPid ()));     //      Release the device

        vf_free(mOutputFrame);
        free(mAudio.data);
}

static const NTV2ReferenceSource        inputSrcToRefSrc [] = { NTV2_REFERENCE_INPUT1,
        NTV2_REFERENCE_ANALOG_INPUT, NTV2_REFERENCE_INPUT2, NTV2_REFERENCE_HDMI_INPUT,
        NTV2_NUM_REFERENCE_INPUTS, NTV2_NUM_REFERENCE_INPUTS, NTV2_REFERENCE_INPUT3,
        NTV2_REFERENCE_INPUT4, NTV2_NUM_REFERENCE_INPUTS, NTV2_NUM_REFERENCE_INPUTS,
        NTV2_NUM_REFERENCE_INPUTS, NTV2_NUM_REFERENCE_INPUTS, NTV2_NUM_REFERENCE_INPUTS,
        NTV2_NUM_REFERENCE_INPUTS, NTV2_REFERENCE_INPUT5, NTV2_REFERENCE_INPUT6, NTV2_REFERENCE_INPUT7,
        NTV2_REFERENCE_INPUT8, NTV2_NUM_REFERENCE_INPUTS
};

static const unordered_map<NTV2Standard, interlacing_t, hash<int>> interlacing_map = {
        { NTV2_STANDARD_525, INTERLACED_MERGED },
        { NTV2_STANDARD_625, INTERLACED_MERGED },
        { NTV2_STANDARD_1080, INTERLACED_MERGED }, // or SEGMENTED_FRAME
        { NTV2_STANDARD_720, PROGRESSIVE },
        { NTV2_STANDARD_1080p, PROGRESSIVE },
        { NTV2_STANDARD_2K, SEGMENTED_FRAME},
};

static const unordered_map<NTV2FrameBufferFormat, codec_t, hash<int>> codec_map = {
        { NTV2_FBF_10BIT_YCBCR, v210 },
        { NTV2_FBF_8BIT_YCBCR, UYVY },
        { NTV2_FBF_RGBA, RGBA },
        { NTV2_FBF_10BIT_RGB, R10k },
        { NTV2_FBF_8BIT_YCBCR_YUY2, YUYV },
        { NTV2_FBF_24BIT_RGB, RGB },
        { NTV2_FBF_24BIT_BGR, BGR },
};

AJAStatus vidcap_state_aja::SetupVideo()
{
        //      Set the video format to match the incomming video format.
        //      Does the device support the desired input source?
        if (!::NTV2BoardCanDoInputSource (mDeviceID, mInputSource))
                return AJA_STATUS_BAD_PARAM;    //      Nope

        //      Determine the input video signal format...
        mVideoFormat = mDevice.GetInputVideoFormat (mInputSource, mProgressive);
        //mVideoFormat = NTV2_FORMAT_4x1920x1080p_2500;
        //mVideoFormat = NTV2_FORMAT_1080p_5000_A;
        if (mVideoFormat == NTV2_FORMAT_UNKNOWN)
        {
                cerr << "## ERROR:  No input signal or unknown format" << endl;
                return AJA_STATUS_NOINPUT;      //      Sorry, can't handle this format
        }

        //      Set the device video format to whatever we detected at the input...
        mDevice.SetReferenceSource (inputSrcToRefSrc [mInputSource]);
        mDevice.SetVideoFormat (mVideoFormat, false, false, mInputChannel);

        //      Set the frame buffer pixel format for all the channels on the device
        //      (assuming it supports that pixel format -- otherwise default to 8-bit YCbCr)...
        if (!::NTV2BoardCanDoFrameBufferFormat (mDeviceID, mPixelFormat))
                mPixelFormat = NTV2_FBF_8BIT_YCBCR;

        mDevice.SetFrameBufferFormat (mInputChannel, mPixelFormat);

        //      Enable and subscribe to the interrupts for the channel to be used...
        mDevice.EnableInputInterrupt (mInputChannel);
        mDevice.SubscribeInputVerticalEvent (mInputChannel);

        NTV2Standard std = GetNTV2StandardFromVideoFormat(mVideoFormat);
        if (codec_map.find(mPixelFormat) == codec_map.end()) {
                cerr << "Cannot find valid mapping from AJA pixel format to UltraGrid" << endl;
                return AJA_STATUS_NOINPUT;
        }
        video_desc desc{GetDisplayWidth(mVideoFormat), GetDisplayHeight(mVideoFormat),
                codec_map.at(mPixelFormat),
                GetFramesPerSecond(GetNTV2FrameRateFromVideoFormat(mVideoFormat)),
                interlacing_map.find(std) == interlacing_map.end() ? PROGRESSIVE : interlacing_map.at(std),
                1};
        cout << "[AJA] Detected input video mode: " << desc << endl;
        mOutputFrame = vf_alloc_desc(desc);

        return AJA_STATUS_SUCCESS;
}

AJAStatus vidcap_state_aja::SetupAudio (void)
{
        //      Have the audio system capture audio from the designated device input...
        mDevice.SetAudioSystemInputSource (mAudioSystem, mInputSource);

        mMaxAudioChannels = ::NTV2BoardGetMaxAudioChannels (mDeviceID);
        mDevice.SetNumberAudioChannels (mMaxAudioChannels, mInputChannel);
        mDevice.SetAudioRate (NTV2_AUDIO_48K, mInputChannel);

        //      How big should the on-device audio buffer be?   1MB? 2MB? 4MB? 8MB?
        //      For this demo, 4MB will work best across all platforms (Windows, Mac & Linux)...
        mDevice.SetAudioBufferSize (NTV2_AUDIO_BUFFER_BIG, mInputChannel);

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

        return AJA_STATUS_SUCCESS;

}       //      SetupAudio

void vidcap_state_aja::SetupHostBuffers (void)
{
        //      Let my circular buffer know when it's time to quit...
        mAVCircularBuffer.SetAbortFlag (&mGlobalQuit);

        mVancEnabled = false;
        mWideVanc = false;
        mDevice.GetEnableVANCData (&mVancEnabled, &mWideVanc);
        mVideoBufferSize = GetVideoWriteSize (mVideoFormat, mPixelFormat, mVancEnabled, mWideVanc);
        mAudioBufferSize = NTV2_AUDIOSIZE_MAX;

        //      Allocate and add each in-host AVDataBuffer to my circular buffer member variable...
        for (unsigned bufferNdx = 0; bufferNdx < CIRCULAR_BUFFER_SIZE; bufferNdx++ )
        {
                mAVHostBuffer [bufferNdx].fVideoBuffer = reinterpret_cast <uint32_t *> (new uint8_t [mVideoBufferSize]);
                mAVHostBuffer [bufferNdx].fVideoBufferSize = mVideoBufferSize;
                mAVHostBuffer [bufferNdx].fAudioBuffer = reinterpret_cast <uint32_t *> (new uint8_t [mAudioBufferSize]);
                mAVHostBuffer [bufferNdx].fAudioBufferSize = mAudioBufferSize;
                mAVCircularBuffer.Add (& mAVHostBuffer [bufferNdx]);
        }       //      for each AVDataBuffer

}       //      SetupHostBuffers

void vidcap_state_aja::RouteInputSignal (void)
{
        //      For this simple example, tie the user-selected input to frame buffer 1.
        //      Is this user-selected input supported on the device?
        if(!::NTV2BoardCanDoInputSource (mDeviceID, mInputSource))
                mInputSource = NTV2_INPUTSOURCE_SDI1;

        ULWord  inputIdentifier (NTV2_XptSDIIn1);
        switch (mInputSource)
        {
                case NTV2_INPUTSOURCE_SDI2: inputIdentifier = NTV2_XptSDIIn2; break;
                case NTV2_INPUTSOURCE_SDI3: inputIdentifier = NTV2_XptSDIIn3; break;
                case NTV2_INPUTSOURCE_SDI4: inputIdentifier = NTV2_XptSDIIn4; break;
                case NTV2_INPUTSOURCE_HDMI: inputIdentifier = NTV2_XptHDMIIn; break;
                case NTV2_INPUTSOURCE_ANALOG: inputIdentifier = NTV2_XptAnalogIn; break;
                default: break;
        }

        //      Use a "Routing" object, which handles the details of writing
        //      the appropriate values into the appropriate device registers...
        CNTV2SignalRouter       router;
        if (IsRGBFormat (mPixelFormat))
        {
                //      If the frame buffer is configured for RGB pixel format, incoming YUV must be converted.
                //      This routes the video signal from the input through a color space converter before
                //      connecting to the RGB frame buffer...
                router.addWithValue (::GetCSC1VidInputSelectEntry (), inputIdentifier);
                router.addWithValue (::GetFrameBuffer1InputSelectEntry(), NTV2_XptCSC1VidRGB);
        }
        else
        {
                //      This routes the YCbCr signal directly from the input to the frame buffer...
                router.addWithValue (::GetFrameBuffer1InputSelectEntry(), inputIdentifier);
        }
        //      Disable SDI output from the SDI input being used,
        //      but only if the device supports bi-directional SDI,
        //      and only if the input being used is an SDI input...
        if (::NTV2BoardHasBiDirectionalSDI (mDeviceID)
                        && mInputSource != NTV2_INPUTSOURCE_HDMI && mInputSource != NTV2_INPUTSOURCE_ANALOG)
                switch (mInputSource)
                {
                        case NTV2_INPUTSOURCE_SDI1:     mDevice.SetSDITransmitEnable (NTV2_CHANNEL1, false);    break;
                        case NTV2_INPUTSOURCE_SDI2:     mDevice.SetSDITransmitEnable (NTV2_CHANNEL2, false);    break;
                        case NTV2_INPUTSOURCE_SDI3:     mDevice.SetSDITransmitEnable (NTV2_CHANNEL3, false);    break;
                        case NTV2_INPUTSOURCE_SDI4:     mDevice.SetSDITransmitEnable (NTV2_CHANNEL4, false);    break;
                        default:                                                                                                                                                        break;
                }

        //      Replace the device's current signal routing with this new one...
        mDevice.ApplySignalRoute (router, true);

}       //      RouteInputSignal

static NTV2Crosspoint   channelToInputCrosspoint []     = {     NTV2CROSSPOINT_INPUT1,  NTV2CROSSPOINT_INPUT2,  NTV2CROSSPOINT_INPUT3,  NTV2CROSSPOINT_INPUT4,
        NTV2CROSSPOINT_INPUT5,  NTV2CROSSPOINT_INPUT6,  NTV2CROSSPOINT_INPUT7,  NTV2CROSSPOINT_INPUT8,  NTV2_NUM_CROSSPOINTS};


void vidcap_state_aja::SetupInputAutoCirculate (void)
{
        mDevice.StopAutoCirculate (NTV2CROSSPOINT_INPUT1);

        ::memset (&mInputTransferStruct,                0,      sizeof (mInputTransferStruct));
        ::memset (&mInputTransferStatusStruct,  0,      sizeof (mInputTransferStatusStruct));

        mInputTransferStruct.channelSpec                        = channelToInputCrosspoint [mInputChannel];
        mInputTransferStruct.videoBufferSize            = mVideoBufferSize;
        mInputTransferStruct.videoDmaOffset                     = 0;
        mInputTransferStruct.audioBufferSize            = mAudioBufferSize;
        mInputTransferStruct.frameRepeatCount           = 1;
        mInputTransferStruct.desiredFrame                       = -1;
        mInputTransferStruct.frameBufferFormat          = mPixelFormat;
        mInputTransferStruct.bDisableExtraAudioInfo     = true;

        //      Tell capture AutoCirculate to use frame buffers 0-9 on the device...
        const uint32_t  startFrame      (0);
        const uint32_t  endFrame        (9);

        mDevice.InitAutoCirculate (mInputTransferStruct.channelSpec, startFrame, endFrame,
                        1,                                              //      Number of channels
                        NTV2_AUDIOSYSTEM_1,             //      Which audio system?
                        true,                                   //      With audio?
                        true,                                   //      With RP188?
                        false,                                  //      Allow frame buffer format changes?
                        false,                                  //      With color correction?
                        false,                                  //      With vidProc?
                        false,                                  //      With custom ANC data?
                        false,                                  //      With LTC?
                        false);                                 //      With audio2?
}       //      SetupInputAutoCirculate

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
        if (mFrameCaptured) {
                mAVCircularBuffer.EndConsumeNextBuffer ();
        }
        //      Set the global 'quit' flag, and wait for the threads to go inactive...
        mGlobalQuit = true;

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
        //      Start AutoCirculate running...
        mDevice.StartAutoCirculate (mInputTransferStruct.channelSpec);

        while (!mGlobalQuit)
        {
                AUTOCIRCULATE_STATUS_STRUCT     acStatus;
                mDevice.GetAutoCirculate (NTV2CROSSPOINT_INPUT1, &acStatus);

                if (acStatus.state == NTV2_AUTOCIRCULATE_RUNNING && acStatus.bufferLevel > 1)
                {
                        //      At this point, there's at least one fully-formed frame available in the device's
                        //      frame buffer to transfer to the host. Reserve an AVDataBuffer to "produce", and
                        //      use it in the next transfer from the device...
                        AVDataBuffer *  captureData     (mAVCircularBuffer.StartProduceNextBuffer ());
                        if (captureData == nullptr)
                                continue;

                        mInputTransferStruct.videoBuffer                = captureData->fVideoBuffer;
                        mInputTransferStruct.videoBufferSize    = captureData->fVideoBufferSize;
                        mInputTransferStruct.audioBuffer                = captureData->fAudioBuffer;
                        mInputTransferStruct.audioBufferSize    = captureData->fAudioBufferSize;

                        //      Do the transfer from the device into our host AVDataBuffer...
                        mDevice.TransferWithAutoCirculate (&mInputTransferStruct, &mInputTransferStatusStruct);
                        captureData->fAudioBufferSize = mInputTransferStatusStruct.audioBufferSize;

                        //      "Capture" timecode into the host AVDataBuffer while we have full access to it...
                        captureData->fRP188Data = mInputTransferStatusStruct.frameStamp.currentRP188;

                        //      Signal that we're done "producing" the frame, making it available for future "consumption"...
                        mAVCircularBuffer.EndProduceNextBuffer ();
                }       //      if A/C running and frame(s) are available for transfer
                else
                {
                        //      Either AutoCirculate is not running, or there were no frames available on the device to transfer.
                        //      Rather than waste CPU cycles spinning, waiting until a frame becomes available, it's far more
                        //      efficient to wait for the next input vertical interrupt event to get signaled...
                        mDevice.WaitForInputVerticalInterrupt (mInputChannel);
                }
        }       //      loop til quit signaled

        //      Stop AutoCirculate...
        mDevice.StopAutoCirculate (mInputTransferStruct.channelSpec);

}       //      CaptureFrames

#define NTV2_AUDIOSIZE_48K (48 * 1024)

struct video_frame *vidcap_state_aja::grab(struct audio_frame **audio)
{
        if (mFrameCaptured) {
                mAVCircularBuffer.EndConsumeNextBuffer ();
        }
        if (!mGlobalQuit)
        {
                //      Wait for the next frame to become ready to "consume"...
                AVDataBuffer *  playData        (mAVCircularBuffer.StartConsumeNextBuffer ());
                if (playData)
                {
                        mOutputFrame->tiles[0].data = (char *) playData->fVideoBuffer;
                        mFrameCaptured = true;
                        mFrames += 1;

                        for (unsigned int i = 0; i < audio_capture_channels; i++) {
                                remux_channel(mAudio.data, (char *) playData->fAudioBuffer, mAudio.bps,
                                                playData->fAudioBufferSize, mMaxAudioChannels,
                                                mAudio.ch_count, i, i);
                        }
                        mAudio.data_len = playData->fAudioBufferSize / mMaxAudioChannels *
                                audio_capture_channels;
                        *audio = &mAudio;

                        chrono::system_clock::time_point now = chrono::system_clock::now();
                        double seconds = chrono::duration_cast<chrono::microseconds>(now - mT0).count() / 1000000.0;
                        if (seconds >= 5) {
                                LOG(LOG_LEVEL_INFO) << "[AJA] " << mFrames << " frames in "
                                        << seconds << " seconds = " <<  mFrames / seconds << " FPS\n";
                                mT0 = now;
                                mFrames = 0;
                        }

                        return mOutputFrame;
                }
        }       //      loop til quit signaled

        return NULL;
}

void *vidcap_aja_init(const struct vidcap_params *params)
{
        unordered_map<string, string> parameters_map;
        char *tmp = strdup(vidcap_params_get_fmt(params));
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
        } catch (...) {
                delete ret;
                return NULL;
        }
        return ret;
}

void vidcap_aja_done(void *state)
{
        auto s = static_cast<vidcap_state_aja *>(state);
        s->Quit();
        delete s;
}

struct video_frame *vidcap_aja_grab(void *state, struct audio_frame **audio)
{
        return ((vidcap_state_aja *) state)->grab(audio);
}

struct vidcap_type *vidcap_aja_probe(bool)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = 0x52095a04;
                vt->name = "aja";
                vt->description = "AJA capture card";
        }
        return vt;
}


/**
 * @file   video_display/aja.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2019 CESNET, z. s. p. o.
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
/*
 * Internal structure heavily inspired by and much code taken from NTV2Player Demo.
 * The use of ping-pong buffer technique is based on NTV2LLBurn.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "config_msvc.h"

#include <ajantv2/includes/ntv2utils.h>
#include <ajatypes.h>
#include <ntv2debug.h>
#include <ntv2democommon.h>
#include <ntv2devicescanner.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video_display.h"
#include "rang.hpp"
#include "video.h"

#include "aja_common.h" // should be included last (overrides log_msg etc.)

#define DEFAULT_MAX_FRAME_QUEUE_LEN 1
#define MODULE_NAME "[AJA display] "
/**
 * The maximum number of bytes of 48KHz audio that can be transferred for two frames.
 * Worst case, assuming 16 channels of audio (max), 4 bytes per sample, and 67 msec per frame
 * (assuming the lowest possible frame rate of 14.98 fps)...
 * 48,000 samples per second requires 6,408 samples x 4 bytes/sample x 16 = 410,112 bytes
 * 402K will suffice, with 1536 bytes to spare
 */
#define BPS                     4
#define SAMPLE_RATE             48000

#ifdef _MSC_VER
extern "C" __declspec(dllexport) int *aja_display_init_noerr;
int *aja_display_init_noerr;
#else
int *aja_display_init_noerr = &display_init_noerr;
#endif

#ifdef _MSC_VER
#define LINK_SPEC extern "C" __declspec(dllexport)
#else
#define LINK_SPEC static
#endif

using std::cerr;
using std::chrono::duration;
using std::chrono::steady_clock;
using std::condition_variable;
using std::cout;
using std::endl;
using std::hash;
using std::lock_guard;
using std::min;
using std::mutex;
using std::ostringstream;
using std::queue;
using std::runtime_error;
using std::string;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

#define CHECK_EX(cmd, msg, action_failed) do { bool ret = cmd; if (!ret) {\
        LOG(LOG_LEVEL_WARNING) << MODULE_NAME << (msg) << "\n";\
        action_failed;\
}\
} while(0)
#define NOOP ((void)0)
#define CHECK(cmd) CHECK_EX(cmd, #cmd " failed", NOOP)

namespace ultragrid {
namespace aja {

struct display {
        const struct configuration {
                string deviceId{"0"};
                NTV2OutputDestination outputDestination = NTV2_OUTPUTDESTINATION_SDI1;
                NTV2Channel outputChannel = NTV2_CHANNEL_INVALID; ///< if invalid, select according to output destination
                bool withAudio = false;
                bool novsync = false;
                int bufLen = DEFAULT_MAX_FRAME_QUEUE_LEN;
                bool clearRouting = false;
                int doMultiChannel = -1; /* -1/0/1 */
                codec_t forceOutputColorSpace = VIDEO_CODEC_NONE; ///< force output color space for SDI - RGB or UYVY, VIDEO_CODEC_NONE for default
                bool setupAll = true;
                bool setupRoute = true;
                bool smpteRange = true;
        } mConf;

        queue<struct video_frame *> frames;
        unsigned int max_frame_queue_len;
        mutex frames_lock;
        condition_variable frame_ready;
        thread worker;
        void join();
        void process_frames();

        bool mOutIsRGB = false;
        static const ULWord app = AJA_FOURCC ('U','L','G','R');

        CNTV2Card mDevice;
        NTV2DeviceID mDeviceID;
        struct video_desc desc{};
        bool mDoMultiChannel; ///< Use multi-format
        NTV2EveryFrameTaskMode mSavedTaskMode = NTV2_TASK_MODE_INVALID; ///< Used to restore the prior task mode
        NTV2Channel mOutputChannel;
        NTV2VideoFormat mVideoFormat = NTV2_FORMAT_UNKNOWN;
        NTV2FrameBufferFormat mPixelFormat = NTV2_FBF_INVALID;
        bool mDoLevelConversion = false;
        bool mEnableVanc = false;

        unique_ptr<char[]> mAudioBuffer {new char[NTV2_AUDIOSIZE_MAX]};
        size_t mAudioLen = 0;
        NTV2AudioSystem mAudioSystem = NTV2_AUDIOSYSTEM_1;
        mutex mAudioLock;
        bool mAudioIsReset = false;
        uint32_t mAudioOutWrapAddress = 0u;
        uint32_t mAudioOutLastAddress = 0u;

        uint32_t mCurrentOutFrame;
        uint32_t mFramesProcessed = 0u;
        uint32_t mFramesDropped = 0u;
        uint32_t mFramesDiscarded = 0u;

        steady_clock::time_point mT0 = steady_clock::now();
        int mFrames = 0;

public:
        display(struct configuration &configuration);
        ~display();
        void Init();
        AJAStatus SetUpVideo();
        AJAStatus SetUpAudio();
        void RouteOutputSignal();
        int Putf(struct video_frame *frame, int nonblock);

        static NTV2FrameRate getFrameRate(double fps);
        void print_stats();
        static void show_help();
};

display::display(struct configuration &conf)
        : mConf(conf), max_frame_queue_len(conf.bufLen), mOutputChannel(conf.outputChannel)
{
        if (!CNTV2DeviceScanner::GetFirstDeviceFromArgument(conf.deviceId, mDevice)) {
                throw runtime_error(string("Device '") + conf.deviceId + "' not found!");
        }

        if (!mDevice.IsDeviceReady(false)) {
                throw runtime_error(string("Device '") + conf.deviceId + "' not ready!");
        }

        if (conf.clearRouting) {
                CHECK(mDevice.ClearRouting());
        }

        mDeviceID = mDevice.GetDeviceID(); // Keep this ID handy -- it's used frequently

        bool canDoMultiChannel = NTV2DeviceCanDoMultiFormat(mDeviceID);
        if (!canDoMultiChannel) {
                if (conf.doMultiChannel != 0) {
                        LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Device " << conf.deviceId << " cannot simultaneously handle different video formats.\n";
                }
                mDoMultiChannel = false;
        } else {
                mDoMultiChannel = conf.doMultiChannel != 0;
        }
        if (!mDoMultiChannel) {
                if (!mDevice.AcquireStreamForApplication(app, static_cast <uint32_t> (getpid()))) {
                        throw runtime_error("Device busy!\n"); // Device is in use by another app -- fail
                }
                CHECK(mDevice.GetEveryFrameServices (mSavedTaskMode)); // Save the current service level
        }

        CHECK(mDevice.SetEveryFrameServices(NTV2_OEM_TASKS));

        if (::NTV2DeviceCanDoMultiFormat(mDeviceID) && mDoMultiChannel) {
                CHECK(mDevice.SetMultiFormatMode(true));
        } else if (::NTV2DeviceCanDoMultiFormat(mDeviceID)) {
                CHECK(mDevice.SetMultiFormatMode (false));
        }

        if (mOutputChannel == NTV2_CHANNEL_INVALID) {
                if (!NTV2_OUTPUT_DEST_IS_SDI(mConf.outputDestination)) {
                        LOG(LOG_LEVEL_NOTICE) << MODULE_NAME "Non-SDI destination detected - we will use "
                                "probably channel 1. Consider passing \"channel\" option (see help).\n";
                }

                mOutputChannel = ::NTV2OutputDestinationToChannel(mConf.outputDestination);

                //      Beware -- some devices (e.g. Corvid1) can only output from FrameStore 2...
                if ((mOutputChannel == NTV2_CHANNEL1) && (!::NTV2DeviceCanDoFrameStore1Display (mDeviceID))) {
                        mOutputChannel = NTV2_CHANNEL2;
                }
        }
        if (UWord (mOutputChannel) >= ::NTV2DeviceGetNumFrameStores (mDeviceID)) {
                ostringstream oss;
                oss    << "## ERROR:  Cannot use channel '" << mOutputChannel+1 << "' -- device only supports channel 1"
                        << (::NTV2DeviceGetNumFrameStores (mDeviceID) > 1  ?  string (" thru ") + string (1, uint8_t (::NTV2DeviceGetNumFrameStores (mDeviceID)+'0'))  :  "");
                throw runtime_error(oss.str());
        }

        mCurrentOutFrame = mOutputChannel * 2u;
}

display::~display() {
        CHECK(mDevice.UnsubscribeOutputVerticalEvent (mOutputChannel));

        if (!mDoMultiChannel) {
                CHECK_EX(mDevice.SetEveryFrameServices (mSavedTaskMode), "Restore Service Mode", NOOP); // Restore the previously saved service level
                CHECK(mDevice.ReleaseStreamForApplication (app, static_cast <uint32_t>(getpid())));     // Release the device
        }

        //      Don't leave the audio system active after we exit
        CHECK_EX(mDevice.StopAudioOutput(mAudioSystem), "Restore Audio", NOOP);
}

void display::Init()
{
        AJAStatus status;
        status = SetUpVideo();
        if (AJA_FAILURE(status)) {
                ostringstream oss;
                oss << "Unable to initialize video: " << ::AJAStatusToString(status);
                throw runtime_error(oss.str());
        }
        RouteOutputSignal();

        //      Before the main loop starts, ping-pong the buffers so the hardware will use
        //      different buffers than the ones it was using while idling...
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                mCurrentOutFrame ^= 1;
                CHECK(mDevice.SetOutputFrame(chan, mCurrentOutFrame + 2 * i));

                mCurrentOutFrame ^= 1;
                CHECK(mDevice.SetOutputFrame(chan, mCurrentOutFrame + 2 * i));
        }
}

AJAStatus display::SetUpVideo ()
{
        if (!mConf.setupAll) {
                return AJA_STATUS_SUCCESS;
        }

        if (!::NTV2DeviceCanDoVideoFormat (mDeviceID, mVideoFormat)) {
                cerr << "## ERROR:  This device cannot handle '" << ::NTV2VideoFormatToString (mVideoFormat) << "'" << endl;
                return AJA_STATUS_UNSUPPORTED;
        }

        //      Configure the device to handle the requested video format...
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                if (!mDevice.SetVideoFormat (mVideoFormat, false, false, chan)) {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Cannot set format "
                                << NTV2VideoFormatToString(mVideoFormat)
                                << " for output " << chan << "\n";
                        return AJA_STATUS_FAIL;
                }
        }

        mOutIsRGB = !mConf.forceOutputColorSpace ? (NTV2_OUTPUT_DEST_IS_HDMI(mConf.outputDestination) ? true : ::IsRGBFormat(mPixelFormat)) : mConf.forceOutputColorSpace == RGB;
        //      If device has no RGB conversion capability for the desired channel, use FBF instead
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                if (UWord (mOutputChannel) + UWord(i) > ::NTV2DeviceGetNumCSCs (mDeviceID)) {
                        if (mConf.forceOutputColorSpace && mOutIsRGB != ::IsRGBFormat(mPixelFormat)) {
                                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Not enough CSCs found, found " << ::NTV2DeviceGetNumCSCs (mDeviceID) << " CSCs, "
                                        "overriding output color spec preference.\n";
                        }
                        mOutIsRGB = ::IsRGBFormat(mPixelFormat);
                }
        }

        if (!::NTV2DeviceCanDo3GLevelConversion (mDeviceID) && mDoLevelConversion && ::IsVideoFormatA (mVideoFormat)) {
                mDoLevelConversion = false;
        }
        if (mDoLevelConversion) {
                CHECK(mDevice.SetSDIOutLevelAtoLevelBConversion (mOutputChannel, mDoLevelConversion));
        }

        //      Set the frame buffer pixel format for all the channels on the device.
        //      If the device doesn't support it, fall back to 8-bit YCbCr...
        if (!::NTV2DeviceCanDoFrameBufferFormat (mDeviceID, mPixelFormat))
        {
                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Device cannot handle '" << ::NTV2FrameBufferFormatString (mPixelFormat) << "'\n";
                return AJA_STATUS_FAIL;
        }

        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                if (!mDevice.SetFrameBufferFormat (chan, mPixelFormat)) {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Cannot set format "
                                << NTV2FrameBufferFormatString(mPixelFormat)
                                << " for output " << mOutputChannel + i << "\n";
                        return AJA_STATUS_FAIL;
                }
        }
        if(mDeviceID == DEVICE_ID_KONAIP_1RX_1TX_2110 ||
                        mDeviceID == DEVICE_ID_KONAIP_2110) {
                CHECK_EX(mDevice.SetReference(NTV2_REFERENCE_SFP1_PTP), "Set Reference SFP1_PTP", NOOP);
        } else {
                CHECK_EX(mDevice.SetReference (NTV2_REFERENCE_FREERUN), "Set Reference FREERUN", NOOP);
        }
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                if (!mDevice.EnableChannel(chan)) {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Cannot enable channel "
                                <<  chan << "\n";
                        return AJA_STATUS_FAIL;
                }
        }

        if (mEnableVanc && !::IsRGBFormat (mPixelFormat) && NTV2_IS_HD_VIDEO_FORMAT (mVideoFormat))
        {
                //      Try enabling VANC...
                CHECK_EX(mDevice.SetEnableVANCData(true), "SetEnableVANCData true", NOOP); // Enable VANC for non-SD formats, to pass thru captions, etc.
                if (::Is8BitFrameBufferFormat (mPixelFormat)) {
                        //      8-bit FBFs require VANC bit shift...
                        CHECK_EX(mDevice.SetVANCShiftMode(mOutputChannel, NTV2_VANCDATA_8BITSHIFT_ENABLE), "SetVANCShiftMode", NOOP);
                }
        } else {
                CHECK_EX(mDevice.SetEnableVANCData(false), "SetEnableVANCData false", NOOP);;      //      No VANC with RGB pixel formats (for now)
        }

        if (NTV2_OUTPUT_DEST_IS_HDMI(mConf.outputDestination)) {
                // convert all to RGB SMPTE range
                if (IsRGBFormat (mPixelFormat) && mOutIsRGB && mConf.smpteRange) { // set LUT
                        NTV2LutType lutType = NTV2_LUTRGBRangeFull_SMPTE;
                        const int bank = 1; // Bank 0 (RGB->YUV, SMPTE->Full), Bank 1 (YUV->RGB, Full->SMPTE)
                        CHECK_EX(mDevice.SetColorCorrectionOutputBank(mOutputChannel, bank), "LUT Bank", NOOP);
                        CHECK_EX(mDevice.WriteRegister(kVRegLUTType, (ULWord) lutType), "LUT Bank Reg", NOOP);
                        NTV2DoubleArray table(1024);
                        CHECK_EX(mDevice.GenerateGammaTable(lutType, bank, table), "Generate Gamma", NOOP);
                        CHECK_EX(mDevice.DownloadLUTToHW(table, table, table, mOutputChannel, bank), "Download LUT", NOOP);
                }
                if (IsRGBFormat (mPixelFormat) != mOutIsRGB) { // set CSC
			CHECK(mDevice.SetColorSpaceMatrixSelect(NTV2_Rec709Matrix, mOutputChannel));
			CHECK(mDevice.SetColorSpaceRGBBlackRange(NTV2_CSC_RGB_RANGE_SMPTE, mOutputChannel));
			CHECK(mDevice.SetColorSpaceMethod(NTV2_CSC_Method_Original, mOutputChannel));
			CHECK(mDevice.SetColorSpaceMakeAlphaFromKey(false, mOutputChannel));
                        CHECK_EX(mDevice.SetColorSpaceRGBBlackRange(NTV2_CSC_RGB_RANGE_SMPTE, mOutputChannel), "CSC RGB Range", NOOP);
                }

                CHECK_EX(mDevice.SetHDMIOutProtocol(NTV2_HDMIProtocolHDMI), "HDMI Output Protocol", NOOP);
                CHECK_EX(mDevice.SetHDMIOutSampleStructure(mOutIsRGB ? NTV2_HDMI_RGB : NTV2_HDMI_YC422),
                                "HDMI Sample Struct", NOOP);
                CHECK_EX(mDevice.SetHDMIOutRange(NTV2_HDMIRangeSMPTE), "HDMI Range", NOOP);
                CHECK_EX(mDevice.SetHDMIOutColorSpace(mOutIsRGB ? NTV2_HDMIColorSpaceRGB : NTV2_HDMIColorSpaceYCbCr),
                                "HDMI Color Space", NOOP);
                //CHECK_EX(mDevice.SetHDMIOutForceConfig(true), "HDMI Force Config", NOOP);
                //CHECK(mDevice.SetHDMIOutPrefer420(false));
        }

        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                //	Enable and subscribe to the output interrupts (though it's enabled by default)...
                CHECK(mDevice.EnableOutputInterrupt(chan));
                CHECK(mDevice.SubscribeOutputVerticalEvent(chan));
                //	Set the Frame Store modes
                CHECK(mDevice.SetMode(chan, NTV2_MODE_DISPLAY));
        }

        //      Subscribe the output interrupt -- it's enabled by default...

        return AJA_STATUS_SUCCESS;

}       //      SetUpVideo

AJAStatus display::SetUpAudio ()
{
        if (!mConf.setupAll) {
                return AJA_STATUS_SUCCESS;
        }

        mAudioSystem = NTV2ChannelToAudioSystem(mOutputChannel);
        CHECK(mDevice.StartAudioOutput(mAudioSystem));

        //      It's best to use all available audio channels...
        CHECK_EX(mDevice.SetNumberAudioChannels(::NTV2DeviceGetMaxAudioChannels(mDeviceID), mAudioSystem), "Unable to set audio channels!", return AJA_STATUS_FAIL);

        //      Assume 48kHz PCM...
        CHECK(mDevice.SetAudioRate (NTV2_AUDIO_48K, mAudioSystem));

        //      4MB device audio buffers work best...
        CHECK(mDevice.SetAudioBufferSize(NTV2_AUDIO_BUFFER_BIG, mAudioSystem));

        //      Set the SDI output audio embedders to embed audio samples from the output of mAudioSystem...
        if (NTV2_OUTPUT_DEST_IS_SDI(mConf.outputDestination)) {
                CHECK_EX(mDevice.SetSDIOutputAudioSystem(NTV2OutputDestinationToChannel(mConf.outputDestination), mAudioSystem), "Unable to set SDI output audio system!", return AJA_STATUS_FAIL);
                CHECK_EX(mDevice.SetSDIOutputDS2AudioSystem(NTV2OutputDestinationToChannel(mConf.outputDestination), mAudioSystem), "Unable to set SDI output audio system!", return AJA_STATUS_FAIL);
        } else {
                CHECK(mDevice.SetHDMIOutAudioChannels(NTV2_HDMIAudio8Channels));
                CHECK_EX(mDevice.SetHDMIOutAudioSource8Channel(NTV2_AudioChannel1_8, mAudioSystem), "Unable to set HDMI output audio system!", return AJA_STATUS_FAIL);
        }

        //
        //      Loopback mode plays whatever audio appears in the input signal when it's
        //      connected directly to an output (i.e., "end-to-end" mode). If loopback is
        //      left enabled, the video will lag the audio as video frames get briefly delayed
        //      in our ring buffer. Audio, therefore, needs to come out of the (buffered) frame
        //      data being played, so loopback must be turned off...
        //
        CHECK_EX(mDevice.SetAudioLoopBack (NTV2_AUDIO_LOOPBACK_OFF, mAudioSystem), "Disable Audio Loopback", NOOP);

        CHECK_EX(mDevice.SetAudioOutputEraseMode(mAudioSystem, true), "Set erase mode", return AJA_STATUS_FAIL);

        //      Reset both the input and output sides of the audio system so that the buffer
        //      pointers are reset to zero and inhibited from advancing.
        CHECK(mDevice.StopAudioOutput(mAudioSystem));
        mAudioIsReset = true;

        CHECK(mDevice.GetAudioWrapAddress(mAudioOutWrapAddress, mAudioSystem));
        mAudioOutLastAddress = 0;

        return AJA_STATUS_SUCCESS;

}       //      SetupAudio

const unordered_map<NTV2Channel, NTV2OutputCrosspointID, hash<int>> chanToLutSrc = {
        {NTV2_CHANNEL1, NTV2_XptLUT1RGB},
        {NTV2_CHANNEL2, NTV2_XptLUT2RGB},
        {NTV2_CHANNEL3, NTV2_XptLUT3Out},
        {NTV2_CHANNEL4, NTV2_XptLUT4Out},
        {NTV2_CHANNEL5, NTV2_XptLUT5Out},
        {NTV2_CHANNEL6, NTV2_XptLUT6Out},
        {NTV2_CHANNEL7, NTV2_XptLUT7Out},
        {NTV2_CHANNEL8, NTV2_XptLUT8Out}
};

void display::RouteOutputSignal ()
{
        if (!mConf.setupRoute) {
                return;
        }

        const NTV2Standard              outputStandard  (::GetNTV2StandardFromVideoFormat (mVideoFormat));
        const UWord                     numSDIOutputs (::NTV2DeviceGetNumVideoOutputs (mDeviceID));
        bool                            fbIsRGB                   (::IsRGBFormat (mPixelFormat));

        if (mDoMultiChannel) {
                //	Multiformat --- We may be sharing the device with other processes, so route only one SDI output...
                for (unsigned int i = 0; i < desc.tile_count; ++i) {
                        //      Multiformat --- route the one output to the CSC video output (RGB) or FrameStore output (YUV)...
                        NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                        NTV2OutputCrosspointID  cscVidOutXpt    (::GetCSCOutputXptFromChannel (chan,  false/*isKey*/,  !fbIsRGB/*isRGB*/));
                        NTV2OutputCrosspointID  fsVidOutXpt             (::GetFrameBufferOutputXptFromChannel (chan,  fbIsRGB/*isRGB*/,  false/*is425*/));
                        if (fbIsRGB != mOutIsRGB) {
                                CHECK_EX(mDevice.Connect(::GetCSCInputXptFromChannel (chan, false/*isKeyInput*/), fsVidOutXpt),
                                                "Connnect to CSC", NOOP);
                        }

                        if (NTV2_OUTPUT_DEST_IS_SDI(mConf.outputDestination)) {
                                if (::NTV2DeviceHasBiDirectionalSDI (mDeviceID)) {
                                        CHECK(mDevice.SetSDITransmitEnable(chan, true));
                                }

                                CHECK(mDevice.SetSDIOutputStandard(chan, outputStandard));
                                //mDevice.Connect (::GetSDIOutputInputXpt (chan, false/*isDS2*/),  fbIsRGB ? cscVidOutXpt : fsVidOutXpt);
                                if (mOutIsRGB) {
                                        CHECK(mDevice.Connect (::GetDLOutInputXptFromChannel(chan), fbIsRGB ? ::GetFrameBufferOutputXptFromChannel(chan, NTV2_IS_FBF_RGB(mPixelFormat)) : cscVidOutXpt)); // DLOut <== FBRGB/CSC
                                        CHECK(mDevice.Connect (::GetOutputDestInputXpt(mConf.outputDestination, false), ::GetDLOutOutputXptFromChannel(chan, false))); // SDIOut <== DLOut
                                        CHECK(mDevice.Connect (::GetOutputDestInputXpt(mConf.outputDestination, true),  ::GetDLOutOutputXptFromChannel(chan, true)));  // SDIOutDS <== DLOutDS
                                } else {
                                        CHECK_EX(mDevice.Connect(::GetOutputDestInputXpt(mConf.outputDestination), fbIsRGB ? cscVidOutXpt : fsVidOutXpt),
                                                        "Connect to CSC", NOOP);
                                }
                        } else if (NTV2_OUTPUT_DEST_IS_HDMI(mConf.outputDestination)) {
				// convert all to RGB SMPTE range
                                if (fbIsRGB && mOutIsRGB) { // connect to LUT to convert full->SMPTE range
                                        if (mConf.smpteRange) {
                                                auto lutInXpt = (NTV2InputCrosspointID) ((unsigned int) NTV2_XptLUT1Input + (unsigned int) chan);
                                                CHECK_EX(mDevice.Connect(lutInXpt, fsVidOutXpt), "Connect to LUT", NOOP);
                                                CHECK_EX(mDevice.Connect(::GetOutputDestInputXpt(mConf.outputDestination), chanToLutSrc.at(chan)),
                                                                "Connect from LUT", NOOP);
                                        } else {
                                                CHECK_EX(mDevice.Connect(::GetOutputDestInputXpt(mConf.outputDestination), fsVidOutXpt), "Connect to FS to HDMI", NOOP);
                                        }
                                } else { // connect output to FB or CSC
                                        CHECK_EX(mDevice.Connect(::GetOutputDestInputXpt(mConf.outputDestination), fbIsRGB != mOutIsRGB ? cscVidOutXpt : fsVidOutXpt),
                                                        "Connect from CSC", NOOP);
                                }
                        } else {
                                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Routing for " << NTV2OutputDestinationToString(mConf.outputDestination)
                                       << " may be incorrect. Please report to " PACKAGE_BUGREPORT ".\n" << endl;
                                CHECK_EX(mDevice.Connect(::GetOutputDestInputXpt(mConf.outputDestination), fbIsRGB ? cscVidOutXpt : fsVidOutXpt),
                                                "Connect from CSC or frame store", NOOP);
                        }
                }
        } else { //	Not multiformat:  We own the whole device, so connect all possible SDI outputs...
                 //     Route all possible SDI outputs to CSC video output (RGB) or FrameStore output (YUV)...
                const NTV2OutputCrosspointID cscVidOutXpt(::GetCSCOutputXptFromChannel (mOutputChannel,  false/*isKey*/,  !fbIsRGB/*isRGB*/));
                const NTV2OutputCrosspointID fsVidOutXpt(::GetFrameBufferOutputXptFromChannel (mOutputChannel,  fbIsRGB/*isRGB*/,  false/*is425*/));
		const UWord	numFrameStores(::NTV2DeviceGetNumFrameStores(mDeviceID));
                CHECK(mDevice.ClearRouting());		//	Start with clean slate

                CHECK(mDevice.Connect (::GetCSCInputXptFromChannel (mOutputChannel, false/*isKeyInput*/),  fsVidOutXpt));

                for (NTV2Channel chan(NTV2_CHANNEL1);  ULWord(chan) < numSDIOutputs;  chan = NTV2Channel(chan+1))
                {
                        if (chan != mOutputChannel  &&  chan < numFrameStores)
                                CHECK(mDevice.DisableChannel(chan));              // Disable unused FrameStore
                        if (::NTV2DeviceHasBiDirectionalSDI (mDeviceID))
                                CHECK(mDevice.SetSDITransmitEnable (chan, true)); // Make it an output

                        CHECK(mDevice.Connect (::GetSDIOutputInputXpt (chan, false/*isDS2*/),  fbIsRGB ? cscVidOutXpt : fsVidOutXpt));
                        CHECK(mDevice.SetSDIOutputStandard (chan, outputStandard));
                }	//	for each SDI output spigot

                //	And connect analog video output, if the device has one...
                if (::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtAnalogOut1))
                        CHECK(mDevice.Connect (::GetOutputDestInputXpt (NTV2_OUTPUTDESTINATION_ANALOG),  fbIsRGB ? cscVidOutXpt : fsVidOutXpt));

                //	And connect HDMI video output, if the device has one...
                if (::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v2)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v3)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v4))
                        CHECK(mDevice.Connect (::GetOutputDestInputXpt (NTV2_OUTPUTDESTINATION_HDMI),  fbIsRGB ? fsVidOutXpt : cscVidOutXpt));

                // connect eventual additional tiles
                for (unsigned int i = 1; i < desc.tile_count; ++i) {
                        NTV2Channel chan = (NTV2Channel) ((int) mOutputChannel + i);
                        const NTV2OutputCrosspointID fsVidOutXpt(::GetFrameBufferOutputXptFromChannel (chan,  fbIsRGB/*isRGB*/,  false/*is425*/));
                        if (fbIsRGB)
                                CHECK(mDevice.Connect (::GetCSCInputXptFromChannel (chan, false/*isKeyInput*/),  fsVidOutXpt)); // CSC
                        CHECK(mDevice.Connect (::GetSDIOutputInputXpt (chan, false/*isDS2*/),  fbIsRGB ? cscVidOutXpt : fsVidOutXpt));
                }
        }
}

void display::join()
{
        unique_lock<mutex> lk(frames_lock);
        frames.push(nullptr); // poison pill
        lk.unlock();
        frame_ready.notify_one();
        worker.join();
}

void display::process_frames()
{
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                CHECK(mDevice.SubscribeOutputVerticalEvent(chan));
        }

        while (true) {
                unique_lock<mutex> lk(frames_lock);
                frame_ready.wait(lk, [this]{ return frames.size() > 0; });
                struct video_frame *frame = frames.front();
                frames.pop();
                lk.unlock();
                if (!frame) { // poison pill
                        break;
                }

                CHECK(mDevice.WaitForOutputVerticalInterrupt(mOutputChannel));
                //      Flip sense of the buffers again to refer to the buffers that the hardware isn't using (i.e. the off-screen buffers)...
                mCurrentOutFrame ^= 1;
                for (unsigned int i = 0; i < frame->tile_count; ++i) {
                        CHECK(mDevice.DMAWriteFrame(mCurrentOutFrame + 2 * i, reinterpret_cast<ULWord*>(frame->tiles[i].data), frame->tiles[i].data_len));
                }

                for (unsigned int i = 0; i < frame->tile_count; ++i) {
                        //      Check for dropped frames by ensuring the hardware has not started to process
                        //      the buffers that were just filled....
                        uint32_t readBackOut;
                        CHECK(mDevice.GetOutputFrame((NTV2Channel)((unsigned int) mOutputChannel + i), readBackOut));

                        if (readBackOut == mCurrentOutFrame + 2 * i) {
                                LOG(LOG_LEVEL_WARNING)    << "## WARNING:  Drop detected: current out " << mCurrentOutFrame + 2 * i << ", readback out " << readBackOut << endl;
                                mFramesDropped++;
                        } else {
                                mFramesProcessed++;
                        }
                        //      Tell the hardware which buffers to start using at the beginning of the next frame...
                        CHECK(mDevice.SetOutputFrame((NTV2Channel)((unsigned int) mOutputChannel + i), mCurrentOutFrame + 2 * i));
                }

                if (mConf.withAudio) {
                        lock_guard<mutex> lk(mAudioLock);
                        if (mAudioIsReset && mAudioLen > 0) {
                                //      Now that the audio system has some samples to play, playback can be taken out of reset...
                                CHECK(mDevice.StartAudioOutput(mAudioSystem));
                                mAudioIsReset = false;
                        }

                        if (mAudioLen > 0) {
                                uint32_t val;
                                CHECK(mDevice.ReadAudioLastOut(val, mOutputChannel));
                                int channels = ::NTV2DeviceGetMaxAudioChannels (mDeviceID);
                                int latency_ms = ((mAudioOutLastAddress + mAudioOutWrapAddress - val) % mAudioOutWrapAddress) / (SAMPLE_RATE / 1000) / BPS / channels;
                                if (latency_ms > 135) {
                                        LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Buffer length: " << latency_ms << " ms, possible wrap-around.\n";
                                        mAudioOutLastAddress = ((val + (SAMPLE_RATE / 1000) * 70 /* ms */ * BPS * channels) % mAudioOutWrapAddress) / 128 * 128;
                                } else {
                                        LOG(LOG_LEVEL_DEBUG) << MODULE_NAME "Audio latency: " << latency_ms << "\n";
                                }

                                int len = min<int>(mAudioLen, mAudioOutWrapAddress - mAudioOutLastAddress); // length to the wrap-around
                                CHECK(mDevice.DMAWriteAudio(mAudioSystem, reinterpret_cast<ULWord*>(mAudioBuffer.get()), mAudioOutLastAddress, len));
                                if (mAudioLen - len > 0) {
                                        CHECK(mDevice.DMAWriteAudio(mAudioSystem, reinterpret_cast<ULWord*>(mAudioBuffer.get() + len), 0, mAudioLen - len));
                                        mAudioOutLastAddress = mAudioLen - len;
                                } else {
                                        mAudioOutLastAddress += len;
                                }
                                mAudioLen = 0;
                        }
                }

                mFrames += 1;
                print_stats();

                vf_free(frame);
        }
}

int display::Putf(struct video_frame *frame, int flags) {
        if (frame == nullptr) {
                return 1;
        }

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
                return 0;
        }

        unique_lock<mutex> lk(frames_lock);
        // PUTF_BLOCKING is not currently honored
        if (frames.size() > max_frame_queue_len) {
                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Frame dropped!\n";
                if (++mFramesDiscarded % 20 == 0) {
                        LOG(LOG_LEVEL_NOTICE) << MODULE_NAME << mFramesDiscarded << " frames discarded - try to increase buffer count or set \"novsync\" (see \"-d aja:help\" for details).\n";
                }
                vf_free(frame);
                return 1;
        }
        frames.push(frame);
        lk.unlock();
        frame_ready.notify_one();

        return 0;
}

void aja::display::print_stats() {
        auto now = steady_clock::now();
        double seconds = duration<double>(now - mT0).count();
        if (seconds > 5) {
                LOG(LOG_LEVEL_INFO) << MODULE_NAME << mFrames << " frames in "
                        << seconds << " seconds = " <<  mFrames / seconds << " FPS\n";
                mFrames = 0;
                mT0 = now;
        }
}

void aja::display::show_help() {
        cout << "Usage:\n"
                "\t" << rang::style::bold << rang::fg::red << "-d aja" << rang::fg::reset <<
                "[[:buffers=<b>][:channel=<ch>][:clear-routing][:connection=<c>][:device=<d>][:[no-]multi-channel][:novsync][:no-setup[-route]][:RGB|:YUV][:{smpte|full}-range]|:help] [-r embedded]\n" << rang::style::reset <<
                "where\n";

        cout << rang::style::bold << "\tbuffers\n" << rang::style::reset <<
                "\t\tuse <b> output buffers (default is " << DEFAULT_MAX_FRAME_QUEUE_LEN << ") - higher values increase stability\n"
                "\t\tbut may also increase latency (when VBlank is enabled)\n";

        cout << rang::style::bold << "\tclear-routing\n" << rang::style::reset <<
                "\t\tremove all existing signal paths for device\n";

        cout << rang::style::bold << "\tconnection\n" << rang::style::reset <<
                "\t\tone of: ";
        NTV2OutputDestination dest = NTV2OutputDestination();
        while (dest != NTV2_OUTPUTDESTINATION_INVALID) {
                if (dest > 0) {
                        cout << ", ";
                }
                cout << NTV2OutputDestinationToString(dest, true);
                // should be this, but GetNTV2InputSourceForIndex knows only SDIs
                //source = ::GetNTV2InputSourceForIndex(::GetIndexForNTV2InputSource(source) + 1);
                dest = (NTV2OutputDestination) ((int) dest + 1);
        }
        cout << "\n";

        cout << rang::style::bold << "\tchannel\n" << rang::style::reset <<
                "\t\tchannel number to use (indexed from 1). Doesn't need to be set for SDI, useful for HDMI (capture and display should have different channel numbers if both used, also other than 1 if SDI1 is in use, see \"-t aja:help\" to see number of available channels).\n";

        cout << rang::style::bold << "\tdevice\n" << rang::style::reset <<
                "\t\tdevice identifier (number or name)\n";

        cout << rang::style::bold << "\t[no-]multi-channel\n" << rang::style::reset <<
                "\t\tdo (not) treat the device as a multi-channel\n";

        cout << rang::style::bold << "\tno-setup[-route]\n" << rang::style::reset <<
                "\t\tdo not setup anything/routing (user must set it eg. in Cables app)\n";

        cout << rang::style::bold << "\tnovsync\n" << rang::style::reset <<
                "\t\tdisable sync on VBlank (may improve latency at the expense of tearing)\n";

        cout << rang::style::bold << "\tRGB|YUV\n" << rang::style::reset <<
                "\t\tforce SDI output to be RGB or YCbCr, otherwise UG keeps the colorspace\n";

        cout << rang::style::bold << "\tsmpte-|full-range\n" << rang::style::reset <<
                "\t\tuse either SMPTE (default) or full RGB range\n";

        cout << rang::style::bold << "\t-r embedded\n" << rang::style::reset <<
                "\t\treceive also audio and embed it to SDI\n";

        cout << "\n";
        cout << "Available devices:\n";

        CNTV2DeviceScanner      deviceScanner;
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                cout << "\t" << rang::style::bold << i << rang::style::reset << ") " << rang::style::bold << info.deviceIdentifier << rang::style::reset << ". " << info;
                cout << "\n";
        }
        if (deviceScanner.GetNumDevices() == 0) {
                cout << rang::fg::red << "\tno devices found\n" << rang::fg::reset;
        }
        cout << "\n";
}

NTV2FrameRate aja::display::getFrameRate(double fps)
{
        NTV2FrameRate result = NTV2_FRAMERATE_UNKNOWN;
        double best = 1000.0;

        for(int i = 0; i < NTV2_NUM_FRAMERATES; i++)
        {
                auto candidate = static_cast<NTV2FrameRate>(i);
                double distance = fabs(GetFramesPerSecond(candidate) - fps);
                if (distance < best)
                {
                        result = candidate;
                        best = distance;
                }
        }

        return result;
}

} // end of namespace aja
} // end of namespace ultragrid

namespace aja = ultragrid::aja;

LINK_SPEC void display_aja_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        CNTV2DeviceScanner      deviceScanner;
        *count = deviceScanner.GetNumDevices();
        *available_cards = static_cast<struct device_info *>(calloc(deviceScanner.GetNumDevices(), sizeof(struct device_info)));
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                snprintf((*available_cards)[i].dev, sizeof (*available_cards)[i].dev, ":device=%d", i);
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                strncpy((*available_cards)[i].name, info.deviceIdentifier.c_str(), sizeof (*available_cards)[0].name - 1);
                (*available_cards)[i].repeatable = false;
        }
}

LINK_SPEC void display_aja_run(void * /* arg */)
{
}

/**
 * format is a quad-link format
 */
static bool display_aja_is_quad_format(NTV2VideoFormat fmt) {
#if (AJA_NTV2_SDK_VERSION_MAJOR > 15 || (AJA_NTV2_SDK_VERSION_MAJOR == 15 && AJA_NTV2_SDK_VERSION_MINOR >= 2))
        if ((fmt >= NTV2_FORMAT_FIRST_4K_DEF_FORMAT && fmt < NTV2_FORMAT_END_4K_DEF_FORMATS)
                        || (fmt >= NTV2_FORMAT_FIRST_4K_DEF_FORMAT2 && fmt < NTV2_FORMAT_END_4K_DEF_FORMATS2)
                        || (fmt >= NTV2_FORMAT_FIRST_UHD2_DEF_FORMAT && fmt < NTV2_FORMAT_END_UHD2_DEF_FORMATS)
                        || (fmt >= NTV2_FORMAT_FIRST_UHD2_FULL_DEF_FORMAT && fmt < NTV2_FORMAT_END_UHD2_FULL_DEF_FORMATS)) {
                return true;
        }
#else
        UNUSED(fmt);
#endif
        return false;
}

/*
 * modified GetFirstMatchingVideoFormat
 */
static NTV2VideoFormat display_aja_get_first_matching_video_format(const NTV2FrameRate inFrameRate, const UWord inHeightLines, const UWord inWidthPixels, const bool inIsInterlaced, const bool inIsLevelB, bool skipQuadFormats)
{
        for (NTV2VideoFormat fmt(NTV2_FORMAT_FIRST_HIGH_DEF_FORMAT); fmt < NTV2_MAX_NUM_VIDEO_FORMATS;  fmt = NTV2VideoFormat(fmt + 1)) {
                if (display_aja_is_quad_format(fmt) && skipQuadFormats) {
                        continue; // skip NTV2_FORMAT_4x formats
                }
                if (inFrameRate == ::GetNTV2FrameRateFromVideoFormat(fmt))
                        if (inHeightLines == ::GetDisplayHeight(fmt))
                                if (inWidthPixels == ::GetDisplayWidth(fmt))
                                        if (inIsInterlaced == !::IsProgressiveTransport(fmt) && !IsPSF(fmt))
                                                if (NTV2_VIDEO_FORMAT_IS_B(fmt) == inIsLevelB)
                                                        return fmt;
	}
        return NTV2_FORMAT_UNKNOWN;
}


LINK_SPEC int display_aja_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<struct aja::display *>(state);
        if (s->desc.color_spec != VIDEO_CODEC_NONE) {
                s->join();
                s->desc = {};
        }

        if (desc.tile_count != 1 && desc.tile_count != 4) {
                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Unsupported tile count: " << desc.tile_count << "\n";
                return FALSE;
        }

        bool interlaced = desc.interlacing == INTERLACED_MERGED;
        // first try to skip quad split formats (aka quad link)
        s->mVideoFormat = display_aja_get_first_matching_video_format(aja::display::getFrameRate(desc.fps),
                        desc.height, desc.width, interlaced, false, false);
        // if not found or supported, include quad formats
        if (s->mVideoFormat == NTV2_FORMAT_UNKNOWN || !::NTV2DeviceCanDoVideoFormat (s->mDeviceID, s->mVideoFormat)) {
                s->mVideoFormat = display_aja_get_first_matching_video_format(aja::display::getFrameRate(desc.fps),
                                desc.height, desc.width, interlaced, false, true);
        }
        if (s->mVideoFormat == NTV2_FORMAT_UNKNOWN) {
                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Unsupported resolution"
#ifndef _MSC_VER
                        ": " << desc
#endif
                        << "\n";
                return FALSE;
        }
        s->mPixelFormat = NTV2_FBF_INVALID;
        for (auto & c : aja::codec_map) {
                if (c.second == desc.color_spec) {
                        s->mPixelFormat = c.first;
                }
        }
        assert(s->mPixelFormat != NTV2_FBF_INVALID);

        // deinit?
        s->desc = desc;
        try {
                s->Init();
        } catch (runtime_error &e) {
                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Reconfiguration failed: " << e.what() << "\n";
                s->desc = {};
                return FALSE;
        }

        s->worker = thread(&aja::display::process_frames, s);

        return TRUE;
}

LINK_SPEC void *display_aja_init(struct module * /* parent */, const char *fmt, unsigned int flags)
{
        struct aja::display::configuration conf;
        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);

        char *item;
        while ((item = strtok(tmp, ":")) != nullptr) {
                if (strcmp("help", item) == 0) {
                        aja::display::show_help();
                        return aja_display_init_noerr;
                } else if (strstr(item, "buffers=") == item) {
                        conf.bufLen = atoi(item + strlen("buffers="));
                        if (conf.bufLen <= 0) {
                                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Buffers must be positive!\n";
                                return nullptr;
                        }
                } else if (strcmp("clear-routing", item) == 0) {
                        conf.clearRouting = true;
                } else if (strstr(item, "connection=") != nullptr) {
                        string connection = item + strlen("connection=");
                        NTV2OutputDestination dest = NTV2OutputDestination();
                        while (dest != NTV2_OUTPUTDESTINATION_INVALID) {
                                if (NTV2OutputDestinationToString(dest, true) == connection) {
                                        conf.outputDestination = dest;
                                        break;
                                }
                                // should be this, but GetNTV2InputSourceForIndex knows only SDIs
                                //source = ::GetNTV2InputSourceForIndex(::GetIndexForNTV2InputSource(source) + 1);
                                dest = (NTV2OutputDestination) ((int) dest + 1);
                        }
                        if (dest == NTV2_OUTPUTDESTINATION_INVALID) {
                                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Unknown destination: " << connection << "!\n";
                                return nullptr;
                        }
                } else if (strstr(item, "channel=") != nullptr) {
                        conf.outputChannel = (NTV2Channel) (atoi(item + strlen("channel=")) - 1);
                } else if (strstr(item, "device=") != nullptr) {
                        conf.deviceId = item + strlen("device=");
                } else if (strstr(item, "multi-channel") != nullptr) {
                        conf.doMultiChannel = strstr(item, "no-") == nullptr;
                } else if (strstr(item, "no-setup") == item) {
                        conf.setupRoute = false;
                        if (strcmp(item, "no-setup") == 0) {
                                conf.setupAll = false;
                        }
                } else if (strstr(item, "novsync") == item) {
                        conf.novsync = true;
                } else if (strcasecmp(item, "RGB") == 0 || strcasecmp(item, "YUV") == 0) {
                        conf.forceOutputColorSpace = strcasecmp(item, "RGB") == 0 ? RGB : UYVY;
                } else if (strcasecmp(item, "smpte-range") == 0 || strcasecmp(item, "full-range") == 0) {
                        conf.smpteRange = strcasecmp(item, "smpte-range") == 0;
                } else {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Unknown option: " << item << "\n";
                        return nullptr;
                }
                tmp = nullptr;
        }

        try {
                conf.withAudio = (flags & DISPLAY_FLAG_AUDIO_ANY) != 0u;
                auto s = new aja::display(conf);
                return s;
        } catch (runtime_error &e) {
                LOG(LOG_LEVEL_ERROR) << MODULE_NAME << e.what() << "\n";
        }

        return nullptr;
}

LINK_SPEC void display_aja_done(void *state)
{
        auto s = static_cast<struct aja::display *>(state);

        if (s->desc.color_spec != VIDEO_CODEC_NONE) {
                s->join();
                s->desc = {};
        }

        delete s;
}

LINK_SPEC struct video_frame *display_aja_getf(void *state)
{
        auto s = static_cast<struct aja::display *>(state);

        return vf_alloc_desc_data(s->desc);
}

LINK_SPEC int display_aja_putf(void *state, struct video_frame *frame, int nonblock)
{
        auto s = static_cast<struct aja::display *>(state);

        if (frame && frame->color_spec == R12L) {
                char *tmp = (char *) malloc(frame->tiles[0].data_len);
                vc_copylineR12LtoR12A((unsigned char *) tmp, (unsigned char *) frame->tiles[0].data, frame->tiles[0].data_len, 0, 0, 0);
                free(frame->tiles[0].data);
                frame->tiles[0].data = tmp;
        }

        return s->Putf(frame, nonblock);
}

LINK_SPEC int display_aja_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = static_cast<struct aja::display *>(state);

        vector<codec_t> codecs(aja::codec_map.size());
        int count = 0;

        for (auto & c : aja::codec_map) {
                if (::NTV2DeviceCanDoFrameBufferFormat(s->mDeviceID, c.first)) {
                        codecs[count++] = c.second;
                }
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (count * sizeof(codec_t) <= *len) {
                                *len = count * sizeof(codec_t);
                                memcpy(val, codecs.data(), *len);
                        } else {
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len >= sizeof(struct audio_desc));
                                auto desc = static_cast<struct audio_desc *>(val);
                                desc->sample_rate = SAMPLE_RATE;
                                desc->ch_count = ::NTV2DeviceGetMaxAudioChannels (s->mDeviceID);
                                desc->codec = AC_PCM;
                                desc->bps = BPS;
                                *len = sizeof *desc;
                        }
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        if (*len >= sizeof(int)) {
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                                *len = sizeof(int);
                        } else {
                                return FALSE;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

LINK_SPEC void display_aja_put_audio_frame(void *state, struct audio_frame *frame)
{
        auto s = static_cast<struct aja::display *>(state);

        lock_guard<mutex> lk(s->mAudioLock);
        int len = NTV2_AUDIOSIZE_MAX - s->mAudioLen;
        if (frame->data_len > len) {
                LOG(LOG_LEVEL_WARNING) << MODULE_NAME << "Audio buffer overrun!\n";
        } else {
                len = frame->data_len;
        }
        memcpy(s->mAudioBuffer.get() + s->mAudioLen, frame->data, len);
        s->mAudioLen += len;
}

LINK_SPEC int display_aja_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        auto s = static_cast<struct aja::display *>(state);
        assert(quant_samples == BPS * 8 && sample_rate == SAMPLE_RATE && ::NTV2DeviceGetMaxAudioChannels (s->mDeviceID) == channels);

        AJAStatus status = s->SetUpAudio();
        if (AJA_FAILURE(status)) {
                ostringstream oss;
                oss << "Unable to initialize audio: " << AJAStatusToString(status);
                return FALSE;
        }

        return TRUE;
}

#ifndef _MSC_VER
static const struct video_display_info display_aja_info = {
        display_aja_probe,
        display_aja_init,
        display_aja_run,
        display_aja_done,
        display_aja_getf,
        display_aja_putf,
        display_aja_reconfigure,
        display_aja_get_property,
        display_aja_put_audio_frame,
        display_aja_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_MODULE(aja, &display_aja_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
#endif

/* vim: set expandtab sw=8 cino=N-8: */

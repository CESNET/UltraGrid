/**
 * @file   video_display/aja.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "aja_common.h"
#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video_display.h"
#if defined _MSC_VER && _MSC_VER <= 1800 // VS 2013
#define constexpr
#define noexcept
#endif
#include "rang.hpp"
#include "video.h"

#define DEFAULT_MAX_FRAME_QUEUE_LEN 1
#define MODULE_NAME "[AJA display] "
/**
 * The maximum number of bytes of 48KHz audio that can be transferred for two frames.
 * Worst case, assuming 16 channels of audio (max), 4 bytes per sample, and 67 msec per frame
 * (assuming the lowest possible frame rate of 14.98 fps)...
 * 48,000 samples per second requires 6,408 samples x 4 bytes/sample x 16 = 410,112 bytes
 * 402K will suffice, with 1536 bytes to spare
 */
#define NTV2_AUDIOSIZE_MAX      (2 * 201 * 1024)
#define BPS                     4
#define SAMPLE_RATE             48000

#ifdef _MSC_VER
#define log_msg(x, ...) fprintf(stderr, __VA_ARGS__)
#undef LOG
#define LOG(...) std::cerr
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
using std::vector;

namespace ultragrid {
namespace aja {

struct display {
        display(string const &device_id, NTV2OutputDestination outputDestination, bool withAudio, bool novsync, int buf_len);
        ~display();
        void Init();
        AJAStatus SetUpVideo();
        AJAStatus SetUpAudio();
        void RouteOutputSignal();
        int Putf(struct video_frame *frame, int nonblock);

        queue<struct video_frame *> frames;
        unsigned int max_frame_queue_len;
        mutex frames_lock;
        condition_variable frame_ready;
        thread worker;
        void join();
        void process_frames();

        bool mNovsync;
        static const ULWord app = AJA_FOURCC ('U','L','G','R');

        CNTV2Card mDevice;
        NTV2DeviceID mDeviceID;
        struct video_desc desc{};
        bool mDoMultiChannel; ///< Use multi-format
        NTV2EveryFrameTaskMode mSavedTaskMode = NTV2_TASK_MODE_INVALID; ///< Used to restore the prior task mode
        NTV2Channel  mOutputChannel = NTV2_CHANNEL1;
        NTV2VideoFormat mVideoFormat = NTV2_FORMAT_UNKNOWN;
        NTV2FrameBufferFormat mPixelFormat = NTV2_FBF_INVALID;
        const NTV2OutputDestination mOutputDestination;
        bool mDoLevelConversion = false;
        bool mEnableVanc = false;

        bool mWithAudio;
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

        static NTV2FrameRate getFrameRate(double fps);
        void print_stats();
        static void show_help();
};

display::display(string const &device_id, NTV2OutputDestination outputDestination, bool withAudio, bool novsync, int buf_len) : max_frame_queue_len(buf_len), mNovsync(novsync), mOutputDestination(outputDestination), mWithAudio(withAudio) {
        if (!CNTV2DeviceScanner::GetFirstDeviceFromArgument(device_id, mDevice)) {
                throw runtime_error(string("Device '") + device_id + "' not found!");
        }

        if (!mDevice.IsDeviceReady(false)) {
                throw runtime_error(string("Device '") + device_id + "' not ready!");
        }

        mDeviceID = mDevice.GetDeviceID(); // Keep this ID handy -- it's used frequently

        mDoMultiChannel = NTV2DeviceCanDoMultiFormat(mDeviceID);
        if (!mDoMultiChannel) {
                LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Device " << device_id << " cannot simultaneously handle different video formats.\n";
                if (!mDevice.AcquireStreamForApplication(app, static_cast <uint32_t> (getpid()))) {
                        throw runtime_error("Device busy!\n"); // Device is in use by another app -- fail
                }
                mDevice.GetEveryFrameServices (mSavedTaskMode); // Save the current service level
        }

        mDevice.SetEveryFrameServices(NTV2_OEM_TASKS);

        if (::NTV2DeviceCanDoMultiFormat(mDeviceID) && mDoMultiChannel) {
                mDevice.SetMultiFormatMode(true);
        } else if (::NTV2DeviceCanDoMultiFormat(mDeviceID)) {
                mDevice.SetMultiFormatMode (false);
        }

        mOutputChannel = ::NTV2OutputDestinationToChannel(mOutputDestination);

        //      Beware -- some devices (e.g. Corvid1) can only output from FrameStore 2...
        if ((mOutputChannel == NTV2_CHANNEL1) && (!::NTV2DeviceCanDoFrameStore1Display (mDeviceID))) {
                mOutputChannel = NTV2_CHANNEL2;
        }
        if (UWord (mOutputChannel) >= ::NTV2DeviceGetNumFrameStores (mDeviceID)) {
                ostringstream oss;
                oss    << "## ERROR:  Cannot use channel '" << mOutputChannel+1 << "' -- device only supports channel 1"
                        << (::NTV2DeviceGetNumFrameStores (mDeviceID) > 1  ?  string (" thru ") + string (1, uint8_t (::NTV2DeviceGetNumFrameStores (mDeviceID)+'0'))  :  "");
                throw runtime_error(oss.str());
        }

        mCurrentOutFrame = mOutputChannel * 2u;

        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                mDevice.EnableOutputInterrupt(chan);
        }
}

display::~display() {
        mDevice.UnsubscribeOutputVerticalEvent (mOutputChannel);

        if (!mDoMultiChannel) {
                mDevice.SetEveryFrameServices (mSavedTaskMode);                 //      Restore the previously saved service level
                mDevice.ReleaseStreamForApplication (app, static_cast <uint32_t>(getpid()));    //      Release the device
        }

        //      Don't leave the audio system active after we exit
        mDevice.SetAudioOutputReset     (mAudioSystem, true);
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
                mDevice.SetOutputFrame(chan, mCurrentOutFrame + 2 * i);

                mCurrentOutFrame ^= 1;
                mDevice.SetOutputFrame(chan, mCurrentOutFrame + 2 * i);
        }
}

AJAStatus display::SetUpVideo ()
{
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

        if (!::NTV2DeviceCanDo3GLevelConversion (mDeviceID) && mDoLevelConversion && ::IsVideoFormatA (mVideoFormat)) {
                mDoLevelConversion = false;
        }
        if (mDoLevelConversion) {
                mDevice.SetSDIOutLevelAtoLevelBConversion (mOutputChannel, mDoLevelConversion);
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
                        mDeviceID == DEVICE_ID_KONAIP_2110)
        {
                mDevice.SetReference(NTV2_REFERENCE_SFP1_PTP);
        }
        else
        {
                mDevice.SetReference (NTV2_REFERENCE_FREERUN);
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
                mDevice.SetEnableVANCData (true);               //      Enable VANC for non-SD formats, to pass thru captions, etc.
                if (::Is8BitFrameBufferFormat (mPixelFormat))
                {
                        //      8-bit FBFs require VANC bit shift...
                        mDevice.SetVANCShiftMode (mOutputChannel, NTV2_VANCDATA_8BITSHIFT_ENABLE);
                        mDevice.SetVANCShiftMode (mOutputChannel, NTV2_VANCDATA_8BITSHIFT_ENABLE);
                }
        }       //      if HD video format
        else {
                mDevice.SetEnableVANCData (false);      //      No VANC with RGB pixel formats (for now)
        }

        //      Subscribe the output interrupt -- it's enabled by default...
        mDevice.SubscribeOutputVerticalEvent (mOutputChannel);

        return AJA_STATUS_SUCCESS;

}       //      SetUpVideo

#define CHECK_OK(cmd, msg, action_failed) do { bool ret = cmd; if (!ret) {\
        LOG(LOG_LEVEL_WARNING) << MODULE_NAME << (msg) << "\n";\
        action_failed;\
}\
} while(0)
#define NOOP ((void)0)
AJAStatus display::SetUpAudio ()
{
        mAudioSystem = NTV2ChannelToAudioSystem(mOutputChannel);

        //      It's best to use all available audio channels...
        CHECK_OK(mDevice.SetNumberAudioChannels(::NTV2DeviceGetMaxAudioChannels(mDeviceID), mAudioSystem), "Unable to set audio channels!", return AJA_STATUS_FAIL);

        //      Assume 48kHz PCM...
        mDevice.SetAudioRate (NTV2_AUDIO_48K, mAudioSystem);

        //      4MB device audio buffers work best...
        mDevice.SetAudioBufferSize (NTV2_AUDIO_BUFFER_BIG, mAudioSystem);

        //      Set the SDI output audio embedders to embed audio samples from the output of mAudioSystem...
        mDevice.SetSDIOutputAudioSystem (mOutputChannel, mAudioSystem);
        mDevice.SetSDIOutputDS2AudioSystem (mOutputChannel, mAudioSystem);

        //
        //      Loopback mode plays whatever audio appears in the input signal when it's
        //      connected directly to an output (i.e., "end-to-end" mode). If loopback is
        //      left enabled, the video will lag the audio as video frames get briefly delayed
        //      in our ring buffer. Audio, therefore, needs to come out of the (buffered) frame
        //      data being played, so loopback must be turned off...
        //
        mDevice.SetAudioLoopBack (NTV2_AUDIO_LOOPBACK_OFF, mAudioSystem);

        CHECK_OK(mDevice.SetAudioOutputEraseMode(mAudioSystem, true), "Set erase mode", return AJA_STATUS_FAIL);

        //      Reset both the input and output sides of the audio system so that the buffer
        //      pointers are reset to zero and inhibited from advancing.
        mDevice.SetAudioOutputReset     (mAudioSystem, true);
        mAudioIsReset = true;

        mDevice.GetAudioWrapAddress     (mAudioOutWrapAddress,   mAudioSystem);
        mAudioOutLastAddress = 0;

        return AJA_STATUS_SUCCESS;

}       //      SetupAudio

void display::RouteOutputSignal ()
{
        const NTV2Standard              outputStandard  (::GetNTV2StandardFromVideoFormat (mVideoFormat));
        const UWord                             numVideoOutputs (::NTV2DeviceGetNumVideoOutputs (mDeviceID));
        bool                                    isRGB                   (::IsRGBFormat (mPixelFormat));

        //      If device has no RGB conversion capability for the desired channel, use YUV instead
        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                if (UWord (mOutputChannel) + UWord(i) > ::NTV2DeviceGetNumCSCs (mDeviceID))
                        isRGB = false;
        }

        if (mDoMultiChannel) {
                for (unsigned int i = 0; i < desc.tile_count; ++i) {
                        NTV2Channel chan = (NTV2Channel)((unsigned int) mOutputChannel + i);
                        //      Multiformat --- route the one SDI output to the CSC video output (RGB) or FrameStore output (YUV)...
                        if (::NTV2DeviceHasBiDirectionalSDI (mDeviceID)) {
                                mDevice.SetSDITransmitEnable (chan, true);
                        }

                        NTV2OutputCrosspointID  cscVidOutXpt    (::GetCSCOutputXptFromChannel (chan,  false/*isKey*/,  !isRGB/*isRGB*/));
                        NTV2OutputCrosspointID  fsVidOutXpt             (::GetFrameBufferOutputXptFromChannel (chan,  isRGB/*isRGB*/,  false/*is425*/));
                        if (isRGB) {
                                mDevice.Connect (::GetCSCInputXptFromChannel (chan, false/*isKeyInput*/),  fsVidOutXpt);
                        }
                        mDevice.Connect (::GetSDIOutputInputXpt (chan, false/*isDS2*/),  isRGB ? cscVidOutXpt : fsVidOutXpt);
                        mDevice.SetSDIOutputStandard (chan, outputStandard);
                }
        } else {
                NTV2OutputCrosspointID  cscVidOutXpt    (::GetCSCOutputXptFromChannel (mOutputChannel,  false/*isKey*/,  !isRGB/*isRGB*/));
                NTV2OutputCrosspointID  fsVidOutXpt             (::GetFrameBufferOutputXptFromChannel (mOutputChannel,  isRGB/*isRGB*/,  false/*is425*/));

                //      Not multiformat:  Route all possible SDI outputs to CSC video output (RGB) or FrameStore output (YUV)...
                mDevice.ClearRouting ();

                if (isRGB) {
                        for (unsigned int i = 0; i < desc.tile_count; ++i) {
                                mDevice.Connect (::GetCSCInputXptFromChannel ((NTV2Channel)((unsigned int) mOutputChannel + i), false/*isKeyInput*/),  fsVidOutXpt);
                        }
                }

                for (NTV2Channel chan (NTV2_CHANNEL1);  ULWord (chan) < numVideoOutputs;  chan = NTV2Channel (chan + 1))
                {
                        if (::NTV2DeviceHasBiDirectionalSDI (mDeviceID))
                                mDevice.SetSDITransmitEnable (chan, true);              //      Make it an output

                        mDevice.Connect (::GetSDIOutputInputXpt (chan, false/*isDS2*/),  isRGB ? cscVidOutXpt : fsVidOutXpt);
                        mDevice.SetSDIOutputStandard (chan, outputStandard);
                }       //      for each output spigot

                if (::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtAnalogOut1))
                        mDevice.Connect (::GetOutputDestInputXpt (NTV2_OUTPUTDESTINATION_ANALOG),  isRGB ? cscVidOutXpt : fsVidOutXpt);

                if (::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v2)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v3)
                                || ::NTV2DeviceCanDoWidget (mDeviceID, NTV2_WgtHDMIOut1v4))
                        mDevice.Connect (::GetOutputDestInputXpt (NTV2_OUTPUTDESTINATION_HDMI),  isRGB ? cscVidOutXpt : fsVidOutXpt);
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
                mDevice.SubscribeOutputVerticalEvent(chan);
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

                CHECK_OK(mDevice.WaitForOutputVerticalInterrupt(mOutputChannel), "WaitForOutputVerticalInterrupt", NOOP);
                //      Flip sense of the buffers again to refer to the buffers that the hardware isn't using (i.e. the off-screen buffers)...
                mCurrentOutFrame ^= 1;
                for (unsigned int i = 0; i < frame->tile_count; ++i) {
                        mDevice.DMAWriteFrame(mCurrentOutFrame + 2 * i, reinterpret_cast<ULWord*>(frame->tiles[i].data), frame->tiles[i].data_len);
                }

                for (unsigned int i = 0; i < frame->tile_count; ++i) {
                        //      Check for dropped frames by ensuring the hardware has not started to process
                        //      the buffers that were just filled....
                        uint32_t readBackOut;
                        mDevice.GetOutputFrame((NTV2Channel)((unsigned int) mOutputChannel + i), readBackOut);

                        if (readBackOut == mCurrentOutFrame + 2 * i) {
                                LOG(LOG_LEVEL_WARNING)    << "## WARNING:  Drop detected: current out " << mCurrentOutFrame + 2 * i << ", readback out " << readBackOut << endl;
                                mFramesDropped++;
                        } else {
                                mFramesProcessed++;
                        }
                        //      Tell the hardware which buffers to start using at the beginning of the next frame...
                        mDevice.SetOutputFrame((NTV2Channel)((unsigned int)mOutputChannel + i), mCurrentOutFrame + 2 * i);
                }

                if (mWithAudio) {
                        lock_guard<mutex> lk(mAudioLock);
                        if (mAudioIsReset && mAudioLen > 0) {
                                //      Now that the audio system has some samples to play, playback can be taken out of reset...
                                mDevice.SetAudioOutputReset (mAudioSystem, false);
                                mAudioIsReset = false;
                        }

                        if (mAudioLen > 0) {
                                uint32_t val;
                                mDevice.ReadAudioLastOut(val, mOutputChannel);
                                int channels = ::NTV2DeviceGetMaxAudioChannels (mDeviceID);
                                int latency_ms = ((mAudioOutLastAddress + mAudioOutWrapAddress - val) % mAudioOutWrapAddress) / (SAMPLE_RATE / 1000) / BPS / channels;
                                if (latency_ms > 135) {
                                        LOG(LOG_LEVEL_WARNING) << MODULE_NAME "Buffer length: " << latency_ms << " ms, possible wrap-around.\n";
                                        mAudioOutLastAddress = ((val + (SAMPLE_RATE / 1000) * 70 /* ms */ * BPS * channels) % mAudioOutWrapAddress) / 128 * 128;
                                } else {
                                        LOG(LOG_LEVEL_DEBUG) << MODULE_NAME "Audio latency: " << latency_ms << "\n";
                                }

                                int len = min<int>(mAudioLen, mAudioOutWrapAddress - mAudioOutLastAddress); // length to the wrap-around
                                mDevice.DMAWriteAudio(mAudioSystem, reinterpret_cast<ULWord*>(mAudioBuffer.get()), mAudioOutLastAddress, len);
                                if (mAudioLen - len > 0) {
                                        mDevice.DMAWriteAudio(mAudioSystem, reinterpret_cast<ULWord*>(mAudioBuffer.get() + len), 0, mAudioLen - len);
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
                "[:device=<d>][:connection=<c>][:novsync][:buffers=<b>][:help] [-r embedded]\n" << rang::style::reset <<
                "where\n";
        cout << rang::style::bold << "\tdevice\n" << rang::style::reset <<
                "\t\tdevice identifier (number or name)\n";
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

        cout << rang::style::bold << "\tnovsync\n" << rang::style::reset <<
                "\t\tdisable sync on VBlank (may improve latency at the expense of tearing)\n";

        cout << rang::style::bold << "\tbuffers\n" << rang::style::reset <<
                "\t\tuse <b> output buffers (default is " << DEFAULT_MAX_FRAME_QUEUE_LEN << ") - higher values increase stability\n"
                "\t\tbut may also increase latency (when VBlank is enabled)\n";

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

LINK_SPEC void display_aja_probe(struct device_info **available_cards, int *count)
{
        CNTV2DeviceScanner      deviceScanner;
        *count = deviceScanner.GetNumDevices();
        *available_cards = static_cast<struct device_info *>(calloc(deviceScanner.GetNumDevices(), sizeof(struct device_info)));
        for (unsigned int i = 0; i < deviceScanner.GetNumDevices (); i++) {
                snprintf((*available_cards)[i].id, sizeof (*available_cards)[i].id, "aja:device=%d", i);
                NTV2DeviceInfo  info    (deviceScanner.GetDeviceInfoList () [i]);
                strncpy((*available_cards)[i].name, info.deviceIdentifier.c_str(), sizeof (*available_cards)[0].name - 1);
                (*available_cards)[i].repeatable = false;
        }
}

LINK_SPEC void display_aja_run(void * /* arg */)
{
}

/*
 * modified GetFirstMatchingVideoFormat
 */
static NTV2VideoFormat MyGetFirstMatchingVideoFormat (const NTV2FrameRate inFrameRate, const UWord inHeightLines, const UWord inWidthPixels, const bool inIsInterlaced, const bool inIsLevelB)
{
        for (NTV2VideoFormat fmt(NTV2_FORMAT_FIRST_HIGH_DEF_FORMAT);  fmt < NTV2_MAX_NUM_VIDEO_FORMATS;  fmt = NTV2VideoFormat(fmt+1))
                if (inFrameRate == ::GetNTV2FrameRateFromVideoFormat(fmt))
                        if (inHeightLines == ::GetDisplayHeight(fmt))
                                if (inWidthPixels == ::GetDisplayWidth(fmt))
                                        if (inIsInterlaced == !::IsProgressiveTransport(fmt) && !IsPSF(fmt))
                                                if (NTV2_VIDEO_FORMAT_IS_B(fmt) == inIsLevelB)
                                                        return fmt;
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

        s->mVideoFormat = MyGetFirstMatchingVideoFormat(aja::display::getFrameRate(desc.fps),
                        desc.height, desc.width, desc.interlacing == INTERLACED_MERGED,
                        false);
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
        string device_idx{"0"};
        string connection;
        bool novsync = false;
        int buf_len = DEFAULT_MAX_FRAME_QUEUE_LEN;
        NTV2OutputDestination outputDestination = NTV2_OUTPUTDESTINATION_SDI1;
        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);

        char *item;
        while ((item = strtok(tmp, ":")) != nullptr) {
                if (strcmp("help", item) == 0) {
                        aja::display::show_help();
                        return aja_display_init_noerr;
                } else if (strstr(item, "connection=") != nullptr) {
                        string connection = item + strlen("connection=");
                        NTV2OutputDestination dest = NTV2OutputDestination();
                        while (dest != NTV2_OUTPUTDESTINATION_INVALID) {
                                if (NTV2OutputDestinationToString(dest, true) == connection) {
                                        outputDestination = dest;
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
                } else if (strstr(item, "device=") != nullptr) {
                        device_idx = item + strlen("device=");
                } else if (strstr(item, "novsync") == item) {
                        novsync = true;
                } else if (strstr(item, "buffers=") == item) {
                        buf_len = atoi(item + strlen("buffers="));
                        if (buf_len <= 0) {
                                LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Buffers must be positive!\n";
                                return nullptr;
                        }
                } else {
                        LOG(LOG_LEVEL_ERROR) << MODULE_NAME "Unknown option: " << item << "\n";
                        return nullptr;
                }
                tmp = nullptr;
        }

        try {
                auto s = new aja::display(device_idx, outputDestination, (flags & DISPLAY_FLAG_AUDIO_ANY) != 0u, novsync, buf_len);
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
        display_aja_reconfigure_audio
};

REGISTER_MODULE(aja, &display_aja_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
#endif

/* vim: set expandtab sw=8 cino=N-8: */

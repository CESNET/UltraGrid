/**
 * @file   audio/playback/coreaudio.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2016 CESNET, z. s. p. o.
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

#include "config.h"

#ifdef HAVE_COREAUDIO

#include <AudioUnit/AudioUnit.h>
#include <Availability.h>
#include <chrono>
#include <CoreAudio/AudioHardware.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "audio/audio.h"
#include "audio/audio_playback.h"
#include "debug.h"
#include "lib_common.h"
#include "rang.hpp"
#include "utils/ring_buffer.h"
#include "utils/audio_buffer.h"

using namespace std::chrono;
using rang::fg;
using rang::style;
using std::cout;

#define NO_DATA_STOP_SEC 2
#define MOD_NAME "[CoreAudio play.] "
#define CA_DIS_AD_B "ca-disable-adaptive-buf"

struct state_ca_playback {
#ifndef __MAC_10_9
        ComponentInstance
#else
        AudioComponentInstance
#endif
                        auHALComponentInstance;
        struct audio_desc desc;
        void *buffer; // audio buffer
        struct audio_buffer_api *buffer_fns;
        int audio_packet_size;
        steady_clock::time_point last_audio_read;
        bool quiet; ///< do not report buffer underruns if we do not receive data at all for a long period
        bool initialized;
};

static OSStatus theRenderProc(void *inRefCon,
                              AudioUnitRenderActionFlags *inActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumFrames,
                              AudioBufferList *ioData);

static OSStatus theRenderProc(void *inRefCon,
                              AudioUnitRenderActionFlags *inActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumFrames,
                              AudioBufferList *ioData)
{
        UNUSED(inActionFlags);
        UNUSED(inTimeStamp);
        UNUSED(inBusNumber);

        struct state_ca_playback * s = (struct state_ca_playback *) inRefCon;
        int write_bytes = inNumFrames * s->audio_packet_size;
        int ret;

        ret = s->buffer_fns->read(s->buffer, (char *) ioData->mBuffers[0].mData, write_bytes);
        ioData->mBuffers[0].mDataByteSize = ret;

        if(ret < write_bytes) {
                if (!s->quiet) {
                        fprintf(stderr, "[CoreAudio] Audio buffer underflow.\n");
                }
                //memset(ioData->mBuffers[0].mData, 0, write_bytes);
                ioData->mBuffers[0].mDataByteSize = ret;
                if (!s->quiet && duration_cast<seconds>(steady_clock::now() - s->last_audio_read).count() > NO_DATA_STOP_SEC) {
                        fprintf(stderr, "[CoreAudio] No data for %d seconds! Stopping.\n", NO_DATA_STOP_SEC);
                        s->quiet = true;
                }
        } else {
                if (s->quiet) {
                        fprintf(stderr, "[CoreAudio] Starting again.\n");
                }
                s->quiet = false;
                s->last_audio_read = steady_clock::now();
        }
        return noErr;
}

static bool audio_play_ca_ctl(void *state [[gnu::unused]], int request, void *data, size_t *len)
{
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                if (*len >= sizeof(struct audio_desc)) {
                        struct audio_desc desc;
                        memcpy(&desc, data, sizeof desc);
                        desc.codec = AC_PCM;
                        memcpy(data, &desc, sizeof desc);
                        *len = sizeof desc;
                        return true;
                } else{
                        return false;
                }
        default:
                return false;
        }
}

ADD_TO_PARAM(ca_disable_adaptive_buf, CA_DIS_AD_B, "* " CA_DIS_AD_B "\n"
                "  Core Audio - use fixed audio playback buffer instead of an adaptive one\n");
static int audio_play_ca_reconfigure(void *state, struct audio_desc desc)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;
        AudioStreamBasicDescription stream_desc;
        UInt32 size;
        OSErr ret = noErr;
        AURenderCallbackStruct  renderStruct;

        printf("[CoreAudio] Audio reinitialized to %d-bit, %d channels, %d Hz\n",
                        desc.bps * 8, desc.ch_count, desc.sample_rate);

        if (s->initialized) {
                ret = AudioOutputUnitStop(s->auHALComponentInstance);
                if(ret) {
                        fprintf(stderr, "[CoreAudio playback] Cannot stop AUHAL instance.\n");
                        goto error;
                }

                ret = AudioUnitUninitialize(s->auHALComponentInstance);
                if(ret) {
                        fprintf(stderr, "[CoreAudio playback] Cannot uninitialize AUHAL instance.\n");
                        goto error;
                }
                s->initialized = false;
        }

        s->desc = desc;

        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->buffer);
                s->buffer_fns = nullptr;
                s->buffer = nullptr;
        }

        {
                int buf_len_ms = 200; // 200 ms by default
                if (get_commandline_param("audio-buffer-len")) {
                        buf_len_ms = atoi(get_commandline_param("audio-buffer-len"));
                        assert(buf_len_ms > 0 && buf_len_ms < 10000);
                }
                if (get_commandline_param(CA_DIS_AD_B)) {
                        int buf_len = desc.bps * desc.ch_count * (desc.sample_rate * buf_len_ms / 1000);
                        s->buffer = ring_buffer_init(buf_len);
                        s->buffer_fns = &ring_buffer_fns;
                } else {
                        s->buffer = audio_buffer_init(desc.sample_rate, desc.bps, desc.ch_count, buf_len_ms);
                        s->buffer_fns = &audio_buffer_fns;
                }
        }

        size = sizeof(stream_desc);
        ret = AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &stream_desc, &size);
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot get device format from AUHAL instance.\n");
                goto error;
        }
        stream_desc.mSampleRate = desc.sample_rate;
        stream_desc.mFormatID = kAudioFormatLinearPCM;
        stream_desc.mChannelsPerFrame = desc.ch_count;
        stream_desc.mBitsPerChannel = desc.bps * 8;
        stream_desc.mFormatFlags = kAudioFormatFlagIsSignedInteger|kAudioFormatFlagIsPacked;
        stream_desc.mFramesPerPacket = 1;
        s->audio_packet_size = stream_desc.mBytesPerFrame = stream_desc.mBytesPerPacket = stream_desc.mFramesPerPacket * desc.ch_count * desc.bps;

        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &stream_desc, sizeof(stream_desc));
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot set device format to AUHAL instance.\n");
                goto error;
        }

        renderStruct.inputProc = theRenderProc;
        renderStruct.inputProcRefCon = s;
        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_SetRenderCallback,
                        kAudioUnitScope_Input, 0, &renderStruct, sizeof(AURenderCallbackStruct));
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot register audio processing callback.\n");
                goto error;
        }

        ret = AudioUnitInitialize(s->auHALComponentInstance);
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot initialize AUHAL.\n");
                goto error;
        }

        ret = AudioOutputUnitStart(s->auHALComponentInstance);
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot start AUHAL.\n");
                goto error;
        }

        s->initialized = true;

        return TRUE;

error:
        return FALSE;
}

static void audio_play_ca_probe(struct device_info **available_devices, int *count)
{
        *available_devices = (struct device_info *) malloc(sizeof(struct device_info));
        strcpy((*available_devices)[0].id, "ca");
        strcpy((*available_devices)[0].name, "Default OS X audio output");
        *count = 1;
}

static void audio_play_ca_help(const char *driver_name)
{
        UNUSED(driver_name);
        OSErr ret;
        AudioDeviceID *dev_ids;
        int dev_items;
        int i;
        UInt32 size;
        AudioObjectPropertyAddress propertyAddress;

        cout << style::bold << "\tcoreaudio" << style::reset << ": default CoreAudio output\n";
        propertyAddress.mSelector = kAudioHardwarePropertyDevices;
        propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
        propertyAddress.mElement = kAudioObjectPropertyElementMaster;
        ret = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size);
        if(ret) goto error;
        dev_ids = (AudioDeviceID *) malloc(size);
        dev_items = size / sizeof(AudioDeviceID);
        ret = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size, dev_ids);
        if(ret) goto error;

        for(i = 0; i < dev_items; ++i)
        {
                CFStringRef deviceName = NULL;
                char cname[128] = "";

                size = sizeof(deviceName);
                propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString;
                ret = AudioObjectGetPropertyData(dev_ids[i], &propertyAddress, 0, NULL, &size, &deviceName);
                CFStringGetCString(deviceName, (char *) cname, sizeof cname, kCFStringEncodingMacRoman);
                cout << style::bold << "\tcoreaudio:" << dev_ids[i] << style::reset << ": " << cname << "\n";
                CFRelease(deviceName);
        }
        free(dev_ids);

        return;

error:
        fprintf(stderr, "[CoreAudio] error obtaining device list.\n");
}

static void * audio_play_ca_init(const char *cfg)
{
        struct state_ca_playback *s;
        OSErr ret = noErr;
#ifndef __MAC_10_9
        Component comp;
        ComponentDescription comp_desc;
#else
        AudioComponent comp;
        AudioComponentDescription comp_desc;
#endif
        UInt32 size;
        AudioDeviceID device;

        s = new struct state_ca_playback();

        //There are several different types of Audio Units.
        //Some audio units serve as Outputs, Mixers, or DSP
        //units. See AUComponent.h for listing
        comp_desc.componentType = kAudioUnitType_Output;

        //Every Component has a subType, which will give a clearer picture
        //of what this components function will be.
        //comp_desc.componentSubType = kAudioUnitSubType_DefaultOutput;
        comp_desc.componentSubType = kAudioUnitSubType_HALOutput;

        //all Audio Units in AUComponent.h must use
        //"kAudioUnitManufacturer_Apple" as the Manufacturer
        comp_desc.componentManufacturer = kAudioUnitManufacturer_Apple;
        comp_desc.componentFlags = 0;
        comp_desc.componentFlagsMask = 0;

#ifndef __MAC_10_9
        comp = FindNextComponent(NULL, &comp_desc);
        if(!comp) goto error;
        ret = OpenAComponent(comp, &s->auHALComponentInstance);
        if (ret != noErr) goto error;
#else
        comp = AudioComponentFindNext(NULL, &comp_desc);
        if(!comp) goto error;
        ret = AudioComponentInstanceNew(comp, &s->auHALComponentInstance);
        if (ret != noErr) goto error;
#endif

        s->buffer = NULL;

        ret = AudioUnitUninitialize(s->auHALComponentInstance);
        if(ret) goto error;

        size=sizeof(device);
        if(cfg != NULL) {
                if(strcmp(cfg, "help") == 0) {
                        cout << "Core Audio playback usage:\n";
                        cout << style::bold << fg::red << "\t-r coreaudio" << fg::reset <<
                                "[:<index>] [--param audio-buffer-len=<len_ms>] [--param " CA_DIS_AD_B "]\n\n" << style::reset;
                        printf("Available CoreAudio devices:\n");
                        audio_play_ca_help(NULL);
                        delete s;
                        return &audio_init_state_ok;
                } else {
                        device = atoi(cfg);
                }
        } else {
                AudioObjectPropertyAddress propertyAddress;
                UInt32 size = sizeof device;
                propertyAddress.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
                propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
                propertyAddress.mElement = kAudioObjectPropertyElementMaster;
                ret = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size, &device);

                if(ret) goto error;
        }

        if (get_commandline_param(CA_DIS_AD_B) == nullptr) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Using adaptive buffer. "
                        "In case of problems, try \"--param " CA_DIS_AD_B "\" "
                        "option.\n";
        }

        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                         kAudioOutputUnitProperty_CurrentDevice,
                         kAudioUnitScope_Global,
                         1,
                         &device,
                         sizeof(device));
        if(ret) goto error;

        return s;

error:
        delete s;
        return NULL;
}

static void audio_play_ca_put_frame(void *state, struct audio_frame *frame)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        s->buffer_fns->write(s->buffer, frame->data, frame->data_len);
}

static void audio_play_ca_done(void *state)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        if (s->initialized) {
                AudioOutputUnitStop(s->auHALComponentInstance);
                AudioUnitUninitialize(s->auHALComponentInstance);
        }
        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->buffer);
        }
        delete s;
}

static const struct audio_playback_info aplay_coreaudio_info = {
        audio_play_ca_probe,
        audio_play_ca_help,
        audio_play_ca_init,
        audio_play_ca_put_frame,
        audio_play_ca_ctl,
        audio_play_ca_reconfigure,
        audio_play_ca_done
};

REGISTER_MODULE(coreaudio, &aplay_coreaudio_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

#endif /* HAVE_COREAUDIO */


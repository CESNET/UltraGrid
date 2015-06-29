/**
 * @file   audio/playback/coreaudio.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s. p. o.
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

#include "audio/audio.h"
#include "audio/playback/coreaudio.h"
#include "utils/ring_buffer.h"
#include "debug.h"
#include <chrono>
#include <stdlib.h>
#include <string.h>
#include <AudioUnit/AudioUnit.h>
#include <CoreAudio/AudioHardware.h>

using namespace std::chrono;

#define NO_DATA_STOP_SEC 2

struct state_ca_playback {
#if OS_VERSION_MAJOR <= 9
        ComponentInstance
#else
        AudioComponentInstance
#endif
                        auHALComponentInstance;
        struct audio_desc desc;
        struct ring_buffer *buffer;
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

        ret = ring_buffer_read(s->buffer, (char *) ioData->mBuffers[0].mData, write_bytes);
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

int audio_play_ca_reconfigure(void *state, int quant_samples, int channels,
                                                int sample_rate)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;
        AudioStreamBasicDescription stream_desc;
        UInt32 size;
        OSErr ret = noErr;
        AURenderCallbackStruct  renderStruct;

        printf("[CoreAudio] Audio reinitialized to %d-bit, %d channels, %d Hz\n",
                        quant_samples, channels, sample_rate);

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

        s->desc.bps = quant_samples / 8;
        s->desc.ch_count = channels;
        s->desc.sample_rate = sample_rate;

        ring_buffer_destroy(s->buffer);
        s->buffer = NULL;

        s->buffer = ring_buffer_init(quant_samples / 8 * channels * sample_rate);

        size = sizeof(stream_desc);
        ret = AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &stream_desc, &size);
        if(ret) {
                fprintf(stderr, "[CoreAudio playback] Cannot get device format from AUHAL instance.\n");
                goto error;
        }
        stream_desc.mSampleRate = sample_rate;
        stream_desc.mFormatID = kAudioFormatLinearPCM;
        stream_desc.mChannelsPerFrame = channels;
        stream_desc.mBitsPerChannel = quant_samples;
        stream_desc.mFormatFlags = kAudioFormatFlagIsSignedInteger|kAudioFormatFlagIsPacked;
        stream_desc.mFramesPerPacket = 1;
        s->audio_packet_size = stream_desc.mBytesPerFrame = stream_desc.mBytesPerPacket = stream_desc.mFramesPerPacket * channels * (quant_samples / 8);

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


void audio_play_ca_help(const char *driver_name)
{
        UNUSED(driver_name);
        OSErr ret;
        AudioDeviceID *dev_ids;
        int dev_items;
        int i;
        UInt32 size;
        AudioObjectPropertyAddress propertyAddress;

        printf("\tcoreaudio : default CoreAudio output\n");
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
                fprintf(stderr,"\tcoreaudio:%d : %s\n", (int) dev_ids[i], cname);
                CFRelease(deviceName);
        }
        free(dev_ids);

        return;

error:
        fprintf(stderr, "[CoreAudio] error obtaining device list.\n");
}

void * audio_play_ca_init(char *cfg)
{
        struct state_ca_playback *s;
        OSErr ret = noErr;
#if OS_VERSION_MAJOR <= 9
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

#if OS_VERSION_MAJOR <= 9
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

void audio_play_ca_put_frame(void *state, struct audio_frame *frame)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        ring_buffer_write(s->buffer, frame->data, frame->data_len);
}

void audio_play_ca_done(void *state)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        if (s->initialized) {
                AudioOutputUnitStop(s->auHALComponentInstance);
                AudioUnitUninitialize(s->auHALComponentInstance);
        }
        ring_buffer_destroy(s->buffer);
        delete s;
}

#endif /* HAVE_COREAUDIO */


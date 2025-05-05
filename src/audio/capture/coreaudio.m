/**
 * @file   audio/capture/coreaudio.m
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2023 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#include <AVFoundation/AVFoundation.h>
#include <AudioUnit/AudioUnit.h>
#include <Availability.h>
#include <CoreAudio/AudioHardware.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio_capture.h"
#include "audio/playback/coreaudio.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macos.h"
#include "utils/macros.h"
#include "utils/ring_buffer.h"

#define MOD_NAME "[CoreAudio cap.] "

struct state_ca_capture {
        AudioComponentInstance auHALComponentInstance;
        struct audio_frame frame;
        char *tmp;
        struct ring_buffer *buffer;
        int audio_packet_size;
        AudioBufferList *theBufferList;

        pthread_mutex_t lock;
        pthread_cond_t cv;
        volatile int boss_waiting;
        volatile int data_ready;
};

static OSStatus InputProc(void *inRefCon,
                AudioUnitRenderActionFlags *ioActionFlags,
                const AudioTimeStamp *inTimeStamp,
                UInt32 inBusNumber,
                UInt32 inNumberFrames,
                AudioBufferList * ioData);
static AudioBufferList *AllocateAudioBufferList(UInt32 numChannels, UInt32 size);
static void DestroyAudioBufferList(AudioBufferList* list);

// Convenience function to allocate our audio buffers
static AudioBufferList *AllocateAudioBufferList(UInt32 numChannels, UInt32 size)
{
        AudioBufferList*                        list;
        UInt32                                          i;
        
        list = (AudioBufferList*)calloc(1, sizeof(AudioBufferList) + numChannels * sizeof(AudioBuffer));
        if(list == NULL)
        return NULL;
        
        list->mNumberBuffers = numChannels;
        for(i = 0; i < numChannels; ++i) {
                list->mBuffers[i].mNumberChannels = 1;
                list->mBuffers[i].mDataByteSize = size;
                list->mBuffers[i].mData = malloc(size);
                if(list->mBuffers[i].mData == NULL) {
                        DestroyAudioBufferList(list);
                        return NULL;
                }
        }
        return list;
}

// Convenience function to dispose of our audio buffers
static void DestroyAudioBufferList(AudioBufferList* list)
{
        UInt32                                          i;
        
        if(list) {
                for(i = 0; i < list->mNumberBuffers; i++) {
                        if(list->mBuffers[i].mData)
                        free(list->mBuffers[i].mData);
                }
                free(list);
        }
}

static OSStatus InputProc(void *inRefCon,
                AudioUnitRenderActionFlags *ioActionFlags,
                const AudioTimeStamp *inTimeStamp,
                UInt32 inBusNumber,
                UInt32 inNumberFrames,
                AudioBufferList * ioData)
{
	UNUSED(ioData);
        struct state_ca_capture * s = (struct state_ca_capture *) inRefCon;

        OSStatus err =noErr;

        err= AudioUnitRender(s->auHALComponentInstance, ioActionFlags, inTimeStamp, inBusNumber,     //will be '1' for input data
                inNumberFrames, //# of frames requested
                s->theBufferList);

        if(err == noErr) {
                int i;
                int len = inNumberFrames * s->audio_packet_size;
                for(i = 0; i < s->frame.ch_count; ++i)
                        mux_channel(s->tmp, s->theBufferList->mBuffers[i].mData, s->frame.bps, len, s->frame.ch_count, i, 1.0);
                uint32_t write_bytes = len * s->frame.ch_count;

                pthread_mutex_lock(&s->lock);
                ring_buffer_write(s->buffer, s->tmp, write_bytes);
                s->data_ready = TRUE;
                if(s->boss_waiting)
                        pthread_cond_signal(&s->cv);
                pthread_mutex_unlock(&s->lock);
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "writing buffer caused error %s.\n",  get_osstatus_str(err));
        }

        return err;
}

static void audio_cap_ca_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_ca_probe(available_devices, count, -1);
}

static void audio_cap_ca_help()
{
        struct device_info *available_devices;
        int count;
        void (*deleter)(void *) = NULL;
        audio_cap_ca_probe(&available_devices, &count, &deleter);

        for (int i = 0; i < count; ++i) {
                color_printf("\t" TBOLD("coreaudio%-4s") ": %s\n", available_devices[i].dev, available_devices[i].name);
        }
        deleter ? deleter(available_devices) : free(available_devices);
}

#define CA_STRINGIFY(A) #A

#define CHECK_OK(cmd, msg, action_failed) do { OSStatus ret = cmd; if (ret != noErr) {\
        log_msg(strlen(CA_STRINGIFY(action_failed)) == 0 ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR, MOD_NAME "%s: %s\n", (msg), get_osstatus_str(ret));\
        action_failed;\
}\
} while(0)
#define NOOP

#ifdef __MAC_10_14
// http://anasambri.com/ios/accessing-camera-and-photos-in-ios.html
static void (^cb)(BOOL) = ^void(BOOL granted) {
        if (!granted) {
                dispatch_async(dispatch_get_main_queue(), ^{
                                //show alert
                                });
        }
};
#endif // defined __MAC_10_14



static void * audio_cap_ca_init(struct module *parent, const char *cfg)
{
        UNUSED(parent);
        if (strcmp(cfg, "help") == 0) {
                printf("Core Audio capture usage:\n");
                color_printf(TBOLD(TRED("\t-s coreaudio") "[:<index>|:<name>]") "\n");
                color_printf("where\n\t" TBOLD("<name>") " - device name substring (case sensitive)\n\n");
                printf("Available Core Audio capture devices:\n");
                audio_cap_ca_help();
                return INIT_NOERR;
        }
        OSStatus ret = noErr;
        AudioComponentDescription desc;
        AudioDeviceID device;
        UInt32 size = sizeof device;

        if (strlen(cfg) > 0) {
                char *endptr = NULL;
                device = strtol(cfg, &endptr, 0);
                if (*endptr != '\0') {
                        device = audio_ca_get_device_by_name(cfg);
                        if (device == UINT_MAX) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong device index or unrecognized name \"%s\"!\n", cfg);
                                return NULL;
                        }
                }
        } else {
                AudioObjectPropertyAddress propertyAddress;
                propertyAddress.mSelector = kAudioHardwarePropertyDefaultInputDevice;
                propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
                propertyAddress.mElement = kAudioObjectPropertyElementMain;
                if ((ret = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size, &device)) != 0) {
                        log_msg(LOG_LEVEL_ERROR, "Error finding default input device: %s.\n", get_osstatus_str(ret));
                        return NULL;
                }
        }
        char device_name[128];
        audio_ca_get_device_name(device, sizeof device_name, device_name);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Using device: %s\n", device_name);

#ifdef __MAC_10_14
        AVAuthorizationStatus authorization_status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        if (authorization_status == AVAuthorizationStatusRestricted ||
                        authorization_status == AVAuthorizationStatusDenied) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Application is not authorized to capture audio input!\n");
                return NULL;
        }
        if (authorization_status == AVAuthorizationStatusNotDetermined) {
                [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:cb];
        }
#endif // defined __MAC_10_14

        double rate = 0.0;
        size = sizeof(double);
        AudioObjectPropertyAddress propertyAddress;
        propertyAddress.mSelector = kAudioDevicePropertyNominalSampleRate;
        propertyAddress.mScope = kAudioDevicePropertyScopeInput;
        propertyAddress.mElement = kAudioObjectPropertyElementMain;
        ret = AudioObjectGetPropertyData(device, &propertyAddress, 0, NULL, &size, &rate);
        if (ret != noErr || rate == 0.0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get sample rate: %s. Wrong device index?\n", get_osstatus_str(ret));
                return NULL;
        }
        if (audio_capture_sample_rate != 0 && audio_capture_sample_rate != rate) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Requested sample rate %u, got %lf!\n", audio_capture_sample_rate, rate);
        }

        struct state_ca_capture *s = (struct state_ca_capture *) calloc(1, sizeof(struct state_ca_capture));
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);
        s->boss_waiting = FALSE;
        s->data_ready = FALSE;
        s->frame.bps = audio_capture_bps ? audio_capture_bps : 2;
        s->frame.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;

        s->frame.sample_rate = rate;
        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate;
        int nonres_channel_size = s->frame.bps * s->frame.sample_rate;

        s->theBufferList = AllocateAudioBufferList(s->frame.ch_count, nonres_channel_size);
        s->tmp = (char *) malloc(nonres_channel_size * s->frame.ch_count);

        s->buffer = ring_buffer_init(s->frame.max_size);
        s->frame.data = (char *) malloc(s->frame.max_size);

        //There are several different types of Audio Units.
        //Some audio units serve as Outputs, Mixers, or DSP
        //units. See AUComponent.h for listing
        desc.componentType = kAudioUnitType_Output;

        //Every Component has a subType, which will give a clearer picture
        //of what this components function will be.
        //desc.componentSubType = kAudioUnitSubType_DefaultOutput;
        desc.componentSubType = kAudioUnitSubType_HALOutput;

        //all Audio Units in AUComponent.h must use 
        //"kAudioUnitManufacturer_Apple" as the Manufacturer
        desc.componentManufacturer = kAudioUnitManufacturer_Apple;
        desc.componentFlags = 0;
        desc.componentFlagsMask = 0;

        bool failed = true;
        do {
                AudioComponent comp = AudioComponentFindNext(NULL, &desc);
                if(!comp) {
                        fprintf(stderr, "Error finding AUHAL component.\n");
                        break;
                }
                CHECK_OK(AudioComponentInstanceNew(comp, &s->auHALComponentInstance),
                                "Error opening AUHAL component",
                                break);

                //When using AudioUnitSetProperty the 4th parameter in the method
                //refer to an AudioUnitElement. When using an AudioOutputUnit
                //the input element will be '1' and the output element will be '0'.

                UInt32 enableIO = 1;
                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance,
                                        kAudioOutputUnitProperty_EnableIO,
                                        kAudioUnitScope_Input,
                                        1, // input element
                                        &enableIO,
                                        sizeof(enableIO)),
                                "Error enabling input on AUHAL", break);

                enableIO = 0;
                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance,
                                        kAudioOutputUnitProperty_EnableIO,
                                        kAudioUnitScope_Output,
                                        0,   //output element
                                        &enableIO,
                                        sizeof(enableIO)),
                                "Error disabling output on AUHAL", break);

                size=sizeof(device);
                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance,
                                        kAudioOutputUnitProperty_CurrentDevice,
                                        kAudioUnitScope_Global,
                                        0,
                                        &device,
                                        sizeof(device)),
                                "Error setting device to AUHAL instance", break);

                AudioStreamBasicDescription desc;

                size = sizeof(desc);
                CHECK_OK(AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                                1, &desc, &size), "Error getting default device properties", break);

                desc.mChannelsPerFrame = s->frame.ch_count;
                desc.mSampleRate = (double) s->frame.sample_rate;
                desc.mFormatID = kAudioFormatLinearPCM;
                desc.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
                if (desc.mFormatID == kAudioFormatLinearPCM && s->frame.ch_count == 1)
                        desc.mFormatFlags &= ~kLinearPCMFormatFlagIsNonInterleaved;

/*#if __BIG_ENDIAN__
                desc.mFormatFlags |= kAudioFormatFlagIsBigEndian;
#endif */
                desc.mBitsPerChannel = s->frame.bps * 8;
                desc.mBytesPerFrame = desc.mBitsPerChannel / 8;
                desc.mFramesPerPacket = 1;
                desc.mBytesPerPacket = desc.mBytesPerFrame;
                s->audio_packet_size = desc.mBytesPerPacket;

                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output,
                                1, &desc, sizeof(desc)), "Error setting device properties", break);

                AURenderCallbackStruct input;
                input.inputProc = InputProc;
                input.inputProcRefCon = s;
                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance, kAudioOutputUnitProperty_SetInputCallback,
                                kAudioUnitScope_Global, 0, &input, sizeof(input)),
                                "Error setting input callback", break);
                uint32_t numFrames = 128;
                if (get_commandline_param("audio-cap-frames")) {
                        numFrames = atoi(get_commandline_param("audio-cap-frames"));
                }
                CHECK_OK(AudioUnitSetProperty(s->auHALComponentInstance, kAudioDevicePropertyBufferFrameSize,
                                        kAudioUnitScope_Global, 0, &numFrames, sizeof(numFrames)),
                                        "Error setting frames", NOOP);

                CHECK_OK(AudioUnitInitialize(s->auHALComponentInstance), "Error initializing device", break);
                CHECK_OK(AudioOutputUnitStart(s->auHALComponentInstance), "Error starting device", break);
                failed = false;
        } while(0);

        if (!failed) {
                return s;
        }

        // error accured
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        DestroyAudioBufferList(s->theBufferList);
        free(s->frame.data);
        ring_buffer_destroy(s->buffer);
        free(s->tmp);
        free(s);
        return NULL;
}

static struct audio_frame *audio_cap_ca_read(void *state)
{
        struct state_ca_capture *s = (struct state_ca_capture *) state;
        int ret = FALSE;

        pthread_mutex_lock(&s->lock);
        ret = ring_buffer_read(s->buffer, s->frame.data, s->frame.max_size);
        if(!ret) {
                s->data_ready = FALSE;
                s->boss_waiting = TRUE;
                while(!s->data_ready) {
                        pthread_cond_wait(&s->cv, &s->lock);
                }
                s->boss_waiting = FALSE;
                ret = ring_buffer_read(s->buffer, s->frame.data, s->frame.max_size);
        }
        pthread_mutex_unlock(&s->lock);

        s->frame.data_len = ret;

        return &s->frame;
}

static void audio_cap_ca_done(void *state)
{
        struct state_ca_capture *s = (struct state_ca_capture *) state;

        if(!s)
                return;

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        DestroyAudioBufferList(s->theBufferList);
        free(s->frame.data);
        ring_buffer_destroy(s->buffer);
        free(s->tmp);

        free(s);
}

static const struct audio_capture_info acap_coreaudio_info = {
        audio_cap_ca_probe,
        audio_cap_ca_init,
        audio_cap_ca_read,
        audio_cap_ca_done
};

REGISTER_MODULE(coreaudio, &acap_coreaudio_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


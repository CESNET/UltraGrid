/*
 * FILE:    audio/capture/coreaudio.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#include "config.h"
#include "host.h"

#ifdef HAVE_COREAUDIO

#include "audio/audio.h"
#include "audio/utils.h"
#include "audio/capture/coreaudio.h" 
#include "utils/ring_buffer.h"
#include "debug.h"
#include <stdlib.h>
#include <string.h>
#include <AudioUnit/AudioUnit.h>
#include <CoreAudio/AudioHardware.h>
#include <QuickTime/QuickTime.h>
#include <pthread.h>

#ifdef HAVE_SPEEX
#include <speex/speex_resampler.h> 
#endif

struct state_ca_capture {
#if OS_VERSION_MAJOR <= 9
        ComponentInstance 
#else
        AudioComponentInstance
#endif
                        auHALComponentInstance;
        struct audio_frame frame;
        char *tmp;
#ifdef HAVE_SPEEX
        char *resampled;
#endif
        struct ring_buffer *buffer;
        int audio_packet_size;
        AudioBufferList *theBufferList;

        pthread_mutex_t lock;
        pthread_cond_t cv;
        volatile int boss_waiting;
        volatile int data_ready;
        int nominal_sample_rate;
#ifdef HAVE_SPEEX
        SpeexResamplerState *resampler; 
#endif
};

static OSStatus InputProc(void *inRefCon,
                AudioUnitRenderActionFlags *ioActionFlags,
                const AudioTimeStamp *inTimeStamp,
                UInt32 inBusNumber,
                UInt32 inNumberFrames,
                AudioBufferList * ioData);
AudioBufferList *AllocateAudioBufferList(UInt32 numChannels, UInt32 size);
void DestroyAudioBufferList(AudioBufferList* list);

// Convenience function to allocate our audio buffers
AudioBufferList *AllocateAudioBufferList(UInt32 numChannels, UInt32 size)
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
void DestroyAudioBufferList(AudioBufferList* list)
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
#ifdef HAVE_SPEEX
                if(s->nominal_sample_rate != s->frame.sample_rate) {
                        int err;
                        uint32_t in_frames = inNumberFrames;
                        err = speex_resampler_process_interleaved_int(s->resampler, (spx_int16_t *) s->tmp, &in_frames, (spx_int16_t *) s->resampled, &write_bytes);
                        //speex_resampler_process_int(resampler, channelID, in, &in_length, out, &out_length); 
                        write_bytes *= s->frame.bps * s->frame.ch_count;
                        if(err) {
                                fprintf(stderr, "Resampling data error.\n");
                                return err;
                        }
                }
#endif

                pthread_mutex_lock(&s->lock);
#ifdef HAVE_SPEEX
                if(s->nominal_sample_rate != s->frame.sample_rate) 
                        ring_buffer_write(s->buffer, s->resampled, write_bytes);
                else
#endif
                        ring_buffer_write(s->buffer, s->tmp, write_bytes);
                s->data_ready = TRUE;
                if(s->boss_waiting)
                        pthread_cond_signal(&s->cv);
                pthread_mutex_unlock(&s->lock);
        } else {
                fprintf(stderr, "[CoreAudio] writing buffer caused error %i.\n", (int) err);
        }

        return err;
}

void audio_cap_ca_help(const char *driver_name)
{
        UNUSED(driver_name);
        OSErr ret;
        AudioDeviceID *dev_ids;
        int dev_items;
        int i;
        UInt32 size;

        printf("\tcoreaudio : default CoreAudio input\n");
        ret = AudioHardwareGetPropertyInfo(kAudioHardwarePropertyDevices, &size, NULL);
        if(ret) goto error;
        dev_ids = malloc(size);
        dev_items = size / sizeof(AudioDeviceID);
        ret = AudioHardwareGetProperty(kAudioHardwarePropertyDevices, &size, dev_ids);
        if(ret) goto error;

        for(i = 0; i < dev_items; ++i)
        {
                char name[128];
                
                size = sizeof(name);
                ret = AudioDeviceGetProperty(dev_ids[i], 0, 0, kAudioDevicePropertyDeviceName, &size, name);
                fprintf(stderr,"\tcoreaudio:%d : %s\n", (int) dev_ids[i], name);
        }
        free(dev_ids);

        return;

error:
        fprintf(stderr, "[CoreAudio] error obtaining device list.\n");
}

void * audio_cap_ca_init(char *cfg)
{
        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available Core Audio capture devices:\n");
                audio_cap_ca_help(NULL);
                return NULL;
        }
        struct state_ca_capture *s;
        OSErr ret = noErr;
#if OS_VERSION_MAJOR <= 9
        Component comp;
        ComponentDescription desc;
#else
        AudioComponent comp;
        AudioComponentDescription desc;
#endif
        UInt32 size;
        AudioDeviceID device;

        s = (struct state_ca_capture *) calloc(1, sizeof(struct state_ca_capture));

        size=sizeof(device);
        if(cfg != NULL) {
                device = atoi(cfg);
        } else {
                ret = AudioHardwareGetProperty(kAudioHardwarePropertyDefaultInputDevice, &size, &device);
                if(ret) {
                        fprintf(stderr, "Error finding default input device.\n");
                        goto error;
                }
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);
        s->boss_waiting = FALSE;
        s->data_ready = FALSE;
        s->frame.bps = 2;
        s->frame.ch_count = audio_capture_channels;

        double rate;
        size = sizeof(double);
        ret = AudioDeviceGetProperty(device, 0, 0, kAudioDevicePropertyNominalSampleRate, &size, &rate);
        s->nominal_sample_rate =  rate;
#ifndef HAVE_SPEEX
        s->frame.sample_rate = rate;
        fprintf(stderr, "[CoreAudio] Libspeex support not compiled in, resampling won't work (check manual or wiki how to enable it)!\n");
#else
        s->resampler = NULL;
        s->frame.sample_rate = 48000;
        if(s->frame.sample_rate != s->nominal_sample_rate) {
                int err;
                s->resampler = speex_resampler_init(s->frame.ch_count, s->nominal_sample_rate, s->frame.sample_rate, 10, &err); 
                if(err) {
                        s->frame.sample_rate = s->nominal_sample_rate;
                }
        }
#endif

        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate;
        int nonres_channel_size = s->frame.bps * s->nominal_sample_rate;

        s->theBufferList = AllocateAudioBufferList(s->frame.ch_count, nonres_channel_size);
        s->tmp = (char *) malloc(nonres_channel_size * s->frame.ch_count);

        s->buffer = ring_buffer_init(s->frame.max_size);
        s->frame.data = (char *) malloc(s->frame.max_size);
#ifdef HAVE_SPEEX
        s->resampled = (char *) malloc(s->frame.max_size);
#endif

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

#if MACOSX_VERSION_MAJOR > 9
        comp = AudioComponentFindNext(NULL, &desc);
        if(!comp) {
                fprintf(stderr, "Error finding AUHAL component.\n");
                goto error;
        }
        ret = AudioComponentInstanceNew(comp, &s->auHALComponentInstance);
        if (ret != noErr) {
                fprintf(stderr, "Error opening AUHAL component.\n");
                goto error;
        }
#else
        comp = FindNextComponent(NULL, &desc);
        if(!comp) {
                fprintf(stderr, "Error finding AUHAL component.\n");
                goto error;
        }
        ret = OpenAComponent(comp, &s->auHALComponentInstance);
        if (ret != noErr) {
                fprintf(stderr, "Error opening AUHAL component.\n");
                goto error;
        }
#endif
        UInt32 enableIO;

        //When using AudioUnitSetProperty the 4th parameter in the method
        //refer to an AudioUnitElement. When using an AudioOutputUnit
        //the input element will be '1' and the output element will be '0'.


        enableIO = 1;
        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                kAudioOutputUnitProperty_EnableIO,
                kAudioUnitScope_Input,
                1, // input element
                &enableIO,
                sizeof(enableIO));
        if (ret != noErr) {
                fprintf(stderr, "Error enabling input on AUHAL.\n");
                goto error;
        }

        enableIO = 0;
        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                kAudioOutputUnitProperty_EnableIO,
                kAudioUnitScope_Output,
                0,   //output element
                &enableIO,
                sizeof(enableIO));
        if (ret != noErr) {
                fprintf(stderr, "Error disabling output on AUHAL.\n");
                goto error;
        }
        


        size=sizeof(device);
        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                         kAudioOutputUnitProperty_CurrentDevice, 
                         kAudioUnitScope_Global, 
                         0, 
                         &device, 
                         sizeof(device));
        if(ret) {
                fprintf(stderr, "[CoreAudio] Error setting device to AUHAL instance.\n");
                goto error;
        }


        {
                AudioStreamBasicDescription desc;

                size = sizeof(desc);
                ret = AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                                1, &desc, &size);
                if(ret) {
                        fprintf(stderr, "[CoreAudio] Error getting default device properties.\n");
                        goto error;
                }

                desc.mChannelsPerFrame = s->frame.ch_count;
                desc.mSampleRate = (double) s->nominal_sample_rate;
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

                ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output,
                                1, &desc, sizeof(desc));
                if(ret) {
                        fprintf(stderr, "[CoreAudio] Error setting device properties.\n");
                        goto error;
                }

                AURenderCallbackStruct input;
                input.inputProc = InputProc;
                input.inputProcRefCon = s;
                ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioOutputUnitProperty_SetInputCallback,
                                kAudioUnitScope_Global, 0, &input, sizeof(input));
                if(ret) {
                        fprintf(stderr, "[CoreAudio] Error setting input callback.\n");
                        goto error;
                }
        }

        ret = AudioUnitInitialize(s->auHALComponentInstance);
        if(ret) {
                fprintf(stderr, "[CoreAudio] Error initializing device.\n");
                goto error;
        }

        ret = AudioOutputUnitStart(s->auHALComponentInstance);
        if(ret) {
                fprintf(stderr, "[CoreAudio] Error starting device.\n");
                goto error;
        }

        return s;

error:
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        DestroyAudioBufferList(s->theBufferList);
        free(s->frame.data);
        ring_buffer_destroy(s->buffer);
        free(s->tmp);
        free(s);
        return NULL;
}

struct audio_frame *audio_cap_ca_read(void *state)
{
        struct state_ca_capture *s = (struct state_ca_capture *) state;
        int ret = FALSE;

        pthread_mutex_lock(&s->lock);
        ret = ring_buffer_read(s->buffer, s->frame.data, s->frame.max_size);
        if(!ret) {
                s->boss_waiting = TRUE;
                while(!s->data_ready) {
                        pthread_cond_wait(&s->cv, &s->lock);
                }
                s->boss_waiting = FALSE;
                ret = ring_buffer_read(s->buffer, s->frame.data, s->frame.max_size);
                s->data_ready = FALSE;
        }
        pthread_mutex_unlock(&s->lock);

        s->frame.data_len = ret;

        return &s->frame;
}

void audio_cap_ca_finish(void *state)
{
        UNUSED(state);
}

void audio_cap_ca_done(void *state)
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

#ifdef HAVE_SPEEX
        if(s->resampler) {
                speex_resampler_destroy(s->resampler);
        }
        free(s->resampled);
#endif // HAVE_SPEEX

        free(s);
}

#endif /* HAVE_COREAUDIO */


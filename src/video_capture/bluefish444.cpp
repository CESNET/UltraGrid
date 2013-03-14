/*
 * FILE:    video_capture/bluefish444.cpp
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
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <iomanip>
#include <iostream>
#include <queue>

#include "bluefish444_common.h"

#include "audio/audio.h"
#include "debug.h"
#include "host.h"
#include "tv.h"
#include "video.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/bluefish444.h"

#include "video_display/bluefish444.h"

#ifndef UINT
#define UINT uint32_t
#endif

#define BUFFERS 4

#define MAX_BLUE_IN_CHANNELS 4

using namespace std;

struct vidcap_bluefish444_state;

/* prototypes of functions defined in this module */
static void *worker(void *arg);
static void show_help(void);
static void WaitForMajorInterrupt(struct vidcap_bluefish444_state *s);
static void SyncForSignal(struct vidcap_bluefish444_state *s);
#ifdef WIN32
static int CompleteBlueAsyncReq(HANDLE hDevice, LPOVERLAPPED pOverlap);
#endif
static bool should_exit_worker = false;

static void show_help()
{
        int iDevices;
        CBLUEVELVET_H pSDK = bfcFactory();
        bfcEnumerate(pSDK, iDevices);
        bfcDestroy(pSDK);

        printf("Bluefish444 capture\n");
        printf("Usage\n");
        printf("\t-t bluefish444[:framestore|:duplex][:depth={10|8}]"
                        "[:subfield][:4K][:device=<device_id>]\n");

        printf("\t\t4K - use 4 inputs of a SuperNova card\n");

        printf("\t\t<device_id> - ID of the Bluefish device (if more present)\n");
        cout << "\t\t" << iDevices << " Bluefish devices found in this system" << endl;
        if(iDevices == 1) {
                cout << " (default)" << endl;
        } else if(iDevices > 1) {
                cout << ", valid indices [1," << iDevices  << "]" << endl;
                cout << "default is 1" << endl;
        }
}

struct av_frame_base {
        virtual ~av_frame_base() {}
};

struct av_frame_nosignal : public av_frame_base {
};

struct av_frame : public av_frame_base {
        av_frame(struct video_desc desc, int GoldenSize, int BytesPerFrame, int AudioLen) :
                valid(true),
                audio_len(0)
        {
                video = vf_alloc_desc(desc);
                for(int i = 0; i < desc.tile_count; ++i) {
                        video->tiles[i].data_len = BytesPerFrame;
                        video->tiles[i].data = (char*)
                                page_aligned_alloc(GoldenSize);
                }
                if(AudioLen) {
                        audio_data = (char *) page_aligned_alloc(AudioLen);
                } else {
                        audio_data = NULL;
                }
        }

        ~av_frame() {
                page_aligned_free(audio_data);
                for(int i = 0; i < video->tile_count; ++i) {
                        page_aligned_free(video->tiles[i].data);
                }
                vf_free(video);
        }

        char               *audio_data;
        struct video_frame *video;
        bool                valid;
        int                 audio_len;
};

struct vidcap_bluefish444_state {
        CBLUEVELVET_H             pSDK[MAX_BLUE_IN_CHANNELS];
        struct video_desc         video_desc;

        int                       frames;
        struct timeval            t0;

        bool                      is4K;
        int                       iDeviceId;
        int                       attachedDevices;

        uint32_t                  LastFieldCount;
        uint32_t                  InvalidVideoModeFlag;
        uint32_t                  SavedVideoMode;

        uint32_t                  VideoEngine;
        uint32_t                  MemoryFormat;
        uint32_t                  UpdateFormat;
        bool                      SubField;
#ifdef WIN32
        blue_video_sync_struct   *pIrqInfo;
#endif

#ifdef HAVE_BLUE_AUDIO
        struct hanc_decode_struct objHancDecode;
#endif
        bool                      grab_audio;
        struct audio_frame        audio;
        unsigned int             *hanc_buffer;

#ifdef WIN32
        OVERLAPPED                OverlapChA;
#endif

        ULONG                     ScheduleID, CapturingID, DoneID;
        bool                      interlaced;

        struct av_frame_base     *CapturedFrame; // ready to send
        struct av_frame          *NetworkFrame; // frame that is being sent
        queue<struct av_frame*>   FreeFrameQueue;

        pthread_mutex_t           lock;
        pthread_cond_t            CapturedFrameReadyCV;
        pthread_cond_t            CapturedFrameDoneCV;
        pthread_cond_t            FreeFrameQueueNotEmptyCV;

        pthread_t                 worker_id;
};

static void BailOut(CBLUEVELVET_H pSDK);
static void InitInputChannel(CBLUEVELVET_H pSDK, uint32_t DefaultInputChannel,
                uint32_t UpdateFormat, uint32_t MemoryFormat, uint32_t VideoEngine);
void RouteChannel(CBLUEVELVET_H pSDK, uint32_t Source, uint32_t Destination,
                uint32_t LinkType);
bool UpdateVideoMode(struct vidcap_bluefish444_state *s, uint32_t VideoMode);
#ifdef HAVE_BLUE_AUDIO
static bool setup_audio(struct vidcap_bluefish444_state *s, unsigned int flags);
#endif

static void BailOut(CBLUEVELVET_H pSDK)
{
        bfcDetach(pSDK);
#ifdef WIN32
        bfcDestroy(pSDK);
#else
        delete pSDK;
#endif
}

void InitInputChannel(CBLUEVELVET_H pSDK, uint32_t DefaultInputChannel, uint32_t UpdateFormat, uint32_t MemoryFormat, uint32_t VideoEngine)
{
        //MOST IMPORTANT: as the first step set the channel that we want to work with
        bfcSetCardProperty32(pSDK, DEFAULT_VIDEO_INPUT_CHANNEL, DefaultInputChannel);

        //make sure the FIFO hasn't been left running (e.g. application crash before), otherwise we can't change card properties
        bfcVideoCaptureStop(pSDK);

        bfcSetCardProperty32(pSDK, VIDEO_INPUT_UPDATE_TYPE, UpdateFormat);

        bfcSetCardProperty32(pSDK, VIDEO_INPUT_MEMORY_FORMAT, MemoryFormat);

        //Only set the Video Engine after setting up the required update type and memory format and make sure that there is a valid input signal
        bfcSetCardProperty32(pSDK, VIDEO_INPUT_ENGINE, VideoEngine);
}

void RouteChannel(CBLUEVELVET_H pSDK, uint32_t Source, uint32_t Destination, uint32_t LinkType)
{
        uint32_t val = EPOCH_SET_ROUTING(Source, Destination, LinkType);
        bfcSetCardProperty32(pSDK, MR2_ROUTING, val);
}

bool UpdateVideoMode(struct vidcap_bluefish444_state *s, uint32_t VideoMode)
{
        for (int i = 0; i < bluefish_frame_modes_count; ++i) {
                if(bluefish_frame_modes[i].mode == VideoMode) {
                        s->video_desc.interlacing =
                                bluefish_frame_modes[i].interlacing;
                        if(s->video_desc.interlacing == INTERLACED_MERGED) {
                                s->interlaced = true;
                        }  else {
                                s->interlaced = false;
                        }

                        s->video_desc.fps =
                                bluefish_frame_modes[i].fps;
                        s->video_desc.width =
                                bluefish_frame_modes[i].width;
                        s->video_desc.height =
                                bluefish_frame_modes[i].height;

			cout << "[Blue cap] Format changed " <<
                                bluefish_frame_modes[i].width << "x" << 
                                bluefish_frame_modes[i].height <<
				get_interlacing_suffix(bluefish_frame_modes[i].interlacing) <<
				" @" << setprecision(2) << 
                                bluefish_frame_modes[i].fps << endl;
                        return true;
                }
        }
        return false;
}

#ifdef WIN32
static int CompleteBlueAsyncReq(HANDLE hDevice, LPOVERLAPPED pOverlap)
{
        DWORD bytesReturned;
        ResetEvent(pOverlap->hEvent);
        GetOverlappedResult(hDevice, pOverlap, &bytesReturned, TRUE);
        //cout << "Bytes ret: " << bytesReturned << endl;
        return bytesReturned;
}
#endif

static void WaitForMajorInterrupt(struct vidcap_bluefish444_state *s)
{
#ifdef WIN32
        assert(s->attachedDevices == 1);

        BOOL bWaitForField = TRUE;

        //We need to wait for a major interrupt (sub field interrupt == 0) before we schedule a frame to be captured
        UINT SubFieldIrqs = 0;
        UINT VideoMsc = 0;
        DWORD IrqReturn = 0;
        do
        {
                bfcVideoSyncStructSet(s->pSDK[0], s->pIrqInfo, BLUE_VIDEO_INPUT_CHANNEL_A,
                                s->UpdateFormat,
                                IGNORE_SYNC_WAIT_TIMEOUT_VALUE);
                bfcWaitVideoSyncAsync(s->pSDK[0], &s->OverlapChA, s->pIrqInfo);
                IrqReturn = WaitForSingleObject(s->OverlapChA.hEvent, 1000);
                CompleteBlueAsyncReq(bfcGetHandle(s->pSDK[0]), &s->OverlapChA);

                bfcVideoSyncStructGet(s->pSDK[0], s->pIrqInfo, VideoMsc, SubFieldIrqs);
                if(s->video_desc.interlacing == PROGRESSIVE || (VideoMsc & 0x1))
                        bWaitForField = FALSE;

        }while(!((SubFieldIrqs == 0) && !bWaitForField));       //we need to schedule the field capture when SubFieldIrqs is 0
#endif
}

static void SyncForSignal(struct vidcap_bluefish444_state *s)
{
        ULONG FieldCount = 0;

        if(s->VideoEngine == VIDEO_ENGINE_FRAMESTORE) {
                s->ScheduleID = s->CapturingID = s->DoneID = 0;

                if(s->SubField) {
                        WaitForMajorInterrupt(s);
                }

                //start the capture sequence
                // all input channels must be genlocked
                bfcWaitVideoInputSync(s->pSDK[0], s->UpdateFormat, FieldCount);  //this call just synchronises us to the card
                for(int i = 0; i < s->attachedDevices; ++i) {
                        bfcRenderBufferCapture(s->pSDK[i], BlueBuffer_Image_HANC(s->ScheduleID));
                }
                s->CapturingID = s->ScheduleID;
                s->ScheduleID = (++s->ScheduleID%BUFFERS);
                s->LastFieldCount = FieldCount;

                if(s->SubField) {
                        WaitForMajorInterrupt(s);
                }

                bfcWaitVideoInputSync(s->pSDK[0], s->UpdateFormat, FieldCount);  //the first buffer starts to be captured now; this is it's field count
                for(int i = 0; i < s->attachedDevices; ++i) {
                        bfcRenderBufferCapture(s->pSDK[i], BlueBuffer_Image_HANC(s->ScheduleID));
                }
                s->DoneID = s->CapturingID;
                s->CapturingID = s->ScheduleID;
                s->ScheduleID = (++s->ScheduleID%BUFFERS);
                s->LastFieldCount = FieldCount;
        } else {
                for(int i = 0; i < s->attachedDevices; ++i) {
                        bfcWaitVideoInputSync(s->pSDK[i], s->UpdateFormat, FieldCount);
                }
                s->LastFieldCount = FieldCount;
        }
}

struct vidcap_type *
vidcap_bluefish444_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_BLUEFISH444_ID;
		vt->name        = "bluefish444";
		vt->description = "Bluefish444 video capture";
	}
	return vt;
}

#ifdef HAVE_BLUE_AUDIO
static bool setup_audio(struct vidcap_bluefish444_state *s, unsigned int flags)
{
        memset(&s->objHancDecode, 0, sizeof(s->objHancDecode));

        s->objHancDecode.audio_ch_required_mask = 0;
        /* MONO_CHANNEL_9 and _10 are used for analog output */
        switch(audio_capture_channels) {
                case 16:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_18;
                case 15:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_17;
                case 14:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_16;
                case 13:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_15;
                case 12:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_14;
                case 11:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_13;
                case 10:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_12;
                case 9:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_11;
                case 8:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_8;
                case 7:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_7;
                case 6:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_6;
                case 5:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_5;
                case 4:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_4;
                case 3:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_3;
                case 2:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_2;
                case 1:
                        s->objHancDecode.audio_ch_required_mask |= MONO_CHANNEL_1;
                        break;
                default:
                        cerr << "To much outpt channels requested." << endl;
                        return false;
        }
        s->objHancDecode.type_of_sample_required = (AUDIO_CHANNEL_16BIT|AUDIO_CHANNEL_LITTLEENDIAN);
        s->objHancDecode.max_expected_audio_sample_count = 2002; // 1 frame time for 23.98 FPS
        if(flags & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                s->objHancDecode.audio_input_source = AUDIO_INPUT_SOURCE_EMB;
        } else if(flags & VIDCAP_FLAG_AUDIO_AESEBU) {
                s->objHancDecode.audio_input_source = AUDIO_INPUT_SOURCE_AES;
        } else { // flags == VIDCAP_FLAG_AUDIO_ANALOG
                cerr << "[Blue cap] Analog audio not supported." << endl;
                return false;
        }

        s->audio.bps = 2;
        s->audio.ch_count = audio_capture_channels;
        s->audio.sample_rate = 48000; // perhaps the driver does not support different
        s->audio.max_size = 4*4096*16;
        
        s->hanc_buffer = (unsigned int *) page_aligned_alloc(MAX_HANC_SIZE);

        return true;
}
#endif

static void signal_error(struct vidcap_bluefish444_state *s) {
        pthread_mutex_lock(&s->lock);
        if(!s->CapturedFrame) {
                s->CapturedFrame = new av_frame_nosignal;
                pthread_cond_signal(&s->CapturedFrameReadyCV);
        }
        pthread_mutex_unlock(&s->lock);
}

static void *worker(void *arg)
{
        struct vidcap_bluefish444_state *s =
                (struct vidcap_bluefish444_state *) arg;
        uint32_t GoldenSize = 0;
        uint32_t ChunkSize = 0;
        uint32_t nChunks = 1;
        ULONG FieldCount = 0;

        while(!should_exit_worker) {
                unsigned int val32;
                UINT SubFieldIrqs = 0;
                unsigned int FifoSize = 0;
                uint32_t VideoMode = VID_FMT_INVALID;
                unsigned int  DroppedFrameCount, NoFilledFrame, audioChannelMask,
                              frame_timestamp, frame_signal;
                int BufferId = -1;
                static  int field1_hanc_buffer_id=-1;
                int     audioSampleType,audio_samples_per_frame,samples_read,hanc_buffer_id=-1;         // flags required for starting audio capture

#if defined WIN32
                blue_videoframe_info_ex FrameInfo;
#endif

                unsigned int CurrentFieldCount = FieldCount;
                struct av_frame *current_frame = NULL;
                int nOffset = 0;

                // Synchronize
                if(!s->SubField || s->SavedVideoMode == VID_FMT_INVALID) {
                        //Check if we have a valid input signal, all cards should be in sync
                        bfcWaitVideoInputSync(s->pSDK[0], s->UpdateFormat, FieldCount); //synchronise with the card before querying VIDEO_INPUT_SIGNAL_VIDEO_MODE

                        for(int i = 0; i < s->attachedDevices; ++i) {
                                bfcQueryCardProperty32(s->pSDK[i], VIDEO_INPUT_SIGNAL_VIDEO_MODE, val32);
                                if(val32 >= VID_FMT_INVALID)
                                {
                                        cerr << "No valid input signal on channel " <<
                                               (char)('A' + i) << endl;
                                        signal_error(s);
                                        goto next_iteration;
                                }
                                if(i == 0) {
                                        VideoMode = val32;
                                } else {
                                        if(val32 != VideoMode) {
                                                cerr << "Different signal detected on channel " <<
                                                        (char)('A' + i) << " than on " <<
                                                       "channel A" << endl;
                                                signal_error(s);
                                                goto next_iteration;
                                        }
                                }
                        }

                } else {
#ifdef WIN32
                        DWORD IrqReturn = 0;
                        UINT VideoMsc = 0;
                        // we do not allow mode change when in subfield mode
                        VideoMode = s->SavedVideoMode;

                        bfcVideoSyncStructSet(s->pSDK[0], s->pIrqInfo, BLUE_VIDEO_INPUT_CHANNEL_A,
                                        s->UpdateFormat,
                                        IGNORE_SYNC_WAIT_TIMEOUT_VALUE);
                        bfcWaitVideoSyncAsync(s->pSDK[0], &s->OverlapChA, s->pIrqInfo);
                        IrqReturn = WaitForSingleObject(s->OverlapChA.hEvent, 1000);
                        CompleteBlueAsyncReq(bfcGetHandle(s->pSDK[0]), &s->OverlapChA);

                        bfcVideoSyncStructGet(s->pSDK[0], s->pIrqInfo, VideoMsc, SubFieldIrqs);
                        //cerr <<FieldCount <<" " << SubFieldIrqs <<endl;
                        if(SubFieldIrqs == 0) {
                                FieldCount = VideoMsc;
                        }
#endif
                }

                if(s->LastFieldCount + 2 < FieldCount)
                {
                        cout << "Error: dropped " << ((FieldCount - s->LastFieldCount + 2)/2) << " frames" << endl;
                }

                // check for mode change
                if(s->SavedVideoMode != VideoMode) {
                        if(UpdateVideoMode(s, VideoMode)) {
                                // mode changed
                                pthread_mutex_lock(&s->lock);
                                if(s->NetworkFrame) {
                                        s->NetworkFrame->valid = false;
                                }
                                while(s->CapturedFrame) {
                                        pthread_cond_wait(&s->CapturedFrameDoneCV, &s->lock);
                                }
                                while(!s->FreeFrameQueue.empty()) {
                                        struct av_frame *frame = s->FreeFrameQueue.front();
                                        s->FreeFrameQueue.pop();
                                        delete frame;
                                }

                                s->SavedVideoMode = VideoMode;

                                uint32_t val32 = 0; // no subfield interrupts
                                if(s->SubField) {
                                        if(s->interlaced) {
                                                s->UpdateFormat = UPD_FMT_FIELD;
                                                val32 = 2;
                                        } else { // progressive
                                                if(s->video_desc.fps > 25) {
                                                        val32 = 2;
                                                } else {
                                                        val32 = 4;
                                                }
                                        }
                                }

                                GoldenSize = BlueVelvetGolden(VideoMode, s->MemoryFormat, s->UpdateFormat);
                                uint32_t BytesPerFrame = BlueVelvetFrameBytes(VideoMode, s->MemoryFormat, s->UpdateFormat);
                                ChunkSize = BytesPerFrame;
                                nChunks = 1;

                                for(int i = 0; i < 2; ++i) {
                                        s->FreeFrameQueue.push(new av_frame(s->video_desc, GoldenSize, BytesPerFrame,
                                                                s->audio.max_size));
                                }
                                pthread_mutex_unlock(&s->lock);

                                for(int i = 0; i < s->attachedDevices; ++i) {
                                        bfcSetCardProperty32(s->pSDK[i], VIDEO_INPUT_UPDATE_TYPE,
                                                        s->UpdateFormat);
                                        bfcSetCardProperty32(s->pSDK[i], EPOCH_SUBFIELD_INPUT_INTERRUPTS,
                                                        val32);
                                }
                                if(val32) {
                                        ChunkSize /= val32;
                                        nChunks = val32;
                                } else {
                                        ChunkSize = GoldenSize;
                                }
                                cout << "Golden       : " << GoldenSize << endl <<
                                        "BytesPerFrame: " << BytesPerFrame << endl <<
                                        "ChunkSize    : " << ChunkSize << endl;

                        } else {
                                cerr << "[Blue422 cap] Fatal: unknown video mode: " << VideoMode << endl;
                                signal_error(s);
                                goto next_iteration;
                        }
                        SyncForSignal(s);
                }

                if(s->VideoEngine == VIDEO_ENGINE_DUPLEX) {
#ifdef WIN32
                        if(BLUE_FAIL(bfcGetCaptureVideoFrameInfoEx(s->pSDK[0], &s->OverlapChA, FrameInfo,
                                                        0, &FifoSize))) {
                                cerr << "Capture frame failed!" << endl;
                                signal_error(s);
                                goto next_iteration;
                        }

                        if(FrameInfo.nVideoSignalType >= s->InvalidVideoModeFlag) {
                                cerr << "Invalid video mode!" << endl;
                                signal_error(s);
                                goto next_iteration;
                        }

                        if(FrameInfo.BufferId == -1) {
                                cerr << "No buffer!" << endl;
                                signal_error(s);
                                goto next_iteration;
                        }
                        BufferId = FrameInfo.BufferId;
#else
                        //Check if we have a valid input signal
                        if(BLUE_FAIL(s->pSDK[0]->video_capture_get_frame(BufferId, DroppedFrameCount,
                                                        NoFilledFrame,
                                                        frame_timestamp,
                                                        frame_signal)) ||
                                        BufferId == -1) {
                                cerr << "Capture frame failed!" << endl;
                                signal_error(s);
                                goto next_iteration;
                        }

#endif
                } else {
                        BufferId = s->DoneID;
                }

                pthread_mutex_lock(&s->lock);
                while(s->FreeFrameQueue.empty()) {
                        pthread_cond_wait(&s->FreeFrameQueueNotEmptyCV, &s->lock);
                }
                current_frame = s->FreeFrameQueue.front();
                s->FreeFrameQueue.pop();
                pthread_mutex_unlock(&s->lock);

                if(s->SubField) {
                        if(SubFieldIrqs == 0) {
                                nOffset = (nChunks - 1) * ChunkSize;
                                current_frame->video->last_fragment = TRUE;
                        } else {
                                nOffset = (SubFieldIrqs - 1) * ChunkSize;
                                current_frame->video->last_fragment = FALSE;
                        }
                        current_frame->video->tiles[0].data_len = ChunkSize;
                        current_frame->video->fragment = TRUE;
                        current_frame->video->tiles[0].offset = nOffset;
                        current_frame->video->frame_fragment_id = CurrentFieldCount & 0x3fff;
                }

                //DMA the frame from the card to our buffer
                for(int i = 0; i < s->attachedDevices; ++i) {
                        bfcSystemBufferReadAsync(s->pSDK[i], (unsigned char *)
                                        current_frame->video->tiles[i].data,
                                        ChunkSize, NULL,
                                        BlueImage_HANC_DMABuffer(BufferId, BLUE_DATA_IMAGE), nOffset);
                }

#ifdef HAVE_BLUE_AUDIO
                if(s->UpdateFormat == UPD_FMT_FRAME) {
                        hanc_buffer_id = BufferId;
                }

                if(s->grab_audio && hanc_buffer_id != -1) {
                        samples_read = 0;
                        bfcSystemBufferReadAsync(s->pSDK[0], (unsigned char *) s->hanc_buffer, MAX_HANC_SIZE, NULL, BlueImage_HANC_DMABuffer(BufferId, BLUE_DATA_HANC));
                        s->objHancDecode.audio_pcm_data_ptr = current_frame->audio_data;
                        bfcDecodeHancFrameEx(s->pSDK[0], bfcQueryCardType(s->pSDK[0]), (unsigned int *) s->hanc_buffer,
                                        &s->objHancDecode);
                        current_frame->audio_len = s->objHancDecode.no_audio_samples *
                                s->audio.bps;
                }
#endif


                if(s->VideoEngine == VIDEO_ENGINE_FRAMESTORE) {
                        if(!s->SubField || SubFieldIrqs == 0) {
                                //tell the card to capture another frame at the next interrupt
                                for(int i = 0; i < s->attachedDevices; ++i) {
                                        bfcRenderBufferCapture(s->pSDK[i],
                                                        BlueBuffer_Image_HANC(s->ScheduleID));
                                }

                                s->DoneID = s->CapturingID;
                                s->CapturingID = s->ScheduleID;
                                s->ScheduleID = (++s->ScheduleID%BUFFERS);
                        }
                }

                s->LastFieldCount = FieldCount;
                if(!s->SubField || SubFieldIrqs == 0) {
                        s->frames++;

                        struct timeval t;
                        gettimeofday(&t, NULL);
                        double seconds = tv_diff(t, s->t0);
                        if (seconds >= 5) {
                                float fps  = s->frames / seconds;
                                fprintf(stderr, "[Blue cap] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                                s->t0 = t;
                                s->frames = 0;
                        }
                }

                pthread_mutex_lock(&s->lock);
                while(s->CapturedFrame) {
                        pthread_cond_wait(&s->CapturedFrameDoneCV, &s->lock);
                }
                s->CapturedFrame = current_frame;
                pthread_cond_signal(&s->CapturedFrameReadyCV);
                pthread_mutex_unlock(&s->lock);

next_iteration:
		;
        }

        return NULL;
}

static void parse_fmt(struct vidcap_bluefish444_state *s, char *fmt) {
        char *item,
             *save_ptr = NULL;
        if(!fmt) {
                return;
        }
        while((item = strtok_r(fmt, ":", &save_ptr))) {
                if(strcasecmp(item, "framestore") == 0) {
                        s->VideoEngine = VIDEO_ENGINE_FRAMESTORE;
                } else if(strcasecmp(item, "duplex") == 0) {
                        s->VideoEngine = VIDEO_ENGINE_DUPLEX;
                } else if(strncasecmp(item, "depth=", strlen("depth=")) == 0) {
                        int depth = atoi(item + strlen("depth="));
                        if(depth == 8) {
                                s->MemoryFormat = MEM_FMT_2VUY;
                        } else if (depth == 10) {
                                s->MemoryFormat = MEM_FMT_V210;
                        } else {
                                cerr << "[Blue cap] Unsupported bit depth: " << depth << endl;
                        }
                } else if(strcasecmp(item, "subfield") == 0) {
                        s->SubField = true;
                } else if(strncasecmp(item, "device=", strlen("device=")) == 0) {
                        s->iDeviceId = atoi(item + strlen("device="));
                } else if(strcasecmp(item, "4K") == 0) {
                        s->is4K = true;
                } else {
                        cerr << "[Blue cap] Unrecognized option: " << item << endl;
                }

                fmt = NULL;
        }
}

void *
vidcap_bluefish444_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_bluefish444_state *s;
        ULONG InputChannels[4] = {
                BLUE_VIDEO_INPUT_CHANNEL_A,
                BLUE_VIDEO_INPUT_CHANNEL_B,
                BLUE_VIDEO_INPUT_CHANNEL_C,
                BLUE_VIDEO_INPUT_CHANNEL_D
        };
        ULONG Sources[4] = { EPOCH_SRC_SDI_INPUT_A,
                EPOCH_SRC_SDI_INPUT_B,
                EPOCH_SRC_SDI_INPUT_C,
                EPOCH_SRC_SDI_INPUT_D
        };
        ULONG Destinations[4] = { EPOCH_DEST_INPUT_MEM_INTERFACE_CHA,
                EPOCH_DEST_INPUT_MEM_INTERFACE_CHB,
                EPOCH_DEST_INPUT_MEM_INTERFACE_CHC,
                EPOCH_DEST_INPUT_MEM_INTERFACE_CHD
        };

        if(init_fmt && strcmp(init_fmt, "help") == 0) {
                show_help();
                return NULL;
        }

	printf("vidcap_bluefish444_init\n");

        s = new vidcap_bluefish444_state;

	if(s == NULL) {
		printf("Unable to allocate bluefish444 capture state\n");
		return NULL;
	}

        s->MemoryFormat = MEM_FMT_2VUY;
        s->VideoEngine = VIDEO_ENGINE_DUPLEX;
        s->UpdateFormat = UPD_FMT_FRAME;
        s->SubField = false;
        s->iDeviceId = 1;
        s->is4K = false;

        parse_fmt(s, init_fmt);

#ifdef WIN32
        memset(&s->OverlapChA, 0, sizeof(s->OverlapChA));
        s->OverlapChA.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
#endif

        int iDevices = 0;
        s->SavedVideoMode = VID_FMT_INVALID;

        if(s->SubField) {
#ifdef WIN32
                s->VideoEngine = VIDEO_ENGINE_FRAMESTORE;
                s->pIrqInfo = bfcNewVideoSyncStruct(s->pSDK);
#else
                cerr << "Subfields not supported under Linux." << endl;
                goto error;
#endif
        }

        if(s->is4K) {
                s->VideoEngine = VIDEO_ENGINE_FRAMESTORE;
                s->attachedDevices = 4;
                s->video_desc.tile_count = 4;
                if(s->VideoEngine == VIDEO_ENGINE_DUPLEX) {
                        cerr << "4K mode is not supported in duplex mode." << endl;
                        goto error;
                }
                if(s->SubField) {
                        cerr << "Subfields are not supported in 4K mode." << endl;
                        goto error;
                }
        } else {
                s->attachedDevices = 1;
        }

        for(int i = 0; i < s->attachedDevices; ++i) {
                s->pSDK[i] = bfcFactory();
        }

        bfcEnumerate(s->pSDK[0], iDevices);
        if(iDevices < 1) {
                cout << "No Bluefish device detected." << endl;
                goto error;
        }

        for(int i = 0; i < s->attachedDevices; ++i) {
                if(BLUE_FAIL(bfcAttach(s->pSDK[i], s->iDeviceId))) {
                        cerr << "Error on device attach (channel " <<
                                (char)('A' + i) << ")"  << endl;
                        goto error;
                }
        }

        if(s->is4K) {
                uint32_t firmware;
                int cardType = bfcQueryCardType(s->pSDK[0]);
                bfcQueryCardProperty32(s->pSDK[0], EPOCH_GET_PRODUCT_ID, firmware);

                if(cardType != CRD_BLUE_SUPER_NOVA) {
                        cerr << "4K supported only for a SuperNova card!" << endl;
                        goto error;
                }
                if(firmware != ORAC_4SDIINPUT_FIRMWARE_PRODUCTID) {
                         cerr << "Wrong firmware; need 4 input channels (QuadIn)";
                         goto error;
                }
        }

        for(int i = 0; i < s->attachedDevices; ++i) {
                InitInputChannel(s->pSDK[i], InputChannels[i], s->UpdateFormat, s->MemoryFormat, s->VideoEngine);
                RouteChannel(s->pSDK[i], Sources[i], Destinations[i], BLUE_CONNECTOR_PROP_SINGLE_LINK);
        }

        //Get VID_FMT_INVALID flag; this enum has changed over time and might be different depending on which driver this application runs on
        unsigned int val32;
        bfcQueryCardProperty32(s->pSDK[0], INVALID_VIDEO_MODE_FLAG, val32);
        s->InvalidVideoModeFlag = val32;

        s->LastFieldCount = 0;

        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        s->video_desc.tile_count = s->attachedDevices;
        if(s->MemoryFormat == MEM_FMT_2VUY) {
                s->video_desc.color_spec = UYVY;
        } else { // MEM_FMT_V210
                s->video_desc.color_spec = v210;
        }
        
        // make sure that capture is stopped
        for(int i = 0; i < s->attachedDevices; ++i) {
                bfcVideoCaptureStop(s->pSDK[i]);
                if(BLUE_FAIL(bfcVideoCaptureStart(s->pSDK[i]))) {
                        /* is this really needed? Sometimes this command keeps failing but
                         * we can stil go on, so ignore this error */
                        //cerr << "Error video capture start failed on channel A" << endl;
                        //goto error;
                }
        }

        gettimeofday(&s->t0, NULL);

        s->grab_audio = false;
        s->audio.data = NULL;
        s->audio.max_size = 0;
        s->hanc_buffer = NULL;
#ifdef HAVE_BLUE_AUDIO
        if(flags) {
                if(s->SubField) {
                        cerr << "[Blue cap] Unable to grab audio in sub-field mode." << endl;
                        goto error;
                }
                bool ret = setup_audio(s, flags);
                if(ret == false) {
                        cerr << "[Blue cap] Unable to setup audio." << endl;
                        goto error;
                } else {
                        s->grab_audio = true;
                }
        }
#else
        if(flags) {
                cerr << "[Blue cap] Unable to capture audio. Support isn't compiled in." << endl;
                goto error;
        }
#endif

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->CapturedFrameReadyCV, NULL);
        pthread_cond_init(&s->CapturedFrameDoneCV, NULL);
        pthread_cond_init(&s->FreeFrameQueueNotEmptyCV, NULL);
        s->NetworkFrame = NULL;
        s->CapturedFrame = NULL;

        if(pthread_create(&s->worker_id, NULL, worker, (void *) s) != 0) {
                cerr << "[Blue cap] Error initializing thread." << endl;
                goto error;
        }

	return s;

error:
        for(int i = 0; i < s->attachedDevices; ++i) {
                BailOut(s->pSDK[i]);
        }
        free(s);
        return NULL;
}

void
vidcap_bluefish444_finish(void *state)
{
        UNUSED(state);
}

void
vidcap_bluefish444_done(void *state)
{
	struct vidcap_bluefish444_state *s = (struct vidcap_bluefish444_state *) state;

	assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        if(s->NetworkFrame) {
                if(s->NetworkFrame->valid) {
                        s->FreeFrameQueue.push(s->NetworkFrame);
                        pthread_cond_signal(&s->FreeFrameQueueNotEmptyCV);
                } else {
                        delete s->NetworkFrame;
                }
                s->NetworkFrame = NULL;
        }
        if(s->CapturedFrame) {
                delete s->CapturedFrame;
                s->CapturedFrame = NULL;
                pthread_cond_signal(&s->CapturedFrameDoneCV);
        }
        should_exit_worker = true;
        pthread_mutex_unlock(&s->lock);
        pthread_join(s->worker_id, NULL);

#ifdef WIN32
        CloseHandle(s->OverlapChA.hEvent);
#endif

        delete s->CapturedFrame;
        delete s->NetworkFrame;
        while(!s->FreeFrameQueue.empty()) {
                delete s->FreeFrameQueue.front();
                s->FreeFrameQueue.pop();
        }

        page_aligned_free(s->hanc_buffer);

        for(int i = 0; i < s->attachedDevices; ++i) {
                bfcVideoCaptureStop(s->pSDK[i]);
                BailOut(s->pSDK[i]);
        }

        pthread_cond_destroy(&s->CapturedFrameReadyCV);
        pthread_cond_destroy(&s->CapturedFrameDoneCV);
        pthread_cond_destroy(&s->FreeFrameQueueNotEmptyCV);

        delete s;
}

struct video_frame *
vidcap_bluefish444_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_bluefish444_state *s = (struct vidcap_bluefish444_state *) state;

        struct av_frame *frame;
        struct video_frame *res = NULL;

        *audio = NULL;

        pthread_mutex_lock(&s->lock);
        if(s->NetworkFrame) {
                if(s->NetworkFrame->valid) {
                        s->FreeFrameQueue.push(s->NetworkFrame);
                        pthread_cond_signal(&s->FreeFrameQueueNotEmptyCV);
                } else {
                        delete s->NetworkFrame;
                }
                s->NetworkFrame = NULL;
        }

        while(!s->CapturedFrame) {
                pthread_cond_wait(&s->CapturedFrameReadyCV, &s->lock);
        }

        if(dynamic_cast<av_frame_nosignal *>(s->CapturedFrame)) {
		delete s->CapturedFrame;
                s->CapturedFrame = NULL;
                pthread_cond_signal(&s->CapturedFrameDoneCV);
                pthread_mutex_unlock(&s->lock);
                return NULL;
        } else {
                frame = dynamic_cast<av_frame *>(s->CapturedFrame);
        }

        if(frame->audio_len) {
                s->audio.data = frame->audio_data;
                s->audio.data_len = frame->audio_len;
                *audio = &s->audio;
        }

        res = frame->video;

        s->NetworkFrame = frame;
        s->CapturedFrame = NULL;
        pthread_cond_signal(&s->CapturedFrameDoneCV);

        // Merge two fields together if needed (subfield, interlaced)
        if(res->fragment) {
                if(res->frame_fragment_id % 2 == 0) {
                        //upper field
                        res->last_fragment = FALSE;
                } else {
                        //bottom field
                        res->tiles[0].offset += res->tiles[0].data_len * 2;
                        res->frame_fragment_id &= 0xfffffffeu;
                }
                if(s->interlaced) {
                        res->interlacing = UPPER_FIELD_FIRST;
                }
        }

        pthread_mutex_unlock(&s->lock);

        return res;
}


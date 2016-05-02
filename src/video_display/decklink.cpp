/**
 * @file   video_display/decklink.cpp
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2016 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#define MOD_NAME "[Decklink display] "

#include "audio/audio.h"
#include "blackmagic_common.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"

#include <chrono>
#include <iomanip>
#include <mutex>
#include <queue>
#include <vector>

#include "DeckLinkAPIVersion.h"

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

static void print_output_modes(IDeckLink *);
static void display_decklink_done(void *state);

#define MAX_DEVICES 4

using namespace std;

namespace {
class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
public:
        // IUnknown needs only a dummy implementation
        virtual HRESULT STDMETHODCALLTYPE        QueryInterface (REFIID , LPVOID *)        { return E_NOINTERFACE;}
        virtual ULONG STDMETHODCALLTYPE          AddRef ()                                                                       {return 1;}
        virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
	{
                if (result == bmdOutputFrameDisplayedLate){
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Late frame\n";
                } else if (result == bmdOutputFrameDropped){
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Dropped frame\n";
                } else if (result == bmdOutputFrameFlushed){
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Flushed frame\n";
                }

		completedFrame->Release();
		return S_OK;
	}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledPlaybackHasStopped (){
        	return S_OK;
	}
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

class DeckLinkFrame;

struct buffer_pool_t {
        queue<DeckLinkFrame *> frame_queue;
        mutex lock;
};

class DeckLinkTimecode : public IDeckLinkTimecode{
                BMDTimecodeBCD timecode;
        public:
                DeckLinkTimecode() : timecode(0) {}
                /* IDeckLinkTimecode */
                virtual BMDTimecodeBCD STDMETHODCALLTYPE GetBCD (void) { return timecode; }
                virtual HRESULT STDMETHODCALLTYPE GetComponents (/* out */ uint8_t *hours, /* out */ uint8_t *minutes, /* out */ uint8_t *seconds, /* out */ uint8_t *frames) { 
                        *frames =   (timecode & 0xf)              + ((timecode & 0xf0) >> 4) * 10;
                        *seconds = ((timecode & 0xf00) >> 8)      + ((timecode & 0xf000) >> 12) * 10;
                        *minutes = ((timecode & 0xf0000) >> 16)   + ((timecode & 0xf00000) >> 20) * 10;
                        *hours =   ((timecode & 0xf000000) >> 24) + ((timecode & 0xf0000000) >> 28) * 10;
                        return S_OK;
                }
                virtual HRESULT STDMETHODCALLTYPE GetString (/* out */ BMD_STR *timecode) { UNUSED(timecode); return E_FAIL; }
                virtual BMDTimecodeFlags STDMETHODCALLTYPE GetFlags (void)        { return bmdTimecodeFlagDefault; }
                virtual HRESULT STDMETHODCALLTYPE GetTimecodeUserBits (/* out */ BMDTimecodeUserBits *userBits) { if (!userBits) return E_POINTER; else return S_OK; }

                /* IUnknown */
                virtual HRESULT STDMETHODCALLTYPE QueryInterface (REFIID , LPVOID *)        {return E_NOINTERFACE;}
                virtual ULONG STDMETHODCALLTYPE         AddRef ()                                                                       {return 1;}
                virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}
                
                void STDMETHODCALLTYPE SetBCD(BMDTimecodeBCD timecode) { this->timecode = timecode; }
};

class DeckLinkFrame : public IDeckLinkMutableVideoFrame
{
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                unique_ptr<char []> data;

                IDeckLinkTimecode *timecode;

                long ref;

                buffer_pool_t &buffer_pool;
        protected:
                DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool);

        public:
                virtual ~DeckLinkFrame();
                static DeckLinkFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool);

                /* IUnknown */
                virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**);
                virtual ULONG STDMETHODCALLTYPE AddRef();
                virtual ULONG STDMETHODCALLTYPE Release();
                
                /* IDeckLinkVideoFrame */
                long STDMETHODCALLTYPE GetWidth (void);
                long STDMETHODCALLTYPE GetHeight (void);
                long STDMETHODCALLTYPE GetRowBytes (void);
                BMDPixelFormat STDMETHODCALLTYPE GetPixelFormat (void);
                BMDFrameFlags STDMETHODCALLTYPE GetFlags (void);
                HRESULT STDMETHODCALLTYPE GetBytes (/* out */ void **buffer);
                
                HRESULT STDMETHODCALLTYPE GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode);
                HRESULT STDMETHODCALLTYPE GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary);
                

                /* IDeckLinkMutableVideoFrame */
                HRESULT STDMETHODCALLTYPE SetFlags(BMDFrameFlags);
                HRESULT STDMETHODCALLTYPE SetTimecode(BMDTimecodeFormat, IDeckLinkTimecode*);
                HRESULT STDMETHODCALLTYPE SetTimecodeFromComponents(BMDTimecodeFormat, uint8_t, uint8_t, uint8_t, uint8_t, BMDTimecodeFlags);
                HRESULT STDMETHODCALLTYPE SetAncillaryData(IDeckLinkVideoFrameAncillary*);
                HRESULT STDMETHODCALLTYPE SetTimecodeUserBits(BMDTimecodeFormat, BMDTimecodeUserBits);
};

class DeckLink3DFrame : public DeckLinkFrame, public IDeckLinkVideoFrame3DExtensions
{
        private:
                DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool);
                unique_ptr<DeckLinkFrame> rightEye; // rightEye ref count is always >= 1 therefore deleted by owner (this class)

        public:
                ~DeckLink3DFrame();
                static DeckLink3DFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool);
                
                /* IUnknown */
                HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**);
                ULONG STDMETHODCALLTYPE AddRef();
                ULONG STDMETHODCALLTYPE Release();

                /* IDeckLinkVideoFrame3DExtensions */
                BMDVideo3DPackingFormat STDMETHODCALLTYPE Get3DPackingFormat();
                HRESULT STDMETHODCALLTYPE GetFrameForRightEye(IDeckLinkVideoFrame**);
};
} // end of unnamed namespace

#define DECKLINK_MAGIC 0x12de326b

struct device_state {
        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        IDeckLinkConfiguration*         deckLinkConfiguration;
};

struct state_decklink {
        uint32_t            magic;

        struct timeval      tv;

        struct device_state state[MAX_DEVICES];

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        DeckLinkTimecode    *timecode;

        struct video_desc   vid_desc;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                stereo;
        bool                initialized;
        bool                emit_timecode;
        int                 devices_cnt;
        unsigned int        play_audio:1;

        BMDPixelFormat      pixelFormat;

        uint32_t            link;

        buffer_pool_t       buffer_pool;

        bool                low_latency;
 };

static void show_help(void);

static void show_help(void)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
        HRESULT                         result;

        printf("Decklink (output) options:\n");
        printf("\t-d decklink[:device=<device_number(s)>][:timecode][:single-link|:dual-link|:quad-link][:3D[:HDMI3DPacking=<packing>]][:audioConsumerLevels={true|false}][:conversion=<fourcc>][:Use1080pNotPsF={true|false}][:low-latency]\n");
        printf("\t\t<device_number(s)> is coma-separated indices of output devices\n");
        printf("\t\tsingle-link/dual-link specifies if the video output will be in a single-link (HD/3G/6G/12G) or in dual-link HD-SDI mode\n");
        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = create_decklink_iterator(true);
        if (deckLinkIterator == NULL) {
                return;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                BMD_STR          deviceNameString = NULL;
                
                // *** Print the model name of the DeckLink card
                result = deckLink->GetModelName(&deviceNameString);
                if (result == S_OK)
                {
                        const char *deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
                        printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
                        release_bmd_api_str(deviceNameString);
                        free((void *) deviceNameCString);

                        print_output_modes(deckLink);
                } else {
                        printf("\ndevice: %d.) (unable to get name)\n\n",numDevices);
                        print_output_modes(deckLink);
                }
                
                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }
        
        deckLinkIterator->Release();

        decklink_uninitialize();

        // If no DeckLink cards were found in the system, inform the user
        if (numDevices == 0)
        {
                log_msg(LOG_LEVEL_WARNING, "\nNo Blackmagic Design devices were found.\n");
                return;
        } 

        printf("\nHDMI 3D packing can be one of following (optional for HDMI 1.4, mandatory for pre-1.4 HDMI):\n");
        printf("\tSideBySideHalf\n");
        printf("\tLineByLine\n");
        printf("\tTopAndBottom\n");
        printf("\tFramePacking\n");
        printf("\tLeftOnly\n");
        printf("\tRightOnly\n");
        printf("\n");

        printf("audioConsumerLevels if set to true sets audio analog level to maximum attenuation on audio output.\n");
        printf("\n");
        print_decklink_version();
        printf("\n");
}


static struct video_frame *
display_decklink_getf(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        assert(s->magic == DECKLINK_MAGIC);

        struct video_frame *out = vf_alloc_desc(s->vid_desc);
        auto deckLinkFrames =  new vector<IDeckLinkMutableVideoFrame *>(s->devices_cnt);
        out->dispose_udata = (void *) deckLinkFrames;
        out->dispose = [](struct video_frame *frame) {
                delete (vector<IDeckLinkMutableVideoFrame *> *) frame->dispose_udata;
                vf_free(frame);
        };

        if (s->initialized) {
                for (unsigned int i = 0; i < s->vid_desc.tile_count; ++i) {
                        const int linesize = vc_get_linesize(s->vid_desc.width, s->vid_desc.color_spec);
                        IDeckLinkMutableVideoFrame *deckLinkFrame = nullptr;
                        lock_guard<mutex> lg(s->buffer_pool.lock);

                        while (!s->buffer_pool.frame_queue.empty()) {
                                auto tmp = s->buffer_pool.frame_queue.front();
                                IDeckLinkMutableVideoFrame *frame;
                                if (s->stereo)
                                        frame = dynamic_cast<DeckLink3DFrame *>(tmp);
                                else
                                        frame = dynamic_cast<DeckLinkFrame *>(tmp);
                                s->buffer_pool.frame_queue.pop();
                                if (!frame || // wrong type
                                                frame->GetWidth() != (long) s->vid_desc.width ||
                                                frame->GetHeight() != (long) s->vid_desc.height ||
                                                frame->GetRowBytes() != linesize ||
                                                frame->GetPixelFormat() != s->pixelFormat) {
                                        delete tmp;
                                } else {
                                        deckLinkFrame = frame;
                                        deckLinkFrame->AddRef();
                                        break;
                                }
                        }
                        if (!deckLinkFrame) {
                                if (s->stereo)
                                        deckLinkFrame = DeckLink3DFrame::Create(s->vid_desc.width,
                                                        s->vid_desc.height, linesize,
                                                        s->pixelFormat, s->buffer_pool);
                                else
                                        deckLinkFrame = DeckLinkFrame::Create(s->vid_desc.width,
                                                        s->vid_desc.height, linesize,
                                                        s->pixelFormat, s->buffer_pool);
                        }
                        (*deckLinkFrames)[i] = deckLinkFrame;

                        deckLinkFrame->GetBytes((void **) &out->tiles[i].data);

                        if (s->stereo) {
                                IDeckLinkVideoFrame     *deckLinkFrameRight = nullptr;
                                dynamic_cast<DeckLink3DFrame *>(deckLinkFrame)->GetFrameForRightEye(&deckLinkFrameRight);
                                deckLinkFrameRight->GetBytes((void **) &out->tiles[1].data);
                                // release immedieatelly (parent still holds the reference)
                                deckLinkFrameRight->Release();

                                ++i;
                        }
                }
        }

        return out;
}

static void update_timecode(DeckLinkTimecode *tc, double fps)
{
        const float epsilon = 0.005;
        uint8_t hours, minutes, seconds, frames;
        BMDTimecodeBCD bcd;
        bool dropFrame = false;

        if(ceil(fps) - fps > epsilon) { /* NTSCi drop framecode  */
                dropFrame = true;
        }

        tc->GetComponents (&hours, &minutes, &seconds, &frames);
        frames++;

        if((double) frames > fps - epsilon) {
                frames = 0;
                seconds++;
                if(seconds >= 60) {
                        seconds = 0;
                        minutes++;
                        if(dropFrame) {
                                if(minutes % 10 != 0)
                                        seconds = 2;
                        }
                        if(minutes >= 60) {
                                minutes = 0;
                                hours++;
                                if(hours >= 24) {
                                        hours = 0;
                                }
                        }
                }
        }

        bcd = (frames % 10) | (frames / 10) << 4 | (seconds % 10) << 8 | (seconds / 10) << 12 | (minutes % 10)  << 16 | (minutes / 10) << 20 |
                (hours % 10) << 24 | (hours / 10) << 28;

        tc->SetBCD(bcd);
}

static int display_decklink_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_decklink *s = (struct state_decklink *)state;
        struct timeval tv;

        if (frame == NULL)
                return FALSE;

        UNUSED(nonblock);

        assert(s->magic == DECKLINK_MAGIC);

        gettimeofday(&tv, NULL);

#ifdef WIN32
        long unsigned int i;
#else
        uint32_t i;
#endif

        s->state[0].deckLinkOutput->GetBufferedVideoFrameCount(&i);

        auto t0 = chrono::high_resolution_clock::now();

        //if (i > 2) 
        if (0) 
                fprintf(stderr, "Frame dropped!\n");
        else {
                for (int j = 0; j < s->devices_cnt; ++j) {
                        IDeckLinkMutableVideoFrame *deckLinkFrame =
                                (*((vector<IDeckLinkMutableVideoFrame *> *) frame->dispose_udata))[j];
                        if(s->emit_timecode) {
                                deckLinkFrame->SetTimecode(bmdTimecodeRP188Any, s->timecode);
                        }

                        if (s->low_latency) {
                                s->state[j].deckLinkOutput->DisplayVideoFrameSync(deckLinkFrame);
                                deckLinkFrame->Release();
                        } else {
                                s->state[j].deckLinkOutput->ScheduleVideoFrame(deckLinkFrame,
                                                s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                        }
                }
                s->frames++;
                if(s->emit_timecode) {
                        update_timecode(s->timecode, s->vid_desc.fps);
                }
        }

        frame->dispose(frame);

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "putf - " << i << " frames buffered, lasted " << setprecision(2) << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() * 1000.0 << " ms.\n";

        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = (s->frames - s->frames_last) / seconds;
                log_msg(LOG_LEVEL_INFO, MOD_NAME "%lu frames in %g seconds = %g FPS\n",
                        s->frames - s->frames_last, seconds, fps);
                s->tv = tv;
                s->frames_last = s->frames;
        }

        return 0;
}

static BMDDisplayMode get_mode(IDeckLinkOutput *deckLinkOutput, struct video_desc desc, BMDTimeValue *frameRateDuration,
		BMDTimeScale        *frameRateScale)
{	IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;
        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fatal: cannot create display mode iterator.\n");
                return (BMDDisplayMode) -1;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                BMD_STR modeNameString;
                if (deckLinkDisplayMode->GetName(&modeNameString) == S_OK)
                {
                        const char *modeNameCString = get_cstr_from_bmd_api_str(modeNameString);
                        if (deckLinkDisplayMode->GetWidth() == (long) desc.width &&
                                        deckLinkDisplayMode->GetHeight() == (long) desc.height)
                        {
                                double displayFPS;
                                BMDFieldDominance dominance;
                                bool interlaced;

                                dominance = deckLinkDisplayMode->GetFieldDominance();
                                if (dominance == bmdLowerFieldFirst ||
                                                dominance == bmdUpperFieldFirst) {
					if (dominance == bmdLowerFieldFirst) {
						log_msg(LOG_LEVEL_WARNING, MOD_NAME "Lower field first format detected, fields can be switched! If so, please report a bug to " PACKAGE_BUGREPORT "\n");
					}
                                        interlaced = true;
                                } else { // progressive, psf, unknown
                                        interlaced = false;
                                }

                                deckLinkDisplayMode->GetFrameRate(frameRateDuration,
                                                frameRateScale);
                                displayFPS = (double) *frameRateScale / *frameRateDuration;
                                if(fabs(desc.fps - displayFPS) < 0.01 && (desc.interlacing == INTERLACED_MERGED ? interlaced : !interlaced)
                                  )
                                {
                                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Selected mode: %s\n", modeNameCString);
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        break;
                                }
                        }
                        release_bmd_api_str(modeNameString);
                        free((void *) modeNameCString);
                }
                deckLinkDisplayMode->Release();
        }
        displayModeIterator->Release();
        
        return displayMode;
}

static int
display_decklink_reconfigure(void *state, struct video_desc desc)
{
        struct state_decklink            *s = (struct state_decklink *)state;
        
        BMDDisplayMode                    displayMode;
        BMDDisplayModeSupport             supported;

        assert(s->magic == DECKLINK_MAGIC);
        
        s->vid_desc = desc;

	switch (desc.color_spec) {
                case UYVY:
                        s->pixelFormat = bmdFormat8BitYUV;
                        break;
                case v210:
                        s->pixelFormat = bmdFormat10BitYUV;
                        break;
                case RGBA:
                        s->pixelFormat = bmdFormat8BitBGRA;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported pixel format!\n");
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->state[0].deckLinkOutput->StopScheduledPlayback (0, NULL, 0);
        }

	if(s->stereo) {
		displayMode = get_mode(s->state[0].deckLinkOutput, desc, &s->frameRateDuration,
                                                &s->frameRateScale);
                if(displayMode == (BMDDisplayMode) -1)
                        goto error;
		
		s->state[0].deckLinkOutput->DoesSupportVideoMode(displayMode, s->pixelFormat, bmdVideoOutputDualStream3D,
	                                &supported, NULL);
                if(supported == bmdDisplayModeNotSupported)
                {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested parameters combination not supported - %dx%d@%f.\n", desc.width, desc.height, (double)desc.fps);
                        goto error;
                }
                
                s->state[0].deckLinkOutput->EnableVideoOutput(displayMode,  bmdVideoOutputDualStream3D);
                if (!s->low_latency) {
                        s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
                }
        } else {
                if((int) desc.tile_count > s->devices_cnt) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected at most %d streams. Got %d.\n", s->devices_cnt,
                                        desc.tile_count);
                        goto error;
                }

	        for(int i = 0; i < s->devices_cnt; ++i) {
                        BMDVideoOutputFlags outputFlags= bmdVideoOutputFlagDefault;
	                
	                displayMode = get_mode(s->state[i].deckLinkOutput, desc, &s->frameRateDuration,
                                                &s->frameRateScale);
                        if(displayMode == (BMDDisplayMode) -1)
                                goto error;

                        if(s->emit_timecode) {
                                outputFlags = bmdVideoOutputRP188;
                        }
	
	                s->state[i].deckLinkOutput->DoesSupportVideoMode(displayMode, s->pixelFormat, outputFlags,
	                                &supported, NULL);
	                if(supported == bmdDisplayModeNotSupported)
	                {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested parameters "
                                                "combination not supported - %d * %dx%d@%f, timecode %s.\n",
                                                desc.tile_count, desc.width, desc.height, desc.fps,
                                                (outputFlags & bmdVideoOutputRP188 ? "ON": "OFF"));
	                        goto error;
	                }

	                s->state[i].deckLinkOutput->EnableVideoOutput(displayMode, outputFlags);
	        }
	
                if (!s->low_latency) {
                        for(int i = 0; i < s->devices_cnt; ++i) {
                                s->state[i].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
                        }
                }
	}

        s->initialized = true;
        return TRUE;

error:
        return FALSE;
}

static void display_decklink_probe(struct device_info **available_cards, int *count)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;

        *count = 0;
        *available_cards = nullptr;

        deckLinkIterator = create_decklink_iterator(false);
        if (deckLinkIterator == NULL) {
                return;
        }

        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                BMD_STR          deviceNameString = NULL;

                // *** Print the model name of the DeckLink card
                HRESULT result = deckLink->GetModelName(&deviceNameString);

                *count += 1;
                *available_cards = (struct device_info *)
                        realloc(*available_cards, *count * sizeof(struct device_info));
                memset(*available_cards + *count - 1, 0, sizeof(struct device_info));
                sprintf((*available_cards)[*count - 1].id, "decklink:device=%d", *count - 1);
                (*available_cards)[*count - 1].repeatable = false;

                if (result == S_OK)
                {
                        const char *deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
                        strncpy((*available_cards)[*count - 1].name, deviceNameCString,
                                        sizeof (*available_cards)[*count - 1].name - 1);
                        release_bmd_api_str(deviceNameString);
                        free((void *) deviceNameCString);
                }

                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        deckLinkIterator->Release();
        decklink_uninitialize();
}

static bool parse_devices(const char *devices_str, int *cardIdx, int *devices_cnt) {
        if (strlen(devices_str) == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Empty device string!\n");
                return false;
        }
        char *save_ptr;
        char *tmp = strdup(devices_str);
        char *ptr = tmp;
        *devices_cnt = 0;
        char *item;
        while ((item = strtok_r(ptr, ",", &save_ptr))) {
                cardIdx[*devices_cnt] = atoi(item);
                ++*devices_cnt;
                ptr = NULL;
        }
        free(tmp);

        return true;
}

static void *display_decklink_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        struct state_decklink *s;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        int                                             cardIdx[MAX_DEVICES];
        int                                             dnum = 0;
        IDeckLinkConfiguration*         deckLinkConfiguration = NULL;
        // for Decklink Studio which has switchable XLR - analog 3 and 4 or AES/EBU 3,4 and 5,6
        BMDAudioOutputAnalogAESSwitch audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
        BMDVideo3DPackingFormat HDMI3DPacking = (BMDVideo3DPackingFormat) 0;
        int audio_consumer_levels = -1;
        BMDVideoOutputConversionMode conversion_mode = bmdNoVideoOutputConversion;
        bool use1080p_not_psf = true;

        if (!blackmagic_api_version_check()) {
                return NULL;
        }

        s = new state_decklink();
        s->magic = DECKLINK_MAGIC;
        s->stereo = FALSE;
        s->emit_timecode = false;
        s->link = 0;
        cardIdx[0] = 0;
        s->devices_cnt = 1;

        if(fmt == NULL || strlen(fmt) == 0) {
                fprintf(stderr, "Card number unset, using first found (see -d decklink:help)!\n");

        } else if (strcmp(fmt, "help") == 0) {
                show_help();
                delete s;
                return &display_init_noerr;
        } else {
                char *tmp = strdup(fmt);
                char *ptr;
                char *save_ptr = 0ul;

                ptr = strtok_r(tmp, ":", &save_ptr);
                assert(ptr);
                int i = 0;
                bool first_option_is_device = true;
                while (ptr[i] != '\0') {
                        if (isdigit(ptr[i++]))
                                continue;
                        else
                                first_option_is_device = false;
                }
                if (first_option_is_device) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unnamed device index "
                                        "deprecated. Use \"device=%s\" instead.\n", ptr);
                        if (!parse_devices(ptr, cardIdx, &s->devices_cnt)) {
                                delete s;
                                return NULL;
                        }
                        ptr = strtok_r(NULL, ":", &save_ptr);
                }
                
                while (ptr)  {
                        if (strncasecmp(ptr, "device=", strlen("device=")) == 0) {
                               if (!parse_devices(ptr + strlen("device="), cardIdx, &s->devices_cnt)) {
                                       delete s;
                                       return NULL;
                               }

                        } else if (strcasecmp(ptr, "3D") == 0) {
                                s->stereo = true;
                        } else if(strcasecmp(ptr, "timecode") == 0) {
                                s->emit_timecode = true;
                        } else if(strcasecmp(ptr, "single-link") == 0) {
                                s->link = to_fourcc('l', 'c', 's', 'l');
                        } else if(strcasecmp(ptr, "dual-link") == 0) {
                                s->link = to_fourcc('l', 'c', 'd', 'l');
                        } else if(strcasecmp(ptr, "quad-link") == 0) {
                                s->link = to_fourcc('l', 'c', 'q', 'l');
                        } else if(strncasecmp(ptr, "HDMI3DPacking=", strlen("HDMI3DPacking=")) == 0) {
                                char *packing = ptr + strlen("HDMI3DPacking=");
                                if(strcasecmp(packing, "SideBySideHalf") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingSidebySideHalf;
                                } else if(strcasecmp(packing, "LineByLine") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingLinebyLine;
                                } else if(strcasecmp(packing, "TopAndBottom") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingTopAndBottom;
                                } else if(strcasecmp(packing, "FramePacking") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingFramePacking;
                                } else if(strcasecmp(packing, "LeftOnly") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingRightOnly;
                                } else if(strcasecmp(packing, "RightOnly") == 0) {
                                        HDMI3DPacking = bmdVideo3DPackingLeftOnly;
                                } else {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown HDMI 3D packing %s.\n", packing);
                                        delete s;
                                        return NULL;
                                }
                        } else if(strncasecmp(ptr, "audioConsumerLevels=", strlen("audioConsumerLevels=")) == 0) {
                                if (strcasecmp(ptr + strlen("audioConsumerLevels="), "false") == 0) {
                                        audio_consumer_levels = 0;
                                } else {
                                        audio_consumer_levels = 1;
                                }
			} else if(strncasecmp(ptr, "conversion=",
						strlen("conversion=")) == 0) {
				const char *conversion_mode_str = ptr + strlen("conversion=");

				union {
					uint32_t fourcc;
					char tmp[4];
				};
				memcpy(tmp, conversion_mode_str, max(strlen(conversion_mode_str), sizeof(tmp)));
				conversion_mode = (BMDVideoOutputConversionMode) htonl(fourcc);

			} else if(strncasecmp(ptr, "Use1080pNotPsF=",
						strlen("Use1080pNotPsF=")) == 0) {
				const char *levels = ptr + strlen("Use1080pNotPsF=");
				if (strcasecmp(levels, "false") == 0) {
					use1080p_not_psf = false;
				} else {
					use1080p_not_psf = true;
				}
                        } else if (strcasecmp(ptr, "low-latency") == 0) {
                                s->low_latency = true;
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Warning: unknown options in config string.\n");
                                free(tmp);
                                delete s;
                                return NULL;
                        }
                        ptr = strtok_r(NULL, ":", &save_ptr);
                }
                free (tmp);
        }
	assert(!s->stereo || s->devices_cnt == 1);

        gettimeofday(&s->tv, NULL);

        if (!s->low_latency) {
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Consider using low-latency mode "
                        "since it will become default in future.\n";
        }

        // Initialize the DeckLink API
        deckLinkIterator = create_decklink_iterator(true);
        if (!deckLinkIterator)
        {
                delete s;
                return NULL;
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->state[i].delegate = NULL;
                s->state[i].deckLink = NULL;
                s->state[i].deckLinkOutput = NULL;
                s->state[i].deckLinkConfiguration = NULL;
        }

        // Connect to the first DeckLink instance
        IDeckLink    *deckLink;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                bool found = false;
                for(int i = 0; i < s->devices_cnt; ++i) {
                        if (dnum == cardIdx[i]){
                                s->state[i].deckLink = deckLink;
                                found = true;
                        }
                }
                if(!found && deckLink != NULL)
                        deckLink->Release();
                dnum++;
        }
        for(int i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].deckLink == NULL) {
                        log_msg(LOG_LEVEL_ERROR, "No DeckLink PCI card #%d found\n", cardIdx[i]);
                        goto error;
                }
        }

        if(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                s->play_audio = TRUE;
                switch(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                        case DISPLAY_FLAG_AUDIO_EMBEDDED:
                                audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
                                break;
                        case DISPLAY_FLAG_AUDIO_AESEBU:
                                audioConnection = bmdAudioOutputSwitchAESEBU;
                                break;
                        case DISPLAY_FLAG_AUDIO_ANALOG:
                                audioConnection = bmdAudioOutputSwitchAnalog;
                                break;
                        default:
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupporetd audio connection.\n");
                                abort();
                }
        } else {
                s->play_audio = FALSE;
        }
        
        if(s->emit_timecode) {
                s->timecode = new DeckLinkTimecode;
        } else {
                s->timecode = NULL;
        }
        
        for(int i = 0; i < s->devices_cnt; ++i) {
                // Obtain the audio/video output interface (IDeckLinkOutput)
                if ((result = s->state[i].deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->state[i].deckLinkOutput)) != S_OK) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not obtain the IDeckLinkOutput interface: %08x\n", (int) result);
                        goto error;
                }

                // Query the DeckLink for its configuration interface
                result = s->state[i].deckLink->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
                s->state[i].deckLinkConfiguration = deckLinkConfiguration;
                if (result != S_OK)
                {
                        log_msg(LOG_LEVEL_ERROR, "Could not obtain the IDeckLinkConfiguration interface: %08x\n", (int) result);
                        goto error;
                }

		result = deckLinkConfiguration->SetInt(bmdDeckLinkConfigVideoOutputConversionMode, conversion_mode);
		if (result != S_OK) {
			log_msg(LOG_LEVEL_ERROR, "Unable to set conversion mode.\n");
			goto error;
		}

		result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigUse1080pNotPsF, use1080p_not_psf);
		if (result != S_OK) {
			log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to set 1080p P/PsF mode.\n");
		}

                result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigLowLatencyVideoOutput, s->low_latency);
                if (result != S_OK) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to set to low-latency mode.\n");
                        goto error;
                }

                if(HDMI3DPacking != 0) {
                        HRESULT res = deckLinkConfiguration->SetInt(bmdDeckLinkConfigHDMI3DPackingFormat,
                                        HDMI3DPacking);
                        if(res != S_OK) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable set 3D packing.\n");
                        }
                }

                if(s->link != 0) {
#if BLACKMAGIC_DECKLINK_API_VERSION < ((10 << 24) | (5 << 16))
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Compiled with old SDK - dual-/quad-link setting might be incorrect.\n");
                        HRESULT res = deckLinkConfiguration->SetFlag(bmdDeckLinkConfig3GBpsVideoOutput,
                                        s->link == to_fourcc('l', 'c', 's', 'l') ? true : false);
#else
                        HRESULT res = deckLinkConfiguration->SetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, s->link);
#endif
                        if(res != S_OK) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable set output SDI standard.\n");
                        }
                }

                if(s->play_audio == FALSE || i != 0) { //TODO: figure out output from multiple streams
                                s->state[i].deckLinkOutput->DisableAudioOutput();
                } else {
                        /* Actually no action is required to set audio connection because Blackmagic card plays audio through all its outputs (AES/SDI/analog) ....
                         */
                        printf(MOD_NAME "Audio output set to: ");
                        switch(audioConnection) {
                                case 0:
                                        printf("embedded");
                                        break;
                                case bmdAudioOutputSwitchAESEBU:
                                        printf("AES/EBU");
                                        break;
                                case bmdAudioOutputSwitchAnalog:
                                        printf("analog");
                                        break;
                        }
                        printf(".\n");
                         /*
                          * .... one exception is a card that has switchable cables between AES/EBU and analog. (But this applies only for channels 3 and above.)
                         */
                        if (audioConnection != 0) { // not embedded 
                                result = deckLinkConfiguration->SetInt(bmdDeckLinkConfigAudioOutputAESAnalogSwitch,
                                                audioConnection);
                                if(result == S_OK) { // has switchable channels
                                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Card with switchable audio channels detected. Switched to correct format.\n");
                                } else if(result == E_NOTIMPL) {
                                        // normal case - without switchable channels
                                } else {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to switch audio output for channels 3 or above although \n"
                                                        "card shall support it. Check if it is ok. Continuing anyway.\n");
                                }
                        }

                        if (audio_consumer_levels != -1) {
                                result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                                audio_consumer_levels == 1 ? true : false);
                                if(result != S_OK) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable set output audio consumer levels.\n");
                                }
                        }
                }

                s->state[i].delegate = new PlaybackDelegate();
                // Provide this class as a delegate to the audio and video output interfaces
                if (!s->low_latency) {
                        s->state[i].deckLinkOutput->SetScheduledFrameCompletionCallback(s->state[i].delegate);
                }
                //s->state[i].deckLinkOutput->DisableAudioOutput();
        }

        s->frames = 0;
        s->initialized = false;

        return (void *)s;
error:
        display_decklink_done(s);
        return NULL;
}

static void display_decklink_run(void *state)
{
        UNUSED(state);
}

static void display_decklink_done(void *state)
{
        debug_msg("display_decklink_done\n"); /* TOREMOVE */
        struct state_decklink *s = (struct state_decklink *)state;
        HRESULT result;

        assert (s != NULL);

        delete s->timecode;

        for (int i = 0; i < s->devices_cnt; ++i)
        {
                if(s->initialized) {
                        if (!s->low_latency) {
                                result = s->state[i].deckLinkOutput->StopScheduledPlayback (0, NULL, 0);
                                if (result != S_OK) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Cannot stop playback: %08x\n", (int) result);
                                }
                        }

                        if(s->play_audio && i == 0) {
                                result = s->state[i].deckLinkOutput->DisableAudioOutput();
                                if (result != S_OK) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Could disable audio output: %08x\n", (int) result);
                                }
                        }
                        result = s->state[i].deckLinkOutput->DisableVideoOutput();
                        if (result != S_OK) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Could disable video output: %08x\n", (int) result);
                        }
                }

                if(s->state[i].deckLinkConfiguration != NULL) {
                        s->state[i].deckLinkConfiguration->Release();
                }

                if(s->state[i].deckLinkOutput != NULL) {
                        s->state[i].deckLinkOutput->Release();
                }

                if(s->state[i].deckLink != NULL) {
                        s->state[i].deckLink->Release();
                }

                if(s->state[i].delegate != NULL) {
                        delete s->state[i].delegate;
                }
        }

        while (!s->buffer_pool.frame_queue.empty()) {
                auto tmp = s->buffer_pool.frame_queue.front();
                s->buffer_pool.frame_queue.pop();
                delete tmp;
        }

        delete s;

        decklink_uninitialize();
}

static int display_decklink_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decklink *s = (struct state_decklink *)state;
        codec_t codecs[] = {v210, UYVY, RGBA};
        int rgb_shift[] = {16, 8, 0};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        if(s->devices_cnt == 1 && !s->stereo)
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                        else
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                desc->ch_count = 2;
                                if (desc->ch_count > 2) {
                                        desc->ch_count = 8;
                                }
                                if (desc->ch_count > 8) {
                                        desc->ch_count = 16;
                                }
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 4;
                        }
                                break;
                default:
                        return FALSE;
        }
        return TRUE;
}

/*
 * AUDIO
 */
static void display_decklink_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = frame->data_len / (frame->bps *
                        frame->ch_count);

        if(!s->play_audio)
                return;

#ifdef WIN32
        unsigned long int sampleFramesWritten;
#else
        unsigned int sampleFramesWritten;
#endif

        auto t0 = chrono::high_resolution_clock::now();

        if (s->low_latency) {
                HRESULT res = s->state[0].deckLinkOutput->WriteAudioSamplesSync(frame->data, sampleFrameCount,
                                &sampleFramesWritten);
                if (FAILED(res)) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WriteAudioSamplesSync failed.\n");
                }
        } else {
                s->state[0].deckLinkOutput->ScheduleAudioSamples(frame->data, sampleFrameCount, 0,
                                0, &sampleFramesWritten);
                if(sampleFramesWritten != sampleFrameCount)
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "audio buffer underflow!\n");
        }

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "putf audio - lasted " << setprecision(2) << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() * 1000.0 << " ms.\n";
}

static int display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        if (channels != 2 && channels != 8 &&
                        channels != 16) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "requested channel count isn't supported: "
                        "%d\n", channels);
                s->play_audio = FALSE;
                return FALSE;
        }
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        sample_rate != 48000) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "audio format isn't supported: "
                        "samples: %d, sample rate: %d\n",
                        quant_samples, sample_rate);
                s->play_audio = FALSE;
                return FALSE;
        }
        switch(quant_samples) {
                case 16:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 32:
                        sample_type = bmdAudioSampleType32bitInteger;
                        break;
                default:
                        return FALSE;
        }
                        
        s->state[0].deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        channels,
                        bmdAudioOutputStreamContinuous);
        if (!s->low_latency) {
                s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration);
        }
        
        return TRUE;
}

#ifndef WIN32
static bool operator==(const REFIID & first, const REFIID & second){
    return (first.byte0 == second.byte0) &&
        (first.byte1 == second.byte1) &&
        (first.byte2 == second.byte2) &&
        (first.byte3 == second.byte3) &&
        (first.byte4 == second.byte4) &&
        (first.byte5 == second.byte5) &&
        (first.byte6 == second.byte6) &&
        (first.byte7 == second.byte7) &&
        (first.byte8 == second.byte8) &&
        (first.byte9 == second.byte9) &&
        (first.byte10 == second.byte10) &&
        (first.byte11 == second.byte11) &&
        (first.byte12 == second.byte12) &&
        (first.byte13 == second.byte13) &&
        (first.byte14 == second.byte14) &&
        (first.byte15 == second.byte15);
}
#endif

HRESULT DeckLinkFrame::QueryInterface(REFIID, void**)
{
        return E_NOINTERFACE;
}

ULONG DeckLinkFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLinkFrame::Release()
{
        if (--ref == 0) {
                lock_guard<mutex> lg(buffer_pool.lock);
                buffer_pool.frame_queue.push(this);
        }
	return ref;
}

DeckLinkFrame::DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & bp)
	: width(w), height(h), rawBytes(rb), pixelFormat(pf), data(new char[rb * h]), timecode(NULL), ref(1l),
        buffer_pool(bp)
{
        clear_video_buffer(reinterpret_cast<unsigned char *>(data.get()), rawBytes, rawBytes, height,
                        pf == bmdFormat8BitYUV ? UYVY : (pf == bmdFormat10BitYUV ? v210 : RGBA));
}

DeckLinkFrame *DeckLinkFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool)
{
        return new DeckLinkFrame(width, height, rawBytes, pixelFormat, buffer_pool);
}

DeckLinkFrame::~DeckLinkFrame() 
{
}

long DeckLinkFrame::GetWidth ()
{
        return width;
}

long DeckLinkFrame::GetHeight ()
{
        return height;
}

long DeckLinkFrame::GetRowBytes ()
{
        return rawBytes;
}

BMDPixelFormat DeckLinkFrame::GetPixelFormat ()
{
        return pixelFormat;
}

BMDFrameFlags DeckLinkFrame::GetFlags ()
{
        return bmdFrameFlagDefault;
}

HRESULT DeckLinkFrame::GetBytes (/* out */ void **buffer)
{
        *buffer = static_cast<void *>(data.get());
        return S_OK;
}

HRESULT DeckLinkFrame::GetTimecode (/* in */ BMDTimecodeFormat, /* out */ IDeckLinkTimecode **timecode)
{
        *timecode = dynamic_cast<IDeckLinkTimecode *>(this->timecode);
        return S_OK;
}

HRESULT DeckLinkFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **)
{
	return S_FALSE;
}

/* IDeckLinkMutableVideoFrame */
HRESULT DeckLinkFrame::SetFlags (/* in */ BMDFrameFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecode (/* in */ BMDTimecodeFormat, /* in */ IDeckLinkTimecode *timecode)
{
        if(this->timecode)
                this->timecode->Release();
        this->timecode = timecode;
        return S_OK;
}

HRESULT DeckLinkFrame::SetTimecodeFromComponents (/* in */ BMDTimecodeFormat, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ BMDTimecodeFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetAncillaryData (/* in */ IDeckLinkVideoFrameAncillary *)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecodeUserBits (/* in */ BMDTimecodeFormat, /* in */ BMDTimecodeUserBits)
{
        return E_FAIL;
}



DeckLink3DFrame::DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool)
        : DeckLinkFrame(w, h, rb, pf, buffer_pool), rightEye(DeckLinkFrame::Create(w, h, rb, pf, buffer_pool))
{
}

DeckLink3DFrame *DeckLink3DFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool)
{
        DeckLink3DFrame *frame = new DeckLink3DFrame(width, height, rawBytes, pixelFormat, buffer_pool);
        return frame;
}

DeckLink3DFrame::~DeckLink3DFrame()
{
}

ULONG DeckLink3DFrame::AddRef()
{
        return DeckLinkFrame::AddRef();
}

ULONG DeckLink3DFrame::Release()
{
        return DeckLinkFrame::Release();
}

HRESULT DeckLink3DFrame::QueryInterface(REFIID id, void**frame)
{
        HRESULT result = E_NOINTERFACE;

        if(id == IID_IDeckLinkVideoFrame3DExtensions)
        {
                this->AddRef();
                *frame = dynamic_cast<IDeckLinkVideoFrame3DExtensions *>(this);
                result = S_OK;
        }
        return result;
}

BMDVideo3DPackingFormat DeckLink3DFrame::Get3DPackingFormat()
{
        return bmdVideo3DPackingLeftOnly;
}

HRESULT DeckLink3DFrame::GetFrameForRightEye(IDeckLinkVideoFrame ** frame) 
{
        *frame = rightEye.get();
        rightEye->AddRef();
        return S_OK;
}

/* function from DeckLink SDK sample DeviceList */

static void print_output_modes (IDeckLink* deckLink)
{
        IDeckLinkOutput*                        deckLinkOutput = NULL;
        IDeckLinkDisplayModeIterator*           displayModeIterator = NULL;
        IDeckLinkDisplayMode*                   displayMode = NULL;
        HRESULT                                 result;
        int                                     displayModeNumber = 0;

        // Query the DeckLink for its configuration interface
        result = deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&deckLinkOutput);
        if (result != S_OK)
        {
                fprintf(stderr, "Could not obtain the IDeckLinkOutput interface - result = %08x\n", (int) result);
                if (result == E_NOINTERFACE) {
                        printf("Device doesn't support video playback.\n");
                }
                goto bail;
        }

        // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on output
        result = deckLinkOutput->GetDisplayModeIterator(&displayModeIterator);
        if (result != S_OK)
        {
                fprintf(stderr, "Could not obtain the video output display mode iterator - result = %08x\n", (int) result);
                goto bail;
        }

        // List all supported output display modes
        printf("display modes:\n");
        while (displayModeIterator->Next(&displayMode) == S_OK)
        {
                BMD_STR                  displayModeString = NULL;

                result = displayMode->GetName(&displayModeString);

                if (result == S_OK)
                {
                        const char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                        int                             modeWidth;
                        int                             modeHeight;
                        BMDDisplayModeFlags             flags;
                        BMDTimeValue    frameRateDuration;
                        BMDTimeScale    frameRateScale;

                        // Obtain the display mode's properties
                        flags = displayMode->GetFlags();
                        modeWidth = displayMode->GetWidth();
                        modeHeight = displayMode->GetHeight();
                        displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
                        printf("%d.) %-20s \t %d x %d \t %2.2f FPS%s\n",displayModeNumber, displayModeCString,
                                        modeWidth, modeHeight, (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (flags & bmdDisplayModeSupports3D ? "\t (supports 3D)" : ""));
                        release_bmd_api_str(displayModeString);
                        free((void *) displayModeCString);
                }

                // Release the IDeckLinkDisplayMode object to prevent a leak
                displayMode->Release();

                displayModeNumber++;
        }

bail:
        // Ensure that the interfaces we obtained are released to prevent a memory leak
        if (displayModeIterator != NULL)
                displayModeIterator->Release();

        if (deckLinkOutput != NULL)
                deckLinkOutput->Release();
}

static const struct video_display_info display_decklink_info = {
        display_decklink_probe,
        display_decklink_init,
        display_decklink_run,
        display_decklink_done,
        display_decklink_getf,
        display_decklink_putf,
        display_decklink_reconfigure,
        display_decklink_get_property,
        display_decklink_put_audio_frame,
        display_decklink_reconfigure_audio,
};

REGISTER_MODULE(decklink, &display_decklink_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


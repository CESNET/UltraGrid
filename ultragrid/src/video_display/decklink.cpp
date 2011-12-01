/*
 * FILE:    video_display/decklink.cpp
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          Colin Perkins    <csp@isi.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2003 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#ifdef __cplusplus
extern "C" {
#endif

#include "host.h"
#include "debug.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_codec.h"
#include "tv.h"
#include "video_display/decklink.h"
#include "debug.h"
#include "video_capture.h"
#include "DeckLinkAPI.h"
#include "audio/audio.h"
#include "audio/utils.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

// defined int video_capture/decklink.cpp
void print_output_modes(IDeckLink *);

#ifdef HAVE_MACOSX
#define STRING CFStringRef
#else
#define STRING const char *
#endif

#define MAX_DEVICES 4

class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
        struct state_decklink *                 s;
        int                                     i;

public:
        PlaybackDelegate (struct state_decklink* owner, int index);

        // IUnknown needs only a dummy implementation
        virtual HRESULT         QueryInterface (REFIID iid, LPVOID *ppv)        {return E_NOINTERFACE;}
        virtual ULONG           AddRef ()                                                                       {return 1;}
        virtual ULONG           Release ()                                                                      {return 1;}

        virtual HRESULT         ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result);
        virtual HRESULT         ScheduledPlaybackHasStopped ();
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

class DeckLinkFrame;
class DeckLinkFrame : public IDeckLinkVideoFrame
{
                long ref;
                
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                char *data;

        protected:
                DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf); 
                virtual ~DeckLinkFrame();

        public:
        	static DeckLinkFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat);                
                /* IUnknown */
                virtual HRESULT QueryInterface(REFIID, void**);
                virtual ULONG AddRef();
                virtual ULONG Release();
                
                /* IDeckLinkVideoFrame */
                long GetWidth (void);
                long GetHeight (void);
                long GetRowBytes (void);
                BMDPixelFormat GetPixelFormat (void);
                BMDFrameFlags GetFlags (void);
                HRESULT GetBytes (/* out */ void **buffer);
                
                HRESULT GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode);
                HRESULT GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary);
};

class DeckLink3DFrame : public DeckLinkFrame, public IDeckLinkVideoFrame3DExtensions
{
        private:
                DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf); 
                ~DeckLink3DFrame();
                
                long ref;
                
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                DeckLinkFrame *rightEye;

        public:
                static DeckLink3DFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat);
                
                /* IUnknown */
                HRESULT QueryInterface(REFIID, void**);
                ULONG AddRef();
                ULONG Release();

                /* IDeckLinkVideoFrame3DExtensions */
                BMDVideo3DPackingFormat Get3DPackingFormat();
                HRESULT GetFrameForRightEye(IDeckLinkVideoFrame**);
};

#define DECKLINK_MAGIC DISPLAY_DECKLINK_ID

struct device_state {
        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        //IDeckLinkMutableVideoFrame *deckLinkFrame;
        IDeckLinkVideoFrame *deckLinkFrame;
};

struct state_decklink {
        uint32_t            magic;

        struct timeval      tv;

        struct device_state state[MAX_DEVICES];

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        struct audio_frame  audio;
        struct video_frame *frame;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                stereo:1;
        bool                initialized;
        int                 devices_cnt;
        unsigned int        play_audio:1;
        int                 output_audio_channel_count;
 };

static void show_help(void);
static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out); 
void display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);


static void show_help(void)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
        HRESULT                         result;

        printf("Decklink (output) options:\n");
        printf("\t-d decklink:<device_numbers>[:3D] - coma-separated numbers of output devices\n");
        
        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (deckLinkIterator == NULL)
        {
                fprintf(stderr, "A DeckLink iterator could not be created.  The DeckLink drivers may not be installed.\n");
                return;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                STRING          deviceNameString = NULL;
                const char *deviceNameCString;
                
                // *** Print the model name of the DeckLink card
                result = deckLink->GetModelName((STRING *) &deviceNameString);
#ifdef HAVE_MACOSX
                deviceNameCString = (char *) malloc(128);
                CFStringGetCString(deviceNameString, (char *) deviceNameCString, 128, kCFStringEncodingMacRoman);
#else
                deviceNameCString = deviceNameString;
#endif
                if (result == S_OK)
                {
                        printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
                        print_output_modes(deckLink);
#ifdef HAVE_MACOSX
                        CFRelease(deviceNameString);
#endif
                        free((void *)deviceNameCString);
                }
                
                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }
        
        deckLinkIterator->Release();

        // If no DeckLink cards were found in the system, inform the user
        if (numDevices == 0)
        {
                printf("\nNo Blackmagic Design devices were found.\n");
                return;
        } 
}

struct video_frame *
display_decklink_getf(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

        assert(s->magic == DECKLINK_MAGIC);

        if (s->initialized) {
                for(int i = 0; i < s->devices_cnt; ++i)
                        s->state[i].deckLinkFrame->GetBytes((void **) &s->frame->tiles[i].data);
        }

        /* stub -- real frames are taken with get_sub_frame call */
        return s->frame;
}

int display_decklink_putf(void *state, char *frame)
{
        int tmp;
        struct state_decklink *s = (struct state_decklink *)state;
        struct timeval tv;

        UNUSED(frame);

        assert(s->magic == DECKLINK_MAGIC);

        gettimeofday(&tv, NULL);

        uint32_t i;
        s->state[0].deckLinkOutput->GetBufferedVideoFrameCount(&i);
        if (i > 2) 
                fprintf(stderr, "Frame dropped!\n");
        else {
                for (int j = 0; j < s->devices_cnt; ++j)
                        s->state[j].deckLinkOutput->ScheduleVideoFrame(s->state[j].deckLinkFrame,
                                        s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                s->frames++;
        }


        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = (s->frames - s->frames_last) / seconds;
                fprintf(stdout, "%lu frames in %g seconds = %g FPS\n",
                        s->frames - s->frames_last, seconds, fps);
                s->tv = tv;
                s->frames_last = s->frames;
        }

        return TRUE;
}

static BMDDisplayMode get_mode(IDeckLinkOutput *deckLinkOutput, struct video_desc desc, BMDTimeValue *frameRateDuration,
		BMDTimeScale        *frameRateScale, int index)
{	IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;
        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                exit(128);
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                STRING modeNameString;
                const char *modeNameCString;
                if (deckLinkDisplayMode->GetName(&modeNameString) == S_OK)
                {
#ifdef HAVE_MACOSX
                        modeNameCString = (char *) malloc(128);
                        CFStringGetCString(modeNameString, (char *) modeNameCString, 128, kCFStringEncodingMacRoman);
#else
                        modeNameCString = modeNameString;
#endif
                        if (deckLinkDisplayMode->GetWidth() == desc.width &&
                                        deckLinkDisplayMode->GetHeight() == desc.height)
                        {
                                double displayFPS;
                                BMDFieldDominance dominance;
                                bool interlaced;

                                dominance = deckLinkDisplayMode->GetFieldDominance();
                                if (dominance == bmdLowerFieldFirst ||
                                                dominance == bmdUpperFieldFirst)
                                        interlaced = true;
                                else // progressive, psf, unknown
                                        interlaced = false;

                                deckLinkDisplayMode->GetFrameRate(frameRateDuration,
                                                frameRateScale);
                                displayFPS = (double) *frameRateScale / *frameRateDuration;
                                if(fabs(desc.fps - displayFPS) < 0.01// && (aux & AUX_INTERLACED && interlaced || !interlaced)
                                  )
                                {
                                        printf("Device %d - selected mode: %s\n", index, modeNameCString);
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        break;
                                }
                        }
                }
        }
        displayModeIterator->Release();
        
        return displayMode;
}

void
display_decklink_reconfigure(void *state, struct video_desc desc)
{
        struct state_decklink            *s = (struct state_decklink *)state;
        
        bool                              modeFound = false;
        BMDPixelFormat                    pixelFormat;
        BMDDisplayMode                    displayMode;
        BMDDisplayModeSupport             supported;
        int h_align = 0;

        assert(s->magic == DECKLINK_MAGIC);
        
        s->frame->color_spec = desc.color_spec;
        s->frame->aux = desc.aux;
        s->frame->fps = desc.fps;

	switch (desc.color_spec) {
                case UYVY:
                        pixelFormat = bmdFormat8BitYUV;
                        break;
                case v210:
                        pixelFormat = bmdFormat10BitYUV;
                        break;
                case RGBA:
                        pixelFormat = bmdFormat8BitBGRA;
                        break;
                default:
                        fprintf(stderr, "[DeckLink] Unsupported pixel format!\n");
        }

	if(s->stereo) {
		desc.width /= 2; // half of with for each channel
		for (int i = 0; i < 2; ++i) {
			struct tile  *tile = tile_get(s->frame, i, 0);
			tile->width = desc.width;
		        tile->height = desc.height;
	                tile->linesize = vc_get_linesize(tile->width, s->frame->color_spec);
	                tile->data_len = tile->linesize * tile->height;
	        }
		displayMode = get_mode(s->state[0].deckLinkOutput, desc, &s->frameRateDuration,
                                                &s->frameRateScale, 0);
		
		s->state[0].deckLinkOutput->DoesSupportVideoMode(displayMode, pixelFormat, bmdVideoOutputDualStream3D,
	                                &supported, NULL);
                if(supported == bmdDisplayModeNotSupported)
                {
                        fprintf(stderr, "[decklink] Requested parameters combination not supported.\n");
                        exit(128);
                }
                
                s->state[0].deckLinkFrame = DeckLink3DFrame::Create(desc.width, desc.height, tile_get(s->frame, 0, 0)->linesize, pixelFormat);
                        
                s->state[0].deckLinkOutput->EnableVideoOutput(displayMode,  bmdVideoOutputDualStream3D);
                s->state[0].deckLinkFrame->GetBytes((void **) &s->frame->tiles[0].data);
                IDeckLinkVideoFrame *right;
                
                dynamic_cast<DeckLink3DFrame *>(s->state[0].deckLinkFrame)->GetFrameForRightEye(&right);
                right->GetBytes((void **) &s->frame->tiles[1].data);
                s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
	} else {
	        for(int i = 0; i < s->devices_cnt; ++i) {
	        	struct video_desc desc_dev;
	                /* compute position */
	                int x_count = s->frame->grid_width;
	                int y_count = s->frame->grid_height;
	                struct tile  *tile = tile_get(s->frame, i % x_count,
	                                i / x_count);

			desc_dev = desc;	                                
	                
	                desc_dev.width = tile->width = desc.width / x_count;
	                desc_dev.height = tile->height = desc.height / y_count;
	                tile->linesize = vc_get_linesize(tile->width, s->frame->color_spec);
	                tile->data_len = tile->linesize * tile->height;
	                
	                displayMode = get_mode(s->state[i].deckLinkOutput, desc_dev, &s->frameRateDuration,
                                                &s->frameRateScale, i);
	
	                s->state[i].deckLinkOutput->DoesSupportVideoMode(displayMode, pixelFormat, bmdVideoOutputFlagDefault,
	                                &supported, NULL);
	                if(supported == bmdDisplayModeNotSupported)
	                {
	                        fprintf(stderr, "[decklink] Requested parameters combination not supported.\n");
	                        exit(128);
	                }
	
	                IDeckLinkMutableVideoFrame *frame;
	                if (s->state[i].deckLinkOutput->CreateVideoFrame(tile->width, tile->height,
	                                tile->linesize, pixelFormat, bmdFrameFlagDefault,
	                                &frame) != S_OK)
	                {
	                        fprintf(stderr, "[decklink] Failed to create video frame.\n");
	                        exit(128);
	                }
	                
	                s->state[i].deckLinkFrame = frame;
	                        
	                s->state[i].deckLinkOutput->EnableVideoOutput(displayMode, bmdVideoOutputFlagDefault);
	                s->state[i].deckLinkFrame->GetBytes((void **) &s->frame->tiles[i].data);
	        }
	
	        for(int i = 0; i < s->devices_cnt; ++i) {
	                s->state[i].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
	        }
	}

        s->initialized = true;
}


void *display_decklink_init(char *fmt, unsigned int flags)
{
        struct state_decklink *s;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        int                                             cardIdx[MAX_DEVICES];
        int                                             dnum = 0;

        s = (struct state_decklink *)calloc(1, sizeof(struct state_decklink));
        s->magic = DECKLINK_MAGIC;
        s->stereo = FALSE;
        
        if(fmt == NULL) {
                cardIdx[0] = 0;
                s->devices_cnt = 1;
                fprintf(stderr, "Card number unset, using first found (see -d decklink:help)!\n");

        } else if (strcmp(fmt, "help") == 0) {
                show_help();
                return NULL;
        } else  {
                char *tmp = strdup(fmt);
                char *ptr;
                char *saveptr1 = 0ul, *saveptr2 = 0ul;

                ptr = strtok_r(tmp, ":", &saveptr1);                
                char *devices = strdup(ptr);
                s->devices_cnt = 0;
                ptr = strtok_r(devices, ",", &saveptr2);
                do {
                        cardIdx[s->devices_cnt] = atoi(ptr);
                        ++s->devices_cnt;
                } while (ptr = strtok_r(NULL, ",", &saveptr2));
                free(devices);
                
                ptr = strtok_r(NULL, ":", &saveptr1);
                if(ptr) {
                        if(strcasecmp(ptr, "3D") == 0) {
                                s->stereo = TRUE;
                        } else {
                                fprintf(stderr, "[DeckLink] Warning: unknown options in config string.\n");
                        }
                }
                free (tmp);
        }
	assert(!s->stereo || s->devices_cnt == 1);
	
        gettimeofday(&s->tv, NULL);

        // Initialize the DeckLink API
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (!deckLinkIterator)
        {
                fprintf(stderr, "This application requires the DeckLink drivers installed.\n");
                if (deckLinkIterator != NULL)
                        deckLinkIterator->Release();
                return NULL;
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->state[i].deckLink = NULL;
                s->state[i].deckLinkOutput = NULL;
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
                        fprintf(stderr, "No DeckLink PCI card #%d found\n", cardIdx[i]);
                        return NULL;
                }
        }

	s->audio.state = s;
        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                s->play_audio = TRUE;
                s->audio.data = NULL;
                s->audio.reconfigure_audio = display_decklink_reconfigure_audio;
        } else {
                s->play_audio = FALSE;
        }
        
        double x_cnt = sqrt(s->devices_cnt);
        int x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
        int y_count = s->devices_cnt / x_count;
        if(s->stereo) {
        	s->frame = vf_alloc(2, 1);
	} else {
		s->frame = vf_alloc(x_count, y_count);
	}
        if(s->devices_cnt > 1) {
                s->frame->aux = AUX_TILED;
        }
        
        for(int i = 0; i < s->devices_cnt; ++i) {
                // Obtain the audio/video output interface (IDeckLinkOutput)
                if (s->state[i].deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->state[i].deckLinkOutput) != S_OK) {
                        if(s->state[i].deckLinkOutput != NULL)
                                s->state[i].deckLinkOutput->Release();
                        s->state[i].deckLink->Release();
                        return NULL;
                }

                s->state[i].delegate = new PlaybackDelegate(s, i);
                // Provide this class as a delegate to the audio and video output interfaces
                s->state[i].deckLinkOutput->SetScheduledFrameCompletionCallback(s->state[i].delegate);
                //s->state[i].deckLinkOutput->DisableAudioOutput();
        }

        s->frames = 0;
        s->initialized = false;

        return (void *)s;
}

void display_decklink_run(void *state)
{
        UNUSED(state);
}

void display_decklink_done(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

        vf_free(s->frame);
        free(s);
}

display_type_t *display_decklink_probe(void)
{
        display_type_t *dtype;

        dtype = (display_type_t *) malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_DECKLINK_ID;
                dtype->name = "decklink";
                dtype->description = "Blackmagick DeckLink card";
        }
        return dtype;
}

int display_decklink_get_property(void *state, int property, void *val, int *len)
{
        codec_t codecs[] = {v210, UYVY, RGBA};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

PlaybackDelegate::PlaybackDelegate (struct state_decklink * owner, int index) 
        : s(owner), i(index)
{
}

HRESULT         PlaybackDelegate::ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) 
{
        return S_OK;
}

HRESULT         PlaybackDelegate::ScheduledPlaybackHasStopped ()
{
        return S_OK;
}

/*
 * AUDIO
 */
struct audio_frame * display_decklink_get_audio_frame(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        
        if(!s->play_audio)
                return NULL;
        return &s->audio;
}

void display_decklink_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = s->audio.data_len / (s->audio.bps *
                        s->audio.ch_count);
        unsigned int sampleFramesWritten;

        /* we got probably count that cannot be rendered directly (aka 1) */
        if(s->output_audio_channel_count != s->audio.ch_count) {
                assert(s->audio.ch_count == 1); /* only reasonable value so far */
                if (sampleFrameCount * s->output_audio_channel_count 
                                * frame->bps > frame->max_size) {
                        fprintf(stderr, "[decklink] audio buffer overflow!\n");
                        sampleFrameCount = frame->max_size / 
                                        (s->output_audio_channel_count * frame->bps);
                        frame->data_len = sampleFrameCount *
                                        (frame->ch_count * frame->bps);
                }
                
                audio_frame_multiply_channel(frame,
                                s->output_audio_channel_count);
        }
        
	s->state[0].deckLinkOutput->ScheduleAudioSamples (s->audio.data, sampleFrameCount, 0, 		
                0, &sampleFramesWritten);
        if(sampleFramesWritten != sampleFrameCount)
                fprintf(stderr, "[decklink] audio buffer underflow!\n");

}

void display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        if(s->audio.data != NULL)
                free(s->audio.data);
                
        s->audio.bps = quant_samples / 8;
        s->audio.sample_rate = sample_rate;
        s->output_audio_channel_count = s->audio.ch_count = channels;
        
        if (s->audio.ch_count != 1 &&
                        s->audio.ch_count != 2 && s->audio.ch_count != 8 &&
                        s->audio.ch_count != 16) {
                fprintf(stderr, "[decklink] requested channel count isn't supported: "
                        "%d\n", s->audio.ch_count);
                s->play_audio = FALSE;
                return;
        }
        
        /* toggle one channel to supported two */
        if(s->audio.ch_count == 1) {
                 s->output_audio_channel_count = 2;
        }
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        sample_rate != 48000) {
                fprintf(stderr, "[decklink] audio format isn't supported: "
                        "samples: %d, sample rate: %d\n",
                        quant_samples, sample_rate);
                s->play_audio = FALSE;
                return;
        }
        switch(quant_samples) {
                case 16:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 32:
                        sample_type = bmdAudioSampleType32bitInteger;
                        break;
        }
                        
        s->state[0].deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        s->output_audio_channel_count,
                        bmdAudioOutputStreamContinuous);
        s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration);
        
        s->audio.max_size = 5 * (quant_samples / 8) 
                        * s->audio.ch_count
                        * sample_rate;                
        s->audio.data = (char *) malloc (s->audio.max_size);
}

bool operator==(const REFIID & first, const REFIID & second){
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

HRESULT DeckLinkFrame::QueryInterface(REFIID id, void**frame)
{
        return E_NOINTERFACE;
}

ULONG DeckLinkFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLinkFrame::Release()
{
        if(--ref == 0)
                delete this;
	return ref;
}

DeckLinkFrame::DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf)
	: width(w), height(h), rawBytes(rb), pixelFormat(pf), ref(1l)
{
        data = new char[rb * h];
}

DeckLinkFrame *DeckLinkFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat)
{
        return new DeckLinkFrame(width, height, rawBytes, pixelFormat);
}


DeckLinkFrame::~DeckLinkFrame() 
{
	delete[] data;
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
        *buffer = static_cast<void *>(data);
        return S_OK;
}

HRESULT DeckLinkFrame::GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode)
{
	return S_FALSE;
}

HRESULT DeckLinkFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary)
{
	return S_FALSE;
}



DeckLink3DFrame::DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf) 
        : DeckLinkFrame(w, h, rb, pf), ref(1l)
{
        rightEye = DeckLinkFrame::Create(w, h, rb, pf);        
}

DeckLink3DFrame *DeckLink3DFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat)
{
        DeckLink3DFrame *frame = new DeckLink3DFrame(width, height, rawBytes, pixelFormat);
        return frame;
}

DeckLink3DFrame::~DeckLink3DFrame()
{
	rightEye->Release();
}

ULONG DeckLink3DFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLink3DFrame::Release()
{
        if(--ref == 0)
                delete this;
	return ref;
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
        *frame = rightEye;
        rightEye->AddRef();
        return S_OK;
}

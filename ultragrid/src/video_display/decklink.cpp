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

#ifdef __cplusplus
} // END of extern "C"
#endif

#include "DeckLinkAPI.h"

class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
        struct state_decklink *                 s;

public:
        PlaybackDelegate (struct state_decklink* owner);

        // IUnknown needs only a dummy implementation
        virtual HRESULT         QueryInterface (REFIID iid, LPVOID *ppv)        {return E_NOINTERFACE;}
        virtual ULONG           AddRef ()                                                                       {return 1;}
        virtual ULONG           Release ()                                                                      {return 1;}

        virtual HRESULT         ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result);
        virtual HRESULT         ScheduledPlaybackHasStopped ();
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

#define DECKLINK_MAGIC DISPLAY_DECKLINK_ID

struct state_decklink {
        uint32_t            magic;

        struct timeval      tv;

        PlaybackDelegate   *delegate;
        IDeckLink          *deckLink;
        IDeckLinkOutput    *deckLinkOutput;
        IDeckLinkMutableVideoFrame *deckLinkFrame;
        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        struct video_frame  frame;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                initialized;
};

static void show_help(void);

static void show_help(void)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
        HRESULT                         result;

        printf("Decklink (output) options:\n");
        printf("\t-g device_number\n");
        
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
                char *          deviceNameString = NULL;
                
                
                // *** Print the model name of the DeckLink card
                result = deckLink->GetModelName((const char **) &deviceNameString);
                if (result == S_OK)
                {
                        printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameString);
                        free(deviceNameString);
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

        if (s->initialized)
                s->deckLinkFrame->GetBytes((void **) &s->frame.data);

        return &s->frame;
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
        s->deckLinkOutput->GetBufferedVideoFrameCount(&i);
        if (i > 2) 
                fprintf(stderr, "Frame dropped!\n");
        else {
                s->deckLinkOutput->ScheduleVideoFrame(s->deckLinkFrame, s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                s->frames++;
        }


        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = (s->frames - s->frames_last) / seconds;
                fprintf(stdout, "%d frames in %g seconds = %g FPS\n",
                        s->frames - s->frames_last, seconds, fps);
                s->tv = tv;
                s->frames_last = s->frames;
        }

        return TRUE;
}

static void
reconfigure_screen_decklink(void *state, unsigned int width, unsigned int height,
                                   codec_t color_spec, double fps, int aux)
{
        struct state_decklink            *s = (struct state_decklink *)state;
        IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        IDeckLinkDisplayMode*             selectedMode = NULL;
        bool                              modeFound = false;
        BMDPixelFormat                    pixelFormat;
        BMDDisplayMode                    displayMode;
        BMDDisplayModeSupport             supported;

        assert(s->magic == DECKLINK_MAGIC);

        s->frame.color_spec = color_spec;
        s->frame.width = width;
        s->frame.height = height;
        s->frame.dst_bpp = get_bpp(color_spec);
        s->frame.fps = fps;
        s->frame.aux = aux;

        s->frame.data_len = s->frame.width * s->frame.height * s->frame.dst_bpp;
        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;
        s->frame.src_bpp = get_bpp(color_spec);

        s->frame.decoder = (decoder_t)memcpy;

        switch (color_spec) {
                case UYVY:
                case DVS8:
                case Vuy2:
                        pixelFormat = bmdFormat8BitYUV;
                        break;
                case v210:
                        pixelFormat = bmdFormat10BitYUV;
                        break;
                case RGBA:
                        pixelFormat = bmdFormat8BitBGRA;
                        break;
                case R10k:
                        pixelFormat = bmdFormat10BitRGB;
                        break;
                default:
                        fprintf(stderr, "[DeckLink] Unsupported pixel format!\n");
        }


        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (s->deckLinkOutput->GetDisplayModeIterator(&displayModeIterator) != S_OK)
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                exit(128);
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                const char       *modeName;

                if (deckLinkDisplayMode->GetName(&modeName) == S_OK)
                {
                        if (deckLinkDisplayMode->GetWidth() == width &&
                                        deckLinkDisplayMode->GetHeight() == height)
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

                                deckLinkDisplayMode->GetFrameRate(&s->frameRateDuration,
                                                &s->frameRateScale);
                                displayFPS = (double) s->frameRateScale / s->frameRateDuration;
                                if(fabs(fps - displayFPS) < 0.5// && (aux & AUX_INTERLACED && interlaced || !interlaced)
                                  )
                                {
                                        printf("Selected mode: %s\n", modeName);
                                        modeFound = true;
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        break;
                                }
                        }
                }
        }
        displayModeIterator->Release();
        s->deckLinkOutput->DoesSupportVideoMode(displayMode, pixelFormat,
                        &supported);
        if(supported == bmdDisplayModeNotSupported)
        {
                fprintf(stderr, "[decklink] Requested parameters combination not supported.\n");
                exit(128);
        }

        //Generate a frame of black
        long int linesize = vc_getsrc_linesize(width, color_spec);
        if (s->deckLinkOutput->CreateVideoFrame(width, height, linesize, pixelFormat, bmdFrameFlagDefault, &s->deckLinkFrame) != S_OK)
        {
                fprintf(stderr, "[decklink] Failed to create video frame.\n");
                exit(128);
        }
                
        s->deckLinkOutput->EnableVideoOutput(displayMode, bmdVideoOutputFlagDefault);
        s->deckLinkFrame->GetBytes((void **) &s->frame.data);

        s->initialized = true;
        s->deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
}


void *display_decklink_init(char *fmt)
{
        struct state_decklink *s;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        int                                             cardIdx;
        int                                             dnum = 0;

        s = (struct state_decklink *)calloc(1, sizeof(struct state_decklink));
        s->magic = DECKLINK_MAGIC;

        if(fmt == NULL) {
                cardIdx = 0;
                fprintf(stderr, "Card number unset, using first found (see -g help)!\n");

        } else if (strcmp(fmt, "help") == 0) {
                show_help();
                return NULL;
        } else cardIdx = atoi(fmt);

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

        s->deckLink = NULL;

        // Connect to the first DeckLink instance
        while (deckLinkIterator->Next(&s->deckLink) == S_OK)
        {
                if (dnum != cardIdx){
                        if(s->deckLink != NULL)
                                s->deckLink->Release();
                        s->deckLink = NULL;
                }
                else break;
        }
        if(s->deckLink == NULL) {
                fprintf(stderr, "No DeckLink PCI cards found\n");
                return NULL;
        }

        // Obtain the audio/video output interface (IDeckLinkOutput)
        if (s->deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->deckLinkOutput) != S_OK) {
                if(s->deckLinkOutput != NULL)
                        s->deckLinkOutput->Release();
                s->deckLink->Release();
                return NULL;
        }

        s->delegate = new PlaybackDelegate(s);

        // Provide this class as a delegate to the audio and video output interfaces
        s->deckLinkOutput->SetScheduledFrameCompletionCallback(s->delegate);
        s->deckLinkOutput->DisableAudioOutput();


        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)reconfigure_screen_decklink;

        s->frames = 0;
        s->initialized = false;

        return (void *)s;
}

void display_decklink_done(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

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

PlaybackDelegate::PlaybackDelegate (struct state_decklink * owner) : s(owner) 
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

void vidcap_decklink_run(void *state)
{
        UNUSED(state);
}


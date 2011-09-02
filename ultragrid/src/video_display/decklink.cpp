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

#ifdef __cplusplus
} // END of extern "C"
#endif

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

#define DECKLINK_MAGIC DISPLAY_DECKLINK_ID

struct device_state {
        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        IDeckLinkMutableVideoFrame *deckLinkFrame;
};

struct state_decklink {
        uint32_t            magic;

        struct timeval      tv;

        struct device_state state[MAX_DEVICES];

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

	struct audio_frame  audio;
        struct video_frame  frame[MAX_DEVICES];

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                initialized;
        int                 devices_cnt;
        unsigned int        play_audio:1;
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
        printf("\t-g device_numbers - coma-separated numbers of output devices\n");
        
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
                        s->state[i].deckLinkFrame->GetBytes((void **) &s->frame[i].data);
        }

        /* stub -- real frames are taken with get_sub_frame call */
        return &s->frame[0];
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
        int tile_width, tile_height;

        assert(s->magic == DECKLINK_MAGIC);

        for(int i = 0; i < s->devices_cnt; ++i) {
                /* compute position */
                double x_cnt = sqrt(s->devices_cnt);
                s->frame[i].tile_info.x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                s->frame[i].tile_info.y_count = s->devices_cnt / s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_x = i % s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_y = i / s->frame[i].tile_info.x_count;

                tile_width = width / s->frame[i].tile_info.x_count;
                tile_height = height / s->frame[i].tile_info.y_count;
                s->frame[i].dst_x_offset = 0;
                s->frame[i].color_spec = color_spec;
                s->frame[i].width = width;
                s->frame[i].height = height;
                s->frame[i].dst_bpp = get_bpp(color_spec);
                s->frame[i].src_bpp = get_bpp(color_spec);
                s->frame[i].fps = fps;
                s->frame[i].aux = aux;

                s->frame[i].data_len = tile_width * tile_height * s->frame[i].dst_bpp;
                s->frame[i].dst_linesize = tile_width * s->frame[i].dst_bpp;
                s->frame[i].src_linesize = tile_width * s->frame[i].src_bpp;
                s->frame[i].dst_pitch = s->frame[i].dst_linesize;

                s->frame[i].decoder = (decoder_t)memcpy;

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
                if (s->state[i].deckLinkOutput->GetDisplayModeIterator(&displayModeIterator) != S_OK)
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
                                if (deckLinkDisplayMode->GetWidth() == tile_width &&
                                                deckLinkDisplayMode->GetHeight() == tile_height)
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
                                        if(fabs(fps - displayFPS) < 0.01// && (aux & AUX_INTERLACED && interlaced || !interlaced)
                                          )
                                        {
                                                printf("Device %d - selected mode: %s\n", i, modeNameCString);
                                                modeFound = true;
                                                displayMode = deckLinkDisplayMode->GetDisplayMode();
                                                break;
                                        }
                                }
                        }
                }
                displayModeIterator->Release();
                s->state[i].deckLinkOutput->DoesSupportVideoMode(displayMode, pixelFormat, bmdVideoOutputFlagDefault,
                                &supported, NULL);
                if(supported == bmdDisplayModeNotSupported)
                {
                        fprintf(stderr, "[decklink] Requested parameters combination not supported.\n");
                        exit(128);
                }

                //Generate a frame of black
                long int linesize = vc_getsrc_linesize(tile_width, color_spec);
                if (s->state[i].deckLinkOutput->CreateVideoFrame(tile_width, tile_height, linesize, pixelFormat, bmdFrameFlagDefault, &s->state[i].deckLinkFrame) != S_OK)
                {
                        fprintf(stderr, "[decklink] Failed to create video frame.\n");
                        exit(128);
                }
                        
                s->state[i].deckLinkOutput->EnableVideoOutput(displayMode, bmdVideoOutputFlagDefault);
                s->state[i].deckLinkFrame->GetBytes((void **) &s->frame[i].data);
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->state[i].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
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

        if(fmt == NULL) {
                cardIdx[0] = 0;
                s->devices_cnt = 1;
                fprintf(stderr, "Card number unset, using first found (see -g help)!\n");

        } else if (strcmp(fmt, "help") == 0) {
                show_help();
                return NULL;
        } else  {
                char *devices = strdup(fmt);
                char *ptr;
                char *saveptr;

                s->devices_cnt = 0;
                ptr = strtok_r(devices, ",", &saveptr);
                do {
                        cardIdx[s->devices_cnt] = atoi(ptr);
                        ++s->devices_cnt;
                } while (ptr = strtok_r(NULL, ",", &saveptr));
                free (devices);
        }

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

                s->frame[i].state = s;
                s->frame[i].reconfigure = (reconfigure_t) reconfigure_screen_decklink;
                s->frame[i].get_sub_frame = (get_sub_frame_t) get_sub_frame;
        }

        s->frames = 0;
        s->initialized = false;

        return (void *)s;
}

void display_decklink_run(void *state)
{
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

void vidcap_decklink_run(void *state)
{
        UNUSED(state);
}

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_decklink *s = (struct state_decklink *)state;

        if (s->devices_cnt > 1) { /* tiled video
                                     expecting that requested sub_frame matches
                                     exactly one physical device */
                double x_cnt = sqrt(s->devices_cnt);
                int y_cnt;
                int index;
                x_cnt = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                y_cnt = s->devices_cnt / (int) x_cnt;

                assert(x % w == 0 &&
                                y % h == 0 &&
                                s->frame[0].width % w == 0 &&
                                s->frame[0].height % h == 0);

                index = x / w + // row 
                        x_cnt * (y / h); // column

                memcpy(out, &s->frame[index], sizeof(struct video_frame));
        } else { /* dual link */
                assert(x + w <= s->frame[0].width &&
                                y + h <= s->frame[0].height);

                memcpy(out, &s->frame[0], sizeof(struct video_frame));
                out->data +=
                        y * s->frame[0].dst_pitch +
                        (size_t) (x * s->frame[0].dst_bpp);
                out->src_linesize =
                        vc_getsrc_linesize(w, out->color_spec);
                out->dst_linesize =
                        w * out->dst_bpp;

        }
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

void display_decklink_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        const unsigned int sampleFrameCount = s->audio.data_len / (s->audio.bps *
                        s->audio.ch_count);
        unsigned int sampleFramesWritten;

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
        s->audio.ch_count = channels;
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        (channels != 2 && channels != 8 && channels != 16) ||
                        sample_rate != 48000) {
                fprintf(stderr, "[decklink] audio format isn't supported: "
                        "channels: %d, samples: %d, sample rate: %d\n",
                        channels, quant_samples, sample_rate);
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
                        channels,
                        bmdAudioOutputStreamContinuous);
        s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration);
        
        s->audio.max_size = 5 * (quant_samples / 8) * channels *
                        sample_rate;                
        s->audio.data = (char *) malloc (s->audio.max_size);
}


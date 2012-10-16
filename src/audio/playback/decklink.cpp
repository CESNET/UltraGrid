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
#include "audio/playback/decklink.h"
#include "debug.h"
#include "video_capture.h"
#include "audio/audio.h"
#include "audio/utils.h"

#ifdef WIN32
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#endif
#include "DeckLinkAPIVersion.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#ifdef WIN32
#include <objbase.h>
#endif

#ifdef HAVE_MACOSX
#define STRING CFStringRef
#elif defined WIN32
#define STRING BSTR
#else
#define STRING const char *
#endif

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

// defined int video_capture/decklink.cpp
void print_output_modes(IDeckLink *);
static int blackmagic_api_version_check(STRING *current_version);


class PlaybackDelegate;


#define DECKLINK_MAGIC DISPLAY_DECKLINK_ID


struct state_decklink {
        uint32_t            magic;

        struct audio_frame  audio;

        int                 output_audio_channel_count;

        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        IDeckLinkMutableVideoFrame *deckLinkFrame;

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        int frames;
 };

class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
        struct state_decklink *                 s;

public:
        PlaybackDelegate (struct state_decklink* owner) {
                s = owner;
        }


        // IUnknown needs only a dummy implementation
        virtual HRESULT STDMETHODCALLTYPE        QueryInterface (REFIID iid, LPVOID *ppv)        {return E_NOINTERFACE;}
        virtual ULONG STDMETHODCALLTYPE            AddRef ()                                                                       {return 1;}
        virtual ULONG STDMETHODCALLTYPE            Release ()                                                                      {return 1;}

        virtual HRESULT STDMETHODCALLTYPE          ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) {
                        s->deckLinkOutput->ScheduleVideoFrame(s->deckLinkFrame,
                                        s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                        s->frames++;
                        return S_OK;
        }
        virtual HRESULT STDMETHODCALLTYPE          ScheduledPlaybackHasStopped () { return S_OK; } 
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

static int blackmagic_api_version_check(STRING *current_version)
{
        int ret = TRUE;
        *current_version = NULL;
        IDeckLinkAPIInformation *APIInformation = NULL;
	HRESULT result;

#ifdef WIN32
	result = CoCreateInstance(CLSID_CDeckLinkAPIInformation, NULL, CLSCTX_ALL,
		IID_IDeckLinkAPIInformation, (void **) &APIInformation);
	
#else
        APIInformation = CreateDeckLinkAPIInformationInstance();
        if(APIInformation == NULL)
#endif
	{
                return FALSE;
        }
        int64_t value;
        HRESULT res;
        res = APIInformation->GetInt(BMDDeckLinkAPIVersion, &value);
        if(res != S_OK) {
                APIInformation->Release();
                return FALSE;
        }

        if(BLACKMAGIC_DECKLINK_API_VERSION > value) { // this is safe comparision, for internal structure please see SDK documentation
                APIInformation->GetString(BMDDeckLinkAPIVersion, current_version);
                ret  = FALSE;
        }


        APIInformation->Release();
        return ret;
}

void decklink_playback_help(const char *driver_name)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
        HRESULT                         result;

        UNUSED(driver_name);

        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
#ifdef WIN32
	result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
		IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result))
#else
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (deckLinkIterator == NULL)
#endif
        {
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
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
#elif defined WIN32
                deviceNameCString = (char *) malloc(128);
		wcstombs((char *) deviceNameCString, deviceNameString, 128);
#else
                deviceNameCString = deviceNameString;
#endif
                if (result == S_OK)
                {
                        printf("\tdecklink:%d :      Blackmagic %s\n",numDevices, deviceNameCString);
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

void *decklink_playback_init(char *index_str)
{
        struct state_decklink *s;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        IDeckLinkConfiguration*         deckLinkConfiguration = NULL;
        // for Decklink Studio which has switchable XLR - analog 3 and 4 or AES/EBU 3,4 and 5,6
        BMDAudioOutputAnalogAESSwitch audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
        int cardIdx = 0;
        int dnum = 0;

#ifdef WIN32
	// Initialize COM on this thread
	result = CoInitialize(NULL);
	if(FAILED(result)) {
		fprintf(stderr, "Initialization of COM failed - result = "
				"08x.\n", result);
		return NULL;
	}
#endif

        STRING current_version;
        if(!blackmagic_api_version_check(&current_version)) {
		fprintf(stderr, "\nThe DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled against DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                fprintf(stderr, "Vendor download page is http://http://www.blackmagic-design.com/support/ \n");
                if(current_version) {
                        const char *currentVersionCString;
#ifdef HAVE_MACOSX
                        currentVersionCString = (char *) malloc(128);
                        CFStringGetCString(current_version, (char *) currentVersionCString, 128, kCFStringEncodingMacRoman);
#elif defined WIN32
                        currentVersionCString = (char *) malloc(128);
			wcstombs((char *) currentVersionCString, current_version, 128);
#else
                        currentVersionCString = current_version;
#endif
                        fprintf(stderr, "Currently installed version is: %s\n", currentVersionCString);
#ifdef HAVE_MACOSX
                        CFRelease(current_version);
#endif
                        free((void *)currentVersionCString);
                } else {
                        fprintf(stderr, "No installed drivers detected\n");
                }
                fprintf(stderr, "\n");
                return NULL;
        }


        s = (struct state_decklink *)calloc(1, sizeof(struct state_decklink));
        s->magic = DECKLINK_MAGIC;
        
        if(index_str == NULL) {
                cardIdx = 0;
                fprintf(stderr, "Card number unset, using first found (see -d decklink:help)!\n");

        } else if (strcmp(index_str, "help") == 0) {
                decklink_playback_help(NULL);
                return NULL;
        } else  {
                cardIdx = atoi(index_str);
        }

        // Initialize the DeckLink API
#ifdef WIN32
	result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
		IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result))
#else
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (!deckLinkIterator)
#endif
        {
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                return NULL;
        }

        s->deckLink = NULL;
        s->deckLinkOutput = NULL;

        // Connect to the first DeckLink instance
        IDeckLink    *deckLink;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                if (dnum == cardIdx){
                        s->deckLink = deckLink;
                } else {
                        deckLink->Release();
                }
                dnum++;
        }

        if(s->deckLink == NULL) {
                fprintf(stderr, "No DeckLink PCI card #%d found\n", cardIdx);
                return NULL;
        }

        s->audio.data = NULL;
        
        // Obtain the audio/video output interface (IDeckLinkOutput)
        if (s->deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->deckLinkOutput) != S_OK) {
                if(s->deckLinkOutput != NULL)
                        s->deckLinkOutput->Release();
                s->deckLink->Release();
                return NULL;
        }

        // Query the DeckLink for its configuration interface
        result = s->deckLink->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
        if (result != S_OK)
        {
                printf("Could not obtain the IDeckLinkConfiguration interface: %08x\n", (int) result);
                return NULL;
        }

        IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode                    displayMode = bmdModeUnknown;
        int width, height;

        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(s->deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                return NULL;
        }

        if(displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                width = deckLinkDisplayMode->GetWidth();
                height = deckLinkDisplayMode->GetHeight();
                deckLinkDisplayMode->GetFrameRate(&s->frameRateDuration, &s->frameRateScale);
        }

        s->frames = 0;



        s->deckLinkOutput->CreateVideoFrame(width, height, width*2, bmdFormat8BitYUV, bmdFrameFlagDefault, &s->deckLinkFrame);

        s->delegate = new PlaybackDelegate(s);
        // Provide this class as a delegate to the audio and video output interfaces
        s->deckLinkOutput->SetScheduledFrameCompletionCallback(s->delegate);
        //s->state[i].deckLinkOutput->DisableAudioOutput();
        //
        s->deckLinkOutput->EnableVideoOutput(deckLinkDisplayMode->GetDisplayMode(), bmdVideoOutputFlagDefault);

        displayModeIterator->Release();
        deckLinkDisplayMode->Release();


        return (void *)s;
}

struct audio_frame* decklink_get_frame(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        
        return &s->audio;
}

void decklink_put_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = s->audio.data_len / (s->audio.bps *
                        s->audio.ch_count);
#ifdef WIN32
        unsigned long int sampleFramesWritten;
#else
        unsigned int sampleFramesWritten;
#endif

        /* we got probably count that cannot be played directly (probably 1) */
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
        
	s->deckLinkOutput->ScheduleAudioSamples (s->audio.data, sampleFrameCount, 0, 		
                0, &sampleFramesWritten);
        if(sampleFramesWritten != sampleFrameCount)
                fprintf(stderr, "[decklink] audio buffer underflow!\n");

}

int decklink_reconfigure(void *state, int quant_samples, int channels,
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
                return FALSE;
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
                        
        s->deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        s->output_audio_channel_count,
                        bmdAudioOutputStreamContinuous);
        s->deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration);
        
        s->audio.max_size = 5 * (quant_samples / 8) 
                        * s->audio.ch_count
                        * sample_rate;                
        s->audio.data = (char *) malloc (s->audio.max_size);

        return TRUE;
}

void decklink_close_playback(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

        s->deckLinkOutput->DisableAudioOutput();
        s->deckLinkOutput->DisableVideoOutput();
        s->deckLinkFrame->Release();
        s->deckLink->Release();
        s->deckLinkOutput->Release();
        free(s->audio.data);
        free(s);
}


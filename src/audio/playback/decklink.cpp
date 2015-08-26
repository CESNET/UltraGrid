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

#include "host.h"
#include "debug.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_codec.h"
#include "tv.h"
#include "audio/audio_playback.h"
#include "debug.h"
#include "lib_common.h"
#include "video_capture.h"
#include "audio/audio.h"
#include "audio/utils.h"

#ifdef WIN32
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#endif
#include "DeckLinkAPIVersion.h"

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

namespace {
class PlaybackDelegate;
}

#define DECKLINK_MAGIC 0x415f46d0


struct state_decklink {
        uint32_t            magic;

        struct audio_desc  audio_desc;

        int                 output_audio_channel_count;

        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        IDeckLinkMutableVideoFrame *deckLinkFrame;

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        int frames;
        int                     audio_consumer_levels; ///< 0 false, 1 true, -1 default
 };

namespace {
class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
        struct state_decklink *                 s;

public:
        PlaybackDelegate (struct state_decklink* owner) {
                s = owner;
        }


        // IUnknown needs only a dummy implementation
        virtual HRESULT STDMETHODCALLTYPE        QueryInterface (REFIID, LPVOID *)        { return E_NOINTERFACE;}
        virtual ULONG STDMETHODCALLTYPE            AddRef ()                                                                       {return 1;}
        virtual ULONG STDMETHODCALLTYPE            Release ()                                                                      {return 1;}

        virtual HRESULT STDMETHODCALLTYPE          ScheduledFrameCompleted (IDeckLinkVideoFrame *, BMDOutputFrameCompletionResult) {
                s->deckLinkOutput->ScheduleVideoFrame(s->deckLinkFrame,
                                s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                s->frames++;
                return S_OK;
        }
        virtual HRESULT STDMETHODCALLTYPE          ScheduledPlaybackHasStopped () { return S_OK; } 
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};
} // end of unnamed namespace

static int blackmagic_api_version_check(STRING *current_version)
{
        int ret = TRUE;
        *current_version = NULL;
        IDeckLinkAPIInformation *APIInformation = NULL;
	HRESULT result;

#ifdef WIN32
	result = CoCreateInstance(CLSID_CDeckLinkAPIInformation, NULL, CLSCTX_ALL,
		IID_IDeckLinkAPIInformation, (void **) &APIInformation);
        if(FAILED(result))
#else
        APIInformation = CreateDeckLinkAPIInformationInstance();
        if(APIInformation == NULL)
#endif
	{
                return FALSE;
        }
        int64_t value;
        result = APIInformation->GetInt(BMDDeckLinkAPIVersion, &value);
        if(result != S_OK) {
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

static void audio_play_decklink_help(const char *driver_name)
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

static void *audio_play_decklink_init(const char *cfg)
{
        struct state_decklink *s = NULL;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        IDeckLinkConfiguration*         deckLinkConfiguration = NULL;
        // for Decklink Studio which has switchable XLR - analog 3 and 4 or AES/EBU 3,4 and 5,6
        //BMDAudioOutputAnalogAESSwitch audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
        int cardIdx = 0;
        int dnum = 0;
        IDeckLink    *deckLink;
        IDeckLinkDisplayModeIterator     *displayModeIterator = NULL;
        IDeckLinkDisplayMode             *deckLinkDisplayMode = NULL;
        //BMDDisplayMode                    displayMode = bmdModeUnknown;
        int width, height;

        log_msg(LOG_LEVEL_WARNING, "Decklink audio playback is most likely broken! Consider using "
                        "\"-t decklink -r analog\" instead.\n");

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
        s->audio_consumer_levels = -1;
        
        if (cfg == NULL) {
                cardIdx = 0;
                fprintf(stderr, "Card number unset, using first found (see -r decklink:help)!\n");
        } else if (strcmp(cfg, "help") == 0) {
                printf("Available Blackmagic audio playback devices:\n");
                audio_play_decklink_help(NULL);
                printf("Options:\n");
                printf("\t-r decklink[:<index>][:audioConsumerLevels={true|false}]\n");
                printf("audioConsumerLevels\n");
                printf("\tIf set true the analog audio levels are set to maximum gain on audio input.\n");
                printf("\tIf set false the selected analog input gain levels are used.\n");
                free(s);
                return NULL;
        } else  {
                char *tmp = strdup(cfg);
                char *item, *save_ptr;
                item = strtok_r(tmp, ":", &save_ptr);
                if (item) {
                        if(strncasecmp(item, "audioConsumerLevels=",
                                                strlen("audioConsumerLevels=")) == 0) {
                                char *levels = item + strlen("audioConsumerLevels=");
                                if (strcasecmp(levels, "false") == 0) {
                                        s->audio_consumer_levels = 0;
                                } else {
                                        s->audio_consumer_levels = 1;
                                }
                                item = strtok_r(NULL, ":", &save_ptr);
                                if (item) {
                                        cardIdx = atoi(cfg);
                                }
                        } else {
                                cardIdx = atoi(cfg);
                        }
                }
                free(tmp);
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
                goto error;
        }

        // Connect to the first DeckLink instance
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
                goto error;
        }

        // Obtain the audio/video output interface (IDeckLinkOutput)
        if (s->deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->deckLinkOutput) != S_OK) {
                goto error;
        }

        // Query the DeckLink for its configuration interface
        result = s->deckLink->QueryInterface(IID_IDeckLinkConfiguration, (void**)&deckLinkConfiguration);
        if (result != S_OK)
        {
                printf("Could not obtain the IDeckLinkConfiguration interface: %08x\n", (int) result);
                goto error;
        }

        if (s->audio_consumer_levels != -1) {
                result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                s->audio_consumer_levels == 1 ? true : false);
                        if(result != S_OK) {
                                fprintf(stderr, "[DeckLink capture] Unable set input audio consumer levels.\n");
                        }
        }
        deckLinkConfiguration->Release();
        deckLinkConfiguration = NULL;

        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(s->deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                goto error;
        }

        // pick first display mode, no matter which it is
        if(displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                width = deckLinkDisplayMode->GetWidth();
                height = deckLinkDisplayMode->GetHeight();
                deckLinkDisplayMode->GetFrameRate(&s->frameRateDuration, &s->frameRateScale);
        } else {
                fprintf(stderr, "[decklink] Fatal: cannot get any display mode.\n");
                goto error;
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
        displayModeIterator = NULL;
        deckLinkDisplayMode->Release();
        deckLinkDisplayMode = NULL;


        return (void *)s;

error:
        if (displayModeIterator)
                displayModeIterator->Release();
        if (deckLinkDisplayMode)
                deckLinkDisplayMode->Release();
        if (deckLinkConfiguration)
                deckLinkConfiguration->Release();
        if (s->deckLinkOutput != NULL)
                s->deckLinkOutput->Release();
        if (s->deckLink != NULL)
                s->deckLink->Release();
        free(s);
        return NULL;
}

static void audio_play_decklink_put_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = frame->data_len /
                (s->audio_desc.bps * s->audio_desc.ch_count);
#ifdef WIN32
        unsigned long int sampleFramesWritten;
#else
        unsigned int sampleFramesWritten;
#endif
        char *data = frame->data;
        // tmp_frame is used if we need to perform 1->2 channel multiplication
        struct audio_frame tmp_frame;
        tmp_frame.data = NULL;

        /* we got probably channel count that cannot be played directly (probably 1) */
        if(s->output_audio_channel_count != s->audio_desc.ch_count) {
                assert(s->audio_desc.ch_count == 1); /* only supported value so far */
                memcpy(&tmp_frame, frame, sizeof(tmp_frame));
                // allocate enough space to hold resulting data
                tmp_frame.max_size = sampleFrameCount * s->output_audio_channel_count
                        * frame->bps;
                tmp_frame.data = (char *) malloc(tmp_frame.max_size);
                memcpy(tmp_frame.data, frame->data, frame->data_len);
                
                audio_frame_multiply_channel(&tmp_frame,
                                s->output_audio_channel_count);

                data = tmp_frame.data;
        }
        
	s->deckLinkOutput->ScheduleAudioSamples (data, sampleFrameCount, 0,
                0, &sampleFramesWritten);
        if(sampleFramesWritten != sampleFrameCount)
                fprintf(stderr, "[decklink] audio buffer underflow!\n");

        free(tmp_frame.data);

}

static struct audio_desc audio_play_decklink_query_format(void *, struct audio_desc desc) {
        return audio_desc{desc.bps <= 2 ? 2 : 4, 48000, desc.ch_count <= 2 ? 2 :
                (desc.ch_count <= 8 ? 8 : 16), AC_PCM};
}

static int audio_play_decklink_reconfigure(void *state, struct audio_desc desc) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        s->audio_desc = desc;
        s->output_audio_channel_count = desc.ch_count;
        
        if (s->audio_desc.ch_count != 1 &&
                        s->audio_desc.ch_count != 2 &&
                        s->audio_desc.ch_count != 8 &&
                        s->audio_desc.ch_count != 16) {
                fprintf(stderr, "[decklink] requested channel count isn't supported: "
                        "%d\n", s->audio_desc.ch_count);
                return FALSE;
        }
        
        /* toggle one channel to supported two */
        if(s->audio_desc.ch_count == 1) {
                 s->output_audio_channel_count = 2;
        }
        
        if((desc.bps != 2 && desc.bps != 4) ||
                        desc.sample_rate != 48000) {
                LOG(LOG_LEVEL_ERROR) << "[decklink] audio format isn't supported: " <<
                        desc;
                return FALSE;
        }
        switch(desc.bps) {
                case 2:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 4:
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
        
        return TRUE;
}

static void audio_play_decklink_done(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

        s->deckLinkOutput->DisableAudioOutput();
        s->deckLinkOutput->DisableVideoOutput();
        s->deckLinkFrame->Release();
        s->deckLink->Release();
        s->deckLinkOutput->Release();
        free(s);
}

static const struct audio_playback_info aplay_decklink_info = {
        audio_play_decklink_help,
        audio_play_decklink_init,
        audio_play_decklink_put_frame,
        audio_play_decklink_query_format,
        audio_play_decklink_reconfigure,
        audio_play_decklink_done
};

REGISTER_MODULE(decklink, &aplay_decklink_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


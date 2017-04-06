/**
 * @file   src/video_display/deltacast.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2016 CESNET, z. s. p. o.
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

#include "host.h"
#include "debug.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "deltacast_common.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"
#include "video_display/deltacast.h"
#include "debug.h"
#include "audio/audio.h"
#include "audio/utils.h"
#include "utils/ring_buffer.h"

#include <algorithm>

#define DELTACAST_MAGIC 0x01005e02

const struct deltacast_frame_mode_t deltacast_frame_modes[] = {
        {VHD_VIDEOSTD_S274M_1080p_25Hz, "SMPTE 274M 1080p 25 Hz",
                1920u, 1080u, 25.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 29.97 Hz",
                1920u, 1080u, 29.97, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 30 Hz",
                1920u, 1080u, 30.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_50Hz, "SMPTE 274M 1080i 50 Hz",
                1920u, 1080u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 59.94 Hz",
                1920u, 1080u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 60 Hz",
                1920u, 1080u, 30.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_50Hz, "SMPTE 296M 720p 50 Hz",
                1280u, 720u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 59.94 Hz",
                1280u, 720u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 60 Hz",
                1280u, 720u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S259M_PAL, "SMPTE 259M PAL",
                720u, 576u, 25.0, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S259M_NTSC, "SMPTE 259M NTSC",
                720u, 487u, 29.97, UPPER_FIELD_FIRST, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 23.98 Hz",
                1920u, 1080u, 23.98, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 24 Hz",
                1920u, 1080u, 24.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 59.94 Hz",
                1920u, 1080u, 59.94, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 60 Hz",
                1920u, 1080u, 60.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 50 Hz",
                1920u, 1080u, 50.0, PROGRESSIVE, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 23.98 Hz",
                1920u, 1080u, 23.98, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 24 Hz",
                1920u, 1080u, 24.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_25Hz, "SMPTE 274M 1080psf 25 Hz",
                1920u, 1080u, 25.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 29.97 Hz",
                1920u, 1080u, 29.97, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 30 Hz",
                1920u, 1080u, 30.0, SEGMENTED_FRAME, VHD_INTERFACE_AUTO},
        // UHD modes
        {VHD_VIDEOSTD_3840x2160p_24Hz, "3840x2160 24 Hz",
                3840u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_25Hz, "3840x2160 25 Hz",
                3840u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_30Hz, "3840x2160 30 Hz",
                3840u, 2160u, 30.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_3840x2160p_50Hz, "3840x2160 50 Hz",
                3840u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_3840x2160p_60Hz, "3840x2160 60 Hz",
                3840u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_24Hz, "4096x2160 24 Hz",
                4096u, 2160u, 24.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_25Hz, "4096x2160 25 Hz",
                4096u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_25Hz, "4096x2160 25 Hz",
                4096u, 2160u, 25.0, PROGRESSIVE, VHD_INTERFACE_4XHD},
        {VHD_VIDEOSTD_4096x2160p_48Hz, "4096x2160 48 Hz",
                4096u, 2160u, 48.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_50Hz, "4096x2160 50 Hz",
                4096u, 2160u, 50.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
        {VHD_VIDEOSTD_4096x2160p_60Hz, "4096x2160 60 Hz",
                4096u, 2160u, 60.0, PROGRESSIVE, VHD_INTERFACE_4X3G_A},
};

const int deltacast_frame_modes_count = sizeof(deltacast_frame_modes)/sizeof(deltacast_frame_mode_t);

struct state_deltacast {
        uint32_t            magic;

        struct timeval      tv;
        struct video_frame *frame;
        struct tile        *tile;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                initialized;
        HANDLE              BoardHandle, StreamHandle;
        HANDLE              SlotHandle;

        pthread_mutex_t     lock;

        unsigned int play_audio:1;
        unsigned int audio_configured:1;
        VHD_AUDIOINFO     AudioInfo;
        SHORT            *pSample;
        struct audio_desc  audio_desc;
        struct ring_buffer  *audio_channels[16];
        char            *audio_tmp;
 };

static void show_help(void);

static void show_help(void)
{
        printf("deltacast (output) options:\n");
        printf("\t-d deltacast[:device=<index>]\n");

        print_available_delta_boards();

        printf("\nDefault board is 0.\n");

}

static struct video_frame *
display_deltacast_getf(void *state)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        BYTE *pBuffer;
        ULONG BufferSize;
        ULONG Result;

        assert(s->magic == DELTACAST_MAGIC);
        
        if(!s->initialized)
                return s->frame;
        
        Result = VHD_LockSlotHandle(s->StreamHandle, &s->SlotHandle);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to lock slot.\n");
                return NULL;
        }
        Result = VHD_GetSlotBuffer(s->SlotHandle,VHD_SDI_BT_VIDEO,&pBuffer,&BufferSize);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to get buffer.\n");
                return NULL;
        }
        
        s->tile->data = (char *) pBuffer;
        s->tile->data_len = BufferSize;

        return s->frame;
}

static int display_deltacast_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        struct timeval tv;
        int i;
        ULONG Result;

        UNUSED(frame);
        UNUSED(nonblock);

        assert(s->magic == DELTACAST_MAGIC);
        
        pthread_mutex_lock(&s->lock);
        if(s->play_audio && s->audio_configured) {
                /* Retrieve the number of needed samples */
                for(i = 0; i < s->audio_desc.ch_count; ++i) {
                        s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].DataSize = 0;
                }
                Result = VHD_SlotEmbedAudio(s->SlotHandle,&s->AudioInfo);
                if (Result != VHDERR_BUFFERTOOSMALL)
                {
                        fprintf(stderr, "[DELTACAST] ERROR : Cannot embed audio on TX0 stream. Result = 0x%08lX\n", Result);
                } else {
                        for(i = 0; i < s->audio_desc.ch_count; ++i) {
                                int ret;
                                ret = ring_buffer_read(s->audio_channels[i], (char *) s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].pData, s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].DataSize);
                                if(!ret) {
                                        fprintf(stderr, "[DELTACAST] Buffer underflow for channel %d.\n", i);
                                }
                                s->AudioInfo.pAudioGroups[0].pAudioChannels[0].DataSize = ret;
                        }
                }
                /* Embed audio */
                Result = VHD_SlotEmbedAudio(s->SlotHandle,&s->AudioInfo);
                if (Result != VHDERR_NOERROR)
                {
                        fprintf(stderr, "[DELTACAST] ERROR : Cannot embed audio on TX0 stream. Result = 0x%08lX\n",Result);
                }
        }
        pthread_mutex_unlock(&s->lock);

        VHD_UnlockSlotHandle(s->SlotHandle);
        s->SlotHandle = NULL;
        
        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[DELTACAST display] %lu frames in %g seconds = %g FPS\n",
                        s->frames, seconds, fps);
                s->tv = tv;
                s->frames = 0;
        }
        s->frames++;

        return 0;
}

static int
display_deltacast_reconfigure(void *state, struct video_desc desc)
{
        struct state_deltacast            *s = (struct state_deltacast *)state;
        int VideoStandard;
        int i;
        ULONG Result;
        
        if(s->initialized) {
                if(s->SlotHandle)
                        VHD_UnlockSlotHandle(s->SlotHandle);
                VHD_StopStream(s->StreamHandle);
                VHD_CloseStreamHandle(s->StreamHandle);
        }

        assert(desc.tile_count == 1);

        s->tile->width = desc.width;
        s->tile->height = desc.height;
        s->frame->color_spec = desc.color_spec;
        s->frame->interlacing = desc.interlacing;
        s->frame->fps = desc.fps;
        
        for (i = 0; i < deltacast_frame_modes_count; ++i)
        {
                if(fabs(desc.fps - deltacast_frame_modes[i].fps) < 0.01 &&
                                desc.interlacing == deltacast_frame_modes[i].interlacing &&
                                desc.width == deltacast_frame_modes[i].width &&
                                desc.height == deltacast_frame_modes[i].height) {
                        VideoStandard = deltacast_frame_modes[i].mode;
                        fprintf(stderr, "[DELTACAST] %s mode selected.\n", deltacast_frame_modes[i].name);
                        break;
                }
        }
        if(i == deltacast_frame_modes_count) {
                fprintf(stderr, "[DELTACAST] Failed to obtain video format for incoming video: %dx%d @ %2.2f %s\n", desc.width, desc.height,
                                                                        (double) desc.fps, get_interlacing_description(desc.interlacing));

                goto error;
        }
        
        if(desc.color_spec == RAW) {
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_TX0,VHD_SDI_STPROC_RAW,NULL,&s->StreamHandle,NULL);
        } else if (s->play_audio == TRUE) {
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_TX0,VHD_SDI_STPROC_JOINED,NULL,&s->StreamHandle,NULL);
        } else {
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_TX0,VHD_SDI_STPROC_DISJOINED_VIDEO,NULL,&s->StreamHandle,NULL);
        }
        
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Failed to open stream handle.\n");
                goto error;
        }
        
        VHD_SetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD,VideoStandard);
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_DEPTH,2);
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_PRELOAD,0);

        Result = VHD_StartStream(s->StreamHandle);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to start stream.\n");  
                goto error;
        }
        
        s->initialized = TRUE;
        return TRUE;

error:
        return FALSE;
}

static void display_deltacast_probe(struct device_info **available_cards, int *count)
{
        *count = 0;
        *available_cards = nullptr;

        ULONG             Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                return;
        }
        if (NbBoards == 0) {
                return;
        }

        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++)
        {
                ULONG BoardType;
                HANDLE            BoardHandle = NULL;
                ULONG Result = VHD_OpenBoardHandle(i,&BoardHandle,NULL,0);
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);

                *count += 1;
                *available_cards = (struct device_info *)
                        realloc(*available_cards, *count * sizeof(struct device_info));
                memset(*available_cards + *count - 1, 0, sizeof(struct device_info));
                sprintf((*available_cards)[*count - 1].id, "deltacast:device=%d", *count - 1);
                (*available_cards)[*count - 1].repeatable = false;

                if (Result == VHDERR_NOERROR)
                {
                        std::string board{"Unknown DELTACAST type"};
                        auto it = board_type_map.find(BoardType);
                        if (it != board_type_map.end()) {
                                board = it->second;
                        }
                        snprintf((*available_cards)[*count - 1].name,
                                        sizeof (*available_cards)[*count - 1].name - 1,
                                        "DELTACAST %s", board.c_str());
                        VHD_CloseBoardHandle(BoardHandle);
                }
        }

}

static void *display_deltacast_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        struct state_deltacast *s;
        ULONG             Result,DllVersion,NbBoards,ChnType;
        ULONG             BrdId = 0;

        s = (struct state_deltacast *)calloc(1, sizeof(struct state_deltacast));
        s->magic = DELTACAST_MAGIC;
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->frames = 0;
        
        gettimeofday(&s->tv, NULL);
        
        s->initialized = FALSE;
        if(flags & DISPLAY_FLAG_AUDIO_EMBEDDED) {
                s->play_audio = TRUE;
        } else {
                s->play_audio = FALSE;
        }
        
        s->BoardHandle = s->StreamHandle = s->SlotHandle = NULL;
        s->audio_configured = FALSE;

        if(fmt && strcmp(fmt, "help") == 0) {
                show_help();
                vf_free(s->frame);
                free(s);
                return &display_init_noerr;
        }
        
        if(fmt)
        {
                char *tmp = strdup(fmt);
                char *save_ptr = NULL;
                char *tok;
                
                tok = strtok_r(tmp, ":", &save_ptr);
                if(!tok)
                {
                        free(tmp);
                        show_help();
                        goto error;
                }
                if (strncasecmp(tok, "device=", strlen("device=")) == 0) {
                        BrdId = atoi(tok + strlen("device="));
                } else {
                        fprintf(stderr, "Unknown option: %s\n\n", tok);
                        free(tmp);
                        show_help();
                        goto error;
                }
                free(tmp);
        }

        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08lX\n",
                                Result);
                goto error;
        }
        if (NbBoards == 0) {
                fprintf(stderr, "[DELTACAST] No DELTA board detected, exiting...\n");
                goto error;
        }
        
        if(BrdId >= NbBoards) {
                fprintf(stderr, "[DELTACAST] Wrong index %lu. Found %lu cards.\n", BrdId, NbBoards);
                goto error;
        }

        /* Open a handle on first DELTA-hd/sdi/codec board */
        Result = VHD_OpenBoardHandle(BrdId,&s->BoardHandle,NULL,0);
        if (Result != VHDERR_NOERROR)
        {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot open DELTA board %lu handle. Result = 0x%08lX\n", BrdId, Result);
                goto error;
        }

        if (!delta_set_nb_channels(BrdId, s->BoardHandle, 0, 1)) {
                goto error;
        }

        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_TX0_TYPE, &ChnType);
        if((ChnType!=VHD_CHNTYPE_SDSDI)&&(ChnType!=VHD_CHNTYPE_HDSDI)&&(ChnType!=VHD_CHNTYPE_3GSDI)) {
                fprintf(stderr, "[DELTACAST] ERROR : The selected channel is not an SDI one\n");
                goto bad_channel;
        }
        
        /* Disable RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,FALSE);
        
        /* Select a 1/1 clock system */
        VHD_SetBoardProperty(s->BoardHandle,VHD_SDI_BP_CLOCK_SYSTEM,VHD_CLOCKDIV_1);

        pthread_mutex_init(&s->lock, NULL);
                  
	return s;

bad_channel:
        VHD_CloseBoardHandle(s->BoardHandle);
error:
        vf_free(s->frame);
        free(s);
        return NULL;
}

static void display_deltacast_run(void *state)
{
        UNUSED(state);
}

static void display_deltacast_done(void *state)
{
        struct state_deltacast *s = (struct state_deltacast *)state;

        if(s->initialized) {
                if(s->SlotHandle)
                        VHD_UnlockSlotHandle(s->SlotHandle);
                VHD_StopStream(s->StreamHandle);
                VHD_CloseStreamHandle(s->StreamHandle);
                VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
                VHD_CloseBoardHandle(s->BoardHandle);
        }

        vf_free(s->frame);
        free(s);
}

static int display_deltacast_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {v210, UYVY, RAW};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, UPPER_FIELD_FIRST, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};
        
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
                                desc->ch_count = std::max(desc->ch_count, 16);
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 3;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_deltacast_reconfigure_audio(void *state, int quant_samples, int channels,
                                int sample_rate)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        int i;

        assert(channels <= 16);

        pthread_mutex_lock(&s->lock);
        s->audio_configured = FALSE;
        for(i = 0; i < 16; ++i) {
                ring_buffer_destroy(s->audio_channels[i]);
                s->audio_channels[i] = NULL;
        }
        free(s->audio_tmp);

        s->audio_desc.bps = quant_samples / 8;
        s->audio_desc.ch_count = channels;
        s->audio_desc.sample_rate = sample_rate;

        for(i = 0; i < channels; ++i) {
                s->audio_channels[i] = ring_buffer_init(s->audio_desc.bps * s->audio_desc.sample_rate);
        }
        s->audio_tmp = (char *) malloc(s->audio_desc.bps * s->audio_desc.sample_rate);

        /* Configure audio info */
        memset(&s->AudioInfo, 0, sizeof(VHD_AUDIOINFO));
        for(i = 0; i < channels; ++i) {
                VHD_AUDIOCHANNEL *pAudioChn=NULL;
                pAudioChn = &s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4];
                pAudioChn->Mode = VHD_AM_MONO;
                switch(quant_samples) {
                        case 16:
                                pAudioChn->BufferFormat = VHD_AF_16; 
                                break;
                        case 20:
                                pAudioChn->BufferFormat = VHD_AF_20; 
                                break;
                        case 24:
                                pAudioChn->BufferFormat = VHD_AF_24; 
                                break;
                        default:
                                fprintf(stderr, "[DELTACAST] Unsupported PCM audio: %d bits.\n", quant_samples);
                                pthread_mutex_unlock(&s->lock);
                                return FALSE;
                }
                pAudioChn->pData = new BYTE[s->audio_desc.bps * s->audio_desc.sample_rate];
        }

        s->audio_configured = TRUE;
        pthread_mutex_unlock(&s->lock);

        return TRUE;
}

static void display_deltacast_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        int i;
        int channel_len = frame->data_len / frame->ch_count;

        pthread_mutex_lock(&s->lock);
        for(i = 0; i < frame->ch_count; ++i) {
                 demux_channel(s->audio_tmp, frame->data, frame->bps, frame->data_len, frame->ch_count, i);
                 ring_buffer_write(s->audio_channels[i], s->audio_tmp, channel_len);
        }
        pthread_mutex_unlock(&s->lock);
}

static const struct video_display_info display_deltacast_info = {
        display_deltacast_probe,
        display_deltacast_init,
        display_deltacast_run,
        display_deltacast_done,
        display_deltacast_getf,
        display_deltacast_putf,
        display_deltacast_reconfigure,
        display_deltacast_get_property,
        display_deltacast_put_audio_frame,
        display_deltacast_reconfigure_audio,
};

REGISTER_MODULE(deltacast, &display_deltacast_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


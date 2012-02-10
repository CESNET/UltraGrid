/*
 * FILE:    video_display/deltacast.cpp
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
#include "video_display/deltacast.h"
#include "debug.h"
#include "audio/audio.h"
#include "audio/utils.h"
#include "utils/ring_buffer.h"

#include "VideoMasterHD_Core.h"
#include "VideoMasterHD_Sdi.h"
#include "VideoMasterHD_Sdi_Audio.h"

#define DELTACAST_MAGIC DISPLAY_DELTACAST_ID

#ifdef __cplusplus
} // END of extern "C"
#endif

const struct deltacast_frame_mode_t deltacast_frame_modes[] = {
        {VHD_VIDEOSTD_S274M_1080p_25Hz, "SMPTE 274M 1080p 25 Hz", 1920u, 1080u, 25.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S274M_1080p_30Hz, "SMPTE 274M 1080p 30 Hz", 1920u, 1080u, 30.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S274M_1080i_50Hz, "SMPTE 274M 1080i 50 Hz", 1920u, 1080u, 25.0, UPPER_FIELD_FIRST},
        {VHD_VIDEOSTD_S274M_1080i_60Hz, "SMPTE 274M 1080i 60 Hz", 1920u, 1080u, 29.97, UPPER_FIELD_FIRST},
        {VHD_VIDEOSTD_S296M_720p_50Hz, "SMPTE 296M 720p 50 Hz", 1280u, 720u, 50.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S296M_720p_60Hz, "SMPTE 296M 720p 60 Hz", 1280u, 720u, 60.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S259M_PAL, "SMPTE 259M PAL", 720u, 576u, 25.0, UPPER_FIELD_FIRST},
        {VHD_VIDEOSTD_S259M_NTSC, "SMPTE 259M NTSC", 720u, 487u, 29.97, UPPER_FIELD_FIRST},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 24 Hz", 1920u, 1080u, 24.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S274M_1080p_60Hz, "SMPTE 274M 1080p 60 Hz", 1920u, 1080u, 60.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S274M_1080p_24Hz, "SMPTE 274M 1080p 50 Hz", 1920u, 1080u, 50.0, PROGRESSIVE},
        {VHD_VIDEOSTD_S274M_1080psf_24Hz, "SMPTE 274M 1080psf 24 Hz", 1920u, 1080u, 24.0, SEGMENTED_FRAME},
        {VHD_VIDEOSTD_S274M_1080psf_25Hz, "SMPTE 274M 1080psf 25 Hz", 1920u, 1080u, 25.0, SEGMENTED_FRAME},
        {VHD_VIDEOSTD_S274M_1080psf_30Hz, "SMPTE 274M 1080psf 30 Hz", 1920u, 1080u, 30.0, SEGMENTED_FRAME}
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
        struct audio_frame  audio_frame;
        struct ring_buffer  *audio_channels[16];
        char            *audio_tmp;
 };

static void show_help(void);

static void show_help(void)
{
        printf("deltacast (output) options:\n");
        printf("\t-d deltacast:<device_number>\n");
}

struct video_frame *
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

int display_deltacast_putf(void *state, char *frame)
{
        int tmp;
        struct state_deltacast *s = (struct state_deltacast *)state;
        struct timeval tv;
        int i;
        ULONG Result;

        UNUSED(frame);

        assert(s->magic == DELTACAST_MAGIC);
        
        pthread_mutex_lock(&s->lock);
        if(s->play_audio && s->audio_configured) {
                /* Retrieve the number of needed samples */
                for(i = 0; i < s->audio_frame.ch_count; ++i) {
                        s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].DataSize = 0;
                }
                Result = VHD_SlotEmbedAudio(s->SlotHandle,&s->AudioInfo);
                if (Result != VHDERR_BUFFERTOOSMALL)
                {
                        fprintf(stderr, "[DELTACAST] ERROR : Cannot embed audio on TX0 stream. Result = 0x%08X\n",Result);
                } else {
                        for(i = 0; i < s->audio_frame.ch_count; ++i) {
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
                        fprintf(stderr, "[DELTACAST] ERROR : Cannot embed audio on TX0 stream. Result = 0x%08X\n",Result);
                }
        }
        pthread_mutex_unlock(&s->lock);

        VHD_UnlockSlotHandle(s->SlotHandle);
        s->SlotHandle = NULL;
        
        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = s->frames / seconds;
                fprintf(stdout, "%lu frames in %g seconds = %g FPS\n",
                        s->frames, seconds, fps);
                s->tv = tv;
                s->frames = 0;
        }
        s->frames++;

        return TRUE;
}

int
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
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_DEPTH,4);
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_PRELOAD,2);

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


void *display_deltacast_init(char *fmt, unsigned int flags)
{
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
        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                s->play_audio = TRUE;
        } else {
                s->play_audio = FALSE;
        }
        
        s->BoardHandle = s->BoardHandle = s->SlotHandle = NULL;
        s->audio_configured = FALSE;
        pthread_mutex_init(&s->lock, NULL);

        if(fmt && strcmp(fmt, "help") == 0) {
                show_help();
                goto error;
        }
        
        if(fmt)
        {
                char *save_ptr = NULL;
                char *tok;
                
                tok = strtok_r(fmt, ":", &save_ptr);
                if(!tok)
                {
                        show_help();
                        goto error;
                }
                BrdId = atoi(tok);
        }

        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08X\n",
                                Result);
                goto error;
        }
        if (NbBoards == 0) {
                fprintf(stderr, "[DELTACAST] No DELTA board detected, exiting...\n");
                goto error;
        }
        
        if(BrdId >= NbBoards) {
                fprintf(stderr, "[DELTACAST] Wrong index %d. Found %d cards.\n", BrdId, NbBoards);
                goto error;
        }

        /* Open a handle on first DELTA-hd/sdi/codec board */
        Result = VHD_OpenBoardHandle(BrdId,&s->BoardHandle,NULL,0);
        if (Result != VHDERR_NOERROR)
        {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot open DELTA board %u handle. Result = 0x%08X\n",BrdId,Result);
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
                  
	return s;

bad_channel:
        VHD_CloseBoardHandle(s->BoardHandle);
error:
        free(s);
        return NULL;
}

void display_deltacast_run(void *state)
{
        UNUSED(state);
}

void display_deltacast_finish(void *state)
{
        UNUSED(state);
}

void display_deltacast_done(void *state)
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

display_type_t *display_deltacast_probe(void)
{
        display_type_t *dtype;

        dtype = (display_type_t *) malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_DELTACAST_ID;
                dtype->name = "deltacast";
                dtype->description = "DELTACAST card";
        }
        return dtype;
}

int display_deltacast_get_property(void *state, int property, void *val, size_t *len)
{
        codec_t codecs[] = {v210, UYVY, RAW};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, UPPER_FIELD_FIRST, SEGMENTED_FRAME};
        
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
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
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
                default:
                        return FALSE;
        }
        return TRUE;
}

int display_deltacast_reconfigure_audio(void *state, int quant_samples, int channels,
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
        free(s->audio_frame.data);
        free(s->audio_tmp);

        s->audio_frame.bps = quant_samples / 8;
        s->audio_frame.ch_count = channels;
        s->audio_frame.sample_rate = sample_rate;

        s->audio_frame.max_size = s->audio_frame.bps * s->audio_frame.ch_count * s->audio_frame.sample_rate;
        s->audio_frame.data = (char *) malloc(s->audio_frame.max_size);
        for(i = 0; i < channels; ++i) {
                s->audio_channels[i] = ring_buffer_init(s->audio_frame.bps * s->audio_frame.sample_rate);
        }
        s->audio_tmp = (char *) malloc(s->audio_frame.bps * s->audio_frame.sample_rate);

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
                                return FALSE;
                }
                pAudioChn->pData = new BYTE[s->audio_frame.bps * s->audio_frame.sample_rate];
        }

        s->audio_configured = TRUE;
        pthread_mutex_unlock(&s->lock);

        return TRUE;
}

struct audio_frame * display_deltacast_get_audio_frame(void *state)
{
        struct state_deltacast *s = (struct state_deltacast *)state;

        return &s->audio_frame;
}

void display_deltacast_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        int i;
        int channel_len = frame->data_len / frame->ch_count;

        for(i = 0; i < s->audio_frame.ch_count; ++i) {
                 demux_channel(s->audio_tmp, frame->data, frame->bps, frame->data_len, frame->ch_count, i);
                 ring_buffer_write(s->audio_channels[i], s->audio_tmp, channel_len);
        }
}


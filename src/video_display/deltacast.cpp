/**
 * @file   video_display/deltacast.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * code is written by DELTACAST's VideoMaster SDK example SampleTX
 * and SDL_TXAudio (last update according to 6.32)
 *
 * @sa deltacast_common.hpp for common DELTACAST information
 */
/*
 * Copyright (c) 2012-2025 CESNET, zájnové sdružení právnických osob
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

#include <algorithm>                  // for max
#include <cassert>                    // for assert
#include <cmath>                      // for fabs
#include <cstdint>                    // for uint32_t
#include <cstdio>                     // for printf, snprintf
#include <cstdlib>                    // for NULL, free, atoi, calloc, malloc
#include <cstring>                    // for memcpy, memset, strlen, strcmp
#include <pthread.h>                  // for pthread_mutex_unlock, pthread_m...
#include <string>                     // for basic_string, string
#include <sys/time.h>                 // for timeval, gettimeofday
#include <unordered_map>              // for operator!=, unordered_map, _Nod...
#include <utility>                    // for pair

#include "host.h"
#include "debug.h"
#include "deltacast_common.hpp"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "utils/color_out.h"          // for color_printf
#include "utils/macros.h"             // for IS_KEY_PREFIX
#include "utils/ring_buffer.h"

#define DELTACAST_MAGIC to_fourcc('v', 'd', 'D', 'C')
#define MOD_NAME "[DELTACAST display] "

struct state_deltacast {
        uint32_t            magic;

        struct timeval      tv;
        struct video_frame *frame;
        struct tile        *tile;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                started;
        HANDLE              BoardHandle, StreamHandle;
        HANDLE              SlotHandle;
        unsigned            channel;
        ULONG               SlotsDroppedLast;

        pthread_mutex_t     lock;

        bool                play_audio;
        bool                audio_configured;
        VHD_AUDIOINFO     AudioInfo;
        SHORT            *pSample;
        struct audio_desc  audio_desc;
        struct ring_buffer  *audio_channels[16];
        char            *audio_tmp;
};

static void display_deltacast_done(void *state);

static void
show_help(bool full)
{
        printf("deltacast (output) options:\n");
        color_printf("\t" TBOLD(
            TRED("-d deltacast") "[:device=<index>][:channel=<ch>]"
            "[:ch_layout=RT]") "\n");
        color_printf("\t" TBOLD("-d deltacast:[full]help") "\n");

        printf("\nOptions:\n");
        color_printf("\t" TBOLD("device") " - board index\n");
        color_printf("\t" TBOLD("channel") " - card channel index (default 0)\n");
        delta_print_ch_layout_help(full);

        print_available_delta_boards(full);

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

        if (!s->started) {
                return s->frame;
        }

        Result = VHD_LockSlotHandle(s->StreamHandle, &s->SlotHandle);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unable to lock slot.\n");
                return NULL;
        }
        Result = VHD_GetSlotBuffer(s->SlotHandle,VHD_SDI_BT_VIDEO,&pBuffer,&BufferSize);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unable to get buffer.\n");
                return NULL;
        }
        
        s->tile->data = (char *) pBuffer;
        s->tile->data_len = BufferSize;

        return s->frame;
}

static void
print_slot_stats(struct state_deltacast *s, bool final_summary)
{
        ULONG SlotsCount, SlotsDropped;
        VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_SLOTS_DROPPED,
                              &SlotsDropped);
        if (SlotsDropped == s->SlotsDroppedLast && !final_summary) {
                return;
        }
        VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_SLOTS_COUNT,
                              &SlotsCount);
        log_msg(SlotsDropped > 0 ? LOG_LEVEL_WARNING : LOG_LEVEL_INFO,
                MOD_NAME "%" PRIu_ULONG " frames sent (%" PRIu_ULONG
                         " dropped)\n",
                SlotsCount, SlotsDropped);
        s->SlotsDroppedLast = SlotsDropped;
}

static bool display_deltacast_putf(void *state, struct video_frame *frame, long long nonblock)
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
                        MSG(ERROR,
                            "ERROR : Cannot embed audio on TX%d stream. Result "
                            "= 0x%08" PRIX_ULONG "\n",
                            s->channel, Result);
                } else {
                        for(i = 0; i < s->audio_desc.ch_count; ++i) {
                                int ret;
                                ret = ring_buffer_read(s->audio_channels[i], (char *) s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].pData, s->AudioInfo.pAudioGroups[i / 4].pAudioChannels[i % 4].DataSize);
                                if(!ret) {
                                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Buffer underflow for channel %d.\n", i);
                                }
                                s->AudioInfo.pAudioGroups[0].pAudioChannels[0].DataSize = ret;
                        }
                }
                /* Embed audio */
                Result = VHD_SlotEmbedAudio(s->SlotHandle,&s->AudioInfo);
                if (Result != VHDERR_NOERROR)
                {
                        MSG(ERROR,
                            "ERROR : Cannot embed audio on TX%d stream. Result "
                            "= 0x%08" PRIX_ULONG "\n",
                            s->channel, Result);
                }
        }
        pthread_mutex_unlock(&s->lock);

        VHD_UnlockSlotHandle(s->SlotHandle);
        s->SlotHandle = NULL;
        
        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                display_print_fps(MOD_NAME, seconds, (int) s->frames, frame->fps);
                print_slot_stats(s, false);

                s->tv = tv;
                s->frames = 0;
        }
        s->frames++;

        return true;
}

static bool
display_deltacast_reconfigure(void *state, struct video_desc desc)
{
        struct state_deltacast            *s = (struct state_deltacast *)state;
        int VideoStandard = 0;
        VHD_CLOCKDIVISOR clock_system = NB_VHD_CLOCKDIVISORS;
        int i;
        ULONG Result;

        if (s->SlotHandle != nullptr) {
                VHD_UnlockSlotHandle(s->SlotHandle);
                s->SlotHandle = nullptr;
        }

        if (s->started) {
                VHD_StopStream(s->StreamHandle);
                s->started = false;
        }
        if (s->StreamHandle != nullptr) {
                VHD_CloseStreamHandle(s->StreamHandle);
                s->StreamHandle = nullptr;
        }

        assert(desc.tile_count == 1);

        s->tile->width = desc.width;
        s->tile->height = desc.height;
        s->frame->color_spec = desc.color_spec;
        s->frame->interlacing = desc.interlacing;
        s->frame->fps = desc.fps;
        
        for (i = 0; i < NB_VHD_VIDEOSTANDARDS; ++i)
        {
                for (bool is_1001 : { false, true }) {
                        const auto &mode = deltacast_get_mode_info(i, is_1001);
                        if (fabs(desc.fps - mode.fps) > 0.01 ||
                            desc.interlacing != mode.interlacing ||
                            desc.width != mode.width ||
                            desc.height != mode.height) {
                                continue;
                        }
                        VideoStandard = i;
                        clock_system =
                            is_1001 ? VHD_CLOCKDIV_1 : VHD_CLOCKDIV_1001;
                        MSG(NOTICE, "%s mode selected.\n",
                            deltacast_get_mode_name(i, is_1001));
                        break;
                }
        }
        if(i == NB_VHD_VIDEOSTANDARDS) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Failed to obtain video format for incoming video: %dx%d @ %2.2f %s\n", desc.width, desc.height,
                                                                        (double) desc.fps, get_interlacing_description(desc.interlacing));

                return false;
        }
        
        const VHD_STREAMTYPE StrmType = delta_tx_ch_to_stream_t(s->channel);
        ULONG ProcessingMode = 0;
        if (desc.color_spec == RAW) {
                ProcessingMode = VHD_SDI_STPROC_RAW;
        } else if (s->play_audio == TRUE) {
                ProcessingMode = VHD_SDI_STPROC_JOINED;
        } else {
                ProcessingMode = VHD_SDI_STPROC_DISJOINED_VIDEO;
        }
        Result = VHD_OpenStreamHandle(s->BoardHandle, StrmType, ProcessingMode,
                                      nullptr, &s->StreamHandle, nullptr);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Failed to open stream handle.\n");
                return false;
        }
        
        VHD_SetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD,VideoStandard);
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_DEPTH,2);
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFERQUEUE_PRELOAD,0);
        VHD_SetBoardProperty(s->BoardHandle, VHD_SDI_BP_CLOCK_SYSTEM,
                             clock_system);

        Result = VHD_StartStream(s->StreamHandle);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unable to start stream.\n");  
                return false;
        }
        
        s->started = true;
        return true;
}

static void display_deltacast_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        UNUSED(deleter);
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
                snprintf((*available_cards)[*count - 1].dev, sizeof (*available_cards)[*count - 1].dev, ":device=%d", *count - 1);
                snprintf((*available_cards)[*count - 1].extra, sizeof (*available_cards)[*count - 1].extra, R"("embeddedAudioAvailable":"t")");
                (*available_cards)[*count - 1].repeatable = false;

                if (Result == VHDERR_NOERROR)
                {
                        const char *board = delta_get_board_type_name(BoardType);
                        snprintf_ch((*available_cards)[*count - 1].name,
                                    "DELTACAST %s", board);
                        VHD_CloseBoardHandle(BoardHandle);
                }
        }

}

static bool
parse_fmt(struct state_deltacast *s, char *fmt, ULONG *BrdId,
          ULONG *NbRxRequired, ULONG *NbTxRequired)
{
        char *save_ptr = nullptr;
        while (char *tok = strtok_r(fmt, ":", &save_ptr)) {
                fmt = nullptr;
                if (IS_KEY_PREFIX(tok, "device")) {
                        *BrdId = std::stoi(strchr(tok, '=') + 1);
                } else if (IS_KEY_PREFIX(tok, "channel")) {
                        s->channel = std::stoi(strchr(tok, '=') + 1);
                        if (s->channel > MAX_DELTA_CH) {
                                MSG(ERROR, "Index %u out of bound!\n",
                                    s->channel);
                                return false;
                        }
                } else if (IS_KEY_PREFIX(tok, "ch_layout")) {
                        int val = std::stoi(strchr(tok, '=') + 1);
                        *NbRxRequired = val / 10;
                        *NbTxRequired = val % 10;
                } else {
                        MSG(ERROR, "Unknown option: %s\n\n", tok);
                        show_help(false);
                        return false;
                }
        }
        return true;
}

static void *display_deltacast_init(struct module *parent, const char *fmt, unsigned int flags)
{
#define HANDLE_ERROR display_deltacast_done(s); return nullptr;
        UNUSED(parent);
        struct state_deltacast *s;
        ULONG             Result,DllVersion,NbBoards,ChnType;
        ULONG             BrdId = 0;
        ULONG             NbRxRequired = 0;
        ULONG             NbTxRequired = 0;

        if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
                show_help(strcmp(fmt, "fullhelp") == 0);
                return INIT_NOERR;
        }

        s = (struct state_deltacast *)calloc(1, sizeof(struct state_deltacast));
        s->magic = DELTACAST_MAGIC;
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        gettimeofday(&s->tv, NULL);
        pthread_mutex_init(&s->lock, NULL);
        s->play_audio = (flags & DISPLAY_FLAG_AUDIO_EMBEDDED) != 0U;
        
        char *tmp = strdup(fmt);
        if (!parse_fmt(s, tmp, &BrdId, &NbRxRequired, &NbTxRequired)) {
                free(tmp);
                HANDLE_ERROR
        }
        free(tmp);
        
        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "Cannot query VideoMasterHD");
                HANDLE_ERROR
        }
        if (NbBoards == 0) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] No DELTA board detected, exiting...\n");
                HANDLE_ERROR
        }
        
        if(BrdId >= NbBoards) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Wrong index %" PRIu_ULONG ". Found %" PRIu_ULONG " cards.\n", BrdId, NbBoards);
                HANDLE_ERROR
        }

        /* Open a handle on first DELTA-hd/sdi/codec board */
        Result = VHD_OpenBoardHandle(BrdId,&s->BoardHandle,NULL,0);
        if (Result != VHDERR_NOERROR)
        {
                DELTA_PRINT_ERROR(
                    Result, "Cannot open DELTA board %" PRIu_ULONG " handle",
                    BrdId);
                HANDLE_ERROR
        }

        if (NbRxRequired == 0 && NbTxRequired == 0) {
                NbTxRequired = s->channel + 1;
        }
        if (!delta_set_nb_channels(BrdId, s->BoardHandle, NbRxRequired,
                                   NbTxRequired)) {
                HANDLE_ERROR
        }

        const auto Property = (VHD_CORE_BOARDPROPERTY) DELTA_CH_TO_VAL(
            s->channel, VHD_CORE_BP_TX0_TYPE, VHD_CORE_BP_TX4_TYPE);
        VHD_GetBoardProperty(s->BoardHandle, Property, &ChnType);
        if (!delta_chn_type_is_sdi(ChnType)) {
                MSG(ERROR, "ERROR : The selected channel is not a SDI one\n");
                HANDLE_ERROR
        }
        
        /* Disable RX0-TX0 by-pass relay loopthrough */
        delta_set_loopback_state(s->BoardHandle, (int) s->channel, FALSE);

	return s;
#undef HANDLE_ERROR
}

static void display_deltacast_done(void *state)
{
        struct state_deltacast *s = (struct state_deltacast *)state;
        assert(s != nullptr);
        print_slot_stats(s, true);

        if (s->SlotHandle != nullptr) {
                VHD_UnlockSlotHandle(s->SlotHandle);
        }
        if (s->started) {
                VHD_StopStream(s->StreamHandle);
        }
        if (s->StreamHandle != nullptr) {
                VHD_CloseStreamHandle(s->StreamHandle);
        }
        if (s->BoardHandle != nullptr) {
                delta_set_loopback_state(s->BoardHandle, (int) s->channel,
                                         TRUE);
                VHD_CloseBoardHandle(s->BoardHandle);
        }

        pthread_mutex_destroy(&s->lock);
        vf_free(s->frame);
        free(s);
}

static bool
display_deltacast_get_property(void *state, int property, void *val,
                               size_t *len)
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
                                return false;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return false;
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
                                return false;
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
                        return false;
        }
        return true;
}

static bool
display_deltacast_reconfigure_audio(void *state, int quant_samples,
                                    int channels, int sample_rate)
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
                                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unsupported PCM audio: %d bits.\n", quant_samples);
                                pthread_mutex_unlock(&s->lock);
                                return false;
                }
                pAudioChn->pData = new BYTE[s->audio_desc.bps * s->audio_desc.sample_rate];
        }

        s->audio_configured = TRUE;
        pthread_mutex_unlock(&s->lock);

        return true;
}

static void display_deltacast_put_audio_frame(void *state, const struct audio_frame *frame)
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
        NULL, // _run
        display_deltacast_done,
        display_deltacast_getf,
        display_deltacast_putf,
        display_deltacast_reconfigure,
        display_deltacast_get_property,
        display_deltacast_put_audio_frame,
        display_deltacast_reconfigure_audio,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(deltacast, &display_deltacast_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


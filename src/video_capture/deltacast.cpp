/**
 * @file   video_capture/deltacast.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * code is written by DELTACAST's VideoMaster SDK example SampleRX and
 * SampleRX4K
 *
 * @sa deltacast_common.hpp for common DELTACAST information
 */
/*
 * Copyright (c) 2011-2025 CESNET
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
#include <cstdio>                     // for printf, NULL, snprintf
#include <cstdlib>                    // for free, atoi, calloc
#include <cstring>                    // for strlen, strcmp, memset, strdup
#include <ostream>                    // for char_traits, basic_ostream, ope...
#include <sys/time.h>                 // for timeval, gettimeofday

#include "audio/types.h"
#include "audio/utils.h"
#include "compat/strings.h"           // for strncasecmp, strcasecmp
#include "debug.h"
#include "deltacast_common.hpp"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"             // for IS_KEY_PREFIX
#include "video.h"
#include "video_capture.h"
#include "video_capture_params.h"

#define MOD_NAME "[DELTACAST] "

using namespace std;

struct vidcap_deltacast_state {
        struct video_frame *frame;
        struct tile        *tile;
        HANDLE            BoardHandle, StreamHandle;
        HANDLE            SlotHandle;
        unsigned int      channel;

        struct audio_frame audio_frame;
        
        struct       timeval t, t0;
        int          frames;

        ULONG             AudioBufferSize;
        VHD_AUDIOCHANNEL *pAudioChn;
        VHD_AUDIOINFO     AudioInfo;
        ULONG             ClockSystem,VideoStandard;
        unsigned int      grab_audio:1;
        unsigned int      autodetect_format:1;
        bool              quad_channel;

        unsigned int      initialize_flags;
        bool              initialized;
};

static void vidcap_deltacast_done(void *state);

static void
usage(bool full)
{
        color_printf("Usage:\n");
        color_printf("\t" TBOLD(TRED("-t "
                     "deltacast") "[:device=<index>][:channel=<idx>][:quad-link][:ch_layout=RL][:mode=<mode>]["
                     ":codec=<codec>]") "\n");
        color_printf("\t" TBOLD("-t "
                                "deltacast:[full]help") "\n");

        printf("\nOptions:\n");
        color_printf("\t" TBOLD("device") " - board index\n");
        color_printf("\t" TBOLD("channel") " - card channel index (default 0)\n");
        color_printf("\t" TBOLD("quad-link") " - quad-link (4k/8k) capture\n");
        delta_print_ch_layout_help(full);
        color_printf("\t" TBOLD("mode") " - capture mode (see below), if not given autodetected\n");
        color_printf("\t" TBOLD("codec") " - pixel format to capture (see list below)\n");

        print_available_delta_boards(full);
        printf("\n");

        if (full) {
                printf("Available modes:\n");
                for (int i = 0; i < NB_VHD_VIDEOSTANDARDS; ++i) {
                        for (bool is_1001 : {true, false}) {
                                const char *name =
                                    deltacast_get_mode_name(i, is_1001);
                                if (name == nullptr) {
                                        continue;
                                }
                                printf("\t%d: " TBOLD("%s") "\n", i, name);
                        }
                }
        } else {
                printf("(use " TBOLD(":fullhelp") " to list modes)\n");
        }
        printf("\nAvailable codecs:\n");
        printf("\tUYVY\n");
        printf("\tv210\n");
        printf("\traw\n");

        printf("\nDefault board is 0. If mode is omitted, it will be autodetected "
                        "(except of UHD modes). Default codec is UYVY.\n");
}

static void vidcap_deltacast_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;

        ULONG             Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result == VHDERR_NOERROR) {
                *available_cards = (struct device_info *) calloc(NbBoards, sizeof(struct device_info));
                *count = NbBoards;
                for (ULONG i = 0; i < NbBoards; ++i) {
                        auto& card = (*available_cards)[i];
                        snprintf(card.dev, sizeof(card.dev), ":device=%" PRIu_ULONG, i);
                        snprintf(card.name, sizeof(card.name), "DELTACAST SDI board %" PRIu_ULONG, i);
                        snprintf(card.extra, sizeof(card.extra), "\"embeddedAudioAvailable\":\"t\"");
                }
	}
}

class delta_init_exception {
};

#define DELTA_TRY_CMD(cmd, msg) \
        do {\
                Result = cmd;\
                if (Result != VHDERR_NOERROR) {\
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] " msg " Result = 0x%08" PRIX_ULONG "\n", Result);\
                        throw delta_init_exception();\
                }\
        } while(0)

/**
 * Function initialize is intended to be called repeatedly if no signal detected
 * in vidcap_deltacast_init(). This will be tried every time grab is called and until
 * initialized.
 */
static bool wait_for_channel(struct vidcap_deltacast_state *s)
{
        ULONG             Result;
        ULONG             Packing;
        ULONG             Status = 0;
        ULONG             Interface = 0;

        /* Wait for channel locked */
        Result =
#ifdef VHD_MIN_6_21
            /* Check Dual and Link A status */
            VHD_GetChannelProperty(s->BoardHandle, VHD_RX_CHANNEL, s->channel,
                                   VHD_CORE_CP_STATUS, &Status);

#else
            VHD_GetBoardProperty(s->BoardHandle,
                                      DELTA_CH_TO_VAL(s->channel,
                                                      VHD_CORE_BP_RX0_STATUS,
                                                      VHD_CORE_BP_RX4_STATUS),
                                      &Status);
#endif

        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot get channel status. Result = 0x%08" PRIX_ULONG "\n",Result);
                throw delta_init_exception();
        }

        // no signal was detected on the wire
        if(Status & VHD_CORE_RXSTS_UNLOCKED) {
                return false;
        }

        /* Auto-detect clock system */
        Result =
#ifdef VHD_MIN_6_21
            VHD_GetChannelProperty(s->BoardHandle, VHD_RX_CHANNEL, s->channel,
                                   VHD_SDI_CP_CLOCK_DIVISOR, &s->ClockSystem);
#else
            VHD_GetBoardProperty(s->BoardHandle,
                                      DELTA_CH_TO_VAL(s->channel,
                                                      VHD_SDI_BP_RX0_CLOCK_DIV,
                                                      VHD_SDI_BP_RX4_CLOCK_DIV),
                                      &s->ClockSystem);
#endif

        if(Result != VHDERR_NOERROR) {
                MSG(ERROR,
                    "ERROR : Cannot detect incoming clock "
                    "system from RX%u. Result = 0x%08" PRIX_ULONG "\n",
                    s->channel, Result);
                throw delta_init_exception();
        } else {
                printf("\nIncoming clock system : %s\n",(s->ClockSystem==VHD_CLOCKDIV_1)?"European":"American");
        }

        /* Create a logical stream to receive from RX0 connector */
        const VHD_STREAMTYPE StrmType = delta_rx_ch_to_stream_t(s->channel);
        ULONG ProcessingMode = 0;
        if(!s->autodetect_format && s->frame->color_spec == RAW) {
                ProcessingMode = VHD_SDI_STPROC_RAW;
        } else if ((s->initialize_flags & VIDCAP_FLAG_AUDIO_EMBEDDED) != 0U) {
                ProcessingMode = VHD_SDI_STPROC_JOINED;
        } else {
                ProcessingMode = VHD_SDI_STPROC_DISJOINED_VIDEO;
        }
        Result = VHD_OpenStreamHandle(s->BoardHandle, StrmType, ProcessingMode,
                                      nullptr, &s->StreamHandle, nullptr);
        if (Result != VHDERR_NOERROR) {
                MSG(ERROR,
                    "ERROR : Cannot open RX%d stream on DELTA-hd/sdi/codec "
                    "board handle. Result = 0x%08" PRIX_ULONG "\n",
                    s->channel, Result);
                throw delta_init_exception();
        }
        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_SDI_SP_INTERFACE,&Interface);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming interfaced from RX%u.", s->channel)
                throw delta_init_exception();
        }
        delta_print_intefrace_info(Interface);
        if (s->quad_channel && !delta_is_quad_channel_interface(Interface)) {
                delta_single_to_quad_links_interface(Status, &Interface, &s->VideoStandard);
        }

        if(s->autodetect_format) {
                /* Get auto-detected video standard */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD,&s->VideoStandard);

                if ((Result == VHDERR_NOERROR) && (s->VideoStandard != NB_VHD_VIDEOSTANDARDS)) {
                } else {
                        MSG(ERROR,
                            "Cannot detect incoming video standard from RX%u. "
                            "Result = 0x%08" PRIX_ULONG "\n",
                            s->channel, Result);
                        throw delta_init_exception();
                }
        }

        const auto &mode = deltacast_get_mode_info(
            s->VideoStandard, s->ClockSystem == VHD_CLOCKDIV_1001);
        if (mode.width == 0) {
                MSG(ERROR,
                    "Failed to obtain information about video format "
                    "%" PRIu_ULONG ".\n",
                    s->VideoStandard);
                throw delta_init_exception();
        }
        s->frame->fps         = mode.fps;
        s->frame->interlacing = mode.interlacing;
        s->tile->width        = mode.width;
        s->tile->height       = mode.height;
        printf("[DELTACAST] %s mode selected. %dx%d @ %2.2f %s\n",
               deltacast_get_mode_name(s->VideoStandard,
                                        s->ClockSystem == VHD_CLOCKDIV_1001),
               s->tile->width, s->tile->height, (double) s->frame->fps,
               get_interlacing_description(s->frame->interlacing));

        /* Configure stream */
        DELTA_TRY_CMD(VHD_SetStreamProperty(s->StreamHandle, VHD_SDI_SP_INTERFACE, Interface),
                        "Unable to set interface.");
        DELTA_TRY_CMD(VHD_SetStreamProperty(s->StreamHandle, VHD_SDI_SP_VIDEO_STANDARD, s->VideoStandard),
                        "Unable to set video standard.");
        DELTA_TRY_CMD(VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_TRANSFER_SCHEME,
                                VHD_TRANSFER_SLAVED),
                        "Unable to set transfer scheme.");

        if(s->autodetect_format) {
                Result = VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFER_PACKING, &Packing);
                if (Result != VHDERR_NOERROR) {
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unable to get pixel format\n");
                        throw delta_init_exception();
                }
                if(Packing == VHD_BUFPACK_VIDEO_YUV422_10) {
                        s->frame->color_spec = v210;
                } else if(Packing == VHD_BUFPACK_VIDEO_YUV422_8) {
                        s->frame->color_spec = UYVY;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Detected unknown pixel format!\n");
                        throw delta_init_exception();
                }
        }
        if(s->frame->color_spec == v210 || s->frame->color_spec == RAW) {
                Packing = VHD_BUFPACK_VIDEO_YUV422_10;
        } else if(s->frame->color_spec == UYVY) {
                Packing = VHD_BUFPACK_VIDEO_YUV422_8;
        } else {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unsupported pixel format\n");
                throw delta_init_exception();
        }
        printf("[DELTACAST] Pixel format '%s' selected.\n", get_codec_name(s->frame->color_spec));
        Result = VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFER_PACKING, Packing);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Unable to set pixel format\n");
                throw delta_init_exception();
        }

        if ((s->initialize_flags & VIDCAP_FLAG_AUDIO_EMBEDDED) == 0u) {
                s->audio_frame.ch_count = audio_capture_channels > 0 ? audio_capture_channels : max(DEFAULT_AUDIO_CAPTURE_CHANNELS, 2);
                if (s->audio_frame.ch_count != 1 &&
                                s->audio_frame.ch_count != 2) {
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST capture] Unable to handle channel count other than 1 or 2.\n");
                        throw delta_init_exception();
                }
                s->audio_frame.bps = 3;
                s->audio_frame.sample_rate = 48000;
                memset(&s->AudioInfo, 0, sizeof(VHD_AUDIOINFO));
                s->pAudioChn = &s->AudioInfo.pAudioGroups[0].pAudioChannels[0];
                s->pAudioChn->Mode = s->AudioInfo.pAudioGroups[0].pAudioChannels[1].Mode= s->audio_frame.ch_count == 1 ? VHD_AM_MONO : VHD_AM_STEREO;
                s->pAudioChn->BufferFormat = s->AudioInfo.pAudioGroups[0].pAudioChannels[1].BufferFormat=VHD_AF_24;

                /* Get the biggest audio frame size */
                s->AudioBufferSize = VHD_GetBlockSize(s->pAudioChn->BufferFormat, s->pAudioChn->Mode) *
                        VHD_GetNbSamples((VHD_VIDEOSTANDARD) s->VideoStandard, (VHD_CLOCKDIVISOR) s->ClockSystem, VHD_ASR_48000, 0);
                s->audio_frame.max_size = s->AudioBufferSize;
                /* Create audio buffer */
                s->audio_frame.data = (char *) calloc(1, s->audio_frame.max_size);
                s->pAudioChn->pData = (BYTE *)
                        s->audio_frame.data;

                LOG(LOG_LEVEL_NOTICE) << "[DELTACAST] Grabbing audio enabled: " << audio_desc_from_frame(&s->audio_frame) << "\n";
                s->grab_audio = TRUE;
        }

        /* Start stream */
        Result = VHD_StartStream(s->StreamHandle);
        if (Result == VHDERR_NOERROR){
                printf("[DELTACAST] Stream started.\n");
        } else {
                MSG(ERROR,
                    "ERROR : Cannot start RX%u stream on DELTA-hd/sdi/codec "
                    "board handle. Result = 0x%08" PRIX_ULONG "\n",
                    s->channel, Result);
                throw delta_init_exception();
        }

        return true;
}

static bool parse_fmt(struct vidcap_deltacast_state *s, char *init_fmt,
                      ULONG *BrdId, ULONG *NbRxRequired, ULONG *NbTxRequired)
{
        char *save_ptr = NULL;
        char *tok      = NULL;
        char *tmp      = init_fmt;

        while ((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (IS_KEY_PREFIX(tok, "device") ||
                    IS_KEY_PREFIX(
                        tok, "board")) { // compat, should be device= instead
                        *BrdId = atoi(strchr(tok, '=') + 1);
                } else if (IS_KEY_PREFIX(tok, "mode")) {
                        s->VideoStandard     = atoi(strchr(tok, '=') + 1);
                        s->autodetect_format = FALSE;
                } else if (IS_KEY_PREFIX(tok, "quad-link")) {
                        s->quad_channel = true;
                } else if (IS_KEY_PREFIX(tok, "codec")) {
                        tok = strchr(tok, '=') + 1;
                        if (strcasecmp(tok, "raw") == 0)
                                s->frame->color_spec = RAW;
                        else if (strcmp(tok, "UYVY") == 0)
                                s->frame->color_spec = UYVY;
                        else if (strcmp(tok, "v210") == 0)
                                s->frame->color_spec = v210;
                        else {
                                MSG(ERROR, "Wrong codec %s entered.\n", tok);
                                usage(false);
                                return false;
                        }
                } else if (IS_KEY_PREFIX(tok, "channel")) {
                        s->channel = stoi(strchr(tok, '=') + 1);
                        if (s->channel > MAX_DELTA_CH) {
                                MSG(ERROR, "Index %u out of bound!\n",
                                    s->channel);
                                return false;
                        }
                } else if (IS_KEY_PREFIX(tok, "ch_layout")) {
                        int val = stoi(strchr(tok, '=') + 1);
                        *NbRxRequired = val / 10;
                        *NbTxRequired = val % 10;
                } else {
                        MSG(ERROR, "Wrong config option '%s'!\n", tok);
                        usage(false);
                        return false;
                }
                tmp = NULL;
        }
        return true;
}

static int
vidcap_deltacast_init(struct vidcap_params *params, void **state)
{
#define HANDLE_ERROR vidcap_deltacast_done(s); return VIDCAP_INIT_FAIL;
	struct vidcap_deltacast_state *s = nullptr;
        ULONG             Result,DllVersion,NbBoards,ChnType;
        ULONG             BrdId = 0;
        ULONG             NbRxRequired = 0;
        ULONG             NbTxRequired = 0;

	printf("vidcap_deltacast_init\n");

        const char *fmt = vidcap_params_get_fmt(params);
        if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
                usage(strcmp(fmt, "fullhelp") == 0);
                return VIDCAP_INIT_NOERR;
        }

        s = (struct vidcap_deltacast_state *) calloc(1, sizeof(struct vidcap_deltacast_state));

	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
		return VIDCAP_INIT_FAIL;
	}

        gettimeofday(&s->t0, NULL);

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->frames = 0;
        s->grab_audio = FALSE;
        s->autodetect_format = TRUE;
        s->frame->color_spec = UYVY;
        s->audio_frame.data = NULL;

        s->BoardHandle = s->StreamHandle = s->SlotHandle = NULL;

        char *init_fmt = strdup(fmt);
        if (!parse_fmt(s, init_fmt, &BrdId, &NbRxRequired, &NbTxRequired)) {
                free(init_fmt);
                HANDLE_ERROR
        }
        free(init_fmt);

        printf("[DELTACAST] Selected device %" PRIu_ULONG "\n", BrdId);

        if(s->autodetect_format) {
                printf("DELTACAST] We will try to autodetect incoming video format.\n");
        }

        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08" PRIX_ULONG "\n",
                                Result);
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
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot open DELTA board %" PRIu_ULONG " handle. Result = 0x%08" PRIX_ULONG "\n",BrdId,Result);
                HANDLE_ERROR
        }

        if (NbRxRequired == 0 && NbTxRequired == 0) {
                assert(!s->quad_channel || s->channel == 0);
                NbRxRequired = s->quad_channel ? 4 : s->channel + 1;
        }
        if (!delta_set_nb_channels(BrdId, s->BoardHandle, NbRxRequired,
                                   NbTxRequired)) {
                HANDLE_ERROR
        }

        const auto Property = (VHD_CORE_BOARDPROPERTY) DELTA_CH_TO_VAL(
            s->channel, VHD_CORE_BP_RX0_TYPE, VHD_CORE_BP_RX4_TYPE);
        VHD_GetBoardProperty(s->BoardHandle, Property, &ChnType);
        if (!delta_chn_type_is_sdi(ChnType)) {
                MSG(ERROR, "ERROR : The selected channel is not a SDI one\n");
                HANDLE_ERROR
        }

        for (ULONG i = s->channel; i < s->channel + (s->quad_channel ? 4 : 1);
             i++) {
#ifdef VHD_MIN_6_21
                /*Channel mode Setup*/
                if ((ChnType == VHD_CHNTYPE_3GSDI_ASI) ||
                    (ChnType == VHD_CHNTYPE_12GSDI_ASI)) {
                        VHD_SetChannelProperty(s->BoardHandle, VHD_RX_CHANNEL, i,
                                               VHD_CORE_CP_MODE,
                                               VHD_CHANNEL_MODE_SDI);
                }
#endif
                /* Disable RX-TX loopback(s) */
                delta_set_loopback_state(s->BoardHandle, (int) i, FALSE);
        }

        s->initialize_flags = vidcap_params_get_flags(params);
        printf("\nWaiting for channel locked...\n");
        try {
                if(wait_for_channel(s)) {
                        s->initialized = true;
                }
        } catch(delta_init_exception &e) {
                HANDLE_ERROR
        }

        *state = s;
	return VIDCAP_INIT_OK;
#undef HANDLE_ERROR
}

static void
vidcap_deltacast_done(void *state)
{
	struct vidcap_deltacast_state *s = (struct vidcap_deltacast_state *) state;

	assert(s != NULL);

        if(s->SlotHandle)
                VHD_UnlockSlotHandle(s->SlotHandle);
        if(s->StreamHandle) {
                VHD_StopStream(s->StreamHandle);
                VHD_CloseStreamHandle(s->StreamHandle);
        }
        if(s->BoardHandle) {
                /* Re-establish RX-TX by-pass relay loopthrough */
                for (ULONG i = s->channel; i < s->channel + (s->quad_channel ? 4 : 1);
                     i++) {
                        delta_set_loopback_state(s->BoardHandle, i, TRUE);
                }
                VHD_CloseBoardHandle(s->BoardHandle);
        }
        
        vf_free(s->frame);
        free(s->audio_frame.data);
        free(s);
}

static struct video_frame *
vidcap_deltacast_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_deltacast_state   *s = (struct vidcap_deltacast_state *) state;
        
        ULONG             /*SlotsCount, SlotsDropped,*/BufferSize;
        ULONG             Result;
        BYTE             *pBuffer=NULL;

        if (!s->initialized) {
                try {
                        if(wait_for_channel(s)) {
                                s->initialized = true;
                        } else {
                                return NULL;
                        }
                } catch(delta_init_exception &e) {
                }
        }

        *audio = NULL;
        /* Unlock slot */
        if(s->SlotHandle)
                VHD_UnlockSlotHandle(s->SlotHandle);
        Result = VHD_LockSlotHandle(s->StreamHandle,&s->SlotHandle);
        if (Result != VHDERR_NOERROR) {
                if (Result != VHDERR_TIMEOUT) {
                        MSG(ERROR,
                            "ERROR : Cannot lock slot on RX%u stream. Result = "
                            "0x%08" PRIX_ULONG "\n",
                            s->channel, Result);
                }
                else {
                        log_msg(LOG_LEVEL_ERROR, "Timeout \n");
                }
                return NULL;
        }

        if(s->grab_audio) {
                /* Set the audio buffer size */
                s->AudioBufferSize = VHD_GetBlockSize(s->pAudioChn->BufferFormat, s->pAudioChn->Mode) *
                        VHD_GetNbSamples((VHD_VIDEOSTANDARD) s->VideoStandard, (VHD_CLOCKDIVISOR) s->ClockSystem, VHD_ASR_48000, 0);
                s->pAudioChn->DataSize = s->AudioBufferSize;

                /* Extract audio */
                Result = VHD_SlotExtractAudio(s->SlotHandle, &s->AudioInfo);
                if(Result==VHDERR_NOERROR) {
                        s->audio_frame.data_len = s->pAudioChn->DataSize;
                        /* Do audio processing here */
                        *audio = &s->audio_frame;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Audio grabbing error. Result = 0x%08" PRIX_ULONG "\n",Result);
                }
        }

        
         Result = VHD_GetSlotBuffer(s->SlotHandle, VHD_SDI_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "\nERROR : Cannot get slot buffer. Result = 0x%08" PRIX_ULONG "\n",Result);
                return NULL;
         }
         
         s->tile->data = (char*) pBuffer;
         s->tile->data_len = BufferSize;

         /* Print some statistics */
         /*VHD_GetStreamProperty(s->StreamHandle,VHD_CORE_SP_SLOTS_COUNT,&SlotsCount);
         VHD_GetStreamProperty(s->StreamHandle,VHD_CORE_SP_SLOTS_DROPPED,&SlotsDropped);
         printf("%u frames received (%u dropped)            \r",SlotsCount,SlotsDropped);*/
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            log_msg(LOG_LEVEL_INFO, "[DELTACAST cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }
        s->frames++;
        
	return s->frame;
}

static const struct video_capture_info vidcap_deltacast_info = {
        vidcap_deltacast_probe,
        vidcap_deltacast_init,
        vidcap_deltacast_done,
        vidcap_deltacast_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(deltacast, &vidcap_deltacast_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


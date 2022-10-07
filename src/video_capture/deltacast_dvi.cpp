/**
 * @file   video_capture/deltacast_dvi.cpp
 * @author Martin Piatka    <445597@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2021 CESNET, z. s. p. o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#ifndef WIN32
#include <sys/poll.h>
#include <sys/ioctl.h>
#endif
#include <sys/time.h>
#include <semaphore.h>

#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "deltacast_common.hpp"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture.h"

using namespace std;

#define DEFAULT_BUFFERQUEUE_DEPTH 5

struct vidcap_deltacast_dvi_state {
        ULONG             BoardType;
        HANDLE            BoardHandle, StreamHandle;

        struct       timeval t, t0;
        int          frames;

        codec_t      codec;
        bool         configured;
        struct video_desc desc;
        // SlotHandles need to be unlocked within the same thread
        mutex             lock;
        queue<HANDLE>     frames_to_free;
};

#define EEDDIDOK 0
#define BADEEDID 1

// compat
#if defined DELTA_DVI_DEPRECATED
#define VHD_DV_EEDID_DVIA VHD_DV_EEDID_DVIA_DEPRECATED
#define VHD_DV_MODE_DVI_A VHD_DV_MODE_DVI_A_DEPRECATED
#define VHD_DV_MODE_ANALOG_COMPONENT_VIDEO VHD_DV_MODE_ANALOG_COMPONENT_VIDEO_DEPRECATED
#define VHD_DV_SP_DUAL_LINK VHD_DV_SP_DUAL_LINK_DEPRECATED
#define VHD_DV_DVI_A_STANDARD VHD_DV_STANDARD
#define VHD_DV_DVIA_STD_DMT VHD_DV_STD_DMT
#define VHD_PresetDviAStreamProperties VHD_PresetTimingStreamProperties
#endif // defined DELTA_DVI_DEPRECATED


static void usage(void);
static decltype(EEDDIDOK) CheckEEDID(BYTE pEEDIDBuffer[256]);

const static map<VHD_DV_EEDID_PRESET, const char *> edid_presets = {
        { VHD_DV_EEDID_EMPTY,           "empty E-EDID - the host should force its output regardless of the DELTA-dvi E-EDID" },
        { VHD_DV_EEDID_DVIA,            "DVI-A E-EDID" },
        { VHD_DV_EEDID_DVID,            "DVI-D E-EDID" },
        { VHD_DV_EEDID_HDMI,            "HDMI E-EDID" },
        { VHD_DV_EEDID_DVID_DUAL,       "DVI-D E-EDID with dual-link formats" },
        { VHD_DV_EEDID_HDMI_H4K,        "HDMI H4K E-EDID" },
        { VHD_DV_EEDID_DVID_H4K,        "DVI-D H4K E-EDID" },
        { VHD_DV_EEDID_HDMI_H4K2,       "HDMI H4K2 E-EDID" },
        { VHD_DV_EEDID_DVID_H4K2,       "DVI-D H4K2 E-EDID" },
        { VHD_DV_EEDID_DISPLAYPORT_1_2, "DisplayPort 1.2 E-EDID" },
        { VHD_DV_EEDID_HDMI_FLEX_HMI,   "HDMI FLEX-HMI E-EDID" },
        { VHD_DV_EEDID_DVID_FLEX_HMI,   "DVI-D FLEX-HMI E-EDID" },
        { (VHD_DV_EEDID_PRESET) -1,     "avoid E-EDID loading" },
};

static void usage(void)
{
        col() << "Usage:\n";
        col() << SBOLD(SRED("\t-t deltacast-dv") << "[:device=<index>][:channel=<channel>][:codec=<color_spec>][:preset=<preset>|:format=<format>]") << "\n";
        col() << "where\n";
        
        col() << SBOLD("\t<index>") << " - index of DVI card\n";
        print_available_delta_boards();

        col() << SBOLD("\t<channel>") << " may be channel index (for cards which have multiple inputs, max 4)\n";
        
        col() << SBOLD("\t<preset>") << " may be one of following\n";
        for (const auto &it : edid_presets) {
                col() << SBOLD("\t\t " << setw(2) << it.first) << " - " << it.second << "\n";
        }

        col() << SBOLD("\t<color_spec>") << " may be one of following\n";
        col() << SBOLD("\t\tUYVY\n");
        col() << SBOLD("\t\tv210\n");
        col() << SBOLD("\t\tRGBA\n");
        col() << SBOLD("\t\tBGR") << " (default)\n";

        col() << SBOLD("\t<format>") << " may be format description (DVI-A), E-EDID will be ignored\n";
        col() << "\t\tvideo format is in the format " << SBOLD("<width>x<height>@<fps>") << "\n";

}

static decltype(EEDDIDOK) CheckEEDID(BYTE pEEDIDBuffer[256])
{
        int i;
        UBYTE sum1 = 0,sum2 = 0;
        decltype(EEDDIDOK) Return = EEDDIDOK;

        /* Verify checksum */
        for(i=0;i<128;i++)
        {
                sum1 += pEEDIDBuffer[i];
                sum2 += pEEDIDBuffer[i+128];
        }
        if(sum1 != 0 || sum2 != 0)
                Return = BADEEDID;

        /* Verify header */
        if(pEEDIDBuffer[0] != 0x00u || pEEDIDBuffer[1] != 0xFFu || pEEDIDBuffer[2] != 0xFFu || pEEDIDBuffer[3] != 0xFFu
                        || pEEDIDBuffer[4] != 0xFFu || pEEDIDBuffer[5] != 0xFFu || pEEDIDBuffer[6] != 0xFFu || pEEDIDBuffer[7] != 0x00u)
                Return = BADEEDID;

        return Return;
}

static struct vidcap_type *
vidcap_deltacast_dvi_probe(bool verbose, void (**deleter)(void *))
{
	struct vidcap_type*		vt;
        *deleter = free;
    
	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt == NULL) {
                return NULL;
        }

        vt->name        = "deltacast-dv";
        vt->description = "DELTACAST DVI/HDMI card";

        if (!verbose) {
                return vt;
        }

        ULONG Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                return vt;
        }

        vt->cards = (struct device_info *) calloc(NbBoards, sizeof(struct device_info));
        vt->card_count = NbBoards;
        for (ULONG i = 0; i < NbBoards; ++i) {
                string board{"Unknown board type"};
                ULONG BoardType;
                HANDLE BoardHandle = NULL;
                Result = VHD_OpenBoardHandle(i, &BoardHandle, NULL, 0);
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
                if (Result == VHDERR_NOERROR)
                {
                        auto it = board_type_map.find(BoardType);
                        if (it != board_type_map.end()) {
                                board = it->second;
                        }
                }
                VHD_CloseBoardHandle(BoardHandle);

                snprintf(vt->cards[i].dev, sizeof vt->cards[i].dev, ":device=%" PRIu_ULONG, i);
                snprintf(vt->cards[i].name, sizeof vt->cards[i].name, "DELTACAST %s #%" PRIu_ULONG,
                                board.c_str(), i);
        }
	return vt;
}

static bool wait_for_channel_locked(struct vidcap_deltacast_dvi_state *s, bool have_dvi_a_format,
        VHD_DV_MODE DviMode,
        ULONG Width, ULONG Height, ULONG RefreshRate)
{
        BOOL32 Interlaced_B = FALSE;
        ULONG             Result = VHDERR_NOERROR;

        struct timeval t0, t;

        gettimeofday(&t0, NULL);

        if(!have_dvi_a_format) {
                /* Wait for channel locked */
                printf("Waiting for incoming signal...\n");
                do
                {
                        Result = VHD_GetStreamProperty(s->StreamHandle, VHD_DV_SP_MODE, (ULONG *) &DviMode);
                        gettimeofday(&t, NULL);
                        if(tv_diff(t, t0) > 1.0) break;
                } while (Result != VHDERR_NOERROR);

                if(Result != VHDERR_NOERROR)
                        return false;
        }

        printf("\nIncoming Dvi mode detected: ");
        switch(DviMode)
        {
                case VHD_DV_MODE_DVI_D                   : printf("DVI-D\n");break;
                case VHD_DV_MODE_DVI_A                   : printf("DVI-A\n");break;
                case VHD_DV_MODE_ANALOG_COMPONENT_VIDEO  : printf("Analog component video\n");break;
                case VHD_DV_MODE_HDMI                    : printf("HDMI\n");break;
                default                                   : break;
        }

        /* Disable EDID auto load */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_DISABLE_EDID_AUTO_LOAD,TRUE);
        if(Result != VHDERR_NOERROR)
                return false;

        /* Set the DVI mode of this channel to the detected one */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_MODE, DviMode);
        if(Result != VHDERR_NOERROR)
                return false;

        if(DviMode == VHD_DV_MODE_DVI_A)
        {
                VHD_DV_DVI_A_STANDARD DviAStd = VHD_DV_DVIA_STD_DMT;
                if(!have_dvi_a_format) {
                        /* Auto-detection is now available for DVI-A.
                           VHD_DVI_SP_ACTIVE_HEIGHT, VHD_DVI_SP_INTERLACED, VHD_DVI_SP_REFRESH_RATE,
                           VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                           VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                           VHD_DVI_SP_V_FRONT_PORCH properties are required for DVI-A but
                           the VHD_PresetDviAStreamProperties is a helper function to set all these
                           properties according to a resolution, a refresh rate and a graphic timing
                           standard. Manual setting or overriding of these properties is allowed
                           Resolution, refresh rate and graphic timing standard can be auto-detect
                           with VHD_DetectDviAFormat function */
#if ! defined DELTA_DVI_DEPRECATED
                        Result = VHD_DetectDviAFormat(s->StreamHandle,&DviAStd,&Width,&Height,&RefreshRate,
                                        &Interlaced_B);
#else
                        Result = VHDERR_NOTIMPLEMENTED; // VHD_DetectDviAFormat was removed in v6.14
#endif
                }
                if(Result == VHDERR_NOERROR)
                {
                        printf("\nDVI-A format detected: %" PRIu_ULONG "x%" PRIu_ULONG " @%" PRIu_ULONG "Hz (%s)\n", Width, Height, RefreshRate, Interlaced_B ? "Interlaced" : "Progressive");
                        Result = VHD_PresetDviAStreamProperties(s->StreamHandle, DviAStd,Width,Height,
                                        RefreshRate,Interlaced_B);
                        if(Result != VHDERR_NOERROR) {
                                printf("ERROR : Cannot set incoming DVI-A format. Result = 0x%08" PRIX_ULONG "\n", Result);
                        }
                }
                else {
                        printf("ERROR : Cannot detect incoming DVI-A format. Result = 0x%08" PRIX_ULONG ". Use 'preset=' option.\n", Result);
                        return false;
                }
        }
        else if(DviMode == VHD_DV_MODE_DVI_D)
        {
                int Dual_B = FALSE;
                /* Get auto-detected resolution */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,&Width);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,&Height);
                else
                        printf("ERROR : Cannot detect incoming active width from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,(ULONG*)&Interlaced_B);
                else
                        printf("ERROR : Cannot detect incoming active height from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE,&RefreshRate);
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is "
                                        "interlaced or progressive. Result = 0x%08" PRIX_ULONG "\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_DUAL_LINK,(ULONG*)&Dual_B);
                else
                        printf("ERROR : Cannot detect incoming refresh rate from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);

                if(Result == VHDERR_NOERROR)
                        printf("\nIncoming graphic resolution : %" PRIu_ULONG "x%" PRIu_ULONG " @%" PRIu_ULONG "Hz (%s) %s link\n", Width, Height, RefreshRate, Interlaced_B ? "Interlaced" : "Progressive", Dual_B ? "Dual" : "Single");
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is dual or simple link. Result = 0x%08" PRIX_ULONG "\n", Result);

                if(Result != VHDERR_NOERROR) {
                        return false;
                }

                /* Configure stream. Only VHD_DVI_SP_ACTIVE_WIDTH, VHD_DVI_SP_ACTIVE_HEIGHT and
                   VHD_DVI_SP_INTERLACED properties are required for DVI-D.
                   VHD_DVI_SP_REFRESH_RATE,VHD_DVI_SP_DUAL_LINK are optional
                   VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                   VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                   VHD_DVI_SP_V_FRONT_PORCH properties are not applicable for DVI-D */

                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,Width);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,Height);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,Interlaced_B);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE,RefreshRate);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_DUAL_LINK,Dual_B);
        }
        else if(DviMode == VHD_DV_MODE_HDMI || DviMode == VHD_DV_MODE_ANALOG_COMPONENT_VIDEO)
        {
                VHD_DV_CS       InputCS;
                ULONG             PxlClk = 0;
                /* Get auto-detected resolution */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,&Width);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,&Height);
                else
                        printf("ERROR : Cannot detect incoming active width from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,(ULONG*)&Interlaced_B);
                else
                        printf("ERROR : Cannot detect incoming active height from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE,&RefreshRate);
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is "
                                        "interlaced or progressive. Result = 0x%08" PRIX_ULONG "\n", Result);

                if (Result == VHDERR_NOERROR) {
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE,&RefreshRate);
                        if(s->BoardType == VHD_BOARDTYPE_HDMI)
                                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_INPUT_CS,(ULONG*)&InputCS);
                        else
                                printf("ERROR : Cannot detect incoming color space from RX0. Result = 0x%08" PRIX_ULONG " (%s)\n", Result,
                                                delta_get_error_description(Result));
                }

                if (Result == VHDERR_NOERROR) {
                        if (s->BoardType == VHD_BOARDTYPE_HDMI)
                                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_PIXEL_CLOCK,&PxlClk);
                        else
                                printf("ERROR : Cannot detect incoming pixel clock from RX0. Result = 0x%08" PRIX_ULONG " (%s)\n", Result,
                                                delta_get_error_description(Result));
                }

                if(Result == VHDERR_NOERROR)
                        printf("\nIncoming graphic resolution : %" PRIu_ULONG "x%" PRIu_ULONG " @%" PRIu_ULONG "Hz (%s)\n", Width, Height, RefreshRate, Interlaced_B ? "Interlaced" : "Progressive");
                else
                        printf("ERROR : Cannot detect incoming refresh rate from RX0. "
                                        "Result = 0x%08" PRIX_ULONG "\n", Result);

                if(Result != VHDERR_NOERROR) {
                        return false;
                }

                /* Configure stream. Only VHD_DVI_SP_ACTIVE_WIDTH, VHD_DVI_SP_ACTIVE_HEIGHT and
                   VHD_DVI_SP_INTERLACED properties are required for HDMI and Component
                   VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                   VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                   VHD_DVI_SP_V_FRONT_PORCH properties are not applicable for DVI-D, HDMI and Component */

                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,Width);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,Height);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,Interlaced_B);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE, RefreshRate);
                if (s->BoardType == VHD_BOARDTYPE_HDMI) {
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_INPUT_CS, InputCS);
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_PIXEL_CLOCK, PxlClk);
                }
        }

        Result = VHD_StartStream(s->StreamHandle);

        if(Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "Cannot start stream!\n");
                return false;
        }

        s->desc.color_spec = s->codec;
        s->desc.width = Width;
        s->desc.height = Height;
        s->desc.fps = RefreshRate;
        s->desc.interlacing = Interlaced_B ? LOWER_FIELD_FIRST : PROGRESSIVE;
        s->desc.tile_count = 1;

        return true;
}

static const unordered_map<codec_t, ULONG, hash<int>> ug_delta_codec_mapping = {
        { BGR, VHD_BUFPACK_VIDEO_RGB_24 },
        { RGBA, VHD_BUFPACK_VIDEO_RGBA_32 },
        { UYVY, VHD_BUFPACK_VIDEO_YUV422_8 },
        { v210, VHD_BUFPACK_VIDEO_YUV422_10 },
};

static int
vidcap_deltacast_dvi_init(struct vidcap_params *params, void **state)
{
	struct vidcap_deltacast_dvi_state *s;
        ULONG Width = 0, Height = 0, RefreshRate = 0;
        ULONG             Result = VHDERR_NOERROR,DllVersion,NbBoards;
        ULONG             BrdId = 0;
        ULONG             Packing;
        int               edid_preset = -1;
        BYTE              pEEDIDBuffer[256];
        ULONG             pEEDIDBufferSize=256;
        int               channel = 0;
        ULONG             ChannelId;
        bool              have_dvi_a_format = false;
        VHD_DV_MODE       DviMode = NB_VHD_DV_MODES;

	printf("vidcap_deltacast_dvi_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        char *init_fmt = NULL,
             *tmp = NULL;
        if (vidcap_params_get_fmt(params) != NULL)
                tmp = init_fmt = strdup(vidcap_params_get_fmt(params));
        if(init_fmt && strcmp(init_fmt, "help") == 0) {
                free(tmp);
                usage();
                return VIDCAP_INIT_NOERR;
        }

        s = new vidcap_deltacast_dvi_state();
	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
                goto error;
	}

        s->codec = BGR;
        s->configured = false;
        s->BoardHandle = s->StreamHandle = NULL;

        if(init_fmt)
        {
                char *save_ptr = NULL;
                char *tok;
                
                while((tok = strtok_r(init_fmt, ":", &save_ptr)) != NULL) {
                        if (strncasecmp(tok, "device=", strlen("device=")) == 0) {
                                BrdId = atoi(tok + strlen("device="));
                        } else if(strncasecmp(tok, "board=", strlen("board=")) == 0) {
                                // compat, should be device= instead
                                BrdId = atoi(tok + strlen("board="));
                        } else if(strncasecmp(tok, "codec=", strlen("codec=")) == 0) {
                                char *codec_str = tok + strlen("codec=");

                                s->codec = get_codec_from_name(codec_str);
                                if (s->codec == VIDEO_CODEC_NONE) {
                                        log_msg(LOG_LEVEL_ERROR, "Unable to find codec: %s\n",
                                                        codec_str);
                                        goto error;
                                }
                        } else if(strncasecmp(tok, "preset=", strlen("preset=")) == 0) {
                                        edid_preset = atoi(strchr(tok, '=') + 1);
                                        if (edid_preset < 0 || edid_preset >= NB_VHD_DV_EEDID_PRESET) {
                                                log_msg(LOG_LEVEL_ERROR, "[DELTA] Error: Wrong "
                                                                "EDID entered on commandline. "
                                                                "Expected 0-%d, got %d.\n",
                                                                (int) NB_VHD_DV_EEDID_PRESET - 1, edid_preset);
                                                goto error;

                                        }
                        } else if(strncasecmp(tok, "channel=", strlen("channel=")) == 0) {
                                channel = atoi(tok + strlen("channel="));
                        } else if(strncasecmp(tok, "format=", strlen("format=")) == 0) {
                                have_dvi_a_format = true;
                                char *ptr = strchr(tok, '=') + 1;
                                char *save_ptr, *item;
                                if((item = strtok_r(ptr, "x@", &save_ptr))) {
                                        Width = atoi(item);
                                }
                                if((item = strtok_r(NULL, "x@", &save_ptr))) {
                                        Height = atoi(item);
                                }
                                if((item = strtok_r(NULL, "x@", &save_ptr))) {
                                        RefreshRate = atof(item);
                                }
                        } else {
                                log_msg(LOG_LEVEL_ERROR, "[DELTA] Error: Unrecongnized "
                                                "trailing parameter %s\n", tok);
                                goto error;
                        }
                        init_fmt = NULL;
                }
        } else {
                BrdId = 0;
                printf("[DELTACAST] Automatically choosen device nr. 0\n");
        }
        free(tmp);
        tmp = NULL;

        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08" PRIX_ULONG "\n",
                                Result);
                goto error;
        }
        if (NbBoards == 0) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] No DELTA board detected, exiting...\n");
                goto error;
        }
        
        if(BrdId >= NbBoards) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Wrong index %" PRIu_ULONG ". Found %" PRIu_ULONG " cards.\n", BrdId, NbBoards);
                goto error;
        }

        /* Open a handle on first DELTA-hd/sdi/codec board */
        Result = VHD_OpenBoardHandle(BrdId,&s->BoardHandle,NULL,0);
        if (Result != VHDERR_NOERROR)
        {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : Cannot open DELTA board %" PRIu_ULONG " handle. Result = 0x%08" PRIX_ULONG "\n", BrdId, Result);
                goto error;
        }
        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_BOARD_TYPE, &s->BoardType);
        if (s->BoardType != VHD_BOARDTYPE_DVI && s->BoardType != VHD_BOARDTYPE_HDMI) {
                log_msg(LOG_LEVEL_ERROR, "[DELTACAST] ERROR : The selected board is not an DVI or HDMI one\n");
                goto bad_channel;
        }
        
        switch(channel) {
                case 0:
                        ChannelId = VHD_ST_RX0;
                        break;
                case 1:
                        ChannelId = VHD_ST_RX1;
                        break;
                case 2:
                        ChannelId = VHD_ST_RX2;
                        break;
                case 3:
                        ChannelId = VHD_ST_RX3;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "[DELTACAST] Bad channel index!\n");
                        goto no_stream;
        }
        Result = VHD_OpenStreamHandle(s->BoardHandle, ChannelId,
                        s->BoardType == VHD_BOARDTYPE_HDMI ? VHD_DV_STPROC_JOINED : VHD_DV_STPROC_DEFAULT,
                        NULL, &s->StreamHandle, NULL);
        if (Result != VHDERR_NOERROR)
        {
                log_msg(LOG_LEVEL_ERROR, "ERROR : Cannot open RX0 stream on DELTA-DVI board handle. "
                                "Result = 0x%08" PRIX_ULONG "\n", Result);
                goto no_stream;
        }

        /* Configure color space reception (RGBA for no color-space conversion) */
        if (ug_delta_codec_mapping.find(s->codec) != ug_delta_codec_mapping.end()) {
                Packing = ug_delta_codec_mapping.at(s->codec);
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unknown pixel formate entered.\n");
                goto no_format;
        }

        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFER_PACKING,
                        Packing);
        if(Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "Unable to set packing format.\n");
                goto no_format;
        }
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_TRANSFER_SCHEME,
                        VHD_TRANSFER_SLAVED);

        Result = VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFERQUEUE_DEPTH,
                        DEFAULT_BUFFERQUEUE_DEPTH);
        if(Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "[Delta-dvi] Warning: Unable to set buffer queue length.\n");
        }

        if(have_dvi_a_format) {
                DviMode = VHD_DV_MODE_DVI_A;
        } else {
                if (edid_preset >= 0 && edid_preset < NB_VHD_DV_EEDID_PRESET) {
                        VHD_PresetEEDID((VHD_DV_EEDID_PRESET)edid_preset,pEEDIDBuffer,256);
                        VHD_LoadEEDID(s->StreamHandle,pEEDIDBuffer,256);
                }
                /* Read EEDID and check its validity */
                Result = VHD_ReadEEDID(s->BoardHandle,VHD_ST_RX0,pEEDIDBuffer,&pEEDIDBufferSize);
                if(Result == VHDERR_NOTIMPLEMENTED || CheckEEDID(pEEDIDBuffer) == BADEEDID)
                {
                        /* Propose edid preset to user and load */
                        log_msg(LOG_LEVEL_ERROR, "\nNo valid EEDID detected or DELTA-dvi board V1.\n");
                        log_msg(LOG_LEVEL_ERROR, "Please set it as a command-line option.\n");
                        goto no_format;
                }
        }

        s->configured = wait_for_channel_locked(s, have_dvi_a_format, DviMode, Width, Height, RefreshRate);
        if(!s->configured &&
                        have_dvi_a_format) {
                log_msg(LOG_LEVEL_ERROR, "Unable to set preset format!\n");
                goto no_format;
        }

        gettimeofday(&s->t0, NULL);
        s->frames = 0;
        
        *state = s;
	return VIDCAP_INIT_OK;

no_format:
        /* Close stream handle */
        VHD_CloseStreamHandle(s->StreamHandle);

no_stream:
        
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
bad_channel:
        VHD_CloseBoardHandle(s->BoardHandle);
error:
        free(tmp);
        delete s;
        return VIDCAP_INIT_FAIL;
}

static void
vidcap_deltacast_dvi_done(void *state)
{
	struct vidcap_deltacast_dvi_state *s = (struct vidcap_deltacast_dvi_state *) state;

	assert(s != NULL);

        while (!s->frames_to_free.empty()) {
                HANDLE h = s->frames_to_free.front();
                s->frames_to_free.pop();
                VHD_UnlockSlotHandle(h);
        }
        
        VHD_StopStream(s->StreamHandle);
        VHD_CloseStreamHandle(s->StreamHandle);
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
        VHD_CloseBoardHandle(s->BoardHandle);
        
        delete s;
}

struct vidcap_deltacast_dispose_udata {
        struct vidcap_deltacast_dvi_state *s;
        HANDLE SlotHandle;
};

static void vidcap_deltacast_dvi_dispose(struct video_frame *f)
{
        auto data = (struct vidcap_deltacast_dispose_udata *) f->callbacks.dispose_udata;
        data->s->lock.lock();
        data->s->frames_to_free.push(data->SlotHandle);
        data->s->lock.unlock();
        delete data;
        vf_free(f);
}

static struct video_frame *
vidcap_deltacast_dvi_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_deltacast_dvi_state   *s = (struct vidcap_deltacast_dvi_state *) state;
        
        ULONG             /*SlotsCount, SlotsDropped,*/BufferSize;
        ULONG             Result;
        BYTE             *pBuffer=NULL;
        HANDLE            SlotHandle;
        queue<HANDLE>     queue;

        s->lock.lock();
        swap(queue, s->frames_to_free); // empty the synchronized queue and unlock the slots without a lock
        s->lock.unlock();
        while (!queue.empty()) {
                HANDLE h = queue.front();
                queue.pop();
                VHD_UnlockSlotHandle(h);
        }

        if(!s->configured) {
                s->configured = wait_for_channel_locked(s, false, NB_VHD_DV_MODES, 0, 0, 0);
        }
        if(!s->configured) {
                return NULL;
        }

        *audio = NULL;

        Result = VHD_LockSlotHandle(s->StreamHandle, &SlotHandle);
        if (Result != VHDERR_NOERROR) {
                if (Result != VHDERR_TIMEOUT) {
                        log_msg(LOG_LEVEL_ERROR, "ERROR : Cannot lock slot on RX0 stream. Result = 0x%08" PRIX_ULONG "\n", Result);
                }
                else {
                        log_msg(LOG_LEVEL_WARNING, "Timeout \n");
                }
                return NULL;
        }

         Result = VHD_GetSlotBuffer(SlotHandle, VHD_DV_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                log_msg(LOG_LEVEL_ERROR, "\nERROR : Cannot get slot buffer. Result = 0x%08" PRIX_ULONG "\n",Result);
                return NULL;
         }

         struct video_frame *out;
         if (s->desc.color_spec == RGBA) { // DELTACAST uses BGRA
                 out = vf_alloc_desc_data(s->desc);
                 vc_copylineRGBA(reinterpret_cast<unsigned char *>(out->tiles[0].data), pBuffer, out->tiles[0].data_len, 16, 8, 0);
                 VHD_UnlockSlotHandle(SlotHandle);
                 out->callbacks.dispose = vf_free;
         } else {
                 out = vf_alloc_desc(s->desc);
                 out->tiles[0].data = (char*) pBuffer;
                 out->tiles[0].data_len = BufferSize;
                 out->callbacks.dispose_udata = new vidcap_deltacast_dispose_udata{s, SlotHandle};
                 out->callbacks.dispose = vidcap_deltacast_dvi_dispose;
         }

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
        
	return out;
}

static const struct video_capture_info vidcap_deltacast_dvi_info = {
        vidcap_deltacast_dvi_probe,
        vidcap_deltacast_dvi_init,
        vidcap_deltacast_dvi_done,
        vidcap_deltacast_dvi_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(deltacast-dv, &vidcap_deltacast_dvi_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


/**
 * @file   video_capture/deltacast_dvi.cpp
 * @author Martin Piatka    <445597@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Code is written by DELTACAST's VideoMaster SDK example Sample_RX_DVI
 * (later SAMPLE_RX_DVI_D). Consulted also VHD 6.32 Sample_RX_DisplayPort
 * sample.
 *
 * @sa deltacast_common.hpp for common DELTACAST information
 */
/*
 * Copyright (c) 2013-2025 CESNET, zájmové sdružení právnických osob
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


#include <cassert>                 // for assert
#include <cstdio>                  // for printf, NULL, perror, fclose, ferror
#include <cstdlib>                 // for atoi, free, atof, calloc
#include <cstring>                 // for strlen, strtok_r, strchr, strcmp
#include <functional>              // for hash
#include <iomanip>                 // for _Setw, setw, operator<<
#include <map>
#include <mutex>
#include <ostream>                 // for operator<<
#include <queue>
#include <unordered_map>
#include <utility>                 // for pair
// IWYU pragma: no_include <sys/time.h> # via tv.h
// IWYU pragma: no_include <strings.h> # via compact/strings.h

#include "compat/strings.h"        // IWYU pragma: keep
#include "debug.h"
#include "deltacast_common.hpp"
#include "lib_common.h"
#include "pixfmt_conv.h"           // for vc_copylineRGBA
#include "tv.h"
#include "types.h"                 // for video_desc, device_info, tile, vid...
#include "utils/color_out.h"
#include "utils/macros.h"          // for snprintf_ch
#include "video_capture_params.h"  // for vidcap_params_get_flags, vidcap_pa...
#include "video_codec.h"           // for get_codec_from_name
#include "video_frame.h"           // for vf_free, vf_alloc_desc, vf_alloc_d...
#include "video_capture.h"

#define MOD_NAME "[vcap/delta_dv] "

using namespace std;

#define DEFAULT_BUFFERQUEUE_DEPTH 5

struct vidcap_deltacast_dvi_state {
        ULONG             BoardType;
        HANDLE            BoardHandle, StreamHandle;
        ULONG             SlotsDroppedLast; ///< for statistics

        struct       timeval t0;

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
#if !defined VHD_MIN_6_13
#define VHD_DV_EEDID_PRESET VHD_DVI_EEDID_PRESET
#define VHD_DV_EEDID_EMPTY VHD_EEDID_EMPTY
#define VHD_DV_EEDID_DVIA VHD_EEDID_DVIA
#define VHD_DV_EEDID_DVID VHD_EEDID_DVID
#define VHD_DV_EEDID_HDMI VHD_EEDID_HDMI
#define VHD_DV_EEDID_DVID_DUAL VHD_EEDID_DVID_DUAL
#define VHD_DV_EEDID_HDMI_H4K VHD_EEDID_HDMI_H4K
#define VHD_DV_EEDID_DVID_H4K VHD_EEDID_DVID_H4K
#define VHD_DV_MODE VHD_DVI_MODE
#define VHD_DV_MODE_ANALOG_COMPONENT_VIDEO VHD_DVI_MODE_ANALOG_COMPONENT_VIDEO
#define VHD_DV_MODE_DVI_D VHD_DVI_MODE_DVI_D
#define VHD_DV_MODE_DVI_A VHD_DVI_MODE_DVI_A
#define VHD_DV_MODE_HDMI VHD_DVI_MODE_HDMI
#define VHD_DV_MODE_DISPLAYPORT NB_VHD_DV_MODES
#define VHD_DV_SP_MODE VHD_DVI_SP_MODE
#define VHD_DV_SP_DISABLE_EDID_AUTO_LOAD VHD_DVI_SP_DISABLE_EDID_AUTO_LOAD
#define VHD_DV_DVI_A_STANDARD VHD_DVI_A_STANDARD
#define VHD_DV_DVIA_STD_DMT VHD_DVIA_STD_DMT
#define VHD_DV_SP_ACTIVE_WIDTH VHD_DVI_SP_ACTIVE_WIDTH
#define VHD_DV_SP_ACTIVE_HEIGHT VHD_DVI_SP_ACTIVE_HEIGHT
#define VHD_DV_SP_INTERLACED VHD_DVI_SP_INTERLACED
#define VHD_DV_SP_REFRESH_RATE VHD_DVI_SP_REFRESH_RATE
#define VHD_DV_SP_DUAL_LINK VHD_DVI_SP_DUAL_LINK
#define VHD_DV_SP_INPUT_CS VHD_DVI_SP_INPUT_CS
#define VHD_DV_SP_PIXEL_CLOCK VHD_DVI_SP_PIXEL_CLOCK
#define VHD_DV_CS VHD_HDMI_CS
#define VHD_DV_STPROC_DEFAULT VHD_DVI_STPROC_DEFAULT
#define VHD_DV_STPROC_JOINED VHD_DVI_STPROC_DISJOINED_VIDEO
#define VHD_DV_BT_VIDEO VHD_DVI_BT_VIDEO
#define NB_VHD_DV_EEDID_PRESET NB_VHD_EEDID
#define NB_VHD_DV_MODES NB_VHD_DVI_MODES
#define NB_VHD_DV_STREAMPROPERTIES NB_VHD_DVI_STREAMPROPERTIES
#endif

#if !defined VHD_MIN_6_19
#define VHD_DV_SAMPLING ULONG
#endif

#if !defined VHD_MIN_6_30
#define VHD_DV_SP_CABLE_BIT_SAMPLING NB_VHD_DV_STREAMPROPERTIES
#endif

#if !defined HAVE_VHD_STRING
#define VHD_DV_SAMPLING_ToString(x) "UNKNOWN"
#define VHD_DV_CS_ToString(x) "UNKNOWN"
#endif

#if defined VHD_MIN_6_14
#define VHD_DV_EEDID_DVIA VHD_DV_EEDID_DVIA_DEPRECATED
#define VHD_DV_MODE_DVI_A VHD_DV_MODE_DVI_A_DEPRECATED
#define VHD_DV_MODE_ANALOG_COMPONENT_VIDEO VHD_DV_MODE_ANALOG_COMPONENT_VIDEO_DEPRECATED
#define VHD_DV_SP_DUAL_LINK VHD_DV_SP_DUAL_LINK_DEPRECATED
#define VHD_DV_DVI_A_STANDARD VHD_DV_STANDARD
#define VHD_DV_DVIA_STD_DMT VHD_DV_STD_DMT
#define VHD_PresetDviAStreamProperties VHD_PresetTimingStreamProperties
#endif // defined DELTA_DVI_DEPRECATED


static decltype(EEDDIDOK) CheckEEDID(BYTE pEEDIDBuffer[256]);

static const char *
get_edid_preset_name(VHD_DV_EEDID_PRESET preset)
{
        switch (preset) {
        case VHD_DV_EEDID_EMPTY:          return  "empty E-EDID - the host should force its output regardless of the DELTA-dvi E-EDID";
        case VHD_DV_EEDID_DVIA:           return  "DVI-A E-EDID";
        case VHD_DV_EEDID_DVID:           return  "DVI-D E-EDID";
        case VHD_DV_EEDID_HDMI:           return  "HDMI E-EDID";
        case VHD_DV_EEDID_DVID_DUAL:      return  "DVI-D E-EDID with dual-link formats";
        case VHD_DV_EEDID_HDMI_H4K:       return  "HDMI H4K E-EDID";
        case VHD_DV_EEDID_DVID_H4K:       return  "DVI-D H4K E-EDID";
#if defined VHD_MIN_6_13
       case  VHD_DV_EEDID_HDMI_H4K2:       return "HDMI H4K2 E-EDID";
       case  VHD_DV_EEDID_DVID_H4K2:       return "DVI-D H4K2 E-EDID";
       case  VHD_DV_EEDID_DISPLAYPORT_1_2: return "DisplayPort 1.2 E-EDID";
       case  VHD_DV_EEDID_HDMI_FLEX_HMI:   return "HDMI FLEX-HMI E-EDID";
       case  VHD_DV_EEDID_DVID_FLEX_HMI:   return "DVI-D FLEX-HMI E-EDID";
#endif
#if defined VHD_MIN_6_30
       case VHD_DV_EEDID_HDMI_12GxC_HMI:      return "HDMI 12G-xC-hmi E-EDID";
       case VHD_DV_EEDID_DVID_12GxC_HMI:      return "DVI-D 12G-xC-hmi E-EDID";
       case VHD_DV_EEDID_HDMI_DELTA_HMI:      return "HDMI DELTA-hmi E-EDID";
       case VHD_DV_EEDID_HDMI_DELTA_HMI_FRL3: return "HDMI DELTA-hmi E-EDID FRL1/2/3 Support";
       case VHD_DV_EEDID_HDMI_DELTA_HMI_FRL5: return "HDMI DELTA-hmi E-EDID FRL1/2/3/4/5 Support";
#endif
       case NB_VHD_DV_EEDID_PRESET: abort();
       }
       return nullptr; // can occur if new presets add to SDK
}

static void
usage(bool full)
{
        col() << "Usage:\n";
        col() << SBOLD(SRED("\t-t deltacast-dv") << "[:device=<index>][:channel=<channel>][:codec=<color_spec>][:preset=<preset>|:format=<format>]") << "\n";
        col() << SBOLD("\t-t deltacast-dv:[full]help") << "\n";
        col() << "where\n";
        
        col() << SBOLD("\t<index>") << " - index of DVI card\n";

        col() << SBOLD("\t<channel>")
              << " may be channel index (for cards which have multiple inputs, 0-"
              << MAX_DELTA_CH << ")\n";

        col() << SBOLD("\t<preset>") << " may be one of following\n";
        for (unsigned i = 0; i < NB_VHD_DV_EEDID_PRESET; ++i) {
                const char *preset_name =
                    get_edid_preset_name((VHD_DV_EEDID_PRESET) i);
                if (preset_name == nullptr) {
                        continue;
                }
                col() << "\t\t " << setw(2) << SBOLD(i) << " - " << preset_name
                      << "\n";
        }

        col() << SBOLD("\t<color_spec>") << " may be one of following\n";
        col() << SBOLD("\t\tUYVY\n");
        col() << SBOLD("\t\tv210\n");
        col() << SBOLD("\t\tRGBA\n");
        col() << SBOLD("\t\tBGR") << " (default)\n";

        col() << SBOLD("\t<format>") << " may be format description (DVI-A), E-EDID will be ignored\n";
        col() << "\t\tvideo format is in the format " << SBOLD("<width>x<height>@<fps>") << "\n";

        print_available_delta_boards(full);
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

static void vidcap_deltacast_dvi_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 0;
        *available_cards = nullptr;
    
        ULONG Result,DllVersion,NbBoards;
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                return;
        }

        *available_cards = (struct device_info *) calloc(NbBoards, sizeof(struct device_info));
        for (ULONG i = 0; i < NbBoards; ++i) {
                if (!delta_board_type_is_dv(delta_get_board_type(i), true)) {
                        continue; // skip SDI-only boards
                }
                auto &card = (*available_cards)[*count];
                snprintf(card.dev, sizeof card.dev, ":device=%" PRIu_ULONG, i);
                snprintf_ch(card.name, "DELTACAST %s", delta_get_model_name(i));
                *count += 1;
        }
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
#ifdef HAVE_VHD_STRING
        puts(VHD_DV_MODE_ToPrettyString(DviMode));
#else
        switch(DviMode)
        {
                case VHD_DV_MODE_DVI_D                   : printf("DVI-D\n");break;
                case VHD_DV_MODE_DVI_A                   : printf("DVI-A\n");break;
                case VHD_DV_MODE_ANALOG_COMPONENT_VIDEO  : printf("Analog component video\n");break;
                case VHD_DV_MODE_HDMI                    : printf("HDMI\n");break;
                default                                   : break;
        }
#endif

        /* Disable EDID auto load */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_DISABLE_EDID_AUTO_LOAD,TRUE);
        if(Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot disable EDID auto load.");
                return false;
        }

        /* Set the DVI mode of this channel to the detected one */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_MODE, DviMode);
        if(Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot configure RX0 stream mode.")
                return false;
        }

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
#if ! defined VHD_MIN_6_14
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
                                DELTA_PRINT_ERROR(Result, "ERROR : Cannot set incoming DVI-A format.");
                        }
                }
                else {
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming DVI-A format.");
                        return false;
                }
        }
        else
        {
                int Dual_B = FALSE;
                VHD_DV_CS       InputCS;
                ULONG             PxlClk = 0;
                VHD_DV_SAMPLING CableBitSampling;

                /* Get auto-detected resolution */
                if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,&Width)) != VHDERR_NOERROR) {
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming active width from RX0.")
                        return false;
                }
                if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,&Height)) != VHDERR_NOERROR) {
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming active height from RX0.");
                        return false;
                }
                if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,(ULONG*)&Interlaced_B)) != VHDERR_NOERROR) {
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect if incoming stream from RX0 is "
                                        "interlaced or progressive.");
                        return false;
                }
                if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE,&RefreshRate)) != VHDERR_NOERROR) {
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming refresh rate from RX0.");
                        return false;
                }

                if (DviMode == VHD_DV_MODE_HDMI || DviMode == VHD_DV_MODE_DISPLAYPORT) {
                        if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_INPUT_CS,(ULONG*)&InputCS)) != VHDERR_NOERROR) {
                                DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming color space from RX0.");
                                return false;
                        }
                        if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_PIXEL_CLOCK,&PxlClk)) != VHDERR_NOERROR) {
                                DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming pixel clock from RX0.");
                                return false;
                        }
                }
                if (DviMode == VHD_DV_MODE_DVI_D) {
                        if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_DUAL_LINK,(ULONG*)&Dual_B)) != VHDERR_NOERROR) {
                                DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect if incoming stream from RX0 is dual or simple link.");
                                return false;
                        }
                }

                if (DviMode == VHD_DV_MODE_DISPLAYPORT) {
                        if ((Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DV_SP_CABLE_BIT_SAMPLING,(ULONG*)&CableBitSampling)) != VHDERR_NOERROR) {
                                DELTA_PRINT_ERROR(Result, "ERROR : Cannot detect incoming cable bit sampling from RX0." );
                                return false;
                        }
                }

                printf("Incoming graphic resolution : %ux%u (%s)\n", Width, Height, Interlaced_B ? "Interlaced" : "Progressive");
                printf("Refresh rate : %u\n", RefreshRate);

                /* Configure stream. Only VHD_DVI_SP_ACTIVE_WIDTH, VHD_DVI_SP_ACTIVE_HEIGHT and
                   VHD_DVI_SP_INTERLACED properties are required for HDMI and Component
                   VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                   VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                   VHD_DVI_SP_V_FRONT_PORCH properties are not applicable for DVI-D, HDMI and Component */

                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_WIDTH,Width);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_ACTIVE_HEIGHT,Height);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_INTERLACED,Interlaced_B);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_REFRESH_RATE, RefreshRate);
                if (DviMode == VHD_DV_MODE_DVI_D) {
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_DUAL_LINK,Dual_B);
                        printf("%s link\n", Dual_B ? "Dual" : "Single");
                }
                if (DviMode == VHD_DV_MODE_HDMI || DviMode == VHD_DV_MODE_DISPLAYPORT) {
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_INPUT_CS, InputCS);
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DV_SP_PIXEL_CLOCK, PxlClk);
                        printf("Input CS : %s\n", VHD_DV_CS_ToString(InputCS));
                        printf("Pixel clock : %u\n", PxlClk);
                }
                if (DviMode == VHD_DV_MODE_DISPLAYPORT) {
                     VHD_SetStreamProperty(s->StreamHandle, VHD_DV_SP_CABLE_BIT_SAMPLING, CableBitSampling);
                     printf("Cable bit sampling: %s\n", VHD_DV_SAMPLING_ToString(CableBitSampling));
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

static bool load_custom_edid(const char *filename, BYTE *pEEDIDBuffer, ULONG *pEEDIDBufferSize) {
        FILE *edid = fopen(filename, "rb");
        if (!edid) {
                perror("EDID open");
                return false;
        }
        *pEEDIDBufferSize = fread((void *) pEEDIDBuffer, 1, *pEEDIDBufferSize, edid);
        bool ret = ferror(edid);
        if (!ret) {
                perror("EDID fread");
        }
        fclose(edid);
        return ret;
}

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
        bool              have_custom_edid = false;
        bool              have_dvi_a_format = false;
        VHD_DV_MODE       DviMode = NB_VHD_DV_MODES;

	printf("vidcap_deltacast_dvi_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

        const char *cfg = vidcap_params_get_fmt(params);
        if (strcmp(cfg, "help") == 0 ||
            strcmp(cfg, "fullhelp") == 0) {
                usage(strcmp(cfg, "fullhelp") == 0);
                return VIDCAP_INIT_NOERR;
        }

        s = new vidcap_deltacast_dvi_state();
	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
                return VIDCAP_INIT_FAIL;
	}

        s->codec = BGR;
        s->configured = false;
        s->BoardHandle = s->StreamHandle = NULL;

        char *init_fmt = strdup(cfg);
        char *tmp = init_fmt;

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
                        } else if (strncasecmp(tok, "edid=", strlen("edid=")) == 0) {
                                if (!load_custom_edid(strchr(tok, '=') + 1, pEEDIDBuffer, &pEEDIDBufferSize)) {
                                        goto error;
                                }
                                have_custom_edid = true;
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
                printf("[DELTACAST] Automatically chosen device nr. 0\n");
        }
        free(tmp);
        tmp = NULL;

        /* Query VideoMasterHD information */
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot query VideoMasterHD information.");
                goto error;
        }
        if (NbBoards == 0) {
                MSG(ERROR, "No DELTA board detected, exiting...\n");
                goto error;
        }
        
        if(BrdId >= NbBoards) {
                MSG(ERROR, "Wrong index %" PRIu_ULONG ". Found %" PRIu_ULONG " cards.\n", BrdId, NbBoards);
                goto error;
        }

        /* Open a handle on first DELTA-hd/sdi/codec board */
        Result = VHD_OpenBoardHandle(BrdId,&s->BoardHandle,NULL,0);
        if (Result != VHDERR_NOERROR)
        {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot open DELTA board %" PRIu_ULONG " handle.", BrdId);
                goto error;
        }
        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_BOARD_TYPE, &s->BoardType);
        if (not delta_board_type_is_dv((VHD_BOARDTYPE) s->BoardType, true)) {
                MSG(ERROR,
                    "ERROR : The selected board is not a DVI, HDMI or DP "
                    "(flex) one, have: %s\n",
                    delta_get_board_type_name(s->BoardType));
                goto bad_channel;
        }

        ChannelId = delta_rx_ch_to_stream_t(channel);
        if (ChannelId == NB_VHD_STREAMTYPES) {
                goto no_stream;
        }
        Result = VHD_OpenStreamHandle(s->BoardHandle, ChannelId,
                                      s->BoardType == VHD_BOARDTYPE_DVI
                                          ? VHD_DV_STPROC_DEFAULT
                                          : VHD_DV_STPROC_JOINED,
                                      nullptr, &s->StreamHandle, nullptr);
        if (Result != VHDERR_NOERROR)
        {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot open RX0 stream on DELTA-DVI board handle.");
                goto no_stream;
        }

        /* Configure color space reception (RGBA for no color-space conversion) */
        if (ug_delta_codec_mapping.find(s->codec) != ug_delta_codec_mapping.end()) {
                Packing = ug_delta_codec_mapping.at(s->codec);
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unknown pixel format entered.\n");
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
                if (have_custom_edid) {
                        VHD_LoadEEDID(s->StreamHandle,pEEDIDBuffer,pEEDIDBufferSize);
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

        ULONG SlotsCount, SlotsDropped;
        /* Print some statistics */
        VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_SLOTS_DROPPED,
                              &SlotsDropped);
        VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_SLOTS_COUNT,
                              &SlotsCount);
        log_msg(SlotsDropped > 0 ? LOG_LEVEL_WARNING : LOG_LEVEL_INFO,
                "%" PRIu_ULONG " frames %s (%" PRIu_ULONG
                         " dropped)\n",
                SlotsCount, "hh", SlotsDropped);
        
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
                        DELTA_PRINT_ERROR(Result, "ERROR : Cannot lock slot on RX0 stream.");
                }
                else {
                        log_msg(LOG_LEVEL_WARNING, "Timeout \n");
                }
                return NULL;
        }

         Result = VHD_GetSlotBuffer(SlotHandle, VHD_DV_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                DELTA_PRINT_ERROR(Result, "ERROR : Cannot get slot buffer.");
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
        struct timeval t;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, s->t0);    
        if (seconds >= DELTA_DROP_WARN_INT_SEC) {
                delta_print_slot_stats(s->StreamHandle, &s->SlotsDroppedLast,
                                       "received");
                s->t0 = t;
        }
        
	return out;
}

static const struct video_capture_info vidcap_deltacast_dvi_info = {
        vidcap_deltacast_dvi_probe,
        vidcap_deltacast_dvi_init,
        vidcap_deltacast_dvi_done,
        vidcap_deltacast_dvi_grab,
        MOD_NAME,
};

REGISTER_MODULE(deltacast-dv, &vidcap_deltacast_dvi_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


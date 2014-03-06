/*
 * FILE:    video_capture/deltacast.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

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

#ifdef __cplusplus
} // END of extern "C"
#endif

#include "video_capture/deltacast_dvi.h"

#include <VideoMasterHD_Core.h>
#include <VideoMasterHD_Dvi.h>

#define DEFAULT_BUFFERQUEUE_DEPTH 5

struct vidcap_deltacast_dvi_state {
        ULONG             BoardType;
        HANDLE            BoardHandle, StreamHandle;

        struct       timeval t, t0;
        int          frames;

        codec_t      codec;
        bool         configured;
        struct video_desc desc;
};

#define EEDDIDOK 0
#define BADEEDID 1

static void usage(void);
static BOOL CheckEEDID(BYTE pEEDIDBuffer[256]);
static const char * GetErrorDescription(ULONG CodeError) __attribute__((unused));

static void usage(void)
{
        ULONG             Result,DllVersion,NbBoards;
        printf("-t deltacast[:board=<index>][:channel=<channel>][:codec=<color_spec>]"
                        "[:edid=<edid>|preset=<format>]\n");
        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot query VideoMasterHD"
                                " information. Result = 0x%08X\n",
                                Result);
                return;
        }
        if (NbBoards == 0) {
                fprintf(stderr, "[DELTACAST] No DELTA board detected, exiting...\n");
                return;
        }
        
        printf("\t<index> - index of DVI card\n");
        printf("\tAvailable cards:\n");
        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++)
        {
                ULONG BoardType;
                HANDLE            BoardHandle = NULL;
                Result = VHD_OpenBoardHandle(i,&BoardHandle,NULL,0);
                VHD_GetBoardProperty(BoardHandle, VHD_CORE_BP_BOARD_TYPE, &BoardType);
                if (Result == VHDERR_NOERROR)
                {
                        printf("\t\tBoard %d: ", i);
                        switch(BoardType)
                        {
                                case VHD_BOARDTYPE_HD :    printf("HD board type"); break;
                                case VHD_BOARDTYPE_HDKEY : printf("HD key board type"); break;
                                case VHD_BOARDTYPE_SD :    printf("SD board type"); break;
                                case VHD_BOARDTYPE_SDKEY : printf("SD key board type"); break;
                                case VHD_BOARDTYPE_DVI :   printf("DVI board type"); break;
                                case VHD_BOARDTYPE_CODEC : printf("CODEC board type"); break;
                                case VHD_BOARDTYPE_3G :    printf("3G board type"); break;
                                case VHD_BOARDTYPE_3GKEY : printf("3G key board type"); break;
                                case VHD_BOARDTYPE_HDMI  : printf("HDMI board type"); break;
                                default :                  printf("Unknown board type"); break;
                        }
                        printf("\n");
                        VHD_CloseBoardHandle(BoardHandle);
                }
        }

        printf("\t<channel> may be channel index (for cards which have multiple inputs, max 4)\n");
        
        printf("\t<edid> may be one of following\n");
        printf("\t\t0 - load DVI-D EEDID\n");
        printf("\t\t1 - load DVI-A EEDID\n");
        printf("\t\t2 - load HDMI EEDID\n");
        printf("\t\t3 - avoid EEDID loading\n");

        printf("\t<color_spec> may be one of following\n");
        printf("\t\tUYVY");
        printf("\t\tRGBA");
        printf("\t\tBGR (default)");

        printf("\t<preset> may be format description (DVI-A), E-EDID will be ignored\n");
        printf("\t\tvideo format is in the format <width>x<height>@<fps>\n");

}

static BOOL CheckEEDID(BYTE pEEDIDBuffer[256])
{

        int i;
        UBYTE sum1 = 0,sum2 = 0;
        BOOL Return = EEDDIDOK;

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

static const char * GetErrorDescription(ULONG CodeError)
{
        switch (CodeError)
        {
                case VHDERR_NOERROR :               return "No error";
                case VHDERR_FATALERROR :            return "Fatal error occurred (should re-install)";
                case VHDERR_OPERATIONFAILED :       return "Operation failed (undefined error)";
                case VHDERR_NOTENOUGHRESOURCE :     return "Not enough resource to complete the operation";
                case VHDERR_NOTIMPLEMENTED :        return "Not implemented yet";
                case VHDERR_NOTFOUND :              return "Required element was not found";
                case VHDERR_BADARG :                return "Bad argument value";
                case VHDERR_INVALIDPOINTER :        return "Invalid pointer";
                case VHDERR_INVALIDHANDLE :         return "Invalid handle";
                case VHDERR_INVALIDPROPERTY :       return "Invalid property index";
                case VHDERR_INVALIDSTREAM :         return "Invalid stream or invalid stream type";
                case VHDERR_RESOURCELOCKED :        return "Resource is currently locked";
                case VHDERR_BOARDNOTPRESENT :       return "Board is not available";
                case VHDERR_INCOHERENTBOARDSTATE :  return "Incoherent board state or register value";
                case VHDERR_INCOHERENTDRIVERSTATE : return "Incoherent driver state";
                case VHDERR_INCOHERENTLIBSTATE :    return "Incoherent library state";
                case VHDERR_SETUPLOCKED :           return "Configuration is locked";
                case VHDERR_CHANNELUSED :           return "Requested channel is already used or doesn't exist";
                case VHDERR_STREAMUSED :            return "Requested stream is already used";
                case VHDERR_READONLYPROPERTY :      return "Property is read-only";
                case VHDERR_OFFLINEPROPERTY :       return "Property is off-line only";
                case VHDERR_TXPROPERTY :            return "Property is of TX streams";
                case VHDERR_TIMEOUT :               return "Time-out occurred";
                case VHDERR_STREAMNOTRUNNING :      return "Stream is not running";
                case VHDERR_BADINPUTSIGNAL :        return "Bad input signal, or unsupported standard";
                case VHDERR_BADREFERENCESIGNAL :    return "Bad genlock signal or unsupported standard";                                 
                case VHDERR_FRAMELOCKED :           return "Frame already locked";
                case VHDERR_FRAMEUNLOCKED :         return "Frame already unlocked";
                case VHDERR_INCOMPATIBLESYSTEM :    return "Selected video standard is incompatible with running clock system";
                case VHDERR_ANCLINEISEMPTY :        return "ANC line is empty";
                case VHDERR_ANCLINEISFULL :         return "ANC line is full";
                case VHDERR_BUFFERTOOSMALL :        return "Buffer too small";
                case VHDERR_BADANC :                return "Received ANC aren't standard";
                case VHDERR_BADCONFIG :             return "Invalid configuration";
                case VHDERR_FIRMWAREMISMATCH :      return "The loaded firmware is not compatible with the installed driver";
                case VHDERR_LIBRARYMISMATCH :       return "The loaded VideomasterHD library is not compatible with the installed driver";
                case VHDERR_FAILSAFE :              return "The fail safe firmware is loaded. You need to upgrade your firmware";
                case VHDERR_RXPROPERTY :            return "Property is of RX streams";
                default:                            return "Unknown code error";
        }
}

struct vidcap_type *
vidcap_deltacast_dvi_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DELTACAST_DVI_ID;
		vt->name        = "deltacast-dvi";
		vt->description = "DELTACAST DVI card";
	}
	return vt;
}

static bool wait_for_channel_locked(struct vidcap_deltacast_dvi_state *s, bool have_preset,
        VHD_DVI_MODE DviMode,
        ULONG Width, ULONG Height, ULONG RefreshRate)
{
        BOOL Interlaced_B = FALSE;
        ULONG             Result = VHDERR_NOERROR;

        struct timeval t0, t;

        gettimeofday(&t0, NULL);

        if(!have_preset) {
                /* Wait for channel locked */
                printf("Waiting for incoming signal...\n");
                do
                {
                        Result = VHD_GetStreamProperty(s->StreamHandle, VHD_DVI_SP_MODE, (ULONG *) &DviMode);
                        gettimeofday(&t, NULL);
                        if(tv_diff(t, t0) > 1.0) break;
                } while (Result != VHDERR_NOERROR);

                if(Result != VHDERR_NOERROR)
                        return false;
        }

        printf("\nIncoming Dvi mode detected: ");
        switch(DviMode)
        {
                case VHD_DVI_MODE_DVI_D                   : printf("DVI-D\n");break;
                case VHD_DVI_MODE_DVI_A                   : printf("DVI-A\n");break;
                case VHD_DVI_MODE_ANALOG_COMPONENT_VIDEO  : printf("Analog component video\n");break;
                case VHD_DVI_MODE_HDMI                    : printf("HDMI\n");break;
                default                                   : break;
        }

        /* Disable EDID auto load */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_DISABLE_EDID_AUTO_LOAD,TRUE);
        if(Result != VHDERR_NOERROR)
                return false;

        /* Set the DVI mode of this channel to the detected one */
        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_MODE, DviMode);
        if(Result != VHDERR_NOERROR)
                return false;

        if(DviMode == VHD_DVI_MODE_DVI_A)
        {
                VHD_DVI_A_STANDARD DviAStd = VHD_DVIA_STD_DMT;
                if(!have_preset) {
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
                        Result = VHD_DetectDviAFormat(s->StreamHandle,&DviAStd,&Width,&Height,&RefreshRate,
                                        &Interlaced_B);
                }
                if(Result == VHDERR_NOERROR)
                {
                        printf("\nDVI-A format detected: %ux%u @%uHz (%s)\n",Width,Height,RefreshRate
                                        ,Interlaced_B?"Interlaced":"Progressive");
                        Result = VHD_PresetDviAStreamProperties(s->StreamHandle, DviAStd,Width,Height,
                                        RefreshRate,Interlaced_B);
                }
                else {
                        printf("ERROR : Cannot detect incoming DVI-A format. Result = 0x%08X\n",
                                        Result);
                        return false;
                }
        }
        else if(DviMode == VHD_DVI_MODE_DVI_D)
        {
                BOOL Dual_B = FALSE;
                /* Get auto-detected resolution */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_WIDTH,&Width);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_HEIGHT,&Height);
                else
                        printf("ERROR : Cannot detect incoming active width from RX0. "
                                        "Result = 0x%08X\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_INTERLACED,(ULONG*)&Interlaced_B);
                else
                        printf("ERROR : Cannot detect incoming active height from RX0. "
                                        "Result = 0x%08X\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_REFRESH_RATE,&RefreshRate);
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is "
                                        "interlaced or progressive. Result = 0x%08X\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_DUAL_LINK,(ULONG*)&Dual_B);
                else
                        printf("ERROR : Cannot detect incoming refresh rate from RX0. "
                                        "Result = 0x%08X\n", Result);

                if(Result == VHDERR_NOERROR)
                        printf("\nIncoming graphic resolution : %ux%u @%uHz (%s) %s link\n", Width, Height,RefreshRate, Interlaced_B?"Interlaced":"Progressive",Dual_B?"Dual":"Single");
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is dual or simple link. "
                                        "Result = 0x%08X\n", Result);

                if(Result != VHDERR_NOERROR) {
                        return false;
                }

                /* Configure stream. Only VHD_DVI_SP_ACTIVE_WIDTH, VHD_DVI_SP_ACTIVE_HEIGHT and
                   VHD_DVI_SP_INTERLACED properties are required for DVI-D.
                   VHD_DVI_SP_REFRESH_RATE,VHD_DVI_SP_DUAL_LINK are optional
                   VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                   VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                   VHD_DVI_SP_V_FRONT_PORCH properties are not applicable for DVI-D */

                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_WIDTH,Width);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_HEIGHT,Height);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_INTERLACED,Interlaced_B);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_REFRESH_RATE,RefreshRate);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_DUAL_LINK,Dual_B);
        }
        else if(DviMode == VHD_DVI_MODE_HDMI || DviMode == VHD_DVI_MODE_ANALOG_COMPONENT_VIDEO)
        {
                VHD_HDMI_CS       InputCS;
                ULONG             PxlClk = 0;
                /* Get auto-detected resolution */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_WIDTH,&Width);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_HEIGHT,&Height);
                else
                        printf("ERROR : Cannot detect incoming active width from RX0. "
                                        "Result = 0x%08X\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_INTERLACED,(ULONG*)&Interlaced_B);
                else
                        printf("ERROR : Cannot detect incoming active height from RX0. "
                                        "Result = 0x%08X\n", Result);
                if(Result == VHDERR_NOERROR)
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_REFRESH_RATE,&RefreshRate);
                else
                        printf("ERROR : Cannot detect if incoming stream from RX0 is "
                                        "interlaced or progressive. Result = 0x%08X\n", Result);

                if (Result == VHDERR_NOERROR) {
                        Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_REFRESH_RATE,&RefreshRate);
                        if(s->BoardType == VHD_BOARDTYPE_HDMI)
                                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_INPUT_CS,(ULONG*)&InputCS);
                        else
                                printf("ERROR : Cannot detect incoming color space from RX0. Result = 0x%08X (%s)\n", Result,
                                                GetErrorDescription(Result));
                }

                if (Result == VHDERR_NOERROR) {
                        if (s->BoardType == VHD_BOARDTYPE_HDMI)
                                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_DVI_SP_PIXEL_CLOCK,&PxlClk);
                        else
                                printf("ERROR : Cannot detect incoming pixel clock from RX0. Result = 0x%08X (%s)\n", Result,
                                                GetErrorDescription(Result));
                }

                if(Result == VHDERR_NOERROR)
                        printf("\nIncoming graphic resolution : %ux%u @%uHz (%s)\n", Width, Height, RefreshRate, Interlaced_B?"Interlaced":"Progressive");
                else
                        printf("ERROR : Cannot detect incoming refresh rate from RX0. "
                                        "Result = 0x%08X\n", Result);

                if(Result != VHDERR_NOERROR) {
                        return false;
                }

                /* Configure stream. Only VHD_DVI_SP_ACTIVE_WIDTH, VHD_DVI_SP_ACTIVE_HEIGHT and
                   VHD_DVI_SP_INTERLACED properties are required for HDMI and Component
                   VHD_DVI_SP_PIXEL_CLOCK, VHD_DVI_SP_TOTAL_WIDTH, VHD_DVI_SP_TOTAL_HEIGHT,
                   VHD_DVI_SP_H_SYNC, VHD_DVI_SP_H_FRONT_PORCH, VHD_DVI_SP_V_SYNC and
                   VHD_DVI_SP_V_FRONT_PORCH properties are not applicable for DVI-D, HDMI and Component */

                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_WIDTH,Width);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_ACTIVE_HEIGHT,Height);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_INTERLACED,Interlaced_B);
                VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_REFRESH_RATE, RefreshRate);
                if (s->BoardType == VHD_BOARDTYPE_HDMI) {
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_INPUT_CS, InputCS);
                        VHD_SetStreamProperty(s->StreamHandle,VHD_DVI_SP_PIXEL_CLOCK, PxlClk);
                }
        }

        Result = VHD_StartStream(s->StreamHandle);

        if(Result != VHDERR_NOERROR) {
                fprintf(stderr, "Cannot start stream!\n");
                return false;
        }

        s->desc.color_spec = s->codec;
        s->desc.width = Width;
        s->desc.height = Height;
        s->desc.fps = RefreshRate;
        s->desc.interlacing = Interlaced_B ? INTERLACED_MERGED : PROGRESSIVE;
        s->desc.tile_count = 1;

        return true;
}

void *
vidcap_deltacast_dvi_init(const struct vidcap_params *params)
{
	struct vidcap_deltacast_dvi_state *s;
        ULONG Width = 0, Height = 0, RefreshRate = 0;
        ULONG             Result = VHDERR_NOERROR,DllVersion,NbBoards;
        ULONG             BrdId = 0;
        ULONG             Packing;
        int               edid = -1;
        BYTE              pEEDIDBuffer[256];
        ULONG             pEEDIDBufferSize=256;
        int               channel = 0;
        ULONG             ChannelId;
        bool              have_preset = false;
        VHD_DVI_MODE      DviMode = NB_VHD_DVI_MODES;

        int               i;

	printf("vidcap_deltacast_dvi_init\n");

        s = (struct vidcap_deltacast_dvi_state *) calloc(1, sizeof(struct vidcap_deltacast_dvi_state));
        s->codec = BGR;
        s->configured = false;
        
	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
		return NULL;
	}

        s->BoardHandle = s->StreamHandle = NULL;

        char *init_fmt = NULL;
        if (vidcap_params_get_fmt(params) != NULL)
                init_fmt = strdup(vidcap_params_get_fmt(params));
        if(init_fmt && strcmp(init_fmt, "help") == 0) {
                usage();
                return &vidcap_init_noerr;
        }

        if(init_fmt)
        {
                char *save_ptr = NULL;
                char *tok;
                
                while((tok = strtok_r(init_fmt, ":", &save_ptr)) != NULL) {
                        if(strncasecmp(tok, "board=", strlen("board=")) == 0) {
                                BrdId = atoi(tok + strlen("board="));
                        } else if(strncasecmp(tok, "codec=", strlen("codec=")) == 0) {
                                char *codec_str = tok + strlen("codec=");

                                for (i = 0; codec_info[i].name != NULL; i++) {
                                        if (strcmp(codec_str, codec_info[i].name) == 0) {
                                                s->codec = codec_info[i].codec;
                                                break;
                                        }
                                }
                                if(codec_info[i].name == NULL) {
                                        fprintf(stderr, "Unable to find codec: %s\n",
                                                        codec_str);
                                        goto error;
                                }
                        } else if(strncasecmp(tok, "edid=", strlen("edid=")) == 0) {
                                        edid = atoi(tok + strlen("edid="));
                                        if(edid < 0 || edid > 3) {
                                                fprintf(stderr, "[DELTA] Error: Wrong "
                                                                "EDID entered on commandline. "
                                                                "Expected 0-3, got %d.\n", edid);
                                                goto error;

                                        }
                        } else if(strncasecmp(tok, "channel=", strlen("channel=")) == 0) {
                                channel = atoi(tok + strlen("channel="));
                        } else if(strncasecmp(tok, "preset=", strlen("preset=")) == 0) {
                                have_preset = true;
                                char *ptr = tok + strlen("preset=");
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
                                fprintf(stderr, "[DELTA] Error: Unrecongnized "
                                                "trailing parameter %s\n", tok);
                                goto error;
                        }
                        init_fmt = NULL;
                }
        } else {
                BrdId = 0;
                printf("[DELTACAST] Automatically choosen device nr. 0\n");
        }
        free(init_fmt);

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
        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_BOARD_TYPE, &s->BoardType);
        if (s->BoardType != VHD_BOARDTYPE_DVI && s->BoardType != VHD_BOARDTYPE_HDMI) {
                fprintf(stderr, "[DELTACAST] ERROR : The selected board is not an DVI or HDMI one\n");
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
                        fprintf(stderr, "[DELTACAST] Bad channel index!\n");
                        goto no_stream;
        }
        Result = VHD_OpenStreamHandle(s->BoardHandle, ChannelId,
                        s->BoardType == VHD_BOARDTYPE_HDMI ? VHD_DVI_STPROC_JOINED : VHD_DVI_STPROC_DEFAULT,
                        NULL, &s->StreamHandle, NULL);
        if (Result != VHDERR_NOERROR)
        {
                fprintf(stderr, "ERROR : Cannot open RX0 stream on DELTA-DVI board handle. "
                                "Result = 0x%08X\n",Result);
                goto no_stream;
        }

        /* Configure color space reception (RGBA for no color-space conversion) */
        switch(s->codec) {
                case BGR:
                        Packing = VHD_BUFPACK_VIDEO_RGB_24;
                        break;
                case RGBA:
                        Packing = VHD_BUFPACK_VIDEO_RGBA_32;
                        break;
                case UYVY:
                        Packing = VHD_BUFPACK_VIDEO_YUV422_8;
                        break;
                default:
                        fprintf(stderr, "Unknown pixel formate entered.\n");
                        goto no_format;
        }

        Result = VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_BUFFER_PACKING,
                        Packing);
        if(Result != VHDERR_NOERROR) {
                fprintf(stderr, "Unable to set packing format.\n");
                goto no_format;
        }
        VHD_SetStreamProperty(s->StreamHandle,VHD_CORE_SP_TRANSFER_SCHEME,
                        VHD_TRANSFER_SLAVED);

        Result = VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFERQUEUE_DEPTH,
                        DEFAULT_BUFFERQUEUE_DEPTH);
        if(Result != VHDERR_NOERROR) {
                fprintf(stderr, "[Delta-dvi] Warning: Unable to set buffer queue length.\n");
        }

        if(have_preset) {
                DviMode = VHD_DVI_MODE_DVI_A;
        } else {
                switch(edid)
                {
                        case 0 : VHD_PresetEEDID(VHD_EEDID_DVID,pEEDIDBuffer,256);
                                 VHD_LoadEEDID(s->StreamHandle,pEEDIDBuffer,256);
                                 break;
                        case 1 : VHD_PresetEEDID(VHD_EEDID_DVIA,pEEDIDBuffer,256);
                                 VHD_LoadEEDID(s->StreamHandle,pEEDIDBuffer,256);
                                 break;
                        case 2 : VHD_PresetEEDID(VHD_EEDID_HDMI,pEEDIDBuffer,256);
                                 VHD_LoadEEDID(s->StreamHandle,pEEDIDBuffer,256);
                                 break;
                        default : break;
                }
                /* Read EEDID and check its validity */
                Result = VHD_ReadEEDID(s->BoardHandle,VHD_ST_RX0,pEEDIDBuffer,&pEEDIDBufferSize);
                if(Result == VHDERR_NOTIMPLEMENTED || CheckEEDID(pEEDIDBuffer) == BADEEDID)
                {
                        /* Propose edid preset to user and load */
                        fprintf(stderr, "\nNo valid EEDID detected or DELTA-dvi board V1.\n");
                        fprintf(stderr, "Please set it as a command-line option.\n");
                        goto no_format;
                }
        }

        s->configured = wait_for_channel_locked(s, have_preset, DviMode, Width, Height, RefreshRate);
        if(!s->configured &&
                        have_preset) {
                fprintf(stderr, "Unable to set preset format!\n");
                goto no_format;
        }

        gettimeofday(&s->t0, NULL);
        s->frames = 0;
        
	return s;

no_format:
        /* Close stream handle */
        VHD_CloseStreamHandle(s->StreamHandle);

no_stream:
        
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
bad_channel:
        VHD_CloseBoardHandle(s->BoardHandle);
error:
        free(s);
        return NULL;
}

void
vidcap_deltacast_dvi_done(void *state)
{
	struct vidcap_deltacast_dvi_state *s = (struct vidcap_deltacast_dvi_state *) state;

	assert(s != NULL);
        
        VHD_StopStream(s->StreamHandle);
        VHD_CloseStreamHandle(s->StreamHandle);
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
        VHD_CloseBoardHandle(s->BoardHandle);
        
        free(s);
}

static void vidcap_deltacast_dvi_dispose(struct video_frame *f)
{
       VHD_UnlockSlotHandle((HANDLE) f->dispose_udata);
       vf_free(f);
}

struct video_frame *
vidcap_deltacast_dvi_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_deltacast_dvi_state   *s = (struct vidcap_deltacast_dvi_state *) state;
        
        ULONG             /*SlotsCount, SlotsDropped,*/BufferSize;
        ULONG             Result;
        BYTE             *pBuffer=NULL;
        HANDLE            SlotHandle;

        if(!s->configured) {
                s->configured = wait_for_channel_locked(s, false, NB_VHD_DVI_MODES, 0, 0, 0);
        }
        if(!s->configured) {
                return NULL;
        }

        *audio = NULL;

        Result = VHD_LockSlotHandle(s->StreamHandle, &SlotHandle);
        if (Result != VHDERR_NOERROR) {
                if (Result != VHDERR_TIMEOUT) {
                        fprintf(stderr, "ERROR : Cannot lock slot on RX0 stream. Result = 0x%08X\n",Result);
                }
                else {
                        fprintf(stderr, "Timeout \n");
                }
                return NULL;
        }

         Result = VHD_GetSlotBuffer(SlotHandle, VHD_DVI_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "\nERROR : Cannot get slot buffer. Result = 0x%08X\n",Result);
                return NULL;
         }

         if(s->desc.color_spec == RGBA) {
                 vc_copylineRGBA(pBuffer, pBuffer, BufferSize, 16, 8, 0);
         }
         
	 struct video_frame *out = vf_alloc_desc(s->desc);
         out->tiles[0].data = (char*) pBuffer;
         out->tiles[0].data_len = BufferSize;
         out->dispose_udata = SlotHandle;
         out->dispose = vidcap_deltacast_dvi_dispose;

         /* Print some statistics */
         /*VHD_GetStreamProperty(s->StreamHandle,VHD_CORE_SP_SLOTS_COUNT,&SlotsCount);
         VHD_GetStreamProperty(s->StreamHandle,VHD_CORE_SP_SLOTS_DROPPED,&SlotsDropped);
         printf("%u frames received (%u dropped)            \r",SlotsCount,SlotsDropped);*/
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "[DELTACAST cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }
        s->frames++;
        
	return out;
}


/**
 * @file   video_capture/deltacast.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2016 CESNET, z.s.p.o.
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
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"
#include "audio/utils.h"
#include "deltacast_common.h"

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

#include "video_display.h"
#include "video_display/deltacast.h"

using namespace std;

struct vidcap_deltacast_state {
        struct video_frame *frame;
        struct tile        *tile;
        HANDLE            BoardHandle, StreamHandle;
        HANDLE            SlotHandle;

        struct audio_frame audio_frame;
        
        struct       timeval t, t0;
        int          frames;

        ULONG             AudioBufferSize;
        VHD_AUDIOCHANNEL *pAudioChn;
        VHD_AUDIOINFO     AudioInfo;
        ULONG             ClockSystem,VideoStandard;
        unsigned int      grab_audio:1;
        unsigned int      autodetect_format:1;

        unsigned int      initialize_flags;
        bool              initialized;
};

static void usage(void);

static void usage(void)
{
        printf("\t-t deltacast[:device=<index>][:mode=<mode>][:codec=<codec>]\n");

        print_available_delta_boards();

        printf("\nAvailable modes:\n");
        for (int i = 0; i < deltacast_frame_modes_count; ++i)
        {
                printf("\t%d: %s", deltacast_frame_modes[i].mode,
                                deltacast_frame_modes[i].name);
                if (deltacast_frame_modes[i].iface != VHD_INTERFACE_AUTO)
                        printf("\t\t(no autodetection)");
                printf("\n");
        }
        
        printf("\nAvailable codecs:\n");
        printf("\tUYVY\n");
        printf("\tv210\n");
        printf("\traw\n");

        printf("\nDefault board is 0. If mode is omitted, it will be autodetected "
                        "(except of UHD modes). Default codec is UYVY.\n");
}

static struct vidcap_type *
vidcap_deltacast_probe(bool verbose)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->name        = "deltacast";
		vt->description = "DELTACAST card";

                if (verbose) {
                        ULONG             Result,DllVersion,NbBoards;
                        Result = VHD_GetApiInfo(&DllVersion,&NbBoards);
                        if (Result == VHDERR_NOERROR) {
                                vt->cards = (struct device_info *) calloc(NbBoards, sizeof(struct device_info));
                                vt->card_count = NbBoards;
                                for (ULONG i = 0; i < NbBoards; ++i) {
                                        snprintf(vt->cards[i].id, sizeof vt->cards[i].id, "device=%lu", i);
                                        snprintf(vt->cards[i].name, sizeof vt->cards[i].name, "DELTACAST SDI board %lu", i);
                                }
                        }
                }
	}
	return vt;
}

class delta_init_exception {
};

#define DELTA_TRY_CMD(cmd, msg) \
        do {\
                Result = cmd;\
                if (Result != VHDERR_NOERROR) {\
                        fprintf(stderr, "[DELTACAST] " msg " Result = 0x%08lX\n", Result);\
                        throw delta_init_exception();\
                }\
        } while(0)

/**
 * Function initialize is intended to be called repeatedly if no signal detected
 * in vidcap_deltacast_init(). This will be tried everytime grab is called and until
 * initialized.
 */
static bool wait_for_channel(struct vidcap_deltacast_state *s)
{
        ULONG             Result;
        ULONG             Packing;
        ULONG             Status = 0;
        int               i;
        ULONG             Interface = 0;

        /* Wait for channel locked */
        Result = VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_RX0_STATUS, &Status);

        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot get channel status. Result = 0x%08lX\n",Result);
                throw delta_init_exception();
        }

        // no signal was detected on the wire
        if(Status & VHD_CORE_RXSTS_UNLOCKED) {
                return false;
        }

        /* Auto-detect clock system */
        Result = VHD_GetBoardProperty(s->BoardHandle,VHD_SDI_BP_RX0_CLOCK_DIV,&s->ClockSystem);

        if(Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot detect incoming clock system from RX0. Result = 0x%08lX\n",Result);
                throw delta_init_exception();
        } else {
                printf("\nIncoming clock system : %s\n",(s->ClockSystem==VHD_CLOCKDIV_1)?"European":"American");
        }

         /* Select the detected clock system */
        VHD_SetBoardProperty(s->BoardHandle,VHD_SDI_BP_CLOCK_SYSTEM,s->ClockSystem);

        /* Create a logical stream to receive from RX0 connector */
        if(!s->autodetect_format && s->frame->color_spec == RAW)
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_RX0,VHD_SDI_STPROC_RAW,NULL,&s->StreamHandle,NULL);
        else if(s->initialize_flags & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_RX0,VHD_SDI_STPROC_JOINED,NULL,&s->StreamHandle,NULL);
        } else {
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_RX0,VHD_SDI_STPROC_DISJOINED_VIDEO,NULL,&s->StreamHandle,NULL);
        }


        if (Result != VHDERR_NOERROR)
        {
                fprintf(stderr, "ERROR : Cannot open RX0 stream on DELTA-hd/sdi/codec board handle. Result = 0x%08lX\n",Result);
                throw delta_init_exception();
        }

        if(s->autodetect_format) {
                /* Get auto-detected video standard */
                Result = VHD_GetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD,&s->VideoStandard);

                if ((Result == VHDERR_NOERROR) && (s->VideoStandard != NB_VHD_VIDEOSTANDARDS)) {
                } else {
                        fprintf(stderr, "[DELTACAST] Cannot detect incoming video standard from RX0. Result = 0x%08lX\n",Result);
                        throw delta_init_exception();
                }
        }

        for (i = 0; i < deltacast_frame_modes_count; ++i)
        {
                if(s->VideoStandard == (ULONG) deltacast_frame_modes[i].mode) {
                        s->frame->fps = deltacast_frame_modes[i].fps;
                        s->frame->interlacing = deltacast_frame_modes[i].interlacing;
                        s->tile->width = deltacast_frame_modes[i].width;
                        s->tile->height = deltacast_frame_modes[i].height;
                        Interface = deltacast_frame_modes[i].iface;
                        printf("[DELTACAST] %s mode selected. %dx%d @ %2.2f %s\n", deltacast_frame_modes[i].name, s->tile->width, s->tile->height,
                                        (double) s->frame->fps, get_interlacing_description(s->frame->interlacing));
                        break;
                }
        }
        if(i == deltacast_frame_modes_count) {
                fprintf(stderr, "[DELTACAST] Failed to obtain information about video format %lu.\n", s->VideoStandard);
                throw delta_init_exception();
        }

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
                        fprintf(stderr, "[DELTACAST] Unable to get pixel format\n");
                        throw delta_init_exception();
                }
                if(Packing == VHD_BUFPACK_VIDEO_YUV422_10) {
                        s->frame->color_spec = v210;
                } else if(Packing == VHD_BUFPACK_VIDEO_YUV422_8) {
                        s->frame->color_spec = UYVY;
                } else {
                        fprintf(stderr, "[DELTACAST] Detected unknown pixel format!\n");
                        throw delta_init_exception();
                }
        }
        if(s->frame->color_spec == v210 || s->frame->color_spec == RAW) {
                Packing = VHD_BUFPACK_VIDEO_YUV422_10;
        } else if(s->frame->color_spec == UYVY) {
                Packing = VHD_BUFPACK_VIDEO_YUV422_8;
        } else {
                fprintf(stderr, "[DELTACAST] Unsupported pixel format\n");
                throw delta_init_exception();
        }
        printf("[DELTACAST] Pixel format '%s' selected.\n", get_codec_name(s->frame->color_spec));
        Result = VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFER_PACKING, Packing);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to set pixel format\n");
                throw delta_init_exception();
        }

        if(s->initialize_flags & DISPLAY_FLAG_AUDIO_EMBEDDED) {
                if(audio_capture_channels != 1 &&
                                audio_capture_channels != 2) {
                        fprintf(stderr, "[DELTACAST capture] Unable to handle channel count other than 1 or 2.\n");
                        throw delta_init_exception();
                }
                s->audio_frame.bps = 3;
                s->audio_frame.sample_rate = 48000;
                s->audio_frame.ch_count = audio_capture_channels;
                memset(&s->AudioInfo, 0, sizeof(VHD_AUDIOINFO));
                s->pAudioChn = &s->AudioInfo.pAudioGroups[0].pAudioChannels[0];
                if(audio_capture_channels == 1) {
                        s->pAudioChn->Mode = s->AudioInfo.pAudioGroups[0].pAudioChannels[1].Mode=VHD_AM_MONO;
                } else if(audio_capture_channels == 2) {
                        s->pAudioChn->Mode = s->AudioInfo.pAudioGroups[0].pAudioChannels[1].Mode=VHD_AM_STEREO;
                } else abort();
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
                fprintf(stderr, "[DELTACAST] ERROR : Cannot start RX0 stream on DELTA-hd/sdi/codec board handle. Result = 0x%08lX\n",Result);
                throw delta_init_exception();
        }

        return true;
}

static int
vidcap_deltacast_init(const struct vidcap_params *params, void **state)
{
	struct vidcap_deltacast_state *s = nullptr;
        ULONG             Result,DllVersion,NbBoards,ChnType;
        ULONG             BrdId = 0;

	printf("vidcap_deltacast_init\n");

        char *init_fmt = strdup(vidcap_params_get_fmt(params));
        if (init_fmt && strcmp(init_fmt, "help") == 0) {
                free(init_fmt);
                usage();
                return VIDCAP_INIT_NOERR;
        }

        s = (struct vidcap_deltacast_state *) calloc(1, sizeof(struct vidcap_deltacast_state));

	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
                free(init_fmt);
                free(s);
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

        if (init_fmt) {
                char *save_ptr = NULL;
                char *tok;
                char *tmp = init_fmt;

                while ((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                        if (strncasecmp(tok, "device=", strlen("device=")) == 0) {
                                BrdId = atoi(tok + strlen("device="));
                        } else if (strncasecmp(tok, "board=", strlen("board=")) == 0) {
                                // compat, should be device= instead
                                BrdId = atoi(tok + strlen("board="));
                        } else if (strncasecmp(tok, "mode=", strlen("mode=")) == 0) {
                                s->VideoStandard = atoi(tok + strlen("mode="));
                                s->autodetect_format = FALSE;
                        } else if (strncasecmp(tok, "codec=", strlen("codec=")) == 0) {
                                tok = tok + strlen("codec=");
                                if(strcasecmp(tok, "raw") == 0)
                                        s->frame->color_spec = RAW;
                                else if(strcmp(tok, "UYVY") == 0)
                                        s->frame->color_spec = UYVY;
                                else if(strcmp(tok, "v210") == 0)
                                        s->frame->color_spec = v210;
                                else {
                                        fprintf(stderr, "Wrong "
                                        "codec entered.\n");
                                        usage();
                                        goto error;
                                }
                        } else {
                                fprintf(stderr, "[DELTACAST] Wrong config option '%s'!\n", tok);
                                goto error;
                        }
                        tmp = NULL;
                }
        }
        free(init_fmt);
        init_fmt = NULL;

        printf("[DELTACAST] Selected device %lu\n", BrdId);

        if(s->autodetect_format) {
                printf("DELTACAST] We will try to autodetect incoming video format.\n");
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
                fprintf(stderr, "[DELTACAST] ERROR : Cannot open DELTA board %lu handle. Result = 0x%08lX\n",BrdId,Result);
                goto error;
        }

        if (!delta_set_nb_channels(BrdId, s->BoardHandle, 1, 0)) {
                goto error;
        }

        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_RX0_TYPE, &ChnType);
        if((ChnType!=VHD_CHNTYPE_SDSDI)&&(ChnType!=VHD_CHNTYPE_HDSDI)&&(ChnType!=VHD_CHNTYPE_3GSDI)) {
                fprintf(stderr, "[DELTACAST] ERROR : The selected channel is not an SDI one\n");
                goto error;
        }

        /* Disable RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,FALSE);
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_1,FALSE);
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_2,FALSE);
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_3,FALSE);

        s->initialize_flags = vidcap_params_get_flags(params);
        printf("\nWaiting for channel locked...\n");
        try {
                if(wait_for_channel(s)) {
                        s->initialized = true;
                }
        } catch(delta_init_exception &e) {
                goto error;
        }

        *state = s;
	return VIDCAP_INIT_OK;

error:
        free(init_fmt);

        if (s) {
                if(s->StreamHandle) {
                        /* Close stream handle */
                        VHD_CloseStreamHandle(s->StreamHandle);
                }
                if(s->BoardHandle) {
                        /* Re-establish RX0-TX0 by-pass relay loopthrough */
                        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
                        VHD_CloseBoardHandle(s->BoardHandle);
                }
                vf_free(s->frame);
        }

        free(s);
        return VIDCAP_INIT_FAIL;
}

static void
vidcap_deltacast_done(void *state)
{
	struct vidcap_deltacast_state *s = (struct vidcap_deltacast_state *) state;

	assert(s != NULL);
        
        if(s->SlotHandle)
                VHD_UnlockSlotHandle(s->SlotHandle);
        VHD_StopStream(s->StreamHandle);
        VHD_CloseStreamHandle(s->StreamHandle);
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
        VHD_CloseBoardHandle(s->BoardHandle);
        
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
                        fprintf(stderr, "ERROR : Cannot lock slot on RX0 stream. Result = 0x%08lX\n", Result);
                }
                else {
                        fprintf(stderr, "Timeout \n");
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
                        fprintf(stderr, "[DELTACAST] Audio grabbing error. Result = 0x%08lX\n",Result);
                }
        }

        
         Result = VHD_GetSlotBuffer(s->SlotHandle, VHD_SDI_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "\nERROR : Cannot get slot buffer. Result = 0x%08lX\n",Result);
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
};

REGISTER_MODULE(deltacast, &vidcap_deltacast_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


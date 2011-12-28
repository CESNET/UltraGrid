/*
 * FILE:    deltacast.c
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
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <semaphore.h>

#ifdef __cplusplus
} // END of extern "C"
#endif

#include "video_capture/deltacast.h"
#include "video_display/deltacast.h"

#include <VideoMasterHD_Core.h>
#include <VideoMasterHD_Sdi.h>

struct vidcap_deltacast_state {
        struct video_frame *frame;
        struct tile        *tile;
        HANDLE            BoardHandle, StreamHandle;
        HANDLE            SlotHandle;
        int mode;
        
        struct       timeval t, t0;
        int          frames;
};

static void usage(void);

static void usage(void)
{
        ULONG             Result,DllVersion,NbBoards;
        int               i;
        printf("-t deltacast:<index>:<mode>:<codec>\n");
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
        
        printf("\nAvailable cards:\n");
        /* Query DELTA boards information */
        for (ULONG i = 0; i < NbBoards; i++)
        {
                HANDLE            BoardHandle = NULL;
                Result = VHD_OpenBoardHandle(i,&BoardHandle,NULL,0);
                if (Result == VHDERR_NOERROR)
                {
                        printf("\tBoard %d\n", i);
                        // Here woulc go detais
                        VHD_CloseBoardHandle(BoardHandle);
                }
        }
        printf("\nAvailable modes:\n");
        for (i = 0; i < deltacast_frame_modes_count; ++i)
        {
                printf("\t%d: %s\n", deltacast_frame_modes[i].mode,
                                deltacast_frame_modes[i].name);
        }
        
        printf("\nAvailable codecs:\n");
        printf("\tUYVY\n");
        printf("\tv210\n");
        printf("\traw\n");
}

struct vidcap_type *
vidcap_deltacast_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DELTACAST_ID;
		vt->name        = "deltacast";
		vt->description = "DELTACAST card";
	}
	return vt;
}

void *
vidcap_deltacast_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_deltacast_state *s;
        ULONG             Result,DllVersion,NbBoards,ChnType;
        
        ULONG             BrdId = 0;
        ULONG             ClockSystem,VideoStandard;
        ULONG             Packing;
        ULONG             Status = 0;
        int               i;

	printf("vidcap_deltacast_init\n");

        s = (struct vidcap_deltacast_state *) malloc(sizeof(struct vidcap_deltacast_state));
        
	if(s == NULL) {
		printf("Unable to allocate DELTACAST state\n");
		return NULL;
	}
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->frames = 0;
        
        s->frame->color_spec = UYVY;
        
        s->BoardHandle = s->StreamHandle = s->SlotHandle = NULL;

        if(!init_fmt || strcmp(init_fmt, "help") == 0) {
                usage();
                goto error;
        }
        
        if(init_fmt)
        {
                char *save_ptr = NULL;
                char *tok;
                
                tok = strtok_r(init_fmt, ":", &save_ptr);
                if(!tok)
                {
                        usage();
                        goto error;
                }
                BrdId = atoi(tok);
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        s->mode = atoi(tok);
                } else {
                        usage();
                        goto error;
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        if(strcasecmp(tok, "raw") == 0)
                                s->frame->color_spec = RAW;
                        else if(strcmp(tok, "UYVY") == 0)
                                s->frame->color_spec = UYVY;
                        else if(strcmp(tok, "v210") == 0)
                                s->frame->color_spec = v210;
                        else {
                                usage();
                                goto error;
                        }

                } else {
                        usage();
                        goto error;
                }
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
        VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_RX0_TYPE, &ChnType);
        if((ChnType!=VHD_CHNTYPE_SDSDI)&&(ChnType!=VHD_CHNTYPE_HDSDI)&&(ChnType!=VHD_CHNTYPE_3GSDI)) {
                fprintf(stderr, "[DELTACAST] ERROR : The selected channel is not an SDI one\n");
                goto bad_channel;
        }
        
        /* Disable RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,FALSE);
        
        /* Wait for channel locked */
        printf("\nWaiting for channel locked...\n");
        do 
        {
                Result = VHD_GetBoardProperty(s->BoardHandle, VHD_CORE_BP_RX0_STATUS, &Status);
        
                if (Result != VHDERR_NOERROR)
                        continue;
        
        } while ((Status & VHD_CORE_RXSTS_UNLOCKED) && !should_exit);

        
        /* Auto-detect clock system */
        Result = VHD_GetBoardProperty(s->BoardHandle,VHD_SDI_BP_RX0_CLOCK_DIV,&ClockSystem);

        if(Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] ERROR : Cannot detect incoming clock system from RX0. Result = 0x%08X\n",Result);
                goto no_clock;
        } else {
                printf("\nIncoming clock system : %s\n",(ClockSystem==VHD_CLOCKDIV_1)?"European":"American");
        }
                
         /* Select the detected clock system */
        VHD_SetBoardProperty(s->BoardHandle,VHD_SDI_BP_CLOCK_SYSTEM,ClockSystem);
                  
        /* Create a logical stream to receive from RX0 connector */
        if(s->frame->color_spec == RAW)
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_RX0,VHD_SDI_STPROC_RAW,NULL,&s->StreamHandle,NULL);
        else
                Result = VHD_OpenStreamHandle(s->BoardHandle,VHD_ST_RX0,VHD_SDI_STPROC_DISJOINED_VIDEO,NULL,&s->StreamHandle,NULL);

                
        if (Result != VHDERR_NOERROR)
        {
                fprintf(stderr, "ERROR : Cannot open RX0 stream on DELTA-hd/sdi/codec board handle. Result = 0x%08X\n",Result);
                goto no_stream;
        }
        /* Get auto-detected video standard */                     
        //Result = VHD_GetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD,&VideoStandard);

        //if ((Result == VHDERR_NOERROR) && (VideoStandard != NB_VHD_VIDEOSTANDARDS))
        //{
        
        /* Configure stream */
        VHD_SetStreamProperty(s->StreamHandle,VHD_SDI_SP_VIDEO_STANDARD, s->mode);
        for (i = 0; i < deltacast_frame_modes_count; ++i)
        {
                if(s->mode == deltacast_frame_modes[i].mode) {
                        s->frame->fps = deltacast_frame_modes[i].fps;
                        s->frame->interlacing = deltacast_frame_modes[i].interlacing;
                        s->tile->width = deltacast_frame_modes[i].width;
                        s->tile->height = deltacast_frame_modes[i].height;
                        printf("[DELTACAST] %s mode selected.\n", deltacast_frame_modes[i].name);
                        break;
                }
        }
        if(i == deltacast_frame_modes_count) {
                fprintf(stderr, "[DELTACAST] Failed to obtain information about video format %d.\n", s->mode);
                goto no_format;
        }
        /*} else {
                fprintf(stderr, "ERROR : Cannot detect incoming video standard from RX0. Result = 0x%08X\n",Result);
                goto no_format;
        }*/
        
        
        Result = VHD_GetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFER_PACKING, &Packing);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to get pixel format\n");
                goto no_pixfmt;
        }
        if(s->frame->color_spec == v210)
                Packing = VHD_BUFPACK_VIDEO_YUV422_10;
        if(s->frame->color_spec == UYVY)
                Packing = VHD_BUFPACK_VIDEO_YUV422_8;
        Result = VHD_SetStreamProperty(s->StreamHandle, VHD_CORE_SP_BUFFER_PACKING, Packing);
        if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "[DELTACAST] Unable to set pixel format\n");
                goto no_pixfmt;
        }
        

        /* Start stream */
        Result = VHD_StartStream(s->StreamHandle);
        if (Result == VHDERR_NOERROR){
                printf("Stream started");
        } else {
                fprintf(stderr, "ERROR : Cannot start RX0 stream on DELTA-hd/sdi/codec board handle. Result = 0x%08X\n",Result);
                goto stream_failed;
        }

	return s;
stream_failed:
no_pixfmt:
no_format:
        /* Close stream handle */
        VHD_CloseStreamHandle(s->StreamHandle);
no_stream:
        
no_clock:
        /* Re-establish RX0-TX0 by-pass relay loopthrough */
        VHD_SetBoardProperty(s->BoardHandle,VHD_CORE_BP_BYPASS_RELAY_0,TRUE);
bad_channel:
        VHD_CloseBoardHandle(s->BoardHandle);
error:
        free(s);
        return NULL;
}

void
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
        free(s);
}

struct video_frame *
vidcap_deltacast_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_deltacast_state   *s = (struct vidcap_deltacast_state *) state;
        
        ULONG             /*SlotsCount, SlotsDropped,*/BufferSize;
        ULONG             Result;
        BYTE             *pBuffer=NULL;

        *audio = NULL;
        /* Unlock slot */
        if(s->SlotHandle)
                VHD_UnlockSlotHandle(s->SlotHandle);
        Result = VHD_LockSlotHandle(s->StreamHandle,&s->SlotHandle);
        if (Result != VHDERR_NOERROR) {
                if (Result != VHDERR_TIMEOUT) {
                        fprintf(stderr, "ERROR : Cannot lock slot on RX0 stream. Result = 0x%08X\n",Result);
                }
                else {
                        fprintf(stderr, "Timeout \n");
                }
                return NULL;
        }
        
         Result = VHD_GetSlotBuffer(s->SlotHandle, VHD_SDI_BT_VIDEO, &pBuffer, &BufferSize);
         
         if (Result != VHDERR_NOERROR) {
                fprintf(stderr, "\nERROR : Cannot get slot buffer. Result = 0x%08X\n",Result);
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
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }
        s->frames++;
        
	return s->frame;
}

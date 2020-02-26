/**
 * @file   bluefish444_common.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2020 CESNET, z. s. p. o.
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

#ifndef BLUEFISH444_COMMON_H
#define BLUEFISH444_COMMON_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"

#ifdef WIN32
#include <objbase.h>
#include <BlueVelvetC_UltraGrid.h>
#else
#include <BlueVelvetC.h>
#include <unistd.h>
#endif
#include <BlueVelvetCUtils.h>

#define HAVE_BLUE_AUDIO 1

#include <video.h>

#define MAX_HANC_SIZE (256*1024)

struct bluefish_frame_mode_t {
        unsigned long int  mode;
        unsigned int       width;
        unsigned int       height;
        double             fps;
        enum interlacing_t interlacing;
};

static const struct bluefish_frame_mode_t bluefish_frame_modes[] = {
        { VID_FMT_PAL, 720, 576, 25.0, INTERLACED_MERGED },
        // VID_FMT_576I_5000=0,    /**< 720  x 576  50       Interlaced */
        { VID_FMT_NTSC, 720, 486, 29.97, INTERLACED_MERGED },
        // VID_FMT_486I_5994=1,    /**< 720  x 486  60/1.001 Interlaced */
        { VID_FMT_720P_5994, 1280, 720, 59.97, PROGRESSIVE },              /**< 1280 x 720  60/1.001 Progressive */
        { VID_FMT_720P_6000, 1280, 720, 60, PROGRESSIVE },        /**< 1280 x 720  60       Progressive */
        { VID_FMT_1080PSF_2397, 1920, 1080, 23.98, SEGMENTED_FRAME },  /**< 1920 x 1080 24/1.001 Segment Frame */
        { VID_FMT_1080PSF_2400, 1920, 1080, 24, SEGMENTED_FRAME },  /**< 1920 x 1080 24       Segment Frame */
        { VID_FMT_1080P_2397, 1920, 1080, 23.98, PROGRESSIVE },             /**< 1920 x 1080 24/1.001 Progressive */
        { VID_FMT_1080P_2400, 1920, 1080, 24, PROGRESSIVE },             /**< 1920 x 1080 24       Progressive */
        { VID_FMT_1080I_5000, 1920, 1080, 25, INTERLACED_MERGED },       /**< 1920 x 1080 50       Interlaced */
        { VID_FMT_1080I_5994, 1920, 1080, 29.97, INTERLACED_MERGED },    /**< 1920 x 1080 60/1.001 Interlaced */
        { VID_FMT_1080I_6000, 1920, 1080, 30, INTERLACED_MERGED },       /**< 1920 x 1080 60       Interlaced */
        { VID_FMT_1080P_2500, 1920, 1080, 25, PROGRESSIVE },             /**< 1920 x 1080 25       Progressive */
        { VID_FMT_1080P_2997, 1920, 1080, 29.97, PROGRESSIVE },          /**< 1920 x 1080 30/1.001 Progressive */
        { VID_FMT_1080P_3000, 1920, 1080, 30, PROGRESSIVE },             /**< 1920 x 1080 30       Progressive */
        { VID_FMT_HSDL_1498, 2048, 1556, 14.98, SEGMENTED_FRAME },       /**< 2048 x 1556 15/1.0   Segment Frame */
        { VID_FMT_HSDL_1500, 2048, 1556, 15, SEGMENTED_FRAME },  /**< 2048 x 1556 15   Segment Frame */
        { VID_FMT_720P_5000, 1280, 720, 50, PROGRESSIVE }, /**< 1280 x 720  50 Progressive */
        { VID_FMT_720P_2398, 1280, 720, 23.98, PROGRESSIVE }, /**< 1280 x 720  24/1.001       Progressive */
        { VID_FMT_720P_2400, 1280, 720, 24, PROGRESSIVE },       /**< 1280 x 720  24     Progressive */
        { VID_FMT_2048_1080PSF_2397, 2048, 1080, 23.98, SEGMENTED_FRAME }, /**< 2048 x 1080 24/1.001 Segment Frame */
        { VID_FMT_2048_1080PSF_2400, 2048, 1080, 24, SEGMENTED_FRAME }, /**< 2048 x 1080 24 Segment Frame */
        { VID_FMT_2048_1080P_2397, 2048, 1080, 23.98, PROGRESSIVE }, /**< 2048 x 1080 24/1.001 progressive */ 
        { VID_FMT_2048_1080P_2400, 2048, 1080, 24, PROGRESSIVE }, /**< 2048 x 1080 24 progressive  */
        { VID_FMT_1080PSF_2500, 1920, 1080, 25, SEGMENTED_FRAME },
        { VID_FMT_1080PSF_2997, 1920, 1080, 29.97, SEGMENTED_FRAME },
        { VID_FMT_1080PSF_3000, 1920, 1080, 30, SEGMENTED_FRAME },
        { VID_FMT_1080P_5000, 1920, 1080, 50, PROGRESSIVE },
        { VID_FMT_1080P_5994, 1920, 1080, 59.94, PROGRESSIVE },
        { VID_FMT_1080P_6000, 1920, 1080, 60, PROGRESSIVE },
        { VID_FMT_720P_2500, 1280, 720, 25, PROGRESSIVE },
        { VID_FMT_720P_2997, 1280, 720, 29.97, PROGRESSIVE },
        { VID_FMT_720P_3000, 1280, 720, 30, PROGRESSIVE },
        // VID_FMT_DVB_ASI=32,
        { VID_FMT_2048_1080PSF_2500, 2048, 1080, 25, SEGMENTED_FRAME },
        { VID_FMT_2048_1080PSF_2997, 2048, 1080, 29.97, SEGMENTED_FRAME },
        { VID_FMT_2048_1080PSF_3000, 2048, 1080, 30, SEGMENTED_FRAME },
        { VID_FMT_2048_1080P_2500, 2048, 1080, 25, PROGRESSIVE },
        { VID_FMT_2048_1080P_2997, 2048, 1080, 29.97, PROGRESSIVE },
        { VID_FMT_2048_1080P_3000, 2048, 1080, 30, PROGRESSIVE },
        { VID_FMT_2048_1080P_5000, 2048, 1080, 50, PROGRESSIVE }, 
        { VID_FMT_2048_1080P_5994, 2048, 1080, 59.97, PROGRESSIVE },
        { VID_FMT_2048_1080P_6000, 2048, 1080, 60, PROGRESSIVE },
        { VID_FMT_INVALID, 0, 0, 0, (interlacing_t) 0 }
};

static const int bluefish_frame_modes_count = sizeof(bluefish_frame_modes) /
        sizeof(struct bluefish_frame_mode_t);

static uint32_t GetNumberOfAudioSamplesPerFrame(uint32_t VideoMode, uint32_t FrameNumber)
        __attribute__((unused));

// from BlueFish SDK
static uint32_t GetNumberOfAudioSamplesPerFrame(uint32_t VideoMode, uint32_t FrameNumber)
{
        uint32_t NTSC_frame_seq[]={       1602,1601,1602,1601,1602};
        uint32_t p59_frame_seq[]={        801,800,801,801,801,801,800,801,801,801};
#if 0
        uint32_t p23_frame_seq[]={        2002,2002,2002,2002};
        uint32_t NTSC_frame_offset[]={0,
                                                                1602,
                                                                1602+1601,
                                                                1602+1601+1602,
                                                                1602+1601+1602+1601};

        uint32_t p59_frame_offset[]={     0,
                                                                801,
                                                                801+800,
                                                                801+800+801,
                                                                801+800+801+801,
                                                                801+800+801+801+801,
                                                                801+800+801+801+801+801,
                                                                801+800+801+801+801+801+800,
                                                                801+800+801+801+801+801+800+801,
                                                                801+800+801+801+801+801+800+801+801};

        uint32_t p23_frame_offset[]={     0,
                                                                2002,
                                                                2002+2002,
                                                                2002+2002+2002};
#endif

        switch(VideoMode)
        {
        case VID_FMT_720P_2398:
        case VID_FMT_1080P_2397:
        case VID_FMT_1080PSF_2397:
        case VID_FMT_2048_1080PSF_2397:
        case VID_FMT_2048_1080P_2397:
                return 2002;
        case VID_FMT_NTSC:
        case VID_FMT_720P_2997:
        case VID_FMT_1080P_2997:
        case VID_FMT_1080PSF_2997:
        case VID_FMT_1080I_5994:
        case VID_FMT_2048_1080PSF_2997:
        case VID_FMT_2048_1080P_2997:
                return NTSC_frame_seq[FrameNumber%5];
                break;
        case VID_FMT_720P_5994:
        case VID_FMT_1080P_5994:
                return p59_frame_seq[FrameNumber%10];
                break;
        case VID_FMT_720P_2400:
        case VID_FMT_1080PSF_2400:
        case VID_FMT_1080P_2400:
        case VID_FMT_2048_1080PSF_2400:
        case VID_FMT_2048_1080P_2400:
                return 2000;
                break;
        case VID_FMT_720P_3000:
        case VID_FMT_1080I_6000:
        case VID_FMT_1080P_3000:
        case VID_FMT_1080PSF_3000:
        case VID_FMT_2048_1080PSF_3000:
        case VID_FMT_2048_1080P_3000:
                return 1600;
                break;
        case VID_FMT_720P_6000:
        case VID_FMT_1080P_6000:
                return 800;
                break;
        case VID_FMT_720P_5000:
        case VID_FMT_1080P_5000:
                return 960;
                break;
        case VID_FMT_PAL:
        case VID_FMT_1080I_5000:
        case VID_FMT_1080PSF_2500:
        case VID_FMT_2048_1080PSF_2500:
                return 1920;
                break;
        case VID_FMT_720P_2500:
        case VID_FMT_1080P_2500:
        case VID_FMT_2048_1080P_2500:
        default:
                return 1920;
                break;
        }
}

#endif // defined BLUEFISH444_COMMON_H


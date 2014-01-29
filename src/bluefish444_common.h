/*
 * FILE:    bluefish444_common.h
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
#include <BlueVelvet.h>
#include <malloc.h>
#include <unistd.h>
#endif

#ifdef HAVE_BLUE_AUDIO
#include<BlueHancUtils.h>
#endif

#include <video.h>

#define MAX_HANC_SIZE (256*1024)

#ifdef WIN32
#define CBLUEVELVET_H BLUEVELVETC_HANDLE
#else
#define bfcFactory BlueVelvetFactory
#define CBLUEVELVET_H CBlueVelvet *
#endif 

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

static void *page_aligned_alloc(size_t size) __attribute__((unused));
static void page_aligned_free(void *ptr) __attribute__((unused));
static uint32_t GetNumberOfAudioSamplesPerFrame(uint32_t VideoMode, uint32_t FrameNumber)
        __attribute__((unused));

#ifdef HAVE_LINUX
typedef void OVERLAPPED;

static void bfcDestroy(CBLUEVELVET_H pSDK) __attribute__((unused));
static int bfcQueryCardProperty32(CBLUEVELVET_H pSDK, int property, uint32_t &value) __attribute__((unused));
static int bfcSetCardProperty32(CBLUEVELVET_H pSDK, int property, uint32_t &value) __attribute__((unused));
static int bfcEnumerate(CBLUEVELVET_H pSDK, int &iDevices) __attribute__((unused));
static int bfcAttach(CBLUEVELVET_H pSDK, int &iDeviceId) __attribute__((unused));
static int bfcDetach(CBLUEVELVET_H pSDK) __attribute__((unused));
static int bfcVideoCaptureStart(CBLUEVELVET_H pSDK) __attribute__((unused));
static int bfcVideoCaptureStop(CBLUEVELVET_H pSDK) __attribute__((unused));
static int bfcVideoPlaybackStart(CBLUEVELVET_H pSDK, int iStep, int iLoop) __attribute__((unused));
static int bfcVideoPlaybackStop(CBLUEVELVET_H pSDK, int iWait, int iFlush) __attribute__((unused));
static int bfcWaitVideoInputSync(CBLUEVELVET_H pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount)
        __attribute__((unused));
static int bfcWaitVideoOutputSync(CBLUEVELVET_H pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount)
        __attribute__((unused));
static int bfcQueryCardType(CBLUEVELVET_H pSDK) __attribute__((unused));
#ifdef HAVE_BLUE_AUDIO
static int bfcDecodeHancFrameEx(CBLUEVELVET_H pHandle, unsigned int nCardType, unsigned int* pHancBuffer, struct hanc_decode_struct* pHancDecodeInfo) __attribute__((unused));
static int bfcEncodeHancFrameEx(CBLUEVELVET_H pHandle, unsigned int nCardType, struct hanc_stream_info_struct* pHancEncodeInfo, void *pAudioBuffer, unsigned int nAudioChannels, unsigned int nAudioSamples, unsigned int nSampleType, unsigned int nAudioFlags) __attribute__((unused));
#endif
static int bfcSystemBufferReadAsync(CBLUEVELVET_H pHandle, unsigned char* pPixels, unsigned long ulSize, OVERLAPPED* pOverlap, unsigned long ulBufferID, unsigned long ulOffset=0) __attribute__((unused));
static int bfcSystemBufferWriteAsync(CBLUEVELVET_H pHandle, unsigned char *pPixels, unsigned long ulSize, OVERLAPPED *pOverlap, unsigned long ulBufferID, unsigned long ulOFfset=0) __attribute__((unused));
static int bfcRenderBufferUpdate(CBLUEVELVET_H pHandle, unsigned long ulBufferID) __attribute__((unused));
static int bfcRenderBufferCapture(CBLUEVELVET_H pHandle, unsigned long ulBufferID) __attribute__((unused));

static void bfcDestroy(CBLUEVELVET_H pSDK)
{
        delete pSDK;
}

static int bfcQueryCardProperty32(CBLUEVELVET_H pSDK, int property, uint32_t &value)
{
        BlueFishParamValue varVal;
        varVal.vt = BLUE_PARAM_ULONG_32BIT;

        BErr err = pSDK->QueryCardProperty(property, varVal);
        value = varVal.int32;

        return err;
}

static int bfcSetCardProperty32(CBLUEVELVET_H pSDK, int property, uint32_t &value)
{
        BlueFishParamValue varVal;
        varVal.vt = BLUE_PARAM_ULONG_32BIT;
        varVal.int32 = value;

        BErr err = pSDK->SetCardProperty(property, varVal);

        return err;
}

static int bfcEnumerate(CBLUEVELVET_H pSDK, int &iDevices)
{
        return pSDK->device_enumerate(iDevices);
}

static int bfcAttach(CBLUEVELVET_H pSDK, int &iDeviceId)
{
        return pSDK->device_attach(iDeviceId, 0);
}

static int bfcDetach(CBLUEVELVET_H pSDK)
{
        return pSDK->device_detach();
}

static int bfcVideoCaptureStart(CBLUEVELVET_H pSDK)
{
        return pSDK->video_capture_start(0);
}

static int bfcVideoCaptureStop(CBLUEVELVET_H pSDK)
{
        return pSDK->video_capture_stop();
}

static int bfcVideoPlaybackStart(CBLUEVELVET_H pSDK, int iStep, int iLoop)
{
	return pSDK->video_playback_start(iStep, iLoop);
}

static int bfcVideoPlaybackStop(CBLUEVELVET_H pSDK, int iWait, int iFlush)
{
        return pSDK->video_playback_stop(iWait, iFlush);
}

static int bfcWaitVideoInputSync(CBLUEVELVET_H pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount) {
        return pSDK->wait_input_video_synch(ulUpdateType, ulFieldCount);
}

static int bfcWaitVideoOutputSync(CBLUEVELVET_H pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount) {
        return pSDK->wait_output_video_synch(ulUpdateType, ulFieldCount);
}

static int bfcQueryCardType(CBLUEVELVET_H pSDK)
{
        return pSDK->has_video_cardtype();
};

#ifdef HAVE_BLUE_AUDIO
static int bfcDecodeHancFrameEx(CBLUEVELVET_H, unsigned int nCardType, unsigned int* pHancBuffer, struct hanc_decode_struct* pHancDecodeInfo)
{
        return hanc_decoder_ex(nCardType, pHancBuffer, pHancDecodeInfo);
}

static int bfcEncodeHancFrameEx(CBLUEVELVET_H, unsigned int nCardType, struct hanc_stream_info_struct* pHancEncodeInfo, void *pAudioBuffer, unsigned int nAudioChannels, unsigned int nAudioSamples, unsigned int nSampleType, unsigned int nAudioFlags)
{
        return encode_hanc_frame_ex(nCardType, pHancEncodeInfo, pAudioBuffer, nAudioChannels, nAudioSamples, nSampleType, nAudioFlags);
}
#endif // HAVE_BLUE_AUDIO

static int bfcSystemBufferReadAsync(CBLUEVELVET_H pHandle, unsigned char* pPixels, unsigned long ulSize, OVERLAPPED*, unsigned long ulBufferID, unsigned long ulOffset)
{
        return pHandle->dma_read((char *) pPixels, ulSize, ulBufferID, ulOffset);
}

static int bfcSystemBufferWriteAsync(CBLUEVELVET_H pHandle, unsigned char *pPixels, unsigned long ulSize, OVERLAPPED *, unsigned long ulBufferID, unsigned long ulOffset)
{
        return pHandle->dma_write((char *) pPixels, ulSize, ulBufferID, ulOffset);
}

static int bfcRenderBufferUpdate(CBLUEVELVET_H pHandle, unsigned long ulBufferID)
{
        return pHandle->render_buffer_update(ulBufferID, BLUE_CARDBUFFER_IMAGE);
}

static int bfcRenderBufferCapture(CBLUEVELVET_H pHandle, unsigned long ulBufferID)
{
        return pHandle->render_buffer_capture(ulBufferID, 0);
}

#endif

static void *page_aligned_alloc(size_t size)
{
#ifdef WIN32
        return VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);
#else
        return memalign(sysconf(_SC_PAGE_SIZE), size);
#endif
}

static void page_aligned_free(void *ptr)
{
#ifdef WIN32
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        free(ptr);
#endif
}

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


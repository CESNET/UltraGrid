/*
 * FILE:    video_display/quicktime.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "host.h"
#include "tv.h"
#include "video_codec.h"

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>

#include "compat/platform_semaphore.h"
#include <signal.h>
#include <pthread.h>
#include <assert.h>

#include "video_display.h"
#include "video_display/quicktime.h"

#define MAGIC_QT_DISPLAY 	DISPLAY_QUICKTIME_ID

#define MAX_DEVICES     4

const quicktime_mode_t quicktime_modes[] = {
        {"AJA   8-bit Digitizer", "AJA 525i23.98         8 Bit  (720x486)", 720, 486, 23.98, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 525i29.97         8 Bit  (720x486)", 720, 486, 29.97, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 625i25            8 Bit  (720x576)", 720, 576, 25, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p50            8 Bit  (1280x720)", 1280, 720, 50, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p50            8 Bit  (1280x720) VFR", 1280, 720, 50, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p23.98         8 Bit  (1280x720)", 1280, 720, 23.98, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p59.94         8 Bit  (1280x720)", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p59.94         8 Bit  (1280x720) VFR", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p60            8 Bit  (1280x720)", 1280, 720, 60, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 720p60            8 Bit  (1280x720) VFR", 1280, 720, 60, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i25   8 Bit  (1920x1080)", 1920, 1080, 25, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i25   8 Bit  (1920x1080) DBL", 1920, 1080, 25, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i29.97        8 Bit  (1920x1080)", 1920, 1080, 29.97, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i29.97        8 Bit  (1920x1080) DBL", 1920, 1080, 29.97, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i29.97        8 Bit  (1920x1080) VFR", 1920, 1080, 29.97, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i30   8 Bit  (1920x1080)", 1920, 1080, 30, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080i30   8 Bit  (1920x1080) VFR", 1920, 1080, 30, AUX_INTERLACED|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080sf23.98 8 Bit  (1920x1080) DBL", 1920, 1080, 23.98, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080sf23.98 8 Bit  (1920x1080)", 1920, 1080, 23.98, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080sf24  8 Bit  (1920x1080)", 1920, 1080, 24, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p23.98        8 Bit  (1920x1080)", 1920, 1080, 23.98, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p24   8 Bit  (1920x1080)", 1920, 1080, 24, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p25   8 Bit  (1920x1080)", 1920, 1080, 25, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p29.97        8 Bit  (1920x1080)", 1920, 1080, 29.97, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p30   8 Bit  (1920x1080)", 1920, 1080, 30, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p50b  8 Bit  (1920x1080)", 1920, 1080, 50, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p59.94b 8 Bit  (1920x1080)", 1920, 1080, 59.94, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080p60b  8 Bit  (1920x1080)", 1920, 1080, 60, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080x2Ksf23.98    8 Bit  (2048x1080)", 2048, 1080, 23.98, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080x2Ksf24       8 Bit  (2048x1080)", 2048, 1080, 24, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080x2Kp23.98     8 Bit  (2048x1080)", 2048, 1080, 23.98, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1080x2Kp24                8 Bit  (2048x1080)", 2048, 1080, 24, AUX_PROGRESSIVE|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1556x2Ksf14.98    8 Bit  (2048x1556)", 2048, 1556, 14.98, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA   8-bit Digitizer", "AJA 1556x2Ksf15       8 Bit  (2048x1556)", 2048, 1556, 15, AUX_SF|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 525i23.98  10 Bit  (720x486)", 720, 486, 23.98, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 525i29.97  10 Bit  (720x486)", 720, 486, 29.97, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 625i25             10 Bit  (720x576)", 720, 576, 25, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p50             10 Bit  (1280x720)", 1280, 720, 50, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p50             10 Bit  (1280x720) VFR", 1280, 720, 50, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p23.98  10 Bit  (1280x720)", 1280, 720, 23.98, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p59.94  10 Bit  (1280x720)", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p59.94  10 Bit  (1280x720) VFR", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p60             10 Bit  (1280x720)", 1280, 720, 60, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 720p60             10 Bit  (1280x720) VFR", 1280, 720, 60, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i25    10 Bit  (1920x1080)", 1920, 1080, 25, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i25    10 Bit  (1920x1080) DBL", 1920, 1080, 25, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i29.97         10 Bit  (1920x1080)", 1920, 1080, 29.97, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i29.97         10 Bit  (1920x1080) DBL", 1920, 1080, 29.97, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i29.97         10 Bit  (1920x1080) VFR", 1920, 1080, 29.97, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i30    10 Bit  (1920x1080)", 1920, 1080, 30, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080i30    10 Bit  (1920x1080) VFR", 1920, 1080, 30, AUX_INTERLACED|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080sf23.98 10 Bit  (1920x1080) DBL", 1920, 1080, 23.98, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080sf23.98 10 Bit  (1920x1080)", 1920, 1080, 23.98, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080sf24   10 Bit  (1920x1080)", 1920, 1080, 24, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p23.98         10 Bit  (1920x1080)", 1920, 1080, 23.98, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p24    10 Bit  (1920x1080)", 1920, 1080, 24, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p25    10 Bit  (1920x1080)", 1920, 1080, 25, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p29.97         10 Bit  (1920x1080)", 1920, 1080, 29.97, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p30    10 Bit  (1920x1080)", 1920, 1080, 30, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p50b   10 Bit  (1920x1080)", 1920, 1080, 50, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p59.94b 10 Bit  (1920x1080)", 1920, 1080, 59.94, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080p60b   10 Bit  (1920x1080)", 1920, 1080, 60, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080x2Ksf23.98     10 Bit  (2048x1080)", 2048, 1080, 23.98, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080x2Ksf24        10 Bit  (2048x1080)", 2048, 1080, 24, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080x2Kp23.98      10 Bit  (2048x1080)", 2048, 1080, 23.98, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1080x2Kp24                 10 Bit  (2048x1080)", 2048, 1080, 24, AUX_PROGRESSIVE|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1556x2Ksf14.98     10 Bit  (2048x1556)", 2048, 1556, 14.98, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"AJA 10-bit Digitizer", "AJA 1556x2Ksf15        10 Bit  (2048x1556)", 2048, 1556, 15, AUX_SF|AUX_10Bit|AUX_RGB|AUX_YUV},
        {"Blackmagic 2K", "Blackmagic 2K 23.976 - RGB", 2048, 1556, 23.98, AUX_PROGRESSIVE|AUX_RGB},
        {"Blackmagic 2K", "Blackmagic 2K 24 - RGB", 2048, 1556, 24, AUX_PROGRESSIVE|AUX_RGB},
        {"Blackmagic 2K", "Blackmagic 2K 25 - RGB", 2048, 1556, 25, AUX_PROGRESSIVE|AUX_RGB},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 23.976 - 8 Bit", 1920, 1080, 23.976, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 23.976 - 10 Bit", 1920, 1080, 23.976, AUX_PROGRESSIVE|AUX_10Bit},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 24 - 8 Bit", 1920, 1080, 24, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 24 - 10 Bit", 1920, 1080, 24, AUX_PROGRESSIVE|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 50 - 8 Bit", 1920, 1080, 50, AUX_INTERLACED|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 50 - 10 Bit", 1920, 1080, 50, AUX_INTERLACED|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 59.94 - 8 Bit", 1920, 1080, 59.94, AUX_INTERLACED|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 59.94 - 10 Bit", 1920, 1080, 59.94, AUX_INTERLACED|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 60 - 8 Bit", 1920, 1080, 60, AUX_INTERLACED|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 60 - 10 Bit", 1920, 1080, 60, AUX_INTERLACED|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 23.976 - RGB 10 Bit", 1920, 1080, 23.976, AUX_PROGRESSIVE|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 1080", "Blackmagic HD 1080p 24 - RGB 10 Bit", 1920, 1080, 24, AUX_PROGRESSIVE|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 50 - RGB 10 Bit", 1920, 1080, 50, AUX_INTERLACED|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 59.94 - RGB 10 Bit", 1920, 1080, 59.94, AUX_INTERLACED|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 1080", "Blackmagic HD 1080i 60 - RGB 10 Bit", 1920, 1080, 60, AUX_INTERLACED|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 720", "Blackmagic HD 720p 50 - 8 Bit", 1280, 720, 50, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 50 - 10 Bit", 1280, 720, 50, AUX_PROGRESSIVE|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 59.94 - 8 Bit", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 59.94 - 10 Bit", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 60 - 8 Bit", 1280, 720, 60, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 60 - 10 Bit", 1280, 720, 60, AUX_PROGRESSIVE|AUX_10Bit|AUX_YUV},
        {"Blackmagic HD 720", "Blackmagic HD 720p 50 - RGB 10 Bit", 1280, 720, 50, AUX_PROGRESSIVE|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 720", "Blackmagic HD 720p 59.94 - RGB 10 Bit", 1280, 720, 59.94, AUX_PROGRESSIVE|AUX_RGB|AUX_10Bit},
        {"Blackmagic HD 720", "Blackmagic HD 720p 60 - RGB 10 Bit", 1280, 720, 60, AUX_PROGRESSIVE|AUX_RGB|AUX_10Bit},
        {"Blackmagic NTSC/PAL", "Blackmagic NTSC/PAL - 8 Bit", 720, 486, 24, AUX_PROGRESSIVE|AUX_YUV},
        {"Blackmagic NTSC/PAL", "Blackmagic NTSC/PAL - 10 Bit", 720, 486, 24, AUX_PROGRESSIVE|AUX_10Bit|AUX_YUV},
        {NULL, NULL, 0, 0, 0, 0},
};        

struct state_quicktime {
        ComponentInstance videoDisplayComponentInstance[MAX_DEVICES];
//    Component                 videoDisplayComponent;
        GWorldPtr gworld[MAX_DEVICES];
        ImageSequence seqID[MAX_DEVICES];
        int device[MAX_DEVICES];
        
        int devices_cnt;
        const struct codec_info_t *cinfo;

        /* Thread related information follows... */
        pthread_t thread_id;
        sem_t semaphore;

        struct video_frame frame[MAX_DEVICES];

        uint32_t magic;
};

/* Prototyping */
char *four_char_decode(int format);
void qt_reconfigure_screen(void *state, unsigned int width, unsigned int height,
		codec_t codec, double fps, int aux);
static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out);
static int find_mode(ComponentInstance *ci, int width, int height, 
                const char * codec_name, double fps, int aux);

static void
nprintf(unsigned char *str)
{
        char tmp[((int)str[0]) + 1];

        strncpy(tmp, (char*)(&str[1]), str[0]);
        tmp[(int)str[0]] = 0;
        fprintf(stdout, "%s", tmp);
}


char *four_char_decode(int format)
{
        static char fbuf0[32];
        static char fbuf1[32];
        static int count = 0;
        char *fbuf;

        if (count & 1)
                fbuf = fbuf1;
        else
                fbuf = fbuf0;
        count++;

        if ((unsigned)format < 64)
                sprintf(fbuf, "%d", format);
        else {
                fbuf[0] = (char)(format >> 24);
                fbuf[1] = (char)(format >> 16);
                fbuf[2] = (char)(format >> 8);
                fbuf[3] = (char)(format >> 0);
        }
        return fbuf;
}

static void reconf_common(struct state_quicktime *s)
{
        int i;
        
        for (i = 0; i < s->devices_cnt; ++i)
        {
                int h_align;
                ImageDescriptionHandle imageDesc;
                OSErr ret;
        
                imageDesc =
                    (ImageDescriptionHandle) NewHandle(sizeof(ImageDescription));
        
                (**(ImageDescriptionHandle) imageDesc).idSize =
                    sizeof(ImageDescription);
                (**(ImageDescriptionHandle) imageDesc).cType = s->cinfo->fcc;
                /* 
                 * dataSize is specified in bytes and is specified as 
                 * height*width*bytes_per_luma_instant. v210 sets 
                 * bytes_per_luma_instant to 8/3. 
                 * See http://developer.apple.com/quicktime/icefloe/dispatch019.html#v210
                 */       
                (**(ImageDescriptionHandle) imageDesc).dataSize = s->frame[i].data_len;
                /* 
                 * Beware: must be a multiple of horiz_align_pixels which is 2 for 2Vuy
                 * and 48 for v210. hd_size_x=1920 is a multiple of both. TODO: needs 
                 * further investigation for 2K!
                 */
                h_align = s->frame[i].width / s->frame[i].tile_info.x_count;
                if(s->cinfo->h_align) {
                        h_align = ((h_align + s->cinfo->h_align - 1) / s->cinfo->h_align) * s->cinfo->h_align;
                }
                (**(ImageDescriptionHandle) imageDesc).width = h_align;
                (**(ImageDescriptionHandle) imageDesc).height = s->frame[i].height / s->frame[i].tile_info.y_count;
        
                ret = DecompressSequenceBeginS(&(s->seqID[i]), imageDesc, s->frame[i].data, 
                                               // Size of the buffer, not size of the actual frame data inside
                                               s->frame[i].data_len,    
                                               s->gworld[i],
                                               NULL,
                                               NULL,
                                               NULL,
                                               srcCopy,
                                               NULL,
                                               (CodecFlags) NULL,
                                               codecNormalQuality, bestSpeedCodec);
                if (ret != noErr)
                        fprintf(stderr, "Failed DecompressSequenceBeginS\n");
                DisposeHandle((Handle) imageDesc);
        }
}

void display_quicktime_run(void *arg)
{
        struct state_quicktime *s = (struct state_quicktime *)arg;

        CodecFlags ignore;
        OSErr ret;

        int frames = 0;
        struct timeval t, t0;

        while (!should_exit) {
                int i;
                platform_sem_wait(&s->semaphore);

                for (i = 0; i < s->devices_cnt; ++i) {
                        /* TODO: Running DecompressSequenceFrameWhen asynchronously 
                         * in this way introduces a possible race condition! 
                         */
                        ret = DecompressSequenceFrameWhen(s->seqID[i], s->frame[i].data, 
                                                        s->frame[i].data_len,
                                                       /* If you set asyncCompletionProc to -1, 
                                                        *  the operation is performed asynchronously but 
                                                        * the decompressor does not call the completion 
                                                        * function. 
                                                        */
                                                       0, &ignore, -1,       
                                                          NULL);
                        if (ret != noErr) {
                                fprintf(stderr,
                                        "Failed DecompressSequenceFrameWhen: %d\n",
                                        ret);
                        }
                }

                frames++;
                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, t0);
                if (seconds >= 5) {
                        float fps = frames / seconds;
                        fprintf(stderr, "%d frames in %g seconds = %g FPS\n",
                                frames, seconds, fps);
                        t0 = t;
                        frames = 0;
                }
        }
}

struct video_frame *
display_quicktime_getf(void *state)
{
        struct state_quicktime *s = (struct state_quicktime *)state;
        assert(s->magic == MAGIC_QT_DISPLAY);
        return &s->frame[0];
}

int display_quicktime_putf(void *state, char *frame)
{
        struct state_quicktime *s = (struct state_quicktime *)state;

        UNUSED(frame);
        assert(s->magic == MAGIC_QT_DISPLAY);

        /* ...and signal the worker */
        platform_sem_post(&s->semaphore);
        return 0;
}

static void print_modes(int fullhelp)
{
        ComponentDescription cd;
        Component c = 0;

        cd.componentType = QTVideoOutputComponentType;
        cd.componentSubType = 0;
        cd.componentManufacturer = 0;
        cd.componentFlags = 0;
        cd.componentFlagsMask = kQTVideoOutputDontDisplayToUser;

        //fprintf(stdout, "Number of Quicktime Vido Display components %d\n", CountComponents (&cd));

        fprintf(stdout, "Available playback devices:\n");
        /* Print relevant video output components */
        while ((c = FindNextComponent(c, &cd))) {
		ComponentDescription exportCD;

                Handle componentNameHandle = NewHandle(0);
                GetComponentInfo(c, &exportCD, componentNameHandle, NULL, NULL);
                HLock(componentNameHandle);
                char *cName = *componentNameHandle;

                fprintf(stdout, " Device %d: ", (int)c);
                nprintf(cName);
                fprintf(stdout, "\n");

                HUnlock(componentNameHandle);
                DisposeHandle(componentNameHandle);

                /* Get display modes of selected video output component */
                QTAtomContainer modeListAtomContainer = NULL;
                ComponentInstance videoDisplayComponentInstance;

                videoDisplayComponentInstance = OpenComponent(c);

                int ret =
                    QTVideoOutputGetDisplayModeList
                    (videoDisplayComponentInstance, &modeListAtomContainer);
                if (ret != noErr || modeListAtomContainer == NULL) {
                        fprintf(stdout, "\tNo output modes available\n");
                        CloseComponent(videoDisplayComponentInstance);
                        continue;
                }

                int i = 1;
                QTAtom atomDisplay = 0, nextAtomDisplay = 0;
                QTAtomType type;
                QTAtomID id;

                /* Print modes of current display component */
                while (i <
                       QTCountChildrenOfType(modeListAtomContainer,
                                             kParentAtomIsContainer,
                                             kQTVODisplayModeItem)) {

                        ret =
                            QTNextChildAnyType(modeListAtomContainer,
                                               kParentAtomIsContainer,
                                               atomDisplay, &nextAtomDisplay);
                        // Make sure its a display atom
                        ret =
                            QTGetAtomTypeAndID(modeListAtomContainer,
                                               nextAtomDisplay, &type, &id);
                        if (type != kQTVODisplayModeItem)
                                continue;

                        atomDisplay = nextAtomDisplay;

                        QTAtom atom;
                        long dataSize, *dataPtr;

                        /* Print component ID */
                        fprintf(stdout, "\t - %ld: ", id);

                        /* Print component name */
                        atom =
                            QTFindChildByID(modeListAtomContainer, atomDisplay,
                                            kQTVOName, 1, NULL);
                        ret =
                            QTGetAtomDataPtr(modeListAtomContainer, atom,
                                             &dataSize, (Ptr *) & dataPtr);
                        fprintf(stdout, "  %s; ", (char *)dataPtr);

                        //if (strcmp((char *)dataPtr, mode) == 0) {
                        //      displayMode = id;
                        //}

                        /* Print component other info */
                        atom =
                            QTFindChildByID(modeListAtomContainer, atomDisplay,
                                            kQTVODimensions, 1, NULL);
                        ret =
                            QTGetAtomDataPtr(modeListAtomContainer, atom,
                                             &dataSize, (Ptr *) & dataPtr);
                        fprintf(stdout, "%dx%d px\n",
                                (int)EndianS32_BtoN(dataPtr[0]),
                                (int)EndianS32_BtoN(dataPtr[1]));

                        /* Do not print codecs */
                        if (!fullhelp) {
                                i++;
                                continue;
                        }

                        /* Print supported pixel formats */
                        fprintf(stdout, "\t\t - Codec: ");
                        QTAtom decompressorsAtom;
                        int j = 1;
                        int codecsPerLine = 0;
                        while ((decompressorsAtom =
                                QTFindChildByIndex(modeListAtomContainer,
                                                   atomDisplay,
                                                   kQTVODecompressors, j,
                                                   NULL)) != 0) {
                                atom =
                                    QTFindChildByID(modeListAtomContainer,
                                                    decompressorsAtom,
                                                    kQTVODecompressorType, 1,
                                                    NULL);
                                ret =
                                    QTGetAtomDataPtr(modeListAtomContainer,
                                                     atom, &dataSize,
                                                     (Ptr *) & dataPtr);
                                if (!(codecsPerLine % 9)) {
                                        fprintf(stdout, "\n  \t\t\t");
                                        fprintf(stdout, "%s", (char *)dataPtr);
                                } else {
                                        fprintf(stdout, ", %s",
                                                (char *)dataPtr);
                                }
                                codecsPerLine++;

                                //atom = QTFindChildByID(modeListAtomContainer, decompressorsAtom, kQTVODecompressorComponent, 1, NULL);
                                //ret = QTGetAtomDataPtr(modeListAtomContainer, atom, &dataSize, (Ptr *)&dataPtr);
                                //fprintf(stdout, "%s\n", (char *)dataPtr);

                                j++;
                        }
                        fprintf(stdout, "\n\n");

                        i++;
                        CloseComponent(videoDisplayComponentInstance);
                }
                fprintf(stdout, "\n");
        }
}

static void show_help(int full)
{
        printf("Quicktime output options:\n");
        printf("\tdevice[:mode:codec] | help | fullhelp\n");
        print_modes(full);
}

void *display_quicktime_init(char *fmt)
{
        struct state_quicktime *s;
        int ret;
        int i;
	char *codec_name;
	int mode;

        /* Parse fmt input */
        s = (struct state_quicktime *)calloc(1, sizeof(struct state_quicktime));
        s->magic = MAGIC_QT_DISPLAY;

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
                        show_help(0);
                        free(s);
                        return NULL;
                }
                if (strcmp(fmt, "fullhelp") == 0) {
                        show_help(1);
                        free(s);
                        return NULL;
                }
                char *tmp = strdup(fmt);
                char *tok;

                tok = strtok(tmp, ":");
                if (tok == NULL) {
                        show_help(0);
                        free(s);
                        free(tmp);
                        return NULL;
                } else {
                        s->devices_cnt = 0;
                        char *tmp = strdup(tok);
                        char *item, *save_ptr = NULL;
                        
                        item = strtok_r(tmp, ",", &save_ptr);
                        do {
                                s->device[s->devices_cnt] = atoi(item);
                                s->devices_cnt++;
                                assert(s->devices_cnt <= MAX_DEVICES);
                        } while((item = strtok_r(NULL, ",", &save_ptr)));
                        
                        free(tmp);
                }
                
                tok = strtok(NULL, ":");
                if (tok == NULL) {
			mode = 0;
                } else {
			mode = atol(tok);
			tok = strtok(NULL, ":");
			if (tok == NULL) {
				show_help(0);
				free(s);
				free(tmp);
				return NULL;
			}
			codec_name = strdup(tok);
		}
        } else {
		show_help(0);
		free(s);
		return NULL;
	}


        for (i = 0; i < s->devices_cnt; ++i) {
                /* compute position */
                double x_cnt = sqrt(s->devices_cnt);
                
                s->videoDisplayComponentInstance[i] = 0;
                s->seqID[i] = 0;
                
                s->frame[i].tile_info.x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                s->frame[i].tile_info.y_count = s->devices_cnt / s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_x = i % s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_y = i / s->frame[i].tile_info.x_count;
        }

        InitCursor();
        EnterMovies();

	if(mode != 0) {
		for (i = 0; codec_info[i].name != NULL; i++) {
			if (strcmp(codec_name, codec_info[i].name) == 0) {
				s->cinfo = &codec_info[i];
			}
		}
		free(codec_name);

                for (i = 0; i < s->devices_cnt; ++i) {
                        /* Open device */
                        s->videoDisplayComponentInstance[i] = OpenComponent((Component) s->device[i]);
        
                        /* Set the display mode */
                        ret =
                            QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance[i],
                                                        mode);
                        if (ret != noErr) {
                                fprintf(stderr, "Failed to set video output display mode.\n");
                                return NULL;
                        }
        
                        /* We don't want to use the video output component instance echo port */
                        ret = QTVideoOutputSetEchoPort(s->videoDisplayComponentInstance[i], nil);
                        if (ret != noErr) {
                                fprintf(stderr, "Failed to set video output echo port.\n");
                                return NULL;
                        }
        
                        /* Register Ultragrid with instande of the video outpiut */
                        ret =
                            QTVideoOutputSetClientName(s->videoDisplayComponentInstance[i],
                                                       (ConstStr255Param) "Ultragrid");
                        if (ret != noErr) {
                                fprintf(stderr,
                                        "Failed to register Ultragrid with selected video output instance.\n");
                                return NULL;
                        }
        
                        /* Call QTVideoOutputBegin to gain exclusive access to the video output */
                        ret = QTVideoOutputBegin(s->videoDisplayComponentInstance[i]);
                        if (ret != noErr) {
                                fprintf(stderr,
                                        "Failed to get exclusive access to selected video output instance.\n");
                                return NULL;
                        }
        
                        /* Get a pointer to the gworld used by video output component */
                        ret =
                            QTVideoOutputGetGWorld(s->videoDisplayComponentInstance[i],
                                                   &s->gworld[i]);
                        if (ret != noErr) {
                                fprintf(stderr,
                                        "Failed to get selected video output instance GWorld.\n");
                                return NULL;
                        }
        
                        ImageDescriptionHandle gWorldImgDesc = NULL;
                        PixMapHandle gWorldPixmap = (PixMapHandle) GetGWorldPixMap(s->gworld[i]);
        
                        /* Determine width and height */
                        ret = MakeImageDescriptionForPixMap(gWorldPixmap, &gWorldImgDesc);
                        if (ret != noErr) {
                                fprintf(stderr, "Failed to determine width and height.\n");
                                return NULL;
                        }
                        s->frame[i].width = (**gWorldImgDesc).width;
                        s->frame[i].height = (**gWorldImgDesc).height;
                        s->frame[i].aux = 0;
        
                        int aligned_x=s->frame[i].width;
        
                        if (s->cinfo->h_align) {
                                aligned_x =
                                    ((aligned_x + s->cinfo->h_align -
                                      1) / s->cinfo->h_align) * s->cinfo->h_align;
                        }
        
                        s->frame[i].dst_bpp = s->cinfo->bpp;
                        s->frame[i].src_bpp = s->cinfo->bpp;
                        s->frame[i].dst_linesize = aligned_x * s->cinfo->bpp;
                        s->frame[i].dst_pitch = s->frame[i].dst_linesize;
                        s->frame[i].src_linesize = aligned_x * s->cinfo->bpp;
                        s->frame[i].color_spec = s->cinfo->codec;
                        s->frame[i].dst_x_offset = 0;
        
                        fprintf(stdout, "Selected mode: %d(%d)x%d, %fbpp\n", s->frame[i].width,
                                aligned_x, s->frame[i].height, s->cinfo->bpp);
        
                        s->frame[i].data_len = s->frame[i].dst_linesize * s->frame[i].height;
                        s->frame[i].data = calloc(s->frame[i].data_len, 1);
                }
		reconf_common(s);
	} else {
                for (i = 0; i < s->devices_cnt; ++i) {
                        s->frame[i].width = 0;
                        s->frame[i].height = 0;
                        s->frame[i].data = NULL;
                }
	}

        for (i = 0; i < s->devices_cnt; ++i) {
                s->frame[i].state = s;
                s->frame[i].reconfigure = (reconfigure_t) qt_reconfigure_screen;
                s->frame[i].get_sub_frame = (get_sub_frame_t) get_sub_frame;
                s->frame[i].decoder = (decoder_t)memcpy;
                
                /* update tiles width/h to represent whole frame's dimension */
                s->frame[i].width *= s->frame[i].tile_info.x_count;
                s->frame[i].height *= s->frame[i].tile_info.y_count;
        }

	platform_sem_init(&s->semaphore, 0, 0);

        /*if (pthread_create
            (&(s->thread_id), NULL, display_thread_quicktime, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/

        return (void *)s;
}

void display_quicktime_done(void *state)
{
        struct state_quicktime *s = (struct state_quicktime *)state;
        int ret;
        int i;

        assert(s->magic == MAGIC_QT_DISPLAY);
        for (i = 0; i < s->devices_cnt; ++i) {
                ret = QTVideoOutputEnd(s->videoDisplayComponentInstance[i]);
                if (ret != noErr) {
                        fprintf(stderr,
                                "Failed to release the video output component.\n");
                }
        
                ret = CloseComponent(s->videoDisplayComponentInstance[i]);
                if (ret != noErr) {
                        fprintf(stderr,
                                "Failed to close the video output component.\n");
                }
        
                DisposeGWorld(s->gworld[i]);
        }
}

display_type_t *display_quicktime_probe(void)
{
        display_type_t *dtype;

        dtype = malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_QUICKTIME_ID;
                dtype->name = "quicktime";
                dtype->description = "QuickTime display device";
        }
        return dtype;
}

void qt_reconfigure_screen(void *state, unsigned int width, unsigned int height,
		codec_t codec, double fps, int aux)
{
	struct state_quicktime *s = (struct state_quicktime *) state;
	int i;
        int ret;
	const char *codec_name;

        for (i = 0; codec_info[i].name != NULL; i++) {
                if (codec_info[i].codec == codec) {
                        s->cinfo = &codec_info[i];
			codec_name = s->cinfo->name;
                }
        }
	if(codec == UYVY || codec == DVS8) /* just aliases for 2vuy,
				            * but would confuse QT */
		codec_name = "2vuy";
         
        if(s->frame[0].data != NULL)
                display_quicktime_done(s);
                
        fprintf(stdout, "Selected mode: %dx%d, %fbpp\n", width,
                        height, s->cinfo->bpp);

        for(i = 0; i < s->devices_cnt; ++i) {
                int tile_width, tile_height;
                /* compute position */
                double x_cnt = sqrt(s->devices_cnt);
                s->frame[i].tile_info.x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                s->frame[i].tile_info.y_count = s->devices_cnt / s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_x = i % s->frame[i].tile_info.x_count;
                s->frame[i].tile_info.pos_y = i / s->frame[i].tile_info.x_count;

                tile_width = width / s->frame[i].tile_info.x_count;
                tile_height = height / s->frame[i].tile_info.y_count;
                s->frame[i].dst_x_offset = 0;
                s->frame[i].color_spec = codec;
                s->frame[i].width = width;
                s->frame[i].height = height;
                s->frame[i].dst_bpp = get_bpp(codec);
                s->frame[i].src_bpp = get_bpp(codec);
                s->frame[i].fps = fps;
                s->frame[i].aux = aux;

                s->frame[i].data_len = tile_width * tile_height * s->frame[i].dst_bpp;
                int aligned_x=tile_width;
                aligned_x =
                    ((aligned_x + s->cinfo->h_align -
                      1) / s->cinfo->h_align) * s->cinfo->h_align;
                s->frame[i].dst_linesize = aligned_x * s->frame[i].dst_bpp;
                s->frame[i].src_linesize = aligned_x * s->frame[i].src_bpp;
                s->frame[i].dst_pitch = s->frame[i].dst_linesize;
                
                s->frame[i].state = s;
                
                if(s->frame[i].data != NULL) {
                        free(s->frame[i].data);
                }
                
                s->frame[i].data = calloc(s->frame[i].data_len, 1);
                
                s->videoDisplayComponentInstance[i] = OpenComponent((Component) s->device[i]);
                
                /* Set the display mode */
                ret =
                    QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance[i],
                                                find_mode(&s->videoDisplayComponentInstance[i],
                                                        tile_width, tile_height, codec_name, fps, aux));
                if (ret != noErr) {
                        fprintf(stderr, "Failed to set video output display mode.\n");
                        exit(128);
                }
                
                /* We don't want to use the video output component instance echo port */
                ret = QTVideoOutputSetEchoPort(s->videoDisplayComponentInstance[i], nil);
                if (ret != noErr) {
                        fprintf(stderr, "Failed to set video output echo port.\n");
                        exit(128);
                }

                /* Register Ultragrid with instande of the video outpiut */
                ret =
                    QTVideoOutputSetClientName(s->videoDisplayComponentInstance[i],
                                               (ConstStr255Param) "Ultragrid");
                if (ret != noErr) {
                        fprintf(stderr,
                                "Failed to register Ultragrid with selected video output instance.\n");
                        exit(128);
                }

                ret = QTVideoOutputBegin(s->videoDisplayComponentInstance[i]);
                if (ret != noErr) {
                        fprintf(stderr,
                                "Failed to get exclusive access to selected video output instance.\n");
                        exit(128);
                }
                /* Get a pointer to the gworld used by video output component */
                ret =
                    QTVideoOutputGetGWorld(s->videoDisplayComponentInstance[i],
                                           &s->gworld[i]);
                if (ret != noErr) {
                        fprintf(stderr,
                                "Failed to get selected video output instance GWorld.\n");
                        exit(128);
                }
        }

	reconf_common(s);
}

static int find_mode(ComponentInstance *ci, int width, int height, 
                const char * codec_name, double fps, int aux)
{
        UNUSED(aux);

	QTAtom atomDisplay = 0, nextAtomDisplay = 0;
	QTAtomType type;
	QTAtomID id;
	QTAtomContainer modeListAtomContainer = NULL;
	int found = FALSE;
        int i = 1;

        int ret =
            QTVideoOutputGetDisplayModeList
            (*ci, &modeListAtomContainer);
        if (ret != noErr || modeListAtomContainer == NULL) {
                fprintf(stdout, "\tNo output modes available\n");
                return 0;
        }

	/* Print modes of current display component */
	while (!found && i <
	       QTCountChildrenOfType(modeListAtomContainer,
				     kParentAtomIsContainer,
				     kQTVODisplayModeItem)) {

		ret =
		    QTNextChildAnyType(modeListAtomContainer,
				       kParentAtomIsContainer,
				       atomDisplay, &nextAtomDisplay);
		// Make sure its a display atom
		ret =
		    QTGetAtomTypeAndID(modeListAtomContainer,
				       nextAtomDisplay, &type, &id);
		if (type != kQTVODisplayModeItem) {
			++i;
			continue;
		}

		atomDisplay = nextAtomDisplay;

		QTAtom atom;
		long dataSize, *dataPtr;

		/* Print component other info */
		atom =
		    QTFindChildByID(modeListAtomContainer, atomDisplay,
				    kQTVODimensions, 1, NULL);
		ret =
		    QTGetAtomDataPtr(modeListAtomContainer, atom,
				     &dataSize, (Ptr *) & dataPtr);
		if(width != (int)EndianS32_BtoN(dataPtr[0]) ||
				height != (int)EndianS32_BtoN(dataPtr[1])) {
			++i;
                        debug_msg("[quicktime] mode %dx%d not selected.\n",
                                        (int)EndianS32_BtoN(dataPtr[0]), 
                                        (int)EndianS32_BtoN(dataPtr[1]));
			continue;
		}
		atom =
		    QTFindChildByID(modeListAtomContainer, atomDisplay,
				    kQTVORefreshRate, 1, NULL);
		ret =
		    QTGetAtomDataPtr(modeListAtomContainer, atom,
				     &dataSize, (Ptr *) & dataPtr);
                /* Following computation is in Fixed data type - its real value
                 * is 65536 bigger than coresponding integer (Fixed cast to int)
                 */
		if(fabs(fps * 65536 - EndianS32_BtoN(dataPtr[0]) > 0.01 * 65536)) {
			++i;
                        debug_msg("[quicktime] mode %dx%d@%0.2f not selected.\n",
                                        width, height, EndianS32_BtoN(dataPtr[0])/65536.0);
			continue;
		}

		/* Get supported pixel formats */
		QTAtom decompressorsAtom;
		int j = 1;
		while ((decompressorsAtom =
			QTFindChildByIndex(modeListAtomContainer,
					   atomDisplay,
					   kQTVODecompressors, j,
					   NULL)) != 0) {
			atom =
			    QTFindChildByID(modeListAtomContainer,
					    decompressorsAtom,
					    kQTVODecompressorType, 1,
					    NULL);
			ret =
			    QTGetAtomDataPtr(modeListAtomContainer,
					     atom, &dataSize,
					     (Ptr *) & dataPtr);
			if(strcasecmp((char *) dataPtr, codec_name) == 0) {
                                debug_msg("[quicktime] mode %dx%d@%0.2f (%s) SELECTED.\n",
                                        width, height, fps, codec_name);
				found = TRUE;
				break;
			} else {
                                debug_msg("[quicktime] mode %dx%d@%0.2f (%s) not selected.\n",
                                        width, height, fps, codec_name);
                        }

			j++;
		}

		i++;
	}

        if(found) {
                debug_msg("Selected format: %ld\n", id);
                return id;
        } else {
                fprintf(stderr, "[quicktime] mode %dx%d@%0.2f (%s) NOT FOUND.\n",
                                width, height, fps, codec_name);
                return 0;
        }
}

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_quicktime *s = (struct state_quicktime *)state;

        if(s->devices_cnt == 1) {/* dual link or 4K framebuffer (aka Kona 3G)  */
                memcpy(out, &s->frame, sizeof(struct video_frame));
                out->data +=
                        y * s->frame[0].dst_pitch +
                        (size_t) (x * s->frame[0].dst_bpp);
                out->src_linesize =
                        vc_getsrc_linesize(w, out->color_spec);
                out->dst_linesize =
                        w * out->dst_bpp;
        } else { /* tiled video
                     expecting that requested sub_frame matches
                     exactly one physical device */
                double x_cnt = sqrt(s->devices_cnt);
                int y_cnt;
                int index;
                /* cnt == 1 -> 1x1, 2 -> 2x1, 3 -> 3x1, 4 -> 2x2, 5 -> 5x1 etc. */
                x_cnt = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
                y_cnt = s->devices_cnt / (int) x_cnt;

                /* check if have tiles of equal size (required) */
                assert(x % w == 0 &&
                                y % h == 0 &&
                                s->frame[0].width % w == 0 &&
                                s->frame[0].height % h == 0);

                /* determine which preprepared bufer to put */
                index = x / w + // row 
                        x_cnt * (y / h); // column
                /* and give it to the caller */
                memcpy(out, &s->frame[index], sizeof(struct video_frame));
        }
}

#endif                          /* HAVE_MACOSX */

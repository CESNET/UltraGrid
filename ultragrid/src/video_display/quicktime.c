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
#include <AudioUnit/AudioUnit.h>

#include "compat/platform_semaphore.h"
#include <signal.h>
#include <pthread.h>
#include <assert.h>

#include "video_display.h"
#include "video_display/quicktime.h"

#include "audio/audio.h"

#define MAGIC_QT_DISPLAY        DISPLAY_QUICKTIME_ID

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

/* for audio see TN2091 (among others) */
struct state_quicktime {
        ComponentInstance videoDisplayComponentInstance[MAX_DEVICES];
#if MACOSX_VERSION_MAJOR <= 9
        ComponentInstance 
#else
        AudioComponentInstance
#endif
                                auHALComponentInstance;

        CFStringRef audio_name;
//    Component                 videoDisplayComponent;
        GWorldPtr gworld[MAX_DEVICES];
        ImageSequence seqID[MAX_DEVICES];
        int device[MAX_DEVICES];
        int mode;
        int devices_cnt;
        const struct codec_info_t *cinfo;

        /* Thread related information follows... */
        pthread_t thread_id;
        sem_t semaphore;

        struct video_frame *frame;

        struct audio_frame audio;
        int audio_packet_size;
        int audio_start, audio_end, max_audio_data_len;
        char *audio_data;
        unsigned play_audio:1;
        unsigned mode_set_manually:1;

        uint32_t magic;
};

/* Prototyping */
char *four_char_decode(int format);
static int find_mode(ComponentInstance *ci, int width, int height, 
                const char * codec_name, double fps, int aux);
void display_quicktime_audio_init(struct state_quicktime *s);
void display_quicktime_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);

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
                struct tile *tile = tile_get(s->frame, i % s->frame->grid_width,
                                                        i / s->frame->grid_width);
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
                (**(ImageDescriptionHandle) imageDesc).dataSize = tile->data_len;
                /* 
                 * Beware: must be a multiple of horiz_align_pixels which is 2 for 2Vuy
                 * and 48 for v210. hd_size_x=1920 is a multiple of both. TODO: needs 
                 * further investigation for 2K!
                 */
                (**(ImageDescriptionHandle) imageDesc).width = get_haligned(tile->width, s->frame->color_spec);
                (**(ImageDescriptionHandle) imageDesc).height = tile->height;
        
                ret = DecompressSequenceBeginS(&(s->seqID[i]), imageDesc, tile->data, 
                                               // Size of the buffer, not size of the actual frame data inside
                                               tile->data_len,    
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
                        struct tile *tile = tile_get(s->frame, i % s->frame->grid_width,
                                                        i / s->frame->grid_width);
                        /* TODO: Running DecompressSequenceFrameWhen asynchronously 
                         * in this way introduces a possible race condition! 
                         */
                        ret = DecompressSequenceFrameWhen(s->seqID[i], tile->data, 
                                                        tile->data_len,
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
        printf("\t-d quicktime:<device>[:<mode>:<codec>] | help | fullhelp\n");
        printf("\t\tIf you set mode, the output format will be forced and if not matched with input, the output will be scaled.");
        print_modes(full);
}

void *display_quicktime_init(char *fmt, unsigned int flags)
{
        struct state_quicktime *s;
        int ret;
        int i;
        char *codec_name;

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
                        s->mode = 0;
                        s->mode_set_manually = FALSE;
                } else {
                        s->mode_set_manually = TRUE;
                        s->mode = atol(tok);
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

        double x_cnt = sqrt(s->devices_cnt);
        int x_count = x_cnt - round(x_cnt) == 0.0 ? x_cnt : s->devices_cnt;
        int y_count = s->devices_cnt / x_count;
        
        s->frame = vf_alloc(x_count, y_count);
        
        for (i = 0; i < s->devices_cnt; ++i) {
                s->videoDisplayComponentInstance[i] = 0;
                s->seqID[i] = 0;
        }

        InitCursor();
        EnterMovies();

        if(s->mode != 0) {
                for (i = 0; codec_info[i].name != NULL; i++) {
                        if (strcmp(codec_name, codec_info[i].name) == 0) {
                                s->cinfo = &codec_info[i];
                        }
                }
                free(codec_name);
                s->frame->color_spec = s->cinfo->codec;
                s->frame->aux = 0;

                for (i = 0; i < s->devices_cnt; ++i) {
                        int pos_x = i % x_count;
                        int pos_y = i / y_count;
                        struct tile *tile = tile_get(s->frame, pos_x, pos_y);
                        /* Open device */
                        s->videoDisplayComponentInstance[i] = OpenComponent((Component) s->device[i]);
        
                        /* Set the display mode */
                        ret =
                            QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance[i],
                                                        s->mode);
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
                        
                        tile->width = (**gWorldImgDesc).width;
                        tile->height = (**gWorldImgDesc).height;

                        tile->linesize = vc_get_linesize(tile->width, s->cinfo->codec);
                        tile->data_len = tile->linesize * tile->height;
                        tile->data = calloc(1, tile->data_len);
        
                        fprintf(stdout, "Selected mode: %dx%d, %fbpp\n", tile->width,
                                tile->height, s->cinfo->bpp);
                }
                reconf_common(s);
        } else {
                for (i = 0; i < s->devices_cnt; ++i) {
                        s->frame->tiles[i].width = 0;
                        s->frame->tiles[i].height = 0;
                        s->frame->tiles[i].data = NULL;
                }
        }
        
        s->audio.state = s;
        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                display_quicktime_audio_init(s);
        } else {
                s->play_audio = FALSE;
        }

        platform_sem_init(&s->semaphore, 0, 0);

        /*if (pthread_create
            (&(s->thread_id), NULL, display_thread_quicktime, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/

        return (void *)s;
}

void display_quicktime_audio_init(struct state_quicktime *s)
{
        OSErr ret = noErr;
#if MACOSX_VERSION_MAJOR <= 9
        Component comp;
        ComponentDescription desc;
#else
        AudioComponent comp;
        AudioComponentDescription desc;
#endif
        //There are several different types of Audio Units.
        //Some audio units serve as Outputs, Mixers, or DSP
        //units. See AUComponent.h for listing
        desc.componentType = kAudioUnitType_Output;

        //Every Component has a subType, which will give a clearer picture
        //of what this components function will be.
        //desc.componentSubType = kAudioUnitSubType_DefaultOutput;
        desc.componentSubType = kAudioUnitSubType_HALOutput;

        //all Audio Units in AUComponent.h must use 
        //"kAudioUnitManufacturer_Apple" as the Manufacturer
        desc.componentManufacturer = kAudioUnitManufacturer_Apple;
        desc.componentFlags = 0;
        desc.componentFlagsMask = 0;

#if MACOSX_VERSION_MAJOR <= 9
        comp = FindNextComponent(NULL, &desc);
        if(!comp) goto audio_error;
        ret = OpenAComponent(comp, &s->auHALComponentInstance);
        if (ret != noErr) goto audio_error;
#else
        comp = AudioComponentFindNext(NULL, &desc);
        if(!comp) goto audio_error;
        ret = AudioComponentInstanceNew(comp, &s->auHALComponentInstance);
        if (ret != noErr) goto audio_error;
#endif
        
        if(s->frame->tiles[0].data == NULL) /* if the output is not open - open it temporarily */
                 ret = OpenAComponent((Component) s->device[0], &s->videoDisplayComponentInstance[0]);
        if (ret != noErr) goto audio_error;
        s->audio.data = NULL;
        s->audio_data = NULL;
        s->audio.max_size = 0;
        s->audio.reconfigure_audio = display_quicktime_reconfigure_audio;
        Component audioComponent;
        ret = QTVideoOutputGetIndSoundOutput(s->videoDisplayComponentInstance[0], 1, &audioComponent);
        if (ret != noErr) goto audio_error;

        Handle componentNameHandle = NewHandle(0);
        ret = GetComponentInfo(audioComponent, 0, componentNameHandle, NULL, NULL);
        if (ret != noErr) goto audio_error;
        HLock(componentNameHandle);
        s->audio_name = CFStringCreateWithPascalString(NULL, *componentNameHandle,
                        kCFStringEncodingMacRoman);
        HUnlock(componentNameHandle);
        DisposeHandle(componentNameHandle);
        if(s->frame->tiles[0].data == NULL) /* ...and close temporarily opened input */
                CloseComponent(s->videoDisplayComponentInstance[0]);
        s->play_audio = TRUE;
        return;
audio_error:
        fprintf(stderr, "There is no audio support (%x).\n", ret);
        s->play_audio = FALSE;
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

int display_quicktime_get_property(void *state, int property, void *val, int *len)
{
        UNUSED(state);
        codec_t codecs[] = {v210, UYVY, RGBA};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

void display_quicktime_reconfigure(void *state, struct video_desc desc)
{
        struct state_quicktime *s = (struct state_quicktime *) state;
        int i;
        int ret;
        const char *codec_name;

        for (i = 0; codec_info[i].name != NULL; i++) {
                if (codec_info[i].codec == desc.color_spec) {
                        s->cinfo = &codec_info[i];
                        codec_name = s->cinfo->name;
                }
        }
        if(desc.color_spec == UYVY || desc.color_spec == DVS8) /* just aliases for 2vuy,
                                            * but would confuse QT */
                codec_name = "2vuy";
         
        if(s->frame->tiles[0].data != NULL)
                display_quicktime_done(s);
                
        fprintf(stdout, "Selected mode: %dx%d, %fbpp\n", desc.width,
                        desc.height, s->cinfo->bpp);
        s->frame->color_spec = desc.color_spec;
        s->frame->fps = desc.fps;
        s->frame->aux = desc.aux;

        for(i = 0; i < s->devices_cnt; ++i) {
                int tile_width = desc.width / s->frame->grid_width;
                int tile_height = desc.height / s->frame->grid_height;
                int pos_x = i % s->frame->grid_width;
                int pos_y = i / s->frame->grid_width;
                struct tile * tile = tile_get(s->frame, pos_x, pos_y);
                
                tile->width = tile_width;
                tile->height = tile_height;
                tile->linesize = vc_get_linesize(tile_width, desc.color_spec);
                tile->data_len = tile->linesize * tile->height;
                
                if(tile->data != NULL) {
                        free(tile->data);
                }
                
                tile->data = calloc(1, tile->data_len);
                
                s->videoDisplayComponentInstance[i] = OpenComponent((Component) s->device[i]);
                
                if(!s->mode_set_manually)
                        s->mode = find_mode(&s->videoDisplayComponentInstance[i],
                                                        tile_width, tile_height, codec_name, desc.fps, desc.aux);
                /* Set the display mode */
                ret =
                    QTVideoOutputSetDisplayMode(s->videoDisplayComponentInstance[i],
                                                s->mode);
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

struct audio_frame *display_quicktime_get_audio_frame(void *state)
{
        struct state_quicktime *s = (struct state_quicktime *)state;

        if(s->play_audio)
                return &s->audio;
        else
                return NULL;
}

void display_quicktime_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_quicktime * s = (struct state_quicktime *) state;
        int to_end = frame->data_len;

        if(frame->data_len > s->max_audio_data_len - s->audio_end)
                to_end = s->max_audio_data_len - s->audio_end;

        memcpy(s->audio_data + s->audio_end, frame->data, to_end);
        memcpy(s->audio_data, frame->data + to_end, frame->data_len - to_end);
        s->audio_end = (s->audio_end + frame->data_len) % s->max_audio_data_len;
}

static OSStatus theRenderProc(void *inRefCon,
                              AudioUnitRenderActionFlags *inActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumFrames,
                              AudioBufferList *ioData)
{
        struct state_quicktime * s = (struct state_quicktime *) inRefCon;
        int write_bytes = inNumFrames * s->audio_packet_size;
        int bytes_in_buffer = s->audio_end - s->audio_start;
        int to_end;

        if (bytes_in_buffer < 0)
                bytes_in_buffer += s->max_audio_data_len;
        
        if(write_bytes > bytes_in_buffer)
                write_bytes = bytes_in_buffer;
        to_end = s->max_audio_data_len - s->audio_start;
        if(to_end > write_bytes)
                to_end = write_bytes;

        memcpy(ioData->mBuffers[0].mData, s->audio_data + s->audio_start, to_end);
        memcpy(ioData->mBuffers[0].mData + to_end, s->audio_data, write_bytes - to_end);
        ioData->mBuffers[0].mDataByteSize = write_bytes;
        s->audio_start = (s->audio_start + write_bytes) % s->max_audio_data_len;

        if(!write_bytes) {
                fprintf(stderr, "[quicktime] Audio buffer underflow.\n");
                //usleep(10 * 1000 * 1000);
        }  
        return noErr;
}

void display_quicktime_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) 
{
        struct state_quicktime *s = (struct state_quicktime *)state;
        OSErr ret;
        UInt32 size;
        AURenderCallbackStruct  renderStruct;
        AudioStreamBasicDescription desc;
        AudioDeviceID device;
        AudioDeviceID *dev_ids;
        int dev_items;
        int i;

        fprintf(stderr, "[quicktime] Audio reinitialized to %d-bit, %d channels, %d Hz\n", 
                        quant_samples, channels, sample_rate);
        ret = AudioUnitUninitialize(s->auHALComponentInstance);
        if(ret) goto error;

        s->audio.bps = quant_samples / 8;
        s->audio.ch_count = channels;
        s->audio.sample_rate = sample_rate;

        free(s->audio_data);
        free(s->audio.data);
        s->audio_start = 0;
        s->audio_end = 0;
        s->audio.max_size = s->max_audio_data_len = quant_samples / 8 * channels * sample_rate * 5;
        s->audio_data = (char *) malloc(s->max_audio_data_len);
        s->audio.data = (char *) malloc(s->audio.max_size);

        ret = AudioHardwareGetPropertyInfo(kAudioHardwarePropertyDevices, &size, NULL);
        if(ret) goto error;
        dev_ids = malloc(size);
        dev_items = size / sizeof(AudioDeviceID);
        ret = AudioHardwareGetProperty(kAudioHardwarePropertyDevices, &size, dev_ids);
        if(ret) goto error;

        for(i = 0; i < dev_items; ++i)
        {
                CFStringRef name;
                
                size = sizeof(name);
                ret = AudioDeviceGetProperty(dev_ids[i], 0, 0, kAudioDevicePropertyDeviceNameCFString, &size, &name);
                if(CFStringCompare(name, s->audio_name,0) == kCFCompareEqualTo) device = dev_ids[i];
                CFRelease(name);
        }
        free(dev_ids);

        size=sizeof(device);
        //ret = AudioHardwareGetProperty(kAudioHardwarePropertyDefaultOutputDevice, &size, &device);
        //if(ret) goto error;
        
        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                         kAudioOutputUnitProperty_CurrentDevice, 
                         kAudioUnitScope_Global, 
                         1, 
                         &device, 
                         sizeof(device));
        if(ret) goto error;

        size = sizeof(desc);
        ret = AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &desc, &size);
        if(ret) goto error;
        desc.mSampleRate = sample_rate;
        desc.mFormatID = kAudioFormatLinearPCM;
        desc.mChannelsPerFrame = channels;
        desc.mBitsPerChannel = quant_samples;
        desc.mFormatFlags = kAudioFormatFlagIsSignedInteger|kAudioFormatFlagIsPacked;
        desc.mFramesPerPacket = 1;
        s->audio_packet_size = desc.mBytesPerFrame = desc.mBytesPerPacket = desc.mFramesPerPacket * channels * (quant_samples / 8);
        
        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &desc, sizeof(desc));
        if(ret) goto error;

        renderStruct.inputProc = theRenderProc;
        renderStruct.inputProcRefCon = s;
        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_SetRenderCallback,
                        kAudioUnitScope_Input, 0, &renderStruct, sizeof(AURenderCallbackStruct));
        if(ret) goto error;

        ret = AudioUnitInitialize(s->auHALComponentInstance);
        if(ret) goto error;

        ret = AudioOutputUnitStart(s->auHALComponentInstance);
        if(ret) goto error;

        return;
error:
        fprintf(stderr, "Audio setting error, disabling audio.\n");
        debug_msg("[quicktime] error: %d", ret);
        free(s->audio_data);
        free(s->audio.data);
        s->audio_data = NULL;
        s->audio.data = NULL;

        s->play_audio = FALSE;
}

#endif                          /* HAVE_MACOSX */


/*
 * FILE:    video_display/dvs.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          Colin Perkins    <csp@isi.edu>
 *
 * Copyright (c) 2005-2015 CESNET z.s.p.o.
 * Copyright (c) 2001-2003 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "host.h"

#ifdef HAVE_DVS           /* From config.h */

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "video_display/dvs.h"
#include "audio/audio.h"
#include "tv.h"
#include "utils/ring_buffer.h"

#include "dvs_clib.h"           /* From the DVS SDK */
#include "dvs_fifo.h"           /* From the DVS SDK */

#define HDSP_MAGIC	0x12345678

const hdsp_mode_table_t hdsp_mode_table[] = {
        {SV_MODE_PAL, 720, 576, 25.0, AUX_INTERLACED},  /* pal          720x576 25.00hz Interlaced        */
        {SV_MODE_NTSC, 720, 486, 29.97, AUX_INTERLACED},  /* ntsc         720x486 29.97hz Interlaced        */
        {SV_MODE_PALHR, 960, 576, 25.50, AUX_INTERLACED},  /* pal          960x576 25.00hz Interlaced        */
        {SV_MODE_NTSCHR, 960, 486, 29.97, AUX_INTERLACED}, /* ntsc         960x486 29.97hz Interlaced        */
        {SV_MODE_PALFF, 720, 592, 25.00, AUX_INTERLACED},  /* pal          720x592 25.00hz Interlaced        */
        {SV_MODE_NTSCFF, 720, 502, 29.97, AUX_INTERLACED},  /* ntsc         720x502 29.97hz Interlaced        */
        {SV_MODE_PAL608, 720, 608, 25.00, AUX_INTERLACED},  /* pal608       720x608 25.00hz Interlaced        */
        {SV_MODE_HD360, 960, 504, 29.97, AUX_INTERLACED},  /* HD360        960x504 29.97hz Compressed HDTV   */
        {SV_MODE_SMPTE293_59P, 720, 483, 59.94, AUX_PROGRESSIVE}, /* SMPTE293/59P 720x483 59.94hz Progressive       */
        {SV_MODE_PAL_24I, 720, 576, 24.0, AUX_INTERLACED},  /* SLOWPAL      720x576 24.00hz Interlaced        */
        //#define SV_MODE_TEST  /*              Test Raster                       */
        {SV_MODE_VESASDI_1024x768_60P, 1024, 768, 60, AUX_PROGRESSIVE},
        {SV_MODE_PAL_25P, 720, 576, 25, AUX_PROGRESSIVE},  /* pal          25Hz (1:1)                        */
        {SV_MODE_PAL_50P, 720, 576, 50, AUX_PROGRESSIVE},  /* pal          50Hz (1:1)                        */
        {SV_MODE_PAL_100P, 720, 576, 100, AUX_PROGRESSIVE},  /* pal         100Hz (1:1)                        */
        {SV_MODE_NTSC_29P, 720, 576, 29.97, AUX_PROGRESSIVE}, /* ntsc         29.97Hz (1:1) */
        {SV_MODE_NTSC_59P, 720, 576, 59.94, AUX_PROGRESSIVE},   /* ntsc         59.94Hz (1:1) */
        {SV_MODE_NTSC_119P, 720, 486, 119.88, AUX_PROGRESSIVE}, /* ntsc        119.88Hz (1:1) */
        {SV_MODE_SMPTE274_25sF, 1920, 1080, 25.0, AUX_SF},  /*              1920x1080 25.00hz Segmented Frame */
        {SV_MODE_SMPTE274_29sF, 1920, 1080, 29.97, AUX_SF},  /*              1920x1080 29.97hz Segmented Frame */
        {SV_MODE_SMPTE274_30sF, 1920, 1080, 30.00, AUX_SF},  /*              1920x1080 30.00hz Segmented Frame */
        //#define SV_MODE_EUREKA
        {SV_MODE_SMPTE240_30I, 1920, 1035, 30.0, AUX_INTERLACED}, /*              1920x1035 30.00hz Interlaced      */
        {SV_MODE_SMPTE274_30I, 1920, 1080, 30.0, AUX_INTERLACED},  /*              1920x1080 30.00hz Interlaced      */
        {SV_MODE_SMPTE296_60P, 1280, 720, 60.0, AUX_PROGRESSIVE},  /*               1280x720 60.00hz Progressive     */
        {SV_MODE_SMPTE240_29I, 1920, 1035, 29.97, AUX_INTERLACED}, /*              1920x1035 29.97hz Interlaced      */
        {SV_MODE_SMPTE274_29I, 1920, 1080, 29.97, AUX_INTERLACED},  /*              1920x1080 29.97hz Interlaced      */
        {SV_MODE_SMPTE296_59P, 1280, 720, 59.94, AUX_PROGRESSIVE},  /*               1280x720 59.94hz Progressive     */
        {SV_MODE_SMPTE295_25I, 1920, 1080, 25.0, AUX_INTERLACED},  /*              1920x1080 1250/25Hz Interlaced    */
        {SV_MODE_SMPTE274_25I, 1920, 1080, 25.0, AUX_INTERLACED},  /*              1920x1080 25.00hz Interlaced      */
        {SV_MODE_SMPTE274_24sF, 1920, 1080, 24.0, AUX_SF},  /*              1920x1080 24.00hz Segmented Frame */
        {SV_MODE_SMPTE274_23sF, 1920, 1080, 23.98, AUX_SF},  /*              1920x1080 23.98hz Segmented Frame */
        {SV_MODE_SMPTE274_24P, 1920, 1080, 24.0, AUX_SF},  /*              1920x1080 24.00hz Progressive     */
        {SV_MODE_SMPTE274_23P, 1920, 1080, 23.98, AUX_PROGRESSIVE},  /*              1920x1080 23.98hz Progressive     */
        {SV_MODE_SMPTE274_25P, 1920, 1080, 25.00, AUX_PROGRESSIVE},  /*              1920x1080 25.00hz Progressive     */
        {SV_MODE_SMPTE274_29P, 1920, 1080, 29.97, AUX_PROGRESSIVE},  /*              1920x1080 29.97hz Progressive     */
        {SV_MODE_SMPTE274_30P, 1920, 1080, 30.00, AUX_PROGRESSIVE},  /*              1920x1080 30.00hz Progressive     */

        {SV_MODE_SMPTE296_72P, 1280, 720, 72.00, AUX_PROGRESSIVE},  /*              1280x720 72.00hz Progressive      */
        {SV_MODE_SMPTE296_71P, 1280, 720, 71.93, AUX_PROGRESSIVE},  /*              1280x720 71.93hz Progressive      */
        {SV_MODE_SMPTE296_72P_89MHZ, 1280, 720, 72.00, AUX_PROGRESSIVE},  /*             1280x720 72.00hz Progressive Analog 89 Mhz */
        {SV_MODE_SMPTE296_71P_89MHZ, 1280, 720, 71.93, AUX_PROGRESSIVE},  /*             1280x720 71.93hz Progressive Analog 89 Mhz */
        
        {SV_MODE_SMPTE274_23I, 1920, 1080, 23.98, AUX_INTERLACED},  /*             1920x1080 23.98hz Interlaced      */
        {SV_MODE_SMPTE274_24I, 1920, 1080, 24.00, AUX_INTERLACED},  /*             1920x1080 24.00hz Interlaced      */
        
        {SV_MODE_SMPTE274_47P, 1920, 1080, 47.95, AUX_PROGRESSIVE},  /*             1920x1080 47.95hz Progressive     */
        {SV_MODE_SMPTE274_48P, 1920, 1080, 48.00, AUX_PROGRESSIVE},  /*             1920x1080 48.00hz Progressive     */
        {SV_MODE_SMPTE274_59P, 1920, 1080, 59.94, AUX_PROGRESSIVE},  /*             1920x1080 59.94hz Progressive     */
        {SV_MODE_SMPTE274_60P, 1920, 1080, 60.00, AUX_PROGRESSIVE},  /*             1920x1080 60.00hz Progressive     */
        {SV_MODE_SMPTE274_71P, 1920, 1080, 71.93, AUX_PROGRESSIVE}, /*             1920x1080 71.93hz Progressive     */
        {SV_MODE_SMPTE274_72P, 1920, 1080, 72.00, AUX_PROGRESSIVE},  /*             1920x1080 72.00hz Progressive     */
        
        //{SV_MODE_SMPTE274_2560_24P
        //{SV_MODE_SMPTE274_2560_23P  ( | SV_MODE_FLAG_DROPFRAME)
        //{SV_MODE_SMPTE296_24P_30MHZ, 1280, 720, 30, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_24P, 1280, 720, 24, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_23P, 1280, 720, 23.98, AUX_PROGRESSIVE},
        
        {SV_MODE_FILM2K_1556_12P, 2048, 1556, 12.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_1556_6P, 2048, 1556, 6.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_1556_3P, 2048, 1556, 3.0, AUX_PROGRESSIVE},
        
        {SV_MODE_FILM2K_2048x1536_24P, 2048, 1536, 24.0, 0},  /* Telecine formats with 4:3 aspect ratio   */
        {SV_MODE_FILM2K_2048x1536_24sF, 2048, 1536, 24.0, AUX_SF},
        {SV_MODE_FILM2K_2048x1536_48P, 2048, 1536, 48.0, AUX_PROGRESSIVE},
        
        {SV_MODE_FILM2K_2048x1556_24P, 2048, 1556, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1556_23P, 2048, 1556, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1556_24sF, 2048, 1556, 24.0, AUX_SF},
        {SV_MODE_FILM2K_2048x1556_23sF, 2048, 1556, 23.98, AUX_SF},
        {SV_MODE_FILM2K_2048x1556_48P, 2048, 1556, 48.0, AUX_PROGRESSIVE},
        
        /*{SV_MODE_SGI_12824_NTSC 0x4f
        {SV_MODE_SGI_12824_59P  0x50
        {SV_MODE_SGI_12848_29I  0x51*/
   
        {SV_MODE_FILM2K_2048x1556_14sF, 2048, 1556, 14.0, AUX_SF},
        {SV_MODE_FILM2K_2048x1556_15sF, 2048, 1556, 15.0, AUX_SF},
        
        {SV_MODE_VESA_1024x768_30I, 1024, 768, 30.0, AUX_INTERLACED},
        {SV_MODE_VESA_1024x768_29I, 1024, 768, 29.97, AUX_INTERLACED},
        {SV_MODE_VESA_1280x1024_30I, 1280, 1024, 30.0, AUX_INTERLACED},
        {SV_MODE_VESA_1280x1024_29I, 1280, 1024, 29.97, AUX_INTERLACED},
        {SV_MODE_VESA_1600x1200_30I, 1600, 1200, 30.0, AUX_INTERLACED},
        {SV_MODE_VESA_1600x1200_29I, 1600, 1200, 29.97, AUX_INTERLACED},
        
        {SV_MODE_VESA_640x480_60P, 640, 480, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_640x480_59P, 640, 480, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_VESA_800x600_60P, 800, 600, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_800x600_59P, 800, 600, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1024x768_60P, 1024, 768, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1024x768_59P, 1024, 768, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1280x1024_60P, 1280, 1024, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1280x1024_59P, 1280, 1024, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1600x1200_60P, 1600, 1200, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1600x1200_59P, 1600, 1200, 59.94, AUX_PROGRESSIVE},
        
        {SV_MODE_VESA_640x480_72P, 640, 480, 72.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_640x480_71P, 640, 480, 71.93, AUX_PROGRESSIVE},
        {SV_MODE_VESA_800x600_72P, 800, 600, 72.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_800x600_71P, 800, 600, 71.93, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1024x768_72P, 1024, 768, 72.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1024x768_71P, 1024, 768, 71.93, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1280x1024_72P, 1280, 1024, 72.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1280x1024_71P, 1280, 1024, 71.93, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1600x1200_72P, 1600, 1200, 72.0, AUX_PROGRESSIVE},
        {SV_MODE_VESA_1600x1200_71P, 1600, 1200, 71.93, AUX_PROGRESSIVE}, 
        
        {SV_MODE_SMPTE296_25P, 1280, 720, 25.0, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_29P, 1280, 720, 29.97, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_30P, 1280, 720, 30.0, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_50P, 1280, 720, 50.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x1556_29sF, 2048, 1556, 29.97, AUX_SF},
        {SV_MODE_FILM2K_2048x1556_30sF, 2048, 1556, 30.0, AUX_SF},
        {SV_MODE_FILM2K_2048x1556_36sF, 2048, 1556, 36.0, AUX_SF},

        {SV_MODE_FILM2K_2048x1080_23sF, 2048, 1080, 23.98, AUX_SF},
        {SV_MODE_FILM2K_2048x1080_24sF, 2048, 1080, 24.0, AUX_SF},
        {SV_MODE_FILM2K_2048x1080_23P, 2048, 1080, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_24P, 2048, 1080, 24.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x1556_19sF, 2048, 1556, 19, AUX_SF}, 
        {SV_MODE_FILM2K_2048x1556_20sF, 2048, 1556, 20.0, AUX_SF}, 

        {SV_MODE_1980x1152_25I, 1980, 1152, 25.0, AUX_INTERLACED},

        {SV_MODE_SMPTE274_50P, 1920, 1080, 50.0, AUX_PROGRESSIVE}, 

        {SV_MODE_FILM4K_4096x2160_24sF, 4096, 2160, 24.0, AUX_SF}, 
        {SV_MODE_FILM4K_4096x2160_24P, 4096, 2160, 24.0, AUX_PROGRESSIVE},

        {SV_MODE_3840x2400_24sF, 3840, 240, 24.0, AUX_SF},
        {SV_MODE_3840x2400_24P, 3840, 2400, 24.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM4K_3112_24sF, 4096, 3112, 24.0, AUX_SF},
        {SV_MODE_FILM4K_3112_24P, 4096, 3112, 24.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM4K_3112_5sF, 4096, 3112, 5, AUX_SF},

        {SV_MODE_3840x2400_12P, 3840, 2440, 12, AUX_PROGRESSIVE},
        {SV_MODE_ARRI_1920x1080_47P, 1920, 1080, 47.95, AUX_PROGRESSIVE},
        {SV_MODE_ARRI_1920x1080_48P, 1920, 1080, 48.0, AUX_PROGRESSIVE},
        {SV_MODE_ARRI_1920x1080_50P, 1920, 1080, 50, AUX_PROGRESSIVE},
        {SV_MODE_ARRI_1920x1080_59P, 1920, 1080, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_ARRI_1920x1080_60P, 1920, 1080, 60.0, AUX_PROGRESSIVE},

        {SV_MODE_SMPTE296_100P, 1280, 720, 100.0, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_119P, 1280, 720, 119.88, AUX_PROGRESSIVE},
        {SV_MODE_SMPTE296_120P, 1280, 720, 120.0, AUX_PROGRESSIVE},

        {SV_MODE_1920x1200_24P, 1920, 1200, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_1920x1200_60P, 1920, 1200, 60.0, AUX_PROGRESSIVE},

        {SV_MODE_WXGA_1366x768_50P, 1366, 768, 50.0, AUX_PROGRESSIVE},
        {SV_MODE_WXGA_1366x768_59P, 1360, 768, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_WXGA_1366x768_60P, 1366, 768, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_WXGA_1366x768_90P, 1366, 768, 90.0, AUX_PROGRESSIVE},
        {SV_MODE_WXGA_1366x768_120P, 1366, 768, 120.0, AUX_PROGRESSIVE},

        {SV_MODE_1400x1050_60P, 1440, 1050, 60.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x858_23sF, 2048, 858, 23.98, AUX_SF},
        {SV_MODE_FILM2K_2048x858_24sF, 2048, 858, 24.0, AUX_SF},
        {SV_MODE_FILM2K_1998x1080_23sF, 1998, 1080, 23.98, AUX_SF},
        {SV_MODE_FILM2K_1998x1080_24sF, 1998, 1080, 24.0, AUX_SF},

        {SV_MODE_ANALOG_1920x1080_47P, 1920, 1080, 47.95, AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_1920x1080_48P, 1920, 1080, 48.0, AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_1920x1080_50P, 1920, 1080, 50.0, AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_1920x1080_59P, 1920, 1080, 59.94 , AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_1920x1080_60P, 1920, 1080, 60.0, AUX_PROGRESSIVE},

        {SV_MODE_QUADDVI_3840x2160_48P, 3840, 2160, 48.0, AUX_PROGRESSIVE},
        {SV_MODE_QUADDVI_3840x2160_48Pf2, 3840, 2160, 48.0, 0},			 /* wtf is Pf2 ?? */
        {SV_MODE_QUADDVI_3840x2160_60P, 3840, 2160, 60.0, AUX_PROGRESSIVE},
        {SV_MODE_QUADDVI_3840x2160_60Pf2, 3840, 2160, 60.0, AUX_SF},

        {SV_MODE_QUADSDI_3840x2160_23P, 3840, 2160, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_QUADSDI_3840x2160_24P, 3840, 2160, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_QUADSDI_4096x2160_23P, 3840, 2160, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_QUADSDI_4096x2160_24P, 3840, 2160, 24.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x1080_25P, 2048, 1080, 25.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x1744_24P, 2048, 1774, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_47P, 2048, 1080, 47.95, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_48P, 2048, 1080, 48.0, AUX_PROGRESSIVE},

        {SV_MODE_FILM2K_2048x1080_25sF, 2048, 1080, 25.0, AUX_SF},
        
        {SV_MODE_FILM2K_2048x1080_29P, 2048, 1080, 29.97 , AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_30P , 2048, 1080, 30.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_50P , 2048, 1080, 50.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_59P, 2048, 1080, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x1080_60P, 2048, 1080, 60.0, AUX_PROGRESSIVE},
        
        {SV_MODE_FILM2K_2048x858_23P, 2048, 858, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_2048x858_24P, 2048, 858, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_1998x1080_23P, 1998, 1080, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_FILM2K_1998x1080_24P, 1998, 1080, 24.0, AUX_PROGRESSIVE},
        
        {SV_MODE_ANALOG_2048x1080_50P, 2048, 1080, 50.0, AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_2048x1080_59P, 2048, 1080, 59.94, AUX_PROGRESSIVE},
        {SV_MODE_ANALOG_2048x1080_60P, 2048, 1080, 60.0, AUX_PROGRESSIVE},
        
        {SV_MODE_QUADSDI_3840x2400_23P, 3840, 2400, 23.98, AUX_PROGRESSIVE},
        {SV_MODE_QUADSDI_3840x2400_24P, 3840, 2400, 24.0, AUX_PROGRESSIVE},
        {SV_MODE_QUADDVI_3840x2160_30P, 3840, 1260, 30.0, AUX_PROGRESSIVE},
        {SV_MODE_QUADDVI_3840x2400_24P, 3840, 2400, 24.0, AUX_PROGRESSIVE},
        
        {SV_MODE_FILM2K_2048x1772_24P, 2048, 1772, 24.0, AUX_PROGRESSIVE},
        
        {0, 0, 0, 0, 0},
};

struct state_hdsp {
        pthread_t thread_id;
        sv_handle *sv;
        sv_fifo *fifo;
        sv_fifo_buffer *fifo_buffer;
        sv_fifo_buffer *display_buffer;
        sv_fifo_buffer *tmp_buffer;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        pthread_cond_t worker_cv;
volatile int work_to_do;
volatile int boss_waiting;
volatile int worker_waiting;
        uint32_t magic;
        char *bufs[2];
        int bufs_index;
        struct video_frame *frame;
        struct tile *tile;
        const hdsp_mode_table_t *mode;
        unsigned first_run:1;
        unsigned play_audio:1;

        pthread_mutex_t audio_reconf_lock;
        pthread_cond_t audio_reconf_possible_cv;
        pthread_cond_t audio_reconf_done_cv;
        volatile int audio_reconf_pending;
        volatile int audio_reconf_possible;
        volatile int audio_reconf_done;
        
        struct ring_buffer * audio_ring_buffer;
        char *audio_fifo_data; /* temporary memory for the data that gets immediatelly
                                    decoded */
        int audio_fifo_required_size;
        
        unsigned int mode_set_manually:1;

        int                     frames;
        struct timeval          t, t0;

        bool should_exit;
};

static void show_help(void);

static void show_help(void)
{
        int i;
        sv_handle *sv;
        int res;
        char name[128];
        int card_idx = 0;

        printf("DVS options:\n\n");
        printf("\t -d dvs[:<mode>:<codec>][:<card>] | help\n\n");
        printf("\t eg. '-d dvs' or '-d dvs:PCI,card:0'");
	printf("\t(the mode needn't to be set and you shouldn't to want set it)\n\n");
        snprintf(name, 128, "PCI,card:%d", card_idx);

        res = sv_openex(&sv, name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
        if (res != SV_OK) {
                printf
                    ("Unable to open grabber: sv_open() failed (no card present or driver not loaded?)\n");
                printf("Error %s\n", sv_geterrortext(res));
                return;
        }
        while (res == SV_OK) {
                printf("\tCard %s - supported modes:\n\n", name);
                for(i=0; hdsp_mode_table[i].width !=0; i++) {
                        int res;
                        sv_query(sv, SV_QUERY_MODE_AVAILABLE, hdsp_mode_table[i].mode, & res);
                        if(res) {
                                const char *interlacing;
                                if(hdsp_mode_table[i].aux & AUX_INTERLACED) {
                                                interlacing = "interlaced";
                                } else if(hdsp_mode_table[i].aux & AUX_PROGRESSIVE) {
                                                interlacing = "progressive";
                                } else if(hdsp_mode_table[i].aux & AUX_SF) {
                                                interlacing = "progressive segmented";
                                } else {
                                                interlacing = "unknown (!)";
                                }
                                printf ("\t%4d:  %4d x %4d @ %2.2f %s\n", hdsp_mode_table[i].mode, 
                                        hdsp_mode_table[i].width, hdsp_mode_table[i].height, 
                                        hdsp_mode_table[i].fps, interlacing);
                        }
                }
                sv_close(sv);
                card_idx++;
                snprintf(name, 128, "PCI,card:%d", card_idx);
                res = sv_openex(&sv, name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
                printf("\n");
        }
        printf("\n");
}

static void display_dvs_run(void *arg)
{
        struct state_hdsp *s = (struct state_hdsp *)arg;
        int res;

        while (1) {
                pthread_mutex_lock(&s->lock);

                while (s->work_to_do == FALSE) {
                        s->worker_waiting = TRUE;
                        pthread_cond_wait(&s->worker_cv, &s->lock);
                        s->worker_waiting = FALSE;
                }

                s->display_buffer = s->tmp_buffer;
                s->work_to_do = FALSE;

                if (s->boss_waiting) {
                        pthread_cond_signal(&s->boss_cv);
                }

                if(s->should_exit) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                /* audio - copy appropriate amount of data from ring buffer */
                if(s->play_audio) {
                        int read_b;
                        read_b = ring_buffer_read(s->audio_ring_buffer, s->audio_fifo_data,
                                        s->audio_fifo_required_size);
                        if(read_b != s->audio_fifo_required_size) {
                                fprintf(stderr, "[dvs] Audio buffer underflow\n");
                        }
                }
                res =
                    sv_fifo_putbuffer(s->sv, s->fifo, s->display_buffer, NULL);
                if (res != SV_OK) {
                        fprintf(stderr, "Error %s\n", sv_geterrortext(res));
                        /*  Here is probably bug in recent DVS SDK (4.2.1.1) that triggers
                         *  this error multiple times after device reconfiguration. After few
                         *  frames it disappears.
                         *  TODO: If ther is no longer the error, remove this comment and exit here.
                         */
                }

                gettimeofday(&s->t, NULL);
                double seconds = tv_diff(s->t, s->t0);    

                s->frames++;

                if (seconds >= 5) {
                    float fps  = s->frames / seconds;
                    log_msg(LOG_LEVEL_INFO, "[DVS disp.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                    s->t0 = s->t;
                    s->frames = 0;
                }  
        }
}

static struct video_frame *
display_dvs_getf(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        int res;

        assert(s->magic == HDSP_MAGIC);

	if(s->mode != NULL) {
                int fifo_flags = SV_FIFO_FLAG_FLUSH | SV_FIFO_FLAG_NODMAADDR;

                if(!s->play_audio)
                        fifo_flags |= SV_FIFO_FLAG_VIDEOONLY;
                else {
                        pthread_mutex_lock(&s->audio_reconf_lock);
                        if(s->audio_reconf_pending) {
                                s->audio_reconf_possible = TRUE;
                                pthread_cond_signal(&s->audio_reconf_possible_cv);
                        }
                        while(!s->audio_reconf_done)
                                pthread_cond_wait(&s->audio_reconf_done_cv, &s->audio_reconf_lock);
                        s->audio_reconf_possible = FALSE;
                        pthread_mutex_unlock(&s->audio_reconf_lock);
                }
		/* Prepare the new RTP buffer... */
		res =
		    sv_fifo_getbuffer(s->sv, s->fifo, &s->fifo_buffer, NULL,
				      fifo_flags);
		if (res != SV_OK) {
			fprintf(stderr, "Error %s\n", sv_geterrortext(res));
			return NULL;
		}      

		s->bufs_index = (s->bufs_index + 1) % 2;
		s->tile->data = s->bufs[s->bufs_index];
		assert(s->tile->data != NULL);
		s->fifo_buffer->video[0].addr = s->tile->data;
		s->fifo_buffer->video[0].size = s->tile->data_len;

                if(s->play_audio) {
                        s->audio_fifo_required_size = s->fifo_buffer->audio[0].size;
                        s->fifo_buffer->audio[0].addr[0] = s->audio_fifo_data;
                }
	}
	s->first_run = FALSE;

        return s->frame;
}

static int display_dvs_putf(void *state, struct video_frame *frame, int flags)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        assert(s->magic == HDSP_MAGIC);

        pthread_mutex_lock(&s->lock);
        if(s->work_to_do && flags == PUTF_NONBLOCK) {
                pthread_mutex_unlock(&s->lock);
                return EWOULDBLOCK;
        }

        /* Wait for the worker to finish... */
        while (s->work_to_do) {
                s->boss_waiting = TRUE;
                pthread_cond_wait(&s->boss_cv, &s->lock);
                s->boss_waiting = FALSE;
        }

        // pass poisoned pill
        if(!frame) {
                s->should_exit = true;
        }

        /* ...and give it more to do... */
        s->tmp_buffer = s->fifo_buffer;
        s->fifo_buffer = NULL;
        s->work_to_do = TRUE;

        /* ...and signal the worker */
        if (s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
        }
        pthread_mutex_unlock(&s->lock);

        return 0;
}

static int display_dvs_reconfigure(void *state,
                                struct video_desc desc)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        int i, res;
        int hd_video_mode;
        
        /* Wait for the worker to finish... */
        while (!s->worker_waiting);

        s->frame->color_spec = desc.color_spec;
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->tile->width = desc.width;
        s->tile->height = desc.height;
        
        if(s->mode_set_manually) return TRUE;

        s->mode = NULL;
        for(i=0; hdsp_mode_table[i].width != 0; i++) {
                if(hdsp_mode_table[i].width == desc.width &&
                   hdsp_mode_table[i].height == desc.height &&
                   fabs(desc.fps - hdsp_mode_table[i].fps) < 0.01 ) {
                        if ((desc.interlacing == INTERLACED_MERGED && hdsp_mode_table[i].aux == AUX_INTERLACED ) ||
                                        (desc.interlacing == PROGRESSIVE && hdsp_mode_table[i].aux == AUX_PROGRESSIVE) ||
                                        (desc.interlacing == SEGMENTED_FRAME && hdsp_mode_table[i].aux == AUX_SF)) {
                                s->mode = &hdsp_mode_table[i];
                                break;
                        }
                }
        }

        if(s->mode == NULL) {
                fprintf(stderr, "Reconfigure failed. Expect troubles pretty soon..\n"
                                "\tRequested: %dx%d, color space %d, fps %f,%s\n",
                                desc.width, desc.height, desc.color_spec, desc.fps, 
                                get_interlacing_description(desc.interlacing));
                return FALSE;
        }

        hd_video_mode = SV_MODE_STORAGE_FRAME;

        switch(s->frame->color_spec) {
                case DVS10:
                        hd_video_mode |= SV_MODE_COLOR_YUV422 | SV_MODE_NBIT_10BDVS;
                        break;
                case UYVY:
                        hd_video_mode |= SV_MODE_COLOR_YUV422;
                        break;
                case RGBA:
                        hd_video_mode |= SV_MODE_COLOR_RGBA;
                        break;
                case RGB:
                        hd_video_mode |= SV_MODE_COLOR_RGB_RGB;
                        break;
                default:
                        fprintf(stderr, "[dvs] Unsupported video codec passed!");
                        return FALSE;
        }

        hd_video_mode |= s->mode->mode;
        //s->hd_video_mode |= SV_MODE_AUDIO_NOAUDIO;

        if(s->fifo)
                sv_fifo_free(s->sv, s->fifo);

        //res = sv_videomode(s->sv, hd_video_mode);
        res = sv_option(s->sv, SV_OPTION_VIDEOMODE, hd_video_mode);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot set videomode %s\n", sv_geterrortext(res));
                return FALSE;
        }
        res = sv_sync_output(s->sv, SV_SYNCOUT_BILEVEL);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot enable sync-on-green %s\n",
                          sv_geterrortext(res));
                return FALSE;
        }


        res = sv_fifo_init(s->sv, &s->fifo, 0, /* must be zero for output */
                        0, /* obsolete, must be zero */
                        SV_FIFO_DMA_ON,
                        SV_FIFO_FLAG_NODMAADDR, /* SV_FIFO_FLAG_* */
                        0); /* default maximal numer of FIFO buffer frames */
        if (res != SV_OK) {
                fprintf(stderr, "Cannot initialize video display FIFO %s\n",
                          sv_geterrortext(res));
                return FALSE;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot start video display FIFO  %s\n",
                          sv_geterrortext(res));
                return FALSE;
        }

        s->tile->data_len = vc_get_linesize(s->tile->width, desc.color_spec) *
                s->tile->height;

        free(s->bufs[0]);
        free(s->bufs[1]);
        s->bufs[0] = malloc(s->tile->data_len);
        s->bufs[1] = malloc(s->tile->data_len);
        s->bufs_index = 0;
        memset(s->bufs[0], 0, s->tile->data_len);
        memset(s->bufs[1], 0, s->tile->data_len);

	if(!s->first_run)
		display_dvs_getf(s); /* update s->frame.data */

        return TRUE;
}

static void display_dvs_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        UNUSED(deleter);
        *available_cards = NULL;
        *count = 0;
}


static void *display_dvs_init(struct module *parent, const char *cfg, unsigned int flags)
{
        UNUSED(parent);
        struct state_hdsp *s;
        int i;
        char *name = strdup("");
        int res;
        char *fmt = NULL;

        if (cfg && strcmp(cfg, "help") == 0) {
                show_help();
                free(name);
                return &display_init_noerr;
        }

        s = (struct state_hdsp *)calloc(1, sizeof(struct state_hdsp));
        s->magic = HDSP_MAGIC;

        s->mode_set_manually = FALSE;
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

        gettimeofday(&s->t0, NULL);

        if (cfg != NULL) {
                fmt = strdup(cfg);
                if(strncmp(fmt, "PCI", 3) == 0) {
                        free(name);
                        name = strdup(fmt);
                } else {
                        char *tmp;
                        int mode_index;

                        tmp = strtok(fmt, ":");

                        mode_index = atoi(tmp);
                        tmp = strtok(NULL, ":");
                        
                        if (tmp) {
                                s->frame->color_spec = get_codec_from_name(tmp);
                                if (s->frame->color_spec == VIDEO_CODEC_NONE) {
                                        fprintf(stderr, "dvs: unknown codec: %s\n", tmp);
                                        goto error;
                                }
                                for(i=0; hdsp_mode_table[i].width != 0; i++) {
                                        if(hdsp_mode_table[i].mode == mode_index) {
                                                s->mode = &hdsp_mode_table[i];
                                                break;
                                        }
                                }
                                if(s->mode == NULL) {
                                        fprintf(stderr, "dvs: unknown video mode: %d\n", mode_index);
                                        goto error;
                               }
                                tmp = strtok(NULL, ":");
                                if(tmp)
                                {
                                        if (strtok(NULL, ":") != NULL) { // DVS device name contains ':'
                                                                         // which was removed by strtok
                                                tmp[strlen(tmp)] = ':';
                                                free(name);
                                                name = strdup(tmp);
                                        } else {
                                                fprintf(stderr, "dvs: Malformed device name!");
                                                goto error;
                                        }
                                }
                        }
                }
                free(fmt);
                fmt = NULL;
        }

        s->audio_ring_buffer = NULL;
        s->audio_fifo_data = NULL;
        if(flags & DISPLAY_FLAG_AUDIO_EMBEDDED) {
                s->play_audio = TRUE;
        } else {
                s->play_audio = FALSE;
                sv_option(s->sv, SV_OPTION_AUDIOMUTE, TRUE);
        }
        
        /* Start the display thread... */
        res = sv_openex(&s->sv, name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot open DVS display device.\n");
                fprintf(stderr, "Error %s\n", sv_geterrortext(res));
                goto error;
        }

        s->worker_waiting = TRUE;
	s->first_run = TRUE;

        if(s->mode) {
                struct video_desc desc;
                desc.width = s->mode->width;
                desc.height = s->mode->height;
                desc.color_spec = s->frame->color_spec;
                switch(s->mode->aux) {
                        case AUX_INTERLACED:
                                desc.interlacing = INTERLACED_MERGED;
                                break;
                        case AUX_PROGRESSIVE:
                                desc.interlacing = PROGRESSIVE;
                                break;
                        case AUX_SF:
                                desc.interlacing = SEGMENTED_FRAME;
                                break;
                        default:
                                /* could not reach here */
                                abort();
                }
                desc.fps = s->mode->fps;

                display_dvs_reconfigure(s,
                                desc);
                s->mode_set_manually = TRUE;
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        s->work_to_do = FALSE;
        s->boss_waiting = FALSE;
        s->worker_waiting = FALSE;
        s->display_buffer = NULL;

        pthread_mutex_init(&s->audio_reconf_lock, NULL);
        pthread_cond_init(&s->audio_reconf_possible_cv, NULL);
        pthread_cond_init(&s->audio_reconf_done_cv, NULL);
        s->audio_reconf_possible = FALSE;
        s->audio_reconf_pending = FALSE;
        s->audio_reconf_done = FALSE;

        /*if (pthread_create(&(s->thread_id), NULL, display_thread_hd, (void *)s)
            != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/
        
        free(name);
        return (void *)s;

error:
        free(name);
        free(fmt);
        vf_free(s->frame);
        free(s);
        return NULL;
}

static void display_dvs_done(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        sv_fifo_free(s->sv, s->fifo);
        sv_close(s->sv);
        vf_free(s->frame);
        free(s);
}

static int display_dvs_get_property(void *state, int property, void *val, size_t *len)
{
        codec_t codecs[] = {DVS10, UYVY, RGBA, RGB};
        int rgb_shift[] = {0, 8, 16};

        UNUSED(state);
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                desc->ch_count = 2;
                                if (desc->ch_count > 2) {
                                        desc->ch_count = 16;
                                }
                                desc->codec = AC_PCM;
                                desc->bps = 4;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

/*
 * AUDIO
 */
static void display_dvs_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        
        if(!s->play_audio)
                return;

        ring_buffer_write(s->audio_ring_buffer, frame->data, frame->data_len);
}

static int display_dvs_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        int ret;
        struct state_hdsp *s = (struct state_hdsp *)state;
        int res = SV_OK;
        
        pthread_mutex_lock(&s->audio_reconf_lock);
        s->audio_reconf_pending = TRUE;
        while(!s->audio_reconf_possible)
                pthread_cond_wait(&s->audio_reconf_possible_cv, &s->audio_reconf_lock);

        if(s->sv)
                sv_fifo_free(s->sv, s->fifo);

        ring_buffer_destroy(s->audio_ring_buffer);
        free(s->audio_fifo_data);
        s->audio_ring_buffer = NULL;
        s->audio_fifo_data = NULL;
                
        if (quant_samples != 32) {
                fprintf(stderr, "[dvs] Unsupported number of audio samples: %d\n",
                                quant_samples);
                goto error;
        }
        
        if (channels != 2 && channels != 16) {
                fprintf(stderr, "[DVS] requested channel count isn't supported: "
                        "%d\n", channels);
                goto error;
        }
        
        if(sample_rate != 48000) {
                fprintf(stderr, "[dvs] Unsupported audio sample rate: %d\n",
                                sample_rate);
                goto error;
        }

        res = sv_option(s->sv, SV_OPTION_AUDIOCHANNELS, channels/2); /* channels are in pairs ! */
        if (res != SV_OK) {
                goto error;
        }
        res = sv_option(s->sv, SV_OPTION_AUDIOFREQ, sample_rate);
        if (res != SV_OK) {
                goto error;
        }
        res = sv_option(s->sv, SV_OPTION_AUDIOBITS, 32);
        if (res != SV_OK) {
                goto error;
        }        

        int ring_buffer_size = 5 * sizeof(int32_t) * channels * sample_rate;
        s->audio_ring_buffer = ring_buffer_init(ring_buffer_size);
        
        s->audio_fifo_data = (char *) calloc(1, 
                        ring_get_size(s->audio_ring_buffer));

        res = sv_fifo_init(s->sv, &s->fifo, 0, /* must be zero for output */
                        0, /* obsolete, must be zero */
                        SV_FIFO_DMA_ON,
                        SV_FIFO_FLAG_NODMAADDR, /* SV_FIFO_FLAG_* */
                        0); /* default maximal numer of FIFO buffer frames */
        if (res != SV_OK) {
                fprintf(stderr, "Cannot initialize audio FIFO %s\n",
                          sv_geterrortext(res));
                goto error;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot start audio FIFO  %s\n",
                          sv_geterrortext(res));
                goto error;
        }

        s->audio_reconf_done = FALSE;

        ret = TRUE;
        goto unlock;
error:
        fprintf(stderr, "Setting audio error  %s\n",
                  sv_geterrortext(res));
        s->play_audio = FALSE;
        ret = FALSE;

unlock:
        s->audio_reconf_done = TRUE;
        pthread_cond_signal(&s->audio_reconf_done_cv);
        pthread_mutex_unlock(&s->audio_reconf_lock);

        return ret;
}

static const struct video_display_info display_dvs_info = {
        display_dvs_probe,
        display_dvs_init,
        display_dvs_run,
        display_dvs_done,
        display_dvs_getf,
        display_dvs_putf,
        display_dvs_reconfigure,
        display_dvs_get_property,
        display_dvs_put_audio_frame,
        display_dvs_reconfigure_audio,
};

REGISTER_MODULE(dvs, &display_dvs_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

#endif                          /* HAVE_DVS */

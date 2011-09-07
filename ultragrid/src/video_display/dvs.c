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
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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

#ifdef HAVE_DVS           /* From config.h */

#include "debug.h"
#include "video_display.h"
#include "video_display/dvs.h"
#include "video_codec.h"
#include "audio/audio.h"
#include "tv.h"

#include "dvs_clib.h"           /* From the DVS SDK */
#include "dvs_fifo.h"           /* From the DVS SDK */

#define HDSP_MAGIC	0x12345678

extern int should_exit;

/* TODO: set properties for commented-out formats (if known) */

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
        {SV_MODE_SMPTE274_30I, 1920, 1038, 30.0, AUX_INTERLACED},  /*              1920x1038 30.00hz Interlaced      */
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
        
        {SV_MODE_FILM2K_2048x1556_25sF, 2048, 1556, 25.0, AUX_SF},
        
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
        struct audio_frame audio;
        struct video_frame frame;
        const hdsp_mode_table_t *mode;
        unsigned first_run:1;
        unsigned play_audio:1;

        pthread_mutex_t audio_reconf_lock;
        pthread_cond_t audio_reconf_possible_cv;
        pthread_cond_t audio_reconf_done_cv;
        volatile int audio_reconf_pending;
        volatile int audio_reconf_possible;
        volatile int audio_reconf_done;
        char *audio_device_data; /* ring buffer holding data in format suitable
                                    for playing on DVS card */
        int audio_device_data_len;
        volatile int audio_start; /* start of the above ring buffer in bytes */
        volatile int audio_end; /* its end */
        char *audio_fifo_data; /* temporary memory for the data that gets immediatelly
                                    decoded */
        int audio_fifo_required_size;
};

static void show_help(void);
static void get_sub_frame(void *s, int x, int y, int w, int h, struct video_frame *out);
void display_dvs_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);
struct audio_frame * display_dvs_get_audio_frame(void *state);
void display_dvs_put_audio_frame(void *state, const struct audio_frame *frame);

static void show_help(void)
{
        int i;
        sv_handle *sv = sv_open("");
        if (sv == NULL) {
                printf
                    ("Unable to open grabber: sv_open() failed (no card present or driver not loaded?)\n");
                return;
        }
	printf("DVS options:\n\n");
	printf("\t[mode:codec] | help\n");
	printf("\t(the mode needn't to be set and you shouldn't to want set it)\n\n");
	printf("\tSupported modes:\n");
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
	printf("\n");
	show_codec_help("dvs");
	sv_close(sv);
}

void display_dvs_run(void *arg)
{
        struct state_hdsp *s = (struct state_hdsp *)arg;
        int res;

        while (!should_exit) {
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
                pthread_mutex_unlock(&s->lock);
                
                /* audio - copy appropriate amount of data from s->audio_device_data */
                if(s->play_audio) {
                        int bytes_in_buffer;
                        int audio_end = s->audio_end; /* to avoid changes under our hands */

                        bytes_in_buffer = audio_end - s->audio_start;
                        if(bytes_in_buffer < 0) bytes_in_buffer += s->audio_device_data_len;

                        if(bytes_in_buffer >= s->audio_fifo_required_size) {
                                if(s->audio_start + s->audio_fifo_required_size <= s->audio_device_data_len) {
                                        memcpy(s->audio_fifo_data, s->audio_device_data +
                                                        s->audio_start, s->audio_fifo_required_size);
                                        s->audio_start = (s->audio_start + s->audio_fifo_required_size) 
                                                % s->audio_device_data_len;
                                } else {
                                        int to_end = s->audio_device_data_len - s->audio_start;
                                        memcpy(s->audio_fifo_data, s->audio_device_data +
                                                        s->audio_start, to_end);
                                        memcpy(s->audio_fifo_data + to_end, s->audio_device_data, 
                                                        s->audio_fifo_required_size - to_end);
                                        s->audio_start = s->audio_fifo_required_size - to_end;
                                }
                        } /* otherwise - do not copy anything, we'll need some (small) buffer then */
                }
                res =
                    sv_fifo_putbuffer(s->sv, s->fifo, s->display_buffer, NULL);
                if (res != SV_OK) {
                        debug_msg("Error %s\n", sv_geterrortext(res));
                        return;
                }
        }
}

struct video_frame *
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
		s->frame.data = s->bufs[s->bufs_index];
		assert(s->frame.data != NULL);
		s->fifo_buffer->video[0].addr = s->frame.data;
		s->fifo_buffer->video[0].size = s->frame.data_len;

                if(s->play_audio) {
                        s->audio_fifo_required_size = s->fifo_buffer->audio[0].size;
                        s->fifo_buffer->audio[0].addr[0] = s->audio_fifo_data;
                }
	}
	s->first_run = FALSE;

        return &s->frame;
}

int display_dvs_putf(void *state, char *frame)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        UNUSED(frame);

        assert(s->magic == HDSP_MAGIC);

        pthread_mutex_lock(&s->lock);
        /* Wait for the worker to finish... */
        while (s->work_to_do) {
                s->boss_waiting = TRUE;
                pthread_cond_wait(&s->boss_cv, &s->lock);
                s->boss_waiting = FALSE;
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

        return TRUE;
}

static void
reconfigure_screen(void *state, unsigned int width, unsigned int height,
                                   codec_t color_spec, double fps, int aux)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        int i, res;
        int hd_video_mode;
        /* Wait for the worker to finish... */
        while (!s->worker_waiting);

        s->mode = NULL;
        for(i=0; hdsp_mode_table[i].width != 0; i++) {
                if(hdsp_mode_table[i].width == width &&
                   hdsp_mode_table[i].height == height &&
                   aux & hdsp_mode_table[i].aux &&
                   fabs(fps - hdsp_mode_table[i].fps) < 0.01 ) {
                    s->mode = &hdsp_mode_table[i];
                        break;
                }
        }

        if(s->mode == NULL) {
                fprintf(stderr, "Reconfigure failed. Expect troubles pretty soon..\n"
                                "\tRequested: %dx%d, color space %d, fps %f, aux: %d\n",
                                width, height, color_spec, fps, aux);
                return;
        }

        s->frame.color_spec = color_spec;
        s->frame.width = width;
        s->frame.height = height;
        s->frame.dst_bpp = get_bpp(color_spec);
        s->frame.src_bpp = s->frame.dst_bpp; /* memcpy */
        s->frame.fps = fps;
        s->frame.aux = aux;

        hd_video_mode = SV_MODE_COLOR_YUV422 | SV_MODE_STORAGE_FRAME;

        if (s->frame.color_spec == DVS10) {
                hd_video_mode |= SV_MODE_NBIT_10BDVS;
        }

        hd_video_mode |= s->mode->mode;
        //s->hd_video_mode |= SV_MODE_AUDIO_NOAUDIO;

        if(s->fifo)
                sv_fifo_free(s->sv, s->fifo);

        //res = sv_videomode(s->sv, hd_video_mode);
        res = sv_option(s->sv, SV_OPTION_VIDEOMODE, hd_video_mode);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot set videomode %s\n", sv_geterrortext(res));
                return;
        }
        res = sv_sync_output(s->sv, SV_SYNCOUT_BILEVEL);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot enable sync-on-green %s\n",
                          sv_geterrortext(res));
                return;
        }


        res = sv_fifo_init(s->sv, &s->fifo, 0, /* must be zero for output */
                        0, /* obsolete, must be zero */
                        SV_FIFO_DMA_ON,
                        SV_FIFO_FLAG_NODMAADDR, /* SV_FIFO_FLAG_* */
                        0); /* default maximal numer of FIFO buffer frames */
        if (res != SV_OK) {
                fprintf(stderr, "Cannot initialize video display FIFO %s\n",
                          sv_geterrortext(res));
                return;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot start video display FIFO  %s\n",
                          sv_geterrortext(res));
                return;
        }

        s->frame.data_len = s->frame.width * s->frame.height * s->frame.dst_bpp;
        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.src_linesize = s->frame.dst_linesize;
        s->frame.dst_pitch = s->frame.dst_linesize;

        free(s->bufs[0]);
        free(s->bufs[1]);
        s->bufs[0] = malloc(s->frame.data_len);
        s->bufs[1] = malloc(s->frame.data_len);
        s->bufs_index = 0;
        memset(s->bufs[0], 0, s->frame.data_len);
        memset(s->bufs[1], 0, s->frame.data_len);

	if(!s->first_run)
		display_dvs_getf(s); /* update s->frame.data */
}


void *display_dvs_init(char *fmt, unsigned int flags)
{
        struct state_hdsp *s;
        int i;

        s = (struct state_hdsp *)calloc(1, sizeof(struct state_hdsp));
        s->magic = HDSP_MAGIC;

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
			show_help();

                        return 0;
                }

                char *tmp;
		int mode_index;

                tmp = strtok(fmt, ":");

                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        free(s);
                        return 0;
                }
                mode_index = atoi(tmp);
                tmp = strtok(NULL, ":");
                if (tmp) {
                        s->frame.color_spec = 0xffffffff;
                        for (i = 0; codec_info[i].name != NULL; i++) {
                                if (strcmp(tmp, codec_info[i].name) == 0) {
                                        s->frame.color_spec = codec_info[i].codec;
                                        s->frame.src_bpp = codec_info[i].bpp;
                                }
                        }
                        if (s->frame.color_spec == 0xffffffff) {
                                fprintf(stderr, "dvs: unknown codec: %s\n", tmp);
                                free(s);
                                return 0;
                        }
                        for(i=0; hdsp_mode_table[i].width != 0; i++) {
                                if(hdsp_mode_table[i].mode == mode_index) {
                                        s->mode = &hdsp_mode_table[i];
                                        break;
                                }
                        }
                        if(s->mode == NULL) {
                                fprintf(stderr, "dvs: unknown video mode: %d\n", mode_index);
                                free(s);
                                return 0;
                       }
                }
        }

        s->audio.state = s;
        s->audio.data = NULL;
        s->audio_device_data = NULL;
        s->audio_fifo_data = NULL;
        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                s->play_audio = TRUE;
                s->audio.reconfigure_audio = display_dvs_reconfigure_audio;
                s->audio.ch_count = 0;
        } else {
                s->play_audio = FALSE;
                sv_option(s->sv, SV_OPTION_AUDIOMUTE, TRUE);
        }
        s->audio_start = 0;
        s->audio_end = 0;
        
        /* Start the display thread... */
        s->sv = sv_open("");
        if (s->sv == NULL) {
                fprintf(stderr, "Cannot open DVS display device.\n");
                return NULL;
        }

        s->worker_waiting = TRUE;
	s->first_run = TRUE;

        if(s->mode) {
                reconfigure_screen(s, s->mode->width, s->mode->height, s->frame.color_spec, s->mode->fps, s->mode->aux);
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
        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)reconfigure_screen;
        s->frame.get_sub_frame = (get_sub_frame_t) get_sub_frame;
        s->frame.decoder = (decoder_t)memcpy;     
        return (void *)s;
}

void display_dvs_done(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        sv_fifo_free(s->sv, s->fifo);
        sv_close(s->sv);
        free(s);
}

display_type_t *display_dvs_probe(void)
{
        display_type_t *dtype;

        dtype = malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_DVS_ID;
                dtype->name = "dvs";
                dtype->description = "DVS card";
        }
        return dtype;
}

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        UNUSED(h);

        memcpy(out, &s->frame, sizeof(struct video_frame));
        out->data +=
                y * s->frame.dst_pitch +
                (size_t) (x * s->frame.dst_bpp);
        out->src_linesize =
                vc_getsrc_linesize(w, out->color_spec);
        out->dst_linesize =
                w * out->dst_bpp;

}

/*
 * AUDIO
 */
struct audio_frame * display_dvs_get_audio_frame(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        
        if(!s->play_audio)
                return NULL;
        return &s->audio;
}

void display_dvs_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        int i;
        char *src = s->audio.data;
        char *dst = s->audio_device_data + s->audio_end;
        const int dst_data_len = (s->audio.data_len / s->audio.bps) * 4;
        
        int len_to_end; /* size in samples(!) for every channel */
        int len_from_start;
        if(s->audio_end + dst_data_len <= s->audio_device_data_len) {
                len_to_end = s->audio.data_len / s->audio.bps;
                len_from_start = 0;
        } else {
                len_to_end = (dst_data_len - s->audio_end) / 4;
                len_from_start = s->audio.data_len / s->audio.bps - len_to_end;
        }

        for(i = 0; i < len_to_end; i++) {
                *((int *) dst) = *((int *) src) << (32 - s->audio.bps * 8);
                src += s->audio.bps;
                dst += 4;
        }
        
        dst = s->audio_device_data;
        src = s->audio.data + len_to_end * s->audio.bps;
        
        for(i = 0; i < len_from_start; i++) {
                *((int *) dst) = *((int *) src) << (32 - s->audio.bps * 8);
                src += s->audio.bps;
                dst += 4;
        }
        
        s->audio_end = (s->audio_end + ((s->audio.data_len / s->audio.bps) * 4)) % s->audio_device_data_len;
}

void display_dvs_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_hdsp *s = (struct state_hdsp *)state;
        int res;
        
        pthread_mutex_lock(&s->audio_reconf_lock);
        s->audio_reconf_pending = TRUE;
        while(!s->audio_reconf_possible)
                pthread_cond_wait(&s->audio_reconf_possible_cv, &s->audio_reconf_lock);

        if(s->sv)
                sv_fifo_free(s->sv, s->fifo);

        free(s->audio.data);
        free(s->audio_device_data);
        free(s->audio_fifo_data);
        s->audio.data = NULL;
        s->audio_device_data = NULL;
                
        s->audio.bps = quant_samples / 8;
        s->audio.sample_rate = sample_rate;
        s->audio.ch_count = channels;
        
        if(quant_samples != 8 && quant_samples != 16 && quant_samples != 24
                        && quant_samples != 32) {
                fprintf(stderr, "[dvs] Unsupported number of audio samples: %d\n",
                                quant_samples);
                goto error;
        }
        
        if(channels != 2 && channels != 16) {
                fprintf(stderr, "[dvs] Unsupported number of audio channels: %d\n",
                                channels);
                goto error;
        }
        
        if(sample_rate != 48000) {
                fprintf(stderr, "[dvs] Unsupported audio sample rate: %d\n",
                                sample_rate);
                goto error;
        }

        res = sv_option(s->sv, SV_OPTION_AUDIOCHANNELS, channels/2); /* items in pairs */
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

        s->audio.max_size = 5 * (quant_samples / 8) * channels *
                        sample_rate;                
        s->audio.data = (char *) malloc (s->audio.max_size);
        s->audio_device_data_len = 5 * 4 * channels * sample_rate;
        s->audio_device_data = (char *) malloc(s->audio_device_data_len);
        s->audio_fifo_data = (char *) calloc(1, s->audio_device_data_len);

        res = sv_fifo_init(s->sv, &s->fifo, 0, /* must be zero for output */
                        0, /* obsolete, must be zero */
                        SV_FIFO_DMA_ON,
                        SV_FIFO_FLAG_NODMAADDR, /* SV_FIFO_FLAG_* */
                        0); /* default maximal numer of FIFO buffer frames */
        if (res != SV_OK) {
                fprintf(stderr, "Cannot initialize video display FIFO %s\n",
                          sv_geterrortext(res));
                goto error;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                fprintf(stderr, "Cannot start video display FIFO  %s\n",
                          sv_geterrortext(res));
                goto error;
        }

        s->audio_reconf_done = FALSE;

        goto unlock;
error:
        fprintf(stderr, "Setting audio error  %s\n",
                  sv_geterrortext(res));
        s->audio.max_size = 0;
        s->play_audio = FALSE;

unlock:
        s->audio_reconf_done = TRUE;
        pthread_cond_signal(&s->audio_reconf_done_cv);
        pthread_mutex_unlock(&s->audio_reconf_lock);
}

#endif                          /* HAVE_DVS */

/*
 * FILE:    v_codec.c
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
#include <stdio.h>
#include "v_codec.h"

const struct codec_info_t codec_info[] = {
        {RGBA, "RGBA", 0, 0, 4.0, 1},
        {UYVY, "UYVY", 846624121, 0, 2, 0},
        {Vuy2, "2vuy", '2vuy', 0, 2, 0},
        {DVS8, "DVS8", 0, 0, 2, 0},
        {R10k, "R10k", 1378955371, 0, 4, 1},
        {v210, "v210", 1983000880, 48, 8.0/3.0, 0},
        {DVS10, "DVS10", 0, 48, 8.0/3.0, 0},
        {0, NULL, 0, 0, 0.0, 0}};


void
show_codec_help(void)
{
        printf("\tSupported codecs:\n");
        printf("\t\t8bits\n");
        printf("\t\t\t'RGBA' - Red Green Blue Alpha 32bit\n");
        printf("\t\t\t'UYVY' - YUV 4:2:2\n");
        printf("\t\t\t'2vuy' - YUV 4:2:2\n");
        printf("\t\t\t'DVS8' - Centaurus 8bit YUV 4:2:2\n");
        printf("\t\t10bits\n");
        printf("\t\t\t'R10k' - RGB 4:4:4\n");
        printf("\t\t\t'v210' - YUV 4:2:2\n");
        printf("\t\t\t'DVS10' - Centaurus 10bit YUV 4:2:2\n");
}

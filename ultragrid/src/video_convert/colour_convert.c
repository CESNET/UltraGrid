/*
 * FILE:   colour_convert.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
 * Copyright (c) 2001-2002 University of Southern California
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
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "colour_convert.h"

void 
yuv_to_rgb(signed char *frame_yuv, signed char *frame_rgb, int offset, int size)
{
	signed char	 Cb, Cr; 
	signed char	 Y0, Y1, fB, fG, fR;
	signed char	*yuv, *yuv_max, *rgb;

	yuv     = frame_yuv;
	yuv_max = yuv + size * 2;
	rgb     = frame_rgb + offset * 4;
	do {
		 Cb = *yuv++ - 128;
		 Y0 = *yuv++;
		 Cr = *yuv++ - 128;
		 fB = (1816 * Cb) >> 10;
		*rgb++ = Y0 + fB;		/* B */
		 fG = (Cr >> 1) - (Cb >> 3);
		 fR = (1540 * Cr) >> 10;
		*rgb++ = Y0 - fG;		/* G */
		*rgb++ = Y0 + fR;		/* R */
		 Y1 = *yuv++;
		 rgb++;
		*rgb++ = Y1 + fB;		/* B */
		*rgb++ = Y1 - fG;		/* G */
		*rgb++ = Y1 + fR;		/* R */
		 rgb++;
	} while (yuv != yuv_max);
}


/*
 * FILE:   video_capture/testcard_common.c
 * AUTHOR: Colin Perkins <csp@csperkins.org
 *         Alvaro Saurin <saurin@dcs.gla.ac.uk>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2006 University of Glasgow
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video.h"
#include "video_capture/testcard_common.h"

const int rect_colors[] = {
        0xff0000ff,
        0xff00ff00,
        0xffff0000,
        0xff00ffff,
        0xffffff00,
        0xffff00ff
};

/**
 * Converts UYVY to v210
 */
unsigned char *tov210(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned int linesize = vc_get_linesize(width, v210);

        unsigned char *dst = (unsigned char *)malloc(linesize * height);
        decoder_t vc_copylineUYVYtoV210 = get_decoder_from_to(UYVY, v210, true);
        assert(vc_copylineUYVYtoV210 != NULL);

        for (unsigned j = 0; j < height; j++) {
                vc_copylineUYVYtoV210(dst, in, linesize, 0, 0, 0);
                dst += linesize;
                in += width * 2;
        }
        return dst;
}

/**
 * @param[in] in buffer in UYVY
 * @retval       buffer in I420 (must be deallocated by the caller)
 * @note
 * Caller must deallocate returned buffer
 */
char *toI420(const char *input, unsigned int width, unsigned int height)
{
        const unsigned char *in = (const unsigned char *) input;
        int w_ch = (width + 1) / 2;
        int h_ch = (height + 1) / 2;
        unsigned char *out = malloc(width * height + 2 * w_ch * h_ch);
        unsigned char *y = out;
        unsigned char *u0 = out + width * height;
        unsigned char *v0 = out + width * height + w_ch * h_ch;
        unsigned char *u1 = u0, *v1 = v0;

        for (unsigned int i = 0; i < height; i += 1) {
                for (unsigned int j = 0; j < ((width + 1) & ~1); j += 2) {
                        // U
                        if (i % 2 == 0) {
                                *u0++ = *in++;
                        } else { // average with every 2nd row
                                *u1 = (*u1 + *in++) / 2;
                                u1++;
                        }
                        // Y
                        *y++ = *in++;
                        // V
                        if (i % 2 == 0) {
                                *v0++ = *in++;
                        } else { // average with every 2nd row
                                *v1 = (*v1 + *in++) / 2;
                                v1++;
                        }
                        // Y
                        if (j + 1 == width) {
                                in++;
                        } else {
                                *y++ = *in++;
                        }
                }
        }
        return (char *) out;
}

void toR10k(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned char *dst = (void *)in;

        decoder_t decoder = get_decoder_from_to(RGBA, R10k, true);
        assert(decoder != NULL);

        for (unsigned j = 0; j < height; j++) {
                decoder(dst, in, width * 4, 0, 0, 0);
                dst += width * 4;
                in += width * 4;
        }
}

char *toRGB(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned int i;
        unsigned char *ret = malloc(width * height * 3);
        for(i = 0; i < height; ++i) {
                vc_copylineRGBAtoRGB(ret + i * width * 3, in + i * width * 4, width * 3, 0, 8, 16);
        }
        return (char *) ret;
}


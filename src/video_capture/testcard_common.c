/*
 * FILE:   testcard_common.c
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

unsigned char *tov210(unsigned char *in, unsigned int width,
                      unsigned int aligned_x, unsigned int height, double bpp)
{
        struct {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } *p;
        unsigned int i, j;

        unsigned int linesize = aligned_x * bpp;

        unsigned char *dst = (unsigned char *)malloc(aligned_x * height * bpp);
        unsigned char *src;
        unsigned char *ret = dst;

        for (j = 0; j < height; j++) {
                p = (void *)dst;
                dst += linesize;
                src = in;
                in += width * 2;
                for (i = 0; i < width; i += 3) {
                        unsigned int u, y, v;

                        u = *(src++);
                        y = *(src++);
                        v = *(src++);

                        p->a = u << 2;
                        p->b = y << 2;
                        p->c = v << 2;
                        p->p1 = 0;

                        p++;

                        u = *(src++);
                        y = *(src++);
                        v = *(src++);

                        p->a = u << 2;
                        p->b = y << 2;
                        p->c = v << 2;
                        p->p1 = 0;

                        p++;
                }
        }
        return ret;
}

void toR10k(unsigned char *in, unsigned int width, unsigned int height)
{
        struct {
                unsigned r:8;

                unsigned gh:6;
                unsigned p1:2;

                unsigned bh:4;
                unsigned p2:2;
                unsigned gl:2;

                unsigned p3:2;
                unsigned p4:2;
                unsigned bl:4;
        } *d;

        unsigned int i, j;

        d = (void *)in;

        for (j = 0; j < height; j++) {
                for (i = 0; i < width; i++) {
                        unsigned int r, g, b;

                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++;

                        d->r = r;
                        d->gh = g >> 2;
                        d->gl = g & 0x3;
                        d->bh = b >> 4;
                        d->bl = b & 0xf;

                        d->p1 = 0;
                        d->p2 = 0;
                        d->p3 = 0;
                        d->p4 = 0;

                        d++;
                }
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


/*
 * FILE:   quicktime.c
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
 * $Revision: 1.8.2.6 $
 * $Date: 2010/02/05 13:59:24 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "tv.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/testcard.h"
#include "host.h"
#include "v_codec.h"
#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>

void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height);
unsigned char * tov210(unsigned char *in, unsigned int width, unsigned int height, double bpp);
void toR10k(unsigned char *in, unsigned int width, unsigned int height);

struct testcard_state {
        struct timeval  last_frame_time;
        int             fps;
        int             count;
        unsigned int    width;
        unsigned int    height;
        int             size;
        char            *frame;
        int             linesize;
        int             pan;
        SDL_Surface     *surface;
        struct timeval  t0;
};

const int rect_colors[] = {
                0xff0000ff,
                0xff00ff00,
                0xffff0000,
                0xff00ffff,
                0xffffff00,
                0xffff00ff };

#define COL_NUM 6

void
rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned int i,j;
        int r,g,b;
        int y,u,v,y1,u1,v1;
        unsigned char *dst;

        dst = in;

        for(j = 0; j < height; j++) {
                for(i = 0; i < width; i+=2) {
                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++; /*skip alpha*/

                        y = r * 0.299 + g * 0.587 + b * 0.114;
                        u = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v = r * 0.5 - g * 0.418688 -b * 0.081312;
                        //y -= 16;
                        if(y > 255)
                                y = 255;
                        if(y < 0)
                                y = 0;
                        if(u < -128)
                                u = -128;
                        if(u > 127)
                                u = 127;
                        if(v < -128)
                                v = -128;
                        if(v > 127)
                                v = 127;
                        u += 128;
                        v += 128;

                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++; /*skip alpha*/
                        
                        y1 = r * 0.299 + g * 0.587 + b * 0.114;
                        u1 = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v1 = r * 0.5 - g * 0.418688 -b * 0.081312;
                        if(y1 > 255)
                                y1 = 255;
                        if(y1 < 0)
                                y1 = 0;
                        if(u1 < -128)
                                u1 = -128;
                        if(u1 > 127)
                                u1 = 127;
                        if(v1 < -128)
                                v1 = -128;
                        if(v1 > 127)
                                v1 = 127;
                        u1 += 128;
                        v1 += 128;
                        
                        *(dst++) = (u + u1)/2;
                        *(dst++) = y;
                        *(dst++) = (v + v1)/2;
                        *(dst++) = y1;
                }
        }
}

unsigned char *
tov210(unsigned char *in, unsigned int width, unsigned int aligned_x, unsigned int height, double bpp)
{
        struct {
                 unsigned a:10;
                 unsigned b:10;
                 unsigned c:10;
                 unsigned p1:2;
        } *p;
        unsigned int i,j;

        unsigned int linesize = aligned_x * bpp;

        unsigned char *dst = (unsigned char*)malloc(aligned_x*height*bpp);
        unsigned char *src;
        unsigned char *ret = dst;

        for(j=0; j < height; j++) {
                p = (void*)dst;
                dst += linesize;
                src = in;
                in += width*2;
                for(i=0; i < width; i+=3) {
                        unsigned int u,y,v;

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

void
toR10k(unsigned char *in, unsigned int width, unsigned int height)
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
        
        unsigned int i,j;

        d = (void*)in;

        for(j = 0; j < height; j++) {
                for(i = 0; i < width; i++) {
                        unsigned int r,g,b;
        
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


void *
vidcap_testcard_init(char *fmt)
{
        struct testcard_state   *s;
        char                    *filename;
        FILE                    *in;
        struct stat             sb;
        unsigned int            i,j;
        unsigned int            rect_size=COL_NUM;
        codec_t                 codec;
        int                     aligned_x;

        if(strcmp(fmt, "help")==0) {
                printf("testcard options:\n");
                printf("\twidth:height:fps:codec[:filename][:p}\n");
                printf("\tp - pan with frame\n");
                show_codec_help();
                return NULL;
        }

        s = malloc(sizeof(struct testcard_state));
        if(!s)
                return NULL;

        char *tmp;

        tmp = strtok(fmt, ":");
        if(!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        s->width = atoi(tmp);
        tmp = strtok(NULL, ":");
        if(!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        s->height = atoi(tmp);
        tmp = strtok(NULL, ":");
        if(!tmp) {
                free(s->frame);
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return NULL;
        }

        s->fps   = atoi(tmp);

        tmp=strtok(NULL, ":");
        if(!tmp) {
                free(s->frame);
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return NULL;
        }

        int h_align=0;
        double bpp=0;

        for(i = 0; codec_info[i].name != NULL; i++) {
                if(strcmp(tmp, codec_info[i].name) == 0) {
                        h_align = codec_info[i].h_align;
                        bpp = codec_info[i].bpp;
                        codec = codec_info[i].codec;
                        break;
                }
        }


        aligned_x = s->width;
        if(h_align) {
                aligned_x = (aligned_x + h_align - 1)/h_align*h_align;
        }

        rect_size = (s->width + rect_size-1) / rect_size;

        s->linesize = aligned_x * bpp;

        s->size = aligned_x*s->height*bpp;

        filename = strtok(NULL, ":");
        if(filename && strcmp(filename, "p") != 0) {
                s->frame = malloc(s->size);
                if(stat(filename, &sb)) {
                        perror("stat");
                        free(s);
                        return NULL;
                }

                in = fopen(filename, "r");

                if(s->size < sb.st_size) {
                        fprintf(stderr, "Error wrong file size for selected "
                                        "resolution and codec. File size %d, "
                                        "computed size %d\n",
                                (int)sb.st_size, s->size);
                        free(s->frame);
                        free(s);
                        return NULL;
                }
        
                if(!in || fread(s->frame, sb.st_size, 1, in) == 0) {
                        fprintf(stderr, "Cannot read file %s\n", filename);
                        free(s->frame);
                        free(s);
                        return NULL;
                }

                fclose(in);
        } else {
                SDL_Rect r;
                int col_num=0;
                s->surface = SDL_CreateRGBSurface(SDL_SWSURFACE, aligned_x, s->height, 
                                32, 0xff, 0xff00, 0xff0000, 0xff000000);
                if(filename && filename[0] == 'p') {
                        s->pan = 48;
                }

                for(j=0; j < s->height; j+=rect_size) {
                        int grey=0xff010101;
                        if(j==rect_size*2) {
                                r.w= s->width;
                                r.h = rect_size/4;
                                r.x = 0;
                                r.y = j;
                                SDL_FillRect(s->surface, &r, 0xffffffff);
                                r.y = j+rect_size*3/4;
                                SDL_FillRect(s->surface, &r, 0);
                        }
                        for(i=0; i < s->width; i+=rect_size) {
                                r.w = rect_size;
                                r.h = rect_size;
                                r.x = i;
                                r.y = j;
                                printf("Fill rect at %d,%d\n", r.x, r.y );
                                if(j!=rect_size*2) {
                                         SDL_FillRect(s->surface, &r, rect_colors[col_num]);
                                         col_num = (col_num+1)%COL_NUM;                                
                                } else {
                                         r.h = rect_size/2;
                                         r.y += rect_size/4;
                                         SDL_FillRect(s->surface, &r, grey);
                                         grey += 0x00010101*(255/COL_NUM);
                                }
                        }
                }
                s->frame = s->surface->pixels;        
                if(codec == UYVY || codec == v210 || codec == Vuy2) {
                        rgb2yuv422((unsigned char*)s->frame, s->width, s->height);
                } 

                if(codec == v210) {
                        s->frame = (char*)tov210((unsigned char*)s->frame, s->width, aligned_x, s->height, bpp);
                }

                if(codec == R10k) {
                        toR10k((unsigned char*)s->frame, s->width, s->height);
                }
        }

        tmp = strtok(NULL, ":");
        if(tmp) {
                if(tmp[0] == 'p') {
                        s->pan = 48;
                }
        }

        s->count = 0;
        gettimeofday(&(s->last_frame_time), NULL);

        printf("Testcard set to %dx%d, bpp %f\n", s->width, s->width, bpp);

        return s;
}

void
vidcap_testcard_done(void *state)
{
        struct testcard_state *s = state;
        if(s->frame != s->surface->pixels)
                free(s->frame);
        if(s->surface)
                SDL_FreeSurface(s->surface);
        free(s);
}

struct video_frame *
vidcap_testcard_grab(void *arg)
{
        struct timeval               curr_time;
        struct testcard_state        *state;
        struct video_frame           *vf;

        state = (struct testcard_state *) arg;

        gettimeofday(&curr_time, NULL);
        if (tv_diff(curr_time, state->last_frame_time) > 1.0/(double)state->fps) {
                state->last_frame_time = curr_time;
                state->count++;

                double seconds = tv_diff(curr_time, state->t0);    
                if (seconds >= 5) {
                        float fps  = state->count / seconds;
                        fprintf(stderr, "%d frames in %g seconds = %g FPS\n", state->count, seconds, fps);
                        state->t0 = curr_time;
                        state->count = 0;
                }

                vf = (struct video_frame *) malloc(sizeof(struct video_frame));
                if (vf != NULL) {
                        char            line[state->linesize*2+state->pan];
                        unsigned int    i;
                        vf->width     = state->width;
                        vf->height    = state->height;
                        vf->data      = state->frame;
                        vf->data_len  = state->size;
                        memcpy(line, state->frame, state->linesize*2+state->pan);
                        for(i=0; i < hd_size_y-3; i++) {
                                memcpy(state->frame+i*state->linesize, 
                                       state->frame+(i+2)*state->linesize+state->pan, state->linesize);
                        }
                        memcpy(state->frame+i*state->linesize,
                               state->frame+(i+2)*state->linesize+state->pan, state->linesize-state->pan);
                        memcpy(state->frame+(hd_size_y-2)*state->linesize-state->pan, 
                               line, state->linesize*2+state->pan);
                        /*if(!(state->count % 2)) {
                                unsigned int *p = state->frame;
                                for(i=0; i < state->linesize*hd_size_y/4; i++) {
                                        *p = *p ^ 0x00ffffffL;
                                        p++;
                                }
                        }*/
                }
                return vf;
        }
        return NULL;
}

struct vidcap_type *
vidcap_testcard_probe(void)
{
        struct vidcap_type        *vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_TESTCARD_ID;
                vt->name        = "testcard";
                vt->description = "Video testcard";
        }
        return vt;
}


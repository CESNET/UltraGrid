/*
 * FILE:   testcard.c
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
#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>

void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height);
unsigned char *tov210(unsigned char *in, unsigned int width, unsigned int align_x,
                      unsigned int height, double bpp);
void toR10k(unsigned char *in, unsigned int width, unsigned int height);

struct testcard_state {
        struct timeval last_frame_time;
        int count;
        int size;
        int pan;
        SDL_Surface *surface;
        struct timeval t0;
        struct video_frame frame;
        struct video_frame *tiles;
};

const int rect_colors[] = {
        0xff0000ff,
        0xff00ff00,
        0xffff0000,
        0xff00ffff,
        0xffffff00,
        0xffff00ff
};

#define COL_NUM 6

void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned int i, j;
        int r, g, b;
        int y, u, v, y1, u1, v1;
        unsigned char *dst;

        dst = in;

        for (j = 0; j < height; j++) {
                for (i = 0; i < width; i += 2) {
                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++;   /*skip alpha */

                        y = r * 0.299 + g * 0.587 + b * 0.114;
                        u = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v = r * 0.5 - g * 0.418688 - b * 0.081312;
                        //y -= 16;
                        if (y > 255)
                                y = 255;
                        if (y < 0)
                                y = 0;
                        if (u < -128)
                                u = -128;
                        if (u > 127)
                                u = 127;
                        if (v < -128)
                                v = -128;
                        if (v > 127)
                                v = 127;
                        u += 128;
                        v += 128;

                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++;   /*skip alpha */

                        y1 = r * 0.299 + g * 0.587 + b * 0.114;
                        u1 = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v1 = r * 0.5 - g * 0.418688 - b * 0.081312;
                        if (y1 > 255)
                                y1 = 255;
                        if (y1 < 0)
                                y1 = 0;
                        if (u1 < -128)
                                u1 = -128;
                        if (u1 > 127)
                                u1 = 127;
                        if (v1 < -128)
                                v1 = -128;
                        if (v1 > 127)
                                v1 = 127;
                        u1 += 128;
                        v1 += 128;

                        *(dst++) = (u + u1) / 2;
                        *(dst++) = y;
                        *(dst++) = (v + v1) / 2;
                        *(dst++) = y1;
                }
        }
}

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

static int configure_tiling(struct testcard_state *s, const char *fmt)
{
        char *tmp, *token, *saveptr;

        if(fmt[1] != '=') return 1;
        s->frame.aux |= AUX_TILED;
        tmp = strdup(&fmt[2]);
        token = strtok_r(tmp, "x", &saveptr);
        s->frame.tile_info.x_count = atoi(token);
        token = strtok_r(NULL, "x", &saveptr);
        s->frame.tile_info.y_count = atoi(token);
        free(tmp);
        if(s->frame.tile_info.x_count < 1u ||
                        s->frame.tile_info.y_count < 1u) {
                fprintf(stderr, "There must be at least 1x1 tile.\n");
                return 1;
        }
        s->tiles = (struct video_frame *) 
                malloc(s->frame.tile_info.x_count *
                                s->frame.tile_info.y_count *
                        sizeof(struct video_frame));
        vf_split(s->tiles, &s->frame, s->frame.tile_info.x_count,
                        s->frame.tile_info.y_count, 1); /*prealloc*/
        return 0;
}

void *vidcap_testcard_init(char *fmt)
{
        struct testcard_state *s;
        char *filename;
        const char *strip_fmt = NULL;
        FILE *in;
        struct stat sb;
        unsigned int i, j;
        unsigned int rect_size = COL_NUM;
        codec_t codec=0;
        int aligned_x;

        if (strcmp(fmt, "help") == 0) {
                printf("testcard options:\n");
                printf("\twidth:height:fps:codec[:filename][:p][:s=XxY]\n");
                printf("\tp - pan with frame\n");
                printf("\ts - split the frames into XxY separate tiles\n");
                show_codec_help("testcard");
                return NULL;
        }

        s = calloc(1, sizeof(struct testcard_state));
        if (!s)
                return NULL;

        char *tmp;

        tmp = strtok(fmt, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        s->frame.width = atoi(tmp);
        tmp = strtok(NULL, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        s->frame.height = atoi(tmp);
        tmp = strtok(NULL, ":");
        if (!tmp) {
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return NULL;
        }

        s->frame.fps = atof(tmp);

        tmp = strtok(NULL, ":");
        if (!tmp) {
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return NULL;
        }

        int h_align = 0;
        double bpp = 0;

        for (i = 0; codec_info[i].name != NULL; i++) {
                if (strcmp(tmp, codec_info[i].name) == 0) {
                        h_align = codec_info[i].h_align;
                        bpp = codec_info[i].bpp;
                        codec = codec_info[i].codec;
                        break;
                }
        }

        s->frame.color_spec = codec;

        if(bpp == 0) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                return NULL;
        }

        aligned_x = s->frame.width;
        if (h_align) {
                aligned_x = (aligned_x + h_align - 1) / h_align * h_align;
        }

        rect_size = (s->frame.width + rect_size - 1) / rect_size;

        s->frame.src_linesize = aligned_x * bpp;

        s->size = aligned_x * s->frame.height * bpp;

        filename = strtok(NULL, ":");
        if (filename && strcmp(filename, "p") != 0
                        && strncmp(filename, "s=", 2ul) != 0) {
                s->frame.data = malloc(s->size);
                if (stat(filename, &sb)) {
                        perror("stat");
                        free(s);
                        return NULL;
                }

                in = fopen(filename, "r");

                if (s->size < sb.st_size) {
                        fprintf(stderr, "Error wrong file size for selected "
                                "resolution and codec. File size %d, "
                                "computed size %d\n", (int)sb.st_size, s->size);
                        free(s->frame.data);
                        free(s);
                        return NULL;
                }

                if (!in || fread(s->frame.data, sb.st_size, 1, in) == 0) {
                        fprintf(stderr, "Cannot read file %s\n", filename);
                        free(s->frame.data);
                        free(s);
                        return NULL;
                }

                fclose(in);
        } else {
                SDL_Rect r;
                int col_num = 0;
                s->surface =
                    SDL_CreateRGBSurface(SDL_SWSURFACE, aligned_x, s->frame.height,
                                         32, 0xff, 0xff00, 0xff0000,
                                         0xff000000);
                if (filename) {
                        if(filename[0] == 'p')
                                s->pan = 48;
                        else if(filename[0] == 's')
                                strip_fmt = filename;
                }

                for (j = 0; j < s->frame.height; j += rect_size) {
                        int grey = 0xff010101;
                        if (j == rect_size * 2) {
                                r.w = s->frame.width;
                                r.h = rect_size / 4;
                                r.x = 0;
                                r.y = j;
                                SDL_FillRect(s->surface, &r, 0xffffffff);
                                r.y = j + rect_size * 3 / 4;
                                SDL_FillRect(s->surface, &r, 0);
                        }
                        for (i = 0; i < s->frame.width; i += rect_size) {
                                r.w = rect_size;
                                r.h = rect_size;
                                r.x = i;
                                r.y = j;
                                printf("Fill rect at %d,%d\n", r.x, r.y);
                                if (j != rect_size * 2) {
                                        SDL_FillRect(s->surface, &r,
                                                     rect_colors[col_num]);
                                        col_num = (col_num + 1) % COL_NUM;
                                } else {
                                        r.h = rect_size / 2;
                                        r.y += rect_size / 4;
                                        SDL_FillRect(s->surface, &r, grey);
                                        grey += 0x00010101 * (255 / COL_NUM);
                                }
                        }
                }
                s->frame.data = s->surface->pixels;
                if (codec == UYVY || codec == v210 || codec == Vuy2) {
                        rgb2yuv422((unsigned char *)s->frame.data, s->frame.width,
                                   s->frame.height);
                }

                if (codec == v210) {
                        s->frame.data =
                            (char *)tov210((unsigned char *)s->frame.data, s->frame.width,
                                           aligned_x, s->frame.height, bpp);
                }

                if (codec == R10k) {
                        toR10k((unsigned char *)s->frame.data, s->frame.width, s->frame.height);
                }
        }

        tmp = strtok(NULL, ":");
        if (tmp) {
                if (tmp[0] == 'p') {
                        s->pan = 48;
                } else if (tmp[0] == 's') {
                        strip_fmt = tmp;
                }
        }
        tmp = strtok(NULL, ":");
        if (tmp) {
                if (tmp[0] == 's') {
                        strip_fmt = tmp;
                }
        }

        s->count = 0;
        gettimeofday(&(s->last_frame_time), NULL);

        printf("Testcard capture set to %dx%d, bpp %f\n", s->frame.width, s->frame.height, bpp);

        s->frame.state = s;
        s->frame.data_len = s->size;

        if(strip_fmt != NULL) {
                if(configure_tiling(s, strip_fmt) != 0)
                        return NULL;
        } else {
                s->frame.aux &= ~AUX_TILED;
        }

        return s;
}

void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = state;
        if (s->frame.data != s->surface->pixels)
                free(s->frame.data);
        if (s->surface)
                SDL_FreeSurface(s->surface);
        if (s->frame.aux & AUX_TILED)
        {
                unsigned int i;
                for (i = 0u; i < s->frame.tile_info.x_count *
                                s->frame.tile_info.y_count; ++i)
                        free(s->tiles[i].data);
                free(s->tiles);
        }
        free(s);
}

struct video_frame *vidcap_testcard_grab(void *arg, int *count)
{
        struct timeval curr_time;
        struct testcard_state *state;

        state = (struct testcard_state *)arg;

        gettimeofday(&curr_time, NULL);
        if (tv_diff(curr_time, state->last_frame_time) >
            1.0 / (double)state->frame.fps) {
                state->last_frame_time = curr_time;
                state->count++;

                double seconds = tv_diff(curr_time, state->t0);
                if (seconds >= 5) {
                        float fps = state->count / seconds;
                        fprintf(stderr, "%d frames in %g seconds = %g FPS\n",
                                state->count, seconds, fps);
                        state->t0 = curr_time;
                        state->count = 0;
                }

                char line[state->frame.src_linesize * 2 + state->pan];
                unsigned int i;
                memcpy(line, state->frame.data,
                       state->frame.src_linesize * 2 + state->pan);
                for (i = 0; i < state->frame.height - 3; i++) {
                        memcpy(state->frame.data + i * state->frame.src_linesize,
                               state->frame.data + (i + 2) * state->frame.src_linesize +
                               state->pan, state->frame.src_linesize);
                }
                memcpy(state->frame.data + i * state->frame.src_linesize,
                       state->frame.data + (i + 2) * state->frame.src_linesize +
                       state->pan, state->frame.src_linesize - state->pan);
                memcpy(state->frame.data +
                       (state->frame.height - 2) * state->frame.src_linesize - state->pan,
                       line, state->frame.src_linesize * 2 + state->pan);
#ifdef USE_EPILEPSY                        
                if(!(state->count % 2)) {
                        unsigned int *p = state->frame.data;
                        for(i=0; i < state->frame.src_linesize*state->frame.height/4; i++) {
                                *p = *p ^ 0x00ffffffL;
                                p++;
                        }
                }
#endif
                if (state->frame.aux & AUX_TILED) {
                        vf_split(state->tiles, &state->frame,
                                       state->frame.tile_info.x_count,
                                       state->frame.tile_info.y_count, 0);
                        *count = state->frame.tile_info.x_count *
                                state->frame.tile_info.y_count;
                        return state->tiles;
                }
                *count = 1;
                return &state->frame;
        }
        return NULL;
}

struct vidcap_type *vidcap_testcard_probe(void)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = VIDCAP_TESTCARD_ID;
                vt->name = "testcard";
                vt->description = "Video testcard";
        }
        return vt;
}

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
#include "song1.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_LIBSDL_MIXER
#include <SDL/SDL.h>
#include <SDL/SDL_mixer.h>
#endif /* HAVE_LIBSDL_MIXER */
#include "audio/audio.h"

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_BPS 2
#define AUDIO_CHANNELS 2
#define BUFFER_SEC 1
#define AUDIO_BUFFER_SIZE (AUDIO_SAMPLE_RATE * AUDIO_BPS * \
                AUDIO_CHANNELS * BUFFER_SEC)

struct testcard_rect {
        int x, y, w, h;
};
struct testcard_pixmap {
        int w, h;
        char *data;
};

void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height);
unsigned char *tov210(unsigned char *in, unsigned int width, unsigned int align_x,
                      unsigned int height, double bpp);
void toR10k(unsigned char *in, unsigned int width, unsigned int height);
void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, int color);
char * toRGB(unsigned char *in, unsigned int width, unsigned int height);

struct testcard_state {
        struct timeval last_frame_time;
        int count;
        int size;
        int pan;
        struct testcard_pixmap pixmap;
        char *data;
        struct timeval t0;
        struct video_frame *frame;
        struct video_frame *tiled;
        
        struct audio_frame audio;
        char **tiles_data;
        
        char *audio_data;
        volatile int audio_start, audio_end;
        unsigned int grab_audio:1;
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

void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, int color)
{
        int cur_x, cur_y;
        int *data = (int *) s->data;
        
        for (cur_x = r->x; cur_x < r->x + r->w; ++cur_x)
                for(cur_y = r->y; cur_y < r->y + r->h; ++cur_y)
                        if(cur_x < s->w)
                                *(data + s->w * cur_y + cur_x) = color;
}

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

char *toRGB(unsigned char *in, unsigned int width, unsigned int height)
{
        int i;
        char *ret = malloc(width * height * 3);
        for(i = 0; i < height; ++i) {
                vc_copylineRGBAtoRGB(ret + i * width * 3, in + i * width * 4, width * 3);
        }
        return ret;
}

static void grab_audio(int chan, void *stream, int len, void *udata)
{
        UNUSED(chan);
        struct testcard_state *s = (struct testcard_state *) udata;
        
        if(s->audio_end + len <= AUDIO_BUFFER_SIZE) {
                memcpy(s->audio_data + s->audio_end, stream, len);
                s->audio_end += len;
        } else {
                int offset = AUDIO_BUFFER_SIZE - s->audio_end;
                memcpy(s->audio_data + s->audio_end, stream, offset);
                memcpy(s->audio_data, stream + offset, len - offset);
                s->audio_end = len - offset;
        }
        /* just hack - Mix_Volume doesn't mute correctly the audio */
        memset(stream, 0, len);
}

static int configure_audio(struct testcard_state *s)
{
        UNUSED(s);
        
#if defined HAVE_LIBSDL_MIXER && ! defined HAVE_MACOSX
        char *filename;
        int fd;
        Mix_Music *music;
        ssize_t bytes_written = 0l;
        
        SDL_Init(SDL_INIT_AUDIO);
        
        if( Mix_OpenAudio( AUDIO_SAMPLE_RATE, AUDIO_S16LSB,
                        AUDIO_CHANNELS, 4096 ) == -1 ) {
                fprintf(stderr,"[testcard] error initalizing sound\n");
                return -1;
        }
        filename = strdup("/tmp/uv.midiXXXXXX");
        fd = mkstemp(filename);
        
        do {
                ssize_t ret;
                ret = write(fd, song1 + bytes_written,
                                sizeof(song1) - bytes_written);
                if(ret < 0) return -1;
                bytes_written += ret;
        } while (bytes_written < (ssize_t) sizeof(song1));
        close(fd);
        music = Mix_LoadMUS(filename);
        free(filename);

        s->audio_data = calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        s->audio_start = 0;
        s->audio_end = 0;
        s->audio.bps = AUDIO_BPS;
        s->audio.ch_count = AUDIO_CHANNELS;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        
        // register grab as a postmix processor
        if(!Mix_RegisterEffect(MIX_CHANNEL_POST, grab_audio, NULL, s)) {
                printf("[testcard] Mix_RegisterEffect: %s\n", Mix_GetError());
                return -1;
        }

        if(Mix_PlayMusic(music,-1)==-1){
                fprintf(stderr, "[testcard] error playing midi\n");
                return -1;
        }
        Mix_Volume(-1, 0);
        
        printf("[testcard] playing audio\n");
        
        return 0;
#else
        return -2;
#endif
}

static int configure_tiling(struct testcard_state *s, const char *fmt)
{
        char *tmp, *token, *saveptr = NULL;
        int tile_cnt;
        int x;
        
        int grid_w, grid_h;

        if(fmt[1] != '=') return 1;
        
        tmp = strdup(&fmt[2]);
        token = strtok_r(tmp, "x", &saveptr);
        grid_w = atoi(token);
        token = strtok_r(NULL, "x", &saveptr);
        grid_h = atoi(token);
        free(tmp);
        
        s->tiled = vf_alloc(grid_w, grid_h);
        s->tiled->color_spec = s->frame->color_spec;
        s->tiled->fps = s->frame->fps;
        s->tiled->aux = s->frame->aux | AUX_TILED;

        tile_cnt = grid_w *
                                grid_h;
        assert(tile_cnt >= 1);

        s->tiles_data = (char **) malloc(tile_cnt *
                        sizeof(char *));
        /* split only horizontally!!!!!! */
        vf_split(s->tiled, s->frame, grid_w,
                        1, 1 /*prealloc*/);
        /* for each row, make the tile data correct.
         * .data pointers of same row point to same block,
         * but different row */
        for(x = 0; x < grid_w; ++x) {
                int y;

                s->tiles_data[x] = s->tiled->tiles[x].data;
                
                s->tiled->tiles[x].width = s->frame->tiles[0].width/ grid_w;
                s->tiled->tiles[x].height = s->frame->tiles[0].height / grid_h;
                s->tiled->tiles[x].data_len = s->frame->tiles[0].data_len / (grid_w * grid_h);
                s->tiled->tiles[x].tile_info.x_count = grid_w;
                s->tiled->tiles[x].tile_info.y_count = grid_h;
                s->tiled->tiles[x].tile_info.pos_x = x;
                s->tiled->tiles[x].tile_info.pos_y = 0;

                s->tiled->tiles[x].data =
                        s->tiles_data[x] = (char *) realloc(s->tiled->tiles[x].data, 
                                s->tiled->tiles[x].data_len * grid_h * 2);
                                
                               
                memcpy(s->tiled->tiles[x].data + s->tiled->tiles[x].data_len  * grid_h,
                                s->tiled->tiles[x].data, s->tiled->tiles[x].data_len * grid_h);                                
                /* recopy tiles vertically */
                for(y = 1; y < grid_h; ++y) {
                        memcpy(&s->tiled->tiles[y * grid_w + x], 
                                        &s->tiled->tiles[x], sizeof(struct tile));
                        /* make the pointers correct */
                        s->tiles_data[y * grid_w + x] =
                                s->tiles_data[x] +
                                y * s->tiled->tiles[x].height * s->tiled->tiles[x].linesize;
                                
                        s->tiled->tiles[y * grid_w + x].data =
                                s->tiles_data[x] + 
                                y * s->tiled->tiles[x].height * s->tiled->tiles[x].linesize;
                        s->tiled->tiles[y * grid_w + x].tile_info.pos_y = y;
                }
        }

        return 0;
}

void *vidcap_testcard_init(char *fmt, unsigned int flags)
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

        if (fmt == NULL || strcmp(fmt, "help") == 0) {
                printf("testcard options:\n");
                printf("\t-t testcard:<width>:<height>:<fps>:<codec>[:<filename>][:p][:s=<X>x<Y>]\n");
                printf("\tp - pan with frame\n");
                printf("\ts - split the frames into XxY separate tiles\n");
                show_codec_help("testcard");
                return NULL;
        }

        s = calloc(1, sizeof(struct testcard_state));
        if (!s)
                return NULL;

        s->frame = vf_alloc(1, 1);
        
        char *tmp;

        tmp = strtok(fmt, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        tile_get(s->frame, 0, 0)->width = atoi(tmp);
        if(tile_get(s->frame, 0, 0)->width % 2 != 0) {
                fprintf(stderr, "[testcard] Width must be multiple of 2.\n");
                free(s);
                return NULL;
        }
        tmp = strtok(NULL, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return NULL;
        }
        tile_get(s->frame, 0, 0)->height = atoi(tmp);
        tmp = strtok(NULL, ":");
        if (!tmp) {
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return NULL;
        }

        s->frame->fps = atof(tmp);

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

        s->frame->color_spec = codec;

        if(bpp == 0) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                return NULL;
        }

        aligned_x = tile_get(s->frame, 0, 0)->width;
        if (h_align) {
                aligned_x = (aligned_x + h_align - 1) / h_align * h_align;
        }

        rect_size = (tile_get(s->frame, 0, 0)->width + rect_size - 1) / rect_size;
        
        tile_get(s->frame, 0, 0)->linesize = aligned_x * bpp;
        s->frame->aux = AUX_PROGRESSIVE;
        s->size = aligned_x * tile_get(s->frame, 0, 0)->height * bpp;

        filename = strtok(NULL, ":");
        if (filename && strcmp(filename, "p") != 0
                        && strncmp(filename, "s=", 2ul) != 0) {
                s->data = malloc(s->size * 2);
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
                        free(s->data);
                        free(s);
                        return NULL;
                }

                if (!in || fread(s->data, sb.st_size, 1, in) == 0) {
                        fprintf(stderr, "Cannot read file %s\n", filename);
                        free(s->data);
                        free(s);
                        return NULL;
                }

                fclose(in);
        } else {
                struct testcard_rect r;
                int col_num = 0;
                s->pixmap.w = aligned_x;
                s->pixmap.h = tile_get(s->frame, 0, 0)->height * 2;
                s->pixmap.data = malloc(s->pixmap.w * s->pixmap.h * sizeof(int));

                if (filename) {
                        if(filename[0] == 'p')
                                s->pan = 48;
                        else if(filename[0] == 's')
                                strip_fmt = filename;
                }

                for (j = 0; j < tile_get(s->frame, 0, 0)->height; j += rect_size) {
                        int grey = 0xff010101;
                        if (j == rect_size * 2) {
                                r.w = tile_get(s->frame, 0, 0)->width;
                                r.h = rect_size / 4;
                                r.x = 0;
                                r.y = j;
                                testcard_fillRect(&s->pixmap, &r, 0xffffffff);
                                r.y = j + rect_size * 3 / 4;
                                testcard_fillRect(&s->pixmap, &r, 0);
                        }
                        for (i = 0; i < tile_get(s->frame, 0, 0)->width; i += rect_size) {
                                r.w = rect_size;
                                r.h = rect_size;
                                r.x = i;
                                r.y = j;
                                printf("Fill rect at %d,%d\n", r.x, r.y);
                                if (j != rect_size * 2) {
                                        testcard_fillRect(&s->pixmap, &r,
                                                     rect_colors[col_num]);
                                        col_num = (col_num + 1) % COL_NUM;
                                } else {
                                        r.h = rect_size / 2;
                                        r.y += rect_size / 4;
                                        testcard_fillRect(&s->pixmap, &r, grey);
                                        grey += 0x00010101 * (255 / COL_NUM);
                                }
                        }
                }
                s->data = s->pixmap.data;
                if (codec == UYVY || codec == v210 || codec == Vuy2) {
                        rgb2yuv422((unsigned char *) s->data, aligned_x,
                                   tile_get(s->frame, 0, 0)->height);
                }

                if (codec == v210) {
                        s->data =
                            (char *)tov210((unsigned char *) s->data, aligned_x,
                                           aligned_x, tile_get(s->frame, 0, 0)->height, bpp);
                }

                if (codec == R10k) {
                        toR10k((unsigned char *) s->data, tile_get(s->frame, 0, 0)->width,
                                        tile_get(s->frame, 0, 0)->height);
                }
                
                if(codec == RGB) {
                        s->data =
                            (char *)toRGB((unsigned char *) s->data, tile_get(s->frame, 0, 0)->width,
                                           tile_get(s->frame, 0, 0)->height);
                }
                
                if(codec == RGBA) {
                        toRGB((unsigned char *) s->data, tile_get(s->frame, 0, 0)->width,
                                           tile_get(s->frame, 0, 0)->height);
                }
        }

        tile_get(s->frame, 0, 0)->data = malloc(2 * s->size);

        memcpy(tile_get(s->frame, 0, 0)->data, s->data, s->size);
        memcpy(tile_get(s->frame, 0, 0)->data + s->size, tile_get(s->frame, 0, 0)->data, s->size);
        if(s->pixmap.data)
                free(s->pixmap.data);
        else
                free(tile_get(s->frame, 0, 0)->data);

        s->data = tile_get(s->frame, 0, 0)->data;

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

        printf("Testcard capture set to %dx%d, bpp %f\n", tile_get(s->frame, 0, 0)->width,
                        tile_get(s->frame, 0, 0)->height, bpp);

        s->frame->state = s;
        tile_get(s->frame, 0, 0)->data_len = s->size;

        if(strip_fmt != NULL) {
                if(configure_tiling(s, strip_fmt) != 0)
                        return NULL;
        } else {
                s->frame->aux &= ~AUX_TILED;
        }
        
        if(flags & VIDCAP_FLAG_ENABLE_AUDIO) {
                s->grab_audio = TRUE;
                if(configure_audio(s) != 0) {
                        s->grab_audio = FALSE;
                        fprintf(stderr, "[testcard] Disabling audio output. "
                                        "SDL-mixer missing, running on Mac or other problem.\n");
                }
        } else {
                s->grab_audio = FALSE;
        }

        return s;
}

void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = state;
#if 0
        free(s->data);
        if (tile_get(s->frame, 0, 0)->aux & AUX_TILED) {
                int i;
                for (i = 0; i < s->frame.grid_width; ++i) {
                        free(s->tiles_data[i]);
                }
                free(s->tiles);
        }
        if(s->audio_data) {
                free(s->audio_data);
        }
        free(s);
#endif
}

struct video_frame *vidcap_testcard_grab(void *arg, struct audio_frame **audio)
{
        struct timeval curr_time;
        struct testcard_state *state;

        state = (struct testcard_state *)arg;

        gettimeofday(&curr_time, NULL);
        if (tv_diff(curr_time, state->last_frame_time) >
            1.0 / (double)state->frame->fps) {
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

                if (state->grab_audio) {
#ifdef HAVE_LIBSDL_MIXER
                        state->audio.data = state->audio_data + state->audio_start;
                        if(state->audio_start <= state->audio_end) {
                                int tmp = state->audio_end;
                                state->audio.data_len = tmp - state->audio_start;
                                state->audio_start = tmp;
                        } else {
                                state->audio.data_len = 
                                                AUDIO_BUFFER_SIZE -
                                                state->audio_start;
                                state->audio_start = 0;
                        }
                        if(state->audio.data_len > 0)
                                *audio = &state->audio; 
                        else
                                *audio = NULL;
#endif                        
                } else {
                        *audio = NULL;
                }

                
                tile_get(state->frame, 0, 0)->data += tile_get(state->frame, 0, 0)->linesize;
                if(tile_get(state->frame, 0, 0)->data > state->data + state->size)
                        tile_get(state->frame, 0, 0)->data = state->data;

                /*char line[state->frame.src_linesize * 2 + state->pan];
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
                */
                if (state->tiled) {
                        /* update tile data instead */
                        int i;
                        int count = state->tiled->grid_width *
                                state->tiled->grid_height;
                                
                        for (i = 0; i < count; ++i) {
                                /* shift - for semantics of vars refer to configure_tiling*/
                                state->tiled->tiles[i].data += state->tiled->tiles[i].linesize;
                                /* if out of data, move to beginning
                                 * keep in mind that we have two "pictures" for
                                 * every tile stored sequentially */
                                if(state->tiled->tiles[i].data >= state->tiles_data[i] +
                                                state->tiled->tiles[i].data_len * state->tiled->grid_height) {
                                        state->tiled->tiles[i].data = state->tiles_data[i];
                                }
                        }

                        return state->tiled;
                }
                return state->frame;
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

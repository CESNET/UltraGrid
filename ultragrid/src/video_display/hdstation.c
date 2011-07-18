/*
 * FILE:    video_display/hdstation.c
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
 * $Revision: 1.3.2.2 $
 * $Date: 2010/01/29 11:26:04 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "host.h"

#ifdef HAVE_HDSTATION           /* From config.h */

#include "debug.h"
#include "video_display.h"
#include "video_display/hdstation.h"
#include "video_codec.h"
#include "tv.h"

#include "dvs_clib.h"           /* From the DVS SDK */
#include "dvs_fifo.h"           /* From the DVS SDK */

#define HDSP_MAGIC	0x12345678

const hdsp_mode_table_t hdsp_mode_table[] = {
        {"SMPTE274", SV_MODE_SMPTE274_25P, 25, 1920, 1080, 0},
        {"SMPTE274", SV_MODE_SMPTE274_29I, 29, 1920, 1080, 1},
        {NULL, 0, 0, 0, 0, 0},
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
        int hd_video_mode;
        struct video_frame frame;
        const hdsp_mode_table_t *mode;
        unsigned interlaced:1;       
};

static void show_help(void);

static void show_help(void)
{
	printf("hdstation options:\n");
	printf("\t[fps:[mode:[codec:[i|p]]]]\n");
	printf("\tSupported modes:\n");
	printf("\t\tSMPTE274\n");
	show_codec_help(strdup("hdstation"));
}

void display_hdstation_run(void *arg)
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
                pthread_mutex_unlock(&s->lock);

                res =
                    sv_fifo_putbuffer(s->sv, s->fifo, s->display_buffer, NULL);
                if (res != SV_OK) {
                        debug_msg("Error %s\n", sv_geterrortext(res));
                        return;
                }
        }
}

struct video_frame *
display_hdstation_getf(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;
        int res;

        assert(s->magic == HDSP_MAGIC);

        /* Prepare the new RTP buffer... */
        res =
            sv_fifo_getbuffer(s->sv, s->fifo, &s->fifo_buffer, NULL,
                              SV_FIFO_FLAG_VIDEOONLY | SV_FIFO_FLAG_FLUSH);
        if (res != SV_OK) {
                debug_msg("Error %s\n", sv_geterrortext(res));
                return NULL;
        }      

        s->bufs_index = (s->bufs_index + 1) % 2;
        s->frame.data = s->bufs[s->bufs_index];
        s->fifo_buffer->dma.addr = s->frame.data;
        s->fifo_buffer->dma.size = s->frame.data_len;

        return &s->frame;
}

int display_hdstation_putf(void *state, char *frame)
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

        /* Wait for the worker to finish... */
        while (!s->worker_waiting);

        s->mode = NULL;
        for(i=0; hdsp_mode_table[i].name != NULL; i++) {
                if(hdsp_mode_table[i].width == width &&
                   hdsp_mode_table[i].height == height &&
                   aux == hdsp_mode_table[i].interlaced &&
                   fps == hdsp_mode_table[i].fps) {
                    s->mode = &hdsp_mode_table[i];
                        break;
                }
        }

        if(s->mode == NULL) {
                fprintf(stderr, "Reconfigure failed. Expect troubles pretty soon..\n"
                                "\tRequested: %dx%d, color space %d, fps %f, interlaced: %d\n",
                                width, height, color_spec, fps, s->interlaced);
                return;
        }

        s->frame.color_spec = color_spec;
        s->frame.width = width;
        s->frame.height = height;
        s->frame.dst_bpp = get_bpp(color_spec);
        s->frame.fps = fps;
        s->frame.aux = aux;
        if(aux & AUX_INTERLACED)
                s->interlaced = 1;
        else
                s->interlaced = 0;

        s->hd_video_mode = SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;

        if (s->frame.color_spec == DVS10) {
                s->hd_video_mode |= SV_MODE_NBIT_10BDVS;
        }

        s->hd_video_mode |= s->mode->mode;

        res = sv_videomode(s->sv, s->hd_video_mode | SV_MODE_AUDIO_NOAUDIO);
        if (res != SV_OK) {
                debug_msg("Cannot set videomode %s\n", sv_geterrortext(res));
                return;
        }
        res = sv_sync_output(s->sv, SV_SYNCOUT_BILEVEL);
        if (res != SV_OK) {
                debug_msg("Cannot enable sync-on-green %s\n",
                          sv_geterrortext(res));
                return;
        }

        if(s->fifo)
                sv_fifo_free(s->sv, s->fifo);

        res = sv_fifo_init(s->sv, &s->fifo, 0, 1, 1, 0, 0);
        if (res != SV_OK) {
                debug_msg("Cannot initialize video display FIFO %s\n",
                          sv_geterrortext(res));
                return;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                debug_msg("Cannot start video display FIFO  %s\n",
                          sv_geterrortext(res));
                return;
        }

        s->frame.data_len = s->frame.width * s->frame.height * s->frame.dst_bpp;
        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;

        free(s->bufs[0]);
        free(s->bufs[1]);
        s->bufs[0] = malloc(s->frame.data_len);
        s->bufs[1] = malloc(s->frame.data_len);
        s->bufs_index = 0;
        memset(s->bufs[0], 0, s->frame.data_len);
        memset(s->bufs[1], 0, s->frame.data_len);
}


void *display_hdstation_init(char *fmt)
{
        struct state_hdsp *s;
        double fps;
        int i;

        s = (struct state_hdsp *)calloc(1, sizeof(struct state_hdsp));
        s->magic = HDSP_MAGIC;

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
			show_help();

                        return 0;
                }

                char *tmp;
                char *mode;

                tmp = strtok(fmt, ":");

                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        free(s);
                        return 0;
                }
                fps = atof(tmp);
                tmp = strtok(NULL, ":");
                if (tmp) {
                        mode = tmp;
                        tmp = strtok(NULL, ":");
                        if (!tmp) {
                                fprintf(stderr, "Wrong config %s\n", fmt);
                                free(s);
                                return 0;
                        }
                        s->frame.color_spec = 0xffffffff;
                        for (i = 0; codec_info[i].name != NULL; i++) {
                                if (strcmp(tmp, codec_info[i].name) == 0) {
                                        s->frame.color_spec = codec_info[i].codec;
                                        s->frame.src_bpp = codec_info[i].bpp;
                                }
                        }
                        if (s->frame.color_spec == 0xffffffff) {
                                fprintf(stderr, "hdstation: unknown codec: %s\n", tmp);
                                free(s);
                                return 0;
                        }
                        tmp = strtok(NULL, ":");
                        if(tmp) {
                                if(tmp[0] == 'i') {
                                        s->interlaced = 1;
                                } else if(tmp[0] == 'p') {
                                        s->interlaced = 0;                               
                                }
                        }
                        for(i=0; hdsp_mode_table[i].name != NULL; i++) {
                                if(strcmp(mode, hdsp_mode_table[i].name) == 0 &&
                                   s->interlaced == hdsp_mode_table[i].interlaced &&
                                   fps == hdsp_mode_table[i].fps) {
                                        s->mode = &hdsp_mode_table[i];
                                        break;
                                }
                        }
                        if(s->mode == NULL) {
                                fprintf(stderr, "hdstation: unknown video mode: %s\n", mode);
                                free(s);
                                return 0;
                       }
                }
        }

        /* Start the display thread... */
        s->sv = sv_open("");
        if (s->sv == NULL) {
                debug_msg("Cannot open HDTV display device\n");
                return NULL;
        }

        if(s->mode) {
                reconfigure_screen(s, s->mode->width, s->mode->height, s->frame.color_spec, s->mode->fps, s->interlaced);
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        s->work_to_do = FALSE;
        s->boss_waiting = FALSE;
        s->worker_waiting = FALSE;
        s->display_buffer = NULL;

        /*if (pthread_create(&(s->thread_id), NULL, display_thread_hd, (void *)s)
            != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/
        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)reconfigure_screen;
        s->frame.decoder = (decoder_t)memcpy;     

        return (void *)s;
}

void display_hdstation_done(void *state)
{
        struct state_hdsp *s = (struct state_hdsp *)state;

        sv_fifo_free(s->sv, s->fifo);
        sv_close(s->sv);
        free(s);
}

display_type_t *display_hdstation_probe(void)
{
        display_type_t *dtype;

        dtype = malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_HDSTATION_ID;
                dtype->name = "hdtv";
                dtype->description = "DVS HDstation (1080i/60 YUV 4:2:2)";
        }
        return dtype;
}

#endif                          /* HAVE_HDSTATION */

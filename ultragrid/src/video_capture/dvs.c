/*
 * FILE:   video_capture/dvs.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
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

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifdef HAVE_DVS           /* From config.h */

#include "debug.h"
#include "video_capture.h"
#include "video_capture/dvs.h"
#include "video_display/dvs.h"
#include "video_codec.h"
#include "tv.h"
#include "dvs_clib.h"           /* From the DVS SDK */
#include "dvs_fifo.h"           /* From the DVS SDK */

struct vidcap_dvs_state {
        sv_handle *sv;
        sv_fifo *fifo;
        sv_fifo_buffer *dma_buffer;
        char *rtp_buffer;
        char *tmp_buffer;
        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        pthread_cond_t worker_cv;
        int boss_waiting;
        int worker_waiting;
        int work_to_do;
        char *bufs[2];
        int bufs_index;
        uint32_t hd_video_mode;
        struct video_frame frame;
        const hdsp_mode_table_t *mode;
};

static void show_help(void);

static void *vidcap_dvs_grab_thread(void *arg)
{
        int res;
        struct vidcap_dvs_state *s = (struct vidcap_dvs_state *)arg;

        while (1) {
                s->dma_buffer = NULL;
                res = sv_fifo_vsyncwait(s->sv, s->fifo);

                res =
                    sv_fifo_getbuffer(s->sv, s->fifo, &(s->dma_buffer), NULL,
                                      SV_FIFO_FLAG_VIDEOONLY |
                                      SV_FIFO_FLAG_FLUSH);
                if (res != SV_OK) {
                        printf("Unable to getbuffer %s\n",
                               sv_geterrortext(res));
                        continue;
                }
                s->bufs_index = (s->bufs_index + 1) % 2;
                s->dma_buffer->dma.addr = s->bufs[s->bufs_index];
                s->dma_buffer->dma.size = s->frame.data_len;

                res = sv_fifo_putbuffer(s->sv, s->fifo, s->dma_buffer, NULL);
                if (res != SV_OK) {
                        printf("Unable to putbuffer %s\n",
                               sv_geterrortext(res));
                }

                pthread_mutex_lock(&(s->lock));

                while (s->work_to_do == FALSE) {
                        s->worker_waiting = TRUE;
                        pthread_cond_wait(&(s->worker_cv), &(s->lock));
                        s->worker_waiting = FALSE;
                }

                s->tmp_buffer = s->dma_buffer->dma.addr;
                s->work_to_do = FALSE;

                if (s->boss_waiting) {
                        pthread_cond_signal(&(s->boss_cv));
                }
                pthread_mutex_unlock(&(s->lock));
        }
        return NULL;
}

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
	printf("\tmode:codec | help\n\n");
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

/* External API ***********************************************************************************/

struct vidcap_type *vidcap_dvs_probe(void)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = VIDCAP_DVS_ID;
                vt->name = "dvs";
                vt->description = "DVS (SMPTE 274M/25i)";
        }
        return vt;
}

void *vidcap_dvs_init(char *fmt)
{
        struct vidcap_dvs_state *s;
	int h_align = 0;
	int aligned_x;
        int i;
        int res;
        int mode_index = 0;
        char *mode;

        s = (struct vidcap_dvs_state *)
            calloc(1, sizeof(struct vidcap_dvs_state));
        if (s == NULL) {
                debug_msg("Unable to allocate DVS state\n");
                return NULL;
        }

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
			show_help();
                        return 0;
                }

                char *tmp;

                tmp = strtok(fmt, ":");
                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        return 0;
                }
                mode_index = atoi(tmp);
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

                tmp = strtok(NULL, ":");
                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        return 0;
                }

                s->frame.color_spec = 0xffffffff;
                for (i = 0; codec_info[i].name != NULL; i++) {
                        if (strcmp(tmp, codec_info[i].name) == 0) {
                                s->frame.color_spec = codec_info[i].codec;
                                s->frame.src_bpp = codec_info[i].bpp;
				h_align = codec_info[i].h_align;
                        }
                }
                if (s->frame.color_spec == 0xffffffff) {
                        fprintf(stderr, "dvs: unknown codec: %s\n", tmp);
                        free(tmp);
                        return 0;
                }
	
        } else {
		show_help();
                return 0;
        }

        s->hd_video_mode = SV_MODE_COLOR_YUV422 | SV_MODE_STORAGE_FRAME;

        if (s->frame.color_spec == DVS10) {
                s->hd_video_mode |= SV_MODE_NBIT_10BDVS;
        }

        s->hd_video_mode |= s->mode->mode;

        s->frame.width = s->mode->width;
        s->frame.height = s->mode->height;
	s->frame.aux = s->mode->aux;

	aligned_x = s->frame.width;
	if (h_align) {
		aligned_x = (aligned_x + h_align - 1) / h_align * h_align;
	}
	s->frame.src_linesize = aligned_x * s->frame.src_bpp;
	s->frame.data_len = aligned_x * s->frame.height * s->frame.src_bpp;

        s->sv = sv_open("");
        if (s->sv == NULL) {
                printf
                    ("Unable to open grabber: sv_open() failed (no card present or driver not loaded?)\n");
                free(s);
                return NULL;
        }

        res = sv_videomode(s->sv, s->hd_video_mode | SV_MODE_AUDIO_NOAUDIO);
        if (res != SV_OK) {
                goto error;
        }
        res = sv_black(s->sv);
        if (res != SV_OK) {
                goto error;
        }
        res = sv_fifo_init(s->sv, &(s->fifo), 1, 1, 1, 0, 0);
        if (res != SV_OK) {
                goto error;
        }
        res = sv_fifo_start(s->sv, s->fifo);
        if (res != SV_OK) {
                goto error;
        }

        pthread_mutex_init(&(s->lock), NULL);
        pthread_cond_init(&(s->boss_cv), NULL);
        pthread_cond_init(&(s->worker_cv), NULL);

        s->rtp_buffer = NULL;
        s->dma_buffer = NULL;
        s->tmp_buffer = NULL;
        s->boss_waiting = FALSE;
        s->worker_waiting = FALSE;
        s->work_to_do = FALSE;
        s->bufs[0] = malloc(s->frame.data_len);
        s->bufs[1] = malloc(s->frame.data_len);
        s->bufs_index = 0;
        s->frame.state = s;

        if (pthread_create
            (&(s->thread_id), NULL, vidcap_dvs_grab_thread, s) != 0) {
                perror("Unable to create grabbing thread");
                return NULL;
        }

        printf("Testcard capture set to %dx%d, bpp %f\n", s->frame.width, s->frame.height, s->frame.src_bpp);

        debug_msg("DVS capture device enabled\n");
        return s;
 error:
        free(s);
        printf("Chyba %s\n", sv_geterrortext(res));
        debug_msg("Unable to open grabber: %s\n", sv_geterrortext(res));
        return NULL;
}

void vidcap_dvs_done(void *state)
{
        struct vidcap_dvs_state *s =
            (struct vidcap_dvs_state *)state;

        sv_fifo_free(s->sv, s->fifo);
        sv_close(s->sv);
        free(s);
}

struct video_frame *vidcap_dvs_grab(void *state, int *count, struct audio_frame **audio)
{
        struct vidcap_dvs_state *s =
            (struct vidcap_dvs_state *)state;

        pthread_mutex_lock(&(s->lock));

        *audio = NULL; /* currently no audio */
        
        /* Wait for the worker to finish... */
        while (s->work_to_do) {
                s->boss_waiting = TRUE;
                pthread_cond_wait(&(s->boss_cv), &(s->lock));
                s->boss_waiting = FALSE;
        }

        /* ...and give it more to do... */
        s->rtp_buffer = s->tmp_buffer;
        s->work_to_do = TRUE;

        /* ...and signal the worker... */
        if (s->worker_waiting) {
                pthread_cond_signal(&(s->worker_cv));
        }

        pthread_mutex_unlock(&(s->lock));

        if (s->rtp_buffer != NULL) {
                s->frame.data = s->rtp_buffer;
                *count = 1;
                return &s->frame;
        }
        *count = 0;
        return NULL;
}

#endif                          /* HAVE_DVS */

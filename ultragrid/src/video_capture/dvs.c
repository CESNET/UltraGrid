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
#include "audio/audio.h"

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
        char *audio_bufs[2];
        int bufs_index;
        uint32_t hd_video_mode;
        struct video_frame *frame;
        struct tile *tile;
        struct audio_frame audio;
        const hdsp_mode_table_t *mode;
        unsigned grab_audio:1;
        unsigned int detect_mode:1;

        int frames;
        struct       timeval t, t0;
};

static void show_help(void);

static void *vidcap_dvs_grab_thread(void *arg)
{
        int res;
        struct vidcap_dvs_state *s = (struct vidcap_dvs_state *)arg;
        int fifo_flags = SV_FIFO_FLAG_FLUSH |
                                      SV_FIFO_FLAG_NODMAADDR;

        if(!s->grab_audio)
                fifo_flags |= SV_FIFO_FLAG_VIDEOONLY;

        while (!should_exit) {
                s->dma_buffer = NULL;
                res = sv_fifo_vsyncwait(s->sv, s->fifo);

                res =
                    sv_fifo_getbuffer(s->sv, s->fifo, &(s->dma_buffer), NULL,
                                      fifo_flags);
                if (res != SV_OK) {
                        printf("Unable to getbuffer %s\n",
                               sv_geterrortext(res));
                        continue;
                }
                s->bufs_index = (s->bufs_index + 1) % 2;
                s->dma_buffer->video[0].addr = s->bufs[s->bufs_index];
                s->dma_buffer->video[0].size = s->tile->data_len;
                if(s->grab_audio) {
                        s->dma_buffer->audio[0].addr[0] = s->audio_bufs[s->bufs_index];
                        //fprintf(stderr, "%d ", s->dma_buffer->audio[0].size);
                        //s->dma_buffer->audio[0].size = 12800;
                }

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

                s->tmp_buffer = s->dma_buffer->video[0].addr;

                s->audio.data = s->dma_buffer->audio[0].addr[0];
                s->audio.data_len = s->dma_buffer->audio[0].size;

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
        int card_idx = 0;
        sv_handle *sv;
        char name[128];
        int res;

	printf("DVS options:\n\n");
	printf("\t -t dvs[:<mode>:<codec>][:<card>] | help\n\n");
        snprintf(name, 128, "PCI,card:%d", card_idx);

        //sv = sv_open(name);
        res = sv_openex(&sv, name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
        if (res != SV_OK) {
                printf
                    ("Unable to open grabber: sv_open() failed (no card present or driver not loaded?)\n");
                printf("Error %s\n", sv_geterrortext(res));
                return;
        }
	while (res == SV_OK) {
                printf("\tCard %s - supported modes:\n\n", name);
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
                sv_close(sv);
                card_idx++;
                snprintf(name, 128, "PCI,card:%d", card_idx);
                res = sv_openex(&sv, name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
                printf("\n");
        }
	printf("\n");
	show_codec_help("dvs");
}

/* External API ***********************************************************************************/

void *vidcap_dvs_init(char *fmt, unsigned int flags)
{
        struct vidcap_dvs_state *s;
        int i;
        int res;
        int mode_index = 0;
        char *card_name = "";

        s = (struct vidcap_dvs_state *)
            calloc(1, sizeof(struct vidcap_dvs_state));
            
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->frames = 0;

        s->detect_mode = FALSE;
        
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

                if(strncmp(fmt, "PCI", 3) == 0) {
                        card_name = fmt;
                        printf("[DVS] Choosen card: %s.\n", card_name);
                        s->detect_mode = TRUE;
                } else {
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

                        s->frame->color_spec = 0xffffffff;
                        for (i = 0; codec_info[i].name != NULL; i++) {
                                if (strcmp(tmp, codec_info[i].name) == 0) {
                                        s->frame->color_spec = codec_info[i].codec;
                                }
                        }
                        if (s->frame->color_spec == 0xffffffff) {
                                fprintf(stderr, "dvs: unknown codec: %s\n", tmp);
                                free(tmp);
                                return 0;
                        }

                        /* card name */
                        tmp = strtok(NULL, ":");
                        if(tmp) {
                                card_name = tmp;
                                tmp[strlen(card_name)] = ':';
                                printf("[DVS] Choosen card: %s.\n", card_name);
                        }
                }
        } else {
                s->detect_mode = TRUE;
        }

        res = sv_openex(&s->sv, card_name, SV_OPENPROGRAM_DEFAULT, SV_OPENTYPE_DEFAULT, 0, 0);
        if (s->sv == NULL) {
                printf
                    ("Unable to open grabber: sv_open() failed (no card present or driver not loaded?)\n");
                printf("Error %s\n", sv_geterrortext(res));
                free(s);
                return NULL;
        }

        if(flags & VIDCAP_FLAG_ENABLE_AUDIO) {
                s->grab_audio = TRUE;
        } else {
                s->grab_audio = FALSE;
        }

        s->hd_video_mode = 0;

        if(!s->detect_mode) {
                switch(s->frame->color_spec) {
                        case DVS10:
                                s->hd_video_mode |= SV_MODE_COLOR_YUV422 | SV_MODE_NBIT_10BDVS;
                                break;
                        case DVS8:
                        case Vuy2:
                        case UYVY:
                                s->hd_video_mode |= SV_MODE_COLOR_YUV422;
                                break;
                        case RGBA:
                                s->hd_video_mode |= SV_MODE_COLOR_RGBA;
                                break;
                        case RGB:
                                s->hd_video_mode |= SV_MODE_COLOR_RGB_RGB;
                                break;
                        default:
                                fprintf(stderr, "[dvs] Unsupported video codec passed!");
                                return NULL;
                }

                s->hd_video_mode |= s->mode->mode;
        } else {
                int val;
                int res;
                res = sv_query(s->sv, SV_QUERY_INPUTRASTER, 0, &val);
                if(res != SV_OK) {
                        printf("[DVS] Could not detect video format %s\n",
                               sv_geterrortext(res));
                        goto error_detect;
                }
                if(val == -1) {
                        printf("[DVS] No signal detected, cannot autodetect format.\n");
                        goto error_detect;
                }
                for(i=0; hdsp_mode_table[i].width != 0; i++) {
                        if(hdsp_mode_table[i].mode == val) {
                                  s->mode = &hdsp_mode_table[i];
                                break;
                        }
                }
                s->hd_video_mode |= val;
                printf("[DVS] Autodetected video mode: %dx%d @ %2.2fFPS.\n", s->mode->width, s->mode->height, s->mode->fps);

                res = sv_query(s->sv, SV_QUERY_IOMODE, 0, &val);
                if(res != SV_OK) {
                        printf("Could not detect IO mode %s\n",
                               sv_geterrortext(res));
                        goto error_detect;
                }
                printf("[DVS] Autodetected IO mode: %d.\n", val);

                switch (val) {
                        case SV_IOMODE_YUV422:
                        case SV_IOMODE_YUV444:
                        case SV_IOMODE_YUV422A:
                        case SV_IOMODE_YUV444A:
                        case SV_IOMODE_YUV422_12:
                        case SV_IOMODE_YUV444_12:
                                s->hd_video_mode |= SV_MODE_COLOR_YUV422;
                                s->frame->color_spec = UYVY;
                                break;
                        case SV_IOMODE_RGB:
                        case SV_IOMODE_RGB_12:
                                s->hd_video_mode |= SV_MODE_COLOR_RGBA;
                                s->frame->color_spec = RGB;
                                break;
                        case SV_IOMODE_RGBA:
                                s->hd_video_mode |= SV_MODE_COLOR_RGB_RGB;
                                s->frame->color_spec = RGBA;
                                break;
                }
        }
        s->hd_video_mode |= SV_MODE_STORAGE_FRAME;

        s->tile->width = s->mode->width;
        s->tile->height = s->mode->height;
        s->frame->fps = s->mode->fps;
        switch(s->mode->aux) {
                case AUX_PROGRESSIVE:
                        s->frame->interlacing = PROGRESSIVE;
                        break;
                case AUX_INTERLACED:
                        s->frame->interlacing = INTERLACED_MERGED;
                        break;
                case AUX_SF:
                        s->frame->interlacing = SEGMENTED_FRAME;
                        break;
        }


	s->tile->linesize = vc_get_linesize(s->tile->width, s->frame->color_spec);
	s->tile->data_len = s->tile->linesize * s->tile->height;


        //res = sv_videomode(s->sv, s->hd_video_mode);
        res = sv_option(s->sv, SV_OPTION_VIDEOMODE, s->hd_video_mode);
        if (res != SV_OK) {
                goto error;
        }
        res = sv_black(s->sv);
        if (res != SV_OK) {
                goto error;
        }



        if(s->grab_audio) {
                int i;
                res = sv_option(s->sv, SV_OPTION_AUDIOINPUT, SV_AUDIOINPUT_AIV);
                if (res != SV_OK) {
                        goto error;
                }
                /* TODO: figure out real channels number
                 * for now it suffices stereo */
                res = sv_option(s->sv, SV_OPTION_AUDIOCHANNELS, 1);
                if (res != SV_OK) {
                        goto error;
                }
                s->audio.ch_count = 2;

                sv_query(s->sv, SV_QUERY_AUDIOBITS, 0, &i);
                s->audio.bps = i / 8;
                sv_query(s->sv, SV_QUERY_AUDIOFREQ, 0, &i);
                s->audio.sample_rate = i;
                s->audio.data_len = 0;

                /* two 1-sec buffers */
                s->audio_bufs[0] = malloc(s->audio.sample_rate * s->audio.ch_count * s->audio.bps);
                s->audio_bufs[1] = malloc(s->audio.sample_rate * s->audio.ch_count * s->audio.bps);
        }

        res = sv_fifo_init(s->sv, &(s->fifo), 1, /* jack - must be 1 for default input FIFO */
                        0, /* obsolete - must be 0 */
                        SV_FIFO_DMA_ON, 
                        SV_FIFO_FLAG_NODMAADDR,
                        0 /*  frames in FIFO - 0 meens let API set the default maximal value*/
                        );
        if (res != SV_OK) {
                goto error;
                abort();
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
        s->bufs[0] = malloc(s->tile->data_len);
        s->bufs[1] = malloc(s->tile->data_len);
        s->bufs_index = 0;

        if (pthread_create
            (&(s->thread_id), NULL, vidcap_dvs_grab_thread, s) != 0) {
                perror("Unable to create grabbing thread");
                return NULL;
        }

        printf("Testcard capture set to %dx%d, bpp %f\n", s->tile->width, s->tile->height, get_bpp(s->frame->color_spec));

        debug_msg("DVS capture device enabled\n");
        return s;
error_detect:
         sv_close(s->sv);
error:
        free(s);
        printf("Error %s\n", sv_geterrortext(res));
        debug_msg("Unable to open grabber: %s\n", sv_geterrortext(res));
        return NULL;
}

void vidcap_dvs_finish(void *state)
{
        struct vidcap_dvs_state *s =
            (struct vidcap_dvs_state *)state;

        pthread_mutex_lock(&(s->lock));
        if(s->boss_waiting) {
                s->work_to_do = FALSE;
                pthread_cond_signal(&(s->boss_cv));
                while (!s->work_to_do) {
                        s->worker_waiting = TRUE;
                        pthread_cond_wait(&(s->worker_cv), &(s->lock));
                        s->worker_waiting = FALSE;
                }
        }

        s->work_to_do = TRUE;
        pthread_cond_signal(&(s->worker_cv));

        pthread_mutex_unlock(&(s->lock));
}

void vidcap_dvs_done(void *state)
{
        struct vidcap_dvs_state *s =
            (struct vidcap_dvs_state *)state;
        
         pthread_join(s->thread_id, NULL);

         sv_fifo_free(s->sv, s->fifo);
         sv_close(s->sv);
         vf_free(s->frame);
         free(s);
}

struct video_frame *vidcap_dvs_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_dvs_state *s =
            (struct vidcap_dvs_state *)state;

        pthread_mutex_lock(&(s->lock));

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
                s->tile->data = s->rtp_buffer;
        
                s->frames++;
                gettimeofday(&s->t, NULL);
                double seconds = tv_diff(s->t, s->t0);    
                if (seconds >= 5) {
                    float fps  = s->frames / seconds;
                    fprintf(stderr, "[DVS cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                    s->t0 = s->t;
                    s->frames = 0;
                }  

                if(s->grab_audio && s->audio.data_len) {
                        *audio = &s->audio;
                } else {
                        *audio = NULL;
                }
                return s->frame;
        }

        return NULL;
}

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


#endif                          /* HAVE_DVS */

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

#ifdef HAVE_HDSTATION		/* From config.h */

#include "debug.h"
#include "video_display.h"
#include "video_display/hdstation.h"
#include "video_codec.h"
#include "tv.h"

#include "dvs_clib.h"		/* From the DVS SDK */
#include "dvs_fifo.h"		/* From the DVS SDK */

#define HDSP_MAGIC	0x12345678

struct state_hdsp {
    pthread_t        thread_id;
    sv_handle       *sv;
    sv_fifo         *fifo;
    char            *frame_buffer;
    int              frame_size;
    sv_fifo_buffer  *fifo_buffer;
    sv_fifo_buffer  *display_buffer;
    sv_fifo_buffer  *tmp_buffer;
    pthread_mutex_t  lock;
    pthread_cond_t   boss_cv;
    pthread_cond_t   worker_cv;
    int              work_to_do;
    int              boss_waiting;
    int              worker_waiting;
    uint32_t         magic;
    char            *bufs[2];
    int              bufs_index;
    codec_t          codec;
    int              hd_video_mode;
    double           bpp;
};

static void*
display_thread_hd(void *arg)
{
	struct state_hdsp       *s = (struct state_hdsp *) arg;
	int			 res;

	while (1) {
		pthread_mutex_lock(&s->lock);

		while (s->work_to_do == FALSE) {
			s->worker_waiting = TRUE;
			pthread_cond_wait(&s->worker_cv, &s->lock);
			s->worker_waiting = FALSE;
		}

		s->display_buffer = s->tmp_buffer;
		s->work_to_do     = FALSE;

		if (s->boss_waiting) {
			pthread_cond_signal(&s->boss_cv);
		}
		pthread_mutex_unlock(&s->lock);

		res = sv_fifo_putbuffer(s->sv, s->fifo, s->display_buffer, NULL);
		if (res != SV_OK) {
			debug_msg("Error %s\n", sv_geterrortext(res));
			return NULL;
		}
	}
	return NULL;
}

char *
display_hdstation_getf(void *state)
{
	struct state_hdsp	*s = (struct state_hdsp *) state;
	int			 res;

	assert(s->magic == HDSP_MAGIC);

	/* Prepare the new RTP buffer... */
	res = sv_fifo_getbuffer(s->sv, s->fifo, &s->fifo_buffer, NULL, SV_FIFO_FLAG_VIDEOONLY | SV_FIFO_FLAG_FLUSH);
	if (res != SV_OK) {
		debug_msg("Error %s\n", sv_geterrortext(res));
		return NULL;
	}
	s->bufs_index   = (s->bufs_index + 1) % 2;
	s->frame_buffer = s->bufs[s->bufs_index];
	s->frame_size   = hd_size_x * hd_size_y * s->bpp;
	s->fifo_buffer->dma.addr = s->frame_buffer;
	s->fifo_buffer->dma.size = s->frame_size;

	return s->frame_buffer;
}

int
display_hdstation_putf(void *state, char *frame)
{
	struct state_hdsp	*s = (struct state_hdsp *) state;

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
	s->tmp_buffer  = s->fifo_buffer;
	s->fifo_buffer = NULL;
	s->work_to_do  = TRUE;

	/* ...and signal the worker */
	if (s->worker_waiting) {
		pthread_cond_signal(&s->worker_cv);
	}
	pthread_mutex_unlock(&s->lock);

	return TRUE;
}

void *
display_hdstation_init(char *fmt)
{
	struct state_hdsp   *s;
    int                  fps;
    int                  i;
	int                  res;

    if (fmt != NULL) {
        if (strcmp(fmt, "help") == 0) {
            printf("hdstation options:\n");
            printf("\tfps:codec\n");

            return 0;
        }   

        char *tmp;

        tmp = strtok(fmt, ":");
        if (!tmp) {
            fprintf(stderr, "Wrong config %s\n", fmt);
            return 0;
        }
        fps = atoi(tmp);
        tmp = strtok(NULL, ":");
        if (!tmp) {
            fprintf(stderr, "Wrong config %s\n", fmt);
            return 0;
        }
        s->codec = 0xffffffff;
        for(i = 0; codec_info[i].name != NULL; i++) {
                if(strcmp(tmp, codec_info[i].name) == 0) {
                    s->codec = codec_info[i].codec;
                    s->bpp = codec_info[i].bpp;
                }
        }
        if(s->codec == 0xffffffff) {
            fprintf(stderr, "hdstation: unknown codec: %s\n", tmp);
            free(s);
            free(tmp);
            return 0;
        }       
    }

    s->hd_video_mode=SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;

    if (s->codec == DVS10) {
        s->hd_video_mode |= SV_MODE_NBIT_10BDVS;
    }
    if (fps == 25) {
        s->hd_video_mode |= SV_MODE_SMPTE274_25P;
    }
    else if (fps == 29) {
        s->hd_video_mode |= SV_MODE_SMPTE274_29I;
    }
    else {
        fprintf(stderr, "Wrong framerate in config %s\n", fmt);
        return 0;
    }

	/* Start the display thread... */
	s = (struct state_hdsp *) malloc(sizeof(struct state_hdsp));
	s->magic        = HDSP_MAGIC;
	s->frame_size   = 0;
	s->frame_buffer = 0;

	s->sv  = sv_open("");
	if (s->sv == NULL) {
		debug_msg("Cannot open HDTV display device\n");
		return NULL;
	}
	res = sv_videomode(s->sv, s->hd_video_mode | SV_MODE_AUDIO_NOAUDIO);
	if (res != SV_OK) {
		debug_msg("Cannot set videomode %s\n", sv_geterrortext(res));
	        return NULL;
	}
	res = sv_sync_output(s->sv, SV_SYNCOUT_BILEVEL);
	if (res != SV_OK) {
		debug_msg("Cannot enable sync-on-green %s\n", sv_geterrortext(res));
	        return NULL;
	}

	res = sv_fifo_init(s->sv, &s->fifo, 0, 1, 1, 0, 0);
	if (res != SV_OK) {
		debug_msg("Cannot initialize video display FIFO %s\n", sv_geterrortext(res));
	        return NULL;
	}
	res = sv_fifo_start(s->sv, s->fifo);
	if (res != SV_OK) {
		debug_msg("Cannot start video display FIFO  %s\n", sv_geterrortext(res));
	        return NULL;
	}

	pthread_mutex_init(&s->lock, NULL);
	pthread_cond_init(&s->boss_cv, NULL);
	pthread_cond_init(&s->worker_cv, NULL);
	s->work_to_do     = FALSE;
	s->boss_waiting   = FALSE;
	s->worker_waiting = FALSE;
	s->display_buffer = NULL;

	s->bufs[0] = malloc(hd_size_x * hd_size_y * s->bpp);
	s->bufs[1] = malloc(hd_size_x * hd_size_y * s->bpp);
	s->bufs_index = 0;
	memset(s->bufs[0], 0, hd_size_x * hd_size_y * s->bpp);
	memset(s->bufs[1], 0, hd_size_x * hd_size_y * s->bpp);

	if (pthread_create(&(s->thread_id), NULL, display_thread_hd, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	return (void *) s;
}

void
display_hdstation_done(void *state)
{
	struct state_hdsp *s = (struct state_hdsp *) state;

	sv_fifo_free(s->sv, s->fifo);
	sv_close(s->sv);
	free(s);
}

display_colour_t
display_hdstation_colour(void *state)
{
	struct state_hdsp *s = (struct state_hdsp *) state;

	assert(s->magic == HDSP_MAGIC);

	return DC_YUV;
}

display_type_t *
display_hdstation_probe(void)
{
	display_type_t		*dtype;
	display_format_t	*dformat;
	sv_handle		*sv;
	
	/* Probe the hardware... */
	sv  = sv_open("");
	if (sv == NULL) {
		debug_msg("Cannot probe HDTV display device\n");
		return NULL;
	}
	sv_close(sv);

	dformat = malloc(sizeof(display_format_t));
	if (dformat == NULL) {
		return NULL;
	}
	dformat->size        = DS_1920x1080;
	dformat->colour_mode = DC_YUV;
	dformat->num_images  = 1;

	dtype = malloc(sizeof(display_type_t));
	if (dtype != NULL) {
		dtype->id	   = DISPLAY_HDSTATION_ID;
		dtype->name        = "hdtv";
		dtype->description = "DVS HDstation (1080i/60 YUV 4:2:2)";
		dtype->formats     = dformat;
		dtype->num_formats = 1;
	}
	return dtype;
}

#endif /* HAVE_HDSTATION */


/*
 * FILE:    quad.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/aggregate.h"
#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>

#include "video_display.h"
#include "video.h"

/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Aggregate capture\n");
        printf("Usage\n");
        printf("\t-t aggregate:<dev1_config>#<dev2_config>[#....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in an aggregate device\n");

}



struct vidcap_aggregate_state {
        struct vidcap     **devices;
        int                 devices_cnt;

        struct video_frame       *frame; 
        int frames;
        struct       timeval t, t0;

        unsigned int        grab_audio:1; /* wheather we process audio or not */
};


struct vidcap_type *
vidcap_aggregate_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_AGGREGATE_ID;
		vt->name        = "aggregate";
		vt->description = "Aggregate video capture";
	}
	return vt;
}

void *
vidcap_aggregate_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_aggregate_state *s;
        char *save_ptr = NULL;
        char *item;
        char *parse_string;
        char *tmp;
        int i;

	printf("vidcap_aggregate_init\n");


        s = (struct vidcap_aggregate_state *) calloc(1, sizeof(struct vidcap_aggregate_state));
	if(s == NULL) {
		printf("Unable to allocate aggregate capture state\n");
		return NULL;
	}

        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(!init_fmt || strcmp(init_fmt, "help") == 0) {
                show_help();
                return NULL;
        }


        s->devices_cnt = 0;
        tmp = parse_string = strdup(init_fmt);
        while(strtok_r(tmp, "#", &save_ptr)) {
                s->devices_cnt++;
                tmp = NULL;
        }
        free(parse_string);

        s->devices = calloc(1, s->devices_cnt * sizeof(struct vidcap *));
        i = 0;
        tmp = parse_string = strdup(init_fmt);
        while((item = strtok_r(tmp, "#", &save_ptr))) {
                char *device;
                char *config = strdup(item);
                char *device_cfg = NULL;
                unsigned int dev_flags = 0u;
                device = config;
		if(strchr(config, ':')) {
			char *delim = strchr(config, ':');
			*delim = '\0';
			device_cfg = delim + 1;
		}

                if(i == 0) {
                        dev_flags = flags;
                } else { // do not grab from second and other devices
                        dev_flags = flags & ~(VIDCAP_FLAG_AUDIO_EMBEDDED | VIDCAP_FLAG_AUDIO_AESEBU | VIDCAP_FLAG_AUDIO_ANALOG);
                }

                s->devices[i] = initialize_video_capture(device,
                                               device_cfg, dev_flags);
                free(config);
                if(!s->devices[i]) {
                        fprintf(stderr, "[aggregate] Unable to initialize device %d (%s:%s).\n", i, device, device_cfg);
                        goto error;
                }
                ++i;
                tmp = NULL;
        }
        free(parse_string);

        s->frame = vf_alloc(s->devices_cnt);
        
	return s;

error:
        if(s->devices) {
                int i;
                for (i = 0u; i < s->devices_cnt; ++i) {
                        if(s->devices[i]) {
                                 vidcap_done(s->devices[i]);
                        }
                }
        }
        free(s);
        return NULL;
}

void
vidcap_aggregate_finish(void *state)
{
	struct vidcap_aggregate_state *s = (struct vidcap_aggregate_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->devices_cnt; ++i) {
                         vidcap_finish(s->devices[i]);
		}
	}
}

void
vidcap_aggregate_done(void *state)
{
	struct vidcap_aggregate_state *s = (struct vidcap_aggregate_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->devices_cnt; ++i) {
                         vidcap_done(s->devices[i]);
		}
	}
        
        vf_free(s->frame);
}

struct video_frame *
vidcap_aggregate_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_aggregate_state *s = (struct vidcap_aggregate_state *) state;
        struct audio_frame *audio_frame;
        struct video_frame *frame = NULL;
        int i;

        while(!frame) {
                frame = vidcap_grab(s->devices[0], &audio_frame);
        }
        s->frame->color_spec = frame->color_spec;
        s->frame->interlacing = frame->interlacing;
        s->frame->fps = frame->fps;
        vf_get_tile(s->frame, 0)->width = vf_get_tile(frame, 0)->width;
        vf_get_tile(s->frame, 0)->height = vf_get_tile(frame, 0)->height;
        vf_get_tile(s->frame, 0)->data_len = vf_get_tile(frame, 0)->data_len;
        vf_get_tile(s->frame, 0)->data = vf_get_tile(frame, 0)->data;
        if(audio_frame) {
                *audio = audio_frame;
        } else {
                *audio = NULL;
        }
        for(i = 1; i < s->devices_cnt; ++i) {
                frame = NULL;
                while(!frame) {
                        frame = vidcap_grab(s->devices[i], &audio_frame);
                }
                if(frame->color_spec != s->frame->color_spec ||
                                frame->fps != s->frame->fps ||
                                frame->interlacing != s->frame->interlacing) {
                        fprintf(stderr, "[aggregate] Different format detected: ");
                        if(frame->color_spec != s->frame->color_spec)
                                fprintf(stderr, "codec");
                        if(frame->interlacing != s->frame->interlacing)
                                fprintf(stderr, "interlacing");
                        if(frame->fps != s->frame->fps)
                                fprintf(stderr, "FPS (%.2f and %.2f)", frame->fps, s->frame->fps);
                        fprintf(stderr, "\n");
                        
                        return NULL;
                }
                vf_get_tile(s->frame, i)->width = vf_get_tile(frame, 0)->width;
                vf_get_tile(s->frame, i)->height = vf_get_tile(frame, 0)->height;
                vf_get_tile(s->frame, i)->data_len = vf_get_tile(frame, 0)->data_len;
                vf_get_tile(s->frame, i)->data = vf_get_tile(frame, 0)->data;
        }
        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "[aggregate cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

	return s->frame;
}


/**
 * @file   video_capture/aggregate.c
 * @author Martin Pulec <pulec@cesnet.cz>
 *
 * @brief Aggregate video capture driver
 */
/*
 * Copyright (c) 2012-2023 CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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
 */
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/types.h"

#include <stdio.h>
#include <stdlib.h>

/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Aggregate capture\n");
        printf("Usage\n");
        printf("\t-t aggregate -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in an aggregate device\n");

}

struct vidcap_aggregate_state {
        struct vidcap     **devices;
        int                 devices_cnt;

        struct video_frame      **captured_frames;
        struct video_frame       *frame; 
        int frames;
        struct       timeval t, t0;

        int          audio_source_index;
};


static void vidcap_aggregate_probe(struct device_info **cards, int *count, void (**deleter)(void *))
{
        *cards = NULL;
        *count = 0;
        *deleter = free;
}

static int
vidcap_aggregate_init(struct vidcap_params *params, void **state)
{
	struct vidcap_aggregate_state *s;

	printf("vidcap_aggregate_init\n");


        s = (struct vidcap_aggregate_state *) calloc(1, sizeof(struct vidcap_aggregate_state));
	if(s == NULL) {
		printf("Unable to allocate aggregate capture state\n");
		return VIDCAP_INIT_FAIL;
	}

        s->audio_source_index = -1;
        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "") != 0) {
                show_help();
                free(s);
                return VIDCAP_INIT_NOERR;
        }


        s->devices_cnt = 0;
        struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
                if (vidcap_params_get_driver(tmp) != NULL)
                        s->devices_cnt++;
                else
                        break;
        }

        s->devices = calloc(s->devices_cnt, sizeof(struct vidcap *));
        tmp = params;
        for (int i = 0; i < s->devices_cnt; ++i) {
                tmp = vidcap_params_get_next(tmp);

                int ret = initialize_video_capture(NULL, (struct vidcap_params *) tmp, &s->devices[i]);
                if(ret != 0) {
                        fprintf(stderr, "[aggregate] Unable to initialize device %d (%s:%s).\n",
                                        i, vidcap_params_get_driver(tmp),
                                        vidcap_params_get_fmt(tmp));
                        goto error;
                }
        }

        s->captured_frames = calloc(s->devices_cnt, sizeof(struct video_frame *));

        s->frame = vf_alloc(s->devices_cnt);
        
        *state = s;
	return VIDCAP_INIT_OK;

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
        return VIDCAP_INIT_FAIL;
}

static void
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

static struct video_frame *
vidcap_aggregate_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_aggregate_state *s = (struct vidcap_aggregate_state *) state;
        struct audio_frame *audio_frame = NULL;
        struct video_frame *frame = NULL;

        for (int i = 0; i < s->devices_cnt; ++i) {
                VIDEO_FRAME_DISPOSE(s->captured_frames[i]);
        }

        *audio = NULL;

        for (int i = 0; i < s->devices_cnt; ++i) {
                frame = NULL;
                while(!frame) {
                        frame = vidcap_grab(s->devices[i], &audio_frame);
                }
                if (i == 0) {
                        s->frame->color_spec = frame->color_spec;
                        s->frame->interlacing = frame->interlacing;
                        s->frame->fps = frame->fps;
                }
                if (s->audio_source_index == -1 && audio_frame != NULL) {
                        fprintf(stderr, "[aggregate] Locking device #%d as an audio source.\n",
                                        i);
                        s->audio_source_index = i;
                }
                if (s->audio_source_index == i) {
                        *audio = audio_frame;
                }
                if (frame->color_spec != s->frame->color_spec ||
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
                s->captured_frames[i] = frame;
        }
        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            log_msg(LOG_LEVEL_INFO, "[aggregate cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

	return s->frame;
}

static const struct video_capture_info vidcap_aggregate_info = {
        vidcap_aggregate_probe,
        vidcap_aggregate_init,
        vidcap_aggregate_done,
        vidcap_aggregate_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(aggregate, &vidcap_aggregate_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


/*
 * FILE:    video_display/sage.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"
#include "video_display/aggregate.h"

#include <assert.h>
#include <host.h>

#include <video_codec.h>

#define MAGIC_AGGREGATE DISPLAY_AGGREGATE_ID

/* defined in main.c */
struct display *initialize_video_display(const char *requested_display,
                char *fmt, unsigned int flags);


struct display_aggregate_state {
        struct display        **devices;
        pthread_t              *threads;
        unsigned int            devices_cnt;
        struct video_frame     *frame;
        struct tile            *tile;

        /* For debugging... */
        uint32_t magic;

        int                     frames;
        struct timeval          t, t0;
};

static void show_help(void);
static void *aggregate_thread(void *);

static void *aggregate_thread(void *arg)
{
        display_run((struct display *) arg);
        return NULL;
}

static void show_help() {
        printf("Aggregate display\n");
        printf("Usage:\n");
        printf("\t-t aggregate:<dev1_config>#<dev2_config>[#....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in an aggregate device\n");
}

void display_aggregate_run(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *) state;
        unsigned int i;

        for (i = 0; i < s->devices_cnt; i++) {
                pthread_create(&s->threads[i], NULL,
                                  aggregate_thread, s->devices[i]);
        }
        for (i = 0; i < s->devices_cnt; i++) {
                pthread_join(s->threads[i], NULL);
        }
}

void *display_aggregate_init(char *fmt, unsigned int flags)
{
        UNUSED(fmt);
        UNUSED(flags);
        struct display_aggregate_state *s;
        char *save_ptr = NULL;
        char *item;
        char *parse_string;
        char *tmp;
        int i;


        if(!fmt || strcmp(fmt, "help") == 0) {
                show_help();
                return NULL;
        }
        
        s = (struct display_aggregate_state *) calloc(1, sizeof(struct display_aggregate_state));
        s->magic = MAGIC_AGGREGATE;

        s->devices_cnt = 0;
        tmp = parse_string = strdup(fmt);
        while(strtok_r(tmp, "#", &save_ptr)) {
                s->devices_cnt++;
                tmp = NULL;
        }
        free(parse_string);

        s->devices = calloc(1, s->devices_cnt * sizeof(struct display *));
        i = 0;
        tmp = parse_string = strdup(fmt);
        while((item = strtok_r(tmp, "#", &save_ptr))) {
                char *device;
                char *config = strdup(item);
                char *save_ptr_dev = NULL;
                char *device_cfg;
                unsigned int dev_flags = 0u;
                device = strtok_r(config, ":", &save_ptr_dev);
                device_cfg = save_ptr_dev;
                if(i == 0 && flags == DISPLAY_FLAG_ENABLE_AUDIO) {
                        dev_flags = DISPLAY_FLAG_ENABLE_AUDIO;
                }

                s->devices[i] = initialize_video_display(device,
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
        s->threads = calloc(1, sizeof(pthread_t));

        return (void *)s;

error:
        if(s->devices) {
                unsigned int i;
                for (i = 0u; i < s->devices_cnt; ++i) {
                        if(s->devices[i]) {
                                 display_done(s->devices[i]);
                         }
                }
        }
        free(s);
        return NULL;
}

void display_aggregate_done(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *) state;

        assert(s != NULL);
        assert(s->magic == MAGIC_AGGREGATE);

        if (s != NULL) {
                unsigned int i;
                for (i = 0; i < s->devices_cnt; ++i) {
                         display_done(s->devices[i]);
                 }
        }
                                        
        vf_free(s->frame);
}

void display_aggregate_finish(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *) state;

        assert(s != NULL);
        assert(s->magic == MAGIC_AGGREGATE);

        if (s != NULL) {
                unsigned int i;
                for (i = 0; i < s->devices_cnt; ++i) {
                         display_finish(s->devices[i]);
                 }
        }
}

struct video_frame *display_aggregate_getf(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        unsigned int i;
        struct video_frame *frame;

        assert(s != NULL);
        assert(s->magic == MAGIC_AGGREGATE);


        for(i = 0; i < s->devices_cnt; ++i) {
                frame = display_get_frame(s->devices[i]);
                vf_get_tile(s->frame, i)->data = frame->tiles[0].data;
                vf_get_tile(s->frame, i)->data_len = frame->tiles[0].data_len;
        }

        return s->frame;
}

int display_aggregate_putf(void *state, char *frame)
{
        unsigned int i;
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;

        assert(s->magic == MAGIC_AGGREGATE);
        UNUSED(frame);
        for(i = 0; i < s->devices_cnt; ++i) {
                display_put_frame(s->devices[i], frame);
        }

        return 0;
}

int display_aggregate_reconfigure(void *state, struct video_desc desc)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        unsigned int i;
        int ret = FALSE;

        assert(s->magic == MAGIC_AGGREGATE);
        
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->frame->color_spec = desc.color_spec;

        for(i = 0; i < s->devices_cnt; ++i) {
                ret = display_reconfigure(s->devices[i], desc);
                if(!ret)
                        break;
        }


        return ret;
}

display_type_t *display_aggregate_probe(void)
{
        display_type_t *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_AGGREGATE_ID;
                dt->name = "aggregate";
                dt->description = "Aggregate video display";
        }
        return dt;
}

int display_aggregate_get_property(void *state, int property, void *val, size_t *len)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        unsigned int i;
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        {
                                unsigned int codec_idx;
                                unsigned int save_codec_idx = 0;
                                codec_t **codecs = malloc(s->devices_cnt * sizeof(codec_t *));
                                size_t *lens = malloc(s->devices_cnt * sizeof(size_t));

                                for (i = 0; i < s->devices_cnt; ++i) {
                                        codecs[i] = malloc(*len);
                                        lens[i] = *len;
                                }
                                for (i = 0; i < s->devices_cnt; ++i) {
                                        int ret;
                                        ret = display_get_property(s->devices[i], DISPLAY_PROPERTY_CODECS, codecs[i], &lens[i]);
                                        if(!ret) {
                                                goto err_codecs;
                                        }
                                }

                                /* for each codec check if is included in all drivers */
                                for(codec_idx = 0; codec_idx < lens[0] / sizeof(codec_t); codec_idx++) {
                                        unsigned int found = 0;
                                        codec_t examined = codecs[0][codec_idx];
                                        for (i = 1; i < s->devices_cnt; ++i) {
                                                unsigned int sub_codec;
                                                found = FALSE;
                                                for(sub_codec = 0; sub_codec < lens[i] / sizeof(codec_t); ++sub_codec) {
                                                        if(examined == codecs[i][sub_codec]) {
                                                                ++found;
                                                        }
                                                }
                                        }

                                        /* is included in all drivers */
                                        if(found == s->devices_cnt - 1) {
                                                ((codec_t *) val)[save_codec_idx++] = examined;
                                        }
                                }

                                *len = save_codec_idx * sizeof(codec_t);
                                for(i = 0; i < s->devices_cnt; ++i) {
                                        free(codecs[i]);
                                }
                                free(codecs);
                                free(lens);
                                return TRUE;
err_codecs:
                                for(i = 0; i < s->devices_cnt; ++i) {
                                        free(codecs[i]);
                                }
                                free(codecs);
                                free(lens);
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                case DISPLAY_PROPERTY_GSHIFT:
                case DISPLAY_PROPERTY_BSHIFT:
                case DISPLAY_PROPERTY_BUF_PITCH:
                        {
                                int ret;
                                int first_val;
                                size_t size;
                                ret = display_get_property(s->devices[0], property, &first_val, &size);
                                if(!ret) goto err;

                                for (i = 1; i < s->devices_cnt; ++i) {
                                        int new_val;
                                        ret = display_get_property(s->devices[i], property, &new_val, &size);
                                        if(!ret) goto err;
                                        if(new_val != first_val)
                                                goto err;

                                }
                                *len = size;
                                *(int *) val = first_val;
                                return TRUE;
err:
                                return FALSE;

                        }
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        if(s->devices_cnt == 1)
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                        else
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        break;

                default:
                        return FALSE;
        }
        return TRUE;
}

struct audio_frame * display_aggregate_get_audio_frame(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        return display_get_audio_frame(s->devices[0]);
}

void display_aggregate_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        double seconds = tv_diff(s->t, s->t0);    

        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "[aggregate disp.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

        display_put_audio_frame(s->devices[0], frame);
}

int display_aggregate_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        return display_reconfigure_audio(s->devices[0], quant_samples, channels, sample_rate);
}

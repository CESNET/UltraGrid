/**
 * @file   video_display/aggregate.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s. p. o.
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


#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"

#include <assert.h>
#include <host.h>

#define MAGIC_AGGREGATE 0xbbcaa321

struct display_aggregate_state {
        struct display        **devices;
        pthread_t              *threads;
        unsigned int            devices_cnt;
        struct video_frame     *frame;
        struct video_frame     **dev_frames;
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

static void display_aggregate_run(void *state)
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

static void display_aggregate_probe(struct device_info **available_cards, int *count)
{
        *available_cards = NULL;
        *count = 0;
}

static void *display_aggregate_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        struct display_aggregate_state *s;
        char *save_ptr = NULL;
        char *item;
        char *parse_string = NULL;;
        char *tmp;
        int i;

        if(!fmt || strcmp(fmt, "help") == 0) {
                show_help();
                return &display_init_noerr;
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
                } else { // push audio only to first device
                        dev_flags = flags & ~(DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG);
                }

                int ret = initialize_video_display(parent, device,
                                               device_cfg, dev_flags, NULL, &s->devices[i]);
                if(ret != 0) {
                        fprintf(stderr, "[aggregate] Unable to initialize device %d (%s:%s).\n", i, device, device_cfg);
                        free(config);
                        goto error;
                }
                free(config);
                ++i;
                tmp = NULL;
        }
        free(parse_string);

        s->frame = vf_alloc(s->devices_cnt);
        s->dev_frames = calloc(s->devices_cnt, sizeof(struct video_frame *));
        s->threads = calloc(s->devices_cnt, sizeof(pthread_t));

        return (void *)s;

error:
        free(parse_string);
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

static void display_aggregate_done(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *) state;

        assert(s != NULL);
        assert(s->magic == MAGIC_AGGREGATE);

        for (unsigned int i = 0; i < s->devices_cnt; ++i) {
                display_done(s->devices[i]);
        }
                                        
        vf_free(s->frame);
        free(s->dev_frames);
        free(s);
}

static struct video_frame *display_aggregate_getf(void *state)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        unsigned int i;
        struct video_frame *frame;

        assert(s != NULL);
        assert(s->magic == MAGIC_AGGREGATE);


        for(i = 0; i < s->devices_cnt; ++i) {
                frame = display_get_frame(s->devices[i]);
                s->dev_frames[i] = frame;
                vf_get_tile(s->frame, i)->data = frame->tiles[0].data;
                vf_get_tile(s->frame, i)->data_len = frame->tiles[0].data_len;
        }

        return s->frame;
}

static int display_aggregate_putf(void *state, struct video_frame *frame, int nonblock)
{
        unsigned int i;
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;

        assert(s->magic == MAGIC_AGGREGATE);
        for(i = 0; i < s->devices_cnt; ++i) {
                struct video_frame *dev_frame = NULL;
                if(frame)
                        dev_frame = s->dev_frames[i];
                int ret = display_put_frame(s->devices[i], dev_frame, nonblock);
                if(ret != 0)
                        return ret;
        }

        return 0;
}

static int display_aggregate_reconfigure(void *state, struct video_desc desc)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        unsigned int i;
        int ret = FALSE;

        assert(s->magic == MAGIC_AGGREGATE);
        
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->frame->color_spec = desc.color_spec;

	desc.tile_count = 1;
        for(i = 0; i < s->devices_cnt; ++i) {
                ret = display_reconfigure(s->devices[i], desc, VIDEO_NORMAL);
                if(!ret)
                        break;
        }

        return ret;
}

static int display_aggregate_get_property(void *state, int property, void *val, size_t *len)
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
                case DISPLAY_PROPERTY_RGB_SHIFT:
                case DISPLAY_PROPERTY_BUF_PITCH:
                        {
                                int ret;
                                char first_val[128];
                                size_t first_size = sizeof(first_val);
                                ret = display_get_property(s->devices[0], property, &first_val, &first_size);
                                if(!ret) goto err;

                                for (i = 1; i < s->devices_cnt; ++i) {
                                        char new_val[128];
                                        size_t new_size = sizeof(new_val);
                                        ret = display_get_property(s->devices[i], property, &new_val, &new_size);
                                        if(!ret) goto err;
                                        if(new_size != first_size || memcmp(first_val, new_val, first_size) != 0)
                                                goto err;

                                }
                                if (first_size > *len)
                                        goto err;
                                *len = first_size;
                                memcpy(val, first_val, first_size);
                                return TRUE;
err:
                                return FALSE;

                        }
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

static void display_aggregate_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        double seconds = tv_diff(s->t, s->t0);    

        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            log_msg(LOG_LEVEL_INFO, "[aggregate disp.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

        display_put_audio_frame(s->devices[0], frame);
}

static int display_aggregate_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct display_aggregate_state *s = (struct display_aggregate_state *)state;
        return display_reconfigure_audio(s->devices[0], quant_samples, channels, sample_rate);
}

static const struct video_display_info display_aggregate_info = {
        display_aggregate_probe,
        display_aggregate_init,
        display_aggregate_run,
        display_aggregate_done,
        display_aggregate_getf,
        display_aggregate_putf,
        display_aggregate_reconfigure,
        display_aggregate_get_property,
        display_aggregate_put_audio_frame,
        display_aggregate_reconfigure_audio,
};

REGISTER_MODULE(aggregate, &display_aggregate_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


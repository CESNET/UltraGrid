/**
 * @file   video_capture/muxer.c
 * @author Gerard Castillo <gerard.castillo@i2cat.net>
 *
 * @brief Muxer of video capturers, N to 1.
 */
/*
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 *
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
 * 3. Neither the name of Fundació i2CAT nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
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
#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video.h"
#include "video_capture.h"

#include "capture_filter.h"
#include "messaging.h"
#include "module.h"

#include "tv.h"

#include "video_capture/muxer.h"
#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>

struct vidcap_muxer_state {
        struct module mod;

        struct vidcap     **devices;
        int                 devices_cnt;

        struct video_frame       *frame;
        struct video_frame       *prev_frame;
        int frames;
        struct       timeval t, t0;

        int         dev_index_curr;
        int         dev_index_next;

        int          audio_source_index;
};

/**
 * MUXER REMOTE CONTROL
 */
static int init(struct module *parent,void *state);
static void done(void *state);
static bool parse(struct vidcap_muxer_state *s, char *cfg)
{
        char *item, *save_ptr;
        while ((item = strtok_r(cfg, ":", &save_ptr)))
            s->dev_index_next = atoi(item);

        return true;
}

static int init(struct module *parent, void *state)
{
        struct vidcap_muxer_state *s = calloc(1, sizeof(struct vidcap_muxer_state));
        assert(s);

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, parent);

        return 0;
}
static void done(void *state)
{
        struct vidcap_muxer_state *s = state;
        module_done(&s->mod);
        if(state!=NULL)free(state);
}
static void process_message(struct vidcap_muxer_state *s, struct msg_universal *msg)
{
        parse(s, msg->text);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct vidcap_muxer_state *s = (struct vidcap_muxer_state *) state;
        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                process_message(s, (struct msg_universal *) msg);
                free_message(msg);
        }
        return in;
}


struct capture_filter_info capture_filter_muxer = {
        .name = "muxer",
        .init = init,
        .done = done,
        .filter = filter,
};
/**
 * MUXER CAPTURE
 */
/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Muxer capture\n");
        printf("Usage\n");
        printf("\t-t muxer -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in the muxer device\n");

}


struct vidcap_type *
vidcap_muxer_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_MUXER_ID;
		vt->name        = "muxer";
		vt->description = "Muxer video capture";
	}
	return vt;
}

void *
vidcap_muxer_init(const struct vidcap_params *params)
{
    struct vidcap_muxer_state *s;
    int i;

    printf("vidcap_muxer_init\n");

    s = (struct vidcap_muxer_state *) calloc(1,
        sizeof(struct vidcap_muxer_state));
    if (s == NULL) {
        printf("Unable to allocate muxer capture state\n");
        return NULL;
    }

    s->audio_source_index = -1;
    s->frames = 0;
    s->dev_index_curr = 0;
    s->dev_index_next = 0;

    gettimeofday(&s->t0, NULL);

    if (vidcap_params_get_fmt(params)
        && strcmp(vidcap_params_get_fmt(params), "") != 0)
    {
        show_help();
        return &vidcap_init_noerr;
    }

    s->devices_cnt = 0;
    const struct vidcap_params *tmp = params;
    while ((tmp = vidcap_params_get_next(tmp))) {
        if (vidcap_params_get_driver(tmp) != NULL)
            s->devices_cnt++;
        else
            break;
    }

    s->devices = calloc(s->devices_cnt, sizeof(struct vidcap *));
    i = 0;
    tmp = params;
    for (int i = 0; i < s->devices_cnt; ++i) {
        tmp = vidcap_params_get_next(tmp);

        int ret = initialize_video_capture(NULL, tmp, &s->devices[i]);
        if (ret != 0) {
            fprintf(stderr, "[muxer] Unable to initialize device %d (%s:%s).\n",
                i, vidcap_params_get_driver(tmp), vidcap_params_get_fmt(tmp));
            goto error;
        }
    }

    s->frame = vf_alloc(1);

    return s;

    error: if (s->devices) {
        int i;
        for (i = 0u; i < s->devices_cnt; ++i) {
            if (s->devices[i]) {
                vidcap_done(s->devices[i]);
            }
        }
    }
    free(s);
    return NULL;
}

void
vidcap_muxer_done(void *state)
{
	struct vidcap_muxer_state *s = (struct vidcap_muxer_state *) state;

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
vidcap_muxer_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_muxer_state *s = (struct vidcap_muxer_state *) state;
    struct audio_frame *audio_frame = NULL;
    struct video_frame *frame_curr = NULL;
    struct video_frame *frame_next = NULL;
    struct video_frame *frame = NULL;

    /**
     * remote control (now keyboard handler)
     */
    set_conio_terminal_mode();
    int c;
    if (kbhit()) {
        c = (int) getch();
        debug_msg("num %d pressed...\r\n", c);
    }
    if(c >= 48 && c < 48+s->devices_cnt){
       // s->dev_index_curr = s->dev_index_next;
        s->dev_index_next = c - 48;
        printf("\n[MUXER] device %d (previous was: %d) selected of %d devices...\r\n",s->dev_index_next,s->dev_index_curr,s->devices_cnt);
    }
    reset_terminal_mode();

//    /**
//     * REMOTE CONTROL
//     */
//    struct message *msg;
//    while ((msg = check_message(&s->mod))) {
//            process_message(s,((struct msg_universal *) msg)->text);
//            free_message(msg);
//    }

    /**
     * vidcap_grap
     */
    while (!frame_curr) {
        frame_curr = vidcap_grab(s->devices[s->dev_index_curr], &audio_frame);
    }
    if(s->dev_index_next != s->dev_index_curr){
        while (!frame_next) {
                frame_next = vidcap_grab(s->devices[s->dev_index_next], &audio_frame);
        }

        printf("\n\n[MUXER] frameNext is std = %d...\n\n",frame_next->isStd);

        if(frame_next->isStd == TRUE){
            printf("\n\n[MUXER] RTSP INCOMING FRAME...\n");
            if(frame_next->h264_iframe == TRUE){
                s->dev_index_curr = s->dev_index_next;
                frame_curr = frame_next;
                printf("[MUXER] GOT INTRA FRAME! SWITCHING...\n\n");
            }else printf("[MUXER] NO INTRA FRAME YET...\n");
        }else{
            frame_curr = frame_next;
            s->dev_index_curr = s->dev_index_next;
        }
    }
    frame = frame_curr;

    if (audio_frame) {
        *audio = audio_frame;
    } else {
        *audio = NULL;
    }
    if (s->audio_source_index == -1 && audio_frame != NULL) {
        fprintf(stderr, "[muxer] Locking device #%d as an audio source.\n",
            s->dev_index_curr);
        s->audio_source_index = s->dev_index_curr;
    }
    if (s->audio_source_index == s->dev_index_curr) {
        *audio = audio_frame;
    }
//                if (frame->color_spec != s->frame->color_spec ||
//                                frame->fps != s->frame->fps ||
//                                frame->interlacing != s->frame->interlacing) {
//                        fprintf(stderr, "[muxer] Different format detected: ");
//                        if(frame->color_spec != s->frame->color_spec)
//                                fprintf(stderr, "codec");
//                        if(frame->interlacing != s->frame->interlacing)
//                                fprintf(stderr, "interlacing");
//                        if(frame->fps != s->frame->fps)
//                                fprintf(stderr, "FPS (%.2f and %.2f)", frame->fps, s->frame->fps);
//                        fprintf(stderr, "\n");
//
//                        //return NULL;
//                }
    if (frame != NULL) {
        vf_get_tile(s->frame, 0)->width = vf_get_tile(frame, 0)->width;
        vf_get_tile(s->frame, 0)->height = vf_get_tile(frame, 0)->height;
        vf_get_tile(s->frame, 0)->data_len = vf_get_tile(frame, 0)->data_len;
        vf_get_tile(s->frame, 0)->data = vf_get_tile(frame, 0)->data;
        s->frame->color_spec = frame->color_spec;
        s->frame->interlacing = frame->interlacing;
        if (frame->fps == 0)
            s->frame->fps = 15;
        else
            s->frame->fps = frame->fps;
    } else
        return NULL;

    s->frames++;
    gettimeofday(&s->t, NULL);
    double seconds = tv_diff(s->t, s->t0);
    if (seconds >= 5) {
        float fps = s->frames / seconds;
        fprintf(stderr, "[muxer] %d frames in %g seconds = %g FPS\n", s->frames,
            seconds, fps);
        s->t0 = s->t;
        s->frames = 0;
    }

	return s->frame;
}

/**
 * KEYBOARD HANDLER
 */

struct termios orig_termios;

void reset_terminal_mode()
{
    tcsetattr(0, TCSANOW, &orig_termios);
}

void set_conio_terminal_mode()
{
    struct termios new_termios;

    /* take two copies - one for now, one for later */
    tcgetattr(0, &orig_termios);
    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

    /* register cleanup handler, and set the new terminal mode */
    atexit(reset_terminal_mode);
    cfmakeraw(&new_termios);
    tcsetattr(0, TCSANOW, &new_termios);
}

int kbhit()
{
    struct timeval tv = { 0L, 0L };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

int getch()
{
    int r;
    unsigned char c;
    if ((r = read(0, &c, sizeof(c))) < 0) {
        return r;
    } else {
    return c;
    }
}


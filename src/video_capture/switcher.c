/**
 * @file   video_capture/switcher.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET z.s.p.o.
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
#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/switcher.h"
#include "audio/audio.h"
#include "module.h"

#include <stdio.h>
#include <stdlib.h>

/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("switcher capture\n");
        printf("Usage\n");
        printf("\t-t switcher[:excl_init] -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\tdevn_config is a configuration of device to be switched\n");
        printf("\t\texcl_init - devices will be initialized after switching to and deinitialized after switching to another\n");

}

struct vidcap_switcher_state {
        struct module       mod;
        struct vidcap     **devices;
        int                 devices_cnt;

        int                 selected_device;

        const struct vidcap_params *params;
        bool                excl_init;
};


struct vidcap_type *
vidcap_switcher_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = 0x1D3E1956;
		vt->name        = "switcher";
		vt->description = "Video switcher pseudodevice";
	}
	return vt;
}

void *
vidcap_switcher_init(const struct vidcap_params *params)
{
	struct vidcap_switcher_state *s;
        int i;

	printf("vidcap_switcher_init\n");

        s = (struct vidcap_switcher_state *) calloc(1, sizeof(struct vidcap_switcher_state));
	if(s == NULL) {
		printf("Unable to allocate switcher capture state\n");
		return NULL;
	}

        if (vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "") != 0) {
                if (strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                        show_help();
                        return &vidcap_init_noerr;
                } else if (strcmp(vidcap_params_get_fmt(params), "excl_init") == 0) {
                        s->excl_init = true;
                } else {
                        show_help();
                        fprintf(stderr, "[switcher] Unknown initialization option!\n");
                        return NULL;
                }
        }

        s->devices_cnt = 0;
        const struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
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

                if (!s->excl_init || i == s->selected_device) {
                        int ret = initialize_video_capture(NULL, tmp, &s->devices[i]);
                        if(ret != 0) {
                                fprintf(stderr, "[switcher] Unable to initialize device %d (%s:%s).\n",
                                                i, vidcap_params_get_driver(tmp),
                                                vidcap_params_get_fmt(tmp));
                                goto error;
                        }
                }
        }

        s->params = params;

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, vidcap_params_get_parent(params));

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
vidcap_switcher_done(void *state)
{
	struct vidcap_switcher_state *s = (struct vidcap_switcher_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->devices_cnt; ++i) {
                        if (!s->excl_init || i == s->selected_device) {
                                vidcap_done(s->devices[i]);
                        }
		}
	}
        module_done(&s->mod);
        free(s);
}

struct video_frame *
vidcap_switcher_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_switcher_state *s = (struct vidcap_switcher_state *) state;
        struct audio_frame *audio_frame = NULL;
        struct video_frame *frame = NULL;

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                struct msg_universal *msg_univ = (struct msg_universal *) msg;
                int new_selected_device = atoi(msg_univ->text);

                if (new_selected_device >= 0 && new_selected_device < s->devices_cnt)
                        if (s->excl_init) {
                                vidcap_done(s->devices[s->selected_device]);
                                int ret = initialize_video_capture(NULL,
                                                vidcap_params_get_nth(s->params, new_selected_device + 1),
                                                &s->devices[new_selected_device]);
                                assert(ret == 0);
                        }

                        s->selected_device = new_selected_device;

                free_message(msg);
        }

        frame = vidcap_grab(s->devices[s->selected_device], &audio_frame);
        *audio = audio_frame;;

	return frame;
}


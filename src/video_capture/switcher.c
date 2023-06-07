/**
 * @file   video_capture/switcher.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET z.s.p.o.
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
#include "utils/macros.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/types.h"
#include "module.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define MOD_NAME "[switcher] "

/* prototypes of functions defined in this module */
static void show_help(void);
static void vidcap_switcher_done(void *state);

static void show_help()
{
        printf("switcher capture\n");
        printf("Usage\n");
        printf("\t--control-port <port> -t switcher[:excl_init][:fallback] -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\t<devn_config> is a configuration of device to be switched\n");
        printf("\t\t<port> specifies port which should be used to control switching\n");
        printf("\t\texcl_init - devices will be initialized after switching to and deinitialized after switching to another\n");
        printf("\t\tfallback - in case that capture doesn't return a frame (in time), capture from next available device(s)\n");
}

struct vidcap_switcher_state {
        struct module       mod;
        struct vidcap     **devices;
        unsigned int        devices_cnt;

        unsigned int        selected_device;

        struct vidcap_params *params;
        bool                excl_init;
        bool                fallback;
};


static struct vidcap_type *
vidcap_switcher_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        *deleter = free;
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->name        = "switcher";
		vt->description = "Video switcher pseudodevice";
	}
	return vt;
}

static void vidcap_switcher_register_keyboard_ctl(struct vidcap_switcher_state *s) {
        for (unsigned int i = 0U; i < MIN(s->devices_cnt, 10); ++i) {
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                sprintf(m->text, "map %d capture.data %d#switch to video input %d", i + 1, i, i + 1);
                struct response *r = send_message_sync(get_root_module(&s->mod), "keycontrol", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) != RESPONSE_OK) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot register keyboard control for video switcher (error %d)!\n", response_get_status(r));
                }
                free_response(r);
        }
}

static int
vidcap_switcher_init(struct vidcap_params *params, void **state)
{
	struct vidcap_switcher_state *s;

	printf("vidcap_switcher_init\n");

        s = (struct vidcap_switcher_state *) calloc(1, sizeof(struct vidcap_switcher_state));
	if(s == NULL) {
		printf("Unable to allocate switcher capture state\n");
		return VIDCAP_INIT_FAIL;
	}

        const char *cfg_c = vidcap_params_get_fmt(params);
        if (cfg_c && strcmp(cfg_c, "") != 0) {
                char *cfg = strdup(cfg_c);
                char *save_ptr, *item;
                char *tmp = cfg;
                assert(cfg != NULL);
                while ((item = strtok_r(cfg, ":", &save_ptr))) {
                        if (strcmp(item, "help") == 0) {
                                show_help();
                                free(tmp);
                                free(s);
                                return VIDCAP_INIT_NOERR;
                        } else if (strcmp(item, "excl_init") == 0) {
                                s->excl_init = true;
                        } else if (strcmp(item, "fallback") == 0) {
                                s->fallback = true;
                        } else if (strncasecmp(item, "select=", strlen("select=")) == 0) {
                                char *val_s = item + strlen("select=");
                                char *endptr = NULL;;
                                errno = 0;
                                long val = strtol(val_s, &endptr, 0);
                                if (errno != 0 || *val_s == '\0' || *endptr != '\0' || val < 0 || (uintmax_t) val > UINT_MAX) {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong value: %s\n", val_s);
                                        free(tmp);
                                        goto error;
                                }
                                s->selected_device = val;
                        } else {
                                fprintf(stderr, "[switcher] Unknown initialization option!\n");
                                show_help();
                                free(tmp);
                                free(s);
                                return VIDCAP_INIT_FAIL;
                        }
                        cfg = NULL;
                }
                free(tmp);
        }

        if (s->excl_init && s->fallback) {
                fprintf(stderr, MOD_NAME "Options \"excl_init\" and \"fallback\" are mutualy incompatible!\n");
                goto error;
        }

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, vidcap_params_get_parent(params));
        s->devices_cnt = 0;
        struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
                if (vidcap_params_get_driver(tmp) == NULL) {
                        break;
                }
                s->devices_cnt++;
        }

        if (s->selected_device >= s->devices_cnt) {
                fprintf(stderr, "[switcher] Error: device #%d not available!\n", s->selected_device);
                goto error;
        }

        s->devices = calloc(s->devices_cnt, sizeof(struct vidcap *));
        tmp = params;
        for (unsigned int i = 0; i < s->devices_cnt; ++i) {
                tmp = vidcap_params_get_next(tmp);

                if (!s->excl_init || i == s->selected_device) {
                        int ret = initialize_video_capture(&s->mod, tmp, &s->devices[i]);
                        if(ret != 0) {
                                fprintf(stderr, "[switcher] Unable to initialize device %d (%s:%s).\n",
                                                i, vidcap_params_get_driver(tmp),
                                                vidcap_params_get_fmt(tmp));
                                goto error;
                        }
                }
        }

        s->params = params;

        vidcap_switcher_register_keyboard_ctl(s);

        *state = s;
	return VIDCAP_INIT_OK;

error:
        vidcap_switcher_done(s);
        return VIDCAP_INIT_FAIL;
}

static void
vidcap_switcher_done(void *state)
{
	struct vidcap_switcher_state *s = (struct vidcap_switcher_state *) state;
        if (s == NULL) {
                return;
        }

	if (s->devices != NULL) {
		for (unsigned int i = 0U; i < s->devices_cnt; ++i) {
                        if (s->devices[i] != NULL) {
                                vidcap_done(s->devices[i]);
                        }
		}
	}
        module_done(&s->mod);
        free(s);
}

static struct video_frame *
vidcap_switcher_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_switcher_state *s = (struct vidcap_switcher_state *) state;
        struct audio_frame *audio_frame = NULL;
        struct video_frame *frame = NULL;

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                struct msg_universal *msg_univ = (struct msg_universal *) msg;
                char *endptr = NULL;
                errno = 0;
                long val = strtol(msg_univ->text, &endptr, 0);
                if (errno != 0 || val < 0 || (uintmax_t) val > UINT_MAX || msg_univ->text[0] == '\0' || *endptr != '\0') {
                        log_msg(LOG_LEVEL_ERROR, "[switcher] Cannot switch to device %s. Wrong value.\n", msg_univ->text);
                        free_message(msg, new_response(RESPONSE_BAD_REQUEST, NULL));
                        continue;
                }
                unsigned int new_selected_device = val;
                struct response *r;

                if (new_selected_device < s->devices_cnt){
                        log_msg(LOG_LEVEL_NOTICE, "[switcher] Switched to device %d.\n", new_selected_device);
                        if (s->excl_init) {
                                vidcap_done(s->devices[s->selected_device]);
                                s->devices[s->selected_device] = NULL;
                                int ret = initialize_video_capture(NULL,
                                                vidcap_params_get_nth((struct vidcap_params *) s->params, new_selected_device + 1),
                                                &s->devices[new_selected_device]);
                                assert(ret == 0);
                        }

                        s->selected_device = new_selected_device;
                        r = new_response(RESPONSE_OK, NULL);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[switcher] Cannot switch to device %d. Device out of bounds.\n", new_selected_device);
                        r = new_response(RESPONSE_BAD_REQUEST, NULL);
                }
                free_message(msg, r);
        }

        frame = vidcap_grab(s->devices[s->selected_device], &audio_frame);
        *audio = audio_frame;

        if (frame || !s->fallback) {
                return frame;
        }
        // if frame was not returned but we have a fallback behavior, try also other devices
        for (unsigned int i = (s->selected_device + 1U) % s->devices_cnt;
                        i != s->selected_device;
                        i = (s->selected_device + 1U) % s->devices_cnt) {
                frame = vidcap_grab(s->devices[i], &audio_frame);
                *audio = audio_frame;
                if (frame != NULL) {
                        break;
                }
        }
        return frame;
}

static const struct video_capture_info vidcap_switcher_info = {
        vidcap_switcher_probe,
        vidcap_switcher_init,
        vidcap_switcher_done,
        vidcap_switcher_grab,
        MOD_NAME,
};

REGISTER_MODULE(switcher, &vidcap_switcher_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


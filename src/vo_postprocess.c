/**
 * @file   vo_postprocess.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2020 CESNET, z. s. p. o.
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
/**
 * @file
 * @todo
 * Fix complex filters that 1. change video properties (tiling mode, codec) or
 * 2. producing more frames (eg. double_framerate).
 *
 * Ideas for 2.:
 * - change API - do not count output frames but always pass NULL to flush the queue
 *   until there are frames
 * - process the filter queueue backwards - first pass NULL to filter <n>, then <n>-1
 *   up to first
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <string.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/list.h"
#include "video_codec.h"
#include "video_display.h"
#include "vo_postprocess.h"

#define MOD_NAME "[vo_postprocess] "

struct vo_postprocess_state_single {
        const struct vo_postprocess_info *funcs;
        void *state;
        struct video_frame *f;
};

struct vo_postprocess_state {
        struct simple_linked_list *postprocessors;
};

void show_vo_postprocess_help(bool full)
{
        printf("Usage:\n");
        color_printf(TERM_BOLD "\t-p <postprocess_module>[:<args>] | help\n" TERM_RESET);
        printf("\n");
        printf("Possible postprocess modules:\n");
        list_modules(LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION, full);
}

static _Bool init(struct vo_postprocess_state *s, const char *config_string) {
        char *cpy = strdup(config_string);
        char *tmp = cpy;
        char *save_ptr = NULL;
        char *item = NULL;

        while ((item = strtok_r(tmp, ",", &save_ptr)) != NULL) {
                const char *vo_postprocess_options = "";
                char *lib_name = item;
                if (strchr(lib_name, ':')) {
                        vo_postprocess_options = strchr(lib_name, ':') + 1;
                        *strchr(lib_name, ':') = '\0';
                }
                const struct vo_postprocess_info *funcs = load_library(lib_name, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
                if (!funcs) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown postprocess module: %s\n", lib_name);
                        free(cpy);
                        return 0;
                }
                struct vo_postprocess_state_single *state = malloc(sizeof(struct vo_postprocess_state_single));
                state->funcs = funcs;
                state->state = state->funcs->init(vo_postprocess_options);
                if (!state->state) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Postprocessing initialization failed: %s\n", config_string);
                        free(cpy);
                        free(state);
                        return 0;
                }
                simple_linked_list_append(s->postprocessors, state);

                tmp = NULL;
        }
        free(cpy);
        return 1;
}

struct vo_postprocess_state *vo_postprocess_init(const char *config_string)
{
        if (!config_string) {
                return NULL;
        }

        if (strcmp(config_string, "help") == 0 || strcmp(config_string, "fullhelp") == 0) {
                show_vo_postprocess_help(strcmp(config_string, "fullhelp") == 0);
                return NULL;
        }

        struct vo_postprocess_state *s = (struct vo_postprocess_state *) malloc(sizeof(struct vo_postprocess_state));
        s->postprocessors = simple_linked_list_init();

        if (!init(s, config_string)) {
                vo_postprocess_done(s);
                return NULL;
        }

        return s;
}

int vo_postprocess_reconfigure(struct vo_postprocess_state *s,
                struct video_desc desc)
{
        if (s == NULL) {
                return FALSE;
        }

        bool filter_complex = false;

        for(void *it = simple_linked_list_it_init(s->postprocessors); it != NULL; ) {
                struct vo_postprocess_state_single *state = simple_linked_list_it_next(&it);
                int ret = state->funcs->reconfigure(state->state, desc);
                if (ret == FALSE) {
                        simple_linked_list_it_destroy(it);
                        return FALSE;
                }
                // get desc for next iteration
                int display_mode = 0;
                int out_frames_count = 0;
                state->funcs->get_out_desc(state->state, &desc, &display_mode, &out_frames_count);

                // check if convert is simple (doesn't change display mode and out_fr_count == 1); if not, only one filter allowed
                if (display_mode != DISPLAY_PROPERTY_VIDEO_MERGED || out_frames_count != 1) {
                        filter_complex = true;
                }
        }

        if (filter_complex && simple_linked_list_size(s->postprocessors) > 1) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "One of postprocessors is complex (changing properties) and "
                                "cannot be used in filter chain (more postprocessors)!\n");
                return FALSE;
        }

        return TRUE;
}

struct video_frame * vo_postprocess_getf(struct vo_postprocess_state *s)
{
        if (s == NULL) {
                return NULL;
        }

        struct video_frame *out = NULL;

        for(void *it = simple_linked_list_it_init(s->postprocessors); it != NULL; ) {
                struct vo_postprocess_state_single *state = simple_linked_list_it_next(&it);
                state->f = state->funcs->getf(state->state);
                if (state->f == NULL) {
                        simple_linked_list_it_destroy(it);
                        return NULL;
                }
                if (out == NULL) {
                        out = state->f;
                }
        }

        return out;
}

/**
 * param in  will be NULL if vo_postprocess_get_out_desc() returns out_frames_count > 1 and will be called
 *           out_frames_count-1 times
 */
bool vo_postprocess(struct vo_postprocess_state *s, struct video_frame *in,
                struct video_frame *out, int req_pitch)
{
        if (s == NULL) {
                return FALSE;
        }
        assert(in == ((struct vo_postprocess_state_single *) simple_linked_list_first(s->postprocessors))->f || in == NULL);

        for(void *it = simple_linked_list_it_init(s->postprocessors); it != NULL; ) {
                struct vo_postprocess_state_single *state = simple_linked_list_it_next(&it);
                struct video_frame *next = out;
                if (it != NULL) {
                        struct vo_postprocess_state_single *state_next = simple_linked_list_it_peek_next(it);
                        next = state_next->f;
                }

                int pitch = vc_get_linesize(next->tiles[0].width, next->color_spec);
                if (it == NULL) {
                        pitch = req_pitch;
                }
                bool ret = state->funcs->vo_postprocess(state->state, in, next, pitch);
                if (!ret) {
                        return false;
                }
                in = next;
        }

        return true;
}

void vo_postprocess_done(struct vo_postprocess_state *s)
{
        if (s == NULL) {
                return;
        }
        for(void *it = simple_linked_list_it_init(s->postprocessors); it != NULL; ) {
                struct vo_postprocess_state_single *state = simple_linked_list_it_next(&it);
                state->funcs->done(state->state);
                free(state);
        }
        simple_linked_list_destroy(s->postprocessors);
        free(s);
}

void vo_postprocess_get_out_desc(struct vo_postprocess_state *s, struct video_desc *out, int *display_mode, int *out_frames_count)
{
        if (s == NULL) {
                return;
        }
        struct vo_postprocess_state_single *state = simple_linked_list_last(s->postprocessors);
        state->funcs->get_out_desc(state->state, out, display_mode, out_frames_count);
}

bool vo_postprocess_get_property(struct vo_postprocess_state *s, int property, void *val, size_t *len)
{
        if (s == NULL) {
                return false;
        }

        if (simple_linked_list_size(s->postprocessors) == 1 || property == VO_PP_DOES_CHANGE_TILING_MODE) {
                struct vo_postprocess_state_single *state = simple_linked_list_last(s->postprocessors);
                return state->funcs->get_property(state, property, val, len);
        }

        /** @todo
         * This is not corrrect in a generic case - the codec may be acceptable for first filter but
         * the output of the filter may be unacceptable for the following filter (or later on).
         */
        if (property == VO_PP_PROPERTY_CODECS) {
                struct vo_postprocess_state_single *state = simple_linked_list_first(s->postprocessors);
                return state->funcs->get_property(state, property, val, len);
        }

        return false;
}


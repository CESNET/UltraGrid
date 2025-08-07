/**
 * @file   capture_filter/disrupt.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Capture filter aimed to development to introduce some disruptions that may
 * occur during regular use like jitter between frames.
 */
/*
 * Copyright (c) 2021-2025 CESNET
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


#include <stdio.h>            // for printf, NULL
#include <stdlib.h>           // for free, atof, calloc
#include <string.h>           // for strchr, strcmp, strlen, strstr

#include "capture_filter.h"

#include "compat/usleep.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/random.h"

#define MOD_NAME "[disrupt c. f.] "

struct module;
struct video_frame;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_disrupt {
        double jitter_ms;
};

static void usage() {
        printf("Disrupts video frame flow\n\n");
        printf("disrupt usage:\n");
        printf("\tdisrupt:jitter[=<ms>]\n");
        printf("where:\n");
        printf("\tjitter - add random jitter to frame timing (value in ms - maximal delay)\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (cfg == NULL || strlen(cfg) == 0 || strcmp(cfg, "help") == 0) {
                usage();
                return cfg && strcmp(cfg, "help") == 0 ? 1 : -1;
        }

        struct state_disrupt *s = calloc(1, sizeof *s);

        if (strstr(cfg, "jitter") == cfg) {
                if (strchr(cfg, '=') != NULL) {
                        s->jitter_ms = atof(strchr(cfg, '=') + 1);
                }
        } else {
                usage();
                color_printf(TERM_FG_RED MOD_NAME "Currently only supported filter is jitter\n" TERM_FG_RESET);
                free(s);
                return -1;
        }

        *state = s;

        return 0;
}

static void done(void *state)
{
        free(state);
}


static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_disrupt *s = state;

        usleep(1000.0 * ug_drand() * s->jitter_ms);

        return in;
}

static const struct capture_filter_info capture_filter_disrupt = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_HIDDEN_MODULE(disrupt, &capture_filter_disrupt, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);


/*
 * FILE:    capture_filter.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"
#include "utils/list.h"
#include "video.h"

#include "capture_filter/blank.h"
#include "capture_filter/every.h"

static struct capture_filter_info *capture_filters[] = {
        &capture_filter_blank,
        &capture_filter_every
};

struct capture_filter {
        struct simple_linked_list *filters;
};

struct capture_filter_instance {
        int index;
        void *state;
};

int capture_filter_init(const char *cfg, struct capture_filter **state)
{
        struct capture_filter *s = calloc(1, sizeof(struct capture_filter));
        char *item, *save_ptr;
        assert(s);
        char *filter_list_str = NULL,
             *tmp = NULL;

        s->filters = simple_linked_list_init();

        if(cfg) {
                if(strcasecmp(cfg, "help") == 0) {
                        printf("Available capture filters:\n");
                        for(unsigned int i = 0;
                                        i < sizeof(capture_filters) / sizeof(struct capture_filter_info *); ++i) {
                                printf("\t%s\n", capture_filters[i]->name);
                        }
                        return 1;
                }
                filter_list_str = tmp = strdup(cfg);

                while((item = strtok_r(filter_list_str, ",", &save_ptr))) {
                        unsigned int i;
                        char filter_name[128];
                        char *options = NULL;
                        strncpy(filter_name, item, sizeof(filter_name));
                        if(strchr(filter_name, ':')) {
                                options = strchr(filter_name, ':') + 1;
                                *strchr(filter_name, ':') = '\0';
                        }
                        for(i = 0; i < sizeof(capture_filters) / sizeof(struct capture_filter_info *); ++i) {
                                if(strcasecmp(capture_filters[i]->name, filter_name) == 0) {
                                        struct capture_filter_instance *instance =
                                                malloc(sizeof(struct capture_filter_instance));
                                        instance->index = i;
                                        int ret = capture_filters[i]->init(options, &instance->state);
                                        if(ret < 0) {
                                                fprintf(stderr, "Unable to initialize capture filter: %s\n",
                                                                filter_name);
                                        }
                                        if(ret != 0) {
                                                return ret;
                                        }
                                        simple_linked_list_append(s->filters, instance);
                                        break;
                                }
                        }
                        if(i == sizeof(capture_filters) / sizeof(struct capture_filter_info *)) {
                                fprintf(stderr, "Unable to find capture filter: %s\n",
                                                filter_name);
                                return -1;
                        }
                        filter_list_str = NULL;
                }
        }

        free(tmp);

        *state = s;

        return 0;
}

void capture_filter_destroy(struct capture_filter *state)
{
        struct capture_filter *s = state;

        while(simple_linked_list_size(s->filters) > 0) {
                struct capture_filter_instance *inst = simple_linked_list_pop(s->filters);
                capture_filters[inst->index]->done(inst->state);
                free(inst);
        }

        simple_linked_list_destroy(s->filters);

        free(state);
}

struct video_frame *capture_filter(struct capture_filter *state, struct video_frame *frame) {
        struct capture_filter *s = state;

        for(void *it = simple_linked_list_it_init(s->filters);
                        it != NULL;
           ) {
                struct capture_filter_instance *inst = simple_linked_list_it_next(&it);
                frame = capture_filters[inst->index]->filter(inst->state, frame);
                if(!frame)
                        return NULL;
        }
        return frame;
}


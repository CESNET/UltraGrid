/**
 * @file   capture_filter.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET
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

#include "capture_filter.h"

#include <cassert>            // for assert
#include <cstdio>             // for printf, fprintf, stderr
#include <cstdlib>            // for free, NULL, atoi, calloc, malloc
#include <cstring>            // for strchr, strcmp, strdup, strlen, strncmp

#include "compat/strings.h"   // for strcasecmp
#include "lib_common.h"       // for get_libraries_for_class, library_class
#include "messaging.h"        // for msg_universal, new_response, RESPONSE_I...
#include "module.h"           // for module, module_done, module_init_default
#include "utils/color_out.h"  // for color_printf, TERM_BOLD, TERM_RESET
#include "utils/list.h"       // for simple_linked_list_pop, simple_linked_l...

using namespace std;

struct capture_filter {
        struct module mod;
        struct simple_linked_list *filters;
};

struct capture_filter_instance {
        const struct capture_filter_info *functions;
        void *state;
};

static int create_filter(struct capture_filter *s, char *cfg)
{
        bool found = false;
        const char *options = "";
        char *filter_name = cfg;
        if(strchr(filter_name, ':')) {
                options = strchr(filter_name, ':') + 1;
                *strchr(filter_name, ':') = '\0';
        }
        const auto & capture_filters = get_libraries_for_class(LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
        for (auto && item : capture_filters) {
                auto capture_filter_info = static_cast<const struct capture_filter_info*>(item.second);
                if(strcasecmp(item.first.c_str(), filter_name) == 0) {
                        struct capture_filter_instance *instance = (struct capture_filter_instance *)
                                malloc(sizeof(struct capture_filter_instance));
                        instance->functions = capture_filter_info;
                        int ret = capture_filter_info->init(&s->mod, options, &instance->state);
                        if(ret < 0) {
                                fprintf(stderr, "Unable to initialize capture filter: %s\n",
                                                filter_name);
                        }
                        if(ret != 0) {
                                free(instance);
                                return ret;
                        }
                        simple_linked_list_append(s->filters, instance);
                        found = true;
                        break;
                }
        }
        if (!found) {
                fprintf(stderr, "Unable to find capture filter: %s\n",
                                filter_name);
                return -1;
        }
        return 0;
}

int capture_filter_init(struct module *parent, const char *cfg, struct capture_filter **state)
{
        if (cfg && (strcasecmp(cfg, "help") == 0 || strcasecmp(cfg, "fullhelp") == 0)) {
                printf("Usage:\n");
                color_printf(TERM_BOLD "\t--capture-filter <filter1>[:opts][,<filter2>[:opts][,<filter3>[:<opts>]]]" TERM_RESET " -t <capture>\n\n");
                printf("Available capture filters:\n");
                list_modules(LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION, strcasecmp(cfg, "fullhelp") == 0);
                if (strcasecmp(cfg, "fullhelp") != 0) {
                        printf("(use \"fullhelp\" to show hidden filters)\n");
                }
                return 1;
        }

        struct capture_filter *s = (struct capture_filter *) calloc(1, sizeof(struct capture_filter));
        char *item, *save_ptr;
        assert(s);
        char *filter_list_str = NULL,
             *tmp = NULL;

        s->filters = simple_linked_list_init();

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_FILTER;
        module_register(&s->mod, parent);

        if(cfg) {
                filter_list_str = tmp = strdup(cfg);

                while((item = strtok_r(filter_list_str, ",", &save_ptr))) {
                        char filter_name[128];
                        strncpy(filter_name, item, sizeof filter_name - 1);

                        int ret = create_filter(s, filter_name);
                        if (ret != 0) {
                                free(tmp);
                                capture_filter_destroy(s);
                                return ret;
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
                struct capture_filter_instance *inst = (struct capture_filter_instance *) simple_linked_list_pop(s->filters);
                inst->functions->done(inst->state);
                free(inst);
        }

        simple_linked_list_destroy(s->filters);

        module_done(&s->mod);

        free(state);
}

static struct response *process_message(struct capture_filter *s, struct msg_universal *msg)
{
        if (strncmp("delete ", msg->text, strlen("delete ")) == 0) {
                int index = atoi(msg->text + strlen("delete "));
                struct capture_filter_instance *inst = (struct capture_filter_instance *)
                        simple_linked_list_remove_index(s->filters, index);
                if (!inst) {
                        fprintf(stderr, "Unable to remove capture filter index %d.\n",
                                        index);
                        return new_response(RESPONSE_INT_SERV_ERR, NULL);
                } else {
                        printf("Capture filter #%d removed successfully.\n", index);
                        inst->functions->done(inst->state);
                        free(inst);
                }
        } else if (strcmp("flush", msg->text) == 0) {
                while(simple_linked_list_size(s->filters) > 0) {
                        struct capture_filter_instance *inst = (struct capture_filter_instance *) simple_linked_list_pop(s->filters);
                        inst->functions->done(inst->state);
                        free(inst);
                }
        } else if (strcmp("help", msg->text) == 0) {
                printf("Capture filter control:\n"
                                "\tflush      - remove all filters\n"
                                "\tdelete <x> - delete x-th filter\n"
                                "\t<filter>   - append a filter named <filter>\n");
        } else {
                char *fmt = strdup(msg->text);
                if (create_filter(s, fmt) != 0) {
                        fprintf(stderr, "Cannot create capture filter: %s.\n",
                                        msg->text);
                        free(fmt);
                        return new_response(RESPONSE_INT_SERV_ERR, NULL);
                } else {
                        printf("Capture filter \"%s\" created successfully.\n",
                                        msg->text);
                }
                free(fmt);
        }

        return new_response(RESPONSE_OK, NULL);
}

struct video_frame *capture_filter(struct capture_filter *state, struct video_frame *frame) {
        struct capture_filter *s = state;

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                struct response *r = process_message(s, (struct msg_universal *) msg);
                free_message(msg, r);
        }

        for(void *it = simple_linked_list_it_init(s->filters);
                        it != NULL;
           ) {
                struct capture_filter_instance *inst = (struct capture_filter_instance *) simple_linked_list_it_next(&it);
                frame = inst->functions->filter(inst->state, frame);
                if(!frame)
                        return NULL;
        }
        return frame;
}


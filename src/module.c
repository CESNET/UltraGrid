/*
 * FILE:    module.c
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
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "module.h"
#include "utils/list.h"

void module_init_default(struct module *module_data)
{
        memset(module_data, 0, sizeof(struct module));

        pthread_mutexattr_t attr;
        assert(pthread_mutexattr_init(&attr) == 0);
        assert(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) == 0);
        assert(pthread_mutex_init(&module_data->lock, &attr) == 0);
        pthread_mutexattr_destroy(&attr);

        module_data->childs = simple_linked_list_init();
        module_data->msg_queue = simple_linked_list_init();

        module_data->magic = MODULE_MAGIC;
}

void module_register(struct module *module_data, struct module *parent)
{
        if(parent) {
                module_data->parent = parent;
                pthread_mutex_lock(&module_data->parent->lock);
                simple_linked_list_append(module_data->parent->childs, module_data);
                pthread_mutex_unlock(&module_data->parent->lock);
        }
}

void module_done(struct module *module_data)
{
        if(!module_data)
                return;

        assert(module_data->magic == MODULE_MAGIC);

        if(module_data->parent) {
                pthread_mutex_lock(&module_data->parent->lock);
                int found;

                found = simple_linked_list_remove(module_data->parent->childs, module_data);

                assert(found == TRUE);
                pthread_mutex_unlock(&module_data->parent->lock);
        }

        // we assume that deleter may dealloc space where are structure stored
        pthread_mutex_lock(&module_data->lock);
        struct module tmp;
        memcpy(&tmp, module_data, sizeof(struct module));
        pthread_mutex_unlock(&module_data->lock);

        if(module_data->deleter)
                module_data->deleter(module_data);

        pthread_mutex_destroy(&tmp.lock);

        if(simple_linked_list_size(tmp.childs) > 0) {
                fprintf(stderr, "Warning: Child database not empty!\n");
        }
        simple_linked_list_destroy(tmp.childs);

        if(simple_linked_list_size(tmp.msg_queue) > 0) {
                fprintf(stderr, "Warning: Message queue not empty!\n");
        }
        simple_linked_list_destroy(tmp.msg_queue);
}

const char *module_class_name_pairs[] = {
        [MODULE_CLASS_ROOT] = "root",
        [MODULE_CLASS_PORT] = "port",
        [MODULE_CLASS_COMPRESS] = "compress",
        [MODULE_CLASS_DATA] = "data",
        [MODULE_CLASS_SENDER] = "sender",
        [MODULE_CLASS_RECEIVER] = "receiver",
        [MODULE_CLASS_TX] = "transmit",
        [MODULE_CLASS_AUDIO] = "audio",
        [MODULE_CLASS_CONTROL] = "control",
};

const char *module_class_name(enum module_class cls)
{
        if((unsigned int) cls > sizeof(module_class_name_pairs)/sizeof(const char *))
                return NULL;
        else
                return module_class_name_pairs[cls];
}

void append_message_path(char *buf, int buflen, enum module_class modules[])
{
        enum module_class *mod = modules;

        while(*mod != MODULE_CLASS_NONE) {
                if(strlen(buf) > 0) {
                        strncat(buf, ".", buflen - strlen(buf) - 1);
                }
                const char *node_name = module_class_name(*mod);
                assert(node_name != NULL);

                strncat(buf, node_name, buflen - strlen(buf) - 1);
                mod += 1;
        }
}

struct module *get_root_module(struct module *node)
{
        while(node->parent) {
                node = node->parent;
        }
        assert(node->cls == MODULE_CLASS_ROOT);

        return node;
}

static struct module *find_child(struct module *node, const char *node_name, int index)
{
        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = (struct module *) simple_linked_list_it_next(&it);
                if(strcasecmp(module_class_name(child->cls), node_name) == 0) {
                        if(index-- == 0) {
                                return child;
                        }
                }
        }
        return NULL;
}

static void get_receiver_index(char *node_str, int *index) {
        *index = 0;
        if(strchr(node_str, '[')) {
                *index = atoi(strchr(node_str, '[') + 1);
                *strchr(node_str, '[') = '\0';
        }
}

struct module *get_module(struct module *root, const char *const_path)
{
        struct module *receiver = root;
        char *path, *tmp;
        char *item, *save_ptr;

        assert(root != NULL);

        pthread_mutex_lock(&receiver->lock);

        tmp = path = strdup(const_path);
        while ((item = strtok_r(path, ".", &save_ptr))) {
                struct module *old_receiver = receiver;
                int index;
                get_receiver_index(item, &index);
                receiver = find_child(receiver, item, index);
                if(!receiver) {
                        pthread_mutex_unlock(&old_receiver->lock);
                        break;
                }
                pthread_mutex_lock(&receiver->lock);
                pthread_mutex_unlock(&old_receiver->lock);

                path = NULL;

        }
        free(tmp);

        return receiver;
}

void dump_tree(struct module *node, int indent) {
        for(int i = 0; i < indent; ++i) putchar(' ');

        printf("%s\n", module_class_name(node->cls));

        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = simple_linked_list_it_next(&it);
                dump_tree(child, indent + 2);
        }
}

void unlock_module(struct module *module)
{
        pthread_mutex_unlock(&module->lock);
}


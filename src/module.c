/**
 * @file   module.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET z.s.p.o.
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

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "module.h"
#include "utils/list.h"

#define MOD_NAME "[module] "

void module_init_default(struct module *module_data)
{
        int ret = 0;
        memset(module_data, 0, sizeof(struct module));

        pthread_mutexattr_t attr;
        ret |= pthread_mutexattr_init(&attr);
        ret |= pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        ret |= pthread_mutex_init(&module_data->lock, &attr);
        ret |= pthread_mutex_init(&module_data->msg_queue_lock, &attr);
        ret |= pthread_mutexattr_destroy(&attr);
        assert(ret == 0 && "Unable to create mutex or set attributes");

        module_data->childs = simple_linked_list_init();
        module_data->msg_queue = simple_linked_list_init();
        module_data->msg_queue_childs = simple_linked_list_init();

        module_data->magic = MODULE_MAGIC;
}

static void
module_mutex_lock(pthread_mutex_t *lock)
{
        MSG(DEBUG, "Locking lock %p\n", lock);
        pthread_mutex_lock(lock);
}

static void
module_mutex_unlock(pthread_mutex_t *lock)
{
        MSG(DEBUG, "Unlocking lock %p\n", lock);
        pthread_mutex_unlock(lock);
}

void module_register(struct module *module_data, struct module *parent)
{
        if(parent) {
                module_data->parent = parent;
                module_mutex_lock(&module_data->parent->lock);
                simple_linked_list_append(module_data->parent->childs, module_data);
                module_check_undelivered_messages(module_data->parent);
                module_mutex_unlock(&module_data->parent->lock);
        }
}

void module_done(struct module *module_data)
{
        if(!module_data)
                return;

        if (module_data->cls == MODULE_CLASS_NONE)
                return;

        assert(module_data->magic == MODULE_MAGIC);

        if(module_data->parent) {
                module_mutex_lock(&module_data->parent->lock);
                bool found = simple_linked_list_remove(
                    module_data->parent->childs, module_data);
                assert(found);
                module_mutex_unlock(&module_data->parent->lock);
        }

        // we assume that deleter may dealloc space where are structure stored
        module_mutex_lock(&module_data->lock);
        struct module tmp;
        memcpy(&tmp, module_data, sizeof(struct module));
        module_mutex_unlock(&module_data->lock);

        module_data->cls = MODULE_CLASS_NONE;

        if(module_data->deleter)
                module_data->deleter(module_data);

        if(simple_linked_list_size(tmp.childs) > 0) {
                log_msg(LOG_LEVEL_WARNING, "Warning: Child database not empty! Remaining:\n");
                dump_tree(&tmp, 0);
                module_mutex_lock(&tmp.lock);
                for(void *it = simple_linked_list_it_init(module_data->childs); it != NULL; ) {
                        struct module *child = simple_linked_list_it_next(&it);
                        module_mutex_lock(&child->lock);
                        child->parent = NULL;
                        module_mutex_unlock(&child->lock);
                }
                module_mutex_unlock(&tmp.lock);
        }
        simple_linked_list_destroy(tmp.childs);

        if(simple_linked_list_size(tmp.msg_queue) > 0) {
                fprintf(stderr, "Warning: Message queue not empty!\n");
                if (log_level >= LOG_LEVEL_VERBOSE) {
                        printf("Path: ");
                        dump_parents(&tmp);
                }
                struct message *m;
                while ((m = check_message(&tmp))) {
                        free_message(m, NULL);
                }
        }
        simple_linked_list_destroy(tmp.msg_queue);

        while (simple_linked_list_size(tmp.msg_queue_childs) > 0) {
                struct message *m = simple_linked_list_pop(tmp.msg_queue_childs);
                free_message_for_child(m, NULL);
        }
        simple_linked_list_destroy(tmp.msg_queue_childs);

        pthread_mutex_destroy(&tmp.lock);

        free(tmp.name);
}

static const char *module_class_name_pairs[] = {
        [MODULE_CLASS_ROOT] = "root",
        [MODULE_CLASS_PORT] = "port",
        [MODULE_CLASS_COMPRESS] = "compress",
        [MODULE_CLASS_DATA] = "data",
        [MODULE_CLASS_SENDER] = "sender",
        [MODULE_CLASS_RECEIVER] = "receiver",
        [MODULE_CLASS_TX] = "transmit",
        [MODULE_CLASS_AUDIO] = "audio",
        [MODULE_CLASS_CONTROL] = "control",
        [MODULE_CLASS_CAPTURE] = "capture",
        [MODULE_CLASS_FILTER] = "filter",
        [MODULE_CLASS_DISPLAY] = "display",
        [MODULE_CLASS_DECODER] = "decoder",
        [MODULE_CLASS_EXPORTER] = "exporter",
        [MODULE_CLASS_KEYCONTROL] = "keycontrol",
};

const char *module_class_name(enum module_class cls)
{
        if((unsigned int) cls < sizeof(module_class_name_pairs)/sizeof(const char *))
                return module_class_name_pairs[cls];
        else
                return NULL;
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
        assert(node);
        while(node->parent) {
                node = node->parent;
        }
        assert(node->cls == MODULE_CLASS_ROOT);

        return node;
}

struct module *get_parent_module(struct module *node)
{
        return node->parent;
}

/**
 * id_num and id_name distinguish in case that the node has more child nodes with
 * the same node_name (the same class). If id_name == NULL, id_num is used.
 */
static struct module *find_child(struct module *node, const char *node_name, int id_num,
                const char *id_name)
{
        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = (struct module *) simple_linked_list_it_next(&it);
                const char *child_name = module_class_name(child->cls);
                assert(child_name != NULL);
                if(strcasecmp(child_name, node_name) == 0) {
                        if (id_name != NULL) {
                                if (child->name && strcmp(child->name, id_name) == 0) {
                                        simple_linked_list_it_destroy(it);
                                        return child;
                                }
                        } else if (id_num-- == 0) {
                                simple_linked_list_it_destroy(it);
                                return child;
                        }
                }
        }
        return NULL;
}

/**
 * Parses path element, which might be either in form a single-word item (eg "display")
 * or an array, either indexed by a number, in which case non-zero index will be returned
 * or a module name (module::name member) which would be returned in name return value.
 * Both indexing options are mutually exclusive.
 */
static void get_receiver_index(char *node_str, int *index, char **name) {
        if (strchr(node_str, '[') && (strchr(node_str, ']') > strchr(node_str, '['))) {
                char *item = strchr(node_str, '[') + 1;
                *strchr(node_str, '[') = '\0';
                *strchr(item, ']') = '\0';
                if (isdigit(item[0])) {
                        *index = atoi(item);
                } else {
                        *name = item;
                }
        }
}

struct module *get_module(struct module *root, const char *const_path)
{
        assert(root != NULL);
        assert(const_path != NULL);

        struct module *receiver = root;
        char *path, *tmp;
        char *item, *save_ptr;

        module_mutex_lock(&root->lock);

        tmp = path = strdup(const_path);
        assert(path != NULL);
        while ((item = strtok_r(path, ".", &save_ptr))) {
                struct module *old_receiver = receiver;

                receiver = get_matching_child(receiver, item);

                if (!receiver) {
                        module_mutex_unlock(&old_receiver->lock);
                        free(tmp);
                        return NULL;
                }
                module_mutex_lock(&receiver->lock);
                module_mutex_unlock(&old_receiver->lock);

                path = NULL;

        }
        free(tmp);

        module_mutex_unlock(&receiver->lock);

        return receiver;
}

struct module *get_matching_child(struct module *node, const char *const_path)
{
        struct module *receiver = node;
        char *path, *tmp;
        char *item, *save_ptr = NULL;

        assert(node != NULL);

        tmp = path = strdup(const_path);
        assert(path != NULL);
        if ((item = strtok_r(path, ".", &save_ptr))) {
                int id_num = 0;
                char *id_name = NULL;
                get_receiver_index(item, &id_num, &id_name);
                receiver = find_child(receiver, item, id_num, id_name);
                if (!receiver) {
                        free(tmp);
                        return NULL;
                }
                free(tmp);
                return receiver;
        }

        free(tmp);

        return NULL;
}

void dump_tree(struct module *node, int indent) {
        for(int i = 0; i < indent; ++i) putchar(' ');

        printf("%s\n", module_class_name(node->cls));

        module_mutex_lock(&node->lock);
        for(void *it = simple_linked_list_it_init(node->childs); it != NULL; ) {
                struct module *child = simple_linked_list_it_next(&it);
                dump_tree(child, indent + 2);
        }
}

void
dump_parents(struct module *node)
{
        bool first = true;
        while (node != NULL) {
                if (!first) {
                        printf("<-");
                }
                printf("%s", module_class_name(node->cls));
                first = false;
                node  = get_parent_module(node);
        }
        printf("\n");
}

/**
 * @returns module_class_name() if we are first child of our class
 * @returns cls_name[idx] otherwise (idx > 0)
 */
static const char *get_module_identifier(struct module *mod)
{
        const char *cls_name = module_class_name(mod->cls);
        struct module *parent = get_parent_module(mod);
        if (parent == NULL) {
                return cls_name;
        }

        module_mutex_lock(&parent->lock);

        int our_index = 0;
        for (void *it = simple_linked_list_it_init(parent->childs);
             it != NULL;) {
                struct module *child = simple_linked_list_it_next(&it);
                if (child->cls != mod->cls) {
                        continue;
                }
                if (child == mod) { // found our node
                        break;
                }
                our_index += 1;
        }
        if (our_index == 0) {
                module_mutex_unlock(&parent->lock);
                return cls_name;
        }
        // append our index if >0
        _Thread_local static char name[128];
        name[sizeof name - 1] = '\0';
        int ret = snprintf(name, sizeof name, "%s[%d]", cls_name, our_index);
        assert((unsigned)ret < sizeof name);

        module_mutex_unlock(&parent->lock);

        return name;
}

/**
 * Gets textual representation of path from root to module.
 *
 * Currently only simple paths are created (class names only)
 */
bool module_get_path_str(struct module *mod, char *buf, size_t buflen) {
        assert(buflen > 0);
        buf[0] = '\0';

        while (mod) {
                const char *cur_name = get_module_identifier(mod);
                if (sizeof(buf) + 1 + sizeof(cur_name) >= buflen) {
                        return false;
                }
                // move content of buf strlen(cur_name) right
                if (strlen(buf) > 0) {
                        memmove(buf + strlen(cur_name) + 1, buf, strlen(buf) + 1);
                        buf[strlen(cur_name)] = '.'; // and separate with '.'
                } else { // rightmost (first written) element
                        buf[strlen(cur_name)] = '\0';
                }
                memcpy(buf, cur_name, strlen(cur_name));
                mod = mod->parent;
        }

        return true;
}


/**
 * @file   module.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup module
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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
 * @file module.h
 *
 * @defgroup module Module infrastructure
 * @{
 *
 * ### Module registration
 * Example:
 * ```
 * struct state {
 *      struct module mod;
 *      ...
 * } s;
 * module_init_default(&s->mod);
 * s->mod.cls = MODULE_CLASS_<NAME>; // allways needed
 * s->mod.priv_data = s;             // optional
 * s->mod.deleter = deleter;         // only used for generic modules, see a note bellow
 * module_register(&s->mod, s->parent);
 * ```
 *
 * @note
 * Struct module should be first inside the struct in order to be able to call
 * @ref CAST_MODULE to it (however, it is currently strictly required only for
 * @ref tx and @ref video_compress_info modules).
 */

#ifndef MODULE_H_
#define MODULE_H_

#include "messaging.h"
#include "utils/macros.h"

#include <pthread.h>
#ifndef __cplusplus
#include <stdalign.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MODULE_MAGIC to_fourcc('M', 'O', 'D', ' ')

enum module_class {
        MODULE_CLASS_NONE = 0,
        MODULE_CLASS_ROOT,
        MODULE_CLASS_PORT,
        MODULE_CLASS_COMPRESS,
        MODULE_CLASS_DATA,
        MODULE_CLASS_SENDER,
        MODULE_CLASS_RECEIVER,
        MODULE_CLASS_TX,
        MODULE_CLASS_AUDIO,
        MODULE_CLASS_CONTROL,
        MODULE_CLASS_CAPTURE,
        MODULE_CLASS_FILTER,
        MODULE_CLASS_DISPLAY,
        MODULE_CLASS_DECODER,
        MODULE_CLASS_EXPORTER,
        MODULE_CLASS_KEYCONTROL,
};

struct module;
struct simple_linked_list;

typedef void (*module_deleter_t)(struct module *);
typedef void (*notify_t)(struct module *);

/**
 * @struct module
 * Only members cls, deleter, priv_data and msg_queue may be directly touched
 * by user. The others should be considered private.
 */
struct module {
        alignas(8) uint32_t magic;
        uint32_t priv_magic; ///< optional magic number that can a module set
        pthread_mutex_t lock;
        enum module_class cls;
        struct module *parent;
        struct simple_linked_list *childs;
        module_deleter_t deleter;
        notify_t new_message; ///< if set, notifies module that new message is in queue, receiver lock is hold during the call

        pthread_mutex_t msg_queue_lock; // protects msg_queue
        struct simple_linked_list *msg_queue;

        struct simple_linked_list *msg_queue_childs; ///< messages for childern that were not delivered

        void *priv_data;
        //uint32_t id;

        char *name; ///< optional name of the module. May be used for indexing. Will be freed by
                    ///< module_done(). Must not start with a digit.

#ifdef __cplusplus
        module() = default;
        // don't be tempted to copy/move module because parent holds pointer to struct module
        // which would be dangling thereafter
        module(module const &) = delete;
        module(module &&) = delete;
        module &operator=(module const &) = delete;
        module &operator=(module &&) = delete;
#endif
};

void module_init_default(struct module *module_data);
void module_register(struct module *module_data, struct module *parent);
void module_done(struct module *module_data);
const char *module_class_name(enum module_class cls);
void append_message_path(char *buf, int buflen, enum module_class modules[]);
bool module_get_path_str(struct module *mod, char *buf, size_t buflen);

#ifdef __cplusplus
class module_raii{
public:
        module_raii(enum module_class type, module *parent, void *priv){
                module_init_default(&mod);
                mod.priv_data = priv;
                mod.cls = type;
                module_register(&mod, parent);
        }

        ~module_raii(){
                module_done(&mod);
        }

        module *get(){ return &mod; }
private:
        module mod;
};
#endif

/**
 * @retval NULL if not found
 * @retval non-NULL pointer to the module
 */
struct module *get_module(struct module *root, const char *path);

/**
 * IMPORTANT: module given as parameter should be locked within the calling thread.
 *
 * @retval NULL if not found
 * @retval non-NULL pointer to the module
 */
struct module *get_matching_child(struct module *node, const char *path);

/**
 * Returns pointer to root module.
 *
 * @retval root module
 */
struct module *get_root_module(struct module *node);

struct module *get_parent_module(struct module *node);

void dump_tree(struct module *root, int indent);
void dump_parents(struct module *node);

#define CAST_MODULE(x) ((struct module *) x)

#ifdef __cplusplus
}
#endif

#endif
/**
 * @} // group module
 */

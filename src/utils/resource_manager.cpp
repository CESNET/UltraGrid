/**
 * @file   utils/resource_manager.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Seldom used, currently only in x11_common.c. But the managed resources
 * cannot be easily replaced by static variables there because there will be
 * multiple instances of static vars in case of modular UltraGrid build (every
 * library compiled with that file defining the var).
 */
/*
 * Copyright (c) 2013 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "resource_manager.h"

#include "utils/lock_guard.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#define TYPE_LOCK 0
#define TYPE_SHM 1
#define TYPE_SINGLETON 2
typedef int type_t;

using std::logic_error;
using std::map;
using std::pair;
using std::string;
using ultragrid::pthread_mutex_guard;

class options_t {
        public:
                virtual ~options_t() {}
};

class no_opts : public options_t {
};

class shm_opts : public options_t {
        public:
                shm_opts(int size) : m_size(size) {}
        private:
                int m_size;
                friend class shm;
};

class singleton_opts : public options_t {
        public:
                singleton_opts(singleton_initializer_t init,
                                void *data, singleton_deleter_t done) :
                        m_init(init), m_init_data(data), m_done(done) {}
        private:
                singleton_initializer_t m_init;
                void *m_init_data;
                singleton_deleter_t m_done;
                friend class singleton;
};

class resource {
        public:
                virtual ~resource() {}
                static resource *create(type_t type, options_t const & options);
                static string get_suffix(type_t type);
};

class lock : public resource {
        public:
                lock() {
                        pthread_mutex_init(&m_lock, NULL);
                }

                ~lock() {
                        pthread_mutex_destroy(&m_lock);
                }

                pthread_mutex_t *get() {
                        return &m_lock;
                }
        private:
                pthread_mutex_t m_lock;
};

class shm : public resource {
        public:
                shm(shm_opts const &opts) {
                        m_data = calloc(1, opts.m_size);
                }

                ~shm() {
                        free(m_data);
                }

                void *get() {
                        return m_data;
                }

        private:
                void *m_data;
};

class singleton : public resource {
        public:
                singleton(singleton_opts const &opts) : m_opts(opts) {
                        m_data = opts.m_init(opts.m_init_data);
                }

                ~singleton() {
                        m_opts.m_done(m_data);
                }

                void *get() {
                        return m_data;
                }

        private:
                singleton_opts m_opts;
                void *m_data;
};

static void func_delete(pair<string, pair<resource *, int> > arg);
static void func_delete(pair<string, pair<resource *, int> > arg) {
        delete arg.second.first;
}

class resource_manager_t {
        public:
                typedef map<string, pair<resource *, int> > obj_map_t;
                
                resource_manager_t() {
                        pthread_mutex_init(&m_access_lock, NULL);
                        pthread_mutex_init(&m_excl_lock, NULL);
                }

                ~resource_manager_t() {
                        for_each(m_objs.begin(),
                                        m_objs.end(), func_delete);
                        pthread_mutex_destroy(&m_access_lock);
                        pthread_mutex_destroy(&m_excl_lock);
                }

                void lock() {
                        pthread_mutex_lock(&m_excl_lock);
                }

                void unlock() {
                        pthread_mutex_unlock(&m_excl_lock);
                }

                resource *acquire(string name, type_t type, options_t const & options) {
                        resource *ret;
                        pthread_mutex_guard lock(m_access_lock);
                        string item_name = name + "#" + resource::get_suffix(type);

                        obj_map_t::iterator it = m_objs.find(item_name);
                        if(it == m_objs.end()) {
                                // create
                                ret = resource::create(type, options);
                                m_objs[item_name] = pair<resource *, int>(ret,
                                                1);
                        } else {
                                it->second.second += 1;
                                ret = it->second.first;
                        }
                        return ret;
                }

                void release(string name, type_t type) {
                        pthread_mutex_guard lock(m_access_lock);
                        string item_name = name + "#" + resource::get_suffix(type);

                        obj_map_t::iterator it = m_objs.find(item_name);
                        if(it == m_objs.end()) {
                                // create
                                throw logic_error("No such object.");
                        } else {
                                it->second.second -= 1;
                                if(it->second.second == 0) { // ref count == 0
                                        delete it->second.first;
                                        m_objs.erase(it);
                                }
                        }
                }

        private:
                pthread_mutex_t m_access_lock;
                pthread_mutex_t m_excl_lock;
                obj_map_t m_objs;

};

static resource_manager_t resource_manager;

resource *resource::create(type_t type, options_t const & options)
{
        if(type == TYPE_LOCK) {
                return new lock;
        } else if(type == TYPE_SHM) {
                return new shm(dynamic_cast<const shm_opts &>(options));
        } else if(type == TYPE_SINGLETON) {
                return new singleton(dynamic_cast<const singleton_opts &>(options));
        } else {
                throw logic_error("Wrong typeid");
        }
}

string resource::get_suffix(type_t type)
{
        if(type == TYPE_LOCK) {
                return string("mutex");
        } else if(type == TYPE_SHM) {
                return string("SHM");
        } else if(type == TYPE_SINGLETON) {
                return string("singleton");
        } else {
                throw logic_error("Wrong typeid");
        }
}

pthread_mutex_t *rm_acquire_shared_lock(const char *name)
{
        lock *l = dynamic_cast<lock *>(resource_manager.acquire(
                                string(name), TYPE_LOCK, no_opts()));
        if(l) {
                return l->get();
        } else {
                return NULL;
        }
}

void rm_release_shared_lock(const char *name)
{
        resource_manager.release(string(name), TYPE_LOCK);
}

void rm_lock()
{
        resource_manager.lock();
}

void rm_unlock()
{
        resource_manager.unlock();
}

void *rm_get_shm(const char *name, int size)
{
        shm *s = dynamic_cast<shm *>(resource_manager.acquire(
                                string(name), TYPE_SHM, shm_opts(size)));
        if(s) {
                return s->get();
        } else {
                return NULL;
        }
}

void *rm_singleton(const char *name, singleton_initializer_t initializer, void *data,
                singleton_deleter_t done)
{
        singleton *s = dynamic_cast<singleton *>(resource_manager.acquire(
                                string(name), TYPE_SINGLETON, singleton_opts(initializer, data, done)));
        if(s) {
                return s->get();
        } else {
                return NULL;
        }
}


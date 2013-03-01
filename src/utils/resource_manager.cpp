/*
 * FILE:    utils/resource_manager.h
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
#endif // HAVE_CONFIG_H

#include "resource_manager.h"

#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#define TYPE_LOCK 0
typedef int type_t;

using namespace std;

class resource {
        public:
                virtual ~resource() {}
                static resource *create(type_t type);
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

class lock_holder {
        public:
                lock_holder(pthread_mutex_t lock) :
                        m_lock(lock)
                {
                        pthread_mutex_lock(&m_lock);
                }

                ~lock_holder()
                {
                        pthread_mutex_unlock(&m_lock);
                }
        private:
                pthread_mutex_t &m_lock;

};

class resource_manager_t {
        public:
                typedef map<string, pair<resource *, int> > obj_map_t;
                
                resource_manager_t() {
                        pthread_mutex_init(&m_lock, NULL);
                }

                ~resource_manager_t() {
                        pthread_mutex_destroy(&m_lock);
                }

                resource *acquire(string name, type_t type) {
                        resource *ret;
                        lock_holder lock(m_lock);
                        string item_name = name + "#" + resource::get_suffix(type);

                        obj_map_t::iterator it = m_objs.find(item_name);
                        if(it == m_objs.end()) {
                                // create
                                ret = resource::create(type);
                                m_objs[item_name] = pair<resource *, int>(ret,
                                                1);
                        } else {
                                it->second.second += 1;
                                ret = it->second.first;
                        }
                        return ret;
                }

                void release(string name, type_t type) {
                        lock_holder lock(m_lock);
                        string item_name = name + "#" + resource::get_suffix(type);

                        obj_map_t::iterator it = m_objs.find(item_name);
                        if(it == m_objs.end()) {
                                // create
                                throw logic_error("Not such object.");
                        } else {
                                it->second.second -= 1;
                                if(it->second.second == 0) { // ref count == 0
                                        delete it->second.first;
                                        m_objs.erase(it);
                                }
                        }
                }

        private:
                pthread_mutex_t m_lock;
                obj_map_t m_objs;

};

static resource_manager_t resource_manager;

resource *resource::create(type_t type)
{
        if(type == TYPE_LOCK) {
                return new lock;
        } else {
                throw logic_error("Wrong typeid");
        }
}

string resource::get_suffix(type_t type)
{
        if(type == TYPE_LOCK) {
                return string("mutex");
        } else {
                throw logic_error("Wrong typeid");
        }
}
 
pthread_mutex_t *rm_acquire_shared_lock(const char *name)
{
        lock *l = dynamic_cast<lock *>(resource_manager.acquire(
                                string(name), TYPE_LOCK));
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


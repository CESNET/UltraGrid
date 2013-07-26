/*
 * FILE:    utils/worker.cpp
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

#include "utils/worker.h"

#include <algorithm>
#include <queue>
#include <set>

using namespace std;

struct wp_worker;

struct worker_state_observer {
        virtual ~worker_state_observer() {}
        virtual void notify(wp_worker *) = 0;
};

/**
 * @brief Holds data to be passed to worker.
 */
struct wp_task_data {
        wp_task_data(task_t task, void *data, wp_worker *w) : m_task(task), m_data(data),
                m_result(0), m_returned(false), m_w(w) {}
        task_t m_task;
        void *m_data;
        void *m_result;
        bool m_returned;
        struct wp_worker *m_w;
};

/**
 * @brief This class represents a worker that is called from within the pool
 */
struct wp_worker {
        wp_worker(worker_state_observer &observer) :
                m_state_observer(observer)
        {
                pthread_mutex_init(&m_lock, NULL);
                pthread_cond_init(&m_task_ready_cv, NULL);
                pthread_cond_init(&m_task_completed_cv, NULL);

                pthread_create(&m_thread_id, NULL, wp_worker::enter_loop, this);
        }
        ~wp_worker() {
                wp_task_data *poisoned = new wp_task_data(NULL, NULL, this);
                this->push(poisoned);

                pthread_join(m_thread_id, NULL);
                pthread_mutex_destroy(&m_lock);
                pthread_cond_destroy(&m_task_ready_cv);
                pthread_cond_destroy(&m_task_completed_cv);
        }
        static void      *enter_loop(void *args);
        void              run();

        void              push(wp_task_data *);
        void             *pop(wp_task_data *);

        queue<wp_task_data*> m_data;
        pthread_mutex_t   m_lock;
        pthread_cond_t    m_task_ready_cv;
        pthread_cond_t    m_task_completed_cv;
        pthread_t         m_thread_id;

        worker_state_observer &m_state_observer;
};

void *wp_worker::enter_loop(void *args) {
        wp_worker *instance = (wp_worker *) args;
        instance->run();

        return NULL;
}

void wp_worker::run() {
        while(1) {
                struct wp_task_data *data;
                pthread_mutex_lock(&m_lock);
                while(m_data.empty()) {
                        pthread_cond_wait(&m_task_ready_cv, &m_lock);
                }
                data = m_data.front();
                m_data.pop();
                pthread_mutex_unlock(&m_lock);

                // poisoned pill
                if(data->m_task == NULL) {
                        delete data;
                        return;
                }

                void *res = data->m_task(data->m_data);

                pthread_mutex_lock(&m_lock);
                data->m_result = res;
                data->m_returned = true;
                pthread_cond_signal(&m_task_completed_cv);
                m_state_observer.notify(this);
                pthread_mutex_unlock(&m_lock);
        }
}

void wp_worker::push(wp_task_data *data) {
        pthread_mutex_lock(&m_lock);
        assert(m_data.size() == 0);
        m_data.push(data);
        pthread_cond_signal(&m_task_ready_cv);
        pthread_mutex_unlock(&m_lock);
}

void *wp_worker::pop(wp_task_data *d) {
        void *res = NULL;

        pthread_mutex_lock(&m_lock);
        while(!d->m_returned) {
                pthread_cond_wait(&m_task_completed_cv, &m_lock);
        }
        res = d->m_result;
        delete d;
        pthread_mutex_unlock(&m_lock);

        return res;
}

static void func_delete(wp_worker *arg) {
        delete arg;
}

class worker_pool : public worker_state_observer
{
        public:
                worker_pool() {
                        pthread_mutex_init(&m_lock, NULL);
                }

                ~worker_pool() {
                        for_each(m_empty_workers.begin(),
                                        m_empty_workers.end(), func_delete);
                        for_each(m_occupied_workers.begin(),
                                        m_occupied_workers.end(), func_delete);
                        pthread_mutex_destroy(&m_lock);
                }

                void notify(wp_worker *w) {
                        pthread_mutex_lock(&m_lock);
                        m_occupied_workers.erase(w);
                        m_empty_workers.insert(w);
                        pthread_mutex_unlock(&m_lock);
                }

                task_result_handle_t run_async(task_t task, void *data);
                void *wait_task(task_result_handle_t handle);

        private:
                set<wp_worker*>    m_empty_workers;
                set<wp_worker*>    m_occupied_workers;
                pthread_mutex_t m_lock;
};

task_result_handle_t worker_pool::run_async(task_t task, void *data)
{
        wp_worker *w;
        pthread_mutex_lock(&m_lock);
        if(m_empty_workers.empty()) {
                m_empty_workers.insert(new wp_worker(*this));
        }
        set<wp_worker*>::iterator it = m_empty_workers.begin();
        assert(it != m_empty_workers.end());
        w = *it;
        /// @todo: really weird - it seems like that 'it' instead of 'w' caused some problems
        m_empty_workers.erase(w);
        m_occupied_workers.insert(w);

        wp_task_data *d = new wp_task_data(task, data, w);
        w->push(d);
        pthread_mutex_unlock(&m_lock);

        return d;
}

void *worker_pool::wait_task(task_result_handle_t handle)
{
        wp_task_data *d = (wp_task_data *) handle;
        wp_worker *w = d->m_w;
        return w->pop(d);
}

static class worker_pool instance;

task_result_handle_t task_run_async(task_t task, void *data)
{
        return instance.run_async(task, data);
}

void *wait_task(task_result_handle_t handle)
{
        return instance.wait_task(handle);
}


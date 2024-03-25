/**
 * @file   utils/worker.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2024 CESNET, z. s. p. o.
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

#include <algorithm>
#include <cassert>
#include <pthread.h>
#include <queue>
#include <set>
#include <vector>

#include "utils/macros.h" // for MAX_CPU_CORES
#include "utils/misc.h"   // get_cpu_core_count
#include "utils/thread.h"
#include "utils/worker.h"

using std::min;
using std::queue;
using std::set;
using std::vector;

struct wp_worker;

struct worker_state_observer {
        virtual ~worker_state_observer() {}
        virtual void notify(wp_worker *) = 0;
};

/**
 * @brief Holds data to be passed to worker.
 */
struct wp_task_data {
        wp_task_data(runnable_t task, void *data, wp_worker *w, bool detached) : m_task(task), m_data(data),
                m_result(0), m_returned(false), m_w(w), m_detached(detached) {}
        runnable_t m_task;
        void *m_data;
        void *m_result;
        bool m_returned;
        struct wp_worker *m_w;
        bool m_detached;
};

/**
 * @brief This class represents a worker that is called from within the pool
 */
struct wp_worker {
        wp_worker(worker_state_observer &observer) :
                m_state_observer(observer)
        {
                int ret;
                ret = pthread_mutex_init(&m_lock, NULL);
                assert(ret == 0);
                ret = pthread_cond_init(&m_task_ready_cv, NULL);
                assert(ret == 0);
                ret = pthread_cond_init(&m_task_completed_cv, NULL);
                assert(ret == 0);

                ret = pthread_create(&m_thread_id, NULL, wp_worker::enter_loop, this);
                assert(ret == 0);
        }
        ~wp_worker() {
                wp_task_data *poisoned = new wp_task_data(NULL, NULL, this, false);
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
        set_thread_name("worker");
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
                if (data->m_detached) {
                        delete data;
                }
                pthread_mutex_unlock(&m_lock);
        }
}

void wp_worker::push(wp_task_data *data) {
        pthread_mutex_lock(&m_lock);
        assert(m_data.size() == 0);
        m_data.push(data);
        pthread_mutex_unlock(&m_lock);
        pthread_cond_signal(&m_task_ready_cv);
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
                        pthread_cond_init(&m_worker_finished, NULL);
                }

                ~worker_pool() {
                        pthread_mutex_lock(&m_lock);
                        while (m_occupied_workers.size() > 0) {
                                pthread_cond_wait(&m_worker_finished, &m_lock);
                        }
                        pthread_mutex_unlock(&m_lock);

                        for_each(m_empty_workers.begin(),
                                        m_empty_workers.end(), func_delete);
                        pthread_cond_destroy(&m_worker_finished);
                        pthread_mutex_destroy(&m_lock);
                }

                void notify(wp_worker *w) {
                        pthread_mutex_lock(&m_lock);
                        m_occupied_workers.erase(w);
                        m_empty_workers.insert(w);
                        pthread_mutex_unlock(&m_lock);
                        pthread_cond_signal(&m_worker_finished);
                }

                task_result_handle_t run_async(runnable_t task, void *data, bool detached);
                void *wait_task(task_result_handle_t handle);

        private:
                set<wp_worker*>    m_empty_workers;
                set<wp_worker*>    m_occupied_workers;
                pthread_mutex_t m_lock;
                pthread_cond_t     m_worker_finished;
};

task_result_handle_t worker_pool::run_async(runnable_t task, void *data, bool detached)
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

        wp_task_data *d = new wp_task_data(task, data, w, detached);
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

/**
 * @brief Runs task asynchronously.
 *
 * @param   task callback to be run
 * @param   data additional data to be passed to the callback
 * @returns      handle to the task
 *
 * @note
 * If you use this call wait_task() must be run.
 */
task_result_handle_t task_run_async(runnable_t task, void *data)
{
        return instance.run_async(task, data, false);
}

/**
 * @brief Runs task asynchronously in a detached state
 *
 * Detached task should own its resources. Moreover, it must not use any static variables/objects.
 *
 * @param   task callback to be run
 * @param   data additional data to be passed to the callback
 */
void task_run_async_detached(runnable_t task, void *data)
{
        instance.run_async(task, data, true);
}

void *wait_task(task_result_handle_t handle)
{
        return instance.wait_task(handle);
}

/**
 * This combines task_run_async() + wait_task()
 *
 * @param task         task to be run
 * @param worker_count number of workers to be run
 * @param data         pointer to data array to be passed to task
 * @param data_size    size of element of data
 * @param res          (optional) pointer to result array, may be NULL
 */
void task_run_parallel(runnable_t task, int worker_count, void *data, size_t data_size, void **res)
{
        if (worker_count == 1) {
                task(data);
                return;
        }

        vector<task_result_handle_t> tasks(worker_count);
        for (int i = 0; i < worker_count; ++i) {
                tasks[i] = task_run_async(task, (void *)((char *) data + i * data_size));
        }
        for (int i = 0; i < worker_count; ++i) {
                if (res != nullptr) {
                        res[i] = wait_task(tasks[i]);
                } else {
                        wait_task(tasks[i]);
                }
        }
}

struct respawn_parallel_data {
        respawn_parallel_callback_t c;
        void *in;
        void *out;
        size_t data_len;
        void *udata;
};
static void *respawn_parallel_task(void *arg) {
        auto data = (struct respawn_parallel_data *) arg;
        data->c(data->in, data->out, data->data_len, data->udata);
        return NULL;
}
/**
 * Automatically respawns threads to convert in to out
 *
 * Botn input and output elements must currently have the same size (can be changed in future).
 * Option semantics is similar to qsort().
 */
void respawn_parallel(void *in, void *out, size_t nmemb, size_t size, respawn_parallel_callback_t c, void *udata)
{
        const int threads = min<int>(get_cpu_core_count(), MAX_CPU_CORES);
        struct respawn_parallel_data data[MAX_CPU_CORES];

        for (int i = 0; i < threads; ++i) {
                data[i].c = c;
                data[i].in = (char *) in + i * (nmemb / threads) * size;
                data[i].out = (char *) out + i * (nmemb / threads) * size;
                data[i].data_len = (nmemb / threads) * size;
                data[i].udata = udata;
                if (i == threads - 1) {
                        data[i].data_len = size * (nmemb - (threads - 1) * (nmemb / threads));
                }
        }

        task_run_parallel(respawn_parallel_task, threads, data, sizeof data[0], NULL);
}


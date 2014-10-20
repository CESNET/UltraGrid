/**
 * @file   utils/wait_obj.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#ifndef UTILS_WAIT_OBJ_H_
#define UTILS_WAIT_OBJ_H_

#ifdef __cplusplus
#include <condition_variable>
#include <mutex>

struct wait_obj {
        public:
                wait_obj() : m_val(false) {
                }
                void wait() {
                        std::unique_lock<std::mutex> lk(m_lock);
                        m_cv.wait(lk, [this] { return m_val; });
                }
                void reset() {
                        std::lock_guard<std::mutex> lk(m_lock);
                        m_val = false;
                }
                void notify() {
                        std::unique_lock<std::mutex> lk(m_lock);
                        m_val = true;
                        lk.unlock();
                        m_cv.notify_one();
                }
        private:
                std::mutex m_lock;
                std::condition_variable m_cv;
                bool m_val;
};
#endif // __cplusplus

struct wait_obj;

//
// C wrapper
//
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct wait_obj *wait_obj_init(void);
void wait_obj_reset(struct wait_obj *);
void wait_obj_wait(struct wait_obj *);
void wait_obj_notify(struct wait_obj *);
void wait_obj_done(struct wait_obj *);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // VIDEO_H_


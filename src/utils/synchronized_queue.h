/**
 * @file   utils/synchronized_queue.h
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

#ifndef SYNCHRONIZED_QUEUE_H_
#define SYNCHRONIZED_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

struct msg {
        virtual ~msg() {}
};

struct msg_quit : public msg {};

/**
 * @brief simple blocking synchronized queue
 *
 * Queue blocks if it size is higher than max_len on push. It also blocks on pop call
 * if there is no element in the queue.
 *
 * @tparam T type to be stored
 * @tparam max_len maximal length of the queue until it bloks (-1 means unlimited)
 */
template<typename T = struct msg *, int max_len = 1>
class synchronized_queue {
public:
        int size()
        {
                std::unique_lock<std::mutex> l(m_lock);
                return m_queue.size();
        }

        void push(T const & message)
        {
                std::unique_lock<std::mutex> l(m_lock);
                if (max_len != -1) {
                        m_queue_decremented.wait(l, [this]{return m_queue.size() < (unsigned int) max_len;});
                }
                m_queue.push(message);
                l.unlock();
                m_queue_incremented.notify_one();
        }

        void push(T && message)
        {
                std::unique_lock<std::mutex> l(m_lock);
                if (max_len != -1) {
                        m_queue_decremented.wait(l, [this]{return m_queue.size() < (unsigned int) max_len;});
                }
                m_queue.push(std::move(message));
                l.unlock();
                m_queue_incremented.notify_one();
        }

        T pop(bool nonblocking = false)
        {
                std::unique_lock<std::mutex> l(m_lock);
                if (m_queue.size() == 0 && nonblocking) {
                        return T();
                }

                m_queue_incremented.wait(l, [this]{return m_queue.size() > 0;});
                T ret = std::move(m_queue.front());
                m_queue.pop();

                l.unlock();
                m_queue_decremented.notify_one();
                return ret;
        }

        template<typename Rep, typename Period>
        bool timed_pop(T& result, std::chrono::duration<Rep, Period> const& timeout)
        {
                std::unique_lock<std::mutex> l(m_lock);
                if (!m_queue_incremented.wait_for(l, timeout, [this]{return m_queue.size() > 0;})) {
                        return false;
                }
                result = std::move(m_queue.front());
                m_queue.pop();
                l.unlock();
                m_queue_decremented.notify_one();
                return true;
        }

private:
        std::queue<T>           m_queue;
        std::mutex              m_lock;
        std::condition_variable m_queue_decremented;
        std::condition_variable m_queue_incremented;
};

#ifndef NO_EXTERN_MSGQ_MSG
extern template class synchronized_queue<msg *, -1>;
extern template class synchronized_queue<msg *, 1>;
#endif

#endif // SYNCHRONIZED_QUEUE_H_


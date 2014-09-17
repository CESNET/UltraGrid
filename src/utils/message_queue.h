/**
 * @file   utils/message_queue.h
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

#ifndef MESSAGE_QUEUE_H_
#define MESSAGE_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

struct msg {
        virtual ~msg() {}
};

struct msg_quit : public msg {};

template<typename T = struct msg *>
class message_queue {
public:

        message_queue(int max_len = -1) :
                m_max_len(max_len)
        {
        }

        virtual ~message_queue()
        {
        }

        int size()
        {
                std::unique_lock<std::mutex> l(m_lock);
                return m_queue.size();
        }

        void push(T const & message)
        {
                std::unique_lock<std::mutex> l(m_lock);
                if (m_max_len != -1) {
                        m_queue_decremented.wait(l, [this]{return m_queue.size() < (unsigned int) m_max_len;});
                }
                m_queue.push(message);
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
                T ret = m_queue.front();
                m_queue.pop();

                l.unlock();
                m_queue_decremented.notify_one();
                return ret;
        }


private:
        int                     m_max_len;
        std::queue<T>           m_queue;
        std::mutex              m_lock;
        std::condition_variable m_queue_decremented;
        std::condition_variable m_queue_incremented;
};

#ifndef NO_EXTERN_MSGQ_MSG
extern template class message_queue<msg *>;
#endif

#endif // MESSAGE_QUEUE_H_


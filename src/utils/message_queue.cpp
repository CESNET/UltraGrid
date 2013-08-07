/**
 * @file   utils/message_queue.cpp
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "utils/lock_guard.h"
#include "utils/message_queue.h"

message_queue::message_queue(int max_len) :
        m_max_len(max_len)
{
        pthread_mutex_init(&m_lock, NULL);
        pthread_cond_init(&m_queue_incremented, NULL);
        pthread_cond_init(&m_queue_decremented, NULL);
}

message_queue::~message_queue()
{
        pthread_mutex_destroy(&m_lock);
        pthread_cond_destroy(&m_queue_incremented);
        pthread_cond_destroy(&m_queue_decremented);
}

void message_queue::push(msg *message)
{
        lock_guard guard(m_lock);
        if (m_max_len != -1) {
                while (m_queue.size() >= (unsigned int) m_max_len) {
                        pthread_cond_wait(&m_queue_decremented, &m_lock); 
                }
        }
        m_queue.push(message);
        pthread_cond_signal(&m_queue_incremented); 
}

msg *message_queue::pop()
{
        lock_guard guard(m_lock);
        while (m_queue.size() == 0) {
                pthread_cond_wait(&m_queue_incremented, &m_lock); 
        }
        msg *ret = m_queue.front();
        m_queue.pop();

        pthread_cond_signal(&m_queue_decremented); 
        return ret;
}


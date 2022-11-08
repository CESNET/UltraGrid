/**
 * @file   video_display/concurrent_queue.hpp
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2021-2022 CESNET, z. s. p. o.
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

#pragma once

#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <queue>


namespace vulkan_display_detail{

constexpr size_t unlimited_size = SIZE_MAX;

template<typename T, size_t max_size = unlimited_size>
class ConcurrentQueue{
        std::queue<T> queue;
        mutable std::mutex mutex{};
        std::condition_variable queue_decremented_cv{};
        std::condition_variable queue_incremented_cv{};

        void push_and_unlock(std::unique_lock<std::mutex>& lock, T&& item){
                queue.push(std::move(item));
                lock.unlock();
                queue_incremented_cv.notify_one();
        }

        T pop(bool nonblocking = false)
        {
                std::unique_lock<std::mutex> lock{mutex};
                if (queue.size() == 0 && nonblocking) {
                        return T();
                }

                queue_incremented_cv.wait(lock, [this]{return queue.size() > 0;});
                T result = std::move(queue.front());
                queue.pop();

                lock.unlock();
                queue_decremented_cv.notify_one();
                return result;
        }
public:
        T try_pop(){
                return pop(true);
        }

        T wait_pop(){
                return pop(false);
        }

        template<typename Rep, typename Period>
        T timed_pop(std::chrono::duration<Rep, Period> timeout){
                std::unique_lock<std::mutex> lock{mutex};

                queue_incremented_cv.wait_for(lock, timeout, [this]{return queue.size() > 0;});
                if (queue.empty()){
                        return T{};
                }
                T result = std::move(queue.front());
                queue.pop();

                lock.unlock();
                queue_decremented_cv.notify_one();
                return result;
        }

        T force_push(T item){
                std::unique_lock lock{mutex};
                T result = {};
                if (queue.size() >= max_size) {
                        result = std::move(queue.front());
                        queue.pop();
                }
                push_and_unlock(lock, std::move(item));
                return result;
        }

        bool try_push(T item){
                std::unique_lock lock{mutex};
                if (queue.size() >= max_size) {
                        return false;
                }
                push_and_unlock(lock, std::move(item));
                return true;
        }

        void wait_push(T item){
                std::unique_lock lock{mutex};
                if (max_size != unlimited_size) {
                        queue_decremented_cv.wait(lock, [this]{return queue.size() < max_size;});
                }
                push_and_unlock(lock, std::move(item));
        }
};

} //namespace vulkan_display_detail




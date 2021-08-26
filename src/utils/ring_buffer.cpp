/**
 * @file   utils/ring_buffer.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2019 CESNET, z. s. p. o.
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

#include "utils/ring_buffer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>

struct ring_buffer {
        char *data;
        int len;
        std::atomic<int> start, end;
};

struct ring_buffer *ring_buffer_init(int size) {
        struct ring_buffer *buf;
        
        buf = (struct ring_buffer *) malloc(sizeof(struct ring_buffer));
        buf->data = (char *) malloc(size);
        buf->len = size;
        buf->start = 0;
        buf->end = 0;
        return buf;
}

void ring_buffer_destroy(struct ring_buffer *ring) {
        if(ring) {
                free(ring->data);
                free(ring);
        }
}

int ring_buffer_read(struct ring_buffer * ring, char *out, int max_len) {
        /* end index is modified by the writer thread, use acquire order to ensure
         * that all writes by the writer thread made before the modification are
         * observable in this (reader) thread */
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_acquire);
        // start index is modified only by this (reader) thread, so relaxed is enough
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_relaxed);
        int read_len = end - start;
        
        if(read_len < 0)
                read_len += ring->len;
        if(read_len > max_len)
                read_len = max_len;
        
        if(start + read_len <= ring->len) {
                memcpy(out, ring->data + start, read_len);
        } else {
                int to_end = ring->len - start;
                memcpy(out, ring->data + start, to_end);
                memcpy(out + to_end, ring->data, read_len - to_end);
        }

        /* Use release order to ensure that all reads are completed (no reads
         * or writes in the current thread can be reordered after this store).
         */
        std::atomic_store_explicit(&ring->start, (start + read_len) % ring->len, std::memory_order_release);
        return read_len;
}

void ring_buffer_flush(struct ring_buffer * buf) {
        /* This should only be called while the buffer is not being read or
         * written. The only way to safely flush without locking is by reading
         * all available data from the reader thread.
         */
        buf->start = 0;
        buf->end = 0;
}

void ring_buffer_write(struct ring_buffer * ring, const char *in, int len) {
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);

        // end index is modified only by this (writer) thread, so relaxed is enough
        int end = std::atomic_load_explicit(&ring->start, std::memory_order_relaxed);


        if(len > ring->len) {
                fprintf(stderr, "Warning: too long write request for ring buffer (%d B)!!!\n", len);
                return;
        }
        /* detect overrun */
        {
                int read_len_old = end - start;
                int read_len_new = ((end + len) % ring->len) - start;
                
                if(read_len_old < 0)
                        read_len_old += ring->len;
                if(read_len_new < 0)
                        read_len_new += ring->len;
                if(read_len_new < read_len_old) {
                        fprintf(stderr, "Warning: ring buffer overflow!!!\n");
                }
        }
        
        int to_end = ring->len - end;
        if(len <= to_end) {
                memcpy(ring->data + end, in, len);
        } else {
                memcpy(ring->data + end, in, to_end);
                memcpy(ring->data, in + to_end, len - to_end);
        }

        /* Use release order to ensure that all writes to the buffer are
         * completed before advancing the end index (no reads or writes in the
         * current thread can be reordered after this store).
         */
        std::atomic_store_explicit(&ring->end, (end + len) % ring->len, std::memory_order_release);
}

int ring_get_size(struct ring_buffer * ring) {
        return ring->len;
}

int ring_get_current_size(struct ring_buffer * ring)
{
        /* This is called from both reader and writer thread.
         *
         * Writer case:
         * If the reader modifies start index under our feet, it doesn't
         * matter, because reader can only make the current size smaller. That
         * means the writer may calculate less free space, but never more than
         * really available.
         *
         * Reader case:
         * If the writer modifies end index under our feet, it doesn't matter,
         * because the writer can only make current size bigger. That means the
         * reader may calculate less size for reading, but the read data is
         * always valid.
         */
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_acquire);
        return (end - start + ring->len) % ring->len;
}

int ring_get_available_write_size(struct ring_buffer * ring){
        /* Ring buffer needs at least one free byte, otherwise start == end
         * and the ring would appear empty */
        return ring_get_size(ring) - ring_get_current_size(ring) - 1;
}

struct audio_buffer_api ring_buffer_fns = {
        (void (*)(void *)) ring_buffer_destroy,
        (int (*)(void *, char *, int)) ring_buffer_read,
        (void (*)(void *, const char *, int)) ring_buffer_write
};


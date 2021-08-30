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

static int calculate_avail_read(int start, int end, int buf_len){
        return (end - start + buf_len) % buf_len;
}

static int calculate_avail_write(int start, int end, int buf_len){
        /* Ring buffer needs at least one free byte, otherwise start == end
         * and the ring would appear empty */
        return buf_len - calculate_avail_read(start, end, buf_len) - 1; 
}


static int ring_get_read_regions(struct ring_buffer *ring, int max_len,
                void **ptr1, int *size1,
                void **ptr2, int *size2)
{
        /* end index is modified by the writer thread, use acquire order to ensure
         * that all writes by the writer thread made before the modification are
         * observable in this (reader) thread */
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_acquire);
        // start index is modified only by this (reader) thread, so relaxed is enough
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_relaxed);

        int read_len = calculate_avail_read(start, end, ring->len);
        if(read_len > max_len)
                read_len = max_len;

        int to_end = ring->len - start;
        *ptr1 = ring->data + start;
        if(read_len <= to_end) {
                *size1 = read_len;
                *ptr2 = nullptr;
                *size2 = 0;
        } else {
                *size1 = to_end;
                *ptr2 = ring->data;
                *size2 = read_len - to_end;
        }

        return read_len;
}

static void ring_advance_read_idx(struct ring_buffer *ring, int amount) {
        // start index is modified only by this (reader) thread, so relaxed is enough
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_relaxed);

        /* Use release order to ensure that all reads are completed (no reads
         * or writes in the current thread can be reordered after this store).
         */
        std::atomic_store_explicit(&ring->start, (start + amount) % ring->len, std::memory_order_release);
}

int ring_buffer_read(struct ring_buffer * ring, char *out, int max_len) {
        void *ptr1;
        int size1;
        void *ptr2;
        int size2;

        int read_len = ring_get_read_regions(ring, max_len, &ptr1, &size1, &ptr2, &size2);
        
        memcpy(out, ptr1, size1);
        if(ptr2) {
                memcpy(out + size1, ptr2, size2);
        }

        ring_advance_read_idx(ring, read_len);
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

static bool ring_get_write_regions(struct ring_buffer *ring, int requested_len,
                void **ptr1, int *size1,
                void **ptr2, int *size2)
{
        *ptr1 = nullptr;
        *size1 = 0;
        *ptr2 = nullptr;
        *size2 = 0;

        int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);
        // end index is modified only by this (writer) thread, so relaxed is enough
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_relaxed);

        if(requested_len >= ring->len) {
                return false;
        }

        int to_end = ring->len - end;
        *ptr1 = ring->data + end;
        *size1 = requested_len < to_end ? requested_len : to_end;
        if(*size1 < requested_len){
                *ptr2 = ring->data;
                *size2 = requested_len - *size1;
        }

        return true;
}

static bool ring_advance_write_idx(struct ring_buffer *ring, int amount) {
        const int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);
        // end index is modified only by this (writer) thread, so relaxed is enough
        const int end = std::atomic_load_explicit(&ring->end, std::memory_order_relaxed);

        /* Use release order to ensure that all writes to the buffer are
         * completed before advancing the end index (no reads or writes in the
         * current thread can be reordered after this store).
         */
        std::atomic_store_explicit(&ring->end, (end + amount) % ring->len, std::memory_order_release);

        return amount > calculate_avail_write(start, end, ring->len);
}

void ring_buffer_write(struct ring_buffer * ring, const char *in, int len) {
        void *ptr1;
        int size1;
        void *ptr2;
        int size2;
        if(!ring_get_write_regions(ring, len, &ptr1, &size1, &ptr2, &size2)){
                fprintf(stderr, "Warning: too long write request for ring buffer (%d B)!!!\n", len);
                return;
        }

        memcpy(ptr1, in, size1);
        if(ptr2){
                memcpy(ptr2, in + size1, size2);
        }

        if(ring_advance_write_idx(ring, len)) {
                fprintf(stderr, "Warning: ring buffer overflow!!!\n");
        }
}

int ring_get_size(struct ring_buffer * ring) {
        return ring->len;
}

/* ring_get_current_size and ring_get_available_write_size can be called from
 * both reader and writer threads.
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
int ring_get_current_size(struct ring_buffer * ring) {
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_acquire);
        return calculate_avail_read(start, end, ring->len);
}

int ring_get_available_write_size(struct ring_buffer * ring) {
        int start = std::atomic_load_explicit(&ring->start, std::memory_order_acquire);
        int end = std::atomic_load_explicit(&ring->end, std::memory_order_acquire);
        return calculate_avail_write(start, end, ring->len);
}

struct audio_buffer_api ring_buffer_fns = {
        (void (*)(void *)) ring_buffer_destroy,
        (int (*)(void *, char *, int)) ring_buffer_read,
        (void (*)(void *, const char *, int)) ring_buffer_write
};


/**
 * @file   utils/ring_buffer.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
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
 
 /*
  * Provides abstraction for ring buffers.
  * Note that it doesn't offer advanced synchronization primitives and
  * therefore is mainly intended for one producer and one consumer.
  */
#ifndef __RING_BUFFER_H
#define __RING_BUFFER_H

#include "audio_buffer.h" // audio_buffer_api

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @warining ring_buffer is generally not thread safe. The exception is when
 * one thread reads and the other writes to the ring buffer (producer-consumer).
 */
struct ring_buffer;
typedef struct ring_buffer ring_buffer_t;

struct ring_buffer *ring_buffer_init(int size);
void ring_buffer_destroy(struct ring_buffer * ring);
/*
 * @param ring           ring buffer structure
 * @param out            allocated buffer to read to
 * @param max_len        maximal amount of data
 * @return               actual data length read (ranges between 0 and max_len)
 */
int ring_buffer_read(struct ring_buffer * ring, char *out, int max_len);
void ring_buffer_write(struct ring_buffer * ring, const char *in, int len);
int ring_get_size(struct ring_buffer * ring);
/**
 * Flushes all data from ring buffer. Not thread safe - needs external
 * synchronization to ensure that this is not called while the buffer is being
 * read or written.
 */
void ring_buffer_flush(struct ring_buffer *ring);
/**
 * Returns actual buffer usage
 */
int ring_get_current_size(struct ring_buffer * ring);

/**
 * Returns size available for writing
 */
int ring_get_available_write_size(struct ring_buffer * ring);


/**
 * Returns pointers to memory available for reading. After reading use 
 * ring_advance_read_idx() to mark the memory as free for writing.
 *
 * @param max_len      cap returned size to this value
 * @param ptr1         pointer to the first region
 * @param size1        size of available data in the first region
 * @param ptr2         pointer to the second region
 * @param size2        size of available data in the second region
 * @return             size1 + size2
 */
int ring_get_read_regions(struct ring_buffer *ring, int max_len,
                void **ptr1, int *size1,
                void **ptr2, int *size2);

/**
 * Marks memory as already read and free for writing. Use after reading using
 * ring_get_read_regions(), or simply for discarding unwanted unread data.
 *
 * @param amount      amount in bytes to discard
 */
void ring_advance_read_idx(struct ring_buffer *ring, int amount);

/**
 * Returns pointers to memory available for writing. After writing use 
 * ring_advance_write_idx() to mark the memory as available for reading.
 *
 * Does not check for overflow, returns regions of requested size even if
 * buffer is full.
 *
 * If the requested_len is longer than the whole buffer, returns 0.
 *
 * @param requested_len    amount you want to write
 * @param ptr1             pointer to the first region
 * @param size1            size of the first region
 * @param ptr2             pointer to the second region
 * @param size2            size of the second region
 * @return                 size1 + size2
 */
int ring_get_write_regions(struct ring_buffer *ring, int requested_len,
                void **ptr1, int *size1,
                void **ptr2, int *size2);

/**
 * Marks written memory as ready for reading. Use after writing using
 * ring_get_write_regions().
 *
 * @param amount      amount in bytes to discard
 * @return            true if an overflow occured
 */
bool ring_advance_write_idx(struct ring_buffer *ring, int amount);

extern struct audio_buffer_api ring_buffer_fns;

#ifdef __cplusplus
}
#endif

#endif /* __RING_BUFFER_H */

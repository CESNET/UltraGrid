/*
 * FILE:    utils/ring_buffer.h
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
 
 /*
  * Provides abstraction for ring buffers.
  * Note that it doesn't offer advanced synchronization primitives and
  * therefore is mainly intended for one producer and one consumer.
  */
#ifndef __RING_BUFFER_H

#define __RING_BUFFER_H

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
 * Flushes all data from ring buffer
 */
void ring_buffer_flush(struct ring_buffer *ring);
/**
 * Returns actual buffer usage
 */
int ring_get_current_size(struct ring_buffer * ring);

#ifdef __cplusplus
}
#endif

#endif /* __RING_BUFFER_H */

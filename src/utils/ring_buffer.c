/*
 * FILE:    utils/ring_buffer.c
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

#include "utils/ring_buffer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ring_buffer {
        char *data;
        int len;
        volatile int start, end;
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
        int end = ring->end; /* to avoid changes under our hand */
        int read_len = end - ring->start;
        
        if(read_len < 0)
                read_len += ring->len;
        if(read_len > max_len)
                read_len = max_len;
        
        if(ring->start + read_len <= ring->len) {
                memcpy(out, ring->data + ring->start, read_len);
        } else {
                int to_end = ring->len - ring->start;
                memcpy(out, ring->data + ring->start, to_end);
                memcpy(out + to_end, ring->data, read_len - to_end);
        }
        ring->start = (ring->start + read_len) % ring->len;
        return read_len;
}

void ring_buffer_flush(struct ring_buffer * buf) {
        buf->start = buf->end = 0;
}

void ring_buffer_write(struct ring_buffer * ring, const char *in, int len) {
        int to_end;

        if(len > ring->len) {
                fprintf(stderr, "Warning: too long write request for ring buffer (%d B)!!!\n", len);
                return;
        }
        /* detect overrun */
        {
                int start = ring->start;
                int read_len_old = ring->end - start;
                int read_len_new = ((ring->end + len) % ring->len) - start;
                
                if(read_len_old < 0)
                        read_len_old += ring->len;
                if(read_len_new < 0)
                        read_len_new += ring->len;
                if(read_len_new < read_len_old) {
                        fprintf(stderr, "Warning: ring buffer overflow!!!\n");
                }
        }
        
        to_end = ring->len - ring->end;
        if(len <= to_end) {
                memcpy(ring->data + ring->end, in, len);
        } else {
                memcpy(ring->data + ring->end, in, to_end);
                memcpy(ring->data, in + to_end, len - to_end);
        }
        ring->end = (ring->end + len) % ring->len;
}

int ring_get_size(struct ring_buffer * ring) {
        return ring->len;
}

int ring_get_current_size(struct ring_buffer * ring)
{
        return (ring->end - ring->start + ring->len) % ring->len;
}


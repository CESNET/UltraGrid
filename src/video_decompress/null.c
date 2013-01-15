/*
 * FILE:    video_decompress/dxt_glsl.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "video_decompress/null.h"
#include <stdlib.h>

struct state_decompress_null {
        uint32_t magic;
};

void * null_decompress_init(void)
{
        struct state_decompress_null *s;

        s = (struct state_decompress_null *) malloc(sizeof(struct state_decompress_null));
        s->magic = NULL_MAGIC;
        return s;
}

int null_decompress_reconfigure(void *state, struct video_desc desc,
                        int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_null *s = (struct state_decompress_null *) state;
        UNUSED(desc);
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        UNUSED(pitch);
        UNUSED(out_codec);

        assert(s->magic == NULL_MAGIC);
        return TRUE;
}

int null_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq)
{
        struct state_decompress_null *s = (struct state_decompress_null *) state;
        assert(s->magic == NULL_MAGIC);

        UNUSED(dst);
        UNUSED(buffer);
        UNUSED(src_len);
        UNUSED(frame_seq);

        return TRUE;
}

int null_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;

        UNUSED(s);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return FALSE;
}

void null_decompress_done(void *state)
{
        struct state_decompress_null *s = (struct state_decompress_null *) state;

        if(!s)
                return;
        assert(s->magic == NULL_MAGIC);
        free(s);
}


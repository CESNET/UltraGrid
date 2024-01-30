/**
 * @file   video_decompress/i420.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * implementation of I420->UYVY "decompression", because for the planar
 * format there is no line decoder and when the display iw unable to display
 * I420 directly, it is not possible to present it.
 *
 * As this is implementation is quite short, it can also hold as a decompress
 * module template.
 */
/*
 * Copyright (c) 2024 CESNET z.s.p.o.
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

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "lib_common.h"
#include "types.h"
#include "video_codec.h"
#include "video_decompress.h"

struct i420_decompress_state {
        struct video_desc desc;
};

static void *
i420_decompress_init(void)
{
        return malloc(sizeof(struct i420_decompress_state));
}

static int
i420_decompress_reconfigure(void *state, struct video_desc desc, int rshift,
                            int gshift, int bshift, int pitch,
                            codec_t out_codec)
{
        (void) rshift, (void) gshift, (void) bshift;
        assert(out_codec == UYVY);
        assert(pitch == (int) desc.width * 2); // implement other ir needed
        struct i420_decompress_state *s = state;
        s->desc                         = desc;
        return true;
}

static decompress_status
i420_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq,
                struct video_frame_callbacks *callbacks,
                struct pixfmt_desc           *internal_prop)
{
        (void) src_len, (void) frame_seq, (void) callbacks,
            (void) internal_prop;
        struct i420_decompress_state *s = state;
        i420_8_to_uyvy((int) s->desc.width, (int) s->desc.height, buffer, dst);
        return DECODER_GOT_FRAME;
}

static int
i420_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        (void) state;

        if (property == DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME) {
                assert(*len >= sizeof(int));
                *(int *) val = true;
                *len         = sizeof(int);
                return true;
        }
        return false;
}

static void
i420_decompress_done(void *state)
{
        free(state);
}

static int
i420_decompress_get_priority(codec_t compression, struct pixfmt_desc internal,
                             codec_t ugc)
{
        (void) internal;
        enum {
                PRIO_NA     = -1,
                PRIO_NORMAL = 500,
        };
        // probing (ugc==VC_NONE) is skipped (optional, not necessary)
        return compression == I420 && ugc == UYVY ? PRIO_NORMAL : PRIO_NA;
}

static const struct video_decompress_info i420_info = {
        i420_decompress_init, i420_decompress_reconfigure,
        i420_decompress,      i420_decompress_get_property,
        i420_decompress_done, i420_decompress_get_priority,
};

REGISTER_MODULE(i420, &i420_info, LIBRARY_CLASS_VIDEO_DECOMPRESS,
                VIDEO_DECOMPRESS_ABI_VERSION);

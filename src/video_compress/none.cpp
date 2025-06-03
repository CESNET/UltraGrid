/**
 * @file   video_compress/none.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2025 CESNET
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <stdlib.h>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video_codec.h"
#include "video_compress.h"
#include "video_frame.h"

namespace {

#define MAGIC 0x45bb3321

static void none_compress_done(void *state);

struct state_video_compress_none {
        uint32_t magic;
};

void* none_compress_init(module *parent, const char *)
{
        (void) parent;
        auto *s = (struct state_video_compress_none *) malloc(
            sizeof(struct state_video_compress_none));
        s->magic = MAGIC;

        return s;
}

std::shared_ptr<video_frame> none_compress(void *state, std::shared_ptr<video_frame> tx)
{
        auto *s = (struct state_video_compress_none *) state;

        assert(s->magic == MAGIC);

        return tx;
}

static void none_compress_done(void  *state)
{
        auto *s = (struct state_video_compress_none *) state;

        assert(s->magic == MAGIC);

        free(s);
}

const struct video_compress_info none_info = {
        none_compress_init,
        none_compress_done,
        none_compress,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};

REGISTER_MODULE(none, &none_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace


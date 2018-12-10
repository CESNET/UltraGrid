/**
 * @file   capture_filter/preview.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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

#include "capture_filter.h"

#include "debug.h"
#include "lib_common.h"

#include "video.h"
#include "video_codec.h"

#include "shared_mem_frame.hpp"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_preview_filter{
        Shared_mem shared_mem;
};


static int init(struct module *parent, const char *cfg, void **state){
        UNUSED(parent);
        UNUSED(cfg);

        struct state_preview_filter *s = new state_preview_filter();
        s->shared_mem.setKey("ultragrid_preview_capture");
        s->shared_mem.create();

        *state = s;

        return 0;
}

static void done(void *state){
        struct state_preview_filter *s = (state_preview_filter *) state;
        delete (state_preview_filter *) state;
}

static struct video_frame *filter(void *state, struct video_frame *in){
        struct state_preview_filter *s = (state_preview_filter *) state;

        s->shared_mem.put_frame(in);
        
        return in;
}


static const struct capture_filter_info capture_filter_preview = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(preview, &capture_filter_preview, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);


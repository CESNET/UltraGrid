/**
 * @file   vo_postprocess/capture_filter_wrapper.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Wrapper encapsulating caputre filter to video postprocessor.
 * In addition to capture filter API, the wrapped module must implement 
 * (void (*set_out_buf)(void *state, char *buffer)) function to provide
 * output buffer to decode to.
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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

#ifndef CAPTURE_FILTER_WRAPPER_H_4106D5BE_4B1D_4367_B506_67B358514C50
#define CAPTURE_FILTER_WRAPPER_H_4106D5BE_4B1D_4367_B506_67B358514C50

#include <assert.h>

#include "capture_filter.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct vo_pp_capture_filter_wrapper {
        void *state;           ///< capture filter state
        struct video_frame *f;
};

#define CF_WRAPPER_MERGE(a,b)  a##b

#define ADD_VO_PP_CAPTURE_FILTER_WRAPPER(name, init, filter, done, set_out_buf) \
static void *CF_WRAPPER_MERGE(vo_pp_init_, name)(const char *cfg) {\
        void *state;\
        if (init(NULL, cfg, &state) != 0) {\
                return NULL;\
        }\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) \
                        calloc(1, sizeof(struct vo_pp_capture_filter_wrapper));\
        s->state = state;\
        return s;\
}\
\
static bool CF_WRAPPER_MERGE(vo_pp_reconfigure_, name)(void *state, struct video_desc desc) {\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) state;\
        s->f = vf_alloc_desc_data(desc);\
        return true;\
}\
\
static bool CF_WRAPPER_MERGE(vo_pp_get_property_, name)(void *state, int property, void *val, size_t *len) {\
        UNUSED(state);\
        UNUSED(property);\
        UNUSED(val);\
        UNUSED(len);\
        return false;\
}\
static struct video_frame *CF_WRAPPER_MERGE(vo_pp_getf_, name)(void *state) {\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) state;\
        return s->f;\
}\
\
static bool CF_WRAPPER_MERGE(vo_pp_postprocess_, name)(void *state, struct video_frame *in, struct video_frame *out, int req_out_pitch) {\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) state;\
        assert(req_out_pitch == vc_get_linesize(out->tiles[0].width, out->color_spec));\
        set_out_buf(s->state, out->tiles[0].data);\
        struct video_frame *dst = filter(s->state, in);\
        VIDEO_FRAME_DISPOSE(dst);\
        UNUSED(req_out_pitch);\
        return true;\
}\
static void CF_WRAPPER_MERGE(vo_pp_get_out_desc_, name)(void *state, struct video_desc *out, int *in_tile_mode, int *out_frame_count) {\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) state;\
        struct video_frame *tmp_out = filter(s->state, s->f);\
        *out = video_desc_from_frame(tmp_out);\
        VIDEO_FRAME_DISPOSE(tmp_out);\
\
        *in_tile_mode = DISPLAY_PROPERTY_VIDEO_MERGED;\
        *out_frame_count = 1;\
\
        UNUSED(in_tile_mode);\
}\
static void CF_WRAPPER_MERGE(vo_pp_get_out_desc_, done)(void *state) {\
        struct vo_pp_capture_filter_wrapper *s = (struct vo_pp_capture_filter_wrapper *) state;\
        done(s->state);\
        vf_free(s->f);\
        free(s);\
}\
\
static const struct vo_postprocess_info CF_WRAPPER_MERGE(vo_pp_, name) = {\
        CF_WRAPPER_MERGE(vo_pp_init_, name),\
        CF_WRAPPER_MERGE(vo_pp_reconfigure_, name),\
        CF_WRAPPER_MERGE(vo_pp_getf_, name),\
        CF_WRAPPER_MERGE(vo_pp_get_out_desc_, name),\
        CF_WRAPPER_MERGE(vo_pp_get_property_, name),\
        CF_WRAPPER_MERGE(vo_pp_postprocess_, name),\
        CF_WRAPPER_MERGE(vo_pp_get_out_desc_, done)\
};\
REGISTER_MODULE(name, &CF_WRAPPER_MERGE(vo_pp_, name), LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);

#endif // defined CAPTURE_FILTER_WRAPPER_H_4106D5BE_4B1D_4367_B506_67B358514C50

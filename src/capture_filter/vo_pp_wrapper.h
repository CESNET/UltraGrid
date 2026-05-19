// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * analogy of ../vo_postprocess/capture_filter_wrapper.h
 *
 * @note
 * can be used only if the postprocess callback is willing
 * to postprocess any buffer (not just the one returned from
 * vo_pp .reconfigure callback - the wrapper is not using .getf)
 */

#ifndef VO_PP_WRAPPER_H_11C0EFD4_BB57_4E7B_A2F3_7846F4B11209
#define VO_PP_WRAPPER_H_11C0EFD4_BB57_4E7B_A2F3_7846F4B11209

#include <assert.h> // for assert
#include <stdint.h> // for uint32_t

#include "compat/c23.h"     // IWYU pragma: keep
#include "capture_filter.h" // for CAPTURE_FILTER_ABI_VERSION
#include "types.h"          // for video_desc
#include "utils/macros.h"   // for to_fourcc

#define VPPW_MAGIC to_fourcc('C', 'F', 'V', 'P')

struct capture_filter_vo_pp_wrapper {
        uint32_t          magic;
        struct video_desc saved_desc;
        struct video_desc out_desc;
        void             *state; ///< vo pp state
};

#define VO_PP_WRAPPER_MERGE(a, b) a##b

#define ADD_CAPTURE_FILTER_VO_PP_WRAPPER(name, pp_init, reconfigure,           \
                                         get_out_desc, postprocess, pp_done)   \
        static int VO_PP_WRAPPER_MERGE(cf_init_, name)(                        \
            struct module * parent, const char *cfg, void **out_state)         \
        {                                                                      \
                (void) parent;                                                 \
                void *state = pp_init(cfg);                                    \
                if (state == nullptr || state == INIT_NOERR) {                 \
                        return -1;                                             \
                }                                                              \
                struct capture_filter_vo_pp_wrapper *s = calloc(1, sizeof *s); \
                s->magic                               = VPPW_MAGIC;           \
                s->state                               = state;                \
                *out_state                             = s;                    \
                return 0;                                                      \
        }                                                                      \
                                                                               \
        static struct video_frame *VO_PP_WRAPPER_MERGE(cf_filter_, name)(      \
            void *state, struct video_frame *in)                               \
        {                                                                      \
                struct capture_filter_vo_pp_wrapper *s = state;                \
                if (in == nullptr) {                                           \
                        if (s->saved_desc.width == 0) { /* not configured */   \
                                return nullptr;                                \
                        }                                                      \
                } else {                                                       \
                        struct video_desc in_desc = video_desc_from_frame(in); \
                        if (!video_desc_eq(in_desc, s->saved_desc)) {          \
                                reconfigure(s->state, in_desc);                \
                                int display_mode_unused = 0;                   \
                                get_out_desc(s->state, &s->out_desc,           \
                                             &display_mode_unused);            \
                                s->saved_desc = in_desc;                       \
                        }                                                      \
                }                                                              \
                struct video_frame *f = vf_alloc_desc_data(s->out_desc);       \
                f->callbacks.dispose  = vf_free;                               \
                size_t out_pitch =                                             \
                    vc_get_linesize(f->tiles[0].width, f->color_spec);         \
                bool ret = postprocess(s->state, in, f, out_pitch);            \
                if (ret) {                                                     \
                        return f;                                              \
                }                                                              \
                vf_free(f);                                                    \
                return nullptr;                                                \
        }                                                                      \
        static void VO_PP_WRAPPER_MERGE(cf_done_, name)(void *state)           \
        {                                                                      \
                struct capture_filter_vo_pp_wrapper *s = state;                \
                assert(s->magic == VPPW_MAGIC);                                \
                pp_done(s->state);                                             \
                free(s);                                                       \
        }                                                                      \
                                                                               \
        static const struct capture_filter_info VO_PP_WRAPPER_MERGE(cf_,       \
                                                                    name) = {  \
                .init   = VO_PP_WRAPPER_MERGE(cf_init_, name),                 \
                .done   = VO_PP_WRAPPER_MERGE(cf_done_, name),                 \
                .filter = VO_PP_WRAPPER_MERGE(cf_filter_, name),               \
        };                                                                     \
        REGISTER_MODULE(name, &VO_PP_WRAPPER_MERGE(cf_, name),                 \
                        LIBRARY_CLASS_CAPTURE_FILTER,                          \
                        CAPTURE_FILTER_ABI_VERSION);

#endif // defined VO_PP_WRAPPER_H_11C0EFD4_BB57_4E7B_A2F3_7846F4B11209

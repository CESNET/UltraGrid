/**
 * @file   capture_filter.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#ifndef CAPTURE_FILTER_H_
#define CAPTURE_FILTER_H_

#define CAPTURE_FILTER_ABI_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

struct module;

/**
 * @brief Performs actual filtering
 * @note
 * Frame management note
 * When input frame is no longer used (eg. returned new output frame),
 * VIDEO_FRAME_DISPOSE(f) has to be called. Also, if you create new output
 * frame, you may use its .dispose and .dispose_udata member to manage
 * video_frame lifetime. If .dispose==nullptr, the callback is not called (in
 * this case its lifetime must be managed by capture_filter module; repeated
 * call of the function or capture_filter_info.done means that the previously
 * returned frame is no longer used by the caller).
 *
 * @param f  input frame - can be nullptr, in which case return either nullptr
 *           or a subsequent frame produced from previous non-null frame
 */
typedef struct video_frame *capture_filter_filter_fn(void               *state,
                                                     struct video_frame *f);

struct capture_filter_info {
        /// @brief Initializes capture filter
        /// @param      parent parent module
        /// @param      cfg    configuration string from cmdline, not-NULL
        /// @param[out] state  output state
        /// @retval     0      if initialized successfully
        /// @retval     <0     if error
        /// @retval     >0     no error but state was not returned, eg. showing help
        int (*init)(struct module *parent, const char *cfg, void **state);
        void (*done)(void *state);
        capture_filter_filter_fn *filter;
};

struct capture_filter;
struct module;
struct video_frame;

/**
 * @see display_init
 */
int capture_filter_init(struct module *parent, const char *cfg, struct capture_filter **state);
void capture_filter_destroy(struct capture_filter *state);
struct video_frame *capture_filter(struct capture_filter *state, struct video_frame *frame);

#ifdef __cplusplus
}
#endif

#endif /* CAPTURE_FILTER_H_ */


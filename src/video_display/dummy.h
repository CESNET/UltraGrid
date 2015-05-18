/**
 * @file   video_display/dummy.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This is an umbrella header for video functions.
 */
/*
 * Copyright (c) 2015 CESNET z.s.p.o.
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

struct audio_frame;
struct video_desc;
struct video_frame;

#ifdef __cplusplus
extern "C" {
#endif

display_type_t		*display_dummy_probe(void);
void 			*display_dummy_init(struct module *parent, const char *fmt, unsigned int flags);
void 			 display_dummy_run(void *state);
void 			 display_dummy_done(void *state);
struct video_frame	*display_dummy_getf(void *state);
int 			 display_dummy_putf(void *state, struct video_frame *frame,
                int nonblock);
int                      display_dummy_reconfigure(void *state, struct video_desc desc);
int                      display_dummy_get_property(void *state, int property, void *val, size_t *len);

void                     display_dummy_put_audio_frame(void *state, struct audio_frame *frame);
int                      display_dummy_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);

#ifdef __cplusplus
}
#endif


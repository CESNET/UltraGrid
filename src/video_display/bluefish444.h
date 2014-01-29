/*
 * FILE:   video_display/bluefish444.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
 * Copyright (c) 2001-2002 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#define DISPLAY_BLUEFISH444_ID	0x52f45430

struct audio_frame;
struct video_desc;
struct video_frame;

#ifdef __cplusplus
extern "C" {
#endif

display_type_t      *display_bluefish444_probe(void);
void                *display_bluefish444_init(char *fmt, unsigned int flags);
void                 display_bluefish444_run(void *state);
void                 display_bluefish444_done(void *state);
struct video_frame  *display_bluefish444_getf(void *state);
int                  display_bluefish444_putf(void *state, struct video_frame *frame,
                int nonblock);
int                  display_bluefish444_reconfigure(void *state,
                                struct video_desc desc);
int                  display_bluefish444_get_property(void *state, int property, void *val, size_t *len);

void                 display_bluefish444_put_audio_frame(void *state, struct audio_frame *frame);
int                  display_bluefish444_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);

#ifdef __cplusplus
} // END extern "C"
#endif

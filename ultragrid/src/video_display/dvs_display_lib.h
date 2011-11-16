/*
 * FILE:   display_dvs.h
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
#include <video_display.h>

#define DISPLAY_DVS_ID	0x74ac3e0f

struct audio_frame;
struct state_decoder;

typedef struct {
        int mode;
        unsigned int width;
        unsigned int height;
        double fps;
        int aux;
} hdsp_mode_table_t;

extern const hdsp_mode_table_t hdsp_mode_table[];

void                *display_dvs_init_impl(char *fmt, unsigned int flags);
void                 display_dvs_run_impl(void *state);
void                 display_dvs_done_impl(void *state);
struct video_frame  *display_dvs_getf_impl(void *state);
int                  display_dvs_putf_impl(void *state, char *frame);
void                 display_dvs_reconfigure_impl(void *state,
                                struct video_desc desc);
int                  display_dvs_get_property_impl(void *state, int property, void *val, int *len);

struct audio_frame * display_dvs_get_audio_frame_impl(void *state);
void display_dvs_put_audio_frame_impl(void *state, struct audio_frame *frame);


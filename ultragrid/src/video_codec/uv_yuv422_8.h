/*
 * FILE:   uv_yuv.h
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2004 University of Glasgow
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
 *    must display the following acknowledgement: "This product includes 
 *    software developed by the University of Glasgow".
 * 
 * 4. The name of the University may not be used to endorse or promote 
 *    products derived from this software without specific prior written 
 *    permission.
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
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#ifndef _UV_YUV_H
#define _UV_YUV_H

int                  uv_yuv_init(void);
void                 uv_yuv_exit(void);
const char *         uv_yuv_get_name(void);
const char *         uv_yuv_get_desc(void);
int                  uv_yuv_can_encode(void);
int                  uv_yuv_can_decode(void);
struct vcodec_state *uv_yuv_encoder_create(void);
void                 uv_yuv_encoder_destroy(struct vcodec_state *state);
int                  uv_yuv_encode(struct vcodec_state *state, 
                                   struct video_frame *in, struct coded_data *out);
struct vcodec_state *uv_yuv_decoder_create(void);
void                 uv_yuv_decoder_destroy(struct vcodec_state *state);
int                  uv_yuv_decode(struct vcodec_state *state, 
                                   struct coded_data *in, struct video_frame *out);

#endif /* _UV_VUY_H */

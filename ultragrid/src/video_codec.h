/*
 * FILE:   video_codec.h
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

#ifndef _VCODEC_H
#define _VCODEC_H

typedef int vcodec_id_t;

/*
 * Initialisation, probing of available codecs, name and RTP 
 * payload type mapping functions. 
 */

void        vcodec_init(void);
void        vcodec_done(void);
unsigned    vcodec_get_num_codecs (void);

const char *vcodec_get_name       (unsigned id);	/* Single word name */
const char *vcodec_get_description(unsigned id);	/* Descriptive text */

int         vcodec_can_encode     (unsigned id);
int         vcodec_can_decode     (unsigned id);

int         vcodec_map_payload    (uint8_t pt, unsigned id);
int         vcodec_unmap_payload  (uint8_t pt);
uint8_t     vcodec_get_payload    (unsigned id);
unsigned    vcodec_get_by_payload (uint8_t pt);

/*
 * Video encoder and decoder functions. These operate on particular
 * instances of a codec, identified by the "state" parameter.
 */

struct vcodec_state;

struct vcodec_state *vcodec_encoder_create (unsigned id);
void                 vcodec_encoder_destroy(struct vcodec_state *state);
int                  vcodec_encode         (struct vcodec_state *state,
                                            struct video_frame  *in,
					    struct coded_data   *out);

struct vcodec_state *vcodec_decoder_create (unsigned id);
void                 vcodec_decoder_destroy(struct vcodec_state *state);
int                  vcodec_decode         (struct vcodec_state *state,
					    struct coded_data   *in,
                                            struct video_frame  *out);

#endif /* _VCODEC_H */


/*
 * FILE:   ldgm.h
 * AUTHOR: Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 1998-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London.
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 */

#ifndef __LDGM_H__
#define __LDGM_H__

#include "rtp/ll.h"

#define LDGM_MAXIMAL_SIZE_RATIO 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct ldgm_desc {
        unsigned int k, m, c;
        unsigned int seed;
};

/* 
 * @param packet_size - approximate size of packet payload
 * @param frame_size - approximate size of whole protected frame
 */ 
void *ldgm_encoder_init_with_cfg(char *cfg);
void *ldgm_encoder_init_with_param(int packet_size, int frame_size, double max_expected_loss);
unsigned int ldgm_encoder_get_k(void *state);
unsigned int ldgm_encoder_get_m(void *state);
unsigned int ldgm_encoder_get_c(void *state);
unsigned int ldgm_encoder_get_seed(void *state);
void ldgm_encoder_encode(void *state, const char *hdr, int hdr_len,
                const char *body, int body_len, char **out, int *len);
void ldgm_encoder_free_buffer(void *state, char *buffer);
void ldgm_encoder_destroy(void *state);

void * ldgm_decoder_init(unsigned int k, unsigned int m, unsigned int c, unsigned int seed);
void ldgm_decoder_decode(void *state, const char *in, int in_len, char **out, int *len, struct linked_list *ll);
void ldgm_decoder_destroy(void *state);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifdef __cplusplus
#include <map>
void ldgm_decoder_decode_map(void *state, const char *in, int in_len, char **out, int *len,
                const std::map<int, int> &);
#endif

#endif /* __LDGM_H__ */

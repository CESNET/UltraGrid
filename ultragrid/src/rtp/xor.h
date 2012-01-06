/*
 * FILE:   xor.h
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

#ifndef __XOR_H__
#define __XOR_H__

struct xor_session;

struct xor_session * xor_init(int header_len, int max_payload_len);
void xor_add_packet(struct xor_session *session, const char *hdr, const char *payload, int payload_len);
void xor_emit_xor_packet(struct xor_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len);
void xor_clear(struct xor_session *session);
void xor_destroy(struct xor_session * session);
int xor_get_hdr_size(void);

struct xor_session *xor_restore_init(void);
void xor_restore_start(struct xor_session *session, char *data);
int xor_restore_packet(struct xor_session *session, char **pkt, uint16_t *len);
void xor_restore_destroy(struct xor_session *xor);
void xor_restore_invalidate(struct xor_session *xor);

#endif /* __XOR_H__ */

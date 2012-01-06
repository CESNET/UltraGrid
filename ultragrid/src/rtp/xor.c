/*
 * FILE:     xor.c
 * AUTHOR:   Martin Pulec <pulec@cesnet.cz>
 *
 * The routines in this file implement the Real-time Transport Protocol,
 * RTP, as specified in RFC1889 with current updates under discussion in
 * the IETF audio/video transport working group. Portions of the code are
 * derived from the algorithms published in that specification.
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 * Copyright (c) 1998-2001 University College London
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
 *      This product includes software developed by the Computer Science
 *      Department at University College London and by the University of
 *      Southern California Information Sciences Institute. This product also
 *      includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Department, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *    
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "memory.h"
#include "debug.h"
#include "net_udp.h"
#include "crypto/random.h"
#include "compat/drand48.h"
#include "compat/gettimeofday.h"
#include "crypto/crypt_des.h"
#include "crypto/crypt_aes.h"
#include "tv.h"
#include "crypto/md5.h"
#include "ntp.h"
#include "xor.h"


struct xor_pkt_hdr {
        uint32_t pkt_count;
        uint16_t payload_len_xor;
        uint16_t header_len;
        uint16_t payload_len;
}__attribute__((packed));

struct xor_session {
        char            *header_xor;
        char            *payload_xor; /* weak ref */
        int              header_len;
        int              max_payload_len;
        int              pkt_count;
        uint16_t              payload_len_xor;

        struct xor_pkt_hdr xor_hdr;
};

struct xor_session * xor_init(int header_len, int max_payload_len)
{
        struct xor_session *s;
        assert (header_len % 4 == 0);
        /* TODO: check if it shouldn't be less */

        s = (struct xor_session *) malloc(sizeof(struct xor_session));
        s->pkt_count = 0;
        s->payload_len_xor = 0;
        s->header_len = header_len;
        /* calloc is really needed here, because we will xor with this place incomming packets */
        s->header_xor = (char *) calloc(1, header_len + max_payload_len);
        s->max_payload_len = max_payload_len;
        s->payload_xor = (char *) s->header_xor + header_len;

        return s;
}

void xor_add_packet(struct xor_session *session, const char *hdr, const char *payload, int payload_len)
{
        int linepos;
        register unsigned int *line1;
        register const unsigned int *line2;

        session->pkt_count++;

        session->payload_len_xor ^= (uint16_t) payload_len;

        line1 = (unsigned int *) session->header_xor;
        line2 = (const unsigned int *) hdr;
        for(linepos = 0; linepos < session->header_len; linepos += 4) {
                *line1 ^= *line2;
                line1 += 1;
                line2 += 1;
        }

        line1 = (unsigned int *) session->payload_xor;
        line2 = (const unsigned int *) payload;
        for(linepos = 0; linepos < (payload_len - 15); linepos += 16) {
                asm volatile ("movdqu (%0), %%xmm0\n"
                        "movdqu (%1), %%xmm1\n"
                        "pxor %%xmm1, %%xmm0\n"
                        "movdqu %%xmm0, (%0)\n"
                        ::"r" ((unsigned long *) line1),
                        "r"((const unsigned long *) line2));
                line1 += 4;
                line2 += 4;
        }
        if(linepos != payload_len) {
                char *line1c = (char *) line1;
                char *line2c = (char *) line1;
                for(; linepos < payload_len; linepos += 1) {
                        *line1c ^= *line2c;
                        line1c += 1;
                        line2c += 1;
                }
        }

}

void xor_emit_xor_packet(struct xor_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len)
{
        session->xor_hdr.pkt_count = htonl(session->pkt_count);
        session->xor_hdr.header_len = htons(session->header_len);
        session->xor_hdr.payload_len = htons(session->max_payload_len);
        session->xor_hdr.payload_len_xor = htons(session->payload_len_xor);

        *hdr = (char *)  &session->xor_hdr;
        *hdr_len = (size_t) sizeof(struct xor_pkt_hdr);
        *payload = (const char *) session->header_xor;
        *payload_len = (size_t) (session->header_len + session->max_payload_len);
}

void xor_clear(struct xor_session *session)
{
        session->pkt_count = 0;
        session->payload_len_xor = 0;
        memset(session->header_xor, 0, session->header_len + session->max_payload_len);
}

void xor_destroy(struct xor_session * session)
{
        free(session->header_xor);
        free(session);
}


struct xor_session *xor_restore_init()
{
        struct xor_session *session;
        session = (struct xor_session *) malloc(sizeof(struct xor_session));
        return session;
}

void xor_restore_start(struct xor_session *session, char *data)
{
        session->pkt_count = - ntohl(((const struct xor_pkt_hdr *) data)->pkt_count);
        session->header_len = ntohs(((const struct xor_pkt_hdr *) data)->header_len);
        session->max_payload_len = ntohs(((const struct xor_pkt_hdr *) data)->payload_len);
        session->payload_len_xor = ntohs(((const struct xor_pkt_hdr *) data)->payload_len_xor);

        session->header_xor = (char *) data + sizeof(struct xor_pkt_hdr);
        session->payload_xor = data + sizeof(struct xor_pkt_hdr) + session->header_len;
}

int xor_restore_packet(struct xor_session *session, char **pkt, uint16_t *len)
{
        if(session->pkt_count == 0)
        {
                /* no packet restored */
                return FALSE;
        }
        if(session->pkt_count < -1) {
                debug_msg("Restoring packed failed - missing data.\n");
                return FALSE;
        }
        if(session->pkt_count > -1) {
                debug_msg("Restoring packed failed - missing xor.\n");
                return FALSE;
        }

        debug_msg("Restoring packed.\n");
        *pkt = session->header_xor;
        *len = session->payload_len_xor;
        return TRUE;
}

void xor_restore_destroy(struct xor_session *xor)
{
        free(xor);
}

void xor_restore_invalidate(struct xor_session *session)
{
        session->pkt_count = 0;
}


int xor_get_hdr_size()
{
        return sizeof(struct xor_pkt_hdr);
}


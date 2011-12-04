/*
 * FILE:     fec.c
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
#include "fec.h"


struct fec_pkt_hdr {
        uint32_t pkt_count;
#ifdef WORDS_BIGENDIAN
        uint16_t header_len;
        uint16_t payload_len;
#else
        uint16_t payload_len;
        uint16_t header_len;
#endif
}__attribute__((packed));

struct fec_session {
        char            *header_fec;
        char            *payload_fec; /* weak ref */
        int              header_len;
        int              max_payload_len;
        int              pkt_count;

        struct fec_pkt_hdr fec_hdr;
};

struct fec_session * fec_init(int header_len, int max_payload_len)
{
        struct fec_session *s;
        assert (header_len % 4 == 0);
        /* TODO: check if it shouldn't be less */
        assert(header_len + max_payload_len + 40 + sizeof(struct fec_pkt_hdr) < 9000);

        s = (struct fec_session *) malloc(sizeof(struct fec_session));
        s->pkt_count = 0;
        s->header_len = header_len;
        /* calloc is really needed here, because we will xor with this place incomming packets */
        s->header_fec = (char *) calloc(1, header_len + max_payload_len);
        s->max_payload_len = max_payload_len;
        s->payload_fec = (char *) s->header_fec + header_len;

        return s;
}

void fec_add_packet(struct fec_session *session, const char *hdr, const char *payload, int payload_len)
{
        int linepos;
        register unsigned int *line1;
        register const unsigned int *line2;

        session->pkt_count++;

        line1 = (unsigned int *) session->header_fec;
        line2 = (const unsigned int *) hdr;
        for(linepos = 0; linepos < session->header_len; linepos += 4) {
                *line1 ^= *line2;
                line1 += 1;
                line2 += 1;
        }

        line1 = (unsigned int *) session->payload_fec;
        line2 = (const unsigned int *) payload;
        for(linepos = 0; linepos < (payload_len - 15); linepos += 16) {
                asm volatile ("movdqu (%0), %%xmm0\n"
                        "movdqu (%1), %%xmm1\n"
                        "pxor %%xmm1, %%xmm0\n"
                        "movdqu %%xmm0, (%0)\n"
                        ::"r" ((unsigned long *) line1),
                        "r"((unsigned long *) line2));
                line1 += 4;
                line2 += 4;
        }
        if(linepos != payload_len) {
                char *line1c = line1;
                char *line2c = line1;
                for(; linepos < payload_len; linepos += 1) {
                        *line1c ^= *line2c;
                        line1c += 1;
                        line2c += 1;
                }
        }

}

void fec_emit_fec_packet(struct fec_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len)
{
        session->fec_hdr.pkt_count = htonl(session->pkt_count);
        session->fec_hdr.header_len = htons(session->header_len);
        session->fec_hdr.payload_len = htons(session->max_payload_len);

        *hdr = (char *)  &session->fec_hdr;
        *hdr_len = (size_t) sizeof(struct fec_pkt_hdr);
        *payload = (const char *) session->header_fec;
        *payload_len = (size_t) (session->header_len + session->max_payload_len);
}

void fec_clear(struct fec_session *session)
{
        session->pkt_count = 0;
        memset(session->header_fec, 0, session->header_len + session->max_payload_len);
}

void fec_destroy(struct fec_session * session)
{
        free(session->header_fec);
        free(session);
}


struct fec_session *fec_restore_init()
{
        struct fec_session *session;
        session = (struct fec_session *) malloc(sizeof(struct fec_session));
        return session;
}

void fec_restore_start(struct fec_session *session, const char *data)
{
        session->pkt_count = - ntohl(((struct fec_pkt_hdr *) data)->pkt_count);
        session->header_len = ntohs(((struct fec_pkt_hdr *) data)->header_len);
        session->max_payload_len = ntohs(((struct fec_pkt_hdr *) data)->payload_len);

        session->header_fec = data + sizeof(struct fec_pkt_hdr);
        session->payload_fec = data + sizeof(struct fec_pkt_hdr) + session->header_len;
}

int fec_restore_packet(struct fec_session *session, char **pkt)
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
                debug_msg("Restoring packed failed - missing FEC.\n");
                return FALSE;
        }

        debug_msg("Restoring packed.\n");
        *pkt = session->header_fec;
        return TRUE;
}

void fec_restore_destroy(struct fec_session *fec)
{
        free(fec);
}

void fec_restore_invalidate(struct fec_session *session)
{
        session->pkt_count = 0;
}



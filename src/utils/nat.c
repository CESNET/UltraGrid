/**
 * @file   utils/nat.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "utils/nat.h"

#define ENABLE_STRNATPMPERR 1
#define STATICLIB 1
#include "ext-deps/libnatpmp-20150609/natpmp.h"

#define ALLOCATION_TIMEOUT (4 * 3600)

struct ug_nat_traverse {
        enum {
                UG_NAT_TRAVERSE_NONE,
                UG_NAT_TRAVERSE_NAT_PMP,
        } traverse;
        int audio_rx_port;
        int video_rx_port;
};

static bool nat_pmp_add_mapping(natpmp_t *natpmp, int privateport, int publicport, int lifetime)
{
        if (privateport == 0 && publicport == 0) {
                return true;
        }

        int r = 0;
        /* sendnewportmappingrequest() */
        r = sendnewportmappingrequest(natpmp, NATPMP_PROTOCOL_UDP,
                        privateport, publicport,
                        lifetime);
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE,
                        "[NAT PMP] sendnewportmappingrequest returned %d (%s)\n",
                        r, r == 12 ? "SUCCESS" : "FAILED");
        if (r < 0) {
                return false;
        }

        natpmpresp_t response;
        do {
                fd_set fds;
                struct timeval timeout;
                FD_ZERO(&fds);
                FD_SET(natpmp->s, &fds);
                getnatpmprequesttimeout(natpmp, &timeout);
                select(FD_SETSIZE, &fds, NULL, NULL, &timeout);
                r = readnatpmpresponseorretry(natpmp, &response);
                log_msg(LOG_LEVEL_VERBOSE, "[NAT PMP] readnatpmpresponseorretry returned %d (%s)\n",
                                r, r == 0 ? "OK" : (r == NATPMP_TRYAGAIN ? "TRY AGAIN" : "FAILED"));
        } while(r==NATPMP_TRYAGAIN);
        if(r<0) {
                log_msg(LOG_LEVEL_ERROR, "[NAT PMP] readnatpmpresponseorretry() failed : %s\n",
                                strnatpmperr(r));
                return false;
        }

        log_msg(LOG_LEVEL_INFO, "[NAT PMP] Mapped public port %hu protocol %s to local port %hu "
                        "liftime %u\n",
                        response.pnu.newportmapping.mappedpublicport,
                        response.type == NATPMP_RESPTYPE_UDPPORTMAPPING ? "UDP" :
                        (response.type == NATPMP_RESPTYPE_TCPPORTMAPPING ? "TCP" :
                         "UNKNOWN"),
                        response.pnu.newportmapping.privateport,
                        response.pnu.newportmapping.lifetime);
        log_msg(LOG_LEVEL_DEBUG, "[NAT PMP] epoch = %u\n", response.epoch);

        return true;
}

static bool setup_nat_pmp(int video_rx_port, int audio_rx_port, int lifetime)
{
        struct in_addr gateway_in_use = { 0 };
        natpmp_t natpmp;
        int r = 0;
        r = initnatpmp(&natpmp, 0, 0);
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "[NAT PMP] initnatpmp returned %d (%s)\n", r,
                        r ? "FAILED" : "SUCCESS");
        if (r < 0) {
                return false;
        }
        gateway_in_use.s_addr = natpmp.gateway;
        log_msg(LOG_LEVEL_NOTICE, "[NAT PMP] using gateway: %s\n", inet_ntoa(gateway_in_use));

        /* sendpublicaddressrequest() */
        r = sendpublicaddressrequest(&natpmp);
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "[NAT PMP] sendpublicaddressrequest returned %d (%s)\n",
                        r, r == 2 ? "SUCCESS" : "FAILED");
        if (r < 0) {
                return false;
        }

        natpmpresp_t response;
        do {
                fd_set fds;
                struct timeval timeout;
                FD_ZERO(&fds);
                FD_SET(natpmp.s, &fds);
                getnatpmprequesttimeout(&natpmp, &timeout);
                r = select(FD_SETSIZE, &fds, NULL, NULL, &timeout);
                if(r<0) {
                        log_msg(LOG_LEVEL_ERROR, "[NAT PMP] select()\n");
                        return false;
                }
                r = readnatpmpresponseorretry(&natpmp, &response);
                int sav_errno = errno;
                log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, "[NAT PMP] readnatpmpresponseorretry returned %d (%s)\n",
                                r, r == 0 ? "OK" : ( r== NATPMP_TRYAGAIN ? "TRY AGAIN" : "FAILED"));
                if (r < 0 && r != NATPMP_TRYAGAIN) {
                        log_msg(LOG_LEVEL_ERROR, "[NAT_PMP] readnatpmpresponseorretry() failed : %s\n",
                                        strnatpmperr(r));
                        log_msg(LOG_LEVEL_ERROR, "[NAT PMP]  errno=%d '%s'\n",
                                        sav_errno, strerror(sav_errno));
                }
        } while(r==NATPMP_TRYAGAIN);

        if (r < 0) {
                return false;
        }

        log_msg(LOG_LEVEL_NOTICE, "[NAT PMP] Public IP address: %s\n", inet_ntoa(response.pnu.publicaddress.addr));
        log_msg(LOG_LEVEL_DEBUG, "[NAT PMP] epoch = %u\n", response.epoch);

        if (!nat_pmp_add_mapping(&natpmp, video_rx_port, video_rx_port, lifetime) ||
                        !nat_pmp_add_mapping(&natpmp, audio_rx_port, audio_rx_port, lifetime)) {
                return false;
        }

        r = closenatpmp(&natpmp);
        log_msg(LOG_LEVEL_VERBOSE, "[NAT PMP] closenatpmp() returned %d (%s)\n", r, r==0?"SUCCESS":"FAILED");
        return r >= 0;
}

struct ug_nat_traverse *start_nat_traverse(int video_rx_port, int audio_rx_port)
{
        assert(video_rx_port >= 0 && video_rx_port <= 65535 && audio_rx_port >= 0 && audio_rx_port <= 65535);
        struct ug_nat_traverse s = { .audio_rx_port = audio_rx_port, .video_rx_port = video_rx_port };
        if (setup_nat_pmp(video_rx_port, audio_rx_port, ALLOCATION_TIMEOUT)) {
                log_msg(LOG_LEVEL_NOTICE, "Successfully set NAT traversal with NAT PMP. Sender can send to external IP address.\n");
                s.traverse = UG_NAT_TRAVERSE_NAT_PMP;
                return memcpy(malloc(sizeof s), &s, sizeof s);
        }
        // other techniques may follow
        return NULL;
}

void stop_nat_traverse(struct ug_nat_traverse *s)
{
        if (s == NULL) {
                return;
        }

        switch (s->traverse) {
        case UG_NAT_TRAVERSE_NAT_PMP:
                setup_nat_pmp(s->video_rx_port, s->audio_rx_port, 0);
                break;
        default:
                break;
        }

        free(s);
}

/* vim: set expandtab sw=8: */

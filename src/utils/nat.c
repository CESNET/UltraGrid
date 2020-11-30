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

#ifdef HAVE_PCP
#include <pcp-client/pcp.h>
#endif // defined HAVE_PCP

#include "debug.h"
#include "utils/nat.h"

#define ENABLE_STRNATPMPERR 1
#define STATICLIB 1
#include "ext-deps/libnatpmp-20150609/natpmp.h"

#define ALLOCATION_TIMEOUT (4 * 3600)
#define MOD_NAME "[NAT] "

struct ug_nat_traverse {
        enum {
                UG_NAT_TRAVERSE_NONE,
                UG_NAT_TRAVERSE_NAT_PMP,
                UG_NAT_TRAVERSE_PCP,
        } traverse;
        union {
#ifdef HAVE_PCP
                struct pcp_state {
                        pcp_ctx_t *ctx;
                        pcp_flow_t *audio_flow;
                        pcp_flow_t *video_flow;
                } pcp_state;
#endif // defined HAVE_PCP
        };
        int audio_rx_port;
        int video_rx_port;
};

#ifdef HAVE_PCP
static int check_flow_info(pcp_flow_t* f)
{
    size_t cnt=0;
    pcp_flow_info_t *info_buf = NULL;
    pcp_flow_info_t *ret = pcp_flow_get_info(f,&cnt);
    int ret_val = 2;
    info_buf=ret;
    for (; cnt>0; cnt--, ret++) {
        switch(ret->result)
        {
        case pcp_state_succeeded:
            printf("\nFlow signaling succeeded.\n");
            ret_val = 0;
            break;
        case pcp_state_short_lifetime_error:
            printf("\nFlow signaling failed. Short lifetime error received.\n");
            ret_val = 3;
            break;
        case pcp_state_failed:
            printf("\nFlow signaling failed.\n");
            ret_val = 4;
            break;
        default:
            continue;
        }
        break;
    }

    if (info_buf) {
        free(info_buf);
    }

    return ret_val;
}

static const char* decode_fresult(pcp_fstate_e s)
{
    switch (s) {
    case pcp_state_short_lifetime_error:
        return "slerr";
    case pcp_state_succeeded:
        return "succ";
    case pcp_state_failed:
        return "fail";
    default:
        return "proc";
    }
}

#ifdef WIN32
static char *ctime_r(const time_t *timep, char *buf)
{
    ctime_s(buf, 26, timep);
    return buf;
}
#endif

static void print_ext_addr(pcp_flow_t* f)
{
    size_t cnt=0;
    pcp_flow_info_t *info_buf = NULL;
    pcp_flow_info_t *ret = pcp_flow_get_info(f,&cnt);
    info_buf=ret;

    printf("%-20s %-4s %-20s %5s   %-20s %5s   %-20s %5s %3s %5s %s\n",
            "PCP Server IP",
            "Prot",
            "Int. IP", "port",
            "Dst. IP", "port",
            "Ext. IP", "port",
            "Res", "State","Ends");
    for (; cnt>0; cnt--, ret++) {
        char ntop_buffs[4][INET6_ADDRSTRLEN];
        char timebuf[32];

        printf("%-20s %-4s %-20s %5hu   %-20s %5hu   %-20s %5hu %3d %5s %s",
                inet_ntop(AF_INET6, &ret->pcp_server_ip, ntop_buffs[0],
                    sizeof(ntop_buffs[0])),
                ret->protocol == IPPROTO_TCP ? "TCP" : (
                   ret->protocol == IPPROTO_UDP ? "UDP" : "UNK"),
                inet_ntop(AF_INET6, &ret->int_ip, ntop_buffs[1],
                    sizeof(ntop_buffs[1])),
                ntohs(ret->int_port),
                inet_ntop(AF_INET6, &ret->dst_ip, ntop_buffs[2],
                    sizeof(ntop_buffs[2])),
                ntohs(ret->dst_port),
                inet_ntop(AF_INET6, &ret->ext_ip, ntop_buffs[3],
                    sizeof(ntop_buffs[3])),
                ntohs(ret->ext_port),
                ret->pcp_result_code,
                decode_fresult(ret->result),
                ret->recv_lifetime_end == 0 ? " -\n" :
                        ctime_r(&ret->recv_lifetime_end, timebuf));
    }
    if (info_buf) {
        free(info_buf);
    }
}

static void done_pcp(struct pcp_state *s)
{
        pcp_terminate(s->ctx, 1);
}

#define PCP_ASSERT_EQ(expr, val) { int rc = expr; if (rc != (val)) abort(); }
#define PCP_ASSERT_NEQ(expr, val) { int rc = expr; if (rc == (val)) abort(); }
#define PCP_WAIT_MS 500

static bool setup_pcp(struct pcp_state *s, int video_rx_port, int audio_rx_port, int lifetime)
{
        struct sockaddr_in src = { 0 };
        struct sockaddr_in dst = { 0 };
        socklen_t src_len = sizeof src;

        s->ctx = pcp_init(ENABLE_AUTODISCOVERY, NULL);
        // handle errors

        // get our outbound IP address
        dst.sin_family = AF_INET;
        dst.sin_port = htons(80);
        PCP_ASSERT_EQ(inet_pton(AF_INET, "93.184.216.34", &dst.sin_addr.s_addr), 1);
        int fd = socket(AF_INET, SOCK_DGRAM, 0);
        PCP_ASSERT_NEQ(fd, -1);
        PCP_ASSERT_EQ(connect(fd, (struct sockaddr *) &dst, sizeof dst), 0);
        PCP_ASSERT_EQ(getsockname(fd, (struct sockaddr *) &src, &src_len), 0);
        CLOSESOCKET(fd);

        bool ret = true;
        if (video_rx_port) {
                src.sin_port = htons(video_rx_port);
                s->video_flow = pcp_new_flow(s->ctx, (struct sockaddr*) &src, NULL, NULL, IPPROTO_UDP, lifetime, NULL);
                if (s->video_flow == NULL) {
                        ret = false;
                }
        }
        if (audio_rx_port) {
                src.sin_port = htons(audio_rx_port);
                s->audio_flow = pcp_new_flow(s->ctx, (struct sockaddr*) &src, NULL, NULL, IPPROTO_UDP, lifetime, NULL);
                if (s->audio_flow == NULL) {
                        ret = false;
                }
        }
        if (!ret) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "PCP - cannot create flow!\n");
                done_pcp(s);
                return false;
        }
        if (video_rx_port != 0) {
                pcp_wait(s->video_flow, PCP_WAIT_MS, 0);
                ret = ret && check_flow_info(s->video_flow) == 0;
                print_ext_addr(s->video_flow);
        }
        if (audio_rx_port != 0) {
                pcp_wait(s->audio_flow, PCP_WAIT_MS, 0);
                ret = ret && check_flow_info(s->audio_flow) == 0;
                print_ext_addr(s->audio_flow);
        }

        if (!ret) {
                done_pcp(s);
        }

        return ret;
}
#endif // defined HAVE_PCP

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

#ifdef HAVE_PCP
        if (setup_pcp(&s.pcp_state, video_rx_port, audio_rx_port, ALLOCATION_TIMEOUT)) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Successfully set NAT traversal with PCP. Sender can send to external IP address.\n");
                s.traverse = UG_NAT_TRAVERSE_PCP;
                return memcpy(malloc(sizeof s), &s, sizeof s);
        }
#else
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "PCP support not compiled in!\n");
#endif // defined HAVE_PCP

        if (setup_nat_pmp(video_rx_port, audio_rx_port, ALLOCATION_TIMEOUT)) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Successfully set NAT traversal with NAT PMP. Sender can send to external IP address.\n");
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
#ifdef HAVE_PCP
        case UG_NAT_TRAVERSE_PCP:
                done_pcp(&s->pcp_state);
                break;
#endif // defined HAVE_PCP
        case UG_NAT_TRAVERSE_NAT_PMP:
                setup_nat_pmp(s->video_rx_port, s->audio_rx_port, 0);
                break;
        default:
                break;
        }

        free(s);
}

/* vim: set expandtab sw=8: */

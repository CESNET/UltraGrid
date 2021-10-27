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

#include <pthread.h>

#ifdef HAVE_NATPMP
#define ENABLE_STRNATPMPERR 1
#define STATICLIB 1
#include <natpmp.h>
#endif // defined HAVE_NATPMP
#ifdef HAVE_PCP
#include <pcp-client/pcp.h>
#endif // defined HAVE_PCP

#include "debug.h"
#include "rtp/net_udp.h" // socket_error
#include "utils/color_out.h"
#include "utils/nat.h"
#include "utils/net.h"

#define DEFAULT_ALLOCATION_TIMEOUT_S 1800
#define DEFAULT_NAT_PMP_TIMEOUT 5
#define MOD_NAME "[NAT] "
#define PREALLOCATE_S 30 ///< number of seconds that repeated allocation is performed before timeout

struct ug_nat_traverse {
        enum traverse_t {
                UG_NAT_TRAVERSE_NONE,
                UG_NAT_TRAVERSE_PCP,
                UG_NAT_TRAVERSE_FIRST = UG_NAT_TRAVERSE_PCP,
                UG_NAT_TRAVERSE_NAT_PMP,
                UG_NAT_TRAVERSE_LAST = UG_NAT_TRAVERSE_NAT_PMP,
        } traverse;
        union {
#ifdef HAVE_PCP
                struct pcp_state {
                        pcp_ctx_t *ctx;
                        pcp_flow_t *audio_flow;
                        pcp_flow_t *video_flow;
                } pcp_state;
#endif // defined HAVE_PCP
        } nat_state;
        int audio_rx_port;
        int video_rx_port;
        int allocation_duration;

        bool initialized; ///< whether the state was initialized
        pthread_t keepalive_thread;
        bool keepalive_should_exit;
        pthread_mutex_t keepalive_mutex;
        pthread_cond_t keepalive_cv;
};

static bool setup_pcp(struct ug_nat_traverse *state, int video_rx_port, int audio_rx_port, int lifetime);
static void done_pcp(struct ug_nat_traverse *state);
static bool setup_nat_pmp(struct ug_nat_traverse *state, int video_rx_port, int audio_rx_port, int lifetime);
static void done_nat_pmp(struct ug_nat_traverse *state);

static const struct nat_traverse_info_t {
        const char *name_short; ///< for command-line specifiction
        const char *name_long; ///< for output
        bool (*init)(struct ug_nat_traverse *state, int video_rx_port, int audio_rx_port, int lifetime);
        void (*done)(struct ug_nat_traverse *state);
} nat_traverse_info[] = {
        [ UG_NAT_TRAVERSE_PCP ] = { "pcp", "PCP", setup_pcp, done_pcp },
        [ UG_NAT_TRAVERSE_NAT_PMP ] = { "natpmp", "NAT PMP", setup_nat_pmp, done_nat_pmp },
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
        case pcp_state_processing:
            printf("\nFlow signaling processing.\n");
            continue;
        case pcp_state_partial_result:
            printf("\nFlow signaling partial result.\n");
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
#endif // defined HAVE_PCP

static void done_pcp(struct ug_nat_traverse *state)
{
#ifdef HAVE_PCP
        struct pcp_state *s = &state->nat_state.pcp_state;
        pcp_terminate(s->ctx, 1);
#else
        UNUSED(state);
#endif
}

#define PCP_WAIT_MS 500

#define NAT_HANDLE_ERROR(msg) { if (fd != -1) CLOSESOCKET(fd); socket_error(msg); return false; }
#define NAT_ASSERT_EQ(expr, val) { int rc = expr; if (rc != (val)) NAT_HANDLE_ERROR(#expr) }
#define NAT_ASSERT_NEQ(expr, val) { int rc = expr; if (rc == (val)) NAT_HANDLE_ERROR(#expr) }
static bool get_outbound_ip(struct sockaddr_in *out) {
        struct sockaddr_in dst = { 0 };
        socklen_t src_len = sizeof *out;

        dst.sin_family = AF_INET;
        dst.sin_port = htons(80);
        int fd = -1;
        NAT_ASSERT_EQ(inet_pton(AF_INET, "93.184.216.34", &dst.sin_addr.s_addr), 1);
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        NAT_ASSERT_NEQ(fd, -1);
        NAT_ASSERT_EQ(connect(fd, (struct sockaddr *) &dst, sizeof dst), 0);
        NAT_ASSERT_EQ(getsockname(fd, (struct sockaddr *) out, &src_len), 0);
        CLOSESOCKET(fd);

        return true;
}

static bool setup_pcp(struct ug_nat_traverse *state, int video_rx_port, int audio_rx_port, int lifetime)
{
#ifdef HAVE_PCP
        struct pcp_state *s = &state->nat_state.pcp_state;
        struct sockaddr_in src = { 0 };

        s->ctx = pcp_init(ENABLE_AUTODISCOVERY, NULL);
        // handle errors

        // get our outbound IP address
        if (!get_outbound_ip(&src)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "PCP - cannot get outbound address!\n");
                return false;
        }

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
                done_pcp(state);
                return false;
        }
        if (video_rx_port != 0) {
                pcp_wait(s->video_flow, PCP_WAIT_MS, 0);
                ret = ret && check_flow_info(s->video_flow) == 0;
                if (ret) {
                        print_ext_addr(s->video_flow);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "PCP - cannot create video flow!\n");
                }
        }
        if (audio_rx_port != 0) {
                pcp_wait(s->audio_flow, PCP_WAIT_MS, 0);
                ret = ret && check_flow_info(s->audio_flow) == 0;
                if (ret) {
                        print_ext_addr(s->audio_flow);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "PCP - cannot create video flow!\n");
                }
        }

        if (!ret) {
                done_pcp(state);
        }

        return ret;
#else
        UNUSED(state);
        UNUSED(video_rx_port);
        UNUSED(audio_rx_port);
        UNUSED(lifetime);
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "PCP support not compiled in!\n");
        return false;
#endif // defined HAVE_PCP
}

#ifdef HAVE_NATPMP
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
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, MOD_NAME
                        "NAT PMP - sendnewportmappingrequest returned %d (%s)\n",
                        r, r == 12 ? "SUCCESS" : "FAILED");
        if (r < 0) {
                return false;
        }

        natpmpresp_t response = { 0 };
        time_t t0 = time(NULL);
        do {
                fd_set fds;
                struct timeval timeout = { 0 };
                FD_ZERO(&fds);
                FD_SET(natpmp->s, &fds);
                r = getnatpmprequesttimeout(natpmp, &timeout);
                if (r != 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "NAT PMP - getnatpmprequesttimeout returned %d (%s)\n",
                                        r, strnatpmperr(r));
                        break;
                }
                select(FD_SETSIZE, &fds, NULL, NULL, &timeout);
                r = readnatpmpresponseorretry(natpmp, &response);
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "NAT PMP - readnatpmpresponseorretry returned %d (%s)\n",
                                r, r == 0 ? "OK" : (r == NATPMP_TRYAGAIN ? "TRY AGAIN" : "FAILED"));
        } while (r == NATPMP_TRYAGAIN && time(NULL) - t0 < DEFAULT_NAT_PMP_TIMEOUT);
        if(r<0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "NAT PMP - readnatpmpresponseorretry() failed : %s\n",
                                strnatpmperr(r));
                return false;
        }

        log_msg(LOG_LEVEL_INFO, MOD_NAME "NAT PMP - Mapped public port %hu protocol %s to local port %hu "
                        "liftime %u\n",
                        response.pnu.newportmapping.mappedpublicport,
                        response.type == NATPMP_RESPTYPE_UDPPORTMAPPING ? "UDP" :
                        (response.type == NATPMP_RESPTYPE_TCPPORTMAPPING ? "TCP" :
                         "UNKNOWN"),
                        response.pnu.newportmapping.privateport,
                        response.pnu.newportmapping.lifetime);
        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "NAT PMP - epoch = %u\n", response.epoch);

        return true;
}
#endif // defined HAVE_NATPMP

static bool setup_nat_pmp(struct ug_nat_traverse *state, int video_rx_port, int audio_rx_port, int lifetime)
{
        UNUSED(state);
#ifdef HAVE_NATPMP
        struct in_addr gateway_in_use = { 0 };
        natpmp_t natpmp;
        int r = 0;
        r = initnatpmp(&natpmp, 0, 0);
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, MOD_NAME "NAT PMP - initnatpmp returned %d (%s)\n", r,
                        r ? "FAILED" : "SUCCESS");
        if (r < 0) {
                return false;
        }
        gateway_in_use.s_addr = natpmp.gateway;
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "NAT PMP - using gateway: %s\n", inet_ntoa(gateway_in_use));

        /* sendpublicaddressrequest() */
        r = sendpublicaddressrequest(&natpmp);
        log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, MOD_NAME "NAT PMP - sendpublicaddressrequest returned %d (%s)\n",
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
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "NAT PMP - select()\n");
                        return false;
                }
                r = readnatpmpresponseorretry(&natpmp, &response);
                int sav_errno = errno;
                log_msg(r < 0 ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE, MOD_NAME "NAT PMP - readnatpmpresponseorretry returned %d (%s)\n",
                                r, r == 0 ? "OK" : ( r== NATPMP_TRYAGAIN ? "TRY AGAIN" : "FAILED"));
                if (r < 0 && r != NATPMP_TRYAGAIN) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "NAT PMP - readnatpmpresponseorretry() failed : %s\n",
                                        strnatpmperr(r));
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "NAT PMP - errno=%d '%s'\n",
                                        sav_errno, strerror(sav_errno));
                }
        } while(r==NATPMP_TRYAGAIN);

        if (r < 0) {
                return false;
        }

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "NAT PMP - Public IP address: %s\n", inet_ntoa(response.pnu.publicaddress.addr));
        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "NAT PMP - epoch = %u\n", response.epoch);

        if (!nat_pmp_add_mapping(&natpmp, video_rx_port, video_rx_port, lifetime) ||
                        !nat_pmp_add_mapping(&natpmp, audio_rx_port, audio_rx_port, lifetime)) {
                return false;
        }

        r = closenatpmp(&natpmp);
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "NAT PMP - closenatpmp() returned %d (%s)\n", r, r==0?"SUCCESS":"FAILED");
        return r >= 0;
#else
        UNUSED(video_rx_port);
        UNUSED(audio_rx_port);
        UNUSED(lifetime);
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "NAT-PMP support not compiled in!\n");
        return false;
#endif // defined HAVE_NATPMP
}

static void done_nat_pmp(struct ug_nat_traverse *state) {
        setup_nat_pmp(state, state->video_rx_port, state->audio_rx_port, 0);
}

static void *nat_traverse_keepalive(void *state) {
        struct ug_nat_traverse *s = state;

        struct timespec timeout = { .tv_sec = time(NULL) + s->allocation_duration - PREALLOCATE_S, .tv_nsec = 0 };

        while (1) {
                pthread_mutex_lock(&s->keepalive_mutex);
                int rc = 0;
                if (!s->keepalive_should_exit) {
                        rc = pthread_cond_timedwait(&s->keepalive_cv, &s->keepalive_mutex, &timeout);
                }
                pthread_mutex_unlock(&s->keepalive_mutex);
                if (s->keepalive_should_exit) {
                        break;
                }

                if (rc != ETIMEDOUT) {
                        perror(__func__);
                        continue;
                }

                if (nat_traverse_info[s->traverse].init(s, s->video_rx_port, s->audio_rx_port, s->allocation_duration)) {
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Mapping renewed successfully for %d seconds.\n", s->allocation_duration);
                        timeout.tv_sec = time(NULL) + s->allocation_duration - PREALLOCATE_S;
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Mapping renewal failed! Trying again in 5 seconds\n");
                        timeout.tv_sec = time(NULL) + 5;
                }
        }

        return NULL;
}

/**
 * @param config NULL  - do not enable NAT traversal
 *                ""   - enable with default arguments
 *               other - start with configuration
 * @returns state, NULL on error (or help)
 */
struct ug_nat_traverse *start_nat_traverse(const char *config, const char *remote_host, int video_rx_port, int audio_rx_port)
{
        if (config == NULL) {
                if ((video_rx_port != 0 || audio_rx_port != 0) && (!is_host_private(remote_host) && !is_host_loopback(remote_host))) {
                        struct sockaddr_in out;
                        if (get_outbound_ip(&out) && is_addr_private((struct sockaddr *) &out)) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Private outbound IPv4 address detected and binding as a receiver. Consider adding '-N' option for NAT traversal.\n");
                        }

                }
                return calloc(1, sizeof(struct ug_nat_traverse));
        }

        assert(video_rx_port >= 0 && video_rx_port <= 65535 && audio_rx_port >= 0 && audio_rx_port <= 65535);

        if (strcmp(config, "help") == 0) {
                printf("Usage:\n");
                color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-N");
                color_out(COLOR_OUT_BOLD, "[protocol[:renewal-interval]]\n");
                printf("where:\n");
                color_out(COLOR_OUT_BOLD, "\tprotocol");
                printf(" - one of:");
                for (int i = UG_NAT_TRAVERSE_FIRST; i <= UG_NAT_TRAVERSE_LAST; ++i) {
                        color_out(COLOR_OUT_BOLD, " %s", nat_traverse_info[i].name_short);
                }
                printf("\n");
                color_out(COLOR_OUT_BOLD, "\trenewal-interval");
                printf(" - mapping renew interval (in seconds, min: %d)\n", PREALLOCATE_S + 1);
                return NULL;
        }

        struct ug_nat_traverse *s = calloc(1, sizeof(struct ug_nat_traverse));
        s->audio_rx_port = audio_rx_port;
        s->video_rx_port = video_rx_port;
        s->allocation_duration = DEFAULT_ALLOCATION_TIMEOUT_S;

        bool not_found = true;
        char protocol[strlen(config) + 1];
        strcpy(protocol, config);
        if (strchr(protocol, ':') != NULL) {
                s->allocation_duration = atoi(strchr(protocol, ':') + 1);
                if (s->allocation_duration < PREALLOCATE_S + 1) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong renewal interval: %s, minimal: %d\n", strchr(protocol, ':') + 1, PREALLOCATE_S + 1);
                        free(s);
                        return NULL;
                }
                *strchr(protocol, ':') = '\0';
        }
        for (int i = UG_NAT_TRAVERSE_FIRST; i <= UG_NAT_TRAVERSE_LAST; ++i) {
                if (strlen(protocol) > 0 && strcmp(nat_traverse_info[i].name_short, protocol) != 0) {
                        continue;
                }
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Trying: %s\n", nat_traverse_info[i].name_long);
                not_found = false;
                if (nat_traverse_info[i].init(s, video_rx_port, audio_rx_port, s->allocation_duration)) {
                        s->traverse = i;
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Set NAT traversal with %s for %d seconds (auto-renewed). Sender can send to external IP address.\n", nat_traverse_info[i].name_long, s->allocation_duration);
                        break;
                }
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s initialization failed!\n", nat_traverse_info[i].name_long);
        }

        if (s->traverse == UG_NAT_TRAVERSE_NONE) {
                if (not_found) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong module name: %s.\n", protocol);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not initialize any NAT traversal.\n");
                }
                free(s);
                return NULL;
        }

        int rc = 0;
        rc |= pthread_mutex_init(&s->keepalive_mutex, NULL);
        rc |= pthread_cond_init(&s->keepalive_cv, NULL);
        rc |= pthread_create(&s->keepalive_thread, NULL, nat_traverse_keepalive, s);
        assert(rc == 0);

        s->initialized = true;

        return s;
}

void stop_nat_traverse(struct ug_nat_traverse *s)
{
        if (s == NULL || s->initialized == false) {
                free(s);
                return;
        }

        pthread_mutex_lock(&s->keepalive_mutex);
        s->keepalive_should_exit = true;
        pthread_mutex_unlock(&s->keepalive_mutex);
        pthread_cond_signal(&s->keepalive_cv);
        pthread_join(s->keepalive_thread, NULL);

        pthread_mutex_destroy(&s->keepalive_mutex);
        pthread_cond_destroy(&s->keepalive_cv);

        if (nat_traverse_info[s->traverse].done) {
                nat_traverse_info[s->traverse].done(s);
        }

        free(s);
}

/* vim: set expandtab sw=8: */

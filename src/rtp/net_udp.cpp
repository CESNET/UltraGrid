/*
 * FILE:     net_udp.c
 * AUTHOR:   Colin Perkins 
 * MODIFIED: Orion Hodson & Piers O'Hanlon
 *           David Cassany   <david.cassany@i2cat.net>
 *           Gerard Castillo <gerard.castillo@i2cat.net>
 *           Martin Pulec    <pulec@cesnet.cz>
 * 
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2015 CESNET z.s.p.o.
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
 *      Department at University College London
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
 */

/* If this machine supports IPv6 the symbol HAVE_IPv6 should */
/* be defined in either config_unix.h or config_win32.h. The */
/* appropriate system header files should also be included   */
/* by those files.                                           */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "memory.h"
#include "compat/platform_semaphore.h"
#include "compat/vsnprintf.h"
#include "net_udp.h"
#include "rtp.h"

#ifdef NEED_ADDRINFO_H
#include "addrinfo.h"
#endif

#include <condition_variable>
#include <chrono>
#include <mutex>

using std::condition_variable;
using std::mutex;
using std::unique_lock;

#define MAX_UDP_READER_QUEUE_LEN 10

static int resolve_address(socket_udp *s, const char *addr);
static void *udp_reader(void *arg);

#define IPv4	4
#define IPv6	6

#ifdef WIN2K_IPV6
const struct in6_addr in6addr_any = { IN6ADDR_ANY_INIT };
#endif

#ifdef WIN32
typedef char *sockopt_t;
#else
typedef void *sockopt_t;
#endif

#ifdef WIN32
#define CLOSESOCKET closesocket
#else
#define CLOSESOCKET close
#endif

/* This is pretty nasty but it's the simplest way to get round */
/* the Detexis bug that means their MUSICA IPv6 stack uses     */
/* IPPROTO_IP instead of IPPROTO_IPV6 in setsockopt calls      */
/* We also need to define in6addr_any */
#ifdef  MUSICA_IPV6
#define	IPPROTO_IPV6	IPPROTO_IP
struct in6_addr in6addr_any = { IN6ADDR_ANY_INIT };

/* These DEF's are required as MUSICA's winsock6.h causes a clash with some of the 
 * standard ws2tcpip.h definitions (eg struct in_addr6).
 * Note: winsock6.h defines AF_INET6 as 24 NOT 23 as in winsock2.h - I have left it
 * set to the MUSICA value as this is used in some of their function calls. 
 */
//#define AF_INET6        23
#define IP_MULTICAST_LOOP      11       /*set/get IP multicast loopback */
#define	IP_MULTICAST_IF		9       /* set/get IP multicast i/f  */
#define	IP_MULTICAST_TTL       10       /* set/get IP multicast ttl */
#define	IP_MULTICAST_LOOP      11       /*set/get IP multicast loopback */
#define	IP_ADD_MEMBERSHIP      12       /* add an IP group membership */
#define	IP_DROP_MEMBERSHIP     13       /* drop an IP group membership */

#define IN6_IS_ADDR_UNSPECIFIED(a) (((a)->s6_addr32[0] == 0) && \
									((a)->s6_addr32[1] == 0) && \
									((a)->s6_addr32[2] == 0) && \
									((a)->s6_addr32[3] == 0))
struct ip_mreq {
        struct in_addr imr_multiaddr;   /* IP multicast address of group */
        struct in_addr imr_interface;   /* local IP address of interface */
};
#endif

#ifndef INADDR_NONE
#define INADDR_NONE 0xffffffff
#endif

struct item {
    struct item *next;
    long size;
    uint8_t *buf;
};

struct _socket_udp {
        int mode;               /* IPv4 or IPv6 */
        char *addr;
        uint16_t rx_port;
        uint16_t tx_port;
        ttl_t ttl;
        fd_t fd;
        struct sockaddr_storage sock;
        socklen_t sock_len;
        bool multithreaded;

        // for multithreaded processing
        pthread_t thread_id;
        struct item queue[MAX_UDP_READER_QUEUE_LEN];
        struct item *queue_head;
        struct item *queue_tail;
        mutex lock;
        condition_variable boss_cv;
        condition_variable reader_cv;

        void *to_be_freed; /// Data that should be freed with free() (may have remained from killed thread)
#ifdef WIN32
        WSAOVERLAPPED *overlapped;
        WSAEVENT *overlapped_events;
        void **dispose_udata;
        bool overlapping_active;
        int overlapped_max;
        int overlapped_count;
#endif
};

static void udp_clean_async_state(socket_udp *s);

#ifdef WIN32
/* Want to use both Winsock 1 and 2 socket options, but since
* IPv6 support requires Winsock 2 we have to add own backwards
* compatibility for Winsock 1.
*/
#define SETSOCKOPT winsock_versions_setsockopt
#else
#define SETSOCKOPT setsockopt
#endif                          /* WIN32 */

#define GETSOCKOPT getsockopt

/*****************************************************************************/
/* Support functions...                                                      */
/*****************************************************************************/

void socket_error(const char *msg, ...)
{
        char buffer[255];
        uint32_t blen = sizeof(buffer) / sizeof(buffer[0]);
        va_list ap;

#ifdef WIN32
#define WSERR(x) {#x,x}
        struct wse {
                char errname[20];
                int errno_code;
        };
        struct wse ws_errs[] = {
                WSERR(WSANOTINITIALISED), WSERR(WSAENETDOWN), WSERR(WSAEACCES),
                WSERR(WSAEINVAL), WSERR(WSAEINTR), WSERR(WSAEINPROGRESS),
                WSERR(WSAEFAULT), WSERR(WSAENETRESET), WSERR(WSAENOBUFS),
                WSERR(WSAENOTCONN), WSERR(WSAENOTSOCK), WSERR(WSAEOPNOTSUPP),
                WSERR(WSAESHUTDOWN), WSERR(WSAEWOULDBLOCK), WSERR(WSAEMSGSIZE),
                WSERR(WSAEHOSTUNREACH), WSERR(WSAECONNABORTED),
                    WSERR(WSAECONNRESET),
                WSERR(WSAEADDRNOTAVAIL), WSERR(WSAEAFNOSUPPORT),
                    WSERR(WSAEDESTADDRREQ),
                WSERR(WSAENETUNREACH), WSERR(WSAETIMEDOUT), WSERR(0)
        };

        int i, e = WSAGetLastError();
        i = 0;
        while (ws_errs[i].errno_code && ws_errs[i].errno_code != e) {
                i++;
        }
        va_start(ap, msg);
        _vsnprintf(buffer, blen, msg, ap);
        va_end(ap);
        if (e != WSAECONNRESET)
                printf("ERROR: %s, (%d - %s)\n", msg, e, ws_errs[i].errname);
#else
        va_start(ap, msg);
        vsnprintf(buffer, blen, msg, ap);
        va_end(ap);
        perror(buffer);
#endif
}

#ifdef WIN32
#define socket_herror socket_error
#else
static void socket_herror(const char *msg, ...)
{
        va_list ap;
        char buffer[255];
        uint32_t blen = sizeof(buffer) / sizeof(buffer[0]);

        va_start(ap, msg);
        vsnprintf(buffer, blen, msg, ap);
        va_end(ap);
        herror(buffer);
}
#endif

#ifdef WIN32
/* ws2tcpip.h defines these constants with different values from
* winsock.h so files that use winsock 2 values but try to use 
* winsock 1 fail.  So what was the motivation in changing the
* constants ?
*/
#define WS1_IP_MULTICAST_IF     2       /* set/get IP multicast interface   */
#define WS1_IP_MULTICAST_TTL    3       /* set/get IP multicast timetolive  */
#define WS1_IP_MULTICAST_LOOP   4       /* set/get IP multicast loopback    */
#define WS1_IP_ADD_MEMBERSHIP   5       /* add  an IP group membership      */
#define WS1_IP_DROP_MEMBERSHIP  6       /* drop an IP group membership      */

/* winsock_versions_setsockopt tries 1 winsock version of option 
* optname and then winsock 2 version if that failed.
*/

static int
winsock_versions_setsockopt(SOCKET s, int level, int optname,
                            const char FAR * optval, int optlen)
{
        int success = -1;
        switch (optname) {
        case IP_MULTICAST_IF:
                success =
                    setsockopt(s, level, WS1_IP_MULTICAST_IF, optval, optlen);
                break;
        case IP_MULTICAST_TTL:
                success =
                    setsockopt(s, level, WS1_IP_MULTICAST_TTL, optval, optlen);
                break;
        case IP_MULTICAST_LOOP:
                success =
                    setsockopt(s, level, WS1_IP_MULTICAST_LOOP, optval, optlen);
                break;
        case IP_ADD_MEMBERSHIP:
                success =
                    setsockopt(s, level, WS1_IP_ADD_MEMBERSHIP, optval, optlen);
                break;
        case IP_DROP_MEMBERSHIP:
                success =
                    setsockopt(s, level, WS1_IP_DROP_MEMBERSHIP, optval,
                               optlen);
                break;
        }
        if (success != -1) {
                return success;
        }
        return setsockopt(s, level, optname, optval, optlen);
}
#endif

#ifdef NEED_INET_ATON
#ifdef NEED_INET_ATON_STATIC
static
#endif
int inet_aton(const char *name, struct in_addr *addr);

int inet_aton(const char *name, struct in_addr *addr)
{
        addr->s_addr = inet_addr(name);
        return (addr->s_addr != (in_addr_t) INADDR_NONE);
}
#endif

#ifdef NEED_IN6_IS_ADDR_MULTICAST
#define IN6_IS_ADDR_MULTICAST(addr) ((addr)->s6_addr[0] == 0xffU)
#endif

#if defined(NEED_IN6_IS_ADDR_UNSPECIFIED) && defined(MUSICA_IPV6)
#define IN6_IS_ADDR_UNSPECIFIED(addr) IS_UNSPEC_IN6_ADDR(*addr)
#endif

/*****************************************************************************/
/* IPv4 specific functions...                                                */
/*****************************************************************************/

static int udp_addr_valid4(const char *dst)
{
        struct in_addr addr4;
        struct hostent *h;

        if (inet_pton(AF_INET, dst, &addr4)) {
                return TRUE;
        }

        h = gethostbyname(dst);
        if (h != NULL) {
                return TRUE;
        }
        socket_herror("Can't resolve IP address for %s", dst);

        return FALSE;
}

static int udp_join_mcast_grp4(unsigned long addr, int fd, int ttl, unsigned int ifindex)
{
        if (IN_MULTICAST(ntohl(addr))) {
#ifndef WIN32
                char loop = 1;
#endif
                struct ip_mreq imr;

                imr.imr_multiaddr.s_addr = addr;
                imr.imr_interface.s_addr = ifindex;

                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ip_mreq)) != 0) {
                        socket_error("setsockopt IP_ADD_MEMBERSHIP");
                        return FALSE;
                }
#ifndef WIN32
                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop,
                     sizeof(loop)) != 0) {
                        socket_error("setsockopt IP_MULTICAST_LOOP");
                        return FALSE;
                }
#endif
                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_MULTICAST_TTL, (char *)&ttl,
                     sizeof(ttl)) != 0) {
                        socket_error("setsockopt IP_MULTICAST_TTL");
                        return FALSE;
                }
                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_MULTICAST_IF,
                     (char *)&ifindex, sizeof(ifindex)) != 0) {
                        socket_error("setsockopt IP_MULTICAST_IF");
                        return FALSE;
                }
        }
        return TRUE;
}

static void udp_leave_mcast_grp4(unsigned long addr, int fd)
{
        if (IN_MULTICAST(ntohl(addr))) {
                struct ip_mreq imr;
                imr.imr_multiaddr.s_addr = addr;
                imr.imr_interface.s_addr = INADDR_ANY;
                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ip_mreq)) != 0) {
                        socket_error("setsockopt IP_DROP_MEMBERSHIP");
                        abort();
                }
                debug_msg("Dropped membership of multicast group\n");
        }
}

static char *udp_host_addr4(void)
{
        char *hname = (char *) calloc(MAXHOSTNAMELEN + 1, 1);
        struct hostent *hent;
        struct in_addr iaddr;

        if (gethostname(hname, MAXHOSTNAMELEN) != 0) {
                debug_msg("Cannot get hostname!");
                abort();
        }
        hent = gethostbyname(hname);
        if (hent == NULL) {
                socket_herror("Can't resolve IP address for %s", hname);
                return NULL;
        }
        assert(hent->h_addrtype == AF_INET);
        memcpy(&iaddr.s_addr, hent->h_addr, sizeof(iaddr.s_addr));
        strncpy(hname, inet_ntoa(iaddr), MAXHOSTNAMELEN);
        return hname;
}

/*****************************************************************************/
/* IPv6 specific functions...                                                */
/*****************************************************************************/

static int udp_addr_valid6(const char *dst)
{
#ifdef HAVE_IPv6
        struct in6_addr addr6;
        switch (inet_pton(AF_INET6, dst, &addr6)) {
        case 1:
                return TRUE;
                break;
        case 0:
                return FALSE;
                break;
        case -1:
                debug_msg("inet_pton failed\n");
                errno = 0;
        }
#endif                          /* HAVE_IPv6 */
        UNUSED(dst);
        return FALSE;
}

static int udp_join_mcast_grp6(struct in6_addr sin6_addr, int fd, int ttl, unsigned int ifindex)
{
#ifdef HAVE_IPv6
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr)) {
                unsigned int loop = 1;
                struct ipv6_mreq imr;
#ifdef MUSICA_IPV6
                imr.i6mr_interface = 1;
                imr.i6mr_multiaddr = sin6_addr;
#else
                imr.ipv6mr_multiaddr = sin6_addr;
                imr.ipv6mr_interface = ifindex;
#endif

                if (SETSOCKOPT
                    (fd, IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ipv6_mreq)) != 0) {
                        socket_error("setsockopt IPV6_ADD_MEMBERSHIP");
                        return FALSE;
                }

                if (SETSOCKOPT
                    (fd, IPPROTO_IPV6, IPV6_MULTICAST_LOOP, (char *)&loop,
                     sizeof(loop)) != 0) {
                        socket_error("setsockopt IPV6_MULTICAST_LOOP");
                        return FALSE;
                }
                if (SETSOCKOPT
                    (fd, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, (char *)&ttl,
                     sizeof(ttl)) != 0) {
                        socket_error("setsockopt IPV6_MULTICAST_HOPS");
                        return FALSE;
                }
                if (SETSOCKOPT(fd, IPPROTO_IPV6, IPV6_MULTICAST_IF,
                                        (char *)&ifindex, sizeof(ifindex)) != 0) {
                        socket_error("setsockopt IPV6_MULTICAST_IF");
                        return FALSE;
                }
        }

        return TRUE;
#endif
}

static void udp_leave_mcast_grp6(struct in6_addr sin6_addr, int fd)
{
#ifdef HAVE_IPv6
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr)) {
                struct ipv6_mreq imr;
#ifdef MUSICA_IPV6
                imr.i6mr_interface = 1;
                imr.i6mr_multiaddr = sin6_addr;
#else
                imr.ipv6mr_multiaddr = sin6_addr;
                imr.ipv6mr_interface = 0;
#endif

                if (SETSOCKOPT
                    (fd, IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ipv6_mreq)) != 0) {
                        socket_error("setsockopt IPV6_DROP_MEMBERSHIP");
                        abort();
                }
        }
#else
        UNUSED(s);
#endif                          /* HAVE_IPv6 */
}

static char *udp_host_addr6(socket_udp * s)
{
#ifdef HAVE_IPv6
        char *hname = (char *) calloc(MAXHOSTNAMELEN + 1, 1);
        int gai_err, newsock;
        struct addrinfo hints, *ai;
        struct sockaddr_in6 local, addr6;
        socklen_t len = sizeof(local);
        int result = 0;

        newsock = socket(AF_INET6, SOCK_DGRAM, 0);
        if (newsock == -1) {
                perror("socket");
                free(hname);
                return NULL;
        }
        memset((char *)&addr6, 0, len);
        addr6.sin6_family = AF_INET6;
#ifdef HAVE_SIN6_LEN
        addr6.sin6_len = len;
#endif
        result = bind(newsock, (struct sockaddr *)&addr6, len);
        if (result != 0) {
                perror("Cannot bind");
        }
        addr6.sin6_addr = ((struct sockaddr_in6 *)&s->sock)->sin6_addr;
        addr6.sin6_port = htons(s->rx_port);
        addr6.sin6_scope_id = ((struct sockaddr_in6 *)&s->sock)->sin6_scope_id;
        result = connect(newsock, (struct sockaddr *)&addr6, len);
        if (result != 0) {
                perror("connect");
        }

        memset((char *)&local, 0, len);
        if ((result =
             getsockname(newsock, (struct sockaddr *)&local, &len)) < 0) {
                local.sin6_addr = in6addr_any;
                local.sin6_port = 0;
                error_msg("getsockname failed\n");
        }

        CLOSESOCKET(newsock);

        if (IN6_IS_ADDR_UNSPECIFIED(&local.sin6_addr)
            || IN6_IS_ADDR_MULTICAST(&local.sin6_addr)) {
                if (gethostname(hname, MAXHOSTNAMELEN) != 0) {
                        error_msg("gethostname failed\n");
                        return NULL;
                }

                hints.ai_protocol = 0;
                hints.ai_flags = 0;
                hints.ai_family = AF_INET6;
                hints.ai_socktype = SOCK_DGRAM;
                hints.ai_addrlen = 0;
                hints.ai_canonname = NULL;
                hints.ai_addr = NULL;
                hints.ai_next = NULL;

                if ((gai_err = getaddrinfo(hname, NULL, &hints, &ai))) {
                        error_msg("getaddrinfo: %s: %s\n", hname,
                                  gai_strerror(gai_err));
                        return NULL;
                }

                struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)(void *)
                        ai->ai_addr;
                if (inet_ntop
                    (AF_INET6,
                     &(addr6->sin6_addr),
                     hname, MAXHOSTNAMELEN) == NULL) {
                        error_msg("inet_ntop: %s: \n", hname);
                        freeaddrinfo(ai);
                        return NULL;
                }
                freeaddrinfo(ai);
                return hname;
        }
        if (inet_ntop(AF_INET6, &local.sin6_addr, hname, MAXHOSTNAMELEN) ==
            NULL) {
                error_msg("inet_ntop: %s: \n", hname);
                return NULL;
        }
        return hname;
#else                           /* HAVE_IPv6 */
        UNUSED(s);
        return "::";            /* The unspecified address... */
#endif                          /* HAVE_IPv6 */
}

/*****************************************************************************/
/* Generic functions, which call the appropriate protocol specific routines. */
/*****************************************************************************/

/**
 * udp_addr_valid:
 * @addr: string representation of IPv4 or IPv6 network address.
 *
 * Returns TRUE if @addr is valid, FALSE otherwise.
 **/

int udp_addr_valid(const char *addr)
{
        return udp_addr_valid4(addr) | udp_addr_valid6(addr);
}

/**
 * udp_init:
 * Creates a session for sending and receiving UDP datagrams over IP
 * networks. 
 *
 * @param addr    character string containing an IPv4 or IPv6 network address.
 * @param rx_port receive port.
 * @param tx_port transmit port.
 * @param ttl     time-to-live value for transmitted packets.
 *
 * Returns: a pointer to a valid socket_udp structure on success, NULL otherwise.
 **/
socket_udp *udp_init(const char *addr, uint16_t rx_port, uint16_t tx_port,
                     int ttl, bool use_ipv6, bool multithreaded)
{
        return udp_init_if(addr, NULL, rx_port, tx_port, ttl, use_ipv6, multithreaded);
}

// we assume that only colon-separated address are IPv6
static bool address_is_ipv6(const char *addr)
{
        return strchr(addr, ':') != NULL;
}

/**
 * udp_init_if:
 * Creates a session for sending and receiving UDP datagrams over IP
 * networks.  The session uses @iface as the interface to send and
 * receive datagrams on.
 *
 * @param addr    character string containing an IPv4 or IPv6 network address.
 * @param iface   character string containing an interface name. If NULL, default is used.
 * @param rx_port receive port.
 * @param tx_port transmit port.
 * @param ttl     time-to-live value for transmitted packets.
 * @param use_ipv6     whether to use IPv6 for hostname
 * @param multithreaded receiving in a separate thread than processing
 *
 * @returns a pointer to a socket_udp structure on success, NULL otherwise.
 **/
socket_udp *udp_init_if(const char *addr, const char *iface, uint16_t rx_port,
                        uint16_t tx_port, int ttl, bool use_ipv6, bool multithreaded)
{
        int ret;
        int ip_family;
        int reuse = 1;
        int ipv6only = 0;
        struct sockaddr_storage s_in{};
        socklen_t sin_len;
        unsigned int ifindex;
        socket_udp *s = new socket_udp();
        s->addr = NULL;
        s->rx_port = rx_port;
        s->tx_port = tx_port;
        s->ttl = ttl;
        s->fd = INVALID_SOCKET;

	if (!address_is_ipv6(addr) && !use_ipv6) {
                s->mode = IPv4;
                ip_family = AF_INET;
	} else {
#ifdef HAVE_IPv6
                s->mode = IPv6;
                ip_family = AF_INET6;
#else
                fprintf(stderr, "IPv6 support not compiled in!\n");
                delete s;
                return NULL;
#endif
	}

        if ((ret = resolve_address(s, addr)) != 0) {
                fprintf(stderr, "Can't resolve IP address for %s: %s\n", addr,
                                gai_strerror(ret));
                goto error;
        }
        if (iface != NULL) {
#ifdef HAVE_IF_NAMETOINDEX
                if ((ifindex = if_nametoindex(iface)) == 0) {
                        debug_msg("Illegal interface specification\n");
                        goto error;
                }
#else
                fprintf(stderr, "Cannot set interface name, if_nametoindex not supported.\n");
#endif
        } else {
                ifindex = 0;
        }
#ifdef WIN32
        s->fd = WSASocket(ip_family, SOCK_DGRAM, IPPROTO_UDP, NULL, 0, WSA_FLAG_OVERLAPPED);
#else
        s->fd = socket(ip_family, SOCK_DGRAM, 0);
#endif
        if (s->fd == INVALID_SOCKET) {
                socket_error("Unable to initialize socket");
                goto error;
        }
        if (s->mode == IPv6) {
                if (SETSOCKOPT(s->fd, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&ipv6only,
                                        sizeof(ipv6only)) != 0) {
                        socket_error("setsockopt IPV6_V6ONLY");
                        goto error;
                }
        }
#ifdef SO_REUSEPORT
        if (SETSOCKOPT
            (s->fd, SOL_SOCKET, SO_REUSEPORT, (int *)&reuse,
             sizeof(reuse)) != 0) {
                socket_error("setsockopt SO_REUSEPORT");
                goto error;
        }
#endif
        if (SETSOCKOPT
            (s->fd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse,
             sizeof(reuse)) != 0) {
                socket_error("setsockopt SO_REUSEADDR");
                goto error;
        }
        if (s->mode == IPv4) {
                struct sockaddr_in *s_in4 = (struct sockaddr_in *) &s_in;
                s_in4->sin_family = AF_INET;
                s_in4->sin_addr.s_addr = htonl(INADDR_ANY);
                s_in4->sin_port = htons(rx_port);
                sin_len = sizeof(struct sockaddr_in);
        } else if (s->mode == IPv6) {
#ifdef HAVE_IPv6
                struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *) &s_in;
                s_in6->sin6_family = AF_INET6;
                s_in6->sin6_addr = in6addr_any;
                s_in6->sin6_port = htons(rx_port);
#ifdef HAVE_SIN6_LEN
                s_in6->sin6_len = sizeof(struct sockaddr_in6);
#endif
                sin_len = sizeof(struct sockaddr_in6);
#endif
        } else {
                abort();
        }
        if (bind(s->fd, (struct sockaddr *)&s_in, sin_len) != 0) {
                socket_error("bind");
#ifdef WIN32
                fprintf(stderr, "Check if there is no service running on UDP port %d. ", rx_port);
                if (rx_port == 5004 || rx_port == 5005)
                        fprintf(stderr, "Windows Media Services is usually a good candidate to check and disable.\n");
#endif
                goto error;
        }

        // if we do not set tx port, fake that is the same as we are bound to
        if (s->tx_port == 0) {
                struct sockaddr_storage sin;
                socklen_t addrlen = sizeof(sin);
                if (getsockname(s->fd, (struct sockaddr *)&sin, &addrlen) == 0 &&
                                sin.ss_family == ip_family) {
                        if (s->mode == IPv4) {
                                struct sockaddr_in *s_in4 = (struct sockaddr_in *) &sin;
                                s->tx_port = ntohs(s_in4->sin_port);
                                ((struct sockaddr_in *) &s->sock)->sin_port = s_in4->sin_port;
                        } else if (s->mode == IPv6) {
#ifdef HAVE_IPv6
                                struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *) &s_in;
                                s->tx_port = ntohs(s_in6->sin6_port);
                                ((struct sockaddr_in6 *) &s->sock)->sin6_port = s_in6->sin6_port;
#endif
                        } else {
                                abort();
                        }
                }
        }

        switch (s->mode) {
        case IPv4:
                if (!udp_join_mcast_grp4(((struct sockaddr_in *)&s->sock)->sin_addr.s_addr, s->fd, ttl, ifindex)) {
                        goto error;
                }
                break;
        case IPv6:
                if (!udp_join_mcast_grp6(((struct sockaddr_in6 *)&s->sock)->sin6_addr, s->fd, ttl, ifindex)) {
                        goto error;
                }
                break;
        default:
                abort();
        }

        s->addr = strdup(addr);

        s->multithreaded = multithreaded;
        if (multithreaded) {
                for (int i = 0; i < MAX_UDP_READER_QUEUE_LEN; i++)
                        s->queue[i].next = s->queue + i + 1;
                s->queue[MAX_UDP_READER_QUEUE_LEN - 1].next = s->queue;
                s->queue_head = s->queue_tail = s->queue;

                pthread_create(&s->thread_id, NULL, udp_reader, s);
        }

        return s;

error:
        if (s->fd != INVALID_SOCKET) {
                CLOSESOCKET(s->fd);
        }
        delete s;
        return NULL;
}

static fd_set rfd;
static fd_t max_fd;

/**
 * udp_fd_zero:
 * 
 * Clears file descriptor from set associated with UDP sessions (see select(2)).
 * 
 **/
void udp_fd_zero(void)
{
        FD_ZERO(&rfd);
        max_fd = 0;
}

void udp_fd_zero_r(struct udp_fd_r *fd_struct)
{
        FD_ZERO(&fd_struct->rfd);
        fd_struct->max_fd = 0;
}

/**
 * udp_exit:
 * @s: UDP session to be terminated.
 *
 * Closes UDP session.
 * 
 **/
void udp_exit(socket_udp * s)
{
        switch (s->mode) {
        case IPv4:
                udp_leave_mcast_grp4(((struct sockaddr_in *)&s->sock)->sin_addr.s_addr, s->fd);
                break;
        case IPv6:
                udp_leave_mcast_grp6(((struct sockaddr_in6 *)&s->sock)->sin6_addr, s->fd);
                break;
        default:
                abort();
        }

        if (s->multithreaded) {
                pthread_cancel(s->thread_id);
#ifdef WIN32
                closesocket(s->fd);
#endif
                pthread_join(s->thread_id, NULL);
                while (s->queue_tail != s->queue_head) {
                        free(s->queue_tail->buf);
                        s->queue_tail = s->queue_tail->next;
                }
        }

        udp_clean_async_state(s);

        // close socket if haven't been already closed (either not on Winows or not multithreaded)
#ifdef WIN32
        if (!s->multithreaded)
#endif
                CLOSESOCKET(s->fd);

        free(s->addr);
        free(s->to_be_freed);
        delete s;
}

/**
 * udp_send:
 * @s: UDP session.
 * @buffer: pointer to buffer to be transmitted.
 * @buflen: length of @buffer.
 * 
 * Transmits a UDP datagram containing data from @buffer.
 * 
 * Return value: 0 on success, -1 on failure.
 **/
int udp_send(socket_udp * s, char *buffer, int buflen)
{
        assert(s != NULL);
        assert(buffer != NULL);
        assert(buflen > 0);

        return sendto(s->fd, buffer, buflen, 0, (struct sockaddr *)&s->sock,
                      s->sock_len);
}

int udp_sendto(socket_udp * s, char *buffer, int buflen, struct sockaddr *dst_addr, socklen_t addrlen)
{
        return sendto(s->fd, buffer, buflen, 0, dst_addr, addrlen);
}

#ifdef WIN32
int udp_sendv(socket_udp * s, LPWSABUF vector, int count, void *d)
{
        assert(s != NULL);

        assert(!s->overlapping_active || s->overlapped_count < s->overlapped_max);

	DWORD bytesSent;
	int ret = WSASendTo(s->fd, vector, count, &bytesSent, 0,
		(struct sockaddr *) &s->sock,
		s->sock_len, s->overlapping_active ? &s->overlapped[s->overlapped_count] : NULL, NULL);
        if (s->overlapping_active) {
                s->dispose_udata[s->overlapped_count] = d;
        } else {
                free(d);
        }
        s->overlapped_count++;
        if (ret == 0 || WSAGetLastError() == WSA_IO_PENDING)
                return 0;
        else {
                return ret;
        }
}
#else
int udp_sendv(socket_udp * s, struct iovec *vector, int count, void *d)
{
        struct msghdr msg;

        assert(s != NULL);

        msg.msg_name = (void *) & s->sock;
        msg.msg_namelen = s->sock_len;
        msg.msg_iov = vector;
        msg.msg_iovlen = count;
        /* Linux needs these... solaris does something different... */
        msg.msg_control = 0;
        msg.msg_controllen = 0;
        msg.msg_flags = 0;

        int ret = sendmsg(s->fd, &msg, 0);
        free(d);
        return ret;
}
#endif // WIN32

/**
 * When receiving data in separate thread, this function fetches data
 * from socket and puts it in queue.
 *
 * @todo
 * Remove ugly thread cancellation stuff.
 */
static void *udp_reader(void *arg)
{
        socket_udp *s = (socket_udp *) arg;

        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

        uint8_t *packet = (uint8_t *) malloc(RTP_MAX_PACKET_LEN);
        uint8_t *buffer = ((uint8_t *) packet) + RTP_PACKET_HEADER_SIZE;
        s->to_be_freed = packet;

        while (1) {
                pthread_testcancel();
                int size = recvfrom(s->fd, (char *) buffer,
                                RTP_MAX_PACKET_LEN - RTP_PACKET_HEADER_SIZE,
                                0, 0, 0);

                if (size <= 0) {
                        /// @todo
                        /// In MSW, this block is called as often as packet is sent if
                        /// we got WSAECONNRESET error (noone is listening). This can have
                        /// negative performance impact.
                        socket_error("recvfrom");
                        pthread_testcancel();
                        continue;
                }

                unique_lock<mutex> lk(s->lock);
                s->reader_cv.wait(lk, [s]{return s->queue_head->next != s->queue_tail;});

                s->queue_head->size = size;
                s->queue_head->buf = packet;
                s->queue_head = s->queue_head->next;
                s->to_be_freed = nullptr;

                lk.unlock();
                s->boss_cv.notify_one();

                packet = (uint8_t *) malloc(RTP_MAX_PACKET_LEN);
                buffer = ((uint8_t *) packet) + RTP_PACKET_HEADER_SIZE;
                s->to_be_freed = packet;
        }

        return NULL;
}

static int udp_do_recv(socket_udp * s, char *buffer, int buflen, int flags, struct sockaddr *src_addr, socklen_t *addrlen)
{
        /* Reads data into the buffer, returning the number of bytes read.   */
        /* If no data is available, this returns the value zero immediately. */
        /* Note: since we don't care about the source address of the packet  */
        /* we receive, this function becomes protocol independent.           */
        int len;

        assert(buffer != NULL);
        assert(buflen > 0);

        len = recvfrom(s->fd, buffer, buflen, flags, src_addr, addrlen);
        if (len > 0) {
                return len;
        }
        if (errno != ECONNREFUSED) {
                socket_error("recvfrom");
        }
        return 0;
}

/**
 * udp_peek:
 * @s: UDP session.
 * @buffer: buffer to read data into.
 * @buflen: length of @buffer.
 * 
 * Reads from datagram queue associated with UDP session. The data is left
 * in the queue, for future reading using udp_recv().
 *
 * Return value: number of bytes read, returns 0 if no data is available.
 **/
int udp_peek(socket_udp * s, char *buffer, int buflen)
{
        return udp_do_recv(s, buffer, buflen, MSG_PEEK, 0, 0);
}

/**
 * @brief
 * Checks whether there are data ready to read in socket
 *
 * @note
 * Designed only for multithreaded socket!
 */
bool udp_not_empty(socket_udp * s, struct timeval *timeout)
{
        assert(s->multithreaded);

        unique_lock<mutex> lk(s->lock);
        if (timeout) {
                std::chrono::microseconds tmout_us =
                        std::chrono::microseconds(timeout->tv_sec * 1000000ll + timeout->tv_usec);
                s->boss_cv.wait_for(lk, tmout_us, [s]{return s->queue_head != s->queue_tail;});
        } else {
                s->boss_cv.wait(lk, [s]{return s->queue_head != s->queue_tail;});
        }
        return (s->queue_head != s->queue_tail);
}

/**
 * udp_recv:
 * @s: UDP session.
 * @buffer: buffer to read data into.
 * @buflen: length of @buffer.
 * 
 * Reads from datagram queue associated with UDP session.
 *
 * Return value: number of bytes read, returns 0 if no data is available.
 **/
int udp_recv(socket_udp * s, char *buffer, int buflen)
{
        /* Reads data into the buffer, returning the number of bytes read.   */
        /* If no data is available, this returns the value zero immediately. */
        /* Note: since we don't care about the source address of the packet  */
        /* we receive, this function becomes protocol independent.           */
        return udp_do_recv(s, buffer, buflen, 0, 0, 0);
}

int udp_recvfrom(socket_udp *s, char *buffer, int buflen, struct sockaddr *src_addr, socklen_t *addrlen)
{
        return udp_do_recv(s, buffer, buflen, 0, src_addr, addrlen);
}

/**
 * Receives data from multithreaded socket.
 *
 * @param[in] s       UDP socket state
 * @param[out] buffer data received from socket. Must be freed by caller!
 * @returns           length of the received datagram
 */
int udp_recv_data(socket_udp * s, char **buffer)
{
        assert(s->multithreaded);
        int ret;
        unique_lock<mutex> lk(s->lock);

        *buffer = (char *) s->queue_tail->buf;
        ret = s->queue_tail->size;
        s->queue_tail->buf = NULL;
        s->queue_tail = s->queue_tail->next;
        lk.unlock();
        s->reader_cv.notify_one();

        return ret;
}

#ifndef WIN32
int udp_recvv(socket_udp * s, struct msghdr *m)
{
        if (recvmsg(s->fd, m, 0) == -1) {
                perror("recvmsg");
                return 1;
        }
        return 0;
}
#endif // WIN32

/**
 * udp_fd_set:
 * @s: UDP session.
 * 
 * Adds file descriptor associated of @s to set associated with UDP sessions.
 *
 * @deprecated Use thread-safe udp_fd_set_r() instead.
 **/
void udp_fd_set(socket_udp * s)
{
        FD_SET(s->fd, &rfd);
        if (s->fd > (fd_t) max_fd) {
                max_fd = s->fd;
        }
}

void udp_fd_set_r(socket_udp *s, struct udp_fd_r *fd_struct)
{
        FD_SET(s->fd, &fd_struct->rfd);
        if (s->fd > (fd_t) fd_struct->max_fd) {
                fd_struct->max_fd = s->fd;
        }
}

/**
 * udp_fd_isset:
 * @s: UDP session.
 * 
 * Checks if file descriptor associated with UDP session is ready for
 * reading.  This function should be called after udp_select().
 *
 * Returns: non-zero if set, zero otherwise.
 *
 * @deprecated Use thread-safe udp_fd_set_r() instead.
 **/
int udp_fd_isset(socket_udp * s)
{
        return FD_ISSET(s->fd, &rfd);
}

int udp_fd_isset_r(socket_udp *s, struct udp_fd_r *fd_struct)
{
        return FD_ISSET(s->fd, &fd_struct->rfd);
}


/**
 * udp_select:
 * @timeout: maximum period to wait for data to arrive.
 * 
 * Waits for data to arrive for UDP sessions.
 * 
 * Return value: number of UDP sessions ready for reading.
 **/
int udp_select(struct timeval *timeout)
{
        return select(max_fd + 1, &rfd, NULL, NULL, timeout);
}

int udp_select_r(struct timeval *timeout, struct udp_fd_r * fd_struct)
{
        return select(fd_struct->max_fd + 1, &fd_struct->rfd, NULL, NULL, timeout);
}

/**
 * udp_host_addr:
 * @s: UDP session.
 * 
 * Return value: character string containing network address
 * associated with session @s. Returned value needs to be freed by caller.
 **/
char *udp_host_addr(socket_udp * s)
{
        switch (s->mode) {
        case IPv4:
                return udp_host_addr4();
        case IPv6:
                return udp_host_addr6(s);
        default:
                abort();
        }
        return NULL;
}

/**
 * udp_fd:
 * @s: UDP session.
 * 
 * This function allows applications to apply their own socketopt()'s
 * and ioctl()'s to the UDP session.
 * 
 * Return value: file descriptor of socket used by session @s.
 **/
int udp_fd(socket_udp * s)
{
        if (s && s->fd > 0) {
                return s->fd;
        }
        return 0;
}

static int resolve_address(socket_udp *s, const char *addr)
{
        struct addrinfo hints, *res0;
        int err;

        memset(&hints, 0, sizeof(hints));
        switch (s->mode) {
        case IPv4:
                hints.ai_family = AF_INET;
                break;
        case IPv6:
                hints.ai_family = AF_INET6;
                break;
        default:
                abort();
        }
        hints.ai_socktype = SOCK_DGRAM;

        char tx_port_str[7];
        sprintf(tx_port_str, "%u", s->tx_port);
        if ((err = getaddrinfo(addr, tx_port_str, &hints, &res0)) != 0) {
                /* We should probably try to do a DNS lookup on the name */
                /* here, but I'm trying to get the basics going first... */
                return err;
        } else {
                memcpy(&s->sock, res0->ai_addr, res0->ai_addrlen);
                s->sock_len = res0->ai_addrlen;
        }
        freeaddrinfo(res0);

        return 0;
}

int udp_set_recv_buf(socket_udp *s, int size)
{
        int opt = 0;
        socklen_t opt_size;
        if(SETSOCKOPT (s->fd, SOL_SOCKET, SO_RCVBUF, (const sockopt_t)&size,
                        sizeof(size)) != 0) {
                perror("Unable to set socket buffer size");
                return FALSE;
        }

        opt_size = sizeof(opt);
        if(GETSOCKOPT (s->fd, SOL_SOCKET, SO_RCVBUF, (sockopt_t)&opt,
                        &opt_size) != 0) {
                perror("Unable to get socket buffer size");
                return FALSE;
        }

        if(opt < size) {
                return FALSE;
        }

        debug_msg("Socket buffer size set to %d B.\n", opt);

        return TRUE;
}

int udp_set_send_buf(socket_udp *s, int size)
{
        int opt = 0;
        socklen_t opt_size;
        if(SETSOCKOPT (s->fd, SOL_SOCKET, SO_SNDBUF, (const sockopt_t)&size,
                        sizeof(size)) != 0) {
                perror("Unable to set socket buffer size");
                return FALSE;
        }

        opt_size = sizeof(opt);
        if(GETSOCKOPT (s->fd, SOL_SOCKET, SO_SNDBUF, (sockopt_t)&opt,
                        &opt_size) != 0) {
                perror("Unable to get socket buffer size");
                return FALSE;
        }

        if(opt < size) {
                return FALSE;
        }

        debug_msg("Socket buffer size set to %d B.\n", opt);

        return TRUE;
}

/*
 * TODO: This should be definitely removed. We need to solve audio burst avoidance first.
 */
void udp_flush_recv_buf(socket_udp *s)
{
        const int len = 1024 * 1024;
        fd_set select_fd;
        struct timeval timeout;

        char *buf = (char *) malloc(len);

        assert (buf != NULL);

        timeout.tv_sec = 0;
        timeout.tv_usec = 0;

        FD_ZERO(&select_fd);
        FD_SET(s->fd, &select_fd);

        while(select(s->fd + 1, &select_fd, NULL, NULL, &timeout) > 0) {
                ssize_t bytes = recv(s->fd, buf, len, 0);
                if(bytes <= 0)
                        break;
        }

        free(buf);
}

/**
 * By calling this function under MSW, caller indicates that following packets
 * can be send in asynchronous manner. Caller should then call udp_async_wait()
 * to ensure that all packets were actually sent.
 */
void udp_async_start(socket_udp *s, int nr_packets)
{
#ifdef WIN32
        if (nr_packets > s->overlapped_max) {
                s->overlapped = (OVERLAPPED *) realloc(s->overlapped, nr_packets * sizeof(WSAOVERLAPPED));
                s->overlapped_events = (void **) realloc(s->overlapped_events, nr_packets * sizeof(WSAEVENT));
                s->dispose_udata = (void **) realloc(s->dispose_udata, nr_packets * sizeof(void *));
                for (int i = s->overlapped_max; i < nr_packets; ++i) {
                        memset(&s->overlapped[i], 0, sizeof(WSAOVERLAPPED));
                        s->overlapped[i].hEvent = s->overlapped_events[i] = WSACreateEvent();
                        assert(s->overlapped[i].hEvent != WSA_INVALID_EVENT);
                }
                s->overlapped_max = nr_packets;
        }

        s->overlapped_count = 0;
        s->overlapping_active = true;
#else
        UNUSED(nr_packets);
        UNUSED(s);
#endif
}

void udp_async_wait(socket_udp *s)
{
#ifdef WIN32
        if (!s->overlapping_active)
                return;
        for(int i = 0; i < s->overlapped_count; i += WSA_MAXIMUM_WAIT_EVENTS)
        {
                int count = WSA_MAXIMUM_WAIT_EVENTS;
                if (s->overlapped_count - i < WSA_MAXIMUM_WAIT_EVENTS)
                        count = s->overlapped_count - i;
                DWORD ret = WSAWaitForMultipleEvents(count, s->overlapped_events + i, TRUE, INFINITE, TRUE);
                if (ret == WSA_WAIT_FAILED) {
                        socket_error("WSAWaitForMultipleEvents");
                }
        }
        for (int i = 0; i < s->overlapped_count; i++) {
                if (WSAResetEvent(s->overlapped[i].hEvent) == FALSE) {
                        socket_error("WSAResetEvent");
                }
                free(s->dispose_udata[i]);
        }
        s->overlapping_active = false;
#else
        UNUSED(s);
#endif
}

static void udp_clean_async_state(socket_udp *s)
{
#ifdef WIN32
        for (int i = 0; i < s->overlapped_max; i++) {
                WSACloseEvent(s->overlapped[i].hEvent);
        }
        free(s->overlapped);
        free(s->overlapped_events);
        free(s->dispose_udata);
#else
        UNUSED(s);
#endif
}

bool udp_is_ipv6(socket_udp *s)
{
        return s->mode == IPv6;
}

bool udp_port_pair_is_free(const char *addr, bool use_ipv6, int even_port)
{
        bool ipv6;
	if (!address_is_ipv6(addr) && !use_ipv6) {
                ipv6 = false;
        } else {
                ipv6 = true;
        }

        for (int i = 0; i < 2; ++i) {
                fd_t fd;

                struct sockaddr_storage s_st = {};
                socklen_t len;

                if (ipv6) {
                        struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *) &s_st;
                        int ipv6only = 0;
                        s_in6->sin6_family = AF_INET6;
                        s_in6->sin6_port = htons(even_port + i);
#ifdef HAVE_SIN6_LEN
                        s_in6->sin6_len = sizeof(s_in);
#endif
                        s_in6->sin6_addr = in6addr_any;
                        len = sizeof(struct sockaddr_in6);
                        fd = socket(AF_INET6, SOCK_DGRAM, 0);
                        if (fd != INVALID_SOCKET) {
                                if (SETSOCKOPT
                                                (fd, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&ipv6only,
                                                 sizeof(ipv6only)) != 0) {
                                        socket_error("setsockopt IPV6_V6ONLY");
                                        CLOSESOCKET(fd);
                                        return false;
                                }
                        }
                } else {
                        struct sockaddr_in *s_in = (struct sockaddr_in *) &s_st;
                        s_in->sin_family = AF_INET;
                        s_in->sin_addr.s_addr = INADDR_ANY;
                        s_in->sin_port = htons(even_port + i);
                        len = sizeof(struct sockaddr_in);
                        fd = socket(AF_INET, SOCK_DGRAM, 0);
                }

                if (fd == INVALID_SOCKET) {
                        socket_error("Unable to initialize socket");
                        return false;
                }

                if (bind(fd, (struct sockaddr *)&s_st, len) != 0) {
                        CLOSESOCKET(fd);
                        return false;
                }

                CLOSESOCKET(fd);
        }
        return true;
}

#ifdef WIN32
/**
 * This function sends buffer asynchronously with provided Winsock parameters.
 *
 * This is different approach from udp_async_start() and udp_async_wait() because
 * this function lets caller provide parameters and passes them directly.
 */
int udp_send_wsa_async(socket_udp *s, char *buffer, int buflen, LPWSAOVERLAPPED_COMPLETION_ROUTINE l, LPWSAOVERLAPPED o)
{
        assert(s != NULL);

        DWORD bytesSent;
        WSABUF vector;
        vector.buf = buffer;
        vector.len = buflen;
        int ret = WSASendTo(s->fd, &vector, 1, &bytesSent, 0,
                        (struct sockaddr *) &s->sock,
                        s->sock_len, o, l);
        if (ret == 0 || WSAGetLastError() == WSA_IO_PENDING)
                return 0;
        else {
                return ret;
        }
}
#endif

/**
 * Tries to receive data from socket and if empty, waits at most specified
 * amount of time.
 */
int udp_recv_timeout(socket_udp *s, char *buffer, int buflen, struct timeval *timeout)
{
        struct udp_fd_r fd;
        int len = 0;

        if (s->multithreaded) {
                if (udp_not_empty(s, timeout)) {
                        char *data = NULL;
                        len = udp_recv_data(s, (char **) &data);
                        if (len > 0) {
                                memcpy(buffer, data, len);
                        }
                        free(data);
                }
        } else {
                udp_fd_zero_r(&fd);
                udp_fd_set_r(s, &fd);
                if (udp_select_r(timeout, &fd) > 0) {
                        if (udp_fd_isset_r(s, &fd)) {
                                len = udp_recv(s, buffer, buflen);
                                if (len < 0) {
                                        len = 0;
                                }
                        }
                }
        }
        return len;
}


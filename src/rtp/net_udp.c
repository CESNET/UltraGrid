/*
 * FILE:     net_udp.c
 * AUTHOR:   Colin Perkins 
 * MODIFIED: Orion Hodson & Piers O'Hanlon
 *           David Cassany   <david.cassany@i2cat.net>
 *           Gerard Castillo <gerard.castillo@i2cat.net>
 *           Martin Pulec    <pulec@cesnet.cz>
 * 
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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

#include <pthread.h>
#include <stdalign.h>
#include <stdint.h>       // for uint16_t, uintmax_t

#ifndef _WIN32
#include <ifaddrs.h>
#endif

#include "debug.h"
#include "host.h"
#include "memory.h"
#include "compat/platform_pipe.h"
#include "compat/vsnprintf.h"
#include "net_udp.h"
#include "rtp.h"
#include "utils/list.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/net.h"
#include "utils/thread.h"
#include "utils/windows.h"

#ifdef NEED_ADDRINFO_H
#include "addrinfo.h"
#endif

#define DEFAULT_MAX_UDP_READER_QUEUE_LEN (1920/3*8*1080/1152) //< 10-bit FullHD frame divided by 1280 MTU packets (minus headers)

static unsigned get_ifindex(const char *iface);
static int resolve_address(socket_udp *s, const char *addr, uint16_t tx_port);
static void *udp_reader(void *arg);
static void     udp_leave_mcast_grp4(unsigned long addr, int fd,
                                     const char iface[]);

#define IPv4	4
#define IPv6	6

#define ERRBUF_SIZE 255
#define MOD_NAME "[RTP UDP] "

#ifdef WIN2K_IPV6
const struct in6_addr in6addr_any = { IN6ADDR_ANY_INIT };
#endif

#ifdef _WIN32
typedef char *sockopt_t;
#else
typedef void *sockopt_t;
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
    uint8_t *buf;
    int size;
    struct sockaddr *src_addr;
    socklen_t addrlen;
};

#define ALIGNED_SOCKADDR_STORAGE_OFF ((RTP_MAX_PACKET_LEN + alignof(struct sockaddr_storage) - 1) / alignof(struct sockaddr_storage) * alignof(struct sockaddr_storage))
#define ALIGNED_ITEM_OFF (((ALIGNED_SOCKADDR_STORAGE_OFF + sizeof(struct sockaddr_storage)) + alignof(struct item) - 1) / alignof(struct item) * alignof(struct item))

/*
 * Local part of the socket
 *
 * Can be shared across multiple RTP sessions.
 */
struct socket_udp_local {
        int mode;               /* IPv4 or IPv6 */
        fd_t rx_fd;
        fd_t tx_fd;
        bool multithreaded;
#ifdef _WIN32
        bool is_wsa_overlapped;
#endif

        // for multithreaded receiving
        pthread_t thread_id;
        struct simple_linked_list *packets;
        unsigned int max_packets;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        pthread_cond_t reader_cv;

        bool should_exit;
        fd_t should_exit_fd[2];
};

/*
 * Complete socket including remote host
 */
struct _socket_udp {
        struct sockaddr_storage sock;
        socklen_t sock_len;
        char iface[IF_NAMESIZE]; ///< iface for multicast

        struct socket_udp_local *local;
        bool local_is_slave; // whether is the local

#ifdef _WIN32
        WSAOVERLAPPED *overlapped;
        WSAEVENT *overlapped_events;
        void **dispose_udata;
        bool overlapping_active;
        int overlapped_max;
        int overlapped_count;
#endif
};

static void udp_clean_async_state(socket_udp *s);

#ifdef _WIN32
/* Want to use both Winsock 1 and 2 socket options, but since
* IPv6 support requires Winsock 2 we have to add own backwards
* compatibility for Winsock 1.
*/
#define SETSOCKOPT winsock_versions_setsockopt
#else
#define SETSOCKOPT setsockopt
#endif                          /* _WIN32 */

#define GETSOCKOPT getsockopt

/*****************************************************************************/
/* Support functions...                                                      */
/*****************************************************************************/

void socket_error(const char *msg, ...)
{
        va_list ap;
        char buffer[ERRBUF_SIZE] = "";

#ifdef _WIN32
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
                WSERR(WSAECONNRESET), WSERR(WSAEADDRINUSE),
                WSERR(WSAEADDRNOTAVAIL), WSERR(WSAEAFNOSUPPORT),
                    WSERR(WSAEDESTADDRREQ),
                WSERR(WSAENETUNREACH), WSERR(WSAETIMEDOUT), WSERR(WSAENOPROTOOPT),
                WSERR(0)
        };

        int i = 0;
        int e = WSAGetLastError();
        if (e == WSAECONNRESET) {
                return;
        }
        while (ws_errs[i].errno_code && ws_errs[i].errno_code != e) {
                i++;
        }
        va_start(ap, msg);
        _vsnprintf(buffer, sizeof buffer, msg, ap);
        va_end(ap);

        const char *errname = ws_errs[i].errno_code == 0 ? get_win32_error(e)
                                                         : ws_errs[i].errname;
        log_msg(LOG_LEVEL_ERROR, "ERROR: %s, (%d - %s)\n", buffer, e, errname);
#else
        char strerror_buf[ERRBUF_SIZE] = "unknown";
        va_start(ap, msg);
        vsnprintf(buffer, sizeof buffer, msg, ap);
        va_end(ap);
#if ! defined _POSIX_C_SOURCE || (_POSIX_C_SOURCE >= 200112L && !  _GNU_SOURCE)
        strerror_r(errno, strerror_buf, sizeof strerror_buf); // XSI version
        log_msg(LOG_LEVEL_ERROR, "%s: %s\n", buffer, strerror_buf);
#else // GNU strerror_r version
        log_msg(LOG_LEVEL_ERROR, "%s: %s\n", buffer, strerror_r(errno, strerror_buf, sizeof strerror_buf));
#endif
#endif
}

#ifdef _WIN32
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

#ifdef _WIN32
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

static bool udp_addr_valid4(const char *dst)
{
        struct in_addr addr4;
        struct hostent *h;

        if (inet_pton(AF_INET, dst, &addr4)) {
                return true;
        }

        h = gethostbyname(dst);
        if (h != NULL) {
                return true;
        }
        socket_herror("Can't resolve IP address for %s", dst);

        return false;
}

/**
 * @returns 1. 0 (htonl(INADDR_ANY)) if iface is "";
 *          2. representation of the address if iface is an adress
 *          3a. [Windows only] interface index
 *          3b. [otherwise] iface local address
 *          4a. [error] htonl(INADDR_ANY) if no IPv4 address on iface
 *          4b. [error] ((unsigned) -1) on wrong iface spec or help
 */
static in_addr_t
get_iface_local_addr4(const char iface[])
{
        if (iface[0] == '\0') {
                return htonl(INADDR_ANY);
        }

        // address specified by bind address
        struct in_addr iface_addr;
        if (inet_pton(AF_INET, iface, &iface_addr) == 1) {
                return iface_addr.s_addr;
        }

        unsigned int ifindex = get_ifindex(iface);
        if (ifindex == (unsigned) -1) {
                return ifindex;
        }

#ifdef _WIN32
        // Windows allow the interface specification by ifindex
        return htonl(ifindex);
#else
        struct ifaddrs *a        = NULL;
        getifaddrs(&a);
        struct ifaddrs *p = a;
        while (NULL != p) {
                if (p->ifa_addr != NULL && p->ifa_addr->sa_family == AF_INET &&
                    strcmp(iface, p->ifa_name) == 0) {
                        struct sockaddr_in *sin = (void *) p->ifa_addr;
                        in_addr_t ret = sin->sin_addr.s_addr;
                        freeifaddrs(a);
                        return ret;
                }
                p = p->ifa_next;
        }
        freeifaddrs(a);
        MSG(ERROR, "Interface %s has assigned no IPv4 address!\n", iface);
        return htonl(INADDR_ANY);
#endif
}

static bool
udp_join_mcast_grp4(unsigned long addr, int rx_fd, int tx_fd, int ttl,
                    const char iface[])
{
        if (IN_MULTICAST(ntohl(addr))) {
#ifndef _WIN32
                char loop = 1;
#endif
                struct ip_mreq imr;

                in_addr_t iface_addr = get_iface_local_addr4(iface);
                if (iface_addr == (unsigned) -1) {
                        return false;
                }
                char buf[IN4_MAX_ASCII_LEN + 1] = "(err. unknown)";
                inet_ntop(AF_INET, &iface_addr, buf, sizeof buf);
                MSG(INFO, "mcast4 iface bound to: %s\n", buf);

                imr.imr_multiaddr.s_addr = addr;
                imr.imr_interface.s_addr = iface_addr;

                if (SETSOCKOPT
                    (rx_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ip_mreq)) != 0) {
                        socket_error("setsockopt IP_ADD_MEMBERSHIP");
                        return false;
                }
#ifndef _WIN32
                if (SETSOCKOPT
                    (tx_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop,
                     sizeof(loop)) != 0) {
                        socket_error("setsockopt IP_MULTICAST_LOOP");
                        return false;
                }
#endif
                if (ttl > -1) {
                        if (SETSOCKOPT
                            (tx_fd, IPPROTO_IP, IP_MULTICAST_TTL, (char *)&ttl,
                             sizeof(ttl)) != 0) {
                                socket_error("setsockopt IP_MULTICAST_TTL");
                                return false;
                        }
                } else {
                        MSG(WARNING, "Using multicast but not setting TTL.\n");
                }
                if (SETSOCKOPT
                    (tx_fd, IPPROTO_IP, IP_MULTICAST_IF,
                     (char *) &iface_addr, sizeof iface_addr) != 0) {
                        socket_error("setsockopt IP_MULTICAST_IF");
                        return false;
                }
        }
        return true;
}

static void
udp_leave_mcast_grp4(unsigned long addr, int fd, const char iface[])
{
        if (IN_MULTICAST(ntohl(addr))) {
                struct ip_mreq imr;
                imr.imr_multiaddr.s_addr = addr;
                imr.imr_interface.s_addr = get_iface_local_addr4(iface);
                if (SETSOCKOPT
                    (fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ip_mreq)) != 0) {
                        socket_error("setsockopt IP_DROP_MEMBERSHIP");
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
                free(hname);
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

static bool udp_addr_valid6(const char *dst)
{
#ifdef HAVE_IPv6
        struct in6_addr addr6;
        switch (inet_pton(AF_INET6, dst, &addr6)) {
        case 1:
                return true;
                break;
        case 0:
                return false;
                break;
        case -1:
                debug_msg("inet_pton failed\n");
                errno = 0;
        }
#endif                          /* HAVE_IPv6 */
        UNUSED(dst);
        return false;
}

static bool
udp_join_mcast_grp6(struct in6_addr sin6_addr, int rx_fd, int tx_fd, int ttl,
                    const char iface[])
{
#ifdef HAVE_IPv6
        // macOS handles v4-mapped addresses transparently as other v6 addrs;
        // Linux/Win need to use the v4 sockpts
#ifdef __APPLE__
        in_addr_t v4mapped = 0;
        memcpy((void *) &v4mapped, sin6_addr.s6_addr + 12, sizeof v4mapped);
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr) ||
            (IN6_IS_ADDR_V4MAPPED(&sin6_addr) &&
             IN_MULTICAST(ntohl(v4mapped)))) {
#else
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr)) {
#endif
                unsigned int loop = 1;
                struct ipv6_mreq imr;
#ifdef MUSICA_IPV6
                imr.i6mr_interface = 1;
                imr.i6mr_multiaddr = sin6_addr;
#else
                unsigned int ifindex = get_ifindex(iface);
                if (ifindex == (unsigned) -1) {
                        return false;
                }
                imr.ipv6mr_multiaddr = sin6_addr;
                imr.ipv6mr_interface = ifindex;
#endif

                if (SETSOCKOPT
                    (rx_fd, IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ipv6_mreq)) != 0) {
                        socket_error("setsockopt IPV6_ADD_MEMBERSHIP");
                        return false;
                }

                if (SETSOCKOPT
                    (tx_fd, IPPROTO_IPV6, IPV6_MULTICAST_LOOP, (char *)&loop,
                     sizeof(loop)) != 0) {
                        socket_error("setsockopt IPV6_MULTICAST_LOOP");
                        return false;
                }
                if (ttl > -1) {
                        if (SETSOCKOPT
                            (tx_fd, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, (char *)&ttl,
                             sizeof(ttl)) != 0) {
                                socket_error("setsockopt IPV6_MULTICAST_HOPS");
                                return false;
                        }
                } else {
                        MSG(WARNING, "Using multicast but not setting TTL.\n");
                }
                if (ifindex != 0) {
#ifdef __APPLE__
                        MSG(WARNING,
                            "Interface specification may not work "
                            "with IPv6 sockets on macOS.\n%s\n",
                            IN6_IS_ADDR_MULTICAST(&sin6_addr)
                                ? "Please contact us for details or resolution."
                                : "Use '-4' to enforce IPv4 socket");
#endif
                        if (SETSOCKOPT(tx_fd, IPPROTO_IPV6, IPV6_MULTICAST_IF,
                                       (char *) &ifindex,
                                       sizeof(ifindex)) != 0) {
                                socket_error("setsockopt IPV6_MULTICAST_IF");
                                return false;
                        }
                }

                return true;
        }
        if (IN6_IS_ADDR_V4MAPPED(&sin6_addr)) {
                in_addr_t v4mapped = 0;
                memcpy((void *) &v4mapped, sin6_addr.s6_addr + 12,
                       sizeof v4mapped);
                return udp_join_mcast_grp4(v4mapped, rx_fd, tx_fd, ttl,
                                           iface);
        }

        return true;
#else
        return false;
#endif
}

static void
udp_leave_mcast_grp6(struct in6_addr sin6_addr, int fd, const char iface[])
{
#ifdef HAVE_IPv6
        // see udp_join_mcast_grp for mac/Linux+Win difference
#ifdef __APPLE__
        in_addr_t v4mapped = 0;
        memcpy((void *) &v4mapped, sin6_addr.s6_addr + 12, sizeof v4mapped);
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr) ||
            (IN6_IS_ADDR_V4MAPPED(&sin6_addr) &&
             IN_MULTICAST(ntohl(v4mapped)))) {
#else
        if (IN6_IS_ADDR_MULTICAST(&sin6_addr)) {
#endif
                struct ipv6_mreq imr;
#ifdef MUSICA_IPV6
                imr.i6mr_interface = 1;
                imr.i6mr_multiaddr = sin6_addr;
#else
                unsigned int ifindex = get_ifindex(iface);
                if (ifindex == (unsigned) -1) {
                        return;
                }
                imr.ipv6mr_multiaddr = sin6_addr;
                imr.ipv6mr_interface = ifindex;
#endif

                if (SETSOCKOPT
                    (fd, IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, (char *)&imr,
                     sizeof(struct ipv6_mreq)) != 0) {
                        socket_error("setsockopt IPV6_DROP_MEMBERSHIP");
                }
        } else if (IN6_IS_ADDR_V4MAPPED(&sin6_addr)) {
                in_addr_t v4mapped = 0;
                memcpy((void *) &v4mapped, sin6_addr.s6_addr + 12,
                       sizeof v4mapped);
                udp_leave_mcast_grp4(v4mapped, fd, iface);
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
        struct sockaddr_in6 local, addr6, bound;
        socklen_t len = sizeof(local);
        int result = 0;

        newsock = socket(AF_INET6, SOCK_DGRAM, 0);
        if (newsock == -1) {
                socket_error("socket");
                free(hname);
                return NULL;
        }
        int ipv6only = 0;
        if (SETSOCKOPT(newsock, IPPROTO_IPV6, IPV6_V6ONLY,
                                (char *) &ipv6only,
                                sizeof(ipv6only)) != 0) {
                socket_error("newsock setsockopt IPV6_V6ONLY");
        }
        memset((char *)&addr6, 0, len);
        addr6.sin6_family = AF_INET6;
#ifdef HAVE_SIN6_LEN
        addr6.sin6_len = len;
#endif
        addr6.sin6_addr = in6addr_any;
        result = bind(newsock, (struct sockaddr *)&addr6, len);
        if (result != 0) {
                socket_error("Cannot bind");
        }
        addr6.sin6_addr = ((struct sockaddr_in6 *)&s->sock)->sin6_addr;
	len = sizeof bound;
	if (getsockname(s->local->rx_fd, (struct sockaddr *)&bound, &len) == -1) {
		socket_error("getsockname");
		addr6.sin6_port = 0;
	} else {
		addr6.sin6_port = bound.sin6_port;
	}
        addr6.sin6_scope_id = ((struct sockaddr_in6 *)&s->sock)->sin6_scope_id;
	len = sizeof addr6;
        result = connect(newsock, (struct sockaddr *)&addr6, len);
        if (result != 0) {
                socket_error("connect");
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
                        free(hname);
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
                        error_msg(MOD_NAME "udp_host_addr6: getaddrinfo: %s: %s\n", hname,
                                  ug_gai_strerror(gai_err));
                        free(hname);
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
                        free(hname);
                        return NULL;
                }
                freeaddrinfo(ai);
                return hname;
        }
        if (inet_ntop(AF_INET6, &local.sin6_addr, hname, MAXHOSTNAMELEN) ==
            NULL) {
                error_msg("inet_ntop: %s: \n", hname);
                free(hname);
                return NULL;
        }
        return hname;
#else                           /* HAVE_IPv6 */
        UNUSED(s);
        return strdup("::");    /* The unspecified address... */
#endif                          /* HAVE_IPv6 */
}

/*****************************************************************************/
/* Generic functions, which call the appropriate protocol specific routines. */
/*****************************************************************************/

/**
 * udp_addr_valid:
 * @addr: string representation of IPv4 or IPv6 network address.
 *
 * @retval true if addr is valid, false otherwise.
 **/

bool udp_addr_valid(const char *addr)
{
        return udp_addr_valid4(addr) || udp_addr_valid6(addr);
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
                     int ttl, int force_ip_version, bool multithreaded)
{
        return udp_init_if(addr, NULL, rx_port, tx_port, ttl, force_ip_version, multithreaded);
}

/// @param ipv6   socket is IPv6 (including v4-mapped)
static bool set_sock_opts_and_bind(fd_t fd, bool ipv6, uint16_t rx_port, int ttl) {
        struct sockaddr_storage s_in;
        memset(&s_in, 0, sizeof s_in);
        socklen_t sin_len;
        int reuse = 1;
        int ipv6only = 0;

        if (ipv6) {
                if (SETSOCKOPT(fd, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&ipv6only,
                                        sizeof(ipv6only)) != 0) {
                        socket_error("setsockopt IPV6_V6ONLY");
                        return false;
                }
                if (ttl > 0 && SETSOCKOPT(fd, IPPROTO_IPV6, IPV6_UNICAST_HOPS, (char *)&ttl,
                                        sizeof(ttl)) != 0) {
                        socket_error("setsockopt IPV6_UNICAST_HOPS");
                        return false;
                }
        }
        // For macs, this is set only for IPv4 socket (not IPv4-mapped IPv6), on the other
        // hand, Linux+Win requires this setting for IPv4-mapped IPv6 sockets for the TTL work.
        if (ttl > 0 &&
#ifdef __APPLE__
                        !ipv6 &&
#endif
                        SETSOCKOPT(fd, IPPROTO_IP, IP_TTL, (char *)&ttl,
                                sizeof(ttl)) != 0) {
                socket_error("setsockopt IP_TTL");
                return false;
        }
#ifdef SO_REUSEPORT
        if (SETSOCKOPT
            (fd, SOL_SOCKET, SO_REUSEPORT, (int *)&reuse,
             sizeof(reuse)) != 0) {
                socket_error("setsockopt SO_REUSEPORT");
                handle_error(EXIT_FAIL_NETWORK);
        }
#endif
        if (SETSOCKOPT
            (fd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse,
             sizeof(reuse)) != 0) {
                socket_error("setsockopt SO_REUSEADDR");
                handle_error(EXIT_FAIL_NETWORK);
        }

        if (!ipv6) {
                struct sockaddr_in *s_in4 = (struct sockaddr_in *) &s_in;
                s_in4->sin_family = AF_INET;
                s_in4->sin_addr.s_addr = htonl(INADDR_ANY);
                s_in4->sin_port = htons(rx_port);
                sin_len = sizeof(struct sockaddr_in);
        } else {
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
        }
        if (bind(fd, (struct sockaddr *)&s_in, sin_len) != 0) {
                socket_error("bind");
#ifdef _WIN32
                log_msg(LOG_LEVEL_ERROR, "Check if there is no service running on UDP port %d. ", rx_port);
                if (rx_port == 5004 || rx_port == 5005)
                        log_msg(LOG_LEVEL_ERROR, "Windows Media Services is usually a good candidate to check and disable.\n");
#endif
                return false;
        }

        return true;
}

/**
 * - checks force_ip_version validity
 * - for macOS with ipv4 mcast addr and iface set, return 4 if 0 requested
 * - otherwise return force_ip_version param
 */
static int
adjust_ip_version(int force_ip_version, const char *addr, const char *iface)
{
        assert(force_ip_version == 0 || force_ip_version == 4 ||
               force_ip_version == 6);
#ifdef __APPLE__
        if (force_ip_version == 0 && iface != NULL && is_addr4(addr) &&
            is_addr_multicast(addr)) {
                MSG(INFO, "enforcing IPv4 mode on macOS for v4 mcast address "
                            "and iface set\n");
                return IPv4;
        }
        return force_ip_version;
#else
        (void) addr, (void) iface;
        return force_ip_version;
#endif
}

static unsigned
get_ifindex(const char *iface)
{
        if (iface[0] == '\0') {
                return 0; // default
        }
        if (strcmp(iface, "help") == 0) {
                printf(
                    "mcast interface specification can be one of following:\n");
                printf( "1. interface name\n");
                printf( "2. interface index\n");
                printf( "3. bind address (IPv4 only)\n");
                printf("\n");
                return (unsigned) -1;
        }
        const unsigned ret = if_nametoindex(iface);
        if (ret != 0) {
                return ret;
        }
        // check if the value isn't the index itself
        char *endptr = NULL;
        const long val = strtol(iface, &endptr, 0);
        if (*endptr == '\0' && val >= 0 && (uintmax_t) val <= UINT_MAX) {
                char buf[IF_NAMESIZE];
                if (if_indextoname(val, buf) != NULL) {
                        MSG(INFO, "Using mcast interface %s\n", buf);
                        return val;
                }
        }
        struct in_addr iface_addr;
        if (inet_pton(AF_INET, iface, &iface_addr) == 1) {
                error_msg("Interface identified with addres %s not allowed "
                          "here. Try '-4'...\n",
                          iface);
                return (unsigned) -1;
        }
        error_msg("Illegal interface specification '%s': %s\n", iface,
                  ug_strerror(errno));
        return (unsigned) -1;
}

ADD_TO_PARAM("udp-queue-len",
                "* udp-queue-len=<l>\n"
                "  Use different queue size than default DEFAULT_MAX_UDP_READER_QUEUE_LEN\n");
#ifdef _WIN32
ADD_TO_PARAM("udp-disable-multi-socket",
                "* udp-disable-multi-socket\n"
                "  Disable separate sockets for RX and TX (Win only). Separated RX/TX is a workaround\n"
                "  to some locking issues (thr. in recv() while no data are received and concurr. send()).\n");
#endif
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
 * @param force_ip_version     force specific IP version
 * @param multithreaded receiving in a separate thread than processing
 *
 * @returns a pointer to a socket_udp structure on success, NULL otherwise.
 **/
socket_udp *udp_init_if(const char *addr, const char *iface, uint16_t rx_port,
                        uint16_t tx_port, int ttl, int force_ip_version, bool multithreaded)
{
        int ret;
        socket_udp *s = (socket_udp *) calloc(1, sizeof *s);
        s->local = (struct socket_udp_local*) calloc(1, sizeof(*s->local));
        s->local->packets = simple_linked_list_init();
        s->local->rx_fd =
                s->local->tx_fd = INVALID_SOCKET;
        pthread_mutex_init(&s->local->lock, NULL);
        pthread_cond_init(&s->local->boss_cv, NULL);
        pthread_cond_init(&s->local->reader_cv, NULL);

        s->local->mode = adjust_ip_version(force_ip_version, addr, iface);

        if ((ret = resolve_address(s, addr, tx_port)) != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Can't resolve IP address for %s: %s\n", addr,
                                ug_gai_strerror(ret));
                goto error;
        }
        if (iface != NULL) {
                snprintf_ch(s->iface, "%s", iface);
        }
#ifdef _WIN32
        if (!is_host_loopback(addr)) {
                s->local->is_wsa_overlapped = true;
        }

        s->local->rx_fd = WSASocket(s->sock.ss_family, SOCK_DGRAM, IPPROTO_UDP, NULL, 0, s->local->is_wsa_overlapped ? WSA_FLAG_OVERLAPPED : 0);
        if (get_commandline_param("udp-disable-multi-socket")) {
                s->local->tx_fd = s->local->rx_fd;
        } else {
                s->local->tx_fd = WSASocket(s->sock.ss_family, SOCK_DGRAM, IPPROTO_UDP, NULL, 0, s->local->is_wsa_overlapped ? WSA_FLAG_OVERLAPPED : 0);
        }
#else
        s->local->rx_fd =
                s->local->tx_fd = socket(s->sock.ss_family, SOCK_DGRAM, 0);
#endif
        if (s->local->rx_fd == INVALID_SOCKET ||
                        s->local->tx_fd == INVALID_SOCKET) {
                socket_error("Unable to initialize socket");
                goto error;
        }
        // Set properties and bind
        // The order here is important if using separate socket for RX and TX - in
        // MSW, first bound socket receives data, in Linux (with Wine) the
        // second.
        if (!set_sock_opts_and_bind(s->local->rx_fd, s->local->mode == IPv6, rx_port, ttl) ||
                        (s->local->rx_fd != s->local->tx_fd && !set_sock_opts_and_bind(s->local->tx_fd, s->local->mode == IPv6, rx_port, ttl))) {
                goto error;
        }
        if (is_wine()) {
                SWAP(s->local->rx_fd, s->local->tx_fd);
        }

        // if we do not set tx port, fake that is the same as we are bound to
        if (tx_port == 0) {
                struct sockaddr_storage sin;
                socklen_t addrlen = sizeof(sin);
                if (getsockname(s->local->rx_fd, (struct sockaddr *)&sin, &addrlen) == 0 &&
                                sin.ss_family == s->sock.ss_family) {
                        if (s->local->mode == IPv4) {
                                struct sockaddr_in *s_in4 = (struct sockaddr_in *) &sin;
                                tx_port = ntohs(s_in4->sin_port);
                                ((struct sockaddr_in *) &s->sock)->sin_port = s_in4->sin_port;
                        } else if (s->local->mode == IPv6) {
#ifdef HAVE_IPv6
                                struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *) &sin;
                                tx_port = ntohs(s_in6->sin6_port);
                                ((struct sockaddr_in6 *) &s->sock)->sin6_port = s_in6->sin6_port;
#endif
                        } else {
                                abort();
                        }
                }
        }

        switch (s->local->mode) {
        case IPv4:
                if (!udp_join_mcast_grp4(
                        ((struct sockaddr_in *) &s->sock)->sin_addr.s_addr,
                        s->local->rx_fd, s->local->tx_fd, ttl, s->iface)) {
                        goto error;
                }
                break;
        case IPv6:
                if (!udp_join_mcast_grp6(
                        ((struct sockaddr_in6 *) &s->sock)->sin6_addr,
                        s->local->rx_fd, s->local->tx_fd, ttl, s->iface)) {
                        goto error;
                }
                break;
        default:
                abort();
        }

        s->local->multithreaded = multithreaded;
        if (multithreaded) {
                if (!get_commandline_param("udp-queue-len")) {
                        s->local->max_packets = DEFAULT_MAX_UDP_READER_QUEUE_LEN;
                } else {
                        s->local->max_packets = atoi(get_commandline_param("udp-queue-len"));
                }
                platform_pipe_init(s->local->should_exit_fd);
                pthread_create(&s->local->thread_id, NULL, udp_reader, s);
        }

        return s;

error:
        udp_exit(s);
        return NULL;
}

/// @returns true if the socket was initialize with
///               IN6_BLACKHOLE_SERVER_MODE_STR
bool
udp_is_server_mode_blackhole(socket_udp *s)
{
        struct in6_addr ipv6_blackhole_server_mode;
        inet_pton(AF_INET6, IN6_BLACKHOLE_SERVER_MODE_STR,
                  &ipv6_blackhole_server_mode);
        return s->sock.ss_family == AF_INET6 &&
               memcmp(&((struct sockaddr_in6 *) &s->sock)->sin6_addr,
                      &ipv6_blackhole_server_mode,
                      sizeof ipv6_blackhole_server_mode) == 0;
}

void udp_set_receiver(socket_udp *s, struct sockaddr *sa, socklen_t len) {
        memcpy(&s->sock, sa, len);
        s->sock_len = len;
}

socket_udp *udp_init_with_local(struct socket_udp_local *l, struct sockaddr *sa, socklen_t len)
{
        if ((sa->sa_family == AF_INET && l->mode != IPv4) ||
                        (sa->sa_family == AF_INET6 && l->mode != IPv6) ||
                        (sa->sa_family != AF_INET && sa->sa_family != AF_INET6)) {
                log_msg(LOG_LEVEL_ERROR, "[NET UDP] udp_init_with_local: Protocol mismatch!\n");
                return NULL;
        }

        struct _socket_udp *s = (socket_udp *) calloc(1, sizeof *s);
        s->local = l;
        s->local_is_slave = true;

        udp_set_receiver(s, sa, len);

        return s;
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
        if (s == NULL) {
                return;
        }
        switch (s->local->mode) {
        case IPv4:
                udp_leave_mcast_grp4(
                    ((struct sockaddr_in *) &s->sock)->sin_addr.s_addr,
                    s->local->rx_fd, s->iface);
                break;
        case IPv6:
                udp_leave_mcast_grp6(
                    ((struct sockaddr_in6 *) &s->sock)->sin6_addr,
                    s->local->rx_fd, s->iface);
                break;
        }

        if (!s->local_is_slave) {
                if (s->local->multithreaded) {
                        char c = 0;
                        int ret = PLATFORM_PIPE_WRITE(s->local->should_exit_fd[1], &c, 1);
                        assert (ret == 1);
                        s->local->should_exit = true;
                        pthread_cond_signal(&s->local->reader_cv);
                        pthread_join(s->local->thread_id, NULL);
                        while (simple_linked_list_size(s->local->packets) > 0) {
                                struct item *item = (struct item *) simple_linked_list_pop(s->local->packets);
                                free(item->buf);
                        }
                        platform_pipe_close(s->local->should_exit_fd[1]);
                }
                CLOSESOCKET(s->local->rx_fd);
                if (s->local->tx_fd != s->local->rx_fd) {
                        CLOSESOCKET(s->local->tx_fd);
                }
                simple_linked_list_destroy(s->local->packets);
                pthread_mutex_destroy(&s->local->lock);
                pthread_cond_destroy(&s->local->boss_cv);
                pthread_cond_destroy(&s->local->reader_cv);
                free(s->local);
        }

        udp_clean_async_state(s);

        free(s);
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

        return sendto(s->local->tx_fd, buffer, buflen, 0, (struct sockaddr *)&s->sock,
                      s->sock_len);
}

int udp_sendto(socket_udp * s, char *buffer, int buflen, struct sockaddr *dst_addr, socklen_t addrlen)
{
        return sendto(s->local->tx_fd, buffer, buflen, 0, dst_addr, addrlen);
}

#ifdef _WIN32
int udp_sendv(socket_udp * s, LPWSABUF vector, int count, void *d)
{
        assert(s != NULL);

        assert(!s->overlapping_active || s->overlapped_count < s->overlapped_max);

	DWORD bytesSent;
	int ret = WSASendTo(s->local->tx_fd, vector, count, &bytesSent, 0,
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

        int ret = sendmsg(s->local->tx_fd, &msg, 0);
        free(d);
        return ret;
}
#endif // _WIN32

/**
 * When receiving data in separate thread, this function fetches data
 * from socket and puts it in queue.
 */
static void *udp_reader(void *arg)
{
        set_thread_name(__func__);
        socket_udp *s = (socket_udp *) arg;

        while (1) {
                fd_set fds;
                FD_ZERO(&fds);
                FD_SET(s->local->rx_fd, &fds);
                FD_SET(s->local->should_exit_fd[0], &fds);
                int nfds = MAX(s->local->rx_fd, s->local->should_exit_fd[0]) + 1;

                int rc = select(nfds, &fds, NULL, NULL, NULL);
                if (rc <= 0) {
                        socket_error("select");
                        continue;
                }
                if (FD_ISSET(s->local->should_exit_fd[0], &fds)) {
                        break;
                }
                uint8_t *packet = (uint8_t *) malloc(ALIGNED_ITEM_OFF + sizeof(struct item));
                uint8_t *buffer = ((uint8_t *) packet) + RTP_PACKET_HEADER_SIZE;
                struct sockaddr *src_addr = (struct sockaddr *)(void *)(packet + ALIGNED_SOCKADDR_STORAGE_OFF);
                socklen_t addrlen = sizeof(struct sockaddr_storage);
                int size = recvfrom(s->local->rx_fd, (char *) buffer,
                                RTP_MAX_PACKET_LEN - RTP_PACKET_HEADER_SIZE,
                                0, src_addr, &addrlen);

                if (size <= 0) {
                        /// @todo
                        /// In MSW, this block is called as often as packet is sent if
                        /// we got WSAECONNRESET error (noone is listening). This can have
                        /// negative performance impact.
                        socket_error("recvfrom");
                        free(packet);
                        continue;
                }

                pthread_mutex_lock(&s->local->lock);
                while (simple_linked_list_size(s->local->packets) >= (int) s->local->max_packets && !s->local->should_exit) {
                        pthread_cond_wait(&s->local->reader_cv, &s->local->lock);
                }
                if (s->local->should_exit) {
                        free(packet);
                        pthread_mutex_unlock(&s->local->lock);
                        break;
                }

                struct item *i = (struct item *)(void *)(packet + ALIGNED_ITEM_OFF);
                *i = (struct item){packet, size, src_addr, addrlen};
                simple_linked_list_append(s->local->packets, i);

                pthread_mutex_unlock(&s->local->lock);
                pthread_cond_signal(&s->local->boss_cv);
        }

        platform_pipe_close(s->local->should_exit_fd[0]);

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

        len = recvfrom(s->local->rx_fd, buffer, buflen, flags, src_addr, addrlen);
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
        assert(s->local->multithreaded);

        pthread_mutex_lock(&s->local->lock);
        if (timeout) {
                struct timeval tv;
                gettimeofday(&tv, NULL);
                tv.tv_sec += timeout->tv_sec;
                tv.tv_usec += timeout->tv_usec;
                if (tv.tv_usec >= 1000000) {
                        tv.tv_sec += 1;
                        tv.tv_usec -= 1000000;
                }
                struct timespec tmout_ts = { tv.tv_sec, tv.tv_usec * 1000 };
                int rc = 0;
                while (rc != ETIMEDOUT && simple_linked_list_size(s->local->packets) == 0) {
                        rc = pthread_cond_timedwait(&s->local->boss_cv, &s->local->lock, &tmout_ts);
                }
        } else {
                while (simple_linked_list_size(s->local->packets) == 0) {
                        pthread_cond_wait(&s->local->boss_cv, &s->local->lock);
                }
        }
        bool ret = simple_linked_list_size(s->local->packets) > 0;
        pthread_mutex_unlock(&s->local->lock);
        return ret;
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
int udp_recvfrom_data(socket_udp * s, char **buffer,
                struct sockaddr *src_addr, socklen_t *addrlen)
{
        assert(s->local->multithreaded);
        int ret;

        pthread_mutex_lock(&s->local->lock);
        struct item *it = (struct item *)(simple_linked_list_pop(s->local->packets));
        *buffer = (char *) it->buf;
        if(src_addr){
                if(it->src_addr){
                        memcpy(src_addr, it->src_addr, it->addrlen);
                        *addrlen = it->addrlen;
                } else {
                        *addrlen = 0;
                }
        }
        ret = it->size;

        pthread_mutex_unlock(&s->local->lock);
        pthread_cond_signal(&s->local->reader_cv);

        return ret;
}
int udp_recv_data(socket_udp * s, char **buffer){
        return udp_recvfrom_data(s, buffer, NULL, NULL);
}

#ifndef _WIN32
int udp_recvv(socket_udp * s, struct msghdr *m)
{
        if (recvmsg(s->local->rx_fd, m, 0) == -1) {
                perror("recvmsg");
                return 1;
        }
        return 0;
}
#endif // _WIN32

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
        FD_SET(s->local->rx_fd, &rfd);
        if (s->local->rx_fd > (fd_t) max_fd) {
                max_fd = s->local->rx_fd;
        }
}

void udp_fd_set_r(socket_udp *s, struct udp_fd_r *fd_struct)
{
        FD_SET(s->local->rx_fd, &fd_struct->rfd);
        if (s->local->rx_fd > (fd_t) fd_struct->max_fd) {
                fd_struct->max_fd = s->local->rx_fd;
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
        return FD_ISSET(s->local->rx_fd, &rfd);
}

int udp_fd_isset_r(socket_udp *s, struct udp_fd_r *fd_struct)
{
        return FD_ISSET(s->local->rx_fd, &fd_struct->rfd);
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
        switch (s->local->mode) {
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
        if (s && s->local->rx_fd > 0) {
                return s->local->rx_fd;
        }
        return 0;
}

/**
 * @param[in]  mode  IP version preference (4 or 6) or 0 for none;
                     if 0 is requested, v4-mapped IPv6 address is returned
 * @param[out] mode  sockaddr struct version - 4 for input mode 4, 6 otherwise
 *
 * @returns 0 on success, GAI error on failure
 */
int resolve_addrinfo(const char *addr, uint16_t tx_port,
                struct sockaddr_storage *dst, socklen_t *len, int *mode)
{
        struct addrinfo hints, *res0;
        int err;
        memset(&hints, 0, sizeof(hints));
        switch (*mode) {
        case 0:
                hints.ai_family = AF_INET6;
                hints.ai_flags = AI_V4MAPPED | AI_ALL;
                break;
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
        snprintf(tx_port_str, sizeof tx_port_str, "%u", tx_port);
        if ((err = getaddrinfo(addr, tx_port_str, &hints, &res0)) != 0) {
                /* We should probably try to do a DNS lookup on the name */
                /* here, but I'm trying to get the basics going first... */
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "resolve_address: getaddrinfo: %s: %s\n", addr, ug_gai_strerror(err));
                return err;
        }
        memcpy(dst, res0->ai_addr, res0->ai_addrlen);
        *len = res0->ai_addrlen;
        *mode = (res0->ai_family == AF_INET ? IPv4 : IPv6);
        freeaddrinfo(res0);

        return 0;
}

static int resolve_address(socket_udp *s, const char *addr, uint16_t tx_port)
{
        int ret = resolve_addrinfo(addr, tx_port, &s->sock, &s->sock_len, &s->local->mode);
        if(ret == 0)
                verbose_msg("Connected IP version %d\n", s->local->mode);

        return ret;
}

int
udp_get_recv_buf(socket_udp *s)
{
        int       opt      = 0;
        socklen_t opt_size = sizeof opt;
        if (GETSOCKOPT(s->local->tx_fd, SOL_SOCKET, SO_RCVBUF, (sockopt_t) &opt,
                       &opt_size) != 0) {
                socket_error("Unable to get socket buffer size");
                return -1;
        }
        return opt;
}

static bool
udp_set_buf(socket_udp *s, int sockopt, int size)
{
        assert(sockopt == SO_RCVBUF || sockopt == SO_SNDBUF);
        const fd_t fd =
            sockopt == SO_RCVBUF ? s->local->rx_fd : s->local->tx_fd;
        if (SETSOCKOPT(fd, SOL_SOCKET, sockopt, (sockopt_t) &size,
                       sizeof(size)) != 0) {
                socket_error("Unable to set socket buffer size");
                return false;
        }

        int       opt      = 0;
        socklen_t opt_size = sizeof opt;
        if (GETSOCKOPT(fd, SOL_SOCKET, sockopt, (sockopt_t) &opt, &opt_size) !=
            0) {
                socket_error("Unable to get socket buffer size");
                return false;
        }

        const bool ret = opt >= size;
        log_msg(LOG_LEVEL_VERBOSE,
                "Socket %s buffer size set to %d B%srequested %d B%s\n",
                sockopt == SO_RCVBUF ? "recv" : "send", opt,
                ret ? " (" : ", ", size, ret ? ")" : "!");
        return ret;
}

bool
udp_set_recv_buf(socket_udp *s, int size)
{
        return udp_set_buf(s, SO_RCVBUF, size);
}

bool
udp_set_send_buf(socket_udp *s, int size)
{
        return udp_set_buf(s, SO_SNDBUF, size);
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
        FD_SET(s->local->rx_fd, &select_fd);

        while(select(s->local->rx_fd + 1, &select_fd, NULL, NULL, &timeout) > 0) {
                ssize_t bytes = recv(s->local->rx_fd, buf, len, 0);
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
#ifdef _WIN32
        if (!s->local->is_wsa_overlapped) {
                return;
        }

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
#ifdef _WIN32
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
#ifdef _WIN32
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
        return s->local->mode == IPv6 && !IN6_IS_ADDR_V4MAPPED(&((struct sockaddr_in6 *) &s->sock)->sin6_addr);
}

/**
 * @retval  0 success
 * @retval -1 port pair is not free
 * @retval -2 another error
 */
int udp_port_pair_is_free(int force_ip_version, int even_port)
{
        struct addrinfo hints;
        memset(&hints, 0, sizeof hints);
        struct addrinfo *res0 = NULL;
        hints.ai_family = force_ip_version == 4 ? AF_INET : AF_INET6;
        hints.ai_flags = AI_NUMERICSERV | AI_PASSIVE;
        hints.ai_socktype = SOCK_DGRAM;
        char tx_port_str[7];
        snprintf(tx_port_str, sizeof tx_port_str, "%hu", (uint16_t) even_port);
        int err = getaddrinfo(NULL, tx_port_str, &hints, &res0);
        if (err) {
                /* We should probably try to do a DNS lookup on the name */
                /* here, but I'm trying to get the basics going first... */
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s getaddrinfo port %d: %s\n", __func__, even_port, gai_strerror(err));
                return -2;
        }

        for (int i = 0; i < 2; ++i) {
                struct sockaddr *sin = res0->ai_addr;
                fd_t fd;

                if (sin->sa_family == AF_INET6) {
                        struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *)(void *) sin;
                        int ipv6only = 0;
                        s_in6->sin6_port = htons(even_port + i);
                        fd = socket(AF_INET6, SOCK_DGRAM, 0);
                        if (fd != INVALID_SOCKET) {
                                if (SETSOCKOPT
                                                (fd, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&ipv6only,
                                                 sizeof(ipv6only)) != 0) {
                                        socket_error("%s - setsockopt IPV6_V6ONLY for port %d", __func__, even_port + 1);
                                        CLOSESOCKET(fd);
                                        freeaddrinfo(res0);
                                        return -2;
                                }
                        }
                } else {
                        struct sockaddr_in *s_in = (struct sockaddr_in *)(void *) sin;
                        s_in->sin_port = htons(even_port + i);
                        fd = socket(AF_INET, SOCK_DGRAM, 0);
                }

                if (fd == INVALID_SOCKET) {
                        socket_error("%s - unable to initialize socket for port %d", __func__, even_port + 1);
                        freeaddrinfo(res0);
                        return -2;
                }

                if (bind(fd, (struct sockaddr *) sin, res0->ai_addrlen) != 0) {
                        int ret = 0;
#ifdef _WIN32
                        if (WSAGetLastError() == WSAEADDRINUSE) {
#else
                        if (errno == EADDRINUSE) {
#endif
                                ret = -1;
                        } else {
                                ret = -2;
                                socket_error("%s - cannot bind port %d", __func__, even_port + 1);
                        }
                        freeaddrinfo(res0);
                        CLOSESOCKET(fd);
                        return ret;
                }

                CLOSESOCKET(fd);
        }

        freeaddrinfo(res0);

        return 0;
}

#ifdef _WIN32
/**
 * This function sends buffer asynchronously with provided Winsock parameters.
 *
 * This is different approach from udp_async_start() and udp_async_wait() because
 * this function lets caller provide parameters and passes them directly.
 */
int udp_sendto_wsa_async(socket_udp *s, char *buffer, int buflen, LPWSAOVERLAPPED_COMPLETION_ROUTINE l, LPWSAOVERLAPPED o, struct sockaddr *addr, socklen_t addrlen)
{
        assert(s != NULL);

        DWORD bytesSent;
        WSABUF vector;
        vector.buf = buffer;
        vector.len = buflen;
        int ret = WSASendTo(s->local->tx_fd, &vector, 1, &bytesSent, 0, addr, addrlen, o, l);
        if (ret == 0 || WSAGetLastError() == WSA_IO_PENDING)
                return 0;
        else {
                return ret;
        }
}

int udp_send_wsa_async(socket_udp *s, char *buffer, int buflen, LPWSAOVERLAPPED_COMPLETION_ROUTINE l, LPWSAOVERLAPPED o)
{
        return udp_sendto_wsa_async(s, buffer, buflen, l, o, (struct sockaddr *) &s->sock, s->sock_len);
}
#endif

/**
 * Tries to receive data from socket and if empty, waits at most specified
 * amount of time.
 */
int udp_recvfrom_timeout(socket_udp *s, char *buffer, int buflen,
                struct timeval *timeout,
                struct sockaddr *src_addr, socklen_t *addrlen)
{
        struct udp_fd_r fd;
        int len = 0;

        if (s->local->multithreaded) {
                if (udp_not_empty(s, timeout)) {
                        char *data = NULL;
                        len = udp_recvfrom_data(s, (char **) &data, src_addr, addrlen);
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
                                len = udp_recvfrom(s, buffer, buflen, src_addr, addrlen);
                                if (len < 0) {
                                        len = 0;
                                }
                        }
                }
        }
        return len;
}

int udp_recv_timeout(socket_udp *s, char *buffer, int buflen, struct timeval *timeout)
{
        return udp_recvfrom_timeout(s, buffer, buflen, timeout, NULL, NULL);
}

struct socket_udp_local *udp_get_local(socket_udp *s)
{
        return s->local;
}

int udp_get_udp_rx_port(socket_udp *s)
{
        struct sockaddr_storage ss;
        socklen_t len = sizeof ss;
        if (getsockname(s->local->rx_fd, (struct sockaddr *) &ss, &len) != 0) {
                socket_error(__func__);
                return -1;
        }
        assert(ss.ss_family == AF_INET || ss.ss_family == AF_INET6);

        return htons(ss.ss_family == AF_INET ? ((struct sockaddr_in *) &ss)->sin_port : ((struct sockaddr_in6 *) &ss)->sin6_port);
}


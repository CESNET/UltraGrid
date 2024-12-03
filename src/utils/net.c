/**
 * @file   utils/net.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016-2024 CESNET
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
 /**
  * @todo
  * move/use common network defs from compat/net.h
  */

#include "utils/net.h"

#ifdef _WIN32
#include <winsock2.h>
#include <iphlpapi.h>
#include <tchar.h>
#include <ws2tcpip.h>
typedef SOCKET fd_t;
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#define INVALID_SOCKET (-1)
typedef int fd_t;
#endif

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rtp/net_udp.h"   // for resolve_addrinfo
#include "utils/macros.h"  // for STR_LEN
#include "utils/windows.h"

#include "debug.h"

#define IPV4_LL_PREFIX ((169u<<8u) | 254u)

// MSW does not have the macro defined
#ifndef IN_LOOPBACKNET
#define IN_LOOPBACKNET 127
#endif

#define MOD_NAME "[utils/net] "

bool is_addr_linklocal(struct sockaddr *sa)
{
        switch (sa->sa_family) {
        case AF_INET:
        {
                struct sockaddr_in *sin = (struct sockaddr_in *)(void *) sa;
                uint32_t addr = ntohl(sin->sin_addr.s_addr);
                if ((addr >> 16u) == IPV4_LL_PREFIX) {
                        return true;
                }
                return false;
        }
        case AF_INET6:
        {
                struct sockaddr_in6 *sin = (struct sockaddr_in6 *)(void *) sa;
                if (IN6_IS_ADDR_V4MAPPED(&sin->sin6_addr)) {
                        uint32_t v4_addr = ntohl(*((uint32_t*)(void *)(sin->sin6_addr.s6_addr + 12)));
                        if ((v4_addr >> 16u) == IPV4_LL_PREFIX) {
                                return true;
                        }
                        return false;
                }
                return IN6_IS_ADDR_LINKLOCAL(&sin->sin6_addr);
        }
        default:
                return false;
        }
}

bool is_addr_loopback(struct sockaddr *sa)
{
        switch (sa->sa_family) {
        case AF_UNIX:
                return true;
        case AF_INET:
        {
                struct sockaddr_in *sin = (struct sockaddr_in *)(void *) sa;
                uint32_t addr = ntohl(sin->sin_addr.s_addr);
                if ((addr >> 24u) == IN_LOOPBACKNET) {
                        return true;
                }
                return false;
        }
        case AF_INET6:
        {
                struct sockaddr_in6 *sin = (struct sockaddr_in6 *)(void *) sa;
                if (IN6_IS_ADDR_V4MAPPED(&sin->sin6_addr)) {
                        uint32_t v4_addr = ntohl(*((uint32_t*)(void *)(sin->sin6_addr.s6_addr + 12)));
                        if ((v4_addr >> 24u) == IN_LOOPBACKNET) {
                                return true;
                        }
                        return false;
                }
                return IN6_IS_ADDR_LOOPBACK(&sin->sin6_addr);
        }
        default:
                return false;
        }
}

bool is_addr_private(struct sockaddr *sa)
{
        switch (sa->sa_family) {
        case AF_UNIX:
                return true;
        case AF_INET:
        {
                struct sockaddr_in *sin = (struct sockaddr_in *)(void *) sa;
                uint32_t addr = ntohl(sin->sin_addr.s_addr);
                return ((addr >> 24U) == 10) || ((addr >> 20U) == 2753) || ((addr >> 16U) == 49320);
        }
        case AF_INET6:
        {
                struct sockaddr_in6 *sin = (struct sockaddr_in6 *)(void *) sa;
                return (sin->sin6_addr.s6_addr[0] & 0xFEU) == 0xFCU; // ULA
        }
        default:
                return false;
        }
}

static struct addrinfo *resolve_host(const char *hostname, const char *err_prefix)
{
        int gai_err = 0;
        struct addrinfo *ai = NULL;
        struct addrinfo hints = { .ai_family = AF_UNSPEC, .ai_socktype = SOCK_DGRAM };

        if ((gai_err = getaddrinfo(hostname, NULL, &hints, &ai)) != 0) {
                error_msg("%sgetaddrinfo: %s: %s\n", err_prefix, hostname,
                                ug_gai_strerror(gai_err));
                return NULL;
        }
        return ai;
}

bool is_host_loopback(const char *hostname)
{
        struct addrinfo *ai = resolve_host(hostname, "is_host_loopback: ");
        if (ai == NULL) {
                return false;
        }

        bool ret = is_addr_loopback(ai->ai_addr);
        freeaddrinfo(ai);

        return ret;
}

bool is_host_private(const char *hostname)
{
        struct addrinfo *ai = resolve_host(hostname, "is_host_private: ");
        if (ai == NULL) {
                return false;
        }

        bool ret = is_addr_loopback(ai->ai_addr);
        freeaddrinfo(ai);

        return ret;
}

uint16_t socket_get_recv_port(int fd)
{
        struct sockaddr_storage ss;
        socklen_t len = sizeof(ss);
        if (getsockname(fd, (struct sockaddr *)&ss, &len) == -1) {
                perror("getsockname");
                return 0;
        }
        switch (ss.ss_family) {
        case AF_INET:
                return ntohs(((struct sockaddr_in *) &ss)->sin_port);
        case AF_INET6:
                return ntohs(((struct sockaddr_in6 *) &ss)->sin6_port);
        default:
                return 0;
        }
}

/**
 * Returns (in output parameter) list of local public IP addresses
 *
 * Code was taken perhaps from:
 * https://social.msdn.microsoft.com/Forums/en-US/3b6a92ac-93d3-4f59-a914-340c1ba41cff/how-to-retrieve-ip-addresses-of-the-network-cards-with-getadaptersaddresses
 *
 * @param[out] addrs      storage allocate to copy the addresses
 * @param[in]  len        length of allocated space (in bytes)
 * @param[out] len        length of returned addresses (in parameter addrs, in bytes)
 * @param      ip_version 6 for IPv6, 4 for IPv6, 0 both
 */
bool get_local_addresses(struct sockaddr_storage *addrs, size_t *len, int ip_version)
{
        assert(ip_version == 0 || ip_version == 4 || ip_version == 6);
#ifdef _WIN32
#define WORKING_BUFFER_SIZE 15000
#define MAX_TRIES 3
	/* Declare and initialize variables */
	DWORD dwRetVal = 0;
	size_t len_remaining = *len;
	*len = 0;

	PIP_ADAPTER_ADDRESSES pAddresses = NULL;
	ULONG outBufLen = 0;
	ULONG Iterations = 0;

	PIP_ADAPTER_ADDRESSES pCurrAddresses = NULL;
	PIP_ADAPTER_UNICAST_ADDRESS pUnicast = NULL;

	// Allocate a 15 KB buffer to start with.
	outBufLen = WORKING_BUFFER_SIZE;

	do {
		pAddresses = (IP_ADAPTER_ADDRESSES *) malloc(outBufLen);
		if (pAddresses == NULL) {
			printf
				("Memory allocation failed for IP_ADAPTER_ADDRESSES struct\n");
			return false;
		}

                ULONG family = ip_version == 4 ? AF_INET
                        : ip_version == 6 ? AF_INET6
                        : AF_UNSPEC; // both
		dwRetVal =
			GetAdaptersAddresses(family, 0, NULL, pAddresses, &outBufLen);

		if (dwRetVal == ERROR_BUFFER_OVERFLOW) {
			free(pAddresses);
			pAddresses = NULL;
		} else {
			break;
		}
		Iterations++;
	} while ((dwRetVal == ERROR_BUFFER_OVERFLOW) && (Iterations < MAX_TRIES));

	if (dwRetVal == NO_ERROR) {
		// If successful, output some information from the data we received
		pCurrAddresses = pAddresses;
		while (pCurrAddresses) {

			pUnicast = pCurrAddresses->FirstUnicastAddress;
                        while (pUnicast != NULL) {
                                if (len_remaining < sizeof(addrs[0])) {
                                        printf("Warning: insufficient space for all addresses.\n");
                                        return true;
                                }
                                size_t sa_len = pUnicast->Address.lpSockaddr->sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
                                memcpy(addrs, pUnicast->Address.lpSockaddr, sa_len);
                                addrs += 1;
                                *len += sizeof(addrs[0]);
                                len_remaining -= sizeof(addrs[0]);
                                pUnicast = pUnicast->Next;
			}

			pCurrAddresses = pCurrAddresses->Next;
		}
	} else {
		printf("Call to GetAdaptersAddresses failed with error: %ld\n",
				dwRetVal);
		if (dwRetVal == ERROR_NO_DATA)
			printf("\tNo addresses were found for the requested parameters\n");
		else {
                        log_msg(LOG_LEVEL_ERROR, "Error: %s\n",
                                get_win32_error(dwRetVal));
                        if (pAddresses)
                                free(pAddresses);
                        return false;
		}
	}

	if (pAddresses) {
		free(pAddresses);
	}

	return true;
#else // ! defined _WIN32
	size_t available_len = *len;
	*len = 0;
	struct ifaddrs* a = NULL;
	getifaddrs(&a);
	struct ifaddrs* p = a;
	while (NULL != p) {
                if (p->ifa_addr == NULL ||
                    (ip_version == 0 && (p->ifa_addr->sa_family != AF_INET &&
                                         p->ifa_addr->sa_family != AF_INET6)) ||
                    (ip_version == 4 && p->ifa_addr->sa_family != AF_INET) ||
                    (ip_version == 6 && p->ifa_addr->sa_family != AF_INET6)) {
                        p = p->ifa_next;
                        continue;
                }
                if (available_len < sizeof addrs[0]) {
                        MSG(WARNING,
                            "Warning: insufficient space for all addresses.\n");
                        return true;
                }
                size_t sa_len = p->ifa_addr->sa_family == AF_INET
                                    ? sizeof(struct sockaddr_in)
                                    : sizeof(struct sockaddr_in6);
                memcpy(addrs, p->ifa_addr, sa_len);
                addrs += 1;
                *len += sizeof(addrs[0]);
                available_len -= sizeof(addrs[0]);

                p = p->ifa_next;
        }
	if (NULL != a) {
		freeifaddrs(a);
	}
	return true;
#endif
}

/**
 * Checks if the address is a dot-separated numeric IPv4 address.
 */
static bool is_addr4(const char *addr) {
        // only basic check
        while (*addr != '\0') {
                if (!isdigit(*addr) && *addr != '.') {
                        return false;
                }
                addr++;
        }
        return true;
}

/**
 * Checks if the address is a colon-separated numeric IPv6 address
 * (with optional zone index).
 */
static bool is_addr6(const char *addr) {
        while (*addr != '\0') {
                if (*addr == '%') { // skip zone identification at the end
                        return true;
                }
                // only basic check
                if (!isxdigit(*addr) && *addr != ':') {
                        return false;
                }
                addr++;
        }
        return true;
}

bool is_addr_multicast(const char *addr)
{
        if (is_addr4(addr)) {
                unsigned int first_addr_byte = atoi(addr);
                if ((first_addr_byte >> 4) == 0xe) {
                        return true;
                }
                return false;

        }
        if (is_addr6(addr)) {
                if (strlen(addr) >= 2 && addr[0] == 'f' && addr[1] == 'f') {
                        return true;
                }
                return false;

        }
        return false;
}

bool is_ipv6_supported(void)
{
        fd_t fd = socket(AF_INET6, SOCK_DGRAM, 0);
        if (fd == INVALID_SOCKET && errno == EAFNOSUPPORT) {
                return false;
        }
        if (fd != INVALID_SOCKET) {
                CLOSESOCKET(fd);
        }
        return true;
}

/**
 * @brief writes string "host:port" for given sockaddr
 *
 * IPv6 addresses will be enclosed in brackets [], scope ID is output as well if
 * defined.
 *
 * @param n size of buf; must be at least @ref ADDR_STR_BUF_LEN
 * @returns the input buffer (buf) pointer with result
 */
char *
get_sockaddr_str(const struct sockaddr *sa, unsigned sa_len, char *buf,
                 size_t n)
{
        assert(n >= ADDR_STR_BUF_LEN);

        char       *buf_ptr = buf; // ptr to be appended to
        const char *buf_end = buf + n; // endptr to check

        if (sa->sa_family == AF_INET6) {
                buf_ptr += snprintf(buf_ptr, buf_end - buf_ptr, "[");
        }
        char port[IN_PORT_STR_LEN + 1];
        const int rc =
            getnameinfo(sa, sa_len, buf_ptr, buf_end - buf_ptr, port,
                        sizeof port, NI_NUMERICHOST | NI_NUMERICSERV);
        if (rc != 0) {
                MSG(ERROR, "%s getnameinfo: %s\n", __func__, gai_strerror(rc));
                snprintf(buf, n, "(error)");
                return buf;
        }
        buf_ptr += strlen(buf_ptr);

        if (sa->sa_family == AF_INET6) {
                buf_ptr += snprintf(buf_ptr, buf_end - buf_ptr, "]");
        }
        buf_ptr += snprintf(buf_ptr, buf_end - buf_ptr, ":%s", port);

        return buf;
}

/**
 * @brief counterpart of get_sockaddr_str()
 *
 * @param mode  mode to enforce 4, 6 or 0 (auto, as @ref resolve_addrinfo
 * returns v4-mapped IPv6 address for 0, the returned address will be aiways in
 * sockaddr_in6, either native or v4-mapped)
 *
 * @note
 * Even for dot-decimal IPv4 address, sockaddr_in6 struct with v4-mapped
 * address may be returned (Linux).
 *
 * Converts from textual representation of <host>:<port> to sockaddr_storage.
 * IPv6 numeric addresses must be enclosed in [] brackets.
 */
struct sockaddr_storage
get_sockaddr(const char *hostport, int mode)
{
        struct sockaddr_storage ret;
        socklen_t               socklen_unused = 0;
        char host[STR_LEN];

        ret.ss_family = AF_UNSPEC;
        const char *const rightmost_colon = strrchr(hostport, ':');
        if (rightmost_colon == NULL) {
                MSG(ERROR, "Address %s not in format host:port!\n", hostport);
                return ret;
        }
        if (rightmost_colon == hostport) {
                MSG(ERROR, "Empty host spec: %s!\n", hostport);
                return ret;
        }

        const char *const port_str = rightmost_colon + 1;
        char             *endptr   = NULL;
        long port = strtol(port_str, &endptr, 10);
        if (*endptr != '\0') {
                MSG(ERROR, "Wrong port value: %s\n", port_str);
                return ret;
        }
        if (port < 0 || port > UINT16_MAX) {
                MSG(ERROR, "Port %ld out of range!\n", port);
                return ret;
        }

        const char *host_start = hostport;
        const char *host_end = rightmost_colon;
        if (*host_start == '[') { // skip IPv6 []
                host_start += 1;
                if (host_end[-1] != ']') {
                        MSG(ERROR, "Malformed IPv6 host (missing ]): %s\n",
                            hostport);
                        return ret;
                }
                host_end -= 1;
        }

        const size_t len = host_end - host_start;
        snprintf(host, MIN(sizeof host, len + 1), "%s", host_start);
        resolve_addrinfo(host, port, &ret, &socklen_unused, &mode);
        return ret;
}

/**
 * If addr6 is not a v4-mapped address, return it.
 * Otherwise convert to sockaddr_in AF_INET address.
 */
static const struct sockaddr *
v4_unmap(const struct sockaddr_in6 *addr6, struct sockaddr_in *buf4)
{
        if (!IN6_IS_ADDR_V4MAPPED(&addr6->sin6_addr)) {
                return (const struct sockaddr *) addr6;
        }
        buf4->sin_family = AF_INET;
        buf4->sin_port = addr6->sin6_port;
        union {
                uint8_t bytes[4];
                uint32_t word;
        } u;
        for (int i = 0; i < 4; ++i) {
                u.bytes[i] = addr6->sin6_addr.s6_addr[12 + i];
        }
        buf4->sin_addr.s_addr = u.word;
        return (struct sockaddr *) buf4;
}

/**
 * @retval <0 struct represents "smaller" address (port)
 * @retval 0  addresses equal
 * @retval >0 struct represents "bigger" address (port)
 * @note
 * v4-mapped ipv6 address is considered eqaul to corresponding AF_INET addr
 */
int
sockaddr_compare(const struct sockaddr *x, const struct sockaddr *y)
{
        struct sockaddr_in tmp1;
        if (x->sa_family == AF_INET6) {
                x = v4_unmap((const void *) x, &tmp1);
        }
        struct sockaddr_in tmp2;
        if (y->sa_family == AF_INET6) {
                y = v4_unmap((const void *) y, &tmp2);
        }

        if (x->sa_family != y->sa_family) {
                return y->sa_family - x->sa_family;
        }

        if (x->sa_family == AF_INET) {
                const struct sockaddr_in *sin_x = (const void *) x;
                const struct sockaddr_in *sin_y = (const void *) y;

                if (sin_x->sin_addr.s_addr != sin_y->sin_addr.s_addr) {
                        return ntohl(sin_x->sin_addr.s_addr) <
                                       ntohl(sin_y->sin_addr.s_addr)
                                   ? -1
                                   : 1;
                }
                return ntohs(sin_y->sin_port) - ntohs(sin_x->sin_port);
        }
        if (x->sa_family == AF_INET6) {
                const struct sockaddr_in6 *sin_x = (const void *) x;
                const struct sockaddr_in6 *sin_y = (const void *) y;

                for (int i = 0; i < 16; ++i) {
                        if (sin_x->sin6_addr.s6_addr[i] !=
                            sin_y->sin6_addr.s6_addr[i]) {
                                return sin_y->sin6_addr.s6_addr[i] -
                                       sin_x->sin6_addr.s6_addr[i];
                        }
                }

                // sin6_scope_id is opaque so do not cope with endianity
                // (actually it is host order on both Linux and Windows)
                if (IN6_IS_ADDR_LINKLOCAL(&sin_x->sin6_addr) &&
                    sin_x->sin6_scope_id != sin_y->sin6_scope_id) {
                        return sin_x->sin6_scope_id < sin_y->sin6_scope_id ? -1
                                                                           : 1;
                }

                return ntohs(sin_y->sin6_port) - ntohs(sin_x->sin6_port);
        }
        MSG(ERROR, "Unsupported address class %d!", (int) x->sa_family);
        abort();
}

const char *ug_gai_strerror(int errcode)
{
#ifdef _WIN32
        UNUSED(errcode);
        // also `win_wstr_to_str(gai_strerrorW(errcode);` would work; it is localized, but with correct diacritics
        return get_win32_error(WSAGetLastError());
#else
        return gai_strerror(errcode);
#endif // ! defined _WIN32
}


/**
 * @file   utils/net.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016-2023 CESNET z.s.p.o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef WIN32
#include <tchar.h>
#else
#include <ifaddrs.h>
#include <netinet/ip.h>
#include <sys/types.h>
#endif

#include "utils/net.h"
#include "utils/windows.h"

#include "debug.h"

#define IPV4_LL_PREFIX ((169u<<8u) | 254u)

// MSW does not have the macro defined
#ifndef IN_LOOPBACKNET
#define IN_LOOPBACKNET 127
#endif

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
#ifdef WIN32
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
                        log_msg(LOG_LEVEL_ERROR, "Error: %s\n", get_win_error(dwRetVal));
                        if (pAddresses)
                                free(pAddresses);
                        return false;
		}
	}

	if (pAddresses) {
		free(pAddresses);
	}

	return true;
#else // ! defined WIN32
	size_t available_len = *len;
	*len = 0;
	struct ifaddrs* a = NULL;
	getifaddrs(&a);
	struct ifaddrs* p = a;
	while (NULL != p) {
		if ((ip_version == 0 && (p->ifa_addr->sa_family == AF_INET || p->ifa_addr->sa_family == AF_INET6)) ||
                                (ip_version == 4 && p->ifa_addr->sa_family == AF_INET) ||
                                (ip_version == 6 && p->ifa_addr->sa_family == AF_INET6)) {
			if (available_len >= sizeof addrs[0]) {
                                size_t sa_len = p->ifa_addr->sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
				memcpy(addrs, p->ifa_addr, sa_len);
				addrs += 1;
				*len += sizeof(addrs[0]);
				available_len -= sizeof(addrs[0]);
			} else {
				printf("Warning: insufficient space for all addresses.\n");
				return true;
			}
		}
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

unsigned get_sockaddr_addr_port(struct sockaddr *sa){
        unsigned port = 0;
        if (sa->sa_family == AF_INET6) {
                port = ntohs(((struct sockaddr_in6 *)(void *) sa)->sin6_port);
        } else if (sa->sa_family == AF_INET) {
                port = ntohs(((struct sockaddr_in *)(void *) sa)->sin_port);
        } else {
                return UINT_MAX;
        }

        return port;
}

void get_sockaddr_addr_str(struct sockaddr *sa, char *buf, size_t n){
        assert(n >= IN6_MAX_ASCII_LEN + 3 /* []: */ + 1 /* \0 */);
        const void *src = NULL;
        if (sa->sa_family == AF_INET6) {
                strcpy(buf, "[");
                src = &((struct sockaddr_in6 *)(void *) sa)->sin6_addr;
        } else if (sa->sa_family == AF_INET) {
                src = &((struct sockaddr_in *)(void *) sa)->sin_addr;
        } else {
                strcpy(buf, "(unknown)");
                return;
        }
        if (inet_ntop(sa->sa_family, src, buf + strlen(buf), n - strlen(buf)) == NULL) {
                perror("get_sockaddr_str");
                strcpy(buf, "(error)");
                return;
        }

        if (sa->sa_family == AF_INET6) {
                strcat(buf, "]");
        }
}

const char *get_sockaddr_str(struct sockaddr *sa)
{
        enum { ADDR_LEN = IN6_MAX_ASCII_LEN + 3 /* []: */ + 5 /* port */ + 1 /* \0 */ };
        _Thread_local static char addr[ADDR_LEN] = "";

        get_sockaddr_addr_str(sa, addr, sizeof(addr));

        unsigned port = get_sockaddr_addr_port(sa);
        if(port == UINT_MAX)
                return addr;
        snprintf(addr + strlen(addr), ADDR_LEN - strlen(addr), ":%u", port);

        return addr;
}

const char *ug_gai_strerror(int errcode)
{
#ifdef _WIN32
        UNUSED(errcode);
        // also `win_wstr_to_str(gai_strerrorW(errcode);` would work; it is localized, but with correct diacritics
        return get_win_error(WSAGetLastError());
#else
        return gai_strerror(errcode);
#endif // ! defined _WIN32
}


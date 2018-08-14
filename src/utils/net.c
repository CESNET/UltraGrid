/**
 * @file   utils/net.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016 CESNET z.s.p.o.
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

#ifndef WIN32
#include <netinet/ip.h>
#include <sys/types.h>
#include <ifaddrs.h>
#endif

#include "utils/net.h"

#include "debug.h"

// MSW does not have the macro defined
#ifndef IN_LOOPBACKNET
#define IN_LOOPBACKNET 127
#endif

bool is_addr_loopback(struct sockaddr *sa)
{
        switch (sa->sa_family) {
        case AF_UNIX:
                return true;
        case AF_INET:
        {
                struct sockaddr_in *sin = (struct sockaddr_in *) sa;
                uint32_t addr = ntohl(sin->sin_addr.s_addr);
                if ((addr >> 24) == IN_LOOPBACKNET) {
                        return true;
                } else {
                        return false;
                }
        }
        case AF_INET6:
        {
                struct sockaddr_in6 *sin = (struct sockaddr_in6 *) sa;
                if (IN6_IS_ADDR_V4MAPPED(&sin->sin6_addr)) {
                        uint32_t v4_addr = ntohl(*((uint32_t*)(sin->sin6_addr.s6_addr + 12)));
                        if ((v4_addr >> 24) == IN_LOOPBACKNET) {
                                return true;
                        } else {
                                return false;
                        }
                } else {
                        return IN6_IS_ADDR_LOOPBACK(&sin->sin6_addr);
                }
        }
        default:
                return false;
        }
}

bool is_host_loopback(const char *hostname)
{
	int gai_err;
	struct addrinfo hints, *ai;
        bool ret;

	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_DGRAM;

	if ((gai_err = getaddrinfo(hostname, NULL, &hints, &ai))) {
		error_msg("getaddrinfo: %s: %s\n", hostname,
				gai_strerror(gai_err));
		return false;
	}

        ret = is_addr_loopback(ai->ai_addr);
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
        } else {
                switch (ss.ss_family) {
                case AF_INET:
                        return ntohs(((struct sockaddr_in *) &ss)->sin_port);
                case AF_INET6:
                        return ntohs(((struct sockaddr_in6 *) &ss)->sin6_port);
                default:
                        return 0;
                }
        }
}

bool get_local_ipv4_addresses(struct sockaddr_in *addrs, size_t *len)
{
#ifdef WIN32
#define WORKING_BUFFER_SIZE 15000
#define MAX_TRIES 3
	/* Declare and initialize variables */
	DWORD dwSize = 0;
	DWORD dwRetVal = 0;
	unsigned int i = 0;
	size_t len_remaining = *len;
	*len = 0;

	LPVOID lpMsgBuf = NULL;

	PIP_ADAPTER_ADDRESSES pAddresses = NULL;
	ULONG outBufLen = 0;
	ULONG Iterations = 0;

	PIP_ADAPTER_ADDRESSES pCurrAddresses = NULL;
	PIP_ADAPTER_UNICAST_ADDRESS pUnicast = NULL;
	PIP_ADAPTER_ANYCAST_ADDRESS pAnycast = NULL;
	PIP_ADAPTER_MULTICAST_ADDRESS pMulticast = NULL;
	IP_ADAPTER_DNS_SERVER_ADDRESS *pDnServer = NULL;
	IP_ADAPTER_PREFIX *pPrefix = NULL;

	// Allocate a 15 KB buffer to start with.
	outBufLen = WORKING_BUFFER_SIZE;

	do {
		pAddresses = (IP_ADAPTER_ADDRESSES *) malloc(outBufLen);
		if (pAddresses == NULL) {
			printf
				("Memory allocation failed for IP_ADAPTER_ADDRESSES struct\n");
			return false;
		}

		dwRetVal =
			GetAdaptersAddresses(AF_INET, 0, NULL, pAddresses, &outBufLen);

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
			if (pUnicast != NULL) {
				for (i = 0; pUnicast != NULL; i++) {
					if (len_remaining >= sizeof(addrs[0])) {
						*addrs = *(struct sockaddr_in *) pUnicast->Address.lpSockaddr;
						addrs += 1;
						*len += sizeof(addrs[0]);
						len_remaining -= sizeof(addrs[0]);
					} else {
						printf("Warning: insufficient space for all addresses.\n");
						return true;
					}
					pUnicast = pUnicast->Next;
				}
			}

			pCurrAddresses = pCurrAddresses->Next;
		}
	} else {
		printf("Call to GetAdaptersAddresses failed with error: %d\n",
				dwRetVal);
		if (dwRetVal == ERROR_NO_DATA)
			printf("\tNo addresses were found for the requested parameters\n");
		else {
			if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
						FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
						NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
						// Default language
						(LPTSTR) & lpMsgBuf, 0, NULL)) {
				printf("\tError: %s", lpMsgBuf);
				LocalFree(lpMsgBuf);
				if (pAddresses)
					free(pAddresses);
				return false;
			}
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
		if (p->ifa_addr->sa_family == AF_INET) {
			if (available_len >= sizeof addrs[0]) {
				*addrs = *(struct sockaddr_in *) p->ifa_addr;
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


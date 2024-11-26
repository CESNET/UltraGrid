/**
 * @file   utils/net.h
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

#ifndef UTILS_NET_H_
#define UTILS_NET_H_

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif // __cplusplus

#ifdef _WIN32
#include <winsock2.h>   // dep for netioapi.h, ntddnidis.h
#include <ntddndis.h>   // dep for IF_NAMESIZE
#include <iphlpapi.h>   // for IF_NAMESIZE
#else
#include <net/if.h>     // for IF_NAMESIZE
#endif

enum {
        /// maximal length of textual representation of IPv6 address including
        /// eventual scope ID but without terminating NUL byte
        IN6_MAX_ASCII_LEN = 40 /* 32 nibbles + 7 colons + "%" */ + IF_NAMESIZE,
        /// not including terminating NUL
        IN_PORT_STR_LEN = 5,
        /**
         * buffer for host:port represenation. If IPv6 address is presented,
         * enclosed in []. Including terminating NUL byte.
         * @sa get_sockaddr_str
         */
        ADDR_STR_BUF_LEN =
            IN6_MAX_ASCII_LEN + 3 /* []: */ + IN_PORT_STR_LEN + 1 /* \0 */,
};
// RFC 6666 prefix 100::/64, suffix 'UltrGrS'
#define IN6_BLACKHOLE_SERVER_MODE_STR "100::556C:7472:4772:6453"

#ifdef __cplusplus
extern "C" {
#endif

struct sockaddr;
struct sockaddr_storage;
bool is_addr_linklocal(struct sockaddr *sa);
bool is_addr_loopback(struct sockaddr *sa);
bool is_addr_private(struct sockaddr *sa);
bool is_addr_multicast(const char *addr);
bool is_host_loopback(const char *hostname);
bool is_host_private(const char *hostname);
uint16_t socket_get_recv_port(int fd);
bool get_local_addresses(struct sockaddr_storage *addrs, size_t *len, int ip_version);
bool is_ipv6_supported(void);
char *get_sockaddr_str(const struct sockaddr *sa, unsigned sa_len, char *buf,
                       size_t n);
struct sockaddr_storage get_sockaddr(const char *hostport, int mode);
const char *ug_gai_strerror(int errcode);
int sockaddr_compare(const struct sockaddr *x, const struct sockaddr *y);

#ifdef _WIN32
#define CLOSESOCKET closesocket
#else
#define CLOSESOCKET close
#endif

#ifdef __cplusplus
}
#endif

#endif// UTILS_NET_H_


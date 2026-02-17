/**
 * @file   compat/net.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file includes the correct header for network-related functions
 * (also htonl and the family).
 */
/*
 * Copyright (c) 2024-2025 CESNET
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

#ifndef COMPAT_NET_H_EF7499D8_4939_4F86_A585_2EED8221D056
#define COMPAT_NET_H_EF7499D8_4939_4F86_A585_2EED8221D056

// IWYU pragma: begin_exports
#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET fd_t;
#else
#include <arpa/inet.h>      // for htonl, ntohl
#include <netdb.h>          // for getaddrinfo
#include <netinet/in.h>     // for sockaddr_in[6]
#include <sys/socket.h>     // for sockaddr, sockaddr_storage
typedef int fd_t;
#define INVALID_SOCKET (-1)
#endif
// IWYU pragma: end_exports

#ifndef _WIN32
#ifdef HAVE_IPv6

#ifdef HAVE_NETINET6_IN6_H
/* Expect IPV6_{JOIN,LEAVE}_GROUP in in6.h, otherwise expect */
/* IPV_{ADD,DROP}_MEMBERSHIP in in.h                         */
#include <netinet6/in6.h>
#else
#include <netinet/in.h>
#endif /* HAVE_NETINET_IN6_H */

#ifndef IPV6_ADD_MEMBERSHIP
#ifdef  IPV6_JOIN_GROUP
#define IPV6_ADD_MEMBERSHIP IPV6_JOIN_GROUP
#else
#error  No definition of IPV6_ADD_MEMBERSHIP
#endif /* IPV6_JOIN_GROUP     */
#endif /* IPV6_ADD_MEMBERSHIP */

#ifndef IPV6_DROP_MEMBERSHIP
#ifdef  IPV6_LEAVE_GROUP
#define IPV6_DROP_MEMBERSHIP IPV6_LEAVE_GROUP
#else
#error  No definition of IPV6_LEAVE_GROUP
#endif  /* IPV6_LEAVE_GROUP     */
#endif  /* IPV6_DROP_MEMBERSHIP */

#endif /* HAVE_IPv6 */
#endif /*_WIN32*/

#endif // defined COMPAT_NET_H_EF7499D8_4939_4F86_A585_2EED8221D056

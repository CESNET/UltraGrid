/*
 * FILE:    rtp/net_udp.h
 * AUTHORS: Colin Perkins
 * 
 * Copyright (c) 1998-2000 University College London
 * Copyright (c) 2005-2021 CESNET, z. s. p. o.
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
 * 
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */

#ifndef _NET_UDP
#define _NET_UDP

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

typedef struct _socket_udp socket_udp; 
struct socket_udp_local;

#if defined(__cplusplus)
extern "C" {
#endif

bool        udp_addr_valid(const char *addr);
socket_udp *udp_init(const char *addr, uint16_t rx_port, uint16_t tx_port, int ttl, int force_ip_version, bool multithreaded);
socket_udp *udp_init_if(const char *addr, const char *iface, uint16_t rx_port, uint16_t tx_port, int ttl, int force_ip_version, bool multithreaded);
void        udp_exit(socket_udp *s);

int         udp_peek(socket_udp *s, char *buffer, int buflen);
int         udp_recv(socket_udp *s, char *buffer, int buflen);
int         udp_recv_timeout(socket_udp *s, char *buffer, int buflen, struct timeval *timeout);
int         udp_recvfrom_timeout(socket_udp *s, char *buffer, int buflen,
                struct timeval *timeout,
                struct sockaddr *src_addr, socklen_t *addrlen);
int         udp_recvfrom(socket_udp *s, char *buffer, int buflen, struct sockaddr *src_addr, socklen_t *addrlen);
int         udp_send(socket_udp *s, char *buffer, int buflen);
int         udp_sendto(socket_udp *s, char *buffer, int buflen, struct sockaddr *dst_addr, socklen_t addrlen);

int         udp_recvv(socket_udp *s, struct msghdr *m);
void        udp_async_start(socket_udp *s, int nr_packets);
void        udp_async_wait(socket_udp *s);
#ifdef WIN32
int         udp_sendv(socket_udp *s, LPWSABUF vector, int count, void *d);
#else
int         udp_sendv(socket_udp *s, struct iovec *vector, int count, void *d);
#endif

char       *udp_host_addr(socket_udp *s);
int         udp_fd(socket_udp *s);

int         udp_select(struct timeval *timeout);
void	    udp_fd_zero(void);
void        udp_fd_set(socket_udp *s);
int         udp_fd_isset(socket_udp *s);

bool        udp_set_recv_buf(socket_udp *s, int size);
bool        udp_set_send_buf(socket_udp *s, int size);
void        udp_flush_recv_buf(socket_udp *s);

struct udp_fd_r {
        fd_set rfd;
        fd_t max_fd;
};

int         udp_select_r(struct timeval *timeout, struct udp_fd_r *);
void	    udp_fd_zero_r(struct udp_fd_r *);
void        udp_fd_set_r(socket_udp *s, struct udp_fd_r *);
int         udp_fd_isset_r(socket_udp *s, struct udp_fd_r *);

int         udp_recv_data(socket_udp * s, char **buffer);
int         udp_recvfrom_data(socket_udp * s, char **buffer,
                struct sockaddr *src_addr, socklen_t *addrlen);
bool        udp_not_empty(socket_udp *s, struct timeval *timeout);
int         udp_port_pair_is_free(int force_ip_version, int even_port);
bool        udp_is_ipv6(socket_udp *s);

void        socket_error(const char *msg, ...);

#ifdef WIN32
int         udp_send_wsa_async(socket_udp *s, char *buffer, int buflen, LPWSAOVERLAPPED_COMPLETION_ROUTINE, LPWSAOVERLAPPED);
#endif

struct socket_udp_local *udp_get_local(socket_udp *s);
int udp_get_udp_rx_port(socket_udp *s);
socket_udp *udp_init_with_local(struct socket_udp_local *l, struct sockaddr *sa, socklen_t len);
void udp_set_receiver(socket_udp *s, struct sockaddr *sa, socklen_t len);
bool udp_is_blackhole(socket_udp *s);

/*************************************************************************************************/
#if defined(__cplusplus)
}
#endif

#endif


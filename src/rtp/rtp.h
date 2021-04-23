/*
 * FILE:   rtp.h
 * AUTHOR: Colin Perkins <c.perkins@cs.ucl.ac.uk>
 *
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
 *      Department at University College London.
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
 * 
 */

#ifndef __RTP_H__
#define __RTP_H__

#ifdef __cplusplus
#include <cstddef> // offsetof
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h> // offsetof
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RTP_VERSION 2
#define RTP_MAX_MTU (16*1024) // VR specific change
#define RTP_MAX_PACKET_LEN (RTP_MAX_MTU+RTP_PACKET_HEADER_SIZE)

#if !defined(WORDS_BIGENDIAN) && !defined(WORDS_SMALLENDIAN)
#error RTP library requires WORDS_BIGENDIAN or WORDS_SMALLENDIAN to be defined.
#endif

struct rtp;

/* XXX gtkdoc doesn't seem to be able to handle functions that return
 * struct *'s. */
typedef struct rtp *rtp_t;

/*
 * TODO: this usage of bit-array is incorrect and the attribute "packed"
 * doesn't ensure proper behaviour. With GCC-incompatible mode of Intel
 * C Compiler this is broken.
 */
typedef struct __attribute__((packed)) {
	/* The following are pointers to the data in the packet as    */
	/* it came off the wire. The packet it read in such that the  */
	/* header maps onto the latter part of this struct, and the   */
	/* fields in this first part of the struct point into it. The */
	/* entire packet can be freed by freeing this struct, without */
	/* having to free the csrc, data and extn blocks separately.  */
	/* WARNING: Don't change the size of the first portion of the */
	/* struct without changing RTP_PACKET_HEADER_SIZE to match.   */
	uint32_t	*csrc;
	char		*data;
	int		 data_len;
	unsigned char	*extn;
	uint16_t	 extn_len;	/* Size of the extension in 32 bit words minus one */
	uint16_t	 extn_type;	/* Extension type field in the RTP packet header   */
	/* The following map directly onto the RTP packet header...   */
#ifdef WORDS_BIGENDIAN
	unsigned short   v:2;		/* packet type                */
	unsigned short   p:1;		/* padding flag               */
	unsigned short   x:1;		/* header extension flag      */
	unsigned short   cc:4;		/* CSRC count                 */
	unsigned short   m:1;		/* marker bit                 */
	unsigned short   pt:7;		/* payload type               */
#else
	unsigned short   cc:4;		/* CSRC count                 */
	unsigned short   x:1;		/* header extension flag      */
	unsigned short   p:1;		/* padding flag               */
	unsigned short   v:2;		/* packet type                */
	unsigned short   pt:7;		/* payload type               */
	unsigned short   m:1;		/* marker bit                 */
#endif
	uint16_t          seq;		/* sequence number            */
	uint32_t          ts;		/* timestamp                  */
	uint32_t          ssrc;		/* synchronization source     */
	/* The csrc list, header extension and data follow, but can't */
	/* be represented in the struct.                              */
	uint32_t	  send_ts;
        uint32_t          rtt;
} rtp_packet;

#define RTP_PACKET_HEADER_SIZE ((int) (offsetof(rtp_packet, extn_type) - offsetof(rtp_packet, csrc) + sizeof ((rtp_packet *) 0)->extn_type))

typedef struct {
	uint32_t         ssrc;
	uint32_t         ntp_sec;
	uint32_t         ntp_frac;
	uint32_t         rtp_ts;
	uint32_t         sender_pcount;
	uint32_t         sender_bcount;
} rtcp_sr;

typedef struct {
	uint32_t	ssrc;		/* The ssrc to which this RR pertains */
#ifdef WORDS_BIGENDIAN
	uint32_t	fract_lost:8;
	uint32_t	total_lost:24;
#else
	uint32_t	total_lost:24;
	uint32_t	fract_lost:8;
#endif	
	uint32_t	last_seq;
	uint32_t	jitter;
	uint32_t	lsr;
	uint32_t	dlsr;
} rtcp_rr;

typedef struct {
        uint32_t	ts;
	uint32_t	tdelay;
	uint32_t	x_recv;
	uint32_t	p;
} rtcp_rx;

typedef struct {
#ifdef WORDS_BIGENDIAN
	unsigned short  version:2;	/* RTP version            */
	unsigned short  p:1;		/* padding flag           */
	unsigned short  subtype:5;	/* application dependent  */
#else
	unsigned short  subtype:5;	/* application dependent  */
	unsigned short  p:1;		/* padding flag           */
	unsigned short  version:2;	/* RTP version            */
#endif
	unsigned short  pt:8;		/* packet type            */
	uint16_t        length;		/* packet length          */
	uint32_t        ssrc;
	char            name[4];        /* four ASCII characters  */
	char            data[1];        /* variable length field  */
} rtcp_app;

/* rtp_event type values. */
typedef enum {
        RX_RTP,
	RX_RTP_IOV,
        RX_SR,
        RX_RR,
	RX_TFRC_RX,	/* received TFRC extended report */
        RX_SDES,
        RX_BYE,         /* Source is leaving the session, database entry is still valid                           */
        SOURCE_CREATED,
        SOURCE_DELETED, /* Source has been removed from the database                                              */
        RX_RR_EMPTY,    /* We've received an empty reception report block                                         */
        RX_RTCP_START,  /* Processing a compound RTCP packet about to start. The SSRC is not valid in this event. */
        RX_RTCP_FINISH,	/* Processing a compound RTCP packet finished. The SSRC is not valid in this event.       */
        RR_TIMEOUT,
        RX_APP,
	PEEK_RTP
} rtp_event_type;

typedef struct {
	uint32_t	 ssrc;
	rtp_event_type	 type;
	void		*data;
} rtp_event;

/* Callback types */
typedef void (*rtp_callback)(struct rtp *session, rtp_event *e);
typedef rtcp_app* (*rtcp_app_callback)(struct rtp *session, uint32_t rtp_ts, int max_size);

/* SDES packet types... */
typedef enum  {
        RTCP_SDES_END   = 0,
        RTCP_SDES_CNAME = 1,
        RTCP_SDES_NAME  = 2,
        RTCP_SDES_EMAIL = 3,
        RTCP_SDES_PHONE = 4,
        RTCP_SDES_LOC   = 5,
        RTCP_SDES_TOOL  = 6,
        RTCP_SDES_NOTE  = 7,
        RTCP_SDES_PRIV  = 8
} rtcp_sdes_type;

typedef struct {
	uint8_t		type;		/* type of SDES item              */
	uint8_t		length;		/* length of SDES item (in bytes) */
	char		data[1];	/* text, not zero-terminated      */
} rtcp_sdes_item;

/* RTP options */
typedef enum {
        RTP_OPT_PROMISC 	  = 1,
        RTP_OPT_WEAK_VALIDATION	  = 2,
        RTP_OPT_FILTER_MY_PACKETS = 3,
	RTP_OPT_REUSE_PACKET_BUFS = 4,	/* Each data packet is written into the same buffer, */
	                                /* rather than malloc()ing a new buffer each time.   */
	RTP_OPT_PEEK              = 5,
	RTP_OPT_RECORD_SOURCE     = 6   /* Record network source (sockaddr_storage) at the   */
                                        /* end of the packet                                 */
} rtp_option;

struct socket_udp_local;

/* API */
rtp_t		rtp_init(const char *addr, 
			 uint16_t rx_port, uint16_t tx_port, 
			 int ttl, double rtcp_bw, 
			 int tfrc_on,
			 rtp_callback callback,
			 uint8_t *userdata,
                         int force_ip_version, bool multithreaded);
rtp_t		rtp_init_if(const char *addr, const char *iface,
			    uint16_t rx_port, uint16_t tx_port, 
			    int ttl, double rtcp_bw, 
			    int tfrc_on,
			    rtp_callback callback,
			    uint8_t *userdata,
                            int force_ip_version, bool multithreaded);
rtp_t            rtp_init_with_udp_socket(struct socket_udp_local *l, struct sockaddr *sa, socklen_t len, rtp_callback callback);

void		 rtp_send_bye(struct rtp *session);
void		 rtp_done(struct rtp *session);

int 		 rtp_set_option(struct rtp *session, rtp_option optname, int optval);
int 		 rtp_get_option(struct rtp *session, rtp_option optname, int *optval);

int 		 rtp_recv(struct rtp *session, 
			  struct timeval *timeout, uint32_t curr_rtp_ts) __attribute__((deprecated));
int 		 rtp_recv_r(struct rtp *session, 
			  struct timeval *timeout, uint32_t curr_rtp_ts);
int 		 rtcp_recv_r(struct rtp *session,
			  struct timeval *timeout, uint32_t curr_rtp_ts);
int 		 rtp_recv_poll_r(struct rtp **sessions, 
			  struct timeval *timeout, uint32_t curr_rtp_ts);
int 		 rtp_send_raw_rtp_data(struct rtp *session, char *buffer, int buffer_len);

int 		 rtp_send_data(struct rtp *session, 
			       uint32_t rtp_ts, char pt, int m, 
			       int cc, uint32_t csrc[], 
                               char *data, int data_len, 
			       char *extn, uint16_t extn_len, uint16_t extn_type);
int 		 rtp_send_data_hdr(struct rtp *session, 
			       uint32_t rtp_ts, char pt, int m, 
			       int cc, uint32_t csrc[], 
                               char *phdr, int phdr_len, 
                               char *data, int data_len,
			       char *extn, uint16_t extn_len, uint16_t extn_type);
void 		 rtp_send_ctrl(struct rtp *session, uint32_t rtp_ts, 
			       rtcp_app_callback appcallback, struct timeval curr_time);
void 		 rtp_send_rtcp_app(struct rtp *session, const char *name, int length, char *data);
void 		 rtp_update(struct rtp *session, struct timeval curr_time);

uint32_t	 rtp_my_ssrc(struct rtp *session);
int		 rtp_add_csrc(struct rtp *session, uint32_t csrc);
int		 rtp_del_csrc(struct rtp *session, uint32_t csrc);

int		 rtp_set_sdes(struct rtp *session, uint32_t ssrc, 
			      rtcp_sdes_type type, 
			      const char *value, int length);
const char	*rtp_get_sdes(struct rtp *session, uint32_t ssrc, rtcp_sdes_type type);

const rtcp_sr	*rtp_get_sr(struct rtp *session, uint32_t ssrc);
const rtcp_rr	*rtp_get_rr(struct rtp *session, uint32_t reporter, uint32_t reportee);

int              rtp_set_encryption_key(struct rtp *session, const char *passphrase);
int              rtp_set_my_ssrc(struct rtp *session, uint32_t ssrc);

uint8_t		*rtp_get_userdata(struct rtp *session);
void 		 rtp_set_recv_iov(struct rtp *session, struct msghdr *m);

int              rtp_set_recv_buf(struct rtp *session, int bufsize);
int              rtp_set_send_buf(struct rtp *session, int bufsize);

void             rtp_flush_recv_buf(struct rtp *session);
uint64_t         rtp_get_bytes_sent(struct rtp *session);
int              rtp_compute_fract_lost(struct rtp *session, uint32_t ssrc);
bool             rtp_is_ipv6(struct rtp *session);

/*
 * Async API - MSW specific
 *
 * Using async API hugely improves performance.
 * Usage is simple - prior to sending a bulk of packets (eg. video frame), rtp_async_start()
 * is started. Then, all packets are sent as usual, exept that neither data nor headers should
 * be altered up to rtp_async_wait() call, which waits upon completition of async operations
 * started after rtp_async_start(). Caller is responsible that rtp_send_data_hdr() is not called
 * more than nr_packet times.
 */
void             rtp_async_start(struct rtp *session, int nr_packets);
void             rtp_async_wait(struct rtp *session);

struct socket_udp_local *rtp_get_udp_local_socket(struct rtp *session);

#ifdef __cplusplus
}
#endif

#endif /* __RTP_H__ */

/*
 * FILE:    ihdtv/ihdtv.h
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifndef WIN32

#ifndef _IHDTV_H_
#define _IHDTV_H_

#include <ctype.h>

#include "video_codec.h"

void ihdtv_receive_frames(void);

typedef struct {
	uint32_t stream_id;
	uint32_t offset;	// offset into frame
	uint64_t frame_number;
	uint8_t data[65520];
} __attribute__((packed)) ihdtv_packet;

typedef struct {
	unsigned int rx_port_1;	// receiving ports
	unsigned int rx_port_2;

	unsigned int tx_port;

	unsigned int video_packets_per_frame;
	unsigned int video_packets_per_half_frame;
	uint64_t current_frame;	// internal use

	unsigned long bytes_per_frame;
	unsigned long bytes_per_half_frame;
	
	long video_data_per_packet;	// size of ihdtv_packet.data used
	long video_data_per_last_packet;	// size of the last packet of each half of a frame

// private:
	int rx_socket_1;
	int rx_socket_2;

	int tx_socket_1;
	int tx_socket_2;

	int pending_packet;	// flag of pending packet (ppacket)
	ihdtv_packet ppacket; // first packet of actual frame in case it came before the end of previous frame(partly lost in ether)
	unsigned int ppacket_size;
	unsigned int ppacket_offset_add;	// we use this to distinguish between upper and lower half of a frame


	struct sockaddr_in target_sockaddress_1;
	struct sockaddr_in target_sockaddress_2;

	int check_peer_ip;	// if set, in rx mode we check, whether the sender address is one of peers addresses
	struct in_addr peer_address_1;
	struct in_addr peer_address_2;
} ihdtv_connection;

int
ihdtv_init_rx_session(ihdtv_connection *connection, const char* address_1, const char* address_2, unsigned int port1, unsigned int port2, long mtu);

int
ihdtv_init_tx_session(ihdtv_connection* connection, const char *address_1, const char *address_2, long mtu);


int
ihdtv_receive(ihdtv_connection *rx_connection, char *buffer, unsigned long buffer_length);

int
ihdtv_send(ihdtv_connection *tx_connection, struct video_frame *tx_frame, unsigned long buffer_length);

#endif

#endif // WIN32


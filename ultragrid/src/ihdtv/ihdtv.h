#ifndef _IHDTV_H_
#define _IHDTV_H_

#include <stdint.h>
#include <ctype.h>
#include <sys/socket.h>

#include "video_types.h"

void ihdtv_recieve_frames(void);

typedef struct {
	uint32_t stream_id;
	uint32_t offset;	// offset into frame
	uint64_t frame_number;
	uint8_t data[65520];
} __attribute__((packed)) ihdtv_packet;

typedef struct {
	unsigned int rx_port_1;	// recieving ports
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
ihdtv_recieve(ihdtv_connection *rx_connection, char *buffer, unsigned long buffer_length);

int
ihdtv_send(ihdtv_connection *tx_connection, struct video_frame *tx_frame, unsigned long buffer_length);

#endif

/*
 * FILE:    ihdtv/ihdtv.c
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>

#include "ihdtv.h"
#include "debug.h"

void init_reciever(void);

int
ihdtv_init_rx_session(ihdtv_connection * connection, const char *address_1,
                      const char *address_2, unsigned int port1,
                      unsigned int port2, long mtu)
{
        connection->rx_port_1 = port1;
        connection->rx_port_2 = port2;
        connection->tx_port = 0;
        connection->tx_socket_1 = 0;
        connection->tx_socket_2 = 0;

        connection->check_peer_ip = 0;

        if ((connection->rx_socket_1 = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
                fprintf(stderr, "Error creating first recieving socket");
                exit(1);
        }

        if ((connection->rx_socket_2 = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
                fprintf(stderr, "Error creating second recieving socket");
                exit(1);
        }

        if (address_1 != NULL)  // we want to accept packets only from certain address so we connect() the socket
        {
                struct hostent *target = gethostbyname(address_1);      // FIXME: should free?
                if (target == NULL) {
                        printf("Unknown target 1");
                        exit(1);        // FIXME: maybe should just return but it requires not returnin ihdtv_connection
                }

                connection->peer_address_1 = *(struct in_addr *)target->h_addr;

                connection->check_peer_ip = 1;
        }

        if (address_2 != NULL)  // we want to accept packets only from certain address so we connect() the socket
        {
                struct hostent *target = gethostbyname(address_2);      // FIXME: should free?
                if (target == NULL) {
                        printf("Unknown target 2");
                        exit(1);        // FIXME: maybe should just return but it requires not returnin ihdtv_connection
                }

                connection->peer_address_2 = *(struct in_addr *)target->h_addr;

                connection->check_peer_ip = 1;
        }

        struct sockaddr_in rx_address;
        rx_address.sin_family = AF_INET;
        rx_address.sin_addr.s_addr = htonl(INADDR_ANY);
        rx_address.sin_port = htons(connection->rx_port_1);
        memset(rx_address.sin_zero, '\0', sizeof(rx_address.sin_zero));
        if (bind
            (connection->rx_socket_1, (struct sockaddr *)&rx_address,
             sizeof(rx_address)) == -1) {
                fprintf(stderr, "Error binding reciever to the first socket");
                exit(1);
        }

        rx_address.sin_family = AF_INET;
        rx_address.sin_addr.s_addr = htonl(INADDR_ANY);
        rx_address.sin_port = htons(connection->rx_port_2);
        memset(rx_address.sin_zero, '\0', sizeof(rx_address.sin_zero));
        if (bind
            (connection->rx_socket_2, (struct sockaddr *)&rx_address,
             sizeof(rx_address)) == -1) {
                fprintf(stderr, "Error binding reciever to the second socket");
                exit(1);
        }

        connection->video_data_per_packet = mtu;
        if (mtu == 0) {
                fprintf(stderr, "Error: mtu must be > 0");
                return -1;
        }

        connection->bytes_per_frame = 5529600;
        connection->bytes_per_half_frame = connection->bytes_per_frame / 2;

        // we will count the number of packets/frame and the size of the last packet
        connection->video_packets_per_half_frame =
            connection->bytes_per_half_frame /
            connection->video_data_per_packet;
        if (connection->bytes_per_half_frame % connection->video_data_per_packet)       // we have the final packet of each half of the frame smaller than video_data_per_packet
        {
                connection->video_data_per_last_packet =
                    connection->bytes_per_half_frame %
                    connection->video_data_per_packet;
                connection->video_packets_per_half_frame++;
        } else {
                connection->video_data_per_last_packet =
                    connection->video_data_per_packet;
        }

        connection->video_packets_per_frame =
            2 * connection->video_packets_per_half_frame;
//      fprintf(stderr,"Video_packets_per_frame=%u, video_data_per_packet=%d, video_data_per_last_packet %d\n", connection->video_packets_per_frame, (int)connection->video_data_per_packet, connection->video_data_per_last_packet);

/*
	connection->video_packets_per_frame = 682;
	connection->bytes_per_frame = 5529600;	// 1920 * 1080 v210
	connection->bytes_per_half_frame = connection->bytes_per_frame / 2;
*/

        connection->current_frame = 0;
        connection->pending_packet = 0;

        // set sockets to nonblocking(will utilitize 100% of a CPU)
        if (fcntl(connection->rx_socket_1, F_SETFL, O_NONBLOCK) < 0) {
                fprintf(stderr, "Error setting rx_socket_1 as nonblocking\n");
                exit(1);
        }

        if (fcntl(connection->rx_socket_2, F_SETFL, O_NONBLOCK) < 0) {
                fprintf(stderr, "Error setting rx_socket_2 as nonblocking\n");
                exit(1);
        }

        return 0;
}

int
ihdtv_init_tx_session(ihdtv_connection * connection, const char *address_1,
                      const char *address_2, long mtu)
{
        connection->rx_port_1 = 0;
        connection->rx_port_2 = 0;
        connection->tx_port = 0;        // not used so far
        connection->tx_socket_1 = 0;
        connection->tx_socket_2 = 0;

        struct hostent *target_1 = gethostbyname(address_1);
        if (target_1 == NULL) {
                printf("Unknown target 1");
                exit(1);        // FIXME: maybe should just return but it requires not returnin ihdtv_connection
        }

        struct hostent *target_2 = gethostbyname(address_2);
        if (target_2 == NULL) {
                printf("Unknown target 2");
                exit(1);        // FIXME: maybe should just return but it requires not returnin ihdtv_connection
        }

        if ((connection->tx_socket_1 = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
                fprintf(stderr, "Error creating sending socket 1");
                exit(1);
        }
        if ((connection->tx_socket_2 = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
                fprintf(stderr, "Error creating sending socket 2");
                exit(1);
        }

        connection->target_sockaddress_1.sin_family = AF_INET;
        connection->target_sockaddress_1.sin_port = htons(3000);        // FIXME: target port
        connection->target_sockaddress_1.sin_addr =
            *((struct in_addr *)target_1->h_addr);
        memset(connection->target_sockaddress_1.sin_zero, '\0',
               sizeof(connection->target_sockaddress_1.sin_zero));
        connection->target_sockaddress_2.sin_family = AF_INET;
        connection->target_sockaddress_2.sin_port = htons(3001);        // FIXME: target port
        connection->target_sockaddress_2.sin_addr =
            *((struct in_addr *)target_2->h_addr);
        memset(connection->target_sockaddress_2.sin_zero, '\0',
               sizeof(connection->target_sockaddress_2.sin_zero));

        connection->video_data_per_packet = mtu;
        if (mtu == 0) {
                fprintf(stderr, "Error: mtu must be > 0");
                return -1;
        }

        connection->bytes_per_frame = 5529600;
        connection->bytes_per_half_frame = connection->bytes_per_frame / 2;

        // we will count the number of packets/frame and the size of the last packet
        connection->video_packets_per_half_frame =
            connection->bytes_per_half_frame /
            connection->video_data_per_packet;
        if (connection->bytes_per_half_frame % connection->video_data_per_packet)       // we have the final packet of each half of the frame smaller than video_data_per_packet
        {
                connection->video_data_per_last_packet =
                    connection->bytes_per_half_frame %
                    connection->video_data_per_packet;
                connection->video_packets_per_half_frame++;
        } else {
                connection->video_data_per_last_packet =
                    connection->video_data_per_packet;
        }

        connection->video_packets_per_frame =
            2 * connection->video_packets_per_half_frame;

//      fprintf(stderr,"Video_packets_per_frame=%u, video_data_per_packet=%d, video_data_per_last_packet %d\n", connection->video_packets_per_frame, (int)connection->video_data_per_packet, connection->video_data_per_last_packet);

        connection->current_frame = 0;
        connection->pending_packet = 0;

        return 0;
}

int
ihdtv_send(ihdtv_connection * connection, struct video_frame *tx_frame,
           unsigned long buffer_length)
{
        UNUSED(buffer_length);
        struct msghdr msg1, msg2;

        msg1.msg_name = &connection->target_sockaddress_1;
        msg2.msg_name = &connection->target_sockaddress_2;

        msg1.msg_namelen = sizeof(connection->target_sockaddress_1);
        msg2.msg_namelen = sizeof(connection->target_sockaddress_2);

        // header + data
        struct iovec vector1[2];
        struct iovec vector2[2];

        /* TODO: is this correct ?? */
        msg1.msg_iov = (struct iovec *) &vector1;
        msg2.msg_iov = (struct iovec *) &vector2;

        msg1.msg_iovlen = 2;
        msg2.msg_iovlen = 2;

        msg1.msg_control = 0;
        msg2.msg_control = 0;
        msg1.msg_controllen = 0;
        msg2.msg_controllen = 0;
        msg1.msg_flags = 0;
        msg2.msg_flags = 0;

        // only used for headers
        ihdtv_packet packet_1;
        ihdtv_packet packet_2;

        packet_1.frame_number = connection->current_frame;
        packet_2.frame_number = connection->current_frame;

        packet_1.stream_id = 0; // top half
        packet_2.stream_id = 1; // bottom half

        unsigned int offset_absolut_1 = 0;      // offset to buffer ( in bytes )
        unsigned int offset_absolut_2 = connection->bytes_per_half_frame;

        unsigned int packet_relative_offset = 0;        // this is the number we put into buffer

        unsigned int i;
        for (i = 0; i < connection->video_packets_per_half_frame - 1; i++) {
                packet_1.offset = packet_relative_offset;
                vector1[0].iov_base = &packet_1;
                vector1[0].iov_len = 16;        // we only use the header
                vector1[1].iov_base = vf_get_tile(tx_frame, 0)->data + offset_absolut_1;
                vector1[1].iov_len = connection->video_data_per_packet;
                if (sendmsg(connection->tx_socket_1, &msg1, 0) == -1) {
                        perror("Sending data to address 1");
                        exit(1);
                }

                packet_2.offset = packet_relative_offset;
                vector2[0].iov_base = &packet_2;
                vector2[0].iov_len = 16;        // we only use the header
                vector2[1].iov_base = vf_get_tile(tx_frame, 0)->data + offset_absolut_2;
                vector2[1].iov_len = connection->video_data_per_packet;
                if (sendmsg(connection->tx_socket_2, &msg2, 0) == -1) {
                        perror("Sending data to address 2");
                        exit(1);
                }

                ++packet_relative_offset;
                offset_absolut_1 += connection->video_data_per_packet;
                offset_absolut_2 += connection->video_data_per_packet;
        }

        packet_1.offset = packet_relative_offset;
        vector1[0].iov_base = &packet_1;
        vector1[0].iov_len = 16;        // we only use the header
        vector1[1].iov_base = vf_get_tile(tx_frame, 0)->data + offset_absolut_1;
        vector1[1].iov_len = connection->video_data_per_last_packet;
        if (sendmsg(connection->tx_socket_1, &msg1, 0) == -1) {
                perror("Sending data to address 1");
                exit(1);
        }

        packet_2.offset = packet_relative_offset;
        vector2[0].iov_base = &packet_2;
        vector2[0].iov_len = 16;        // we only use the header
        vector2[1].iov_base = vf_get_tile(tx_frame, 0)->data + offset_absolut_2;
        vector2[1].iov_len = connection->video_data_per_last_packet;
        if (sendmsg(connection->tx_socket_2, &msg2, 0) == -1) {
                perror("Sending data to address 2");
                exit(1);
        }

        ++(connection->current_frame);

        return 0;
}

inline static int packet_to_buffer(const ihdtv_connection * connection, char *buffer, const unsigned long buffer_length, const ihdtv_packet * packet, const int packet_length)  // returns number of written packets (1 or 0)
{
        if (buffer_length < (unsigned long int)
            (packet->offset * connection->video_data_per_packet +
             (packet_length - 16)))
                return 0;

        if (packet->stream_id == 0) {
                memcpy(buffer +
                       (packet->offset) * connection->video_data_per_packet,
                       packet->data, packet_length - 16);
                return 1;
        }

        if (packet->stream_id == 1) {
                memcpy(buffer +
                       (packet->offset) * connection->video_data_per_packet +
                       connection->bytes_per_half_frame, packet->data,
                       packet_length - 16);
                return 1;
        }

        return 0;
}

int
ihdtv_recieve(ihdtv_connection * connection, char *buffer,
              const unsigned long buffer_length)
{
        int packets_number = 0;

        ihdtv_packet packet;
        int num_bytes;
        struct sockaddr_in sender_address;      // where did the packets come from
        socklen_t sender_address_length = sizeof(sender_address);

        int dont_check_ip = !(connection->check_peer_ip);

        if (buffer == NULL) {
                fprintf(stderr,
                        "iHDTV reciever: buffer is empty, not recieving.\n");
                return -1;
        }
//      memset(buffer, 0, buffer_length);       // make holes in frame more visible -- your computer might not be fast enough for this, if you experience performance problems, try to comment this out

        // if we have some data from previous session, we use it
        if (connection->pending_packet) {
                packet_to_buffer(connection, buffer, buffer_length,
                                 &(connection->ppacket),
                                 connection->ppacket_size);
//              printf("processing pending packet %ld\n", connection->ppacket.frame_number);    // FIXME: delete this

                connection->pending_packet = 0;
                ++packets_number;
        }

        while (1) {
                if ((num_bytes =
                     recvfrom(connection->rx_socket_1, &packet, sizeof(packet),
                              0, (struct sockaddr *)&sender_address,
                              &sender_address_length)) > -1) {
                        if ((dont_check_ip)
                            || (connection->peer_address_1.s_addr ==
                                sender_address.sin_addr.s_addr)
                            || (connection->peer_address_2.s_addr ==
                                sender_address.sin_addr.s_addr)) {
                                if (packet.stream_id == 0 || packet.stream_id == 1) {   // it is strange, but ihdtv seems to be sending video and audio with different frame numbers in some cases, so we only work with video here
                                        if ((connection->current_frame < packet.frame_number) || (packet.frame_number + 10 < connection->current_frame))        // we just recieved frame we didn't expect
                                        {
//                                              printf("current frame: %llu (packets: %u)   incoming frame: %llu\n", connection->current_frame, packets_number, packet.frame_number);
                                                connection->current_frame =
                                                    packet.frame_number;
                                                connection->pending_packet++;
                                                connection->ppacket = packet;
                                                connection->ppacket_size =
                                                    num_bytes;

                                                break;
                                        }

                                        if (packet.frame_number ==
                                            connection->current_frame)
                                                packets_number +=
                                                    packet_to_buffer(connection,
                                                                     buffer,
                                                                     buffer_length,
                                                                     &packet,
                                                                     num_bytes);

                                        if ((unsigned int) packets_number == connection->video_packets_per_frame)      // we have all packets of a frame
                                        {
                                                connection->current_frame++;
                                                break;
                                        }
                                }
                        }

                } else if (errno != EWOULDBLOCK) {
                        fprintf(stderr, "Error recieving packet\n");
                        exit(1);
                }

                if ((num_bytes =
                     recvfrom(connection->rx_socket_2, &packet, sizeof(packet),
                              0, (struct sockaddr *)&sender_address,
                              &sender_address_length)) > -1) {
                        if ((dont_check_ip)
                            || (connection->peer_address_1.s_addr ==
                                sender_address.sin_addr.s_addr)
                            || (connection->peer_address_2.s_addr ==
                                sender_address.sin_addr.s_addr)) {
                                if (packet.stream_id == 0 || packet.stream_id == 1) {   // it is strange, but ihdtv seems to be sending video and audio with different frame numbers in some cases, so we only work with video here
                                        if ((connection->current_frame < packet.frame_number) || (packet.frame_number + 10 < connection->current_frame))        // we just recieved frame we didn't expect
                                        {
//                                              printf("current frame: %llu (packets: %u)   incoming frame: %llu\n", connection->current_frame, packets_number, packet.frame_number);
                                                connection->current_frame =
                                                    packet.frame_number;
                                                connection->pending_packet++;
                                                connection->ppacket = packet;
                                                connection->ppacket_size =
                                                    num_bytes;

                                                break;
                                        }
                                        if (packet.frame_number ==
                                            connection->current_frame)
                                                packets_number +=
                                                    packet_to_buffer(connection,
                                                                     buffer,
                                                                     buffer_length,
                                                                     &packet,
                                                                     num_bytes);

                                        if ((unsigned int ) packets_number == connection->video_packets_per_frame)      // we have all packets of a frame
                                        {
                                                connection->current_frame++;
                                                break;
                                        }
                                }
                        }

                } else if (errno != EWOULDBLOCK) {
                        fprintf(stderr, "Error recieving packet\n");
                        exit(1);
                }
        }

        return 0;
}

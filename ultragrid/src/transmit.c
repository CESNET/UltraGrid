/*
 * FILE:     transmit.c
 * AUTHOR:   Colin Perkins <csp@csperkins.org>
 * MODIFIED: Ladan Gharai
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 * $Revision: 1.3 $
 * $Date: 2009/05/20 14:55:24 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "video_types.h"
#include "rtp/rtp.h"
#include "tv.h"
#include "transmit.h"
#include "host.h"

#define TRANSMIT_MAGIC	0xe80ab15f
#define DXT_HEIGHT 1080/4
#define DXT_WIDTH 1920/4
#define DXT_DEPTH 8

extern long packet_rate;

#if HAVE_MACOSX
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_usec - start.tv_usec) * 1000L
#else /* HAVE_MACOSX */
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = stop.tv_nsec - start.tv_nsec
#endif /* HAVE_MACOSX */

struct video_tx {
	uint32_t	 magic;
	unsigned	 mtu;
};

struct video_tx *
tx_init(unsigned mtu)
{
	struct video_tx	*tx;

	tx = (struct video_tx *) malloc(sizeof(struct video_tx));
	if (tx != NULL) {
		tx->magic = TRANSMIT_MAGIC;
		tx->mtu   = mtu;
	}
	return tx;
}

void
tx_done(struct video_tx *tx)
{
	assert(tx->magic == TRANSMIT_MAGIC);
	free(tx);
}

#define HD_WIDTH hd_size_x
#define HD_HEIGHT hd_size_y
#define HD_DEPTH    hd_color_bpp
#define TS_WRAP     4294967296

typedef struct {
	uint16_t	scan_line;	/* pixels */
	uint16_t	scan_offset;	/* pixels */
	uint16_t	length;		/* octets */
	uint16_t	flags;
} payload_hdr_t;

void
dxt_tx_send(struct video_tx *tx, struct video_frame *frame, struct rtp *rtp_session)
{
	int		 m, x, y, first_x, first_y, data_len, l, payload_count, octets_left_this_line, octets_left_this_packet;
	payload_hdr_t	 payload_hdr[10];
	int		 pt = 96;	/* A dynamic payload type for the tests... */
	static uint32_t	 ts = 0;
	char		*data;

	assert(tx->magic == TRANSMIT_MAGIC);

	/* Note: We are operating on DXT macroblocks instead of individual pixles. A macro-block corresponds
	 * to a 4x4 block of real pixels, with 64 bits (8 bytes) as a length. We set this to our depth to ensure
	 * that we only send a complete macro block. --iwsmith
	 */ 
	assert(frame->data_len == 1920*1080/16*8);
	data = frame->data + 1920*1080/16*8;

	m = 0;
	x = 0; 
	y = 0;
	payload_count = 0;
	data_len = 0;
	first_x  = x;
	first_y  = y;
	ts = get_local_mediatime();
	do {
		if (payload_count == 0) {
			data_len = 0;
			first_x  = x;
			first_y  = y;
		}

		octets_left_this_line   = (DXT_WIDTH - x) * DXT_DEPTH;
		octets_left_this_packet = tx->mtu - 40 - data_len - (8 * (payload_count + 1));
		if (octets_left_this_packet < octets_left_this_line) {
			l = octets_left_this_packet;
		} else {
			l = octets_left_this_line;
		}
		while ((l % DXT_DEPTH) != 0) {	/* Only send complete pixels */
			l--;
		}

		payload_hdr[payload_count].scan_line   = htons(y);
		payload_hdr[payload_count].scan_offset = htons(x);
		payload_hdr[payload_count].length      = htons(l);
		payload_hdr[payload_count].flags       = htons(0);
		payload_count++;

		data_len = data_len + l;

		x += (l / DXT_DEPTH);
		if (x == (int)DXT_WIDTH) {
			x = 0;
			y++;
		}
		if (y == (int)DXT_HEIGHT) {
			m = 1;
		}

		/* Is it time to send this packet? */
		if ((y == (int)DXT_HEIGHT) || (payload_count == 10) || ((40u + data_len + (8u * (payload_count + 1)) + DXT_DEPTH) > tx->mtu)) {
#if HAVE_MACOSX
			struct timeval start, stop;
#else /* HAVE_MACOSX */
			struct timespec start, stop;
#endif /* HAVE_MACOSX */
			long delta;
			payload_hdr[payload_count - 1].flags = htons(1<<15);
			data = frame->data + (first_y * DXT_WIDTH * DXT_DEPTH) + (first_x * DXT_DEPTH);
			GET_STARTTIME;
			rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0, (char *) payload_hdr, 8 * payload_count, data, data_len, 0, 0, 0);
			do {
				GET_STOPTIME;
				GET_DELTA;
				if(delta < 0)
					delta += 1000000000L;
			} while(packet_rate - delta > 0);
			payload_count = 0;
		}
	} while (y < (int)DXT_HEIGHT);
}

void
tx_send(struct video_tx *tx, struct video_frame *frame, struct rtp *rtp_session)
{
	int		 m, x, y, first_x, first_y, data_len, l, payload_count, octets_left_this_line, octets_left_this_packet;
	payload_hdr_t	 payload_hdr[10];
	int		 pt = 96;	/* A dynamic payload type for the tests... */
	static uint32_t	 ts = 0;
	char		*data;

	assert(tx->magic == TRANSMIT_MAGIC);

	assert(frame->data_len == (hd_size_x * hd_size_y * hd_color_bpp));
	data = frame->data + (hd_size_x * hd_size_y * hd_color_bpp);

	m = 0;
	x = 0; 
	y = 0;
	payload_count = 0;
	data_len = 0;
	first_x  = x;
	first_y  = y;
	ts = get_local_mediatime();
	do {
		if (payload_count == 0) {
			data_len = 0;
			first_x  = x;
			first_y  = y;
		}

		octets_left_this_line   = (HD_WIDTH - x) * HD_DEPTH;
		octets_left_this_packet = tx->mtu - 40 - data_len - (8 * (payload_count + 1));
		if (octets_left_this_packet < octets_left_this_line) {
			l = octets_left_this_packet;
		} else {
			l = octets_left_this_line;
		}
		while ((l % HD_DEPTH) != 0) {	/* Only send complete pixels */
			l--;
		}

		payload_hdr[payload_count].scan_line   = htons(y);
		payload_hdr[payload_count].scan_offset = htons(x);
		payload_hdr[payload_count].length      = htons(l);
		payload_hdr[payload_count].flags       = htons(0);
		payload_count++;

		data_len = data_len + l;

		x += (l / HD_DEPTH);
		if (x == (int)HD_WIDTH) {
			x = 0;
			y++;
		}
		if (y == (int)HD_HEIGHT) {
			m = 1;
		}

		/* Is it time to send this packet? */
		if ((y == (int)HD_HEIGHT) || (payload_count == 10) || ((40u + data_len + (8u * (payload_count + 1)) + HD_DEPTH) > tx->mtu)) {
#if HAVE_MACOSX
			struct timeval start, stop;
#else /* HAVE_MACOSX */
			struct timespec start, stop;
#endif /* HAVE_MACOSX */
			long delta;
			payload_hdr[payload_count - 1].flags = htons(1<<15);
			data = frame->data + (first_y * HD_WIDTH * HD_DEPTH) + (first_x * HD_DEPTH);
			GET_STARTTIME;
			rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0, (char *) payload_hdr, 8 * payload_count, data, data_len, 0, 0, 0);
			do {
				GET_STOPTIME;
				GET_DELTA;
				if(delta < 0)
					delta += 1000000000L;
			} while(packet_rate - delta > 0);
			payload_count = 0;
		}
	} while (y < (int)HD_HEIGHT);
}

void
audio_tx_send(struct rtp *rtp_session, audio_frame *buffer)
{
      audio_frame_to_network_buffer(buffer->tmp_buffer, buffer);

      //uint32_t timestamp = get_local_mediatime();
      static uint32_t timestamp;
      timestamp++;

      int marker_bit = 0;     // FIXME: probably should define last packet of a frame, but is payload dependand so...think this through
      
      rtp_send_data(rtp_session, timestamp, audio_payload_type, marker_bit,
                      0, /* contributing sources */
                      0, /* contributing sources length*/
                      buffer->tmp_buffer, buffer->samples_per_channel * 3 * 8, 0, 0, 0);
}

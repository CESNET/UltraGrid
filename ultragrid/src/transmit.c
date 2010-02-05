/*
 * FILE:     transmit.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 * $Revision: 1.5.2.6 $
 * $Date: 2010/02/05 13:56:49 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "audio/audio.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tv.h"
#include "transmit.h"
#include "host.h"
#include "video_codec.h"

#define TRANSMIT_MAGIC	0xe80ab15f

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

void
tx_send(struct video_tx *tx, struct video_frame *frame, struct rtp *rtp_session)
{
	int		        m, data_len;
	payload_hdr_t	payload_hdr;
	int		        pt = 96;	/* A dynamic payload type for the tests... */
	static uint32_t	ts = 0;
	char		    *data;
        unsigned int    pos;
#if HAVE_MACOSX
	struct timeval  start, stop;
#else /* HAVE_MACOSX */
        struct timespec start, stop;
#endif /* HAVE_MACOSX */
	long            delta;

	assert(tx->magic == TRANSMIT_MAGIC);

	m = 0;
	ts = get_local_mediatime();
        pos = 0;

        payload_hdr.width = htons(frame->width);
        payload_hdr.height = htons(frame->height);
        payload_hdr.colorspc = frame->color_spec;

	do {
		payload_hdr.offset = htonl(pos);
		payload_hdr.flags  = htons(1<<15);

		data = frame->data + pos;
                data_len = tx->mtu - 40 - (sizeof(payload_hdr_t));
                if(pos + data_len > frame->data_len) {
                    m = 1;
                    data_len = frame->data_len - pos;
                }
                pos += data_len;
		payload_hdr.length = htons(data_len);
		GET_STARTTIME;
    	        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0, (char *)&payload_hdr, 
                                sizeof(payload_hdr_t), data, data_len, 0, 0, 0);
		do {
			GET_STOPTIME;
			GET_DELTA;
			if(delta < 0)
				delta += 1000000000L;
		} while(packet_rate - delta > 0);

	} while (pos < frame->data_len);
}

#ifdef HAVE_AUDIO
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
#endif /* HAVE_AUDIO */

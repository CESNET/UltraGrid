/*
 * AUTHOR:   N.Cihan Tas
 * MODIFIED: Ladan Gharai
 *           Colin Perkins
 *           Martin Benes     <martinbenesh@gmail.com>
 *           Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *           Petr Holub       <hopet@ics.muni.cz>
 *           Milos Liska      <xliska@fi.muni.cz>
 *           Jiri Matela      <matela@ics.muni.cz>
 *           Dalibor Matura   <255899@mail.muni.cz>
 *           Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 * 
 * This file implements a linked list for the playout buffer.
 *
 * Copyright (c) 2003-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/ptime.h"
#include "rtp/pbuf.h"
#include "rtp/audio_decoders.h"
#include "audio/audio.h"
#include "audio/utils.h"

#include "utils/packet_counter.h"

#include <time.h>

#define AUDIO_DECODER_MAGIC 0x12ab332bu
struct state_audio_decoder {
        uint32_t magic;

        struct timeval t0;

        struct packet_counter *packet_counter;
};

void *audio_decoder_init(void)
{
        struct state_audio_decoder *s;

        s = (struct state_audio_decoder *) malloc(sizeof(struct state_audio_decoder));
        s->magic = AUDIO_DECODER_MAGIC;

        gettimeofday(&s->t0, NULL);
        s->packet_counter = NULL;

        return s;
}

void audio_decoder_destroy(void *state)
{
        struct state_audio_decoder *s = (struct state_audio_decoder *) state;

        assert(s != NULL);
        assert(s->magic == AUDIO_DECODER_MAGIC);

        packet_counter_destroy(s->packet_counter);

        free(s);
}

int decode_audio_frame(struct coded_data *cdata, void *data)
{
        struct pbuf_audio_data *s = (struct pbuf_audio_data *) data;
        struct audio_frame *buffer = s->buffer;
        struct state_audio_decoder *decoder = s->decoder;

        int total_channels = 0;
        int bps, sample_rate, channel;
        static int prints = 0;

        while (cdata != NULL) {
                char *data;
                audio_payload_hdr_t *hdr = 
                        (audio_payload_hdr_t *) cdata->data->data;
                        
                /* we receive last channel first (with m bit, last packet) */
                /* thus can be set only with m-bit packet */
                if(cdata->data->m) {
                        total_channels = ((ntohl(hdr->substream_bufnum) >> 22) & 0x3ff) + 1;
                }
                assert(total_channels > 0);

                channel = (ntohl(hdr->substream_bufnum) >> 22) & 0x3ff;
                int bufnum = ntohl(hdr->substream_bufnum) & 0x3fffff;
                sample_rate = ntohl(hdr->quant_sample_rate) & 0x3fffff;
                bps = (ntohl(hdr->quant_sample_rate) >> 26) / 8;
                
                if(s->saved_channels != total_channels ||
                                s->saved_bps != bps ||
                                s->saved_sample_rate != sample_rate) {
                        if(audio_reconfigure(s->audio_state, bps * 8, total_channels,
                                                sample_rate) != TRUE) {
                                fprintf(stderr, "Audio reconfiguration failed!");
                                return FALSE;
                        }
                        else fprintf(stderr, "Audio reconfiguration succeeded.");
                        fprintf(stderr, " (%d channels, %d bps, %d Hz)\n", total_channels,
                                        bps, sample_rate);
                        s->saved_channels = total_channels;
                        s->saved_bps = bps;
                        s->saved_sample_rate = sample_rate;
                        buffer = audio_get_frame(s->audio_state);
                        packet_counter_destroy(decoder->packet_counter);
                        decoder->packet_counter = packet_counter_init(total_channels);
                }
                
                data = cdata->data->data + sizeof(audio_payload_hdr_t);
                
                int length = cdata->data->data_len - sizeof(audio_payload_hdr_t);

                int offset = ntohl(hdr->offset);
                //fprintf(stderr, "%d-%d-%d ", length, bufnum, channel);
                packet_counter_register_packet(decoder->packet_counter, channel, bufnum, offset, length);
                if(length * total_channels <= ((int) buffer->max_size) - offset) {
                        mux_channel(buffer->data + offset * total_channels, data, bps, length, total_channels, channel);
                        //memcpy(buffer->data + ntohl(hdr->offset), data, ntohs(hdr->length));
                } else { /* discarding data - buffer to small */
                        int copy_len = buffer->max_size - offset * total_channels;

                        if(copy_len > 0)
                                mux_channel(buffer->data + offset * total_channels, data, bps, copy_len, total_channels, channel);
                                //memcpy(buffer->data + ntohl(hdr->offset), data, 
                                //        copy_len);
                        if(++prints % 100 == 0)
                                fprintf(stdout, "Warning: "
                                        "discarding audio data "
                                        "- buffer too small (audio init failed?)\n");
                }
                
                /* buffer size same for every packet of the frame */
                if(ntohl(hdr->length) <= buffer->max_size) {
                        buffer->data_len = ntohl(hdr->length) * total_channels;
                } else { /* overflow */
                        buffer->data_len = buffer->max_size;
                }
                
                cdata = cdata->nxt;
        }

        double seconds;
        struct timeval t;

        gettimeofday(&t, 0);
        seconds = tv_diff(t, decoder->t0);
        if(seconds > 5.0) {
                int bytes_received = packet_counter_get_total_bytes(decoder->packet_counter);
                fprintf(stderr, "[Audio decoder] Received and decoded %u bytes (%d channels, %d samples) in last %f seconds (expected %d).\n",
                                bytes_received, total_channels,
                                bytes_received / (bps * total_channels),
                                seconds,
                                packet_counter_get_all_bytes(decoder->packet_counter));
                decoder->t0 = t;
                packet_counter_clear(decoder->packet_counter);
        }
        
        return TRUE;
}


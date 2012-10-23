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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

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

#include <ctype.h>
#include <time.h>
#include <string.h>

#define AUDIO_DECODER_MAGIC 0x12ab332bu

struct scale_data {
        double vol_avg;
        int samples;

        double scale;
};

struct channel_map {
        int **map; // index is source channel, content is output channels
        int *sizes;
        int size;
        int max_output;
};

struct state_audio_decoder {
        uint32_t magic;

        struct timeval t0;

        struct packet_counter *packet_counter;

        unsigned int channel_remapping:1;
        struct channel_map channel_map;

        struct scale_data *scale;
        bool fixed_scale;
};

static int validate_mapping(struct channel_map *map);
static void compute_scale(struct scale_data *scale_data, float vol_avg, int samples, int sample_rate);

static int validate_mapping(struct channel_map *map)
{
        int ret = TRUE;

        for(int i = 0; i < map->size; ++i) {
                for(int j = 0; j < map->sizes[i]; ++j) {
                        if(map->map[i][j] < 0) {
                                fprintf(stderr, "Audio channel mapping - negative parameter occured.\n");
                                ret = FALSE;
                                goto return_value;
                        }
                }
        }

return_value:
        return ret;
}

static void compute_scale(struct scale_data *scale_data, float vol_avg, int samples, int sample_rate)
{
        scale_data->vol_avg = scale_data->vol_avg * (scale_data->samples / ((double) scale_data->samples + samples)) +
                vol_avg * (samples / ((double) scale_data->samples + samples));
        scale_data->samples += samples;

        if(scale_data->samples > sample_rate * 6) { // 10 sec
                double ratio = 0.0;

                if(scale_data->vol_avg < 0.01 && scale_data->vol_avg > 0.0001) {
                        ratio = 1.1;
                } else if(scale_data->vol_avg > 0.25) {
                        ratio = 1/1.1;
                } else if(scale_data->vol_avg > 0.05 && scale_data->scale > 1.0) {
                        ratio = 1/1.1;
                } else if(scale_data->vol_avg < 0.20 && scale_data->scale < 1.0) {
                        ratio = 1.1;
                }

                if(ratio != 0.0) {
                        scale_data->scale *= ratio;
                        scale_data->vol_avg *= ratio;
                }

                debug_msg("Audio scale adjusted to: %f (average volume was %f)\n", scale_data->scale, scale_data->vol_avg);

                scale_data->samples = 4 * sample_rate;
        }
}

void *audio_decoder_init(char *audio_channel_map, const char *audio_scale)
{
        struct state_audio_decoder *s;
        bool scale_auto = false;
        double scale_factor = 1.0;

        assert(audio_scale != NULL);

        s = (struct state_audio_decoder *) malloc(sizeof(struct state_audio_decoder));
        s->magic = AUDIO_DECODER_MAGIC;

        gettimeofday(&s->t0, NULL);
        s->packet_counter = NULL;



        if(audio_channel_map) {
                char *save_ptr = NULL;
                char *item;
                char *tmp;
                char *ptr;
                tmp = ptr = strdup(audio_channel_map);

                s->channel_map.size = 0;
                while((item = strtok_r(ptr, ",", &save_ptr))) {
                        ptr = NULL;
                        // item is in format x1:y1
                        if(isdigit(item[0])) {
                                s->channel_map.size = max(s->channel_map.size, atoi(item) + 1);
                        }
                }
                
                s->channel_map.map = (int **) malloc(s->channel_map.size * sizeof(int *));
                s->channel_map.sizes = (int *) malloc(s->channel_map.size * sizeof(int));
                s->channel_map.max_output = -1;

                /* default value, do not process */
                for(int i = 0; i < s->channel_map.size; ++i) {
                        s->channel_map.map[i] = NULL;
                        s->channel_map.sizes[i] = 0;
                }

                free (tmp);
                tmp = ptr = strdup(audio_channel_map);

                while((item = strtok_r(ptr, ",", &save_ptr))) {
                        ptr = NULL;

                        assert(strchr(item, ':') != NULL);
                        int src;
                        if(isdigit(item[0])) {
                                src = atoi(item);
                        } else {
                                src = -1;
                        }
                        int dst = atoi(strchr(item, ':') + 1);
                        if(src >= 0) {
                                s->channel_map.sizes[src] += 1;
                                if(s->channel_map.map[src] == NULL) {
                                        s->channel_map.map[src] = (int *) malloc(1 * sizeof(int));
                                } else {
                                        s->channel_map.map[src] = realloc(s->channel_map.map[src], s->channel_map.sizes[src] * sizeof(int));
                                }
                                s->channel_map.map[src][s->channel_map.sizes[src] - 1] = dst;
                        }
                        s->channel_map.max_output = max(dst, s->channel_map.max_output);
                }


                if(!validate_mapping(&s->channel_map)) {
                        free(s);
                        fprintf(stderr, "Wrong audio mapping.\n");
                        return NULL;
                } else {
                        s->channel_remapping = TRUE;
                }

                free (tmp);
        } else {
                s->channel_remapping = FALSE;
                s->channel_map.map = NULL;
                s->channel_map.sizes = NULL;
                s->channel_map.size = 0;
        } 

        if(strcasecmp(audio_scale, "mixauto") == 0) {
                if(s->channel_remapping) {
                        scale_auto = true;
                } else {
                        scale_auto = false;
                }
        } else if(strcasecmp(audio_scale, "auto") == 0) {
                scale_auto = true;
        } else if(strcasecmp(audio_scale, "none") == 0) {
                scale_auto = false;
                scale_factor = 1.0;
        } else {
                scale_auto = false;
                scale_factor = atof(audio_scale);
                if(scale_factor <= 0.0) {
                        fprintf(stderr, "Invalid audio scaling factor!\n");
                        free(s);
                        return NULL;
                }
        }

        s->fixed_scale = scale_auto ? false : true;

        if(s->fixed_scale) {
                s->scale = (struct scale_data *) malloc(sizeof(struct scale_data));
                s->scale->samples = 0;
                s->scale->vol_avg = 1.0;
                s->scale->scale = scale_factor;
        } else {
                s->scale = NULL; // will allocated by decoder reconfiguration
        }

        return s;
}

void audio_decoder_destroy(void *state)
{
        struct state_audio_decoder *s = (struct state_audio_decoder *) state;

        assert(s != NULL);
        assert(s->magic == AUDIO_DECODER_MAGIC);

        free(s->scale);
        free(s->channel_map.map);
        free(s->channel_map.sizes);
        packet_counter_destroy(s->packet_counter);

        free(s);
}

int decode_audio_frame(struct coded_data *cdata, void *data)
{
        struct pbuf_audio_data *s = (struct pbuf_audio_data *) data;
        struct audio_frame *buffer = s->buffer;
        struct state_audio_decoder *decoder = s->decoder;

        int input_channels = 0;
        int output_channels = 0;
        int bps, sample_rate, channel;
        static int prints = 0;
        int ret = TRUE;

        if(!cdata) {
                ret = FALSE;
                goto cleanup;
        }

        while (cdata != NULL) {
                char *data;
                // for definition see rtp_callbacks.h
                uint32_t *hdr = (uint32_t *) cdata->data->data;
                        
                /* we receive last channel first (with m bit, last packet) */
                /* thus can be set only with m-bit packet */
                if(cdata->data->m) {
                        input_channels = ((ntohl(hdr[0]) >> 22) & 0x3ff) + 1;
                }

                // we have:
                // 1) last packet, then we have just set total channels
                // 2) not last, but the last one was processed at first
                assert(input_channels > 0);

                channel = (ntohl(hdr[0]) >> 22) & 0x3ff;
                int bufnum = ntohl(hdr[0]) & 0x3fffff;
                sample_rate = ntohl(hdr[3]) & 0x3fffff;
                bps = (ntohl(hdr[3]) >> 26) / 8;
                
                output_channels = decoder->channel_remapping ? decoder->channel_map.max_output + 1: input_channels;

                if(s->saved_channels != input_channels ||
                                s->saved_bps != bps ||
                                s->saved_sample_rate != sample_rate) {
                        if(audio_reconfigure(s->audio_state, bps * 8,
                                                output_channels,
                                                sample_rate) != TRUE) {
                                fprintf(stderr, "Audio reconfiguration failed!");
                                ret = FALSE;
                                goto cleanup;
                        }
                        else {
                                fprintf(stderr, "Audio reconfiguration succeeded.");
                        }

                        s->reconfigured = true;

                        fprintf(stderr, " (%d channels, %d bps, %d Hz)\n", output_channels,
                                        bps, sample_rate);
                        if(!decoder->fixed_scale) {
                                free(decoder->scale);
                                decoder->scale = (struct scale_data *) malloc(output_channels * sizeof(struct scale_data));

                                for(int i = 0; i < output_channels; ++i) {
                                        decoder->scale[i].samples = 0;
                                        decoder->scale[i].vol_avg = 1.0;
                                        decoder->scale[i].scale = 1.0;
                                }
                        }
                        s->saved_channels = input_channels;
                        s->saved_bps = bps;
                        s->saved_sample_rate = sample_rate;
                        buffer = audio_get_frame(s->audio_state);
                        packet_counter_destroy(decoder->packet_counter);
                        decoder->packet_counter = packet_counter_init(input_channels);
                }
                
                data = cdata->data->data + sizeof(audio_payload_hdr_t);
                
                unsigned int length = cdata->data->data_len - sizeof(audio_payload_hdr_t);

                unsigned int offset = ntohl(hdr[1]);
                unsigned int buffer_len = ntohl(hdr[2]);
                //fprintf(stderr, "%d-%d-%d ", length, bufnum, channel);


                if(decoder->channel_remapping) {
                        // first packet, so we clean the buffer first
                        if(cdata->data->m) {
                                assert(buffer_len <= buffer->max_size);

                                memset(buffer->data, 0, buffer_len * output_channels);
                        }

                        // there is a mapping for channel
                        if(channel < decoder->channel_map.size) {
                                for(int i = 0; i < decoder->channel_map.sizes[channel]; ++i) {
                                        // there might be packet mutiplication in which case the result would be wrong
                                        if(!packet_counter_has_packet(decoder->packet_counter, channel, bufnum, offset, length)) {
                                                mux_and_mix_channel(buffer->data + offset * output_channels, data, bps, length, output_channels,
                                                                decoder->channel_map.map[channel][i], 
                                                                decoder->scale[decoder->fixed_scale ? 0 : decoder->channel_map.map[channel][i]].scale);
                                        }
                                }
                        }

                } else {
                        if(length * input_channels <= buffer->max_size - offset) {
                                mux_channel(buffer->data + offset * input_channels, data, bps, length, input_channels,
                                                channel, decoder->scale[decoder->fixed_scale ? 0 : input_channels].scale);
                        } else { /* discarding data - buffer to small */
                                if(++prints % 100 == 0)
                                        fprintf(stdout, "Warning: "
                                                        "discarding audio data "
                                                        "- buffer too small (audio init failed?)\n");
                        }
                }

                packet_counter_register_packet(decoder->packet_counter, channel, bufnum, offset, length);

                /* buffer size same for every packet of the frame */
                if(buffer_len <= buffer->max_size) {
                        buffer->data_len = buffer_len * output_channels;
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
                                bytes_received, input_channels,
                                bytes_received / (bps * input_channels),
                                seconds,
                                packet_counter_get_all_bytes(decoder->packet_counter));
                decoder->t0 = t;
                packet_counter_clear(decoder->packet_counter);
        }

        if(!decoder->fixed_scale) {
                for(int i = 0; i <= decoder->channel_map.max_output; ++i) {
                        double avg = get_avg_volume(buffer->data, bps, buffer->data_len / output_channels, output_channels, i);
                        compute_scale(&decoder->scale[i], avg, buffer->data_len / output_channels / bps, sample_rate);
                }
        }
        
cleanup:

        return ret;
}


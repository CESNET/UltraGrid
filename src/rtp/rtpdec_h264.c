/*
 * AUTHOR:   David Cassany   <david.cassany@i2cat.net>,
 *           Ignacio Contreras <ignacio.contreras@i2cat.net>,
 *           Gerard Castillo <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/rtpdec_h264.h"
#include "utils/h264_stream.h"
#include "utils/bs.h"
#include "video_frame.h"

#define H264_NAL 32 ///< type was 1-23 representing H.264 NAL type

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

int fill_coded_frame_from_sps(struct video_frame *rx_data, unsigned char *data, int data_len);

/**
 * This function extracts important data for futher processing of the stream, eg. frame type - for prepending
 * RTSP/SDP sprop-parameter-sets to I-frame and parsing dimensions from SPS NAL
 *
 * @retval H264_NAL   type was H.264 NAL type that can be directly stored to buffer, using the H264_NAL
 *                    placeholder to represent all
 * @retval !=H264_NAL RTC type that doesn't represent H.264 NAL unit, eg. aggregate or fragment units
 */
static uint8_t process_nal(uint8_t nal, struct video_frame *frame, uint8_t *data, int data_len) {
    uint8_t type = nal & 0x1f;
    uint8_t nri = nal & 0x60;
    debug_msg("NAL type %d\n", (int) type);

    if (type == 7){
        fill_coded_frame_from_sps(frame, data, data_len);
    }

    if (type >= 1 && type <= 23) {
        if(frame->frame_type != INTRA && (type == 5 || type == 6)) {
            frame->frame_type = INTRA;
        } else if (frame->frame_type == BFRAME && nri != 0){
            frame->frame_type = OTHER;
        }
        return H264_NAL;
    }
    return type;
}

static _Bool decode_nal_unit(struct video_frame *frame, int *total_length, int pass, unsigned char **dst, uint8_t *data, int data_len) {
    uint8_t nal = data[0];
    uint8_t type = process_nal(nal, frame, data, data_len);
    int fu_length = 0;

    switch (type) {
        case H264_NAL:
            if (pass == 0) {
                *total_length += sizeof(start_sequence) + data_len;
            } else {
                *dst -= data_len + sizeof(start_sequence);
                memcpy(*dst, start_sequence, sizeof(start_sequence));
                memcpy(*dst + sizeof(start_sequence), data, data_len);
            }
            break;
        case 24:
        {
            int nal_sizes[100];
            unsigned nal_count = 0;
            data++;
            data_len--;

            while (data_len > 2) {
                //TODO: Not properly tested
                uint16_t nal_size;
                memcpy(&nal_size, data, sizeof(uint16_t));
                nal_size = ntohs(nal_size);

                data += 2;
                data_len -= 2;

                if (log_level >= LOG_LEVEL_DEBUG) {
                    debug_msg("STAP-A subpacket NAL type %d\n", (int) (data[0] & 0x1f));
                }

                if (nal_size <= data_len) {
                    if (pass == 0) {
                        *total_length += sizeof(start_sequence) + nal_size;
                    } else {
                        assert(nal_count < sizeof nal_sizes / sizeof nal_sizes[0] - 1);
                        nal_sizes[nal_count++] = nal_size;
                        process_nal(data[0], frame, data, data_len);
                    }
                } else {
                    error_msg("NAL size exceeds length: %u %d\n", nal_size, data_len);
                    return FALSE;
                }
                data += nal_size;
                data_len -= nal_size;

                if (data_len < 0) {
                    error_msg("Consumed more bytes than we got! (%d)\n", data_len);
                    return FALSE;
                }

            }
            if (pass > 0) {
                for (int i = nal_count - 1; i >= 0; i--) {
                    int nal_size = nal_sizes[i];
                    data -= nal_size;
                    *dst -= nal_size + sizeof(start_sequence);
                    memcpy(*dst, start_sequence, sizeof(start_sequence));
                    memcpy(*dst + sizeof(start_sequence), data, nal_size);
                    data -= 2;
                }
            }
            break;
        }
        case 0:
        case 25:
        case 26:
        case 27:
        case 29:
            error_msg("Unhandled NAL type %d\n", type);
            return FALSE;
        case 28:
            data++;
            data_len--;

            if (data_len > 1) {
                uint8_t fu_header = *data;
                uint8_t start_bit = fu_header >> 7;
                uint8_t end_bit       = (fu_header & 0x40) >> 6;
                uint8_t nal_type = fu_header & 0x1f;
                uint8_t reconstructed_nal;

                // Reconstruct this packet's true nal; only the data follows.
                /* The original nal forbidden bit and NRI are stored in this
                 * packet's nal. */
                reconstructed_nal = nal & 0xe0;
                reconstructed_nal |= nal_type;

                // skip the fu_header
                data++;
                data_len--;

                if (pass == 0) {
                    if (start_bit) {
                        *total_length += sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                    } else {
                        *total_length += data_len;
                    }
                } else {
                    if (end_bit) {
                        fu_length = data_len;
                    } else {
                        fu_length += data_len;
                    }
                    if (start_bit) {
                        *dst -= sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                        memcpy(*dst, start_sequence, sizeof(start_sequence));
                        memcpy(*dst + sizeof(start_sequence), &reconstructed_nal, sizeof(reconstructed_nal));
                        memcpy(*dst + sizeof(start_sequence) + sizeof(reconstructed_nal), data, data_len);
                        process_nal(reconstructed_nal, frame, data, fu_length);
                    } else {
                        *dst -= data_len;
                        memcpy(*dst, data, data_len);
                    }
                }
            } else {
                error_msg("Too short data for FU-A H264 RTP packet\n");
                return FALSE;
            }
            break;
        default:
            error_msg("Unknown NAL type\n");
            return FALSE;
    }
    return TRUE;
}

int decode_frame_h264(struct coded_data *cdata, void *decode_data) {
    struct coded_data *orig = cdata;

    int total_length = 0;

    struct decode_data_h264 *data = (struct decode_data_h264 *) decode_data;
    struct video_frame *frame = data->frame;
    frame->frame_type = BFRAME;

    for (int pass = 0; pass < 2; pass++) {
        unsigned char *dst = NULL;

        if (pass > 0) {
            cdata = orig;
            if(frame->frame_type == INTRA){
                total_length+=data->offset_len;
            }
            frame->tiles[0].data_len = total_length;
            dst = (unsigned char *) frame->tiles[0].data + total_length;
        }

        while (cdata != NULL) {
            rtp_packet *pckt = cdata->data;

            if (!decode_nal_unit(frame, &total_length, pass, &dst, (uint8_t *) pckt->data, pckt->data_len)) {
                return FALSE;
            }

            cdata = cdata->nxt;
        }
    }

    return TRUE;
}

int fill_coded_frame_from_sps(struct video_frame *rx_data, unsigned char *data, int data_len){
    uint32_t width, height;
    sps_t* sps = (sps_t*)malloc(sizeof(sps_t));
    uint8_t* rbsp_buf = (uint8_t*)malloc(data_len);
    if (nal_to_rbsp(data, &data_len, rbsp_buf, &data_len) < 0){
        free(rbsp_buf);
        free(sps);
        return -1;
    }
    bs_t* b = bs_new(rbsp_buf, data_len);
    if(read_seq_parameter_set_rbsp(sps,b) < 0){
        bs_free(b);
        free(rbsp_buf);
        free(sps);
        return -1;
    }
    width = (sps->pic_width_in_mbs_minus1 + 1) * 16;
    height = (2 - sps->frame_mbs_only_flag) * (sps->pic_height_in_map_units_minus1 + 1) * 16;
    //NOTE: frame_mbs_only_flag = 1 --> only progressive frames
    //      frame_mbs_only_flag = 0 --> some type of interlacing (there are 3 types contemplated in the standard)
    if (sps->frame_cropping_flag){
        width -= (sps->frame_crop_left_offset*2 + sps->frame_crop_right_offset*2);
        height -= (sps->frame_crop_top_offset*2 + sps->frame_crop_bottom_offset*2);
    }

    if((width != rx_data->tiles[0].width) || (height != rx_data->tiles[0].height)) {
        vf_get_tile(rx_data, 0)->width = width;
        vf_get_tile(rx_data, 0)->height = height;
//        free(rx_data->tiles[0].data);
//        rx_data->tiles[0].data = calloc(1, rx_data->h264_width * rx_data->h264_height);
    }

    bs_free(b);
    free(rbsp_buf);
    free(sps);

    return 0;
}

int width_height_from_SDP(int *widthOut, int *heightOut , unsigned char *data, int data_len){
    uint32_t width, height;
    sps_t* sps = (sps_t*)malloc(sizeof(sps_t));
    uint8_t* rbsp_buf = (uint8_t*)malloc(data_len);
    if (nal_to_rbsp(data, &data_len, rbsp_buf, &data_len) < 0){
        free(rbsp_buf);
        free(sps);
        return -1;
    }
    bs_t* b = bs_new(rbsp_buf, data_len);
    if(read_seq_parameter_set_rbsp(sps,b) < 0){
        bs_free(b);
        free(rbsp_buf);
        free(sps);
        return -1;
    }
    width = (sps->pic_width_in_mbs_minus1 + 1) * 16;
    height = (2 - sps->frame_mbs_only_flag) * (sps->pic_height_in_map_units_minus1 + 1) * 16;
    //NOTE: frame_mbs_only_flag = 1 --> only progressive frames
    //      frame_mbs_only_flag = 0 --> some type of interlacing (there are 3 types contemplated in the standard)
    if (sps->frame_cropping_flag){
        width -= (sps->frame_crop_left_offset*2 + sps->frame_crop_right_offset*2);
        height -= (sps->frame_crop_top_offset*2 + sps->frame_crop_bottom_offset*2);
    }

    debug_msg("\n\n[width_height_from_SDP] width: %d   height: %d\n\n",width,height);


    if(width > 0){
        *widthOut = width;
    }
    if(height > 0){
        *heightOut = height;
    }

    bs_free(b);
    free(rbsp_buf);
    free(sps);

    return 0;
}

// vi: set et sw=4 :

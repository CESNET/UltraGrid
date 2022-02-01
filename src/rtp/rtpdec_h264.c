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

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

int fill_coded_frame_from_sps(struct video_frame *rx_data, unsigned char *data, int data_len);

int decode_frame_h264(struct coded_data *cdata, void *decode_data) {
    rtp_packet *pckt = NULL;
    struct coded_data *orig = cdata;

    uint8_t nal;
    uint8_t type;
    uint8_t nri;

    int pass;
    int total_length = 0;

    unsigned char *dst = NULL;
    int src_len;

    struct decode_data_h264 *data = (struct decode_data_h264 *) decode_data;
    struct video_frame *frame = data->frame;
    frame->frame_type = BFRAME;

    for (pass = 0; pass < 2; pass++) {

        if (pass > 0) {
            cdata = orig;
            if(frame->frame_type == INTRA){
                total_length+=data->offset_len;
            }
            frame->tiles[0].data_len = total_length;
            dst = (unsigned char *) frame->tiles[0].data + total_length;
        }

        while (cdata != NULL) {
            pckt = cdata->data;

            nal = (uint8_t) pckt->data[0];
            type = nal & 0x1f;
            nri = nal & 0x60;

            if (type == 7){
                fill_coded_frame_from_sps(frame, (unsigned char*) pckt->data, pckt->data_len);
            }

            if (type >= 1 && type <= 23) {
                if(frame->frame_type != INTRA && (type == 5 || type == 6)) {
                    frame->frame_type = INTRA;
                } else if (frame->frame_type == BFRAME && nri != 0){
                    frame->frame_type = OTHER;
                }

                type = 1;
            }

            const uint8_t *src = NULL;

            switch (type) {
                case 0:
                case 1:
                    if (pass == 0) {
                        debug_msg("NAL type 1\n");
                        total_length += sizeof(start_sequence) + pckt->data_len;
                    } else {
                        dst -= pckt->data_len + sizeof(start_sequence);
                        memcpy(dst, start_sequence, sizeof(start_sequence));
                        memcpy(dst + sizeof(start_sequence), pckt->data, pckt->data_len);
                    }
                    break;
                case 24:
                    src = (const uint8_t *) pckt->data;
                    src_len = pckt->data_len;

                    src++;
                    src_len--;

                    while (src_len > 2) {
                        //TODO: Not properly tested
                        //TODO: bframes and iframes detection
                        uint16_t nal_size;
                        memcpy(&nal_size, src, sizeof(uint16_t));

                        src += 2;
                        src_len -= 2;

                        if (nal_size <= src_len) {
                            if (pass == 0) {
                                total_length += sizeof(start_sequence) + nal_size;
                            } else {
                                dst -= nal_size + sizeof(start_sequence);
                                memcpy(dst, start_sequence, sizeof(start_sequence));
                                memcpy(dst + sizeof(start_sequence), src, nal_size);
                            }
                        } else {
                            error_msg("NAL size exceeds length: %u %d\n", nal_size, src_len);
                            return FALSE;
                        }
                        src += nal_size;
                        src_len -= nal_size;

                        if (src_len < 0) {
                            error_msg("Consumed more bytes than we got! (%d)\n", src_len);
                            return FALSE;
                        }
                    }
                    break;

                case 25:
                case 26:
                case 27:
                case 29:
                    error_msg("Unhandled NAL type\n");
                    return FALSE;
                case 28:
                    src = (const uint8_t *) pckt->data;
                    src_len = pckt->data_len;

                    src++;
                    src_len--;

                    if (src_len > 1) {
                        uint8_t fu_header = *src;
                        uint8_t start_bit = fu_header >> 7;
                        //uint8_t end_bit       = (fu_header & 0x40) >> 6;
                        uint8_t nal_type = fu_header & 0x1f;
                        uint8_t reconstructed_nal;

                        if(frame->frame_type != INTRA && (nal_type == 5 || nal_type == 6)){
                            frame->frame_type = INTRA;
                        } else if (frame->frame_type == BFRAME && nri != 0){
                            frame->frame_type = OTHER;
                        }

                        // Reconstruct this packet's true nal; only the data follows.
                        /* The original nal forbidden bit and NRI are stored in this
                         * packet's nal. */
                        reconstructed_nal = nal & 0xe0;
                        reconstructed_nal |= nal_type;

                        // skip the fu_header
                        src++;
                        src_len--;

                        if (pass == 0) {
                            if (start_bit) {
                                total_length += sizeof(start_sequence) + sizeof(reconstructed_nal) + src_len;
                            } else {
                                total_length += src_len;
                            }
                        } else {
                            if (start_bit) {
                                dst -= sizeof(start_sequence) + sizeof(reconstructed_nal) + src_len;
                                memcpy(dst, start_sequence, sizeof(start_sequence));
                                memcpy(dst + sizeof(start_sequence), &reconstructed_nal, sizeof(reconstructed_nal));
                                memcpy(dst + sizeof(start_sequence) + sizeof(reconstructed_nal), src, src_len);
                            } else {
                                dst -= src_len;
                                memcpy(dst, src, src_len);
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

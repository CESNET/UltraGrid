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

#include <assert.h>             // for assert
#include <stdbool.h>            // for false, true
#include <stdint.h>             // for uint8_t, uint32_t, uint16_t
#include <stdlib.h>             // for free, malloc, NULL
#include <string.h>             // for memcpy

#include "compat/net.h"         // for ntohs
#include "debug.h"
#include "rtp/rtp.h"
#include "rtp/pbuf.h"
#include "rtp/rtpenc_h264.h"    // for get_nalu_name
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpdec_state.h"
#include "types.h"              // for tile, video_frame, frame_type
#include "utils/h264_stream.h"
#include "utils/hevc_stream.h"
#include "utils/bs.h"
#include "video_frame.h"

/// type was 1-23 representing H.264 NAL type
#define H264_NAL   32

#define MOD_NAME "[rtpdec_h2645] "

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

/**
 * This function extracts important data for futher processing of the stream,
 * eg. frame type - for prepending RTSP/SDP sprop-parameter-sets to I-frame and
 * parsing dimensions from SPS NAL.
 *
 * Should be run in 1st pass - it sets frame_type according to which does the
 * decoder deduce offset in 2nd pass.
 *
 * @retval H.264 or RTP NAL type
 */
static uint8_t process_h264_nal(uint8_t nal, struct video_frame *frame, uint8_t *data, int data_len) {
    uint8_t type = H264_NALU_HDR_GET_TYPE(&nal);
    uint8_t nri = H264_NALU_HDR_GET_NRI(&nal);
    log_msg(LOG_LEVEL_DEBUG2, "NAL type %s (%d; nri: %d)\n",
            get_h264_nalu_name(type), (int) type, (int) nri);

    if (type == NAL_H264_SPS) {
        width_height_from_h264_sps((int *) &frame->tiles[0].width,
                                   (int *) &frame->tiles[0].height, data,
                                   data_len);
    }

    if (type >= NAL_H264_MIN && type <= NAL_H264_MAX) {
        if (type == NAL_H264_IDR || type == NAL_H264_SEI) {
            frame->frame_type = INTRA;
        } else if (frame->frame_type == BFRAME && nri != 0){
            frame->frame_type = OTHER;
        }
    }
    return type;
}

static uint8_t process_hevc_nal(const uint8_t nal[2], struct video_frame *frame, uint8_t *data, int data_len) {
    unsigned char forbidden = nal[0] >> 7;
    assert(forbidden == 0);
    enum hevc_nal_type type = HEVC_NALU_HDR_GET_TYPE(nal);
    unsigned layer_id = HEVC_NALU_HDR_GET_LAYER_ID(nal);
    unsigned tid = HEVC_NALU_HDR_GET_TID(nal);
    log_msg(LOG_LEVEL_DEBUG2, "HEVC NAL type %s (%d; layerID: %u, TID: %u)\n",
            get_hevc_nalu_name(type), (int) type, layer_id, tid);

    if (type == NAL_HEVC_SPS) {
        width_height_from_hevc_sps((int *) &frame->tiles[0].width,
                                   (int *) &frame->tiles[0].height, data,
                                   data_len);
    }

    frame->frame_type = type >= NAL_HEVC_CODED_SLC_FIRST && type <= NAL_HEVC_MAX
                            ? INTRA
                            : OTHER;
    return type;
}

/**
 * @param pass  0 - extracts frame metadata (type; width height if SPS present)
 * and computes required out buffer length;
 *              1 - copy NAL units to output buffer separated by start codes
 * ([0,]0,0,1i; Annex-B)
 */
static bool
decode_h264_nal_unit(struct video_frame *frame, int *total_length, int pass,
                     unsigned char **dst, uint8_t *data, int data_len)
{
    int fu_length = 0;
    uint8_t nal = data[0];
    uint8_t type      = pass == 0 ? process_h264_nal(nal, frame, data, data_len)
                                  : H264_NALU_HDR_GET_TYPE(&nal);
    if (type >= NAL_H264_MIN && type <= NAL_H264_MAX) {
        type = H264_NAL;
    }

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
        case RTP_STAP_A:
        {
            int nal_sizes[100];
            unsigned nal_count = 0;
            data++;
            data_len--;

            while (data_len > 2) {
                uint16_t nal_size;
                memcpy(&nal_size, data, sizeof(uint16_t));
                nal_size = ntohs(nal_size);

                data += 2;
                data_len -= 2;

                log_msg(LOG_LEVEL_DEBUG2,
                        "STAP-A subpacket NAL type %d (nri: %d)\n",
                        (int) H264_NALU_HDR_GET_TYPE(data),
                        (int) H264_NALU_HDR_GET_NRI(&nal));

                if (nal_size <= data_len) {
                    if (pass == 0) {
                        *total_length += sizeof(start_sequence) + nal_size;
                        process_h264_nal(data[0], frame, data, data_len);
                    } else {
                        assert(nal_count < sizeof nal_sizes / sizeof nal_sizes[0] - 1);
                        nal_sizes[nal_count++] = nal_size;
                    }
                } else {
                    error_msg("NAL size exceeds length: %u %d\n", nal_size, data_len);
                    return false;
                }
                data += nal_size;
                data_len -= nal_size;

                if (data_len < 0) {
                    error_msg("Consumed more bytes than we got! (%d)\n", data_len);
                    return false;
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
        case RTP_STAP_B:
        case RTP_MTAP16:
        case RTP_MTAP24:
        case RTP_FU_B:
            error_msg("Unhandled NAL type %d\n", type);
            return false;
        case RTP_FU_A:
            data++;
            data_len--;

            if (data_len > 1) {
                uint8_t fu_header = *data;
                uint8_t start_bit = fu_header >> 7;
                uint8_t end_bit       = (fu_header & 0x40) >> 6;
                uint8_t nal_type = H264_NALU_HDR_GET_TYPE(&fu_header);
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
                    if (end_bit) {
                        fu_length = data_len;
                    } else {
                        fu_length += data_len;
                    }
                    if (start_bit) {
                        *total_length += sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                        process_h264_nal(reconstructed_nal, frame, data, fu_length);
                    } else {
                        *total_length += data_len;
                    }
                } else {
                    if (start_bit) {
                        *dst -= sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                        memcpy(*dst, start_sequence, sizeof(start_sequence));
                        memcpy(*dst + sizeof(start_sequence), &reconstructed_nal, sizeof(reconstructed_nal));
                        memcpy(*dst + sizeof(start_sequence) + sizeof(reconstructed_nal), data, data_len);
                    } else {
                        *dst -= data_len;
                        memcpy(*dst, data, data_len);
                    }
                }
            } else {
                error_msg("Too short data for FU-A H264 RTP packet\n");
                return false;
            }
            break;
        default:
            error_msg("Unknown NAL type %d\n", type);
            return false;
    }
    return true;
}

static bool
decode_hevc_nal_unit(struct video_frame *frame, int *total_length, int pass,
                     unsigned char **dst, uint8_t *data, int data_len)
{
    uint8_t nal[2] = { data[0], data[1] };
    int fu_length = 0;
    enum hevc_nal_type type = pass == 0
                                  ? process_hevc_nal(nal, frame, data, data_len)
                                  : HEVC_NALU_HDR_GET_TYPE(nal);

    if (type <= NAL_HEVC_MAX) {
            if (pass == 0) {
                *total_length += sizeof(start_sequence) + data_len;
            } else {
                *dst -= data_len + sizeof(start_sequence);
                memcpy(*dst, start_sequence, sizeof(start_sequence));
                memcpy(*dst + sizeof(start_sequence), data, data_len);
            }
    } else if (type == NAL_RTP_HEVC_AP) {
            // untested (copy from decode_h264_nal_unit)
            int nal_sizes[100];
            unsigned nal_count = 0;
            data += 2;
            data_len -= 2;

            while (data_len > 2) {
                uint16_t nal_size;
                memcpy(&nal_size, data, sizeof(uint16_t));
                nal_size = ntohs(nal_size);

                data += 2;
                data_len -= 2;

                log_msg(LOG_LEVEL_DEBUG2,
                        "HEVC AP subpacket NAL type %d\n",
                        (int) HEVC_NALU_HDR_GET_TYPE(data));

                if (nal_size <= data_len) {
                    if (pass == 0) {
                        *total_length += sizeof(start_sequence) + nal_size;
                        process_hevc_nal(data, frame, data, data_len);
                    } else {
                        assert(nal_count < sizeof nal_sizes / sizeof nal_sizes[0] - 1);
                        nal_sizes[nal_count++] = nal_size;
                    }
                } else {
                    error_msg("NAL size exceeds length: %u %d\n", nal_size, data_len);
                    return false;
                }
                data += nal_size;
                data_len -= nal_size;

                if (data_len < 0) {
                    error_msg("Consumed more bytes than we got! (%d)\n", data_len);
                    return false;
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
    } else if (type == NAL_RTP_HEVC_FU) {
            data += 2;
            data_len -= 2;

            if (data_len > 1) {
                uint8_t fu_header = *data;
                uint8_t start_bit = fu_header >> 7;
                uint8_t end_bit       = (fu_header & 0x40) >> 6;
                enum hevc_nal_type nal_type = fu_header & 0x3f;
                uint8_t reconstructed_nal[2];

                // Reconstruct this packet's true nal; only the data follows.
                /* The original nal forbidden bit, layer ID and TID are stored
                 * packet's nal. */
                reconstructed_nal[0] = nal[0] & 0x81; // keep 1st and last bit
                reconstructed_nal[0] |= nal_type << 1;
                reconstructed_nal[1] = nal[1];

                // skip the fu_header
                data++;
                data_len--;
                // + skip DONL if present

                if (pass == 0) {
                    if (end_bit) {
                        fu_length = data_len;
                    } else {
                        fu_length += data_len;
                    }
                    if (start_bit) {
                        *total_length += sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                        process_hevc_nal(reconstructed_nal, frame, data, fu_length);
                    } else {
                        *total_length += data_len;
                    }
                } else {
                    if (start_bit) {
                        *dst -= sizeof(start_sequence) + sizeof(reconstructed_nal) + data_len;
                        memcpy(*dst, start_sequence, sizeof(start_sequence));
                        memcpy(*dst + sizeof(start_sequence), &reconstructed_nal, sizeof(reconstructed_nal));
                        memcpy(*dst + sizeof(start_sequence) + sizeof(reconstructed_nal), data, data_len);
                    } else {
                        *dst -= data_len;
                        memcpy(*dst, data, data_len);
                    }
                }
            } else {
                error_msg("Too short data for FU HEVC RTP packet\n");
                return false;
            }
    } else {
            MSG(ERROR, "%s type not implemented!\n", get_hevc_nalu_name(type));
            return false;
    }
    return true;
}

static void
write_sps_pps(struct video_frame *frame, struct decode_data_rtsp *decode_data) {
        memcpy(frame->tiles[0].data, decode_data->h264.offset_buffer,
               decode_data->offset_len);
}

/**
 * This is not mandatory and is merely an optimization - we can emit PPS/SPS
 * early (otherwise it is prepended only to IDR frames). The aim is to allow
 * the receiver to probe the format while allowing it to reconfigure, it can
 * then display the following IDR (otherwise it would be used to probe and the
 * only next would be displayed).
 */
struct video_frame *
get_sps_pps_frame(const struct video_desc *desc,
             struct decode_data_rtsp *decode_data)
{
        if (decode_data->offset_len == 0) {
                return NULL;
        }
        struct video_frame *frame = vf_alloc_desc_data(*desc);
        frame->tiles[0].data_len  = decode_data->offset_len;
        frame->callbacks.dispose  = vf_free;
        write_sps_pps(frame, decode_data);
        return frame;
}

int decode_frame_h2645(struct coded_data *cdata, void *decode_data) {
    struct coded_data *orig = cdata;

    int total_length = 0;

    struct decode_data_rtsp *data = decode_data;
    struct video_frame *frame = data->frame;
    frame->frame_type = BFRAME;
    assert(frame->color_spec == H264 || frame->color_spec == H265);
    bool (*const decode_nal_unit)(struct video_frame *, int *, int,
                                  unsigned char **, uint8_t *, int) =
        frame->color_spec == H264 ? decode_h264_nal_unit : decode_hevc_nal_unit;

    for (int pass = 0; pass < 2; pass++) {
        unsigned char *dst = NULL;

        if (pass > 0) {
            cdata = orig;
            if(frame->frame_type == INTRA){
                total_length+=data->offset_len;
            }
            frame->tiles[0].data_len = total_length;
            assert(frame->tiles[0].data == NULL);
            frame->tiles[0].data = malloc(total_length);
            frame->callbacks.data_deleter = vf_data_deleter;
            dst = (unsigned char *) frame->tiles[0].data + total_length;
        }

        while (cdata != NULL) {
            rtp_packet *pckt = cdata->data;

            if (!decode_nal_unit(frame, &total_length, pass, &dst, (uint8_t *) pckt->data, pckt->data_len)) {
                return false;
            }

            cdata = cdata->nxt;
        }
    }

    if (frame->frame_type == INTRA) {
        write_sps_pps(frame, data);
    }

    return true;
}

int
width_height_from_h264_sps(int *widthOut, int *heightOut, unsigned char *data,
                           int data_len)
{
    uint32_t width, height;
    sps_t* sps = (sps_t*)malloc(sizeof(sps_t));
    uint8_t* rbsp_buf = (uint8_t*)malloc(data_len);
    if (nal_to_rbsp(data, &data_len, rbsp_buf, &data_len) < 0){
        free(rbsp_buf);
        free(sps);
        return -1;
    }
    bs_t* b = bs_new(rbsp_buf, data_len);
    /* nal->forbidden_zero_bit */ bs_skip_u(b, 1);
    /* nal->nal_ref_idc = */ bs_read_u(b, 2);
    /* nal->nal_unit_type = */ bs_read_u(b, 5);
    read_seq_parameter_set_rbsp(sps,b);
    width = (sps->pic_width_in_mbs_minus1 + 1) * 16;
    height = (2 - sps->frame_mbs_only_flag) * (sps->pic_height_in_map_units_minus1 + 1) * 16;
    //NOTE: frame_mbs_only_flag = 1 --> only progressive frames
    //      frame_mbs_only_flag = 0 --> some type of interlacing (there are 3 types contemplated in the standard)
    if (sps->frame_cropping_flag){
        width -= (sps->frame_crop_left_offset + sps->frame_crop_right_offset);
        height -= (sps->frame_crop_top_offset + sps->frame_crop_bottom_offset);
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

int
width_height_from_hevc_sps(int *widthOut, int *heightOut, unsigned char *data,
                           int data_len)
{
    uint32_t width, height;

    /** @todo the more correct way than using the low-level functions will be
     * the following, but we will use the low-level approach as for H.264 now:
     * ```
     * hevc_stream_t* h = hevc_new();
     * read_hevc_nal_unit(h, data, data_len);
     * ...process..
     * hevc_free(h);
     * ````
     * Also this allow keeping just the SPS parser part from the HEVC streem
     * decoder. */

    uint8_t* rbsp_buf = (uint8_t*)malloc(data_len);
    if (nal_to_rbsp(data, &data_len, rbsp_buf, &data_len) < 0){
        free(rbsp_buf);
        return -1;
    }
    bs_t* b = bs_new(rbsp_buf, data_len);
    /* forbidden_zero_bit */ bs_skip_u(b, 1);
    /* nal->nal_unit_type = */ bs_read_u(b, 6);
    /* nal->nal_layer_id = */ bs_read_u(b, 6);
    /* nal->nal_temporal_id_plus1 = */ bs_read_u(b, 3);
    hevc_sps_t sps;
    read_hevc_seq_parameter_set_rbsp(&sps, b);

    width = sps.pic_width_in_luma_samples;
    height = sps.pic_height_in_luma_samples;

    if (sps.conformance_window_flag) {
        width -= sps.conf_win_left_offset + sps.conf_win_right_offset;
        height -= sps.conf_win_top_offset + sps.conf_win_bottom_offset;
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

    return 0;
}


// vi: set et sw=4 :

/*
 * h264bitstream - a library for reading and writing H.264 video
 * Copyright (C) 2005-2007 Auroras Entertainment, LLC
 * Copyright (C) 2008-2011 Avail-TVN
 *
 * Written by Alex Izvorski <aizvorski@gmail.com> and Alex Giladi <alex.giladi@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef _H264_STREAM_H
#define _H264_STREAM_H        1

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "bs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
   Sequence Parameter Set
   @see 7.3.2.1 Sequence parameter set RBSP syntax
   @see read_seq_parameter_set_rbsp
   @see write_seq_parameter_set_rbsp
   @see debug_sps
*/
typedef struct
{
    int profile_idc;
    int constraint_set0_flag;
    int constraint_set1_flag;
    int constraint_set2_flag;
    int constraint_set3_flag;
    int constraint_set4_flag;
    int constraint_set5_flag;
    int reserved_zero_2bits;
    int level_idc;
    int seq_parameter_set_id;
    int chroma_format_idc;
    int residual_colour_transform_flag;
    int bit_depth_luma_minus8;
    int bit_depth_chroma_minus8;
    int qpprime_y_zero_transform_bypass_flag;
    int seq_scaling_matrix_present_flag;
      int seq_scaling_list_present_flag[8];
      int* ScalingList4x4[6];
      int UseDefaultScalingMatrix4x4Flag[6];
      int* ScalingList8x8[2];
      int UseDefaultScalingMatrix8x8Flag[2];
    int log2_max_frame_num_minus4;
    int pic_order_cnt_type;
      int log2_max_pic_order_cnt_lsb_minus4;
      int delta_pic_order_always_zero_flag;
      int offset_for_non_ref_pic;
      int offset_for_top_to_bottom_field;
      int num_ref_frames_in_pic_order_cnt_cycle;
      int offset_for_ref_frame[256];
    int num_ref_frames;
    int gaps_in_frame_num_value_allowed_flag;
    int pic_width_in_mbs_minus1;
    int pic_height_in_map_units_minus1;
    int frame_mbs_only_flag;
    int mb_adaptive_frame_field_flag;
    int direct_8x8_inference_flag;
    int frame_cropping_flag;
      int frame_crop_left_offset;
      int frame_crop_right_offset;
      int frame_crop_top_offset;
      int frame_crop_bottom_offset;
    int vui_parameters_present_flag;

    struct
    {
        int aspect_ratio_info_present_flag;
          int aspect_ratio_idc;
            int sar_width;
            int sar_height;
        int overscan_info_present_flag;
          int overscan_appropriate_flag;
        int video_signal_type_present_flag;
          int video_format;
          int video_full_range_flag;
          int colour_description_present_flag;
            int colour_primaries;
            int transfer_characteristics;
            int matrix_coefficients;
        int chroma_loc_info_present_flag;
          int chroma_sample_loc_type_top_field;
          int chroma_sample_loc_type_bottom_field;
        int timing_info_present_flag;
          int num_units_in_tick;
          int time_scale;
          int fixed_frame_rate_flag;
        int nal_hrd_parameters_present_flag;
        int vcl_hrd_parameters_present_flag;
          int low_delay_hrd_flag;
        int pic_struct_present_flag;
        int bitstream_restriction_flag;
          int motion_vectors_over_pic_boundaries_flag;
          int max_bytes_per_pic_denom;
          int max_bits_per_mb_denom;
          int log2_max_mv_length_horizontal;
          int log2_max_mv_length_vertical;
          int num_reorder_frames;
          int max_dec_frame_buffering;
    } vui;

    struct
    {
        int cpb_cnt_minus1;
        int bit_rate_scale;
        int cpb_size_scale;
          int bit_rate_value_minus1[32]; // up to cpb_cnt_minus1, which is <= 31
          int cpb_size_value_minus1[32];
          int cbr_flag[32];
        int initial_cpb_removal_delay_length_minus1;
        int cpb_removal_delay_length_minus1;
        int dpb_output_delay_length_minus1;
        int time_offset_length;
    } hrd;

} sps_t;

/**
   H264 stream
   Contains data structures for all NAL types that can be handled by this library.
   When reading, data is read into those, and when writing it is written from those.
   The reason why they are all contained in one place is that some of them depend on others, we need to
   have all of them available to read or write correctly.
 */

int nal_to_rbsp(const uint8_t* nal_buf, int* nal_size, uint8_t* rbsp_buf, int* rbsp_size);

int read_seq_parameter_set_rbsp(sps_t* h, bs_t* b);
void read_scaling_list(bs_t* b, int* scalingList, int sizeOfScalingList, int useDefaultScalingMatrixFlag );
void read_vui_parameters(sps_t* h, bs_t* b);
void read_hrd_parameters(sps_t* h, bs_t* b);

void read_rbsp_trailing_bits(bs_t* b);

void debug_sps(sps_t* sps);

#define SAR_Extended      255

#ifdef __cplusplus
}
#endif

#endif

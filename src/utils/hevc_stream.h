/*
 * Copied from <https://github.com/leslie-wang/hevcbitstream> but
 * removed all parts except the parsing of SPS.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
/*
  hevc_stream.h
  Created by leslie_qiwa@gmail.com on 6/8/17
 */


#ifndef _HEVC_STREAM_H
#define _HEVC_STREAM_H        1

#include "bs.h"
// #include "h264_sei.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NUM_SUBLAYERS            32
#define MAX_NUM_HRD_PARAM            10
#define MAX_CPB_CNT                  32
#define MAX_NUM_NEGATIVE_PICS        32
#define MAX_NUM_POSITIVE_PICS        32
#define MAX_NUM_REF_PICS_L0          32
#define MAX_NUM_REF_PICS_L1          32
#define MAX_NUM_SHORT_TERM_REF_PICS  32
#define MAX_NUM_LONG_TERM_REF_PICS   32
#define MAX_NUM_LONG_TERM_REF_PICS   32
#define MAX_NUM_PALLETTE_PREDICTOR   32

/**
   hrd_parameters
   @see E.2.2 HRD parameters syntax
*/
typedef struct
{
    int bit_rate_value_minus1[MAX_CPB_CNT];
    int cpb_size_value_minus1[MAX_CPB_CNT];
    int cpb_size_du_value_minus1[MAX_CPB_CNT];
    int bit_rate_du_value_minus1[MAX_CPB_CNT];
    int cbr_flag[MAX_CPB_CNT];
} hevc_sub_layer_hrd_t;
/**
   hrd_parameters
   @see E.2.2 HRD parameters syntax
*/
typedef struct
{
    int nal_hrd_parameters_present_flag;
    int vcl_hrd_parameters_present_flag;
    int sub_pic_hrd_params_present_flag;
    int tick_divisor_minus2;
    int du_cpb_removal_delay_increment_length_minus1;
    int sub_pic_cpb_params_in_pic_timing_sei_flag;
    int dpb_output_delay_du_length_minus1;
    int bit_rate_scale;
    int cpb_size_scale;
    int cpb_size_du_scale;
    int initial_cpb_removal_delay_length_minus1;
    int au_cpb_removal_delay_length_minus1;
    int dpb_output_delay_length_minus1;
    int fixed_pic_rate_general_flag[MAX_NUM_SUBLAYERS];
    int fixed_pic_rate_within_cvs_flag[MAX_NUM_SUBLAYERS];
    int elemental_duration_in_tc_minus1[MAX_NUM_SUBLAYERS];
    int low_delay_hrd_flag[MAX_NUM_SUBLAYERS];
    int cpb_cnt_minus1[MAX_NUM_SUBLAYERS];
    hevc_sub_layer_hrd_t sub_layer_hrd_nal[MAX_NUM_SUBLAYERS];
    hevc_sub_layer_hrd_t sub_layer_hrd_vcl[MAX_NUM_SUBLAYERS];
} hevc_hrd_t;

/**
   Profile, tier and level
   @see 7.3 Profile, tier and level syntax
*/
typedef struct
{
    //profile parameters
    int general_profile_space;
    int general_tier_flag;
    int general_profile_idc;
    int general_profile_compatibility_flag[32];
    int general_progressive_source_flag;
    int general_interlaced_source_flag;
    int general_non_packed_constraint_flag;
    int general_frame_only_constraint_flag;
    int general_max_12bit_constraint_flag;
    int general_max_10bit_constraint_flag;
    int general_max_8bit_constraint_flag;
    int general_max_422chroma_constraint_flag;
    int general_max_420chroma_constraint_flag;
    int general_max_monochrome_constraint_flag;
    int general_intra_constraint_flag;
    int general_one_picture_only_constraint_flag;
    int general_lower_bit_rate_constraint_flag;
    int general_max_14bit_constraint_flag;
    int general_inbld_flag;
    // level parameters
    int general_level_idc;
    int sub_layer_profile_present_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_level_present_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_profile_space[MAX_NUM_SUBLAYERS];
    int sub_layer_tier_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_profile_idc[MAX_NUM_SUBLAYERS];
    int sub_layer_profile_compatibility_flag[MAX_NUM_SUBLAYERS][32];
    int sub_layer_progressive_source_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_interlaced_source_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_non_packed_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_frame_only_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_12bit_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_10bit_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_8bit_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_422chroma_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_420chroma_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_monochrome_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_intra_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_one_picture_only_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_lower_bit_rate_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_max_14bit_constraint_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_inbld_flag[MAX_NUM_SUBLAYERS];
    int sub_layer_level_idc[MAX_NUM_SUBLAYERS];
} hevc_profile_tier_level_t;

/**
   Scaling List Data
   @see 7.3.4 scaling list data syntax
*/
typedef struct
{
    int scaling_list_pred_mode_flag[4][6];
    int scaling_list_pred_matrix_id_delta[4][6];
    int scaling_list_dc_coef_minus8[2][6];
    int scaling_list_delta_coef[4][64];
} hevc_scaling_list_data_t;

/**
   Sequence Parameter Set
   @see 7.3.2.2 Sequence parameter set RBSP syntax
*/
typedef struct
{
    int inter_ref_pic_set_prediction_flag;
    int delta_idx_minus1;
    int delta_rps_sign;
    int abs_delta_rps_minus1;
    int used_by_curr_pic_flag[MAX_NUM_SHORT_TERM_REF_PICS];
    int use_delta_flag[MAX_NUM_SHORT_TERM_REF_PICS];
    int num_negative_pics;
    int num_positive_pics;
    int delta_poc_s0_minus1[MAX_NUM_NEGATIVE_PICS];
    int used_by_curr_pic_s0_flag[MAX_NUM_NEGATIVE_PICS];
    int delta_poc_s1_minus1[MAX_NUM_POSITIVE_PICS];
    int used_by_curr_pic_s1_flag[MAX_NUM_NEGATIVE_PICS];
} hevc_st_ref_pic_set_t;

/**
   Sequence Parameter Set
   @see 7.3.2.2 Sequence parameter set RBSP syntax
*/
typedef struct
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
    int neutral_chroma_indication_flag;
    int field_seq_flag;
    int frame_field_info_present_flag;
    int default_display_window_flag; 
    int def_disp_win_left_offset;
    int def_disp_win_right_offset;
    int def_disp_win_top_offset;
    int def_disp_win_bottom_offset;
    int vui_timing_info_present_flag;
    int vui_num_units_in_tick;
    int vui_time_scale;
    int vui_poc_proportional_to_timing_flag;
    int vui_num_ticks_poc_diff_one_minus1;
    int vui_hrd_parameters_present_flag;
    hevc_hrd_t hrd;
    int bitstream_restriction_flag;
    int tiles_fixed_structure_flag;
    int motion_vectors_over_pic_boundaries_flag;
    int restricted_ref_pic_lists_flag;
    int min_spatial_segmentation_idc;
    int max_bytes_per_pic_denom;
    int max_bits_per_min_cu_denom;
    int log2_max_mv_length_horizontal;
    int log2_max_mv_length_vertical;
} hevc_vui_t;

/**
   Sequence Parameter Set range extension syntax
   @see 7.3.2.2.2 Sequence parameter set range extension syntax
*/
typedef struct
{
    int transform_skip_rotation_enabled_flag;
    int transform_skip_context_enabled_flag;
    int implicit_rdpcm_enabled_flag;
    int explicit_rdpcm_enabled_flag;
    int extended_precision_processing_flag;
    int intra_smoothing_disabled_flag;
    int high_precision_offsets_enabled_flag;
    int persistent_rice_adaptation_enabled_flag;
    int cabac_bypass_alignment_enabled_flag;
} hevc_sps_range_ext_t;

/**
   Sequence Parameter Set
   @see 7.3.2.2 Sequence parameter set RBSP syntax
   @see read_hevc_seq_parameter_set_rbsp
   @see write_hevc_seq_parameter_set_rbsp
   @see debug_sps
*/
typedef struct
{
    int sps_video_parameter_set_id;
    int sps_max_sub_layers_minus1;
    int sps_temporal_id_nesting_flag;
    hevc_profile_tier_level_t ptl;
    int sps_seq_parameter_set_id;
    int chroma_format_idc;
    int separate_colour_plane_flag;
    int pic_width_in_luma_samples;
    int pic_height_in_luma_samples;
    int conformance_window_flag;
    int conf_win_left_offset;
    int conf_win_right_offset;
    int conf_win_top_offset;
    int conf_win_bottom_offset;
    int bit_depth_luma_minus8;
    int bit_depth_chroma_minus8;
    int log2_max_pic_order_cnt_lsb_minus4;
    int sps_sub_layer_ordering_info_present_flag;
    int sps_max_dec_pic_buffering_minus1[MAX_NUM_SUBLAYERS];
    int sps_max_num_reorder_pics[MAX_NUM_SUBLAYERS];
    int sps_max_latency_increase_plus1[MAX_NUM_SUBLAYERS];
    int log2_min_luma_coding_block_size_minus3;
    int log2_diff_max_min_luma_coding_block_size;
    int log2_min_luma_transform_block_size_minus2;
    int log2_diff_max_min_luma_transform_block_size;
    int max_transform_hierarchy_depth_inter;
    int max_transform_hierarchy_depth_intra;
    int scaling_list_enabled_flag;
    int sps_scaling_list_data_present_flag;
    hevc_scaling_list_data_t scaling_list_data;
    int amp_enabled_flag;
    int sample_adaptive_offset_enabled_flag;
    int pcm_enabled_flag;
    int pcm_sample_bit_depth_luma_minus1;
    int pcm_sample_bit_depth_chroma_minus1;
    int log2_min_pcm_luma_coding_block_size_minus3;
    int log2_diff_max_min_pcm_luma_coding_block_size;
    int pcm_loop_filter_disabled_flag;
    int num_short_term_ref_pic_sets;
    hevc_st_ref_pic_set_t st_ref_pic_set[MAX_NUM_SHORT_TERM_REF_PICS];
    int long_term_ref_pics_present_flag;
    int num_long_term_ref_pics_sps;
    int lt_ref_pic_poc_lsb_sps[MAX_NUM_LONG_TERM_REF_PICS];
    int used_by_curr_pic_lt_sps_flag[MAX_NUM_LONG_TERM_REF_PICS];
    int sps_temporal_mvp_enabled_flag;
    int strong_intra_smoothing_enabled_flag;
    int vui_parameters_present_flag;
    hevc_vui_t vui;
    int sps_extension_present_flag;
    int sps_range_extension_flag;
    int sps_multilayer_extension_flag;
    int sps_3d_extension_flag;
    int sps_extension_5bits;
    hevc_sps_range_ext_t sps_range_ext;
    //TODO: support sps_extension_data_flag;
    //TODO: support SVC/MVC extensions
} hevc_sps_t;

void read_hevc_seq_parameter_set_rbsp(hevc_sps_t* sps, bs_t* b);

#ifdef __cplusplus
}
#endif

#endif

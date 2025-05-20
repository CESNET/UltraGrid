/*
 * Copied from <https://github.com/leslie-wang/hevcbitstream> but reduced:
 * - marked functions as internally linked except of
 * read_debug_hevc_seq_parameter_set_rbsp() that is used directly
 * - removed what not needed (see point above; the included parts are not
 * modified, so the original file may replace this if more advanced parsing
 * needed)
 * - added smoe GCC diag pragmas (suppress warnings)
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
/*
  hevc_stream.c
  Created by leslie_qiwa@gmail.com on 6/8/17
 */

#include <string.h>

#include "bs.h"
#include "h264_stream.h"
#include "hevc_stream.h"
// #include "h264_sei.h"

#pragma GCC diagnostic ignored "-Waddress"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int NumDeltaPocs[MAX_NUM_SHORT_TERM_REF_PICS];
static int NumNegativePics[MAX_NUM_NEGATIVE_PICS];
static int NumPositivePics[MAX_NUM_POSITIVE_PICS];
static int DeltaPocS0[MAX_NUM_REF_PICS_L0][MAX_NUM_NEGATIVE_PICS];
static int UsedByCurrPicS0[MAX_NUM_REF_PICS_L0][MAX_NUM_NEGATIVE_PICS];
static int DeltaPocS1[MAX_NUM_REF_PICS_L1][MAX_NUM_POSITIVE_PICS];
static int UsedByCurrPicS1[MAX_NUM_REF_PICS_L1][MAX_NUM_POSITIVE_PICS];

static void updateNumDeltaPocs(hevc_st_ref_pic_set_t *st_ref_pic_set, int stRpsIdx) {
    int RefRpsIdx = stRpsIdx - ( st_ref_pic_set->delta_idx_minus1 + 1 );
    
    if( st_ref_pic_set->inter_ref_pic_set_prediction_flag ) {
        int i, j, dPoc;
        int deltaRps = ( 1 - 2 * st_ref_pic_set->delta_rps_sign ) * ( st_ref_pic_set->abs_delta_rps_minus1 + 1 );
        i = 0;
        for( j = NumPositivePics[ RefRpsIdx ] - 1; j >= 0; j-- ) {
            dPoc = DeltaPocS1[ RefRpsIdx ][ j ] + deltaRps;
            if( dPoc < 0 && st_ref_pic_set->use_delta_flag[ NumNegativePics[ RefRpsIdx ] + j ] ) {
                DeltaPocS0[ stRpsIdx ][ i ] = dPoc;
                UsedByCurrPicS0[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ NumNegativePics[ RefRpsIdx ] + j ];
            }
        }
        if( deltaRps < 0 && st_ref_pic_set->use_delta_flag[ NumDeltaPocs[ RefRpsIdx ] ] ) {
            DeltaPocS0[ stRpsIdx ][ i ] = deltaRps;
            UsedByCurrPicS0[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ NumDeltaPocs[ RefRpsIdx ] ];
        }
        for( j = 0; j < NumNegativePics[ RefRpsIdx ]; j++ ) {
            dPoc = DeltaPocS0[ RefRpsIdx ][ j ] + deltaRps;
            if( dPoc < 0 && st_ref_pic_set->use_delta_flag[ j ] ) {
                DeltaPocS0[ stRpsIdx ][ i ] = dPoc;
                UsedByCurrPicS0[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ j ];
            }
        }
        NumNegativePics[ stRpsIdx ] = i;
        i = 0;
        for( j = NumNegativePics[ RefRpsIdx ] - 1; j >= 0; j-- ) {
            dPoc = DeltaPocS0[ RefRpsIdx ][ j ] + deltaRps;
            if( dPoc > 0 && st_ref_pic_set->use_delta_flag[ j ] ) {
                DeltaPocS1[ stRpsIdx ][ i ] = dPoc;
                UsedByCurrPicS1[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ j ];
            }
        }
        if( deltaRps > 0 && st_ref_pic_set->use_delta_flag[ NumDeltaPocs[ RefRpsIdx ] ] ) {
            DeltaPocS1[ stRpsIdx ][ i ] = deltaRps;
            UsedByCurrPicS1[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ NumDeltaPocs[ RefRpsIdx ] ];
        }
        for( j = 0; j < NumPositivePics[ RefRpsIdx ]; j++) {
            dPoc = DeltaPocS1[ RefRpsIdx ][ j ] + deltaRps;
            if( dPoc > 0 && st_ref_pic_set->use_delta_flag[ NumNegativePics[ RefRpsIdx ] + j ] ) {
                DeltaPocS1[ stRpsIdx ][ i ] = dPoc;
                UsedByCurrPicS1[ stRpsIdx ][ i++ ] = st_ref_pic_set->used_by_curr_pic_flag[ NumNegativePics[ RefRpsIdx ] + j ];
            }
        }
        NumPositivePics[ stRpsIdx ] = i;
    } else {
        NumNegativePics[ stRpsIdx ] = st_ref_pic_set->num_negative_pics;
        NumPositivePics[ stRpsIdx ] = st_ref_pic_set->num_positive_pics;
    }
    
    NumDeltaPocs[ stRpsIdx ] = NumNegativePics[ stRpsIdx ] + NumPositivePics[ stRpsIdx ];
}


void read_hevc_seq_parameter_set_rbsp(hevc_sps_t* sps, bs_t* b);
static void read_hevc_sps_range_extension(hevc_sps_range_ext_t* sps_range_ext, bs_t* b);
static void read_hevc_profile_tier_level(hevc_profile_tier_level_t* ptl, bs_t* b, int profilePresentFlag, int maxNumSubLayersMinus1);
static void read_hevc_scaling_list_data( hevc_scaling_list_data_t *sld, bs_t* b );
static void read_hevc_st_ref_pic_set( hevc_st_ref_pic_set_t *st_ref_pic_set, bs_t* b, int stRpsIdx, int num_short_term_ref_pic_sets );
static void read_hevc_vui_parameters(hevc_sps_t* sps, bs_t* b);
static void read_hevc_hrd_parameters(hevc_hrd_t* hrd, bs_t* b, int commonInfPresentFlag, int maxNumSubLayersMinus1);
static void read_hevc_sub_layer_hrd_parameters(hevc_sub_layer_hrd_t* sub_layer_hrd, bs_t* b, int CpbCnt, int sub_pic_hrd_params_present_flag);

//7.3.2.2 Sequence parameter set RBSP syntax
void read_hevc_seq_parameter_set_rbsp(hevc_sps_t* sps, bs_t* b)
{
    int i;

    if( 1 )
    {
        memset(sps, 0, sizeof(hevc_sps_t));
    }
 
    sps->sps_video_parameter_set_id = bs_read_u(b, 4);
    sps->sps_max_sub_layers_minus1 = bs_read_u(b, 3);
    sps->sps_temporal_id_nesting_flag = bs_read_u1(b);
    read_hevc_profile_tier_level(&sps->ptl, b, 1, sps->sps_max_sub_layers_minus1); 
    sps->sps_seq_parameter_set_id = bs_read_ue(b);
    sps->chroma_format_idc = bs_read_ue(b);
    if( sps->chroma_format_idc == 3 ) {
        sps->separate_colour_plane_flag = bs_read_u1(b);
    }
    sps->pic_width_in_luma_samples = bs_read_ue(b);
    sps->pic_height_in_luma_samples = bs_read_ue(b);
    sps->conformance_window_flag = bs_read_u1(b);
    if( sps->conformance_window_flag ) {
        sps->conf_win_left_offset = bs_read_ue(b);
        sps->conf_win_right_offset = bs_read_ue(b);
        sps->conf_win_top_offset = bs_read_ue(b);
        sps->conf_win_bottom_offset = bs_read_ue(b);
    }
    sps->bit_depth_luma_minus8 = bs_read_ue(b);
    sps->bit_depth_chroma_minus8 = bs_read_ue(b);
    sps->log2_max_pic_order_cnt_lsb_minus4 = bs_read_ue(b);
    sps->sps_sub_layer_ordering_info_present_flag = bs_read_u1(b);
    for( i = ( sps->sps_sub_layer_ordering_info_present_flag ? 0 : sps->sps_max_sub_layers_minus1 ); 
            i <= sps->sps_max_sub_layers_minus1; i++ ) {
        sps->sps_max_dec_pic_buffering_minus1 [ i ] = bs_read_ue(b);
        sps->sps_max_num_reorder_pics [ i ] = bs_read_ue(b);
        sps->sps_max_latency_increase_plus1 [ i ] = bs_read_ue(b);
    }
    sps->log2_min_luma_coding_block_size_minus3 = bs_read_ue(b);
    sps->log2_diff_max_min_luma_coding_block_size = bs_read_ue(b);
    sps->log2_min_luma_transform_block_size_minus2 = bs_read_ue(b);
    sps->log2_diff_max_min_luma_transform_block_size = bs_read_ue(b);
    sps->max_transform_hierarchy_depth_inter = bs_read_ue(b);
    sps->max_transform_hierarchy_depth_intra = bs_read_ue(b);
    sps->scaling_list_enabled_flag = bs_read_u1(b);
    
    if( sps->scaling_list_enabled_flag ) {
        sps->sps_scaling_list_data_present_flag = bs_read_u1(b);
        if( sps->sps_scaling_list_data_present_flag ) {
            read_hevc_scaling_list_data(&sps->scaling_list_data, b); 
        }
    }
    
    sps->amp_enabled_flag = bs_read_u1(b);
    sps->sample_adaptive_offset_enabled_flag = bs_read_u1(b);
    sps->pcm_enabled_flag = bs_read_u1(b);
    if( sps->pcm_enabled_flag ) {
        sps->pcm_sample_bit_depth_luma_minus1 = bs_read_u(b, 4);
        sps->pcm_sample_bit_depth_chroma_minus1 = bs_read_u(b, 4);
        sps->log2_min_pcm_luma_coding_block_size_minus3 = bs_read_ue(b);
        sps->log2_diff_max_min_pcm_luma_coding_block_size = bs_read_ue(b);
        sps->pcm_loop_filter_disabled_flag = bs_read_u1(b);
    }
    sps->num_short_term_ref_pic_sets = bs_read_ue(b);
    for( i = 0; i < sps->num_short_term_ref_pic_sets; i++) {
        read_hevc_st_ref_pic_set(&sps->st_ref_pic_set[i], b, i, sps->num_short_term_ref_pic_sets);
    }
    
    sps->long_term_ref_pics_present_flag = bs_read_u1(b);
    if( sps->long_term_ref_pics_present_flag ) {
        sps->num_long_term_ref_pics_sps = bs_read_ue(b);
        for( i = 0; i < sps->num_long_term_ref_pics_sps; i++ ) {
            sps->lt_ref_pic_poc_lsb_sps[ i ] = bs_read_u(b,  sps->log2_max_pic_order_cnt_lsb_minus4 + 4 );
            sps->used_by_curr_pic_lt_sps_flag[ i ] = bs_read_u1(b);
        }
    }
    sps->sps_temporal_mvp_enabled_flag = bs_read_u1(b);
    sps->strong_intra_smoothing_enabled_flag = bs_read_u1(b);
    sps->vui_parameters_present_flag = bs_read_u1(b);
    if( sps->vui_parameters_present_flag ) {
        read_hevc_vui_parameters(sps, b);
    }
    sps->sps_extension_present_flag = bs_read_u1(b);
    
    if( sps->sps_extension_present_flag ) {
        sps->sps_range_extension_flag = bs_read_u1(b);
        sps->sps_multilayer_extension_flag = bs_read_u1(b);
        sps->sps_3d_extension_flag = bs_read_u1(b);
        sps->sps_extension_5bits = bs_read_u(b, 5);
    }
    if( sps->sps_range_extension_flag ) {
        read_hevc_sps_range_extension( &sps->sps_range_ext, b);
    }
    
    // if( 1 )
    // {
    //     memcpy(h->sps_table[sps->sps_seq_parameter_set_id], h->sps, sizeof(hevc_sps_t));
    // }
}

//7.3.2.2.2 Sequence parameter set range extension syntax
void read_hevc_sps_range_extension(hevc_sps_range_ext_t* sps_range_ext, bs_t* b)
{
    sps_range_ext->transform_skip_rotation_enabled_flag = bs_read_u1(b);
    sps_range_ext->transform_skip_context_enabled_flag = bs_read_u1(b);
    sps_range_ext->implicit_rdpcm_enabled_flag = bs_read_u1(b);
    sps_range_ext->explicit_rdpcm_enabled_flag = bs_read_u1(b);
    sps_range_ext->extended_precision_processing_flag = bs_read_u1(b);
    sps_range_ext->intra_smoothing_disabled_flag = bs_read_u1(b);
    sps_range_ext->high_precision_offsets_enabled_flag = bs_read_u1(b);
    sps_range_ext->persistent_rice_adaptation_enabled_flag = bs_read_u1(b);
    sps_range_ext->cabac_bypass_alignment_enabled_flag = bs_read_u1(b);
}

//7.3.3 Profile, tier and level syntax
void read_hevc_profile_tier_level(hevc_profile_tier_level_t* ptl, bs_t* b, int profilePresentFlag, int maxNumSubLayersMinus1)
{
    int i, j;
    if( profilePresentFlag ) {
        ptl->general_profile_space = bs_read_u(b, 2);
        ptl->general_tier_flag = bs_read_u1(b);
        ptl->general_profile_idc = bs_read_u(b, 5);
        for( i = 0; i < 32; i++ ) {
            ptl->general_profile_compatibility_flag[ i ] = bs_read_u1(b);
        }
        ptl->general_progressive_source_flag = bs_read_u1(b);
        ptl->general_interlaced_source_flag = bs_read_u1(b);
        ptl->general_non_packed_constraint_flag = bs_read_u1(b);
        ptl->general_frame_only_constraint_flag = bs_read_u1(b);
        if( ptl->general_profile_idc == 4 || ptl->general_profile_compatibility_flag[ 4 ] || 
            ptl->general_profile_idc == 5 || ptl->general_profile_compatibility_flag[ 5 ] || 
            ptl->general_profile_idc == 6 || ptl->general_profile_compatibility_flag[ 6 ] || 
            ptl->general_profile_idc == 7 || ptl->general_profile_compatibility_flag[ 7 ] ) {
                
            ptl->general_max_12bit_constraint_flag = bs_read_u1(b);
            ptl->general_max_10bit_constraint_flag = bs_read_u1(b);
            ptl->general_max_8bit_constraint_flag = bs_read_u1(b);
            ptl->general_max_422chroma_constraint_flag = bs_read_u1(b);
            ptl->general_max_420chroma_constraint_flag = bs_read_u1(b);
            ptl->general_max_monochrome_constraint_flag = bs_read_u1(b);
            ptl->general_intra_constraint_flag = bs_read_u1(b);
            ptl->general_one_picture_only_constraint_flag = bs_read_u1(b);
            ptl->general_lower_bit_rate_constraint_flag = bs_read_u1(b);
            /* general_reserved_zero_34bits */ bs_skip_u(b, 34);
        } else {
            /* general_reserved_zero_43bits */ bs_skip_u(b, 43);
        }
        if( ( ptl->general_profile_idc >= 1 && ptl->general_profile_idc <= 5 ) ||
              ptl->general_profile_compatibility_flag[ 1 ] ||
              ptl->general_profile_compatibility_flag[ 2 ] ||
              ptl->general_profile_compatibility_flag[ 3 ] ||
              ptl->general_profile_compatibility_flag[ 4 ] ||
              ptl->general_profile_compatibility_flag[ 5 ] ) {

            ptl->general_inbld_flag = bs_read_u1(b);
        } else {
            /* general_reserved_zero_bit */ bs_skip_u(b, 1);
        }
        ptl->general_level_idc = bs_read_u8(b);
        for( i = 0; i < maxNumSubLayersMinus1; i++ ) {
            ptl->sub_layer_profile_present_flag[ i ] = bs_read_u1(b);
            ptl->sub_layer_level_present_flag[ i ] = bs_read_u1(b);
        }
        if( maxNumSubLayersMinus1 > 0 ) {
            for( i = maxNumSubLayersMinus1; i < 8; i++ ) {
                /* reserved_zero_xxbits */ bs_skip_u(b, 2);
            }
        }
        for( i = 0; i < maxNumSubLayersMinus1; i++ ) { 
            if( ptl->sub_layer_profile_present_flag[ i ] ) {
                ptl->sub_layer_profile_space[ i ] = bs_read_u(b, 2);
                ptl->sub_layer_tier_flag[ i ] = bs_read_u1(b);
                ptl->sub_layer_profile_idc[ i ] = bs_read_u(b, 5);
                for( j = 0; j < 32; j++ ) {
                    ptl->sub_layer_profile_compatibility_flag[ i ][ j ] = bs_read_u(b, 1);
                }
                ptl->sub_layer_progressive_source_flag[ i ] = bs_read_u1(b);
                ptl->sub_layer_interlaced_source_flag[ i ] = bs_read_u1(b);
                ptl->sub_layer_non_packed_constraint_flag[ i ] = bs_read_u1(b);
                ptl->sub_layer_frame_only_constraint_flag[ i ] = bs_read_u1(b);
                if( ptl->sub_layer_profile_idc[ i ] == 4 ||
                    ptl->sub_layer_profile_compatibility_flag[ i ][ 4 ] ||
                    ptl->sub_layer_profile_idc[ i ] == 5 ||
                    ptl->sub_layer_profile_compatibility_flag[ i ][ 5 ] ||
                    ptl->sub_layer_profile_idc[ i ] == 6 ||
                    ptl->sub_layer_profile_compatibility_flag[ i ][ 6 ] ||
                    ptl->sub_layer_profile_idc[ i ] == 7 ||
                    ptl->sub_layer_profile_compatibility_flag[ i ][ 7 ] ) {
                    ptl->sub_layer_max_12bit_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_max_10bit_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_max_8bit_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_max_422chroma_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_max_420chroma_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_max_monochrome_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_intra_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_one_picture_only_constraint_flag[ i ] = bs_read_u1(b);
                    ptl->sub_layer_lower_bit_rate_constraint_flag[ i ] = bs_read_u1(b);
                    /* sub_layer_reserved_zero_34bits */ bs_skip_u(b, 34);
                } else {
                    /* sub_layer_reserved_zero_43bits */ bs_skip_u(b, 43);
                }
            
                if( ( ptl->sub_layer_profile_idc[ i ] >= 1 && ptl->sub_layer_profile_idc[ i ] <= 5 ) ||
                   ptl->sub_layer_profile_compatibility_flag[ 1 ] ||
                   ptl->sub_layer_profile_compatibility_flag[ 2 ] ||
                   ptl->sub_layer_profile_compatibility_flag[ 3 ] ||
                   ptl->sub_layer_profile_compatibility_flag[ 4 ] ||
                   ptl->sub_layer_profile_compatibility_flag[ 5 ] ) {
                    ptl->sub_layer_inbld_flag[ i ] = bs_read_u1(b);
                } else {
                    /* sub_layer_reserved_zero_bit */ bs_skip_u(b, 1);
                }
            }
            if( ptl->sub_layer_level_present_flag[ i ] ) {
                ptl->sub_layer_level_idc[ i ] = bs_read_u8(b);
            }
        }
    }
}

//7.3.4 Scaling list data syntax
void read_hevc_scaling_list_data( hevc_scaling_list_data_t *sld, bs_t* b )
{
    int nextCoef, coefNum;
    for( int sizeId = 0; sizeId < 4; sizeId++ )
        for( int matrixId = 0; matrixId < 6; matrixId += ( sizeId == 3 ) ? 3 : 1 ) {
            sld->scaling_list_pred_mode_flag[ sizeId ][ matrixId ] = bs_read_u1(b);
            if( !sld->scaling_list_pred_mode_flag[ sizeId ][ matrixId ] ) {
                sld->scaling_list_pred_matrix_id_delta[ sizeId ][ matrixId ] = bs_read_ue(b);
            } else {
                nextCoef = 8;
                coefNum=MIN(64, (1 << (4+(sizeId << 1))));
                if( sizeId > 1 ) {
                    sld->scaling_list_dc_coef_minus8[ sizeId - 2 ][ matrixId ] = bs_read_se(b);
                }
 
                for( int i = 0; i < coefNum; i++) {
                    sld->scaling_list_delta_coef[ sizeId ][ matrixId ] = bs_read_se(b);
                }
            }
 
        }
}

//7.3.7 Short-term reference picture set syntax
void read_hevc_st_ref_pic_set( hevc_st_ref_pic_set_t *st_ref_pic_set, bs_t* b, int stRpsIdx, int num_short_term_ref_pic_sets )
{
    int i, j;
    
    if( stRpsIdx != 0 ) {
        st_ref_pic_set->inter_ref_pic_set_prediction_flag = bs_read_u1(b);
    }
    if( st_ref_pic_set->inter_ref_pic_set_prediction_flag ) {
        if( stRpsIdx == num_short_term_ref_pic_sets ) {
            st_ref_pic_set->delta_idx_minus1 = bs_read_ue(b);
        }
        st_ref_pic_set->delta_rps_sign = bs_read_u1(b);
        st_ref_pic_set->abs_delta_rps_minus1 = bs_read_ue(b);
        
        int RefRpsIdx = stRpsIdx - ( st_ref_pic_set->delta_idx_minus1 + 1 );
        
        for( j = 0; j <= NumDeltaPocs[ RefRpsIdx ]; j++ ) {
            st_ref_pic_set->used_by_curr_pic_flag[ j ] = bs_read_u1(b);
            if( !st_ref_pic_set->used_by_curr_pic_flag[ j ] ) {
                st_ref_pic_set->use_delta_flag[ j ] = bs_read_u1(b);
            }
        }
    } else {
        st_ref_pic_set->num_negative_pics = bs_read_ue(b);
        st_ref_pic_set->num_positive_pics = bs_read_ue(b);
        for( i = 0; i < st_ref_pic_set->num_negative_pics; i++ ) {
            st_ref_pic_set->delta_poc_s0_minus1[ i ] = bs_read_ue(b);
            st_ref_pic_set->used_by_curr_pic_s0_flag[ i ] = bs_read_u1(b);
            
            //update derived field
            UsedByCurrPicS0[ stRpsIdx ][ i ] = st_ref_pic_set->used_by_curr_pic_s0_flag[ i ];
            
            if( i == 0 ) {
                DeltaPocS0[ stRpsIdx ][ i ] = -1 * ( st_ref_pic_set->delta_poc_s0_minus1[ i ] + 1 );
            } else {
                DeltaPocS0[ stRpsIdx ][ i ] = DeltaPocS0[ stRpsIdx ][ i - 1 ] - ( st_ref_pic_set->delta_poc_s0_minus1[ i ] + 1 );
            }
        }
        for( i = 0; i < st_ref_pic_set->num_positive_pics; i++ ) {
            st_ref_pic_set->delta_poc_s1_minus1[ i ] = bs_read_ue(b);
            st_ref_pic_set->used_by_curr_pic_s1_flag[ i ] = bs_read_u1(b);
        
            //update derived field
            UsedByCurrPicS1[ stRpsIdx ][ i ] = st_ref_pic_set->used_by_curr_pic_s1_flag[ i ];
            
            if( i == 0 ) {
                DeltaPocS1[ stRpsIdx ][ i ] = st_ref_pic_set->delta_poc_s1_minus1[ i ] + 1;
            } else {
                DeltaPocS1[ stRpsIdx ][ i ] = DeltaPocS1[ stRpsIdx ][ i - 1 ] + ( st_ref_pic_set->delta_poc_s1_minus1[ i ] + 1 );
            }
        }
    }
    updateNumDeltaPocs( st_ref_pic_set, stRpsIdx);
}

//Appendix E.2.1 VUI parameters syntax
void read_hevc_vui_parameters(hevc_sps_t* sps, bs_t* b)
{
    hevc_vui_t* vui = &sps->vui;
    vui->aspect_ratio_info_present_flag = bs_read_u1(b);
    if( vui->aspect_ratio_info_present_flag )
    {
        vui->aspect_ratio_idc = bs_read_u8(b);
        if( vui->aspect_ratio_idc == SAR_Extended )
        {
            vui->sar_width = bs_read_u(b, 16);
            vui->sar_height = bs_read_u(b, 16);
        }
    }
    vui->overscan_info_present_flag = bs_read_u1(b);
    if( vui->overscan_info_present_flag ) {
        vui->overscan_appropriate_flag = bs_read_u1(b);
    }
    vui->video_signal_type_present_flag = bs_read_u1(b);
    if( vui->video_signal_type_present_flag ) {
        vui->video_format = bs_read_u(b, 3);
        vui->video_full_range_flag = bs_read_u1(b);
        vui->colour_description_present_flag = bs_read_u1(b);
        if( vui->colour_description_present_flag ) {
            vui->colour_primaries = bs_read_u8(b);
            vui->transfer_characteristics = bs_read_u8(b);
            vui->matrix_coefficients = bs_read_u8(b);
        }
    }
    vui->chroma_loc_info_present_flag = bs_read_u1(b);
    if( vui->chroma_loc_info_present_flag ) {
        vui->chroma_sample_loc_type_top_field = bs_read_ue(b);
        vui->chroma_sample_loc_type_bottom_field = bs_read_ue(b);
    }
    
    vui->neutral_chroma_indication_flag = bs_read_u1(b);
    vui->field_seq_flag = bs_read_u1(b);
    vui->frame_field_info_present_flag = bs_read_u1(b);
    vui->default_display_window_flag = bs_read_u1(b);
    if( vui->default_display_window_flag ) {
        vui->def_disp_win_left_offset = bs_read_ue(b);
        vui->def_disp_win_right_offset = bs_read_ue(b);
        vui->def_disp_win_top_offset = bs_read_ue(b);
        vui->def_disp_win_bottom_offset = bs_read_ue(b);
    }
    vui->vui_timing_info_present_flag = bs_read_u1(b);
    if( vui->vui_timing_info_present_flag ) {
        vui->vui_num_units_in_tick = bs_read_u(b, 32);
        vui->vui_time_scale = bs_read_u(b, 32);
        vui->vui_poc_proportional_to_timing_flag = bs_read_u1(b);
        if( vui->vui_poc_proportional_to_timing_flag ) {
            vui->vui_num_ticks_poc_diff_one_minus1 = bs_read_ue(b);
        }
        vui->vui_hrd_parameters_present_flag = bs_read_u1(b);
        if( vui->vui_hrd_parameters_present_flag ) {
            read_hevc_hrd_parameters( &vui->hrd, b, 1, sps->sps_max_sub_layers_minus1 );
        }
    }
    vui->bitstream_restriction_flag = bs_read_u1(b);
    if( vui->bitstream_restriction_flag )
    {
        vui->tiles_fixed_structure_flag = bs_read_u1(b);
        vui->motion_vectors_over_pic_boundaries_flag = bs_read_u1(b);
        vui->restricted_ref_pic_lists_flag = bs_read_u1(b);
        vui->min_spatial_segmentation_idc = bs_read_ue(b);
        vui->max_bytes_per_pic_denom = bs_read_ue(b);
        vui->max_bits_per_min_cu_denom = bs_read_ue(b);
        vui->log2_max_mv_length_horizontal = bs_read_ue(b);
        vui->log2_max_mv_length_vertical = bs_read_ue(b);
    }
}

//Appendix E.2.2 HRD parameters syntax
void read_hevc_hrd_parameters(hevc_hrd_t* hrd, bs_t* b, int commonInfPresentFlag, int maxNumSubLayersMinus1)
{
    if( commonInfPresentFlag ) {
        hrd->nal_hrd_parameters_present_flag = bs_read_u1(b);
        hrd->vcl_hrd_parameters_present_flag = bs_read_u1(b);
        if( hrd->nal_hrd_parameters_present_flag || hrd->vcl_hrd_parameters_present_flag ){
            hrd->sub_pic_hrd_params_present_flag = bs_read_u1(b);
            if( hrd->sub_pic_hrd_params_present_flag ) {
                hrd->tick_divisor_minus2 = bs_read_u8(b);
                hrd->du_cpb_removal_delay_increment_length_minus1 = bs_read_u(b, 5);
                hrd->sub_pic_cpb_params_in_pic_timing_sei_flag = bs_read_u1(b);
                hrd->dpb_output_delay_du_length_minus1 = bs_read_u(b, 5);
            }
            hrd->bit_rate_scale = bs_read_u(b, 4);
            hrd->cpb_size_scale = bs_read_u(b, 4);
            if( hrd->sub_pic_hrd_params_present_flag ) {
                hrd->cpb_size_du_scale = bs_read_u(b, 4);
            }
            hrd->initial_cpb_removal_delay_length_minus1 = bs_read_u(b, 5);
            hrd->au_cpb_removal_delay_length_minus1 = bs_read_u(b, 5);
            hrd->dpb_output_delay_length_minus1 = bs_read_u(b, 5);
        }
    }
    
    for( int i = 0; i <= maxNumSubLayersMinus1; i++ ) {
        hrd->fixed_pic_rate_general_flag[ i ] = bs_read_u1(b);
        if( !hrd->fixed_pic_rate_general_flag[ i ] ) {
            hrd->fixed_pic_rate_within_cvs_flag[ i ] = bs_read_u1(b);
        }
        if( hrd->fixed_pic_rate_within_cvs_flag[ i ] ) {
            hrd->elemental_duration_in_tc_minus1[ i ] = bs_read_ue(b);
        } else {
            hrd->low_delay_hrd_flag[ i ] = bs_read_u1(b);
        }
        if( hrd->low_delay_hrd_flag[ i ] ) {
            hrd->cpb_cnt_minus1[ i ] = bs_read_ue(b);
        }
        if( hrd->nal_hrd_parameters_present_flag ) {
            read_hevc_sub_layer_hrd_parameters(&hrd->sub_layer_hrd_nal[i], b, hrd->cpb_cnt_minus1[ i ] + 1, hrd->sub_pic_hrd_params_present_flag);
        }
        if( hrd->vcl_hrd_parameters_present_flag ) {
            read_hevc_sub_layer_hrd_parameters(&hrd->sub_layer_hrd_vcl[i], b, hrd->cpb_cnt_minus1[ i ] + 1, hrd->sub_pic_hrd_params_present_flag);
        }
    }
}

//Appendix E.2.3 Sub-layer HRD parameters syntax
void read_hevc_sub_layer_hrd_parameters(hevc_sub_layer_hrd_t* sub_layer_hrd, bs_t* b, int CpbCnt, int sub_pic_hrd_params_present_flag)
{
    for( int i = 0; i <= CpbCnt; i++ ) {
        sub_layer_hrd->bit_rate_value_minus1[i] = bs_read_ue(b);
        sub_layer_hrd->cpb_size_value_minus1[i] = bs_read_ue(b);
        if( sub_pic_hrd_params_present_flag ) {
            sub_layer_hrd->cpb_size_du_value_minus1[i] = bs_read_ue(b);
            sub_layer_hrd->bit_rate_du_value_minus1[i] = bs_read_ue(b);
        }
        sub_layer_hrd->cbr_flag[i] = bs_read_u1(b);
    }
}


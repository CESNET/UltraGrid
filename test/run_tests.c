/**
 * @file    run_tests.c
 * @author  Colin Perkins
 * @author  Martin Pulec
 */
/*
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of
 *      Glasgow Department of Computing Science
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdbool.h>

#include "debug.h"
#include "host.h"

#include "test_host.h"
#include "test_aes.h"
#include "test_alpha_blend.h"
#include "test_overlay_config.h"
#include "test_overlay_layout.h"
#include "test_overlay_pam.h"
#include "test_overlay_scale.h"
#include "test_overlay_soft_edge.h"
#include "test_overlay_watch.h"
#include "test_bitstream.h"
#include "test_des.h"
#include "test_md5.h"
#include "test_tv.h"
#include "test_net_udp.h"
#include "test_video_capture.h"
#include "test_video_display.h"
#include "test_sdp_parser.h"

#define TEST_AV_HW 1

/* These globals should be fixed in the future as well */
uint32_t hd_size_x = 1920;
uint32_t hd_size_y = 1080;
uint32_t hd_color_bpp = 3;
uint32_t bitdepth = 10;
uint32_t progressive = 0;
uint32_t hd_video_mode;

long packet_rate = 13600;

#define DECLARE_TEST(func) int func(void)
#define DEFINE_QUIET_TEST(func) { #func, func, true } // original tests that print status by itselves
#define DEFINE_TEST(func) { #func, func, false }

DECLARE_TEST(codec_conversion_test_testcard_uyvy_to_i420);
DECLARE_TEST(codec_conversion_test_y216_to_p010le);
DECLARE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r10k);
DECLARE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r12l);
DECLARE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48);
DECLARE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48_out_of_range);
DECLARE_TEST(ff_codec_conversions_test_pX10_from_to_v210);
DECLARE_TEST(get_framerate_test_2997);
DECLARE_TEST(get_framerate_test_3000);
DECLARE_TEST(get_framerate_test_free);
DECLARE_TEST(gpujpeg_test_simple);
DECLARE_TEST(libavcodec_test_get_decoder_from_uv_to_uv);
DECLARE_TEST(misc_test_color_coeff_range);
DECLARE_TEST(misc_test_net_getsockaddr);
DECLARE_TEST(misc_test_net_sockaddr_compare_v4_mapped);
DECLARE_TEST(misc_test_replace_all);
DECLARE_TEST(misc_test_unit_evaluate);
DECLARE_TEST(misc_test_video_desc_io_op_symmetry);
DECLARE_TEST(alpha_blend_test_rgba_alpha_zero);
DECLARE_TEST(alpha_blend_test_rgba_alpha_max);
DECLARE_TEST(alpha_blend_test_rgba_half_alpha);
DECLARE_TEST(alpha_blend_test_rgb_alpha_zero);
DECLARE_TEST(alpha_blend_test_rgb_alpha_max);
DECLARE_TEST(alpha_blend_test_uyvy_alpha_zero);
DECLARE_TEST(alpha_blend_test_uyvy_alpha_max_white);
DECLARE_TEST(alpha_blend_test_uyvy_alpha_max_red);
DECLARE_TEST(alpha_blend_test_yuyv_alpha_zero);
DECLARE_TEST(alpha_blend_test_yuyv_alpha_max_white);
DECLARE_TEST(alpha_blend_test_yuyv_alpha_max_red);
DECLARE_TEST(alpha_blend_test_y416_alpha_zero);
DECLARE_TEST(alpha_blend_test_y416_alpha_max_white);
DECLARE_TEST(alpha_blend_test_i420_alpha_zero);
DECLARE_TEST(alpha_blend_test_i420_alpha_max_white);
DECLARE_TEST(alpha_blend_test_i420_chroma_alpha_averaging);
DECLARE_TEST(alpha_blend_test_i420_subregion_strides);
DECLARE_TEST(alpha_blend_test_rg48_alpha_zero);
DECLARE_TEST(alpha_blend_test_rg48_alpha_max_white);
DECLARE_TEST(alpha_blend_test_v210_alpha_zero);
DECLARE_TEST(alpha_blend_test_v210_alpha_max_white);
DECLARE_TEST(alpha_blend_test_r10k_alpha_zero);
DECLARE_TEST(alpha_blend_test_r10k_alpha_max_white);
DECLARE_TEST(alpha_blend_test_r12l_alpha_zero);
DECLARE_TEST(alpha_blend_test_r12l_alpha_max_white);
DECLARE_TEST(overlay_pam_test_load_8bit_rgba);
DECLARE_TEST(overlay_pam_test_load_8bit_rgb_adds_alpha);
DECLARE_TEST(overlay_pam_test_load_16bit_rgba);
DECLARE_TEST(overlay_pam_test_rejects_missing_file);
DECLARE_TEST(overlay_pam_test_rejects_grayscale);
DECLARE_TEST(overlay_pam_test_rejects_intermediate_maxval);
DECLARE_TEST(overlay_layout_test_center);
DECLARE_TEST(overlay_layout_test_corners);
DECLARE_TEST(overlay_layout_test_custom_negative_from_edge);
DECLARE_TEST(overlay_layout_test_block_pixel_alignment);
DECLARE_TEST(overlay_layout_test_overlay_larger_than_frame);
DECLARE_TEST(overlay_layout_test_oversized_center);
DECLARE_TEST(overlay_layout_test_oversized_right);
DECLARE_TEST(overlay_layout_test_oversized_custom_positive);
DECLARE_TEST(overlay_layout_test_block_lines_alignment);
DECLARE_TEST(overlay_scale_test_identity);
DECLARE_TEST(overlay_scale_test_upscale_solid_colour);
DECLARE_TEST(overlay_scale_test_downscale_average);
DECLARE_TEST(overlay_scale_test_returns_null_on_bad_dims);
DECLARE_TEST(overlay_scale_test_source_buffer_unchanged);
DECLARE_TEST(overlay_scaler_test_create_destroy);
DECLARE_TEST(overlay_scaler_test_reuses_context_same_dims);
DECLARE_TEST(overlay_scaler_test_rebuilds_context_on_dim_change);
DECLARE_TEST(overlay_scaler_test_scale_into_no_alloc);
DECLARE_TEST(overlay_scaler_test_filter_nearest);
DECLARE_TEST(overlay_scaler_test_filter_bilinear);
DECLARE_TEST(overlay_soft_edge_test_zero_width_is_noop);
DECLARE_TEST(overlay_soft_edge_test_edge_pixel_zeroed);
DECLARE_TEST(overlay_soft_edge_test_linear_ramp);
DECLARE_TEST(overlay_soft_edge_test_centre_untouched);
DECLARE_TEST(overlay_soft_edge_test_rgb_components_unchanged);
DECLARE_TEST(overlay_soft_edge_test_oversized_width_clamps);
DECLARE_TEST(overlay_soft_edge_test_non_square);
DECLARE_TEST(overlay_soft_edge_test_exact_half_dimension);
DECLARE_TEST(overlay_soft_edge_test_scales_existing_alpha);
DECLARE_TEST(overlay_soft_edge_test_degenerate_one_row);
DECLARE_TEST(overlay_watch_test_init_no_change);
DECLARE_TEST(overlay_watch_test_detects_size_change);
DECLARE_TEST(overlay_watch_test_detects_mtime_change);
DECLARE_TEST(overlay_watch_test_ack_on_missing_file_preserves_baseline);
DECLARE_TEST(overlay_watch_test_missing_file_no_change);
DECLARE_TEST(overlay_watch_test_file_appears_after_init);
DECLARE_TEST(overlay_watch_test_detects_atomic_rename);
DECLARE_TEST(overlay_watch_test_changed_does_not_consume_baseline);
DECLARE_TEST(overlay_watch_test_ack_commits_baseline);
DECLARE_TEST(overlay_config_test_minimal_file_only);
DECLARE_TEST(overlay_config_test_position_keywords);
DECLARE_TEST(overlay_config_test_custom_xy);
DECLARE_TEST(overlay_config_test_help);
DECLARE_TEST(overlay_config_test_rejects_missing_file);
DECLARE_TEST(overlay_config_test_rejects_unknown_key);
DECLARE_TEST(overlay_config_test_rejects_bad_position);
DECLARE_TEST(overlay_config_test_rejects_non_integer_xy);
DECLARE_TEST(overlay_config_test_rejects_null_and_empty_value);
DECLARE_TEST(overlay_config_test_soft_edge);
DECLARE_TEST(overlay_config_test_scale);
DECLARE_TEST(overlay_config_test_scale_frame);
DECLARE_TEST(overlay_config_test_scale_frame_overrides_wxh);
DECLARE_TEST(overlay_config_test_perf);
DECLARE_TEST(overlay_config_test_scale_filter);
DECLARE_TEST(overlay_config_test_blend_threads);
DECLARE_TEST(overlay_config_test_rejects_oversize_options);

struct {
        const char *name;
        int (*test)(void);
        bool quiet;
} tests[] = {
        //DEFINE_QUIET_TEST(test_bitstream),
        DEFINE_QUIET_TEST(test_des),
        //DEFINE_QUIET_TEST(test_aes),
        DEFINE_QUIET_TEST(test_md5),
        DEFINE_QUIET_TEST(test_tv),
        DEFINE_QUIET_TEST(test_net_udp),
#ifdef TEST_AV_HW
        DEFINE_QUIET_TEST(test_video_capture),
        DEFINE_QUIET_TEST(test_video_display),
#endif
        DEFINE_TEST(codec_conversion_test_y216_to_p010le),
        DEFINE_TEST(codec_conversion_test_testcard_uyvy_to_i420),
#if defined HAVE_LAVC
        DEFINE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r10k),
        DEFINE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r12l),
        DEFINE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48),
        DEFINE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48_out_of_range),
        DEFINE_TEST(ff_codec_conversions_test_pX10_from_to_v210),
#endif // defined HAVE_LAVC
        DEFINE_TEST(get_framerate_test_2997),
        DEFINE_TEST(get_framerate_test_3000),
        DEFINE_TEST(get_framerate_test_free),
        DEFINE_TEST(gpujpeg_test_simple),
        DEFINE_TEST(libavcodec_test_get_decoder_from_uv_to_uv),
        DEFINE_TEST(misc_test_color_coeff_range),
        DEFINE_TEST(misc_test_net_getsockaddr),
        DEFINE_TEST(misc_test_net_sockaddr_compare_v4_mapped),
        DEFINE_TEST(misc_test_replace_all),
        DEFINE_TEST(misc_test_unit_evaluate),
        DEFINE_TEST(misc_test_video_desc_io_op_symmetry),
        DEFINE_TEST(test_sdp_parser),
        DEFINE_TEST(alpha_blend_test_rgba_alpha_zero),
        DEFINE_TEST(alpha_blend_test_rgba_alpha_max),
        DEFINE_TEST(alpha_blend_test_rgba_half_alpha),
        DEFINE_TEST(alpha_blend_test_rgb_alpha_zero),
        DEFINE_TEST(alpha_blend_test_rgb_alpha_max),
        DEFINE_TEST(alpha_blend_test_uyvy_alpha_zero),
        DEFINE_TEST(alpha_blend_test_uyvy_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_uyvy_alpha_max_red),
        DEFINE_TEST(alpha_blend_test_yuyv_alpha_zero),
        DEFINE_TEST(alpha_blend_test_yuyv_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_yuyv_alpha_max_red),
        DEFINE_TEST(alpha_blend_test_y416_alpha_zero),
        DEFINE_TEST(alpha_blend_test_y416_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_i420_alpha_zero),
        DEFINE_TEST(alpha_blend_test_i420_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_i420_chroma_alpha_averaging),
        DEFINE_TEST(alpha_blend_test_i420_subregion_strides),
        DEFINE_TEST(alpha_blend_test_rg48_alpha_zero),
        DEFINE_TEST(alpha_blend_test_rg48_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_v210_alpha_zero),
        DEFINE_TEST(alpha_blend_test_v210_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_r10k_alpha_zero),
        DEFINE_TEST(alpha_blend_test_r10k_alpha_max_white),
        DEFINE_TEST(alpha_blend_test_r12l_alpha_zero),
        DEFINE_TEST(alpha_blend_test_r12l_alpha_max_white),
        DEFINE_TEST(overlay_pam_test_load_8bit_rgba),
        DEFINE_TEST(overlay_pam_test_load_8bit_rgb_adds_alpha),
        DEFINE_TEST(overlay_pam_test_load_16bit_rgba),
        DEFINE_TEST(overlay_pam_test_rejects_missing_file),
        DEFINE_TEST(overlay_pam_test_rejects_grayscale),
        DEFINE_TEST(overlay_pam_test_rejects_intermediate_maxval),
        DEFINE_TEST(overlay_layout_test_center),
        DEFINE_TEST(overlay_layout_test_corners),
        DEFINE_TEST(overlay_layout_test_custom_negative_from_edge),
        DEFINE_TEST(overlay_layout_test_block_pixel_alignment),
        DEFINE_TEST(overlay_layout_test_overlay_larger_than_frame),
        DEFINE_TEST(overlay_layout_test_oversized_center),
        DEFINE_TEST(overlay_layout_test_oversized_right),
        DEFINE_TEST(overlay_layout_test_oversized_custom_positive),
        DEFINE_TEST(overlay_layout_test_block_lines_alignment),
        DEFINE_TEST(overlay_scale_test_identity),
        DEFINE_TEST(overlay_scale_test_upscale_solid_colour),
        DEFINE_TEST(overlay_scale_test_downscale_average),
        DEFINE_TEST(overlay_scale_test_returns_null_on_bad_dims),
        DEFINE_TEST(overlay_scale_test_source_buffer_unchanged),
        DEFINE_TEST(overlay_scaler_test_create_destroy),
        DEFINE_TEST(overlay_scaler_test_reuses_context_same_dims),
        DEFINE_TEST(overlay_scaler_test_rebuilds_context_on_dim_change),
        DEFINE_TEST(overlay_scaler_test_scale_into_no_alloc),
        DEFINE_TEST(overlay_scaler_test_filter_nearest),
        DEFINE_TEST(overlay_scaler_test_filter_bilinear),
        DEFINE_TEST(overlay_soft_edge_test_zero_width_is_noop),
        DEFINE_TEST(overlay_soft_edge_test_edge_pixel_zeroed),
        DEFINE_TEST(overlay_soft_edge_test_linear_ramp),
        DEFINE_TEST(overlay_soft_edge_test_centre_untouched),
        DEFINE_TEST(overlay_soft_edge_test_rgb_components_unchanged),
        DEFINE_TEST(overlay_soft_edge_test_oversized_width_clamps),
        DEFINE_TEST(overlay_soft_edge_test_non_square),
        DEFINE_TEST(overlay_soft_edge_test_exact_half_dimension),
        DEFINE_TEST(overlay_soft_edge_test_scales_existing_alpha),
        DEFINE_TEST(overlay_soft_edge_test_degenerate_one_row),
        DEFINE_TEST(overlay_watch_test_init_no_change),
        DEFINE_TEST(overlay_watch_test_detects_size_change),
        DEFINE_TEST(overlay_watch_test_detects_mtime_change),
        DEFINE_TEST(overlay_watch_test_ack_on_missing_file_preserves_baseline),
        DEFINE_TEST(overlay_watch_test_missing_file_no_change),
        DEFINE_TEST(overlay_watch_test_file_appears_after_init),
        DEFINE_TEST(overlay_watch_test_detects_atomic_rename),
        DEFINE_TEST(overlay_watch_test_changed_does_not_consume_baseline),
        DEFINE_TEST(overlay_watch_test_ack_commits_baseline),
        DEFINE_TEST(overlay_config_test_minimal_file_only),
        DEFINE_TEST(overlay_config_test_position_keywords),
        DEFINE_TEST(overlay_config_test_custom_xy),
        DEFINE_TEST(overlay_config_test_help),
        DEFINE_TEST(overlay_config_test_rejects_missing_file),
        DEFINE_TEST(overlay_config_test_rejects_unknown_key),
        DEFINE_TEST(overlay_config_test_rejects_bad_position),
        DEFINE_TEST(overlay_config_test_rejects_non_integer_xy),
        DEFINE_TEST(overlay_config_test_rejects_null_and_empty_value),
        DEFINE_TEST(overlay_config_test_soft_edge),
        DEFINE_TEST(overlay_config_test_scale),
        DEFINE_TEST(overlay_config_test_scale_frame),
        DEFINE_TEST(overlay_config_test_scale_frame_overrides_wxh),
        DEFINE_TEST(overlay_config_test_perf),
        DEFINE_TEST(overlay_config_test_scale_filter),
        DEFINE_TEST(overlay_config_test_blend_threads),
        DEFINE_TEST(overlay_config_test_rejects_oversize_options),
};

static bool test_helper(const char *name, int (*func)(), bool quiet) {
        int ret = func();
        if (!quiet) {
                char msg_start[] = "Testing ";
                size_t len = sizeof msg_start + strlen(name);
                fprintf(stderr, "%s%s ", msg_start, name);
                for (int i = len; i < 74; ++i) {
                        fprintf(stderr, ".");
                }
                fprintf(stderr, " %s\n", ret == 0 ? "Ok" : ret < 0 ? "FAIL" : "--");
        }
        return ret >= 0;
}

static bool run_tests(const char *test)
{
        if (test) {
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        if (strcmp(test, tests[i].name) == 0) {
                                return test_helper(tests[i].name, tests[i].test, tests[i].quiet);
                        }
                }
                fprintf(stderr, "No such a test named \"%s!\"\n", test);
                return false;
        }
        bool ret = true;
        for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                ret = test_helper(tests[i].name, tests[i].test, tests[i].quiet) && ret;
        }
        return ret;
}

int main(int argc, char **argv)
{
        if (argc > 1 && (strcmp("-h", argv[1]) == 0 || strcmp("--help", argv[1]) == 0)) {
                printf("Usage:\n\t%s [-V] [ all | <test_name> | -h | --help ]\n", argv[0]);
                printf("\nwhere\n"
                       "\t-V[V[V]] | --verbose - verbose (use UG log level verbose/debug/debug2, default fatal)\n"
                       "\t     <test_name>     - run only test of given name\n");
                printf("\nAvailable tests:\n");
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        printf(" - %s\n", tests[i].name);
                }
                return 0;
        }

        struct init_data *init = NULL;
        if ((init = common_preinit(argc, argv)) == NULL) {
                return 2;
        }
        log_level = LOG_LEVEL_FATAL;

        argc -= 1;
        argv += 1;
        if (argc >= 1 && (strstr(argv[0], "-V") == argv[0] || strstr(argv[0], "--verbose") == argv[0])) { // handled in common_preinit
                argc -= 1;
                argv += 1;
        }

        const char *test_name = NULL;;
        if (argc == 1) {
                if (strcmp("all", argv[0]) == 0) {
                } else {
                        test_name = argv[0];
                }
        }

        bool success = run_tests(test_name);

        common_cleanup(init);

        // Return error code 1 if the one of test failed.
        return success ? 0 : 1;
}


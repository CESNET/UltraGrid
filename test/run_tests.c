/**
 * @file    run_tests.c
 * @author  Colin Perkins
 * @author  Martin Pulec
 */
/*
 * Copyright (c) 2004 University of Glasgow
 * Copyright (c) 2005-2023 CESNET, z. s .p .o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <stdbool.h>

#include "debug.h"
#include "host.h"

#include "test_host.h"
#include "test_aes.h"
#include "test_bitstream.h"
#include "test_des.h"
#include "test_md5.h"
#include "test_random.h"
#include "test_tv.h"
#include "test_net_udp.h"
#include "test_rtp.h"
#include "test_video_capture.h"
#include "test_video_display.h"

#define TEST_AV_HW 1

/* These globals should be fixed in the future as well */
uint32_t hd_size_x = 1920;
uint32_t hd_size_y = 1080;
uint32_t hd_color_bpp = 3;
uint32_t bitdepth = 10;
uint32_t progressive = 0;
uint32_t hd_video_mode;

long packet_rate = 13600;

void exit_uv(int status);

void exit_uv(int status)
{
        exit(status);
}

#define DECLARE_TEST(func) int func(void)
#define DEFINE_QUIET_TEST(func) { #func, func, true } // original tests that print status by itselves
#define DEFINE_TEST(func) { #func, func, false }

DECLARE_TEST(codec_conversion_test_testcard_uyvy_to_i420);
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
DECLARE_TEST(misc_test_replace_all);
DECLARE_TEST(misc_test_video_desc_io_op_symmetry);

struct {
        const char *name;
        int (*test)(void);
        bool quiet;
} tests[] = {
        DEFINE_QUIET_TEST(test_bitstream),
        DEFINE_QUIET_TEST(test_des),
        //DEFINE_QUIET_TEST(test_aes),
        DEFINE_QUIET_TEST(test_md5),
        DEFINE_QUIET_TEST(test_random),
        DEFINE_QUIET_TEST(test_tv),
        DEFINE_QUIET_TEST(test_net_udp),
        DEFINE_QUIET_TEST(test_rtp),
#ifdef TEST_AV_HW
        DEFINE_QUIET_TEST(test_video_capture),
        DEFINE_QUIET_TEST(test_video_display),
#endif
        DEFINE_TEST(codec_conversion_test_testcard_uyvy_to_i420),
        DEFINE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r10k),
        DEFINE_TEST(ff_codec_conversions_test_yuv444pXXle_from_to_r12l),
        DEFINE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48),
        DEFINE_TEST(ff_codec_conversions_test_yuv444p16le_from_to_rg48_out_of_range),
        DEFINE_TEST(ff_codec_conversions_test_pX10_from_to_v210),
        DEFINE_TEST(get_framerate_test_2997),
        DEFINE_TEST(get_framerate_test_3000),
        DEFINE_TEST(get_framerate_test_free),
        DEFINE_TEST(gpujpeg_test_simple),
        DEFINE_TEST(libavcodec_test_get_decoder_from_uv_to_uv),
        DEFINE_TEST(misc_test_replace_all),
        DEFINE_TEST(misc_test_video_desc_io_op_symmetry),
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
                if (getenv("GITHUB_REPOSITORY") != NULL && strcmp(tests[i].name, "test_net_udp") == 0) {
                        continue; // skip this test in CI
                }
                ret = test_helper(tests[i].name, tests[i].test, tests[i].quiet) && ret;
        }
        return ret;
}

int main(int argc, char **argv)
{
        if (argc > 1 && (strcmp("-h", argv[1]) == 0 || strcmp("--help", argv[1]) == 0)) {
                printf("Usage:\n\t%s [-V] [ all | <test_name> | -h | --help ]\n", argv[0]);
                printf("\nwhere\n"
                       "\t  -V[V[V]]  - verbose (use UG log level verbose/debug/debug2, default fatal)\n"
                       "\t<test_name> - run only test of given name\n");
                printf("\nAvailable tests:\n");
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        printf(" - %s\n", tests[i].name);
                }
                return 0;
        }

        log_level = LOG_LEVEL_FATAL;
        struct init_data *init = NULL;
        if ((init = common_preinit(argc, argv)) == NULL) {
                return 2;
        }

        argc -= 1;
        argv += 1;
        if (argc >= 1 && strncmp(argv[0], "-V", 2) == 0) { // handled in common_preinit
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


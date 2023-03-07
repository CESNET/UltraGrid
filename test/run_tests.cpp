/*
 * FILE:    run_tests.cpp
 * AUTHORS: Colin Perkins
 *
 * Copyright (c) 2004 University of Glasgow
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
 *
 * $Revision: 1.2 $
 * $Date: 2008/01/10 11:07:42 $
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <iostream>
#include <string>

#include "debug.h"
#include "host.h"

extern "C" {
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
}

using std::cerr;
using std::cout;
using std::string;

#define TEST_AV_HW 1

/* These globals should be fixed in the future as well */
uint32_t hd_size_x = 1920;
uint32_t hd_size_y = 1080;
uint32_t hd_color_bpp = 3;
uint32_t bitdepth = 10;
uint32_t progressive = 0;
uint32_t hd_video_mode;

long packet_rate = 13600;

extern "C" void exit_uv(int status);

void exit_uv(int status)
{
        exit(status);
}

static bool run_standard_tests()
{
        bool success = true;

        if (test_bitstream() != 0)
                success = false;
        if (test_des() != 0)
                success = false;
#if 0
        if (test_aes() != 0)
                success = false;
#endif
        if (test_md5() != 0)
                success = false;
        if (test_random() != 0)
                success = false;
        if (test_tv() != 0)
                success = false;
        if (test_net_udp() != 0)
                success = getenv("GITHUB_REPOSITORY") != NULL; // ignore failure if run in CI
        if (test_rtp() != 0)
                success = false;

#ifdef TEST_AV_HW
        if (test_video_capture() != 0)
                success = false;
        if (test_video_display() != 0)
                success = false;
#endif

        return success;
}

#define DECLARE_TEST(func) extern "C" bool func(void)
#define DEFINE_TEST(func) { #func, func }

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
        bool (*test)(void);
} tests[] {
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

static bool test_helper(const char *name, bool (*func)()) {
        bool ret = func();
        char msg_start[] = "Testing ";
        size_t len = sizeof msg_start + strlen(name);
        cerr << msg_start << name << " ";
        for (int i = len; i < 74; ++i) {
                cerr << ".";
        }
        cerr << " " << (ret ? "Ok" : "FAIL" ) << "\n";
        return ret;
}

static bool run_unit_tests([[maybe_unused]] string const &test)
{
        bool ret = true;
        if (!test.empty()) {
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        if (test == tests[i].name) {
                                return test_helper(tests[i].name, tests[i].test);
                        }
                }
        } else {
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        ret = test_helper(tests[i].name, tests[i].test) && ret;
                }
        }
        return ret;
}

int main(int argc, char **argv)
{
        if (argc > 1 && (strcmp("-h", argv[1]) == 0 || strcmp("--help", argv[1]) == 0)) {
                cout << "Usage:\n\t" << argv[0] << " [ unit | standard | all | <test_name> | -h | --help ]\n";
                cout << "where\n\t<test_name> - run only unit test of given name\n";
                cout << "\nAvailable unit tests:\n";
                for (unsigned i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
                        cout << " - " << tests[i].name << "\n";
                }
                return 0;
        }

        struct init_data *init = nullptr;
        if ((init = common_preinit(argc, argv)) == nullptr) {
                return 2;
        }

        bool run_standard = true;
        bool run_unit = true;
        string run_unit_test_name{};
        if (argc == 2) {
                run_standard = run_unit = false;
                if (strcmp("unit", argv[1]) == 0) {
                        run_unit = true;
                } else if (strcmp("standard", argv[1]) == 0) {
                        run_standard = true;
                } else if (strcmp("all", argv[1]) == 0) {
                        run_standard = run_unit = true;
                } else {
                        run_unit_test_name = argv[1];
                        run_unit = true;
                }
        }

        bool success = (run_standard ? run_standard_tests() : true);
        success = (run_unit ? run_unit_tests(run_unit_test_name) : true) && success;

        common_cleanup(init);

        // Return error code 1 if the one of test failed.
        return success ? 0 : 1;
}


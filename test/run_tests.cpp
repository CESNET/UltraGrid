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

#ifdef HAVE_CPPUNIT
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#endif
#include <iostream>

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

using std::clog;
using std::cout;

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
        if (!getenv("UG_SKIP_NET_TESTS") && test_net_udp() != 0)
                success = false;
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

static bool run_unit_tests()
{
#ifdef HAVE_CPPUNIT
        std::clog << "Running CppUnit tests:\n";
        // Get the top level suite from the registry
        CPPUNIT_NS::Test *suite = CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest();

        // Adds the test to the list of test to run
        CPPUNIT_NS::TextUi::TestRunner runner;
        runner.addTest( suite );

        // Change the default outputter to a compiler error format outputter
        runner.setOutputter( new CPPUNIT_NS::CompilerOutputter( &runner.result(),
                                CPPUNIT_NS::stdCOut() ) );
        // Run the test.
        return runner.run();
#endif
        std::clog << "CppUnit was not found, skipping CppUnit tests!\n";
        return true;
}

int main(int argc, char **argv)
{
        if (argc > 1 && (strcmp("-h", argv[1]) == 0 || strcmp("--help", argv[1]) == 0)) {
                cout << "Usage:\n\t" << argv[0] << " [unit|standard|all|-h|--help]\n";
                return 0;
        }

        struct init_data *init = nullptr;
        if ((init = common_preinit(argc, argv, nullptr)) == nullptr) {
                return 2;
        }

        bool run_standard = true;
        bool run_unit = true;
        if (argc == 2) {
                run_standard = run_unit = false;
                if (strcmp("unit", argv[1]) == 0) {
                        run_unit = true;
                }
                if (strcmp("standard", argv[1]) == 0) {
                        run_standard = true;
                }
                if (strcmp("all", argv[1]) == 0) {
                        run_standard = run_unit = true;
                }
        }

        bool success = (run_standard ? run_standard_tests() : true);
        success = (run_unit ? run_unit_tests() : true) && success;

        common_cleanup(init);

        // Return error code 1 if the one of test failed.
        return success ? 0 : 1;
}


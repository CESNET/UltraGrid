#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cmath>
#include <iostream>
#include <string>
#include "unit_common.h"
#include "utils/misc.h"

/// for 10 FPS the difference between 10000/1000 and 10000/1001 is ~0.001
/// and this needs to be less than a half of that
#define EPS 0.0049

using std::cerr;
using std::string;
using std::to_string;

extern "C" {
        bool get_framerate_test();
        bool get_framerate_test_2997();
        bool get_framerate_test_3000();
        bool get_framerate_test_free();
}

bool get_framerate_test_2997()
{
        // approx
        ASSERT_EQUAL(30000, get_framerate_n(29.97));
        ASSERT_EQUAL(1001, get_framerate_d(29.97));

        // "exactly" 30000/1001
        ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001));
        ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001));

        // with epsilon
        ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001 + EPS));
        ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001 + EPS));
        ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001 - EPS));
        ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001 - EPS));
        return true;
}

bool get_framerate_test_3000()
{
        ASSERT_EQUAL(30000, get_framerate_n(30));
        ASSERT_EQUAL(1000, get_framerate_d(30));

        ASSERT_EQUAL(30000, get_framerate_n(30 + EPS));
        ASSERT_EQUAL(1000, get_framerate_d(30 + EPS));
        ASSERT_EQUAL(30000, get_framerate_n(30 - EPS));
        ASSERT_EQUAL(1000, get_framerate_d(30 - EPS));
        return true;
}

bool get_framerate_test_free()
{
        for (int i = 10; i < 480; ++i) {
                // base 1000
                string num_str = to_string(i);
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i));
                ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i));
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i + EPS));
                ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i + EPS));
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i - EPS));
                ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i - EPS));

                // base 1001
                double num = i * 1000.0 / 1001.0;
                num_str = to_string(num);
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num));
                ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num));
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num + EPS));
                ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num + EPS));
                ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num - EPS));
                ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num - EPS));

                // halves
                num = i + 0.5;
                num_str = to_string(num);
                ASSERT_EQUAL_MESSAGE(num_str, (int) round(num * 1000), get_framerate_n(num));
                ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(num));
        }
        return true;
}


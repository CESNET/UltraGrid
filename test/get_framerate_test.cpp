#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_CPPUNIT

#include <cppunit/config/SourcePrefix.h>
#include <cmath>
#include <string>
#include "get_framerate_test.hpp"
#include "utils/misc.h"

/// for 10 FPS the difference between 10000/1000 and 10000/1001 is ~0.001
/// and this needs to be less than a half of that
#define EPS 0.0049

using std::string;
using std::to_string;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( get_framerate_test );

get_framerate_test::get_framerate_test()
{
}

get_framerate_test::~get_framerate_test()
{
}

void
get_framerate_test::setUp()
{
}


void
get_framerate_test::tearDown()
{
}

void
get_framerate_test::test_2997()
{
        // approx
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(29.97));
        CPPUNIT_ASSERT_EQUAL(1001, get_framerate_d(29.97));

        // "exactly" 30000/1001
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001));
        CPPUNIT_ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001));

        // with epsilon
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001 + EPS));
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30000.0/1001 + EPS));
        CPPUNIT_ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001 - EPS));
        CPPUNIT_ASSERT_EQUAL(1001, get_framerate_d(30000.0/1001 - EPS));
}

void
get_framerate_test::test_3000()
{
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30));
        CPPUNIT_ASSERT_EQUAL(1000, get_framerate_d(30));

        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30 + EPS));
        CPPUNIT_ASSERT_EQUAL(1000, get_framerate_d(30 + EPS));
        CPPUNIT_ASSERT_EQUAL(30000, get_framerate_n(30 - EPS));
        CPPUNIT_ASSERT_EQUAL(1000, get_framerate_d(30 - EPS));
}

void
get_framerate_test::test_free()
{
        for (int i = 10; i < 480; ++i) {
                // base 1000
                string num_str = to_string(i);
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i + EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i + EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(i - EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(i - EPS));

                // base 1001
                double num = i * 1000.0 / 1001.0;
                num_str = to_string(num);
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num + EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num + EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, i * 1000, get_framerate_n(num - EPS));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1001, get_framerate_d(num - EPS));

                // halves
                num = i + 0.5;
                num_str = to_string(num);
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, (int) round(num * 1000), get_framerate_n(num));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(num_str, 1000, get_framerate_d(num));
        }
}

#endif // defined HAVE_CPPUNIT

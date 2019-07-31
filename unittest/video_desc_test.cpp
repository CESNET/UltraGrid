#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cppunit/config/SourcePrefix.h>
#include "video_desc_test.h"

#include <sstream>
#include "video.h"

using namespace std;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( video_desc_test );

video_desc_test::video_desc_test() : m_test_desc{{1920, 1080, UYVY, 25, INTERLACED_MERGED, 1},
        {1920, 1080, DXT5, 60, PROGRESSIVE, 4},
        {640, 480, H264, 15, PROGRESSIVE, 1}}
{
}

video_desc_test::~video_desc_test()
{
}

void
video_desc_test::setUp()
{
}


void
video_desc_test::tearDown()
{
}

void
video_desc_test::testIOOperatorSymetry()
{
        for (const auto & i : m_test_desc) {
                video_desc tmp;

                ostringstream oss;
                oss << i;
                istringstream iss(oss.str());
                iss >> tmp;

                // Check
                ostringstream oss2;
                oss2 << tmp;
                string err_elem = oss.str() + " vs " +  oss2.str();
                CPPUNIT_ASSERT_MESSAGE(err_elem, video_desc_eq(tmp, i));
                CPPUNIT_ASSERT_EQUAL_MESSAGE(err_elem, tmp, i);
        }
}


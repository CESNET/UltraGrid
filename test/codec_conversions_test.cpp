#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_CPPUNIT

#include <cppunit/config/SourcePrefix.h>
#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "codec_conversions_test.h"
#include "video_codec.h"
#include "video_capture/testcard_common.h"

using std::list;
using std::pair;
using std::string;
using std::to_string;
using std::ostringstream;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( codec_conversions_test );

codec_conversions_test::codec_conversions_test()
{
}

codec_conversions_test::~codec_conversions_test()
{
}

void
codec_conversions_test::setUp()
{
}

void
codec_conversions_test::tearDown()
{
}

void
codec_conversions_test::test_testcard_uyvy_to_i420()
{
        list<pair<size_t,size_t>> sizes = { {1, 2}, {2, 1}, { 16, 1}, {16, 16}, {127, 255} };
        for (auto &i : sizes) {
                size_t size_x = i.first;
                size_t size_y = i.second;
                /// @todo Check also if chroma is horizontally interpolated in UV planes
                unsigned char uyvy_pattern[4] = { 'u', 'y', 'v', 'Y' };
                size_t uyvy_buf_size = ((size_x + 1) & ~1) * 2 * size_y;

                unsigned char uyvy_buf[uyvy_buf_size];
                for (size_t i = 0; i < size_y * 2 * ((size_x + 1) & ~1); ++i) {
                        uyvy_buf[i] = uyvy_pattern[i % 4];
                }

                auto *i420_buf = (unsigned char *) malloc(vc_get_datalen(size_x, size_y, I420));
                testcard_convert_buffer(UYVY, I420, i420_buf, uyvy_buf, size_x, size_y);
                unsigned char *y_ptr = i420_buf;
                for (size_t i = 0; i < size_y; ++i) {
                        for (size_t j = 0; j < size_x; ++j) {
                                ostringstream oss;
                                unsigned char expected = uyvy_pattern[((2 * j + 1) % 4)];
                                unsigned char actual = *y_ptr++;
                                oss << size_x << "X" << size_y << ": [" << i << ", " << j << "] expected " << expected << ", actual: " << actual << "\n";
                                CPPUNIT_ASSERT_EQUAL_MESSAGE(oss.str(), expected, actual);
                        }
                }
                // U
                unsigned char *u_ptr = i420_buf + size_x * size_y;
                for (size_t i = 0; i < (size_y + 1) / 2; ++i) {
                        for (size_t j = 0; j < (size_x + 1) / 2; ++j) {
                                ostringstream oss;
                                unsigned char expected = 'u';
                                unsigned char actual = *u_ptr++;
                                oss << "[" << i << ", " << j << "] expected " << expected << ", actual: " << actual << "\n";
                                CPPUNIT_ASSERT_EQUAL_MESSAGE(oss.str(), expected, actual);
                        }
                }
                // V
                unsigned char *v_ptr = i420_buf + size_x * size_y + ((size_x + 1) / 2) * ((size_y + 1) / 2);
                for (size_t i = 0; i < (size_y + 1) / 2; ++i) {
                        for (size_t j = 0; j < (size_x + 1) / 2; ++j) {
                                ostringstream oss;
                                unsigned char expected = 'v';
                                unsigned char actual = *v_ptr++;
                                oss << "[" << i << ", " << j << "] expected " << expected << ", actual: " << actual << "\n";
                                CPPUNIT_ASSERT_EQUAL_MESSAGE(oss.str(), expected, actual);
                        }
                }
                free(i420_buf);
        }
}

#endif // defined HAVE_CPPUNIT

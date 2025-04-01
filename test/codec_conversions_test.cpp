#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pixfmt_conv.h"
#include "unit_common.h"
#include "video_codec.h"
#include "video_capture/testcard_common.h"

using std::cerr;
using std::list;
using std::pair;
using std::string;
using std::to_string;
using std::ostringstream;
using std::vector;

extern "C" int codec_conversion_test_testcard_uyvy_to_i420(void);
extern "C" int codec_conversion_test_y216_to_p010le(void);

/**
 * The test is intended mainly for support for odd sizes, the pattern is fixed,
 * which is suboptimal.
 */
int codec_conversion_test_testcard_uyvy_to_i420(void)
{
        list<pair<size_t,size_t>> sizes = { {1, 2}, {2, 1}, { 16, 1}, {16, 16}, {127, 255} };
        for (auto &i : sizes) {
                size_t size_x = i.first;
                size_t size_y = i.second;
                /// @todo Check also if chroma is horizontally interpolated in UV planes
                unsigned char uyvy_pattern[4] = { 'u', 'y', 'v', 'Y' };
                size_t uyvy_buf_size = ((size_x + 1) & ~1) * 2 * size_y;

                vector<unsigned char> uyvy_buf(uyvy_buf_size);
                for (size_t i = 0; i < size_y * 2 * ((size_x + 1) & ~1); ++i) {
                        uyvy_buf[i] = uyvy_pattern[i % 4];
                }

                auto *i420_buf = (unsigned char *) malloc(vc_get_datalen(size_x, size_y, I420));
                testcard_convert_buffer(UYVY, I420, i420_buf, uyvy_buf.data(),
                                        size_x, size_y);
                unsigned char *y_ptr = i420_buf;
                for (size_t i = 0; i < size_y; ++i) {
                        for (size_t j = 0; j < size_x; ++j) {
                                ostringstream oss;
                                unsigned char expected = uyvy_pattern[((2 * j + 1) % 4)];
                                unsigned char actual = *y_ptr++;
                                oss << "size: " << size_x << "x" << size_y
                                    << ": [" << i << ", " << j << "]";
                                ASSERT_EQUAL_MESSAGE(oss.str(), expected,
                                                     actual);
                        }
                }
                // U
                unsigned char *u_ptr = i420_buf + size_x * size_y;
                for (size_t i = 0; i < (size_y + 1) / 2; ++i) {
                        for (size_t j = 0; j < (size_x + 1) / 2; ++j) {
                                ostringstream oss;
                                unsigned char expected = 'u';
                                unsigned char actual = *u_ptr++;
                                oss << "size: " << size_x << "x" << size_y
                                    << ": [" << i << ", " << j << "]";
                                ASSERT_EQUAL_MESSAGE(oss.str(), expected, actual);
                        }
                }
                // V
                unsigned char *v_ptr = i420_buf + size_x * size_y + ((size_x + 1) / 2) * ((size_y + 1) / 2);
                for (size_t i = 0; i < (size_y + 1) / 2; ++i) {
                        for (size_t j = 0; j < (size_x + 1) / 2; ++j) {
                                ostringstream oss;
                                unsigned char expected = 'v';
                                unsigned char actual = *v_ptr++;
                                oss << "[" << i << ", " << j << "]";
                                ASSERT_EQUAL_MESSAGE(oss.str(), expected, actual);
                        }
                }
                free(i420_buf);
        }
        return 0;
}

/**
 * The test is intended mainly for support for odd sizes, the pattern is fixed,
 * which is suboptimal.
 */
int
codec_conversion_test_y216_to_p010le(void)
{
        list<pair<size_t, size_t>> sizes = {
                { 1,   1   },
                { 1,   2   },
                { 2,   1   },
                { 2,   2   },
                { 2,   3   },
                { 15,  1   },
                { 16,  1   },
                { 16,  16  },
                { 127, 255 },
                { 128, 256 },
                { 255, 1   },
                { 255, 2   },
        };
        for (auto &i : sizes) {
                size_t size_x = i.first;
                size_t size_y = i.second;
                /// @todo Check also if chroma is horizontally interpolated in UV planes
                constexpr uint16_t u              = 'U' << 8 | 'u';
                constexpr uint16_t y1             = 'Y' << 8 | '1';
                constexpr uint16_t v              = 'V' << 8 | 'v';
                constexpr uint16_t y2             = 'Y' << 8 | '2';
                constexpr uint16_t y216_pattern[] = { y1, u, y2, v };
                size_t y216_buf_size = vc_get_linesize(size_x, Y216) * size_y;

                vector<uint16_t> y216_buf(y216_buf_size);
                for (size_t i = 0; i < size_y; ++i) {
                        uint16_t * line = y216_buf.data() + i * ((size_x + 1) & ~1) * 2;
                        for (size_t j = 0; j < ((size_x + 1) & ~1) * 2; ++j) {
                                line[j] = y216_pattern[j % 4];
                        }
                }

                vector<uint16_t> p010_buf(vc_get_datalen(size_x, size_y, I420) * 2);

                int            out_linesize[] = { (int) size_x * 2,
                                                  (int) ((size_x + 1) & ~1) * 2 };
                unsigned char *out_data[]     = {
                        (unsigned char *) &p010_buf[0],
                        (unsigned char *) &p010_buf[size_x * size_y]
                };
                y216_to_p010le(out_data, out_linesize,
                               (unsigned char *) &y216_buf[0], size_x, size_y);

                auto *y_ptr = (uint16_t *) (void *) out_data[0];
                for (size_t i = 0; i < size_y; ++i) {
                        for (size_t j = 0; j < size_x; ++j) {
                                ostringstream oss;
                                auto expected = y216_pattern[((2 * j) % 4)];
                                auto actual = *y_ptr++;
                                oss << "size: " << size_x << "x" << size_y
                                    << ": [" << i << ", " << j << "]";
                                ASSERT_EQUAL_MESSAGE(oss.str(), expected,
                                                     actual);
                        }
                }
                // UV combined
                for (size_t i = 0; i < (size_y + 1) / 2; ++i) {
                        auto *uv_ptr = ((uint16_t *) (void *) out_data[1]) +
                                       i * ((size_x + 1) & ~1);
                        for (size_t j = 0; j < ((size_x + 1) & ~1); ++j) {
                                ostringstream oss;
                                auto expected = y216_pattern[((2 * j + 1) % 4)];
                                auto actual = *uv_ptr++;
                                oss << "size: " << size_x << "x" << size_y
                                    << ": [" << i << ", " << j << "]";
                                ASSERT_EQUAL_MESSAGE(oss.str(), expected,
                                                     actual);
                        }
                }
        }
        return 0;
}

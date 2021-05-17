#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#if defined HAVE_CPPUNIT && defined HAVE_LAVC

#include <algorithm>
#include <array>
#include <cppunit/config/SourcePrefix.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ff_codec_conversions_test.h"
#include "libavcodec_common.h"
#include "tv.h"
#include "video_capture/testcard_common.h"
#include "video_codec.h"

using std::array;
using std::copy;
using std::cout;
using std::default_random_engine;
using std::ifstream;
using std::min;
using std::max;
using std::to_string;
using std::vector;

constexpr int MIN_12B = 16;
constexpr int MAX_12B = 4079;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( ff_codec_conversions_test );

ff_codec_conversions_test::ff_codec_conversions_test()
{
}

ff_codec_conversions_test::~ff_codec_conversions_test()
{
}

void
ff_codec_conversions_test::setUp()
{
}

void
ff_codec_conversions_test::tearDown()
{
}

#define TIMER(t) struct timeval t{}; gettimeofday(&(t), nullptr)
void
ff_codec_conversions_test::test_yuv444pXXle_from_to_r10k()
{
        using namespace std::string_literals;

        constexpr int width = 320;
        constexpr int height = 240;
        vector <unsigned char> rgba_buf(width * height * 4);

        /// @todo Use 10-bit natively
        auto test_pattern = [&](AVPixelFormat avfmt) {
                vector <unsigned char> r10k_buf(width * height * 4);
                copy(rgba_buf.begin(), rgba_buf.end(), r10k_buf.begin());
                toR10k(r10k_buf.data(), width, height);

                AVFrame frame;
                frame.format = avfmt;
                frame.width = width;
                frame.height = height;

                /* the image can be allocated by any means and av_image_alloc() is
                 * just the most convenient way if av_malloc() is to be used */
                assert(av_image_alloc(frame.data, frame.linesize,
                                        width, height, (AVPixelFormat) frame.format, 32) >= 0);

                auto from_conv = get_uv_to_av_conversion(R10k, frame.format);
                auto to_conv = get_av_to_uv_conversion(frame.format, R10k);
                assert(to_conv != nullptr && from_conv != nullptr);

                TIMER(t0);
                from_conv(&frame, r10k_buf.data(), width, height);
                TIMER(t1);
                to_conv(reinterpret_cast<char*>(r10k_buf.data()), &frame, width, height, vc_get_linesize(width, R10k), nullptr);
                TIMER(t2);

                if (getenv("PERF") != nullptr) {
                        cout << "test_yuv444p16le_from_to_r10k: duration - enc " << tv_diff(t1, t0) << ", dec " <<tv_diff(t2, t1) << "\n";
                }

                av_freep(frame.data);

                vector <unsigned char> rgba_buf_res(width * height * 4);
                vc_copyliner10k(rgba_buf_res.data(), r10k_buf.data(), height * vc_get_linesize(width, RGBA), 0, 8, 16);

                int max_diff = 0;
                for (size_t i = 0; i < width * height; ++i) {
                        for (int j = 0; j < 3; ++j) {
                                max_diff = max<int>(max_diff, abs(rgba_buf[4 * i + j] - rgba_buf_res[4 * i + j]));
                                //fprintf(stderr, "%d %d\n", (int) rgba_buf[4 * i + j], (int) rgba_buf_res[4 * i + j]);
                        }
                        //fprintf(stderr, "R in 10 bits = %d\n", (int) (r10k_buf[4 * i] << 2) + (r10k_buf[4 * i + 1] >> 6));
                }

                if (getenv("DEBUG_DUMP") != nullptr) {
                        FILE *out = fopen("out.rgba","w");
                        fwrite(rgba_buf_res.data(), width * height * 4, 1, out);
                        fclose(out);
                }

                CPPUNIT_ASSERT_MESSAGE("Maximal allowed difference 1, found "s + to_string(max_diff), max_diff <= 1);
        };

        for (auto f : { AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE, AV_PIX_FMT_YUV444P16LE }) {
                int i = 0;
                for_each(rgba_buf.begin(), rgba_buf.end(), [&](unsigned char & c) { c = (i++ / 4) % 0x100; });
                test_pattern(f);

                array<unsigned char, 4> pattern{ 0xFFU, 0, 0, 0xFFU };
                for_each(rgba_buf.begin(), rgba_buf.end(), [&](unsigned char & c) { c = pattern[i++ % 4]; });
                test_pattern(f);

                default_random_engine rand_gen;
                for_each(rgba_buf.begin(), rgba_buf.end(), [&](unsigned char & c) { c = rand_gen() % 0x100; });
                test_pattern(f);
        }
}

void
ff_codec_conversions_test::test_yuv444pXXle_from_to_r12l()
{
        using namespace std::string_literals;

        constexpr int width = 320;
        constexpr int height = 240;
        vector <unsigned char> rgb_buf(width * height * 3);

        /// @todo Use 12-bit natively
        auto test_pattern = [&](AVPixelFormat avfmt) {
                vector <unsigned char> r12l_buf(vc_get_datalen(width, height, R12L));
                decoder_t rgb_to_r12l = get_decoder_from_to(RGB, R12L, true);
                rgb_to_r12l(r12l_buf.data(), rgb_buf.data(), vc_get_datalen(width, height, R12L), 0, 8, 16);

                AVFrame frame;
                frame.format = avfmt;
                frame.width = width;
                frame.height = height;

                /* the image can be allocated by any means and av_image_alloc() is
                 * just the most convenient way if av_malloc() is to be used */
                assert(av_image_alloc(frame.data, frame.linesize,
                                        width, height, (AVPixelFormat) frame.format, 32) >= 0);

                auto from_conv = get_uv_to_av_conversion(R12L, frame.format);
                auto to_conv = get_av_to_uv_conversion(frame.format, R12L);
                assert(to_conv != nullptr && from_conv != nullptr);

                TIMER(t0);
                from_conv(&frame, r12l_buf.data(), width, height);
                TIMER(t1);
                to_conv(reinterpret_cast<char*>(r12l_buf.data()), &frame, width, height, vc_get_linesize(width, R12L), nullptr);
                TIMER(t2);

                if (getenv("PERF") != nullptr) {
                        cout << "test_yuv444p16le_from_to_r12l: duration - enc " << tv_diff(t1, t0) << ", dec " <<tv_diff(t2, t1) << "\n";
                }

                av_freep(frame.data);

                vector <unsigned char> rgb_buf_res(width * height * 3);
                decoder_t r12l_to_rgb = get_decoder_from_to(R12L, RGB, true);
                r12l_to_rgb(rgb_buf_res.data(), r12l_buf.data(), vc_get_datalen(width, height, RGB), 0, 8, 16);

                int max_diff = 0;
                for (size_t i = 0; i < width * height; ++i) {
                        for (int j = 0; j < 3; ++j) {
                                max_diff = max<int>(max_diff, abs(rgb_buf[3 * i + j] - rgb_buf_res[3 * i + j]));
                        }
                }

                if (getenv("DEBUG_DUMP") != nullptr) {
                        FILE *out = fopen("out.rgb","w");
                        fwrite(rgb_buf_res.data(), width * height * 3, 1, out);
                        fclose(out);
                }

                CPPUNIT_ASSERT_MESSAGE("Maximal allowed difference 1, found "s + to_string(max_diff), max_diff <= 1);
        };

        for (auto f : { AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE, AV_PIX_FMT_YUV444P16LE }) {
                int i = 0;
                array<unsigned char, 3> pattern{ 0xFFU, 0, 0 };
                for_each(rgb_buf.begin(), rgb_buf.end(), [&](unsigned char & c) { c = pattern[i++ % 3]; });
                test_pattern(f);

                for_each(rgb_buf.begin(), rgb_buf.end(), [&](unsigned char & c) { c = (i++ / 3) % 0x100; });
                test_pattern(f);

                default_random_engine rand_gen;
                for_each(rgb_buf.begin(), rgb_buf.end(), [&](unsigned char & c) { c = rand_gen() % 0x100; });
                test_pattern(f);
        }
}

static void yuv444p16le_rg48_encode_decode(int width, int height, char *in, char *out) {
        AVFrame frame{};
        frame.format = AV_PIX_FMT_YUV444P16LE;
        frame.width = width;
        frame.height = height;

        /* the image can be allocated by any means and av_image_alloc() is
         * just the most convenient way if av_malloc() is to be used */
        if (av_image_alloc(frame.data, frame.linesize,
                                width, height, (AVPixelFormat) frame.format, 32) < 0) {
                abort();
        }

        auto from_conv = get_uv_to_av_conversion(RG48, frame.format);
        auto to_conv = get_av_to_uv_conversion(frame.format, RG48);
        assert(to_conv != nullptr && from_conv != nullptr);

        TIMER(t0);
        from_conv(&frame, reinterpret_cast<unsigned char *>(in), width, height);
        TIMER(t1);
        to_conv(reinterpret_cast<char*>(out), &frame, width, height, vc_get_linesize(width, RG48), nullptr);
        TIMER(t2);

        if (getenv("PERF") != nullptr) {
                cout << "test_yuv444p16le_from_to_rg48: duration - enc " << tv_diff(t1, t0) << ", dec " <<tv_diff(t2, t1) << "\n";
        }

        av_freep(frame.data);
}

/**
 * Tests 12-bit values 0-4096 - max allowed range is MIN_12B-MAX_12B which should be
 * also the output - so comparing the output values against clamped input value. Also
 * using lower delta to accept because of the claming.
 */
void ff_codec_conversions_test::test_yuv444p16le_from_to_rg48_out_of_range()
{
        using namespace std::string_literals;

        constexpr int MAX_DIFF = 16;

        constexpr int width = 4096;
        constexpr int height = 5;
        vector <uint16_t> rg48_buf(width * height * 3);
        vector <uint16_t> rg48_buf_res(width * height * 3);

        for (int i = 0; i < 4096; i += 1) { // grayscale tones
                rg48_buf[3 * i] =
                        rg48_buf[3 * i + 1] =
                        rg48_buf[3 * i + 2] = 65535 * i / 4095;
        }
        for (int line = 0; line < 3; ++line) { // R, G, B scale
                for (int i = 0; i < 4096; i += 1) {
                        rg48_buf[(line + 1) * (width * 3) + 3 * i + line] = 65535 * i / 4095;
                }
        }
        for (int i = 0; i < 4096; i += 1) { // custom
                rg48_buf[4 * (width * 3) + 3 * i] = 65535;
                rg48_buf[4 * (width * 3) + 3 * i + 1] = 65535;
                rg48_buf[4 * (width * 3) + 3 * i + 2] = 0;
        }

        yuv444p16le_rg48_encode_decode(width, height, reinterpret_cast<char *>(rg48_buf.data()),
                        reinterpret_cast<char *>(rg48_buf_res.data()));

        int max_diff = 0;
        for (size_t i = 0; i < width * height; ++i) {
                for (int j = 0; j < 3; ++j) {
                        int diff = static_cast<int>(rg48_buf[3 * i + j]) - static_cast<int>(rg48_buf_res[3 * i + j]);
                        if (abs(diff) >= 1 && getenv("DEBUG") != nullptr) {
                                cout << "different value at pos: " << i << "," << j << " diff: " << diff << "\n";
                        }
                        max_diff = max<int>(max_diff, abs(diff));
                }
        }

        if (getenv("DEBUG_DUMP") != nullptr) {
                std::ofstream in("in.rg48", std::ifstream::out | std::ifstream::binary);
                in.write(reinterpret_cast<char *>(rg48_buf.data()), rg48_buf.size() * sizeof(decltype(rg48_buf)::value_type));
                std::ofstream out("out.rg48", std::ifstream::out | std::ifstream::binary);
                out.write(reinterpret_cast<char *>(rg48_buf_res.data()), rg48_buf_res.size() * sizeof(decltype(rg48_buf_res)::value_type));
        }

        CPPUNIT_ASSERT_MESSAGE("Maximal allowed difference "s + to_string (MAX_DIFF) + "/65535, found "s + to_string(max_diff), max_diff <= MAX_DIFF);
}

/**
 * Tests RG48<->YUV444P16LE conversions with 12-bit RGB input values
 * (full-range with the SDI small headroom)
 */
void ff_codec_conversions_test::test_yuv444p16le_from_to_rg48()
{
        using namespace std::string_literals;
        constexpr int MAX_DIFF = 16; /// @todo look at the conversions to yield better precision

        int width = 2048;
        int height = 1;
        vector <uint16_t> rg48_buf(width * height * 3, MIN_12B << 4); // SDI minimal value
        vector <uint16_t> rg48_buf_res(width * height * 3);

        for (int i = 16; i < 2040; i += 1) {
                rg48_buf[3 * i] =
                        rg48_buf[3 * i + 1] =
                        rg48_buf[3 * i + 2] = (i * 2) << 4;
        }
        for (int i = 2040; i < 2048; i += 1) {
                rg48_buf[3 * i] =
                        rg48_buf[3 * i + 1] =
                        rg48_buf[3 * i + 2] = MAX_12B << 4;
        }

        if (getenv("UG_TEST_RG48_INPUT") != nullptr) {
                char *cpy = strdup(getenv("UG_TEST_RG48_INPUT"));
                char *filename = strtok(cpy, ":");
                width = atoi(strtok(nullptr, ":"));
                height = atoi(strtok(nullptr, ":"));
                rg48_buf.resize(width * height * 3);
                rg48_buf_res.resize(width * height * 3);
                ifstream in(filename, std::ifstream::in | std::ifstream::binary);
                in.read(reinterpret_cast<char *>(rg48_buf.data()), rg48_buf.size() * sizeof(decltype(rg48_buf)::value_type));
        }

        yuv444p16le_rg48_encode_decode(width, height, reinterpret_cast<char *>(rg48_buf.data()),
                        reinterpret_cast<char *>(rg48_buf_res.data()));

        int max_diff = 0;
        for (int i = 0; i < width * height; ++i) {
                for (int j = 0; j < 3; ++j) {
                        int in = rg48_buf[3 * i + j];
                        int out = rg48_buf_res[3 * i + j];
                        int diff = in - out;
                        if (abs(diff) >= MAX_DIFF && getenv("DEBUG") != nullptr) {
                                cout << "different value at pos: " << i << "," << j << " diff: " << diff << " in: " << in << " out: " << out << "\n";
                        }
                        max_diff = max<int>(max_diff, abs(diff));
                        out >>= 4;
                        CPPUNIT_ASSERT_MESSAGE("Value "s + to_string(out) + " out of range "s + to_string(MIN_12B) + "-"s + to_string(MAX_12B), out >= MIN_12B && out <= MAX_12B);
                }
        }

        if (getenv("DEBUG_DUMP") != nullptr) {
                std::ofstream in("in.rg48", std::ifstream::out | std::ifstream::binary);
                in.write(reinterpret_cast<char *>(rg48_buf.data()), rg48_buf.size() * sizeof(decltype(rg48_buf)::value_type));
                std::ofstream out("out.rg48", std::ifstream::out | std::ifstream::binary);
                out.write(reinterpret_cast<char *>(rg48_buf_res.data()), rg48_buf_res.size() * sizeof(decltype(rg48_buf_res)::value_type));
        }

        CPPUNIT_ASSERT_MESSAGE("Maximal allowed difference "s + to_string (MAX_DIFF) + "/65535, found "s + to_string(max_diff), max_diff <= MAX_DIFF);
}

#endif // defined HAVE_CPPUNIT && HAVE_LAVC

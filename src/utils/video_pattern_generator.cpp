/**
 * @file   utils/video_pattern_generator.cpp
 * @author Colin Perkins <csp@csperkins.org
 * @author Alvaro Saurin <saurin@dcs.gla.ac.uk>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 * @author Martin Pulec <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2005-2006 University of Glasgow
 * Copyright (c) 2005-2021 CESNET z.s.p.o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file
 * @todo
 * Do the rendering in 16 bits
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // defined HAVE_CONFIG_H
#include "config_msvc.h"
#include "config_unix.h"
#include "config_win32.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "debug.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture/testcard_common.h"
#include "video_pattern_generator.hpp"

constexpr size_t headroom = 128; // headroom for cases when dst color_spec has wider block size
constexpr const char *MOD_NAME = "[vid. patt. generator] ";
constexpr int rg48_bpp = 6;

using namespace std::string_literals;
using std::array;
using std::copy;
using std::cout;
using std::default_random_engine;
using std::exception;
using std::for_each;
using std::make_unique;
using std::max;
using std::min;
using std::move;
using std::string;
using std::swap;
using std::unique_ptr;
using std::uniform_int_distribution;
using std::vector;

struct testcard_rect {
        int x, y, w, h;
};
struct testcard_pixmap {
        int w, h;
        void *data;
};

enum class generator_depth {
        bits8,
        bits16
};

static void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, uint32_t color);

class image_pattern {
        public:
                static unique_ptr<image_pattern> create(string const & config);
                auto init(int width, int height, enum generator_depth depth) {
                        auto delarr_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ delete [] ptr; });
                        size_t data_len = width * height * rg48_bpp + headroom;
                        auto out = unique_ptr<unsigned char[], void (*)(unsigned char*)>(new unsigned char[data_len], delarr_deleter);
                        auto actual_bit_depth = fill(width, height, out.get());
                        if (depth == generator_depth::bits8 && actual_bit_depth == generator_depth::bits16) {
                                convert_rg48_to_rgba(width, height, out.get());
                        }
                        if (depth == generator_depth::bits16 && actual_bit_depth == generator_depth::bits8) {
                                convert_rgba_to_rg48(width, height, out.get());
                        }
                        return out;
                }
                virtual ~image_pattern() = default;
                image_pattern() = default;
                image_pattern(const image_pattern &) = delete;
                image_pattern & operator=(const image_pattern &) = delete;
                image_pattern(image_pattern &&) = delete;
                image_pattern && operator=(image_pattern &&) = delete;
        private:
                /// @retval bit depth used by the generator (either 8 or 16)
                virtual enum generator_depth fill(int width, int height, unsigned char *data) = 0;

                /// @note in-place
                virtual void convert_rgba_to_rg48(int width, int height, unsigned char *data) {
                        for (int y = height - 1; y >= 0; --y) {
                                for (int x = width - 1; x >= 0; --x) {
                                        unsigned char *in_pix = data + 4 * (y * width + x);
                                        unsigned char *out_pix = data + 6 * (y * width + x);
                                        *out_pix++ = 0;
                                        *out_pix++ = *in_pix++;
                                        *out_pix++ = 0;
                                        *out_pix++ = *in_pix++;
                                        *out_pix++ = 0;
                                        *out_pix++ = *in_pix++;
                                }
                        }
                }
                /// @note in-place
                virtual void convert_rg48_to_rgba(int width, int height, unsigned char *data) {
                        for (int y = 0; y < height; ++y) {
                                for (int x = 0; x < width; ++x) {
                                        unsigned char *in_pix = data + 6 * (y * width + x);
                                        unsigned char *out_pix = data + 4 * (y * width + x);
                                        *out_pix++ = in_pix[1];
                                        *out_pix++ = in_pix[3];
                                        *out_pix++ = in_pix[5];
                                        *out_pix++ = 0xFFU;
                                }
                        }
                }
};

class image_pattern_bars : public image_pattern {
        enum generator_depth fill(int width, int height, unsigned char *data) override {
                int col_num = 0;
                int rect_size = COL_NUM;
                struct testcard_rect r{};
                struct testcard_pixmap pixmap{};
                pixmap.w = width;
                pixmap.h = height;
                pixmap.data = data;
                rect_size = (width + rect_size - 1) / rect_size;
                for (int j = 0; j < height; j += rect_size) {
                        uint32_t grey = 0xFF010101U;
                        if (j == rect_size * 2) {
                                r.w = width;
                                r.h = rect_size / 4;
                                r.x = 0;
                                r.y = j;
                                testcard_fillRect(&pixmap, &r, 0xFFFFFFFFU);
                                r.h = rect_size - (rect_size * 3 / 4);
                                r.y = j + rect_size * 3 / 4;
                                testcard_fillRect(&pixmap, &r, 0xFF000000U);
                        }
                        for (int i = 0; i < width; i += rect_size) {
                                r.x = i;
                                r.y = j;
                                r.w = rect_size;
                                r.h = min<int>(rect_size, height - r.y);
                                printf("Fill rect at %d,%d\n", r.x, r.y);
                                if (j != rect_size * 2) {
                                        testcard_fillRect(&pixmap, &r,
                                                        rect_colors[col_num]);
                                        col_num = (col_num + 1) % COL_NUM;
                                } else {
                                        r.h = rect_size / 2;
                                        r.y += rect_size / 4;
                                        testcard_fillRect(&pixmap, &r, grey);
                                        grey += 0x00010101U * (255 / COL_NUM);
                                }
                        }
                }
                return generator_depth::bits8;
        }
};

/**
 * @todo Proper SMPTE test pattern has different in bottom third.
 */
template<uint8_t f, int columns>
class image_pattern_ebu_smpte_bars : public image_pattern {
        static constexpr array bars{
                uint32_t{0xFFU << 24U | f  << 16U | f  << 8U | f  },
                        uint32_t{0xFFU << 24U | 0U << 16U | f  << 8U | f  },
                        uint32_t{0xFFU << 24U | f  << 16U | f  << 8U | 0U },
                        uint32_t{0xFFU << 24U | 0U << 16U | f  << 8U | 0U },
                        uint32_t{0xFFU << 24U | f  << 16U | 0U << 8U | f  },
                        uint32_t{0xFFU << 24U | 0U << 16U | 0U << 8U | f  },
                        uint32_t{0xFFU << 24U | f  << 16U | f  << 8U | 0U },
                        uint32_t{0xFFU << 24U | f  << 16U | 0U << 8U | 0U },
        };
        enum generator_depth fill(int width, int height, unsigned char *data) override {
                int col_num = 0;
                const int rect_size = (width + columns - 1) / columns;
                struct testcard_rect r{};
                struct testcard_pixmap pixmap{};
                pixmap.w = width;
                pixmap.h = height;
                pixmap.data = data;
                for (int j = 0; j < height; j += rect_size) {
                        for (int i = 0; i < width; i += rect_size) {
                                r.x = i;
                                r.y = j;
                                r.w = rect_size;
                                r.h = min<int>(rect_size, height - r.y);
                                printf("Fill rect at %d,%d\n", r.x, r.y);
                                testcard_fillRect(&pixmap, &r,
                                                bars.at(col_num));
                                col_num = (col_num + 1) % columns;
                        }
                }
                return generator_depth::bits8;
        }
};

class image_pattern_blank : public image_pattern {
        public:
                explicit image_pattern_blank(uint32_t c = 0xFF000000U) : color(c) {}

        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        for (int i = 0; i < width * height; ++i) {
                                (reinterpret_cast<uint32_t *>(data))[i] = color;
                        }
                        return generator_depth::bits8;
                }
                uint32_t color;
};

class image_pattern_gradient : public image_pattern {
        public:
                explicit image_pattern_gradient(uint32_t c) : color(c) {}
                static constexpr uint32_t red = 0xFFU;
        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        auto *ptr = reinterpret_cast<uint32_t *>(data);
                        for (int j = 0; j < height; j += 1) {
                                uint8_t r = sin(static_cast<double>(j) / height * M_PI) * (color & 0xFFU);
                                uint8_t g = sin(static_cast<double>(j) / height * M_PI) * ((color >> 8) & 0xFFU);
                                uint8_t b = sin(static_cast<double>(j) / height * M_PI) * ((color >> 16) & 0xFFU);
                                uint32_t val = (0xFFU << 24U) | (b << 16) | (g << 8) | r;
                                for (int i = 0; i < width; i += 1) {
                                        *ptr++ = val;
                                }
                        }
                        return generator_depth::bits8;
                }
                uint32_t color;
};

class image_pattern_gradient2 : public image_pattern {
        public:
                explicit image_pattern_gradient2(long maxval = 0XFFFFU) : val_max(maxval) {}
        private:
                const unsigned int val_max;
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        width = max(width, 2); // avoid division by zero
                        auto *ptr = reinterpret_cast<uint16_t *>(data);
                        for (int j = 0; j < height; j += 1) {
                                for (int i = 0; i < width; i += 1) {
                                        unsigned int gray = i * val_max / (width - 1);
                                        *ptr++ = gray;
                                        *ptr++ = gray;
                                        *ptr++ = gray;
                                }
                        }
                        return generator_depth::bits16;
                }
};

class image_pattern_noise : public image_pattern {
        default_random_engine rand_gen;
        enum generator_depth fill(int width, int height, unsigned char *data) override {
                uniform_int_distribution<> dist(0, 0xFFFF);
                for_each(reinterpret_cast<uint16_t *>(data), reinterpret_cast<uint16_t *>(data) + 3 * width * height, [&](uint16_t & c) { c = dist(rand_gen); });
                return generator_depth::bits16;
        }
};

class image_pattern_raw : public image_pattern {
        public:
                explicit image_pattern_raw(string config) {
                        if (config.empty()) {
                                throw ug_runtime_error("Empty raw pattern is not allowed!");
                        }
                        while (!config.empty()) {
                                unsigned char byte = 0;
                                if (sscanf(config.c_str(), "%2hhx", &byte) == 1) {
                                        m_pattern.push_back(byte);
                                }
                                config = config.substr(min<size_t>(config.size(), 2));
                        }
                }

                void raw_fill(unsigned char *data, size_t data_len)  {
                        while (data_len >= m_pattern.size()) {
                                copy(m_pattern.begin(), m_pattern.end(), data);
                                data += m_pattern.size();
                                data_len -= m_pattern.size();
                        }
                }
        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        memset(data, 0, width * height * 3); // placeholder only
                        return generator_depth::bits8;
                }
                vector<unsigned char> m_pattern;
};


unique_ptr<image_pattern> image_pattern::create(string const &config) {
        if (config == "help") {
                cout << "Pattern to use, one of: " << BOLD("bars, blank, ebu_bars, gradient[=0x<AABBGGRR>], gradient2, noise, raw=0xXX[YYZZ..], smpte_bars, 0x<AABBGGRR>\n");
                cout << "\t\t- patterns 'gradient2' and 'noise' generate full bit-depth patterns with " << BOLD("RG48") << ", " << BOLD("R12L") << " and " << BOLD("R10k\n");
                cout << "\t\t- pattern 'raw' generates repeating sequence of given bytes without any color conversion\n";
                cout << "\t\t- pattern 'smpte' uses the top bars from top 2 thirds only (doesn't render bottom third differently)\n";
                return {};
        }
        string pattern = config;
        string params;
        if (string::size_type delim = config.find('='); delim != string::npos) {
                pattern = config.substr(0, delim);
                params = config.substr(delim + 1);
        }
        if (config == "bars") {
                return make_unique<image_pattern_bars>();
        }
        if (config == "blank") {
                return make_unique<image_pattern_blank>();
        }
        if (config == "ebu_bars") {
                return make_unique<image_pattern_ebu_smpte_bars<0xFFU, 8>>();
        }
        if (pattern == "gradient2") {
                if (!params.empty()) {
                        if (params == "help"s) {
                                cout << "Testcard gradient2 usage:\n\t-t testcard:gradient2[=maxval] - maxval is 16-bit resolution\n";
                                return {};
                        }
                        return make_unique<image_pattern_gradient2>(stol(params, nullptr, 0));
                }
                return make_unique<image_pattern_gradient2>();
        }
        if (pattern == "gradient") {
                uint32_t color = image_pattern_gradient::red;
                if (!params.empty()) {
                        color = stol(params, nullptr, 0);
                }
                return make_unique<image_pattern_gradient>(color);
        }
        if (config == "noise") {
                return make_unique<image_pattern_noise>();
        }
        if (config.substr(0, "raw=0x"s.length()) == "raw=0x") {
                return make_unique<image_pattern_raw>(config.substr("raw=0x"s.length()));
        }
        if (config == "smpte_bars") {
                return make_unique<image_pattern_ebu_smpte_bars<0xBFU, 7>>();
        }
        if (config.substr(0, "0x"s.length()) == "0x") {
                uint32_t blank_color = 0U;
                if (sscanf(config.substr("0x"s.length()).c_str(), "%x", &blank_color) == 1) {
                        return make_unique<image_pattern_blank>(blank_color);
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong color!\n";
                }
        }
        throw ug_runtime_error("Unknown pattern: "s +  config + "!"s);
}

static void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, uint32_t color)
{
        auto *data = static_cast<uint32_t *>(s->data);

        for (int cur_x = r->x; cur_x < r->x + r->w; ++cur_x) {
                for (int cur_y = r->y; cur_y < r->y + r->h; ++cur_y) {
                        if (cur_x < s->w) {
                                *(data + s->w * cur_y + cur_x) = color;
                        }
                }
        }
}

unique_ptr<unsigned char [],void (*)(unsigned char*)>
video_pattern_generate(std::string const & config, int width, int height, codec_t color_spec)
{
        assert(width > 0 && height > 0);
        auto free_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ free(ptr); });
        auto delarr_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ delete [] ptr; });

        unique_ptr<image_pattern> generator;
        try {
                generator = image_pattern::create(config);
        } catch (exception const &e) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
                return {nullptr, free_deleter};
        }
        if (!generator) {
                return {nullptr, free_deleter};
        }

        auto data = generator->init(width, height, generator_depth::bits8);
        codec_t codec_src = RGBA;
        if (color_spec == RG48 || color_spec == R12L || color_spec == R10k) {
                data = generator->init(width, height, generator_depth::bits16);
                codec_src = RG48;
        }

        auto src = move(data);
        data = decltype(data)(new unsigned char[vc_get_datalen(width, height, color_spec)], delarr_deleter);
        testcard_convert_buffer(codec_src, color_spec, data.get(), src.get(), width, height);

        if (auto *raw_generator = dynamic_cast<image_pattern_raw *>(generator.get())) {
                raw_generator->raw_fill(data.get(), vc_get_datalen(width, height, color_spec));
        }

        return data;
}
/* vim: set expandtab sw=8: */

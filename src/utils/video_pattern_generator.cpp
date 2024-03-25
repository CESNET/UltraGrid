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
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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
#include "config_unix.h"
#include "config_win32.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "color.h"
#include "debug.h"
#include "pixfmt_conv.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/string_view_utils.hpp"
#include "utils/text.h"
#include "video.h"
#include "video_capture/testcard_common.h"
#include "video_pattern_generator.h"

constexpr size_t headroom = 128; // headroom for cases when dst color_spec has wider block size
#define MOD_NAME "[vid. patt. generator] "
constexpr int rg48_bpp = 6;

using namespace std::string_literals;
using std::array;
using std::copy;
using std::cout;
using std::default_random_engine;
using std::exception;
using std::for_each;
using std::make_unique;
using std::min;
using std::stoi;
using std::stoll;
using std::string;
using std::string_view;
using std::unique_ptr;
using std::uniform_int_distribution;
using std::vector;

enum class generator_depth {
        bits8,
        bits16
};

class image_pattern {
        public:
                static unique_ptr<image_pattern> create(string const &pattern, string const &params);
                auto init(int width, int height, enum generator_depth depth) noexcept {
                        size_t data_len = width * height * rg48_bpp + headroom;
                        vector<unsigned char> out(data_len);
                        auto actual_bit_depth = fill(width, height, out.data());
                        if (depth == generator_depth::bits8 && actual_bit_depth == generator_depth::bits16) {
                                convert_rg48_to_rgba(width, height, out.data());
                        }
                        if (depth == generator_depth::bits16 && actual_bit_depth == generator_depth::bits8) {
                                convert_rgba_to_rg48(width, height, out.data());
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
                                        unsigned char r = *in_pix++;
                                        unsigned char g = *in_pix++;
                                        unsigned char b = *in_pix++;
                                        *out_pix++ = 0;
                                        *out_pix++ = r;
                                        *out_pix++ = 0;
                                        *out_pix++ = g;
                                        *out_pix++ = 0;
                                        *out_pix++ = b;
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
        public:
        explicit image_pattern_bars(string const &init) {
                if (init == "help"s) {
                        col() << "Testcard bar usage:\n\t" << SBOLD(SRED("-t testcard:pattern=bars") << "[=text]") << " - optionally annotate with text" << "\n";
                        throw 1;
                }
                annotate = init;
        }
        private:
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
                                LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "Fill rect at " << r.x << "," << r.y << "\n";
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
                if (!annotate.empty()) {
                        draw_line((char *) data, width * 4, annotate.c_str(), 0xFFFFFFFF, true);
                }
                return generator_depth::bits8;
        }
        string annotate;
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
                        uint32_t{0xFFU << 24U | f  << 16U | 0U << 8U | 0U },
                        uint32_t{0xFFU << 24U | 0U << 16U | 0U << 8U | 0U },
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
                                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Fill rect at %d,%d\n", r.x, r.y);
                                testcard_fillRect(&pixmap, &r,
                                                bars.at(col_num));
                                col_num = (col_num + 1) % columns;
                        }
                }
                return generator_depth::bits8;
        }
        friend class image_pattern_smpte_bars;
};

class image_pattern_smpte_bars : public image_pattern_ebu_smpte_bars<0xBFU, 7> {
        static constexpr array bottom_bars{
                uint32_t{0xFFU << 24U | 105 << 16U | 63 << 8U | 0U  },
                uint32_t{0xFFFFFFFFU },
                uint32_t{0xFFU << 24U | 119U << 16U | 0U << 8U | 0U },
                uint32_t{0xFF000000U },
                uint32_t{0xFF000000U },
                uint32_t{0xFF000000U },
        };
        enum generator_depth fill(int width, int height, unsigned char *data) override {
                auto ret = image_pattern_ebu_smpte_bars<0xBFU, 7>::fill(width, height, data); // upper 2 3rds
                assert(ret == generator_depth::bits8);
                int columns = 7;
                struct testcard_pixmap pixmap{ .w = width, .h = height, .data = data };
                const int mid_strip_height = height / 3 - width / 6;
                struct testcard_rect r{ .x = 0, .y = height / 3 * 2, .w = (width + columns - 1) / columns, .h = mid_strip_height};
                for (int i = 0; i < columns; i += 1) {
                        r.x = i * r.w;
                        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Fill rect at %d,%d\n", r.x, r.y);
                        if (i % 2 == 1) testcard_fillRect(&pixmap, &r, 0);
                        else testcard_fillRect(&pixmap, &r, image_pattern_ebu_smpte_bars<0xBFU, 7>::bars.at(columns - 1 - i));
                }
                columns = 6;
                r.w = (width + columns - 1) / columns;
                r.h = width / 6;
                r.y += mid_strip_height;
                for (int i = 0; i < columns; i += 1) {
                        r.x = i * r.w;
                        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Fill rect at %d,%d\n", r.x, r.y);
                        testcard_fillRect(&pixmap, &r,
                                        bottom_bars.at(i));
                }
                // pluge - skipping a "superblack" and black bar
                r.x = 5 * (width / 7);
                r.w = (width / 7) / 3;
                r.x += 2 * r.w;
                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Fill rect at %d,%d\n", r.x, r.y);
                testcard_fillRect(&pixmap, &r,
                                0xFFU << 24 | 0x0A0A0A);
                return generator_depth::bits8;
        }
};

class image_pattern_blank : public image_pattern {
        public:
                explicit image_pattern_blank(string const &init) {
                        if (!init.empty()) {
                                color = stoll(init, nullptr, 0);
                        }
                }

        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        for (int i = 0; i < width * height; ++i) {
                                (reinterpret_cast<uint32_t *>(data))[i] = color;
                        }
                        return generator_depth::bits8;
                }
                uint32_t color = 0xFF000000U;
};

class image_pattern_gradient : public image_pattern {
        public:
                explicit image_pattern_gradient(const string &config) {
                        if (!config.empty()) {
                                color = stol(config, nullptr, 0);
                        }
                }
                static constexpr uint32_t red = 0xFFU;
        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        auto *ptr = reinterpret_cast<uint16_t *>(data);
                        for (int j = 0; j < height; j += 1) {
                                uint16_t r = sin(static_cast<double>(j) / height * M_PI) * (color & 0xFFU) / 0xFFU * 0xFF'FFU;
                                uint16_t g = sin(static_cast<double>(j) / height * M_PI) * ((color >> 8) & 0xFFU) / 0xFFU * 0xFF'FFU;
                                uint16_t b = sin(static_cast<double>(j) / height * M_PI) * ((color >> 16) & 0xFFU) / 0xFFU * 0xFF'FFU;
                                for (int i = 0; i < width; i += 1) {
                                        *ptr++ = r;
                                        *ptr++ = g;
                                        *ptr++ = b;
                                }
                        }
                        return generator_depth::bits16;
                }
                uint32_t color = image_pattern_gradient::red;;
};

class image_pattern_gradient2 : public image_pattern {
        public:
                explicit image_pattern_gradient2(string const &config) {
                        if (config.empty()) {
                                return;
                        }
                        if (config == "help"s) {
                                cout << "Testcard gradient2 usage:\n\t-t testcard:gradient2[=maxval] - maxval is 16-bit number\n";
                                throw 1;
                        }
                        val_max = stol(config, nullptr, 0);
                }
        private:
                unsigned int val_max = 0XFFFFU;
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        assert(width > 1); // avoid division by zero
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

class image_pattern_uv_plane : public image_pattern {
        public:
                explicit image_pattern_uv_plane(string const &y_lvl) {
                        if (!y_lvl.empty()) {
                                y_level = LIMIT_LO(16) + stof(y_lvl) * (LIMIT_HI_Y(16) - LIMIT_LO(16));
                        }
                }
        private:
                int y_level = (LIMIT_HI_Y(16) + LIMIT_LO(16)) / 2;
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        assert(width > 1 && height > 1); // avoid division by zero
                        auto *ptr = reinterpret_cast<uint16_t *>(data);
                        auto *conv = get_decoder_from_to(Y416, RG48);
                        int scale_cbcr = LIMIT_HI_CBCR(16) - LIMIT_LO(16);
                        for (int j = 0; j < height; j += 1) {
                                for (int i = 0; i < width; i += 1) {
                                        uint16_t uyva[4];
                                        uyva[0] = LIMIT_LO(16) + i * scale_cbcr / (width - 1);
                                        uyva[1] = y_level;
                                        uyva[2] = LIMIT_LO(16) + j * scale_cbcr / (height - 1);
                                        uyva[3] = 0xFF'FF;
                                        conv((unsigned char *) ptr, (unsigned char *) uyva, 6, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                                        ptr += 3;
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
                        if (config.substr(0, "0x"s.length()) == "0x") { // strip optional "0x"
                                config = config.substr(2);
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

class image_pattern_text : public image_pattern {
        public:
                explicit image_pattern_text(const string & config) {
                        if (config == "help"s) {
                                col() << "Testcard text usage:\n\t" << SBOLD(SRED("-t testcard:pattern=text") << "[=pattern][,bg=0x<AABBGGRR<][,fg=0x<AABBGGRR>]") << "\n";
                                throw 1;
                        }
                        if (!config.empty()) {
                                string_view sv = config;
                                bool text_set = false;
                                while (!sv.empty()) {
                                        auto tok = tokenize(sv, ',');
                                        if (tok.substr(0,3) == "bg=") {
                                                bg = stol((string) tok.substr(3), nullptr, 0);
                                        } else if (tok.substr(0,3) == "fg=") {
                                                fg = stol((string) tok.substr(3), nullptr, 0);
                                        } else if (!text_set) {
                                                text = tok;
                                                text_set = true;
                                        } else {
                                                throw ug_runtime_error("Testcard text - wrong option: " + string(tok));
                                        }
                                }
                        }
                }

        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        std::fill((uint32_t *)(void *) data, (uint32_t *)(void *) (data + width * height * 4), bg);
                        string line;
                        for (int i = 0; i < width; i += 8 * (text.size() + 1)) {
                                line += " " + text;
                        }
                        for (int i = 0; i < height / 16; ++i) {
                                draw_line((char *) data + (i * 16 * width * 4), width * 4, line.c_str(), fg, false);
                        }
                        return generator_depth::bits8;
                }
                string text = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
                uint32_t bg = 0xFFCC00CCU;
                uint32_t fg = 0xFFFFFFFFU;
};

class image_pattern_diagonal : public image_pattern{
        public:
                explicit image_pattern_diagonal(const string & config) {
                        if (config == "help"s) {
                                col() << "Testcard diagonal usage:\n\t" << SBOLD(SRED("-t testcard:pattern=diagonal") << "[,bg=0x<AABBGGRR<][,fg=0x<AABBGGRR>][,stride=<stride>][,line_width=<width>]") << "\n";
                                throw 1;
                        }
                        if (!config.empty()) {
                                string_view sv = config;
                                while (!sv.empty()) {
                                        auto tok = tokenize(sv, ',');
                                        auto key = tokenize(tok, '=');
                                        auto val = tokenize(tok, '=');
                                        if (key == "bg") {
                                                bg = stol((string) val, nullptr, 0);
                                        } else if (key == "fg") {
                                                fg = stol((string) val, nullptr, 0);
                                        } else if (key == "stride") {
                                                stride = stol((string) val, nullptr, 0);
                                        } else if (key == "line_width") {
                                                line_width = stol((string) val, nullptr, 0);
                                        } else {
                                                throw ug_runtime_error("Testcard diagonal - wrong option: " + string(tok));
                                        }
                                }
                        }
                }

        private:
                enum generator_depth fill(int width, int height, unsigned char *data) override {
                        std::fill((uint32_t *)(void *) data, (uint32_t *)(void *) (data + width * height * 4), bg);
                        for(int y = 0; y < height; y++){
                                for(int x = y % stride; x < width - line_width; x += stride){
                                        for(int i = 0; i < line_width; i++){
                                                *((uint32_t *) (void *) (data + (y*width + x + i) * 4)) = fg;
                                        }
                                }
                                for(int x = stride - (y % stride); x < width - line_width; x += stride){
                                        for(int i = 0; i < line_width; i++){
                                                *((uint32_t *) (void *) (data + (y*width + x + i) * 4)) = fg;
                                        }
                                }
                        }

                        return generator_depth::bits8;
                }
                uint32_t bg = 0xFF000000U;
                uint32_t fg = 0xFFFFFFFFU;
                int stride = 80;
                int line_width = 4;
};

unique_ptr<image_pattern> image_pattern::create(string const &pattern, string const &params) {
        if (pattern == "bars") {
                return make_unique<image_pattern_bars>(params);
        }
        if (pattern == "blank") {
                return make_unique<image_pattern_blank>(params);
        }
        if (pattern == "ebu_bars") {
                return make_unique<image_pattern_ebu_smpte_bars<0xFFU, 8>>();
        }
        if (pattern == "gradient") {
                return make_unique<image_pattern_gradient>(params);
        }
        if (pattern == "gradient2") {
                return make_unique<image_pattern_gradient2>(params);
        }
        if (pattern == "noise") {
                return make_unique<image_pattern_noise>();
        }
        if (pattern == "raw") {
                return make_unique<image_pattern_raw>(params);
        }
        if (pattern == "smpte_bars") {
                return make_unique<image_pattern_smpte_bars>();
        }
        if (pattern == "text") {
                return make_unique<image_pattern_text>(params);
        }
        if (pattern == "uv_plane") {
                return make_unique<image_pattern_uv_plane>(params);
        }
        if (pattern == "diagonal") {
                return make_unique<image_pattern_diagonal>(params);
        }
        throw ug_runtime_error("Unknown pattern: "s +  pattern + "!"s);
}

struct video_pattern_generator {
        virtual char *get_next() = 0;
        virtual ~video_pattern_generator() {}
};

struct still_image_video_pattern_generator : public video_pattern_generator {
        still_image_video_pattern_generator(string const &pattern, string const &params, int w, int h, codec_t c, int o)
                : width(w), height(h), color_spec(c), offset(o)
        {
                unique_ptr<image_pattern> generator;
                try {
                        generator = image_pattern::create(pattern, params);
                } catch (exception const &e) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
                        throw 1;
                }
                if (!generator) {
                        throw 2;
                }

                data = generator->init(width, height, generator_depth::bits8);
                codec_t codec_src = RGBA;
                if (get_decoder_from_to(RG48, color_spec) != NULL) {
                        data = generator->init(width, height, generator_depth::bits16);
                        codec_src = RG48;
                }

                vector<unsigned char> src;
                data.swap(src);
                long data_len = vc_get_datalen(width, height, color_spec);
                data.resize(data_len * 2);
                testcard_convert_buffer(codec_src, color_spec, data.data(), src.data(), width, height);

                if (auto *raw_generator = dynamic_cast<image_pattern_raw *>(generator.get())) {
                        raw_generator->raw_fill(data.data(), data_len);
                }

                memcpy(data.data() + data_len, data.data(), data_len);
        }
        int width;
        int height;
        codec_t color_spec;
        vector<unsigned char> data;
        int offset;
        long cur_pos = 0;
        long data_len = vc_get_datalen(width, height, color_spec);
        long linesize = vc_get_linesize(width, color_spec);

        char *get_next() override {
                auto ret = (char *) data.data() + cur_pos;
                cur_pos += offset;
                if (cur_pos >= data_len) {
                        cur_pos = 0;
                }
                return ret;
        }
};

struct gray_video_pattern_generator : public video_pattern_generator {
        gray_video_pattern_generator(int w, int h, codec_t c, string const& opts)
                : width(w), height(h), color_spec(c)
        {
                if (!opts.empty()) {
                        if (opts == "help") {
                                col() << "Usage:\n\t" SBOLD("gray[:step]") << " - interframe color increment (default " << DEFAULT_STEP << ")\n";
                                throw 1;
                        }
                        step = stoi(opts);
                }
                int col = 0;
                while (col < 0xFF) {
                        int pixels = get_pf_block_pixels(color_spec);
                        unsigned char rgba[MAX_PFB_SIZE * 4];
                        for (int i = 0; i < pixels * 4; ++i) {
                                rgba[i] = (i + 1) % 4 != 0 ? col : 0xFFU; // handle alpha
                        }
                        int dst_bs = get_pf_block_bytes(color_spec);
                        unsigned char dst[MAX_PFB_SIZE];
                        testcard_convert_buffer(RGBA, color_spec, dst, rgba, pixels, 1);

                        auto next_frame = vector<unsigned char>(data_len);
                        for (int y = 0; y < height; ++y) {
                                for (int x = 0; x < width / pixels; x += 1) {
                                        memcpy(next_frame.data() + y * vc_get_linesize(width, color_spec) + x * dst_bs, dst, dst_bs);
                                }
                        }
                        data.push_back(std::move(next_frame));
                        col += step;
                }
        }
        char *get_next() override {
                auto *out = (char *) data[cur_idx++].data();
                if (cur_idx * step >= 0xFF) {
                        cur_idx = 0;
                }

                return out;
        }
private:
        constexpr static int DEFAULT_STEP = 1;
        int step = DEFAULT_STEP;
        int width;
        int height;
        codec_t color_spec;
        int cur_idx = 0;
        long data_len = vc_get_datalen(width, height, color_spec);
        vector<vector<unsigned char>> data;
};

struct interlaced_video_pattern_generator : public video_pattern_generator {
        interlaced_video_pattern_generator(int w, int h, codec_t color_spec)
                : width(w), height(h), linesize(vc_get_linesize(width, color_spec))
        {
                size_t rgb_linesize = vc_get_linesize(width, RGB);
                vector<char> rgb(3 * h * rgb_linesize + 4 * (width / step) * rgb_linesize);
                memset(rgb.data(), 255, h * rgb_linesize);
                char *ptr = rgb.data() + h * rgb_linesize;
                auto fill = [&](int col1, int col2) {
                        for (int i = 0; i < width; i += step) {
                                size_t fill_len = rgb_linesize - i * 3;
                                memset(ptr, col1, fill_len);
                                memset(ptr + fill_len, col2, rgb_linesize - fill_len);
                                ptr += rgb_linesize;
                                fill_len = MAX(0, (int) rgb_linesize - (i + 2 * step) * 3);
                                memset(ptr, col1, fill_len);
                                memset(ptr + fill_len, col2, rgb_linesize - fill_len);
                                ptr += rgb_linesize;
                        }
                };
                fill(255, 0);
                memset(ptr, 0, h * rgb_linesize);
                ptr += h * rgb_linesize;
                fill(0, 255);
                memset(ptr, 255, h * rgb_linesize);
                vector<char> rgba(rgb.size() / 3 * 4);
                for (unsigned i = 0; i < rgb.size(); i += 3) {
                        rgba[i / 3 * 4] = rgb[i];
                        rgba[i / 3 * 4 + 1] = rgb[i + 1];
                        rgba[i / 3 * 4 + 2] = rgb[i + 2];
                        rgba[i / 3 * 4 + 3] = 0xff;
                }
                data.resize(3 * h * linesize + 4 * linesize * (w / step));
                testcard_convert_buffer(RGBA, color_spec, (unsigned char *) data.data(), (unsigned char *) rgba.data(), width, 3 * height + 4 * (width / step));
        }
        char *get_next() override {
                auto *out = (char *) data.data() + cur_idx * linesize;
                cur_idx += 8;
                if (cur_idx >= 2 * height + 4 * width / step) {
                        cur_idx = 0;
                }

                return out;
        }
private:
        constexpr static int step = 3;
        int width;
        int height;
        size_t linesize;
        int cur_idx = 0;
        vector<char> data;
};

video_pattern_generator_t
video_pattern_generator_create(const char *config, int width, int height, codec_t color_spec, int offset)
{
        if (string(config) == "help") {
                col() << "Pattern to use, one of: " << SBOLD("bars, blank[=0x<AABBGGRR>], ebu_bars, gradient[=0x<AABBGGRR>], gradient2*, gray, interlaced, noise, raw=0xXX[YYZZ..], smpte_bars, uv_plane[=<y_lvl>], diagonal*\n");
                col() << "\t\t- patterns " SBOLD("'gradient'") ", " SBOLD("'gradient2'") ", " SBOLD("'noise'") " and " SBOLD("'uv_plane'") " generate higher bit-depth patterns with";
                for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = static_cast<codec_t>(static_cast<int>(c) + 1)) {
                        if (get_decoder_from_to(RG48, c) != NULL && get_bits_per_component(c) > 8) {
                                col() << " " << SBOLD(get_codec_name(c));
                        }
                }
                col() << "\n";
                col() << "\t\t- pattern " << SBOLD("'raw'") " generates repeating sequence of given bytes without any color conversion\n";
                col() << "\t\t- patterns marked with " << SBOLD("'*'") " provide help as its option\n";
                return nullptr;
        }
        assert(width > 0 && height > 0);
        try {
                string pattern = config;
                string params;
                if (string::size_type delim = pattern.find('='); delim != string::npos) {
                        params = pattern.substr(delim + 1);
                        pattern = pattern.substr(0, delim);
                }
                if (pattern == "gray" || pattern == "grey") {
                        return new gray_video_pattern_generator{width, height, color_spec, params};
                }
                if (pattern == "interlaced") {
                        return new interlaced_video_pattern_generator{width, height, color_spec};
                }
                return new still_image_video_pattern_generator{pattern, params, width, height, color_spec, offset};
        } catch (exception const &e) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
                return nullptr;
        } catch (...) {
                return nullptr;
        }
}

char *video_pattern_generator_next_frame(video_pattern_generator_t s)
{
        return s->get_next();
}

void video_pattern_generator_fill_data(video_pattern_generator_t s, const char *data)
{
        auto &state = dynamic_cast<still_image_video_pattern_generator &>(*s);
        memcpy(state.data.data(), data, state.data_len);
        memcpy(state.data.data() + state.data_len, data, state.data_len);
}

void video_pattern_generator_destroy(video_pattern_generator_t s)
{
        delete s;
}

/* vim: set expandtab sw=8: */

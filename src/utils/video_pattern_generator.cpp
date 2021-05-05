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
#include <iostream>
#include <random>
#include <utility>

#include "debug.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture/testcard_common.h"
#include "video_pattern_generator.hpp"

constexpr size_t headroom = 128; // headroom for cases when dst color_spec has wider block size
constexpr int rg48_bpp = 6;

using namespace std::string_literals;
using std::cout;
using std::default_random_engine;
using std::for_each;
using std::make_unique;
using std::min;
using std::move;
using std::string;
using std::swap;
using std::unique_ptr;

struct testcard_rect {
        int x, y, w, h;
};
struct testcard_pixmap {
        int w, h;
        void *data;
};

static void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, uint32_t color);

class image_pattern {
        public:
                static unique_ptr<image_pattern> create(string const & config) noexcept;
                auto init(int width, int height, int out_bit_depth) {
                        assert(out_bit_depth == 8 || out_bit_depth == 16);
                        auto delarr_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ delete [] ptr; });
                        size_t data_len = width * height * rg48_bpp + headroom;
                        auto out = unique_ptr<unsigned char[], void (*)(unsigned char*)>(new unsigned char[data_len], delarr_deleter);
                        int actual_bit_depth = fill(width, height, out.get());
                        assert(actual_bit_depth == 8 || actual_bit_depth == 16);
                        if (out_bit_depth == 8 && actual_bit_depth == 16) {
                                convert_rg48_to_rgba(width, height, out.get());
                        }
                        if (out_bit_depth == 16 && actual_bit_depth == 8) {
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
                virtual int fill(int width, int height, unsigned char *data) = 0;

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
        int fill(int width, int height, unsigned char *data) override {
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
                return 8;
        }
};

class image_pattern_blank : public image_pattern {
        public:
                explicit image_pattern_blank(uint32_t c = 0xFF000000U) : color(c) {}

        private:
                int fill(int width, int height, unsigned char *data) override {
                        for (int i = 0; i < width * height; ++i) {
                                (reinterpret_cast<uint32_t *>(data))[i] = color;
                        }
                        return 8;
                }
                uint32_t color;
};

class image_pattern_gradient : public image_pattern {
        public:
                explicit image_pattern_gradient(uint32_t c) : color(c) {}
                static constexpr uint32_t red = 0xFFU;
        private:
                int fill(int width, int height, unsigned char *data) override {
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
                        return 8;
                }
                uint32_t color;
};

class image_pattern_gradient2 : public image_pattern {
        public:
                explicit image_pattern_gradient2(long maxval = 0XFFFFU) : val_max(maxval) {}
        private:
                const unsigned int val_max;
                int fill(int width, int height, unsigned char *data) override {
                        width = min(width, 2); // avoid division by zero
                        auto *ptr = reinterpret_cast<uint16_t *>(data);
                        for (int j = 0; j < height; j += 1) {
                                for (int i = 0; i < width; i += 1) {
                                        unsigned int gray = i * val_max / (width - 1);
                                        *ptr++ = gray;
                                        *ptr++ = gray;
                                        *ptr++ = gray;
                                }
                        }
                        return 16;
                }
};

class image_pattern_noise : public image_pattern {
        default_random_engine rand_gen;
        int fill(int width, int height, unsigned char *data) override {
                for_each(reinterpret_cast<uint16_t *>(data), reinterpret_cast<uint16_t *>(data) + 3 * width * height, [&](uint16_t & c) { c = rand_gen() % 0xFFFFU; });
                return 16;
        }
};

unique_ptr<image_pattern> image_pattern::create(string const &config) noexcept {
        if (config == "help") {
                cout << "Pattern to use, one of: " << BOLD("bars, blank, gradient[=0x<AABBGGRR>], gradient2, noise, 0x<AABBGGRR>\n");
                cout << "\t\t- patterns 'gradient2' and 'noise' generate full bit-depth patterns for RG48 and R12L\n";
                return {};
        }
        if (config == "bars") {
                return make_unique<image_pattern_bars>();
        }
        if (config == "blank") {
                return make_unique<image_pattern_blank>();
        }
        if (config.substr(0, "gradient"s.length()) == "gradient") {
                uint32_t color = image_pattern_gradient::red;
                if (config.substr(0, "gradient="s.length()) == "gradient=") {
                        auto val = config.substr("gradient="s.length());
                        color = stol(val, nullptr, 0);
                }
                return make_unique<image_pattern_gradient>(color);
        }
        if (config.substr(0, "gradient2"s.length()) == "gradient2") {
                if (config.substr(0, "gradient2="s.length()) == "gradient2=") {
                        auto val = config.substr("gradient2="s.length());
                        if (val == "help"s) {
                                cout << "Testcard gradient2 usage:\n\t-t testcard:gradient2[=maxval] - maxval is 16-bit resolution\n";
                                return {};
                        }
                        return make_unique<image_pattern_gradient2>(stol(val, nullptr, 0));
                }
                return make_unique<image_pattern_gradient2>();
        }
        if (config == "noise") {
                return make_unique<image_pattern_noise>();
        }
        if (config.substr(0, "0x"s.length()) == "0x") {
                uint32_t blank_color = 0U;
                if (sscanf(config.substr("0x"s.length()).c_str(), "%x", &blank_color) == 1) {
                        return make_unique<image_pattern_blank>(blank_color);
                } else {
                        LOG(LOG_LEVEL_ERROR) << "[testcard] Wrong color!\n";
                }
        }
        return {};
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

        auto generator = image_pattern::create(config);
        if (!generator) {
                return {nullptr, free_deleter};
        }

        auto data = generator->init(width, height, 8);

        codec_t codec_intermediate = color_spec;

        /// first step - conversion from RGBA
        if (color_spec == RG48 || color_spec == R12L) {
                data = generator->init(width, height, 16);
                if (color_spec == R12L) {
                        codec_intermediate = RG48;
                }
        } else if (color_spec != RGBA) {
                // these codecs do not have direct conversion from RGBA - use @ref second_conversion_step
                if (color_spec == I420 || color_spec == v210 || color_spec == YUYV || color_spec == Y216) {
                        codec_intermediate = UYVY;
                }
                if (color_spec == R12L) {
                        codec_intermediate = RGB;
                }

                auto decoder = get_decoder_from_to(RGBA, codec_intermediate, true);
                assert(decoder != nullptr);
                auto src = move(data);
                data = decltype(data)(new unsigned char [height * vc_get_linesize(width, codec_intermediate) + headroom], delarr_deleter);
                size_t src_linesize = vc_get_linesize(width, RGBA);
                size_t dst_linesize = vc_get_linesize(width, codec_intermediate);
                auto *in = src.get();
                auto *out = data.get();
                for (int i = 0; i < height; ++i) {
                        decoder(out, in, dst_linesize, 0, 0, 0);
                        in += src_linesize;
                        out += dst_linesize;
                }
        }

        /// @anchor second_conversion_step for some codecs
        if (color_spec == I420) {
                auto src = move(data);
                data = decltype(data)(reinterpret_cast<unsigned char *>((toI420(reinterpret_cast<char *>(src.get()), width, height))), free_deleter);
        } else if (color_spec == YUYV) {
                for (int i = 0; i < width * height * 2; i += 2) {
                        swap(data[i], data[i + 1]);
                }
        } else if (codec_intermediate != color_spec) {
                auto src = move(data);
                data = decltype(data)(new unsigned char[vc_get_datalen(width, height, color_spec)], delarr_deleter);
                auto decoder = get_decoder_from_to(codec_intermediate, color_spec, true);
                assert(decoder != nullptr);

                int src_linesize = vc_get_linesize(width, codec_intermediate);
                int dst_linesize = vc_get_linesize(width, color_spec);
                for (int i = 0; i < height; ++i) {
                        decoder(data.get() + i * dst_linesize,
                                        src.get() + i * src_linesize, dst_linesize, 0, 8, 16);
                }
        }
        return data;
}
/* vim: set expandtab sw=8: */

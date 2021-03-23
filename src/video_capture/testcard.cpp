/**
 * @file   video_capture/testcard.cpp
 * @author Colin Perkins <csp@csperkins.org
 * @author Alvaro Saurin <saurin@dcs.gla.ac.uk>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 */
/*
 * Copyright (c) 2005-2006 University of Glasgow
 * Copyright (c) 2005-2020 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/testcard_common.h"
#include "song1.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/vf_split.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <random>
#ifdef HAVE_LIBSDL_MIXER
#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>
#else
#include <SDL/SDL.h>
#include <SDL/SDL_mixer.h>
#endif // defined HAVE_SDL2
#endif /* HAVE_LIBSDL_MIXER */
#include <vector>
#include "audio/audio.h"

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_BPS 2
#define BUFFER_SEC 1
constexpr int AUDIO_BUFFER_SIZE(int ch_count) { return AUDIO_SAMPLE_RATE * AUDIO_BPS * ch_count * BUFFER_SEC; }
#define MOD_NAME "[testcard] "
constexpr video_desc default_format = { 1920, 1080, UYVY, 25.0, INTERLACED_MERGED, 1 };
constexpr size_t headroom = 128; // headroom for cases when dst color_spec has wider block size

using rang::fg;
using rang::style;
using namespace std;

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
                static unique_ptr<image_pattern> create(const char *pattern) noexcept;
                auto init(int width, int height, int out_bit_depth) {
                        assert(out_bit_depth == 8 || out_bit_depth == 16);
                        auto delarr_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ delete [] ptr; });
                        size_t data_len = width * height * 6 + headroom;
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

unique_ptr<image_pattern> image_pattern::create(const char *pattern) noexcept {
        if (strcmp(pattern, "bars") == 0) {
                return make_unique<image_pattern_bars>();
        } else if (strcmp(pattern, "blank") == 0) {
                return make_unique<image_pattern_blank>();
        } else if (strstr(pattern, "gradient2") != nullptr) {
                if (strstr(pattern, "gradient2=") != nullptr) {
                        auto val = string(pattern).substr("gradient2="s.length());
                        if (val == "help"s) {
                                cout << "Testcard gradient2 usage:\n\t-t testcard:gradient2[=maxval] - maxval is 16-bit resolution\n";
                                return {};
                        }
                        return make_unique<image_pattern_gradient2>(stol(val, nullptr, 0));
                }
                return make_unique<image_pattern_gradient2>();
        } else if (strstr(pattern, "gradient") != nullptr) {
                uint32_t color = image_pattern_gradient::red;
                if (strstr(pattern, "gradient=") != nullptr) {
                        auto val = string(pattern).substr("gradient="s.length());
                        color = stol(val, nullptr, 0);
                }
                return make_unique<image_pattern_gradient>(color);
        } else if (strcmp(pattern, "noise") == 0) {
                return make_unique<image_pattern_noise>();
        } else if (strstr(pattern, "0x") == pattern) {
                uint32_t blank_color = 0U;
                if (sscanf(pattern + 2, "%x", &blank_color) == 1) {
                        return make_unique<image_pattern_blank>(blank_color);
                } else {
                        LOG(LOG_LEVEL_ERROR) << "[testcard] Wrong color!\n";
                }
        }
        return {};
}

struct testcard_state {
        std::chrono::steady_clock::time_point last_frame_time;
        int pan;
        char *data {nullptr};
        std::chrono::steady_clock::time_point t0;
        struct video_frame *frame{nullptr};
        int frame_linesize;
        struct video_frame *tiled;

        struct audio_frame audio;
        char **tiles_data;
        int tiles_cnt_horizontal;
        int tiles_cnt_vertical;

        vector <char> audio_data;
        struct ring_buffer *midi_buf{};
        enum class grab_audio_t {
                NONE,
                ANY,
                MIDI,
                SINE,
        } grab_audio = grab_audio_t::NONE;

        unsigned int still_image;
        unique_ptr<image_pattern> pattern {new image_pattern_bars};
};

static void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, uint32_t color)
{
        auto *data = static_cast<uint32_t *>(s->data);

        for (int cur_x = r->x; cur_x < r->x + r->w; ++cur_x)
                for(int cur_y = r->y; cur_y < r->y + r->h; ++cur_y)
                        if(cur_x < s->w)
                                *(data + s->w * cur_y + cur_x) = color;
}

#if defined HAVE_LIBSDL_MIXER && ! defined HAVE_MACOSX
static void midi_audio_callback(int chan, void *stream, int len, void *udata)
{
        UNUSED(chan);
        struct testcard_state *s = (struct testcard_state *) udata;

        ring_buffer_write(s->midi_buf, static_cast<const char *>(stream), len);
}
#endif

static auto configure_sdl_mixer_audio(struct testcard_state *s) {
#if defined HAVE_LIBSDL_MIXER && ! defined HAVE_MACOSX
        char filename[1024] = "";
        int fd;
        Mix_Music *music;
        ssize_t bytes_written = 0l;

        SDL_Init(SDL_INIT_AUDIO);

        if( Mix_OpenAudio( AUDIO_SAMPLE_RATE, AUDIO_S16LSB,
                                s->audio.ch_count, 4096 ) == -1 ) {
                fprintf(stderr,"[testcard] error initalizing sound\n");
                return false;
        }
        strncpy(filename, "/tmp/uv.midiXXXXXX", sizeof filename - 1);
        fd = mkstemp(filename);
        if (fd < 0) {
                perror("mkstemp");
                return false;
        }

        do {
                ssize_t ret;
                ret = write(fd, song1 + bytes_written,
                                sizeof(song1) - bytes_written);
                if(ret < 0) return false;
                bytes_written += ret;
        } while (bytes_written < (ssize_t) sizeof(song1));
        close(fd);
        music = Mix_LoadMUS(filename);

        // register grab as a postmix processor
        if (!Mix_RegisterEffect(MIX_CHANNEL_POST, midi_audio_callback, nullptr, s)) {
                printf("[testcard] Mix_RegisterEffect: %s\n", Mix_GetError());
                return false;
        }

        if(Mix_PlayMusic(music,-1)==-1){
                fprintf(stderr, "[testcard] error playing midi\n");
                return false;
        }
        Mix_Volume(-1, 0);

        s->midi_buf = ring_buffer_init(AUDIO_BUFFER_SIZE(s->audio.ch_count) /* 1 sec */);

        cout << MOD_NAME << "Initialized MIDI\n";

        return true;
#else
        UNUSED(s);
        return false;
#endif
}

static void configure_fallback_audio(struct testcard_state *s) {
        static_assert(AUDIO_BPS == sizeof(int16_t), "Only 2-byte audio is supported for testcard audio at the moment");
        const int frequency = 1000;
        const double scale = 0.1;

        for (int i = 0; i < AUDIO_BUFFER_SIZE(s->audio.ch_count) / AUDIO_BPS; i += 1) {
                *(reinterpret_cast<int16_t*>(&s->audio_data[i * AUDIO_BPS])) = round(sin((static_cast<double>(i) / (static_cast<double>(AUDIO_SAMPLE_RATE) / frequency)) * M_PI * 2. ) * ((1LL << (AUDIO_BPS * 8)) / 2 - 1) * scale);
        }
}

static auto configure_audio(struct testcard_state *s)
{
        s->audio.bps = AUDIO_BPS;
        s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        s->audio.max_size = AUDIO_BUFFER_SIZE(s->audio.ch_count);
        s->audio_data.resize(s->audio.max_size);
        s->audio.data = s->audio_data.data();

        if (s->grab_audio != testcard_state::grab_audio_t::SINE) {
                if (configure_sdl_mixer_audio(s)) {
                        s->grab_audio = testcard_state::grab_audio_t::MIDI;
                        return true;
                }
                if (s->grab_audio == testcard_state::grab_audio_t::MIDI) {
                        return false;
                }
        }

        LOG(LOG_LEVEL_WARNING) << MOD_NAME "SDL-mixer missing, running on Mac or other problem - using fallback audio.\n";
        configure_fallback_audio(s);
        s->grab_audio = testcard_state::grab_audio_t::SINE;

        return true;
}


static int configure_tiling(struct testcard_state *s, const char *fmt)
{
        char *tmp, *token, *saveptr = NULL;
        int tile_cnt;
        int x;

        int grid_w, grid_h;

        if(fmt[1] != '=') return 1;

        tmp = strdup(&fmt[2]);
        token = strtok_r(tmp, "x", &saveptr);
        grid_w = atoi(token);
        token = strtok_r(NULL, "x", &saveptr);
        grid_h = atoi(token);
        free(tmp);

        s->tiled = vf_alloc(grid_w * grid_h);
        s->tiles_cnt_horizontal = grid_w;
        s->tiles_cnt_vertical = grid_h;
        s->tiled->color_spec = s->frame->color_spec;
        s->tiled->fps = s->frame->fps;
        s->tiled->interlacing = s->frame->interlacing;

        tile_cnt = grid_w *
                grid_h;
        assert(tile_cnt >= 1);

        s->tiles_data = (char **) malloc(tile_cnt *
                        sizeof(char *));
        /* split only horizontally!!!!!! */
        vf_split(s->tiled, s->frame, grid_w,
                        1, 1 /*prealloc*/);
        /* for each row, make the tile data correct.
         * .data pointers of same row point to same block,
         * but different row */
        for(x = 0; x < grid_w; ++x) {
                int y;

                s->tiles_data[x] = s->tiled->tiles[x].data;

                s->tiled->tiles[x].width = s->frame->tiles[0].width/ grid_w;
                s->tiled->tiles[x].height = s->frame->tiles[0].height / grid_h;
                s->tiled->tiles[x].data_len = s->frame->tiles[0].data_len / (grid_w * grid_h);

                s->tiled->tiles[x].data =
                        s->tiles_data[x] = (char *) realloc(s->tiled->tiles[x].data,
                                        s->tiled->tiles[x].data_len * grid_h * 2);


                memcpy(s->tiled->tiles[x].data + s->tiled->tiles[x].data_len  * grid_h,
                                s->tiled->tiles[x].data, s->tiled->tiles[x].data_len * grid_h);
                /* recopy tiles vertically */
                for(y = 1; y < grid_h; ++y) {
                        memcpy(&s->tiled->tiles[y * grid_w + x],
                                        &s->tiled->tiles[x], sizeof(struct tile));
                        /* make the pointers correct */
                        s->tiles_data[y * grid_w + x] =
                                s->tiles_data[x] +
                                y * s->tiled->tiles[x].height *
                                vc_get_linesize(s->tiled->tiles[x].width, s->tiled->color_spec);

                        s->tiled->tiles[y * grid_w + x].data =
                                s->tiles_data[x] +
                                y * s->tiled->tiles[x].height *
                                vc_get_linesize(s->tiled->tiles[x].width, s->tiled->color_spec);
                }
        }

        return 0;
}

static const codec_t codecs_8b[] = {I420, RGBA, RGB, UYVY, YUYV, VIDEO_CODEC_NONE};
static const codec_t codecs_10b[] = {R10k, v210, VIDEO_CODEC_NONE};
static const codec_t codecs_12b[] = {Y216, RG48, R12L, VIDEO_CODEC_NONE};

static auto parse_format(char **fmt, char **save_ptr) {
        struct video_desc desc{};
        desc.tile_count = 1;
        desc.interlacing = PROGRESSIVE;
        char *tmp;

        tmp = strtok_r(*fmt, ":", save_ptr);
        *fmt = nullptr;
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", *fmt);
                return video_desc{};
        }
        desc.width = max<long long>(strtol(tmp, nullptr, 0), 0);
        tmp = strtok_r(nullptr, ":", save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", *fmt);
                return video_desc{};
        }
        desc.height = max<long long>(strtol(tmp, nullptr, 0), 0);
        tmp = strtok_r(nullptr, ":", save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", *fmt);
                return video_desc{};
        }

        if (desc.width * desc.height == 0) {
                fprintf(stderr, "Wrong dimensions for testcard.\n");
                return video_desc{};
        }

        char *endptr;
        desc.fps = strtod(tmp, &endptr);
        if (endptr[0] != '\0') { // optional interlacing suffix
                desc.interlacing = get_interlacing_from_suffix(endptr);
                if (desc.interlacing != PROGRESSIVE &&
                                desc.interlacing != SEGMENTED_FRAME &&
                                desc.interlacing != INTERLACED_MERGED) { // tff or bff
                        log_msg(LOG_LEVEL_ERROR, "Unsuppored interlacing format!\n");
                        return video_desc{};
                }
                if (desc.interlacing == INTERLACED_MERGED) {
                        desc.fps /= 2;
                }
        }

        tmp = strtok_r(nullptr, ":", save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", *fmt);
                return video_desc{};
        }

        desc.color_spec = get_codec_from_name(tmp);
        if (desc.color_spec == VIDEO_CODEC_NONE) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                return video_desc{};
        }
        {
                const codec_t *sets[] = {codecs_8b, codecs_10b, codecs_12b};
                bool supported = false;
                for (int i = 0; i < (int) (sizeof sets / sizeof sets[0]); ++i) {
                        const codec_t *it = sets[i];
                        while (*it != VIDEO_CODEC_NONE) {
                                if (desc.color_spec == *it++) {
                                        supported = true;
                                }
                        }
                }
                if (!supported) {
                        log_msg(LOG_LEVEL_ERROR, "Unsupported codec '%s'\n", tmp);
                        return video_desc{};
                }
        }

        return desc;
}

static int vidcap_testcard_init(struct vidcap_params *params, void **state)
{
        struct testcard_state *s = nullptr;
        char *filename;
        const char *strip_fmt = NULL;
        FILE *in = NULL;
        char *save_ptr = NULL;
        char *tmp;

        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                printf("testcard options:\n");
                cout << BOLD(RED("\t-t testcard") << ":<width>:<height>:<fps>:<codec>[:filename=<filename>][:p][:s=<X>x<Y>][:i|:sf][:still][:pattern=<pattern>][:apattern=sine|midi]\n");
                cout << "where\n";
                cout << BOLD("\t<filename>") << " - use file named filename instead of default bars\n";
                cout << BOLD("\tp") << " - pan with frame\n";
                cout << BOLD("\ts") << " - split the frames into XxY separate tiles\n";
                cout << BOLD("\ti|sf") << " - send as interlaced or segmented frame (if none of those is set, progressive is assumed)\n";
                cout << BOLD("\tstill") << " - send still image\n";
                cout << BOLD("\tpattern") << " - pattern to use, one of: " << BOLD("bars, blank, gradient[=0x<AABBGGRR>], gradient2, noise, 0x<AABBGGRR>\n");
                cout << "\t\t- patterns 'gradient2' and 'noise' generate full bit-depth patterns for RG48 and R12L\n";
                cout << BOLD("\tapattern") << " - audio pattern to use - \"sine\" or an included \"midi\"\n";
                show_codec_help("testcard", codecs_8b, codecs_10b, codecs_12b);
                return VIDCAP_INIT_NOERR;
        }

        s = new testcard_state();
        if (!s)
                return VIDCAP_INIT_FAIL;

        char *fmt = strdup(vidcap_params_get_fmt(params));
        char *ptr = fmt;

        struct video_desc desc = [&]{ return strlen(ptr) == 0 || !isdigit(ptr[0]) ? default_format : parse_format(&ptr, &save_ptr);}();
        if (!desc) {
                goto error;
        }

        s->still_image = FALSE;
        s->frame = vf_alloc_desc(desc);
        vf_get_tile(s->frame, 0)->data = static_cast<char *>(malloc(s->frame->tiles[0].data_len * 2));
        s->frame_linesize = vc_get_linesize(desc.width, desc.color_spec);

        filename = NULL;

        tmp = strtok_r(ptr, ":", &save_ptr);
        while (tmp) {
                if (strcmp(tmp, "p") == 0) {
                        s->pan = 48;
                } else if (strncmp(tmp, "filename=", strlen("filename=")) == 0) {
                        filename = tmp + strlen("filename=");
                        in = fopen(filename, "r");
                        if (!in) {
                                perror("fopen");
                                goto error;
                        }
                        fseek(in, 0L, SEEK_END);
                        long filesize = ftell(in);
                        assert(filesize >= 0);
                        fseek(in, 0L, SEEK_SET);

                        if (s->frame->tiles[0].data_len != filesize) {
                                int level = s->frame->tiles[0].data_len < filesize ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR;
                                LOG(level) << MOD_NAME  << "Wrong file size for selected "
                                        "resolution and codec. File size " << filesize << ", "
                                        "computed size " << s->frame->tiles[0].data_len << "\n";
                                filesize = s->frame->tiles[0].data_len;
                                if (level == LOG_LEVEL_ERROR) {
                                        goto error;
                                }
                        }

                        if (in == nullptr || fread(vf_get_tile(s->frame, 0)->data, filesize, 1, in) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "Cannot read file %s\n", filename);
                                goto error;
                        }

                        fclose(in);
                        in = NULL;
                } else if (strncmp(tmp, "s=", 2) == 0) {
                        strip_fmt = tmp;
                } else if (strcmp(tmp, "i") == 0) {
                        s->frame->interlacing = INTERLACED_MERGED;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'i' option. Use format testcard:1920:1080:50i:UYVY instead!\n");
                } else if (strcmp(tmp, "sf") == 0) {
                        s->frame->interlacing = SEGMENTED_FRAME;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'sf' option. Use format testcard:1920:1080:25sf:UYVY instead!\n");
                } else if (strcmp(tmp, "still") == 0) {
                        s->still_image = TRUE;
                } else if (strncmp(tmp, "pattern=", strlen("pattern=")) == 0) {
                        const char *pattern = tmp + strlen("pattern=");
                        s->pattern = image_pattern::create(pattern);
                        if (!s->pattern) {
                                fprintf(stderr, "[testcard] Unknown pattern!\n");;
                                goto error;
                        }
                } else if (strstr(tmp, "apattern=") == tmp) {
                        s->grab_audio = strcasecmp(tmp + strlen("apattern="), "sine") == 0 ? testcard_state::grab_audio_t::SINE : testcard_state::grab_audio_t::MIDI;
                } else {
                        fprintf(stderr, "[testcard] Unknown option: %s\n", tmp);
                        goto error;
                }
                tmp = strtok_r(NULL, ":", &save_ptr);
        }

        if (!filename) {
                auto data = s->pattern->init(s->frame->tiles[0].width, s->frame->tiles[0].height, 8);
                auto free_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ free(ptr); });
                auto delarr_deleter = static_cast<void (*)(unsigned char*)>([](unsigned char *ptr){ delete [] ptr; });

                codec_t codec_intermediate = s->frame->color_spec;

                /// first step - conversion from RGBA
                if (s->frame->color_spec == RG48 || s->frame->color_spec == R12L) {
                        data = s->pattern->init(s->frame->tiles[0].width, s->frame->tiles[0].height, 16);
                        if (s->frame->color_spec == R12L) {
                                codec_intermediate = RG48;
                        }
                } else if (s->frame->color_spec != RGBA) {
                        // these codecs do not have direct conversion from RGBA - use @ref second_conversion_step
                        if (s->frame->color_spec == I420 || s->frame->color_spec == v210 || s->frame->color_spec == YUYV || s->frame->color_spec == Y216) {
                                codec_intermediate = UYVY;
                        }
                        if (s->frame->color_spec == R12L) {
                                codec_intermediate = RGB;
                        }

                        auto decoder = get_decoder_from_to(RGBA, codec_intermediate, true);
                        assert(decoder != nullptr);
                        auto src = move(data);
                        data = decltype(data)(new unsigned char [s->frame->tiles[0].height * vc_get_linesize(s->frame->tiles[0].width, codec_intermediate) + headroom], delarr_deleter);
                        size_t src_linesize = vc_get_linesize(s->frame->tiles[0].width, RGBA);
                        size_t dst_linesize = vc_get_linesize(s->frame->tiles[0].width, codec_intermediate);
                        auto *in = src.get();
                        auto *out = data.get();
                        for (unsigned int i = 0; i < s->frame->tiles[0].height; ++i) {
                                decoder(out, in, dst_linesize, 0, 0, 0);
                                in += src_linesize;
                                out += dst_linesize;
                        }
                }

                /// @anchor second_conversion_step for some codecs
                if (s->frame->color_spec == I420) {
                        auto src = move(data);
                        data = decltype(data)(reinterpret_cast<unsigned char *>((toI420(reinterpret_cast<char *>(src.get()), s->frame->tiles[0].width, s->frame->tiles[0].height))), free_deleter);
                } else if (s->frame->color_spec == YUYV) {
                        for (unsigned int i = 0; i < s->frame->tiles[0].data_len; i += 2) {
                                swap(data[i], data[i + 1]);
                        }
                } else if (codec_intermediate != s->frame->color_spec) {
                        auto src = move(data);
                        data = decltype(data)(new unsigned char[s->frame->tiles[0].data_len], delarr_deleter);
                        auto decoder = get_decoder_from_to(codec_intermediate, s->frame->color_spec, true);
                        assert(decoder != nullptr);

                        int src_linesize = vc_get_linesize(s->frame->tiles[0].width, codec_intermediate);
                        int dst_linesize = vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec);
                        for (int i = 0; i < (int) vf_get_tile(s->frame, 0)->height; ++i) {
                                decoder(data.get() + i * dst_linesize,
                                                src.get() + i * src_linesize, dst_linesize, 0, 8, 16);
                        }
                }

                memcpy(vf_get_tile(s->frame, 0)->data, data.get(), s->frame->tiles[0].data_len);
        }

        // duplicate the image to allow scrolling
        memcpy(vf_get_tile(s->frame, 0)->data + vf_get_tile(s->frame, 0)->data_len, vf_get_tile(s->frame, 0)->data, vf_get_tile(s->frame, 0)->data_len);

        if (!s->still_image && codec_is_planar(s->frame->color_spec)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Planar pixel format '%s', using still picture.\n", get_codec_name(s->frame->color_spec));
                s->still_image = true;
        }

        s->last_frame_time = std::chrono::steady_clock::now();

        printf("Testcard capture set to %dx%d, bpp %f\n", vf_get_tile(s->frame, 0)->width,
                        vf_get_tile(s->frame, 0)->height, get_bpp(s->frame->color_spec));

        if(strip_fmt != NULL) {
                if(configure_tiling(s, strip_fmt) != 0) {
                        goto error;
                }
        }

        if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                if (s->grab_audio == testcard_state::grab_audio_t::NONE) {
                        s->grab_audio = testcard_state::grab_audio_t::ANY;
                }
                if (!configure_audio(s)) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot initialize audio!\n";
                        goto error;
                }
        } else {
                s->grab_audio = testcard_state::grab_audio_t::NONE;
        }

        free(fmt);

        s->data = s->frame->tiles[0].data;

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(fmt);
        free(s->data);
        vf_free(s->frame);
        if (in)
                fclose(in);
        delete s;
        return VIDCAP_INIT_FAIL;
}

static void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = (struct testcard_state *) state;
        free(s->data);
        if (s->tiled) {
                int i;
                for (i = 0; i < s->tiles_cnt_horizontal; ++i) {
                        free(s->tiles_data[i]);
                }
                vf_free(s->tiled);
        }
        vf_free(s->frame);
        ring_buffer_destroy(s->midi_buf);
        delete s;
}

static struct video_frame *vidcap_testcard_grab(void *arg, struct audio_frame **audio)
{
        struct testcard_state *state;
        state = (struct testcard_state *)arg;

        std::chrono::steady_clock::time_point curr_time =
                std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::duration<double>>(curr_time - state->last_frame_time).count() <
                        1.0 / state->frame->fps) {
                return NULL;
        }

        state->last_frame_time = curr_time;

        if (state->grab_audio != testcard_state::grab_audio_t::NONE) {
                if (state->grab_audio == testcard_state::grab_audio_t::MIDI) {
                        state->audio.data_len = ring_buffer_read(state->midi_buf, state->audio.data, state->audio.max_size);
                } else if (state->grab_audio == testcard_state::grab_audio_t::SINE) {
                        state->audio.data_len = state->audio.ch_count * state->audio.bps * AUDIO_SAMPLE_RATE / state->frame->fps;
                        state->audio.data += state->audio.data_len;
                        if (state->audio.data + state->audio.data_len > state->audio_data.data() + AUDIO_BUFFER_SIZE(state->audio.ch_count)) {
                                state->audio.data = state->audio_data.data();
                        }
                } else {
                        abort();
                }
                if(state->audio.data_len > 0)
                        *audio = &state->audio;
                else
                        *audio = NULL;
        } else {
                *audio = NULL;
        }

        if(!state->still_image) {
                vf_get_tile(state->frame, 0)->data += state->frame_linesize + state->pan;
        }
        if (vf_get_tile(state->frame, 0)->data > state->data + state->frame->tiles[0].data_len) {
                vf_get_tile(state->frame, 0)->data = state->data;
        }

        if (state->tiled) {
                /* update tile data instead */
                int i;
                int count = state->tiled->tile_count;

                for (i = 0; i < count; ++i) {
                        /* shift - for semantics of vars refer to configure_tiling*/
                        state->tiled->tiles[i].data += vc_get_linesize(
                                        state->tiled->tiles[i].width, state->tiled->color_spec);
                        /* if out of data, move to beginning
                         * keep in mind that we have two "pictures" for
                         * every tile stored sequentially */
                        if(state->tiled->tiles[i].data >= state->tiles_data[i] +
                                        state->tiled->tiles[i].data_len * state->tiles_cnt_vertical) {
                                state->tiled->tiles[i].data = state->tiles_data[i];
                        }
                }

                return state->tiled;
        }
        return state->frame;
}

static struct vidcap_type *vidcap_testcard_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type *vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name = "testcard";
        vt->description = "Video testcard";

        if (!verbose) {
                return vt;
        }

        vt->card_count = 1;
        vt->cards = (struct device_info *) calloc(vt->card_count, sizeof(struct device_info));
        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Testing signal");

        struct {
                int width;
                int height;
        } sizes[] = {
                {1280, 720},
                {1920, 1080},
                {3840, 2160},
        };
        int framerates[] = {24, 30, 60};
        const char * const pix_fmts[] = {"UYVY", "RGB"};

        snprintf(vt->cards[0].modes[0].name,
                        sizeof vt->cards[0].modes[0].name, "Default");
        snprintf(vt->cards[0].modes[0].id,
                        sizeof vt->cards[0].modes[0].id,
                        "{\"width\":\"\", "
                        "\"height\":\"\", "
                        "\"format\":\"\", "
                        "\"fps\":\"\"}");

        int i = 1;
        for(const auto &pix_fmt : pix_fmts){
                for(const auto &size : sizes){
                        for(const auto &fps : framerates){
                                snprintf(vt->cards[0].modes[i].name,
                                                sizeof vt->cards[0].name,
                                                "%dx%d@%d %s",
                                                size.width, size.height,
                                                fps, pix_fmt);
                                snprintf(vt->cards[0].modes[i].id,
                                                sizeof vt->cards[0].modes[0].id,
                                                "{\"width\":\"%d\", "
                                                "\"height\":\"%d\", "
                                                "\"format\":\"%s\", "
                                                "\"fps\":\"%d\"}",
                                                size.width, size.height,
                                                pix_fmt, fps);
                                i++;
                        }
                }
        }
        return vt;
}

static const struct video_capture_info vidcap_testcard_info = {
        vidcap_testcard_probe,
        vidcap_testcard_init,
        vidcap_testcard_done,
        vidcap_testcard_grab,
        true
};

REGISTER_MODULE(testcard, &vidcap_testcard_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

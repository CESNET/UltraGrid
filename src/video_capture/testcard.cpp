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
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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
 * * fix broken tiling (perhaps wrapper over pattern generator)
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
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/ring_buffer.h"
#include "utils/string.h"
#include "utils/vf_split.h"
#include "utils/pam.h"
#include "utils/y4m.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include "audio/types.h"
#include "utils/video_pattern_generator.hpp"
#include "video_capture/testcard_common.h"

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_BPS 2
#define BUFFER_SEC 1
constexpr int AUDIO_BUFFER_SIZE(int ch_count) { return AUDIO_SAMPLE_RATE * AUDIO_BPS * ch_count * BUFFER_SEC; }
#define MOD_NAME "[testcard] "
constexpr video_desc default_format = { 1920, 1080, UYVY, 25.0, INTERLACED_MERGED, 1 };

using namespace std;

struct testcard_state {
        std::chrono::steady_clock::time_point last_frame_time;
        int pan = 0;
        video_pattern_generator_t generator;
        std::chrono::steady_clock::time_point t0;
        struct video_frame *frame{nullptr};
        struct video_frame *tiled;

        struct audio_frame audio;
        char **tiles_data;
        int tiles_cnt_horizontal;
        int tiles_cnt_vertical;

        vector <char> audio_data;
        struct ring_buffer *midi_buf{};
        bool grab_audio = false;
        bool still_image = false;
        string pattern{"bars"};
};

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

        configure_fallback_audio(s);
        s->grab_audio = true;

        return true;
}

#if 0
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
#endif

static bool parse_fps(const char *fps, struct video_desc *desc) {
        char *endptr = nullptr;
        desc->fps = strtod(fps, &endptr);
        desc->interlacing = PROGRESSIVE;
        if (strlen(endptr) != 0) { // optional interlacing suffix
                desc->interlacing = get_interlacing_from_suffix(endptr);
                if (desc->interlacing != PROGRESSIVE &&
                                desc->interlacing != SEGMENTED_FRAME &&
                                desc->interlacing != INTERLACED_MERGED) { // tff or bff
                        log_msg(LOG_LEVEL_ERROR, "Unsuppored interlacing format: %s!\n", endptr);
                        return false;
                }
                if (desc->interlacing == INTERLACED_MERGED) {
                        desc->fps /= 2;
                }
        }
        return true;
}

static auto parse_format(char **fmt, char **save_ptr) {
        struct video_desc desc{};
        desc.tile_count = 1;
        char *tmp = strtok_r(*fmt, ":", save_ptr);
        if (!tmp) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Missing width!\n";
                return video_desc{};
        }
        desc.width = max<long long>(strtol(tmp, nullptr, 0), 0);

        if ((tmp = strtok_r(nullptr, ":", save_ptr)) == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Missing height!\n";
                return video_desc{};
        }
        desc.height = max<long long>(strtol(tmp, nullptr, 0), 0);

        if (desc.width * desc.height == 0) {
                fprintf(stderr, "Wrong dimensions for testcard.\n");
                return video_desc{};
        }

        if ((tmp = strtok_r(nullptr, ":", save_ptr)) == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Missing FPS!\n";
                return video_desc{};
        }
        if (!parse_fps(tmp, &desc)) {
                return video_desc{};
        }

        if ((tmp = strtok_r(nullptr, ":", save_ptr)) == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Missing pixel format!\n";
                return video_desc{};
        }
        desc.color_spec = get_codec_from_name(tmp);
        if (desc.color_spec == VIDEO_CODEC_NONE) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unknown codec '" << tmp << "'\n";
                return video_desc{};
        }
        if (!testcard_has_conversion(desc.color_spec)) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unsupported codec '" << tmp << "'\n";
                return video_desc{};
        }

        *fmt = nullptr;
        return desc;
}

static bool testcard_load_from_file_pam(const char *filename, struct video_desc *desc, vector<char>& in_file_contents) {
        struct pam_metadata info;
        unsigned char *data = nullptr;
        if (pam_read(filename, &info, &data, malloc) == 0) {
                return false;
        }
        assert(info.depth == 3);
        desc->width = info.width;
        desc->height = info.height;
        desc->color_spec = info.maxval == 255 ? RGB : RG48;
        in_file_contents.resize(vc_get_datalen(desc->width, desc->height, desc->color_spec));
        if (desc->color_spec == RGB) {
                memcpy(in_file_contents.data(), data, info.width * info.height * 3);
        } else {
                uint16_t *in = (uint16_t *)(void *) data;
                uint16_t *out = (uint16_t *)(void *) in_file_contents.data();
                for (size_t i = 0; i < (size_t) info.width * info.height * 3; ++i) {
                        *out++ = ntohs(*in++) * ((1<<16U) / (info.maxval + 1));
                }
        }
        free(data);
        return true;
}

static bool testcard_load_from_file_y4m(const char *filename, struct video_desc *desc, vector<char>& in_file_contents) {
        struct y4m_metadata info;
        unsigned char *data = nullptr;
        if (y4m_read(filename, &info, &data, malloc) == 0) {
                return false;
        }
        assert((info.subsampling == Y4M_SUBS_422 && info.bitdepth == 8) || (info.subsampling == Y4M_SUBS_444 && info.bitdepth > 8));
        desc->width = info.width;
        desc->height = info.height;
        desc->color_spec = info.bitdepth == 8 ? UYVY : Y416;
        in_file_contents.resize(vc_get_datalen(desc->width, desc->height, desc->color_spec));
        if (info.bitdepth == 8) {
                i422_8_to_uyvy(desc->width, desc->height, (char *) data, in_file_contents.data());
        } else {
                i444_16_to_y416(desc->width, desc->height, (char *) data, in_file_contents.data(), info.bitdepth);
        }
        free(data);
        return true;
}

static bool testcard_load_from_file(const char *filename, struct video_desc *desc, vector<char>& in_file_contents, bool deduce_pixfmt) {
        if (ends_with(filename, ".pam") || ends_with(filename, ".pnm") || ends_with(filename, ".ppm")) {
                return testcard_load_from_file_pam(filename, desc, in_file_contents);
        } else if (ends_with(filename, ".y4m")) {
                return testcard_load_from_file_y4m(filename, desc, in_file_contents);
        }

        if (deduce_pixfmt && strchr(filename, '.') != nullptr && get_codec_from_file_extension(strrchr(filename, '.') + 1)) {
                desc->color_spec = get_codec_from_file_extension(strrchr(filename, '.') + 1);
        }
        long data_len = vc_get_datalen(desc->width, desc->height, desc->color_spec);
        in_file_contents.resize(data_len);
        char *data = in_file_contents.data();
        bool ret = true;
        FILE *in = fopen(filename, "r");
        if (in == nullptr) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << filename << " fopen: " << ug_strerror(errno) << "\n";
                return false;
        }
        fseek(in, 0L, SEEK_END);
        long filesize = ftell(in);
        if (filesize == -1) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "ftell: " << ug_strerror(errno) << "\n";
                filesize = data_len;
        }
        fseek(in, 0L, SEEK_SET);

        do {
                if (data_len != filesize) {
                        int level = data_len < filesize ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR;
                        LOG(level) << MOD_NAME  << "Wrong file size for selected "
                                "resolution and codec. File size " << filesize << ", "
                                "computed size " << data_len << "\n";
                        filesize = data_len;
                        if (level == LOG_LEVEL_ERROR) {
                                ret = false; break;
                        }
                }

                if (fread(data, filesize, 1, in) != 1) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot read file %s\n", filename);
                        ret = false; break;
                }
        } while (false);

        fclose(in);
        return ret;
}

static int vidcap_testcard_init(struct vidcap_params *params, void **state)
{
        struct testcard_state *s = nullptr;
        char *filename = nullptr;
        const char *strip_fmt = NULL;
        char *save_ptr = NULL;
        int ret = VIDCAP_INIT_FAIL;
        char *tmp;
        vector<char> in_file_contents;

        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                printf("testcard options:\n");
                col() << TBOLD(TRED("\t-t testcard") << "[:size=<width>x<height>][:fps=<fps>][:codec=<codec>]") << "[:file=<filename>][:p][:s=<X>x<Y>][:i|:sf][:still][:pattern=<pattern>] " << TBOLD("| -t testcard:help\n");
                col() << "or\n";
                col() << TBOLD(TRED("\t-t testcard") << ":<width>:<height>:<fps>:<codec>") << "[:other_opts]\n";
                col() << "where\n";
                col() << TBOLD("\t<filename>") << " - use file named filename instead of default bars\n";
                col() << TBOLD("\tp") << "          - pan with frame\n";
                col() << TBOLD("\ts") << "          - split the frames into XxY separate tiles\n";
                col() << TBOLD("\ti|sf") << "       - send as interlaced or segmented frame (if none of those is set, progressive is assumed)\n";
                col() << TBOLD("\tstill") << "      - send still image\n";
                col() << TBOLD("\tpattern") << "    - pattern to use, use \"" << TBOLD("pattern=help") << "\" for options\n";
                col() << "\n";
                testcard_show_codec_help("testcard", false);
                col() << TBOLD("Note:") << " only certain codec and generator combinations produce full-depth samples (not up-sampled 8-bit), use " << TBOLD("pattern=help") << " for details.\n";
                return VIDCAP_INIT_NOERR;
        }

        s = new testcard_state();
        if (!s)
                return VIDCAP_INIT_FAIL;

        char *fmt = strdup(vidcap_params_get_fmt(params));
        char *ptr = fmt;

        bool pixfmt_default = true;
        struct video_desc desc = [&]{ return strlen(ptr) == 0 || !isdigit(ptr[0]) ? default_format : (pixfmt_default = false, parse_format(&ptr, &save_ptr));}();
        if (!desc) {
                goto error;
        }

        tmp = strtok_r(ptr, ":", &save_ptr);
        while (tmp) {
                if (strcmp(tmp, "p") == 0) {
                        s->pan = 48;
                } else if (strstr(tmp, "file=") == tmp || strstr(tmp, "filename=") == tmp) {
                        filename = strchr(tmp, '=') + 1;
                } else if (strncmp(tmp, "s=", 2) == 0) {
                        strip_fmt = tmp;
                } else if (strcmp(tmp, "i") == 0) {
                        s->frame->interlacing = INTERLACED_MERGED;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'i' option. Use format testcard:1920:1080:50i:UYVY instead!\n");
                } else if (strcmp(tmp, "sf") == 0) {
                        s->frame->interlacing = SEGMENTED_FRAME;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'sf' option. Use format testcard:1920:1080:25sf:UYVY instead!\n");
                } else if (strcmp(tmp, "still") == 0) {
                        s->still_image = true;
                } else if (strncmp(tmp, "pattern=", strlen("pattern=")) == 0) {
                        const char *pattern = tmp + strlen("pattern=");
                        s->pattern = pattern;
                } else if (strstr(tmp, "codec=") == tmp) {
                        desc.color_spec = get_codec_from_name(strchr(tmp, '=') + 1);
                        pixfmt_default = false;
                } else if (strstr(tmp, "size=") == tmp && strchr(tmp, 'x') != nullptr) {
                        desc.width = stoi(strchr(tmp, '=') + 1);
                        desc.height = stoi(strchr(tmp, 'x') + 1);
                } else if (strstr(tmp, "fps=") == tmp) {
                        if (!parse_fps(strchr(tmp, '=') + 1, &desc)) {
                                goto error;
                        }
                } else {
                        fprintf(stderr, "[testcard] Unknown option: %s\n", tmp);
                        goto error;
                }
                tmp = strtok_r(NULL, ":", &save_ptr);
        }

        if (desc.color_spec == VIDEO_CODEC_NONE || desc.width <= 0 || desc.height <= 0 || desc.fps <= 0.0) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong video format: " << desc << "\n";
                goto error;
        }

        if (filename) {
                if (!testcard_load_from_file(filename, &desc, in_file_contents, pixfmt_default)) {
                        goto error;
                }
        }

        if (!s->still_image && codec_is_planar(desc.color_spec)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Planar pixel format '%s', using still picture.\n", get_codec_name(desc.color_spec));
                s->still_image = true;
        }

        s->frame = vf_alloc_desc(desc);

        s->generator = video_pattern_generator_create(s->pattern.c_str(), s->frame->tiles[0].width, s->frame->tiles[0].height, s->frame->color_spec,
                        s->still_image ? 0 : vc_get_linesize(desc.width, desc.color_spec) + s->pan);
        if (!s->generator) {
                ret = s->pattern.find("help") != string::npos ? VIDCAP_INIT_NOERR : VIDCAP_INIT_FAIL;
                goto error;
        }
        if (in_file_contents.size() > 0) {
                video_pattern_generator_fill_data(s->generator, in_file_contents.data());
        }

        s->last_frame_time = std::chrono::steady_clock::now();

        LOG(LOG_LEVEL_INFO) << MOD_NAME << "capture set to " << desc << ", bpc "
                << get_bits_per_component(s->frame->color_spec) << ", pattern: " << s->pattern
                << ", audio " << (s->grab_audio ? "on" : "off") << "\n";

        if (strip_fmt != NULL) {
                LOG(LOG_LEVEL_ERROR) << "Multi-tile testcard (stripping) is currently broken, you can use eg. \"-t aggregate -t testcard[args] -t testcard[args]\" instead!\n";
                goto error;
#if 0
                if(configure_tiling(s, strip_fmt) != 0) {
                        goto error;
                }
#endif
        }

        if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                if (!configure_audio(s)) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot initialize audio!\n";
                        goto error;
                }
        }

        free(fmt);

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(fmt);
        vf_free(s->frame);
        delete s;
        return ret;
}

static void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = (struct testcard_state *) state;
        if (s->tiled) {
                int i;
                for (i = 0; i < s->tiles_cnt_horizontal; ++i) {
                        free(s->tiles_data[i]);
                }
                vf_free(s->tiled);
        }
        vf_free(s->frame);
        ring_buffer_destroy(s->midi_buf);
        video_pattern_generator_destroy(s->generator);
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

        if (state->grab_audio) {
                state->audio.data_len = state->audio.ch_count * state->audio.bps * AUDIO_SAMPLE_RATE / state->frame->fps;
                state->audio.data += state->audio.data_len;
                if (state->audio.data + state->audio.data_len > state->audio_data.data() + AUDIO_BUFFER_SIZE(state->audio.ch_count)) {
                        state->audio.data = state->audio_data.data();
                }
                *audio = &state->audio;
        } else {
                *audio = NULL;
        }

        vf_get_tile(state->frame, 0)->data = video_pattern_generator_next_frame(state->generator);

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

static void vidcap_testcard_probe(device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;

        *count = 1;
        *available_devices = (struct device_info *) calloc(*count, sizeof(struct device_info));
        auto& card = **available_devices;
        snprintf(card.name, sizeof card.name, "Testing signal");

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

        snprintf(card.modes[0].name,
                        sizeof card.modes[0].name, "Default");
        snprintf(card.modes[0].id,
                        sizeof card.modes[0].id,
                        "{\"width\":\"\", "
                        "\"height\":\"\", "
                        "\"format\":\"\", "
                        "\"fps\":\"\"}");

        int i = 1;
        for(const auto &pix_fmt : pix_fmts){
                for(const auto &size : sizes){
                        for(const auto &fps : framerates){
                                snprintf(card.modes[i].name,
                                                sizeof card.name,
                                                "%dx%d@%d %s",
                                                size.width, size.height,
                                                fps, pix_fmt);
                                snprintf(card.modes[i].id,
                                                sizeof card.modes[0].id,
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
        dev_add_option(&card, "Still", "Send still image", "still", ":still", true);
        dev_add_option(&card, "Pattern", "Pattern to use", "pattern", ":pattern=", false);
}

static const struct video_capture_info vidcap_testcard_info = {
        vidcap_testcard_probe,
        vidcap_testcard_init,
        vidcap_testcard_done,
        vidcap_testcard_grab,
        MOD_NAME,
};

REGISTER_MODULE(testcard, &vidcap_testcard_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

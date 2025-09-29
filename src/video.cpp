/**
 * @file   video.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This file defines some common video functions.
 *
 * These function are neither video frame nor video codec related.
 */
/*
 * Copyright (c) 2013 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "rtp/rtp_types.h" // FPS_MAX
#include "utils/macros.h"
#include "video.h"

#include <iomanip>
#include <sstream>
#include <map>

#define MOD_NAME "[video] "

using namespace std;

typedef struct {
        const char *name;
        int x;
        int y;
} video_mode_info_t;

const static map<enum video_mode, video_mode_info_t> video_mode_info = {
        { VIDEO_UNKNOWN, { "(unknown)", 0, 0 }},
        { VIDEO_NORMAL, { "normal", 1, 1 }},
        { VIDEO_DUAL, { "dual-link", 1, 2 }},
        { VIDEO_STEREO, { "3D", 2, 1 }},
        { VIDEO_4K, { "tiled-2x2", 2, 2 }},
        { VIDEO_3X1, { "3x1", 3, 1 }},
};

/**
 * This function matches string representation of video mode with its
 * respective enumeration value.
 *
 * @param reuqested_mode textual representation of video mode
 * @return valid member of enum video mode
 */
enum video_mode get_video_mode_from_str(const char *requested_mode) {
        if(strcasecmp(requested_mode, "help") == 0) {
                printf("Video mode options:\n");
                auto it = video_mode_info.begin();
                while (it != video_mode_info.end()) {
                        if (it == video_mode_info.begin()) { // omit unknown
                                ++it;
                                continue;
                        }
                        printf("\t%s\n", it->second.name);
                        if (++it != video_mode_info.end()) {
                        }
                }
                return VIDEO_UNKNOWN;
        } else {
                for (auto const &it : video_mode_info) {
                        if (strcasecmp(requested_mode, it.second.name) == 0) {
                                return it.first;
                        }
                }
                fprintf(stderr, "Unknown video mode (see -M help)\n");
                return VIDEO_UNKNOWN;
        }
}

int get_video_mode_tiles_x(enum video_mode video_mode)
{
        return video_mode_info.at(video_mode).x;
}

int get_video_mode_tiles_y(enum video_mode video_mode)
{
        return video_mode_info.at(video_mode).y;
}

const char *get_video_mode_description(enum video_mode video_mode)
{
        return video_mode_info.at(video_mode).name;
}

enum video_mode guess_video_mode(int num_substreams)
{
        assert(num_substreams > 0);

        switch (num_substreams) {
                case 1:
                        return VIDEO_NORMAL;
                case 2:
                        return VIDEO_STEREO;
                case 3:
                        return VIDEO_3X1;
                case 4:
                        return VIDEO_4K;
                default:
                        return VIDEO_UNKNOWN;
        }
}

std::ostream& operator<<(std::ostream& os, const video_desc& desc)
{
        std::streamsize p = os.precision();
        ios_base::fmtflags f = os.flags();
        if (desc.tile_count > 1) {
                os << desc.tile_count << "*";
        }
        os << desc.width << "x" << desc.height << " @" << setprecision(2) << setiosflags(ios_base::fixed)
                << desc.fps * (desc.interlacing == PROGRESSIVE || desc.interlacing == SEGMENTED_FRAME ? 1 : 2)
                << get_interlacing_suffix(desc.interlacing) << ", codec "
                << get_codec_name(desc.color_spec);
        os.precision(p);
        os.flags(f);
        return os;
}

std::ostream& operator<<(std::ostream& os, const codec_t& color_spec)
{
        os << get_codec_name(color_spec);
        return os;
}

std::istream& operator>>(std::istream& is, video_desc& desc)
{
        video_desc out;
        char buf[1024];
        char *line = buf;

        char interl_suffix[4] = "";
        char codec_name[11] = "";

        out.tile_count = 1;

        unsigned int i = 0;
        int num_spaces = 0;
        while (is.good() && i < sizeof buf - 1) {
                char c;
                is.get(c);
                if (!is.good()) {
                        break;
                }
                if (num_spaces == 3 &&
                                !(isalnum(c) || c == '.')) // allowed symbols in codec name
                {
                        is.unget();
                        break;
                } else {
                        buf[i++] = c;
                }
                if (c == ' ') {
                        num_spaces += 1;
                }
        }
        buf[i] = '\0';

        /// @todo regex matching would be better
        if (strchr(line, '*')) {
                out.tile_count = atoi(line);
                line = strchr(line, '*') + 1;
        }
        int ret = sscanf(line, "%dx%d @%lf%3[a-z], codec %10s", &out.width, &out.height,
                &out.fps, interl_suffix, codec_name);

        if (ret != 5) {
                is.setstate(std::ios::failbit);
                log_msg(LOG_LEVEL_ERROR, "Malformed video_desc representation!\n");
                return is;
        }

        out.color_spec = get_codec_from_name(codec_name);
        out.interlacing = get_interlacing_from_suffix(interl_suffix);
        out.fps = out.fps / (out.interlacing == PROGRESSIVE || out.interlacing == SEGMENTED_FRAME ? 1 : 2);
        desc = out;
        return is;
}

bool video_desc::operator==(video_desc const & other) const
{
        return video_desc_eq(*this, other);

}

bool video_desc::operator!=(video_desc const & other) const
{
        return !video_desc_eq(*this, other);

}

bool video_desc::operator!() const
{
        return color_spec == VIDEO_CODEC_NONE;
}

video_desc::operator string() const
{
        ostringstream oss;
        oss << *this;
        return oss.str();
}

const char *video_desc_to_string(struct video_desc d)
{
        thread_local string desc_str;
        desc_str = d;
        return desc_str.c_str();
}

/**
 * @retval 0.0  FPS not specified
 * @retval <0.0 wrong FPS specification
 */
static double
parse_fps(const char *string, bool interlaced)
{
        const char *fps_str = strpbrk(string, "DKdikp");
        assert(fps_str != nullptr);
        fps_str += 1;
        if (strlen(fps_str) == 0) {
                return 0.0;
        }
        char      *endptr = nullptr;
        const long fps_l  = strtol(fps_str, &endptr, 10);
        if (fps_l <= 0 || fps_l > FPS_MAX || *endptr != '\0') {
                MSG(ERROR, "Wrong mode FPS specification: %s\n", string);
                return -1;
        }
        double fps = 0;
        // adjust FPS (Hi29 has FPS 29.97)
        switch (fps_l) {
        case 23:
                fps = 23.98;
                break;
        case 29:
                fps = 29.97;
                break;
        case 59:
                fps = 59.94;
                break;
        default:
                fps = (double) fps_l;
                break;
        }
        // convert field per sec to FPS if needed (interlaced)
        return interlaced ? fps / 2.0 : fps;
}

/**
 * @returns video description according to the configuration string, allowing
 * BMD-like FourCC (pal, ntsc, Hi59 etc.)
 */
struct video_desc get_video_desc_from_string(const char *string)
{
        const struct {
                const char *name;
                struct video_desc desc;
        } map[] = {
                {"microvision", { 16, 16, UYVY, 10, PROGRESSIVE, 1 }},
                {"sqcif", { 126, 96, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"qcif", { 176, 144, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"cif", { 352, 288, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"4cif", { 704, 576, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"ntsc", { 720, 480, UYVY, 29.97, INTERLACED_MERGED, 1 }},
                {"pal", { 720, 576, UYVY, 25, INTERLACED_MERGED, 1 }},
                {"cga", { 320, 200, UYVY, 60, PROGRESSIVE, 1 }},
                {"vga", { 640, 480, UYVY, 60, PROGRESSIVE, 1 }},
                {"svga", { 800, 600, UYVY, 60, PROGRESSIVE, 1 }},
                {"xga", { 1024, 768, UYVY, 60, PROGRESSIVE, 1 }},
                {"wxga", { 1280, 768, UYVY, 60, PROGRESSIVE, 1 }},
                {"sxga", { 1280, 1024, UYVY, 60, PROGRESSIVE, 1 }},
                {"uxga", { 1600, 1200, UYVY, 60, PROGRESSIVE, 1 }},
                {"wuxga", { 1920, 1200, UYVY, 60, PROGRESSIVE, 1 }},
                {"23ps", { 1920, 1080, UYVY, 23.98, PROGRESSIVE, 1 }},
                {"24ps", { 1920, 1080, UYVY, 24, PROGRESSIVE, 1 }},
                {"hd", { 1280, 720, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"720p", { 1280, 720, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"fhd", { 1920, 1080, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"1080p", { 1920, 1080, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"uhd", { 3840, 2160, UYVY, 29.97, PROGRESSIVE, 1 }},
                {"dci", { 2048, 1080, UYVY, 23.98, PROGRESSIVE, 1 }},
                {"dci4", { 4096, 2160, UYVY, 23.98, PROGRESSIVE, 1 }},
        };
        if (strcasecmp(string, "help") == 0) {
                printf("Recognized video formats: ");
                for (unsigned i = 0; i < sizeof map / sizeof map[0]; ++i) {
                        color_printf(TBOLD("%s") ", ", map[i].name);
                }
                color_printf(TBOLD("{Hp,Hi,hp,2d,4k,4d,1080i,1080p,720p,2160p}{23,24,25,29,30,59,60}") "\n");
                return {};

        }
        for (unsigned i = 0; i < sizeof map / sizeof map[0]; ++i) {
                if (strcasecmp(map[i].name, string) == 0 || (strlen(string) == 4 && string[3] == ' ' && strncasecmp(map[i].name, string, 3) == 0)) {
                        return map[i].desc;
                }
        }
        // BMD-like syntax (+ line-counted syntax like 1080p<FPS>)
        struct video_desc ret{};
        ret.color_spec = UYVY;
        ret.tile_count  = 1;
        if (strncmp(string, "Hp", 2) == 0 || starts_with(string, "1080p")) {
                ret.width = 1920;
                ret.height = 1080;
        } else if (strncmp(string, "Hi", 2) == 0 || starts_with(string, "1080i")) {
                ret.width = 1920;
                ret.height = 1080;
                ret.interlacing = INTERLACED_MERGED;
        } else if (strncmp(string, "hp", 2) == 0 || starts_with(string, "720p")) {
                ret.width = 1280;
                ret.height = 720;
        } else if (strncasecmp(string, "2d", 2) == 0) {
                ret.width = 2048;
                ret.height = 1080;
        } else if (strncasecmp(string, "4k", 2) == 0 || starts_with(string, "2160p")) {
                ret.width = 3840;
                ret.height = 2160;
        } else if (strncasecmp(string, "4d", 2) == 0) {
                ret.width = 4096;
                ret.height = 2160;
        }
        if (ret.width == 0) {
                MSG(ERROR, "Unrecognized video mode: %s\n", string);
                return {};
        }
        ret.fps = parse_fps(string, ret.interlacing == INTERLACED_MERGED);
        if (ret.fps < .0) {
                return {};
        }

        return ret;
}

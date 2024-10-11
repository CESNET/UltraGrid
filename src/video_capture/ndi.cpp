/**
 * @file   video_capture/ndi.cpp
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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
 * Parsing IPv6 URLs (vidcap_state_ndi::requested_url) - what is the syntax?
 * Looks like NDI doesn't support that yet (question is the position of ":" which
 * delimits host and port)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <type_traits> // static_assert

#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "ndi_common.h"
#include "utils/color_out.h"
#include "utils/macros.h" // OPTIMIZED_FOR
#include "video.h"
#include "video_capture.h"

#if __has_include(<ndi_version.h>)
#include <ndi_version.h>
#endif

static constexpr double DEFAULT_AUDIO_DIVISOR = 1;
static constexpr const char *MOD_NAME = "[NDI cap.] ";

using std::array;
using std::cout;
using std::max;
using std::min;
using std::string;
using std::chrono::duration_cast;
using std::chrono::steady_clock;

static void vidcap_ndi_done(void *state);

struct vidcap_state_ndi {
        static_assert(NDILIB_CPP_DEFAULT_CONSTRUCTORS == 0, "Don't use default C++ NDI constructors - we are using run-time dynamic lib load");
        LIB_HANDLE lib{};
        const NDIlib_t *NDIlib{};
        NDIlib_recv_instance_t pNDI_recv = nullptr;
        NDIlib_find_instance_t pNDI_find = nullptr;
        array<struct audio_frame, 2> audio;
        int audio_buf_idx = 0;
        bool capture_audio = false;
        struct video_desc last_desc{};

        NDIlib_video_frame_v2_t field_0{}; ///< stored to asssemble interleaved interlaced video together with field 1

        string requested_name; // if not empty recv from requested NDI name
        string requested_url; // if not empty recv from requested URL (either addr or addr:port)
        NDIlib_find_create_t find_create_settings{true, nullptr, nullptr};
        NDIlib_recv_create_v3_t create_settings{NDIlib_source_t(), NDIlib_recv_color_format_best, NDIlib_recv_bandwidth_highest, true, NULL};

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        int frames = 0;

        /// sample divisor derived from audio reference level - 1 for 0 dB, 10 for 20 dB
        double audio_divisor = DEFAULT_AUDIO_DIVISOR; // NOLINT

        void print_stats() {
                auto now = steady_clock::now();
                double seconds = duration_cast<std::chrono::microseconds>(now - t0).count() / 1000000.0;
                if (seconds > 5) {
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << frames << " frames in "
                                << seconds << " seconds = " <<  frames / seconds << " FPS\n";
                        frames = 0;
                        t0 = now;
                }
        }
        ~vidcap_state_ndi() {
                free(const_cast<char *>(find_create_settings.p_extra_ips));
        }
};

static void show_help(struct vidcap_state_ndi *s) {
        col() << "Usage:\n"
                "\t" << SBOLD(SRED("-t ndi") << "[:help][:name=<n>][:url=<u>][:audio_level=<l>][:bandwidth=<b>][:color=<c>][:extra_ips=<ip>][:progressive] | -t ndi:help[:extra_ips=<ip>]") << "\n" <<
                "where\n";

        col() << TBOLD("\tname\n") <<
                "\t\tname of the NDI source in form "
                "\"MACHINE_NAME (NDI_SOURCE_NAME)\"\n";
        col() << TBOLD("\turl\n") <<
                "\t\tURL, typically <ip> or <ip>:<port>\n";
        col() << TBOLD("\taudio_level\n") <<
                "\t\taudio headroom above reference level (in dB, or mic/line, default " << 20 * log(DEFAULT_AUDIO_DIVISOR) / log(10) << ")\n";
        col() << TBOLD("\tbandwidth\n") <<
                "\t\trequired bandwidth, " << TBOLD(<< NDIlib_recv_bandwidth_audio_only << ) << " - audio only, " TBOLD(<< NDIlib_recv_bandwidth_lowest <<) " - lowest, " TBOLD(<< NDIlib_recv_bandwidth_highest <<) " - highest (default)\n";
        col() << TBOLD("\tcolor\n") <<
                "\t\tcolor format, " << TBOLD(<< NDIlib_recv_color_format_BGRX_BGRA <<) << " - BGRX/BGRA, " << TBOLD(<< NDIlib_recv_color_format_UYVY_BGRA <<)  << " - UYVY/BGRA, " <<
                TBOLD(<< NDIlib_recv_color_format_RGBX_RGBA <<) << " - RGBX/RGBA, " << TBOLD(<< NDIlib_recv_color_format_UYVY_RGBA <<)  << " - UYVY/RGBA, " << TBOLD(<< NDIlib_recv_color_format_fastest <<) <<
                " - fastest (UYVY), " << TBOLD(<< NDIlib_recv_color_format_best <<) << " - best (default, P216/UYVY)\n"
                "\t\tSelection is on NDI runtime and usually depends on presence of alpha channel. UG ignores alpha channel for YCbCr codecs.\n";
        col() << TBOLD("\textra_ips\n") <<
                "\t\tadditional IP addresses for query in format \"12.0.0.8,13.0.12.8\". Can be used if DNS-SD is not usable.\n";
        col() << TBOLD("\tprogressive\n") <<
                "\t\tprefer progressive capture for interlaced input\n"
                "\n";

        cout << "available sources (tentative, format: name - url):\n";
        auto *pNDI_find = s->NDIlib->find_create_v2(&s->find_create_settings);
        if (pNDI_find == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot create finder object!\n";
                return;
        }

        uint32_t nr_sources = 0;
        const NDIlib_source_t* p_sources = nullptr;
        // Give sources some time to occur
        usleep(500 * 1000);
        // we do not usea NDIlib_find_wait_for_sources() here because: 1) if there is
        // no source, it will still wait requested amount of time and 2) if there are
        // more sources, it will continue after first source found while there can be more
        p_sources = s->NDIlib->find_get_current_sources(pNDI_find, &nr_sources);
        for (int i = 0; i < static_cast<int>(nr_sources); ++i) {
                cout << "\t" << p_sources[i].p_ndi_name << " - " << p_sources[i].p_url_address << "\n";
        }
        if (nr_sources == 0) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "No sources found!\n";
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "You can pass \"extra_ips\" if mDNS is not operable.\n";
        }
        cout << "\n";
        s->NDIlib->find_destroy(pNDI_find);
#ifdef NDI_VERSION
        cout << NDI_VERSION "\n";
#elif defined USE_NDI_VERSION
        cout << "NDI version " << USE_NDI_VERSION << "\n";
#endif
}

static int vidcap_ndi_init(struct vidcap_params *params, void **state)
{
        NDI_PRINT_COPYRIGHT(void);
        using namespace std::string_literals;
        using std::stoi;
        auto s = new vidcap_state_ndi();
        s->NDIlib = NDIlib_load(&s->lib);
        if (s->NDIlib == NULL) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot open NDI library!\n";
                delete s;
                return VIDCAP_INIT_FAIL;
        }
        // Not required, but "correct" (see the SDK documentation)
        if (!s->NDIlib->initialize()) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot initialize NDI!\n";
                delete s;
                return VIDCAP_INIT_FAIL;
        }
        if ((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) != 0u) {
                s->capture_audio = true;
        }

        const char *fmt = vidcap_params_get_fmt(params);
        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);
        char *item, *save_ptr;
        bool req_show_help = false;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != nullptr) {
                tmp = nullptr;

                if (strcmp(item, "help") == 0) {
                        req_show_help = true; // defer execution to honor extra_ips option (if present);
                        continue;
                }
                if (strncmp(item, "name=", strlen("name=")) == 0) {
                        s->requested_name = item + strlen("name=");
                } else if (strncmp(item, "url=", strlen("url=")) == 0) {
                        s->requested_url = item + strlen("url=");
                        if (isdigit(*save_ptr)) { // ...:url=abc:123:bbb
                                s->requested_url += string(":") + strtok_r(nullptr, ":", &save_ptr);
                                cout << s->requested_url;
                        }
                } else if (strstr(item, "audio_level=") == item) {
                        char *val = item + strlen("audio_level=");
                        long ref_level = 0;
                        if (strcasecmp(val, "mic") == 0) {
                                ref_level = 0;
                        } else if (strcasecmp(val, "line") == 0) {
                                ref_level = 20; // NOLINT
                        } else {
                                char *endptr = nullptr;
                                ref_level = strtol(val, &endptr, 0);
                                if (ref_level < 0 || ref_level >= INT_MAX || *val == '\0' || *endptr != '\0') {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong value: " << val << "!\n";
                                        delete s;
                                        return VIDCAP_INIT_NOERR;
                                }
                        }
                        s->audio_divisor = pow(10.0, ref_level / 20.0); // NOLINT
                } else if (strstr(item, "extra_ips=") == item) {
                        s->find_create_settings.p_extra_ips = strdup(item + "extra_ips="s.length());
                } else if (strstr(item, "bandwidth=") == item) {
                        s->create_settings.bandwidth = static_cast<NDIlib_recv_bandwidth_e>(stoi(strchr(item, '=') + 1));
                } else if (strstr(item, "color=") == item) {
                        s->create_settings.color_format = static_cast<NDIlib_recv_color_format_e>(stoi(item + "color="s.length()));
                } else if (strstr(item, "progressive") == item) {
                        s->create_settings.allow_video_fields = false;
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unknown option: " << item << "\n";
                        delete s;
                        return VIDCAP_INIT_NOERR;
                }
        }

        if (req_show_help) {
                show_help(s);
                vidcap_ndi_done(s);
                return VIDCAP_INIT_NOERR;
        }

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_ndi_done(void *state)
{
        auto s = static_cast<struct vidcap_state_ndi *>(state);

        for (auto & i : s->audio) {
                free(i.data);
        }

        if (s->field_0.p_data != nullptr) {
                s->NDIlib->recv_free_video_v2(s->pNDI_recv, &s->field_0);
        }

        // Destroy the NDI finder.
        s->NDIlib->find_destroy(s->pNDI_find);

        s->NDIlib->recv_destroy(s->pNDI_recv);

        // Not required, but nice
        s->NDIlib->destroy();
        close_ndi_library(s->lib);

        delete s;
}

static void audio_append_pcm(struct vidcap_state_ndi *s, NDIlib_audio_frame_v3_t *frame)
{
        struct audio_desc d{4, frame->sample_rate, static_cast<int>(audio_capture_channels > 0 ? audio_capture_channels : frame->no_channels), AC_PCM};

        if (frame->no_channels != d.ch_count && s->audio[s->audio_buf_idx].ch_count == 0 && s->audio_buf_idx == 0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Requested " << s->audio[s->audio_buf_idx].ch_count
                        << " channels, stream has " << frame->no_channels << "!\n";
        }

        if (!audio_desc_eq(d, audio_desc_from_frame(&s->audio[s->audio_buf_idx]))) {
                free(s->audio[s->audio_buf_idx].data);
                s->audio[s->audio_buf_idx].bps = 4;
                s->audio[s->audio_buf_idx].sample_rate = frame->sample_rate;
                s->audio[s->audio_buf_idx].ch_count = d.ch_count;
                s->audio[s->audio_buf_idx].data_len = 0;
                s->audio[s->audio_buf_idx].max_size =
                        4 * d.ch_count * frame->sample_rate / 5; // 200 ms
                s->audio[s->audio_buf_idx].data = static_cast<char *>(malloc(s->audio[s->audio_buf_idx].max_size));
        }

        for (int i = 0; i < frame->no_samples; ++i) {
                float *in = (float *)(void *) frame->p_data + i;
                int32_t *out = (int32_t *)(void *) s->audio[s->audio_buf_idx].data + i * d.ch_count;
                int j = 0;
                for (; j < min(d.ch_count, frame->no_channels); ++j) {
                        if (s->audio[s->audio_buf_idx].data_len >= s->audio[s->audio_buf_idx].max_size) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Audio frame too small!\n";
                                return;
                        }
                        *out++ = max<double>(INT32_MIN, min<double>(INT32_MAX, *in / s->audio_divisor * INT32_MAX));
                        in += frame->channel_stride_in_bytes / sizeof(float);
                        s->audio[s->audio_buf_idx].data_len += sizeof(int32_t);
                }
                for (; j < d.ch_count; ++j) { // fill excess channels with zeros
                        *out++ = 0;
                        s->audio[s->audio_buf_idx].data_len += sizeof(int32_t);
                }
        }
}

static const NDIlib_source_t *get_matching_source(struct vidcap_state_ndi *s, const NDIlib_source_t *sources, int nr_sources) {
        if (s->requested_name.empty() && s->requested_url.empty()) {
                return sources + 0;
        }

        for (int i = 0; i < nr_sources; ++i) {
                if (!s->requested_name.empty()) {
                        if (s->requested_name != sources[i].p_ndi_name &&
                                        // match also '.*(name)' if name was given and it doesn't contain whole name (`hostname (name)` -> brace pair is checked)
                                        (std::regex_search(s->requested_name, std::regex("(.*)", std::regex::basic)) ||
                                         !std::regex_match(sources[i].p_ndi_name, std::regex(string(".*(") + s->requested_name + ")", std::regex::basic)))) {
                                continue;
                        }
                }
                if (!s->requested_url.empty()) {
                        const char *url_c = s->requested_url.c_str();
                        if (strncmp(url_c, sources[i].p_url_address, strlen(url_c)) != 0) { // doesn't match at all, requested url is not prefix of actual (so it cannot be <host> and <host>:<port>)
                                continue;
                        }

                        const char *url_suffix = sources[i].p_url_address + strlen(url_c);
                        if (strlen(url_suffix) > 0 && url_suffix[0] != ':') { // case "xy:z" (user requested "x") or "a:bc" (user requested "a:b")
                                continue;
                        }
                }
                return sources + i;
        }

        return nullptr;
}

using convert_t = void (*)(struct video_frame *, const uint8_t *, int in_stride, int field_idx, int total_fields);

static void convert_BGRA_RGBA(struct video_frame *out, const uint8_t *data, int in_stride, int field_idx, int total_fields)
{
        auto *out_p = reinterpret_cast<uint32_t *>(out->tiles[0].data) + field_idx * out->tiles[0].width;
        unsigned int width = out->tiles[0].width;
        for (unsigned int i = 0; i < out->tiles[0].height / total_fields; i += 1) {
                const auto *in_p = reinterpret_cast<const uint32_t *>(data + i * in_stride);
                OPTIMIZED_FOR (unsigned int j = 0; j < width; j++) {
                        uint32_t argb = *in_p++;
                        *out_p++ = (argb & 0xFF000000U) | ((argb & 0xFFU) << 16U) | (argb & 0xFF00U) | ((argb & 0xFF0000U) >> 16U);
                }
                out_p += (total_fields - 1) * out->tiles[0].width;
        }
}

static void convert_P216_Y216(struct video_frame *out, const uint8_t *data, int in_stride, int field_idx, int total_fields)
{
        auto *out_p = reinterpret_cast<uint16_t *>(out->tiles[0].data) + 2 * field_idx * out->tiles[0].width;
        unsigned int width = out->tiles[0].width;
        for (unsigned int i = 0; i < out->tiles[0].height / total_fields; i += 1) {
                const auto *in_y = reinterpret_cast<const uint16_t *>(data + i * in_stride);
                const auto *in_cb_cr = reinterpret_cast<const uint16_t *>(data + in_stride * (i + out->tiles[0].height / total_fields));
                OPTIMIZED_FOR (unsigned int j = 0; j < (width + 1) / 2; j += 1) {
                        *out_p++ = *in_y++;
                        *out_p++ = *in_cb_cr++;
                        *out_p++ = *in_y++;
                        *out_p++ = *in_cb_cr++;
                }
                out_p += (total_fields - 1) * out->tiles[0].width * 2;
        }
}

static void convert_memcpy(struct video_frame *out, const uint8_t *data, int in_stride, int field_idx, int total_fields)
{
        size_t linesize = vc_get_linesize(out->tiles[0].width, out->color_spec);
        auto *out_p = out->tiles[0].data + field_idx * linesize;
        for (unsigned int i = 0; i < out->tiles[0].height / total_fields; i += 1) {
                memcpy(out_p, data, linesize);
                out_p += total_fields * linesize;
                data += in_stride;
        }
}

static struct video_frame *vidcap_ndi_grab(void *state, struct audio_frame **audio)
{
        auto s = static_cast<struct vidcap_state_ndi *>(state);

        if (s->pNDI_find == nullptr) {
                // Create a finder
                s->pNDI_find = s->NDIlib->find_create_v2(&s->find_create_settings);
                if (s->pNDI_find == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot create object!\n";
                        return nullptr;
                }
        }

        if (s->pNDI_recv == nullptr) {
                // Wait until there is one source
                uint32_t nr_sources = 0;
                const NDIlib_source_t* p_sources = nullptr;
                // Wait until the sources on the nwtork have changed
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "Looking for source(s)...\n";
                s->NDIlib->find_wait_for_sources(s->pNDI_find, 100 /* 100 ms */);
                p_sources = s->NDIlib->find_get_current_sources(s->pNDI_find, &nr_sources);
                if (nr_sources == 0) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "No sources.\n";
                        return nullptr;
                }

                const NDIlib_source_t *source = get_matching_source(s, p_sources, nr_sources);
                if (source == nullptr) {
                        return nullptr;
                }

                // We now have at least one source, so we create a receiver to look at it.
                s->pNDI_recv = s->NDIlib->recv_create_v3(&s->create_settings);
                if (s->pNDI_recv == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to create receiver!\n";
                        return nullptr;
                }
                // Connect to our sources
                s->NDIlib->recv_connect(s->pNDI_recv, source);

                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Receiving from source: " << source->p_ndi_name << ", URL: " << source->p_url_address << "\n";
        }

        NDIlib_video_frame_v2_t video_frame;
        NDIlib_audio_frame_v3_t audio_frame;

        struct video_frame *out = nullptr;
        video_desc out_desc;

        switch (s->NDIlib->recv_capture_v3(s->pNDI_recv, &video_frame, &audio_frame, nullptr, 200))
        {       // No data
        case NDIlib_frame_type_none:
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "No data received.\n";
                // check disconnect
                if (s->NDIlib->recv_get_no_connections(s->pNDI_recv) > 0) {
                        break;
                }
                MSG(WARNING, "The source has disconnected, starting "
                             "new lookup!\n");
                s->NDIlib->recv_destroy(s->pNDI_recv);
                s->pNDI_recv = nullptr;
                return nullptr;

                // Video data
        case NDIlib_frame_type_video:
        {
                convert_t convert = nullptr;

                out_desc = {static_cast<unsigned int>(video_frame.xres),
                                static_cast<unsigned int>(video_frame.yres) *
                                       (video_frame.frame_format_type == NDIlib_frame_format_type_field_0 ||
                                       video_frame.frame_format_type == NDIlib_frame_format_type_field_1 ? 2 : 1),
                                VIDEO_CODEC_NONE,
                                static_cast<double>(video_frame.frame_rate_N) / video_frame.frame_rate_D,
                                video_frame.frame_format_type == NDIlib_frame_format_type_progressive ? PROGRESSIVE : INTERLACED_MERGED,
                                1};
                switch (video_frame.FourCC) {
                        case NDIlib_FourCC_type_UYVA:
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Detected input format UYVA, dropping alpha! Please report if not desired.\n";
                                // fall through
                        case NDIlib_FourCC_type_UYVY:
                                out_desc.color_spec = UYVY;
                                break;
                        case NDIlib_FourCC_type_PA16:
                        case NDIlib_FourCC_type_P216:
                                convert = convert_P216_Y216;
                                out_desc.color_spec = Y216;
                                break;
                        case NDIlib_FourCC_type_BGRA:
                        case NDIlib_FourCC_type_BGRX:
                                convert = convert_BGRA_RGBA;
                                // fall through
                        case NDIlib_FourCC_type_RGBA:
                        case NDIlib_FourCC_type_RGBX:
                                out_desc.color_spec = RGBA;
                                break;
                        default:
                                if (video_frame.FourCC == to_fourcc('H', '2', '6', '4') || video_frame.FourCC == to_fourcc('h', '2', '6', '4')) {
                                        out_desc.color_spec = H264;
                                } else if (video_frame.FourCC == to_fourcc('H', 'E', 'V', 'C') || video_frame.FourCC == to_fourcc('h', 'e', 'v', 'c')) {
                                        out_desc.color_spec = H265;
                                } else {
                                        array<char, sizeof(uint32_t) + 1> fcc_s{};
                                        memcpy(fcc_s.data(), &video_frame.FourCC, sizeof(uint32_t));
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unsupported codec '" << fcc_s.data() << "', please report to " PACKAGE_BUGREPORT "!\n";
                                        return {};
                                }
                }
                if (s->last_desc != out_desc) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Received video changed: " << out_desc << "\n";
                        s->last_desc = out_desc;
                        if (s->field_0.p_data != nullptr) {
                                s->NDIlib->recv_free_video_v2(s->pNDI_recv, &s->field_0);
                                s->field_0 = NDIlib_video_frame_v2_t{};
                        }
                        if (out_desc.color_spec == Y216) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Receiving 16-bit YCbCr, if not needed, consider using \"color=\" "
                                        "option to reduce required processing power.\n";
                        }
                }

                if (video_frame.frame_format_type == NDIlib_frame_format_type_field_0) {
                        s->field_0 = video_frame;
                        return nullptr;
                }
                if (video_frame.frame_format_type == NDIlib_frame_format_type_field_1) {
                        if (convert == nullptr) {
                                convert = convert_memcpy;
                        }
                }

                if (convert != nullptr) {
                        out = vf_alloc_desc_data(out_desc);
                        int stride = video_frame.line_stride_in_bytes != 0 ? video_frame.line_stride_in_bytes : vc_get_linesize(video_frame.xres, out_desc.color_spec);
                        int field_count = video_frame.frame_format_type == NDIlib_frame_format_type_field_1 ? 2 : 1;
                        if (field_count > 1) {
                                if (s->field_0.p_data == nullptr) {
                                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Missing corresponding field!\n";
                                } else {
                                        convert(out, s->field_0.p_data, stride, 0, field_count);
                                        s->NDIlib->recv_free_video_v2(s->pNDI_recv, &s->field_0);
                                        s->field_0 = NDIlib_video_frame_v2_t{};
                                }
                                convert(out, video_frame.p_data, stride, 1, field_count);
                        } else {
                                convert(out, video_frame.p_data, stride, 0, 1);
                        }
                        s->NDIlib->recv_free_video_v2(s->pNDI_recv, &video_frame);
                        out->callbacks.dispose = vf_free;
                } else {
                        out = vf_alloc_desc(out_desc);
                        out->tiles[0].data = reinterpret_cast<char*>(video_frame.p_data);
                        if (is_codec_opaque(out_desc.color_spec)) {
                                out->tiles[0].data_len = video_frame.data_size_in_bytes;
                        }
                        struct dispose_udata_t {
                                NDIlib_video_frame_v2_t video_frame;
                                NDIlib_recv_instance_t pNDI_recv;
                                void(*recv_free_video_v2)(NDIlib_recv_instance_t p_instance, const NDIlib_video_frame_v2_t* p_video_data);
                        };
                        out->callbacks.dispose_udata = new dispose_udata_t{video_frame, s->pNDI_recv, s->NDIlib->recv_free_video_v2};
                        static auto dispose = [](struct video_frame *f) { auto du = static_cast<dispose_udata_t *>(f->callbacks.dispose_udata);
                                du->recv_free_video_v2(du->pNDI_recv, &du->video_frame);
                                delete du;
                                free(f);
                        };
                        out->callbacks.dispose = dispose;
                }
                s->frames += 1;
                s->print_stats();
                return out;
                break;
        }
                // Audio data
        case NDIlib_frame_type_audio:
                *audio = nullptr;
                if (s->capture_audio) {
                        if (audio_frame.FourCC == NDIlib_FourCC_audio_type_FLTP) {
                                audio_append_pcm(s, &audio_frame);
                                *audio = &s->audio[s->audio_buf_idx];
                                s->audio_buf_idx = (s->audio_buf_idx + 1) % 2;
                                s->audio[s->audio_buf_idx].data_len = 0;
                        } else {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unsupported audio codec 0x" << std::hex << audio_frame.FourCC << std::dec << ", please report!\n";
                        }
                }
                s->NDIlib->recv_free_audio_v3(s->pNDI_recv, &audio_frame);
                break;

        case NDIlib_frame_type_metadata:
                break;

        case NDIlib_frame_type_error:
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Error occured!\n";
                break;

        case NDIlib_frame_type_status_change:
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Status changed!\n";
                break;
        case NDIlib_frame_type_max:
                assert(0 && "NDIlib_frame_type_max is invalid!");
        }

        return nullptr;
}

static void vidcap_ndi_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_cards = nullptr;
        *count = 0;

        LIB_HANDLE tmp = nullptr;
        const NDIlib_t *NDIlib = NDIlib_load(&tmp);
        if (NDIlib == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot open NDI library!\n";
                return;
        }
        std::unique_ptr<std::remove_pointer<LIB_HANDLE>::type, void (*)(LIB_HANDLE)> lib(tmp, close_ndi_library);

        auto pNDI_find = NDIlib->find_create_v2(nullptr);
        if (pNDI_find == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot create finder object!\n";
                return;
        }

        uint32_t nr_sources = 0;
        const NDIlib_source_t* p_sources = nullptr;
        // Give sources some time to occur
        usleep(100 * 1000);
        // we do not usea NDIlib_find_wait_for_sources() here because: 1) if there is
        // no source, it will still wait requested amount of time and 2) if there are
        // more sources, it will continue after first source found while there can be more
        p_sources = NDIlib->find_get_current_sources(pNDI_find, &nr_sources);

        *available_cards = (struct device_info *) calloc(nr_sources, sizeof(struct device_info));
        if (*available_cards == nullptr) {
                NDIlib->find_destroy(pNDI_find);
                return;
        }
        *count = nr_sources;
        for (int i = 0; i < static_cast<int>(nr_sources); ++i) {
                auto& card = (*available_cards)[i];
                snprintf(card.dev, sizeof card.dev, ":url=%s", p_sources[i].p_url_address);
                snprintf(card.name, sizeof card.name, "%s", p_sources[i].p_ndi_name);
                snprintf(card.extra, sizeof card.extra, "\"embeddedAudioAvailable\":\"t\"");
                card.repeatable = true;
        }

        NDIlib->find_destroy(pNDI_find);
}

static const struct video_capture_info vidcap_ndi_info = {
        vidcap_ndi_probe,
        vidcap_ndi_init,
        vidcap_ndi_done,
        vidcap_ndi_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(ndi, &vidcap_ndi_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */


/**
 * @file   video_capture/ndi.cpp
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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

#include <Processing.NDI.Lib.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifdef _WIN64
#pragma comment(lib, "Processing.NDI.Lib.x64.lib")
#else // _WIN64
#pragma comment(lib, "Processing.NDI.Lib.x86.lib")
#endif // _WIN64
#endif // _WIN32

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "rang.hpp"
#include "video.h"
#include "video_capture.h"

static constexpr double DEFAULT_AUDIO_DIVISOR = 1;
static constexpr const char *MOD_NAME = "[NDI] ";

using std::array;
using std::cout;
using std::max;
using std::min;
using std::string;
using std::chrono::duration_cast;
using std::chrono::steady_clock;

struct vidcap_state_ndi {
        NDIlib_recv_instance_t pNDI_recv = nullptr;
        NDIlib_find_instance_t pNDI_find = nullptr;
        array<struct audio_frame, 2> audio;
        int audio_buf_idx = 0;
        bool capture_audio = false;
        struct video_desc last_desc{};

        string requested_name; // if not empty recv from requested NDI name
        string requested_url; // if not empty recv from requested URL (either addr or addr:port)
        int requested_color = -1;

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        int frames = 0;

        /// sample divisor derived from audio reference level - 1 for 0 dB, 10 for 20 dB
        double audio_divisor = DEFAULT_AUDIO_DIVISOR; // NOLINT

        void print_stats() {
                auto now = steady_clock::now();
                double seconds = duration_cast<std::chrono::microseconds>(now - t0).count() / 1000000.0;
                if (seconds > 5) {
                        LOG(LOG_LEVEL_INFO) << "[NDI] " << frames << " frames in "
                                << seconds << " seconds = " <<  frames / seconds << " FPS\n";
                        frames = 0;
                        t0 = now;
                }
        }
};

static void show_help() {
        cout << "Usage:\n"
                "\t" << rang::style::bold << rang::fg::red << "-t ndi" << rang::fg::reset <<
                "[:help][:name=<n>][:url=<u>][:audio_level=<l>][color=<c>]\n" << rang::style::reset <<
                "\twhere\n"
                << rang::style::bold << "\t\tname\n" << rang::style::reset <<
                "\t\t\tname of the NDI source in form "
                "\"MACHINE_NAME (NDI_SOURCE_NAME)\"\n"
                << rang::style::bold << "\t\turl\n" << rang::style::reset <<
                "\t\t\tURL, typically <ip> or <ip>:<port>\n"
                << rang::style::bold << "\t\taudio_level\n" << rang::style::reset <<
                "\t\t\taudio headroom above reference level (in dB, or mic/line, default " << 20 * log(DEFAULT_AUDIO_DIVISOR) / log(10) << ")\n"
                << rang::style::bold << "\t\tcolor\n" << rang::style::reset <<
                "\t\t\tcolor format, 0 - BGRX/BGRA, 1 - UYVY/BGRA, 2 - RGBX/RGBA, 3 - UYVY/RGBA, 100 - fastest (UYVY), 101 - best (default, P216/UYVY)\n"
                "\t\t\tSelection is on NDI runtime and usually depends on presence of alpha channel. UG ignores alpha channel for YCbCr codecs.\n"
                "\n";

        cout << "\tavailable sources (tentative, format: name - url):\n";
        auto pNDI_find = NDIlib_find_create_v2();
        if (pNDI_find == nullptr) {
                LOG(LOG_LEVEL_ERROR) << "[NDI] Cannot create finder object!\n";
                return;
        }

        uint32_t nr_sources = 0;
        const NDIlib_source_t* p_sources = nullptr;
        // Give sources some time to occur
        usleep(500 * 1000);
        // we do not usea NDIlib_find_wait_for_sources() here because: 1) if there is
        // no source, it will still wait requested amount of time and 2) if there are
        // more sources, it will continue after first source found while there can be more
        p_sources = NDIlib_find_get_current_sources(pNDI_find, &nr_sources);
        for (int i = 0; i < static_cast<int>(nr_sources); ++i) {
                cout << "\t\t" << p_sources[i].p_ndi_name << " - " << p_sources[i].p_url_address << "\n";
        }
        cout << "\n";
        NDIlib_find_destroy(pNDI_find);
}

static int vidcap_ndi_init(struct vidcap_params *params, void **state)
{
        using namespace std::string_literals;
        using std::stoi;
        // Not required, but "correct" (see the SDK documentation)
        if (!NDIlib_initialize()) {
                LOG(LOG_LEVEL_ERROR) << "[NDI] Cannot initialize NDI!\n";
                return VIDCAP_INIT_FAIL;
        }
        auto s = new vidcap_state_ndi();
        if ((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) != 0u) {
                s->capture_audio = true;
        }

        const char *fmt = vidcap_params_get_fmt(params);
        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);
        char *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != nullptr) {
                if (strcmp(item, "help") == 0) {
                        show_help();
                        delete s;
                        return VIDCAP_INIT_NOERR;
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
                } else if (strstr(item, "color=") == item) {
                        s->requested_color = stoi(item + "color="s.length());
                } else {
                        LOG(LOG_LEVEL_ERROR) << "[NDI] Unknown option: " << item << "\n";
                        delete s;
                        return VIDCAP_INIT_NOERR;
                }

                tmp = nullptr;
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

        // Destroy the NDI finder.
        NDIlib_find_destroy(s->pNDI_find);

        NDIlib_recv_destroy(s->pNDI_recv);

        // Not required, but nice
        NDIlib_destroy();

        delete s;
}

static void audio_append(struct vidcap_state_ndi *s, NDIlib_audio_frame_v2_t *frame)
{
        struct audio_desc d{4, frame->sample_rate, static_cast<int>(audio_capture_channels > 0 ? audio_capture_channels : frame->no_channels), AC_PCM};
        if (!audio_desc_eq(d, audio_desc_from_audio_frame(&s->audio[s->audio_buf_idx]))) {
                free(s->audio[s->audio_buf_idx].data);
                s->audio[s->audio_buf_idx].bps = 4;
                s->audio[s->audio_buf_idx].sample_rate = frame->sample_rate;
                s->audio[s->audio_buf_idx].ch_count = d.ch_count;
                s->audio[s->audio_buf_idx].data_len = 0;
                s->audio[s->audio_buf_idx].max_size =
                        4 * d.ch_count * frame->sample_rate / 5; // 200 ms
                s->audio[s->audio_buf_idx].data = static_cast<char *>(malloc(s->audio[s->audio_buf_idx].max_size));
        }

        if (frame->no_channels > s->audio[s->audio_buf_idx].ch_count) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Requested " << s->audio[s->audio_buf_idx].ch_count << " channels, stream has only "
                        << frame->no_channels << "!\n";
        }

        for (int i = 0; i < frame->no_samples; ++i) {
                float *in = frame->p_data + i;
                int32_t *out = (int32_t *) s->audio[s->audio_buf_idx].data + i * d.ch_count;
                for (int j = 0; j < max(d.ch_count, frame->no_channels); ++j) {
                        if (s->audio[s->audio_buf_idx].data_len >= s->audio[s->audio_buf_idx].max_size) {
                                LOG(LOG_LEVEL_WARNING) << "[NDI] Audio frame too small!\n";
                                return;
                        }
                        *out++ = max<double>(INT32_MIN, min<double>(INT32_MAX, *in / s->audio_divisor * INT32_MAX));
                        in += frame->channel_stride_in_bytes / sizeof(float);
                        s->audio[s->audio_buf_idx].data_len += sizeof(int32_t);
                }
        }
}

static const NDIlib_source_t *get_matching_source(struct vidcap_state_ndi *s, const NDIlib_source_t *sources, int nr_sources) {
        assert(nr_sources > 0);
        if (s->requested_name.empty() && s->requested_url.empty()) {
                return sources + 0;
        }

        for (int i = 0; i < nr_sources; ++i) {
                if (!s->requested_name.empty()) {
                        if (s->requested_name != sources[i].p_ndi_name) {
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

using convert_t = void (*)(struct video_frame *, const uint8_t *);

static void convert_BGRA_RGBA(struct video_frame *out, const uint8_t *data)
{
        const auto *in_p = reinterpret_cast<const uint32_t *>(data);
        auto *out_p = reinterpret_cast<uint32_t *>(out->tiles[0].data);
        for (unsigned int i = 0; i < out->tiles[0].width * out->tiles[0].height; ++i) {
                uint32_t argb = *in_p++;
                *out_p++ = (argb & 0xFF000000U) | ((argb & 0xFFU) << 16U) | (argb & 0xFF00U) | ((argb & 0xFF0000U) >> 16U);
        }
}

static void convert_P216_Y216(struct video_frame *out, const uint8_t *data)
{
        const auto *in_y = reinterpret_cast<const uint16_t *>(data);
        const auto *in_cb_cr = reinterpret_cast<const uint16_t *>(data) + out->tiles[0].width * out->tiles[0].height;
        auto *out_p = reinterpret_cast<uint16_t *>(out->tiles[0].data);
        for (unsigned int i = 0; i < out->tiles[0].width; ++i) {
                for (unsigned int j = 0; j < out->tiles[0].height; ++j) {
                        *out_p++ = *in_y++;
                        *out_p++ = *in_cb_cr++;
                        *out_p++ = *in_y++;
                        *out_p++ = *in_cb_cr++;
                }
        }
}

static struct video_frame *vidcap_ndi_grab(void *state, struct audio_frame **audio)
{
        auto s = static_cast<struct vidcap_state_ndi *>(state);

        if (s->pNDI_find == nullptr) {
                // Create a finder
                s->pNDI_find = NDIlib_find_create_v2();
                if (s->pNDI_find == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "[NDI] Cannot create object!\n";
                        return nullptr;
                }
        }

        if (s->pNDI_recv == nullptr) {
                // Wait until there is one source
                uint32_t nr_sources = 0;
                const NDIlib_source_t* p_sources = nullptr;
                // Wait until the sources on the nwtork have changed
                LOG(LOG_LEVEL_INFO) << "[NDI] Looking for source(s)...\n";
                NDIlib_find_wait_for_sources(s->pNDI_find, 100 /* 100 ms */);
                p_sources = NDIlib_find_get_current_sources(s->pNDI_find, &nr_sources);
                if (nr_sources == 0) {
                        LOG(LOG_LEVEL_WARNING) << "[NDI] No sources.\n";
                        return nullptr;
                }

                const NDIlib_source_t *source = get_matching_source(s, p_sources, nr_sources);
                if (source == nullptr) {
                        return nullptr;
                }

                // We now have at least one source, so we create a receiver to look at it.
                NDIlib_recv_create_v3_t create_settings{};
                create_settings.color_format = s->requested_color == -1 ? NDIlib_recv_color_format_best : static_cast<NDIlib_recv_color_format_e>(s->requested_color);
                create_settings.allow_video_fields = false;
                s->pNDI_recv = NDIlib_recv_create_v3(&create_settings);
                if (s->pNDI_recv == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "[NDI] Unable to create receiver!\n";
                        return nullptr;
                }
                // Connect to our sources
                NDIlib_recv_connect(s->pNDI_recv, source);

                LOG(LOG_LEVEL_NOTICE) << "[NDI] Receiving from source: " << source->p_ndi_name << ", URL: " << source->p_url_address << "\n";
        }

        NDIlib_video_frame_v2_t video_frame;
        NDIlib_audio_frame_v2_t audio_frame;

        struct video_frame *out = nullptr;
        video_desc out_desc;

        switch (NDIlib_recv_capture_v2(s->pNDI_recv, &video_frame, &audio_frame, nullptr, 200))
        {       // No data
        case NDIlib_frame_type_none:
                cout << "No data received.\n";
                break;

                // Video data
        case NDIlib_frame_type_video:
        {
                convert_t convert = nullptr;
                if (video_frame.frame_format_type != NDIlib_frame_format_type_progressive && video_frame.frame_format_type != NDIlib_frame_format_type_interleaved) {
                        LOG(LOG_LEVEL_ERROR) << "[NDI] Unsupported interlacing, please report to " PACKAGE_BUGREPORT "!\n";
                }
                out_desc = {static_cast<unsigned int>(video_frame.xres),
                                static_cast<unsigned int>(video_frame.yres),
                                VIDEO_CODEC_NONE,
                                static_cast<double>(video_frame.frame_rate_N) / video_frame.frame_rate_D,
                                video_frame.frame_format_type == NDIlib_frame_format_type_progressive ? PROGRESSIVE : INTERLACED_MERGED,
                                1};
                switch (video_frame.FourCC) {
                        case NDIlib_FourCC_type_UYVA:
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
                                out_desc.color_spec = RGBA;
                                break;
                        default:
                        {
                                array<char, sizeof(uint32_t) + 1> fcc_s{};
                                memcpy(fcc_s.data(), &video_frame.FourCC, sizeof(uint32_t));
                                LOG(LOG_LEVEL_ERROR) << "[NDI] Unsupported codec '" << fcc_s.data() << "', please report to " PACKAGE_BUGREPORT "!\n";
                                return {};
                        }
                }
                if (s->last_desc != out_desc) {
                        LOG(LOG_LEVEL_NOTICE) << "[NDI] Received video changed: " << out_desc << "\n";
                        s->last_desc = out_desc;
                }
                if (convert != nullptr) {
                        out = vf_alloc_desc_data(out_desc);
                        assert(video_frame.line_stride_in_bytes == vc_get_linesize(video_frame.xres, out_desc.color_spec));
                        convert(out, video_frame.p_data);
                        NDIlib_recv_free_video_v2(s->pNDI_recv, &video_frame);
                        out->callbacks.dispose = vf_free;
                } else {
                        out = vf_alloc_desc(out_desc);
                        out->tiles[0].data = reinterpret_cast<char*>(video_frame.p_data);
                        struct dispose_udata_t {
                                NDIlib_video_frame_v2_t video_frame;
                                NDIlib_recv_instance_t pNDI_recv;
                        };
                        out->callbacks.dispose_udata = new dispose_udata_t{video_frame, s->pNDI_recv};
                        out->callbacks.dispose = [](struct video_frame *f) { auto du = static_cast<dispose_udata_t *>(f->callbacks.dispose_udata);
                                NDIlib_recv_free_video_v2(du->pNDI_recv, &du->video_frame);
                                delete du;
                        };
                }
                s->frames += 1;
                s->print_stats();
                return out;
                break;
        }
                // Audio data
        case NDIlib_frame_type_audio:
                if (s->capture_audio) {
                        audio_append(s, &audio_frame);
                        *audio = &s->audio[s->audio_buf_idx];
                        s->audio_buf_idx = (s->audio_buf_idx + 1) % 2;
                        s->audio[s->audio_buf_idx].data_len = 0;
                } else {
                        *audio = nullptr;
                }
                NDIlib_recv_free_audio_v2(s->pNDI_recv, &audio_frame);
                break;

        case NDIlib_frame_type_metadata:
                break;

        case NDIlib_frame_type_error:
                LOG(LOG_LEVEL_ERROR) << "[NDI] Error occured!\n";
                break;

        case NDIlib_frame_type_status_change:
                LOG(LOG_LEVEL_NOTICE) << "[NDI] Status changed!\n";
                break;
        case NDIlib_frame_type_max:
                assert(0 && "NDIlib_frame_type_max is invalid!");
        }

        return nullptr;
}

static struct vidcap_type *vidcap_ndi_probe(bool verbose, void (**deleter)(void *))
{
        *deleter = free;
        auto vt = static_cast<struct vidcap_type *>(calloc(1, sizeof(struct vidcap_type)));
        if (vt == nullptr) {
                return nullptr;
        }
        vt->name = "ndi";
        vt->description = "NDI source";

        if (!verbose) {
                return vt;
        }

        auto pNDI_find = NDIlib_find_create_v2();
        if (pNDI_find == nullptr) {
                LOG(LOG_LEVEL_ERROR) << "[NDI] Cannot create finder object!\n";
                return vt;
        }

        uint32_t nr_sources = 0;
        const NDIlib_source_t* p_sources = nullptr;
        // Give sources some time to occur
        usleep(100 * 1000);
        // we do not usea NDIlib_find_wait_for_sources() here because: 1) if there is
        // no source, it will still wait requested amount of time and 2) if there are
        // more sources, it will continue after first source found while there can be more
        p_sources = NDIlib_find_get_current_sources(pNDI_find, &nr_sources);

        vt->cards = (struct device_info *) calloc(nr_sources, sizeof(struct device_info));
        if (vt->cards == nullptr) {
                NDIlib_find_destroy(pNDI_find);
                return vt;
        }
        vt->card_count = nr_sources;
        for (int i = 0; i < static_cast<int>(nr_sources); ++i) {
                snprintf(vt->cards[i].id, sizeof vt->cards[i].id, "url=%s", p_sources[i].p_url_address);
                snprintf(vt->cards[i].name, sizeof vt->cards[i].name, "%s", p_sources[i].p_ndi_name);
                vt->cards[i].repeatable = true;
        }

        NDIlib_find_destroy(pNDI_find);

        return vt;
}

static const struct video_capture_info vidcap_ndi_info = {
        vidcap_ndi_probe,
        vidcap_ndi_init,
        vidcap_ndi_done,
        vidcap_ndi_grab,
        false
};

REGISTER_MODULE(ndi, &vidcap_ndi_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */


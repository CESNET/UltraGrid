/**
 * @file   video_capture/spout.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2021 CESNET z.s.p.o.
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
#endif

#include "gl_context.h" // it looks like it needs to be included prior to Spout.h

#include <array>
#include <cctype>
#include <chrono>
#include <iostream>
#include <SpoutLibrary.h>
#include <string>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "spout_sender.h" // spout_set_log_level
#include "utils/color_out.h"
#include "utils/text.h" // urlencode, urldecode
#include "video.h"
#include "video_capture.h"

#define DEFAULT_FPS 60.0
#define DEFAULT_CODEC RGB

static constexpr const char *MOD_NAME = "[Spout] ";

using std::array;
using std::cout;
using std::shared_ptr;
using std::stoi;
using std::string;

/**
 * Class state_vidcap_spout must be value-initialized
 */
struct state_vidcap_spout {
        struct video_desc desc;
        shared_ptr<SPOUTLIBRARY> spout_state;
        struct gl_context glc;
        GLenum gl_format;

        char server_name[256];

        std::chrono::steady_clock::time_point last_frame_captured = std::chrono::steady_clock::now();
        int frames;
};

static void usage()
{
        col() << "Usage:\n";
        col() << "\t" << TBOLD(TRED("-t spout") << "[:name=<server_name>>][:fps=<fps>][:codec=<codec>]") << "\n";
        col() << "where\n";
        col() << "\t" << TBOLD("name") << "\n\t\tSpout server name\n";
        col() << "\t" << TBOLD("fps") << "\n\t\tFPS count (default: " << DEFAULT_FPS << ")\n";
        col() << "\t" << TBOLD("codec") << "\n\t\tvideo codec (default: " << get_codec_name(DEFAULT_CODEC) << ")\n";
        col() << "\nServers:\n";
        auto receiver = shared_ptr<SPOUTLIBRARY>(GetSpout(), [](auto s) {s->Release();});
        int count = receiver->GetSenderCount();

        for (int i = 0; i < count; ++i) {
                array<char, 256> name{};
                if (!receiver->GetSender(i, name.data(), name.size())) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot get name for server #" << i << "\n";
                        continue;
                }
                unsigned int width = 0;
                unsigned int height = 0;
                HANDLE dxShareHandle = 0;
                DWORD dwFormat = 0;
                if (!receiver->GetSenderInfo(name.data(), width, height, dxShareHandle, dwFormat)) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot get server " << name.data() << "details\n";
                }
                col() << "\t" << TBOLD(<< name.data() <<) << ") width: " << width << ", height: " << height << "\n";
        }
}

static int vidcap_spout_init(struct vidcap_params *params, void **state)
{
        state_vidcap_spout *s = new state_vidcap_spout();

        double fps = DEFAULT_FPS;
        codec_t codec = DEFAULT_CODEC;

        char *opts = strdup(vidcap_params_get_fmt(params));
        char *item, *save_ptr;
        int ret = VIDCAP_INIT_OK;

        item = strtok_r(opts, ":", &save_ptr);
        while (item) {
                if (strcmp(item, "help") == 0) {
                        usage();
                        ret = VIDCAP_INIT_NOERR;
                        break;
                } else if (strstr(item, "name=") == item) {
                        char *name = item + strlen("name=");
                        if (strstr(name, "urlencoded=") == name) {
                                name += strlen("urlencoded=");
                                if (urldecode(s->server_name, sizeof s->server_name, name) == 0) {
                                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Improperly formatted name: " << item + strlen("name=") << "\n";
                                        ret = VIDCAP_INIT_FAIL;
                                }
                        } else {
                                strncpy(s->server_name, item + strlen("name="), sizeof(s->server_name) - 1);
                        }
                } else if (strstr(item, "fps=") == item) {
                        fps = atof(item + strlen("fps="));
                } else if (strstr(item, "codec=") == item) {
                        codec = get_codec_from_name(item + strlen("codec="));
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unknown argument - " << item << "\n";
                        ret = VIDCAP_INIT_FAIL;
                        break;
                }
                item = strtok_r(NULL, ":", &save_ptr);
        }

        free(opts);

        if (ret != VIDCAP_INIT_OK) {
                delete s;
                return ret;
        }

        switch (codec) {
        case RGB:
                s->gl_format = GL_RGB;
                break;
        case RGBA:
                s->gl_format = GL_RGBA;
                break;
        default:
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unsupported codec: " <<  get_codec_name(codec) << "! Currently only RGB and RGBA are supported.\n";
                delete s;
                return VIDCAP_INIT_FAIL;
        }


        if (!init_gl_context(&s->glc, GL_CONTEXT_ANY)) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to initialize GL context!\n";
                delete s;
                return VIDCAP_INIT_FAIL;
        }
        gl_context_make_current(&s->glc);

        s->spout_state = shared_ptr<SPOUTLIBRARY>(GetSpout(), [](auto s) {s->Release();});
        spout_set_log_level(s->spout_state.get(), log_level);
        if (strlen(s->server_name) != 0) {
                s->spout_state->SetReceiverName(s->server_name);
        }

        gl_context_make_current(NULL);
        s->desc = video_desc{0, 0, codec, fps, PROGRESSIVE, 1};

        *state = s;

        return VIDCAP_INIT_OK;
}

static void vidcap_spout_done(void *state)
{
        state_vidcap_spout *s = (state_vidcap_spout *) state;

        gl_context_make_current(&s->glc);
        s->spout_state->ReleaseReceiver();
        destroy_gl_context(&s->glc);

        delete s;
}

static struct video_frame *vidcap_spout_grab(void *state, struct audio_frame **audio)
{
        state_vidcap_spout *s = (state_vidcap_spout *) state;

        gl_context_make_current(&s->glc);
        struct video_frame *out = vf_alloc_desc_data(s->desc);
        out->callbacks.dispose = vf_free;

        bool ret = s->spout_state->ReceiveImage((unsigned char *) out->tiles[0].data, s->gl_format);
        if (!ret) {
                vf_free(out);
                gl_context_make_current(NULL);
                return NULL;
        }
        if (s->spout_state->IsUpdated()) {
                s->desc.width = s->spout_state->GetSenderWidth();
                s->desc.height = s->spout_state->GetSenderHeight();
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Connection updated - server name: " << s->spout_state->GetSenderName() << ", width: " << s->desc.width << ", height: " << s->desc.height << ", fps: " << s->desc.fps << ", codec: " << get_codec_name_long(s->desc.color_spec) << "\n";
                vf_free(out);
                gl_context_make_current(NULL);
                return NULL;
        }
        gl_context_make_current(NULL);

	// wait until this frame is due
	decltype(s->last_frame_captured) t;
	do {
		t = std::chrono::steady_clock::now();
	} while (std::chrono::duration_cast<std::chrono::duration<double>>(t - s->last_frame_captured).count() < 1 / out->fps);
	s->last_frame_captured = t;

        *audio = NULL;
        return out;
}

static struct vidcap_type *vidcap_spout_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        *deleter = free;
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "spout";
                vt->description = "Spout capture client";
        }

        if (!verbose) {
                return vt;
        }

        auto receiver = shared_ptr<SPOUTLIBRARY>(GetSpout(), [](auto s) {s->Release();});

        int count = receiver->GetSenderCount();

        vt->cards = (struct device_info *) calloc(count, sizeof(struct device_info));
        if (vt->cards == nullptr) {
                return vt;
        }
        vt->card_count = count;

        for (int i = 0; i < count; ++i) {
                array<char, 256> name{};
                if (!receiver->GetSender(i, name.data(), name.size())) {
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Cannot get name for server #" << i << "\n";
                        snprintf(vt->cards[i].dev, sizeof vt->cards[i].dev, ":device=%d", i);
                        snprintf(vt->cards[i].name, sizeof vt->cards[i].name, "Spout #%d", i);
                } else {
                        snprintf(vt->cards[i].dev, sizeof vt->cards[i].dev, ":name=urlencoded=");
                        urlencode(vt->cards[i].dev + strlen(vt->cards[i].dev),
                                        sizeof vt->cards[i].dev - strlen(vt->cards[i].dev),
                                        name.data(), urlencode_rfc3986_eval, false);
                        snprintf(vt->cards[i].name, sizeof vt->cards[i].name, "Spout %s", name.data());
                }
                vt->cards[i].repeatable = true;
        }
        return vt;
}

static const struct video_capture_info vidcap_spout_info = {
        vidcap_spout_probe,
        vidcap_spout_init,
        vidcap_spout_done,
        vidcap_spout_grab,
        MOD_NAME,
};

REGISTER_MODULE(spout, &vidcap_spout_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


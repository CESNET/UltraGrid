/**
 * @file audio/playback/dump.c
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include <string>

#include "debug.h"
#include "audio/export.h"
#include "audio/audio_playback.h"
#include "audio/types.h"
#include "lib_common.h"

namespace{
        struct Export_state_deleter{
                void operator()(struct audio_export* e) { audio_export_destroy(e); }
        };
}

struct audio_dump_state{
        std::unique_ptr<struct audio_export, Export_state_deleter> exporter;
        struct audio_desc desc;

        std::string filename;
        unsigned file_name_num;
};

static void usage() {
        printf("dump usage:\n"
                        "\t-r dump[:<path>]\n"
                        "where\n"
                        "\tpath - path prefix to use (without .wav extension)\n"
                        "\n");
}

static void * audio_play_dump_init(const char *cfg){
        struct audio_dump_state *s = new audio_dump_state();

        if (cfg != nullptr) {
                if (strcmp(cfg, "help") == 0) {
                        usage();
                        delete s;
                        return nullptr;
                }
                s->filename = cfg;
        } else {
                s->filename = "audio_dump";
        }

        return s;
}

static void audio_play_dump_done(void *state){
        auto *s = static_cast<audio_dump_state *>(state);

        delete s;
}

static void audio_play_dump_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = NULL;
        *count = 0;
}

static void audio_play_dump_help(const char *)
{
        printf("\tdump: dump audio\n");
}

static bool query_fmt(void *data, size_t *len){
        struct audio_desc desc;
        if (*len < sizeof desc) {
                return false;
        } else {
                memcpy(&desc, data, sizeof desc);
        }

        desc.codec = AC_PCM;
        memcpy(data, &desc, sizeof desc);

        return true;
}

static bool audio_play_dump_ctl(void *, int request, void *data, size_t *len)
{
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return query_fmt(data, len);
        default:
                return false;
        }
}

static void audio_play_dump_put_frame(void *state, const struct audio_frame *f)
{
        auto *s = static_cast<audio_dump_state *>(state);

        audio_export(s->exporter.get(), f);
}

static int audio_play_dump_reconfigure(void *state, struct audio_desc new_desc)
{
        auto *s = static_cast<audio_dump_state *>(state);
        s->desc = new_desc;

        std::string filename = s->filename;
        if(s->file_name_num){
                filename += "_" + std::to_string(s->file_name_num);
        }
        filename += ".wav";
        s->file_name_num++;
        s->exporter.reset(audio_export_init(filename.c_str()));

        return true;
}

static const struct audio_playback_info aplay_dump_info = {
        audio_play_dump_probe,
        audio_play_dump_help,
        audio_play_dump_init,
        audio_play_dump_put_frame,
        audio_play_dump_ctl,
        audio_play_dump_reconfigure,
        audio_play_dump_done
};

REGISTER_MODULE(dump, &aplay_dump_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

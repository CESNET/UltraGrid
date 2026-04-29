// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file   audio/capture/wav.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */

#include <assert.h> // for assert
#include <errno.h>  // for errno
#include <stdint.h> // for uint32_t
#include <stdio.h>  // for FILE, SEEK_SET, fclose, feof, fopen
#include <stdlib.h> // for free, calloc, malloc
#include <string.h> // for strchr, strcmp
#include <time.h>   // for nanosleep, timespec

#include "audio/audio_capture.h" // for AUDIO_CAPTURE_ABI_VERSION, audio_ca...
#include "audio/types.h"         // for audio_frame
#include "audio/wav_reader.h"    // for wav_metadata, WAV_HDR_PARSE_OK, get...
#include "compat/c23.h"          // IWYU pragma: keep
#include "debug.h"               // for MSG, LOG_LEVEL_ERROR, LOG_LEVEL_WAR...
#include "host.h"                // for INIT_NOERR
#include "lib_common.h"          // for REGISTER_MODULE, library_class
#include "tv.h"                  // for NS_IN_SEC, get_time_in_ns, time_ns_t
#include "utils/color_out.h"     // for TBOLD, color_printf, TRED
#include "utils/macros.h"        // for to_fourcc, IS_KEY_PREFIX
#include "utils/misc.h"          // for ug_strerror

struct device_info;
struct module;

#define AUDIO_CAPTURE_WAV_MAGIC to_fourcc('A', 'C', 'w', 'a')
#define CHUNK_SIZE              512
#define MOD_NAME                "[acap/wav] "

struct state_audio_capture_wav {
        uint32_t            magic;
        FILE               *wav_file;
        struct wav_metadata wav_metadata;
        struct audio_frame  audio_frame;
        time_ns_t           next_grab_time;
};

static void audio_cap_wav_done(void *state);

static void
audio_cap_wav_probe(struct device_info **available_devices, int *count,
                    void (**deleter)(void *))
{
        *deleter           = free;
        *available_devices = nullptr;
        *count             = 0;
}

static void
usage()
{
        color_printf("Audio capture " TBOLD("wav")
                     " is capable to play a WAV file.\n\n");
        color_printf("Usage:\n\t" TBOLD(TRED("wav") ":file=<filename>")
                     "\n\n");
        color_printf("Input WAV file is looped infinitely.\n\n");
}

static const char *
parse_fmt(const char *fmt)
{
        if (!IS_KEY_PREFIX(fmt, "file")) {
                usage();
                return nullptr;
        }
        return strchr(fmt, '=') + 1;
}

static void *
audio_cap_wav_init(struct module * /*parent */, const char *cfg)
{
        if (strcmp(cfg, "help") == 0) {
                usage();
                return INIT_NOERR;
        }
        const char *filename = parse_fmt(cfg);
        if (filename == nullptr) {
                return nullptr;
        }
        struct state_audio_capture_wav *s = calloc(1, sizeof *s);
        assert(s != nullptr);
        s->magic    = AUDIO_CAPTURE_WAV_MAGIC;
        s->wav_file = fopen(filename, "rb");
        if (s->wav_file == nullptr) {
                MSG(ERROR, "Cannot open input file %s: %s\n", filename,
                    ug_strerror(errno));
                audio_cap_wav_done(s);
                return nullptr;
        }
        int ret = read_wav_header(s->wav_file, &s->wav_metadata);
        if (ret != WAV_HDR_PARSE_OK) {
                MSG(ERROR, "%s!\n", get_wav_error(ret));
                audio_cap_wav_done(s);
                return nullptr;
        }

        s->audio_frame.ch_count    = s->wav_metadata.ch_count;
        s->audio_frame.sample_rate = s->wav_metadata.sample_rate;
        s->audio_frame.bps         = s->wav_metadata.bits_per_sample / 8;
        s->audio_frame.max_size =
            s->audio_frame.bps * s->audio_frame.ch_count * CHUNK_SIZE;
        s->audio_frame.data = malloc(s->audio_frame.max_size);

        s->next_grab_time = get_time_in_ns();

        return s;
}

static const struct audio_frame *
audio_cap_wav_read(void *state)
{
        struct state_audio_capture_wav *s = state;

        time_ns_t curr_time = get_time_in_ns();
        if (s->next_grab_time > curr_time) {
                nanosleep(&(struct timespec){ .tv_nsec = s->next_grab_time -
                                                         curr_time },
                          nullptr);
        } else {
                // we missed more than 2 "frame times"
                if ((curr_time - s->next_grab_time) >
                    (long long int) (2 * NS_IN_SEC * CHUNK_SIZE /
                                     s->audio_frame.sample_rate)) {
                        s->next_grab_time = curr_time;
                        MSG(WARNING, "Warning: late grab call!\n");
                }
        }
        s->next_grab_time +=
            NS_IN_SEC * CHUNK_SIZE / s->audio_frame.sample_rate;

        unsigned frame_size  = s->audio_frame.bps * s->audio_frame.ch_count;
        size_t   frames_read = wav_read(s->audio_frame.data, CHUNK_SIZE,
                                        s->wav_file, &s->wav_metadata);
        if (frames_read == 0) {
                if (feof(s->wav_file)) {
                        MSG(INFO, "File ended, rewinding...\n");
                        wav_seek(s->wav_file, 0, SEEK_SET, &s->wav_metadata);
                }
                return nullptr;
        }
        s->audio_frame.data_len = frames_read * frame_size;

        return &s->audio_frame;
}

static void
audio_cap_wav_done(void *state)
{
        struct state_audio_capture_wav *s = state;
        assert(s->magic == AUDIO_CAPTURE_WAV_MAGIC);

        if (s->wav_file != nullptr) {
                fclose(s->wav_file);
        }
        free(s);
}

static const struct audio_capture_info acap_wav_info = {
        .probe = audio_cap_wav_probe,
        .init  = audio_cap_wav_init,
        .read  = audio_cap_wav_read,
        .done  = audio_cap_wav_done
};

REGISTER_MODULE(wav, &acap_wav_info, LIBRARY_CLASS_AUDIO_CAPTURE,
                AUDIO_CAPTURE_ABI_VERSION);

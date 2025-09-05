/**
 * @file   audio/capture/fluidsynth.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET
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

#include <assert.h>               // for assert
#include <fluidsynth.h>           // for fluid_player_play, fluid_synth_writ...
#include <fluidsynth/types.h>     // for fluid_player_t, fluid_settings_t
#include <stdbool.h>              // for bool
#include <stdio.h>                // for NULL, fclose, size_t, FILE, fopen
#include <stdlib.h>               // for free, getenv, malloc, calloc
#include <string.h>               // for strdup, strlen, strncat, strcmp
#include <unistd.h>               // for unlink

#include "audio/audio_capture.h"  // for AUDIO_CAPTURE_ABI_VERSION, audio_ca...
#include "audio/types.h"          // for audio_frame
#include "audio/utils.h"          // for mux_channel
#include "compat/usleep.h"        // for usleep
#include "debug.h"                // for LOG_LEVEL_ERROR, MSG, log_msg, LOG_...
#include "host.h"                 // for audio_capture_sample_rate, INIT_NOERR
#include "lib_common.h"           // for REGISTER_MODULE, library_class
#include "song1.h"                // for song1
#include "tv.h"                   // for get_time_in_ns, time_ns_t, NS_IN_SE...
#include "types.h"                // for device_info
#include "utils/color_out.h"      // for color_printf, TBOLD, TRED
#include "utils/fs.h"             // for get_install_root, get_temp_file
#include "utils/macros.h"         // for ARR_COUNT, IS_KEY_PREFIX

struct module;

enum {
        FLUIDSYNTH_BPS                 = 2,
        DEFAULT_FLUIDSYNTH_SAMPLE_RATE = 48000,
        CHUNK_SIZE                     = 480,
};

#define MOD_NAME "[fluidsynth] "

struct state_fluidsynth_capture {
        struct audio_frame audio;
        unsigned char     *left;
        unsigned char     *right;

        char       *req_filename;
        const char *tmp_filename;

        time_ns_t next_frame_time;
        time_ns_t frame_interval;
        ;

        fluid_settings_t *settings;
        fluid_synth_t    *synth;
        fluid_player_t   *player;
};

static void audio_cap_fluidsynth_done(void *state);

static void
audio_cap_fluidsynth_probe(struct device_info **available_devices, int *count,
                           void (**deleter)(void *))
{
        *deleter           = free;
        *count             = 1;
        *available_devices = calloc(1, sizeof **available_devices);
        strncat((*available_devices)[0].dev, "fluidsynth",
                sizeof(*available_devices)[0].dev - 1);
        strncat((*available_devices)[0].name, "Sample midi song",
                sizeof(*available_devices)[0].name - 1);
}

static void
usage()
{
        color_printf(
            TBOLD("fluidsynth") " is a capture device capable playing MIDI.\n\n"
                                "The main functional difference to " TBOLD(
                                    "file") " video capture (that is able to "
                                            "play audio\n"
                                            "files as well) is the support "
                                            "for " TBOLD(
                                                "MIDI") " (and also having one "
                                                        "song bundled).\n\n");
        color_printf("Usage:\n");
        color_printf(TBOLD(TRED("\t-s fluidsynth") "[:file=<filename>]") "\n");
        color_printf("where\n");
        color_printf(TBOLD("\t<filename>") " - name of file to be used\n");
        color_printf("\n");
        color_printf(TBOLD(
            "FLUIDSYNTH_SF") "        - environment variable with path to "
                             "sound fonts for MIDI playback (eg. freepats)\n");
        color_printf(
            TBOLD("ULTRAGRID_BUNDLED_SF") " - set this environment variable to "
                                          "1 to skip loading system default "
                                          "sound font\n\n");
}

static int
parse_opts(struct state_fluidsynth_capture *s, char *cfg)
{
        char *save_ptr = NULL;
        char *item     = NULL;
        while ((item = strtok_r(cfg, ":", &save_ptr)) != NULL) {
                cfg = NULL;
                if (strcmp(item, "help") == 0) {
                        usage();
                        return 1;
                }
                if (IS_KEY_PREFIX(item, "file")) {
                        s->req_filename = strdup(strchr(item, '=') + 1);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong option: %s!\n",
                                item);
                        color_printf("Use " TBOLD(
                            "-s fluidsynth:help") " to see available "
                                                  "options.\n");
                        return -1;
                }
        }
        return 0;
}

static const char *
load_song1()
{
        const char *filename = NULL;
        FILE       *f        = get_temp_file(&filename);
        if (f == NULL) {
                perror("fopen audio");
                return NULL;
        }
        size_t nwritten = fwrite(song1, sizeof song1, 1, f);
        fclose(f);
        if (nwritten != 1) {
                unlink(filename);
                return NULL;
        }
        return filename;
}

/**
 * Try to preload a sound font.
 *
 * This is mainly intended to allow loading sound fonts from application bundle
 * on various platforms (get_install_root is relative to executable). But use
 * to system default font, if available.
 */
static char *
get_soundfont()
{
        const char *env_fs = getenv("FLUIDSYNTH_SF");
        if (env_fs != NULL) {
                return strdup(env_fs);
        }
        char bundled[MAX_PATH_SIZE];
        snprintf_ch(bundled, "%s/%s", get_ug_data_path(),
                    "TimGM6mb_but_fixed__piano_.sf3");
        const char *sf_candidates[] = {
                "/usr/share/soundfonts/default.sf2",
                "/usr/share/soundfonts/default.sf3",
                "/usr/share/sounds/sf2/default-GM.sf2",
                "/usr/share/sounds/sf3/default-GM.sf3", // Ubuntu
                bundled,
        };

        const char *force_bundled_sf = getenv("ULTRAGRID_BUNDLED_SF");
        if (force_bundled_sf != NULL && strcmp(force_bundled_sf, "1") == 0) {
                for (size_t i = ARR_COUNT(sf_candidates) - 1; i > 0; --i) {
                        sf_candidates[i] = sf_candidates[i - 1];
                }
                sf_candidates[0] = bundled;
        }

        for (size_t i = 0; i < ARR_COUNT(sf_candidates); ++i) {
                const char *path = sf_candidates[i];
                FILE *f = fopen(path, "rb");
                debug_msg(MOD_NAME
                          "Trying to open sound font '%s': %s\n",
                          path, f ? "success, setting" : "failed");
                if (!f) {
                        continue;
                }
                fclose(f);
                return strdup(path);
        }
        MSG(ERROR, "Cannot find any suitable sound font!\n");
        return NULL;
}

static void *
audio_cap_fluidsynth_init(struct module *parent, const char *cfg)
{
        (void) parent;
        struct state_fluidsynth_capture *s    = calloc(1, sizeof *s);
        char                            *ccfg = strdup(cfg);
        int                              ret  = parse_opts(s, ccfg);
        free(ccfg);
        if (ret != 0) {
                audio_cap_fluidsynth_done(s);
                return ret < 0 ? NULL : INIT_NOERR;
        }

        char *sf = get_soundfont();
        if (sf == NULL) {
                audio_cap_fluidsynth_done(s);
                return NULL;
        }

        /// @todo add other if some-one needs that...
        if (audio_capture_bps != 0 && audio_capture_bps != FLUIDSYNTH_BPS) {
                MSG(ERROR, "Only %d bits-per-second supported so far...\n", FLUIDSYNTH_BPS);
                goto error;
        }
        if (audio_capture_channels > 2) {
                MSG(ERROR, "Only 1 or 2 channels currently supported...\n");
                goto error;
        }

        s->audio.bps         = FLUIDSYNTH_BPS;
        s->audio.ch_count    = audio_capture_channels < 2 ? 1 : 2;
        s->audio.sample_rate = audio_capture_sample_rate > 0
                                   ? audio_capture_sample_rate
                                   : DEFAULT_FLUIDSYNTH_SAMPLE_RATE;

        const char *filename = s->req_filename;
        if (!filename) {
                filename = s->tmp_filename = load_song1();
                if (!filename) {
                        goto error;
                }
        }

        s->settings = new_fluid_settings();
        fluid_settings_setnum(s->settings, "synth.sample-rate",
                              s->audio.sample_rate);

        s->synth = new_fluid_synth(s->settings);
        if (fluid_synth_sfload(s->synth, sf, 1) < 0) {
                MSG(ERROR, "Failed to load SF2: %s\n", sf);
                goto error;
        }
        s->player = new_fluid_player(s->synth);
        if (fluid_player_add(s->player, filename) != FLUID_OK) {
                MSG(ERROR, "Failed to add MIDI: %s\n", s->req_filename);
                goto error;
        }
        fluid_player_play(s->player);

        s->audio.max_size = s->audio.data_len =
            s->audio.ch_count * s->audio.bps * CHUNK_SIZE;
        s->audio.data = malloc(s->audio.data_len);
        s->left       = malloc(s->audio.data_len / s->audio.ch_count);
        s->right      = malloc(s->audio.data_len / s->audio.ch_count);

        s->frame_interval  = CHUNK_SIZE * NS_IN_SEC_DBL / s->audio.sample_rate;
        s->next_frame_time = get_time_in_ns() + s->frame_interval;

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Initialized fluidsynth\n");

        free(sf);
        return s;
error:
        audio_cap_fluidsynth_done(s);
        free(sf);
        return NULL;
}

static struct audio_frame *
audio_cap_fluidsynth_read(void *state)
{
        struct state_fluidsynth_capture *s = state;

        if (fluid_player_get_status(s->player) == FLUID_PLAYER_DONE) {
                MSG(VERBOSE, "Rewinding...\n");
                fluid_player_play(s->player);
        }

        if (s->audio.ch_count == 1) {
                fluid_synth_write_s16(s->synth, CHUNK_SIZE, s->audio.data, 0, 1,
                                      s->right, 0, 1);
        } else { // drop right channel, keep the left
                assert(s->audio.ch_count == 2);
                fluid_synth_write_s16(s->synth, CHUNK_SIZE, s->left, 0, 1,
                                      s->right, 0, 1);
                mux_channel(s->audio.data, (char *) s->left, s->audio.bps,
                            s->audio.bps * CHUNK_SIZE, s->audio.ch_count, 0,
                            1.);
                mux_channel(s->audio.data, (char *) s->right, s->audio.bps,
                            s->audio.bps * CHUNK_SIZE, s->audio.ch_count, 1,
                            1.);
        }

        time_ns_t t = get_time_in_ns();
        if (t > s->next_frame_time + s->frame_interval) {
                MSG(WARNING, "Some data missed!\n");
                t = s->next_frame_time;
        } else if (t < s->next_frame_time){
                usleep((s->next_frame_time - t) / US_IN_NS);
        }
        s->next_frame_time += s->frame_interval;

        return &s->audio;
}

static void
audio_cap_fluidsynth_done(void *state)
{
        struct state_fluidsynth_capture *s = state;
        free(s->audio.data);
        free(s->req_filename);
        free(s->left);
        free(s->right);
        if (s->tmp_filename) {
                unlink(s->tmp_filename);
        }
        free(s);
}

static const struct audio_capture_info acap_fluidsynth_info = {
        audio_cap_fluidsynth_probe, audio_cap_fluidsynth_init,
        audio_cap_fluidsynth_read, audio_cap_fluidsynth_done
};

REGISTER_MODULE(fluidsynth, &acap_fluidsynth_info, LIBRARY_CLASS_AUDIO_CAPTURE,
                AUDIO_CAPTURE_ABI_VERSION);
REGISTER_MODULE_WITH_FLAG(sdl_mixer, &acap_fluidsynth_info, LIBRARY_CLASS_AUDIO_CAPTURE,
                AUDIO_CAPTURE_ABI_VERSION, MODULE_FLAG_ALIAS);

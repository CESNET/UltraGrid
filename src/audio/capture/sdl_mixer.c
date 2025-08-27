/**
 * @file   audio/capture/sdl_mixer.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2025 CESNET
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
 * @todo errata (SDL3 vs SDL2)
 * 1. 1 channel capture (-a ch=1) seem no longer work but there is a workaround
 * 2. insufficient performance (generates overflow even in default config)
 */

#include "config.h"               // for HAVE_SDL3

#ifdef HAVE_SDL3
#include <SDL3/SDL.h>             // for SDL_Init, SDL_INIT_AUDIO
#include <SDL3/SDL_audio.h>       // for AUDIO_S16LSB, AUDIO_S32LSB, AUDIO_S8
#include <SDL3_mixer/SDL_mixer.h> // for MIX_MAX_VOLUME, Mix_GetError, Mix_C...
#else
#include <SDL.h>                  // for SDL_Init, SDL_INIT_AUDIO
#include <SDL_audio.h>            // for AUDIO_S16LSB, AUDIO_S32LSB, AUDIO_S8
#include <SDL_mixer.h>            // for MIX_MAX_VOLUME, Mix_GetError, Mix_C...
#endif
#include <stdio.h>                // for NULL, fclose, fopen, size_t, FILE
#include <stdlib.h>               // for free, calloc, getenv, atoi, malloc
#include <string.h>               // for strlen, strncat, strchr, strcmp
#include <unistd.h>               // for unlink

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "song1.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/fs.h"
#include "utils/macros.h"
#include "utils/ring_buffer.h"

#define DEFAULT_SDL_MIXER_BPS 2
#define DEFAULT_MIX_MAX_VOLUME (MIX_MAX_VOLUME / 4)
#define SDL_MIXER_SAMPLE_RATE 48000
#define MOD_NAME "[SDL_mixer] "

#ifdef HAVE_SDL3
#define Mix_GetError SDL_GetError
#define SDL_ERR false
#else
#define SDL_AUDIO_S8 AUDIO_S8
#define SDL_AUDIO_S16LE AUDIO_S16LSB
#define SDL_AUDIO_S32LE AUDIO_S32LSB
#define SDL_ERR (-1)
#endif

struct state_sdl_mixer_capture {
        Mix_Music *music;
        struct audio_frame audio;
        struct ring_buffer *sdl_mixer_buf;
        int volume;
        char *req_filename;
};

static void audio_cap_sdl_mixer_done(void *state);

static void audio_cap_sdl_mixer_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_devices = calloc(1, sizeof **available_devices);
        strncat((*available_devices)[0].dev, "sdl_mixer", sizeof (*available_devices)[0].dev - 1);
        strncat((*available_devices)[0].name, "Sample midi song", sizeof (*available_devices)[0].name - 1);
}

static void sdl_mixer_audio_callback(int chan, void *stream, int len, void *udata)
{
        UNUSED(chan);
        struct state_sdl_mixer_capture *s = udata;

        ring_buffer_write(s->sdl_mixer_buf, stream, len);
        memset(stream, 0, len); // do not playback anything to PC output
}

static int parse_opts(struct state_sdl_mixer_capture *s, char *cfg) {
        char *save_ptr = NULL;
        char *item = NULL;
        while ((item = strtok_r(cfg, ":", &save_ptr)) != NULL) {
                cfg = NULL;
                if (strcmp(item, "help") == 0) {
                        color_printf(TBOLD("sdl_mixer") " is a capture device capable playing various audio files like FLAC,\n"
                                        "MIDI, mp3, Vorbis or WAV.\n\n"
                                        "The main functional difference to " TBOLD("file") " video capture (that is able to play audio\n"
                                        "files as well) is the support for " TBOLD("MIDI") " (and also having one song bundled).\n\n");
                        color_printf("Usage:\n");
                        color_printf(TBOLD(TRED("\t-s sdl_mixer") "[:file=<filename>][:volume=<vol>]") "\n");
                        color_printf("where\n");
                        color_printf(TBOLD("\t<filename>") " - name of file to be used\n");
                        color_printf(TBOLD("\t<vol>     ") " - volume [0..%d], default %d\n", MIX_MAX_VOLUME, DEFAULT_MIX_MAX_VOLUME);
                        color_printf("\n");
                        color_printf(TBOLD("SDL_SOUNDFONTS") "       - environment variable with path to sound fonts for MIDI playback (eg. freepats)\n");
                        color_printf(TBOLD("ULTRAGRID_BUNDLED_SF") " - set this environment variable to 1 to skip loading system default sound font\n\n");
                        return 1;
                }
                if (strstr(item, "file=") == item) {
                        s->req_filename = strdup(strchr(item, '=') + 1);
                } else if (strstr(item, "volume=") == item) {
                        s->volume = atoi(strchr(item, '=') + 1);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong option: %s!\n", item);
                        color_printf("Use " TBOLD("-s sdl_mixer:help") " to see available options.\n");
                        return -1;
                }
        }
        return 0;
}

static const char *load_song1() {
        const char *filename = NULL;
        FILE *f = get_temp_file(&filename);
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
static void try_open_soundfont() {
        const _Bool force_bundled_sf = getenv("ULTRAGRID_BUNDLED_SF") != NULL && strcmp(getenv("ULTRAGRID_BUNDLED_SF"), "1") == 0;
        if (!force_bundled_sf) {
                const char *default_soundfont = Mix_GetSoundFonts();
                if (default_soundfont) {
                        FILE *f = fopen(default_soundfont, "rb");
                        if (f) {
                                debug_msg(MOD_NAME "Default sound font '%s' seems usable, not trying to load additional fonts.\n", default_soundfont);
                                fclose(f);
                                return;
                        }
                }
                debug_msg(MOD_NAME "Unable to open default sound font '%s'\n", default_soundfont ? default_soundfont : "(no font)");
        }
        const char *roots[] = { get_install_root(), "/usr" };
        const char *sf_candidates[] = { // without install prefix
                "/share/soundfonts/default.sf2", "/share/soundfonts/default.sf3",
                "/share/sounds/sf2/default-GM.sf2", "/share/sounds/sf3/default-GM.sf3", // Ubuntu
        };
        for (size_t i = 0; i < sizeof roots / sizeof roots[0]; ++i) {
                for (size_t j = 0; j < sizeof sf_candidates / sizeof sf_candidates[0]; ++j) {
                        const char *root = roots[i];
                        const size_t len = strlen(root) + strlen(sf_candidates[j]) + 1;
                        char path[len];
                        strncpy(path, root, len - 1);
                        strncat(path, sf_candidates[j], len - strlen(path) - 1);
                        FILE *f = fopen(path, "rb");
                        debug_msg(MOD_NAME "Trying to open sound font '%s': %s\n", path, f ? "success, setting" : "failed");
                        if (!f) {
                                continue;
                        }
                        fclose(f);
                        Mix_SetSoundFonts(path);
                        return;
                }
        }
}

/// handle SDL 3.0.0 mixer not being able to capture mono
static void
adjust_ch_count(struct state_sdl_mixer_capture *s)
{
        int             frequency = 0;
        SDL_AudioFormat format = { 0 };
        int             channels = 0;
        Mix_QuerySpec(&frequency, &format, &channels);
        if (audio_capture_channels > 0 &&
            channels != (int) audio_capture_channels) {
                MSG(WARNING,
                    "%d channel capture seem to be broken with SDL3 mixer - "
                    "capturing %d channels and dropping the excessive later.\n",
                    s->audio.ch_count, channels);
                s->audio.ch_count = channels;
        }
}

static void * audio_cap_sdl_mixer_init(struct module *parent, const char *cfg)
{
        MSG(WARNING, "SDL_mixer is deprecated, used fluidsynth...\n");
        UNUSED(parent);
        SDL_Init(SDL_INIT_AUDIO);

        struct state_sdl_mixer_capture *s = calloc(1, sizeof *s);
        s->volume = DEFAULT_MIX_MAX_VOLUME;
        char *ccfg = strdup(cfg);
        int ret = parse_opts(s, ccfg);
        free(ccfg);
        if (ret != 0) {
                audio_cap_sdl_mixer_done(s);
                return ret < 0 ? NULL : INIT_NOERR;
        }

        s->audio.bps = audio_capture_bps ? audio_capture_bps : DEFAULT_SDL_MIXER_BPS;
        s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels
                                                       : MIX_DEFAULT_CHANNELS;
        s->audio.sample_rate = SDL_MIXER_SAMPLE_RATE;

        int audio_format = 0;
        switch (s->audio.bps) {
                case 1: audio_format = SDL_AUDIO_S8; break;
                case 2: audio_format = SDL_AUDIO_S16LE; break;
                case 4: audio_format = SDL_AUDIO_S32LE; break;
                default: UG_ASSERT(0 && "BPS can be only 1, 2 or 4");
        }

#ifdef HAVE_SDL3
        SDL_AudioSpec spec = {
                .format   = audio_format,
                .channels = s->audio.ch_count,
                .freq     = s->audio.sample_rate,
        };
        if (!Mix_OpenAudio(0, &spec)) {
#else
        if( Mix_OpenAudio(SDL_MIXER_SAMPLE_RATE, audio_format,
                                s->audio.ch_count, 4096 ) == -1 ) {
#endif
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error initializing sound: %s\n", Mix_GetError());
                goto error;
        }
        adjust_ch_count(s);
        const char *filename = s->req_filename;
        if (!filename) {
                filename = load_song1();
                if (!filename) {
                        goto error;
                }
        }
        try_open_soundfont();
        s->music = Mix_LoadMUS(filename);
        if (filename != s->req_filename) {
                unlink(filename);
        }
        if (s->music == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error loading file: %s\n", Mix_GetError());
                goto error;
        }

        s->audio.max_size =
                s->audio.data_len = s->audio.ch_count * s->audio.bps * s->audio.sample_rate /* 1 sec */;
        s->audio.data = malloc(s->audio.data_len);
        s->sdl_mixer_buf = ring_buffer_init(s->audio.data_len);

        // register grab as a postmix processor
        if (!Mix_RegisterEffect(MIX_CHANNEL_POST, sdl_mixer_audio_callback, NULL, s)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Mix_RegisterEffect: %s\n", Mix_GetError());
                goto error;
        }

        Mix_VolumeMusic(s->volume);
        if (Mix_PlayMusic(s->music, -1) == SDL_ERR) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "error playing file: %s\n", Mix_GetError());
                goto error;
        }

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Initialized SDL_mixer\n");

        return s;
error:
        audio_cap_sdl_mixer_done(s);
        return NULL;
}

static struct audio_frame *audio_cap_sdl_mixer_read(void *state)
{
        struct state_sdl_mixer_capture *s = state;
        s->audio.data_len = ring_buffer_read(s->sdl_mixer_buf, s->audio.data, s->audio.max_size);
        if (s->audio.data_len == 0) {
                return NULL;
        }
        return &s->audio;
}

static void audio_cap_sdl_mixer_done(void *state)
{
        struct state_sdl_mixer_capture *s = state;
        Mix_HaltMusic();
        Mix_FreeMusic(s->music);
        Mix_CloseAudio();
        free(s->audio.data);
        free(s->req_filename);
        free(s);
}

static const struct audio_capture_info acap_sdl_mixer_info = {
        audio_cap_sdl_mixer_probe,
        audio_cap_sdl_mixer_init,
        audio_cap_sdl_mixer_read,
        audio_cap_sdl_mixer_done
};

REGISTER_MODULE(sdl_mixer, &acap_sdl_mixer_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);


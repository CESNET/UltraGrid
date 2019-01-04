/**
 * @file   audio/capture/alsa.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2016 CESNET, z. s. p. o.
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
 * @note
 * In code below, state_alsa_capture::frame::ch_count can differ from
 * state_alsa_capture::min_device_channels, if device doesn't support
 * mono and we want to capture it. In this case state_alsa_capture::frame::ch_count
 * is set to 1 and min_device_channels to minimal number of channels
 * that device supports.
 */

/* Use the newer ALSA API */
#define ALSA_PCM_NEW_HW_PARAMS_API

#include "config.h"
#include "config_unix.h"

#include "host.h"

#ifdef HAVE_ALSA

#include "alsa_common.h"
#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/utils.h"

#include "debug.h"
#include "lib_common.h"
#include "tv.h"
#include <stdlib.h>
#include <string.h>

#include <alsa/asoundlib.h>

#define MOD_NAME "[ALSA cap.] "

struct state_alsa_capture {
        snd_pcm_t *handle;
        struct audio_frame frame;
        char *tmp_data;

        snd_pcm_uframes_t frames;
        unsigned int min_device_channels;

        struct timeval start_time;
        long long int captured_samples;

        bool non_interleaved;
};

static void audio_cap_alsa_probe(struct device_info **available_devices, int *count)
{
        *available_devices = malloc(sizeof(struct device_info));
        strcpy((*available_devices)[0].id, "alsa");
        strcpy((*available_devices)[0].name, "Default Linux audio input");
        *count = 1;
}

static void audio_cap_alsa_help(const char *driver_name)
{
        UNUSED(driver_name);
        audio_alsa_help();
}

static const int bps_preference[] = { 2, 4, 3, 1 };

#define DEFAULT_SAMPLE_RATE 48000

static void * audio_cap_alsa_init(const char *cfg)
{
        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Usage\n");
                color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-s alsa\n");
                color_out(COLOR_OUT_BOLD, "\t-s alsa:<device>\n");
                color_out(COLOR_OUT_BOLD, "\t-s alsa:<device>:opts=<opts>\n");
                color_out(COLOR_OUT_BOLD, "\t-s alsa:opts=<opts>\n\n");
                color_out(COLOR_OUT_BOLD, "\t<opts>");
                printf(" can be in format key1=value1:key2=value2, options are:\n");
                color_out(COLOR_OUT_BOLD, "\t\tframes=<frames>");
                printf(" number of audio frames captured at a moment\n");

                printf("\nAvailable ALSA capture devices\n");
                audio_cap_alsa_help(NULL);
                return &audio_init_state_ok;
        }
        struct state_alsa_capture *s;
        int rc;
        snd_pcm_hw_params_t *params;
        unsigned int val;
        int dir;
        const char *name = "default";
        char *opts = NULL;
        int format;
        char *tmp = NULL;

        s = calloc(1, sizeof(struct state_alsa_capture));

        if (cfg && strlen(cfg) > 0) {
                tmp = strdup(cfg);
                if (strncmp(tmp, "opts=", strlen("opts")) == 0) {
                        opts = tmp + strlen("opts=");
                } else {
                        name = tmp;
                        if (strstr(tmp, ":opts=") != NULL) {
                                opts = strstr(tmp, ":opts=") + strlen(":opts=");
                                *strstr(tmp, ":opts=") = '\0';
                        }
                }
        }

        gettimeofday(&s->start_time, NULL);
        s->frame.bps = audio_capture_bps;
        s->frame.sample_rate = audio_capture_sample_rate;
        s->min_device_channels = s->frame.ch_count = audio_capture_channels;
        s->tmp_data = NULL;

        /* Set period size to 128 frames or more. */
        s->frames = 128;

        if (opts) {
                char *item, *save_ptr;
                while ((item = strtok_r(opts, ":", &save_ptr)) != NULL) {
                        if (strncmp(item, "frames=", strlen("frames=")) == 0) {
                                s->frames = atoi(item + strlen("frames="));
                        } else {
                                fprintf(stderr, "[ALSA cap.] Unknown option: %s\n", item);
                                goto error;
                        }
                        opts = NULL;
                }
        }

        /* Open PCM device for recording (capture). */
        rc = snd_pcm_open(&s->handle, name,
                SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) {
                fprintf(stderr, MOD_NAME "unable to open pcm device: %s\n",
                        snd_strerror(rc));
                goto error;
        }
        print_alsa_device_info(s->handle, MOD_NAME);

        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        rc = snd_pcm_hw_params_any(s->handle, params);
        if (rc < 0) {
                fprintf(stderr, MOD_NAME "unable to set default parameters: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* Find appropriate parameters if not given by user */
        if (s->frame.bps < 0 || s->frame.bps > 4) {
                log_msg(LOG_LEVEL_ERROR, "[ALSA] %d bits per second are not supported by UG.\n",
                                s->frame.bps * 8);
                goto error;
        }

        if (s->frame.bps == 0) {
                for (unsigned int i = 0; i < sizeof bps_preference / sizeof(bps_preference[0]); i++) {
                        if (!snd_pcm_hw_params_test_format(s->handle, params, bps_to_snd_fmts[bps_preference[i]])) {
                                s->frame.bps = bps_preference[i];
                                break;
                        }
                }
                if (s->frame.bps == 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot find suitable sample format, "
                                        "please contact %s.\n", PACKAGE_BUGREPORT);
                        goto error;
                }
        }

        if (s->frame.sample_rate == 0) {
                s->frame.sample_rate = get_rate_near(s->handle, params, DEFAULT_SAMPLE_RATE);
                if (s->frame.sample_rate == 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot find usable sample rate!\n");
                        goto error;
                }
        }

        if (!snd_pcm_hw_params_test_access(s->handle, params, SND_PCM_ACCESS_RW_INTERLEAVED)) {
                s->non_interleaved = false;
        } else if (!snd_pcm_hw_params_test_access(s->handle, params, SND_PCM_ACCESS_RW_NONINTERLEAVED)) {
                if (audio_capture_channels > 1) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Non-interleaved mode "
                                        "available only when capturing mono!\n");
                        goto error;
                } else {
                        s->non_interleaved = true;
                }
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported access mode!\n");
                goto error;

        }

        /* Set the desired hardware parameters. */

        /* Access mode */
        rc = snd_pcm_hw_params_set_access(s->handle, params,
                s->non_interleaved ? SND_PCM_ACCESS_RW_NONINTERLEAVED : SND_PCM_ACCESS_RW_INTERLEAVED);
        if (rc < 0) {
                fprintf(stderr, MOD_NAME "unable to set interleaved mode: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* Sample format */
        format = bps_to_snd_fmts[s->frame.bps];
        rc = snd_pcm_hw_params_set_format(s->handle, params,
                format);
        if (rc < 0) {
                fprintf(stderr, MOD_NAME "unable to set capture format: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* we want to resample if device doesn't support default sample rate */
        val = 1;
        rc = snd_pcm_hw_params_set_rate_resample(s->handle,
                        params, val);
        if(rc < 0) {
                fprintf(stderr, MOD_NAME "Warning: Unable to set resampling: %s\n",
                        snd_strerror(rc));
        }

        /* set sampling rate */
        val = s->frame.sample_rate;
        dir = 0;
        rc = snd_pcm_hw_params_set_rate_near(s->handle, params,
                &val, &dir);
        if (rc < 0) {
                fprintf(stderr, "[ALSA cap.] unable to set sampling rate (%s %d): %s\n",
                        dir == 0 ? "=" : (dir == -1 ? "<" : ">"),
                        val, snd_strerror(rc));
                goto error;
        }

        /* Channels count */
        rc = snd_pcm_hw_params_set_channels(s->handle, params, s->frame.ch_count);
        if (rc < 0) {
                if (s->frame.ch_count == 1) { // some devices cannot do mono
                        snd_pcm_hw_params_set_channels_first(s->handle, params, &s->min_device_channels);
                } else {
                        fprintf(stderr, MOD_NAME "unable to set channel count: %s\n",
                                        snd_strerror(rc));
                        goto error;
                }
        }

        /* This must be set after setting of sample rate for Chat 150 which increases
         * value to 1024. But if this setting precedes, setting sample rate of 48000
         * fails (1024 period) or does not work properly (128).
         * */
        dir = 0;
        rc = snd_pcm_hw_params_set_period_size_near(s->handle,
                params, &s->frames, &dir);
        if (rc < 0) {
                fprintf(stderr, "[ALSA cap.] unable to set frame period (%ld): %s\n",
                                s->frames, snd_strerror(rc));
        }

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                fprintf(stderr, MOD_NAME "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* Use a buffer large enough to hold one period */
        snd_pcm_hw_params_get_period_size(params, &s->frames, &dir);
        s->frame.max_size = s->frames  * s->frame.ch_count * s->frame.bps;
        s->frame.data = (char *) malloc(s->frame.max_size);

        s->tmp_data = malloc(s->frames  * s->min_device_channels * s->frame.bps);

        log_msg(LOG_LEVEL_NOTICE, "ALSA capture configuration: %d channel%s, %d Bps, %d Hz, "
                       "%ld samples per frame.\n", s->frame.ch_count,
                       s->frame.ch_count == 1 ? "" : "s", s->frame.bps,
                       s->frame.sample_rate, s->frames);

        free(tmp);
        return s;

error:
        free(s);
        free(tmp);
        return NULL;
}

static struct audio_frame *audio_cap_alsa_read(void *state)
{
        struct state_alsa_capture *s = (struct state_alsa_capture *) state;
        int rc;
        char *discard_data;

        char *read_ptr[s->min_device_channels];
        read_ptr[0] = s->frame.data;
        if((int) s->min_device_channels > s->frame.ch_count && s->frame.ch_count == 1) {
                read_ptr[0] = s->tmp_data;
        }

        if (s->non_interleaved) {
                assert(audio_capture_channels == 1);
                discard_data = (char *) alloca(s->frames * s->frame.bps * (s->min_device_channels-1));
                for (unsigned int i = 1; i < s->min_device_channels; ++i) {
                        read_ptr[i] = discard_data + (i - 1) * s->frames * s->frame.bps;
                }
                rc = snd_pcm_readn(s->handle, (void **) read_ptr, s->frames);
        } else {
                rc = snd_pcm_readi(s->handle, read_ptr[0], s->frames);
        }
        if (rc == -EPIPE) {
                /* EPIPE means overrun */
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "overrun occurred\n");
                snd_pcm_prepare(s->handle);
        } else if (rc < 0) {
		log_msg(LOG_LEVEL_WARNING, MOD_NAME "error from read: %s\n", snd_strerror(rc));
		if (rc == -ENODEV) { // someone pulled the device (eg. videoconferencing
			// microphone) out - taking that as a fatal error
			log_msg(LOG_LEVEL_FATAL, MOD_NAME "Device removed, exiting!\n");
			exit_uv(EXIT_FAIL_AUDIO);
		}
        } else if (rc != (int)s->frames) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "short read, read %d frames\n", rc);
        }

        if(rc > 0) {
                if ((int) s->min_device_channels > s->frame.ch_count && s->frame.ch_count == 1) {
                        demux_channel(s->frame.data, (char *) s->tmp_data, s->frame.bps,
                                        rc * s->frame.bps * s->min_device_channels,
                                        s->min_device_channels, /* channels (originally) */
                                        0 /* we want first channel */
                                );
                }
                s->frame.data_len = rc * s->frame.bps * s->frame.ch_count;
                if (s->frame.bps == 1) {
                        // should be unsigned2signed but it works in both directions
                        signed2unsigned(s->frame.data, s->frame.data, s->frame.data_len);
                }
                s->captured_samples += rc;
                return &s->frame;
        } else {
                return NULL;
        }
}

static void audio_cap_alsa_done(void *state)
{
        struct state_alsa_capture *s = (struct state_alsa_capture *) state;
        struct timeval t;

        gettimeofday(&t, NULL);
        printf("[ALSA cap.] Captured %lld samples in %f seconds (%f samples per second).\n",
                        s->captured_samples, tv_diff(t, s->start_time),
                        s->captured_samples / tv_diff(t, s->start_time));
        snd_pcm_drop(s->handle);
        snd_pcm_close(s->handle);
        free(s->frame.data);
        free(s->tmp_data);
        free(s);
}

static const struct audio_capture_info acap_alsa_info = {
        audio_cap_alsa_probe,
        audio_cap_alsa_help,
        audio_cap_alsa_init,
        audio_cap_alsa_read,
        audio_cap_alsa_done
};

REGISTER_MODULE(alsa, &acap_alsa_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

#endif /* HAVE_ALSA */

/**
 * @file audio/playback/alsa.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2014 CESNET, z. s. p. o.
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

/*
 * Changes should use Safe ALSA API (http://0pointer.de/blog/projects/guide-to-sound-apis).
 *
 * Please, report all differencies from it here:
 * - used format SND_PCM_FORMAT_S24_LE
 * - used "default" device for arbitrary number of channels
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#ifdef HAVE_ALSA

#include <alsa/asoundlib.h>
#include <stdlib.h>
#include <string.h>

#include "alsa_common.h"
#include "audio/audio.h"
#include "audio/audio_playback.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "tv.h"

#define BUFFER_MIN 41
#define BUFFER_MAX 200

struct state_alsa_playback {
        snd_pcm_t *handle;
        struct audio_desc desc;

        unsigned int min_device_channels;
        struct timeval start_time;
        long long int played_samples;

        /* Local configuration with handle_underrun workaround set for PulseAudio
           ALSA plugin.  Will be NULL if the PA ALSA plugin is not in use or the
           workaround is not required. */
        snd_config_t * local_config;
};

static int audio_play_alsa_reconfigure(void *state, int quant_samples, int channels,
                                int sample_rate)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        snd_pcm_hw_params_t *params;
        snd_pcm_format_t format;
        unsigned int val;
        int dir;
        int rc;
        unsigned int frames;

        s->desc.bps = quant_samples / 8;
        s->min_device_channels = s->desc.ch_count = channels;
        s->desc.sample_rate = sample_rate;


        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        rc = snd_pcm_hw_params_any(s->handle, params);
        if (rc < 0) {
                fprintf(stderr, "cannot obtain default hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        rc = snd_pcm_hw_params_set_access(s->handle, params,
                        SND_PCM_ACCESS_RW_INTERLEAVED);
        if (rc < 0) {
                fprintf(stderr, "cannot set interleaved hw access: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        switch(quant_samples) {
                case 8:
                        format = SND_PCM_FORMAT_U8;
                        break;
                case 16:
                        format = SND_PCM_FORMAT_S16_LE;
                        break;
                case 24:
                        format = SND_PCM_FORMAT_S24_3LE;
                        break;
                case 32:
                        format = SND_PCM_FORMAT_S32_LE;
                        break;
                default:
                        fprintf(stderr, "[ALSA playback] Unsupported BPS for audio (%d).\n", quant_samples);
                        return FALSE;
        }
        /* Signed 16-bit little-endian format */
        rc = snd_pcm_hw_params_set_format(s->handle, params,
                        format);
        if (rc < 0) {
                fprintf(stderr, "cannot set format: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Two channels (stereo) */
        rc = snd_pcm_hw_params_set_channels(s->handle, params, channels);
        if (rc < 0) {
                if(channels == 1) {
                        snd_pcm_hw_params_set_channels_first(s->handle, params, &s->min_device_channels);
                } else {
                        fprintf(stderr, "cannot set requested channel count: %s\n",
                                        snd_strerror(rc));
                        return FALSE;
                }
        }

        /* we want to resample if device doesn't support default sample rate */
        val = 1;
        rc = snd_pcm_hw_params_set_rate_resample(s->handle,
                        params, val);
        if(rc < 0) {
                fprintf(stderr, "[ALSA play.] Warning: Unable to set resampling: %s\n",
                        snd_strerror(rc));
        }


        /* 44100 bits/second sampling rate (CD quality) */
        val = sample_rate;
        dir = 0;
        rc = snd_pcm_hw_params_set_rate_near(s->handle, params,
                        &val, &dir);
        if (rc < 0) {
                fprintf(stderr, "cannot set requested sample rate: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Set period to its minimal size.
         * Do not use snd_pcm_hw_params_set_period_size_near,
         * since it allows to set also unsupported value without notifying.
         * See also http://www.alsa-project.org/main/index.php/FramesPeriods */
        frames = 1;
        dir = 1;
        rc = snd_pcm_hw_params_set_period_time_min(s->handle,
                        params, &frames, &dir);
        if (rc < 0) {
                fprintf(stderr, "[ALSA play.] Warning: cannot set period time: %s\n",
                        snd_strerror(rc));
        }

        val = BUFFER_MIN * 1000;
        dir = 1;
        rc = snd_pcm_hw_params_set_buffer_time_min(s->handle, params,
                        &val, &dir); 
        if (rc < 0) {
                fprintf(stderr, "Warining - unable to set minimal buffer size: %s\n",
                        snd_strerror(rc));
        }

        val = BUFFER_MAX * 1000;
        dir = -1;
        rc = snd_pcm_hw_params_set_buffer_time_max(s->handle, params,
                        &val, &dir); 
        if (rc < 0) {
                fprintf(stderr, "Warining - unable to set maximal buffer size: %s\n",
                        snd_strerror(rc));
        }

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                fprintf(stderr,
                        "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        return TRUE;
}

static void audio_play_alsa_help(const char *driver_name)
{
        UNUSED(driver_name);
        audio_alsa_help();
}

static bool is_default_pulse(void)
{
        void **hints;
        bool default_pulse = false;
        bool pulse_present = false;

        snd_device_name_hint(-1, "pcm", &hints);
        while(*hints != NULL) {
                char *tmp = strdup(*(char **) hints);
                char *save_ptr = NULL;
                char *name_part = NULL;
                char *desc = NULL;

                name_part = strtok_r(tmp + 4, "|", &save_ptr);
                desc = strtok_r(NULL, "|", &save_ptr);

                if (strcmp(name_part, "default") == 0) {
                        if (desc && strstr(desc, "PulseAudio")) {
                                default_pulse = true;
                        }
                }

                if (strcmp(name_part, "pulse") == 0) {
                        pulse_present = true;
                }

                hints++;
                free(tmp);
        }

        return default_pulse && pulse_present;
}

/* Work around PulseAudio ALSA plugin bug where the PA server forces a
   higher than requested latency, but the plugin does not update its (and
   ALSA's) internal state to reflect that, leading to an immediate underrun
   situation.  Inspired by WINE's make_handle_underrun_config.
   Reference: http://mailman.alsa-project.org/pipermail/alsa-devel/2012-July/053391.html
              https://github.com/kinetiknz/cubeb/commit/1aa0058d0729eb85505df104cd1ac072432c6d24
              http://en.it-usenet.org/thread/17996/19923/
*/
static snd_config_t *
init_local_config_with_workaround(char const * pcm_node_name)
{
        int r;
        snd_config_t * lconf;
        snd_config_t * device_node;
        snd_config_t * type_node;
        snd_config_t * node;
        char const * type_string;

        lconf = NULL;

        if (snd_config == NULL) {
                snd_config_update();
        }

        r = snd_config_copy(&lconf, snd_config);
        assert(r >= 0);

        r = snd_config_search(lconf, pcm_node_name, &device_node);
        if (r != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        /* Fetch the PCM node's type, and bail out if it's not the PulseAudio plugin. */
        r = snd_config_search(device_node, "type", &type_node);
        if (r != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        r = snd_config_get_string(type_node, &type_string);
        if (r != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        if (strcmp(type_string, "pulse") != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        /* Don't clobber an explicit existing handle_underrun value, set it only
           if it doesn't already exist. */
        r = snd_config_search(device_node, "handle_underrun", &node);
        if (r != -ENOENT) {
                snd_config_delete(lconf);
                return NULL;
        }

        r = snd_config_imake_integer(&node, "handle_underrun", 0);
        if (r != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        r = snd_config_add(device_node, node);
        if (r != 0) {
                snd_config_delete(lconf);
                return NULL;
        }

        return lconf;
}

static void * audio_play_alsa_init(const char *cfg)
{
        int rc;
        struct state_alsa_playback *s;
        const char *name;

        s = calloc(1, sizeof(struct state_alsa_playback));

        gettimeofday(&s->start_time, NULL);

        if(cfg && strlen(cfg) > 0) {
                if(strcmp(cfg, "help") == 0) {
                        printf("Available ALSA playback devices:\n");
                        audio_play_alsa_help(NULL);
                        free(s);
                        return &audio_init_state_ok;
                }
                name = cfg;
        } else {
                if (is_default_pulse()) {
                        name = "pulse";
                } else {
                        name = "default";
                }
        }

        char device[1024] = "pcm.";
        strncat(device, name, sizeof(device) - strlen(device) - 1);
        s->local_config = init_local_config_with_workaround(device);

        if (s->local_config) {
                rc = snd_pcm_open_lconf(&s->handle, name,
                                SND_PCM_STREAM_PLAYBACK, 0, s->local_config);
        } else {
                rc = snd_pcm_open(&s->handle, name,
                                SND_PCM_STREAM_PLAYBACK, 0);
        }

        if (rc < 0) {
                    fprintf(stderr, "unable to open pcm device: %s\n",
                                    snd_strerror(rc));
                    goto error;
        }

        rc = snd_pcm_nonblock(s->handle, 1);
        if(rc < 0) {
                fprintf(stderr, "ALSA Warning: Unable to set nonblock mode.\n");
        }
        
        return s;

error:
        free(s);
        return NULL;
}

static void audio_play_alsa_put_frame(void *state, struct audio_frame *frame)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        int rc;
        char *data = frame->data;
        char *tmp_data = NULL;
        int frames = frame->data_len / (frame->bps * frame->ch_count);

#ifdef DEBUG
        snd_pcm_sframes_t delay;
        snd_pcm_delay(s->handle, &delay);
        //fprintf(stderr, "Alsa delay: %d samples (%u Hz)\n", (int)delay, (unsigned int) s->frame.sample_rate);
#endif

        if(frame->bps == 1) { // convert to unsigned
                signed2unsigned(frame->data, frame->data, frame->data_len);
        }

        if((int) s->min_device_channels > frame->ch_count && frame->ch_count == 1) {
                tmp_data = (char *) malloc(frame->data_len * s->min_device_channels);
                copy_channel(tmp_data, frame->data, frame->bps, frame->data_len, s->min_device_channels);
                data = tmp_data;
        }

        s->played_samples += frame->data_len / frame->bps / frame->ch_count;
    
        rc = snd_pcm_writei(s->handle, data, frames);
        if (rc == -EPIPE) {
                /* EPIPE means underrun */
                fprintf(stderr, "underrun occurred\n");
                snd_pcm_prepare(s->handle);
                /* fill the stream with some sasmples */
                for (double sec = 0.0; sec < BUFFER_MAX / 1000.0; sec += (double) frames / frame->sample_rate) {
                        int frames_to_write = frames;
                        if(sec + (double) frames/frame->sample_rate > BUFFER_MAX / 1000.0) {
                                frames_to_write = (BUFFER_MAX / 1000.0 - sec) * frame->sample_rate;
                        }
                        int rc = snd_pcm_writei(s->handle, data, frames_to_write);
                        if(rc < 0) {
                                fprintf(stderr, "error from writei: %s\n",
                                                snd_strerror(rc));
                                break;
                        }
                }
        } else if (rc < 0) {
                fprintf(stderr, "error from writei: %s\n",
                        snd_strerror(rc));
        }  else if (rc != (int)frames) {
                fprintf(stderr, "short write, written %d frames (overrun)\n", rc);
        }

        free(tmp_data);
}

static void audio_play_alsa_done(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        struct timeval t;

        gettimeofday(&t, NULL);
        printf("[ALSA play.] Played %lld samples in %f seconds (%f samples per second).\n",
                        s->played_samples, tv_diff(t, s->start_time),
                        s->played_samples / tv_diff(t, s->start_time));

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        free(s);
}

static const struct audio_playback_info aplay_alsa_info = {
        audio_play_alsa_help,
        audio_play_alsa_init,
        audio_play_alsa_put_frame,
        audio_play_alsa_reconfigure,
        audio_play_alsa_done
};

static void mod_reg(void)  __attribute__((constructor));

static void mod_reg(void)
{
        register_library("alsa", &aplay_alsa_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);
}

#endif /* HAVE_ALSA */

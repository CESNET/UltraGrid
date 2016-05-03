/**
 * @file audio/playback/alsa.c
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
#define MOD_NAME "[ALSA play.] "

/**
 * Speex jitter buffer use is currently not stable and not ready for production use.
 * Moreover, it is unclear whether is suitable for our use case.
 */
//#define USE_SPEEX_JITTER_BUFFER 1

#ifdef USE_SPEEX_JITTER_BUFFER
#include <speex/speex_jitter.h>
#else
#include "utils/audio_buffer.h"
#endif

struct state_alsa_playback {
        snd_pcm_t *handle;
        struct audio_desc desc;

        struct timeval start_time;
        long long int played_samples;

        /* Local configuration with handle_underrun workaround set for PulseAudio
           ALSA plugin.  Will be NULL if the PA ALSA plugin is not in use or the
           workaround is not required. */
        snd_config_t * local_config;

        bool non_interleaved;
        bool new_api;

        // following variables are used only if new_api == true
        pthread_t thread_id;
#ifdef USE_SPEEX_JITTER_BUFFER
        JitterBuffer *buf;
#else
        audio_buffer_t *buf;
#endif
        pthread_mutex_t lock;
        bool should_exit_thread;
        bool thread_started;
        uint32_t timestamp;
        struct timeval last_audio_read;
};

static void audio_play_alsa_write_frame(void *state, struct audio_frame *frame);

static void *worker(void *args) {
        struct state_alsa_playback *s = args;

        struct audio_frame f = { .bps = s->desc.bps,
                .sample_rate = s->desc.sample_rate,
                .ch_count = s->desc.ch_count };

        size_t len = 256;
        size_t len_100ms = f.bps * f.sample_rate * f.ch_count / 10;
        char *silence = alloca(len_100ms);
        memset(silence, 0, len_100ms);

#ifdef USE_SPEEX_JITTER_BUFFER
	const int pkt_max_len = s->desc.sample_rate / 10;
        JitterBufferPacket pkt;
        pkt.len = pkt_max_len;
        pkt.data = alloca(pkt_max_len);
#else
        char *data = alloca(len);
#endif
        while (1) {
                pthread_mutex_lock(&s->lock);
                if (s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }

#ifdef USE_SPEEX_JITTER_BUFFER
                int start_offset;
                int err = jitter_buffer_get(s->buf, &pkt, len, &start_offset);
			jitter_buffer_tick(s->buf);
#else
                int ret = audio_buffer_read(s->buf, data, len);
#endif
                pthread_mutex_unlock(&s->lock);

#ifdef USE_SPEEX_JITTER_BUFFER
                if (err == JITTER_BUFFER_OK) {
                        f.data_len = pkt.len;
                        f.data = pkt.data;
		} else {
			log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "underrun occurred\n");
			if (err < 0) {
				log_msg(LOG_LEVEL_WARNING, MOD_NAME "Jitter buffer: %s\n",
						JITTER_BUFFER_INTERNAL_ERROR ? "internal error" :
						"invalid argument\n");
			}
                        f.data_len = len;
                        f.data = silence;
		}
		pkt.len = pkt_max_len;
#else
                struct timeval now;
                gettimeofday(&now, NULL);

                if (ret > 0) {
                        f.data_len = ret;
                        f.data = data;
                        s->last_audio_read = now;
                } else {
                        f.data = silence;

                        if (tv_diff(now, s->last_audio_read) < 2.0) {
                                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "underrun occurred\n");
                                f.data_len = len;
                        } else {
                                f.data_len = len_100ms;
                        }
                }
#endif

                audio_play_alsa_write_frame(s, &f);
        }
}

static struct audio_desc audio_play_alsa_query_format(void *state, struct audio_desc desc)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        int rc;
        unsigned int val;
        int dir;
        snd_pcm_hw_params_t *params;
        struct audio_desc ret;

        memset(&ret, 0, sizeof ret);

        ret.codec = AC_PCM;

        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        rc = snd_pcm_hw_params_any(s->handle, params);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to set default parameters: %s\n",
                        snd_strerror(rc));
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        assert (desc.bps > 0 && desc.bps <= 4);
        int bps = desc.bps;
        for ( ; ; ) {
                if (!snd_pcm_hw_params_test_format(s->handle, params, bps_to_snd_fmts[bps])) {
                        break;
                }
                // We try to find nearest higher
                if (bps >= desc.bps) {
                        bps++;
                } else { // or nearest lower
                        bps--;
                }
                if (bps > 4) {
                        bps = desc.bps - 1;
                }
                if (bps == 0) {
                        break;
                }
        }

        ret.bps = bps;

        rc = snd_pcm_hw_params_set_format(s->handle, params, bps_to_snd_fmts[bps]);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to set capture format: %s\n",
                                snd_strerror(rc));
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        ret.sample_rate = get_rate_near(s->handle, params, desc.sample_rate);
        if (!ret.sample_rate) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to find sample rate:\n");
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        /* set sampling rate */
        val = desc.sample_rate;
        dir = 0;
        rc = snd_pcm_hw_params_set_rate_near(s->handle, params,
                &val, &dir);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to set sampling rate (%s %d): %s\n",
                        dir == 0 ? "=" : (dir == -1 ? "<" : ">"),
                        val, snd_strerror(rc));
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        int channels = desc.ch_count;
        unsigned int max_channels;
        rc = snd_pcm_hw_params_get_channels_max(params, &max_channels);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to get max number of channels!\n");
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }
        for ( ; ; ) {
                if (!snd_pcm_hw_params_test_channels(s->handle, params, channels)) {
                        break;
                }
                // We try to find nearest higher
                if (channels >= desc.ch_count) {
                        channels++;
                } else { // or nearest lower
                        channels--;
                }
                if (channels > (int) max_channels) {
                        channels = desc.ch_count - 1;
                }
                if (channels == 0) {
                        break;
                }
        }

        if (channels == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to get usable channe countl!\n");
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        rc = snd_pcm_hw_params_set_channels(s->handle, params, channels);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to set channels!\n");
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        ret.ch_count = channels;

        if (snd_pcm_hw_params_test_access(s->handle, params, SND_PCM_ACCESS_RW_INTERLEAVED)
                && snd_pcm_hw_params_test_access(s->handle, params, SND_PCM_ACCESS_RW_NONINTERLEAVED)) {

                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot find supported access mode!\n");
                return (struct audio_desc){0, 0, 0, AC_NONE};
        }

        return ret;
}

static int audio_play_alsa_reconfigure(void *state, struct audio_desc desc)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        snd_pcm_hw_params_t *params;
        snd_pcm_format_t format;
        unsigned int val;
        int dir;
        int rc;
        unsigned int frames;

        if (s->new_api && s->thread_started) {
                pthread_mutex_lock(&s->lock);
                s->should_exit_thread = true;
                pthread_mutex_unlock(&s->lock);
                pthread_join(s->thread_id, NULL);
                s->should_exit_thread = false;
                s->thread_started = false;
        }

        s->desc.bps = desc.bps;
        s->desc.ch_count = desc.ch_count;
        s->desc.sample_rate = desc.sample_rate;

        snd_pcm_drop(s->handle); // stop the stream

        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        rc = snd_pcm_hw_params_any(s->handle, params);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot obtain default hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        rc = snd_pcm_hw_params_set_access(s->handle, params,
                        SND_PCM_ACCESS_RW_INTERLEAVED);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot set interleaved hw access: %s\n",
                        snd_strerror(rc));
                rc = snd_pcm_hw_params_set_access(s->handle, params,
                                SND_PCM_ACCESS_RW_NONINTERLEAVED);
                if (rc < 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot set non-interleaved hw access: %s\n",
                                        snd_strerror(rc));
                        return FALSE;
                }
                s->non_interleaved = true;
        } else {
                s->non_interleaved = false;
        }

        if (desc.bps > 4 || desc.bps < 1) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported BPS for audio (%d).\n",
                                desc.bps * 8);
                return FALSE;

        }
        format = bps_to_snd_fmts[desc.bps];

        /* Signed 16-bit little-endian format */
        rc = snd_pcm_hw_params_set_format(s->handle, params,
                        format);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot set format: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Two channels (stereo) */
        rc = snd_pcm_hw_params_set_channels(s->handle, params, desc.ch_count);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot set requested channel count: %s\n",
                                snd_strerror(rc));
                return FALSE;
        }

        /* we want to resample if device doesn't support default sample rate */
        val = 1;
        rc = snd_pcm_hw_params_set_rate_resample(s->handle,
                        params, val);
        if(rc < 0) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: Unable to set resampling: %s\n",
                        snd_strerror(rc));
        }


        /* 44100 bits/second sampling rate (CD quality) */
        val = desc.sample_rate;
        dir = 0;
        rc = snd_pcm_hw_params_set_rate_near(s->handle, params,
                        &val, &dir);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "cannot set requested sample rate: %s\n",
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
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: cannot set period time: %s\n",
                        snd_strerror(rc));
        }

        if (s->new_api) {
                int mindir = -1, maxdir = 1;
                unsigned int minval = 0;
                unsigned int maxval = 15000;

                const char *minval_str = get_commandline_param("alsa-playback-bufmin");
                if (minval_str) {
                        minval = atoi(minval_str);
                }
                const char *maxval_str = get_commandline_param("alsa-playback-bufmax");
                if (maxval_str) {
                        maxval = atoi(maxval_str);
                }

                rc = snd_pcm_hw_params_set_buffer_time_minmax(s->handle, params, &minval, &mindir,
                                &maxval, &maxdir);
                if (rc < 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning - unable to set buffer to its size: %s\n",
                                        snd_strerror(rc));
                } else {
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Buffer size: %d-%d ns\n", minval, maxval);
                }
        } else {
                val = BUFFER_MIN * 1000;
                dir = 1;
                rc = snd_pcm_hw_params_set_buffer_time_min(s->handle, params,
                                &val, &dir);
                if (rc < 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning - unable to set minimal buffer size: %s\n",
                                        snd_strerror(rc));
                }

                val = BUFFER_MAX * 1000;
                dir = -1;
                rc = snd_pcm_hw_params_set_buffer_time_max(s->handle, params,
                                &val, &dir);
                if (rc < 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning - unable to set maximal buffer size: %s\n",
                                        snd_strerror(rc));
                }
        }

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        if (s->new_api) {
#ifdef USE_SPEEX_JITTER_BUFFER
		jitter_buffer_reset(s->buf);
#else
                audio_buffer_destroy(s->buf);
                s->buf = audio_buffer_init(s->desc.sample_rate, s->desc.bps, s->desc.ch_count, 20);
#endif
                s->timestamp = 0;
                pthread_create(&s->thread_id, NULL, worker, s);
                s->thread_started = true;
        }

        return TRUE;
}

static void audio_play_alsa_probe(struct device_info **available_devices, int *count)
{
        *available_devices = malloc(sizeof(struct device_info));
        strcpy((*available_devices)[0].id, "alsa");
        strcpy((*available_devices)[0].name, "Default Linux audio output");
        *count = 1;
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

        const char *new_api;
        new_api = get_commandline_param("alsa-playback-api");
        if (new_api) {
                if (strcmp(new_api, "new") == 0) {
                        s->new_api = true;
                } else if (strcmp(new_api, "old") == 0) {
                        s->new_api = false;
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown API string \"%s\"!\n", new_api);
                        free(s);
                        return NULL;
                }
        } else {
		log_msg(LOG_LEVEL_NOTICE, MOD_NAME "You may try to use \"--param alsa-playback-api=new\" to use a new playback API which may become default in future versions.\n");
	}

        gettimeofday(&s->start_time, NULL);

        if(cfg && strlen(cfg) > 0) {
                if(strcmp(cfg, "help") == 0) {
			printf("Usage\n");
			printf("\t-s alsa\n");
			printf("\t-s alsa:<device>\n");
			printf("\t-s alsa --param alsa-playback-api={new|old} # uses new experimental API\n");
			printf("\t-s alsa --param alsa-playback-api=new[:alsa-playback-bufmin=<us>][:alsa-playback-bufmax=<us>] # new api only\n");
			printf("\n");

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

        if (!s->new_api) {
                s->local_config = init_local_config_with_workaround(device);
        }

        if (!s->new_api && s->local_config) {
                rc = snd_pcm_open_lconf(&s->handle, name,
                                SND_PCM_STREAM_PLAYBACK, 0, s->local_config);
        } else {
                rc = snd_pcm_open(&s->handle, name,
                                SND_PCM_STREAM_PLAYBACK, 0);
        }

        if (rc < 0) {
                    log_msg(LOG_LEVEL_ERROR, MOD_NAME "unable to open pcm device: %s\n",
                                    snd_strerror(rc));
                    goto error;
        }

        if (s->new_api) {
#ifdef USE_SPEEX_JITTER_BUFFER
		s->buf = jitter_buffer_init(1);
#endif
                pthread_mutex_init(&s->lock, NULL);
        } else {
                rc = snd_pcm_nonblock(s->handle, 1);
                if(rc < 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: Unable to set nonblock mode.\n");
                }
        }

        /* CC3000e playback needs to be initialized prior to capture
         * (to arbitrary value) in order to work. This hack ensures
         * that. */
        if (strstr(snd_pcm_name(s->handle), "CARD=Speakerph")) {
                audio_play_alsa_reconfigure(s, audio_play_alsa_query_format(s, (struct audio_desc){2, 48000, 1, AC_PCM}));
        }
        
        return s;

error:
        free(s);
        return NULL;
}

static int write_samples(snd_pcm_t *handle, const struct audio_frame *frame, int frames, bool noninterleaved, bool new_api) {
        char *write_ptr[frame->ch_count]; // for non-interleaved
        char *tmp_data = (char *) alloca(frame->data_len);
        if (noninterleaved) {
                interleaved2noninterleaved(tmp_data, frame->data, frame->bps, frame->data_len, frame->ch_count);
        }

        char *data = frame->data; // for interleaved
        int written = 0;
        while (written < frames) {
                int rc;
                if (noninterleaved) {
                        for (int i = 0; i < frame->ch_count; ++i) {
                                write_ptr[i] = tmp_data + frame->data_len / frame->ch_count * i + written * frame->bps;
                        }
                        rc = snd_pcm_writen(handle, (void **) &write_ptr, frames);
                } else {
                        rc = snd_pcm_writei(handle, data, frames - written);
                }

                if (rc < 0) {
                        return rc;
                }
                if (!new_api) { // overrun (in non-blocking mode) or signal
                        return rc;
                }
                written += rc;
                data += written * frame->bps * frame->ch_count;
        }
        return written;
}

static void audio_play_alsa_write_frame(void *state, struct audio_frame *frame)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        int rc;

#ifdef DEBUG
        snd_pcm_sframes_t delay;
        snd_pcm_delay(s->handle, &delay);
        //fprintf(stderr, "Alsa delay: %d samples (%u Hz)\n", (int)delay, (unsigned int) s->frame.sample_rate);
#endif

        if(frame->bps == 1) { // convert to unsigned
                signed2unsigned(frame->data, frame->data, frame->data_len);
        }

        s->played_samples += frame->data_len / frame->bps / frame->ch_count;
    
        int frames = frame->data_len / (frame->bps * frame->ch_count);
        rc = write_samples(s->handle, frame, frames, s->non_interleaved, s->new_api);
        if (rc == -EPIPE) {
                /* EPIPE means underrun */
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "underrun occurred\n");
                snd_pcm_prepare(s->handle);
                /* fill the stream with some sasmples */
                for (double sec = 0.0; sec < BUFFER_MAX / 1000.0; sec += (double) frames / frame->sample_rate) {
                        int frames_to_write = frames;
                        if(sec + (double) frames/frame->sample_rate > BUFFER_MAX / 1000.0) {
                                frames_to_write = (BUFFER_MAX / 1000.0 - sec) * frame->sample_rate;
                        }
                        int rc = write_samples(s->handle, frame, frames_to_write, s->non_interleaved, s->new_api);
                        if(rc < 0) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "error from writei: %s\n",
                                                snd_strerror(rc));
                                break;
                        }
                }
        } else if (rc < 0) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "error from writei: %s\n",
                        snd_strerror(rc));
                snd_pcm_recover(s->handle, rc, 0);
        }  else if (rc != (int)frames) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "short write, written %d frames (overrun)\n", rc);
        }
}

static void audio_play_alsa_put_frame(void *state, struct audio_frame *frame)
{
        struct state_alsa_playback *s = state;

        if (s->new_api) {
               pthread_mutex_lock(&s->lock);
#ifdef USE_SPEEX_JITTER_BUFFER
                JitterBufferPacket pkt;
                pkt.data = frame->data;
                pkt.len = frame->data_len;
                pkt.timestamp = s->timestamp;
                pkt.span = frame->data_len / s->desc.bps / s->desc.ch_count;
                s->timestamp += pkt.span;
                jitter_buffer_put(s->buf, &pkt);
#else
                audio_buffer_write(s->buf, frame->data, frame->data_len);
#endif
                pthread_mutex_unlock(&s->lock);
        } else {
                audio_play_alsa_write_frame(state, frame);
        }
}

static void audio_play_alsa_done(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        struct timeval t;

        if (s->new_api && s->thread_started) {
                pthread_mutex_lock(&s->lock);
                s->should_exit_thread = true;
                pthread_mutex_unlock(&s->lock);
                pthread_join(s->thread_id, NULL);
        }

        gettimeofday(&t, NULL);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Played %lld samples in %f seconds (%f samples per second).\n",
                        s->played_samples, tv_diff(t, s->start_time),
                        s->played_samples / tv_diff(t, s->start_time));

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        if (s->local_config) {
                snd_config_delete(s->local_config);
        }
        if (s->new_api) {
#ifdef USE_SPEEX_JITTER_BUFFER
                jitter_buffer_destroy(s->buf);
#else
                audio_buffer_destroy(s->buf);
#endif
                pthread_mutex_destroy(&s->lock);
        }
        free(s);
}

static const struct audio_playback_info aplay_alsa_info = {
        audio_play_alsa_probe,
        audio_play_alsa_help,
        audio_play_alsa_init,
        audio_play_alsa_put_frame,
        audio_play_alsa_query_format,
        audio_play_alsa_reconfigure,
        audio_play_alsa_done
};

REGISTER_MODULE(alsa, &aplay_alsa_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


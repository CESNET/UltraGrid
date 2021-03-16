/**
 * @file   alsa_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2019 CESNET, z. s. p. o.
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

#ifndef ALSA_COMMON_H
#define ALSA_COMMON_H

#include <alsa/asoundlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "utils/color_out.h"
#include "types.h"

static inline void audio_alsa_probe(struct device_info **available_devices,
                                    int *count,
                                    const char **whitelist,
                                    size_t whitelist_size)
{
        void **hints;
        snd_device_name_hint(-1, "pcm", &hints);

        size_t device_count = 0;
        for (void **it = hints; *it != NULL; it++) device_count++;

        *available_devices = calloc(device_count + 1 , sizeof(struct device_info));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Default Linux audio");
        *count = 1;

        for(void **it = hints; *it != NULL; it++){
                char *id = snd_device_name_get_hint(*it, "NAME");

                bool whitelisted = false;
                for(size_t i = 0; i < whitelist_size; i++){
                        if(strstr(id, whitelist[i])){
                                whitelisted = true;
                                break;
                        }
                }
                if(!whitelisted && whitelist_size > 0) {
                        free(id);
                        continue;
                }

                strcpy((*available_devices)[*count].dev, ":");
                strncat((*available_devices)[*count].dev, id,
                                sizeof (*available_devices)[*count].dev -
                                strlen((*available_devices)[*count].dev) - 1);
                free(id);

                char *name = snd_device_name_get_hint(*it, "DESC");
                char *tok = name;
                while(tok){
                        char *newline = strchr(tok, '\n');
                        if(newline){
                                *newline = '\0';
                                strncat((*available_devices)[*count].name, tok,
                                                sizeof (*available_devices)[*count].name -
                                                strlen((*available_devices)[*count].name) - 1);
                                strncat((*available_devices)[*count].name, " - ",
                                                sizeof (*available_devices)[*count].name -
                                                strlen((*available_devices)[*count].name) - 1);
                                tok = newline + 1;
                        } else {
                                strncat((*available_devices)[*count].name, tok,
                                                sizeof (*available_devices)[*count].name -
                                                strlen((*available_devices)[*count].name) - 1);
                                tok = NULL;
                        }
                }
                free(name);
                (*count)++;
        }

        snd_device_name_free_hint(hints);
}

static inline void audio_alsa_help(void)
{
        struct device_info *available_devices;
        int count;
        audio_alsa_probe(&available_devices, &count, NULL, 0);
        strcpy(available_devices[0].dev, "");
        strcpy(available_devices[0].name, "default ALSA device (same as \"alsa:default\")");
        for(int i = 0; i < count; i++){
                const char * const id = available_devices[i].dev;
                color_out(COLOR_OUT_BOLD, "\talsa%s", id);
                for (int j = 0; j < 30 - (int) strlen(id); ++j) putchar(' ');
                printf(": %s\n", available_devices[i].name);
        }
        free(available_devices);
}

static const snd_pcm_format_t bps_to_snd_fmts[] = {
        [0] = SND_PCM_FORMAT_UNKNOWN,
        [1] = SND_PCM_FORMAT_U8,
        [2] = SND_PCM_FORMAT_S16_LE,
        [3] = SND_PCM_FORMAT_S24_3LE,
        [4] = SND_PCM_FORMAT_S32_LE,
};

/**
 * Finds equal or nearest higher sample rate that device supports. If none exist, pick highest
 * lower value.
 *
 * @returns sample rate, 0 if none was found
 */
static int get_rate_near(snd_pcm_t *handle, snd_pcm_hw_params_t *params, unsigned int approx_val) {
        int ret = 0;
        int dir = 0;
        int rc;
        unsigned int rate = approx_val;
        // try exact sample rate
        rc = snd_pcm_hw_params_set_rate_min(handle, params, &rate, &dir);
        if (rc != 0) {
                dir = 1;
                // or higher
                rc = snd_pcm_hw_params_set_rate_min(handle, params, &rate, &dir);
        }

        if (rc == 0) {
                // read the rate
                rc = snd_pcm_hw_params_get_rate_min(params, &rate, NULL);
                if (rc == 0) {
                        ret = rate;
                }
                // restore configuration space
                rate = 0;
                dir = 1;
                rc = snd_pcm_hw_params_set_rate_min(handle, params, &rate, &dir);
                assert(rc == 0);
        }

        // we did not succeed, try lower sample rate
        if (ret == 0) {
                unsigned int rate = approx_val;
                dir = 0;
                unsigned int orig_max;
                rc = snd_pcm_hw_params_get_rate_max(params, &orig_max, NULL);
                assert(rc == 0);

                rc = snd_pcm_hw_params_set_rate_max(handle, params, &rate, &dir);
                if (rc != 0) {
                        dir = -1;
                        rc = snd_pcm_hw_params_set_rate_max(handle, params, &rate, &dir);
                }

                if (rc == 0) {
                        rc = snd_pcm_hw_params_get_rate_max(params, &rate, NULL);
                        if (rc == 0) {
                                ret = rate;
                        }
                        // restore configuration space
                        dir = 0;
                        rc = snd_pcm_hw_params_set_rate_max(handle, params, &orig_max, &dir);
                        assert(rc == 0);
                }
        }
        return ret;
}

static void print_alsa_device_info(snd_pcm_t *handle, const char *module_name) {
        snd_pcm_info_t *info;
        snd_pcm_info_alloca(&info);
        if (snd_pcm_info(handle, info) == 0) {
                log_msg(LOG_LEVEL_NOTICE, "%sUsing device: %s\n",
                                module_name, snd_pcm_info_get_name(info));
        }
}

static const char *alsa_get_pcm_state_name(snd_pcm_state_t state) __attribute__((unused));
static const char *alsa_get_pcm_state_name(snd_pcm_state_t state) {
        switch (state) {
                case SND_PCM_STATE_OPEN:
                        return "Open";
                case SND_PCM_STATE_SETUP:
                        return "Setup installed";
                case SND_PCM_STATE_PREPARED:
                        return "Ready to start";
                case SND_PCM_STATE_RUNNING:
                        return "Running";
                case SND_PCM_STATE_XRUN:
                        return "Stopped: underrun (playback) or overrun (capture) detected";
                case SND_PCM_STATE_DRAINING:
                        return "Draining: running (playback) or stopped (capture)";
                case SND_PCM_STATE_PAUSED:
                        return "Paused";
                case SND_PCM_STATE_SUSPENDED:
                        return "Hardware is suspended";
                case SND_PCM_STATE_DISCONNECTED:
                        return "Hardware is disconnected";
                default:
                        return "(unknown)";
        }
}

#endif // defined ALSA_COMMON_H


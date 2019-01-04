/**
 * @file   alsa_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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

static inline void audio_alsa_help(void)
{
        void **hints;

        color_out(COLOR_OUT_BOLD, "\talsa");
        printf("%27s default ALSA device (same as \"alsa:default\")\n", ":");
        snd_device_name_hint(-1, "pcm", &hints);
        while(*hints != NULL) {
                char *tmp = strdup(*(char **) hints);
                char *save_ptr = NULL;
                char *name_part = NULL;
                char *desc = NULL;
                char *desc_short = NULL;
                char *desc_long = NULL;
                char *name = NULL;

                name_part = strtok_r(tmp + 4, "|", &save_ptr);
                desc = strtok_r(NULL, "|", &save_ptr);
                if (desc) {
                        desc_short = strtok_r(desc + 4, "\n", &save_ptr);
                        desc_long = strtok_r(NULL, "\n", &save_ptr);
                }

                name = malloc(strlen("alsa:") + strlen(name_part) + 1);
                strcpy(name, "alsa:");
                strcat(name, name_part);

                color_out(COLOR_OUT_BOLD, "\t%s", name);
                int i;

                if (desc_short) {
                        for (i = 0; i < 30 - (int) strlen(name); ++i) putchar(' ');
                        printf(": %s", desc_short);
                        if(desc_long) {
                                printf(" - %s", desc_long);
                        }
                }
                printf("\n");
                hints++;
                free(tmp);
                free(name);
        }
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

#endif // defined ALSA_COMMON_H


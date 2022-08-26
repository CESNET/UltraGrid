/**
 * @file   playback.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "debug.h"
#include "keyboard_control.h"
#include "playback.h"
#include "utils/color_out.h"

#define MOD_NAME "[playback] "

static void  playback_usage(void) {
        color_printf("Usage:\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t--playback [<file>|<dir>|help]" TERM_FG_RESET "[:loop]\n\n" TERM_RESET);
        color_printf("Use ");
        color_printf(TERM_BOLD "-t file:help" TERM_RESET " or " TERM_BOLD "-t import:help" TERM_RESET " to see further specific configuration options.\n\n");
}

int playback_set_device(char *device_string, size_t buf_len, const char *optarg) {
        bool is_import = false;
        if (strcmp(optarg, "help") == 0) {
                playback_usage();
                return 0;
        }
        char *path = strdup(optarg);
        if (strchr(path, ':')) {
                *strchr(path, ':') = '\0';
        }

        struct stat sb;
        if (stat(path, &sb) == -1) {
                perror(MOD_NAME "stat");
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot access file or directory '%s'\n", path);
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Use '--playback help' to see usage.\n");
                free(path);
                return -1;
        }

        if (sb.st_mode & S_IFDIR || strstr(path, "video.info") != NULL) {
                is_import = true;
        }

        snprintf(device_string, buf_len, "%s:%s:opportunistic_audio", is_import ? "import" : "file", optarg);
        free(path);
        return 1;
}

/**
 * @param mod moodule that will receive the playback messages
 */
void playback_register_keyboard_ctl(struct module *mod) {
        struct { int key; const char *msg; } kb[] = {{K_PGUP, "seek +600s"}, {K_PGDOWN, "seek -600s"}, {K_UP, "seek +60"}, {K_DOWN, "seek -60s"}, {K_LEFT, "seek -10s"}, {K_RIGHT, "seek +10s"}, {' ', "pause"}, {'q', "quit"}};
        for (size_t i = 0; i < sizeof kb / sizeof kb[0]; ++i) {
                char description[200];
                snprintf(description, sizeof description, "playback %s", kb[i].msg);
                if (!keycontrol_register_key(mod, kb[i].key, kb[i].msg, description)) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot register key %d control for playback!\n", kb[i].key);
                }
        }
}


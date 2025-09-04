/**
 * @file   playback.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2025 CESNET, zájmové sdružení právnických osob
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

#include "playback.h"

#include <errno.h>             // for errno
#include <stdbool.h>           // for bool, false, true
#include <stdio.h>             // for snprintf
#include <stdlib.h>            // for free
#include <string.h>            // for strchr, strcmp, strdup, strstr

#include "debug.h"             // for LOG_LEVEL_ERROR, LOG_LEVEL_INFO, MSG
#include "keyboard_control.h"  // for keycontrol_register_key, K_DOWN, K_LEFT
#include "utils/color_out.h"   // for color_printf, TBOLD, TERM_BOLD, TERM_R...
#include "utils/fs.h"          // for dir_exists
#include "utils/misc.h"        // for ug_strerror
#include "utils/text.h"        // for wrap_paragraph

struct module;

#define MOD_NAME "[playback] "

static void  playback_usage(void) {
        char desc[] = TBOLD("playback") " can capture data from locally "
                "stored data, which can be either a regular video file, "
                "or a directory with data recorded by UltraGrid itself "
                "(--record option).";
        color_printf("%s\n\n", wrap_paragraph(desc));
        color_printf("Usage:\n");
        color_printf(TBOLD(TRED("\t--playback <file>|<dir>") "[:loop]\n"));
        color_printf(TBOLD(TRED("\t-I{<file>|<dir>}") "[:loop]\n"));
        color_printf(TBOLD("\t-Ihelp | --playback help") "\n\n");
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

        if (!file_exists(path, FT_ANY)) {
                MSG(ERROR, "Cannot access file or directory: %s (%s)\n", path,
                    ug_strerror(errno));
                MSG(INFO, "Use '--playback help' to see usage.\n");
                free(path);
                return -1;
        }
        if (file_exists(path, FT_DIRECTORY) ||
            strstr(path, "video.info") != NULL) {
                is_import = true;
        }

        snprintf(device_string, buf_len, "%s:%s:opportunistic_audio", is_import ? "import" : "file", optarg);
        free(path);
        return 1;
}

/**
 * @param mod module that will receive the playback messages
 */
void playback_register_keyboard_ctl(struct module *mod) {
        struct { int key; const char *msg; } kb[] = {{K_PGUP, "seek +600s"}, {K_PGDOWN, "seek -600s"}, {K_UP, "seek +60s"}, {K_DOWN, "seek -60s"}, {K_LEFT, "seek -10s"}, {K_RIGHT, "seek +10s"}, {' ', "pause"}, {'q', "quit"}};
        for (size_t i = 0; i < sizeof kb / sizeof kb[0]; ++i) {
                char description[200];
                snprintf(description, sizeof description, "playback %s", kb[i].msg);
                if (!keycontrol_register_key(mod, kb[i].key, kb[i].msg, description)) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot register key %d control for playback!\n", kb[i].key);
                }
        }
}


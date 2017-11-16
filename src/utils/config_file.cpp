/**
 * @file   utils/config_file.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  Configuration file for UltraGrid
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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
#include "config_win32.h"
#include "config_unix.h"
#endif // HAVE_CONFIG_H
#include "config_msvc.h"

#include "config_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

struct config_file {
        FILE *f; ///< handle to config file, NULL if opening failed
        char file_name[1024];
};

/**
 * Returns location of default UltraGrid config file
 * @param[out] buf     buffer to be written to
 * @param[in]  buf_len length of the buffer
 * @retval     NULL    if path cannot be constructed
 * @retval     buf     pointer output buffer pointer - if successfully performed
 */
char *default_config_file(char *buf, int buf_len)
{
        const char *rc_suffix = "/.ug.rc";

        if(!getenv("HOME")) {
                return NULL;
        }

        strncpy(buf, getenv("HOME"), buf_len - 1);
        strncat(buf, rc_suffix, buf_len - 1);
        return buf;
}

/**
 * Creates config file instance
 *
 * @param  name name of the config file
 * @return newly created instance. If opening of the config file failed, empty instance
 * is returned and no error is reported!
 */
struct config_file *config_file_open(const char *name)
{
        if (name == NULL) {
                return NULL;
        }
        FILE *f = fopen(name, "r");
        if (f == NULL) {
                return NULL;
        }
        struct config_file *s = (struct config_file *) calloc(1, sizeof(struct config_file));
        s->f = f;
        strncpy(s->file_name, name, sizeof(s->file_name) - 1);
        return s;
}

void config_file_close(struct config_file *s)
{
        if (!s)
                return;
        if (s->f)
                fclose(s->f);

        free(s);
}

static string get_nth_word(struct config_file *s, const char *prefix, int index)
{
        if (!s->f)
                return NULL;

        char line[1024];
        fseek(s->f, 0, SEEK_SET); // rewind
        while (fgets(line, sizeof(line), s->f)) {
                if (strncmp(prefix, line, strlen(prefix)) == 0) {
                        char *suffix = line + strlen(prefix);
                        if (suffix[strlen(suffix) - 1] == '\n')
                                suffix[strlen(suffix) - 1] = '\0';

                        char *item, *save_ptr, *tmp;
                        tmp = suffix;
                        int i = 0;
                        while ((item = strtok_r(tmp, " ", &save_ptr))) {
                                tmp = NULL;
                                if (i++ == index) {
                                        if (strlen(item) == 0) {
                                                return NULL;
                                        }
                                        return item;
                                }
                        }
                }
        }

        return string();
}

/**
 * Returns alias for specified class and item name.
 *
 * Caller must not deallocate returned value.
 * Returned value is valid until corresponding instance of @ref config_file
 * is destroyed.
 *
 * @param s              config_file instance
 * @param req_item_class textual representation of configuration file name
 * @param requested_name requested alias
 *
 * @retval NULL    if not found
 * @retval pointer pointer to string representation of the aliased item
 */
string config_file_get_alias(struct config_file *s, const char *req_item_class,
                const char *requested_name)
{
        if (!s) {
                return {};
        }

        char prefix[1024];
        memset(prefix, 0, sizeof(prefix));

        snprintf(prefix, sizeof(prefix) - 1, "alias %s %s ",
                        req_item_class, requested_name);

        return get_nth_word(s, prefix, 0);
}

string config_file_get_capture_filter_for_alias(struct config_file *s,
                const char *alias)
{
        if (!s) {
                return {};
        }

        char prefix[1024];
        memset(prefix, 0, sizeof(prefix));

        snprintf(prefix, sizeof(prefix) - 1, "capture-filter %s ",
                        alias);

        return get_nth_word(s, prefix, 0);
}

std::list<std::pair<std::string, std::string>> get_configured_capture_aliases(struct config_file *s)
{
        if (!s) {
                return {};
        }

        fseek(s->f, 0, SEEK_SET); // rewind

        std::list<std::pair<std::string, std::string>> ret;

        char line[1024];
        char prefix[1024] = "";
        snprintf(prefix, sizeof prefix, "alias capture ");
        fseek(s->f, 0, SEEK_SET); // rewind
        while (fgets(line, sizeof(line), s->f)) {
                if (line[strlen(line) - 1] == '\n')
                        line[strlen(line) - 1] = '\0';

                if (strncasecmp(prefix, line, strlen(prefix)) == 0) {
                        char *suffix = line + strlen(prefix);
                        char *save_ptr, *item;
                        if ((item = strtok_r(suffix, " ", &save_ptr))) {
                                string name(item);
                                string description;
                                char *cfg = strtok_r(NULL, " ", &save_ptr); // skip config string
                                item = strtok_r(NULL, " ", &save_ptr); // skip config string
                                if (item) {
                                        description += item;
                                        while ((item = strtok_r(NULL, " ", &save_ptr))) {
                                                description += string(" ") + item;
                                        }
                                } else {
                                        description = cfg; // else use config string as description (grrr)
                                }
                                ret.emplace_back(name, description);
                        }
                }
        }

        return ret;
}


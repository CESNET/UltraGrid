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

#include "config_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct config_file {
        FILE *f; ///< handle to config file, NULL if opening failed
        char **tmp_buffer; ///< temporary storage for returned strings
        int tmp_buffer_count;

        char ****tmp_list_buffer; /**< temporary storage of returned lists (without actual
                                   *   strings which are stored in @ref tmp_buffer */
        int tmp_list_buffer_count;
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
        struct config_file *s = (struct config_file *) calloc(1, sizeof(struct config_file));
        s->f = fopen(name, "r");
        s->tmp_buffer = (char **) malloc(s->tmp_buffer_count * sizeof(char *));
        s->tmp_list_buffer = (char ****) malloc(s->tmp_list_buffer_count *
                        sizeof(char ***));
        return s;
}

void config_file_close(struct config_file *s)
{
        if (s->f)
                fclose(s->f);

        while (s->tmp_buffer_count > 0)
                free(s->tmp_buffer[--s->tmp_buffer_count]);
        while (s->tmp_list_buffer_count > 0) {
                char ***ptr = s->tmp_list_buffer[s->tmp_list_buffer_count - 1];
                while (*ptr) {
                        free(*ptr);
                        ptr += 1;
                }
                s->tmp_list_buffer_count -= 1;
        }
        free(s->tmp_buffer);
        free(s->tmp_list_buffer);
        free(s);
}

static char *get_line_suffix(struct config_file *s, const char *prefix)
{
        if (!s->f)
                return NULL;

        char line[1024];
        fseek(s->f, 0, SEEK_SET); // rewind
        while (fgets(line, sizeof(line), s->f)) {
                if (strncmp(prefix, line, strlen(prefix)) == 0) {
                        s->tmp_buffer_count += 1;
                        s->tmp_buffer = (char **) realloc(s->tmp_buffer,
                                        s->tmp_buffer_count * sizeof(char *));
                        char *suffix = line + strlen(prefix);
                        if (suffix[strlen(suffix) - 1] == '\n')
                                suffix[strlen(suffix) - 1] = '\0';
                        s->tmp_buffer[s->tmp_buffer_count - 1] =
                                strdup(suffix);
                        return s->tmp_buffer[s->tmp_buffer_count - 1];
                }
        }

        return NULL;
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
char *config_file_get_alias(struct config_file *s, const char *req_item_class,
                const char *requested_name)
{
        char line[1024];
        fseek(s->f, 0, SEEK_SET); // rewind
        while (fgets(line, sizeof(line), s->f)) {
                char prefix[1024];
                memset(prefix, 0, sizeof(prefix));

                snprintf(prefix, sizeof(prefix) - 1, "alias %s %s ",
                                req_item_class, requested_name);

                return get_line_suffix(s, prefix);
        }

        return NULL;
}

/**
 * Returns list of all available items for specified class
 * @param s              state
 * @param req_item_class class to which returned list belongs
 * @return
 * Null-terminated list is in format
 * [ [ alias, aliased], [alias2, aliased2] ..., NULL ].
 * Caller must not delete returned buffer. Returned values are valid
 * @ref config_file instance is deleted.
 */
char ***config_file_get_aliases_for_class(struct config_file *s,
                                const char *req_item_class)
{
        s->tmp_list_buffer_count += 1;
        s->tmp_list_buffer = (char ****) realloc(s->tmp_list_buffer,
                        s->tmp_list_buffer_count * sizeof(char ***));

        int current_len = 0;
        s->tmp_list_buffer[s->tmp_list_buffer_count - 1] =
                (char ***) calloc(1, sizeof(char **)); // null-terminated

        char ***current_buffer = s->tmp_list_buffer[s->tmp_list_buffer_count - 1];

        if (!s->f)
                return current_buffer; // empty list

        char line[1024];
        fseek(s->f, 0, SEEK_SET); // rewind
        while (fgets(line, sizeof(line), s->f)) {
                char name[128];
                char item_class[128];
                char alias[128];
                memset(name, 0, sizeof(name));
                memset(alias, 0, sizeof(alias));
                if (sscanf(line, "alias %127s %127s %511s", item_class, name, alias) != 3)
                        continue;
                if (strcasecmp(item_class, req_item_class) == 0) {
                        s->tmp_buffer_count += 1;
                        s->tmp_buffer = (char **) realloc(s->tmp_buffer,
                                        s->tmp_buffer_count * sizeof(char *));
                        s->tmp_buffer[s->tmp_buffer_count - 1] =
                                strdup(name);
                        s->tmp_buffer_count += 1;
                        s->tmp_buffer = (char **) realloc(s->tmp_buffer,
                                        s->tmp_buffer_count * sizeof(char *));
                        s->tmp_buffer[s->tmp_buffer_count - 1] =
                                strdup(alias);

                        current_buffer = (char ***) realloc(current_buffer,
                                        sizeof(char **) * (current_len + 1));
                        current_buffer[current_len] =
                               (char **) malloc(sizeof(char *) * 2);
                        current_buffer[current_len][0] =
                                s->tmp_buffer[s->tmp_buffer_count - 2];
                        current_buffer[current_len][1] =
                                s->tmp_buffer[s->tmp_buffer_count - 1];
                        current_len += 1;
                }
        }

        current_buffer[current_len] = NULL;

        return current_buffer;
}


/**
 * @file   utils/fs.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2018-2025 CESNET, zájmové sdružení právnických osob
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

#ifndef UTILS_FS_H_
#define UTILS_FS_H_

// maximal platform path length including terminating null byte
#ifdef _WIN32
#include <windef.h>            /// for MAX_PATH
#define MAX_PATH_SIZE (MAX_PATH + 1)
#define NULL_FILE "NUL"
#define PATH_SEPARATOR "\\"
#else
#include <limits.h>
#define MAX_PATH_SIZE PATH_MAX
#define NULL_FILE "/dev/null"
#define PATH_SEPARATOR "/"
#endif

#ifdef __cplusplus
#include <cstdio>
extern "C" {
#else
#include <stdbool.h>
#include <stdio.h>
#endif

enum check_file_type {
        FT_ANY,
        FT_REGULAR,
        FT_DIRECTORY,
};
bool file_exists(const char *path, enum check_file_type type);
const char *get_temp_dir(void);
FILE *get_temp_file(const char **filename);
const char *get_ug_data_path(void);
char *strdup_path_with_expansion(const char *orig_path);

#ifdef __cplusplus
} // extern "C"
#endif

#endif// UTILS_FS_H_


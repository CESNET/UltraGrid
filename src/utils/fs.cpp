/**
 * @file   utils/fs.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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

#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include "utils/fs.h"
#include "utils/string.h"

/**
 * Returns temporary path ending with path delimiter ('/' or '\' in Windows)
 */
const char *get_temp_dir(void)
{
        static __thread char temp_dir[MAX_PATH_SIZE];

        if (temp_dir[0] != '\0') {
                return temp_dir;
        }

#ifdef WIN32
        if (GetTempPathA(sizeof temp_dir, temp_dir) == 0) {
                return NULL;
        }
#else
        if (char *req_tmp_dir = getenv("TMPDIR")) {
                temp_dir[sizeof temp_dir - 1] = '\0';
                strncpy(temp_dir, req_tmp_dir, sizeof temp_dir - 1);
        } else {
                strcpy(temp_dir, P_tmpdir);
        }
        strncat(temp_dir, "/", sizeof temp_dir - strlen(temp_dir) - 1);
#endif

        return temp_dir;
}

#ifdef _WIN32
int get_exec_path(char* path) {
        return GetModuleFileNameA(NULL, path, MAX_PATH_SIZE) != 0;
}
#endif


#ifdef __linux__
int get_exec_path(char* path) {
        return realpath("/proc/self/exe", path) != NULL;
}
#endif


#ifdef __APPLE__
#include <mach-o/dyld.h> //_NSGetExecutablePath
#include <unistd.h>

int get_exec_path(char* path) {
        char raw_path_name[MAX_PATH_SIZE];
        uint32_t raw_path_size = (uint32_t)(sizeof(raw_path_name));

        if (_NSGetExecutablePath(raw_path_name, &raw_path_size) != 0) {
            return false;
        }
        return realpath(raw_path_name, path) != NULL;
}
#endif  

/**
 * @returns  installation root without trailing '/', eg. installation prefix on
 *           Linux - default "/usr/local", Windows - top-level directory extracted
 *           UltraGrid directory
 */
const char *get_install_root(void) {
        static __thread char exec_path[MAX_PATH_SIZE];
        if (!get_exec_path(exec_path)) {
                return NULL;
        }
        char *last_path_delim = strrpbrk(exec_path, "/\\");
        if (!last_path_delim) {
                return NULL;
        }
        *last_path_delim = '\0'; // cut off executable name
        last_path_delim = strrpbrk(exec_path, "/\\");
        if (!last_path_delim) {
                return exec_path;
        }
        if (strcmp(last_path_delim + 1, "bin") == 0 || strcmp(last_path_delim + 1, "MacOS") == 0) {
                *last_path_delim = '\0'; // remove "bin" suffix if there is one (not in Windows builds) or MacOS in a bundle
        }
        return exec_path;
}

/**
 * opens and returns temporary file and stores its name in filename
 *
 * Caller is resposible for both closing and unlinking the file.
 *
 * Reason for this file is because the linker complains about tmpnam as unsafe
 * thus we create a "safer" workaround (at least for POSIX systems) returning
 * both FILE pointer and file name.
 */
FILE *get_temp_file(const char **filename) {
        static thread_local char filename_buf[MAX_PATH_SIZE];
#ifdef _WIN32
        *filename = tmpnam(filename_buf);
        return fopen(*filename, "wbx");
#else
        *filename = filename_buf;
        strncpy(filename_buf, get_temp_dir(), sizeof filename_buf - 1);
        strncat(filename_buf, "/uv.XXXXXX", sizeof filename_buf - strlen(filename_buf) - 1);
        umask(S_IRWXG|S_IRWXO);
        int fd = mkstemp(filename_buf);
        return fd == -1 ? NULL : fdopen(fd, "wb");
#endif
}

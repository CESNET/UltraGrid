/**
 * @file   utils/fs.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include "utils/fs.h"

/**
 * Returns temporary path ending with path delimiter ('/' or '\' in Windows)
 */
const char *get_temp_dir(void)
{
#ifdef WIN32
        static __thread char temp_dir[MAX_PATH + 1];
#else
        static __thread char temp_dir[MAXPATHLEN + 2];
#endif

        if (temp_dir[0] != '\0') {
                return temp_dir;
        }

#ifdef WIN32
        if (GetTempPathA(sizeof temp_dir, temp_dir) == 0) {
                return NULL;
        }
#else
        strcpy(temp_dir, P_tmpdir);
        strcat(temp_dir, "/");
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





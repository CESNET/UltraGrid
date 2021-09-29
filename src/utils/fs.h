/**
 * @file   utils/fs.h
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

#ifndef UTILS_FS_H_
#define UTILS_FS_H_



#ifdef _WIN32
#include <windows.h>
#define MAX_PATH_SIZE MAX_PATH
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <limits.h>
#define MAX_PATH_SIZE (PATH_MAX + 1)
#endif


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @param path buffer with size MAX_PATH_SIZE where function stores path to executable
 * @return 1 - SUCCESS, 0 - ERROR
 */ 
int get_exec_path(char* path);


const char *get_temp_dir(void);

#ifdef __cplusplus
} // extern "C"
#endif



#ifdef __cplusplus
#include <string>

inline std::string get_executable_path(){
        std::string path(MAX_PATH_SIZE, '\0');
        if (!get_exec_path(path.data())){
            return "";
        }
        path.erase(path.find('\0'));
        return path;
}
#endif


#endif// UTILS_FS_H_


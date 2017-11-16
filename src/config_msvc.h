/**
 * @file   config_msvc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017 CESNET z.s.p.o.
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

#ifndef CONFIG_MSVC_H
#define CONFIG_MSVC_H

#if defined _MSC_VER && !defined WIN32

#include <stdlib.h>
#include <string.h>

static inline char * strtok_r(char *str, const char *delim, char **save);

/*
 * Public domain licensed code taken from:
 * http://en.wikibooks.org/wiki/C_Programming/Strings#The_strtok_function
 */
static inline char *strtok_r(char *s, const char *delimiters, char **lasts)
{
     char *sbegin, *send;
     sbegin = s ? s : *lasts;
     sbegin += strspn(sbegin, delimiters);
     if (*sbegin == '\0') {
         /* *lasts = ""; */
         *lasts = sbegin;
         return NULL;
     }
     send = sbegin + strcspn(sbegin, delimiters);
     if (*send != '\0')
         *send++ = '\0';
     *lasts = send;
     return sbegin;
}

#define ATTRIBUTE(a)

#define snprintf(x, y, ...) sprintf(x, __VA_ARGS__)
#define strncasecmp(x, y, z) strcmp(x, y)

#endif

#endif

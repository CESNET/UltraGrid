/*
 * FILE:   lib_common.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <glob.h>

#include "debug.h"

#include "lib_common.h"

extern char **uv_argv;

static void *lib_common_handle = NULL;

void open_all(const char *pattern) {
        char path[512];
        glob_t glob_buf;

        char *tmp = strdup(uv_argv[0]);
        /* binary not from $PATH */
        if(strchr(uv_argv[0], '/') != NULL) {
                char *dir = dirname(tmp);
                snprintf(path, sizeof(path), "%s/../lib/ultragrid/%s", dir, pattern);
        } else {
                snprintf(path, sizeof(path), TOSTRING(LIB_DIR) "/ultragrid/%s", pattern);
        }
        free(tmp);

        glob(path, 0, NULL, &glob_buf);

        for(unsigned int i = 0; i < glob_buf.gl_pathc; ++i) {
                if(!dlopen(glob_buf.gl_pathv[i], RTLD_NOW|RTLD_GLOBAL))
                        fprintf(stderr, "Library opening warning: %s \n", dlerror());
        }

        globfree(&glob_buf);
}

void *open_library(const char *name)
{
        void *handle = NULL;
        struct stat buf;
        char kLibName[128];
        char path[512];
        char *dir;
        char *tmp;
        
        snprintf(kLibName, sizeof(kLibName), "ultragrid/%s", name);


        /* firstly expect we are opening from a build */
        tmp = strdup(uv_argv[0]);
        /* binary not from $PATH */
        if(strchr(tmp, '/') != NULL) {
                dir = dirname(tmp);
                snprintf(path, sizeof(path), "%s/../lib/%s", dir, kLibName);
                if(!handle && stat(path, &buf) == 0) {
                        handle = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
                        if(!handle)
                                fprintf(stderr, "Library opening warning: %s \n", dlerror());
                }
        }
        free(tmp);

        /* next try $LIB_DIR/ultragrid */
        snprintf(path, sizeof(path), TOSTRING(LIB_DIR) "/%s", kLibName);
        if(!handle && stat(path, &buf) == 0) {
                handle = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
                if(!handle)
                        fprintf(stderr, "Library opening warning: %s \n", dlerror());
        }
        
        if(!handle) {
                fprintf(stderr, "Unable to load %s library.\n", kLibName);
        }
                
        return handle;
}

void init_lib_common(void)
{
        char name[128];
        snprintf(name, sizeof(name), "ug_lib_common.so.%d", COMMON_LIB_ABI_VERSION);

        lib_common_handle = open_library(name);
}

void lib_common_done(void)
{
        if(lib_common_handle) {
                dlclose(lib_common_handle);
        }
}


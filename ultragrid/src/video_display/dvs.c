/*
 * FILE:    video_display/dvs.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          Colin Perkins    <csp@isi.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2003 University of Southern California
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifdef HAVE_DVS           /* From config.h */

#include <dlfcn.h>
#include <pthread.h>
#include "video_display/dvs.h"
#include "video_codec.h"
#include "dvs_clib.h"           /* From the DVS SDK */
#include <sys/stat.h>

extern char ** uv_argv;

#define kLibName "uv_dvs_lib.so"

static pthread_once_t DVSLibraryLoad = PTHREAD_ONCE_INIT;

static void loadLibrary(void);

typedef void *(*display_dvs_init_t)(char *fmt, unsigned int flags);
typedef void (*display_dvs_run_t)(void *state);
typedef void (*display_dvs_done_t)(void *state);
typedef struct video_frame *(*display_dvs_getf_t)(void *state);
typedef int (*display_dvs_putf_t)(void *state, char *frame);
typedef struct audio_frame * (*display_dvs_get_audio_frame_t)(void *state);
typedef void (*display_dvs_put_audio_frame_t)(void *state, struct audio_frame *frame);
typedef void (*display_dvs_reconfigure_t)(void *state,
                                struct video_desc desc);
typedef int (*display_dvs_get_property_t)(void *state, int property, void *val, int *len);

static display_dvs_init_t display_dvs_init_func = NULL;
static display_dvs_run_t display_dvs_run_func = NULL;
static display_dvs_done_t display_dvs_done_func = NULL;
static display_dvs_getf_t display_dvs_getf_func = NULL;
static display_dvs_putf_t display_dvs_putf_func = NULL;
static display_dvs_reconfigure_t display_dvs_reconfigure_func = NULL;
static display_dvs_get_audio_frame_t display_dvs_get_audio_frame_func = NULL;
static display_dvs_put_audio_frame_t display_dvs_put_audio_frame_func = NULL;
static display_dvs_get_property_t display_dvs_get_property_func = NULL;

display_type_t *display_dvs_probe(void)
{
        display_type_t *dtype;

        dtype = malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_DVS_ID;
                dtype->name = "dvs";
                dtype->description = "DVS card";
        }
        return dtype;
}

void * openDVSLibrary()
{
        void *handle = NULL;
        struct stat buf;
        

        handle = dlopen(kLibName, RTLD_NOW|RTLD_GLOBAL);
        if(!handle)
                fprintf(stderr, "Library opening error: %s \n", dlerror());

        
        if(!handle && stat("/usr/local/lib/" kLibName, &buf) == 0) {
                handle = dlopen("/usr/local/lib/" kLibName, RTLD_NOW|RTLD_GLOBAL);
                if(!handle)
                        fprintf(stderr, "Library opening error: %s \n", dlerror());
        }
        
        if(!handle && stat("./" kLibName, &buf) == 0) {
                handle = dlopen("./" kLibName, RTLD_NOW|RTLD_GLOBAL);
                if(!handle)
                        fprintf(stderr, "Library opening error: %s \n", dlerror());
        }
        if(!handle && stat("./lib/" kLibName, &buf) == 0) {
                handle = dlopen("./lib/" kLibName, RTLD_NOW|RTLD_GLOBAL);
                if(!handle)
                        fprintf(stderr, "Library opening error: %s \n", dlerror());
        }
        
        if(!handle) {
                char *path_to_self = strdup(uv_argv[0]);
                char *path;
                if(strrchr(path_to_self, '/')) {
                        *strrchr(path_to_self, '/') = '\0';
                        path = malloc(strlen(path_to_self) +
                                        strlen("/../lib/") +
                                        strlen(kLibName) + 1);
                        strcpy(path, path_to_self);
                        strcat(path, "/../lib/");
                        strcat(path, kLibName);
                } else {
                        path = strdup("../lib/");
                }
                if (stat(path, &buf) == 0) {
                        handle = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
                        if(!handle)
                                fprintf(stderr, "Library opening error: %s \n", dlerror());
                }

                free(path);
                free(path_to_self);
        }
                
        return handle;
        
}

static void loadLibrary()
{
        void *handle = NULL;
        
        handle = openDVSLibrary();
        
        display_dvs_init_func = (display_dvs_init_t) dlsym(handle,
                        "display_dvs_init_impl");
        display_dvs_run_func = (display_dvs_run_t) dlsym(handle,
                        "display_dvs_run_impl");
        display_dvs_done_func = (display_dvs_done_t) dlsym(handle,
                        "display_dvs_done_impl");
        display_dvs_getf_func = (display_dvs_getf_t) dlsym(handle,
                        "display_dvs_getf_impl");
        display_dvs_putf_func = (display_dvs_putf_t) dlsym(handle,
                        "display_dvs_putf_impl");
        display_dvs_reconfigure_func = (display_dvs_reconfigure_t) dlsym(handle,
                        "display_dvs_reconfigure_impl");
        display_dvs_get_property_func = (display_dvs_get_property_t) dlsym(handle,
                        "display_dvs_get_property_impl");
        display_dvs_get_audio_frame_func = (display_dvs_get_audio_frame_t)
                        dlsym(handle, "display_dvs_get_audio_frame_impl");
        display_dvs_put_audio_frame_func = (display_dvs_put_audio_frame_t)
                        dlsym(handle, "display_dvs_put_audio_frame_impl");
}

void display_dvs_run(void *arg)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_run_func == NULL)
                return;
        display_dvs_run_func(arg);
}

struct video_frame *
display_dvs_getf(void *state)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_getf_func == NULL)
                return NULL;
        return display_dvs_getf_func(state);
}

int display_dvs_putf(void *state, char *frame)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_putf_func == NULL)
                return 0;
        return display_dvs_putf_func(state, frame);
}

void *display_dvs_init(char *fmt, unsigned int flags)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_init_func == NULL)
                return NULL;
        return display_dvs_init_func(fmt, flags);
}

void display_dvs_done(void *state)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_done_func == NULL)
                return;
        display_dvs_done_func(state);
}

struct audio_frame * display_dvs_get_audio_frame(void *state)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_get_audio_frame_func == NULL)
                return NULL;
        return display_dvs_get_audio_frame_func(state);
}

void display_dvs_put_audio_frame(void *state, struct audio_frame *frame)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_put_audio_frame_func == NULL)
                return;
        display_dvs_put_audio_frame_func(state, frame);
}

void display_dvs_reconfigure(void *state,
                                struct video_desc desc)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_reconfigure_func == NULL)
                return;
        display_dvs_reconfigure_func(state, desc);
}

int display_dvs_get_property(void *state, int property, void *val, int *len)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (display_dvs_get_property_func == NULL)
                return FALSE;
        return display_dvs_get_property_func(state, property, val, len);
}


#endif                          /* HAVE_DVS */

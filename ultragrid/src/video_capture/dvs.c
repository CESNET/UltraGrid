/*
 * FILE:   video_capture/dvs.c
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

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifdef HAVE_DVS           /* From config.h */

#include <dlfcn.h>
#include <pthread.h>
#include "video_capture.h"
#include "video_capture/dvs.h"
#include "video_codec.h"
#include "video_display/dvs.h"

#define kLibName "uv_dvs_lib.so"

static pthread_once_t DVSLibraryLoad = PTHREAD_ONCE_INIT;

static void loadLibrary(void);

typedef void *(*vidcap_dvs_init_t)(char *fmt, unsigned int flags);
typedef void (*vidcap_dvs_done_t)(void *state);
typedef struct video_frame *(*vidcap_dvs_grab_t)(void *state, struct audio_frame **audio);

static vidcap_dvs_init_t vidcap_dvs_init_func = NULL;
static vidcap_dvs_done_t vidcap_dvs_done_func = NULL;
static vidcap_dvs_grab_t vidcap_dvs_grab_func = NULL;

static void loadLibrary()
{
        void *handle = NULL;
        
        /* defined in src/video_display/dvs.c */
        handle = openDVSLibrary();
        
        vidcap_dvs_init_func = (vidcap_dvs_init_t) dlsym(handle,
                        "vidcap_dvs_init_impl");
        vidcap_dvs_done_func = (vidcap_dvs_done_t) dlsym(handle,
                        "vidcap_dvs_done_impl");
        vidcap_dvs_grab_func = (vidcap_dvs_grab_t) dlsym(handle,
                        "vidcap_dvs_grab_impl");
}

/* External API ***********************************************************************************/

struct vidcap_type *vidcap_dvs_probe(void)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = VIDCAP_DVS_ID;
                vt->name = "dvs";
                vt->description = "DVS (SMPTE 274M/25i)";
        }
        return vt;
}

void *vidcap_dvs_init(char *fmt, unsigned int flags)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (vidcap_dvs_init_func == NULL)
                return NULL;
        return vidcap_dvs_init_func(fmt, flags);
}

void vidcap_dvs_done(void *state)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (vidcap_dvs_done_func == NULL)
                return;
        vidcap_dvs_done_func(state);
}

struct video_frame *vidcap_dvs_grab(void *state, struct audio_frame **audio)
{
        pthread_once(&DVSLibraryLoad, loadLibrary);
        
        if (vidcap_dvs_grab_func == NULL)
                return NULL;
        return vidcap_dvs_grab_func(state, audio);
}

#endif                          /* HAVE_DVS */

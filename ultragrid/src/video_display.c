/*
 * FILE:   video_display.c
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "video_display.h"

#include "video_display/aggregate.h"
#include "video_display/null.h"
#include "video_display/sdl.h"
#include "video_display/decklink.h"
#include "video_display/deltacast.h"
#include "video_display/dvs.h"
#include "video_display/gl.h"
#include "video_display/quicktime.h"
#include "video_display/sage.h"
#include "lib_common.h"

extern char **uv_argv;

/*
 * Interface to probing the valid display types. 
 *
 */

typedef struct {
        display_id_t              id;
        const  char              *library_name;
        display_type_t         *(*func_probe) (void);
        const char               *func_probe_str;
        void                   *(*func_init) (char *fmt, unsigned int flags);
        const char               *func_init_str;
        void                    (*func_run) (void *state);
        const char               *func_run_str;
        void                    (*func_done) (void *state);
        const char               *func_done_str;
        void                    (*func_finish) (void *state);
        const char               *func_finish_str;
        struct video_frame     *(*func_getf) (void *state);
        const char               *func_getf_str;
        int                     (*func_putf) (void *state, char *frame);
        const char               *func_putf_str;
        int                     (*func_reconfigure)(void *state, struct video_desc desc);
        const char               *func_reconfigure_str;
        int                     (*func_get_property)(void *state, int property, void *val, size_t *len);
        const char               *func_get_property_str;
        
        struct audio_frame     *(*func_get_audio_frame) (void *state);
        const char               *func_get_audio_frame_str;
        void                    (*func_put_audio_frame) (void *state, struct audio_frame *frame);
        const char               *func_put_audio_frame_str;
        int                     (*func_reconfigure_audio) (void *state, int quant_samples, int channels,
                        int sample_rate);
        const char               *func_reconfigure_audio_str;

        void                     *handle;
} display_table_t;

static display_table_t display_device_table[] = {
        {
         0,
         NULL,
         MK_STATIC(display_aggregate_probe),
         MK_STATIC(display_aggregate_init),
         MK_STATIC(display_aggregate_run),
         MK_STATIC(display_aggregate_done),
         MK_STATIC(display_aggregate_finish),
         MK_STATIC(display_aggregate_getf),
         MK_STATIC(display_aggregate_putf),
         MK_STATIC(display_aggregate_reconfigure),
         MK_STATIC(display_aggregate_get_property),
         MK_STATIC(display_aggregate_get_audio_frame),
         MK_STATIC(display_aggregate_put_audio_frame),
         MK_STATIC(display_aggregate_reconfigure_audio),
         NULL
         },
#if defined HAVE_SDL || defined BUILD_LIBRARIES
        {
         0,
         "sdl",
         MK_NAME(display_sdl_probe),
         MK_NAME(display_sdl_init),
         MK_NAME(display_sdl_run),
         MK_NAME(display_sdl_done),
         MK_NAME(display_sdl_finish),
         MK_NAME(display_sdl_getf),
         MK_NAME(display_sdl_putf),
         MK_NAME(display_sdl_reconfigure),
         MK_NAME(display_sdl_get_property),
         MK_NAME(display_sdl_get_audio_frame),
         MK_NAME(display_sdl_put_audio_frame),
         MK_NAME(display_sdl_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_SDL */
#if defined HAVE_GL || defined BUILD_LIBRARIES
        {
         0,
         "gl",
         MK_NAME(display_gl_probe),
         MK_NAME(display_gl_init),
         MK_NAME(display_gl_run),
         MK_NAME(display_gl_done),
         MK_NAME(display_gl_finish),
         MK_NAME(display_gl_getf),
         MK_NAME(display_gl_putf),
         MK_NAME(display_gl_reconfigure),
         MK_NAME(display_gl_get_property),
         MK_NAME(display_gl_get_audio_frame),
         MK_NAME(display_gl_put_audio_frame),
         MK_NAME(display_gl_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_GL */
#if defined HAVE_SAGE || defined BUILD_LIBRARIES
        {
         0,
         "sage",
         MK_NAME(display_sage_probe),
         MK_NAME(display_sage_init),
         MK_NAME(display_sage_run),
         MK_NAME(display_sage_done),
         MK_NAME(display_sage_finish),
         MK_NAME(display_sage_getf),
         MK_NAME(display_sage_putf),
         MK_NAME(display_sage_reconfigure),
         MK_NAME(display_sage_get_property),
         MK_NAME(display_sage_get_audio_frame),
         MK_NAME(display_sage_put_audio_frame),
         MK_NAME(display_sage_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_SAGE */
#if defined HAVE_DECKLINK || defined BUILD_LIBRARIES
        {
         0,
         "decklink",
         MK_NAME(display_decklink_probe),
         MK_NAME(display_decklink_init),
         MK_NAME(display_decklink_run),
         MK_NAME(display_decklink_done),
         MK_NAME(display_decklink_finish),
         MK_NAME(display_decklink_getf),
         MK_NAME(display_decklink_putf),
         MK_NAME(display_decklink_reconfigure),
         MK_NAME(display_decklink_get_property),
         MK_NAME(display_decklink_get_audio_frame),
         MK_NAME(display_decklink_put_audio_frame),
         MK_NAME(display_decklink_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_DECKLINK */
#if defined HAVE_DELTACAST || defined BUILD_LIBRARIES
        {
         0,
         "deltacast",
         MK_NAME(display_deltacast_probe),
         MK_NAME(display_deltacast_init),
         MK_NAME(display_deltacast_run),
         MK_NAME(display_deltacast_done),
         MK_NAME(display_deltacast_finish),
         MK_NAME(display_deltacast_getf),
         MK_NAME(display_deltacast_putf),
         MK_NAME(display_deltacast_reconfigure),
         MK_NAME(display_deltacast_get_property),
         MK_NAME(display_deltacast_get_audio_frame),
         MK_NAME(display_deltacast_put_audio_frame),
         MK_NAME(display_deltacast_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_DELTACAST */
#if defined HAVE_DVS || defined BUILD_LIBRARIES
        {
         0,
         "dvs",
         MK_NAME(display_dvs_probe),
         MK_NAME(display_dvs_init),
         MK_NAME(display_dvs_run),
         MK_NAME(display_dvs_done),
         MK_NAME(display_dvs_finish),
         MK_NAME(display_dvs_getf),
         MK_NAME(display_dvs_putf),
         MK_NAME(display_dvs_reconfigure),
         MK_NAME(display_dvs_get_property),
         MK_NAME(display_dvs_get_audio_frame),
         MK_NAME(display_dvs_put_audio_frame),
         MK_NAME(display_dvs_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_DVS */
#ifdef HAVE_MACOSX
        {
         0,
         "quicktime",
         MK_NAME(display_quicktime_probe),
         MK_NAME(display_quicktime_init),
         MK_NAME(display_quicktime_run),
         MK_NAME(display_quicktime_done),
         MK_NAME(display_quicktime_finish),
         MK_NAME(display_quicktime_getf),
         MK_NAME(display_quicktime_putf),
         MK_NAME(display_quicktime_reconfigure),
         MK_NAME(display_quicktime_get_property),
         MK_NAME(display_quicktime_get_audio_frame),
         MK_NAME(display_quicktime_put_audio_frame),
         MK_NAME(display_quicktime_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_MACOSX */
        {
         0,
         NULL,
         MK_STATIC(display_null_probe),
         MK_STATIC(display_null_init),
         MK_STATIC(display_null_run),
         MK_STATIC(display_null_done),
         MK_STATIC(display_null_finish),
         MK_STATIC(display_null_getf),
         MK_STATIC(display_null_putf),
         MK_STATIC(display_null_reconfigure),
         MK_STATIC(display_null_get_property),
         MK_STATIC(display_null_get_audio_frame),
         MK_STATIC(display_null_put_audio_frame),
         MK_STATIC(display_null_reconfigure_audio),
         NULL
         }
};

#define DISPLAY_DEVICE_TABLE_SIZE (sizeof(display_device_table) / sizeof(display_table_t))

static display_type_t *available_devices[DISPLAY_DEVICE_TABLE_SIZE];
static int available_device_count = 0;

#ifdef BUILD_LIBRARIES
static int display_fill_symbols(display_table_t *device);
static void *display_open_library(const char *display_name);

void *open_library(const char *name);
static void *display_open_library(const char *display_name)
{
        char name[128];
        snprintf(name, sizeof(name), "display_%s.so.%d", display_name, VIDEO_DISPLAY_ABI_VERSION);

        return open_library(name);
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
        dir = dirname(tmp);
        if(strcmp(dir, ".") != 0) {
                snprintf(path, sizeof(path), "%s/../lib/%s", dir, kLibName);
                if(!handle && stat(path, &buf) == 0) {
                        handle = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
                        if(!handle)
                                fprintf(stderr, "Library opening error: %s \n", dlerror());
                }
        }
        free(tmp);

        /* next try $LIB_DIR/ultragrid */
        snprintf(path, sizeof(path), TOSTRING(LIB_DIR) "/%s", kLibName);
        if(!handle && stat(path, &buf) == 0) {
                handle = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
                if(!handle)
                        fprintf(stderr, "Library opening error: %s \n", dlerror());
        }
        
        if(!handle) {
                fprintf(stderr, "Unable to find %s library.\n", kLibName);
        }
                
        return handle;
}

static int display_fill_symbols(display_table_t *device)
{
        void *handle = device->handle;

        device->func_probe = (display_type_t *(*) (void))
                dlsym(handle, device->func_probe_str);
        device->func_init = (void *(*) (char *, unsigned int))
                dlsym(handle, device->func_init_str);
        device->func_run = (void (*) (void *))
                dlsym(handle, device->func_run_str);
        device->func_done = (void (*) (void *))
                dlsym(handle, device->func_done_str);
        device->func_finish = (void (*) (void *))
                dlsym(handle, device->func_finish_str);
        device->func_getf = (struct video_frame *(*) (void *))
                dlsym(handle, device->func_getf_str);
        device->func_putf = (int (*) (void *, char *))
                dlsym(handle, device->func_putf_str);
        device->func_reconfigure = (int (*)(void *, struct video_desc))
                dlsym(handle, device->func_reconfigure_str);
        device->func_get_property = (int (*)(void *, int, void *, size_t *))
                dlsym(handle, device->func_get_property_str);
        
        device->func_get_audio_frame = (struct audio_frame *(*) (void *))
                dlsym(handle, device->func_get_audio_frame_str);
        device->func_put_audio_frame = (void (*) (void *, struct audio_frame *))
                dlsym(handle, device->func_put_audio_frame_str);
        device->func_reconfigure_audio = (int (*) (void *, int, int,
                        int))
                dlsym(handle, device->func_reconfigure_audio_str);

        if(!device->func_probe || !device->func_init || !device->func_run ||
                        !device->func_done || !device->func_finish ||
                        !device->func_getf || !device->func_getf ||
                        !device->func_putf || !device->func_reconfigure ||
                        !device->func_get_property || !device->func_get_audio_frame ||
                        !device->func_put_audio_frame || !device->func_reconfigure_audio) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }

        return TRUE;
}
#endif

int display_init_devices(void)
{
        unsigned int i;
        display_type_t *dt;

        assert(available_device_count == 0);

        for (i = 0; i < DISPLAY_DEVICE_TABLE_SIZE; i++) {
#ifdef BUILD_LIBRARIES
                display_device_table[i].handle = NULL;
                if(display_device_table[i].library_name) {
                        display_device_table[i].handle =
                                display_open_library(display_device_table[i].library_name);
                        if(display_device_table[i].handle) {
                                int ret;
                                ret = display_fill_symbols(&display_device_table[i]);
                                if(!ret) continue;
                        } else {
                                continue;
                        }
                }
#endif
                dt = display_device_table[i].func_probe();
                if (dt != NULL) {
                        display_device_table[i].id = dt->id;
                        available_devices[available_device_count++] = dt;
                }
        }
        return 0;
}

void display_free_devices(void)
{
        int i;

        for (i = 0; i < available_device_count; i++) {
                free(available_devices[i]);
                available_devices[i] = NULL;
        }
        available_device_count = 0;
}

int display_get_device_count(void)
{
        return available_device_count;
}

display_type_t *display_get_device_details(int index)
{
        assert(index < available_device_count);
        assert(available_devices[index] != NULL);

        return available_devices[index];
}

display_id_t display_get_null_device_id(void)
{
        return DISPLAY_NULL_ID;
}

/*
 * Display initialisation and playout routines...
 */

#define DISPLAY_MAGIC 0x01ba7ef1

struct display {
        uint32_t magic;
        int index;
        void *state;
};

struct display *display_init(display_id_t id, char *fmt, unsigned int flags)
{
        unsigned int i;

        for (i = 0; i < DISPLAY_DEVICE_TABLE_SIZE; i++) {
                if (display_device_table[i].id == id) {
                        struct display *d =
                            (struct display *)malloc(sizeof(struct display));
                        d->magic = DISPLAY_MAGIC;
                        d->state = display_device_table[i].func_init(fmt, flags);
                        d->index = i;
                        if (d->state == NULL) {
                                debug_msg("Unable to start display 0x%08lx\n",
                                          id);
                                free(d);
                                return NULL;
                        }
                        return d;
                }
        }
        debug_msg("Unknown display id: 0x%08x\n", id);
        return NULL;
}

void display_finish(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_finish(d->state);
}

void display_done(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_done(d->state);
}

void display_run(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_run(d->state);
}

struct video_frame *display_get_frame(struct display *d)
{
        perf_record(UVP_GETFRAME, d);
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_getf(d->state);
}

void display_put_frame(struct display *d, char *frame)
{
        perf_record(UVP_PUTFRAME, frame);
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_putf(d->state, frame);
}

int display_reconfigure(struct display *d, struct video_desc desc)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_reconfigure(d->state, desc);
}

int display_get_property(struct display *d, int property, void *val, size_t *len)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_get_property(d->state, property, val, len);
}

struct audio_frame *display_get_audio_frame(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_get_audio_frame(d->state);
}

void display_put_audio_frame(struct display *d, struct audio_frame *frame)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_put_audio_frame(d->state, frame);
}

int display_reconfigure_audio(struct display *d, int quant_samples, int channels, int sample_rate)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_reconfigure_audio(d->state, quant_samples, channels, sample_rate);
}


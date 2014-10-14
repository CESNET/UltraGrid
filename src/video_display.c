/**
 * @file   video_display.c
 * @author Colin Perkins    <csp@isi.edu>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup display
 */
/*
 * Copyright (c) 2001-2003 University of Southern California
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
#include "video_display/bluefish444.h"
#include "video_display/null.h"
#include "video_display/sdl.h"
#include "video_display/decklink.h"
#include "video_display/deltacast.h"
#include "video_display/dvs.h"
#include "video_display/gl.h"
#include "video_display/pipe.h"
#include "video_display/proxy.h"
#include "video_display/quicktime.h"
#include "video_display/sage.h"
#include "lib_common.h"

#include <string.h>

#define DISPLAY_MAGIC 0x01ba7ef1

/// @brief This struct represents initialized video display state.
struct display {
        uint32_t magic; ///< state of the created video capture driver
        int index;      ///< index to @ref display_device_table
        void *state;    ///< For debugging. Conatins @ref DISPLAY_MAGIC
};

/**This variable represents a pseudostate and may be returned when initialization
 * of module was successful but no state was created (eg. when driver had displayed help). */
int display_init_noerr;

/**
 * This struct describes individual vidcap modules
 * @copydetails decoder_table_t
 */
typedef struct {
        display_id_t              id;           ///< @copydoc decoder_table_t::magic
        const  char              *library_name; ///< @copydoc decoder_table_t::library_name
        display_type_t         *(*func_probe) (void);
        const char               *func_probe_str;
        void                   *(*func_init) (const char *fmt, unsigned int flags);
        const char               *func_init_str;
        void                    (*func_run) (void *state);
        const char               *func_run_str;
        void                    (*func_done) (void *state);
        const char               *func_done_str;
        struct video_frame     *(*func_getf) (void *state);
        const char               *func_getf_str;
        int                     (*func_putf) (void *state, struct video_frame *frame, int nonblock);
        const char               *func_putf_str;
        int                     (*func_reconfigure)(void *state, struct video_desc desc);
        const char               *func_reconfigure_str;
        int                     (*func_get_property)(void *state, int property, void *val, size_t *len);
        const char               *func_get_property_str;
        
        void                    (*func_put_audio_frame) (void *state, struct audio_frame *frame);
        const char               *func_put_audio_frame_str;
        int                     (*func_reconfigure_audio) (void *state, int quant_samples, int channels,
                        int sample_rate);
        const char               *func_reconfigure_audio_str;

        void                     *handle; ///< @copydoc decoder_table_t::handle
} display_table_t;

/**
 * @brief This table contains list of video capture devices compiled with this UltraGrid version.
 * @copydetails decoders
 */
static display_table_t display_device_table[] = {
#ifndef UV_IN_YURI
        {
         0,
         NULL,
         MK_STATIC(display_aggregate_probe),
         MK_STATIC(display_aggregate_init),
         MK_STATIC(display_aggregate_run),
         MK_STATIC(display_aggregate_done),
         MK_STATIC(display_aggregate_getf),
         MK_STATIC(display_aggregate_putf),
         MK_STATIC(display_aggregate_reconfigure),
         MK_STATIC(display_aggregate_get_property),
         MK_STATIC(display_aggregate_put_audio_frame),
         MK_STATIC(display_aggregate_reconfigure_audio),
         NULL
         },
#if defined HAVE_BLUEFISH444 || defined BUILD_LIBRARIES
        {
         0,
         "bluefish444",
         MK_NAME(display_bluefish444_probe),
         MK_NAME(display_bluefish444_init),
         MK_NAME(display_bluefish444_run),
         MK_NAME(display_bluefish444_done),
         MK_NAME(display_bluefish444_getf),
         MK_NAME(display_bluefish444_putf),
         MK_NAME(display_bluefish444_reconfigure),
         MK_NAME(display_bluefish444_get_property),
         MK_NAME(display_bluefish444_put_audio_frame),
         MK_NAME(display_bluefish444_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_SDL */
#if defined HAVE_SDL || defined BUILD_LIBRARIES
        {
         0,
         "sdl",
         MK_NAME(display_sdl_probe),
         MK_NAME(display_sdl_init),
         MK_NAME(display_sdl_run),
         MK_NAME(display_sdl_done),
         MK_NAME(display_sdl_getf),
         MK_NAME(display_sdl_putf),
         MK_NAME(display_sdl_reconfigure),
         MK_NAME(display_sdl_get_property),
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
         MK_NAME(display_gl_getf),
         MK_NAME(display_gl_putf),
         MK_NAME(display_gl_reconfigure),
         MK_NAME(display_gl_get_property),
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
         MK_NAME(display_sage_getf),
         MK_NAME(display_sage_putf),
         MK_NAME(display_sage_reconfigure),
         MK_NAME(display_sage_get_property),
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
         MK_NAME(display_decklink_getf),
         MK_NAME(display_decklink_putf),
         MK_NAME(display_decklink_reconfigure),
         MK_NAME(display_decklink_get_property),
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
         MK_NAME(display_deltacast_getf),
         MK_NAME(display_deltacast_putf),
         MK_NAME(display_deltacast_reconfigure),
         MK_NAME(display_deltacast_get_property),
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
         MK_NAME(display_dvs_getf),
         MK_NAME(display_dvs_putf),
         MK_NAME(display_dvs_reconfigure),
         MK_NAME(display_dvs_get_property),
         MK_NAME(display_dvs_put_audio_frame),
         MK_NAME(display_dvs_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_DVS */
#ifdef HAVE_QUICKTIME
        {
         0,
         "quicktime",
         MK_NAME(display_quicktime_probe),
         MK_NAME(display_quicktime_init),
         MK_NAME(display_quicktime_run),
         MK_NAME(display_quicktime_done),
         MK_NAME(display_quicktime_getf),
         MK_NAME(display_quicktime_putf),
         MK_NAME(display_quicktime_reconfigure),
         MK_NAME(display_quicktime_get_property),
         MK_NAME(display_quicktime_put_audio_frame),
         MK_NAME(display_quicktime_reconfigure_audio),
         NULL
         },
#endif                          /* HAVE_MACOSX */
#endif
        {
         0,
         NULL,
         MK_STATIC(display_null_probe),
         MK_STATIC(display_null_init),
         MK_STATIC(display_null_run),
         MK_STATIC(display_null_done),
         MK_STATIC(display_null_getf),
         MK_STATIC(display_null_putf),
         MK_STATIC(display_null_reconfigure),
         MK_STATIC(display_null_get_property),
         MK_STATIC(display_null_put_audio_frame),
         MK_STATIC(display_null_reconfigure_audio),
         NULL
         },
        {
         0,
         "pipe",
         MK_STATIC(display_pipe_probe),
         MK_STATIC(display_pipe_init),
         MK_STATIC(display_pipe_run),
         MK_STATIC(display_pipe_done),
         MK_STATIC(display_pipe_getf),
         MK_STATIC(display_pipe_putf),
         MK_STATIC(display_pipe_reconfigure),
         MK_STATIC(display_pipe_get_property),
         MK_STATIC(display_pipe_put_audio_frame),
         MK_STATIC(display_pipe_reconfigure_audio),
         NULL
         },
        {
         0,
         "proxy",
         MK_STATIC(display_proxy_probe),
         MK_STATIC(display_proxy_init),
         MK_STATIC(display_proxy_run),
         MK_STATIC(display_proxy_done),
         MK_STATIC(display_proxy_getf),
         MK_STATIC(display_proxy_putf),
         MK_STATIC(display_proxy_reconfigure),
         MK_STATIC(display_proxy_get_property),
         MK_STATIC(display_proxy_put_audio_frame),
         MK_STATIC(display_proxy_reconfigure_audio),
         NULL
         },
};

#define DISPLAY_DEVICE_TABLE_SIZE (sizeof(display_device_table) / sizeof(display_table_t))

/** @brief List of available display devices
 * Initialized with @ref display_init_devices */
static display_type_t *available_display_devices[DISPLAY_DEVICE_TABLE_SIZE];
/** @brief Count of @ref available_display_devices
 * Initialized with @ref display_init_devices */
static int available_display_device_count = 0;

#ifdef BUILD_LIBRARIES
/** Opens display library of given name. */
static void *display_open_library(const char *display_name)
{
        char name[128];
        snprintf(name, sizeof(name), "display_%s.so.%d", display_name, VIDEO_DISPLAY_ABI_VERSION);

        return open_library(name);
}

/** For a given device, load individual functions from library handle (previously opened). */
static int display_fill_symbols(display_table_t *device)
{
        void *handle = device->handle;

        device->func_probe = (display_type_t *(*) (void))
                dlsym(handle, device->func_probe_str);
        device->func_init = (void *(*) (const char *, unsigned int))
                dlsym(handle, device->func_init_str);
        device->func_run = (void (*) (void *))
                dlsym(handle, device->func_run_str);
        device->func_done = (void (*) (void *))
                dlsym(handle, device->func_done_str);
        device->func_getf = (struct video_frame *(*) (void *))
                dlsym(handle, device->func_getf_str);
        device->func_putf = (int (*) (void *, struct video_frame *, int))
                dlsym(handle, device->func_putf_str);
        device->func_reconfigure = (int (*)(void *, struct video_desc))
                dlsym(handle, device->func_reconfigure_str);
        device->func_get_property = (int (*)(void *, int, void *, size_t *))
                dlsym(handle, device->func_get_property_str);
        
        device->func_put_audio_frame = (void (*) (void *, struct audio_frame *))
                dlsym(handle, device->func_put_audio_frame_str);
        device->func_reconfigure_audio = (int (*) (void *, int, int,
                        int))
                dlsym(handle, device->func_reconfigure_audio_str);

        if(!device->func_probe || !device->func_init || !device->func_run ||
                        !device->func_done ||
                        !device->func_getf || !device->func_getf ||
                        !device->func_putf || !device->func_reconfigure ||
                        !device->func_get_property ||
                        !device->func_put_audio_frame || !device->func_reconfigure_audio) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }

        return TRUE;
}
#endif

void list_video_display_devices()
{
        int i;
        display_type_t *dt;

        printf("Available display devices:\n");
        display_init_devices();
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                printf("\t%s\n", dt->name);
        }
        display_free_devices();
}

int initialize_video_display(const char *requested_display,
                const char *fmt, unsigned int flags,
                struct display **out)
{
        display_type_t *dt;
        display_id_t id = 0;
        int i;

        if(!strcmp(requested_display, "none"))
                 id = display_get_null_device_id();

        if (display_init_devices() != 0) {
                printf("Unable to initialise devices\n");
                abort();
        } else {
                debug_msg("Found %d display devices\n",
                          display_get_device_count());
        }
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                if (strcmp(requested_display, dt->name) == 0) {
                        id = dt->id;
                        debug_msg("Found device\n");
                        break;
                } else {
                        debug_msg("Device %s does not match %s\n", dt->name,
                                  requested_display);
                }
        }
        if(i == display_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
                return -1;
        }
        display_free_devices();

        return display_init(id, fmt, flags, out);
}

/**
 * Must be called to initialize list of display devices before actual
 * video display initialization.
 *
 * In modular UltraGrid build, it also opens available libraries.
 */
int display_init_devices(void)
{
        unsigned int i;
        display_type_t *dt;

        assert(available_display_device_count == 0);

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
                        available_display_devices[available_display_device_count++] = dt;
                }
        }
        return 0;
}

/** Should be called after video display is initialized. */
void display_free_devices(void)
{
        int i;

        for (i = 0; i < available_display_device_count; i++) {
                free(available_display_devices[i]);
                available_display_devices[i] = NULL;
        }
        available_display_device_count = 0;
}

/** Returns number of available display devices */
int display_get_device_count(void)
{
        return available_display_device_count;
}

/** Returns metadata about specified device */
display_type_t *display_get_device_details(int index)
{
        assert(index < available_display_device_count);
        assert(available_display_devices[index] != NULL);

        return available_display_devices[index];
}

/** Returns null device */
display_id_t display_get_null_device_id(void)
{
        return DISPLAY_NULL_ID;
}

/*
 * Display initialisation and playout routines...
 */

/**
 * @brief Initializes video display.
 * @param[in] id     video display identifier that will be initialized
 * @param[in] fmt    command-line entered format string
 * @param[in] flags  bit sum of @ref display_flags
 * @param[out] state output display state. Defined only if initialization was successful.
 * @retval    0  if sucessful
 * @retval   -1  if failed
 * @retval    1  if successfully shown help (no state returned)
 */
int display_init(display_id_t id, const char *fmt, unsigned int flags, struct display **state)
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
                                return -1;
                        } else if (d->state == &display_init_noerr) {
                                free(d);
                                return 1;
                        }
                        *state = d;
                        return 0;
                }
        }
        debug_msg("Unknown display id: 0x%08x\n", id);
        return -1;
}

/**
 * @brief This function performs cleanup after done.
 * @param d display do be destroyed (must not be NULL)
 */
void display_done(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_done(d->state);
        free(d);
}

/**
 * @brief Display mainloop function.
 *
 * This call is entered in main thread and the display may stay in this call until end of the program.
 * This is mainly for GUI displays (GL/SDL), which usually need to be run from main thread of the
 * program (OS X).
 *
 * The function must quit after receiving a poisoned pill (frame == NULL) to
 * a display_put_frame() call.

 * @param d display to be run
 */
void display_run(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_run(d->state);
}

/**
 * @brief Returns video framebuffer which will be written to.
 *
 * Currently there is a restriction on number of concurrently acquired frames - only one frame
 * can be hold at the moment. Every obtained frame from this call has to be returned back
 * with display_put_frame()
 *
 * @return               video frame
 */
struct video_frame *display_get_frame(struct display *d)
{
        perf_record(UVP_GETFRAME, d);
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_getf(d->state);
}

/**
 * @brief Puts filled video frame.
 * After calling this function, video frame cannot be used.
 *
 * @param d        display to be putted frame to
 * @param frame    frame that has been obtained from display_get_frame() and has not yet been put.
 *                 Should not be NULL unless we want to quit display mainloop.
 * @param nonblock specifies blocking behavior (@ref display_put_frame_flags)
 */
int display_put_frame(struct display *d, struct video_frame *frame, int nonblock)
{
        perf_record(UVP_PUTFRAME, frame);
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_putf(d->state, frame, nonblock);
}

/**
 * @brief Reconfigure display to new video format.
 *
 * video_desc::color_spec, video_desc::interlacing
 * and video_desc::tile_count are set according
 * to properties obtained from display_get_property().
 *
 * @param d    display to be reconfigured
 * @param desc new video description to be reconfigured to
 * @retval TRUE  if reconfiguration succeeded
 * @retval FALSE if reconfiguration failed
 */
int display_reconfigure(struct display *d, struct video_desc desc)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_reconfigure(d->state, desc);
}

/**
 * @brief Gets property from video display.
 * @param[in]     d         video display state
 * @param[in]     property  one of @ref display_property
 * @param[in]     val       pointer to output buffer where should be the property stored
 * @param[in]     len       provided buffer length
 * @param[out]    len       actual size written
 * @retval      TRUE      if succeeded and result is contained in val and len
 * @retval      FALSE     if the query didn't succeeded (either not supported or error)
 */
int display_get_property(struct display *d, int property, void *val, size_t *len)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_get_property(d->state, property, val, len);
}

/**
 * @brief Puts audio data.
 * @param d     video display
 * @param frame audio frame to be played
 */
void display_put_audio_frame(struct display *d, struct audio_frame *frame)
{
        assert(d->magic == DISPLAY_MAGIC);
        display_device_table[d->index].func_put_audio_frame(d->state, frame);
}

/**
 * This function instructs video driver to reconfigure itself
 *
 * @param               d               video display structure
 * @param               quant_samples   number of bits per sample
 * @param               channels        count of channels
 * @param               sample_rate     samples per second
 * @retval              TRUE            if reconfiguration succeeded
 * @retval              FALSE           if reconfiguration failed
 */
int display_reconfigure_audio(struct display *d, int quant_samples, int channels, int sample_rate)
{
        assert(d->magic == DISPLAY_MAGIC);
        return display_device_table[d->index].func_reconfigure_audio(d->state, quant_samples, channels, sample_rate);
}


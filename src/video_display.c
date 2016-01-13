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
 * Copyright (c) 2005-2015 CESNET z.s.p.o.
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
#include "lib_common.h"
#include "module.h"
#include "perf.h"
#include "video_display.h"

#define DISPLAY_MAGIC 0x01ba7ef1

/// @brief This struct represents initialized video display state.
struct display {
        struct module mod;
        uint32_t magic;    ///< For debugging. Conatins @ref DISPLAY_MAGIC
        const struct video_display_info *funcs;
        void *state;       ///< state of the created video capture driver
        bool started;
};

/**This variable represents a pseudostate and may be returned when initialization
 * of module was successful but no state was created (eg. when driver had displayed help). */
int display_init_noerr;

void list_video_display_devices()
{
        printf("Available display devices:\n");
        list_modules(LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
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
int initialize_video_display(struct module *parent, const char *requested_display,
                const char *fmt, unsigned int flags, struct display **out)
{
        const struct video_display_info *vdi = (const struct video_display_info *)
                        load_library(requested_display, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

        if (vdi) {
                struct display *d = calloc(1, sizeof(struct display));
                d->magic = DISPLAY_MAGIC;
                d->funcs = vdi;

                module_init_default(&d->mod);
                d->mod.cls = MODULE_CLASS_DISPLAY;
                module_register(&d->mod, parent);

                d->state  = d->funcs->init(&d->mod, fmt, flags);

                if (d->state == NULL) {
                        debug_msg("Unable to start display %s\n",
                                        requested_display);
                        module_done(&d->mod);
                        free(d);
                        return -1;
                } else if (d->state == &display_init_noerr) {
                        module_done(&d->mod);
                        free(d);
                        return 1;
                }
                *out = d;
                return 0;
        }

        log_msg(LOG_LEVEL_ERROR, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
        return -1;
}

/**
 * @brief This function performs cleanup after done.
 * @param d display do be destroyed (must not be NULL)
 */
void display_done(struct display *d)
{
        assert(d->magic == DISPLAY_MAGIC);
        d->funcs->done(d->state);
        module_done(&d->mod);
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
        d->started = true;
        d->funcs->run(d->state);
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
        return d->funcs->getf(d->state);
}

/**
 * @brief Puts filled video frame.
 * After calling this function, video frame cannot be used.
 *
 * @param d        display to be putted frame to
 * @param frame    frame that has been obtained from display_get_frame() and has not yet been put.
 *                 Should not be NULL unless we want to quit display mainloop.
 * @param nonblock specifies blocking behavior (@ref display_put_frame_flags)
 * @retval      0  if displayed succesfully
 * @retval      1  if not displayed
 */
int display_put_frame(struct display *d, struct video_frame *frame, int nonblock)
{
        perf_record(UVP_PUTFRAME, frame);
        assert(d->magic == DISPLAY_MAGIC);
        if (!d->started) {
                return 1;
        }
        return d->funcs->putf(d->state, frame, nonblock);
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
        return d->funcs->reconfigure_video(d->state, desc);
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
        return d->funcs->get_property(d->state, property, val, len);
}

/**
 * @brief Puts audio data.
 * @param d     video display
 * @param frame audio frame to be played
 */
void display_put_audio_frame(struct display *d, struct audio_frame *frame)
{
        assert(d->magic == DISPLAY_MAGIC);
        d->funcs->put_audio_frame(d->state, frame);
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
        return d->funcs->reconfigure_audio(d->state, quant_samples, channels, sample_rate);
}


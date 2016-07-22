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
/**
 * @todo
 * On-fly postprocess reconfiguration is not ok in general case. The pitch or
 * supported codecs might have changed due to the reconfiguration, which is,
 * however, not reflected by decoder which queried the data before.
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "lib_common.h"
#include "module.h"
#include "perf.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

#define DISPLAY_MAGIC 0x01ba7ef1

/// @brief This struct represents initialized video display state.
struct display {
        struct module mod;
        uint32_t magic;    ///< For debugging. Conatins @ref DISPLAY_MAGIC
        const struct video_display_info *funcs;
        void *state;       ///< state of the created video capture driver

        struct vo_postprocess_state *postprocess;
        int pp_output_frames_count, display_pitch;
        struct video_desc saved_desc;
        enum video_mode saved_mode;
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
                const char *fmt, unsigned int flags, const char *postprocess, struct display **out)
{
        if (postprocess && strcmp(postprocess, "help") == 0) {
                show_vo_postprocess_help();
                return 1;
        }

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

                if (postprocess) {
                        d->postprocess = vo_postprocess_init(postprocess);
                        if (!d->postprocess) {
                                display_done(d);
                                return 1;
                        }
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
        vo_postprocess_done(d->postprocess);
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
        d->funcs->run(d->state);
}

static struct response *process_message(struct display *d, struct msg_universal *msg)
{
        if (strncasecmp(msg->text, "postprocess ", strlen("postprocess ")) == 0) {
                log_msg(LOG_LEVEL_WARNING, "On fly changing postprocessing is currently "
                                "only an experimental feature! Use with caution!\n");
                const char *text = msg->text + strlen("postprocess ");

                struct vo_postprocess_state *postprocess_old = d->postprocess;

                if (strcmp(text, "flush") != 0) {
                        d->postprocess = vo_postprocess_init(text);
                        if (!d->postprocess) {
                                d->postprocess = postprocess_old;
                                log_msg(LOG_LEVEL_ERROR, "Unable to create postprocess '%s'.\n", text);
                                return new_response(RESPONSE_BAD_REQUEST, NULL);
                        }
                } else {
                        d->postprocess = NULL;
                }

                vo_postprocess_done(postprocess_old);

                display_reconfigure(d, d->saved_desc, d->saved_mode);

                return new_response(RESPONSE_OK, NULL);
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unknown command '%s'.\n", msg->text);
                return new_response(RESPONSE_BAD_REQUEST, NULL);
        }
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
        struct message *msg;
        while((msg = check_message(&d->mod))) {
                struct response *r = process_message(d, (struct msg_universal *) msg);
                free_message(msg, r);
        }

        perf_record(UVP_GETFRAME, d);
        assert(d->magic == DISPLAY_MAGIC);
        if (d->postprocess) {
                return vo_postprocess_getf(d->postprocess);
        } else {
                return d->funcs->getf(d->state);
        }
}

/**
 * @brief Puts filled video frame.
 * After calling this function, video frame cannot be used.
 *
 * @param d        display to be putted frame to
 * @param frame    frame that has been obtained from display_get_frame() and has not yet been put.
 *                 Should not be NULL unless we want to quit display mainloop.
 * @param flags specifies blocking behavior (@ref display_put_frame_flags)
 * @retval      0  if displayed succesfully
 * @retval      1  if not displayed
 */
int display_put_frame(struct display *d, struct video_frame *frame, int flags)
{
        perf_record(UVP_PUTFRAME, frame);
        assert(d->magic == DISPLAY_MAGIC);

        if (!frame) {
                return d->funcs->putf(d->state, frame, flags);
        }

        if (d->postprocess) {
                int display_ret = 0;
		for (int i = 0; i < d->pp_output_frames_count; ++i) {
			struct video_frame *display_frame = d->funcs->getf(d->state);
			int ret = vo_postprocess(d->postprocess,
					frame,
					display_frame,
					d->display_pitch);
                        frame = NULL;
			if (!ret) {
				d->funcs->putf(d->state, display_frame, PUTF_DISCARD);
				return 1;
			}

			display_ret = d->funcs->putf(d->state, display_frame, flags);
		}
                return display_ret;
        } else {
                return d->funcs->putf(d->state, frame, flags);
        }
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
int display_reconfigure(struct display *d, struct video_desc desc, enum video_mode video_mode)
{
        assert(d->magic == DISPLAY_MAGIC);

        d->saved_desc = desc;
        d->saved_mode = video_mode;

        if (d->postprocess) {
                bool pp_does_change_tiling_mode = false;
                size_t len = sizeof(pp_does_change_tiling_mode);
                if (vo_postprocess_get_property(d->postprocess, VO_PP_DOES_CHANGE_TILING_MODE,
                                        &pp_does_change_tiling_mode, &len)) {
                        if(len == 0) {
                                // just for sake of completness since it shouldn't be a case
                                log_msg(LOG_LEVEL_WARNING, "Warning: unable to get pp tiling mode!\n");
                        }
                }
		struct video_desc pp_desc = desc;

                if (!pp_does_change_tiling_mode) {
                        pp_desc.width *= get_video_mode_tiles_x(video_mode);
                        pp_desc.height *= get_video_mode_tiles_y(video_mode);
                        pp_desc.tile_count = 1;
                }
                if (!vo_postprocess_reconfigure(d->postprocess, pp_desc)) {
                        log_msg(LOG_LEVEL_ERROR, "[video dec.] Unable to reconfigure video "
                                        "postprocess.\n");
                        return false;
                }
		struct video_desc display_desc;
                int render_mode; // WTF ?
		vo_postprocess_get_out_desc(d->postprocess, &display_desc, &render_mode, &d->pp_output_frames_count);
		int rc = d->funcs->reconfigure_video(d->state, display_desc);
                len = sizeof d->display_pitch;
                d->display_pitch = PITCH_DEFAULT;
                d->funcs->get_property(d->state, DISPLAY_PROPERTY_BUF_PITCH,
					&d->display_pitch, &len);
                if (d->display_pitch == PITCH_DEFAULT) {
			d->display_pitch = vc_get_linesize(display_desc.width, display_desc.color_spec);
		}

                return rc;
        } else {
		return d->funcs->reconfigure_video(d->state, desc);
	}
}

static void restrict_returned_codecs(codec_t *display_codecs,
                size_t *display_codecs_count, codec_t *pp_codecs,
                int pp_codecs_count)
{
        int i;

        for (i = 0; i < (int) *display_codecs_count; ++i) {
                int j;

                int found = FALSE;

                for (j = 0; j < pp_codecs_count; ++j) {
                        if(display_codecs[i] == pp_codecs[j]) {
                                found = TRUE;
                        }
                }

                if(!found) {
                        memmove(&display_codecs[i], (const void *) &display_codecs[i + 1],
                                        sizeof(codec_t) * (*display_codecs_count - i - 1));
                        --*display_codecs_count;
                        --i;
                }
        }
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
        if (d->postprocess) {
                switch (property) {
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        return TRUE;
		case DISPLAY_PROPERTY_CODECS:
			{
                                codec_t display_codecs[20], pp_codecs[20];
                                size_t display_codecs_count, pp_codecs_count;
                                size_t nlen;
                                bool ret;
                                nlen = sizeof display_codecs;
                                ret = d->funcs->get_property(d->state, DISPLAY_PROPERTY_CODECS, display_codecs, &nlen);
                                if (!ret) return FALSE;
                                display_codecs_count = nlen / sizeof(codec_t);
                                nlen = sizeof pp_codecs;
                                ret = vo_postprocess_get_property(d->postprocess, VO_PP_PROPERTY_CODECS, pp_codecs, &nlen);
                                if (ret) {
					if (nlen == 0) { // problem detected
						log_msg(LOG_LEVEL_ERROR, "[Decoder] Unable to get supported codecs.\n");
						return FALSE;

					}
                                        pp_codecs_count = nlen / sizeof(codec_t);
                                        restrict_returned_codecs(display_codecs, &display_codecs_count,
                                                        pp_codecs, pp_codecs_count);
                                }
                                nlen = display_codecs_count * sizeof(codec_t);
                                if (nlen <= *len) {
                                        *len = nlen;
                                        memcpy(val, display_codecs, nlen);
                                        return TRUE;
                                } else {
                                        return FALSE;
                                }
                        }
			break;
                default:
                        return d->funcs->get_property(d->state, property, val, len);
                }
        } else {
                return d->funcs->get_property(d->state, property, val, len);
        }
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


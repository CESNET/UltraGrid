/**
 * @file   video_capture.c
 * @author Colin Perkins <csp@csperkins.org>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup vidcap
 */
/*
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "module.h"
#include "utils/config_file.h"
#include "utils/resource_manager.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/DirectShowGrabber.h"
#include "video_capture/aggregate.h"
#include "video_capture/avfoundation.h"
#include "video_capture/bluefish444.h"
#include "video_capture/decklink.h"
#include "video_capture/deltacast.h"
#include "video_capture/deltacast_dvi.h"
#include "video_capture/dvs.h"
#include "video_capture/import.h"
#include "video_capture/null.h"
#include "video_capture/quicktime.h"
#include "video_capture/screen_osx.h"
#include "video_capture/screen_x11.h"
#include "video_capture/swmix.h"
#include "video_capture/switcher.h"
#include "video_capture/testcard.h"
#include "video_capture/testcard2.h"
#include "video_capture/v4l2.h"
#include "video_capture/rtsp.h"

#include <string>

using namespace std;

#define VIDCAP_MAGIC	0x76ae98f0

static int vidcap_init_devices(bool verbose);
static void vidcap_free_devices(void);

/**This variable represents a pseudostate and may be returned when initialization
 * of module was successful but no state was created (eg. when driver had displayed help).
 */
int vidcap_init_noerr;

struct vidcap_params;

/**
 * Defines parameters passed to video capture driver.
  */
struct vidcap_params {
        char  *driver; ///< driver name
        char  *fmt;    ///< driver options
        unsigned int flags;  ///< one of @ref vidcap_flags

        char *requested_capture_filter;
        char  *name;   ///< input name (capture alias in config file or complete config if not alias)
        struct vidcap_params *next; /**< Pointer to next vidcap params. Used by aggregate capture drivers.
                                     *   Last device in list has @ref driver set to NULL. */
        struct module *parent;
};

/// @brief This struct represents video capture state.
struct vidcap {
        struct module mod;
        void    *state; ///< state of the created video capture driver
        int      index; ///< index to @ref vidcap_device_table
        uint32_t magic; ///< For debugging. Conatins @ref VIDCAP_MAGIC

        struct capture_filter *capture_filter; ///< capture_filter_state
};

/**
 * This struct describes individual vidcap modules
 * @copydetails decoder_table_t
 */
struct vidcap_device_api {
        vidcap_id_t id;                        ///< @copydoc decoder_table_t::magic

        const char              *library_name; ///< @copydoc decoder_table_t::library_name

        struct vidcap_type    *(*func_probe) (bool verbose);
        const char              *func_probe_str;
        void                  *(*func_init) (const struct vidcap_params *param);
        const char              *func_init_str;
        void                   (*func_done) (void *state);
        const char              *func_done_str;
        struct video_frame    *(*func_grab) (void *state, struct audio_frame **audio);
        const char              *func_grab_str;

        void                    *handle;       ///< @copydoc decoder_table_t::handle
        /** @var func_init
         * @param[in] driver configuration string
         * @param[in] param  driver parameters
         * @retval NULL if initialization failed
         * @retval &vidcap_init_noerr if initialization succeeded but a state was not returned (eg. help)
         * @retval other_ptr if initialization succeeded, contains pointer to state
         */
};

/** @brief This table contains list of video capture devices compiled with this UltraGrid version.
 *  @copydetails decoders */
struct vidcap_device_api vidcap_device_table[] = {
#ifndef UV_IN_YURI
        {
         /* The aggregate capture card */
         0,
         NULL,
         MK_STATIC(vidcap_aggregate_probe),
         MK_STATIC(vidcap_aggregate_init),
         MK_STATIC(vidcap_aggregate_done),
         MK_STATIC(vidcap_aggregate_grab),
         NULL
        },
#if defined HAVE_AVFOUNDATION
        {
         0,
         "avfoundation",
         MK_NAME(vidcap_avfoundation_probe),
         MK_NAME(vidcap_avfoundation_init),
         MK_NAME(vidcap_avfoundation_done),
         MK_NAME(vidcap_avfoundation_grab),
         NULL
        },
#endif
        {
         0,
         NULL,
         MK_STATIC(vidcap_import_probe),
         MK_STATIC(vidcap_import_init),
         MK_STATIC(vidcap_import_done),
         MK_STATIC(vidcap_import_grab),
         NULL
        },
        {
         0,
         NULL,
         MK_STATIC(vidcap_switcher_probe),
         MK_STATIC(vidcap_switcher_init),
         MK_STATIC(vidcap_switcher_done),
         MK_STATIC(vidcap_switcher_grab),
         NULL
        },
#if defined HAVE_RTSP
        {
         0,
         "rtsp",
         MK_NAME(vidcap_rtsp_probe),
         MK_NAME(vidcap_rtsp_init),
         MK_NAME(vidcap_rtsp_done),
         MK_NAME(vidcap_rtsp_grab),
         NULL
        },
#endif
#if defined HAVE_SWMIX || defined BUILD_LIBRARIES
        {
         /* The SW mix capture card */
         0,
         "swmix",
         MK_NAME(vidcap_swmix_probe),
         MK_NAME(vidcap_swmix_init),
         MK_NAME(vidcap_swmix_done),
         MK_NAME(vidcap_swmix_grab),
         NULL
        },
#endif
#if defined HAVE_BLUEFISH444 || defined BUILD_LIBRARIES
        {
         /* The Bluefish444 capture card */
         0,
         "bluefish444",
         MK_NAME(vidcap_bluefish444_probe),
         MK_NAME(vidcap_bluefish444_init),
         MK_NAME(vidcap_bluefish444_done),
         MK_NAME(vidcap_bluefish444_grab),
         NULL
        },
#endif /* HAVE_BLUEFISH444 */
#if defined HAVE_DSHOW || defined BUILD_LIBRARIES
        {
         /* The DirectShow capture card */
         0,
         "dshow",
         MK_NAME(vidcap_dshow_probe),
         MK_NAME(vidcap_dshow_init),
         MK_NAME(vidcap_dshow_done),
         MK_NAME(vidcap_dshow_grab),
         NULL
        },
#endif /* HAVE_DSHOW */
#if defined HAVE_SCREEN_CAP || defined BUILD_LIBRARIES
        {
         /* The screen capture card */
         0,
         "screen",
#ifdef HAVE_LINUX
         MK_NAME(vidcap_screen_x11_probe),
         MK_NAME(vidcap_screen_x11_init),
         MK_NAME(vidcap_screen_x11_done),
         MK_NAME(vidcap_screen_x11_grab),
#else
         MK_NAME(vidcap_screen_osx_probe),
         MK_NAME(vidcap_screen_osx_init),
         MK_NAME(vidcap_screen_osx_done),
         MK_NAME(vidcap_screen_osx_grab),
#endif // ! defined HAVE_LINUX
         NULL
        },
#endif /* HAVE_SCREEN */
#if defined HAVE_DVS || defined BUILD_LIBRARIES
        {
         /* The DVS capture card */
         0,
         "dvs",
         MK_NAME(vidcap_dvs_probe),
         MK_NAME(vidcap_dvs_init),
         MK_NAME(vidcap_dvs_done),
         MK_NAME(vidcap_dvs_grab),
         NULL
        },
#endif                          /* HAVE_DVS */
#if defined HAVE_DECKLINK || defined BUILD_LIBRARIES
        {
         /* The Blackmagic DeckLink capture card */
         0,
         "decklink",
         MK_NAME(vidcap_decklink_probe),
         MK_NAME(vidcap_decklink_init),
         MK_NAME(vidcap_decklink_done),
         MK_NAME(vidcap_decklink_grab),
         NULL
        },
#endif                          /* HAVE_DECKLINK */
#if defined HAVE_DELTACAST || defined BUILD_LIBRARIES
        {
         /* The Blackmagic DeckLink capture card */
         0,
         "deltacast",
         MK_NAME(vidcap_deltacast_probe),
         MK_NAME(vidcap_deltacast_init),
         MK_NAME(vidcap_deltacast_done),
         MK_NAME(vidcap_deltacast_grab),
         NULL
        },
        {
         0,
         "deltacast",
         MK_NAME(vidcap_deltacast_dvi_probe),
         MK_NAME(vidcap_deltacast_dvi_init),
         MK_NAME(vidcap_deltacast_dvi_done),
         MK_NAME(vidcap_deltacast_dvi_grab),
         NULL
        },
#endif                          /* HAVE_DELTACAST */
#if defined HAVE_MACOSX
        {
         /* The QuickTime API */
         0,
         "quicktime",
         MK_NAME(vidcap_quicktime_probe),
         MK_NAME(vidcap_quicktime_init),
         MK_NAME(vidcap_quicktime_done),
         MK_NAME(vidcap_quicktime_grab),
         NULL
        },
#endif                          /* HAVE_MACOSX */
        {
         /* Dummy sender for testing purposes */
         0,
         "testcard",
         MK_NAME(vidcap_testcard_probe),
         MK_NAME(vidcap_testcard_init),
         MK_NAME(vidcap_testcard_done),
         MK_NAME(vidcap_testcard_grab),
         NULL
        },
#if defined HAVE_TESTCARD2 || defined BUILD_LIBRARIES
        {
         /* Dummy sender for testing purposes */
         0,
         "testcard2",
         MK_NAME(vidcap_testcard2_probe),
         MK_NAME(vidcap_testcard2_init),
         MK_NAME(vidcap_testcard2_done),
         MK_NAME(vidcap_testcard2_grab),
         NULL
        },
#endif /* HAVE_TESTCARD2 */
#if defined HAVE_V4L2 || defined BUILD_LIBRARIES
        {
         /* Dummy sender for testing purposes */
         0,
         "v4l2",
         MK_NAME(vidcap_v4l2_probe),
         MK_NAME(vidcap_v4l2_init),
         MK_NAME(vidcap_v4l2_done),
         MK_NAME(vidcap_v4l2_grab),
         NULL
        },
#endif /* HAVE_V4L2 */
#endif
        {
         0,
         NULL,
         MK_STATIC(vidcap_null_probe),
         MK_STATIC(vidcap_null_init),
         MK_STATIC(vidcap_null_done),
         MK_STATIC(vidcap_null_grab),
         NULL
        }
};

#define VIDCAP_DEVICE_TABLE_SIZE (sizeof(vidcap_device_table)/sizeof(struct vidcap_device_api))

/* API for probing capture devices ****************************************************************/

/** @brief List of available vidcap devices
 * Initialized with @ref vidcap_init_devices */
static struct vidcap_type *available_vidcap_devices[VIDCAP_DEVICE_TABLE_SIZE];
/** @brief Count of @ref available_vidcap_devices
 * Initialized with @ref vidcap_init_devices */
static int available_vidcap_device_count = 0;

#ifdef BUILD_LIBRARIES
/** Opens vidcap library of given name. */
static void *vidcap_open_library(const char *vidcap_name)
{
        char name[128];
        snprintf(name, sizeof(name), "vidcap_%s.so.%d", vidcap_name, VIDEO_CAPTURE_ABI_VERSION);

        return open_library(name);
}

/** For a given device, load individual functions from library handle (previously opened). */
static int vidcap_fill_symbols(struct vidcap_device_api *device)
{
        void *handle = device->handle;

        device->func_probe = (struct vidcap_type *(*) (bool))
                dlsym(handle, device->func_probe_str);
        device->func_init = (void *(*) (const struct vidcap_params *))
                dlsym(handle, device->func_init_str);
        device->func_done = (void (*) (void *))
                dlsym(handle, device->func_done_str);
        device->func_grab = (struct video_frame *(*) (void *, struct audio_frame **))
                dlsym(handle, device->func_grab_str);
        if(!device->func_probe || !device->func_init ||
                        !device->func_done || !device->func_grab) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }
        return TRUE;
}
#endif

/** @brief Must be called before initalization of vidcap.
 * In modular UltraGrid build, it also opens available libraries.
 * @todo
 * Figure out where to close libraries. vidcap_free_devices() is not the right place because
 * it is called to early.
 */
static int vidcap_init_devices(bool verbose)
{
        unsigned int i;
        struct vidcap_type *dt;

        assert(available_vidcap_device_count == 0);

        for (i = 0; i < VIDCAP_DEVICE_TABLE_SIZE; i++) {
                //printf("probe: %d\n",i);
#ifdef BUILD_LIBRARIES
                vidcap_device_table[i].handle = NULL;
                if(vidcap_device_table[i].library_name) {
                        vidcap_device_table[i].handle =
                                vidcap_open_library(vidcap_device_table[i].library_name);
                        if(vidcap_device_table[i].handle) {
                                int ret;
                                ret = vidcap_fill_symbols(&vidcap_device_table[i]);
                                if(!ret) continue;
                        } else {
                                continue;
                        }
                }
#endif

                dt = vidcap_device_table[i].func_probe(verbose);
                if (dt != NULL) {
                        vidcap_device_table[i].id = dt->id;
                        available_vidcap_devices[available_vidcap_device_count++] = dt;
                }
        }

        return available_vidcap_device_count;
}

/** Should be called after video capture is initialized. */
static void vidcap_free_devices(void)
{
        int i;

        for (i = 0; i < available_vidcap_device_count; i++) {
                free(available_vidcap_devices[i]->cards);
                free(available_vidcap_devices[i]);
                available_vidcap_devices[i] = NULL;
        }
        available_vidcap_device_count = 0;
}

/** Returns count of available vidcap devices. */
int vidcap_get_device_count(void)
{
        return available_vidcap_device_count;
}

/** Returns vidcap device metadata for given index. */
struct vidcap_type *vidcap_get_device_details(int index)
{
        assert(index < available_vidcap_device_count);
        assert(available_vidcap_devices[index] != NULL);

        return available_vidcap_devices[index];
}

/** Returns index of the noop device. */
vidcap_id_t vidcap_get_null_device_id(void)
{
        return VIDCAP_NULL_ID;
}

void list_video_capture_devices()
{
        int i;
        struct vidcap_type *vt;

        printf("Available capture devices:\n");
        vidcap_init_devices(false);
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                printf("\t%s\n", vt->name);
        }
        vidcap_free_devices();
}

void print_available_capturers()
{
        vidcap_init_devices(true);
        for (int i = 0; i < vidcap_get_device_count(); i++) {
                struct vidcap_type *vt = vidcap_get_device_details(i);
                for (int i = 0; i < vt->card_count; ++i) {
                        printf("(%s:%s;%s)\n", vt->name, vt->cards[i].id, vt->cards[i].name);
                }
        }

        char buf[1024] = "";
        struct config_file *conf = config_file_open(default_config_file(buf, sizeof buf));
        if (conf) {
                auto const & from_config_file = get_configured_capture_aliases(conf);
                for (auto const & it : from_config_file) {
                        printf("(%s;%s)\n", it.first.c_str(), it.second.c_str());
                }
        }
        config_file_close(conf);
        vidcap_free_devices();
}

int initialize_video_capture(struct module *parent,
                struct vidcap_params *params,
                struct vidcap **state)
{
        struct vidcap_type *vt;
        vidcap_id_t id = 0;
        int i;

        if(!strcmp(vidcap_params_get_driver(params), "none"))
                id = vidcap_get_null_device_id();

        // locking here is because listing of the devices is not really thread safe
        pthread_mutex_t *vidcap_lock = rm_acquire_shared_lock("VIDCAP_LOCK");
        pthread_mutex_lock(vidcap_lock);

        vidcap_init_devices(false);
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                if (strcmp(vt->name, vidcap_params_get_driver(params)) == 0) {
                        id = vt->id;
                        break;
                }
        }
        if(i == vidcap_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' capture card "
                        "was not found.\n", vidcap_params_get_driver(params));
                return -1;
        }
        vidcap_free_devices();

        pthread_mutex_unlock(vidcap_lock);
        rm_release_shared_lock("VIDCAP_LOCK");

        return vidcap_init(parent, id, params, state);
}

/** @brief Initializes video capture
 * @param[in] id     index of selected video capture driver
 * @param[in] param  driver parameters
 * @param[out] state returned state
 * @retval 0    if initialization was successful
 * @retval <0   if initialization failed
 * @retval >0   if initialization was successful but no state was returned (eg. only having shown help).
 */
int vidcap_init(struct module *parent, vidcap_id_t id, struct vidcap_params *param,
                struct vidcap **state)
{
        unsigned int i;

        for (i = 0; i < VIDCAP_DEVICE_TABLE_SIZE; i++) {
                if (vidcap_device_table[i].id == id) {
                        struct vidcap *d =
                            (struct vidcap *)malloc(sizeof(struct vidcap));
                        d->magic = VIDCAP_MAGIC;

                        module_init_default(&d->mod);
                        d->mod.cls = MODULE_CLASS_CAPTURE;
                        module_register(&d->mod, parent);

                        param->parent = &d->mod;
                        d->state = vidcap_device_table[i].func_init(param);
                        d->index = i;
                        if (d->state == NULL) {
                                debug_msg
                                    ("Unable to start video capture device 0x%08lx\n",
                                     id);
                                module_done(&d->mod);
                                free(d);
                                return -1;
                        }
                        if(d->state == &vidcap_init_noerr) {
                                module_done(&d->mod);
                                free(d);
                                return 1;
                        }

                        int ret = capture_filter_init(&d->mod, param->requested_capture_filter,
                                        &d->capture_filter);
                        if(ret < 0) {
                                fprintf(stderr, "Unable to initialize capture filter: %s.\n",
                                        param->requested_capture_filter);
                        }

                        if (ret != 0) {
                                module_done(&d->mod);
                                return ret;
                        }

                        *state = d;
                        return 0;
                }
        }
        debug_msg("Unknown video capture device: 0x%08x\n", id);
        return -1;
}

/** @brief Destroys vidap state
 * @param state state to be destroyed (must have been successfully initialized with vidcap_init()) */
void vidcap_done(struct vidcap *state)
{
        assert(state->magic == VIDCAP_MAGIC);
        vidcap_device_table[state->index].func_done(state->state);
        capture_filter_destroy(state->capture_filter);
        module_done(&state->mod);
        free(state);
}

/** @brief Grabs video frame.
 * This function may block for a short period if waiting for incoming frame. This period, however,
 * should not be longer than few frame times, a second at maximum.
 *
 * The decision of blocking behavior is on the vidcap driver.
 *
 * The returned video frame is valid only until next vidcap_grab() call.
 *
 * @param[in]  state vidcap state
 * @param[out] audio contains audio frame if driver is grabbing audio
 * @returns video frame. If no frame was grabbed (or timeout passed) NULL is returned.
 */
struct video_frame *vidcap_grab(struct vidcap *state, struct audio_frame **audio)
{
        assert(state->magic == VIDCAP_MAGIC);
        struct video_frame *frame;
        frame = vidcap_device_table[state->index].func_grab(state->state, audio);
        if (frame != NULL)
                frame = capture_filter(state->capture_filter, frame);
        return frame;
}

/**
 * @brier Allocates blank @ref vidcap_params structure.
 */
struct vidcap_params *vidcap_params_allocate(void)
{
        return (struct vidcap_params *) calloc(1, sizeof(struct vidcap_params));
}

/**
 * @brier Allocates blank @ref vidcap_params structure.
 *
 * Follows curr struct in the virtual list.
 * @param curr structure to be appended after
 * @returns pointer to newly created structure
 */
struct vidcap_params *vidcap_params_allocate_next(struct vidcap_params *curr)
{
        curr->next = vidcap_params_allocate();
        return curr->next;
}

/**
 * @brier Returns next item in virtual @ref vidcap_params list.
 */
struct vidcap_params *vidcap_params_get_next(const struct vidcap_params *curr)
{
        return curr->next;
}

/**
 * @brier Returns n-th item in @ref vidcap_params list.
 */
struct vidcap_params *vidcap_params_get_nth(struct vidcap_params *curr, int index)
{
        struct vidcap_params *ret = curr;
        for (int i = 0; i < index; i++) {
                ret = ret->next;
                if (ret == NULL) {
                        return NULL;
                }
        }
        if (ret->driver == NULL) {
                // this is just the stopper of the list...
                return NULL;
        }
        return ret;
}

/**
 * This function does 2 things:
 * * checks whether @ref vidcap_params::name is not an alias
 * * tries to find capture filter for @ref vidcap_params::name if not given
 * @retval true  if alias dispatched successfully
 * @retval false otherwise
 */
static bool vidcap_dispatch_alias(struct vidcap_params *params)
{
        bool ret;
        char buf[1024];
        string real_capture;
        struct config_file *conf =
                config_file_open(default_config_file(buf,
                                        sizeof(buf)));
        if (conf == NULL)
                return false;
        real_capture = config_file_get_alias(conf, "capture", params->name);
        if (real_capture.empty()) {
                ret = false;
        } else {
                params->driver = strdup(real_capture.c_str());
                if (strchr(params->driver, ':')) {
                        char *delim = strchr(params->driver, ':');
                        params->fmt = strdup(delim + 1);
                        *delim = '\0';
                } else {
                        params->fmt = strdup("");
                }
                ret = true;
        }


        if (params->requested_capture_filter == NULL) {
                string matched_cap_filter = config_file_get_capture_filter_for_alias(conf,
                                        params->name);
                if (!matched_cap_filter.empty())
                        params->requested_capture_filter = strdup(matched_cap_filter.c_str());
        }

        config_file_close(conf);

        return ret;
}

/**
 * Fills the structure with device config string in format either driver[:params] or
 * alias.
 */
void vidcap_params_set_device(struct vidcap_params *params, const char *config)
{
        params->name = strdup(config);

        if (!vidcap_dispatch_alias(params)) {
                params->driver = strdup(config);
                if (strchr(params->driver, ':')) {
                        char *delim = strchr(params->driver, ':');
                        *delim = '\0';
                        params->fmt = strdup(delim + 1);
                } else {
                        params->fmt = strdup("");
                }
        }
}

void vidcap_params_set_capture_filter(struct vidcap_params *params,
                const char *req_capture_filter)
{
        params->requested_capture_filter = strdup(req_capture_filter);
}

void vidcap_params_set_flags(struct vidcap_params *params, unsigned int flags)
{
        params->flags = flags;
}

const char *vidcap_params_get_driver(const struct vidcap_params *params)
{
        return params->driver;
}

const char *vidcap_params_get_fmt(const struct vidcap_params *params)
{
        return params->fmt;
}

unsigned int vidcap_params_get_flags(const struct vidcap_params *params)
{
        return params->flags;
}

const char *vidcap_params_get_name(const struct vidcap_params *params)
{
        return params->name;
}

struct module *vidcap_params_get_parent(const struct vidcap_params *params)
{
        return params->parent;
}

/**
 * Creates deep copy of @ref vidcap_params structure.
 */
struct vidcap_params *vidcap_params_copy(const struct vidcap_params *params)
{
        if (!params)
                return NULL;

        struct vidcap_params *ret = (struct vidcap_params *) calloc(1, sizeof(struct vidcap_params));

        if (params->driver)
                ret->driver = strdup(params->driver);
        if (params->fmt)
                ret->fmt = strdup(params->fmt);
        if (params->requested_capture_filter)
                ret->requested_capture_filter =
                        strdup(params->requested_capture_filter);
        if (params->name)
                ret->name = strdup(params->name);
        if (params->flags)
                ret->flags = params->flags;

        ret->next = NULL; // there is high probability that the pointer will be invalid

        return ret;
}

/**
 * Frees all members of the given structure as well as its members.
 *
 * @param[in] buf structure to be feed
 */
void vidcap_params_free_struct(struct vidcap_params *buf)
{
        if (!buf)
                return;

        free(buf->driver);
        free(buf->fmt);
        free(buf->requested_capture_filter);
        free(buf->name);

        free(buf);
}

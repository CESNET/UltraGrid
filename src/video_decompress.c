/**
 * @file   video_decompress.c
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup video_decompress
 * @brief video compress functions
 */
/*
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <stdio.h>
#include <string.h>
#include "video_codec.h"
#include "video_decompress.h"
#include "video_decompress/dxt_glsl.h"
#include "video_decompress/jpeg.h"
#include "video_decompress/libavcodec.h"
#include "video_decompress/null.h"
#include "video_decompress/jpeg_to_dxt.h"
#include "lib_common.h"

#define DECOMPRESS_MAGIC 0xdff34f21u

/**
 * @brief This struct describes individual decompress modules
 *
 * Initially, in this struct are either callbacks or functions names.
 * For actual initialization of the callbacks/names, @ref MK_STATIC and @ref MK_NAME
 * macros should be used. After initialization, callbacks are set.
 */
typedef struct {
        uint32_t magic;     ///< unique module identifier
        char *library_name; ///< If module is dynamically loadable, this is the name of library.

        decompress_init_t init;
        const char *init_str;
        decompress_reconfigure_t reconfigure;
        const char *reconfigure_str;
        decompress_decompress_t decompress;
        const char *decompress_str;
        decompress_get_property_t get_property;
        const char *get_property_str;
        decompress_done_t done;
        const char *done_str;

        void *handle;       ///< for modular build, dynamically loaded library handle
} decoder_table_t;

/**
 * This struct represents actual decompress state
 */
struct state_decompress {
        uint32_t magic;             ///< selected decoder magic
        decoder_table_t *functions; ///< pointer to selected decoder functions
        void *state;                ///< decoder driver state
};

#ifdef BUILD_LIBRARIES
/** Opens decompress library of given name.  */
static void *decompress_open_library(const char *vidcap_name)
{
        char name[128];
        snprintf(name, sizeof(name), "vdecompress_%s.so.%d", vidcap_name, VIDEO_DECOMPRESS_ABI_VERSION);

        return open_library(name);
}

/** For a given device, load individual functions from library handle (previously opened). */
static int decompress_fill_symbols(decoder_table_t *device)
{
        void *handle = device->handle;

        device->init = (decompress_init_t)
                dlsym(handle, device->init_str);
        device->reconfigure = (decompress_reconfigure_t)
                dlsym(handle, device->reconfigure_str);
        device->decompress = (decompress_decompress_t)
                dlsym(handle, device->decompress_str);
        device->get_property = (decompress_get_property_t)
                dlsym(handle, device->get_property_str);
        device->done = (decompress_done_t)
                dlsym(handle, device->done_str);
        if(!device->init || !device->reconfigure || !device->decompress || 
                        !device->get_property || !device->done) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }
        return TRUE;
}
#endif

/**
 * This array represent list of all known decoders and its capabilities
 */
struct decode_from_to decoders_for_codec[] = {
        { DXT1, RGBA, RTDXT_MAGIC, 500},
        { DXT1_YUV, RGBA, RTDXT_MAGIC, 500 },
        { DXT5, RGBA, RTDXT_MAGIC, 500 },
        { DXT1, UYVY, RTDXT_MAGIC, 500 },
        { DXT1_YUV, UYVY, RTDXT_MAGIC, 500 },
        { DXT5, UYVY, RTDXT_MAGIC, 500 },
        { JPEG, RGB, JPEG_MAGIC, 500 },
        { JPEG, UYVY, JPEG_MAGIC, 500 },
        { H264, UYVY, LIBAVCODEC_MAGIC, 500 },
        { JPEG, UYVY, LIBAVCODEC_MAGIC, 600 },
        { MJPG, UYVY, LIBAVCODEC_MAGIC, 500 },
        { JPEG, DXT1, JPEG_TO_DXT_MAGIC, 900 },
        { JPEG, DXT5, JPEG_TO_DXT_MAGIC, 900 },
        { VP8, UYVY, LIBAVCODEC_MAGIC, 500 },
        { (codec_t) -1, (codec_t) -1, NULL_MAGIC, 0 }
};
/** @brief Length of @ref decoders_for_codec members (items) */
const int decoders_for_codec_count = (sizeof(decoders_for_codec) / sizeof(struct decode_from_to));

/**
 * @brief This table contains list of decoders compiled with this UltraGrid version.
 * If building modular UltraGrid version, hooks for all modules will be created.
 */
decoder_table_t decoders[] = {
#if defined HAVE_DXT_GLSL || defined BUILD_LIBRARIES
        { RTDXT_MAGIC, "rtdxt", MK_NAME(dxt_glsl_decompress_init), MK_NAME(dxt_glsl_decompress_reconfigure),
                MK_NAME(dxt_glsl_decompress), MK_NAME(dxt_glsl_decompress_get_property),
                MK_NAME(dxt_glsl_decompress_done), NULL},
#endif
#if defined HAVE_JPEG || defined BUILD_LIBRARIES
        { JPEG_MAGIC, "jpeg", MK_NAME(jpeg_decompress_init), MK_NAME(jpeg_decompress_reconfigure),
                MK_NAME(jpeg_decompress), MK_NAME(jpeg_decompress_get_property),
                MK_NAME(jpeg_decompress_done), NULL},
#endif 
#if defined HAVE_LAVC || defined BUILD_LIBRARIES
        { LIBAVCODEC_MAGIC, "libavcodec",
                MK_NAME(libavcodec_decompress_init),
                MK_NAME(libavcodec_decompress_reconfigure),
                MK_NAME(libavcodec_decompress),
                MK_NAME(libavcodec_decompress_get_property),
                MK_NAME(libavcodec_decompress_done),
                NULL},
#endif
#if defined HAVE_TRANSCODE && ! defined BUILD_LIBRARIES
        { TRANSCODE_MAGIC, "transcode", MK_STATIC(transcode_decompress_init), MK_STATIC(transcode_decompress_reconfigure),
                MK_STATIC(transcode_decompress), MK_STATIC(transcode_decompress_get_property),
                MK_STATIC(transcode_decompress_done), NULL},
#endif // ! defined BUILD_LIBRARIES && defined HAVE_JPEG || defined HAVE_RTDXT
        { NULL_MAGIC, NULL, MK_STATIC(null_decompress_init), MK_STATIC(null_decompress_reconfigure),
                MK_STATIC(null_decompress), MK_NAME(null_decompress_get_property),
                MK_STATIC(null_decompress_done), NULL}
};

/** @brief length of @ref decoders array */
#define MAX_DECODERS (sizeof(decoders) / sizeof(decoder_table_t))

/** @brief List of available decoders
 * Initialized with @ref initialize_video_decompress */
decoder_table_t *available_decoders[MAX_DECODERS];
/** Count of @ref available_decoders.
 * Initialized with @ref initialize_video_decompress */
int available_decoders_count = -1;

/** must be called before initalization of decoders */
void initialize_video_decompress(void)
{
        available_decoders_count = 0;

        unsigned int i;
        for (i = 0; i < MAX_DECODERS; ++i) {
#ifdef BUILD_LIBRARIES
                decoders[i].handle = NULL;
                if(decoders[i].library_name) {
                        decoders[i].handle = decompress_open_library(decoders[i].library_name);
                        int ret;
                        if(!decoders[i].handle)
                                continue;
                        ret = decompress_fill_symbols(&decoders[i]);
                        if(!ret)
                                continue;
                }
#endif
                available_decoders[available_decoders_count] = &decoders[i];
                available_decoders_count++;
        }
}

/**
 * Checks wheather there is decompressor with given magic present and thus can
 * be initialized with decompress_init
 *
 * @see decompress_init
 * @retval TRUE if decoder is present and can be initialized
 * @retval FALSE if decoder could not be initialized (not found)
 */
int decompress_is_available(unsigned int decoder_index)
{
        int i;
        assert(available_decoders_count > -1);

        for(i = 0; i < available_decoders_count; ++i) {
                if(available_decoders[i]->magic == decoder_index) {
                        return TRUE;
                }
        }
        return FALSE;
}

/**
 * Initializes decompressor or the given magic
 *
 * @retval NULL if initialization failed
 * @retval not-NULL state of new decompressor
 */
struct state_decompress *decompress_init(unsigned int decoder_index)
{
        int i;
        struct state_decompress *s;

        for(i = 0; i < available_decoders_count; ++i) {
                if(available_decoders[i]->magic == decoder_index) {
                        s = (struct state_decompress *) malloc(sizeof(struct state_decompress));
                        s->magic = DECOMPRESS_MAGIC;
                        s->functions = available_decoders[i];
                        s->state = s->functions->init();
                        return s;
                }
        }
        return NULL;
}

/** @copydoc decompress_reconfigure_t */
int decompress_reconfigure(struct state_decompress *s, struct video_desc desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        return s->functions->reconfigure(s->state, desc, rshift, gshift, bshift, pitch, out_codec);
}

/** @copydoc decompress_decompress_t */
int decompress_frame(struct state_decompress *s, unsigned char *dst,
                unsigned char *compressed, unsigned int compressed_len, int frame_seq)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        return s->functions->decompress(s->state, dst, compressed, compressed_len, frame_seq);
}

/** @copydoc decompress_get_property_t */
int decompress_get_property(struct state_decompress *s, int property, void *val, size_t *len)
{
        return s->functions->get_property(s->state, property, val, len);
}

/** @copydoc decompress_done_t */
void decompress_done(struct state_decompress *s)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        s->functions->done(s->state);
        free(s);
}


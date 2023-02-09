/**
 * @file   video_compress.h
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup video_compress
 *
 * @brief API for video compress drivers.
 */
/*
 * Copyright (c) 2009-2023 CESNET, z. s. p. o.
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

/**
 * @defgroup video_compress Video Compress
 * @{
 * ### Adding new video compress module
 * #### Module registration
 * Deleter is currently called through the @ref module API. Therefore the
 * compress module must be initialized and deleter set.
 *
 * Example:
 * ```
 * struct state_vcompress_xy {
 *      struct module mod; // must be first!
 *      ...
 *      <other members>
 * } *s;
 * s = calloc(1, sizeof(struct state_vcompress_xy));
 * module_init_default(&s->mod);
 * s->mod.cls = MODULE_CLASS_DATA;
 * s->mod.priv_data = s;
 * s->mod.deleter = vcompress_xy_free;
 * module_register(&s->mod, s->parent);
 * ```
 */
#ifndef __video_compress_h
#define __video_compress_h

#include "types.h"

#define VIDEO_COMPRESS_ABI_VERSION 9

#ifdef __cplusplus
extern "C" {
#endif

struct compress_state;
struct module;

//
// Begins external API for video compression use
//
/**
 * @brief Initializes video compression
 * 
 * @param[in] parent parent module
 * @param[in] cfg    configuration string
 * @return           driver internal state
 */
typedef struct module *(*compress_init_t)(struct module *parent,
                const char *cfg);
// documented at definition
void show_compress_help(bool full);
// documented at definition
int compress_init(struct module *parent, const char *config_string, struct compress_state **);
// documented at definition
const char *get_compress_name(struct compress_state *);
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
#include <memory>
// documented at definition
void compress_frame(struct compress_state *, std::shared_ptr<video_frame>);
// documented at definition
std::shared_ptr<video_frame> compress_pop(struct compress_state *);

//
// Begins API for individual video compression modules
//
#include <list>
#include <string>
#include <vector>

/**
 * @brief Compresses video frame
 * 
 * @param state        driver internal state
 * @param frame        uncompressed frame
 * @return             compressed frame, may be NULL if compression failed
 */
typedef  std::shared_ptr<video_frame> (*compress_frame_t)(struct module *state, std::shared_ptr<video_frame> frame);

/**
 * @brief Compresses tile of a video frame
 * 
 * @param[in]     state         driver internal state
 * @param[in]     in_frame      uncompressed frame containing exactly one tile
 * @return                      compressed frame with one tile, may be NULL if compression failed
 */
typedef  std::shared_ptr<video_frame> (*compress_tile_t)(struct module *state, std::shared_ptr<video_frame> in_frame);

/**
 * @brief Passes frame to compress module for async processing.
 *
 * compress_frame_async_pop_t() should be called thereafter to fetch compressed frame.
 *
 * @param[in]     state         driver internal state
 * @param[in]     in_frame      uncompressed frame or empty shared_ptr to pass a poisoned pile
 */
typedef void (*compress_frame_async_push_t)(struct module *state, std::shared_ptr<video_frame> in_frame);

/**
 * @brief Fetches compressed frame passed with compress_frame_async_push()
 *
 * @param[in]     state         driver internal state
 * @return                      compressed frame, empty shared_ptr corresponding with poisoned
 *                              pill can be also returned
 */
typedef  std::shared_ptr<video_frame> (*compress_frame_async_pop_t)(struct module *state);

/**
 * @brief Passes tile to compress module for async processing.
 *
 * compress_frame_async_pop_t() should be called thereafter to fetch compressed frame.
 *
 * @param[in]     state         driver internal state
 * @param[in]     in_frame      uncompressed frame or empty shared_ptr to pass a poisoned pile
 */
typedef void (*compress_tile_async_push_t)(struct module *state, std::shared_ptr<video_frame> in_frame);

/**
 * @brief Fetches compressed tile passed with compress_tile_async_push()
 *
 * @param[in]     state         driver internal state
 * @return                      compressed frame, empty shared_ptr corresponding with poisoned
 *                              pill can be also returned
 */
typedef  std::shared_ptr<video_frame> (*compress_tile_async_pop_t)(struct module *state);

struct module_option{
        std::string display_name; //Name displayed to user
        std::string display_desc; //Description displayed to user

        /* internal name of option, options that are used in the same way should
         * have the same key (e.g. both bitrate for libavcodec and quality for jpeg
         * should have the same key). This allows using different labels displayed
         * to user */
        std::string key;
        std::string opt_str; //Option string to pass to ug (e.g. ":bitrate=")

        bool is_boolean; //If true, GUI shows a checkbox instead of text edit
};

struct encoder{
        std::string name;
        std::string opt_str;
};

struct codec{
        std::string name;
        std::vector<encoder> encoders;
        int priority;
};

struct compress_module_info{
        std::string name;
        std::vector<module_option> opts;
        std::vector<codec> codecs;
};

/**
 * There are 4 possible APIs for video compress modules. Each module may choose
 * which one to implement, however, only one should be implemented (there is no
 * "smart" heuristics to pick one if more APIs are implemented). Available options
 * are:
 * 1. Frame API - compress entire frame (all tiles)
 * 2. Tile API - compress one tile
 * 3. Async API - compress a frame asynchronously
 * 4. Async tile API - compress a tile asynchronously
 */
struct video_compress_info {
        const char        * name;         ///< compress (unique) name
        compress_init_t     init_func;           ///< compress driver initialization function

        compress_frame_t    compress_frame_func; ///< compress function for Frame API

        compress_tile_t     compress_tile_func;  ///< compress function for Tile API

        compress_frame_async_push_t compress_frame_async_push_func; ///< Async API
        compress_frame_async_pop_t compress_frame_async_pop_func; ///< Async API

        compress_tile_async_push_t compress_tile_async_push_func; ///< Async tile API
        compress_tile_async_pop_t compress_tile_async_pop_func; ///< Async tile API

        compress_module_info (*get_module_info)();
};

#endif // __cplusplus

#endif /* __video_compress_h */
/** @} */ // end of video_compress


/**
 * @file   video_compress.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup video_compress
 *
 * @brief API for video compress drivers.
 */
/*
 * Copyright (c) 2011-2013 CESNET z.s.p.o.
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
 * @{ */
#ifndef __video_compress_h
#define __video_compress_h

#include "types.h"

#define VIDEO_COMPRESS_ABI_VERSION 7

#ifdef __cplusplus
extern "C" {
#endif

struct compress_state;
struct module;

extern struct module compress_init_noerr;

/** @name API for capture modules
 * @{ */

/**
 * @brief Initializes video compression
 * 
 * @param[in] parent parent module
 * @param[in] cfg    configuration string
 * @return           driver internal state
 */
typedef struct module *(*compress_init_t)(struct module *parent,
                const char *cfg);
/// @}

void show_compress_help(void);
int compress_init(struct module *parent, const char *config_string, struct compress_state **);
const char *get_compress_name(struct compress_state *);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <list>
#include <memory>
#include <string>

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

void compress_frame(struct compress_state *, std::shared_ptr<video_frame>);

struct compress_preset {
        struct compress_prop {
                int latency; // ms
                double cpu_cores;
                double gpu_gflops;
        };

        std::string name;
        int quality;
        long (*compute_bitrate)(const struct video_desc *);
        compress_prop enc_prop;
        compress_prop dec_prop;
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
        std::list<compress_preset> (*get_presets)();    ///< list of available presets
};

std::shared_ptr<video_frame> compress_pop(struct compress_state *);

#endif

#endif /* __video_compress_h */
/** @} */ // end of video_compress


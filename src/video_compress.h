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

#ifdef __cplusplus
extern "C" {
#endif

struct compress_state;
struct module;

extern struct module compress_init_noerr;

/** @name API for capture modules
 * @{ */

struct video_compress_params {
        const char *cfg;
};

/**
 * @brief Initializes video compression
 * 
 * @param[in] parent parent module
 * @param[in] cfg    configuration string
 * @return           driver internal state
 */
typedef struct module *(*compress_init_t)(struct module *parent,
                const struct video_compress_params *params);
/**
 * @brief Compresses video frame
 * 
 * @param state        driver internal state
 * @param frame        uncompressed frame
 * @param buffer_index 0 or 1 - driver should have 2 output buffers, filling the selected one.
 *                     Returned video frame should stay valid until requesting compress with the
 *                     same index.
 * @return             compressed frame, may be NULL if compression failed
 */
typedef  struct video_frame * (*compress_frame_t)(struct module *state, struct video_frame *frame);
/**
 * @brief Compresses tile of a video frame
 * 
 * @param[in]     state         driver internal state
 * @param[in]     in_frame      uncompressed frame
 * @param[in]     tile_index    index of tile to be compressed
 * @param         buffer_index  0 or 1 - driver should have 2 output buffers, filling the selected one.
 *                Returned video frame should stay valid until requesting compress with the
 *                same index.
 * @return                      compressed frame with one tile, may be NULL if compression failed
 */
typedef  struct video_frame * (*compress_tile_t)(struct module *state, struct video_frame *in_frame);
/// @}

void show_compress_help(void);
int compress_init(struct module *parent, const char *config_string, struct compress_state **);
const char *get_compress_name(struct compress_state *);

void compress_frame(struct compress_state *, struct video_frame*);
struct video_frame *compress_pop(struct compress_state *);


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <list>
#include <string>
struct compress_preset {
        struct compress_prop {
                int latency; // ms
                double cpu_cores;
                double gpu_gflops;
        };

        std::string name;
        int quality;
        long bitrate;
        compress_prop enc_prop;
        compress_prop dec_prop;
};

std::list<compress_preset> get_compress_capabilities();
#endif

#endif /* __video_compress_h */
/** @} */ // end of video_compress


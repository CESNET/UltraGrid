/**
 * @file   video_decompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2025 CESNET
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

#include "video_decompress.h"

#include <cassert>        // for assert
#include <cstdint>        // for uint32_t
#include <cstdlib>        // for free, calloc
#include <cstring>        // for strchr, strdup
#include <string>         // for basic_string, char_traits, hash, operator<<

#include "debug.h"        // for LOG, LOG_LEVEL_VERBOSE
#include "host.h"         // for commandline_params, ADD_TO_PARAM
#include "lib_common.h"   // for get_libraries_for_class, library_class
#include "video_codec.h"  // for get_codec_from_name

#define DECOMPRESS_MAGIC 0xdff34f21u

using std::string;

/**
 * This struct represents actual decompress state
 */
struct state_decompress {
        uint32_t magic;             ///< selected decoder magic
        const struct video_decompress_info *functions; ///< pointer to selected decoder functions
        void *state;                ///< decoder driver state
};

ADD_TO_PARAM("decompress", "* decompress=<name>[:<codec>]\n"
                "  Forces specified decompress module (will fail if not able to decompress\n"
                "   the received compression). Optionally also force codec to decode to."
                "   See 'uv --list-modules' to see available decompress modules.\n");
/**
 * @param[in] compression input compression
 * @param[in] out_pixfmt output pixel format
 * @param[in] prio_min minimal priority that can be probed
 * @param[in] prio_max maximal priority that can be probed
 * @param[out] magic if decompressor was found here is stored its magic
 * @retval -1       if no found
 * @retval priority best decoder's priority
 */
static int find_best_decompress(codec_t compression, struct pixfmt_desc internal_prop, codec_t out_pixfmt,
                int prio_min, int prio_max, const struct video_decompress_info **vdi, string & name) {
        auto decomps = get_libraries_for_class(LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

        int best_priority = prio_max + 1;
        string force_module;

        if (commandline_params.find("decompress") != commandline_params.end()) {
                char *tmp = strdup(commandline_params.at("decompress").c_str());
                if (strchr(tmp, ':')) {
                        // if out_pixfmt specified and doesn't match, return
                        if (out_pixfmt != get_codec_from_name(strchr(tmp, ':') + 1)) {
                                free(tmp);
                                return -1;
                        }
                        *strchr(tmp, ':') = '\0';
                }
                force_module = tmp;
                free(tmp);
        }

        for (const auto & d : decomps) {
                // if user has explicitly requested decoder, skip all others
                if (!force_module.empty() && d.first != force_module) {
                        continue;
                }
                // first pass - find the one with best priority (least)
                int priority = static_cast<const video_decompress_info *>(d.second)->get_decompress_priority(compression, internal_prop, out_pixfmt);
                if (priority < 0) { // decoder not valid for given properties combination
                        continue;
                }
                if (priority <= prio_max && priority >= prio_min && priority < best_priority) {
                        best_priority = priority;
                        *vdi = static_cast<const video_decompress_info *>(d.second);
                        name = d.first;
                }
        }

        if(best_priority == prio_max + 1)
                return -1;
        return best_priority;
}

/**
 * Initializes decompressor
 *
 * @retval NULL if initialization failed
 * @retval not-NULL state of new decompressor
 */
static struct state_decompress *decompress_init(const video_decompress_info *vdi)
{
        struct state_decompress *s;

        s = (struct state_decompress *) calloc(1, sizeof(struct state_decompress));
        s->magic = DECOMPRESS_MAGIC;
        s->functions = vdi;
        s->state = s->functions->init();
        if (s->state == NULL) {
                free(s);
                return NULL;
        }
        return s;
}

/**
 * Attempts to initialize decompress of given magic
 *
 * @param[in]  vdi      metadata struct of requested decompressor
 * @param[out] decompress_state decoder state
 * @param[in]  substreams number of decoders to be created
 * @return     true if initialization succeeded
 */
static bool try_initialize_decompress(const video_decompress_info *vdi,
                struct state_decompress **decompress_state, int substreams)
{
        for(int i = 0; i < substreams; ++i) {
                decompress_state[i] = decompress_init(vdi);

                if (!decompress_state[i]) {
                        for(int j = 0; j < i; ++j) {
                                if (decompress_state[j] != nullptr) {
                                        decompress_done(decompress_state[j]);
                                }
                                decompress_state[j] = nullptr;
                        }
                        return false;
                }
        }

        return true;
}

/**
 * @brief Initializes (multiple) decompress states
 *
 * If more than one decompress module is available, load the one with highest priority.
 *
 * @param[in] compression source compression
 * @param[in] out_codec   requested destination pixelformat
 * @param[out] out        pointer (array) to be filled with state_count instances of decompressor
 * @param[in] count       number of decompress states to be created.
 * This is important mainly for interlrame compressions which keeps internal state between individual
 * frames. Different tiles need to have different states then.
 * @retval true           if state_count members of state is filled with valid decompressor
 * @retval false          if initialization failed
 */
bool decompress_init_multi(codec_t compression, struct pixfmt_desc internal_prop, codec_t out_codec, struct state_decompress **out, int count)
{
        int prio_max = 1000;
        int prio_min = 0;
        int prio_cur;
        const video_decompress_info *vdi = nullptr;

        while(1) {
                string name;
                prio_cur = find_best_decompress(compression, internal_prop, out_codec,
                                prio_min, prio_max, &vdi, name);
                // if found, init decoder
                if(prio_cur != -1) {
                        if (try_initialize_decompress(vdi, out, count)) {
                                LOG(LOG_LEVEL_VERBOSE) << "Decompressor \"" << name << "\" initialized successfully.\n";
                                return true;
                        } else {
                                // failed, try to find another one
                                LOG(LOG_LEVEL_VERBOSE) << "Cannot initialize decompressor \"" << name << "\"!\n";
                                prio_min = prio_cur + 1;
                                continue;
                        }
                } else {
                        break;
                }
        }
        return false;
}

/** @copydoc decompress_reconfigure_t */
int decompress_reconfigure(struct state_decompress *s, struct video_desc desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        return s->functions->reconfigure(s->state, desc, rshift, gshift, bshift, pitch, out_codec);
}

/** @copydoc decompress_decompress_t */
decompress_status decompress_frame(
                struct state_decompress *s,
                unsigned char *dst,
                unsigned char *compressed,
                unsigned int compressed_len,
                int frame_seq,
                struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        return s->functions->decompress(s->state,
                        dst,
                        compressed,
                        compressed_len,
                        frame_seq,
                        callbacks,
                        internal_prop);
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


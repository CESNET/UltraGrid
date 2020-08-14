/**
 * @file   video_decompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2019 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <stdio.h>
#include <string.h>
#include <string>
#include "debug.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "lib_common.h"

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
                "   the received compression). Optionaly also force codec to decode to."
                "   See 'uv --list-modules' to see available decompress modules.\n");
/**
 * @param[in] in_codec input codec
 * @param[in] out_codec output codec
 * @param[in] prio_min minimal priority that can be probed
 * @param[in] prio_max maximal priority that can be probed
 * @param[out] magic if decompressor was found here is stored its magic
 * @retval -1       if no found
 * @retval priority best decoder's priority
 */
static int find_best_decompress(codec_t in_codec, codec_t internal, codec_t out_codec,
                int prio_min, int prio_max, const struct video_decompress_info **vdi, string & name) {
        auto decomps = get_libraries_for_class(LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

        int best_priority = prio_max + 1;
        string force_module;

        if (commandline_params.find("decompress") != commandline_params.end()) {
                char *tmp = strdup(commandline_params.at("decompress").c_str());
                if (strchr(tmp, ':')) {
                        // if out_codec specified and doesn't match, return
                        if (out_codec != get_codec_from_name(strchr(tmp, ':') + 1)) {
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
                const struct decode_from_to *f = static_cast<const video_decompress_info *>(d.second)->get_available_decoders();
                while (f->from != VIDEO_CODEC_NONE) {
                        if(in_codec == f->from && internal == f->internal && out_codec == f->to) {
                                int priority = f->priority;
                                if(priority <= prio_max &&
                                                priority >= prio_min &&
                                                priority < best_priority) {
                                        best_priority = priority;
                                        *vdi = static_cast<const video_decompress_info *>(d.second);
                                        name = d.first;
                                }
                        }

                        f++;
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
 * Attemps to initialize decompress of given magic
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
 * @param[in] in_codec    source compression
 * @param[in] out_codec   requested destination pixelformat
 * @param[out] out        pointer (array) to be filled with state_count instances of decompressor
 * @param[in] count       number of decompress states to be created.
 * This is important mainly for interlrame compressions which keeps internal state between individual
 * frames. Different tiles need to have different states then.
 * @retval true           if state_count members of state is filled with valid decompressor
 * @retval false          if initialization failed
 */
bool decompress_init_multi(codec_t in_codec, codec_t internal_codec, codec_t out_codec, struct state_decompress **out, int count)
{
        int prio_max = 1000;
        int prio_min = 0;
        int prio_cur;
        const video_decompress_info *vdi = nullptr;

        while(1) {
                string name;
                prio_cur = find_best_decompress(in_codec, internal_codec, out_codec,
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
        LOG(LOG_LEVEL_VERBOSE) << "Could not find or initialize any suitable decompressor!\n";
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
                codec_t *internal_codec)
{
        assert(s->magic == DECOMPRESS_MAGIC);

        return s->functions->decompress(s->state,
                        dst,
                        compressed,
                        compressed_len,
                        frame_seq,
                        callbacks,
                        internal_codec);
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


/**
 * @file   capture_filter/gamma.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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
#endif /* HAVE_CONFIG_H */

#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "rang.hpp"
#include "utils/color_out.h"
#include "utils/worker.h"
#include "video.h"
#include "video_codec.h"
#include "vo_postprocess/capture_filter_wrapper.h"

constexpr const char *MOD_NAME = "[gamma cap. f.] ";

using std::cout;
using std::exception;
using std::numeric_limits;
using rang::style;
using std::vector;
using std::thread;

struct state_capture_filter_gamma {
public:
        int out_depth; ///< 0, 8 or 16 (0 menas keep)
        void *vo_pp_out_buffer{}; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)

        explicit state_capture_filter_gamma(double gamma, int out_depth) : out_depth(out_depth) {
                for (int i = 0; i <= numeric_limits<uint8_t>::max(); ++i) { // 8->8
                        lut8.push_back(pow(static_cast<double>(i)
                                        / numeric_limits<uint8_t>::max(), gamma)
                                * numeric_limits<uint8_t>::max());
                }
                for (int i = 0; i <= numeric_limits<uint16_t>::max(); ++i) { // 8->16
                        lut16.push_back(pow(static_cast<double>(i)
                                        / numeric_limits<uint16_t>::max(), gamma)
                                * numeric_limits<uint16_t>::max());
                }
                for (int i = 0; i <= numeric_limits<uint8_t>::max(); ++i) { // 8->16
                        lut8_16.push_back(pow(static_cast<double>(i)
                                        / numeric_limits<uint8_t>::max(), gamma)
                                * numeric_limits<uint16_t>::max());
                }
                for (int i = 0; i <= numeric_limits<uint16_t>::max(); ++i) { // 16->8
                        lut16_8.push_back(pow(static_cast<double>(i)
                                        / numeric_limits<uint16_t>::max(), gamma)
                                * numeric_limits<uint8_t>::max());
                }
        }

        void apply_gamma(int in_depth, int out_depth, size_t in_len, void const * __restrict in, void * __restrict out) {
                if (in_depth == CHAR_BIT && out_depth == CHAR_BIT) {
                        apply_lut<uint8_t, uint8_t>(in_len, lut8, in, out);
                } else if (in_depth == 2 * CHAR_BIT && out_depth == 2 * CHAR_BIT) {
                        apply_lut<uint16_t, uint16_t>(in_len, lut16, in, out);
                } else if (in_depth == CHAR_BIT && out_depth == 2 * CHAR_BIT) {
                        apply_lut<uint8_t, uint16_t>(in_len, lut8_16, in, out);
                } else if (in_depth == 2 * CHAR_BIT && out_depth == CHAR_BIT) {
                        apply_lut<uint16_t, uint8_t>(in_len, lut16_8, in, out);
                } else {
                        throw exception();
                }
        }

private:
        template<typename inT, typename outT>
        struct data {
                size_t len;
                const vector<outT> &lut;
                const inT *in;
                outT *out;
        };

        template<typename inT, typename outT>
        static void *compute(void *arg) {
                auto *d = static_cast<struct data<inT, outT> *>(arg);
                for (size_t i = 0; i < d->len; ++i) {
                        d->out[i] = d->lut[d->in[i]];
                }
                return nullptr;
        }

        template<typename inT, typename outT> void apply_lut(size_t in_len, const vector<outT> &lut, void const *in, void *out)
        {
                auto *in_data = static_cast<const inT*>(in);
                auto *out_data = static_cast<outT*>(out);
                unsigned int cpus = thread::hardware_concurrency();
                in_len /= sizeof(inT);
                vector<data<inT, outT>> d;
                vector<task_result_handle_t> handles(cpus);
                for (unsigned int i = 0; i < cpus; i++) {
                        d.push_back({in_len / cpus, lut, in_data + i * (in_len / cpus), out_data + i * (in_len / cpus)});
                }
                for (unsigned int i = 0; i < cpus; i++) {
                        handles[i] = task_run_async(state_capture_filter_gamma::compute<inT, outT>, static_cast<void *>(&(d[i])));
                }
                for (unsigned int i = 0; i < cpus; i++) {
                        wait_task(handles[i]);
                }
        }

        vector<uint8_t>  lut8;
        vector<uint16_t> lut16;
        vector<uint8_t>  lut16_8;
        vector<uint16_t> lut8_16;
};

static auto init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (strlen(cfg) == 0 || strcmp(cfg, "help") == 0) {
                cout << "Performs gamma transformation.\n\n"
                       "usage:\n";
                cout << style::bold << "\t--capture-filter gamma:value[:8|:16]\n" << style::reset;
                cout << "where:\n";
                cout << style::bold << "\t8|16" << style::reset << " - force output to 8 (16) bits regardless the input\n";
                return 1;
        }
        char *endptr = nullptr;
        errno = 0;
        double gamma = strtod(cfg, &endptr);

        if (gamma <= 0.0 || errno != 0 || (*endptr != '\0' && *endptr != ':')) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Using gamma value " << gamma << "\n";
        }

        long int bits = 0;
        if (*endptr != '\0') {
                endptr += 1;
                errno = 0;
                bits = strtol(endptr, &endptr, 0);
                if ((bits != 8 && bits != 16) || *endptr != '\0') {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong number of bits (only 8 or 16)!\n";
                        return -1;
                }
        }

        auto *s = new state_capture_filter_gamma(gamma, bits);

        *state = s;
        return 0;
}

static void done(void *state)
{
        delete static_cast<state_capture_filter_gamma *>(state);
}

static auto filter(void *state, struct video_frame *in) -> video_frame *
{
        if (in->color_spec != RGB && in->color_spec != RG48) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to apply lut on: " << get_codec_name(in->color_spec) << "\n";
                VIDEO_FRAME_DISPOSE(in);
                return nullptr;
        }

        auto *s = static_cast<state_capture_filter_gamma *>(state);
        struct video_desc out_desc = video_desc_from_frame(in);
        if (s->out_depth != 0) {
                out_desc.color_spec = s->out_depth == 8 ? RGB : RG48;
        }
        struct video_frame *out = vf_alloc_desc(out_desc);
        if (s->vo_pp_out_buffer != nullptr) {
                out->tiles[0].data = (char *) s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = (char *) malloc(out->tiles[0].data_len);
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        try {
                s->apply_gamma(get_bits_per_component(in->color_spec), get_bits_per_component(out_desc.color_spec), in->tiles[0].data_len, in->tiles[0].data, out->tiles[0].data);
        } catch(...) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Only 8-bit and 16-bit codecs are currently supported!\n";
                vf_free(out);
                out = nullptr;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        auto *s = (state_capture_filter_gamma *) state;
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_gamma = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(gamma, &capture_filter_gamma, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(gamma, init, filter, done, vo_pp_set_out_buffer)

/* vim: set expandtab sw=8: */

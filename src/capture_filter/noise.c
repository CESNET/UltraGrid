/**
 * @file   capture_filter/noise.c
 * @author Martin Pulec <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#include <limits.h> // for UCHAR_MAX, UINT_MAX
#include <stdlib.h> // for calloc, free, strtol
#include <string.h> // for memcpy, strchr, strcmp, strdup, strtok_r

#include "capture_filter.h"  // for CAPTURE_FILTER_ABI_VERSION, capture_fil...
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "types.h"           // for tile, video_frame, video_frame_callbacks
#include "utils/color_out.h" // for color_printf, TBOLD, TRED
#include "utils/macros.h"    // for IS_KEY_PREFIX
#include "utils/random.h"    // for ug_rand
#include "video_frame.h"     // for vf_alloc_desc, video_desc_from_frame
struct module;

#define MOD_NAME "[cf/noise] "

#if __STDC_VERSION__ < 202311L
#define nullptr NULL
#endif

static void done(void *state);

enum {
        DEFAULT_MAGNITUDE = 200,
};

struct state_noise {
        unsigned magnitude;
};

static void
usage()
{
        color_printf(TRED(TBOLD("noise")) " capture filter adds white noise\n");
        color_printf("\n");

        color_printf("THe main use case is for testing (to add entropy for compression).\n");
        color_printf("See also \"noise\" parameter of " TBOLD("testcard2") ".\n");
        color_printf("\n");

        color_printf("Usage:\n");
        color_printf(
            "\t" TBOLD(TRED("-F noise") "[:magnitude=<m>]") " (default %d)\n",
            DEFAULT_MAGNITUDE);
        color_printf("\t" TBOLD("-F noise:help") "\n");
}

static int
parse_fmt(struct state_noise *s, char *ccfg)
{
        if (strcmp(ccfg, "help") == 0) {
                return 1;
        }
        char *item   = nullptr;
        char *endptr = nullptr;
        while ((item = strtok_r(ccfg, ":", &endptr)) != nullptr) {
                ccfg = nullptr;
                if (IS_KEY_PREFIX(item, "magnitude")) {
                        char *endptr = nullptr;
                        const long val =
                            strtol(strchr(item, '=') + 1, &endptr, 0);
                        if (val <= 0 || val > UINT_MAX || *endptr != '\0') {
                                MSG(ERROR, "Wrong value given: %ld!\n", val);
                                return -1;
                        }
                        s->magnitude = val;
                } else {
                        MSG(ERROR, "Unknown option: %s!", item);
                        return -1;
                }
        }
        return 0;
}

static int
init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;
        struct state_noise *s = calloc(1, sizeof(struct state_noise));
        s->magnitude          = DEFAULT_MAGNITUDE;
        char *ccfg            = strdup(cfg);
        int   rc              = parse_fmt(s, ccfg);
        free(ccfg);
        if (rc != 0) {
                usage();
                done(s);
                return rc;
        }

        *state = s;
        return 0;
}

static void
done(void *state)
{
        free(state);
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        struct state_noise *s = state;

        struct video_frame *out = vf_alloc_desc_data(video_desc_from_frame(in));
        out->callbacks.dispose  = vf_free;

        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

        memcpy(out_data, in->tiles[0].data, in->tiles[0].data_len);

        const unsigned char *const end = out_data + in->tiles[0].data_len;
        out_data += (ug_rand() % s->magnitude);
        while (out_data < end) {
                *out_data = ug_rand() % (UCHAR_MAX + 1);
                out_data += 1 + (ug_rand() % s->magnitude);
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static const struct capture_filter_info capture_filter_noise = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(noise, &capture_filter_noise, LIBRARY_CLASS_CAPTURE_FILTER,
                CAPTURE_FILTER_ABI_VERSION);

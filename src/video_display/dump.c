/**
 * @file   video_display/dump.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016-2023 CESNET, z. s. p. o.
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
 * @file
 * @todo Add audio support
 */
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <stdint.h>

#include "debug.h"
#include "export.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "video.h"
#include "video_display.h"

#define MOD_NAME "[dump] "

struct dump_display_state {
        struct video_frame *f;
        int frames;
        struct exporter *e;
        size_t max_tile_data_len;
        codec_t requested_codec;
};

static void usage()
{
        color_printf("Usage:\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-d dump" TERM_FG_RESET "[:<directory>] [--param decoder-use-codec=<c>]\n" TERM_RESET);
        color_printf("where\n");
        color_printf(TERM_BOLD "\t<directory>" TERM_RESET " - directory to save the dumped stream\n");
        color_printf(TERM_BOLD "\t<c>" TERM_RESET " - codec to use instead of the received (default), must be a way to convert\n");
}

static void *display_dump_init(struct module *parent, const char *cfg, unsigned int flags)
{
        (void) parent, (void) flags;
        if (strcmp(cfg, "help") == 0) {
                usage();
                return INIT_NOERR;
        }
        struct dump_display_state *s = calloc(1, sizeof *s);
        char dirname[128];
        if (strlen(cfg) == 0) {
                time_t     t      = time(NULL);
                struct tm  tm_buf = { 0 };
                localtime_s(&t, &tm_buf);
                strftime(dirname, sizeof dirname, "dump.%Y%m%dT%H%M%S",
                         &tm_buf);
                cfg = dirname;
        }
        s->e = export_init(NULL, cfg, true);
        if (s->e == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[dump] Failed to create export instance!\n");
                free(s);
                return NULL;
        }
        return s;
}

static void display_dump_done(void *state)
{
        struct dump_display_state *s = state;

        vf_free(s->f);
        export_destroy(s->e);
        free(s);
}

static struct video_frame *display_dump_getf(void *state)
{
        struct dump_display_state *s = state;
        for (unsigned int i = 0; i < s->f->tile_count; ++i) {
                if (is_codec_opaque(s->f->color_spec)) {
                        s->f->tiles[i].data_len = s->max_tile_data_len;
                }
        }
        return s->f;
}

static bool display_dump_putf(void *state, struct video_frame *frame, long long flags)
{
        struct dump_display_state *s = state;
        if (frame == NULL || flags == PUTF_DISCARD) {
                return true;
        }
        assert(frame == s->f);
        export_video(s->e, frame);

        return true;
}

static bool display_dump_get_property(void *state, int property, void *val, size_t *len)
{
        (void) state;
        codec_t codecs[VIDEO_CODEC_COUNT - 1];

        for (int i = 0; i < VIDEO_CODEC_COUNT - 1; ++i) {
                codecs[i] = (codec_t) (i + 1); // all codecs (exclude VIDEO_CODEC_NONE)
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                                *len = sizeof(codecs);
                        } else {
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        *len = sizeof(int);
                        break;
                default:
                        return false;
        }
        return true;
}

static bool display_dump_reconfigure(void *state, struct video_desc desc)
{
        struct dump_display_state *s = state;
        vf_free(s->f);
        s->f = vf_alloc_desc(desc);
        s->f->decoder_overrides_data_len = is_codec_opaque(desc.color_spec) != 0 ? TRUE : FALSE;
        s->max_tile_data_len = MAX(8 * desc.width * desc.height, 1000000UL);
        for (unsigned int i = 0; i < s->f->tile_count; ++i) {
                if (is_codec_opaque(desc.color_spec)) {
                        s->f->tiles[i].data_len = s->max_tile_data_len;
                }
                s->f->tiles[i].data = (char *) malloc(s->f->tiles[i].data_len);
        }
        s->f->callbacks.data_deleter = vf_data_deleter;

        return true;
}

static void display_dump_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = NULL;
        *count = 0;
}

static const struct video_display_info display_dump_info = {
        display_dump_probe,
        display_dump_init,
        NULL, // _run
        display_dump_done,
        display_dump_getf,
        display_dump_putf,
        display_dump_reconfigure,
        display_dump_get_property,
        NULL, // _put_audio_frame
        NULL, // _reconfigure_audio,
        MOD_NAME,
};

REGISTER_MODULE(dump, &display_dump_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


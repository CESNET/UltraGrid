/**
 * @file   video_display/dummy.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2023 CESNET, z. s. p. o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"

#define DEFAULT_DUMP_LEN 32
#define MOD_NAME "[dummy] "

static const codec_t default_codecs[] = {I420, UYVY, YUYV, v210, R10k, R12L, RGBA, RGB, BGR, RG48};

struct dummy_display_state {
        struct video_frame *f;
        codec_t codecs[VIDEO_CODEC_COUNT];
        size_t codec_count;
        int rgb_shift[3];

        size_t dump_bytes;
        _Bool dump_to_file;
        _Bool oneshot;
        _Bool raw;
        int dump_to_file_skip_frames;
};

static _Bool dummy_parse_opts(struct dummy_display_state *s, char *fmt) {
        char *item = NULL;
        char *save_ptr = NULL;
        while ((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                fmt = NULL;
                if (strstr(item, "codec=") != NULL) {
                        char *sptr = NULL;
                        char *tok = NULL;
                        char *str = strchr(item, '=') + 1;
                        s->codec_count = 0;
                        while ((tok = strtok_r(str, ",", &sptr))) {
                                str = NULL;
                                assert(s->codec_count < sizeof s->codecs / sizeof s->codecs[0]);
                                s->codecs[s->codec_count++] = get_codec_from_name(tok);
                                if (s->codecs[s->codec_count - 1] == VIDEO_CODEC_NONE) {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong codec spec: %s!\n", tok);
                                        return 0;
                                }
                        }
                } else if (strstr(item, "dump") != NULL) {
                        s->dump_to_file = 1;
                } else if (strstr(item, "hexdump") != NULL) {
                        if (strstr(item, "hexdump=") != NULL) {
                                s->dump_bytes = atoi(item + strlen("hexdump="));
                        } else {
                                s->dump_bytes = DEFAULT_DUMP_LEN;
                        }
                } else if (strstr(item, "rgb_shift=") != NULL) {
                        item += strlen("rgb_shift=");
                        s->rgb_shift[0] = strtol(item, &item, 0);
                        item += 1;
                        s->rgb_shift[1] = strtol(item, &item, 0);
                        item += 1;
                        s->rgb_shift[2] = strtol(item, &item, 0);
                } else if (strstr(item, "skip=") != NULL) {
                        s->dump_to_file_skip_frames = atoi(strchr(item, '=') + 1);
                } else if (strcmp(item, "oneshot") == 0) {
                        s->oneshot = 1;
                } else if (strcmp(item, "raw") == 0) {
                        s->raw = 1;
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unrecognized option: %s\n", item);
                        return 0;
                }
        }
        if (!s->dump_to_file && (s->oneshot || s->dump_to_file_skip_frames > 0 || s->raw)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Some of passed options don't do anything without \":dump\".\n");
        }
        return 1;
}

static void *display_dummy_init(struct module *parent, const char *cfg, unsigned int flags)
{
        UNUSED(parent), UNUSED(flags);
        if (strcmp(cfg, "help") == 0) {
                char desc[] = "Display " TBOLD("dummy") " only consumes the video without displaying it. A difference to avoiding "
                                "display specification is that it forces decoding (otherwise it will be skipped altogether).\n\n"
                                "Additionally, options " TBOLD("hexdump") " and " TBOLD("dump") " are available for debugging.\n\n";
                color_printf("%s", indent_paragraph(desc));
                struct key_val options[] = {
                        { "codec=<codec>[,<codec2>]", "force the use of a codec instead of default set" },
                        { "rgb_shift=<r>,<g>,<b>", "if using output codec RGBA, use specified shifts instead of default (" TOSTRING(DEFAULT_R_SHIFT) ", " TOSTRING(DEFAULT_G_SHIFT) ", " TOSTRING(DEFAULT_B_SHIFT) ")" },
                        { "dump[:skip=<n>][:oneshot][:raw]", "dump first frame to file dummy.<ext> (optionally skip <n> first frames); 'oneshot' - exit after dumping the picture; 'raw' - dump raw data" },
                        { "hexdump[=<n>]", "dump first n (default " TOSTRING(DEFAULT_DUMP_LEN) ") bytes of every frame in hexadecimal format" },
                        { NULL, NULL }
                };
                print_module_usage("-d dummy", options, NULL, 0);
                return INIT_NOERR;
        }
        struct dummy_display_state s = { .codec_count = sizeof default_codecs / sizeof default_codecs[0] };
        memcpy(s.codecs, default_codecs, sizeof default_codecs);
        int rgb_shift_init[] = DEFAULT_RGB_SHIFT_INIT;
        memcpy(s.rgb_shift, &rgb_shift_init, sizeof s.rgb_shift);
        char *ccpy = alloca(strlen(cfg) + 1);
        strcpy(ccpy, cfg);

        if (!dummy_parse_opts(&s, ccpy)) {
                return NULL;
        }

        struct dummy_display_state *ret = malloc(sizeof s);
        memcpy(ret, &s, sizeof s);

        return ret;
}

static void display_dummy_done(void *state)
{
        struct dummy_display_state *s = state;

        vf_free(s->f);
        free(s);
}

static struct video_frame *display_dummy_getf(void *state)
{
        return ((struct dummy_display_state *) state)->f;
}

static void dump_buf(unsigned char *buf, size_t len, int block_size) {
        printf("Frame content: ");
        for (size_t i = 0; i < len; ++i) {
                printf("%02hhx ", *buf++);
                if (block_size > 0 && (i + 1) % block_size == 0) {
                        printf(" ");
                }
        }
        printf("\n");
}

static int display_dummy_putf(void *state, struct video_frame *frame, long long flags)
{
        if (flags == PUTF_DISCARD || frame == NULL) {
                return 0;
        }
        struct dummy_display_state *s = state;
        if (s->dump_bytes > 0) {
                dump_buf((unsigned char *)(frame->tiles[0].data), MIN(frame->tiles[0].data_len, s->dump_bytes), get_pf_block_bytes(frame->color_spec));
        }
        if (s->dump_to_file) {
                if (s->dump_to_file_skip_frames-- == 0) {
                        const char *filename = save_video_frame(frame, "dummy", s->raw);
                        if (filename) {
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Written dump to file %s\n", filename);
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to write dump!\n");
                        }
                        s->dump_to_file = false;
                        if (s->oneshot) {
                                exit_uv(0);
                        }
                }
        }

        return 0;
}

static int display_dummy_get_property(void *state, int property, void *val, size_t *len)
{
        struct dummy_display_state *s = state;

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        {
                                size_t req_len = s->codec_count * sizeof(codec_t);
                                if (req_len > *len) {
                                        return FALSE;
                                }
                                *len = req_len;
                                memcpy(val, s->codecs, *len);
                        }
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if (sizeof s->rgb_shift > *len) {
                                return FALSE;
                        }
                        *len = sizeof s->rgb_shift;
                        memcpy(val, s->rgb_shift, *len);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_dummy_reconfigure(void *state, struct video_desc desc)
{
        struct dummy_display_state *s = state;
        vf_free(s->f);
        s->f = vf_alloc_desc_data(desc);

        return TRUE;
}

static void display_dummy_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = NULL;
        *count = 0;
}

static const struct video_display_info display_dummy_info = {
        display_dummy_probe,
        display_dummy_init,
        NULL, // _run
        display_dummy_done,
        display_dummy_getf,
        display_dummy_putf,
        display_dummy_reconfigure,
        display_dummy_get_property,
        NULL, // _put_audio_frame
        NULL, // _reconfigure_audio
        DISPLAY_DOESNT_NEED_MAINLOOP,
        MOD_NAME,
};

REGISTER_MODULE(dummy, &display_dummy_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


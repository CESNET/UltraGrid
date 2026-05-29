/**
 * @file   utils/overlay_config.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Parser for the overlay postprocessor's configuration string.
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat/c23.h"  // countof
#include "debug.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/overlay_config.h"

#define MOD_NAME "[overlay_config] "

static const struct {
        const char *kw;
        enum overlay_position pos;
} POSITIONS[] = {
        {"center",       OVERLAY_POS_CENTER},
        {"top_left",     OVERLAY_POS_TOP_LEFT},
        {"top_right",    OVERLAY_POS_TOP_RIGHT},
        {"bottom_left",  OVERLAY_POS_BOTTOM_LEFT},
        {"bottom_right", OVERLAY_POS_BOTTOM_RIGHT},
        {"custom",       OVERLAY_POS_CUSTOM},
};

static const struct {
        const char *kw;
        enum overlay_scale_filter filter;
} FILTERS[] = {
        {"nearest",       OVERLAY_SCALE_NEAREST},
        {"fast_bilinear", OVERLAY_SCALE_FAST_BILINEAR},
        {"bilinear",      OVERLAY_SCALE_BILINEAR},
        {"bicubic",       OVERLAY_SCALE_BICUBIC},
        {"lanczos",       OVERLAY_SCALE_LANCZOS},
};

static bool parse_position(const char *val, enum overlay_position *out)
{
        for (size_t i = 0; i < countof(POSITIONS); i++) {
                if (strcmp(val, POSITIONS[i].kw) == 0) {
                        *out = POSITIONS[i].pos;
                        return true;
                }
        }
        return false;
}

static bool parse_scale_filter(const char *val,
                               enum overlay_scale_filter *out)
{
        for (size_t i = 0; i < countof(FILTERS); i++) {
                if (strcmp(val, FILTERS[i].kw) == 0) {
                        *out = FILTERS[i].filter;
                        return true;
                }
        }
        return false;
}

/* Wrap parse_number() (which uses INT_MIN as its error sentinel) so the
 * caller gets a plain success/fail and can still accept INT_MIN+1. */
static bool parse_coord(const char *val, int *out)
{
        const int n = parse_number(val, INT_MIN + 1, 10);
        if (n == INT_MIN) return false;
        *out = n;
        return true;
}

bool overlay_config_parse(const char *opts, struct overlay_config *out)
{
        memset(out, 0, sizeof *out);
        out->position = OVERLAY_POS_CENTER;

        if (opts == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "missing options\n");
                return false;
        }

        /* strtok_r mutates its input — walk a heap copy. */
        char *buf = strdup(opts);
        if (buf == NULL) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "out of memory parsing options\n");
                return false;
        }

        bool ok = false;
        char *saveptr = NULL;
        for (char *tok = strtok_r(buf, ":", &saveptr); tok != NULL;
             tok = strtok_r(NULL, ":", &saveptr)) {
                if (strcmp(tok, "help") == 0) {
                        out->help = true;
                        ok = true;
                        goto out;
                }
                if (strcmp(tok, "perf") == 0) {
                        out->perf = true;
                        continue;
                }
                char *eq = strchr(tok, '=');
                if (eq == NULL) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "missing '=' in '%s'\n", tok);
                        goto out;
                }
                *eq = '\0';
                const char *key = tok;
                char *val = eq + 1;          /* mutable: points into buf */
                if (*val == '\0') {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "empty value for '%s'\n", key);
                        goto out;
                }

                if (strcmp(key, "file") == 0) {
                        const int n = snprintf(out->file, sizeof out->file,
                                               "%s", val);
                        if (n < 0 || (size_t)n >= sizeof out->file) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "file path too long\n");
                                goto out;
                        }
                } else if (strcmp(key, "position") == 0) {
                        if (!parse_position(val, &out->position)) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "unknown position '%s'\n", val);
                                goto out;
                        }
                } else if (strcmp(key, "custom_x") == 0) {
                        if (!parse_coord(val, &out->custom_x)) goto out;
                        out->position = OVERLAY_POS_CUSTOM;
                } else if (strcmp(key, "custom_y") == 0) {
                        if (!parse_coord(val, &out->custom_y)) goto out;
                        out->position = OVERLAY_POS_CUSTOM;
                } else if (strcmp(key, "scale_filter") == 0) {
                        if (!parse_scale_filter(val, &out->scale_filter)) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "unknown scale_filter '%s' "
                                        "(try nearest, fast_bilinear, bilinear, "
                                        "bicubic, lanczos)\n", val);
                                goto out;
                        }
                } else if (strcmp(key, "blend_threads") == 0) {
                        if (!parse_coord(val, &out->blend_threads)) goto out;
                        if (out->blend_threads < 0
                            || out->blend_threads > MAX_CPU_CORES) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "blend_threads must be in [0, %d]\n",
                                        MAX_CPU_CORES);
                                goto out;
                        }
                } else if (strcmp(key, "soft_edge") == 0) {
                        if (!parse_coord(val, &out->soft_edge)) goto out;
                        if (out->soft_edge < 0) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "soft_edge must be >= 0\n");
                                goto out;
                        }
                } else if (strcmp(key, "scale") == 0) {
                        if (strcmp(val, "frame") == 0) {
                                /* scale=frame: track current frame
                                 * dimensions; clear any previous WxH
                                 * (last-one-wins). */
                                out->scale_to_frame = true;
                                out->scale_w = 0;
                                out->scale_h = 0;
                                continue;
                        }
                        char *x = strchr(val, 'x');
                        if (x == NULL) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "scale must be WxH or 'frame' (got '%s')\n",
                                        val);
                                goto out;
                        }
                        *x = '\0';
                        if (!parse_coord(val,   &out->scale_w)) goto out;
                        if (!parse_coord(x + 1, &out->scale_h)) goto out;
                        if (out->scale_w <= 0 || out->scale_h <= 0) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME "scale dimensions must be > 0\n");
                                goto out;
                        }
                        /* Explicit WxH overrides any previous scale=frame. */
                        out->scale_to_frame = false;
                } else {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "unknown key '%s'\n", key);
                        goto out;
                }
        }

        if (out->file[0] == '\0') {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "missing 'file=' option\n");
                goto out;
        }

        /* Auto-default blend_threads when unset (= 0). The work-area guard
         * in vo_postprocess/overlay still routes small overlays to the
         * serial path, so a generous default doesn't waste cores on tiny
         * rects. min(ncpu, 8) — past 8 the dispatch overhead grows faster
         * than the per-thread work shrinks for our blend sizes. Users
         * who want explicit single-threaded blend pass blend_threads=1. */
        if (out->blend_threads == 0) {
                int n = get_cpu_core_count();
                if (n < 1) n = 1;
                if (n > 8) n = 8;
                out->blend_threads = n;
        }
        ok = true;

out:
        free(buf);
        return ok;
}

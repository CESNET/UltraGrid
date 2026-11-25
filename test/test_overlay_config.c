/**
 * @file   test/test_overlay_config.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_config.c
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

#include <stdbool.h>
#include <string.h>

#include "test_overlay_config.h"
#include "unit_common.h"
#include "utils/overlay_config.h"

/* file=PATH alone: position defaults to center; help is false. */
int overlay_config_test_minimal_file_only(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse("file=/tmp/x.pam", &c));
        ASSERT_MESSAGE("file matches",
                       strcmp(c.file, "/tmp/x.pam") == 0);
        ASSERT_EQUAL_MESSAGE("position default",
                             OVERLAY_POS_CENTER, (int)c.position);
        ASSERT_MESSAGE("not help", !c.help);
        return 0;
}

/* All five preset position keywords map to their enum values. */
int overlay_config_test_position_keywords(void)
{
        struct overlay_config c;
        const struct {
                const char *kw;
                enum overlay_position pos;
        } cases[] = {
                {"center",       OVERLAY_POS_CENTER},
                {"top_left",     OVERLAY_POS_TOP_LEFT},
                {"top_right",    OVERLAY_POS_TOP_RIGHT},
                {"bottom_left",  OVERLAY_POS_BOTTOM_LEFT},
                {"bottom_right", OVERLAY_POS_BOTTOM_RIGHT},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
                char buf[64];
                snprintf(buf, sizeof buf, "file=a.pam:position=%s",
                         cases[i].kw);
                ASSERT_MESSAGE(cases[i].kw, overlay_config_parse(buf, &c));
                ASSERT_EQUAL_MESSAGE(cases[i].kw, (int)cases[i].pos,
                                     (int)c.position);
        }
        return 0;
}

/* custom_x / custom_y promote position to OVERLAY_POS_CUSTOM and parse signed
 * integers (negative = from-edge). */
int overlay_config_test_custom_xy(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse(
                               "file=a.pam:custom_x=10:custom_y=-20", &c));
        ASSERT_EQUAL_MESSAGE("position", OVERLAY_POS_CUSTOM, (int)c.position);
        ASSERT_EQUAL_MESSAGE("x",  10, c.custom_x);
        ASSERT_EQUAL_MESSAGE("y", -20, c.custom_y);
        return 0;
}

/* "help" alone is allowed with no file (the postprocessor prints usage). */
int overlay_config_test_help(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse help", overlay_config_parse("help", &c));
        ASSERT_MESSAGE("help flag", c.help);
        return 0;
}

int overlay_config_test_rejects_missing_file(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("no file: reject",
                       !overlay_config_parse("position=center", &c));
        return 0;
}

int overlay_config_test_rejects_unknown_key(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("typo: reject",
                       !overlay_config_parse("file=a.pam:positon=center", &c));
        return 0;
}

int overlay_config_test_rejects_bad_position(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("middle: reject",
                       !overlay_config_parse("file=a.pam:position=middle", &c));
        return 0;
}

int overlay_config_test_rejects_non_integer_xy(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("alpha x: reject",
                       !overlay_config_parse("file=a.pam:custom_x=abc", &c));
        ASSERT_MESSAGE("trailing junk: reject",
                       !overlay_config_parse("file=a.pam:custom_x=10x", &c));
        return 0;
}

int overlay_config_test_rejects_null_and_empty_value(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("NULL opts: reject", !overlay_config_parse(NULL, &c));
        ASSERT_MESSAGE("empty position value: reject",
                       !overlay_config_parse("file=a.pam:position=", &c));
        ASSERT_MESSAGE("empty file value: reject",
                       !overlay_config_parse("file=", &c));
        return 0;
}

int overlay_config_test_soft_edge(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse("file=a.pam:soft_edge=12", &c));
        ASSERT_EQUAL_MESSAGE("soft_edge", 12, c.soft_edge);

        ASSERT_MESSAGE("default zero",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default", 0, c.soft_edge);

        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:soft_edge=-3", &c));
        return 0;
}

int overlay_config_test_scale(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse 320x240",
                       overlay_config_parse("file=a.pam:scale=320x240", &c));
        ASSERT_EQUAL_MESSAGE("scale_w", 320, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h", 240, c.scale_h);

        ASSERT_MESSAGE("default zero",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default w", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("default h", 0, c.scale_h);

        ASSERT_MESSAGE("missing 'x' rejected",
                       !overlay_config_parse("file=a.pam:scale=320", &c));
        ASSERT_MESSAGE("zero rejected",
                       !overlay_config_parse("file=a.pam:scale=0x240", &c));
        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:scale=-10x240", &c));
        return 0;
}

/* scale=frame parses, sets scale_to_frame=true, and zeroes the
 * numeric scale dimensions. */
int overlay_config_test_scale_frame(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse scale=frame",
                       overlay_config_parse("file=a.pam:scale=frame", &c));
        ASSERT_MESSAGE("scale_to_frame set",       c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w zero", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h zero", 0, c.scale_h);

        ASSERT_MESSAGE("default off",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("scale_to_frame default false", !c.scale_to_frame);
        return 0;
}

/* Last-one-wins between scale=WxH and scale=frame. */
int overlay_config_test_scale_frame_overrides_wxh(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("frame after WxH",
                       overlay_config_parse(
                               "file=a.pam:scale=320x240:scale=frame", &c));
        ASSERT_MESSAGE("scale_to_frame set",       c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w cleared", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h cleared", 0, c.scale_h);

        ASSERT_MESSAGE("WxH after frame",
                       overlay_config_parse(
                               "file=a.pam:scale=frame:scale=320x240", &c));
        ASSERT_MESSAGE("scale_to_frame cleared",     !c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w set",   320, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h set",   240, c.scale_h);
        return 0;
}

int overlay_config_test_perf(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse perf flag",
                       overlay_config_parse("file=a.pam:perf", &c));
        ASSERT_MESSAGE("perf set", c.perf);

        ASSERT_MESSAGE("default off",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("perf default false", !c.perf);
        return 0;
}

int overlay_config_test_scale_filter(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("default bicubic",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default", OVERLAY_SCALE_BICUBIC, (int)c.scale_filter);

        const struct {
                const char *name;
                enum overlay_scale_filter expected;
        } cases[] = {
                {"nearest",       OVERLAY_SCALE_NEAREST},
                {"fast_bilinear", OVERLAY_SCALE_FAST_BILINEAR},
                {"bilinear",      OVERLAY_SCALE_BILINEAR},
                {"bicubic",       OVERLAY_SCALE_BICUBIC},
                {"lanczos",       OVERLAY_SCALE_LANCZOS},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
                char buf[64];
                snprintf(buf, sizeof buf, "file=a.pam:scale_filter=%s",
                         cases[i].name);
                ASSERT_MESSAGE(cases[i].name, overlay_config_parse(buf, &c));
                ASSERT_EQUAL_MESSAGE(cases[i].name, (int)cases[i].expected,
                                     (int)c.scale_filter);
        }
        ASSERT_MESSAGE("unknown filter rejected",
                       !overlay_config_parse("file=a.pam:scale_filter=bogus", &c));
        return 0;
}

int overlay_config_test_blend_threads(void)
{
        struct overlay_config c;

        /* Unset -> auto-default of min(ncpu, 8). On any host that's >= 1. */
        ASSERT_MESSAGE("default auto",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("auto >= 1", c.blend_threads >= 1);
        ASSERT_MESSAGE("auto <= 8", c.blend_threads <= 8);

        ASSERT_MESSAGE("explicit 4",
                       overlay_config_parse("file=a.pam:blend_threads=4", &c));
        ASSERT_EQUAL_MESSAGE("4", 4, c.blend_threads);

        /* User who wants single-threaded passes 1 explicitly. */
        ASSERT_MESSAGE("explicit 1",
                       overlay_config_parse("file=a.pam:blend_threads=1", &c));
        ASSERT_EQUAL_MESSAGE("1", 1, c.blend_threads);

        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:blend_threads=-1", &c));
        return 0;
}

int overlay_config_test_rejects_oversize_options(void)
{
        char giant[MAX_PATH_SIZE + 512];
        memset(giant, 'a', sizeof giant - 1);
        giant[sizeof giant - 1] = '\0';
        memcpy(giant, "file=", 5);

        struct overlay_config c;
        ASSERT_MESSAGE("options too long: reject",
                       !overlay_config_parse(giant, &c));
        return 0;
}

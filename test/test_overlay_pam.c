/**
 * @file   test/test_overlay_pam.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_pam.c, registered via run_tests.c
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test_overlay_pam.h"
#include "unit_common.h"
#include "utils/fs.h"
#include "utils/overlay_pam.h"

/* Write a temp PAM file with the given header + data and return its path via
 * *path_out (caller will unlink). Returns true on success. */
static bool write_temp_pam(const char *header, const unsigned char *data,
                           size_t data_len, const char **path_out)
{
        FILE *f = get_temp_file(path_out);
        if (!f) return false;
        const size_t header_len = strlen(header);
        if (fwrite(header, 1, header_len, f) != header_len ||
            fwrite(data, 1, data_len, f) != data_len) {
                fclose(f);
                unlink(*path_out);
                return false;
        }
        if (fclose(f) != 0) {
                unlink(*path_out);
                return false;
        }
        return true;
}

/*
 * 8-bit RGBA PAM (maxval=255) loads with 8-bit values bit-replicated to 16-bit:
 * value 0xAB at 8-bit becomes 0xABAB at 16-bit.
 */
int overlay_pam_test_load_8bit_rgba(void)
{
        const char *header =
                "P7\n"
                "WIDTH 2\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 255\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        const unsigned char data[] = {
                0xFF, 0x00, 0x00, 0xFF,   /* opaque red   */
                0xAB, 0xCD, 0xEF, 0x80,   /* arbitrary, half-alpha */
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("width",  2, w);
        ASSERT_EQUAL_MESSAGE("height", 1, h);

        ASSERT_EQUAL_MESSAGE("p0 R", 0xFFFFu, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("p0 G", 0x0000u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("p0 B", 0x0000u, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("p0 A", 0xFFFFu, (unsigned)out[3]);
        ASSERT_EQUAL_MESSAGE("p1 R", 0xABABu, (unsigned)out[4]);
        ASSERT_EQUAL_MESSAGE("p1 G", 0xCDCDu, (unsigned)out[5]);
        ASSERT_EQUAL_MESSAGE("p1 B", 0xEFEFu, (unsigned)out[6]);
        ASSERT_EQUAL_MESSAGE("p1 A", 0x8080u, (unsigned)out[7]);

        free(out);
        return 0;
}

/* 3-channel RGB PAM (DEPTH 3, no alpha) gets alpha=65535 added. */
int overlay_pam_test_load_8bit_rgb_adds_alpha(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 3\n"
                "MAXVAL 255\n"
                "TUPLTYPE RGB\n"
                "ENDHDR\n";
        const unsigned char data[] = { 0x12, 0x34, 0x56 };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("R", 0x1212u, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("G", 0x3434u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("B", 0x5656u, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("A defaults to opaque", 0xFFFFu, (unsigned)out[3]);
        free(out);
        return 0;
}

/* 16-bit RGBA PAM (maxval=65535): samples are 2 bytes big-endian. Data is read
 * verbatim into the uint16_t buffer with byte-swap on little-endian hosts. */
int overlay_pam_test_load_16bit_rgba(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 65535\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        /* Big-endian 16-bit samples: R=0x1234, G=0x5678, B=0x9ABC, A=0xDEF0 */
        const unsigned char data[] = {
                0x12, 0x34,  0x56, 0x78,  0x9A, 0xBC,  0xDE, 0xF0,
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("R", 0x1234u, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("G", 0x5678u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("B", 0x9ABCu, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("A", 0xDEF0u, (unsigned)out[3]);
        free(out);
        return 0;
}

int overlay_pam_test_rejects_missing_file(void)
{
        uint16_t *out = (uint16_t *)0xDEADBEEF;
        int w = 999, h = 999;
        bool ok = overlay_load_pam_rgba16("/nonexistent/path/no.pam",
                                          &out, &w, &h);
        ASSERT_MESSAGE("load returned false", !ok);
        /* outputs untouched per contract */
        ASSERT_EQUAL_MESSAGE("out untouched", (uintptr_t)0xDEADBEEF, (uintptr_t)out);
        ASSERT_EQUAL_MESSAGE("w untouched",   999, w);
        ASSERT_EQUAL_MESSAGE("h untouched",   999, h);
        return 0;
}

int overlay_pam_test_rejects_grayscale(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 1\n"
                "MAXVAL 255\n"
                "TUPLTYPE GRAYSCALE\n"
                "ENDHDR\n";
        const unsigned char data[] = { 0x80 };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned false", !ok);
        return 0;
}

int overlay_pam_test_rejects_intermediate_maxval(void)
{
        /* 10-bit (maxval=1023) is rejected; values would otherwise be
         * silently treated as raw 16-bit, producing a darker overlay. */
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 1023\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        const unsigned char data[] = {
                0x03, 0xFF,  0x03, 0xFF,  0x03, 0xFF,  0x03, 0xFF,
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned false", !ok);
        return 0;
}

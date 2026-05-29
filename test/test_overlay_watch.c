/**
 * @file   test/test_overlay_watch.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_watch.c
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "test_overlay_watch.h"
#include "unit_common.h"
#include "utils/fs.h"
#include "utils/overlay_watch.h"

static bool write_file(const char **path_out, const char *body, size_t len)
{
        FILE *f = get_temp_file(path_out);
        if (!f) return false;
        bool ok = fwrite(body, 1, len, f) == len;
        if (fclose(f) != 0) ok = false;
        if (!ok) unlink(*path_out);
        return ok;
}

static bool overwrite_file(const char *path, const char *body, size_t len)
{
        FILE *f = fopen(path, "wb");
        if (!f) return false;
        bool ok = fwrite(body, 1, len, f) == len;
        if (fclose(f) != 0) ok = false;
        return ok;
}

/* Stable file: init then immediate poll reports no change. */
int overlay_watch_test_init_no_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("not changed", !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* File grows -> change detected (size differs even if mtime resolution is coarse). */
int overlay_watch_test_detects_size_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite bigger",
                       overwrite_file(path, "hello world", 11));
        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Same size, different mtime -> change detected. Force the baseline mtime
 * into the past via utimes() so we don't depend on timing resolution. */
int overlay_watch_test_detects_mtime_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

        struct timeval tv[2];
        tv[0].tv_sec = 1000000; tv[0].tv_usec = 0;  /* atime */
        tv[1].tv_sec = 1000000; tv[1].tv_usec = 0;  /* mtime: well in the past */
        ASSERT_EQUAL_MESSAGE("utimes", 0, utimes(path, tv));

        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Acking against a missing file is a no-op: baseline survives so the next
 * appearance still triggers reload. Avoids dropping a real edit if the file
 * is briefly absent (e.g. atomic-write rename) at ack time. */
int overlay_watch_test_ack_on_missing_file_preserves_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

        ASSERT_EQUAL_MESSAGE("unlink", 0, unlink(path));
        overlay_watch_ack(&w, path);

        ASSERT_MESSAGE("recreate",
                       overwrite_file(path, "different content", 17));
        ASSERT_MESSAGE("appearance still triggers",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/*
 * The baseline must NOT advance until the caller explicitly acknowledges
 * the change (after a successful reload). Otherwise a failed reload would
 * eat the trigger and a later valid mutation would go undetected.
 */
int overlay_watch_test_changed_does_not_consume_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite #1",
                       overwrite_file(path, "broken!!!", 9));

        /* First poll: change detected. Caller's reload (simulated) fails,
         * so it does NOT call overlay_watch_ack. */
        ASSERT_MESSAGE("first poll changed",
                       overlay_watch_changed(&w, path));

        /* Second poll without ack: still reports change so the caller can
         * retry, even though mtime/size haven't moved further. */
        ASSERT_MESSAGE("second poll still changed (not acked)",
                       overlay_watch_changed(&w, path));

        /* And after a fresh mutation, change is still reported. */
        ASSERT_MESSAGE("rewrite #2",
                       overwrite_file(path, "fixed content version 2", 23));
        ASSERT_MESSAGE("after mutation: changed",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

int overlay_watch_test_ack_commits_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite",
                       overwrite_file(path, "version 2 here", 14));
        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        overlay_watch_ack(&w, path);
        ASSERT_MESSAGE("after ack: quiet",
                       !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Missing file (transiently absent during atomic-write) must not trigger a
 * spurious reload — caller would crash trying to load nothing. */
int overlay_watch_test_missing_file_no_change(void)
{
        struct overlay_watch w;
        overlay_watch_init(&w, "/nonexistent/path/no.pam");
        ASSERT_MESSAGE("missing file: no change",
                       !overlay_watch_changed(&w, "/nonexistent/path/no.pam"));
        return 0;
}

/* File initially missing, then created -> first poll after creation reports
 * change (so the caller's reload kicks in once the file appears). */
int overlay_watch_test_file_appears_after_init(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write+remove", write_file(&path, "x", 1));
        ASSERT_EQUAL_MESSAGE("unlink", 0, unlink(path));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("missing while absent",
                       !overlay_watch_changed(&w, path));

        ASSERT_MESSAGE("recreate", overwrite_file(path, "appeared", 8));
        ASSERT_MESSAGE("change reported on appearance",
                       overlay_watch_changed(&w, path));
        overlay_watch_ack(&w, path);
        ASSERT_MESSAGE("then quiet",
                       !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Atomic-write replacement: write a sibling temp file then rename() over the
 * watched path. The new inode has a different mtime/size and the watcher
 * must detect it. */
int overlay_watch_test_detects_atomic_rename(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "v1", 2));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

        const char *staging = NULL;
        ASSERT_MESSAGE("write staging", write_file(&staging, "version2", 8));
        ASSERT_EQUAL_MESSAGE("rename", 0, rename(staging, path));

        ASSERT_MESSAGE("change detected after rename",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/**
 * @file   utils/overlay_watch.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  mtime/size-based file change detection for overlay hot-reload.
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

#include <sys/stat.h>

#include "utils/overlay_watch.h"

/* Snapshot the file's mtime+size, returning false if the file is absent
 * (transient — e.g. mid-atomic-rename). Centralises the platform-specific
 * timespec spelling: detect by OS rather than _POSIX_C_SOURCE so a TU that
 * hasn't opted into POSIX 2008 still gets nanosecond precision on Linux. */
bool overlay_watch_fingerprint(const char *path,
                               int64_t *mtime_ns, int64_t *size)
{
        struct stat sb;
        if (stat(path, &sb) != 0) return false;
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) \
    || defined(__OpenBSD__) || defined(__DragonFly__)
        *mtime_ns = (int64_t)sb.st_mtimespec.tv_sec * 1000000000
                  + (int64_t)sb.st_mtimespec.tv_nsec;
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__sun) \
    || defined(_AIX) || defined(__GNU__)
        *mtime_ns = (int64_t)sb.st_mtim.tv_sec * 1000000000
                  + (int64_t)sb.st_mtim.tv_nsec;
#else
        *mtime_ns = (int64_t)sb.st_mtime * 1000000000;
#endif
        *size = sb.st_size;
        return true;
}

void overlay_watch_init(struct overlay_watch *w, const char *path)
{
        w->mtime_ns = 0;
        w->size = 0;
        w->valid = overlay_watch_fingerprint(path, &w->mtime_ns, &w->size);
}

bool overlay_watch_changed(const struct overlay_watch *w, const char *path)
{
        int64_t mtime_ns = 0, size = 0;
        if (!overlay_watch_fingerprint(path, &mtime_ns, &size)) return false;
        if (!w->valid) return true;  /* file was missing at init */
        return mtime_ns != w->mtime_ns || size != w->size;
}

void overlay_watch_ack(struct overlay_watch *w, const char *path)
{
        /* Best-effort: on transient stat failure leave the baseline alone
         * so a later appearance still triggers reload. */
        int64_t mtime_ns = 0, size = 0;
        if (!overlay_watch_fingerprint(path, &mtime_ns, &size)) return;
        w->mtime_ns = mtime_ns;
        w->size     = size;
        w->valid    = true;
}

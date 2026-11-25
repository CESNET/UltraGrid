/**
 * @file   utils/overlay_watch.h
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

#ifndef UTILS_OVERLAY_WATCH_H_7F3D9A2B_1E5C_4A8F_B6D3_2C9E5A1F8B4D
#define UTILS_OVERLAY_WATCH_H_7F3D9A2B_1E5C_4A8F_B6D3_2C9E5A1F8B4D

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tracks the last-seen mtime and size of a file. Designed to be polled cheaply
 * once per frame from the overlay postprocessor's hot path: a single stat()
 * call detects on-disk edits so the overlay can be reloaded without restart.
 *
 * Treated as opaque by callers; fields are exposed only so it can be embedded
 * by value in callers' state structs (no heap allocation needed).
 */
struct overlay_watch {
        int64_t mtime_ns;
        int64_t size;
        bool    valid;
};

/*
 * Capture the current mtime+size as the baseline. If the file can't be stat'd,
 * the watch is left invalid and overlay_watch_changed() will return false
 * until the file appears (in which case the next call detects it as a change).
 */
void overlay_watch_init(struct overlay_watch *w, const char *path);

/*
 * True if the on-disk mtime or size differs from the baseline. Does not
 * mutate the watch — caller must call overlay_watch_ack() after successfully
 * consuming the change (e.g. reloaded the file). If the reload fails, do
 * not ack: the next poll keeps reporting the change so the caller retries.
 *
 * Returns false when the file is missing — a transient stat() failure
 * during atomic-write doesn't trigger a spurious reload.
 */
bool overlay_watch_changed(const struct overlay_watch *w, const char *path);

/* Commit the current on-disk mtime+size as the new baseline. No-op if the
 * file is currently missing (baseline preserved). */
void overlay_watch_ack(struct overlay_watch *w, const char *path);

/* Read the on-disk fingerprint (mtime_ns, size) for a path. Returns true
 * on success; on failure (any stat() error — caller cannot distinguish
 * missing-file from transient I/O glitch) leaves *mtime_ns and *size
 * untouched. Centralises the platform-specific timespec spelling so
 * callers don't re-implement the same #ifdef ladder. */
bool overlay_watch_fingerprint(const char *path,
                               int64_t *mtime_ns, int64_t *size);

#ifdef __cplusplus
}
#endif

#endif

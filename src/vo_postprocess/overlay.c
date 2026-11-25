/**
 * @file   vo_postprocess/overlay.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Image overlay postprocessor (native-format alpha blending).
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

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/alpha_blend.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/overlay_config.h"
#include "utils/overlay_layout.h"
#include "utils/overlay_pam.h"
#include "utils/overlay_scale.h"
#include "utils/overlay_soft_edge.h"
#include "utils/overlay_watch.h"
#include "utils/worker.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
#include "vo_postprocess.h"

#define MOD_NAME "[overlay] "

/* Below this many output pixels the per-worker wakeup cost exceeds the
 * saving. ~500k chosen so a 1080p (2.07 Mpx) blend keeps fanning out,
 * while a 720p (0.92 Mpx) blend at a small overlay rect falls through
 * to the single-threaded path. Shared by both single-plane (blend_overlay)
 * and planar (blend_i420) dispatch — the cost model is the same. */
enum { MIN_PARALLEL_PIXELS = 500000 };

/* Effective worker count for a blend rect. Returns 1 (caller takes the
 * serial path) when the rect is too small to amortise dispatch, or when
 * the user asked for serial. Otherwise clamps requested to MAX_CPU_CORES
 * so the per-thread VLAs stay bounded. */
static inline int
parallel_workers(int requested, size_t pixels)
{
        if (pixels < MIN_PARALLEL_PIXELS) return 1;
        if (requested <= 1) return 1;
        return requested > MAX_CPU_CORES ? MAX_CPU_CORES : requested;
}

typedef void (*blend_fn)(uint8_t *dst, const uint16_t *rgba16, int width);

struct state_overlay {
        struct overlay_config cfg;
        struct overlay_watch  watch;
        struct overlay_scaler *scaler;  ///< caches SwsContext across reloads

        uint16_t *overlay_rgba16;    ///< active overlay (blend reads this)
        int       overlay_w, overlay_h;
        size_t    overlay_capacity;  ///< pixels allocated for the active

        /* Async reload pipeline. Watcher fires -> kick a worker that
         * fills `staging` -> next frame harvests, swap into active.
         * Both buffers grow monotonically; a steady stream of same-
         * sized reloads pays no per-reload malloc/free. */
        uint16_t *staging_rgba16;
        size_t    staging_capacity;

        task_result_handle_t reload_handle;
        struct reload_job   *reload_job;
        /* (mtime, size) of the bytes the most recent failed reload
         * actually loaded — captured by the worker before parsing,
         * so the user "fixing" the file mid-load doesn't fool us into
         * skipping a real retry. Suppresses re-kicking against the
         * same broken file at frame rate; cleared on next success. */
        bool    have_failed_fp;
        int64_t failed_mtime_ns;
        int64_t failed_size;

        struct video_desc   saved_desc;
        struct video_frame *in;

        /* Last codec we logged "unsupported by overlay" for. Reset to
         * VIDEO_CODEC_NONE on construction so the first unsupported codec
         * always warns; if the stream switches to a different unsupported
         * codec mid-flight, that one warns too. */
        codec_t warned_codec;

        /* Cumulative timing; see overlay_perf_report(). */
        struct {
                unsigned long frames;
                unsigned long reloads;
                time_ns_t     blend_ns;
                time_ns_t     total_ns;
                time_ns_t     last_report_ns;
        } perf;
};

#define PERF_REPORT_INTERVAL_NS ((time_ns_t)10 * NS_IN_SEC)

static void
overlay_perf_report(struct state_overlay *s, time_ns_t now_ns)
{
        const unsigned long frames = s->perf.frames;
        const double avg_total_ms = frames > 0
                ? (double)s->perf.total_ns / (double)frames / 1.0e6 : 0.0;
        const double avg_blend_ms = frames > 0
                ? (double)s->perf.blend_ns / (double)frames / 1.0e6 : 0.0;
        log_msg(LOG_LEVEL_INFO,
                MOD_NAME TUNDERLINE("Overlay stats (cumulative)")
                " - Frames: " TBOLD("%lu")
                " / Reloads: " TBOLD("%lu")
                " / Avg blend: " TBOLD("%.3f") " ms"
                " / Avg total: " TBOLD("%.3f") " ms\n",
                frames, s->perf.reloads, avg_blend_ms, avg_total_ms);
        s->perf.last_report_ns = now_ns;
}

static inline time_ns_t
perf_t0(const struct state_overlay *s)
{
        return s->cfg.perf ? get_time_in_ns() : 0;
}

/* Accumulate one frame's timings and emit a periodic report if due.
 * frame_t0 is the frame-start timestamp from perf_t0(); blend_ns is the
 * just-measured blend duration. */
static void
perf_tally(struct state_overlay *s, time_ns_t frame_t0, time_ns_t blend_ns)
{
        if (!s->cfg.perf) return;
        const time_ns_t now = get_time_in_ns();
        s->perf.frames++;
        s->perf.blend_ns += blend_ns;
        s->perf.total_ns += now - frame_t0;
        if (now - s->perf.last_report_ns >= PERF_REPORT_INTERVAL_NS) {
                overlay_perf_report(s, now);
        }
}

/* Self-contained job for the async-reload worker. The worker writes its
 * output into the caller-provided staging buffer (no per-reload malloc
 * once dimensions stabilise). */
struct reload_job {
        char  file[MAX_PATH_SIZE];
        int   scale_w, scale_h;
        int   soft_edge;
        struct overlay_scaler *scaler;
        uint16_t *staging;             /* caller-owned, sized for staging_pixels */
        size_t   staging_pixels;       /* worker bails if load needs more */
        int   result_w, result_h;
        bool  success;
        /* Captured by the worker at file-open time so harvest can stamp
         * the *exact* fingerprint that failed/succeeded — closes a
         * TOCTOU where the user could fix the file between worker start
         * and harvest, fooling the failed-fingerprint guard. */
        int64_t loaded_mtime_ns;
        int64_t loaded_size;
        bool    fingerprint_valid;
};

static void *
reload_worker(void *arg)
{
        struct reload_job *j = arg;
        /* Capture (mtime, size) BEFORE loading so the fingerprint matches
         * the bytes the worker actually saw. */
        j->fingerprint_valid =
                overlay_watch_fingerprint(j->file, &j->loaded_mtime_ns,
                                          &j->loaded_size);
        uint16_t *raw = NULL;
        int       rw = 0, rh = 0;
        if (!overlay_load_pam_rgba16(j->file, &raw, &rw, &rh)) return NULL;

        const bool scaling = j->scale_w > 0 && j->scale_h > 0;
        const int  out_w   = scaling ? j->scale_w : rw;
        const int  out_h   = scaling ? j->scale_h : rh;

        /* Refuse to write past the caller-provided buffer. Triggers the
         * realloc-on-next-kick path so the second attempt fits. */
        if ((size_t)out_w * out_h > j->staging_pixels) {
                free(raw);
                return NULL;
        }

        if (scaling) {
                if (!overlay_scaler_scale_into(j->scaler, j->staging, raw,
                                               rw, rh, out_w, out_h)) {
                        free(raw); return NULL;
                }
        } else {
                memcpy(j->staging, raw,
                       (size_t)out_w * out_h * 4 * sizeof *raw);
        }
        free(raw);

        overlay_apply_soft_edge(j->staging, out_w, out_h, j->soft_edge);

        j->result_w = out_w;
        j->result_h = out_h;
        j->success  = true;
        return NULL;
}

/* Drain an in-flight reload. On success: swap the staging buffer into
 * the active slot and remember its dims. On failure: leave the active
 * slot alone, record the failure fingerprint so we don't immediately
 * re-kick against the same broken file. */
static void
harvest_reload(struct state_overlay *s)
{
        struct reload_job *j = s->reload_job;
        wait_task(s->reload_handle);
        s->reload_handle = NULL;
        s->reload_job    = NULL;

        if (j->success) {
                /* Swap active <-> staging. The old active buffer (still
                 * allocated) becomes the staging slot for the next reload,
                 * so a steady stream of same-sized reloads pays no malloc. */
                SWAP_PTR(s->overlay_rgba16, s->staging_rgba16);
                SWAP(s->overlay_capacity, s->staging_capacity);
                s->overlay_w = j->result_w;
                s->overlay_h = j->result_h;

                s->perf.reloads++;
                s->have_failed_fp = false;
                overlay_watch_ack(&s->watch, s->cfg.file);
                log_msg(LOG_LEVEL_VERBOSE,
                        MOD_NAME "loaded %s (%dx%d)\n",
                        s->cfg.file, j->result_w, j->result_h);
        } else {
                /* Use the worker's pre-load fingerprint so we stamp
                 * exactly the bytes that failed; if a fix raced into
                 * place between worker start and harvest, the next
                 * frame's matches_last_failure will compare against
                 * the now-fixed fingerprint and correctly let a retry
                 * fire. */
                if (j->fingerprint_valid) {
                        s->failed_mtime_ns = j->loaded_mtime_ns;
                        s->failed_size     = j->loaded_size;
                        s->have_failed_fp  = true;
                }
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "async reload of %s failed\n", s->cfg.file);
        }
        free(j);
}

static bool
matches_last_failure(const struct state_overlay *s)
{
        if (!s->have_failed_fp) return false;
        int64_t mtime = 0, size = 0;
        if (!overlay_watch_fingerprint(s->cfg.file, &mtime, &size)) return false;
        return mtime == s->failed_mtime_ns && size == s->failed_size;
}

/* Ensure *buf is at least want_pixels*4*sizeof(uint16_t) bytes; updates
 * *capacity to match on success. Grows monotonically — never shrinks,
 * which avoids realloc churn when a stream of similarly-sized PAMs lands.
 * Returns false on allocation failure (leaving *buf and *capacity intact). */
static bool
ensure_pixel_capacity(uint16_t **buf, size_t *capacity, size_t want_pixels)
{
        if (want_pixels <= *capacity) return true;
        uint16_t *grown = realloc(*buf, want_pixels * 4 * sizeof *grown);
        if (grown == NULL) return false;
        *buf      = grown;
        *capacity = want_pixels;
        return true;
}

/* Kick an async reload with explicit scale dimensions. Pass 0/0 for
 * "no scaling" (load at native PAM size). Used directly by the
 * scale=frame reconfigure path; the legacy file-watcher kick is the
 * thin wrapper below. */
static void
kick_async_reload_with_dims(struct state_overlay *s, int scale_w, int scale_h)
{
        const int out_w = scale_w > 0 ? scale_w : s->overlay_w;
        const int out_h = scale_h > 0 ? scale_h : s->overlay_h;
        /* If we don't yet know the source dims (initial load was scaled,
         * subsequent reloads use scale=). For the unscaled-source path
         * the worker discovers the size — leave staging un-presized and
         * the worker can grow it on first use via realloc.
         *
         * In practice the unscaled path is rare: users with a fixed-
         * dimension stream typically pre-scale at config time. The
         * realloc cost shows up only on first reload where source dim
         * differs from active. */
        if (out_w > 0 && out_h > 0) {
                const size_t want = (size_t)out_w * out_h;
                if (!ensure_pixel_capacity(&s->staging_rgba16,
                                           &s->staging_capacity, want)) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "out of memory growing staging "
                                "buffer to %dx%d; reload skipped, will "
                                "retry on next change\n", out_w, out_h);
                        return;
                }
        }

        struct reload_job *j = calloc(1, sizeof *j);
        if (j == NULL) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "out of memory allocating reload job; "
                        "reload skipped, will retry on next change\n");
                return;
        }
        snprintf(j->file, sizeof j->file, "%s", s->cfg.file);
        j->scale_w   = scale_w;
        j->scale_h   = scale_h;
        j->soft_edge = s->cfg.soft_edge;
        j->scaler         = s->scaler;
        j->staging        = s->staging_rgba16;
        j->staging_pixels = s->staging_capacity;

        s->reload_job    = j;
        s->reload_handle = task_run_async(reload_worker, j);
        if (s->reload_handle == NULL) {
                /* Pool refused (shouldn't happen with the current pool
                 * impl, but keep the cleanup so a future implementation
                 * change doesn't leak the job). */
                free(j);
                s->reload_job = NULL;
        }
}

/* File-watcher-driven kick. With scale=frame we use the current frame
 * dimensions (not cfg.scale_w/h, which are 0); with scale=WxH we use
 * the configured dims; otherwise no scaling. */
static void
kick_async_reload(struct state_overlay *s)
{
        int sw = s->cfg.scale_w;
        int sh = s->cfg.scale_h;
        if (s->cfg.scale_to_frame) {
                sw = (int)s->saved_desc.width;
                sh = (int)s->saved_desc.height;
        }
        kick_async_reload_with_dims(s, sw, sh);
}

/* Load (or reload) the overlay PAM. On success the persistent
 * s->overlay_rgba16 buffer holds the (possibly scaled, possibly
 * soft-edged) result. On failure leaves the existing buffer untouched. */
static bool reload_overlay(struct state_overlay *s)
{
        uint16_t *raw = NULL;
        int       raw_w = 0, raw_h = 0;
        if (!overlay_load_pam_rgba16(s->cfg.file, &raw, &raw_w, &raw_h)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "failed to load %s\n",
                        s->cfg.file);
                return false;
        }

        const bool scaling = s->cfg.scale_w > 0 && s->cfg.scale_h > 0;
        const int  out_w   = scaling ? s->cfg.scale_w : raw_w;
        const int  out_h   = scaling ? s->cfg.scale_h : raw_h;

        if (!ensure_pixel_capacity(&s->overlay_rgba16, &s->overlay_capacity,
                                   (size_t)out_w * out_h)) {
                free(raw);
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "out of memory for %dx%d overlay\n",
                        out_w, out_h);
                return false;
        }

        /* Scale before soft-edge fade so the fade width is in final-buffer
         * pixels, not source pixels. */
        if (scaling) {
                if (!overlay_scaler_scale_into(s->scaler, s->overlay_rgba16,
                                               raw, raw_w, raw_h,
                                               out_w, out_h)) {
                        free(raw);
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "failed to scale %s to %dx%d\n",
                                s->cfg.file, out_w, out_h);
                        return false;
                }
                free(raw);
        } else {
                memcpy(s->overlay_rgba16, raw,
                       (size_t)out_w * out_h * 4 * sizeof *raw);
                free(raw);
        }
        overlay_apply_soft_edge(s->overlay_rgba16, out_w, out_h,
                                s->cfg.soft_edge);
        s->overlay_w = out_w;
        s->overlay_h = out_h;
        /* First load is loud (NOTICE); subsequent hot-reloads are quiet
         * (VERBOSE) so a fast-mtime sequence doesn't flood the terminal.
         * Reload count is still surfaced via the perf stats line. */
        const int level = s->perf.reloads == 0
                ? LOG_LEVEL_NOTICE : LOG_LEVEL_VERBOSE;
        s->perf.reloads++;
        if (s->cfg.soft_edge > 0) {
                log_msg(level,
                        MOD_NAME "loaded %s (%dx%d, soft_edge=%d)\n",
                        s->cfg.file, out_w, out_h, s->cfg.soft_edge);
        } else {
                log_msg(level,
                        MOD_NAME "loaded %s (%dx%d)\n",
                        s->cfg.file, out_w, out_h);
        }
        return true;
}

static void
print_help(void)
{
        printf("overlay video postprocessor — blends a 16-bit RGBA PAM "
               "image onto video in its native pixel format.\n\n"
               "Usage:\n"
               "\t-p overlay:file=<path>[:position=<pos>]"
               "[:custom_x=<x>:custom_y=<y>][:soft_edge=<n>]"
               "[:scale=<W>x<H>[:scale_filter=<f>]][:perf]\n\n"
               "Position keywords: center (default), top_left, top_right,\n"
               "                   bottom_left, bottom_right, custom\n\n"
               "custom_x / custom_y count from the right/bottom edge when "
               "negative.\n"
               "soft_edge=<n> applies an N-pixel linear alpha fade on each "
               "edge (default 0).\n"
               "scale=<W>x<H> resizes the loaded overlay to W x H pixels.\n"
               "scale=frame  re-scales the overlay to match the current\n"
               "             frame dimensions on every resolution change\n"
               "             (decode-path renegotiation).\n"
               "scale_filter=<f> picks the resampler: nearest, fast_bilinear,\n"
               "                 bilinear, bicubic, lanczos (default bicubic).\n"
               "                 Use lanczos for sharper output if budget allows;\n"
               "                 nearest/fast_bilinear if running tight at 4K60.\n"
               "blend_threads=<N> parallelises the per-row alpha blend across\n"
               "                 N pthread-pool workers. Default is\n"
               "                 min(ncpu, 8); pass 1 to disable.\n"
               "perf logs cumulative blend / total timing every 10 seconds "
               "(off by default).\n");
}

static void *
overlay_init(const char *config)
{
        struct state_overlay *s = calloc(1, sizeof *s);
        if (s == NULL) return NULL;

        if (!overlay_config_parse(config, &s->cfg)) {
                free(s);
                return NULL;
        }
        if (s->cfg.help) {
                print_help();
                free(s);
                return NULL;
        }
        s->scaler = overlay_scaler_create(s->cfg.scale_filter);
        if (s->scaler == NULL) {
                free(s);
                return NULL;
        }
        if (!reload_overlay(s)) {
                overlay_scaler_destroy(s->scaler);
                free(s);
                return NULL;
        }
        overlay_watch_init(&s->watch, s->cfg.file);
        s->perf.last_report_ns = get_time_in_ns();
        return s;
}

static bool
overlay_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_overlay *s = state;
        s->saved_desc = desc;
        vf_free(s->in);
        s->in = vf_alloc_desc_data(s->saved_desc);
        return s->in != NULL;
}

static struct video_frame *
overlay_getf(void *state)
{
        struct state_overlay *s = state;
        return s->in;
}

/* Single-plane blend functions only — I420 has its own dispatch via
 * blend_i420() because the planar layout takes a different signature. */
static blend_fn
get_blend_fn(codec_t codec)
{
        switch (codec) {
        case RGBA: return alpha_blend_rgba;
        case RGB:  return alpha_blend_rgb;
        case RG48: return alpha_blend_rg48;
        case UYVY: return alpha_blend_uyvy;
        case YUYV: return alpha_blend_yuyv;
        case v210: return alpha_blend_v210;
        case R10k: return alpha_blend_r10k;
        case R12L: return alpha_blend_r12l;
        case Y416: return alpha_blend_y416;
        default:   return NULL;
        }
}

/* Per-stripe job for the parallel I420 dispatch. Each worker calls
 * alpha_blend_i420 with already-offset plane pointers and a slice height,
 * so workers don't coordinate (chroma row pairs split cleanly at even
 * boundaries — the rect snapper guarantees this). */
struct i420_stripe {
        uint8_t        *dst_y;
        uint8_t        *dst_u;
        uint8_t        *dst_v;
        int             y_stride;
        int             uv_stride;
        const uint16_t *src;
        int             src_pixel_stride;
        int             width;
        int             height;
};

static void *
i420_stripe_worker(void *arg)
{
        const struct i420_stripe *j = arg;
        alpha_blend_i420(j->dst_y, j->y_stride,
                         j->dst_u, j->dst_v, j->uv_stride,
                         j->src, j->src_pixel_stride,
                         j->width, j->height);
        return NULL;
}

/* Pointer into the overlay buffer at the visible region's origin.
 * src_x/src_y are 0 in the common in-bounds case (so the result is
 * just `base`); they're non-zero when the overlay is larger than the
 * frame and we're showing its centre or right slice. The "* 4" is
 * the four uint16_t RGBA components per overlay pixel. */
static inline const uint16_t *
overlay_src_at(const uint16_t *base, int overlay_w,
               int src_x, int src_y)
{
        return base + (size_t)src_y * overlay_w * 4
                    + (size_t)src_x * 4;
}

/* I420 needs a planar dispatch — three plane pointers and a different
 * blend signature than the single-plane formats. rect.x/rect.y/rect.width/
 * rect.height are already snapped to even by overlay_calc_rect (block_lines
 * and block_pixels both = 2 for I420). blend_threads matches the other
 * codecs' fan-out; chroma rows split at even-luma boundaries so a 2x2
 * cell is never spanned by two workers. */
static void
blend_i420(struct video_frame *out, const struct overlay_rect *rect,
           const uint16_t *overlay_rgba16, int overlay_w, int blend_threads)
{
        assert((rect->x & 1) == 0 && (rect->y & 1) == 0);
        assert((rect->width & 1) == 0 && (rect->height & 1) == 0);

        const int frame_w = (int)out->tiles[0].width;
        const int frame_h = (int)out->tiles[0].height;
        const int chroma_w = (frame_w + 1) / 2;
        const int chroma_h_plane = (frame_h + 1) / 2;

        uint8_t *plane_y = (uint8_t *)out->tiles[0].data;
        uint8_t *plane_u = plane_y + (size_t)frame_w * frame_h;
        uint8_t *plane_v = plane_u + (size_t)chroma_w * chroma_h_plane;

        uint8_t *dst_y = plane_y + (size_t)rect->y * frame_w + rect->x;
        uint8_t *dst_u = plane_u + (size_t)(rect->y / 2) * chroma_w + (rect->x / 2);
        uint8_t *dst_v = plane_v + (size_t)(rect->y / 2) * chroma_w + (rect->x / 2);

        const uint16_t *src = overlay_src_at(overlay_rgba16, overlay_w,
                                             rect->src_x, rect->src_y);

        const int N = parallel_workers(blend_threads,
                                       (size_t)rect->width * rect->height);
        if (N == 1) {
                alpha_blend_i420(dst_y, frame_w, dst_u, dst_v, chroma_w,
                                 src, overlay_w,
                                 rect->width, rect->height);
                return;
        }

        /* Split the chroma rows evenly; last worker absorbs the remainder.
         * Each stripe owns 2*rows_per luma rows + rows_per chroma rows. */
        const int total_chroma_rows = rect->height / 2;
        const int rows_per = total_chroma_rows / N;
        struct i420_stripe jobs[N];
        for (int i = 0; i < N; i++) {
                const int cy_start = i * rows_per;
                const int cy_end   = (i == N - 1) ? total_chroma_rows
                                                  : (i + 1) * rows_per;
                const int luma_off = cy_start * 2;
                jobs[i] = (struct i420_stripe){
                        .dst_y            = dst_y + (size_t)luma_off * frame_w,
                        .dst_u            = dst_u + (size_t)cy_start * chroma_w,
                        .dst_v            = dst_v + (size_t)cy_start * chroma_w,
                        .y_stride         = frame_w,
                        .uv_stride        = chroma_w,
                        .src              = overlay_src_at(src, overlay_w,
                                                           0, luma_off),
                        .src_pixel_stride = overlay_w,
                        .width            = rect->width,
                        .height           = (cy_end - cy_start) * 2,
                };
        }
        task_run_parallel(i420_stripe_worker, N, jobs, sizeof jobs[0], NULL);
}

/* Below this many bytes the per-worker dispatch cost outweighs the
 * memory-bandwidth saving from threading the memcpy. Picked so a
 * 1080p UYVY frame (~4 MB) still gets parallelised; smaller frames
 * fall through to a single memcpy. */
#define MIN_PARALLEL_MEMCPY_BYTES (1u << 21)   /* 2 MB */

struct memcpy_stripe {
        uint8_t       *dst;
        const uint8_t *src;
        size_t         bytes;
};

static void *
memcpy_stripe_worker(void *arg)
{
        const struct memcpy_stripe *j = arg;
        memcpy(j->dst, j->src, j->bytes);
        return NULL;
}

static void
threaded_memcpy(uint8_t *dst, const uint8_t *src, size_t bytes, int N)
{
        if (N <= 1 || bytes < MIN_PARALLEL_MEMCPY_BYTES) {
                memcpy(dst, src, bytes);
                return;
        }
        /* VLA sized to actual N; the parser clamps N to MAX_CPU_CORES
         * (256) so the stack worst case is bounded. */
        struct memcpy_stripe jobs[N];
        const size_t chunk = bytes / N;
        for (int i = 0; i < N; i++) {
                jobs[i].dst   = dst   + (size_t)i * chunk;
                jobs[i].src   = src   + (size_t)i * chunk;
                jobs[i].bytes = (i == N - 1) ? bytes - (size_t)i * chunk
                                             : chunk;
        }
        task_run_parallel(memcpy_stripe_worker, N, jobs, sizeof jobs[0], NULL);
}

/* Each thread chews through a contiguous row stripe of the rect. The
 * blend functions are pure per-row (output = f(dst_row, src_row, width)),
 * so independent workers cannot collide. */
struct blend_stripe {
        blend_fn        fn;
        uint8_t        *dst_base;        /* out->tiles[0].data + rect.y*pitch */
        const uint16_t *src_base;        /* overlay_rgba16 row 0 */
        size_t          dst_row_stride;  /* req_pitch (bytes) */
        size_t          src_row_stride;  /* overlay_w * 4 (uint16_t elements) */
        int             dst_x_byte_off;  /* (rect.x / block_pixels) * block_bytes */
        int             rect_width;      /* pixels per row */
        int             row_start;
        int             row_end;
};

static void *
blend_stripe_worker(void *arg)
{
        const struct blend_stripe *j = arg;
        for (int r = j->row_start; r < j->row_end; r++) {
                uint8_t *dst_row = j->dst_base
                                 + (size_t)r * j->dst_row_stride
                                 + j->dst_x_byte_off;
                const uint16_t *src_row =
                        j->src_base + (size_t)r * j->src_row_stride;
                j->fn(dst_row, src_row, j->rect_width);
        }
        return NULL;
}

/* Run the overlay blend onto `out` (caller has already pass-through-copied
 * `in`). No-op if there is no overlay loaded or the rect is empty. Logs
 * once when a codec without dispatch arrives. */
static void
blend_overlay(struct state_overlay *s, codec_t cs,
              struct video_frame *out, int req_pitch)
{
        if (s->overlay_rgba16 == NULL) return;

        const int block_pixels = get_pf_block_pixels(cs);
        const int block_lines  = (cs == I420) ? 2 : 1;
        const struct overlay_rect rect = overlay_calc_rect(
                s->cfg.position, s->cfg.custom_x, s->cfg.custom_y,
                (int)out->tiles[0].width, (int)out->tiles[0].height,
                s->overlay_w, s->overlay_h, block_pixels, block_lines);
        if (rect.width <= 0 || rect.height <= 0) return;

        if (cs == I420) {
                blend_i420(out, &rect, s->overlay_rgba16, s->overlay_w,
                           s->cfg.blend_threads);
                return;
        }

        const blend_fn blend = get_blend_fn(cs);
        if (blend == NULL) {
                if (s->warned_codec != cs) {
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "format %s unsupported by overlay\n",
                                get_codec_name(cs));
                        s->warned_codec = cs;
                }
                return;
        }

        const int block_bytes = get_pf_block_bytes(cs);
        /* src_row_stride still walks the full overlay row even when
         * src_x/src_y are non-zero (oversized overlay slice). */
        const uint16_t *src_base = overlay_src_at(s->overlay_rgba16,
                                                  s->overlay_w,
                                                  rect.src_x, rect.src_y);
        struct blend_stripe base = {
                .fn             = blend,
                .dst_base       = (uint8_t *)out->tiles[0].data
                                + (size_t)rect.y * req_pitch,
                .src_base       = src_base,
                .dst_row_stride = req_pitch,
                .src_row_stride = (size_t)s->overlay_w * 4,
                .dst_x_byte_off = (rect.x / block_pixels) * block_bytes,
                .rect_width     = rect.width,
        };

        const int N = parallel_workers(s->cfg.blend_threads,
                                       (size_t)rect.width * rect.height);
        if (N == 1) {
                base.row_start = 0;
                base.row_end   = rect.height;
                blend_stripe_worker(&base);
                return;
        }

        /* task_run_parallel reuses persistent worker-pool threads, so each
         * frame pays only condvar wakeup cost rather than pthread_create.
         * Each worker gets a contiguous stripe; the last absorbs the
         * remainder so total rows == rect.height regardless of N.
         *
         * VLA sized to actual N; the parser clamps N to MAX_CPU_CORES
         * (256) so the stack worst case is bounded. */
        struct blend_stripe jobs[N];
        const int rows_per = rect.height / N;
        for (int i = 0; i < N; i++) {
                jobs[i] = base;
                jobs[i].row_start = i * rows_per;
                jobs[i].row_end   = (i == N - 1) ? rect.height
                                                 : (i + 1) * rows_per;
        }
        task_run_parallel(blend_stripe_worker, N, jobs, sizeof jobs[0], NULL);
}

static bool
overlay_postprocess(void *state, struct video_frame *in,
                    struct video_frame *out, int req_pitch)
{
        struct state_overlay *s = state;
        assert(in->tile_count == 1);
        assert(req_pitch == vc_get_linesize(in->tiles[0].width, in->color_spec));

        const time_ns_t frame_t0 = perf_t0(s);

        threaded_memcpy((uint8_t *)out->tiles[0].data,
                        (const uint8_t *)in->tiles[0].data,
                        in->tiles[0].data_len,
                        s->cfg.blend_threads);

        /* Harvest any completed async reload before checking the watcher,
         * so a fresh change-detect can fire on the same frame the previous
         * reload finishes. */
        if (s->reload_handle != NULL && task_is_done(s->reload_handle)) {
                harvest_reload(s);
        }

        /* scale=frame: re-render the overlay if the source resolution
         * differs from the active overlay buffer. Detected here (per
         * frame, cheap) rather than via the reconfigure callback —
         * not every pipeline path reaches the postprocess via
         * display_reconfigure (testcard→dummy in the e2e harness, for
         * one). The kick is gated on no-reload-in-flight, same as the
         * watcher path. */
        if (s->cfg.scale_to_frame
            && s->reload_handle == NULL
            && ((int)in->tiles[0].width  != s->overlay_w
             || (int)in->tiles[0].height != s->overlay_h)) {
                kick_async_reload_with_dims(s,
                        (int)in->tiles[0].width,
                        (int)in->tiles[0].height);
        }
        /* matches_last_failure suppresses retry-on-broken-file at frame
         * rate; cleared when the file actually changes again. */
        if (s->reload_handle == NULL
            && overlay_watch_changed(&s->watch, s->cfg.file)
            && !matches_last_failure(s)) {
                kick_async_reload(s);
        }

        const time_ns_t blend_t0 = perf_t0(s);
        blend_overlay(s, in->color_spec, out, req_pitch);
        const time_ns_t blend_ns = s->cfg.perf
                ? get_time_in_ns() - blend_t0 : 0;

        perf_tally(s, frame_t0, blend_ns);
        return true;
}

static void
overlay_done(void *state)
{
        struct state_overlay *s = state;
        if (s->cfg.perf && s->perf.frames > 0) {
                overlay_perf_report(s, get_time_in_ns());
        }
        /* Drain any in-flight async reload before tearing down. wait_task
         * blocks here but we're shutting down anyway. */
        if (s->reload_handle != NULL) {
                wait_task(s->reload_handle);
                free(s->reload_job);
        }
        vf_free(s->in);
        free(s->overlay_rgba16);
        free(s->staging_rgba16);
        overlay_scaler_destroy(s->scaler);
        free(s);
}

static void
overlay_get_out_desc(void *state, struct video_desc *out,
                     int *in_display_mode, int *out_frames)
{
        struct state_overlay *s = state;
        *out             = s->saved_desc;
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames      = 1;
}

static bool
overlay_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state); UNUSED(property); UNUSED(val); UNUSED(len);
        return false;
}

static const struct vo_postprocess_info vo_pp_overlay_info = {
        overlay_init,
        overlay_postprocess_reconfigure,
        overlay_getf,
        overlay_get_out_desc,
        overlay_get_property,
        overlay_postprocess,
        overlay_done,
};

REGISTER_MODULE(overlay, &vo_pp_overlay_info,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);

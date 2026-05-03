/**
 * @file   test/test_overlay_watch.h
 * @author Ben Roeder     <ben@sohonet.com>
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
#ifndef TEST_OVERLAY_WATCH_H_2A9D8F1C_4B6E_4D7A_9F3B_5E8C2A1D7B4F
#define TEST_OVERLAY_WATCH_H_2A9D8F1C_4B6E_4D7A_9F3B_5E8C2A1D7B4F

int overlay_watch_test_init_no_change(void);
int overlay_watch_test_detects_size_change(void);
int overlay_watch_test_detects_mtime_change(void);
int overlay_watch_test_ack_on_missing_file_preserves_baseline(void);
int overlay_watch_test_missing_file_no_change(void);
int overlay_watch_test_file_appears_after_init(void);
int overlay_watch_test_detects_atomic_rename(void);
int overlay_watch_test_changed_does_not_consume_baseline(void);
int overlay_watch_test_ack_commits_baseline(void);

#endif

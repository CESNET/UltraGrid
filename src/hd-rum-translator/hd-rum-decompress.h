/**
 * @file   hd-rum-translator/hd-rum-decompress.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET, z. s. p. o.
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
#ifndef HD_RUM_DECOMPRESS_H_d6e9e9cd5dab
#define HD_RUM_DECOMPRESS_H_d6e9e9cd5dab

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct state_recompress;

enum hd_rum_mode_t {NORMAL, BLEND, CONFERENCE};
struct hd_rum_output_conf{
        enum hd_rum_mode_t mode;
        const char *arg;
};

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count);
void *hd_rum_decompress_init(struct module *parent, struct hd_rum_output_conf conf, const char *capture_filter, struct state_recompress *recompress);
void hd_rum_decompress_done(void *state);

#ifdef __cplusplus
}
#endif

struct video_frame;

#endif // HD_RUM_DECOMPRESS_H_d6e9e9cd5dab

/**
 * @file   hd-rum-translator/hd-rum/recompress.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct video_frame;

struct state_recompress;

struct state_recompress *recompress_init(struct module *parent);
void recompress_done(struct state_recompress *state);

uint32_t recompress_get_port_ssrc(struct state_recompress *s, int idx);


int recompress_add_port(struct state_recompress *s,
		const char *host, const char *compress, unsigned short rx_port,
		unsigned short tx_port, int mtu, const char *fec, long long bitrate);

void recompress_remove_port(struct state_recompress *s, int index);

void recompress_port_set_active(struct state_recompress *s,
                int index, bool active);

int recompress_get_num_active_ports(struct state_recompress *s);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <memory>
#include <string>
void recompress_process_async(state_recompress *state, std::shared_ptr<video_frame> frame);
#endif

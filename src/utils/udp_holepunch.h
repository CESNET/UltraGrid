/**
 * @file   utils/udp_holepuch.h
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#ifndef UG_UTILS_UDP_HOLEPUNCH_H
#define UG_UTILS_UDP_HOLEPUNCH_H

#ifndef __cplusplus
#include <stdbool.h>
#endif // ! defined __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

struct Holepunch_config{
        const char *client_name;
        const char *room_name;

        int *video_rx_port;
        int *video_tx_port;
        int *audio_rx_port;
        int *audio_tx_port;

        char *host_addr;
        size_t host_addr_len;

        const char *coord_srv_addr;
        int coord_srv_port;
        const char *stun_srv_addr;
        int stun_srv_port;
};

#ifdef HAVE_LIBJUICE
bool punch_udp(struct Holepunch_config c);
#endif

#ifdef __cplusplus
} //extern "C"
#endif

#endif

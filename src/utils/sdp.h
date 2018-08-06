/*
 * FILE:    sdp.h
 * AUTHORS: Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifndef __sdp_h
#define __sdp_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_STREAMS 2
#define strLength 2048

enum rtp_standard {
        std_H264,
        std_PCM
};

struct stream_info {
    char media_info[strLength];
    char rtpmap[strLength];
    char fmtp[strLength];
};

struct sdp {
    enum rtp_standard std_rtp;
    int port;
    char *version;
    char *origin;
    char *session_name;
    char *connection;
    char *times;
    struct stream_info stream[MAX_STREAMS];
    int stream_count; //between 1 and MAX_STREAMS
};

/*
 * External API
 */
struct sdp *new_sdp(enum rtp_standard std, int port);
bool get_sdp(struct sdp *sdp);

void set_version(struct sdp *sdp);
void get_version(struct sdp *sdp);

void set_origin(struct sdp *sdp);
void get_origin(struct sdp *sdp);

void set_session_name(struct sdp *sdp);
void get_session_name(struct sdp *sdp);

void set_connection(struct sdp *sdp);
void get_connection(struct sdp *sdp);

void set_times(struct sdp *sdp);
void get_times(struct sdp *sdp);

void set_stream(struct sdp *sdp);
void get_stream(struct sdp *sdp, int index);

/*
 * Internal API
 */
bool new_stream(struct sdp *sdp);
char *set_stream_media_info(struct sdp *sdp, int index);
char *set_stream_rtpmap(struct sdp *sdp, int index);
void clean_sdp(struct sdp *sdp);

#ifdef __cplusplus
}
#endif

#endif

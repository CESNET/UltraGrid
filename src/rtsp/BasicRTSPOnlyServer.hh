/*
 * FILE:    rtsp/BasicRTSPOnlyServer.hh
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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

#ifndef _BASIC_RTSP_ONLY_SERVER_HH
#define _BASIC_RTSP_ONLY_SERVER_HH

#include <RTSPServer.hh>
#include <BasicUsageEnvironment.hh>
#include "rtsp/rtsp_utils.h"
#include "audio/types.h"
#include "module.h"



class BasicRTSPOnlyServer {
private:
    BasicRTSPOnlyServer(int port, struct module *mod, rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio);

public:
    static BasicRTSPOnlyServer* initInstance(int port, struct module *mod, rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio);
    static BasicRTSPOnlyServer* getInstance();

    int init_server();

    static void *start_server(void *args);

    int update_server();

private:

    static BasicRTSPOnlyServer* srvInstance;
    int fPort;
    struct module *mod;
    rtps_types_t avType;
    audio_codec_t audio_codec;
    int audio_sample_rate;
    int audio_channels;
    int audio_bps;
    int rtp_port; //server rtp port
    int rtp_port_audio; //server rtp port
    RTSPServer* rtspServer;
    UsageEnvironment* env;
};

#endif

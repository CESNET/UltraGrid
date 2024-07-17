/*
 * FILE:    rtsp/BasicRTSPOnlySubsession.hh
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
#ifndef _BASIC_RTSP_ONLY_SUBSESSION_HH
#define _BASIC_RTSP_ONLY_SUBSESSION_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include <ServerMediaSession.hh>
#endif

#include <liveMedia_version.hh>

#include "rtsp/rtsp_utils.h"
#include "audio/types.h"
#include "module.h"
#include "control_socket.h"

// #ifndef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH
// #include <OnDemandServerMediaSubsession.hh>
// #endif

class Destinations {
public:
    Destinations(struct sockaddr_storage const& destAddr,
        Port const& rtpDestPort,
        Port const& rtcpDestPort)
: isTCP(False), addr(destAddr), rtpPort(rtpDestPort), rtcpPort(rtcpDestPort),
        tcpSocketNum(0), rtpChannelId(0), rtcpChannelId(0)
    {
    }
    Destinations(int tcpSockNum, unsigned char rtpChanId, unsigned char rtcpChanId)
    : isTCP(True), rtpPort(0) /*dummy*/, rtcpPort(0) /*dummy*/,
      tcpSocketNum(tcpSockNum), rtpChannelId(rtpChanId), rtcpChannelId(rtcpChanId) {
    }

public:
    Boolean isTCP;
    struct sockaddr_storage addr;
    Port rtpPort;
    Port rtcpPort;
    int tcpSocketNum;
    unsigned char rtpChannelId, rtcpChannelId;
};

#ifdef __clang__
#define MAYBE_UNUSED_ATTRIBUTE [[maybe_unused]]
#else
#define MAYBE_UNUSED_ATTRIBUTE // GCC complains if [[maybe_used]] is used there
#endif

class BasicRTSPOnlySubsession: public ServerMediaSubsession {

public:
    static BasicRTSPOnlySubsession*
    createNew(UsageEnvironment& env,
        Boolean reuseFirstSource,
        struct module *mod,
        rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio);

protected:

    BasicRTSPOnlySubsession(UsageEnvironment& env, Boolean reuseFirstSource,
        struct module *mod, rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio);

    virtual ~BasicRTSPOnlySubsession();

    virtual char const* sdpLines(int addressFamily);

    virtual void getStreamParameters(unsigned clientSessionId,
        struct sockaddr_storage const &clientAddress,
        Port const& clientRTPPort,
        Port const& clientRTCPPort,
        int tcpSocketNum,
        unsigned char rtpChannelId,
        unsigned char rtcpChannelId,
        TLSState *tlsState,
        struct sockaddr_storage &destinationAddress,
        uint8_t& destinationTTL,
        Boolean& isMulticast,
        Port& serverRTPPort,
        Port& serverRTCPPort,
        void*& streamToken);

    virtual void startStream(unsigned clientSessionId, void* streamToken,
        TaskFunc* rtcpRRHandler, void* rtcpRRHandlerClientData,
        unsigned short& rtpSeqNum,
        unsigned& rtpTimestamp,
        ServerRequestAlternativeByteHandler* serverRequestAlternativeByteHandler,
        void* serverRequestAlternativeByteHandlerClientData);

    virtual void deleteStream(unsigned clientSessionId, void*& streamToken);

#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1701302400 // 2023.11.30
    void getRTPSinkandRTCP(void*, RTPSink*&, RTCPInstance*&) override {}
#else
    void getRTPSinkandRTCP(void *, const RTPSink *&,
                           const RTCPInstance *&) override
    {
    }
#endif

protected:

    char* fSDPLines;
    Destinations* Vdestination;
    Destinations* Adestination;

private:

    void setSDPLines(int addressFamily);

    MAYBE_UNUSED_ATTRIBUTE Boolean fReuseFirstSource;
    MAYBE_UNUSED_ATTRIBUTE void* fLastStreamToken;
    char fCNAME[100];
    struct module *fmod;
    rtps_types_t avType;
    audio_codec_t audio_codec;
    int audio_sample_rate;
    int audio_channels;
    int audio_bps;
    int rtp_port; //server rtp port
    int rtp_port_audio; //server rtp port
};


#endif

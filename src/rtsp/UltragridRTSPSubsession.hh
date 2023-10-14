/*
 * FILE:    rtsp/UltragridRTSPSubsession.hh
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec     <pulec@cesnet.cz>
 *          Jakub Kováč      <xkovac5@mail.muni.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2010-2023 CESNET, z. s. p. o.
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
/**
 * Implement handling functions for RTSP methods
 * 
 * @note for inspiration look in Live555 library 
 *       liveMedia/include/OnDemandServerMediaSubsession.hh
 *       liveMedia/OnDemandServerMediaSubsession.cpp
*/

#ifndef BASIC_RTSP_SUBSESSION_HH
#define BASIC_RTSP_SUBSESSION_HH

#ifdef __clang__
#define MAYBE_UNUSED_ATTRIBUTE [[maybe_unused]]
#else
#define MAYBE_UNUSED_ATTRIBUTE // GCC complains if [[maybe_used]] is used there
#endif

#include <ServerMediaSession.hh>
#include <BasicUsageEnvironment.hh>
#include "module.h"

class UltragridRTSPSubsessionCommon: public ServerMediaSubsession {
protected:
    UltragridRTSPSubsessionCommon(UsageEnvironment& env, struct module *mod, int RTPPort, enum module_class *path_sender);

    /**
    * @note called by Live555 when handling SETUP method
    */
    virtual void getStreamParameters(
        unsigned clientSessionId, // in
        struct sockaddr_storage const& clientAddress, // in
        Port const& clientRTPPort, // in
        Port const& clientRTCPPort, // in
        int tcpSocketNum, // in (-1 means use UDP, not TCP)
        unsigned char rtpChannelId, // in (used if TCP)
        unsigned char rtcpChannelId, // in (used if TCP)
        TLSState* tlsState, // in (used if TCP)
        struct sockaddr_storage& destinationAddress, // in out
        u_int8_t& destinationTTL, // in out
        Boolean& isMulticast, // out
        Port& serverRTPPort, // out
        Port& serverRTCPPort, // out
        void*& streamToken // out
    );

    /**
    * @note called by Live555 when handling PLAY method
    */
    virtual void startStream(
        unsigned clientSessionId,
        void* streamToken,
        TaskFunc* rtcpRRHandler,
        void* rtcpRRHandlerClientData,
        unsigned short& rtpSeqNum,
        unsigned& rtpTimestamp,
        ServerRequestAlternativeByteHandler* serverRequestAlternativeByteHandler,
        void* serverRequestAlternativeByteHandlerClientData
    );

    /**
    * @note called by Live555 when handling TEARDOWN method
    */
    virtual void deleteStream(unsigned clientSessionId, void*& streamToken);

    /**
    * none of the arguments are used in this implementaton of server, so the function does nothing
    */
    virtual void getRTPSinkandRTCP(
        MAYBE_UNUSED_ATTRIBUTE void* /* streamToken */,
        MAYBE_UNUSED_ATTRIBUTE RTPSink *& /* rtpSink */,
        MAYBE_UNUSED_ATTRIBUTE RTCPInstance *& /* rtcp */) {}

    /**
    * tells UltraGrid (SENDER module) where (IP adress and port) to send media
    */
    void redirectStream(const char* destinationAddress, int destinationPort);

    UsageEnvironment& env;
    struct sockaddr_storage destAddress;
    Port destPort;

    struct module *mod;
    int RTPPort;

    enum module_class *path_sender;
    const Boolean fReuseFirstSource = True; // tells Live555 that all clients use same source, eg. no pausing, seeking ...
};

class UltragridRTSPVideoSubsession: public UltragridRTSPSubsessionCommon {
public:
    static UltragridRTSPVideoSubsession* createNew(UsageEnvironment& env, struct module *mod, int RTPPort);
    UltragridRTSPVideoSubsession(UsageEnvironment& env, struct module *mod, int RTPPort);
};

class UltragridRTSPAudioSubsession: public UltragridRTSPSubsessionCommon {
public:
    static UltragridRTSPAudioSubsession* createNew(UsageEnvironment& env, struct module *mod, int RTPPort);
    UltragridRTSPAudioSubsession(UsageEnvironment& env, struct module *mod, int RTPPort);
};

typedef UltragridRTSPSubsessionCommon BasicRTSPOnlySubsession; // kept for legacy maintanance

#undef MAYBE_UNUSED_ATTRIBUTE

#endif // BASIC_RTSP_SUBSESSION_HH

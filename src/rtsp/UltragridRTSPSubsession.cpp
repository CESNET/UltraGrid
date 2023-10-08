/*
 * FILE:    rtsp/UltragridRTSPSubsession.cpp
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

#include "rtsp/UltragridRTSPSubsession.hh"
#include "messaging.h"
#include "host.h"

#ifdef __clang__
#define MAYBE_UNUSED_ATTRIBUTE [[maybe_unused]]
#else
#define MAYBE_UNUSED_ATTRIBUTE // GCC complains if [[maybe_used]] is used there
#endif

UltragridRTSPSubsessionCommon::UltragridRTSPSubsessionCommon(int RTPPort)
    : destPort(0), RTPPort(RTPPort) {}

void UltragridRTSPSubsessionCommon::getStreamParameters(
    MAYBE_UNUSED_ATTRIBUTE unsigned /* clientSessionId */, // in
    MAYBE_UNUSED_ATTRIBUTE struct sockaddr_storage const& clientAddress, // in
    MAYBE_UNUSED_ATTRIBUTE Port const& clientRTPPort, // in
    MAYBE_UNUSED_ATTRIBUTE Port const& /* clientRTCPPort */, // in
    MAYBE_UNUSED_ATTRIBUTE int /* tcpSocketNum */, // in (-1 means use UDP, not TCP)
    MAYBE_UNUSED_ATTRIBUTE unsigned char /* rtpChannelId */, // in (used if TCP)
    MAYBE_UNUSED_ATTRIBUTE unsigned char /* rtcpChannelId */, // in (used if TCP)
    MAYBE_UNUSED_ATTRIBUTE TLSState* /* tlsState */, // in (used if TCP)
    MAYBE_UNUSED_ATTRIBUTE struct sockaddr_storage& destinationAddress, // in out
    MAYBE_UNUSED_ATTRIBUTE u_int8_t& /* destinationTTL */, // in out
    MAYBE_UNUSED_ATTRIBUTE Boolean& /* isMulticast */, // out
    MAYBE_UNUSED_ATTRIBUTE Port& serverRTPPort, // out
    MAYBE_UNUSED_ATTRIBUTE Port& serverRTCPPort, // out
    MAYBE_UNUSED_ATTRIBUTE void*& /* streamToken */ // out
    ) {
    
    // out
    destinationAddress = clientAddress;
    serverRTPPort = Port(RTPPort);
    serverRTCPPort = Port(RTPPort + 1);
    
    // in
    destAddress = destinationAddress;
    destPort = clientRTPPort;
}

#undef MAYBE_UNUSED_ATTRIBUTE

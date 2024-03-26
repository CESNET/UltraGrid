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

UltragridRTSPSubsessionCommon::UltragridRTSPSubsessionCommon(UsageEnvironment& env, struct module *mod, int RTPPort, enum module_class *path_sender)
    : ServerMediaSubsession(env), env(env), destPort(0), mod(mod), RTPPort(RTPPort), path_sender(path_sender) {
        if (mod == NULL)
            throw std::system_error();
}

UltragridRTSPVideoSubsession* UltragridRTSPVideoSubsession::createNew(UsageEnvironment& env, struct module *mod, int RTPPort) {
    return new UltragridRTSPVideoSubsession(env, mod, RTPPort);
}

UltragridRTSPVideoSubsession::UltragridRTSPVideoSubsession(UsageEnvironment& env, struct module *mod, int RTPPort)
    : UltragridRTSPSubsessionCommon(env, mod, RTPPort, path_sender) {}

UltragridRTSPAudioSubsession* UltragridRTSPAudioSubsession::createNew(UsageEnvironment& env, struct module *mod, int RTPPort, int sampleRate, int numOfChannels, audio_codec_t codec) {
    return new UltragridRTSPAudioSubsession(env, mod, RTPPort, sampleRate, numOfChannels, codec);
}

UltragridRTSPAudioSubsession::UltragridRTSPAudioSubsession(UsageEnvironment& env, struct module *mod, int RTPPort, int sampleRate, int numOfChannels, audio_codec_t codec)
    : UltragridRTSPSubsessionCommon(env, mod, RTPPort, path_sender), sampleRate(sampleRate), numOfChannels(numOfChannels), codec(codec) {}

char const* UltragridRTSPVideoSubsession::sdpLines(int addressFamily) {
    // already created
    if (SDPLines.size() != 0)
        return SDPLines.c_str();

    AddressString ipAddressStr(nullAddress(addressFamily));

    SDPLines += "m=video " + std::to_string(RTPPort) + " RTP/AVP 96\r\n";
    SDPLines += std::string("c=IN ") + (addressFamily == AF_INET ? "IP4 " : "IP6 ") + std::string(ipAddressStr.val()) + "\r\n";
    SDPLines += "b=AS:5000\r\n";
    SDPLines += "a=rtcp:" + std::to_string(RTPPort + 1) + "\r\n";
    SDPLines += "a=rtpmap:96 H264/90000\r\n";
    SDPLines += "a=control:" + std::string(trackId()) + "\r\n";

    return SDPLines.c_str();
}

char const* UltragridRTSPAudioSubsession::sdpLines(int addressFamily) {
    // already created
    if (SDPLines.size() != 0)
        return SDPLines.c_str();

    AddressString ipAddressStr(nullAddress(addressFamily));

    std::string audioCodec;
    audioCodec.append(codec == AC_MULAW ? "PCMU" : codec == AC_ALAW ? "PCMA" : "OPUS");

    int RTPPayloadType = calculateRTPPayloadType();

    SDPLines += "m=audio " + std::to_string(RTPPort) + " RTP/AVP " + std::to_string(RTPPayloadType) + "\r\n";
    SDPLines += std::string("c=IN ") + (addressFamily == AF_INET ? "IP4 " : "IP6 ") + std::string(ipAddressStr.val()) + "\r\n";
    SDPLines += "b=AS:384\r\n";
    SDPLines += "a=rtcp:" + std::to_string(RTPPort + 1) + "\r\n";
    SDPLines += "a=rtpmap:" + std::to_string(RTPPayloadType) + " " + std::move(audioCodec) + "/" + std::to_string(sampleRate) + "/" + std::to_string(numOfChannels) + "\r\n";
    SDPLines += "a=control:" + std::string(trackId()) + "\r\n";

    return SDPLines.c_str();
}

int UltragridRTSPAudioSubsession::calculateRTPPayloadType() {
    if (sampleRate != 8000 || numOfChannels != 1)
        return 97;

    if (codec == AC_MULAW)
        return 0;
    if (codec == AC_ALAW)
        return 8;
    return 97;
}

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

void UltragridRTSPSubsessionCommon::startStream(
        MAYBE_UNUSED_ATTRIBUTE unsigned /* clientSessionId */,
        MAYBE_UNUSED_ATTRIBUTE void* /* streamToken */,
        MAYBE_UNUSED_ATTRIBUTE TaskFunc* /* rtcpRRHandler */,
        MAYBE_UNUSED_ATTRIBUTE void* /* rtcpRRHandlerClientData */,
        MAYBE_UNUSED_ATTRIBUTE unsigned short& /* rtpSeqNum */,
        MAYBE_UNUSED_ATTRIBUTE unsigned& /* rtpTimestamp */,
        MAYBE_UNUSED_ATTRIBUTE ServerRequestAlternativeByteHandler* /* serverRequestAlternativeByteHandler */,
        MAYBE_UNUSED_ATTRIBUTE void* /* serverRequestAlternativeByteHandlerClientData */
    ) {

    if (addressIsNull(destAddress) || destPort.num() == 0) {
        env << "[RTSP Server] Error: Failed to start (audio OR video) stream due to empty destination address\n";
        return;
    }
    
    redirectStream(AddressString(destAddress).val(), destPort.num());
}

void UltragridRTSPSubsessionCommon::deleteStream(MAYBE_UNUSED_ATTRIBUTE unsigned /* clientSessionId */, MAYBE_UNUSED_ATTRIBUTE  void*& /* streamToken */) {
    destAddress = sockaddr_storage();
    destPort = Port(0);

    redirectStream("127.0.0.1", RTPPort);
}

void UltragridRTSPSubsessionCommon::redirectStream(const char* destinationAddress, int destinationPort) {
    char pathV[1024];
    memset(pathV, 0, sizeof(pathV));
    append_message_path(pathV, sizeof(pathV), path_sender);
    struct response *resp = NULL;
    // change destination port
    struct msg_sender *msg1 = (struct msg_sender *) new_message(sizeof(struct msg_sender));
    msg1->tx_port = ntohs(destinationPort);
    msg1->type = SENDER_MSG_CHANGE_PORT;
    resp = send_message(mod, pathV, (struct message *) msg1);
    free_response(resp);

    // change destination address
    struct msg_sender *msg2 = (struct msg_sender *) new_message(sizeof(struct msg_sender));
    strncpy(msg2->receiver, destinationAddress, sizeof(msg2->receiver) - 1);
    msg2->type = SENDER_MSG_CHANGE_RECEIVER;
    resp = send_message(mod, pathV, (struct message *) msg2);
    free_response(resp);
}

#undef MAYBE_UNUSED_ATTRIBUTE

/*
 * FILE:    rtsp/BasicRTSPOnlySubsession.cpp
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2014-2026 CESNET, zájmové sdružení právnických osob
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

#include "rtsp/BasicRTSPOnlySubsession.hh"
#include <cassert>
#include <BasicUsageEnvironment.hh>
#include <RTSPServer.hh>
#include <GroupsockHelper.hh>

#include "audio/codec.h"          // get_name_to_audio_codec
#include "debug.h"                // for MSG
#include "messaging.h"
#include "module.h"               // for module_class, append_message...
#include "utils/macros.h"
#include "utils/net.h"
#include "utils/sdp.h"
#include "video_codec.h"          // for get_codec_name

#define MOD_NAME "[RTSP] "

BasicRTSPOnlySubsession*
BasicRTSPOnlySubsession::createNew(UsageEnvironment& env,
		Boolean reuseFirstSource, rtsp_types_t avType, int rtpPort,
		struct rtsp_server_parameters params) {
        assert(avType == rtsp_type_audio || avType == rtsp_type_video);
        return new BasicRTSPOnlySubsession(env, reuseFirstSource, avType,
                                           rtpPort, params);
}

constexpr enum module_class path_sender_audio[] = { MODULE_CLASS_AUDIO,
                                                    MODULE_CLASS_SENDER,
                                                    MODULE_CLASS_NONE };
constexpr enum module_class path_sender_video[] = { MODULE_CLASS_SENDER,
                                                    MODULE_CLASS_VIDEO,
                                                    MODULE_CLASS_NONE };

BasicRTSPOnlySubsession::BasicRTSPOnlySubsession(UsageEnvironment& env,
		Boolean reuseFirstSource, rtsp_types_t avType, int rtpPort,
		struct rtsp_server_parameters params) :
		ServerMediaSubsession(env), fReuseFirstSource(reuseFirstSource),
		fLastStreamToken(nullptr), rtsp_params(params)
{
	assert(avType == rtsp_type_audio || avType == rtsp_type_video);
	destination = NULL;
	gethostname(fCNAME, sizeof fCNAME);
	this->avType = avType;
	this->rtpPort = rtpPort;
        this->sender_msg_path =
            avType == rtsp_type_audio ? path_sender_audio : path_sender_video;
	fCNAME[sizeof fCNAME - 1] = '\0';

	// print (preliminary) SDP
	setSDPLines(AF_UNSPEC);
	delete[] fSDPLines;
	fSDPLines = nullptr;
}

BasicRTSPOnlySubsession::~BasicRTSPOnlySubsession() {
	delete[] fSDPLines;
	delete destination;
}

const static struct media_spec {
        unsigned    estBitrate;
        const char *mname;
} media_params[] = {
        { 0,    nullptr }, // none
        { 384,  "audio" },
        { 5000, "video" },
};
static_assert(rtsp_type_audio == 1); // ensure the above mapping is correct
static_assert(rtsp_type_video == 2);

char const* BasicRTSPOnlySubsession::sdpLines(int addressFamily) {
	if (fSDPLines == NULL) {
                setSDPLines(addressFamily);
	}
	return fSDPLines;
}

void BasicRTSPOnlySubsession::setSDPLines(int addressFamily) {
	//TODO: should be more dynamic
        const char *ip_ver_list_addr = nullptr;
        const char *control  = trackId();
        switch (addressFamily) {
        case AF_INET:
                ip_ver_list_addr = "4 0.0.0.0";
                break;
        case AF_INET6:
                ip_ver_list_addr = "6 ::";
                break;
        case AF_UNSPEC:
                ip_ver_list_addr = "<VER> <TO_BE_FILLED>";
                control = "<CONTROL>"; // is null here
                break;
        default:
                abort();
        }

        const struct media_spec *mspec = &media_params[avType];
        char rtpmapLine[STR_LEN];
        int rtpPayloadType = avType == rtsp_type_audio
                  ? get_audio_rtp_pt_rtpmap(
                      rtsp_params.adesc.codec, rtsp_params.adesc.sample_rate,
                      rtsp_params.adesc.ch_count, rtpmapLine)
                  : get_video_rtp_pt_rtpmap(rtsp_params.video_codec, rtpmapLine);
        if (rtpPayloadType < 0) {
                MSG(ERROR, "Unsupported %s codec %s!\n", mspec->mname,
                    avType == rtsp_type_audio
                        ? get_name_to_audio_codec(rtsp_params.adesc.codec)
                        : get_codec_name(rtsp_params.video_codec));
        }
        //char const* auxSDPLine = "";

        char const *const sdpFmt = "m=%s %u RTP/AVP %d\r\n"
                                   "c=IN IP%s\r\n"
                                   "b=AS:%u\r\n"
                                   "a=rtcp:%d\r\n"
                                   "%s"
                                   "a=control:%s\r\n";
        unsigned sdpFmtSize = strlen(sdpFmt) + strlen(mspec->mname) +
                              5   /* max short len */
                              + 3 /* max char len */
                              + strlen(ip_ver_list_addr) + 20 /* max int len */
                              + strlen(rtpmapLine) + strlen(control);
        char *sdpLines = new char[sdpFmtSize];

        snprintf(sdpLines, sdpFmtSize, sdpFmt, mspec->mname, // m= <media>
                 rtpPort,         // fPortNumForSDP, // m= <port>
                 rtpPayloadType,    // m= <fmt list>
                 ip_ver_list_addr,  // c= address
                 mspec->estBitrate, // b=AS:<bandwidth>
                 rtpPort + 1,
                 rtpmapLine, // a=rtpmap:... (if present)
                 control); // a=control:<track-id>

        fSDPLines = sdpLines;
        MSG(VERBOSE, "SDP%s:\n%s\n",
            addressFamily == AF_UNSPEC ? " (preliminary)" : "", fSDPLines);
}

void BasicRTSPOnlySubsession::getStreamParameters(unsigned /* clientSessionId */,
		struct sockaddr_storage const &clientAddress, Port const& clientRTPPort,
		Port const& clientRTCPPort, int /* tcpSocketNum */,
		unsigned char /* rtpChannelId */, unsigned char /* rtcpChannelId */,
                TLSState * /* tlsState */,
		struct sockaddr_storage& /*destinationAddress*/, uint8_t& /*destinationTTL*/,
		Boolean& /* isMulticast */, Port& serverRTPPort, Port& serverRTCPPort,
		void*& /* streamToken */) {
        delete destination;
        serverRTPPort  = avType == rtsp_type_video ? rtsp_params.rtp_port_video
                                                   : rtsp_params.rtp_port_audio;
        serverRTCPPort = serverRTPPort.num() + 1;
        destination =
            new Destinations(clientAddress, clientRTPPort, clientRTCPPort);
}

void BasicRTSPOnlySubsession::startStream(unsigned /* clientSessionId */,
		void* /* streamToken */, TaskFunc* /* rtcpRRHandler */,
		void* /* rtcpRRHandlerClientData */, unsigned short& /* rtpSeqNum */,
		unsigned& /* rtpTimestamp */,
		ServerRequestAlternativeByteHandler* /* serverRequestAlternativeByteHandler */,
		void* /* serverRequestAlternativeByteHandlerClientData */) {
	struct response *resp = NULL;

	/// @todo shouldn't here be rather assert?
	if (destination == nullptr) {
	        return;
	}

	char path[1024] = "";

	append_message_path(path, sizeof(path), sender_msg_path);

	//CHANGE DST PORT
        auto *msg1 =
            (struct msg_sender *) new_message(sizeof(struct msg_sender));
        msg1->tx_port = ntohs(destination->rtpPort.num());
        msg1->type    = SENDER_MSG_CHANGE_PORT;
        resp = send_message(rtsp_params.parent, path, (struct message *) msg1);
        free_response(resp);

        // CHANGE DST ADDRESS
        auto *msg2 =
            (struct msg_sender *) new_message(sizeof(struct msg_sender));
        char      host[IN6_MAX_ASCII_LEN + 1];
        const int ret = getnameinfo((struct sockaddr *) &destination->addr,
                                    sizeof destination->addr, host, sizeof host,
                                    nullptr, 0, NI_NUMERICHOST);
        assert(ret == 0);
        strncpy(msg2->receiver, host, sizeof(msg2->receiver) - 1);
        msg2->type = SENDER_MSG_CHANGE_RECEIVER;

        resp = send_message(rtsp_params.parent, path, (struct message *) msg2);
        free_response(resp);
}

void BasicRTSPOnlySubsession::deleteStream(unsigned /* clientSessionId */,
		void*& /* streamToken */) {
        char path[1024] = "";
        delete destination;
        destination = nullptr;
        append_message_path(path, sizeof(path), sender_msg_path);

        // CHANGE DST PORT
        auto *msg1 =
            (struct msg_sender *) new_message(sizeof(struct msg_sender));
        msg1->tx_port = rtsp_params.rtp_port_video;
        msg1->type    = SENDER_MSG_CHANGE_PORT;
        struct response *resp =
            send_message(rtsp_params.parent, path, (struct message *) msg1);
        free_response(resp);

        // CHANGE DST ADDRESS
        auto *msg2 =
            (struct msg_sender *) new_message(sizeof(struct msg_sender));
        strncpy(msg2->receiver, "127.0.0.1", sizeof(msg2->receiver) - 1);
        msg2->type = SENDER_MSG_CHANGE_RECEIVER;
        resp = send_message(rtsp_params.parent, path, (struct message *) msg2);
        free_response(resp);
}
/* vi: set noexpandtab: */

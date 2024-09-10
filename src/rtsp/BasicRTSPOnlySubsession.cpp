/*
 * FILE:    rtsp/BasicRTSPOnlySubsession.cpp
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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
        return new BasicRTSPOnlySubsession(env, reuseFirstSource, avType,
                                           rtpPort, params);
}

BasicRTSPOnlySubsession::BasicRTSPOnlySubsession(UsageEnvironment& env,
		Boolean reuseFirstSource, rtsp_types_t avType, int rtpPort,
		struct rtsp_server_parameters params) :
		ServerMediaSubsession(env), fReuseFirstSource(reuseFirstSource),
		fLastStreamToken(nullptr), rtsp_params(params)
{
	assert(avType == rtsp_type_audio || avType == rtsp_type_video);
	Vdestination = NULL;
	Adestination = NULL;
	gethostname(fCNAME, sizeof fCNAME);
	this->avType = avType;
	this->rtpPort = rtpPort;
	fCNAME[sizeof fCNAME - 1] = '\0';

	// print (preliminary) SDP
	setSDPLines(AF_UNSPEC);
	delete[] fSDPLines;
	fSDPLines = nullptr;
}

BasicRTSPOnlySubsession::~BasicRTSPOnlySubsession() {
	delete[] fSDPLines;
	delete Adestination;
	delete Vdestination;
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
                MSG(ERROR, "Unsupported %s codec %s!\n", mspec[avType].mname,
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
	if (avType == rtsp_type_video) {
		Port rtp(rtsp_params.rtp_port_video);
		serverRTPPort = rtp;
		Port rtcp(rtsp_params.rtp_port_video + 1);
		serverRTCPPort = rtcp;

		delete Vdestination;
		Vdestination = new Destinations(clientAddress, clientRTPPort,
				clientRTCPPort);
	}
	if (avType == rtsp_type_audio) {
		Port rtp(rtsp_params.rtp_port_audio);
		serverRTPPort = rtp;
		Port rtcp(rtsp_params.rtp_port_audio + 1);
		serverRTCPPort = rtcp;

		delete Adestination;
		Adestination = new Destinations(clientAddress, clientRTPPort,
				clientRTCPPort);
	}
}

void BasicRTSPOnlySubsession::startStream(unsigned /* clientSessionId */,
		void* /* streamToken */, TaskFunc* /* rtcpRRHandler */,
		void* /* rtcpRRHandlerClientData */, unsigned short& /* rtpSeqNum */,
		unsigned& /* rtpTimestamp */,
		ServerRequestAlternativeByteHandler* /* serverRequestAlternativeByteHandler */,
		void* /* serverRequestAlternativeByteHandlerClientData */) {
	struct response *resp = NULL;

	if (Vdestination != NULL) {
		if (avType == rtsp_type_video) {
			char pathV[1024];

			memset(pathV, 0, sizeof(pathV));
			enum module_class path_sender[] = { MODULE_CLASS_SENDER,
					MODULE_CLASS_NONE };
			append_message_path(pathV, sizeof(pathV), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgV1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			msgV1->tx_port = ntohs(Vdestination->rtpPort.num());
			msgV1->type = SENDER_MSG_CHANGE_PORT;
			resp = send_message(rtsp_params.parent, pathV, (struct message *) msgV1);
			free_response(resp);

			//CHANGE DST ADDRESS
			struct msg_sender *msgV2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
                        char host[IN6_MAX_ASCII_LEN + 1];
                        const int ret =
                            getnameinfo((struct sockaddr *) &Vdestination->addr,
                                        sizeof Vdestination->addr, host,
                                        sizeof host, nullptr, 0, NI_NUMERICHOST);
                        assert(ret == 0);
			strncpy(msgV2->receiver, host,
					sizeof(msgV2->receiver) - 1);
			msgV2->type = SENDER_MSG_CHANGE_RECEIVER;

			resp = send_message(rtsp_params.parent, pathV, (struct message *) msgV2);
			free_response(resp);
		}
	}

	if (Adestination != NULL) {
		if (avType == rtsp_type_audio) {
			char pathA[1024];

			memset(pathA, 0, sizeof(pathA));
			enum module_class path_sender[] = { MODULE_CLASS_AUDIO,
					MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
			append_message_path(pathA, sizeof(pathA), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgA1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			msgA1->tx_port = ntohs(Adestination->rtpPort.num());
			msgA1->type = SENDER_MSG_CHANGE_PORT;
			resp = send_message(rtsp_params.parent, pathA, (struct message *) msgA1);
			free_response(resp);
			resp = NULL;

			//CHANGE DST ADDRESS
			struct msg_sender *msgA2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
                        char host[IN6_MAX_ASCII_LEN + 1];
                        const int ret =
                            getnameinfo((struct sockaddr *) &Adestination->addr,
                                        sizeof Adestination->addr, host,
                                        sizeof host, nullptr, 0, NI_NUMERICHOST);
                        assert(ret == 0);
                        strncpy(msgA2->receiver, host,
					sizeof(msgA2->receiver) - 1);
			msgA2->type = SENDER_MSG_CHANGE_RECEIVER;

			resp = send_message(rtsp_params.parent, pathA, (struct message *) msgA2);
			free_response(resp);
			resp = NULL;
		}
	}
}

void BasicRTSPOnlySubsession::deleteStream(unsigned /* clientSessionId */,
		void*& /* streamToken */) {
	if (Vdestination != NULL) {
		if (avType == rtsp_type_video) {
			char pathV[1024];
			delete Vdestination;
			Vdestination = NULL;
			memset(pathV, 0, sizeof(pathV));
			enum module_class path_sender[] = { MODULE_CLASS_SENDER,
					MODULE_CLASS_NONE };
			append_message_path(pathV, sizeof(pathV), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgV1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			msgV1->tx_port = rtsp_params.rtp_port_video;
			msgV1->type = SENDER_MSG_CHANGE_PORT;
			struct response *resp;
			resp = send_message(rtsp_params.parent, pathV, (struct message *) msgV1);
			free_response(resp);

			//CHANGE DST ADDRESS
			struct msg_sender *msgV2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgV2->receiver, "127.0.0.1", sizeof(msgV2->receiver) - 1);
			msgV2->type = SENDER_MSG_CHANGE_RECEIVER;
			resp = send_message(rtsp_params.parent, pathV, (struct message *) msgV2);
			free_response(resp);
		}
	}

	if (Adestination != NULL) {
		if (avType == rtsp_type_audio) {
			char pathA[1024];
			delete Adestination;
			Adestination = NULL;
			memset(pathA, 0, sizeof(pathA));
			enum module_class path_sender[] = { MODULE_CLASS_AUDIO,
					MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
			append_message_path(pathA, sizeof(pathA), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgA1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));

			//TODO: GET AUDIO PORT SET (NOT A COMMON CASE WHEN RTSP IS ENABLED: DEFAULT -> vport + 2)
			msgA1->tx_port = rtsp_params.rtp_port_audio;
			msgA1->type = SENDER_MSG_CHANGE_PORT;
			struct response *resp;
                        resp = send_message(rtsp_params.parent, pathA, (struct message *) msgA1);
			free_response(resp);

			//CHANGE DST ADDRESS
			struct msg_sender *msgA2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgA2->receiver, "127.0.0.1", sizeof(msgA2->receiver) - 1);
			msgA2->type = SENDER_MSG_CHANGE_RECEIVER;
			resp = send_message(rtsp_params.parent, pathA, (struct message *) msgA2);
			free_response(resp);
		}
	}
}
/* vi: set noexpandtab: */

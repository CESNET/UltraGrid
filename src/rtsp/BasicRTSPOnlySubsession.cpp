/*
 * FILE:    BasicRTSPOnlySubsession.cpp
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

#include "rtsp/BasicRTSPOnlySubsession.hh"
#include <BasicUsageEnvironment.hh>
#include <RTSPServer.hh>
#include <GroupsockHelper.hh>

#include "messaging.h"

BasicRTSPOnlySubsession*
BasicRTSPOnlySubsession::createNew(UsageEnvironment& env,
		Boolean reuseFirstSource, struct module *mod, rtps_types_t avType,
		audio_codec_t audio_codec, int audio_sample_rate, int audio_channels,
		int audio_bps, int rtp_port, int rtp_port_audio) {
	return new BasicRTSPOnlySubsession(env, reuseFirstSource, mod, avType,
			audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio);
}

BasicRTSPOnlySubsession::BasicRTSPOnlySubsession(UsageEnvironment& env,
		Boolean reuseFirstSource, struct module *mod, rtps_types_t avType,
		audio_codec_t audio_codec, int audio_sample_rate, int audio_channels,
		int audio_bps, int rtp_port, int rtp_port_audio) :
		ServerMediaSubsession(env), fSDPLines(NULL), fReuseFirstSource(
				reuseFirstSource), fLastStreamToken(NULL) {
	Vdestination = NULL;
	Adestination = NULL;
	gethostname(fCNAME, sizeof fCNAME);
	this->fmod = mod;
	this->avType = avType;
	this->audio_codec = audio_codec;
	this->audio_sample_rate = audio_sample_rate;
	this->audio_channels = audio_channels;
	this->audio_bps = audio_bps;
	this->rtp_port = rtp_port;
	this->rtp_port_audio = rtp_port_audio;
	fCNAME[sizeof fCNAME - 1] = '\0';
}

BasicRTSPOnlySubsession::~BasicRTSPOnlySubsession() {
	delete[] fSDPLines;
	delete Adestination;
	delete Vdestination;
}

char const* BasicRTSPOnlySubsession::sdpLines() {
	if (fSDPLines == NULL) {
		setSDPLines();
	}
	if (Adestination != NULL || Vdestination != NULL)
		return NULL;
	return fSDPLines;
}

void BasicRTSPOnlySubsession::setSDPLines() {
	//TODO: should be more dynamic
	//VStream
	if (avType == video || avType == av) {
		unsigned estBitrate = 5000;
		char const* mediaType = "video";
		uint8_t rtpPayloadType = 96;
		AddressString ipAddressStr(fServerAddressForSDP);
		char* rtpmapLine = strdup("a=rtpmap:96 H264/90000\n");
		//char const* auxSDPLine = "";

		char const* const sdpFmt = "m=%s %u RTP/AVP %u\r\n"
				"c=IN IP4 %s\r\n"
				"b=AS:%u\r\n"
				"a=rtcp:%d\r\n"
				"%s"
				"a=control:%s\r\n";
		unsigned sdpFmtSize = strlen(sdpFmt) + strlen(mediaType) + 5 /* max short len */
				+ 3 /* max char len */
				+ strlen(ipAddressStr.val()) + 20 /* max int len */
				+ strlen(rtpmapLine) + strlen(trackId());
		char* sdpLines = new char[sdpFmtSize];

		sprintf(sdpLines, sdpFmt, mediaType, // m= <media>
				rtp_port,//fPortNumForSDP, // m= <port>
				rtpPayloadType, // m= <fmt list>
				ipAddressStr.val(), // c= address
				estBitrate, // b=AS:<bandwidth>
				rtp_port + 1,
				rtpmapLine, // a=rtpmap:... (if present)
				trackId()); // a=control:<track-id>

		fSDPLines = sdpLines;
	}
	//AStream
	if (avType == audio || avType == av) {
		unsigned estBitrate = 384;
		char const* mediaType = "audio";
		AddressString ipAddressStr(fServerAddressForSDP);
		uint8_t rtpPayloadType;

		if (audio_sample_rate == 8000 && audio_channels == 1) { //NOW NOT COMPUTING 1 BPS BECAUSE RESAMPLER FORCES TO 2 BPS...
			if (audio_codec == AC_MULAW)
				rtpPayloadType = 0;
			else if (audio_codec == AC_ALAW)
				rtpPayloadType = 8;
			else rtpPayloadType = 97;
		} else {
			rtpPayloadType = 97;
		}

		char* rtpmapLine = strdup("a=rtpmap:97 PCMU/48000/2\n"); //only to alloc max possible size
		//char const* auxSDPLine = "";

		char const* const sdpFmt = "m=%s %u RTP/AVP %u\r\n"
				"c=IN IP4 %s\r\n"
				"b=AS:%u\r\n"
				"a=rtcp:%d\r\n"
				"a=rtpmap:%u %s/%d/%d\r\n"
				"a=control:%s\r\n";
		unsigned sdpFmtSize = strlen(sdpFmt) + strlen(mediaType) + 5 /* max short len */
				+ 3 /* max char len */
				+ strlen(ipAddressStr.val()) + 20 /* max int len */
				+ strlen(rtpmapLine) + strlen(trackId());
		char* sdpLines = new char[sdpFmtSize];

		sprintf(sdpLines, sdpFmt,
				mediaType, // m= <media>
				rtp_port_audio,//fPortNumForSDP, // m= <port>
				rtpPayloadType, // m= <fmt list>
				ipAddressStr.val(), // c= address
				estBitrate, // b=AS:<bandwidth>
				//rtpmapLine, // a=rtpmap:... (if present)
				rtp_port_audio + 1,
				rtpPayloadType,
				audio_codec == AC_MULAW ? "PCMU" : audio_codec == AC_ALAW ? "PCMA" : "OPUS",
				audio_sample_rate,
				audio_channels,
				trackId()); // a=control:<track-id>

		fSDPLines = sdpLines;
	}
}

void BasicRTSPOnlySubsession::getStreamParameters(unsigned /* clientSessionId */,
		netAddressBits clientAddress, Port const& clientRTPPort,
		Port const& clientRTCPPort, int /* tcpSocketNum */,
		unsigned char /* rtpChannelId */, unsigned char /* rtcpChannelId */,
		netAddressBits& destinationAddress, uint8_t& /*destinationTTL*/,
		Boolean& /* isMulticast */, Port& serverRTPPort, Port& serverRTCPPort,
		void*& /* streamToken */) {
	if (Vdestination == NULL && (avType == video || avType == av)) {
		Port rtp(rtp_port);
		serverRTPPort = rtp;
		Port rtcp(rtp_port + 1);
		serverRTCPPort = rtcp;

		if (fSDPLines == NULL) {
			setSDPLines();
		}
		if (destinationAddress == 0) {
			destinationAddress = clientAddress;
		}
		struct in_addr destinationAddr;
		destinationAddr.s_addr = destinationAddress;
		Vdestination = new Destinations(destinationAddr, clientRTPPort,
				clientRTCPPort);
	}
	if (Adestination == NULL && (avType == audio || avType == av)) {
		Port rtp(rtp_port_audio);
		serverRTPPort = rtp;
		Port rtcp(rtp_port_audio + 1);
		serverRTCPPort = rtcp;

		if (fSDPLines == NULL) {
			setSDPLines();
		}
		if (destinationAddress == 0) {
			destinationAddress = clientAddress;
		}
		struct in_addr destinationAddr;
		destinationAddr.s_addr = destinationAddress;
		Adestination = new Destinations(destinationAddr, clientRTPPort,
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
		if (avType == video || avType == av) {
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
			resp = send_message(fmod, pathV, (struct message *) msgV1);
			resp = NULL;

			//CHANGE DST ADDRESS
			struct msg_sender *msgV2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgV2->receiver, inet_ntoa(Vdestination->addr),
					sizeof(msgV2->receiver) - 1);
			msgV2->type = SENDER_MSG_CHANGE_RECEIVER;

			resp = send_message(fmod, pathV, (struct message *) msgV2);
			resp = NULL;
		}
	}

	if (Adestination != NULL) {
		if (avType == audio || avType == av) {
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
			resp = send_message(fmod, pathA, (struct message *) msgA1);
			resp = NULL;

			//CHANGE DST ADDRESS
			struct msg_sender *msgA2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgA2->receiver, inet_ntoa(Adestination->addr),
					sizeof(msgA2->receiver) - 1);
			msgA2->type = SENDER_MSG_CHANGE_RECEIVER;

			resp = send_message(fmod, pathA, (struct message *) msgA2);
			resp = NULL;
		}
	}
}

void BasicRTSPOnlySubsession::deleteStream(unsigned /* clientSessionId */,
		void*& /* streamToken */) {
	if (Vdestination != NULL) {
		if (avType == video || avType == av) {
			char pathV[1024];
			Vdestination = NULL;
			memset(pathV, 0, sizeof(pathV));
			enum module_class path_sender[] = { MODULE_CLASS_SENDER,
					MODULE_CLASS_NONE };
			append_message_path(pathV, sizeof(pathV), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgV1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			msgV1->tx_port = rtp_port;
			msgV1->type = SENDER_MSG_CHANGE_PORT;
			struct response *resp;
			resp = send_message(fmod, pathV, (struct message *) msgV1);
			free_response(resp);

			//CHANGE DST ADDRESS
			struct msg_sender *msgV2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgV2->receiver, "127.0.0.1", sizeof(msgV2->receiver) - 1);
			msgV2->type = SENDER_MSG_CHANGE_RECEIVER;
			resp = send_message(fmod, pathV, (struct message *) msgV2);
			free_response(resp);
		}
	}

	if (Adestination != NULL) {
		if (avType == audio || avType == av) {
			char pathA[1024];
			Adestination = NULL;
			memset(pathA, 0, sizeof(pathA));
			enum module_class path_sender[] = { MODULE_CLASS_AUDIO,
					MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
			append_message_path(pathA, sizeof(pathA), path_sender);

			//CHANGE DST PORT
			struct msg_sender *msgA1 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));

			//TODO: GET AUDIO PORT SET (NOT A COMMON CASE WHEN RTSP IS ENABLED: DEFAULT -> vport + 2)
			msgA1->tx_port = rtp_port_audio;
			msgA1->type = SENDER_MSG_CHANGE_PORT;
			struct response *resp;
                        resp = send_message(fmod, pathA, (struct message *) msgA1);
			free_response(resp);

			//CHANGE DST ADDRESS
			struct msg_sender *msgA2 = (struct msg_sender *) new_message(
					sizeof(struct msg_sender));
			strncpy(msgA2->receiver, "127.0.0.1", sizeof(msgA2->receiver) - 1);
			msgA2->type = SENDER_MSG_CHANGE_RECEIVER;
			resp = send_message(fmod, pathA, (struct message *) msgA2);
			free_response(resp);
		}
	}
}

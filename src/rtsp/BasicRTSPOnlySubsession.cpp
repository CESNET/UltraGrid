#include "rtsp/BasicRTSPOnlySubsession.hh"
#include <BasicUsageEnvironment.hh>
#include <RTSPServer.hh>
#include <GroupsockHelper.hh>

#include "messaging.h"

BasicRTSPOnlySubsession*
BasicRTSPOnlySubsession::createNew(UsageEnvironment& env,
				Boolean reuseFirstSource,
				struct module *mod){
	return new BasicRTSPOnlySubsession(env, reuseFirstSource, mod);
}
 
BasicRTSPOnlySubsession
::BasicRTSPOnlySubsession(UsageEnvironment& env,
				Boolean reuseFirstSource,
				struct module *mod)
  : ServerMediaSubsession(env),
    fSDPLines(NULL), 
    fReuseFirstSource(reuseFirstSource), fLastStreamToken(NULL) {
    destination = NULL;
    gethostname(fCNAME, sizeof fCNAME);
	this->fmod = mod;
	fCNAME[sizeof fCNAME-1] = '\0';
}

BasicRTSPOnlySubsession::~BasicRTSPOnlySubsession() {
	delete[] fSDPLines;
	delete destination;
}

char const* BasicRTSPOnlySubsession::sdpLines() {
	if (fSDPLines == NULL){
		setSDPLines();
	}
	if(destination != NULL) return NULL;
	return fSDPLines;
}

void BasicRTSPOnlySubsession
::setSDPLines() {
	//TODO: should be more dynamic
	unsigned estBitrate = 5000;
	char const* mediaType = "video";
	uint8_t rtpPayloadType = 96;
	AddressString ipAddressStr(fServerAddressForSDP);
	char* rtpmapLine = strdup("a=rtpmap:96 H264/90000\n");
	char const* auxSDPLine = "";

	char const* const sdpFmt =
		"m=%s %u RTP/AVP %u\r\n"
		"c=IN IP4 %s\r\n"
		"b=AS:%u\r\n"
		"%s"
		"a=control:%s\r\n";
	unsigned sdpFmtSize = strlen(sdpFmt)
		+ strlen(mediaType) + 5 /* max short len */ + 3 /* max char len */
		+ strlen(ipAddressStr.val())
		+ 20 /* max int len */
		+ strlen(rtpmapLine)
		+ strlen(trackId());
	char* sdpLines = new char[sdpFmtSize];

	sprintf(sdpLines, sdpFmt,
		mediaType, // m= <media>
		fPortNumForSDP, // m= <port>
		rtpPayloadType, // m= <fmt list>
		ipAddressStr.val(), // c= address
		estBitrate, // b=AS:<bandwidth>
		rtpmapLine, // a=rtpmap:... (if present)
		trackId()); // a=control:<track-id>
	
	fSDPLines = sdpLines;
}

void BasicRTSPOnlySubsession::getStreamParameters(unsigned clientSessionId,
		      netAddressBits clientAddress,
		      Port const& clientRTPPort,
		      Port const& clientRTCPPort,
		      int tcpSocketNum,
		      unsigned char rtpChannelId,
		      unsigned char rtcpChannelId,
		      netAddressBits& destinationAddress,
		      u_int8_t& /*destinationTTL*/,
		      Boolean& isMulticast,
		      Port& serverRTPPort,
		      Port& serverRTCPPort,
		      void*& streamToken) {

    if(destination == NULL){
        if (fSDPLines == NULL){
            setSDPLines();
        }
        if (destinationAddress == 0) {
            destinationAddress = clientAddress;
        }
        struct in_addr destinationAddr;
        destinationAddr.s_addr = destinationAddress;
        destination = new Destinations(destinationAddr, clientRTPPort,clientRTCPPort);
	}
}


void BasicRTSPOnlySubsession::startStream(unsigned clientSessionId,
						void* streamToken,
						TaskFunc* rtcpRRHandler,
						void* rtcpRRHandlerClientData,
						unsigned short& rtpSeqNum,
						unsigned& rtpTimestamp,
						ServerRequestAlternativeByteHandler* serverRequestAlternativeByteHandler,
						void* serverRequestAlternativeByteHandlerClientData) {
	if (destination == NULL){
		return;
	} else {
	    struct response *resp = NULL;
	    char path[1024];
	    memset(path, 0, sizeof(path));
	    enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
	    append_message_path(path, sizeof(path), path_sender);

        //CHANGE DST PORT
        struct msg_sender *msg2 =
            (struct msg_sender *)
            new_message(sizeof(struct msg_sender));
        msg2->port =  ntohs(destination->rtpPort.num());
        msg2->type = SENDER_MSG_CHANGE_PORT;
        resp = send_message(fmod, path, (struct message *) msg2);
        resp = NULL;

        //CHANGE DST ADDRESS
        struct msg_sender *msg1 =
            (struct msg_sender *)
            new_message(sizeof(struct msg_sender));
        strncpy(msg1->receiver, inet_ntoa(destination->addr), sizeof(msg1->receiver) - 1);
        msg1->type = SENDER_MSG_CHANGE_RECEIVER;

        resp = send_message(fmod, path, (struct message *) msg1);
        resp = NULL;
	}
}

void BasicRTSPOnlySubsession::deleteStream(unsigned clientSessionId, void*& streamToken){
    if (destination == NULL){
        return;
    } else {
        destination = NULL;
        char path[1024];
        memset(path, 0, sizeof(path));
        enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
        append_message_path(path, sizeof(path), path_sender);

        //CHANGE DST PORT
        struct msg_sender *msg2 = (struct msg_sender *) new_message(
            sizeof(struct msg_sender));
        msg2->port = 5004;
        msg2->type = SENDER_MSG_CHANGE_PORT;
        send_message(fmod, path, (struct message *) msg2);

        //CHANGE DST ADDRESS
        struct msg_sender *msg1 = (struct msg_sender *) new_message(
            sizeof(struct msg_sender));
        strncpy(msg1->receiver, "127.0.0.1",
            sizeof(msg1->receiver) - 1);
        msg1->type = SENDER_MSG_CHANGE_RECEIVER;
        send_message(fmod, path, (struct message *) msg1);
    }
}

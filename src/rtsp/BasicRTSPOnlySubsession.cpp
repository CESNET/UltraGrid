#include <BasicUsageEnvironment.hh>
#include <RTSPServer.hh>
#include <GroupsockHelper.hh>
#include "rtsp/BasicRTSPOnlySubsession.hh"

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
	//fDestinationsHashTable = HashTable::create(ONE_WORD_HASH_KEYS);
    //Destinations* destination;// = NULL;
    gethostname(fCNAME, sizeof fCNAME);
	this->fmod = mod;
	fCNAME[sizeof fCNAME-1] = '\0'; // just in case
}

BasicRTSPOnlySubsession::~BasicRTSPOnlySubsession() {
  
	delete[] fSDPLines;
//	while (1) {
//		Destinations* destinations
//			= (Destinations*)(fDestinationsHashTable->RemoveNext());
//		if (destinations == NULL) break;
//		delete destinations;
//	}
	//delete fDestinationsHashTable;
	delete destination;
}

char const* BasicRTSPOnlySubsession::sdpLines() {
	if (fSDPLines == NULL){
	    //vir
		setSDPLines();
	}
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

	printf("\n\n[SDP LINES] rtpPayloadType = %d, fPortNumForSDP %d",rtpPayloadType,fPortNumForSDP);

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
	
    if (fSDPLines == NULL){
        //vir
        setSDPLines();
    }

	if (destinationAddress == 0) {
		destinationAddress = clientAddress;
	}
	
	struct in_addr destinationAddr;
	destinationAddr.s_addr = destinationAddress;
	
	//Destinations* destinations;
	if(destination == NULL){
	    destination = new Destinations(destinationAddr, clientRTPPort, clientRTCPPort);
	    printf("\n[getStreamParameters] destinationAddr=%s, clientRTPPort = %d, clientRTCPPort=%d, fPortNumForSDP %d",inet_ntoa(destinationAddr), clientRTPPort, clientRTCPPort,fPortNumForSDP);
	}

	else printf("\n[RTSP Server] no more connections accepted...\n");
	//fDestinationsHashTable->Add((char const*)clientSessionId, destinations);
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
	    printf("\n[RTSP Server] no more connections accepted...\n");
		return;
	} else {
	    struct response *resp = NULL;
	    char path[1024];
	    memset(path, 0, sizeof(path));
	    enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
	    append_message_path(path, sizeof(path), path_sender);

        //CHANGE DST PORT
        printf("\n[RTSP Server] change dst port to %d\n", destination->rtpPort.num());
        struct msg_sender *msg2 =
            (struct msg_sender *)
            new_message(sizeof(struct msg_sender));

        msg2->port = destination->rtpPort.num();
        msg2->type = SENDER_MSG_CHANGE_PORT;
        printf("\n[changing receiver dst] inet_ntoa(msg->receiver) = %s",msg2->receiver);
        resp = send_message(fmod, path, (struct message *) msg2);
        printf("\n[SENDER_MSG_CHANGE_PORT] response--> type = %d, resp = %s",resp->status,resp->text);
        resp = NULL;

        //CHANGE DST ADDRESS
        printf("\n[RTSP Server] change dst address to %s\n",inet_ntoa(destination->addr));
        struct msg_sender *msg1 =
            (struct msg_sender *)
            new_message(sizeof(struct msg_sender));

        strncpy(msg1->receiver, inet_ntoa(destination->addr), sizeof(msg1->receiver) - 1);
        msg1->type = SENDER_MSG_CHANGE_RECEIVER;
        printf("\n[changing receiver dst] inet_ntoa(msg->receiver) = %s",msg1->receiver);

        resp = send_message(fmod, path, (struct message *) msg1);
        printf("\n[SENDER_MSG_CHANGE_RECEIVER] response--> type = %d, resp = %s",resp->status,resp->text);
        resp = NULL;
	}
}

void BasicRTSPOnlySubsession::deleteStream(unsigned clientSessionId, void*& streamToken){
    if (destination == NULL){
        printf("\n[RTSP Server] no more connections accepted...\n");
        return;
    } else {
        char path[1024];
        memset(path, 0, sizeof(path));
        enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
        append_message_path(path, sizeof(path), path_sender);

        //CHANGE DST ADDRESS
        printf("\n[RTSP Server] change dst address to %s\n",
            "127.0.0.1");
        struct msg_sender *msg1 = (struct msg_sender *) new_message(
            sizeof(struct msg_sender));

        strncpy(msg1->receiver, "127.0.0.1",
            sizeof(msg1->receiver) - 1);
        msg1->type = SENDER_MSG_CHANGE_RECEIVER;
        send_message(fmod, path, (struct message *) msg1);

        //CHANGE DST PORT
        printf("\n[RTSP Server] change dst port to %d\n", 23456);
        struct msg_sender *msg2 = (struct msg_sender *) new_message(
            sizeof(struct msg_sender));

        msg2->port = 23456;
        msg2->type = SENDER_MSG_CHANGE_PORT;
        send_message(fmod, path, (struct message *) msg2);

        destination = NULL;
    }
}

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
	
	if (destinationAddress == 0) {
		destinationAddress = clientAddress;
	}
	
	struct in_addr destinationAddr;
	destinationAddr.s_addr = destinationAddress;
	
	//Destinations* destinations;
	
	destination = new Destinations(destinationAddr, clientRTPPort, clientRTCPPort);
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

	Destinations* dst = destination;
		//(Destinations*)(fDestinationsHashTable->Lookup((char const*)clientSessionId));
	if (dst == NULL){
		return;
	} else {
	    //        participant_data_t *participant;
	    //        participant = init_participant(clientSessionId, OUTPUT, inet_ntoa(dst->addr), ntohs(dst->rtpPort.num()));
	    //        add_participant_stream(fStream, participant);
	    char path[1024];
	    memset(path, 0, sizeof(path));
	    enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
	    append_message_path(path, sizeof(path), path_sender);


	    struct msg_sender *msg1 =
	        (struct msg_sender *)
	        new_message(sizeof(struct msg_sender));

	    strncpy(msg1->receiver, inet_ntoa(dst->addr)/*suffix(message, "receiver ")*/, sizeof(msg1->receiver) - 1);
	    msg1->type = SENDER_MSG_CHANGE_RECEIVER;
	    send_message(fmod, path, (struct message *) msg1);


	    struct msg_sender *msg2 =
	        (struct msg_sender *)
	        new_message(sizeof(struct msg_sender));

	    msg2->port = dst->rtpPort.num();//atoi(suffix(message, "sender-port "));
	    msg2->type = SENDER_MSG_CHANGE_PORT;
	    send_message(fmod, path, (struct message *) msg2);
	}
}

void BasicRTSPOnlySubsession::deleteStream(unsigned clientSessionId, void*& streamToken){
    Destinations* dst = destination;
        //(Destinations*)(fDestinationsHashTable->Lookup((char const*)clientSessionId));
    if (dst == NULL){
        return;
    } else {
        //remove_participant(fStream->plist, clientSessionId);
    }
}

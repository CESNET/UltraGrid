#ifndef _BASIC_RTSP_ONLY_SUBSESSION_HH
#define _BASIC_RTSP_ONLY_SUBSESSION_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include <ServerMediaSession.hh>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "module.h"
#include "control_socket.h"

#ifdef __cplusplus
}
#endif

// #ifndef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH
// #include <OnDemandServerMediaSubsession.hh>
// #endif

class Destinations {
public:
  Destinations(struct in_addr const& destAddr,
               Port const& rtpDestPort,
               Port const& rtcpDestPort)
    : isTCP(False), addr(destAddr), rtpPort(rtpDestPort), rtcpPort(rtcpDestPort) {
  }
  Destinations(int tcpSockNum, unsigned char rtpChanId, unsigned char rtcpChanId)
    : isTCP(True), rtpPort(0) /*dummy*/, rtcpPort(0) /*dummy*/,
      tcpSocketNum(tcpSockNum), rtpChannelId(rtpChanId), rtcpChannelId(rtcpChanId) {
  }

public:
  Boolean isTCP;
  struct in_addr addr;
  Port rtpPort;
  Port rtcpPort;
  int tcpSocketNum;
  unsigned char rtpChannelId, rtcpChannelId;
};

class BasicRTSPOnlySubsession: public ServerMediaSubsession {
	
public:
	static BasicRTSPOnlySubsession*
		createNew(UsageEnvironment& env,
			Boolean reuseFirstSource,
			struct module *mod);

protected:
	
	BasicRTSPOnlySubsession(UsageEnvironment& env, Boolean reuseFirstSource,
	    struct module *mod);
	
	virtual ~BasicRTSPOnlySubsession();	
	
	virtual char const* sdpLines();
	
	virtual void getStreamParameters(unsigned clientSessionId,
								  netAddressBits clientAddress,
								  Port const& clientRTPPort,
								  Port const& clientRTCPPort,
								  int tcpSocketNum,
								  unsigned char rtpChannelId,
								  unsigned char rtcpChannelId,
								  netAddressBits& destinationAddress,
								  u_int8_t& destinationTTL,
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

protected:
	
	char* fSDPLines;
	Destinations* destination;
	
private:
	
	void setSDPLines();
	
	Boolean fReuseFirstSource;
	void* fLastStreamToken;
	char fCNAME[100];
	struct module *fmod;
};


#endif

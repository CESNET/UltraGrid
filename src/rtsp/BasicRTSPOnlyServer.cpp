/*
 * FILE:    rtsp/BasicRTSPOnlyServer.cpp
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

#include <GroupsockHelper.hh> // for "weHaveAnIPv*Address()"

#include "rtsp/BasicRTSPOnlyServer.hh"
#include "rtsp/BasicRTSPOnlySubsession.hh"

BasicRTSPOnlyServer *BasicRTSPOnlyServer::srvInstance = NULL;

BasicRTSPOnlyServer::BasicRTSPOnlyServer(int port, struct module *mod, rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio){
    if(mod == NULL){
        exit(1);
    }
    this->fPort = port;
    this->mod = mod;
    this->avType = avType;
    this->audio_codec = audio_codec;
    this->audio_sample_rate = audio_sample_rate;
    this->audio_channels = audio_channels;
    this->audio_bps = audio_bps;
    this->rtp_port = rtp_port;
    this->rtp_port_audio = rtp_port_audio;
    this->rtspServer = NULL;
    this->env = NULL;
    this->srvInstance = this;
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::initInstance(int port, struct module *mod, rtps_types_t avType, audio_codec_t audio_codec, int audio_sample_rate, int audio_channels, int audio_bps, int rtp_port, int rtp_port_audio){
    if (srvInstance != NULL){
        return srvInstance;
    }
    return new BasicRTSPOnlyServer(port, mod, avType, audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio);
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::getInstance(){
    if (srvInstance != NULL){
        return srvInstance;
    }
    return NULL;
}

int BasicRTSPOnlyServer::init_server() {
    
    if (env != NULL || rtspServer != NULL || mod == NULL || (avType >= NUM_RTSP_FORMATS && avType < 0)){
        exit(1);
    }

    //setting livenessTimeoutTask
    unsigned reclamationTestSeconds = 60;

    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);

    UserAuthenticationDatabase* authDB = NULL;
 #ifdef ACCESS_CONTROL
   // To implement client access control to the RTSP server, do the following:
   authDB = new UserAuthenticationDatabase;
   authDB->addUserRecord("i2cat", "ultragrid"); // replace these with real strings
   // Repeat the above with each <username>, <password> that you wish to allow
   // access to the server.
 #endif

    if (fPort == 0){
        fPort = 8554;
    }

    rtspServer = RTSPServer::createNew(*env, fPort, authDB, reclamationTestSeconds);
    if (rtspServer == NULL) {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
        exit(1);
    }
    rtspServer->disableStreamingRTPOverTCP();
    ServerMediaSession* sms;
               sms = ServerMediaSession::createNew(*env, "ultragrid",
                   "UltraGrid RTSP server enabling standard transport",
                   "UltraGrid RTSP server");

               if(avType == av){
                                  sms->addSubsession(BasicRTSPOnlySubsession
                                                    ::createNew(*env, True, mod, audio, audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio));
                                  sms->addSubsession(BasicRTSPOnlySubsession
                                                    ::createNew(*env, True, mod, video, audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio));
               }else if(avType == audio){
                   sms->addSubsession(BasicRTSPOnlySubsession
                                     	 ::createNew(*env, True, mod, audio, audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio));

               }else if(avType == video){
            	   sms->addSubsession(BasicRTSPOnlySubsession
            			   	   	    	 ::createNew(*env, True, mod, video, audio_codec, audio_sample_rate, audio_channels, audio_bps, rtp_port, rtp_port_audio));
               }else{
            	   *env << "\n[RTSP Server] Error when trying to play stream type: \"" << avType << "\"\n";
            	   exit(1);
               }

               rtspServer->addServerMediaSession(sms);

               *env << "\n";
               if (weHaveAnIPv4Address(*env)) {
                   char* url = rtspServer->ipv4rtspURL(sms);
                   *env << "[RTSP Server] Play this stream using the URL \"" << url << "\"\n";
                   delete[] url;
               }
               if (weHaveAnIPv6Address(*env)) {
                   char* url = rtspServer->ipv6rtspURL(sms);
                   *env << "[RTSP Server] Play this stream using the URL \"" << url << "\"\n";
                   delete[] url;
               }

    return 0;
}

void *BasicRTSPOnlyServer::start_server(void *args){
    char* watch = (char*) args;
    BasicRTSPOnlyServer* instance = getInstance();
    
	if (instance == NULL || instance->env == NULL || instance->rtspServer == NULL){
		return NULL;
	}

	instance->env->taskScheduler().doEventLoop(watch); 

    Medium::close(instance->rtspServer);
    delete &instance->env->taskScheduler();
    instance->env->reclaim();

	return NULL;
}

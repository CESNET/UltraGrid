/*
 * FILE:    rtsp/BasicRTSPOnlyServer.cpp
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2014-2025 CESNET
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
#include <cassert>

#include "rtsp/BasicRTSPOnlyServer.hh"
#include "rtsp/BasicRTSPOnlySubsession.hh"
#include "rtsp/rtsp_utils.h"

BasicRTSPOnlyServer *BasicRTSPOnlyServer::srvInstance = NULL;

BasicRTSPOnlyServer::BasicRTSPOnlyServer(struct rtsp_server_parameters params) {
    if (params.parent == NULL) {
        exit(1);
    }
    this->params = params;
    this->rtspServer = NULL;
    this->env = NULL;
    this->srvInstance = this;
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::initInstance(struct rtsp_server_parameters params)
{
    if (srvInstance != NULL){
        return srvInstance;
    }
    return new BasicRTSPOnlyServer(params);
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::getInstance(){
    if (srvInstance != NULL){
        return srvInstance;
    }
    return NULL;
}

int
BasicRTSPOnlyServer::init_server()
{
    assert(env == nullptr && rtspServer == nullptr);
    assert(params.parent != nullptr);
    assert(params.avType > rtsp_type_none && params.avType <= rtsp_av_type_both);

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

    if (params.rtsp_port == 0) {
        params.rtsp_port = 8554;
    }

    rtspServer = RTSPServer::createNew(*env, params.rtsp_port, authDB,
                                       reclamationTestSeconds);
    if (rtspServer == NULL) {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
        exit(1);
    }
    rtspServer->disableStreamingRTPOverTCP();
    ServerMediaSession* sms;
               sms = ServerMediaSession::createNew(*env, "ultragrid",
                   "UltraGrid RTSP server enabling standard transport",
                   "UltraGrid RTSP server");

    if ((params.avType & rtsp_type_audio) != 0) {
        sms->addSubsession(BasicRTSPOnlySubsession ::createNew(
            *env, True, rtsp_type_audio, params.rtp_port_audio, params));
    }
    if ((params.avType & rtsp_type_video) != 0) {
        sms->addSubsession(BasicRTSPOnlySubsession ::createNew(
            *env, True, rtsp_type_video, params.rtp_port_video, params));
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
        auto                *watch    = (EventLoopWatchVariable *) args;
        BasicRTSPOnlyServer *instance = getInstance();

	if (instance == NULL || instance->env == NULL || instance->rtspServer == NULL){
		return NULL;
	}

	instance->env->taskScheduler().doEventLoop(watch); 

    Medium::close(instance->rtspServer);
    delete &instance->env->taskScheduler();
    instance->env->reclaim();

	return NULL;
}

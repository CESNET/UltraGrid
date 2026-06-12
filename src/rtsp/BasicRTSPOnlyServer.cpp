/*
 * FILE:    rtsp/BasicRTSPOnlyServer.cpp
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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

#include <BasicUsageEnvironment.hh>
#include <BasicUsageEnvironment_version.hh> // for BASICUSAGEENVIRONMENT_LI...
#include <GroupsockHelper.hh>               // for "weHaveAnIPv*Address()"
#include <RTSPServer.hh>
#include <UsageEnvironment.hh>              // for UsageEnvironment, TaskSc...
#include <cassert>
#include <pthread.h> // for pthread_create, pthread_...

#include "host.h" // for exit_uv
#include "rtsp/BasicRTSPOnlyServer.hh"
#include "rtsp/BasicRTSPOnlySubsession.hh"
#include "rtsp/rtsp_utils.h"

// compat
#if BASICUSAGEENVIRONMENT_LIBRARY_VERSION_INT < 1752883200
typedef char volatile EventLoopWatchVariable;
#endif

struct BasicRTSPOnlyServer {
      public:
        int init_server(struct rtsp_server_parameters params);
        static void *server_worker(void *args);

        pthread_t              server_th;
        EventLoopWatchVariable watch = 0;

      private:
        RTSPServer       *rtspServer = nullptr;
        UsageEnvironment *env        = nullptr;
};

int
BasicRTSPOnlyServer::init_server(struct rtsp_server_parameters params)
{
    assert(env == nullptr && rtspServer == nullptr);
    assert(params.parent != nullptr);
    assert(params.avType > rtsp_type_none && params.avType <= rtsp_av_type_both);

    //setting livenessTimeoutTask
    unsigned reclamationTestSeconds = 60;

    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);

    UserAuthenticationDatabase* authDB = nullptr;
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
    if (rtspServer == nullptr) {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
        return -1;
    }
    rtspServer->disableStreamingRTPOverTCP();
    ServerMediaSession* sms;
               sms = ServerMediaSession::createNew(*env, "ultragrid",
                   "UltraGrid RTSP server enabling standard transport",
                   "UltraGrid RTSP server");

    if ((params.avType & rtsp_type_audio) != 0) {
        sms->addSubsession(BasicRTSPOnlySubsession ::createNew(
            *env, True, rtsp_type_audio, params.rtp_audio_src_port, params));
    }
    if ((params.avType & rtsp_type_video) != 0) {
        sms->addSubsession(BasicRTSPOnlySubsession ::createNew(
            *env, True, rtsp_type_video, params.rtp_video_src_port, params));
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

void *
BasicRTSPOnlyServer::server_worker(void *args)
{
        auto *srv = (BasicRTSPOnlyServer *) args;

        srv->env->taskScheduler().doEventLoop(&srv->watch);

        Medium::close(srv->rtspServer);
        delete &srv->env->taskScheduler();
        srv->env->reclaim();

        return nullptr;
}

BasicRTSPOnlyServer*
start_rtsp_server(struct rtsp_server_parameters rtsp_params)
{
        auto *srv = new BasicRTSPOnlyServer();
        if (srv->init_server(rtsp_params) == -1) {
                delete srv;
                exit_uv(1);
                return nullptr;
        }
        int ret = pthread_create(&srv->server_th, nullptr,
                                 BasicRTSPOnlyServer::server_worker,
                                 (void *) srv);
        assert(ret == 0);
        return srv;
}

/**
 * stops + joins the RTSP server state and deletes the state
 */
void
stop_rtsp_server(struct BasicRTSPOnlyServer *srv)
{
        if (srv == nullptr) {
                return;
        }
        srv->watch = 1;
        pthread_join(srv->server_th, nullptr);

        delete srv;
}

/*
 * FILE:    rtsp/UltragridRTSPServer.cpp
 * AUTHORS: David Cassany    <david.cassany@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec     <pulec@cesnet.cz>
 *          Jakub Kováč      <xkovac5@mail.muni.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2010-2023 CESNET, z. s. p. o.
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
/**
 * @note This code was created as a set of steps on file from Live555 library testProgs/testOnDemandRTSPServer.cpp
 *       Original file licenece is posted below
 * @note This file also contains function announceURL what was copied from Live555 library testProgs/announceURL.cpp
 *       announceURL.cpp licence is the same as in testOnDemandRTSPServer.cpp
*/
/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// Copyright (c) 1996-2023, Live Networks, Inc.  All rights reserved

#include "rtsp/UltragridRTSPServer.hh"
#include "rtsp/UltragridRTSPSubsession.hh"
#include <GroupsockHelper.hh> // for "weHaveAnIPv*Address()"

UltragridRTSPServer::UltragridRTSPServer(unsigned int rtsp_port, struct module* mod, rtsp_media_type_t media_type, audio_codec_t audio_codec,
        int audio_sample_rate, int audio_channels, int audio_bps, int rtp_video_port, int rtp_audio_port) {
    if(mod == NULL)
        throw std::system_error();

    // Begin by setting up our usage environment:
    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);

    UserAuthenticationDatabase* authDB = NULL;
    #ifdef ACCESS_CONTROL
        // To implement client access control to the RTSP server, do the following:
        authDB = new UserAuthenticationDatabase;
        authDB->addUserRecord("username1", "password1"); // replace these with real strings
        // Repeat the above with each <username>, <password> that you wish to allow
        // access to the server.
    #endif

    if (rtsp_port == 0)
        rtsp_port = 8554; // default port number

    // only trying specified port because using different port than expected could lead to issues
    rtspServer = RTSPServer::createNew(*env, rtsp_port, authDB);
    if (rtspServer == NULL) {
        *env << "[RTSP Server] Error: Failed to create RTSP server: " << env->getResultMsg() << "\n";
        throw std::system_error();
    }

    ServerMediaSession* sms = ServerMediaSession::createNew(*env, "ultragrid", "UltraGrid RTSP server enabling standard transport", "UltraGrid RTSP server");

    if (media_type == video || media_type == av)
        sms->addSubsession(UltragridRTSPVideoSubsession::createNew(*env, mod, rtp_video_port));
    if (media_type == audio || media_type == av)
        sms->addSubsession(UltragridRTSPAudioSubsession::createNew(*env, mod, rtp_audio_port, audio_sample_rate, audio_channels, audio_codec));

    if (media_type == none) {
        *env << "\n[RTSP Server] Error: No media type selected: \"none\"\n";
        throw std::system_error();
    }
    if (media_type < 0 || media_type >= NUM_RTSP_MEDIA_TYPES) {
        *env << "\n[RTSP Server] Error: Incompatible media type for subsession: \"" << media_type << "\"\n";
        throw std::system_error();
    }

    rtspServer->addServerMediaSession(sms);
    announceURL(rtspServer, sms);
}

UltragridRTSPServer::~UltragridRTSPServer() {
    if (rtspServer != NULL) {
        Medium::close(rtspServer);
    }
    if (env != NULL) {
        delete &env->taskScheduler();
        env->reclaim();
    }
}

void UltragridRTSPServer::serverRunner(char* serverStopFlag) {
        env->taskScheduler().doEventLoop(serverStopFlag);
}

// copied from Live555 library live555/testProgs/announceURL.cpp, published under LGPL3 licence
void UltragridRTSPServer::announceURL(RTSPServer* rtspServer, ServerMediaSession* sms) {
  if (rtspServer == NULL || sms == NULL) return; // sanity check

  UsageEnvironment& env = rtspServer->envir();

  env << "Play this stream using the URL ";
  if (weHaveAnIPv4Address(env)) {
    char* url = rtspServer->ipv4rtspURL(sms);
    env << "\"" << url << "\"";
    delete[] url;
    if (weHaveAnIPv6Address(env)) env << " or ";
  }
  if (weHaveAnIPv6Address(env)) {
    char* url = rtspServer->ipv6rtspURL(sms);
    env << "\"" << url << "\"";
    delete[] url;
  }
  env << "\n";
}

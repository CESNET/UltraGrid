/*
 * FILE:    c_basicRTSPOnlyServer.cpp
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
#include "rtsp/c_basicRTSPOnlyServer.h"
#include "rtsp/BasicRTSPOnlyServer.hh"

int c_start_server(rtsp_serv_t* server){
    int ret;
    BasicRTSPOnlyServer *srv = BasicRTSPOnlyServer::initInstance(server->port, server->mod, server->avType);
    srv->init_server();
    ret = pthread_create(&server->server_th, NULL, BasicRTSPOnlyServer::start_server, &server->watch);
    if (ret == 0){
        server->run = TRUE;
    } else {
        server->run = FALSE;
    }
    return ret;
}

rtsp_serv_t *init_rtsp_server(unsigned int port, struct module *mod, uint8_t avType){
    rtsp_serv_t *server = (rtsp_serv_t*) malloc(sizeof(rtsp_serv_t));
    server->port = port;
    server->mod = mod;
    server->watch = 0;
    server->run = FALSE;
    server->avType = avType;
    return server;
}

void c_stop_server(rtsp_serv_t* server){
    server->watch = 1;
    if (server->run){
        pthread_join(server->server_th, NULL);
    }
}

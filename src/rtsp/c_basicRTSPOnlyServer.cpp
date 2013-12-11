#include "rtsp/c_basicRTSPOnlyServer.h"

#include "rtsp/BasicRTSPOnlyServer.hh"

int c_start_server(rtsp_serv_t* server){
    int ret;
    BasicRTSPOnlyServer *srv = BasicRTSPOnlyServer::initInstance(server->port, server->mod);
    srv->init_server();
    ret = pthread_create(&server->server_th, NULL, BasicRTSPOnlyServer::start_server, &server->watch);
    if (ret == 0){
        server->run = TRUE;
    } else {
        server->run = FALSE;
    }
    return ret;
}

rtsp_serv_t *init_rtsp_server(uint port, struct module *mod){//, transmitter_t *transmitter){
    rtsp_serv_t *server = (rtsp_serv_t*) malloc(sizeof(rtsp_serv_t));
    server->port = port;
    //server->transmitter = transmitter;
    server->mod = mod;
    server->watch = 0;
    server->run = FALSE;
    return server;
}

void c_stop_server(rtsp_serv_t* server){
    server->watch = 1;
    if (server->run){
        pthread_join(server->server_th, NULL);
    }
}

//int c_update_server(rtsp_serv_t* server){
//    BasicRTSPOnlyServer *srv = BasicRTSPOnlyServer::getInstance();
//    if (srv == NULL){
//        exit(1);
//    }
//    return srv->update_server();
//}

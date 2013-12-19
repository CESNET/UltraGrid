#ifndef _BASIC_RTSP_ONLY_SERVER_HH
#define _BASIC_RTSP_ONLY_SERVER_HH

#include <RTSPServer.hh>
#include <BasicUsageEnvironment.hh>
#include "module.h"



class BasicRTSPOnlyServer {
private:
    BasicRTSPOnlyServer(int port, struct module *mod);
    
public:
    static BasicRTSPOnlyServer* initInstance(int port, struct module *mod);
    static BasicRTSPOnlyServer* getInstance();
    
    int init_server();

    static void *start_server(void *args);
    
    int update_server();
    
private:
    
    static BasicRTSPOnlyServer* srvInstance;
    int fPort;
    struct module *mod;
    RTSPServer* rtspServer;
    UsageEnvironment* env;
};

#endif

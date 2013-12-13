#include "rtsp/BasicRTSPOnlyServer.hh"
#include "rtsp/BasicRTSPOnlySubsession.hh"

BasicRTSPOnlyServer *BasicRTSPOnlyServer::srvInstance = NULL;

BasicRTSPOnlyServer::BasicRTSPOnlyServer(int port, struct module *mod){
    if(mod == NULL){
        exit(1);
    }
    
    this->fPort = port;
    //this->fTransmitter = transmitter;
    this->mod = mod;
    this->rtspServer = NULL;
    this->env = NULL;
    this->srvInstance = this;
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::initInstance(int port, struct module *mod){
    if (srvInstance != NULL){
        return srvInstance;
    }
    return new BasicRTSPOnlyServer(port, mod);
}

BasicRTSPOnlyServer* 
BasicRTSPOnlyServer::getInstance(){
    if (srvInstance != NULL){
        return srvInstance;
    }
    return NULL;
}

int BasicRTSPOnlyServer::init_server() {
    
    if (env != NULL || rtspServer != NULL || mod == NULL){
        exit(1);
    }
    
    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);

    UserAuthenticationDatabase* authDB = NULL;   
// #ifdef ACCESS_CONTROL
//   // To implement client access control to the RTSP server, do the following:
//   authDB = new UserAuthenticationDatabase;
//   authDB->addUserRecord("username1", "password1"); // replace these with real strings
//   // Repeat the above with each <username>, <password> that you wish to allow
//   // access to the server.
// #endif

    if (fPort == 0){
        fPort = 8554;
    }
  
    printf("RTSP Server port = %d",fPort);
    rtspServer = RTSPServer::createNew(*env, fPort, authDB);
    if (rtspServer == NULL) {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
        exit(1);
    }
    ServerMediaSession* sms;
               sms = ServerMediaSession::createNew(*env, "i2cat_rocks",
                   "i2cat_rocks",
                   "i2cat_rocks");

               sms->addSubsession(BasicRTSPOnlySubsession
                  ::createNew(*env, True, mod));
               rtspServer->addServerMediaSession(sms);

               char* url = rtspServer->rtspURL(sms);
               *env << "\nPlay this stream using the URL \"" << url << "\"\n";
               delete[] url;

    return 0;
}

void *BasicRTSPOnlyServer::start_server(void *args){
    char* watch = (char*) args;
    BasicRTSPOnlyServer* instance = getInstance();
    
	if (instance == NULL || instance->env == NULL || instance->rtspServer == NULL){
		return NULL;
	}

    printf("\n[RTSP Server] starting server thread....");

	instance->env->taskScheduler().doEventLoop(watch); 

    printf("\n[RTSP Server] stopping server thread....");

    Medium::close(instance->rtspServer);
    delete &instance->env->taskScheduler();
    instance->env->reclaim();

	return NULL;
}

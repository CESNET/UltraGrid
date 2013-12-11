#ifndef C_BASIC_RTSP_ONLY_SERVER_H
#define C_BASIC_RTSP_ONLY_SERVER_H
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <pthread.h>
#include <semaphore.h>
//#include "video.h"
#include "control_socket.h"

#include "module.h"

#include "debug.h"

//#include "transmitter.h"

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC typedef struct rtsp_serv {
	uint port;
	//transmitter_t* transmitter;
	//struct control_state *ug_control;
	struct module *mod;
	pthread_t server_th;
    uint8_t watch;
    uint8_t run;
} rtsp_serv_t;

EXTERNC int c_start_server(rtsp_serv_t* server);

EXTERNC void c_stop_server(rtsp_serv_t* server);

EXTERNC rtsp_serv_t* init_rtsp_server(uint port, struct module *mod);//, transmitter_t *transmitter);

#undef EXTERNC


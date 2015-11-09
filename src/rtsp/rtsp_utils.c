#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "rtsp_utils.h"

int get_rtsp_server_port(const char *cconfig){
        int port;
        char *tok;
        char *save_ptr = NULL;
        char *config = strdup(cconfig);
        if(strcmp((strtok_r(config, ":", &save_ptr)),"port") == 0){
                if ((tok = strtok_r(NULL, ":", &save_ptr))) {
                        port = atoi(tok);
                        if (!(port >= 0 && port <= 65535)) {
                                printf("\n[RTSP SERVER] ERROR - please, enter a valid port number.\n");
                                rtps_server_usage();
                                free(config);
                                return -1;
                        } else return port;
                } else {
                        printf("\n[RTSP SERVER] ERROR - please, enter a port number.\n");
                        rtps_server_usage();
                        free(config);
                        return -1;
                }
        } else {
                printf("\n[RTSP SERVER] ERROR - please, check usage.\n");
                rtps_server_usage();
                free(config);
                return -1;
        }
}

void rtps_server_usage(){
        printf("\n[RTSP SERVER] usage:\n");
        printf("\t--rtsp-server[=port:number]\n");
        printf("\t\tdefault rtsp server port number: 8554\n\n");
}


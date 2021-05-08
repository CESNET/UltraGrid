#include <unistd.h>

#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libug.h>

static bool exit_requested = false;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

static void signal_handler(int signal) {
        char msg[] = "Signal XXX caught... Exiting\n";
        char *signum = strstr(msg, "XXX"); // allowed in POSIX.1-2016
        signum[0] = '0' + ((signal / 100) % 10);
        signum[1] = '0' + ((signal / 10) % 10);
        signum[2] = '0' + (signal % 10);
        write(2, msg, sizeof msg - 1);
        pthread_mutex_lock(&lock);
        exit_requested = true;
        pthread_mutex_unlock(&lock);
        pthread_cond_signal(&cv);
}

static void usage(const char *progname) {
        printf("%s [options] [sender[:port]]\n", progname);
        printf("options:\n");
        printf("\t-c I420|RGBA|CUDA_I420|CUDA_RGBA - force decompress to codec\n");
        printf("\t-C - connection count\n");
        printf("\t-d - display (default vrg)\n");
        printf("\t-h - show this help\n");
        printf("\t-p - port\n");
        printf("\t-S - enable strips\n");
        printf("\t-v - increase verbosity (use twice for debug)\n");
}

int main(int argc, char *argv[]) {
        struct ug_receiver_parameters init_params = { 0 };

        int ch = 0;
        while ((ch = getopt(argc, argv, "c:C:hd:p:Sv")) != -1) {
                switch (ch) {
                case 'c':
                        assert(strcmp(optarg, "RGBA") == 0 || strcmp(optarg, "I420") == 0
                                        || strcmp(optarg, "CUDA_RGBA") == 0 || strcmp(optarg, "CUDA_I420") == 0);
                        if (strcmp(optarg, "RGBA") == 0) {
                                init_params.decompress_to = UG_RGBA;
                        } else if (strcmp(optarg, "I420") == 0) {
                                init_params.decompress_to =  UG_I420;
                        } else if (strcmp(optarg, "CUDA_RGBA") == 0) {
                                init_params.decompress_to =  UG_CUDA_RGBA;
                        } else if (strcmp(optarg, "CUDA_I420") == 0) {
                                init_params.decompress_to =  UG_CUDA_I420;
                        }
                        break;
                case 'C':
                        init_params.connections = atoi(optarg);
                        break;
                case 'd':
                        init_params.display = optarg;
                        break;
                case 'h':
                        usage(argv[0]);
                        return 0;
                case 'p':
                        init_params.port = atoi(optarg);
                        break;
                case 'S':
                        init_params.enable_strips = 1;
                        break;
                case 'v':
                        init_params.verbose += 1;
                        break;
                default:
                        usage(argv[0]);
                        return 1;
                }
        }

        argc -= optind;
        argv += optind;

        if (argc > 0) {
                init_params.sender = argv[0];
                if (strchr(argv[0], ':') != NULL) {
                        char *port_str = strchr(argv[0], ':') + 1;
                        *strchr(argv[0], ':') = '\0';
                        init_params.port = atoi(port_str);
                }
        }

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        struct ug_receiver *s = ug_receiver_start(&init_params);
        if (!s) {
                return 2;
        }
        pthread_mutex_lock(&lock);
        while (!exit_requested) {
                pthread_cond_wait(&cv, &lock);
        }
        ug_receiver_done(s);
        printf("Exit\n");
}


#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libug.h>

static void usage(const char *progname) {
        printf("%s [options] [sender[:port]]\n", progname);
        printf("options:\n");
        printf("\t-h - show this help\n");
        printf("\t-d - display (default vrg)\n");
}

int main(int argc, char *argv[]) {
        struct ug_receiver_parameters init_params = { 0 };

        int ch = 0;
        while ((ch = getopt(argc, argv, "d:h")) != -1) {
                switch (ch) {
                case 'd':
                        init_params.display = optarg;
                        break;
                case 'h':
                        usage(argv[0]);
                        return 0;
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
                        *strchr(argv[1], ':') = '\0';
                        init_params.rx_port =
                                init_params.tx_port = atoi(port_str);
                }
        }

        struct ug_receiver *s = ug_receiver_start(&init_params);
        if (!s) {
                return 2;
        }
        sleep(600);
        ug_receiver_done(s);
}


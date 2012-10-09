#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#ifdef HAVE_NCURSES

#include <curses.h>

#include <stdlib.h>
#include <stdio.h>
#include "debug.h"

void usage(const char *progname);
int main(int argc, char *argv[]);
void sig_handler(int signal);

void usage(const char *progname) {
        printf("Usage: %s <hostname> <port>\n", progname);
}

int fd = -1;// = socket(AF_INET6, SOCK_STREAM, 0);

void sig_handler(int signal) {
        UNUSED(signal);
        close(fd);
        endwin();
        exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{
        if(argc != 3) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        signal(SIGINT, sig_handler);
        signal(SIGPIPE, sig_handler);

        const char *hostname = argv[1];
        uint16_t port = atoi(argv[2]);

        struct addrinfo hints, *res, *res0;
        int err;

        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        const char *port_str = argv[2];
        err = getaddrinfo(hostname, port_str, &hints, &res0);

        if(err) {
                fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(err));
                return EXIT_FAILURE;
        }

        char what[1024];

        for (res = res0; res; res = res->ai_next) {
                fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
                if  (fd < 0) {
                        snprintf(what, 1024, "socket failed: %s", strerror(errno));
                        continue;
                }

                if(connect(fd, res->ai_addr, res->ai_addrlen) == -1) {
                        fd = -1;

                        snprintf(what, 1024, "connect failed: %s:%d :%s",  hostname, port, strerror(errno));
                        continue;
                }

                break; /* okay we got one */
        }

        freeaddrinfo(res0);

        if(fd < 0 ) {
                fprintf(stderr, "%s\n", what);
                return EXIT_FAILURE;
        }

        initscr();
        keypad(stdscr, TRUE);
        scrollok(stdscr, TRUE);

        while(1) {
                char message[1024] = { '\0' };
                int key = getch();
                switch(key) {
                        case KEY_UP: // UP
                        case KEY_DOWN: // DOWN
                        case KEY_RIGHT: // RIGHT
                        case KEY_LEFT: // LEFT
                                {
                                        int offset = 10; // sec
                                        int direction = 1; // forward
                                        if(key == KEY_UP || key == KEY_DOWN) {
                                                offset = 60;
                                        }
                                        if(key == KEY_LEFT || key == KEY_DOWN) {
                                                direction = -1;
                                        }
                                        snprintf(message, 1024, "SEEK %c%ds", 
                                                        direction == 1 ? '+' : '-',
                                                        offset);
                                        break;
                                }
                        case 0x20: // space
                                snprintf(message, 1024, "PAUSE");
                                break;
                        case 0x71: // 'q'
                                snprintf(message, 1024, "QUIT");
                                break;
                        default:
                                printw("Unknown key: %d\n", key);
                }

                if(strlen(message) > 0) {
                        printw("Sent message: \"%s\"\n", message);
                        ssize_t total_written = 0;
                        do {
                                ssize_t ret = write(fd, message, strlen(message)+1 - total_written);
                                if(ret <= 0) {
                                        perror("Error sending command");
                                        goto finish;
                                }
                                total_written += ret;
                        } while(total_written < (int) strlen(message) + 1);
                }
        }

finish:
        close(fd);
        endwin();

        return EXIT_SUCCESS;
}

#else // ! HAVE_NCURSES
#include <stdio.h>
int main () { fprintf(stderr, "Recompile with ncurses support!\n"); return 1; }

#endif // HAVE_NCURSES


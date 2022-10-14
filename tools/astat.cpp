#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_unix.h"
#include "config_win32.h"
#include "compat/platform_pipe.h"

#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <mutex>
#include <algorithm>

#include "astat.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif

using namespace std;

static void worker(ug_connection &);

static ssize_t write_all(fd_t fd, const void *buf, size_t count)
{
    char *p = (char *) buf;
    size_t rest = count;
    ssize_t w = 0;

    while (rest > 0 && (w = send(fd, p, rest, MSG_NOSIGNAL)) >= 0) {
        p += w;
        rest -= w;
    }

    if (rest > 0)
        return w;
    else
        return count;
}

struct ug_connection {
        fd_t fd;
        ug_connection(int f, fd_t sef[]) : fd(f), should_exit_fd{sef[0], sef[1]}, t(worker, ref(*this)) {}
        double volpeak[2] = {-INFINITY, -INFINITY};
        double volrms[2] = {0, 0};
        int sample_count = 0;
        mutex lock;

        volatile bool should_exit = false;
        fd_t should_exit_fd[2];
        thread t;

        bool connection_lost = false;
};

bool astat_parse_line(const char *str, double volpeak[2], double volrms[2]){
        int ret = sscanf(str, "stats ARECV volrms0 %lf volpeak0 %lf volrms1 %lf volpeak1 %lf",
                        &volrms[0], &volpeak[0], &volrms[1], &volpeak[1]);

        return ret == 4;
}

// Line format example:
// stats ARECV volrms0 -18.0004 volpeak0 -14.9897 volrms1 -18.0004 volpeak1 -14.9897"
static void parse_and_store(ug_connection &c, const char *str)
{
        double volpeak[2];
        double volrms[2];
        if (!astat_parse_line(str, volpeak, volrms)) {
#ifdef ASTAT_DEBUG
                fprintf(stderr, "Wrong line format!");
#endif
                return;
        }

        lock_guard<mutex> lk(c.lock);
        for (int i = 0; i < 2; i++) {
                c.volpeak[i] = std::max(c.volpeak[i], volpeak[i]);
                c.volrms[i] = (volrms[i] * c.sample_count + volrms[i]) / (c.sample_count + 1);
        }
        c.sample_count += 1;
}

static void worker(ug_connection &c)
{
        char line[10000] = "";
        while (!c.should_exit) {
                char buf[9000] = "";

                fd_set fds;
                FD_ZERO(&fds);
                FD_SET(c.fd, &fds);
                FD_SET(c.should_exit_fd[0], &fds);
                fd_t nfds = std::max<fd_t>(c.fd, c.should_exit_fd[0]) + 1;

                int rc = select(nfds, &fds, NULL, NULL, NULL);
                if (rc <= 0) {
                        continue;
                }

                if (FD_ISSET(c.should_exit_fd[0], &fds)) {
                        break;
                }

                ssize_t ret = recv(c.fd, buf, sizeof buf, 0);
                if (ret == 0) { // connection was closed
                        lock_guard<mutex> lk(c.lock);
                        c.connection_lost = true;
#ifdef ASTAT_DEBUG
                        fprintf(stderr, "Connection lost!\n");
#endif
                        break;
                }

                if (ret < 0) {
#ifdef _WIN32
                        int err = WSAGetLastError();
                        if(err == WSAECONNRESET){
                                lock_guard<mutex> lk(c.lock);
                                c.connection_lost = true;
#ifdef ASTAT_DEBUG
                                fprintf(stderr, "Connection lost!\n");
#endif
                                break;
                        }
                        printf("recv: %d \n", WSAGetLastError());
#else
                        perror("recv");
#endif
                        continue;
                }

                assert(ret + strlen(line) < sizeof line - 1); // TODO...
                memcpy(line + strlen(line), buf, ret);
                // process individual lines
                while (strchr(line, '\n')) {
                        char *next = strchr(line, '\n') + 1;
                        *strchr(line, '\n') = '\0';

                        // process one line
                        if (strstr(line, "stats ARECV") == line) { // only process those lines that belong to us
                                parse_and_store(c, line);
                        }
                        
                        // move the rest
                        size_t rest_len = strlen(next);
                        memmove(line, next, rest_len);
                        memset(line + rest_len, 0, sizeof line - rest_len);
                }
        }
}

/**
 * @param local_port UltraGrid control port to connect to
 */
struct ug_connection *ug_control_connection_init(int local_port) {
        int fd = socket(AF_INET6, SOCK_STREAM, 0);
        if (fd == -1) {
                return nullptr;
        }
        fd_t should_exit_fd[2];
        struct sockaddr_in6 sin;
        memset(&sin, 0, sizeof sin);
        sin.sin6_family = AF_INET6;
        sin.sin6_port = htons(local_port);
        sin.sin6_addr = in6addr_loopback;

        if (connect(fd, (const sockaddr*) &sin, sizeof sin) == -1) {
                CLOSESOCKET(fd);
                return NULL;
        }

	char stats_on[] = "stats on\r\n";
	if (write_all(fd, stats_on, sizeof stats_on) != sizeof stats_on) {
                CLOSESOCKET(fd);
                return NULL;
	}

        platform_pipe_init(should_exit_fd);

        return new ug_connection{fd, should_exit_fd};
}

void ug_control_connection_done(struct ug_connection *c) {
        if(!c){
                return;
        }
        c->should_exit = true;
        char ch = 0;
        int ret = send(c->should_exit_fd[1], &ch, 1, MSG_DONTWAIT);
        if (ret == -1) {
                perror("ug_control_connection_done send");
        }
        c->t.join();
        platform_pipe_close(c->should_exit_fd[0]);
        platform_pipe_close(c->should_exit_fd[1]);
        CLOSESOCKET(c->fd);
        delete c;
}

/**
 * @param peak Returns peak volume for 2 channels. Must be array of 2 doubles.
 * @param rms Returns RMS volume for 2 channels. Must be array of 2 doubles.
 * @retval false if connection was closed
 */
bool ug_control_get_volumes(struct ug_connection *c, double peak[], double rms[], int *count) {
        if (!c)
                return false;

        lock_guard<mutex> lk(c->lock);

        if (c->connection_lost) {
                return false;
        }

        memcpy(peak, c->volpeak, sizeof c->volpeak);
        memcpy(rms, c->volrms, sizeof c->volrms);
        *count = c->sample_count;

        std::fill_n(c->volpeak, sizeof(c->volpeak)/sizeof(*c->volpeak), -INFINITY);
        memset(c->volrms, 0, sizeof c->volrms);
        c->sample_count = 0;

        return true;
}

void ug_control_init(){
#ifdef _WIN32	
        WSADATA wsa;
        WSAStartup(MAKEWORD(2,2), &wsa);
#endif
}

void ug_control_cleanup(){
#ifdef _WIN32
        WSACleanup();
#endif
}

#ifdef DEFINE_TEST_MAIN
#define MAX_RECONNECTS 5

static bool should_exit = false;

static void signal_handler(int signal) {
        should_exit = true;
}

int main() {
        auto connection = ug_control_connection_init(8888);
        while (!connection) {
			sleep(1);
			connection = ug_control_connection_init(8888);
                fprintf(stderr, "Unable to initialize!\n");
        }
        int reconnect_attempt = 0;

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        while (!should_exit) {
                double volpeak[2];
                double volrms[2];
                int sample_count;

                sleep(1);
                bool ret = ug_control_get_volumes(connection, volpeak, volrms, &sample_count);
                if (ret) {
                        printf("samples: %d, volpeak0: %lf, volrms0: %lf, volpeak1: %lf, volrms1: %lf\n", sample_count, volpeak[0], volrms[0], volpeak[1], volrms[1]);
                } else {
                        fprintf(stderr, "Connection was closed!\n");
                        // try to reinit
                        ug_control_connection_done(connection);
                        connection = ug_control_connection_init(8888);
                        if (!connection) {
                                fprintf(stderr, "Unable to initialize!\n");
                                if (reconnect_attempt < MAX_RECONNECTS) {
                                        continue;
                                } else {
                                        return 1;
                                }
                        }
                        fprintf(stderr, "Successfully reconnected.\n");
                }
        }

        ug_control_connection_done(connection);
        printf("Exit.\n");
}
#endif // DEFINE_TEST_MAIN


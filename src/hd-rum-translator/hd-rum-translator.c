#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "debug.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <pthread.h>
#include <fcntl.h>

#include "hd-rum-translator/hd-rum-recompress.h"
#include "hd-rum-translator/hd-rum-decompress.h"

#define EXIT_FAIL_USAGE 1
#define EXIT_INIT_PORT 3

long packet_rate = 13600;

static pthread_t main_thread_id;

/*
 * Prototypes
 */
static int output_socket(unsigned short port, const char *host, int bufsize);
static void *replica_init(const char *host, unsigned short port, int bufsize);
static ssize_t replica_write(void *state, void *buf, size_t count);
static void replica_done(void *state);
static void qinit(int qsize);
static int buffer_size(int sock, int optname, int size);
static void *writer(void *arg);
static void signal_handler(int signal);
static void usage();

static volatile int should_exit = false;

/*
 * this is currently only placeholder to substitute UG default
 */
static void _exit_uv(int status);
static void _exit_uv(int status) {
    UNUSED(status);
    should_exit = true;
    if(!pthread_equal(pthread_self(), main_thread_id))
            pthread_kill(main_thread_id, SIGTERM);
}
void (*exit_uv)(int status) = _exit_uv;

static void signal_handler(int signal)
{
    debug_msg("Caught signal %d\n", signal);
    exit_uv(0);
    return;
}

#define SIZE    10000

typedef ssize_t (*write_t)(void *state, void *buf, size_t count);
typedef void (*done_t)(void *state);

#define REPLICA_MAGIC 0xd2ff3323

struct replica {
    uint32_t magic;
    const char *host;
    unsigned short port;
    int sock;
};

static void *replica_init(const char *host, unsigned short port, int bufsize) {
        struct replica *replica = (struct replica *)
            calloc(1, sizeof(struct replica));

        replica->magic = REPLICA_MAGIC;
        replica->host = host;
        replica->port = port;
        replica->sock = output_socket(port, host,
                                         bufsize);

        return (void *) replica;
}

static ssize_t replica_write(void *state, void *buf, size_t count) {
        struct replica *s = (struct replica *) state;
        return write(s->sock, buf, count);
}

static void replica_done(void *state) {
        struct replica *s = (struct replica *) state;
        assert(s->magic == REPLICA_MAGIC);

        close(s->sock);
        free(s);
}

struct forward {
        void *state;
        write_t write;
        done_t done;
};

struct item {
    struct item *next;
    long size;
    char buf[SIZE];
};

static struct item *queue;
static struct item *qhead;
static struct item *qtail;
static int qempty = 1;
static int qfull = 0;
static pthread_mutex_t qempty_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t qfull_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t qempty_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t qfull_cond = PTHREAD_COND_INITIALIZER;

struct forward *forwards;
int in_port_count;

static void qinit(int qsize)
{
    int i;

    printf("initializing packet queue for %d items\n", qsize);

    queue = (struct item *) calloc(qsize, sizeof(struct item));
    if (queue == NULL) {
        fprintf(stderr, "not enough memory\n");
        exit(2);
    }

    for (i = 0; i < qsize; i++)
        queue[i].next = queue + i + 1;
    queue[qsize - 1].next = queue;

    qhead = qtail = queue;
}


static int buffer_size(int sock, int optname, int size)
{
    socklen_t len = sizeof(int);

    if (setsockopt(sock, SOL_SOCKET, optname, (void *) &size, len)
        || getsockopt(sock, SOL_SOCKET, optname, (void *) &size, &len)) {
        perror("[sg]etsockopt()");
        return -1;
    }

    printf("UDP %s buffer size set to %d bytes\n",
           (optname == SO_SNDBUF) ? "send" : "receiver", size);

    return 0;
}


static int output_socket(unsigned short port, const char *host, int bufsize)
{
    int s;
    struct addrinfo hints;
    struct addrinfo *res;
    char saddr[INET_ADDRSTRLEN];
    char p[6];

    memset(&hints, '\0', sizeof(hints));
    hints.ai_flags = AI_CANONNAME;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_family = PF_INET;

    snprintf(p, 6, "%d", port);
    p[5] = '\0';

    if (getaddrinfo(host, p, &hints, &res)) {
        int err = errno;
        if (err == 0)
            fprintf(stderr, "Address not found: %s\n", host);
        else
            fprintf(stderr, "%s: %s\n", gai_strerror(err), host);
        exit(2);
    }

    inet_ntop(AF_INET, &((struct sockaddr_in *) res->ai_addr)->sin_addr,
              saddr, sizeof(saddr));
    printf("connecting to %s (%s) port %d\n",
           res->ai_canonname, saddr, port);

    if ((s = socket(res->ai_family, res->ai_socktype, 0)) == -1) {
        perror("socket");
        exit(2);
    }

    if (buffer_size(s, SO_SNDBUF, bufsize))
        exit(2);

    if (connect(s, res->ai_addr, res->ai_addrlen)) {
        perror("connect");
        exit(2);
    }

    return s;
}

static void *writer(void *arg)
{
    int i;
    UNUSED(arg);

    while (1) {
        while (qhead != qtail) {
            if(qhead->size == 0) { // poisoned pill
                return NULL;
            }
            for (i = 0; i < in_port_count; i++) {
                forwards[i].write(forwards[i].state, qhead->buf, qhead->size);
            }

            qhead = qhead->next;

            pthread_mutex_lock(&qfull_mtx);
            qfull = 0;
            pthread_cond_signal(&qfull_cond);
            pthread_mutex_unlock(&qfull_mtx);
        }

        pthread_mutex_lock(&qempty_mtx);
        if (qempty)
            pthread_cond_wait(&qempty_cond, &qempty_mtx);
        qempty = 1;
        pthread_mutex_unlock(&qempty_mtx);
    }

    return NULL;
}

static void usage(const char *progname) {
        printf("%s buffer_size port [host1_options] host1 [[host2_options] host2] ...\n",
                progname);
        printf("\twhere hostX_options may be:\n"
                "\t\t-c <compression> - compression\n"
                "\t\t-m <mtu> - MTU size. Will be used only with compression.\n"
                "\t\t-f <fec> - FEC that will be used for transmission.\n"
              );

}

struct host_opts {
    char *addr;
    int mtu;
    char *compression;
    char *fec;
};

static void parse_fmt(int argc, char **argv, char **bufsize, unsigned short *port,
        struct host_opts **host_opts, int *host_opts_count)
{
    *bufsize = argv[1];
    *port = atoi(argv[2]);

    argv += 2;
    argc -= 2;

    *host_opts_count = 0;

    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            *host_opts_count += 1;
        } else {
            if(strlen(argv[i]) != 2) {
                fprintf(stderr, "Error: invalild option '%s'\n", argv[i]);
                exit(EXIT_FAIL_USAGE);
            }

            if(i == argc - 1 || argv[i + 1][0] == '-') {
                fprintf(stderr, "Error: option '-%c' requires an argument\n", argv[i][1]);
                exit(EXIT_FAIL_USAGE);
            }

            *host_opts_count -= 1; // because option argument (mandatory) will be counted as a host
                                   // in next iteration
        }
    }

    *host_opts = calloc(*host_opts_count, sizeof(struct host_opts));
    int host_idx = 0;
    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch(argv[i][1]) {
                case 'm':
                    (*host_opts)[host_idx].mtu = atoi(argv[i + 1]);
                    break;
                case 'c':
                    (*host_opts)[host_idx].compression = argv[i + 1];
                    break;
                case 'f':
                    (*host_opts)[host_idx].fec = argv[i + 1];
                    break;
                default:
                    fprintf(stderr, "Error: invalild option '%s'\n", argv[i]);
                    exit(EXIT_FAIL_USAGE);
            }
            i += 1;
        } else {
            (*host_opts)[host_idx].addr = argv[i];
            host_idx += 1;
        }
    }
}

#ifndef WIN32
int main(int argc, char **argv)
{
    unsigned short port;
    int qsize;
    int bufsize;
    char *bufsize_str;
    struct sockaddr_in addr;
    int sock_in;
    pthread_t thread;
    int err = 0;
    int i;
    struct host_opts *hosts;
    int host_count;

    if (argc < 4) {
        usage(argv[0]);
        return EXIT_FAIL_USAGE;
    }

    main_thread_id = pthread_self();

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
#ifndef WIN32
    sigaction(SIGHUP, &sa, NULL);
#endif
    sigaction(SIGABRT, &sa, NULL);

    parse_fmt(argc, argv, &bufsize_str, &port, &hosts, &host_count);

    if ((bufsize = atoi(bufsize_str)) <= 0) {
        fprintf(stderr, "invalid buffer size: %d\n", bufsize);
        return 1;
    }
    switch (bufsize_str[strlen(bufsize_str) - 1]) {
    case 'K':
    case 'k':
        bufsize *= 1024;
        break;

    case 'M':
    case 'm':
        bufsize *= 1024*1024;
        break;
    }

    qsize = bufsize / 8000;

    printf("using UDP send and receive buffer size of %d bytes\n", bufsize);

    if (port <= 0) {
        fprintf(stderr, "invalid port: %d\n", port);
        return 1;
    }

    qinit(qsize);

    /* input socket */
    if ((sock_in = socket(PF_INET, SOCK_DGRAM, 0)) == -1) {
        perror("input socket");
        return 2;
    }

    if (buffer_size(sock_in, SO_RCVBUF, bufsize))
        exit(2);

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock_in, (struct sockaddr *) &addr,
             sizeof(struct sockaddr_in))) {
        perror("bind");
        return 2;
    }

    printf("listening on *:%d\n", port);

    /* output socket(s) */
    forwards = (struct forward *) calloc(host_count, sizeof(struct replica));
    if (forwards == NULL) {
        fprintf(stderr, "not enough memory for replica array");
        return 2;
    }

    void *decompress = NULL;

    in_port_count = 0;
    for (i = 0; i < host_count; i++) {
        if(hosts[i].compression == NULL) {
            forwards[in_port_count].state =
                replica_init(hosts[i].addr, port, bufsize);
            if(!forwards[in_port_count].state) {
                return EXIT_INIT_PORT;
            }
            forwards[in_port_count].write = replica_write;
            forwards[in_port_count].done = replica_done;
            in_port_count += 1;
        } else {
            // we need only one shared receiver decompressor for all recompressing streams
            if(decompress == NULL) {
                decompress = hd_rum_decompress_init(port + 2);
                forwards[in_port_count].state = decompress;
                if(!forwards[in_port_count].state) {
                    return EXIT_INIT_PORT;
                }
                forwards[in_port_count].write = hd_rum_decompress_write;
                forwards[in_port_count].done = hd_rum_decompress_done;
                ++in_port_count;
            }

            int mtu = 1500;
            if(hosts[i].mtu) {
                mtu = hosts[i].mtu;
            }

            void *recompress = recompress_init(hosts[i].addr, hosts[i].compression,
                    port + (i + 2) * 2, port, mtu, hosts[i].fec);
            if(recompress == 0) {
                fprintf(stderr, "Initializing output port '%s' failed!\n",
                        hosts[i].addr);
                return EXIT_INIT_PORT;
            }
            // we don't care about this clients, we only tell decompressor to
            // take care about them
            hd_rum_decompress_add_port(decompress, recompress);
        }
    }

    if (pthread_create(&thread, NULL, writer, NULL)) {
        fprintf(stderr, "cannot create writer thread\n");
        return 2;
    }

    /* main loop */
    while (!should_exit) {
        while (qtail->next != qhead
               && (qtail->size = read(sock_in, qtail->buf, SIZE)) > 0
               && !should_exit) {

            qtail = qtail->next;

            pthread_mutex_lock(&qempty_mtx);
            qempty = 0;
            pthread_cond_signal(&qempty_cond);
            pthread_mutex_unlock(&qempty_mtx);
        }

        if (qtail->size <= 0)
            break;

        pthread_mutex_lock(&qfull_mtx);
        if (qfull)
            pthread_cond_wait(&qfull_cond, &qfull_mtx);
        qfull = 1;
        pthread_mutex_unlock(&qfull_mtx);
    }

    if (qtail->size < 0 && !should_exit) {
        printf("read: %s\n", strerror(err));
        return 2;
    }

    // pass poisoned pill to the worker
    qtail->size = 0;
    qtail = qtail->next;

    pthread_mutex_lock(&qempty_mtx);
    qempty = 0;
    pthread_cond_signal(&qempty_cond);
    pthread_mutex_unlock(&qempty_mtx);

    pthread_join(thread, NULL);

    for (i = 0; i < in_port_count; i++) {
        forwards[i].done(forwards[i].state);
    }

    printf("Exit\n");

    return 0;
}
#else // WIN32
#include <stdio.h>
int main()
{
    fprintf(stderr, "Reflector is currently not supported under MSW\n");
    return 1;
}
#endif

/* vim: set sw=4 expandtab : */

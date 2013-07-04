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

#include "control_socket.h"
#include "hd-rum-translator/hd-rum-recompress.h"
#include "hd-rum-translator/hd-rum-decompress.h"
#include "module.h"
#include "stats.h"
#include "tv.h"

#define EXIT_FAIL_USAGE 1
#define EXIT_INIT_PORT 3

long packet_rate = 13600;

static pthread_t main_thread_id;
struct forward;
struct item;

struct replica {
    struct module mod;
    uint32_t magic;
    const char *host;
    unsigned short port;

    enum {
        USE_SOCK,
        RECOMPRESS
    } type;
    int sock;
    void *recompress;
};

struct hd_rum_translator_state {
    struct module mod;
    struct item *queue;
    struct item *qhead;
    struct item *qtail;
    int qempty;
    int qfull;
    pthread_mutex_t qempty_mtx;
    pthread_mutex_t qfull_mtx;
    pthread_cond_t qempty_cond;
    pthread_cond_t qfull_cond;

    struct replica *replicas;
    void *decompress;
    int host_count;
};

/*
 * Prototypes
 */
static int output_socket(unsigned short port, const char *host, int bufsize);
static ssize_t replica_write(struct replica *s, void *buf, size_t count);
static void replica_done(struct replica *s);
static struct item *qinit(int qsize);
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

#define REPLICA_MAGIC 0xd2ff3323

static ssize_t replica_write(struct replica *s, void *buf, size_t count)
{
        return write(s->sock, buf, count);
}

static void replica_done(struct replica *s)
{
        assert(s->magic == REPLICA_MAGIC);
        module_done(&s->mod);

        close(s->sock);
}

struct item {
    struct item *next;
    long size;
    char buf[SIZE];
};

static struct item *qinit(int qsize)
{
    struct item *queue;
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

    return queue;
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

    inet_ntop(AF_INET, &((struct sockaddr_in *)(void *) res->ai_addr)->sin_addr,
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
    struct hd_rum_translator_state *s =
        (struct hd_rum_translator_state *) arg;

    while (1) {
        while (s->qhead != s->qtail) {
            if(s->qhead->size == 0) { // poisoned pill
                return NULL;
            }
            for (i = 0; i < s->host_count; i++) {
                if(s->replicas[i].sock != -1) {
                    replica_write(&s->replicas[i],
                            s->qhead->buf, s->qhead->size);
                }
            }

            if(s->decompress) {
                hd_rum_decompress_write(s->decompress,
                        s->qhead->buf, s->qhead->size);
            }

            s->qhead = s->qhead->next;

            pthread_mutex_lock(&s->qfull_mtx);
            s->qfull = 0;
            pthread_cond_signal(&s->qfull_cond);
            pthread_mutex_unlock(&s->qfull_mtx);
        }

        pthread_mutex_lock(&s->qempty_mtx);
        if (s->qempty)
            pthread_cond_wait(&s->qempty_cond, &s->qempty_mtx);
        s->qempty = 1;
        pthread_mutex_unlock(&s->qempty_mtx);
    }

    return NULL;
}

static void usage(const char *progname) {
        printf("%s [global_opts] buffer_size port [host1_options] host1 [[host2_options] host2] ...\n",
                progname);
        printf("\twhere global_opts may be:\n"
                "\t\t--control-port <port_number> - control port to connect to\n");
        printf("\tand hostX_options may be:\n"
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
        struct host_opts **host_opts, int *host_opts_count, int *control_port)
{
    int start_index = 1;

    while(argv[start_index][0] == '-') {
        if(strcmp(argv[start_index], "--control-port") == 0) {
            *control_port = atoi(argv[++start_index]);
        }
        start_index++;
    }

    *bufsize = argv[start_index];
    *port = atoi(argv[start_index + 1]);

    argv += start_index + 1;
    argc -= start_index + 1;

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

static void hd_rum_translator_state_init(struct hd_rum_translator_state *s)
{
    module_init_default(&s->mod);
    s->mod.cls = MODULE_CLASS_ROOT;

    s->qempty = 1;
    s->qfull = 0;
    pthread_mutex_init(&s->qempty_mtx, NULL);
    pthread_mutex_init(&s->qfull_mtx, NULL);
    pthread_cond_init(&s->qempty_cond, NULL);
    pthread_cond_init(&s->qfull_cond, NULL);
    s->decompress = NULL;
}

static void hd_rum_translator_state_destroy(struct hd_rum_translator_state *s)
{
    pthread_mutex_destroy(&s->qempty_mtx);
    pthread_mutex_destroy(&s->qfull_mtx);
    pthread_cond_destroy(&s->qempty_cond);
    pthread_cond_destroy(&s->qfull_cond);

    module_done(&s->mod);
}

#ifndef WIN32
int main(int argc, char **argv)
{
    struct hd_rum_translator_state state;

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
    int control_port = CONTROL_DEFAULT_PORT;
    struct control_state *control_state = NULL;

    if (argc < 4) {
        usage(argv[0]);
        return EXIT_FAIL_USAGE;
    }

    hd_rum_translator_state_init(&state);

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

    parse_fmt(argc, argv, &bufsize_str, &port, &hosts, &host_count, &control_port);

    if (host_count == 0) {
        usage(argv[0]);
        return EXIT_FAIL_USAGE;
    }

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

    state.qhead = state.qtail = state.queue = qinit(qsize);

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
    state.replicas = (struct replica *) calloc(host_count, sizeof(struct replica));
    if (state.replicas == NULL) {
        fprintf(stderr, "not enough memory for replica array");
        return 2;
    }

    // we need only one shared receiver decompressor for all recompressing streams
    state.decompress = hd_rum_decompress_init(port + 2);
    if(!state.decompress) {
        return EXIT_INIT_PORT;
    }

    state.host_count = host_count;
    for (i = 0; i < host_count; i++) {
        state.replicas[i].magic = REPLICA_MAGIC;
        state.replicas[i].host = hosts[i].addr;
        state.replicas[i].port = port;
        state.replicas[i].sock = output_socket(port, hosts[i].addr,
                bufsize);
        module_init_default(&state.replicas[i].mod);
        state.replicas[i].mod.cls = MODULE_CLASS_PORT;

        if(hosts[i].compression == NULL) {
            state.replicas[i].type = USE_SOCK;
            int mtu = 1500;
            char compress[] = "none";
            char *fec = NULL;
            state.replicas[i].recompress = recompress_init(&state.replicas[i].mod,
                    hosts[i].addr, compress,
                    0, port, mtu, fec);
            hd_rum_decompress_add_inactive_port(state.decompress, state.replicas[i].recompress);
        } else {
            state.replicas[i].type = RECOMPRESS;
            int mtu = 1500;
            if(hosts[i].mtu) {
                mtu = hosts[i].mtu;
            }

            state.replicas[i].recompress = recompress_init(&state.replicas[i].mod,
                    hosts[i].addr, hosts[i].compression,
                    0, port, mtu, hosts[i].fec);
            if(state.replicas[i].recompress == 0) {
                fprintf(stderr, "Initializing output port '%s' failed!\n",
                        hosts[i].addr);
                return EXIT_INIT_PORT;
            }
            // we don't care about this clients, we only tell decompressor to
            // take care about them
            hd_rum_decompress_add_port(state.decompress, state.replicas[i].recompress);
        }

        module_register(&state.replicas[i].mod, &state.mod);
    }

    if(control_init(control_port, &control_state, &state.mod) != 0) {
        fprintf(stderr, "Warning: Unable to create remote control.\n");
        if(control_port != CONTROL_DEFAULT_PORT) {
            return EXIT_FAILURE;
        }
    }

    if (pthread_create(&thread, NULL, writer, (void *) &state)) {
        fprintf(stderr, "cannot create writer thread\n");
        return 2;
    }

    struct stats *stat_received = stats_new_statistics(
            control_state,
            "received");
    uint64_t received_data = 0;
    struct timeval t0, t;
    gettimeofday(&t0, NULL);

    /* main loop */
    while (!should_exit) {
        while (state.qtail->next != state.qhead
               && (state.qtail->size = read(sock_in, state.qtail->buf, SIZE)) > 0
               && !should_exit) {
            received_data += state.qtail->size;

            state.qtail = state.qtail->next;

            pthread_mutex_lock(&state.qempty_mtx);
            state.qempty = 0;
            pthread_cond_signal(&state.qempty_cond);
            pthread_mutex_unlock(&state.qempty_mtx);

            gettimeofday(&t, NULL);
            if(tv_diff(t, t0) > 1.0) {
                stats_update_int(stat_received, received_data);
                t0 = t;
            }
        }

        if (state.qtail->size <= 0)
            break;

        pthread_mutex_lock(&state.qfull_mtx);
        if (state.qfull)
            pthread_cond_wait(&state.qfull_cond, &state.qfull_mtx);
        state.qfull = 1;
        pthread_mutex_unlock(&state.qfull_mtx);
    }

    if (state.qtail->size < 0 && !should_exit) {
        printf("read: %s\n", strerror(err));
        return 2;
    }

    // pass poisoned pill to the worker
    state.qtail->size = 0;
    state.qtail = state.qtail->next;

    pthread_mutex_lock(&state.qempty_mtx);
    state.qempty = 0;
    pthread_cond_signal(&state.qempty_cond);
    pthread_mutex_unlock(&state.qempty_mtx);

    stats_destroy(stat_received);
    control_done(control_state);

    pthread_join(thread, NULL);

    if(state.decompress) {
        hd_rum_decompress_done(state.decompress);
    }

    for (i = 0; i < state.host_count; i++) {
        if(state.replicas[i].sock != -1) {
            replica_done(&state.replicas[i]);
        }
    }

    hd_rum_translator_state_destroy(&state);

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

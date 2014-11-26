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
#include "host.h"
#include "hd-rum-translator/hd-rum-recompress.h"
#include "hd-rum-translator/hd-rum-decompress.h"
#include "messaging.h"
#include "module.h"
#include "stats.h"
#include "utils/misc.h"
#include "tv.h"

#include <vector>

#define EXIT_FAIL_USAGE 1
#define EXIT_INIT_PORT 3

using namespace std;

static pthread_t main_thread_id;
struct forward;
struct item;

struct replica {
    struct module mod;
    uint32_t magic;
    const char *host;

    enum type_t {
        NONE,
        USE_SOCK,
        RECOMPRESS
    };
    enum type_t type;
    int sock;
    void *recompress;
    volatile enum type_t change_to_type;
};

struct hd_rum_translator_state {
    hd_rum_translator_state() : mod{}, queue(nullptr), qhead(nullptr), qtail(nullptr), qempty(1),
            qfull(0), decompress(nullptr) {
        module_init_default(&mod);
        mod.cls = MODULE_CLASS_ROOT;
        pthread_mutex_init(&qempty_mtx, NULL);
        pthread_mutex_init(&qfull_mtx, NULL);
        pthread_cond_init(&qempty_cond, NULL);
        pthread_cond_init(&qfull_cond, NULL);
    }
    ~hd_rum_translator_state() {
        pthread_mutex_destroy(&qempty_mtx);
        pthread_mutex_destroy(&qfull_mtx);
        pthread_cond_destroy(&qempty_cond);
        pthread_cond_destroy(&qfull_cond);
        module_done(&mod);
    }
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

    vector<replica *> replicas;
    void *decompress;
};

/*
 * Prototypes
 */
static int output_socket(unsigned short port, const char *host, int bufsize);
static void replica_done(struct replica *s) __attribute__((unused));
static struct item *qinit(int qsize);
static int buffer_size(int sock, int optname, int size);
static void *writer(void *arg);
static void signal_handler(int signal);
void exit_uv(int status);

static volatile int should_exit = false;

/*
 * this is currently only placeholder to substitute UG default
 */
void exit_uv(int status) {
    UNUSED(status);
    should_exit = true;
    if(!pthread_equal(pthread_self(), main_thread_id))
            pthread_kill(main_thread_id, SIGTERM);
}

static void signal_handler(int signal)
{
    debug_msg("Caught signal %d\n", signal);
    exit_uv(0);
    return;
}

#define SIZE    10000

#define REPLICA_MAGIC 0xd2ff3323

static struct response *change_replica_type_callback(struct module *mod, struct message *msg);

static void replica_init(struct replica *s, const char *addr, int tx_port, int bufsize, struct module *parent)
{
        s->magic = REPLICA_MAGIC;
        s->host = addr;
        s->sock = output_socket(tx_port, addr,
                bufsize);
        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_PORT;
        s->mod.msg_callback = change_replica_type_callback;
        s->mod.priv_data = s;
        module_register(&s->mod, parent);
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

#ifdef _WIN32
typedef char sockopt_t;
#else
typedef void sockopt_t;
#endif

static int buffer_size(int sock, int optname, int size)
{
    socklen_t len = sizeof(int);

    if (setsockopt(sock, SOL_SOCKET, optname, (const sockopt_t *) &size, len)
        || getsockopt(sock, SOL_SOCKET, optname, (sockopt_t *) &size, &len)) {
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
    struct addrinfo *res = NULL;
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

    freeaddrinfo(res);

    return s;
}

static struct response *change_replica_type_callback(struct module *mod, struct message *msg)
{
    struct replica *s = (struct replica *) mod->priv_data;

    struct msg_universal *data = (struct msg_universal *) msg;

    enum replica::type_t new_type;
    if (strcasecmp(data->text, "sock") == 0) {
        new_type = replica::type_t::USE_SOCK;
    } else if (strcasecmp(data->text, "recompress") == 0) {
        new_type = replica::type_t::RECOMPRESS;
    } else {
        new_type = replica::type_t::NONE;
    }
    free_message(msg);

    if (new_type == replica::type_t::NONE) {
        return new_response(RESPONSE_BAD_REQUEST, NULL);
    }

    while (s->change_to_type != replica::type_t::NONE)
        ;
    s->change_to_type = new_type;

    return new_response(RESPONSE_OK, NULL);
}

static void *writer(void *arg)
{
    struct hd_rum_translator_state *s =
        (struct hd_rum_translator_state *) arg;

    while (1) {
        // first check messages
        for (unsigned int i = 0; i < s->replicas.size(); i++) {
            if (s->replicas[i]->change_to_type != replica::type_t::NONE) {
                s->replicas[i]->type = s->replicas[i]->change_to_type;
                hd_rum_decompress_set_active(s->decompress, s->replicas[i]->recompress,
                        s->replicas[i]->change_to_type == replica::type_t::RECOMPRESS);
                s->replicas[i]->change_to_type = replica::type_t::NONE;
            }
        }

        struct msg_universal *msg;
        while ((msg = (struct msg_universal *) check_message(&s->mod))) {
            if (strncasecmp(msg->text, "delete-port ", strlen("delete-port ")) == 0) {
                int index = atoi(msg->text + strlen("delete-port "));
                hd_rum_decompress_remove_port(s->decompress, index);
                replica_done(s->replicas[index]);
                delete s->replicas[index];
                s->replicas.erase(s->replicas.begin() + index);
            } else if (strncasecmp(msg->text, "create-port", strlen("create-port")) == 0) {
                s->replicas.push_back(new replica());
                struct replica *rep = s->replicas[s->replicas.size() - 1];
                char *host, *save_ptr;
                strtok_r(msg->text, " ", &save_ptr);
                host = strtok_r(NULL, " ", &save_ptr);
                int tx_port = atoi(strtok_r(NULL, " ", &save_ptr));
                char *compress = strtok_r(NULL, " ", &save_ptr);
                replica_init(rep, host, tx_port, 100*1000, &s->mod);

                rep->type = replica::type_t::USE_SOCK;
                char *fec = NULL;
                rep->recompress = recompress_init(&rep->mod,
                        host, compress,
                        0, tx_port, 1500, fec, RATE_UNLIMITED);
                hd_rum_decompress_append_port(s->decompress, rep->recompress);
            }

            free_message((struct message *) msg);
        }

        // then process incoming packets
        while (s->qhead != s->qtail) {
            if(s->qhead->size == 0) { // poisoned pill
                return NULL;
            }
            for (unsigned int i = 0; i < s->replicas.size(); i++) {
                if(s->replicas[i]->type == replica::type_t::USE_SOCK) {
                    ssize_t ret = send(s->replicas[i]->sock, s->qhead->buf, s->qhead->size, 0);
                    if (ret < 0) {
                        perror("Hd-rum-translator send");
                    }
                }
            }

            ssize_t ret = hd_rum_decompress_write(s->decompress, s->qhead->buf, s->qhead->size);
            if (ret < 0) {
                perror("hd_rum_decompress_write");
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
                "\t\t--control-port <port_number>[:0|:1] - control port to connect to, optionally client/server (default)\n"
                "\t\t--help\n");
        printf("\tand hostX_options may be:\n"
                "\t\t-P <port> - TX port to be used\n"
                "\t\t-c <compression> - compression\n"
                "\t\tFollowing options will be used only if '-c' parameter is set:\n"
                "\t\t-m <mtu> - MTU size\n"
                "\t\t-l <limiting_bitrate> - bitrate to be shaped to\n"
                "\t\t-f <fec> - FEC that will be used for transmission.\n"
              );
}

struct host_opts {
    char *addr;
    int port;
    int mtu;
    char *compression;
    char *fec;
    int64_t bitrate;
};

static bool parse_fmt(int argc, char **argv, char **bufsize, unsigned short *port,
        struct host_opts **host_opts, int *host_opts_count, int *control_port, int *control_connection_type)
{
    int start_index = 1;

    while(start_index < argc && argv[start_index][0] == '-') {
        if(strcmp(argv[start_index], "--control-port") == 0) {
            char *item = argv[++start_index];
            *control_port = atoi(item);
            *control_connection_type = 0;
            if (strchr(item, ':')) {
                *control_connection_type = atoi(strchr(item, ':') + 1);
            }
        } else if(strcmp(argv[start_index], "--capabilities") == 0) {
            print_capabilities(CAPABILITY_COMPRESS);
            return false;
        } else if(strcmp(argv[start_index], "--help") == 0) {
            usage(argv[0]);
            return false;
        }
        start_index++;
    }

    if (argc < start_index + 2) {
        usage(argv[0]);
        return false;
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

    *host_opts = (struct host_opts *) calloc(*host_opts_count, sizeof(struct host_opts));
    // default values
    for(int i = 0; i < *host_opts_count; ++i) {
        (*host_opts)[i].bitrate = RATE_UNLIMITED;
        (*host_opts)[i].mtu = 1500;
    }

    int host_idx = 0;
    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch(argv[i][1]) {
                case 'P':
                    (*host_opts)[host_idx].port = atoi(argv[i + 1]);
                    break;
                case 'm':
                    (*host_opts)[host_idx].mtu = atoi(argv[i + 1]);
                    break;
                case 'c':
                    (*host_opts)[host_idx].compression = argv[i + 1];
                    break;
                case 'f':
                    (*host_opts)[host_idx].fec = argv[i + 1];
                    break;
                case 'l':
                    if (strcmp(argv[i + 1], "unlimited") == 0) {
                        (*host_opts)[host_idx].bitrate = RATE_UNLIMITED;
                    } else if (strcmp(argv[i + 1], "auto") == 0) {
                        (*host_opts)[host_idx].bitrate = RATE_AUTO;
                    } else {
                        (*host_opts)[host_idx].bitrate = unit_evaluate(argv[i + 1]);
                    }
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

    return true;
}

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
    int control_connection_type = 0;
    struct control_state *control_state = NULL;
#ifdef WIN32
    WSADATA wsaData;

    err = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if(err != 0) {
        fprintf(stderr, "WSAStartup failed with error %d.", err);
        return EXIT_FAILURE;
    }
    if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
        fprintf(stderr, "Counld not found usable version of Winsock.\n");
        WSACleanup();
        return EXIT_FAILURE;
    }

#endif
    if (argc == 1) {
        usage(argv[0]);
        return false;
    }


    uv_argc = argc;
    uv_argv = argv;

    main_thread_id = pthread_self();

#ifndef WIN32
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);
#endif

    bool ret = parse_fmt(argc, argv, &bufsize_str, &port, &hosts, &host_count, &control_port,
            &control_connection_type);

    if (ret == false) {
        return EXIT_SUCCESS;
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

    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(sock_in, (struct sockaddr *) &addr,
             sizeof(struct sockaddr_in))) {
        perror("bind");
        return 2;
    }

    printf("listening on *:%d\n", port);

    state.replicas.resize(host_count);
    for (auto && rep : state.replicas) {
        rep = new replica();
    }

    if(control_init(control_port, control_connection_type, &control_state, &state.mod) != 0) {
        fprintf(stderr, "Warning: Unable to create remote control.\n");
        return EXIT_FAILURE;
    }
    control_start(control_state);

    // we need only one shared receiver decompressor for all recompressing streams
    state.decompress = hd_rum_decompress_init(&state.mod);;
    if(!state.decompress) {
        return EXIT_INIT_PORT;
    }

    for (i = 0; i < host_count; i++) {
        int packet_rate;
        if (hosts[i].bitrate != RATE_AUTO && hosts[i].bitrate != RATE_UNLIMITED) {
                packet_rate = compute_packet_rate(hosts[i].bitrate, hosts[i].mtu);
        } else {
                packet_rate = hosts[i].bitrate;
        }

        int tx_port = port;
        if(hosts[i].port) {
            tx_port = hosts[i].port;
        }

        replica_init(state.replicas[i], hosts[i].addr, tx_port, bufsize, &state.mod);

        if(hosts[i].compression == NULL) {
            state.replicas[i]->type = replica::type_t::USE_SOCK;
            char compress[] = "none";
            char *fec = NULL;
            state.replicas[i]->recompress = recompress_init(&state.replicas[i]->mod,
                    hosts[i].addr, compress,
                    0, tx_port, hosts[i].mtu, fec, packet_rate);
            hd_rum_decompress_append_port(state.decompress, state.replicas[i]->recompress);
            hd_rum_decompress_set_active(state.decompress, state.replicas[i]->recompress, false);
        } else {
            state.replicas[i]->type = replica::type_t::RECOMPRESS;

            state.replicas[i]->recompress = recompress_init(&state.replicas[i]->mod,
                    hosts[i].addr, hosts[i].compression,
                    0, tx_port, hosts[i].mtu, hosts[i].fec, packet_rate);
            if(state.replicas[i]->recompress == 0) {
                fprintf(stderr, "Initializing output port '%s' failed!\n",
                        hosts[i].addr);
                return EXIT_INIT_PORT;
            }
            // we don't care about this clients, we only tell decompressor to
            // take care about them
            hd_rum_decompress_append_port(state.decompress, state.replicas[i]->recompress);
            hd_rum_decompress_set_active(state.decompress, state.replicas[i]->recompress, true);
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
    struct timeval t_report;
    gettimeofday(&t0, NULL);
    gettimeofday(&t_report, NULL);

    unsigned long long int last_data = 0ull;

    /* main loop */
    while (!should_exit) {
        while (state.qtail->next != state.qhead
               && (state.qtail->size = recv(sock_in, state.qtail->buf, SIZE, 0)) > 0
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
            double seconds = tv_diff(t, t_report);
            if (seconds > 5.0) {
                unsigned long long int cur_data = (received_data - last_data);
                unsigned long long int bps = cur_data / seconds;
                fprintf(stderr, "Received %llu bytes in %g seconds = %llu B/s.\n", cur_data, seconds, bps);
                t_report = t;
                last_data = received_data;
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

    pthread_join(thread, NULL);

    if(state.decompress) {
        hd_rum_decompress_done(state.decompress);
    }

    for (unsigned int i = 0; i < state.replicas.size(); i++) {
        replica_done(state.replicas[i]);
        delete state.replicas[i];
    }

    control_done(control_state);

#ifdef WIN32
    WSACleanup();
#endif

    printf("Exit\n");

    return 0;
}

/* vim: set sw=4 expandtab : */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "debug.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "control_socket.h"
#include "crypto/random.h"
#include "host.h"
#include "hd-rum-translator/hd-rum-recompress.h"
#include "hd-rum-translator/hd-rum-decompress.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rtp/net_udp.h"
#include "utils/misc.h"
#include "tv.h"

#include <map>
#include <string>
#include <vector>

using namespace std;

static pthread_t main_thread_id;
struct item;

#define REPLICA_MAGIC 0xd2ff3323

struct replica {
    replica(const char *addr, int tx_port, int bufsize, struct module *parent) {
        magic = REPLICA_MAGIC;
        host = addr;
        sock = udp_init(addr, 0, tx_port, 255, false, false);
        if (!sock) {
            throw string("Cannot initialize output port!\n");
        }
        if (udp_set_send_buf(sock, bufsize) != TRUE)
            fprintf(stderr, "Cannot set send buffer!\n");
        module_init_default(&mod);
        mod.cls = MODULE_CLASS_PORT;
        mod.priv_data = this;
        module_register(&mod, parent);
    }

    ~replica() {
        assert(magic == REPLICA_MAGIC);
        module_done(&mod);
        udp_exit(sock);
    }

    struct module mod;
    uint32_t magic;
    string host;

    enum type_t {
        NONE,
        USE_SOCK,
        RECOMPRESS
    };
    enum type_t type;
    socket_udp *sock;
    void *recompress;
};

struct hd_rum_translator_state {
    hd_rum_translator_state() : mod(), control_state(nullptr), queue(nullptr), qhead(nullptr),
            qtail(nullptr), qempty(1), qfull(0), decompress(nullptr) {
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
    struct control_state *control_state;
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
static struct item *qinit(int qsize);
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

struct response *change_replica_type(struct hd_rum_translator_state *s,
        struct module *mod, struct message *msg)
{
    struct replica *r = (struct replica *) mod->priv_data;

    struct msg_universal *data = (struct msg_universal *) msg;

    if (strcasecmp(data->text, "sock") == 0) {
        r->type = replica::type_t::USE_SOCK;
        log_msg(LOG_LEVEL_NOTICE, "Output port is now forwarding.\n");
    } else if (strcasecmp(data->text, "recompress") == 0) {
        r->type = replica::type_t::RECOMPRESS;
        log_msg(LOG_LEVEL_NOTICE, "Output port is now transcoding.\n");
    } else {
        fprintf(stderr, "Unknown replica type \"%s\"\n", data->text);
        return new_response(RESPONSE_BAD_REQUEST, NULL);
    }

    hd_rum_decompress_set_active(s->decompress, r->recompress,
            r->type == replica::type_t::RECOMPRESS);

    return new_response(RESPONSE_OK, NULL);
}

static void *writer(void *arg)
{
    struct hd_rum_translator_state *s =
        (struct hd_rum_translator_state *) arg;

    while (1) {
        // first check messages
        for (unsigned int i = 0; i < s->replicas.size(); i++) {
            struct message *msg;
            while ((msg = check_message(&s->replicas[i]->mod))) {
                struct response *r = change_replica_type(s, &s->replicas[i]->mod, msg);
                free_message(msg, r);
            }
        }

        struct msg_universal *msg;
        while ((msg = (struct msg_universal *) check_message(&s->mod))) {
            struct response *r = NULL;
            if (strncasecmp(msg->text, "delete-port ", strlen("delete-port ")) == 0) {
                int index = atoi(msg->text + strlen("delete-port "));
                hd_rum_decompress_remove_port(s->decompress, index);
                delete s->replicas[index];
                s->replicas.erase(s->replicas.begin() + index);
                log_msg(LOG_LEVEL_NOTICE, "Deleted output port %d.\n", index);
            } else if (strncasecmp(msg->text, "create-port", strlen("create-port")) == 0) {
                char *host, *save_ptr, *item;
                strtok_r(msg->text, " ", &save_ptr);
                host = strtok_r(NULL, " ", &save_ptr);
                item = strtok_r(NULL, " ", &save_ptr);
                if (!host || !item ) {
                    const char *err_msg = "wrong format";
                    log_msg(LOG_LEVEL_ERROR, "%s\n", err_msg);
                    free_message((struct message *) msg, new_response(RESPONSE_BAD_REQUEST, err_msg));
                    continue;
                }
                int tx_port = atoi(item);
                char *compress = strtok_r(NULL, " ", &save_ptr);
                struct replica *rep;
                try {
                    rep = new replica(host, tx_port, 100*1000, &s->mod);
                } catch (string const & s) {
                    fputs(s.c_str(), stderr);
                    const char *err_msg = "cannot create output port (wrong address?)";
                    log_msg(LOG_LEVEL_ERROR, "%s\n", err_msg);
                    free_message((struct message *) msg, new_response(RESPONSE_INT_SERV_ERR, err_msg));
                    continue;
                }
                s->replicas.push_back(rep);

                if (compress) {
                    rep->type = replica::type_t::RECOMPRESS;
                    char *fec = NULL;
                    rep->recompress = recompress_init(&rep->mod,
                            host, compress,
                            0, tx_port, 1500, fec, RATE_UNLIMITED);
                    if (!rep->recompress) {
                        delete s->replicas[s->replicas.size() - 1];
                        s->replicas.erase(s->replicas.end() - 1);

                        log_msg(LOG_LEVEL_ERROR, "Unable to create recompress!\n");
                    } else {
                        hd_rum_decompress_append_port(s->decompress, rep->recompress);
                        hd_rum_decompress_set_active(s->decompress, rep->recompress, true);
                        log_msg(LOG_LEVEL_NOTICE, "Created new transcoding output port.\n");
                    }
                } else {
                    rep->type = replica::type_t::USE_SOCK;
                    char compress[] = "none";
                    char *fec = NULL;
                    rep->recompress = recompress_init(&rep->mod,
                            host, compress,
                            0, tx_port, 1500, fec, RATE_UNLIMITED);
                    hd_rum_decompress_append_port(s->decompress, rep->recompress);
                    hd_rum_decompress_set_active(s->decompress, rep->recompress, false);
                    log_msg(LOG_LEVEL_NOTICE, "Created new forwarding output port.\n");
                }
            } else {
                r = new_response(RESPONSE_BAD_REQUEST, NULL);
            }

            free_message((struct message *) msg, r ? r : new_response(RESPONSE_OK, NULL));
        }

        // then process incoming packets
        while (s->qhead != s->qtail) {
            if(s->qhead->size == 0) { // poisoned pill
                return NULL;
            }
            for (unsigned int i = 0; i < s->replicas.size(); i++) {
                if(s->replicas[i]->type == replica::type_t::USE_SOCK) {
                    ssize_t ret = udp_send(s->replicas[i]->sock, s->qhead->buf, s->qhead->size);
                    if (ret < 0) {
                        perror("Hd-rum-translator send");
                    }
                }
            }

            if (hd_rum_decompress_get_num_active_ports(s->decompress) > 0) {
                ssize_t ret = hd_rum_decompress_write(s->decompress, s->qhead->buf, s->qhead->size);
                if (ret < 0) {
                    perror("hd_rum_decompress_write");
                }
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
                "\t\t--blend - enable blending from original to newly received stream, increases latency\n"
                "\t\t--capture-filter <cfg_string> - apply video capture filter to incoming video\n"
                "\t\t--help\n");
        printf("\tand hostX_options may be:\n"
                "\t\t-P <port> - TX port to be used\n"
                "\t\t-c <compression> - compression\n"
                "\t\tFollowing options will be used only if '-c' parameter is set:\n"
                "\t\t-m <mtu> - MTU size\n"
                "\t\t-l <limiting_bitrate> - bitrate to be shaped to\n"
                "\t\t-f <fec> - FEC that will be used for transmission.\n"
              );
        printf("\tPlease note that blending and capture filter is used only for host for which\n"
               "\tcompression is specified (transcoding is active). If compression is not\n"
               "\tset, simple packet retransmission is used. Compression can be also 'none'\n"
               "\tfor uncompressed transmission (see 'uv -c help' for list).\n");
}

struct host_opts {
    char *addr;
    int port;
    int mtu;
    char *compression;
    char *fec;
    int64_t bitrate;
};

struct cmdline_parameters {
    const char *bufsize;
    unsigned short port;
    struct host_opts *hosts;
    int host_count;
    int control_port = CONTROL_DEFAULT_PORT;
    int control_connection_type = 0;
    bool blend = false;
    const char *capture_filter = NULL;
};

/**
 * @todo
 * Use rather getopt() than manual parsing.
 */
static bool parse_fmt(int argc, char **argv, struct cmdline_parameters *parsed)
{
    int start_index = 1;

    while(start_index < argc && argv[start_index][0] == '-') {
        if(strcmp(argv[start_index], "--control-port") == 0) {
            char *item = argv[++start_index];
            parsed->control_port = atoi(item);
            parsed->control_connection_type = 0;
            if (strchr(item, ':')) {
                parsed->control_connection_type = atoi(strchr(item, ':') + 1);
            }
        } else if(strcmp(argv[start_index], "--capabilities") == 0) {
            print_capabilities(CAPABILITY_COMPRESS, NULL, false);
            return false;
        } else if(strcmp(argv[start_index], "--blend") == 0) {
            parsed->blend = true;
        } else if(strcmp(argv[start_index], "--capture-filter") == 0) {
            parsed->capture_filter = argv[++start_index];
        } else if(strcmp(argv[start_index], "--help") == 0) {
            usage(argv[0]);
            return false;
        } else {
            usage(argv[0]);
            return false;
        }
        start_index++;
    }

    if (argc < start_index + 2) {
        usage(argv[0]);
        return false;
    }

    parsed->bufsize = argv[start_index];
    parsed->port = atoi(argv[start_index + 1]);

    argv += start_index + 1;
    argc -= start_index + 1;

    parsed->host_count = 0;

    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            parsed->host_count += 1;
        } else {
            if(strlen(argv[i]) != 2) {
                fprintf(stderr, "Error: invalild option '%s'\n", argv[i]);
                exit(EXIT_FAIL_USAGE);
            }

            if(i == argc - 1 || argv[i + 1][0] == '-') {
                fprintf(stderr, "Error: option '-%c' requires an argument\n", argv[i][1]);
                exit(EXIT_FAIL_USAGE);
            }

            parsed->host_count -= 1; // because option argument (mandatory) will be counted as a host
                                   // in next iteration
        }
    }

    parsed->hosts = (struct host_opts *) calloc(parsed->host_count, sizeof(struct host_opts));
    // default values
    for(int i = 0; i < parsed->host_count; ++i) {
        parsed->hosts[i].bitrate = RATE_UNLIMITED;
        parsed->hosts[i].mtu = 1500;
    }

    int host_idx = 0;
    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch(argv[i][1]) {
                case 'P':
                    parsed->hosts[host_idx].port = atoi(argv[i + 1]);
                    break;
                case 'm':
                    parsed->hosts[host_idx].mtu = atoi(argv[i + 1]);
                    break;
                case 'c':
                    parsed->hosts[host_idx].compression = argv[i + 1];
                    break;
                case 'f':
                    parsed->hosts[host_idx].fec = argv[i + 1];
                    break;
                case 'l':
                    if (strcmp(argv[i + 1], "unlimited") == 0) {
                        parsed->hosts[host_idx].bitrate = RATE_UNLIMITED;
                    } else if (strcmp(argv[i + 1], "auto") == 0) {
                        parsed->hosts[host_idx].bitrate = RATE_AUTO;
                    } else {
                        parsed->hosts[host_idx].bitrate = unit_evaluate(argv[i + 1]);
                    }
                    break;
                default:
                    fprintf(stderr, "Error: invalild option '%s'\n", argv[i]);
                    exit(EXIT_FAIL_USAGE);
            }
            i += 1;
        } else {
            parsed->hosts[host_idx].addr = argv[i];
            host_idx += 1;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    struct hd_rum_translator_state state;

    int qsize;
    int bufsize;
    socket_udp *sock_in;
    pthread_t thread;
    int err = 0;
    int i;
    struct cmdline_parameters params;

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

    open_all("module_*.so"); // load modules

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

    bool ret = parse_fmt(argc, argv, &params);

    if (ret == false) {
        return EXIT_SUCCESS;
    }

    if ((bufsize = atoi(params.bufsize)) <= 0) {
        fprintf(stderr, "invalid buffer size: %d\n", bufsize);
        return 1;
    }
    switch (params.bufsize[strlen(params.bufsize) - 1]) {
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

    if (params.port <= 0) {
        fprintf(stderr, "invalid port: %d\n", params.port);
        return 1;
    }

    state.qhead = state.qtail = state.queue = qinit(qsize);

    /* input socket */
    if ((sock_in = udp_init_if("::1", NULL, params.port, 0, 255, false, false)) == NULL) {
        perror("input socket");
        return 2;
    }

    if (udp_set_recv_buf(sock_in, bufsize) != TRUE) {
        fprintf(stderr, "Cannot set recv buffer!\n");
    }

    printf("listening on *:%d\n", params.port);

    state.replicas.resize(params.host_count);

    if(control_init(params.control_port, params.control_connection_type, &state.control_state, &state.mod) != 0) {
        fprintf(stderr, "Warning: Unable to create remote control.\n");
        return EXIT_FAILURE;
    }
    control_start(state.control_state);

    // we need only one shared receiver decompressor for all recompressing streams
    state.decompress = hd_rum_decompress_init(&state.mod, params.blend, params.capture_filter);
    if(!state.decompress) {
        return EXIT_FAIL_DECODER;
    }

    for (i = 0; i < params.host_count; i++) {
        int packet_rate;
        if (params.hosts[i].bitrate != RATE_AUTO && params.hosts[i].bitrate != RATE_UNLIMITED) {
                packet_rate = compute_packet_rate(params.hosts[i].bitrate, params.hosts[i].mtu);
        } else {
                packet_rate = params.hosts[i].bitrate;
        }

        int tx_port = params.port;
        if(params.hosts[i].port) {
            tx_port = params.hosts[i].port;
        }

        try {
            state.replicas[i] = new replica(params.hosts[i].addr, tx_port, bufsize, &state.mod);
        } catch (string const &s) {
            fputs(s.c_str(), stderr);
            return EXIT_FAILURE;
        }

        if(params.hosts[i].compression == NULL) {
            state.replicas[i]->type = replica::type_t::USE_SOCK;
            char compress[] = "none";
            char *fec = NULL;
            state.replicas[i]->recompress = recompress_init(&state.replicas[i]->mod,
                    params.hosts[i].addr, compress,
                    0, tx_port, params.hosts[i].mtu, fec, packet_rate);
            hd_rum_decompress_append_port(state.decompress, state.replicas[i]->recompress);
            hd_rum_decompress_set_active(state.decompress, state.replicas[i]->recompress, false);
        } else {
            state.replicas[i]->type = replica::type_t::RECOMPRESS;

            state.replicas[i]->recompress = recompress_init(&state.replicas[i]->mod,
                    params.hosts[i].addr, params.hosts[i].compression,
                    0, tx_port, params.hosts[i].mtu, params.hosts[i].fec, packet_rate);
            if(state.replicas[i]->recompress == 0) {
                fprintf(stderr, "Initializing output port '%s' failed!\n",
                        params.hosts[i].addr);
                return EXIT_FAILURE;
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

    uint64_t received_data = 0;
    struct timeval t0;
    gettimeofday(&t0, NULL);

    unsigned long long int last_data = 0ull;

    /* main loop */
    while (!should_exit) {
        while (state.qtail->next != state.qhead
               && (state.qtail->size = udp_recv(sock_in, state.qtail->buf, SIZE)) > 0
               && !should_exit) {
            received_data += state.qtail->size;

            state.qtail = state.qtail->next;

            pthread_mutex_lock(&state.qempty_mtx);
            state.qempty = 0;
            pthread_cond_signal(&state.qempty_cond);
            pthread_mutex_unlock(&state.qempty_mtx);

            struct timeval t;
            gettimeofday(&t, NULL);
            double seconds = tv_diff(t, t0);
            if (seconds > 5.0) {
                unsigned long long int cur_data = (received_data - last_data);
                unsigned long long int bps = cur_data / seconds;
                fprintf(stderr, "Received %llu bytes in %g seconds = %llu B/s.\n", cur_data, seconds, bps);
                t0 = t;
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

    pthread_join(thread, NULL);

    if(state.decompress) {
        hd_rum_decompress_done(state.decompress);
    }

    for (unsigned int i = 0; i < state.replicas.size(); i++) {
        delete state.replicas[i];
    }

    control_done(state.control_state);

    udp_exit(sock_in);

#ifdef WIN32
    WSACleanup();
#endif

    printf("Exit\n");

    return 0;
}

/* vim: set sw=4 expandtab : */

/**
 * @file   hd-rum-translator/hd-rum-translator.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 *
 * Main part of transcoding reflector. This component provides a runtime
 * for the reflector. Componets are following:
 * - legacy packet reflector (defined in this file) for backward compatiblity.
 *   It handles those recipient that doesn't need transcoding.
 * - decompressor - decompresses the stream if there are some host that need
 *   transcoding
 * - recompressor - for every transcoded host, there is a recompressor that
 *   compresses and sends frame to receiver
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
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

#include "compat/platform_time.h"
#include "control_socket.h"
#include "crypto/random.h"
#include "host.h"
#include "hd-rum-translator/hd-rum-recompress.h"
#include "hd-rum-translator/hd-rum-decompress.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rang.hpp"
#include "rtp/net_udp.h"
#include "utils/misc.h" // format_in_si_units, unit_evaluate
#include "tv.h"
#include "utils/net.h"

#include <cinttypes>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using s = rang::style;
using fg = rang::fg;
constexpr const char *MOD_NAME = "[hd-rum-trans] ";

struct item;

#define REPLICA_MAGIC 0xd2ff3323

static char *get_replica_mod_name(const char *addr, uint16_t tx_port){
        const size_t len = strlen(addr) + 2 /* [ ] for IPv6 addr */ + 6 /* port (including ':') */ + 1 /* '\0' */;
        char *name = (char *) malloc(len);
        bool is_ipv6 = strchr(addr, ':') != NULL;
        bool add_bracket = is_ipv6 && addr[0] != '[';
        sprintf(name, "%s%s%s:%" PRIu16, add_bracket ? "[" : "", addr, add_bracket ? "]" : "", tx_port);

        return name;
}

struct replica {
    replica(const char *addr, uint16_t rx_port, uint16_t tx_port, int bufsize, struct module *parent, int force_ip_version) {
        magic = REPLICA_MAGIC;
        host = addr;
        m_tx_port = tx_port;
        sock = udp_init(addr, rx_port, tx_port, 255, force_ip_version, false);
        if (!sock) {
            throw string("Cannot initialize output port!\n");
        }
        if (udp_set_send_buf(sock, bufsize) != TRUE)
            fprintf(stderr, "Cannot set send buffer!\n");
        module_init_default(&mod);
        mod.cls = MODULE_CLASS_PORT;
        mod.name = get_replica_mod_name(addr, tx_port);
        mod.priv_data = this;
        module_register(&mod, parent);
        type = replica::type_t::NONE;
    }

    ~replica() {
        assert(magic == REPLICA_MAGIC);
        module_done(&mod);
        udp_exit(sock);
    }

    struct module mod;
    uint32_t magic;
    string host;
    int m_tx_port;

    enum type_t {
        NONE,
        USE_SOCK,
        RECOMPRESS
    };
    enum type_t type;
    socket_udp *sock;
};

struct hd_rum_translator_state {
    hd_rum_translator_state() {
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
    struct control_state *control_state = nullptr;
    struct item *queue = nullptr;
    struct item *qhead = nullptr;
    struct item *qtail = nullptr;
    int qempty = 1;
    int qfull = 0;
    pthread_mutex_t qempty_mtx;
    pthread_mutex_t qfull_mtx;
    pthread_cond_t qempty_cond;
    pthread_cond_t qfull_cond;

    vector<replica *> replicas;
    void *decompress = nullptr;
    struct state_recompress *recompress = nullptr;
};

/*
 * Prototypes
 */
static struct item *qinit(int qsize);
static void qdestroy(struct item *queue);
static void *writer(void *arg);
static void signal_handler(int signal);
void exit_uv(int status);

/*
 * this is currently only placeholder to substitute UG default
 */
void exit_uv(int status) {
    UNUSED(status);
    should_exit = true;
}

static void signal_handler(int signal)
{
    if (log_level >= LOG_LEVEL_DEBUG) {
	char msg[] = "Caught signal ";
	char buf[128];
	char *ptr = buf;
	for (size_t i = 0; i < sizeof msg - 1; ++i) {
	    *ptr++ = msg[i];
	}
	if (signal / 10) {
	    *ptr++ = '0' + signal/10;
	}
	*ptr++ = '0' + signal%10;
	*ptr++ = '\n';
	size_t bytes = ptr - buf;
	ptr = buf;
	do {
	    ssize_t written = write(STDERR_FILENO, ptr, bytes);
	    if (written < 0) {
		break;
	    }
	    bytes -= written;
	    ptr += written;
	} while (bytes > 0);
    }
    exit_uv(0);
}

#define MAX_PKT_SIZE 10000

#ifdef WIN32
struct wsa_aux_storage {
    WSAOVERLAPPED *overlapped;
    int ref;
};
#define ALIGNMENT std::alignment_of<wsa_aux_storage>::value
#define OFFSET ((MAX_PKT_SIZE + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT)
#define SIZE (OFFSET + sizeof(wsa_aux_storage))
#else
#define SIZE MAX_PKT_SIZE
#endif

struct item {
    struct item *next;
    long size;
    char *buf;
};

static struct item *qinit(int qsize)
{
    struct item *queue;
    int i;

    if (qsize <= 0) {
        fprintf(stderr, "wrong packet queue size %d items\n", qsize);
        return nullptr;
    }

    printf("initializing packet queue for %d items\n", qsize);

    queue = (struct item *) calloc(qsize, sizeof(struct item));
    if (queue == NULL) {
        fprintf(stderr, "not enough memory\n");
        exit(2);
    }

    for (i = 0; i < qsize; i++) {
        queue[i].buf = (char *) malloc(SIZE);
        queue[i].next = queue + i + 1;
    }
    queue[qsize - 1].next = queue;

    return queue;
}

static void qdestroy(struct item *queue)
{
    if (queue == nullptr) {
        return;
    }

    struct item *q = queue;
    do {
        free(q->buf);
        q = q->next;
    } while (q != queue);
    free(queue);
}

#define prefix_matches(x,y) strncasecmp(x, y, strlen(y)) == 0
static struct response *change_replica_type(struct hd_rum_translator_state *s,
        struct module *mod, struct message *msg, int index)
{
    struct replica *r = (struct replica *) mod->priv_data;

    struct msg_universal *data = (struct msg_universal *) msg;

    if (strcasecmp(data->text, "sock") == 0) {
        r->type = replica::type_t::USE_SOCK;
        log_msg(LOG_LEVEL_NOTICE, "Output port %d is now forwarding.\n", index);
    } else if (strcasecmp(data->text, "recompress") == 0) {
        r->type = replica::type_t::RECOMPRESS;
        log_msg(LOG_LEVEL_NOTICE, "Output port %d is now transcoding.\n", index);
    } else if (prefix_matches(data->text, "compress ")) {
        if(recompress_port_change_compress(s->recompress, index, data->text + strlen("compress "))){
            log_msg(LOG_LEVEL_NOTICE, "Output port %d compression changed.\n", index);
        } else {
            log_msg(LOG_LEVEL_ERROR, "Failed to change port %d compression. Port removed.\n", index);
        }
    } else {
        fprintf(stderr, "Unknown replica type \"%s\"\n", data->text);
        return new_response(RESPONSE_BAD_REQUEST, NULL);
    }

    recompress_port_set_active(s->recompress, index,
            r->type == replica::type_t::RECOMPRESS);

    return new_response(RESPONSE_OK, NULL);
}

#ifdef WIN32
static VOID CALLBACK wsa_deleter(DWORD /* dwErrorCode */,
        DWORD /* dwNumberOfBytesTransfered */,
        LPOVERLAPPED lpOverlapped, long unsigned int) {
    struct wsa_aux_storage *aux = (struct wsa_aux_storage *)(void *) ((char *) lpOverlapped->hEvent + OFFSET);
    if (--aux->ref == 0) {
        free(aux->overlapped);
        free(lpOverlapped->hEvent);
    }
}
#endif

static int create_output_port(struct hd_rum_translator_state *s,
        const char *addr, int rx_port, int tx_port, int bufsize, bool force_ip_version,
        const char *compression, int mtu, const char *fec, int bitrate)
{
        struct replica *rep;
        try {
            rep = new replica(addr, rx_port, tx_port, bufsize, &s->mod, force_ip_version);
        } catch (string const & s) {
            fputs(s.c_str(), stderr);
            const char *err_msg = "cannot create output port (wrong address?)";
            log_msg(LOG_LEVEL_ERROR, "%s\n", err_msg);
            return -1;
        }
        s->replicas.push_back(rep);

        rep->type = compression ? replica::type_t::RECOMPRESS : replica::type_t::USE_SOCK;
        int idx = recompress_add_port(s->recompress, &rep->mod,
                addr, compression ? compression : "none",
                0, tx_port, mtu, fec, bitrate);
        if (idx < 0) {
            fprintf(stderr, "Initializing output port '%s' compression failed!\n", addr);

            delete s->replicas.back();
            s->replicas.pop_back();

            return -1;
        }

        assert((unsigned) idx == s->replicas.size() - 1);
        recompress_port_set_active(s->recompress, idx, compression != nullptr);

        return idx;
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
                struct response *r = change_replica_type(s, &s->replicas[i]->mod, msg, i);
                free_message(msg, r);
            }
        }

        struct msg_universal *msg;
        while ((msg = (struct msg_universal *) check_message(&s->mod))) {
            struct response *r = NULL;
            if (strncasecmp(msg->text, "delete-port ", strlen("delete-port ")) == 0) {
                char *port_spec = msg->text + strlen("delete-port ");
                int index = -1;
                if (isdigit(port_spec[0])) {
                    int i = atoi(port_spec);
                    if (i >= 0 && i < (int) s->replicas.size()) {
                        index = i;
                    } else {
                        log_msg(LOG_LEVEL_WARNING, "Invalid port index: %d. Not removing.\n", i);
                    }
                } else {
                    int i = 0;
                    for (auto r : s->replicas) {
                        if (strcmp(r->mod.name, port_spec) == 0) {
                            index = i;
                            break;
                        }
                        i++;
                    }
                    if (index == -1) {
                        log_msg(LOG_LEVEL_WARNING, "Unknown port name: %s. Not removing.\n", port_spec);
                    }
                }
                if (index >= 0) {
                    recompress_remove_port(s->recompress, index);
                    delete s->replicas[index];
                    s->replicas.erase(s->replicas.begin() + index);
                    log_msg(LOG_LEVEL_NOTICE, "Deleted output port %d.\n", index);
                }
            } else if (strncasecmp(msg->text, "create-port", strlen("create-port")) == 0) {
                // format of parameters is either:
                // <host>:<port> [<compression>]
                // or (for compat with older CoUniverse version)
                // <host> <port> [<compression>]
                char *host_port, *port_str = NULL, *save_ptr;
                char *host;
                int tx_port;
                strtok_r(msg->text, " ", &save_ptr);
                host_port = strtok_r(NULL, " ", &save_ptr);
                if (host_port && (strchr(host_port, ':') != NULL || (port_str = strtok_r(NULL, " ", &save_ptr)) != NULL)) {
                    if (port_str) {
                        host = host_port;
                        tx_port = atoi(port_str);
                    } else {
                        tx_port = atoi(strrchr(host_port, ':') + 1);
                        host = host_port;
                        *strrchr(host_port, ':') = '\0';
                    }
                    // handle square brackets around an IPv6 address
                    if (host[0] == '[' && host[strlen(host) - 1] == ']') {
                        host += 1;
                        host[strlen(host) - 1] = '\0';
                    }
                } else {
                    const char *err_msg = "wrong format";
                    log_msg(LOG_LEVEL_ERROR, "%s\n", err_msg);
                    free_message((struct message *) msg, new_response(RESPONSE_BAD_REQUEST, err_msg));
                    continue;
                }
                char *compress = strtok_r(NULL, " ", &save_ptr);

                int idx = create_output_port(s,
                        host, 0, tx_port, 100*1000, false,
                        compress, 1500, nullptr, RATE_UNLIMITED);

                if(idx < 0) {
                    free_message((struct message *) msg, new_response(RESPONSE_INT_SERV_ERR, "Cannot create output port."));
                    continue;
                }

                if(compress)
                    log_msg(LOG_LEVEL_NOTICE, "Created new transcoding output port %s:%d:0x%08" PRIx32 ".\n", host, tx_port, recompress_get_port_ssrc(s->recompress, idx));
                else
                    log_msg(LOG_LEVEL_NOTICE, "Created new forwarding output port %s:%d.\n", host, tx_port);

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

            // pass it for transcoding if needed
            if (recompress_get_num_active_ports(s->recompress) > 0) {
                ssize_t ret = hd_rum_decompress_write(s->decompress, s->qhead->buf, s->qhead->size);
                if (ret < 0) {
                    perror("hd_rum_decompress_write");
                }
            }

            // distribute it to output ports that don't need transcoding
#ifdef WIN32
            // send it asynchronously in MSW (performance optimalization)
            SleepEx(0, TRUE); // allow system to call our completion routines in APC
            int ref = 0;
            for (unsigned int i = 0; i < s->replicas.size(); i++) {
                if(s->replicas[i]->type == replica::type_t::USE_SOCK) {
                    ref++;
                }
            }
            struct wsa_aux_storage *aux = (struct wsa_aux_storage *)(void *) ((char *) s->qhead->buf + OFFSET);
            memset(aux, 0, sizeof *aux);
            aux->overlapped = (WSAOVERLAPPED *) calloc(ref, sizeof(WSAOVERLAPPED));
            aux->ref = ref;
            int overlapped_idx = 0;
            for (unsigned int i = 0; i < s->replicas.size(); i++) {
                if(s->replicas[i]->type == replica::type_t::USE_SOCK) {
                    aux->overlapped[overlapped_idx].hEvent = s->qhead->buf;
                    ssize_t ret = udp_send_wsa_async(s->replicas[i]->sock, s->qhead->buf, s->qhead->size, wsa_deleter, &aux->overlapped[overlapped_idx]);
                    if (ret < 0) {
                        perror("Hd-rum-translator send");
                    }
                    overlapped_idx += 1;
                }
            }
            // reallocate the buffer since the last one will be freeed automaticaly
            s->qhead->buf = (char *) malloc(SIZE);
#else
            for (unsigned int i = 0; i < s->replicas.size(); i++) {
                if(s->replicas[i]->type == replica::type_t::USE_SOCK) {
                    ssize_t ret = udp_send(s->replicas[i]->sock, s->qhead->buf, s->qhead->size);
                    if (ret < 0) {
                        perror("Hd-rum-translator send");
                    }
                }
            }
#endif
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
        cout << s::bold << fg::red << progname << fg::reset <<
            " [global_opts] buffer_size port [host1_options] host1 [[host2_options] host2] ...\n"
            << s::reset;
        cout << "\twhere " << s::underline << "global_opts" << s::reset << " may be:\n" <<
                s::bold << "\t\t--control-port <port_number>[:0|:1]" << s::reset << " - control port to connect to, optionally client/server (default)\n" <<
                s::bold << "\t\t--blend" << s::reset << " - enable blending from original to newly received stream, increases latency\n" <<
                s::bold << "\t\t--conference <width>:<height>[:fps]" << s::reset << " - enable combining of multiple inputs, increases latency\n" <<
                s::bold << "\t\t--conference-compression <compression>" << s::reset << " - compression for conference participants\n" <<
                s::bold << "\t\t--capture-filter <cfg_string>" << s::reset << " - apply video capture filter to incoming video\n" <<
                s::bold << "\t\t--param" << s::reset << " - additional parameters\n" <<
                s::bold << "\t\t--help\n" << s::reset <<
                s::bold << "\t\t--verbose\n" << s::reset <<
                s::bold << "\t\t-v" << s::reset << " - print version\n";
        cout << "\tand " << s::underline<< "hostX_options" << s::reset << " may be:\n" <<
                s::bold << "\t\t-P [<rx_port>:]<tx_port>" << s::reset << " - TX port to be used (optionally also RX)\n" <<
                s::bold << "\t\t-c <compression>" << s::reset << " - compression\n" <<
                "\t\tFollowing options will be used only if " << s::underline << "'-c'" << s::reset << " parameter is set:\n" <<
                s::bold << "\t\t-m <mtu>" << s::reset << " - MTU size\n" <<
                s::bold << "\t\t-l <limiting_bitrate>" << s::reset << " - bitrate to be shaped to\n" <<
                s::bold << "\t\t-f <fec>" << s::reset << " - FEC that will be used for transmission.\n" <<
                s::bold << "\t\t-6" << s::reset << " - use IPv6\n";
        printf("\tPlease note that blending and capture filter is used only for host for which\n"
               "\tcompression is specified (transcoding is active). If compression is not\n"
               "\tset, simple packet retransmission is used. Compression can be also 'none'\n"
               "\tfor uncompressed transmission (see 'uv -c help' for list).\n");
}

struct host_opts {
    char *addr;
    int rx_port;
    int tx_port;
    int mtu;
    char *compression;
    char *fec;
    int64_t bitrate;
    int force_ip_version;
};

struct cmdline_parameters {
    const char *bufsize;
    unsigned short port;
    vector<struct host_opts> hosts;
    int host_count;
    int control_port = -1;
    int control_connection_type = 0;
    struct hd_rum_output_conf out_conf = {NORMAL, NULL};
    const char *capture_filter = NULL;
    bool verbose = false;
    char *conference_compression = nullptr;
};

/**
 * @todo
 * Use rather getopt() than manual parsing.
 * @retval -1 failure
 * @retval 0 success
 * @retval 1 help shown
 */
static int parse_fmt(int argc, char **argv, struct cmdline_parameters *parsed)
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
            print_capabilities(NULL, false);
            return 1;
        } else if(strcmp(argv[start_index], "--blend") == 0) {
            parsed->out_conf.mode = BLEND;
        } else if(strcmp(argv[start_index], "--conference") == 0) {
            parsed->out_conf.mode = CONFERENCE;
            char *item = argv[++start_index];
            parsed->out_conf.arg = item;
        } else if(strcmp(argv[start_index], "--conference-compression") == 0) {
            char *item = argv[++start_index];
            parsed->conference_compression = item;
        } else if(strcmp(argv[start_index], "--capture-filter") == 0) {
            parsed->capture_filter = argv[++start_index];
        } else if(strcmp(argv[start_index], "-h") == 0 || strcmp(argv[start_index], "--help") == 0) {
            usage(argv[0]);
            return 1;
        } else if(strcmp(argv[start_index], "-v") == 0) {
            return 1;
        } else if(strcmp(argv[start_index], "--verbose") == 0) {
            parsed->verbose = true;
        } else if(strcmp(argv[start_index], "--param") == 0 && start_index < argc - 1) {
            // already handled in common_preinit()
        } else {
            LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Unknown global parameter: " << argv[start_index] << "\n\n";
            usage(argv[0]);
            return -1;
        }
        start_index++;
    }

    if (argc < start_index + 2) {
        LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Missing mandatory parameters!\n\n";
        usage(argv[0]);
        return -1;
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
                fprintf(stderr, "Error: invalid option '%s'\n", argv[i]);
                exit(EXIT_FAIL_USAGE);
            }

            if(i == argc - 1 || argv[i + 1][0] == '-') {
                fprintf(stderr, "Error: option '-%c' requires an argument\n", argv[i][1]);
                return -1;
            }

            parsed->host_count -= 1; // because option argument (mandatory) will be counted as a host
                                   // in next iteration
        }
    }

    parsed->hosts.resize(parsed->host_count);
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
                    if (strchr(argv[i + 1], ':')) {
                        parsed->hosts[host_idx].rx_port = atoi(argv[i + 1]);
                        parsed->hosts[host_idx].tx_port = atoi(strchr(argv[i + 1], ':') + 1);
                    } else {
                        parsed->hosts[host_idx].tx_port = atoi(argv[i + 1]);
                    }
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
                        if (parsed->hosts[host_idx].bitrate <= 0) {
                            LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Error: wrong bitrate - " << argv[i + 1] << "\n";
                            exit(EXIT_FAIL_USAGE);
                        }
                    }
                    break;
                case '4':
                    parsed->hosts[host_idx].force_ip_version = 4;
                    break;
                case '6':
                    parsed->hosts[host_idx].force_ip_version = 6;
                    break;
                default:
                    LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Error: invalid host option " << argv[i] << "\n";
                    exit(EXIT_FAIL_USAGE);
            }
            i += 1;
        } else {
            parsed->hosts[host_idx].addr = argv[i];
            host_idx += 1;
        }
    }

    return 0;
}

static void hd_rum_translator_deinit(struct hd_rum_translator_state *s) {
    if(s->decompress) {
        hd_rum_decompress_done(s->decompress);
    }

    if(s->recompress)
            recompress_done(s->recompress);

    for (unsigned int i = 0; i < s->replicas.size(); i++) {
        delete s->replicas[i];
    }

    control_done(s->control_state);

    qdestroy(s->queue);
}

struct Conf_participant{
        struct sockaddr_storage addr;
        socklen_t addrlen;
        struct timeval last_recv;
};

static bool sockaddr_equal(struct sockaddr *a, struct sockaddr *b){
        if(a->sa_family != b->sa_family)
                return false;

        switch(a->sa_family){
        case AF_INET:{
                auto a_in = reinterpret_cast<struct sockaddr_in *>(a);
                auto b_in = reinterpret_cast<struct sockaddr_in *>(b);
                return a_in->sin_port == b_in->sin_port
                        && a_in->sin_addr.s_addr == b_in->sin_addr.s_addr;
        }
        case AF_INET6:{
                auto a_in = reinterpret_cast<struct sockaddr_in6 *>(a);
                auto b_in = reinterpret_cast<struct sockaddr_in6 *>(b);
                return a_in->sin6_port == b_in->sin6_port
                        && memcmp(a_in->sin6_addr.s6_addr, b_in->sin6_addr.s6_addr, sizeof(struct in6_addr)) == 0;
        }
        default:
                assert(false && "Comparison not implemented for address type");
                abort();
        }
}

#define EXIT(retval) { hd_rum_translator_deinit(&state); if (sock_in != nullptr) udp_exit(sock_in); common_cleanup(init); return retval; }
int main(int argc, char **argv)
{
    struct init_data *init;
    struct hd_rum_translator_state state;

    int qsize;
    int bufsize;
    socket_udp *sock_in = nullptr;
    pthread_t thread;
    int err = 0;
    int i;
    struct cmdline_parameters params = {};

    if ((init = common_preinit(argc, argv)) == nullptr) {
        EXIT(EXIT_FAILURE);
    }

    print_version();
    printf("\n");

    if (argc == 1) {
        usage(argv[0]);
        EXIT(EXIT_FAILURE);
    }

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

    int ret = parse_fmt(argc, argv, &params);

    if (ret != 0) {
        EXIT(ret < 0 ? EXIT_FAILURE : EXIT_SUCCESS);
    }

    if (params.verbose) {
        log_level = LOG_LEVEL_VERBOSE;
    }

    if ((bufsize = atoi(params.bufsize)) <= 0) {
        fprintf(stderr, "invalid buffer size: %d\n", bufsize);
        EXIT(1);
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
        EXIT(EXIT_FAIL_USAGE);
    }

    state.qhead = state.qtail = state.queue = qinit(qsize);
    if (!state.qhead) {
        EXIT(EXIT_FAILURE);
    }

    /* input socket */
    if ((sock_in = udp_init_if("localhost", NULL, params.port, 0, 255, false, false)) == NULL) {
        perror("input socket");
        EXIT(EXIT_FAILURE);
    }

    if (udp_set_recv_buf(sock_in, bufsize) != TRUE) {
        fprintf(stderr, "Cannot set recv buffer!\n");
    }

    printf("listening on *:%d\n", params.port);


    if (params.control_port != -1) {
        if (control_init(params.control_port, params.control_connection_type, &state.control_state, &state.mod, 0) != 0) {
            fprintf(stderr, "Warning: Unable to create remote control.\n");
            EXIT(EXIT_FAIL_CONTROL_SOCK);
        }
        control_start(state.control_state);
    }

	//one shared recompressor, which manages compressions and sends to all hosts
	state.recompress = recompress_init(&state.mod);
    if(!state.recompress) {
        EXIT(EXIT_FAIL_COMPRESS);
    }

    // we need only one shared receiver decompressor for all recompressing streams
    state.decompress = hd_rum_decompress_init(&state.mod, params.out_conf,
			params.capture_filter, state.recompress);
    if(!state.decompress) {
        EXIT(EXIT_FAIL_DECODER);
    }

    for (i = 0; i < params.host_count; i++) {
        int tx_port = params.port; // use default rx port by default
        int rx_port = params.hosts[i].rx_port;
        if (params.hosts[i].tx_port) {
            tx_port = params.hosts[i].tx_port;
        }

        const auto& h = params.hosts[i];

        int idx = create_output_port(&state,
                h.addr, rx_port, tx_port, bufsize, h.force_ip_version,
                h.compression, h.mtu, h.compression ? h.fec : nullptr, h.bitrate);
        if(idx < 0) {
            EXIT(EXIT_FAILURE);
        }
    }

    if (pthread_create(&thread, NULL, writer, (void *) &state)) {
        fprintf(stderr, "cannot create writer thread\n");
        EXIT(2);
    }

    uint64_t received_data = 0;
    struct timeval t0;
    gettimeofday(&t0, NULL);

    unsigned long long int last_data = 0ull;

    std::vector<Conf_participant> participants;

    /* main loop */
    while (!should_exit) {
        while (state.qtail->next != state.qhead && !should_exit) {
            struct timeval timeout = { 1, 0 };

            struct sockaddr_storage sin = {};
            socklen_t addrlen = sizeof(sin);
            state.qtail->size = udp_recvfrom_timeout(sock_in, state.qtail->buf, SIZE, &timeout, (sockaddr *) &sin, &addrlen);
            if(state.qtail->size <= 0)
                break;

            struct timeval t;
            gettimeofday(&t, NULL);

            if(params.out_conf.mode == CONFERENCE){
                    bool seen = false;
                    for(auto it = participants.begin(); it != participants.end();){
                            if(addrlen > 0 && sockaddr_equal((sockaddr *) &it->addr, (sockaddr *) &sin)){
                                    it->last_recv = t;
                                    seen = true;
                            }
                            const double participant_timeout = 10.0;
                            if(tv_diff(t, it->last_recv) > participant_timeout){
                                    std::swap(*it, participants.back());
                                    participants.pop_back();
                                    log_msg(LOG_LEVEL_NOTICE, "Removing participant\n");
                                    std::string msg = "delete-port ";
                                    auto addr = reinterpret_cast<struct sockaddr *>(&it->addr);
                                    char addr_str[128];
                                    get_sockaddr_addr_str(addr, addr_str, sizeof(addr_str));
                                    msg += get_replica_mod_name(addr_str, get_sockaddr_addr_port(addr));

                                    struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                                    strncpy(m->text, msg.c_str(), sizeof(m->text) - 1);
                                    log_msg(LOG_LEVEL_NOTICE, "Msg: %s\n", m->text);
                                    struct response *r = send_message_to_receiver(&state.mod, (struct message *) m);
                                    if (response_get_status(r) != RESPONSE_ACCEPTED) {
                                            log_msg(LOG_LEVEL_ERROR, "Couldn't remove participant (error %d)!\n", response_get_status(r));
                                    }
                                    free_response(r);
                            } else {
                                    it++;
                            }
                    }

                    if(!seen && addrlen > 0){
                            Conf_participant p;
                            p.addr = sin;
                            p.addrlen = addrlen;
                            p.last_recv = t;

                            log_msg(LOG_LEVEL_NOTICE, "New participant\n");
                            std::string msg = "create-port ";
                            msg += get_sockaddr_str(reinterpret_cast<sockaddr *>(&sin));
                            if(params.conference_compression){
                                    msg += " ";
                                    msg += params.conference_compression;
                            } else {
                                    msg += " libavcodec";
                            }

                            struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                            strncpy(m->text, msg.c_str(), sizeof(m->text) - 1);
                            log_msg(LOG_LEVEL_NOTICE, "Msg: %s\n", m->text);
                            struct response *r = send_message_to_receiver(&state.mod, (struct message *) m);
                            if (response_get_status(r) != RESPONSE_ACCEPTED) {
                                    log_msg(LOG_LEVEL_ERROR, "Cannot add new participant (error %d)!\n", response_get_status(r));
                            } else{
                                participants.push_back(p);
                            }
                            free_response(r);
                    }
            }

            received_data += state.qtail->size;

            state.qtail = state.qtail->next;

            pthread_mutex_lock(&state.qempty_mtx);
            state.qempty = 0;
            pthread_cond_signal(&state.qempty_cond);
            pthread_mutex_unlock(&state.qempty_mtx);

            double seconds = tv_diff(t, t0);
            if (seconds > 5.0) {
                unsigned long long int cur_data = (received_data - last_data);
                unsigned long long int bps = cur_data / seconds;
                log_msg(LOG_LEVEL_INFO, "Received %llu bytes in %g seconds = %sbps.\n", cur_data, seconds, format_in_si_units(bps * 8));
                t0 = t;
                last_data = received_data;
            }
        }

        if (state.qtail->size <= 0)
            continue;

        pthread_mutex_lock(&state.qfull_mtx);
        if (state.qfull)
            pthread_cond_wait(&state.qfull_cond, &state.qfull_mtx);
        state.qfull = 1;
        pthread_mutex_unlock(&state.qfull_mtx);
    }

    if (state.qtail->size < 0 && !should_exit) {
        printf("read: %s\n", strerror(err));
        EXIT(2);
    }

    // pass poisoned pill to the worker
    state.qtail->size = 0;
    state.qtail = state.qtail->next;

    pthread_mutex_lock(&state.qempty_mtx);
    state.qempty = 0;
    pthread_cond_signal(&state.qempty_cond);
    pthread_mutex_unlock(&state.qempty_mtx);

    pthread_join(thread, NULL);

    hd_rum_translator_deinit(&state);
    udp_exit(sock_in);
    common_cleanup(init);

    printf("Exit\n");

    return 0;
}

/* vim: set sw=4 expandtab : */

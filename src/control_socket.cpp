/*
 * FILE:    control_socket.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "control_socket.h"

#include <set>

#include "debug.h"
#include "messaging.h"
#include "module.h"
#include "stats.h"
#include "tv.h"

#define DEFAULT_CONTROL_PORT 5054
#define MAX_CLIENTS 16

#ifdef WIN32
typedef const char *sso_val_type;
#else
typedef void *sso_val_type;
#endif                          /* WIN32 */


using namespace std;

struct client {
        fd_t fd;
        char buff[1024];
        int buff_len;

        struct client *prev;
        struct client *next;
};

enum connection_type {
        SERVER,
        CLIENT
};

struct control_state {
        struct module mod;
        pthread_t thread_id;
        /// @var internal_fd is used for internal communication
        fd_t internal_fd[2];
        int local_port;
        int network_port;
        struct module *root_module;

        set<struct stats *> stats;
        pthread_mutex_t stats_lock;

        enum connection_type connection_type;

        fd_t socket_fd;

        bool started;
};

#define CONTROL_EXIT -1
#define CONTROL_CLOSE_HANDLE -2

static bool parse_msg(char *buffer, int buffer_len, /* out */ char *message, int *new_buffer_len);
static int process_msg(struct control_state *s, fd_t client_fd, char *message);
static ssize_t write_all(fd_t fd, const void *buf, size_t count);
static void * control_thread(void *args);
static void send_response(fd_t fd, struct response *resp);

#ifndef HAVE_LINUX
#define MSG_NOSIGNAL 0
#endif

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

/**
 * Creates listening socket for internal communication.
 * This can be closed after accepted.
 *
 * @param[out] port port to be connected to
 * @returns listening socket descriptor
 */
static fd_t create_internal_port(int *port)
{
        fd_t sock;
        sock = socket(AF_INET6, SOCK_STREAM, 0);
        struct sockaddr_in6 s_in;
        memset(&s_in, 0, sizeof(s_in));
        s_in.sin6_family = AF_INET6;
        s_in.sin6_addr = in6addr_any;
        s_in.sin6_port = htons(0);
        assert(bind(sock, (const struct sockaddr *) &s_in,
                        sizeof(s_in)) == 0);
        assert(listen(sock, 10) == 0);
        socklen_t len = sizeof(s_in);
        assert(getsockname(sock, (struct sockaddr *) &s_in, &len) == 0);
        *port = ntohs(s_in.sin6_port);
        return sock;
}


int control_init(int port, struct control_state **state, struct module *root_module)
{
        control_state *s = new control_state;

        s->root_module = root_module;
        s->started = false;

        pthread_mutex_init(&s->stats_lock, NULL);

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_CONTROL;
        s->mod.priv_data = s;

        if(port == -1) {
                s->network_port = DEFAULT_CONTROL_PORT;
                s->connection_type = SERVER;
        } else {
                s->network_port = port;
                s->connection_type = CLIENT;
        }

        if(s->connection_type == SERVER) {
                s->socket_fd = socket(AF_INET6, SOCK_STREAM, 0);
                int val = 1;
                setsockopt(s->socket_fd, SOL_SOCKET, SO_REUSEADDR,
                                (sso_val_type) &val, sizeof(val));

                /* setting address to in6addr_any allows connections to be established
                 * from both IPv4 and IPv6 hosts. This behavior can be modified
                 * using the IPPROTO_IPV6 level socket option IPV6_V6ONLY if required.*/
                struct sockaddr_in6 s_in;
                s_in.sin6_family = AF_INET6;
                s_in.sin6_addr = in6addr_any;
                s_in.sin6_port = htons(s->network_port);

                bind(s->socket_fd, (const struct sockaddr *) &s_in, sizeof(s_in));
                listen(s->socket_fd, MAX_CLIENTS);
        } else {
                s->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
                struct addrinfo hints, *res, *res0;
                int err;

                memset(&hints, 0, sizeof(hints));
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_STREAM;
                char port_str[10];
                snprintf(port_str, 10, "%d", s->network_port);
                err = getaddrinfo("127.0.0.1", port_str, &hints, &res0);

                if(err) {
                        return -1;
                        fprintf(stderr, "Unable to get address: %s\n", gai_strerror(err));
                }
                bool connected = false;

                for (res = res0; res; res = res->ai_next) {
                    if(connect(s->socket_fd, res->ai_addr, res->ai_addrlen) == -1) {
                        continue;
                    }
                    connected = true;
                    break; /* okay we got one */
                }

                freeaddrinfo(res0);

                if(!connected) {
                        fprintf(stderr, "Unable to connect to localhost:%d\n", s->network_port);
                        return -1;
                }
        }

        module_register(&s->mod, root_module);

        *state = s;
        return 0;
}

void control_start(struct control_state *s)
{
        fd_t sock;
        sock = create_internal_port(&s->local_port);

        if(pthread_create(&s->thread_id, NULL, control_thread, s) != 0) {
                fprintf(stderr, "Unable to create thread.\n");
                free(s);
                abort();
        }

        s->internal_fd[0] = accept(sock, NULL, NULL);
        close(sock);
        s->started = true;
}

#define prefix_matches(x,y) strncasecmp(x, y, strlen(y)) == 0
#define suffix(x,y) x + strlen(y)

/**
  * @retval -1 exit thread
  * @retval -2 close handle
  */
static int process_msg(struct control_state *s, fd_t client_fd, char *message)
{
        int ret = 0;
        struct response *resp = NULL;
        char path[1024];
        char buf[1024];

        memset(path, 0, sizeof(path));

        if(prefix_matches(message, "port ")) {
                message = suffix(message, "port ");
                snprintf(path, 1024, "%s[%d]", module_class_name(MODULE_CLASS_PORT), atoi(message));
                while(isdigit(*message))
                        message++;
                while(isspace(*message))
                        message++;
        }

        if(strcasecmp(message, "quit") == 0) {
                return CONTROL_EXIT;
        } else if(prefix_matches(message, "receiver ") || prefix_matches(message, "play") ||
                        prefix_matches(message, "pause") || prefix_matches(message, "sender-port ")) {
                struct msg_sender *msg =
                        (struct msg_sender *)
                        new_message(sizeof(struct msg_sender));
                if(prefix_matches(message, "receiver ")) {
                        strncpy(msg->receiver, suffix(message, "receiver "), sizeof(msg->receiver) - 1);
                        msg->type = SENDER_MSG_CHANGE_RECEIVER;
                } else if(prefix_matches(message, "sender-port ")) {
                        msg->port = atoi(suffix(message, "sender-port "));
                        msg->type = SENDER_MSG_CHANGE_PORT;
                } else if(prefix_matches(message, "play")) {
                        msg->type = SENDER_MSG_PLAY;
                } else if(prefix_matches(message, "pause")) {
                        msg->type = SENDER_MSG_PAUSE;
                } else {
                        abort();
                }

                enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
                append_message_path(path, sizeof(path), path_sender);
                resp =
                        send_message(s->root_module, path, (struct message *) msg);
        } else if (prefix_matches(message, "receiver-port ")) {
                struct msg_receiver *msg =
                        (struct msg_receiver *)
                        new_message(sizeof(struct msg_receiver));
                msg->type = RECEIVER_MSG_CHANGE_RX_PORT;
                msg->new_rx_port = atoi(suffix(message, "receiver-port "));

                enum module_class path_receiver[] = { MODULE_CLASS_RECEIVER, MODULE_CLASS_NONE };
                append_message_path(path, sizeof(path), path_receiver);
                resp =
                        send_message(s->root_module, path, (struct message *) msg);
        } else if(prefix_matches(message, "fec ")) {
                struct msg_change_fec_data *msg = (struct msg_change_fec_data *)
                        new_message(sizeof(struct msg_change_fec_data));
                char *fec = suffix(message, "fec ");

                if(strncasecmp(fec, "audio ", 6) == 0) {
                        msg->media_type = TX_MEDIA_AUDIO;
                        strncpy(msg->fec, fec + 6, sizeof(msg->fec) - 1);
                } else if(strncasecmp(fec, "video ", 6) == 0) {
                        msg->media_type = TX_MEDIA_VIDEO;
                        strncpy(msg->fec, fec + 6, sizeof(msg->fec) - 1);
                } else {
                        resp = new_response(RESPONSE_NOT_FOUND, strdup("unknown media type"));
                }

                if(!resp) {
                        if(msg->media_type == TX_MEDIA_VIDEO) {
                                enum module_class path_tx[] = { MODULE_CLASS_TX, MODULE_CLASS_NONE };
                                append_message_path(path, sizeof(path),
                                                path_tx);
                        } else {
                                enum module_class path_audio_tx[] = { MODULE_CLASS_TX, MODULE_CLASS_AUDIO, MODULE_CLASS_NONE };
                                append_message_path(path, sizeof(path),
                                                path_audio_tx);
                        }
                        resp = send_message(s->root_module, path, (struct message *) msg);
                }
        } else if(prefix_matches(message, "compress ")) {
                struct msg_change_compress_data *msg =
                        (struct msg_change_compress_data *)
                        new_message(sizeof(struct msg_change_compress_data));
                char *compress = suffix(message, "compress ");

                if(prefix_matches(compress, "param ")) {
                        compress = suffix(compress, " param");
                        msg->what = CHANGE_PARAMS;
                } else {
                        msg->what = CHANGE_COMPRESS;
                }
                strncpy(msg->config_string, compress, sizeof(msg->config_string) - 1);

                if(!resp) {
                        enum module_class path_compress[] = { MODULE_CLASS_COMPRESS, MODULE_CLASS_NONE };
                        append_message_path(path, sizeof(path), path_compress);
                        resp = send_message(s->root_module, path, (struct message *) msg);
                }
        } else if(strcasecmp(message, "bye") == 0) {
                ret = CONTROL_CLOSE_HANDLE;
                resp = new_response(RESPONSE_OK, NULL);
        } else { // assume message in format "path message"
                struct msg_universal *msg = (struct msg_universal *)
                        new_message(sizeof(struct msg_universal));

                if (strchr(message, ' ')) {
                        memcpy(path, message, strchr(message, ' ') - message);
                        strncpy(msg->text, strchr(message, ' ') + 1, sizeof(path) - 1);
                } else {
                        strncpy(path, message, sizeof(path) - 1); // empty message ??
                }

                resp = send_message(s->root_module, path, (struct message *) msg);
        }

        if(!resp) {
                fprintf(stderr, "No response received!\n");
                snprintf(buf, sizeof(buf), "(unknown path: %s)", path);
                resp = new_response(RESPONSE_INT_SERV_ERR, strdup(buf));
        }
        send_response(client_fd, resp);

        return ret;
}

static void send_response(fd_t fd, struct response *resp)
{
        char buffer[1024];

        snprintf(buffer, sizeof(buffer) - 2, "%d %s", resp->status,
                        response_status_to_text(resp->status));

        if(resp->text) {
                strncat(buffer, " ", sizeof(buffer) - strlen(buffer) - 1 - 2);
                strncat(buffer, resp->text, sizeof(buffer) - strlen(buffer) - 1 - 2);
        }
        strcat(buffer, "\r\n");

        write_all(fd, buffer, strlen(buffer));

        resp->deleter(resp);
}

static bool parse_msg(char *buffer, int buffer_len, /* out */ char *message, int *new_buffer_len)
{
        bool ret = false;
        int i = 0;

        while (i < buffer_len) {
                if(buffer[i] == '\0' || buffer[i] == '\n' || buffer[i] == '\r') {
                        ++i;
                } else {
                        break;
                }
        }

        int start = i;

        for( ; i < buffer_len; ++i) {
                if(buffer[i] == '\0' || buffer[i] == '\n' || buffer[i] == '\r') {
                        memcpy(message, buffer + start, i - start);
                        message[i - start] = '\0';
                        ret = true;
                        break;
                }
        }

        if(ret) {
                memmove(buffer, buffer + i, buffer_len - i);
                *new_buffer_len = buffer_len - i;
        }

        return ret;
}


#define is_internal_port(x) (x == s->internal_fd[1])

/**
 * connects to internal communication channel
 */
static fd_t connect_to_internal_channel(int local_port)
{
        struct sockaddr_in6 s_in;
        memset(&s_in, 0, sizeof(s_in));
        s_in.sin6_family = AF_INET6;
        s_in.sin6_addr = in6addr_loopback;
        s_in.sin6_port = htons(local_port);
        fd_t fd = socket(AF_INET6, SOCK_STREAM, 0);
        int ret;
        ret = connect(fd, (struct sockaddr *) &s_in,
                                sizeof(s_in));
        assert(ret == 0);

        return fd;
}

static struct client *add_client(struct client *clients, int fd) {
        struct client *new_client = (struct client *)
                malloc(sizeof(struct client));
        new_client->fd = fd;
        new_client->prev = NULL;
        new_client->next = clients;
        if (new_client->next) {
                new_client->next->prev = new_client;
        }
        new_client->buff_len = 0;

        return new_client;
}

static void * control_thread(void *args)
{
        struct control_state *s = (struct control_state *) args;
        struct client *clients = NULL;

        s->internal_fd[1] = connect_to_internal_channel(s->local_port);

        if(s->connection_type == CLIENT) {
                clients = add_client(clients, s->socket_fd);
        }
        struct sockaddr_storage client_addr;
        socklen_t len;

        errno = 0;

        clients = add_client(clients, s->internal_fd[1]);

        bool should_exit = false;

        struct timeval last_report_sent;
        const int report_interval_sec = 5;
        gettimeofday(&last_report_sent, NULL);

        while(!should_exit) {
                fd_t max_fd = 0;

                fd_set fds;
                FD_ZERO(&fds);
                if(s->connection_type == SERVER) {
                        FD_SET(s->socket_fd, &fds);
                        max_fd = s->socket_fd + 1;
                }

                struct client *cur = clients;

                while(cur) {
                        FD_SET(cur->fd, &fds);
                        if(cur->fd + 1 > max_fd) {
                                max_fd = cur->fd + 1;
                        }
                        cur = cur->next;
                }

                struct timeval timeout;
                timeout.tv_sec = report_interval_sec;
                timeout.tv_usec = 0;

                struct timeval *timeout_ptr = NULL;
                if(clients->next != NULL) { // some remote client
                        timeout_ptr = &timeout;
                }

                if(select(max_fd, &fds, NULL, NULL, timeout_ptr) >= 1) {
                        if(s->connection_type == SERVER && FD_ISSET(s->socket_fd, &fds)) {
                                int fd = accept(s->socket_fd, (struct sockaddr *) &client_addr, &len);
                                clients = add_client(clients, fd);
                        }

                        struct client *cur = clients;

                        while(cur) {
                                if(FD_ISSET(cur->fd, &fds)) {
                                        ssize_t ret = recv(cur->fd, cur->buff + cur->buff_len,
                                                        sizeof(cur->buff) - cur->buff_len, 0);
                                        if(ret == -1) {
                                                fprintf(stderr, "Error reading socket!!!\n");
                                        }
                                        if(ret == 0) {
                                                struct client *next;
                                                close(cur->fd);
                                                if(cur->prev) {
                                                        cur->prev->next = cur->next;
                                                } else {
                                                        clients = cur->next;
                                                }
                                                next = cur->next;
                                                free(cur);
                                                cur = next;
                                                continue;
                                        }
                                        cur->buff_len += ret;
                                }
                                cur = cur->next;
                        }
                }

                cur = clients;
                while(cur) {
                        char msg[sizeof(cur->buff) + 1];
                        int cur_buffer_len;
                        while(parse_msg(cur->buff, cur->buff_len, msg, &cur_buffer_len)) {
                                cur->buff_len = cur_buffer_len;

                                int ret = process_msg(s, cur->fd, msg);
                                if(ret == CONTROL_EXIT && is_internal_port(cur->fd)) {
                                        should_exit = true;
                                } else if(ret == CONTROL_CLOSE_HANDLE) {
                                        shutdown(cur->fd, SHUT_RDWR);
                                }
                        }
                        if(cur->buff_len == sizeof(cur->buff)) {
                                fprintf(stderr, "Socket buffer full and no delimited message. Discarding.\n");
                                cur->buff_len = 0;
                        }

                        cur = cur->next;
                }

                struct timeval curr_time;
                gettimeofday(&curr_time, NULL);
                if(tv_diff(curr_time, last_report_sent) > report_interval_sec) {
                        char buffer[1025];
                        bool empty = true;
                        memset(buffer, '\0', sizeof(buffer));
                        strncpy(buffer + strlen(buffer), "stats",  sizeof(buffer) -
                                        strlen(buffer) - 1);
                        pthread_mutex_lock(&s->stats_lock);
                        for(set<struct stats *>::iterator it = s->stats.begin();
                                        it != s->stats.end(); ++it) {
                                empty = false;
                                strncpy(buffer + strlen(buffer), " ",  sizeof(buffer) -
                                                strlen(buffer) - 1);
                                stats_format(*it, buffer + strlen(buffer),
                                                sizeof(buffer) - strlen(buffer));
                        }

                        pthread_mutex_unlock(&s->stats_lock);
                        strncpy(buffer + strlen(buffer), "\r\n",  sizeof(buffer) -
                                        strlen(buffer) - 1);

                        if(strlen(buffer) < 1024 && !empty) {
                                cur = clients;
                                while(cur) {
                                        if(is_internal_port(cur->fd)) { // skip local FD
                                                cur = cur->next;
                                                continue;
                                        }
                                        write_all(cur->fd, buffer, strlen(buffer));
                                        cur = cur->next;
                                }
                        }
                        last_report_sent = curr_time;
                }
        }

        struct client *cur = clients;
        while(cur) {
                struct client *tmp = cur;
                close(cur->fd);
                cur = cur->next;
                free(tmp);
        }

        close(s->internal_fd[1]);

        return NULL;
}

void control_done(struct control_state *s)
{
        if(!s) {
                return;
        }

        module_done(&s->mod);

        if(s->started) {
                write_all(s->internal_fd[0], "quit\r\n", 6);
                pthread_join(s->thread_id, NULL);
                close(s->internal_fd[0]);
        }
        if(s->connection_type == SERVER) {
                // for client, the socket has already been closed
                // by the time of control_thread exit
                close(s->socket_fd);
        }

        pthread_mutex_destroy(&s->stats_lock);

        delete s;
}

void control_add_stats(struct control_state *s, struct stats *stats)
{
        pthread_mutex_lock(&s->stats_lock);
        s->stats.insert(stats);
        pthread_mutex_unlock(&s->stats_lock);
}

void control_remove_stats(struct control_state *s, struct stats *stats)
{
        pthread_mutex_lock(&s->stats_lock);
        s->stats.erase(stats);
        pthread_mutex_unlock(&s->stats_lock);
}


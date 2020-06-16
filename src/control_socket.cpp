/**
 * @file   control_socket.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#include "control_socket.h"
#include "compat/platform_pipe.h"

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>

#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "rtp/net_udp.h" // socket_error
#include "tv.h"
#include "utils/net.h"
#include "utils/thread.h"

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

#define MAX_STAT_EVENT_QUEUE 100

struct control_state {
        struct module mod;
        thread control_thread_id;
        /// @var internal_fd is used for internal communication
        fd_t internal_fd[2];
        int network_port;
        struct module *root_module;

        std::mutex stats_lock;

        enum connection_type connection_type;

        fd_t socket_fd = INVALID_SOCKET;

        bool started;

        thread stat_event_thread_id;
        condition_variable stat_event_cv;
        queue<string> stat_event_queue;

        bool stats_on;
};

#define CONTROL_EXIT -1
#define CONTROL_CLOSE_HANDLE -2

static bool parse_msg(char *buffer, int buffer_len, /* out */ char *message, int *new_buffer_len);
static int process_msg(struct control_state *s, fd_t client_fd, char *message, struct client *clients);
static ssize_t write_all(fd_t fd, const void *buf, size_t count);
static void * control_thread(void *args);
static void * stat_event_thread(void *args);
static void send_response(fd_t fd, struct response *resp);
static void print_control_help();

#ifndef HAVE_LINUX
#define MSG_NOSIGNAL 0
#endif

static ssize_t write_all(fd_t fd, const void *buf, size_t count)
{
    const char *p = (const char *) buf;
    size_t rest = count;
    ssize_t w = 0;

    while (rest > 0 && ((fd > 2 && (w = send(fd, p, rest, MSG_NOSIGNAL)) > 0) ||
                            (w = write(fd, p, rest)) > 0)) {
        p += w;
        rest -= w;
    }

    if (rest > 0)
        return w;
    else
        return count;
}

static void new_message(struct module *m) {
        control_state *s = (control_state *) m->priv_data;
        if(s->started) {
                // just wake up from select
                int ret = write_all(s->internal_fd[1], "noop\r\n", 6);
                if (ret <= 0) {
                        log_msg(LOG_LEVEL_ERROR, "[control] Cannot write to pipe!\n");
                }
        }
}

int control_init(int port, int connection_type, struct control_state **state, struct module *root_module, int force_ip_version)
{
        control_state *s = new control_state();

        s->root_module = root_module;
        s->started = false;

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_CONTROL;
        s->mod.new_message = new_message;
        s->mod.priv_data = s;

        if(connection_type == 0) {
                s->network_port = port;
                s->connection_type = SERVER;
        } else {
                s->network_port = port;
                s->connection_type = CLIENT;
        }

        if(s->connection_type == SERVER) {
                int ip_version = 6;
                if (force_ip_version != 4) {
                        s->socket_fd = socket(AF_INET6, SOCK_STREAM, 0);
                }
                if (force_ip_version == 4 || (s->socket_fd == INVALID_SOCKET && errno == EAFNOSUPPORT)) { // try IPv4
                        ip_version = 4;
                        s->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
                }
                if (s->socket_fd == INVALID_SOCKET) {
                        perror("Control socket - socket");
                        goto error;
                }
                int val = 1;
                int rc;
                rc = setsockopt(s->socket_fd, SOL_SOCKET, SO_REUSEADDR,
                                (sso_val_type) &val, sizeof(val));
                if (rc != 0) {
                        perror("Control socket - setsockopt");
                }

                int ipv6only = 0;
                if (ip_version == 6 && setsockopt(s->socket_fd, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&ipv6only,
                                        sizeof(ipv6only)) != 0) {
                        perror("setsockopt IPV6_V6ONLY");
                }

                /* setting address to in6addr_any allows connections to be established
                 * from both IPv4 and IPv6 hosts. This behavior can be modified
                 * using the IPPROTO_IPV6 level socket option IPV6_V6ONLY if required.*/
                struct sockaddr_storage s_in;
                socklen_t s_len;
                memset(&s_in, 0, sizeof(s_in));
                if (ip_version == 4) {
                        struct sockaddr_in *s_in = (struct sockaddr_in *) &s_in;
                        s_in->sin_family = AF_INET;
                        s_in->sin_addr.s_addr = htonl(INADDR_ANY);
                        s_in->sin_port = htons(s->network_port);
                        s_len = sizeof(*s_in);
                } else {
                        struct sockaddr_in6 *s_in6 = (struct sockaddr_in6 *) &s_in;
                        s_in6->sin6_family = AF_INET6;
                        s_in6->sin6_addr = in6addr_any;
                        s_in6->sin6_port = htons(s->network_port);
                        s_len = sizeof(*s_in6);
                }

                rc = ::bind(s->socket_fd, (const struct sockaddr *) &s_in, s_len);
                if (rc != 0) {
                        perror("Control socket - bind");
                        CLOSESOCKET(s->socket_fd);
                        s->socket_fd = INVALID_SOCKET;
                } else {
                        rc = listen(s->socket_fd, MAX_CLIENTS);
                        if (rc != 0) {
                                perror("Control socket - listen");
                                CLOSESOCKET(s->socket_fd);
                                s->socket_fd = INVALID_SOCKET;
                        }
                }
        } else {
                if (force_ip_version == 6) {
                        log_msg(LOG_LEVEL_ERROR, "Control socket: IPv6 unimplemented in client mode!\n");
                        goto error;
                }
                s->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
                assert(s->socket_fd != INVALID_SOCKET);
                struct addrinfo hints, *res, *res0;
                int err;

                memset(&hints, 0, sizeof(hints));
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_STREAM;
                char port_str[10];
                snprintf(port_str, 10, "%d", s->network_port);
                err = getaddrinfo("127.0.0.1", port_str, &hints, &res0);

                if(err) {
                        fprintf(stderr, "Unable to get address: %s\n", gai_strerror(err));
                        goto error;
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
                        goto error;
                }
        }


        module_register(&s->mod, root_module);

        *state = s;
        return 0;

error:
        if (s->socket_fd != INVALID_SOCKET) {
                CLOSESOCKET(s->socket_fd);
        }
        delete s;
        return -1;
}

void control_start(struct control_state *s)
{
        if (s == NULL) {
                return;
        }

        platform_pipe_init(s->internal_fd);

        s->control_thread_id = thread(control_thread, s);
        s->stat_event_thread_id = thread(stat_event_thread, s);

        s->started = true;
}

#define prefix_matches(x,y) strncasecmp(x, y, strlen(y)) == 0
#define suffix(x,y) x + strlen(y)
#define is_internal_port(x) (x == s->internal_fd[0])

/**
  * @retval -1 exit thread
  * @retval -2 close handle
  */
static int process_msg(struct control_state *s, fd_t client_fd, char *message, struct client *clients)
{
        int ret = 0;
        struct response *resp = NULL;
        char path[1024] = ""; // path for msg receiver (usually video)
        char path_audio[1024] = ""; // auxiliary buffer used when we need to signalize both audio
                                    // and video
        char buf[1048];

        if(prefix_matches(message, "port ")) {
                message = suffix(message, "port ");
                if (isdigit(message[0])) { // index is a number
                        snprintf(path, 1024, "%s[%d]", module_class_name(MODULE_CLASS_PORT), atoi(message));
                        while(isdigit(*message))
                                message++;
                        while(isspace(*message))
                                message++;
                } else { // index is a name
                        char *port_id_name = message;
                        while (!isspace(message[0]) && message[0] != '\0')
                                message++;
                        if (message[0] != '\0') {
                                message[0] = '\0';
                                message++;
                        }
                        snprintf(path, 1024, "%s[%s]", module_class_name(MODULE_CLASS_PORT), port_id_name);
                        while (isspace(message[0]) && message[0] != '\0')
                                message++;
                }
        }

        if(strcasecmp(message, "quit") == 0) {
                return CONTROL_EXIT;
        } else if (strcasecmp(message, "exit") == 0) {
                exit_uv(0);
                resp = new_response(RESPONSE_OK, NULL);
        } else if (strcasecmp(message, "help") == 0) {
                print_control_help();
                return ret;
        } else if (strcasecmp(message, "noop") == 0) {
                return ret;
        } else if (prefix_matches(message, "stats ") || prefix_matches(message, "event ")) {
                if (is_internal_port(client_fd)) {
                        struct client *cur = clients;
                        char *new_msg = NULL;

                        while (cur) {
                                if(is_internal_port(cur->fd)) { // skip local FD
                                        cur = cur->next;
                                        continue;
                                }
                                if (!new_msg) {
                                        // append <CR><LF> again
                                        new_msg = (char *) malloc(strlen(message) + 1 + 2);
                                        strcpy(new_msg, message);
                                        new_msg[strlen(message)] = '\r';
                                        new_msg[strlen(message) + 1] = '\n';
                                        new_msg[strlen(message) + 2] = '\0';
                                }

                                int ret = write_all(cur->fd, new_msg, strlen(new_msg));
                                if (ret != (int) strlen(new_msg)) {
                                        log_msg(LOG_LEVEL_WARNING, "Cannot write stats/event!\n");
                                }
                                cur = cur->next;
                        }
                        free(new_msg);
                        return ret;
                } else if (prefix_matches(message, "stats ")) {
                        const char *toggle = suffix(message, "stats ");
                        if (strcasecmp(toggle, "on") == 0) {
                                s->stats_on = true;
                                resp = new_response(RESPONSE_OK, NULL);
                        } else if (strcasecmp(toggle, "off") == 0) {
                                s->stats_on = false;
                                resp = new_response(RESPONSE_OK, NULL);
                        } else {
                                resp = new_response(RESPONSE_BAD_REQUEST, NULL);
                        }
                }
        } else if (prefix_matches(message, "sender-port ")) {
                struct msg_sender *msg = (struct msg_sender *)
                        new_message(sizeof(struct msg_sender));
                struct msg_sender *msg_audio = (struct msg_sender *)
                        new_message(sizeof(struct msg_sender));
                char *port_str = suffix(message, "sender-port ");

                msg->type = SENDER_MSG_CHANGE_PORT;
                msg_audio->type = SENDER_MSG_CHANGE_PORT;

                if (strchr(port_str, ':')) {
                        char *save_ptr, *item;
                        item = strtok_r(port_str, ":", &save_ptr);
                        msg->rx_port = atoi(item);
                        item = strtok_r(NULL, ":", &save_ptr);
                        msg->tx_port = atoi(item);

                        item = strtok_r(NULL, ":", &save_ptr);
                        if (item) {
                                char *item2 = strtok_r(NULL, ":", &save_ptr);
                                if (item2) { // change only if both RX and TX given
                                        msg_audio->rx_port = atoi(item);
                                        msg_audio->tx_port = atoi(item2);
                                } else {
                                        log_msg(LOG_LEVEL_ERROR, "Not changing audio port!\n");
                                }
                        }
                } else {
                        msg->tx_port = atoi(port_str);
                }

                if (msg_audio->rx_port == 0) {
                        msg_audio->rx_port = msg->rx_port ? msg->rx_port + 2 : 0;
                }

                if (msg_audio->tx_port == 0) {
                        msg_audio->tx_port = msg->tx_port + 2;
                }

                enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
                enum module_class path_sender_audio[] = { MODULE_CLASS_AUDIO, MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
                memcpy(path_audio, path, sizeof(path_audio));
                append_message_path(path, sizeof(path), path_sender);
                append_message_path(path_audio, sizeof(path_audio), path_sender_audio);

                resp =
                        send_message(s->root_module, path, (struct message *) msg);
                struct response *resp_audio =
                        send_message(s->root_module, path_audio, (struct message *) msg_audio);
                free_response(resp_audio);
        } else if(prefix_matches(message, "receiver ") || prefix_matches(message, "play") ||
                        prefix_matches(message, "pause") || prefix_matches(message, "reset-ssrc")) {
                struct msg_sender *msg =
                        (struct msg_sender *)
                        new_message(sizeof(struct msg_sender));
                if(prefix_matches(message, "receiver ")) {
                        strncpy(msg->receiver, suffix(message, "receiver "), sizeof(msg->receiver) - 1);
                        msg->type = SENDER_MSG_CHANGE_RECEIVER;
                } else if(prefix_matches(message, "play")) {
                        msg->type = SENDER_MSG_PLAY;
                } else if(prefix_matches(message, "pause")) {
                        msg->type = SENDER_MSG_PAUSE;
                } else if(prefix_matches(message, "reset-ssrc")) {
                        msg->type = SENDER_MSG_RESET_SSRC;
                } else {
                        abort();
                }

                struct msg_sender *msg_audio = (struct msg_sender *) malloc(sizeof(struct msg_sender));
                memcpy(msg_audio, msg, sizeof(struct msg_sender));

                enum module_class path_sender[] = { MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
                enum module_class path_sender_audio[] = { MODULE_CLASS_AUDIO, MODULE_CLASS_SENDER, MODULE_CLASS_NONE };
                memcpy(path_audio, path, sizeof(path_audio));
                append_message_path(path, sizeof(path), path_sender);
                append_message_path(path_audio, sizeof(path_audio), path_sender_audio);

                resp =
                        send_message(s->root_module, path, (struct message *) msg);
                struct response *resp_audio =
                        send_message(s->root_module, path_audio, (struct message *) msg_audio);
                free_response(resp_audio);
        } else if (prefix_matches(message, "receiver-port ")) {
                struct msg_receiver *msg =
                        (struct msg_receiver *)
                        new_message(sizeof(struct msg_receiver));
                struct msg_receiver *msg_audio =
                        (struct msg_receiver *)
                        new_message(sizeof(struct msg_receiver));
                msg->type = RECEIVER_MSG_CHANGE_RX_PORT;
                msg_audio->type = RECEIVER_MSG_CHANGE_RX_PORT;

                char *port_str = suffix(message, "receiver-port ");

                msg->new_rx_port = atoi(port_str);
                if (strchr(port_str, ':')) {
                        msg_audio->new_rx_port = atoi(strchr(port_str, ':') + 1);
                } else {
                        msg_audio->new_rx_port = msg->new_rx_port + 2;
                }

                enum module_class path_receiver[] = { MODULE_CLASS_RECEIVER, MODULE_CLASS_NONE };
                enum module_class path_audio_receiver[] = { MODULE_CLASS_AUDIO, MODULE_CLASS_RECEIVER, MODULE_CLASS_NONE };
                append_message_path(path, sizeof(path), path_receiver);
                append_message_path(path_audio, sizeof(path_audio), path_audio_receiver);
                resp =
                        send_message(s->root_module, path, (struct message *) msg);
                struct response *resp_audio =
                        send_message(s->root_module, path_audio, (struct message *) msg_audio);
                free_response(resp_audio);
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
                        resp = new_response(RESPONSE_NOT_FOUND, "unknown media type");
                        free(msg);
                        msg = nullptr;
                }

                if (msg) {
                        if(msg->media_type == TX_MEDIA_VIDEO) {
                                enum module_class path_tx[] = { MODULE_CLASS_SENDER, MODULE_CLASS_TX, MODULE_CLASS_NONE };
                                append_message_path(path, sizeof(path),
                                                path_tx);
                        } else {
                                enum module_class path_audio_tx[] = { MODULE_CLASS_AUDIO, MODULE_CLASS_SENDER, MODULE_CLASS_TX, MODULE_CLASS_NONE };
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
                        compress = suffix(compress, "param ");
                        msg->what = CHANGE_PARAMS;
                } else {
                        msg->what = CHANGE_COMPRESS;
                }
                strncpy(msg->config_string, compress, sizeof(msg->config_string) - 1);

                if(!resp) {
                        enum module_class path_compress[] = { MODULE_CLASS_SENDER, MODULE_CLASS_COMPRESS, MODULE_CLASS_NONE };
                        append_message_path(path, sizeof(path), path_compress);
                        resp = send_message(s->root_module, path, (struct message *) msg);
                }
        } else if (prefix_matches(message, "volume ")) {
                strncpy(path, "audio.receiver", sizeof path);
                struct msg_receiver *msg = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                char *dir = suffix(message, "volume ");
                if (strcmp(dir, "up") == 0) {
                        msg->type = RECEIVER_MSG_INCREASE_VOLUME;
                } else if (strcmp(dir, "down") == 0) {
                        msg->type = RECEIVER_MSG_DECREASE_VOLUME;
                } else {
                        free_message((struct message *) msg, NULL);
                        msg = NULL;
                        resp = new_response(RESPONSE_BAD_REQUEST, NULL);
                }

                if (msg) {
                        resp = send_message(s->root_module, path, (struct message *) msg);
                }
        } else if (prefix_matches(message, "mute")) {
                strncpy(path, "audio.receiver", sizeof path);
                struct msg_receiver *msg = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                msg->type = RECEIVER_MSG_MUTE;
                resp = send_message(s->root_module, path, (struct message *) msg);
        } else if (prefix_matches(message, "av-delay ")) {
                int val = atoi(suffix(message, "av-delay "));
                set_audio_delay(val);
                resp = new_response(RESPONSE_OK, NULL);
        } else if (prefix_matches(message, "postprocess ")) {
                strncpy(path, "display", sizeof path);
                struct msg_universal *msg = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strncpy(msg->text, message, sizeof msg->text - 1);
                msg->text[sizeof msg->text - 1] = '\0';
                resp = send_message(s->root_module, path, (struct message *) msg);
        } else if(strcasecmp(message, "bye") == 0) {
                ret = CONTROL_CLOSE_HANDLE;
                resp = new_response(RESPONSE_OK, NULL);
        } else if(strcmp(message, "dump-tree") == 0) {
                dump_tree(s->root_module, 0);
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
                resp = new_response(RESPONSE_INT_SERV_ERR, buf);
        }
        send_response(client_fd, resp);

        return ret;
}

static void send_response(fd_t fd, struct response *resp)
{
        char buffer[1024];

        snprintf(buffer, sizeof(buffer) - 2, "%d %s", response_get_status(resp),
                        response_status_to_text(response_get_status(resp)));

        if (response_get_text(resp)) {
                strncat(buffer, " ", sizeof(buffer) - strlen(buffer) - 1 - 2);
                strncat(buffer, response_get_text(resp), sizeof(buffer) - strlen(buffer) - 1 - 2);
        }
        strcat(buffer, "\r\n");

        ssize_t ret = write_all(fd, buffer, strlen(buffer));
        if (ret < 0) {
                perror("Unable to write response");
        }

        free_response(resp);
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

/**
 * prepends itself at the head of the list
 */
static struct client *add_client(struct client *clients, fd_t fd) {
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

static void process_messages(struct control_state *s)
{
        struct message *msg;
        while((msg = check_message(&s->mod))) {
		struct msg_universal *m = (struct msg_universal *) msg;
		if (strcmp(m->text, "get_port") == 0) {
                        struct response *r;
                        uint16_t port = socket_get_recv_port(s->socket_fd);
                        if (port) {
                                char port_str[6];
                                sprintf(port_str, "%hu", port);
                                r = new_response(RESPONSE_OK, port_str);
                        } else {
                                r = new_response(RESPONSE_INT_SERV_ERR, "get_recv_port");
                        }
                        free_message(msg, r);
                } else if (strstr(m->text, "execute ") == m->text) {
                        process_msg(s, 1, m->text + strlen("execute "), nullptr);
                        free_message(msg, new_response(RESPONSE_OK, nullptr));
                } else {
                        log_msg(LOG_LEVEL_WARNING, "[control] Unrecognized command: %s\n", m->text);
                        free_message(msg, new_response(RESPONSE_NOT_IMPL, NULL));
                        continue;
		}
        }
}

static void set_socket_nonblock(fd_t fd) {
#ifdef _WIN32
    unsigned long ul;
    ioctlsocket(fd, FIONBIO, &ul);
#else
    if (fcntl(fd, F_SETFL, O_NONBLOCK) == -1) {
            perror("fcntl");
    }
#endif
}

ADD_TO_PARAM("control-accept-global", "* control-accept-global\n"
                "  Open control socket to public network.\n");
static void * control_thread(void *args)
{
        set_thread_name(__func__);
        struct control_state *s = (struct control_state *) args;
        struct client *clients = NULL;

        assert(s->socket_fd != INVALID_SOCKET);
        assert(s->internal_fd[0] != INVALID_SOCKET);

        if(s->connection_type == CLIENT) {
                clients = add_client(clients, s->socket_fd);
        }
        struct sockaddr_storage client_addr;
        socklen_t len = sizeof client_addr;

        errno = 0;

        clients = add_client(clients, s->internal_fd[0]);

        bool should_exit = false;

        struct timeval last_report_sent;
        const int report_interval_sec = 5;
        gettimeofday(&last_report_sent, NULL);

        while(!should_exit) {
                process_messages(s);

                fd_t max_fd = 0;
                fd_set fds;
                FD_ZERO(&fds);
                if (s->connection_type == SERVER) {
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

                int rc;

                if ((rc = select(max_fd, &fds, NULL, NULL, timeout_ptr)) >= 1) {
                        if(s->connection_type == SERVER && FD_ISSET(s->socket_fd, &fds)) {
                                fd_t fd = accept(s->socket_fd, (struct sockaddr *) &client_addr, &len);
                                if (fd == INVALID_SOCKET) {
                                        socket_error("[control socket] accept");
                                        continue;
                                }

                                // all remote sockets are written sequentially so
                                // we don't want to block if one gets stuck
                                set_socket_nonblock(fd);

                                // refuse remote connections by default
                                if (!get_commandline_param("control-accept-global") &&
                                                !is_addr_loopback((struct sockaddr *) &client_addr)) {
                                        log_msg(LOG_LEVEL_WARNING, "[control socket] Refusing remote connection. Use \"--param control-accept-global\" to allow UG control from a remote host.\n");
                                        const char *msg = "unauthorized\r\n";
                                        write_all(fd, msg, strlen(msg));
                                        CLOSESOCKET(fd);
                                        continue;
                                }

                                // add to list
                                clients = add_client(clients, fd);
                        }

                        struct client *cur = clients;

                        while(cur) {
                                if(FD_ISSET(cur->fd, &fds)) {
                                        ssize_t ret = recv(cur->fd, cur->buff + cur->buff_len,
                                                        sizeof(cur->buff) - cur->buff_len, 0);
                                        if(ret == -1) {
                                                fprintf(stderr, "Error reading socket, closing!!!\n");
                                        }
                                        if(ret <= 0) {
                                                struct client *next;
                                                CLOSESOCKET(cur->fd);
                                                if (cur->prev) {
                                                        cur->prev->next = cur->next;
                                                } else {
                                                        clients = cur->next;
                                                }
                                                if (cur->next) {
                                                        cur->next->prev = cur->prev;
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
                } else if (rc < 0) {
                        socket_error("[control socket] select");
                }

                cur = clients;
                while(cur) {
                        char msg[sizeof(cur->buff) + 1];
                        int cur_buffer_len;
                        while(parse_msg(cur->buff, cur->buff_len, msg, &cur_buffer_len)) {
                                cur->buff_len = cur_buffer_len;
                                int ret = process_msg(s, cur->fd, msg, clients);
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
        }

        // notify clients about exit
        struct client *cur = clients;
        while(cur) {
                struct client *tmp = cur;
                if (!is_internal_port(cur->fd)) {
                        const char *msg = "event exit\r\n";
                        write_all(cur->fd, msg, strlen(msg));
                }
                CLOSESOCKET(cur->fd);
                cur = cur->next;
                free(tmp);
        }

        platform_pipe_close(s->internal_fd[0]);

        return NULL;
}

static void *stat_event_thread(void *args)
{
        set_thread_name(__func__);
        struct control_state *s = (struct control_state *) args;

        while (1) {
                std::unique_lock<std::mutex> lk(s->stats_lock);
                s->stat_event_cv.wait(lk, [s] { return s->stat_event_queue.size() > 0; });
                string &line = s->stat_event_queue.front();

                if (line.empty()) {
                        break;
                }

                int ret = write_all(s->internal_fd[1], line.c_str(), line.length());
                s->stat_event_queue.pop();
                if (ret <= 0) {
                        fprintf(stderr, "Cannot write stat line!\n");
                }
        }

        return NULL;
}

void control_done(struct control_state *s)
{
        if(!s) {
                return;
        }

        module_done(&s->mod);

        if(s->started) {
                s->stats_lock.lock();
                s->stat_event_queue.push({});
                s->stats_lock.unlock();
                s->stat_event_cv.notify_one();
                s->stat_event_thread_id.join();

                int ret = write_all(s->internal_fd[1], "quit\r\n", 6);
                if (ret > 0) {
                        s->control_thread_id.join();
                        platform_pipe_close(s->internal_fd[1]);
                } else {
                        fprintf(stderr, "Cannot exit control thread!\n");
                }
        }
        if(s->connection_type == SERVER && s->socket_fd != INVALID_SOCKET) {
                // for client, the socket has already been closed
                // by the time of control_thread exit
                CLOSESOCKET(s->socket_fd);
        }

        delete s;
}

static void control_report_stats_event(struct control_state *s, const std::string &report_line)
{
        std::unique_lock<std::mutex> lk(s->stats_lock);

        if (s->stat_event_queue.size() < MAX_STAT_EVENT_QUEUE) {
                s->stat_event_queue.push(report_line);
        } else {
                log_msg(LOG_LEVEL_WARNING, "Cannot write stats/event - queue full!!!");
        }
        lk.unlock();
        s->stat_event_cv.notify_one();
}

void control_report_stats(struct control_state *s, const std::string &report_line)
{
        if (!s || !s->stats_on) {
                return;
        }

        control_report_stats_event(s, "stats " + report_line + "\r\n");
}

void control_report_event(struct control_state *s, const std::string &report_line)
{
        if (!s) {
                return;
        }

        control_report_stats_event(s, "event " + report_line + "\r\n");
}

bool control_stats_enabled(struct control_state *s)
{
        return s && s->stats_on;
}

static void print_control_help() {
        printf("Control internal commands:\n"
                        "\texit\n"
                        "\tpause\n"
                        "\tplay\n"
                        "\treciever {pause|play}\n"
                        "\treset-ssrc\n"
                        "\t{receiver|sender}-port <XY>\n"
                        "\tfec {audio|video} <fec-string>\n"
                        "\tcompress <new-compress>\n"
                        "\tcompress param <new-compress-param>\n"
                        "\tvolume {up|down}\n"
                        "\tav-delay <ms>\n"
                        "\tmute\n"
                                "\t\tthe three items above apply to receiver\n"
                        "\tpostprocess <new_postprocess>|flush\n"
                        "\tdump-tree\n");
        printf("\nOther commands can be issued directly to individual "
                        "modules (see \"dump-tree\"), eg.:\n"
                        "\tcapture.filter mirror\n"
                        "\nSometimes those modules support help (eg. \"capture.filter help)\"\n\n");
}


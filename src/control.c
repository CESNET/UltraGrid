/*
 * FILE:    control.c
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

#include "control.h"

#include "debug.h"
#include "messaging.h"
#include "module.h"

#define DEFAULT_CONTROL_PORT 5054
#define MAX_CLIENTS 16

struct client {
        int fd;
        char buff[1024];
        int buff_len;

        struct client *prev;
        struct client *next;
};

struct control_state {
        pthread_t thread_id;
        int local_fd[2];
        int network_port;
        struct module *root_module;
};

#define CONTROL_EXIT -1
#define CONTROL_CLOSE_HANDLE -2

static bool parse_msg(char *buffer, char buffer_len, /* out */ char *message, int *new_buffer_len);
static int process_msg(struct control_state *s, int client_fd, char *message);
static ssize_t write_all(int fd, const void *buf, size_t count);
static void * control_thread(void *args);
static void send_response(int fd, struct response *resp);

static ssize_t write_all(int fd, const void *buf, size_t count)
{
    unsigned char *p = (unsigned char *) buf;
    size_t rest = count;
    ssize_t w = 0;

    while (rest > 0 && (w = write(fd, p, rest)) >= 0) {
        p += w;
        rest -= w;
    }

    if (rest > 0)
        return w;
    else
        return count;
}

int control_init(int port, struct control_state **state, struct module *root_module)
{
        struct control_state *s = (struct control_state *) malloc(sizeof(struct control_state));

        s->root_module = root_module;

        if(port == -1) {
                s->network_port = DEFAULT_CONTROL_PORT;
        } else {
                s->network_port = port;
        }

        if (socketpair(AF_UNIX, SOCK_STREAM, 0, s->local_fd) < 0) {
                perror("opening stream socket pair");
                free(s);
                return -1;
        }

        if(pthread_create(&s->thread_id, NULL, control_thread, s) != 0) {
                fprintf(stderr, "Unable to create thread.\n");
                free(s);
                return -1;
        }

        *state = s;
        return 0;
}

#define prefix_matches(x,y) strncasecmp(x, y, strlen(y)) == 0
#define suffix(x,y) x + strlen(y)

/**
  * @retval -1 exit thread
  * @retval -2 close handle
  */
static int process_msg(struct control_state *s, int client_fd, char *message)
{
        int ret = 0;
        struct response *resp = NULL;
        char path[1024] = { '\0' };
        char buf[1024];

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
        } else if(prefix_matches(message, "receiver ")) {
                struct msg_change_receiver_address *msg =
                        (struct msg_change_receiver_address *)
                        new_message(sizeof(struct msg_change_receiver_address));
                strncpy(msg->receiver, suffix(message, "receiver "), sizeof(msg->receiver) - 1);

                append_message_path(path, sizeof(path), (enum module_class[]){ MODULE_CLASS_SENDER, MODULE_CLASS_NONE });
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
                        resp = new_response(RESPONSE_NOT_FOUND, NULL);
                }

                if(!resp) {
                        if(msg->media_type == TX_MEDIA_VIDEO) {
                                append_message_path(path, sizeof(path),
                                                (enum module_class[]){ MODULE_CLASS_TX, MODULE_CLASS_NONE });
                        } else {
                                append_message_path(path, sizeof(path),
                                                (enum module_class[]){ MODULE_CLASS_AUDIO, MODULE_CLASS_TX,
                                                MODULE_CLASS_NONE });
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
                        append_message_path(path, sizeof(path), (enum module_class[]){ MODULE_CLASS_COMPRESS, MODULE_CLASS_NONE });
                        resp = send_message(s->root_module, path, (struct message *) msg);
                }
        } else if(prefix_matches(message, "stats ")) {
                struct msg_stats *msg = (struct msg_stats *)
                        new_message(sizeof(struct msg_stats));
                char *stats = suffix(message, "stats ");

                strncpy(msg->what, stats, sizeof(msg->what) - 1);

                // resp = send_message(s->root_module, "blblbla", &data);
                resp = new_response(RESPONSE_NOT_FOUND, strdup("statistics currently not working"));
        } else if(strcasecmp(message, "bye") == 0) {
                ret = CONTROL_CLOSE_HANDLE;
                resp = new_response(RESPONSE_OK, NULL);
        } else {
                snprintf(buf, sizeof(buf), "(unrecognized control sequence: %s)", message);
                resp = new_response(RESPONSE_BAD_REQUEST, strdup(buf));
        }

        if(!resp) {
                fprintf(stderr, "No response received!\n");
                snprintf(buf, sizeof(buf), "(unknown path: %s)", path);
                resp = new_response(RESPONSE_INT_SERV_ERR, strdup(buf));
        }
        send_response(client_fd, resp);

        return ret;
}

static void send_response(int fd, struct response *resp)
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

static bool parse_msg(char *buffer, char buffer_len, /* out */ char *message, int *new_buffer_len)
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

static void * control_thread(void *args)
{
        struct control_state *s = (struct control_state *) args;
        int fd;

        fd = socket(AF_INET6, SOCK_STREAM, 0);
        int val = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));

        /* setting address to in6addr_any allows connections to be established
         * from both IPv4 and IPv6 hosts. This behavior can be modified
         * using the IPPROTO_IPV6 level socket option IPV6_V6ONLY if required.*/
        struct sockaddr_in6 s_in;
        s_in.sin6_family = AF_INET6;
        s_in.sin6_addr = in6addr_any;
        s_in.sin6_port = htons(s->network_port);

        bind(fd, (const struct sockaddr *) &s_in, sizeof(s_in));
        listen(fd, MAX_CLIENTS);
        struct sockaddr_storage client_addr;
        socklen_t len;

        errno = 0;

        struct client *clients = NULL;

        struct client *new_client = malloc(sizeof(struct client));
        new_client->fd = s->local_fd[1];
        new_client->prev = NULL;
        new_client->next = NULL;
        new_client->buff_len = 0;
        clients = new_client;

        bool should_exit = false;

        while(!should_exit) {
                fd_set set;
                FD_ZERO(&set);
                FD_SET(fd, &set);
                int max_fd = fd + 1;

                struct client *cur = clients;

                while(cur) {
                        FD_SET(cur->fd, &set);
                        if(cur->fd + 1 > max_fd) {
                                max_fd = cur->fd + 1;
                        }
                        cur = cur->next;
                }

                if(select(max_fd, &set, NULL, NULL, NULL) >= 1) {
                        if(FD_ISSET(fd, &set)) {
                                struct client *new_client = malloc(sizeof(struct client));
                                new_client->fd = accept(fd, (struct sockaddr *) &client_addr, &len);
                                new_client->prev = NULL;
                                new_client->next = clients;
                                new_client->buff_len = 0;
                                clients = new_client;
                        }

                        struct client *cur = clients;

                        while(cur) {
                                if(FD_ISSET(cur->fd, &set)) {
                                        ssize_t ret = read(cur->fd, cur->buff + cur->buff_len,
                                                        sizeof(cur->buff) - cur->buff_len);
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
                                if(ret == CONTROL_EXIT) {
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

        struct client *cur = clients;
        while(cur) {
                struct client *tmp = cur;
                close(cur->fd);
                cur = cur->next;
                free(tmp);
        }

        close(fd);

        return NULL;
}

void control_done(struct control_state *s)
{
        if(!s) {
                return;
        }

        write_all(s->local_fd[0], "quit\r\n", 6);

        pthread_join(s->thread_id, NULL);

        close(s->local_fd[0]);
}


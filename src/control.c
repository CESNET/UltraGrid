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

};

#define CONTROL_EXIT -1
#define CONTROL_CLOSE_HANDLE -2

static bool parse_msg(char *buffer, char buffer_len, /* out */ char *message, int *new_buffer_len);
static int process_msg(int client_fd, char *message);
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

int control_init(int port, struct control_state **state)
{
        struct control_state *s = (struct control_state *) malloc(sizeof(struct control_state));

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

#define prefix_matches(x) strncasecmp(message, x, strlen(x)) == 0
#define suffix(x) message + strlen(x)

/**
  * @retval -1 exit thread
  * @retval -2 close handle
  */
static int process_msg(int client_fd, char *message)
{
        int ret = 0;
        struct response *resp;

        if(strcasecmp(message, "quit") == 0) {
                return CONTROL_EXIT;
        } else if(prefix_matches("receiver ")) {
                char *receiver = suffix("receiver ");

                resp =
                        send_message(messaging_instance(), MSG_CHANGE_RECEIVER_ADDRESS, receiver);
        } else if(strcasecmp(message, "bye") == 0) {
                ret = CONTROL_CLOSE_HANDLE;
                resp = new_response(RESPONSE_OK);
        } else {
                resp = new_response(RESPONSE_BAD_REQUEST);
        }

        send_response(client_fd, resp);

        return ret;
}

static void send_response(int fd, struct response *resp)
{
        char buffer[1024];

        snprintf(buffer, sizeof(buffer), "%d %s\r\n", resp->status,
                        response_status_to_text(resp->status));

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

                                int ret = process_msg(cur->fd, msg);
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


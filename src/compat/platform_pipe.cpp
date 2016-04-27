/**
 * @file   src/compat/platform_pipe.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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

#include "debug.h"

#include "compat/platform_pipe.h"

#include <thread>

#ifdef WIN32
#define CLOSESOCKET closesocket
#else
#define CLOSESOCKET close
#endif

using std::thread;

static fd_t open_socket(int *port)
{
        fd_t sock;
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
                return INVALID_SOCKET;
        }

        struct sockaddr_in s_in;
        memset(&s_in, 0, sizeof(s_in));
        s_in.sin_family = AF_INET;
        s_in.sin_addr.s_addr = htonl(INADDR_ANY);
        s_in.sin_port = htons(0);
        if (::bind(sock, (const struct sockaddr *) &s_in,
                        sizeof(s_in)) != 0) {
                return INVALID_SOCKET;
        }
        if (listen(sock, 10) != 0) {
                return INVALID_SOCKET;
        }
        socklen_t len = sizeof(s_in);
        if (getsockname(sock, (struct sockaddr *) &s_in, &len) != 0) {
                CLOSESOCKET(sock);
                return INVALID_SOCKET;
        }
        *port = ntohs(s_in.sin_port);
        return sock;
}

static fd_t connect_to_socket(int local_port)
{
        struct sockaddr_in s_in;
        memset(&s_in, 0, sizeof(s_in));
        s_in.sin_family = AF_INET;
        s_in.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        s_in.sin_port = htons(local_port);
        fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd == INVALID_SOCKET) {
                return INVALID_SOCKET;
        }
        int ret;
        ret = connect(fd, (struct sockaddr *) &s_in,
                                sizeof(s_in));
        if (ret != 0) {
                CLOSESOCKET(fd);
                return INVALID_SOCKET;
        }

        return fd;
}

struct params {
        int port;
        fd_t sock;
};

static void * worker(void *args)
{
        struct params *p = (struct params *) args;
        p->sock = connect_to_socket(p->port);

        return NULL;
}


int platform_pipe_init(fd_t p[2])
{
        struct params par;
        fd_t sock = open_socket(&par.port);
        if (sock == INVALID_SOCKET) {
                return -1;
        }

        thread thr(worker, &par);

        p[0] = accept(sock, NULL, NULL);
        if (p[0] == INVALID_SOCKET) {
                CLOSESOCKET(sock);
                return -1;
        }
        thr.join();
        p[1] = par.sock;
        if (p[1] == INVALID_SOCKET) {
                CLOSESOCKET(sock);
                return -1;
        }
        CLOSESOCKET(sock);

        return 0;
}

void platform_pipe_close(fd_t pipe)
{
        CLOSESOCKET(pipe);
}


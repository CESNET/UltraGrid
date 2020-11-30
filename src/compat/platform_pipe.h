/**
 * @file   compat/platform_pipe.h
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

#ifndef platform_pipe_h
#define platform_pipe_h

#include "config_unix.h"
#include "config_win32.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int platform_pipe_init(fd_t pipe[2]);
void platform_pipe_close(fd_t pipe);

#ifdef _WIN32
#define PLATFORM_PIPE_READ(fd, buf, len) recv(fd, buf, len, 0)
#define PLATFORM_PIPE_WRITE(fd, buf, len) send(fd, buf, len, 0)
#else
// the fallback implementation of platform_pipe is a plain pipe for which
// doesn't work send()/recv(). read() and write(), however, works for both
// socket and pipe (but doesn't so in Windows, therefore there is used
// send()/recv()).
//
// PLATFORM_PIPE_READ()/PLATFORM_PIPE_WRITE() can be also safely be used if
// unsure if underlying fd is a socket or a pipe.
#define PLATFORM_PIPE_READ read
#define PLATFORM_PIPE_WRITE write
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* defined platform_pipe_h */

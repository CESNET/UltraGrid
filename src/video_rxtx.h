/**
 * @file   video_rxtx.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET z.s.p.o.
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

#ifndef VIDEO_RXTX_H_
#define VIDEO_RXTX_H_

#include <chrono>
#include <map>
#include <memory>
#include <string>

#include "module.h"

#define VIDEO_RXTX_ABI_VERSION 2

struct display;
struct module;
struct video_compress;
struct video_export;
struct video_frame;

class video_rxtx;

union param_u {
        void * ptr;
        int    i;
        long   l;
        bool   b;
};

struct video_rxtx_info {
        const char *long_name;
        class video_rxtx *(*create)(std::map<std::string, param_u> const &params);
};

#define STREAM_PAUSED_PLAY 1

class video_rxtx {
public:
        virtual ~video_rxtx();
        void send(std::shared_ptr<struct video_frame>);
        static const char *get_long_name(std::string const & short_name);
        static void *receiver_thread(void *arg) {
                video_rxtx *rxtx = static_cast<video_rxtx *>(arg);
                return rxtx->get_receiver_thread()(arg);
        }
        bool supports_receiving() {
                return get_receiver_thread() != NULL;
        }
        virtual void join();
        static video_rxtx *create(std::string const & name, std::map<std::string, param_u> const &);
        int m_port_id;
protected:
        video_rxtx(std::map<std::string, param_u> const &);
        int check_sender_messages();
        bool m_paused;
        bool m_report_paused_play;
        struct module m_sender_mod;
        struct module m_receiver_mod;
        int m_rxtx_mode;
        struct module *m_parent;
private:
        void start();
        virtual void send_frame(std::shared_ptr<video_frame>) = 0;
        virtual void *(*get_receiver_thread())(void *arg) = 0;
        static void *sender_thread(void *args);
        void *sender_loop();
        virtual struct response *process_sender_message(struct msg_sender *, int *status) {
                *status = 0;
                return NULL;
        }

        struct compress_state *m_compression;
        pthread_mutex_t m_lock;
        struct video_export *m_video_exporter;

        pthread_t m_thread_id;
        bool m_poisoned, m_joined;
};

class video_rxtx_loader {
public:
        video_rxtx_loader();
};

#endif // VIDEO_RXTX_H_


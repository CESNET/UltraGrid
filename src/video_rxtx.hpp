/**
 * @file   video_rxtx.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET zájmové sdružení právnických osob
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

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "host.h"
#include "module.h"

#define VIDEO_RXTX_ABI_VERSION 3

struct audio_desc;
struct display;
struct module;
struct video_compress;
struct exporter;
struct video_frame;

/**
 * @todo
 * get rid of this altogether and pass 2 structs (common + video_rxtx opts)
 */
union param_u {
        void          *ptr;
        const void    *cptr;
        volatile void *vptr;
        const char    *str;
        int            i;
        long           l;
        long long      ll;
        bool           b;
};

struct video_rxtx_info {
        const char *long_name;
        struct video_rxtx *(*create)(std::map<std::string, param_u> const &params);
};

struct video_rxtx {
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
        /**
         * If overridden, children must call also video_rxtx::join()
         */
        virtual void join();
        static video_rxtx *create(std::string const & name, std::map<std::string, param_u> const &);
        static void list(bool full);
        virtual void set_audio_spec(const struct audio_desc *desc,
                                    int audio_rx_port, int audio_tx_port,
                                    bool ipv6);
        std::string m_port_id;
protected:
        video_rxtx(std::map<std::string, param_u> const &);
        void check_sender_messages();
        struct module m_sender_mod;
        struct module m_receiver_mod;
        int m_rxtx_mode;
        unsigned long long int m_frames_sent;
        struct common_opts m_common;
        bool m_should_exit = false;

private:
        void start();
        virtual void send_frame(std::shared_ptr<video_frame>) noexcept = 0;
        virtual void *(*get_receiver_thread() noexcept)(void *arg) = 0;
        static void *sender_thread(void *args);
        void *sender_loop();
        virtual struct response *process_sender_message(struct msg_sender *) {
                return NULL;
        }
        static void should_exit(void *state);

        struct compress_state *m_compression = nullptr;
        pthread_mutex_t m_lock;

        pthread_t m_thread_id;
        bool m_poisoned, m_joined;

        video_desc       m_video_desc{};
        std::atomic<codec_t> m_input_codec{};
};

class video_rxtx_loader {
public:
        video_rxtx_loader();
};

#endif // VIDEO_RXTX_H_


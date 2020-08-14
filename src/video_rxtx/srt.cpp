/**
 * @file   video_rxtx/srt.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET z.s.p.o.
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

#include <array>
#include <condition_variable>
#include <memory> // std::unique_ptr
#include <mutex>
#include <srt/srt.h>
#include <string>
#include <thread>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rtp/rtp_types.h"
#include "rtp/video_decoders.h"
#include "transmit.h"
#include "types.h"
#include "ug_runtime_error.hpp"
#include "utils/net.h"
#include "video.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_rxtx/srt.hpp"
#include "video_rxtx.h"

constexpr const char *MOD_NAME = "[SRT] ";
constexpr size_t MAX_BUFFER_LEN = 5'000'000;
constexpr size_t ERR_MSG_BUF_LEN = 1024;

using namespace std;

class srt_video_rxtx: public video_rxtx {
public:
        srt_video_rxtx(std::map<std::string, param_u> const &);
        ~srt_video_rxtx();
private:
        void send_frame(std::shared_ptr<video_frame>) override;
        void *(*get_receiver_thread())(void *arg) override {
                return receiver_thread;
        }
        void listen();
        void connect(std::string receiver, uint16_t tx_port);

        static void *receiver_thread(void *arg) {
                auto *s = static_cast<srt_video_rxtx *>(arg);
                s->receiver_loop();
                return nullptr;
        }
        static void cancel_receiver(void *arg);
        void receiver_loop();
        void reconnect();

        struct display *m_display_device;

        SRTSOCKET     m_socket_listen{};
        SRTSOCKET     m_socket_rx{};
        SRTSOCKET     m_socket_tx{};

        std::thread m_listener_thread;
        std::thread m_caller_thread;

        std::mutex m_lock;
        std::condition_variable m_reconnect_cv;
        std::condition_variable m_connected_cv;

        bool m_should_exit{false};

        struct srt_decoder {
                struct video_desc network_desc;
                state_decompress *decompress;
        } m_decoder;

        enum class connection_mode_t {
                NONE,
                CALLER,
                LISTENER,
                BOTH,
        } connection_mode;
};

static auto create_socket() {
        auto sock = srt_create_socket();
        int file_mode = SRTT_FILE;
        int yes = 1;
        srt_setsockflag(sock, SRTO_TRANSTYPE, &file_mode, sizeof file_mode);
        srt_setsockflag(sock, SRTO_MESSAGEAPI, &yes, sizeof yes);
        return sock;
}

srt_video_rxtx::srt_video_rxtx(map<string, param_u> const &params) :
        video_rxtx(params)
{
        using namespace std::string_literals;

        string caller{params.at("receiver").str};
        int rx_port = params.at("rx_port").i;
        int tx_port = params.at("tx_port").i;
        m_display_device = (struct display *) params.at("display_device").ptr;

        if (caller == SRT_LOOPBACK || (is_host_loopback(caller.c_str()) && (m_rxtx_mode & MODE_SENDER) != 0U
                                && (m_rxtx_mode & MODE_RECEIVER) != 0U)) { // listener
                connection_mode = connection_mode_t::LISTENER;
                m_socket_listen = create_socket();

                sockaddr_in sa{};
                sa.sin_family = AF_INET;
                sa.sin_port = htons(rx_port);
                sa.sin_addr.s_addr = INADDR_ANY;
                int st = srt_bind(m_socket_listen, (struct sockaddr *) &sa, sizeof sa);
                if (st != 0) {
                        array<char, ERR_MSG_BUF_LEN> err_buf{"Unknown error"};
                        strerror_r(errno, err_buf.data(), err_buf.size());
                        throw ug_runtime_error{"srt_bind: "s + err_buf.data()};
                }

                srt_listen(m_socket_listen, 5);

                m_listener_thread = thread(&srt_video_rxtx::listen, this);
        }

        if (is_host_loopback(caller.c_str()) && (m_rxtx_mode & MODE_SENDER) != 0U && (m_rxtx_mode & MODE_RECEIVER) != 0U) { // sending to ourselves (listener+caller)
                connection_mode = connection_mode_t::BOTH;
                m_socket_rx = create_socket();

                sockaddr_in sa{};
                sa.sin_family = AF_INET;
                sa.sin_port = htons(tx_port);
                inet_aton("127.0.0.1", &sa.sin_addr);
                LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Trying to connect to ourselves.\n";
                int rc = srt_connect(m_socket_rx, (struct sockaddr *) &sa, sizeof sa);
                assert(rc >= 0);
        } else if (caller != SRT_LOOPBACK) { // calller only
                connection_mode = connection_mode_t::CALLER;
                m_caller_thread = thread(&srt_video_rxtx::connect, this, caller, tx_port);
                m_reconnect_cv.notify_one();
        }

        if (m_rxtx_mode & MODE_RECEIVER) {
                register_should_exit_callback(m_parent, cancel_receiver, this);
        }
}

void srt_video_rxtx::listen()
{
        struct sockaddr_storage ss;
        int ss_len = sizeof ss;

        int epid = srt_epoll_create();
        int events = SRT_EPOLL_IN | SRT_EPOLL_ERR;
        if (SRT_ERROR == srt_epoll_add_usock(epid, m_socket_listen, &events))
        {
                LOG(LOG_LEVEL_ERROR) << "srt_epoll_add_usock: " << srt_getlasterror_str() << endl;
        }

        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Listening started.\n";

        while (!m_should_exit) {
                int srtrfdslen = 1;
                const int timeout_ms = 300;
                SRTSOCKET srtrfds[srtrfdslen];
                int n = srt_epoll_wait(epid, &srtrfds[0], &srtrfdslen, nullptr, nullptr, timeout_ms, nullptr, nullptr, nullptr, nullptr);

                if (n != 1) {
                        continue;
                }

                try {
                        m_socket_tx = srt_accept(m_socket_listen, (struct sockaddr *) &ss, &ss_len);
                } catch (std::exception &e) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Accept - exception: " << e.what() << "\n";
                        return;
                }

                if (m_socket_tx > 0) {
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Connection accepted.\n";
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to accept: "  << srt_getlasterror_str() << "\n";
                }
                if (connection_mode == connection_mode_t::LISTENER) {
                        m_socket_rx = m_socket_tx;
                }
                m_connected_cv.notify_one();
        }
}

// caller only
void srt_video_rxtx::connect(string receiver, uint16_t tx_port)
{
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Connecting to: " << receiver << ":" << tx_port << "\n";

        while (true) {
                unique_lock<mutex> lk(m_lock);
                m_reconnect_cv.wait(lk, [&]() { return m_should_exit || m_socket_rx == 0 || m_socket_tx == 0; });
                if (m_should_exit) {
                        return;
                }
                lk.unlock();
                struct addrinfo hints{};
                struct addrinfo *res0{nullptr};
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_DGRAM;
                string port_str = to_string(tx_port);
                if (getaddrinfo(receiver.c_str(), port_str.c_str(), &hints, &res0) != 0) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to resolve: " << receiver << "!\n";
                        break;
                }

                int fd = create_socket();

                for (struct addrinfo *res = res0; res != nullptr; res = res->ai_next) {
                        int rc = srt_connect(fd, res->ai_addr, res->ai_addrlen);
                        if (rc >= 0) {
                                m_socket_tx = m_socket_rx = fd;
                                m_connected_cv.notify_one();
                        }
                }
                if (m_socket_tx) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Connected to: " << receiver << ".\n";
                } else {
                        srt_close(fd);
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to connect to " << receiver << ": " << srt_getlasterror_str() << "\n";
                }
        }
}

void srt_video_rxtx::reconnect() {
        unique_lock<mutex> lk(m_lock);
        srt_close(m_socket_tx);
        m_socket_rx = m_socket_tx = 0;
        m_reconnect_cv.notify_one();
}

void srt_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame)
{
        vector<char> buf(tx_frame->tiles[0].data_len + sizeof(video_payload_hdr_t));
        format_video_header(tx_frame.get(), 0, 0 /* unused */, reinterpret_cast<uint32_t *>(buf.data()));
        copy(tx_frame->tiles[0].data, tx_frame->tiles[0].data + tx_frame->tiles[0].data_len, &buf.at(sizeof(video_payload_hdr_t)));

        if (m_socket_tx != 0) {
                if (srt_send(m_socket_tx, buf.data(), buf.size()) == -1) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "srt_send: " << srt_getlasterror_str() << "\n";
                }
        }
}

void srt_video_rxtx::cancel_receiver(void *arg)
{
        auto *s = static_cast<srt_video_rxtx *>(arg);
        s->m_should_exit = true;
        srt_close(s->m_socket_rx);
        s->m_connected_cv.notify_one();
}

void srt_video_rxtx::receiver_loop()
{
        while (!m_should_exit) {
                if (m_socket_rx == 0) {
                        unique_lock<mutex> _(m_lock);
                        m_connected_cv.wait(_, [&]{ return m_should_exit || m_socket_rx != 0; });
                }

                vector<char> buffer(MAX_BUFFER_LEN);
                int size = 0;
                try {
                        size = srt_recv(m_socket_rx, buffer.data(), buffer.size());
                        if (size <= 0) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "srt_recv returned " << size << ": " << srt_getlasterror_str() << "\n";
                                reconnect();
                                continue;
                        }
                } catch (exception const &e) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "srt_recv exception " << e.what() << "!\n";
                        continue;
                }
                video_desc desc;
                if (!parse_video_hdr(reinterpret_cast<uint32_t *>(buffer.data()), &desc)) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to parse buffer header!\n";
                        continue;
                }

                if (desc != m_decoder.network_desc) {
                        video_desc display_desc{desc};

                        if (is_codec_opaque(desc.color_spec)) {
                                display_desc.color_spec = UYVY;

                                if (!decompress_init_multi(desc.color_spec, VIDEO_CODEC_NONE, display_desc.color_spec, &m_decoder.decompress, 1)) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to create decompress!\n";
                                        continue;
                                }

                                if (!decompress_reconfigure(m_decoder.decompress,
                                                        desc, 0, 8, 16, vc_get_linesize(desc.width, display_desc.color_spec), display_desc.color_spec)) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to configure decompress!\n";
                                        continue;
                                }
                        }

                        if (display_reconfigure(m_display_device, display_desc, VIDEO_NORMAL) != TRUE) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot reconfigure display!\n";
                                continue;
                        } else {
                                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Succesfully reconfigured to " << desc << "\n";
                                m_decoder.network_desc = desc;
                        }
                }

                struct video_frame *f = display_get_frame(m_display_device);
                char *video_data = &buffer[sizeof(video_payload_hdr_t)];
                size -= sizeof(video_payload_hdr_t);
                if (is_codec_opaque(desc.color_spec)) {
                        decompress_frame(m_decoder.decompress, reinterpret_cast<unsigned char *>(f->tiles[0].data), reinterpret_cast<unsigned char *>(video_data), size, 0, nullptr, nullptr);
                } else {
                        if ((unsigned int) size > f->tiles[0].data_len) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Discarding data!\n";
                                size = f->tiles[0].data_len;
                        }
                        memcpy(f->tiles[0].data, video_data, size);
                }
                display_put_frame(m_display_device, f, PUTF_BLOCKING);
        }

        display_put_frame(m_display_device, nullptr, PUTF_BLOCKING); // poison pill
}

srt_video_rxtx::~srt_video_rxtx()
{
        {
                unique_lock<mutex> lk(m_lock);
                m_should_exit = true;
                m_reconnect_cv.notify_one();
        }
        if (m_listener_thread.joinable()) {
                m_listener_thread.join();
        }
        if (m_caller_thread.joinable()) {
                m_caller_thread.join();
        }

        srt_close(m_socket_listen);
        srt_close(m_socket_rx);
        if (m_socket_tx != m_socket_rx) {
                srt_close(m_socket_tx);
        }
}

static video_rxtx *create_video_rxtx_srt(std::map<std::string, param_u> const &params)
{
        return new srt_video_rxtx(params);
}

static const struct video_rxtx_info srt_video_rxtx_info = {
        "srt",
        create_video_rxtx_srt
};

REGISTER_MODULE(srt, &srt_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);


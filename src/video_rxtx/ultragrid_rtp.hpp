/**
 * @file   video_rxtx/ultragrid_rtp.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET, zájmové sdružení právnických osob
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

#ifndef VIDEO_RXTX_ULTRAGRID_RTP_H_
#define VIDEO_RXTX_ULTRAGRID_RTP_H_

#include "video_rxtx/rtp.hpp"

#include <condition_variable>
#include <cstddef>             // for size_t
#include <cstdint>             // for uint32_t
#include <list>
#include <map>
#include <memory>              // for shared_ptr
#include <mutex>
#include <string>

#ifdef _WIN32
#include <basetsd.h>           // for SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>         // for ssize_t
#endif

#include "types.h"             // for video_frame (ptr only), video_mode

union param_u;

class ultragrid_rtp_video_rxtx : public rtp_video_rxtx {
public:
        ultragrid_rtp_video_rxtx(std::map<std::string, param_u> const &);
        virtual ~ultragrid_rtp_video_rxtx();
        void join() override;
        uint32_t get_ssrc();

        // transcoder functions
        friend ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count);
private:
        static void *receiver_thread(void *arg);
        virtual void send_frame(std::shared_ptr<video_frame>) noexcept override;
        void *receiver_loop();
        static void *send_frame_async_callback(void *arg);
        virtual void send_frame_async(std::shared_ptr<video_frame>);
        virtual void *(*get_receiver_thread() noexcept)(void *arg) override;

        void receiver_process_messages();
        void remove_display_from_decoders();
        struct vcodec_state *new_video_decoder(struct display *d);
        static void destroy_video_decoder(void *state);

        enum video_mode  m_decoder_mode;
        struct display  *m_display_device;
        std::list<struct display *> m_display_copies; ///< some displays can be "forked"
                                                      ///< and used simultaneously from
                                                      ///< multiple decoders, here are
                                                      ///< saved forked states

        /**
         * This variables serve as a notification when asynchronous sending exits
         * @{ */
        bool             m_async_sending;
        std::condition_variable m_async_sending_cv;
        std::mutex       m_async_sending_lock;
        /// @}

        long long int m_send_bytes_total;
        struct control_state *m_control;

        long long int m_nano_per_frame_actual_cumul = 0;
        long long int m_nano_per_frame_expected_cumul = 0;
        long long int m_compress_millis_cumul = 0;
};

#endif // VIDEO_RXTX_ULTRAGRID_RTP_H_


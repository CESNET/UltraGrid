/**
 * @file   video_rxtx/ultragrid_rtp.h
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

#ifndef VIDEO_RXTX_ULTRAGRID_RTP_H_
#define VIDEO_RXTX_ULTRAGRID_RTP_H_

#include "video_rxtx.h"
#include "video_rxtx/rtp.h"

class ultragrid_rtp_video_rxtx : public rtp_video_rxtx {
public:
        ultragrid_rtp_video_rxtx(struct module *parent, struct video_export *video_exporter,
                        const char *requested_compression, const char *requested_encryption,
                        const char *receiver, int rx_port, int tx_port,
                        bool use_ipv6, const char *mcast_if, const char *requested_video_fec, int mtu,
                        long packet_rate, enum video_mode decoder_mode, const char *postprocess,
                        struct display *display_device) :
                rtp_video_rxtx(parent, video_exporter, requested_compression, requested_encryption,
                                receiver, rx_port, tx_port,
                                use_ipv6, mcast_if, requested_video_fec, mtu, packet_rate)
        {
                gettimeofday(&m_start_time, NULL);
                m_decoder_mode = decoder_mode;
                m_postprocess = postprocess;
                m_display_device = display_device;
                m_requested_encryption = requested_encryption;
        }
        virtual ~ultragrid_rtp_video_rxtx();
        static void *receiver_thread(void *arg) {
                ultragrid_rtp_video_rxtx *s = static_cast<ultragrid_rtp_video_rxtx *>(arg);
                return s->receiver_loop();
        }
        void *receiver_loop();
protected:
        virtual void send_frame(struct video_frame *);
private:
        virtual void *(*get_receiver_thread())(void *arg) {
                return receiver_thread;
        }

        void receiver_process_messages();
        void remove_display_from_decoders();
        struct vcodec_state *new_video_decoder();
        static void destroy_video_decoder(void *state);

        struct timeval m_start_time;

        enum video_mode  m_decoder_mode;
        const char      *m_postprocess;
        struct display  *m_display_device;
        const char      *m_requested_encryption;
};

#endif // VIDEO_RXTX_ULTRAGRID_RTP_H_


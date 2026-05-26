/**
 * @file   video_rxtx/ultragrid_rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include "video_rxtx/ultragrid_rtp.h"

#include <cassert>             // for assert
#include <condition_variable>  // for condition_variable
#include <cstddef>             // for size_t
#include <cstdint>             // for uint32_t
#include <cstdio>              // for fprintf, stderr
#include <cstdlib>             // for free, calloc
#include <cstring>             // for strcmp
#include <list>                // for list
#include <memory>              // for shared_ptr
#include <mutex>               // for mutex
#include <string>              // for basic_string, operator<, operator==
#include <utility>             // for pair
// IWYU pragma: no_include <sys/time.h> # via tv.h
// IWYU pragma: no_include <iterator>   # std::pair is rather in utility

#ifdef _WIN32
#include <basetsd.h>           // for SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>         // for ssize_t
#endif

#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "pdb.h"
#include "rtp/audio_decoders.h" // for decode_audio_frame
#include "rtp/fec.h"            // for fec
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/video_decoders.h"
#include "tfrc.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"            // for video_frame (ptr only), video_mode
#include "utils/color_out.h"  // for TBOLD, color_printf
#include "utils/lock_guard.h" // for ultragrid::pthread_mutex_guard
#include "utils/macros.h"     // for to_fourcc
#include "utils/misc.h"       // for format_in_si_units
#include "utils/text.h"       // for wrap_paragraph
#include "utils/thread.h"
#include "utils/worker.h"
#include "video_display.h"
#include "video_rxtx.h"
#include "video_rxtx/rtp_common.h" // for rtp_video_rxtx

constexpr uint32_t MAGIC = to_fourcc('V', 'X', 'u', 'r');
#define MOD_NAME "[rxtx/ultragrid_rtp] "

using namespace std;
using ultragrid::pthread_mutex_guard;

struct ultragrid_rtp_video_rxtx {
        const uint32_t magic;
        ultragrid_rtp_video_rxtx(const struct vrxtx_params *params,
                                 const struct common_opts  *common);
        virtual ~ultragrid_rtp_video_rxtx();
        virtual void send_frame(std::shared_ptr<video_frame>) noexcept;
        void join();
        static void *receiver_thread(void *arg);

        struct rtp_rxtx_common *m_rtp_common;
        void *receiver_loop();
        static void *send_frame_async_callback(void *arg);
        virtual void send_frame_async(std::shared_ptr<video_frame>);

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
        bool             m_async_sending = false;
        std::condition_variable m_async_sending_cv;
        std::mutex       m_async_sending_lock;
        /// @}

        long long int m_send_bytes_total;
        struct control_state *m_control;
        struct module *m_parent;

        time_ns_t m_start_time;
        // video
        long long int m_nano_per_frame_actual_cumul = 0;
        long long int m_nano_per_frame_expected_cumul = 0;
        long long int m_compress_millis_cumul = 0;

        struct module *m_receiver_mod{};

        bool m_should_exit = false;
        static void should_exit(void *state);
};

ultragrid_rtp_video_rxtx::ultragrid_rtp_video_rxtx(
    const struct vrxtx_params *params, const struct common_opts *common) :
        magic(MAGIC),
        m_decoder_mode(params->decoder_mode),
        m_display_device(params->display_device),
        m_send_bytes_total(0),
        m_control(get_control_state(common->parent)),
        m_parent(common->parent),
        m_start_time(common->start_time),
        m_receiver_mod(params->receiver_mod)
{

        if (get_commandline_param("decoder-use-codec") != nullptr && "help"s == get_commandline_param("decoder-use-codec")) {
                destroy_video_decoder(new_video_decoder(m_display_device));
                throw 1;
        }
        m_rtp_common = rtp_rxtx_common_init(params, common);
        if (m_rtp_common == nullptr) {
                throw -1;
        }

}

ultragrid_rtp_video_rxtx::~ultragrid_rtp_video_rxtx()
{
        for (auto d : m_display_copies) {
                display_done(d);
        }
        rtp_rxtx_common_done(m_rtp_common);
}

void ultragrid_rtp_video_rxtx::join()
{
        unique_lock<mutex> lk(m_async_sending_lock);
        m_async_sending_cv.wait(lk, [this]{return !m_async_sending;});
}

void *ultragrid_rtp_video_rxtx::receiver_thread(void *arg) {
        ultragrid_rtp_video_rxtx *s = static_cast<ultragrid_rtp_video_rxtx *>(arg);
        assert(s->magic == MAGIC);
        return s->receiver_loop();
}

using async_data = pair<ultragrid_rtp_video_rxtx *, shared_ptr<video_frame>>;

void
ultragrid_rtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        struct rtp_rxtx_medium *video =
            &m_rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(m_rtp_common, TX_MEDIA_VIDEO);

        if (video->fec_state != nullptr) {
                struct video_frame *f = fec_encode_video_frame(
                    video->fec_state, tx_frame.get());
                tx_frame =
                    std::shared_ptr<video_frame>(f, f->callbacks.dispose);
        }

        auto *data = new async_data(this, tx_frame);

        unique_lock<mutex> lk(m_async_sending_lock);
        m_async_sending_cv.wait(lk, [this]{return !m_async_sending;});
        m_async_sending = true;
        task_run_async_detached(ultragrid_rtp_video_rxtx::send_frame_async_callback,
                        (void *) data);
}

void *ultragrid_rtp_video_rxtx::send_frame_async_callback(void *arg) {
        auto *data = (async_data *) arg;

        data->first->send_frame_async(data->second);
        delete data;

        return NULL;
}

void
ultragrid_rtp_video_rxtx::send_frame_async(shared_ptr<video_frame> tx_frame)
{
        struct rtp_rxtx_medium *video = &m_rtp_common->medium[TX_MEDIA_VIDEO];
        pthread_mutex_guard lock(video->lock);

        tx_send(video->tx, tx_frame.get(), video->network_device);

        if ((video->rxtx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - m_start_time) / 100'000 * 9; // at 90000 Hz
                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                // receive RTCP
                bool ret = true;
                do {
                        struct timeval timeout { 0, 0 };
                        ret = rtcp_recv_r(video->network_device, &timeout, ts);
                } while (!m_should_exit && ret);
        }

        m_async_sending_lock.lock();
        m_async_sending = false;
        m_async_sending_lock.unlock();
        m_async_sending_cv.notify_all();
}

void ultragrid_rtp_video_rxtx::receiver_process_messages()
{
        struct msg_receiver *msg = nullptr;
        while ((msg = (struct msg_receiver *) check_message(m_receiver_mod))) {
                switch (msg->type) {
                case RECEIVER_MSG_VIDEO_PROP_CHANGED:
                        rtp_rxtx_set_pbuf_delay(
                            &m_rtp_common->medium[TX_MEDIA_VIDEO],
                            1.0 / msg->new_desc.fps);
                        free_message((struct message *) msg,
                                     new_response(RESPONSE_OK, nullptr));
                        break;
                default:
                        assert(0 && "Wrong message passed to "
                                    "ultragrid_rtp_video_rxtx::receiver_"
                                    "process_messages()");
                }
        }
}

/**
 * Removes display from decoders and effectively kills them. They cannot be used
 * until new display assigned.
 */
void ultragrid_rtp_video_rxtx::remove_display_from_decoders() {
        struct rtp_rxtx_medium *video = &m_rtp_common->medium[TX_MEDIA_VIDEO];
        if (video->participants != nullptr) {
                pdb_iter_t it;
                struct pdb_e *cp = pdb_iter_init(video->participants, &it);
                while (cp != NULL) {
                        if (cp->decoder_state) {
                                video_decoder_deactivate(
                                    ((struct vcodec_state *) cp->decoder_state)
                                        ->decoder);
                        }
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
}

void ultragrid_rtp_video_rxtx::destroy_video_decoder(void *state) {
        struct vcodec_state *video_decoder_state = (struct vcodec_state *) state;

        if(!video_decoder_state) {
                return;
        }

        video_decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

struct vcodec_state *ultragrid_rtp_video_rxtx::new_video_decoder(struct display *d) {
        struct vcodec_state *state = (struct vcodec_state *) calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->decoder = video_decoder_init(m_receiver_mod, m_decoder_mode,
                                d, m_rtp_common->encryption);

                if(!state->decoder) {
                        fprintf(stderr, "Error initializing decoder (incorrect '-M' or '-p' option?).\n");
                        free(state);
                        exit_uv(1);
                        return NULL;
                } else {
                        //decoder_register_display(state->decoder, uv->display_device);
                }
        }

        return state;
}

void ultragrid_rtp_video_rxtx::should_exit(void *state) {
        auto *s = (ultragrid_rtp_video_rxtx *) state;
        s->m_should_exit = true;
}

static void
display_buf_increase_warning(int size)
{
        log_msg(LOG_LEVEL_INFO, "\n***\nUnable to set buffer size to %sB.\n",
                format_in_si_units(size));

#if defined _WIN32
        log_msg(LOG_LEVEL_INFO, "See "
                                "https://github.com/CESNET/UltraGrid/wiki/"
                                "Extending-Network-Buffers-%%28Windows%%29 "
                                "for details.\n");
        return;
#endif /* defined _WIN32 */

#ifdef __APPLE__
#define SYSCTL_ENTRY "net.inet.udp.recvspace"
#else
#define SYSCTL_ENTRY "net.core.rmem_max"
#endif
        log_msg(
            LOG_LEVEL_INFO,
            "Please set " SYSCTL_ENTRY " value to %d or greater (see also\n"
            "https://github.com/CESNET/UltraGrid/wiki/OS-Setup-UltraGrid):\n"
#ifdef __APPLE__
            "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
#endif
            "\tsysctl -w " SYSCTL_ENTRY "=%d\n"
            "To make this persistent, add these options (key=value) to "
            "/etc/sysctl.d/60-ultragrid.conf\n"
            "\n***\n\n",
            size,
#ifdef __APPLE__
            size * 4,
#endif /* __APPLE__ */
            size
        );
#undef SYSCTL_ENTRY
}

void *ultragrid_rtp_video_rxtx::receiver_loop()
{
        set_thread_name(__func__);
        struct rtp_rxtx_medium *video = &m_rtp_common->medium[TX_MEDIA_VIDEO];
        struct pdb_e *cp;
        int fr;
        int last_buf_size = rtp_get_recv_buf(video->network_device);

#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_video_decoder(m_display_device);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER

        fr = 1;

        time_ns_t last_not_timeout = 0;

        register_should_exit_callback(m_parent, ultragrid_rtp_video_rxtx::should_exit,
                                      this);

        while (!m_should_exit) {
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (m_start_time - curr_time) / 100'000 * 9; // at 90000 Hz

                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        curr_time = get_time_in_ns();
                        receiver_process_messages();
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                // use longer timeout when we are not receivng any data
                if ((curr_time - last_not_timeout) > NS_IN_SEC) {
                        timeout.tv_usec = 100000;
                } else {
                        timeout.tv_usec = 1000;
                }
                const bool ret = rtp_recv_r(video->network_device, &timeout, ts);

                // timeout
                if (!ret) {
                        // processing is needed here in case we are not receiving any data
                        receiver_process_messages();
                        //printf("Failed to receive data\n");
                } else {
                        last_not_timeout = curr_time;
                }

                /* Decode and render for each participant in the conference... */
                pdb_iter_t it;
                cp = pdb_iter_init(video->participants, &it);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               curr_time));
                        }

                        if(cp->decoder_state == NULL &&
                                        !pbuf_is_empty(cp->playout_buffer)) { // the second check is needed because we want to assign display to participant that really sends data
#ifdef SHARED_DECODER
                                cp->decoder_state = shared_decoder;
#else
                                // we are assigning our display so we make sure it is removed from other display

                                struct multi_sources_supp_info supp_for_mult_sources;
                                size_t len = sizeof(multi_sources_supp_info);
                                int ret = display_ctl_property(m_display_device,
                                                DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES, &supp_for_mult_sources, &len);
                                if (!ret) {
                                        supp_for_mult_sources.val = false;
                                }

                                struct display *d;
                                if (supp_for_mult_sources.val == false) {
                                        remove_display_from_decoders(); // must be called before creating new decoder state
                                        d = m_display_device;
                                } else {
                                        d = supp_for_mult_sources.fork_display(supp_for_mult_sources.state);
                                        assert(d != NULL);
                                        m_display_copies.push_back(d);
                                }

                                cp->decoder_state = new_video_decoder(d);
                                cp->decoder_state_deleter = destroy_video_decoder;

                                if (cp->decoder_state == NULL) {
                                        log_msg(LOG_LEVEL_FATAL, "Fatal: unable to create decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
#endif // SHARED_DECODER
                        }

                        struct vcodec_state *vdecoder_state = (struct vcodec_state *) cp->decoder_state;

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_video_frame, vdecoder_state)) {
                                fr = 1;
                        }

                        if(vdecoder_state && vdecoder_state->decoded % 100 == 99) {
                                int new_size = vdecoder_state->max_frame_size * 110ull / 100;
                                if(new_size > last_buf_size) {
                                        if (rtp_set_recv_buf(video->network_device, new_size)) {
                                                debug_msg("Recv buffer adjusted to %d\n", new_size);
                                        } else {
                                                display_buf_increase_warning(new_size);
                                        }
                                        last_buf_size = new_size;
                                }
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }

        unregister_should_exit_callback(
            m_parent, ultragrid_rtp_video_rxtx::should_exit, this);

#ifdef SHARED_DECODER
        destroy_video_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders();
#endif //  SHARED_DECODER

        // pass poisoned pill to display
        display_put_frame(m_display_device, NULL, PUTF_BLOCKING);

        return 0;
}

static void usage() {
        color_printf("Transport " TBOLD("ultragrid_rtp")
                     " doesn't take any options.\n\n");
        color_printf("Usage:\n\t" TBOLD("-x ultragrid_rtp")
                     "\n");
}

void
ultragrid_rtp_server_mode_help()
{
        color_printf(TBOLD("server mode")
                     " is one of " TBOLD("NAT traversal")
                     " techniques in UG.\n\n");
        char desc[] =
            "It is useful in cases when at least one end is " TBOLD("outside")
            " NAT. "
            "This end will become the \"server\" while the one behind "
            "NAT the client.\n\n";
        color_printf("%s", wrap_paragraph(desc));
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("uv [uv_args] -S")
                     "\n\t\t the server\n");
        color_printf("\t" TBOLD("uv [uv_args] -C <server_address>")
                     "\n\t\t the client\n");
        color_printf("\nSee "
                     "also: <https://github.com/CESNET/UltraGrid/wiki/"
                     "NAT-traversal#server-mode>\nfor more details.\n");
}

static void *
create_video_rxtx_ultragrid_rtp(const struct vrxtx_params *params,
                                const struct common_opts  *common)
{
        if (strlen(params->protocol_opts) != 0) {
                usage();
                return nullptr;
        }
        return new ultragrid_rtp_video_rxtx(params, common);
}

static void done(void *state) {
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static void join(void *state) {
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        s->join();
}

static void
send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        struct rtp_rxtx_medium *audio =
            &s->m_rtp_common->medium[TX_MEDIA_AUDIO];

        rtp_rxtx_sender_do_housekeeping(s->m_rtp_common, TX_MEDIA_AUDIO);

        struct audio_frame2 *fec_frame = nullptr;
        if (audio->fec_state != nullptr) {
                frame = fec_frame =
                    fec_encode_audio_frame(audio->fec_state, frame);
        }

        audio_tx_send(
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
        delete fec_frame;
}

static bool
ultragrid_rtp_ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        assert(s->magic == MAGIC);
        switch (p) {
        case GET_RTP_COMMON_STATE: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof s->m_rtp_common);
                *len = sizeof s->m_rtp_common;
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy(val, (void *) &s->m_rtp_common, *len);
                return true;
        }
        case SET_ULTRAGRID_RTP_MUTLI_OUT: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof(bool));
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy((void *) &s->m_rtp_common
                           ->playback_supports_multiple_streams,
                       val, sizeof(bool));
                return true;
        }
        }
        MSG(WARNING, "Unexpected property %d queried!\n", (int) p);
        return false;
}

static struct rx_audio_frames *
ultragrid_rtp_recv_audio_frame(void *state)
{
        auto *s = static_cast<ultragrid_rtp_video_rxtx *>(state);
        return rtp_recv_audio_frame(s->m_rtp_common, decode_audio_frame);
}

static const struct video_rxtx_info ultragrid_rtp_video_rxtx_info = {
        .long_name          = "UltraGrid RTP",
        .create             = create_video_rxtx_ultragrid_rtp,
        .done               = done,
        .send_audio_frame   = send_audio_frame,
        .recv_audio_frame   = ultragrid_rtp_recv_audio_frame,
        .send_video_frame   = send_frame,
        .video_recv_routine = ultragrid_rtp_video_rxtx::receiver_thread,
        .ctl_property       = ultragrid_rtp_ctl_property,
        .join_sender        = join,
};

REGISTER_MODULE(ultragrid_rtp, &ultragrid_rtp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);


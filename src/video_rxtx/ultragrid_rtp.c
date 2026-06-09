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

#include <assert.h>  // for assert
#include <pthread.h> // for pthread_mutex_lock, pthread_mutex...
#include <stdint.h>  // for uint32_t
#include <stdio.h>   // for fprintf, stderr
#include <stdlib.h>  // for free, calloc
#include <string.h>  // for strcmp
// IWYU pragma: no_include <sys/time.h> # via tv.h

#include "audio/types.h"       // for audio_frame2_delete
#include "compat/c23.h"        // IWYU pragma: keep
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
#include "utils/macros.h"     // for to_fourcc
#include "utils/misc.h"       // for format_in_si_units
#include "utils/pthread.h"    // for CHK_PTHR, ug_pthread_mutex_ini
#include "utils/text.h"       // for wrap_paragraph
#include "utils/thread.h"
#include "utils/worker.h"
#include "video_display.h"
#include "video_rxtx.h"
#include "video_rxtx/rtp_common.h" // for rtp_video_rxtx

struct audio_frame2;
struct display;

#define MAGIC    to_fourcc('V', 'X', 'u', 'r')
#define MOD_NAME "[rxtx/ultragrid_rtp] "

struct ultragrid_rtp_video_rxtx {
        uint32_t magic;

        struct rtp_rxtx_common *rtp_common;

        enum video_mode  decoder_mode;
        struct display  *display_device;

        struct display  **display_copies; ///< some displays can be "forked"
                                         ///< and used simultaneously from
                                         ///< multiple decoders, here are
                                         ///< saved forked states
        unsigned display_copies_count;

        /**
         * This variables serve as a notification when asynchronous sending exits
         * @{ */
        bool            async_sending;
        pthread_cond_t  async_sending_cv;
        pthread_mutex_t async_sending_lock;
        /// @}

        long long int         send_bytes_total;
        struct module        *parent;

        time_ns_t start_time;

        struct module *receiver_mod;

        bool should_exit;
};

// protoypes
static void usage();
static struct vcodec_state *
new_video_decoder(struct ultragrid_rtp_video_rxtx *s, struct display *d);
static void destroy_video_decoder(void *state);

static void done(void *state)
{
        struct ultragrid_rtp_video_rxtx *s = state;
        for (unsigned i = 0; i < s->display_copies_count; ++i) {
                display_done(s->display_copies[i]);
        }
        rtp_rxtx_common_done(s->rtp_common);
        CHK_PTHR(pthread_cond_destroy(&s->async_sending_cv));
        CHK_PTHR(pthread_mutex_destroy(&s->async_sending_lock));
        free(s);
}

static void *
init(const struct vrxtx_params *params)
{
        if (strlen(params->protocol_opts) > 0) {
                usage();
                return strcmp(params->protocol_opts, "help") == 0 ? INIT_NOERR
                                                                  : nullptr;
        }

        struct ultragrid_rtp_video_rxtx *s = calloc(1, sizeof *s);

        s->magic          = MAGIC;
        s->decoder_mode   = params->decoder_mode;
        s->display_device = params->display_device;
        s->parent         = params->parent;
        s->start_time     = params->start_time;
        s->receiver_mod   = params->receiver_mod;
        ug_pthread_mutex_init(&s->async_sending_lock);
        pthread_cond_init(&s->async_sending_cv, nullptr);
        s->rtp_common = rtp_rxtx_common_init(params);
        if (s->rtp_common == nullptr) {
                done(s);
                return nullptr;
        }

        const char *dec_use_codec = get_commandline_param("decoder-use-codec");
        if (dec_use_codec != nullptr && strcmp(dec_use_codec, "help") == 0) {
                destroy_video_decoder(new_video_decoder(s, params->display_device));
                done(s);
                return INIT_NOERR;
        }
        return s;

}


static void join(void *state) {
        struct ultragrid_rtp_video_rxtx *s = state;
        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        while (s->async_sending) {
                pthread_cond_wait(&s->async_sending_cv, &s->async_sending_lock);
        }
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
}

struct async_data {
        struct ultragrid_rtp_video_rxtx *s;
        struct video_frame              *f;
};

static void *send_video_frame_async_callback(void *arg);

static void
send_video_frame(void *state, struct video_frame *tx_frame)
{
        struct ultragrid_rtp_video_rxtx *s = state;
        struct rtp_rxtx_medium *video =
            &s->rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_VIDEO);

        if (video->fec_state != nullptr) {
                struct video_frame *f = fec_encode_video_frame(
                    video->fec_state, tx_frame);
                tx_frame->callbacks.dispose(tx_frame);
                tx_frame = f;
        }

        struct async_data *data = malloc(sizeof *data);
        data->s = s;
        data->f = tx_frame;

        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        while (s->async_sending) {
                pthread_cond_wait(&s->async_sending_cv, &s->async_sending_lock);
        }
        s->async_sending = true;
        task_run_async_detached(send_video_frame_async_callback, (void *) data);
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
}

static void *send_video_frame_async_callback(void *arg) {
        struct async_data *data = arg;
        struct ultragrid_rtp_video_rxtx *s    = data->s;
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];
        struct video_frame *tx_frame = data->f;
        free(data);

        CHK_PTHR(pthread_mutex_lock(&video->lock));

        tx_send(video->tx, tx_frame, video->network_device);
        tx_frame->callbacks.dispose(tx_frame);

        if ((video->rxtx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - s->start_time) / (100 * 1000 * 9); // at 90000 Hz
                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                // receive RTCP
                bool ret = true;
                do {
                        struct timeval timeout = { 0, 0 };
                        ret = rtcp_recv_r(video->network_device, &timeout, ts);
                } while (!s->should_exit && ret);
        }
        CHK_PTHR(pthread_mutex_unlock(&video->lock));

        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        s->async_sending = false;
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
        CHK_PTHR(pthread_cond_signal(&s->async_sending_cv));

        return nullptr;
}

static void
receiver_process_messages(struct ultragrid_rtp_video_rxtx *s)
{
        struct msg_receiver *msg = nullptr;
        while ((msg = (struct msg_receiver *) check_message(s->receiver_mod))) {
                switch (msg->type) {
                case RECEIVER_MSG_VIDEO_PROP_CHANGED:
                        rtp_rxtx_set_pbuf_delay(
                            &s->rtp_common->medium[TX_MEDIA_VIDEO],
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
static void
remove_display_from_decoders(struct ultragrid_rtp_video_rxtx *s)
{
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];
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

static void
destroy_video_decoder(void *state)
{
        struct vcodec_state *video_decoder_state =
            (struct vcodec_state *) state;

        if(!video_decoder_state) {
                return;
        }

        video_decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

static struct vcodec_state *
new_video_decoder(struct ultragrid_rtp_video_rxtx *s, struct display *d)
{
        struct vcodec_state *state = (struct vcodec_state *) calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->decoder = video_decoder_init(s->receiver_mod, s->decoder_mode,
                                d, s->rtp_common->encryption);

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

static void
should_exit(void *state) {
        struct ultragrid_rtp_video_rxtx *s = state;
        s->should_exit = true;
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

static void *
receiver_thread(void *arg)
{
        struct ultragrid_rtp_video_rxtx *s = arg;
        set_thread_name(__func__);
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];
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

        register_should_exit_callback(s->parent, should_exit, s);

        while (!s->should_exit) {
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (s->start_time - curr_time) / (100 * 1000 * 9); // at 90000 Hz

                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        curr_time = get_time_in_ns();
                        receiver_process_messages(s);
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
                        receiver_process_messages(s);
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
                                size_t len = sizeof(struct multi_sources_supp_info);
                                int ret = display_ctl_property(s->display_device,
                                                DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES, &supp_for_mult_sources, &len);
                                if (!ret) {
                                        supp_for_mult_sources.val = false;
                                }

                                struct display *d;
                                if (supp_for_mult_sources.val == false) {
                                        remove_display_from_decoders(s); // must be called before creating new decoder state
                                        d = s->display_device;
                                } else {
                                        d = supp_for_mult_sources.fork_display(supp_for_mult_sources.state);
                                        assert(d != NULL);
                                        s->display_copies = realloc(
                                            s->display_copies,
                                            (s->display_copies_count + 1) *
                                                sizeof *s->display_copies);
                                        s->display_copies[s->display_copies_count++] = d;
                                }

                                cp->decoder_state = new_video_decoder(s, d);
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

        unregister_should_exit_callback(s->parent, should_exit, s);

#ifdef SHARED_DECODER
        destroy_video_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders(s);
#endif //  SHARED_DECODER

        // pass poisoned pill to display
        display_put_frame(s->display_device, nullptr, PUTF_BLOCKING);

        return nullptr;
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

static void
send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct ultragrid_rtp_video_rxtx *s = state;
        struct rtp_rxtx_medium *audio = &s->rtp_common->medium[TX_MEDIA_AUDIO];

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_AUDIO);

        struct audio_frame2 *fec_frame = nullptr;
        if (audio->fec_state != nullptr) {
                frame = fec_frame =
                    fec_encode_audio_frame(audio->fec_state, frame);
        }

        audio_tx_send(
            s->rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
        audio_frame2_delete(fec_frame);
}

static bool
ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        struct ultragrid_rtp_video_rxtx *s = state;
        assert(s->magic == MAGIC);
        switch (p) {
        case GET_RTP_COMMON_STATE: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof s->rtp_common);
                *len = sizeof s->rtp_common;
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy(val, (void *) &s->rtp_common, *len);
                return true;
        }
        case SET_ULTRAGRID_RTP_MUTLI_OUT: {
                assert(*len >= sizeof(bool));
                memcpy(&s->rtp_common->playback_supports_multiple_streams, val,
                       sizeof(bool));
                return true;
        }
        case SET_RTP_AUD_FRM_SZ: {
                int sz = 0;
                assert(*len >= sizeof sz);
                memcpy(&sz, val, sizeof sz);
                rtp_set_recv_buf(
                    s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, sz);
                return true;
        }
        }
        MSG(WARNING, "Unexpected property %d queried!\n", (int) p);
        return false;
}

static struct rx_audio_frames *
recv_audio_frame(void *state)
{
        struct ultragrid_rtp_video_rxtx *s = state;
        return rtp_recv_audio_frame(s->rtp_common, decode_audio_frame);
}

static const struct video_rxtx_info ultragrid_rtp_video_rxtx_info = {
        .long_name    = "UltraGrid RTP",
        .create       = init,
        .done         = done,
        .ctl_property = ctl_property,

        .send_audio_frame = send_audio_frame,
        .recv_audio_frame = recv_audio_frame,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_video_frame,
        .video_recv_routine = receiver_thread,
        .join_video_sender  = join,
};

REGISTER_MODULE(ultragrid_rtp, &ultragrid_rtp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);


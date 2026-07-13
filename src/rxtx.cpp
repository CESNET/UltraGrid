/**
 * @file   rxtx.cpp
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

#include "rxtx.h"

#include "types.h"
#include <atomic>
#include <cassert>           // for assert
#include <cstdio>            // for printf
#include <cstring>           // for NULL, memset, strcasecmp, strcmp
#include <memory>
#include <pthread.h>         // for pthread_join, pthread_create, pthread_equal
#include <sstream>
#include <string>
#include <utility>

#include "audio/types.h"
#include "debug.h"
#include "export.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "utils/macros.h"    // for snprintf_ch, to_fourcc
#include "utils/pthread.h"   // for CHK_PTHR, PTHREAD_NULL
#include "utils/thread.h"
#include "video.h"
#include "video_codec.h"     // for get_codec_name
#include "video_compress.h"
#include "video_frame.h"     // for video_desc_from_frame

constexpr char DEFAULT_VIDEO_COMPRESSION[] = "none";

#define MOD_NAME "[rxtx] "
#define MAGIC to_fourcc('R', 'X', 'T', 'X')

using std::shared_ptr;
using std::ostringstream;
using std::string;

struct rxtx {
public:
        const uint32_t magic = MAGIC;
        virtual ~rxtx() noexcept;
        void               send_vframe(std::shared_ptr<struct video_frame>) noexcept;
        static const char *get_long_name(std::string const &short_name) noexcept;
        /**
         * If overridden, children must call also video_rxtx::join()
         */
        virtual void       join() noexcept;
        static rxtx       *create(std::string const  &proto,
                                  struct rxtx_params *params) noexcept;
        static void        list(bool full) noexcept;
        void set_audio_spec(const struct audio_desc *desc, int audio_rx_port,
                            int audio_tx_port, bool ipv6) noexcept;

        const struct rxtx_info *m_impl_funcs = nullptr;
        void                   *m_impl_state = nullptr;

        enum rxtx_mode rxtx_mode[NUM_TX_MEDIA] = {};

protected:
        rxtx(struct rxtx_params *params);
        void check_sender_messages();

        struct module m_sender_mod;
        struct module m_receiver_mod;
        unsigned long long int m_frames_sent = 0ull;
        struct exporter *m_video_exporter; ///< currently, this is used for video,
                                           ///< audio is exported otherwhere

private:
  static void *video_sender_thread(void *args);
  void        *video_sender_loop();

  struct compress_state *m_video_compression = nullptr;

  pthread_t m_video_sender_thread_id   = PTHREAD_NULL;
  bool      m_video_sender_poisoned    = false;
  pthread_t m_video_receiver_thread_id = PTHREAD_NULL;

  video_desc           m_video_desc{};
  std::atomic<codec_t> m_input_video_codec{};
};

rxtx::rxtx(struct rxtx_params *params)
    : m_video_exporter(params->video_exporter)
{
        module_init_default(&m_sender_mod);
        m_sender_mod.cls = MODULE_CLASS_SENDER;
        module_register(&m_sender_mod, params->parent);

        module_init_default(&m_receiver_mod);
        m_receiver_mod.cls = MODULE_CLASS_RECEIVER;
        module_register(&m_receiver_mod, params->parent);
}

rxtx::~rxtx() noexcept
{
        join();
        // video sender has not been created (error during init) but compression
        // was - we need to stop its thread
        if (!m_video_sender_poisoned && m_video_compression != nullptr) {
                send_vframe(NULL);
                compress_pop(m_video_compression);
        }
        if (m_impl_funcs != nullptr && m_impl_state != nullptr) {
                m_impl_funcs->done(m_impl_state);
        }
        compress_done(m_video_compression);
        module_done(&m_receiver_mod);
        module_done(&m_sender_mod);
}

void
rxtx::join() noexcept
{
        if (!pthread_equal(m_video_receiver_thread_id, PTHREAD_NULL)) {
                CHK_PTHR(pthread_join(m_video_receiver_thread_id, nullptr));
                m_video_receiver_thread_id = PTHREAD_NULL;
        }

        if (!pthread_equal(m_video_sender_thread_id, PTHREAD_NULL)) {
                send_vframe(nullptr); // pass poisoned pill
                CHK_PTHR(pthread_join(m_video_sender_thread_id, nullptr));
                if (m_impl_funcs != nullptr &&
                    m_impl_funcs->join_video_sender != nullptr) {
                        m_impl_funcs->join_video_sender(m_impl_state);
                }
                m_video_sender_thread_id = PTHREAD_NULL;
        }
}

const char *
rxtx::get_long_name(string const &short_name) noexcept
{
        const auto *vri = static_cast<const rxtx_info *>(
            load_library(short_name.c_str(), LIBRARY_CLASS_RXTX,
                         RXTX_ABI_VERSION));

        if (vri != nullptr) {
                return vri->long_name;
        }
        return "(ERROR)";
}

void
rxtx::send_vframe(shared_ptr<video_frame> frame) noexcept
{
        if (!frame && m_video_sender_poisoned) {
                return;
        }
        if (!frame) {
                m_video_sender_poisoned = true;
        } else {
                m_input_video_codec = frame->color_spec;
        }
        compress_frame(m_video_compression, std::move(frame));
}

void *rxtx::video_sender_thread(void *args) {
        return static_cast<rxtx *>(args)->video_sender_loop();
}

void rxtx::check_sender_messages() {
        // process external messages
        struct message *msg_external = nullptr;
        while((msg_external = check_message(&m_sender_mod))) {
                struct response *r = nullptr;
                auto *msg = (struct msg_sender *) msg_external;
                if (msg->type == SENDER_MSG_QUERY_VIDEO_MODE) {
                        if (!m_video_desc) {
                                r = new_response(RESPONSE_NO_CONTENT, nullptr);
                        } else {
                                ostringstream oss;
                                oss << m_video_desc
                                    << " (input " << get_codec_name(m_input_video_codec)
                                    << ")";
                                r = new_response(RESPONSE_OK,
                                                 oss.str().c_str());
                        }
                } else {
                        char buf[200];
                        snprintf_ch(buf, "Unexpected sender message type %d",
                                    msg->type);
                        MSG(WARNING, "%s\n", buf);
                        r = new_response(RESPONSE_BAD_REQUEST, buf);
                }

                free_message(msg_external, r);
        }
}

namespace {
struct shared_ptr_udata {
        shared_ptr<video_frame> frame;
        void (*orig_dispose)(struct video_frame *);
        void *orig_dispose_udata;
};
} // namespace

static void
shared_ptr_dispose(struct video_frame *f)
{
        auto *d = (struct shared_ptr_udata *) f->callbacks.dispose_udata;
        f->callbacks.dispose       = d->orig_dispose;
        f->callbacks.dispose_udata = d->orig_dispose_udata;
        delete d;
}

static struct video_frame *
shared_vf_to_plain(shared_ptr<video_frame> &&frame)
{
        struct video_frame *f = frame.get();
        auto *d = new shared_ptr_udata{ .frame        = std::move(frame),
                                        .orig_dispose = f->callbacks.dispose,
                                        .orig_dispose_udata =
                                            f->callbacks.dispose_udata };
        f->callbacks.dispose       = shared_ptr_dispose;
        f->callbacks.dispose_udata = d;
        return f;
}

void *rxtx::video_sender_loop() {
        set_thread_name(__func__);

        while (true) {
                check_sender_messages();

                shared_ptr<video_frame> tx_frame =
                    compress_pop(m_video_compression);
                if (!tx_frame) {
                        break;
                }

                m_video_desc = video_desc_from_frame(tx_frame.get());
                export_video(m_video_exporter, tx_frame.get());

                if (m_impl_funcs->send_video_frame != nullptr) {
                        m_impl_funcs->send_video_frame(m_impl_state,
                                                       std::move(tx_frame));
                } else {
                        m_impl_funcs->send_video_frame_c(
                            m_impl_state,
                            shared_vf_to_plain(std::move(tx_frame)));
                }
                m_frames_sent += 1;
        }

        check_sender_messages();
        return NULL;
}

/**
 * @returns the rxtx state (not nullptr)
 * @throws 1 help shown
 * @throws -1 on error
 */
rxtx *
rxtx::create(string const              &proto,
                   struct rxtx_params *params) noexcept
{
        const struct rxtx_medium_params *params_audio =
            &params->medium[TX_MEDIA_AUDIO];
        const struct rxtx_medium_params *params_video =
            &params->medium[TX_MEDIA_VIDEO];

        const auto *vri = static_cast<const rxtx_info *>(load_library(
            proto.c_str(), LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION));
        if (vri == nullptr) {
                if (proto != "help") {
                        error_msg("Requested RX/TX %s cannot be created "
                                  "(missing library?)\n", proto.c_str());
                        return nullptr;
                }
                return (rxtx *) INIT_NOERR;
        }

        // check if RX/TX protocol supports needed A/V send/recv
        if ((params_audio->rxtx_mode & MODE_RECEIVER) &&
            vri->recv_audio_frame == nullptr) {
                MSG(ERROR,
                    "Selected RX/TX module doesn't support audio receiving.\n");
                return nullptr;
        }
        if ((params_audio->rxtx_mode & MODE_SENDER) &&
            vri->send_audio_frame == nullptr) {
                MSG(ERROR,
                    "Selected RX/TX module doesn't support audio sending.\n");
                return nullptr;
        }
        if ((params_video->rxtx_mode & MODE_RECEIVER) &&
            vri->video_recv_routine == nullptr) {
                MSG(ERROR, "Selected RX/TX module doesn't support video receiving.\n");
                return nullptr;
        }
        if ((params_video->rxtx_mode & MODE_SENDER) &&
            (vri->send_video_frame == nullptr &&
             vri->send_video_frame_c == nullptr)) {
                MSG(ERROR, "Selected RX/TX module doesn't support video sending.\n");
                return nullptr;
        }

        rxtx *ret = new rxtx(params);

        params->sender_mod  = &ret->m_sender_mod;
        params->receiver_mod  = &ret->m_receiver_mod;
        ret->m_impl_state = vri->create(params);
        if (ret->m_impl_state == nullptr || ret->m_impl_state == INIT_NOERR) {
                void *retval = ret->m_impl_state;
                delete ret;
                return (rxtx *) retval;
        }
        ret->m_impl_funcs = vri;

        if (strlen(params->video_compression) == 0) {
                // not set by user or RXTX mod
                strcpy_ch(params->video_compression, DEFAULT_VIDEO_COMPRESSION);
        }
        const char *video_compression = params->video_compression;
        // "tentatively" is meant to be just print
        /// @todo it may be possible also to store the compression and dispatch
        /// the opportunistic init here, not by a message...
        if (strstr(params->video_compression, "tentatively")) {
                video_compression = DEFAULT_VIDEO_COMPRESSION;
        }
        int rc = compress_init(&ret->m_sender_mod, video_compression,
                               &ret->m_video_compression);
        if (rc != 0) {
                delete ret;
                if (rc < 0) {
                        error_msg("Error initializing compression %s.\n",
                                  params->video_compression);
                        return nullptr;
                }
                return (rxtx *) INIT_NOERR;
        }

        for (int i = 0; i < NUM_TX_MEDIA; ++i) {
                ret->rxtx_mode[i] = params->medium[i].rxtx_mode;
        }

        if ((params_video->rxtx_mode & MODE_RECEIVER) != 0U) {
                int rc = pthread_create(&ret->m_video_receiver_thread_id, nullptr,
                                        ret->m_impl_funcs->video_recv_routine,
                                        ret->m_impl_state);
                assert(rc == 0);
        }

        if (params_video->rxtx_mode & MODE_SENDER) {
                int rc =
                    pthread_create(&ret->m_video_sender_thread_id, nullptr,
                                   rxtx::video_sender_thread, (void *) ret);
                assert(rc == 0);
        }

        return ret;
}

void
rxtx::list(bool full) noexcept
{
        printf("Available TX protocols:\n");
        list_modules(LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION, full);
}

/**
 * @param[in]  params requested parameters
 * @param[out] params actual parameters (some default vals may filled with
 *                    adjusted values)
 * @returns -1 error; 0 OK; 1 help shown (as usual)
 */
int
rxtx_init(const char *proto_name, struct rxtx_params *params,
           struct rxtx **state)
{
        rxtx *ret = rxtx::create(proto_name, params);
        if (ret == nullptr) {
                return -1;
        }
        if (ret == INIT_NOERR) {
                return 1;
        }
        *state = ret;
        return 0;
}

void
rxtx_list_protocols(bool full)
{
        rxtx::list(full);
}

const char *
rxtx_get_proto_long_name(const char *short_name)
{
        return rxtx::get_long_name(short_name);
}

void
rxtx_join(struct rxtx *state)
{
        state->join();
}

void
rxtx_destroy(struct rxtx *state)
{
        delete state;
}

/// @sa vrxtx_send for shared_ptr variant
void
rxtx_send_video(struct rxtx *state, struct video_frame *tx_frame)
{
        state->send_vframe(
            shared_ptr<video_frame>(tx_frame, tx_frame->callbacks.dispose));
}

/// @sa rxtx_send_vide for plain pointer variant
void
vrxtx_send(struct rxtx *state, std::shared_ptr<struct video_frame> f)
{
        state->send_vframe(std::move(f));
}

bool rxtx_ctl_property(struct rxtx *state, enum rxtx_property p,
                              void *val, size_t *len) {
        assert(state->magic == MAGIC);
        if (state->m_impl_funcs->ctl_property == nullptr) {
                return false;
        }
        return state->m_impl_funcs->ctl_property(state->m_impl_state, p, val,
                                                 len);
}

void
rxtx_send_audio(struct rxtx *s, const struct audio_frame2 *frame)
{
        s->m_impl_funcs->send_audio_frame(s->m_impl_state, frame);
}

struct rx_audio_frames *
rxtx_recv_audio_frame(struct rxtx *s)
{
        return s->m_impl_funcs->recv_audio_frame(s->m_impl_state);
}

void
rxtx_free_audio_frames(struct rx_audio_frames *frames)
{
        while (frames != nullptr) {
                free(frames->source);
                delete frames->frame;
                struct rx_audio_frames *tmp = frames;
                frames = frames->next;
                free(tmp);
        }
}

const char *
get_tx_name(enum tx_media_type t)
{
        switch (t) {
        case TX_MEDIA_AUDIO:
                return "audio";
        case TX_MEDIA_VIDEO:
                return "video";
        case NUM_TX_MEDIA:
                MSG(ERROR, "NUM_TX_MEDIA passed to %s!\n", __func__);
                abort();
        }
        MSG(ERROR, "wrong medium type %d passed to %s!\n", t, __func__);
        abort();
}

enum rxtx_mode
rxtx_get_mode(struct rxtx *s, enum tx_media_type t)
{
        return s->rxtx_mode[t];
}



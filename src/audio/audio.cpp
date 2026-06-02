/*
 * FILE:    audio/audio.cpp
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Martin Pulec     <martin.pulec@cesnet.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#include <cassert>
#include <climits>
#include <cmath>      // for log10
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "../export.h" // not audio/export.h
#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/audio_filter.h"
#include "audio/audio_playback.h"
#include "audio/capture/sdi.h"
#include "audio/codec.h"
#include "audio/echo.h"
#include "audio/filter_chain.hpp"
#include "audio/playback/sdi.h"
#include "audio/resampler.hpp"
#include "audio/utils.h"
#include "config.h"                     // for HAVE_SPEEXDSP
#include "debug.h"
#include "host.h"
#include "messaging.h"                  // for check_message...
#include "module.h"
#include "rtp/audio_decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "video_display.h"              // for DISPLAY_FLAG_AUDIO_*
#include "utils/macros.h"               // for STR_LEN, snprintf_ch
#include "utils/misc.h"                 // for get_stat_color
#include "utils/pthread.h"              // for PTHREAD_NULL
#include "utils/string_view_utils.hpp"
#include "utils/thread.h"
#include "utils/worker.h"
#include "video_rxtx.h"                 // for video_rxtx
#include "video_rxtx/rtp_common.h"

using std::ostringstream;
using std::string;
using std::unique_ptr;
using namespace std::string_literals;

const enum module_class path_audio_send_module[] = { MODULE_CLASS_AUDIO,
                                                     MODULE_CLASS_SENDER,
                                                     MODULE_CLASS_NONE };
#define MOD_NAME "[audio] "

struct state_audio {
        explicit state_audio(struct module *parent) :
                mod(MODULE_CLASS_AUDIO, parent, this),
                audio_receiver_module(MODULE_CLASS_RECEIVER, mod.get(), this),
                audio_sender_module(MODULE_CLASS_SENDER,mod.get(), this),
                filter_chain(audio_sender_module.get())
        {
        }

        bool should_exit = false;

        module_raii mod;
        struct state_audio_capture *audio_capture_device = nullptr;
        struct state_audio_playback *audio_playback_device = nullptr;

        module_raii audio_receiver_module;
        module_raii audio_sender_module;

        Filter_chain filter_chain;

        struct audio_codec_state *audio_encoder = nullptr;

        // for statistics
        time_ns_t t0 = get_time_in_ns();
        audio_frame2 captured;

        pthread_t audio_sender_thread_id   = PTHREAD_NULL;
        pthread_t audio_receiver_thread_id = PTHREAD_NULL;

        echo_cancellation_t *echo_state = nullptr;
        struct exporter *exporter = nullptr;
        int resample_to = 0;

        int audio_tx_mode = 0;

        double volume = 1.0; // receiver volume scale
        bool muted_receiver = false;
        bool muted_sender = false;

        int recv_buf_size = DEFAULT_AUDIO_RECV_BUF_SIZE;

        struct video_rxtx *rxtx = nullptr;
        struct state_audio_postprocess *pp = nullptr;
};

/** 
 * Copies one input channel into n output (interlaced).
 * 
 * Input and output data may overlap. 
 */
typedef void (*audio_device_help_t)(void);

static void *audio_sender_thread(void *arg);
static void *audio_receiver_thread(void *arg);
static struct response *audio_receiver_process_message(struct state_audio *s, struct msg_receiver *msg);
static struct response *
audio_sender_process_message(struct state_audio      *s,
                             struct msg_audio_sender *msg);

static void should_exit_audio(void *state) {
        auto *s = (struct state_audio *) state;
        s->should_exit = true;
}

static int
audio_init_real(struct state_audio *s, const struct audio_options *opt,
                const struct common_opts *common)
{
        assert(opt->send_cfg != NULL);
        assert(opt->recv_cfg != NULL);

        s->rxtx = opt->vrxtx;
        s->resample_to = parse_audio_codec_params(opt->codec_cfg).sample_rate;

        s->exporter = common->exporter;

        if (opt->echo_cancellation) {
#ifdef HAVE_SPEEXDSP
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Echo cancellation is currently experimental "
                                "and may not work as expected.\n");
                s->echo_state = echo_cancellation_init();
#else
                fprintf(stderr, "Speex not compiled in. Could not enable echo cancellation.\n");
                return -1;
#endif /* HAVE_SPEEXDSP */
        }

        if(opt->filter_cfg){
                std::string_view cfg_sv = opt->filter_cfg;;
                std::string_view item;
                while(item = tokenize(cfg_sv, '#'), !item.empty()) {
                        if(!s->filter_chain.emplace_new(item)) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to init audio filter\n");
                                return -1;
                        }
                }
        }

        if (strcmp(opt->send_cfg, "none") != 0) {
                const char *cfg = "";
                char *device = strdup(opt->send_cfg);
		if(strchr(device, ':')) {
			char *delim = strchr(device, ':');
			*delim = '\0';
			cfg = delim + 1;
		}

                int ret = audio_capture_init(s->audio_sender_module.get(), device, cfg, &s->audio_capture_device);
                free(device);
                if (ret != 0) {
                        return ret;
                }

                s->audio_tx_mode |= MODE_SENDER;
        } else {
                s->audio_capture_device = audio_capture_init_null_device();
        }
        
        if (strcmp(opt->recv_cfg, "none") != 0) {
                s->audio_tx_mode |= MODE_RECEIVER;
        }
        const char *playback_cfg = "";
        char       *playback_dev = strdup(opt->recv_cfg);
        if (strchr(playback_dev, ':') != nullptr) {
                char *delim  = strchr(playback_dev, ':');
                *delim       = '\0';
                playback_cfg = delim + 1;
        }
        struct audio_playback_opts opts{};
        snprintf_ch(opts.cfg, "%s", playback_cfg);
        opts.parent = s->audio_receiver_module.get();
        int ret =
            audio_playback_init(playback_dev, &opts, &s->audio_playback_device);
        bool playback_is_sdi = (audio_get_display_flags(playback_dev) &
                                DISPLAY_FLAG_AUDIO_ANY) != 0U;
        free(playback_dev);
        if (ret != 0) {
                return ret;
        }
        if (playback_is_sdi) {
                auto *sdi_playback = (struct state_sdi_playback *)
                    audio_playback_get_state_pointer(s->audio_playback_device);
                sdi_register_display(sdi_playback, opt->display);
        }

        if ((s->audio_tx_mode & MODE_SENDER) != 0U || "help"s == opt->codec_cfg) {
                if ((s->audio_encoder = audio_codec_init_cfg(opt->codec_cfg, AUDIO_CODER)) == nullptr) {
                        return -1;
                }
        }

        struct rtp_rxtx_common *rtp_common_state = nullptr;
        size_t                  len =
            sizeof rtp_common_state; // NOLINT(bugprone-sizeof-expression)
        bool rtp_state_rc = rxtx_ctl_property(s->rxtx, GET_RTP_COMMON_STATE,
                                              (void *) &rtp_common_state, &len);
        if (rtp_state_rc) {
                struct rtp_rxtx_medium *audio =
                    &rtp_common_state->medium[TX_MEDIA_AUDIO];
                struct rtp *audio_network_device = audio->network_device;
                if ((s->audio_tx_mode & MODE_RECEIVER) != 0U) {
                        audio_playback_ctl(s->audio_playback_device,
                                           AUDIO_PLAYBACK_PUT_NETWORK_DEVICE,
                                           (void *) &audio_network_device,
                                           &len);
                }
        }

        int pp_ret =
            audio_postprocess_init(opt->channel_map, opt->scale,
                                   s->audio_receiver_module.get(), &s->pp);
        if (pp_ret != 0) {
                return pp_ret;
        }

        return 0;
}

/**
 * take care that addrs can also be comma-separated list of addresses !
 * @retval  0 state successfully initialized
 * @retval <0 error occurred
 * @retval >0 success but no state was created (eg. help printed)
 */
int
audio_init(struct state_audio **state, const struct audio_options *opt,
                const struct common_opts *common) {
        auto *s = new state_audio(common->parent);
        register_should_exit_callback(common->parent, should_exit_audio, s);

        int rc = audio_init_real(s, opt, common);
        if (rc == 0) {
                *state = s;
                return 0;
        }

        audio_done(s);

        return rc;
}

void audio_start(struct state_audio *s) {
        if (s->audio_tx_mode & MODE_SENDER) {
                if (pthread_create
                    (&s->audio_sender_thread_id, NULL, audio_sender_thread, (void *)s) != 0) {
                        log_msg(LOG_LEVEL_FATAL, "Error creating audio thread. Quitting\n");
                        exit_uv(EXIT_FAIL_AUDIO);
		}
        }

        if (s->audio_tx_mode & MODE_RECEIVER) {
                if (pthread_create
                    (&s->audio_receiver_thread_id, NULL, audio_receiver_thread, (void *)s) != 0) {
                        log_msg(LOG_LEVEL_FATAL, "Error creating audio thread. Quitting\n");
                        exit_uv(EXIT_FAIL_AUDIO);
		}
        }
}

/**
 * called to ensure that connected modules (rxtx) will not be called and thus
 * can be safely deleted
 */
void audio_join(struct state_audio *s) {
        if (!s) {
                return;
        }
        if (!pthread_equal(s->audio_receiver_thread_id, PTHREAD_NULL)) {
                pthread_join(s->audio_receiver_thread_id, NULL);
        }
        if (!pthread_equal(s->audio_sender_thread_id, PTHREAD_NULL)) {
                pthread_join(s->audio_sender_thread_id, NULL);
        }
        // Aplay/mixer uses thread that uses rxtx/ultragrid_rtp so destroy it to
        // join that thread as well.
        audio_playback_done(s->audio_playback_device);
        // avoids 2nd done in audio_done (it should be there as well - if
        // something happens and audio_join() is not run before audio_done())
        s->audio_playback_device = nullptr;
}

void audio_done(struct state_audio *s)
{
        if (!s) {
                return;
        }
        audio_playback_done(s->audio_playback_device);
        audio_capture_done(s->audio_capture_device);
        // process remaining messages
        struct message *msg;
        while ((msg = check_message(s->audio_receiver_module.get()))) {
                struct response *r = audio_receiver_process_message(s, (struct msg_receiver *) msg);
                free_message(msg, r);
        }
        while ((msg = check_message(s->audio_sender_module.get()))) {
                struct response *r = audio_sender_process_message(
                    s, (struct msg_audio_sender *) msg);
                free_message(msg, r);
        }

        audio_codec_done(s->audio_encoder);
        audio_postprocess_done(s->pp);

        unregister_should_exit_callback(get_root_module(s->mod.get()),
                                        should_exit_audio, s);

        delete s;
}

#define GET_MUTED(muted_flag, act, color) \
        if (muted_flag) { \
                act   = "muted"; \
                color = TERM_FG_RED; \
        } else { \
                act   = "unmuted"; \
                color = TERM_FG_GREEN; \
        }

static struct response * audio_receiver_process_message(struct state_audio *s, struct msg_receiver *msg)
{
        switch (msg->type) {
        case RECEIVER_MSG_GET_AUDIO_STATUS: {
                double ret             = s->muted_receiver ? 0.0 : s->volume;
                char   volume_str[128] = "";
                snprintf(volume_str, sizeof volume_str, "%lf,%d", ret,
                         (int) s->muted_receiver);
                return new_response(RESPONSE_OK, volume_str);
                break;
        }
        case RECEIVER_MSG_INCREASE_VOLUME:
        case RECEIVER_MSG_DECREASE_VOLUME:
        case RECEIVER_MSG_MUTE:
        case RECEIVER_MSG_UNMUTE:
        case RECEIVER_MSG_MUTE_TOGGLE: {
                if (msg->type == RECEIVER_MSG_INCREASE_VOLUME) {
                        s->volume *= 1.1;
                } else if (msg->type == RECEIVER_MSG_DECREASE_VOLUME) {
                        s->volume /= 1.1;
                } else {
                        s->muted_receiver =
                            msg->type == RECEIVER_MSG_MUTE_TOGGLE
                                ? !s->muted_receiver
                                : msg->type == RECEIVER_MSG_MUTE;
                }
                double new_volume = s->muted_receiver ? 0.0 : s->volume;
                double db         = 20.0 * log10(new_volume);
                if (msg->type == RECEIVER_MSG_MUTE_TOGGLE) {
                        const char *act = nullptr;
                        const char *color = nullptr;
                        GET_MUTED(s->muted_receiver, act, color)
                        color_printf("%sAudio " TBOLD("receiver")
                                     " %s. " TERM_FG_RESET "\n",
                                     color, act);
                } else {
                        log_msg(
                            LOG_LEVEL_INFO,
                            TGREEN("Playback volume: %.2f%% (%+.2f dB)")
                            "\n",
                            new_volume * 100.0, db);
                }
                break;
        }
        case RECEIVER_MSG_VIDEO_PROP_CHANGED:
                abort();
        }

        return new_response(RESPONSE_OK, nullptr);
}

static void audio_update_recv_buf(struct state_audio *s, size_t curr_frame_len)
{
        int new_size = (int) curr_frame_len * 2;

        if (new_size <= s->recv_buf_size) {
                return;
        }
        s->recv_buf_size = new_size;

        size_t len = sizeof new_size;
        bool ret = rxtx_ctl_property(s->rxtx, SET_RTP_AUD_FRM_SZ, (void *) &curr_frame_len,
                          &len);
        if (ret) {
                MSG(VERBOSE, "Recv buffer adjusted to %d\n", new_size);
        } else {
                MSG(VERBOSE, "Cannot adjust recv buffer to %d. Not a RTP RXTX?\n",
                    new_size);
        }
}

static void
change_volume(bool mute, double scale, audio_frame *buffer)
{
        if (mute) {
                memset(buffer->data, 0, buffer->data_len);
                return;
        }
        rescale_audio_buffer(buffer->data, buffer->data_len, buffer->bps,
                             (float) scale);
}

static void
flush_rtp_samples(struct video_rxtx *rxtx)
{
        struct rtp_rxtx_common *rtp_common_state = nullptr;
        size_t                  len =
            sizeof rtp_common_state; // NOLINT(bugprone-sizeof-expression)
        bool ctl_rc = rxtx_ctl_property(rxtx, GET_RTP_COMMON_STATE,
                                        (void *) &rtp_common_state, &len);
        if (!ctl_rc) {
                return;
        }
        struct rtp_rxtx_medium *audio =
            &rtp_common_state->medium[TX_MEDIA_AUDIO];
        rtp_flush_recv_buf(audio->network_device);
}

static void *audio_receiver_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_audio *s = (struct state_audio *) arg;
        // rtp variables
        struct audio_desc device_desc{};
        struct audio_desc network_desc{};
        bool  playback_supports_multiple_streams = false;
        long long int received_bytes_cum         = 0;
        long long int expected_bytes_cum         = 0;

        size_t len = sizeof playback_supports_multiple_streams;
        audio_playback_ctl(s->audio_playback_device, AUDIO_PLAYBACK_CTL_MULTIPLE_STREAMS,
                        &playback_supports_multiple_streams, &len);
        len = sizeof playback_supports_multiple_streams;
        rxtx_ctl_property(s->rxtx, SET_ULTRAGRID_RTP_MUTLI_OUT,
                          &playback_supports_multiple_streams, &len);

        printf("Audio receiving started.\n");
        struct audio_frame f{};
#define CONTINUE                                                               \
        rxtx_free_audio_frames(frames);                                        \
        continue
        while (!s->should_exit) {
                struct message *msg;
                while((msg= check_message(s->audio_receiver_module.get()))) {
                        struct response *r = audio_receiver_process_message(s, (struct msg_receiver *) msg);
                        free_message(msg, r);
                }

                struct rx_audio_frames *frames = nullptr;

                frames = rxtx_recv_audio_frame(s->rxtx);

                if (frames == nullptr) {
                        continue;
                }

                struct audio_desc curr_desc = frames->frame->get_desc();

                if (!audio_desc_eq(network_desc, curr_desc)) {
                        size_t len = sizeof device_desc;
                        // the eventual user override by --audio-channel-map
                        int    output_channels = audio_postprocess_reconfigure(
                            s->pp, curr_desc.ch_count);
                        device_desc =
                            audio_desc{ .bps         = curr_desc.bps,
                                        .sample_rate = curr_desc.sample_rate,
                                        .ch_count    = output_channels,
                                        .codec       = AC_PCM };
                        if (const char *fmt = get_commandline_param("audio-dec-format")) {
                                if (int ret = parse_audio_format(fmt, &device_desc)) {
                                        exit_uv(ret > 0 ? 0 : 1);
                                        CONTINUE;
                                }
                        }
                        if (!audio_playback_ctl(s->audio_playback_device,
                                                AUDIO_PLAYBACK_CTL_QUERY_FORMAT,
                                                &device_desc, &len)) {
                                MSG(ERROR, "Unable to query audio desc!\n");
                                CONTINUE;
                        }
                        if (!audio_playback_reconfigure(
                                s->audio_playback_device,
                                device_desc.bps * CHAR_BIT,
                                device_desc.ch_count,
                                device_desc.sample_rate)) {
                                MSG(ERROR,
                                    "Audio reconfiguration failed (%s)!\n",
                                    ((string) curr_desc).c_str());
                                CONTINUE;
                        }
                        f.bps            = device_desc.bps;
                        f.ch_count       = device_desc.ch_count;
                        f.sample_rate    = device_desc.sample_rate;
                        network_desc     = curr_desc;
                        MSG(NOTICE, "Audio reconfiguration succeeded (%s).\n",
                            ((string) curr_desc).c_str());

                        flush_rtp_samples(s->rxtx);
                }


                struct rx_audio_frames *cur_frame = frames;
                while (cur_frame != nullptr) {
                        audio_update_recv_buf(s, cur_frame->frame->get_data_len());

                        expected_bytes_cum += cur_frame->expected_bytes;
                        received_bytes_cum += cur_frame->received_bytes;

                        if (!decode_audio_frame_postprocess(
                                s->pp, cur_frame->frame, &f, &expected_bytes_cum,
                                &received_bytes_cum)) {
                                CONTINUE;
                        }

                        if (s->muted_receiver || s->volume != 1) {
                                change_volume(s->muted_receiver, s->volume, &f);
                        }

                        if (s->echo_state) {
#ifdef HAVE_SPEEXDSP
                                echo_play(s->echo_state, &f);
#endif
                        }
                        f.network_source = cur_frame->source;
                        audio_playback_put_frame(s->audio_playback_device, &f);

                        cur_frame = cur_frame->next;
                }
                rxtx_free_audio_frames(frames);
        }

        free(f.data);

        return NULL;
}

static struct response *
audio_sender_process_message(struct state_audio      *s,
                             struct msg_audio_sender *msg)
{
        switch (msg->type) {
        case AUDIO_SENDER_MSG_GET_STATUS: {
                char status[128] = "";
                snprintf(status, sizeof status, "%d", (int) s->muted_sender);
                return new_response(RESPONSE_OK, status);
        }
        case AUDIO_SENDER_MSG_MUTE:
        case AUDIO_SENDER_MSG_UNMUTE:
        case AUDIO_SENDER_MSG_MUTE_TOGGLE:
                s->muted_sender = msg->type == AUDIO_SENDER_MSG_MUTE_TOGGLE
                                      ? !s->muted_sender
                                      : msg->type == AUDIO_SENDER_MSG_MUTE;
                const char *act = nullptr;
                const char *color = nullptr;
                GET_MUTED(s->muted_sender, act, color)
                color_printf("%sAudio " TBOLD("sender")
                             " %s. " TERM_FG_RESET "\n",
                             color, act);
                return new_response(RESPONSE_OK, nullptr);
        }
        MSG(ERROR, "Wrong audio sender type %d!\n", msg->type);
        return new_response(RESPONSE_INT_SERV_ERR, nullptr);
}

struct asend_stats_processing_data {
        audio_frame2 frame;
        double seconds;
        bool muted_sender;
};

static void *asend_compute_and_print_stats(void *arg) {
        auto *d = (struct asend_stats_processing_data*) arg;

        const double exp_samples = d->frame.get_sample_rate() * d->seconds;
        const char  *dec_cnt_warn_col =
            get_stat_color(d->frame.get_sample_count() / exp_samples);

        log_msg(LOG_LEVEL_INFO,
                "[Audio sender] Sent %s%d samples" TERM_FG_RESET
                " in last %.2f seconds.\n",
                dec_cnt_warn_col, d->frame.get_sample_count(), d->seconds);

        char volume[STR_LEN];
        char *vol_start = volume;
        for (int i = 0; i < d->frame.get_channel_count(); ++i) {
                double rms = 0.0;
                double peak = 0.0;
                rms = calculate_rms(&d->frame, i, &peak);
                format_audio_channel_volume(i, rms, peak, T256_FG_SYM(T_AMBER),
                                            &vol_start, volume + sizeof volume);
        }

        log_msg(
            LOG_LEVEL_INFO,
            TBOLD(T256_FG(T_AMBER,
                          "[audio send] ")) "Volume: %s dBFS RMS/peak%s\n",
            volume, d->muted_sender ? TBOLD(TRED(" (muted sender)")) : "");

        delete d;

        return NULL;
}

static void process_statistics(struct state_audio *s, const audio_frame *buffer)
{
        if (!s->captured.has_same_prop_as(*buffer)) {
                s->captured.init(buffer->ch_count, AC_PCM,
                                buffer->bps, buffer->sample_rate);
        }
        s->captured.append(*buffer);

        time_ns_t t = get_time_in_ns();
        if (t - s->t0 > 5 * NS_IN_SEC) {
                double seconds = (double)(t - s->t0) / NS_IN_SEC;
                auto d = new asend_stats_processing_data;
                std::swap(d->frame, s->captured);
                d->seconds = seconds;
                d->muted_sender = s->muted_sender;

                task_run_async_detached(asend_compute_and_print_stats, d);
                s->t0 = t;
        }
}

static int find_codec_sample_rate(int sample_rate, const int *supported) {
        if (!supported) {
                return 0;
        }

        int rate_hi = 0, // nearest high
           rate_lo = 0; // nearest low

        const int *tmp = supported;
        while (*tmp != 0) {
                if (*tmp == sample_rate) {
                        return sample_rate;
                }
                if (*tmp > sample_rate && (rate_hi == 0 || *tmp < rate_hi)) {
                        rate_hi = *tmp;
                }

                if (*tmp < sample_rate && *tmp > rate_lo) {
                        rate_lo = *tmp;
                }
                tmp++;
        }

        return rate_hi > 0 ? rate_hi : rate_lo;
}

static void *audio_sender_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_audio *s = (struct state_audio *) arg;
        unique_ptr<audio_frame2_resampler> resampler_state;
        try {
                resampler_state = unique_ptr<audio_frame2_resampler>(new audio_frame2_resampler);
        } catch (ug_runtime_error &e) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", e.what());
                exit_uv(1);
                return NULL;
        }

        printf("Audio sending started.\n");

        while (!s->should_exit) {
                struct message *msg;
                while((msg = check_message(s->audio_sender_module.get()))) {
                        struct response *r = audio_sender_process_message(
                            s, (struct msg_audio_sender *) msg);
                        free_message(msg, r);
                }

                const struct audio_frame *buffer =
                    audio_capture_read(s->audio_capture_device);
                if(buffer) {
                        if(s->echo_state) {
#ifdef HAVE_SPEEXDSP
                                buffer = echo_cancel(s->echo_state, buffer);
                                if(!buffer)
                                        continue;
#endif
                        }
                        process_statistics(s, buffer);
                        if (s->muted_sender) {
                                memset(buffer->data, 0, buffer->data_len);
                        }
                        export_audio(s->exporter, buffer);

                        s->filter_chain.filter(&buffer);

                        if(!buffer)
                                continue;

                        audio_frame2 bf_n(buffer);
                        if (audio_capture_channels != 0 &&
                            (int) audio_capture_channels !=
                                bf_n.get_channel_count()) {
                                bf_n.change_ch_count((int) audio_capture_channels);
                        }

                        // RESAMPLE
                        int resample_to = s->resample_to;
                        if (resample_to == 0) {
                                const int *supp_sample_rates = audio_codec_get_supported_samplerates(s->audio_encoder);
                                resample_to = find_codec_sample_rate(bf_n.get_sample_rate(),
                                                supp_sample_rates);
                        }
                        if (resample_to != 0 && bf_n.get_sample_rate() != resample_to) {
                                if (bf_n.get_bps() != 2) {
                                        bf_n.change_bps(2);
                                }

                                bf_n.resample(*resampler_state, resample_to);
                        }
                        // SEND
                        audio_frame2 *uncompressed = &bf_n;
                        while (audio_frame2 to_send = audio_codec_compress(
                                   s->audio_encoder, uncompressed)) {
                                rxtx_send_audio(s->rxtx, &to_send);
                                uncompressed = NULL;
                        }
                }
        }

        return NULL;
}

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame) {
        void *sdi_capture;
        if (!s->audio_capture_device)
                return;
        if(!audio_capture_get_vidcap_flags(audio_capture_get_driver_name(s->audio_capture_device)))
                return;
        
        sdi_capture = audio_capture_get_state_pointer(s->audio_capture_device);
        sdi_capture_new_incoming_frame(sdi_capture, frame);
}

unsigned int
audio_get_display_flags(const char *playback_dev)
{
        if (strcasecmp(playback_dev, "embedded") == 0) {
                return DISPLAY_FLAG_AUDIO_EMBEDDED;
        }
        if (strcasecmp(playback_dev, "AESEBU") == 0) {
                return DISPLAY_FLAG_AUDIO_AESEBU;
        }
        if (strcasecmp(playback_dev, "analog") == 0) {
                return DISPLAY_FLAG_AUDIO_ANALOG;
        }
        return 0U;
}

std::ostream& operator<<(std::ostream& os, const audio_desc& desc)
{
    os << desc.ch_count << " channel" << (desc.ch_count > 1 ? "s" : "") << ", " << desc.bps << " Bps, " << desc.sample_rate << " Hz, codec: " << get_name_to_audio_codec(desc.codec);
    return os;
}


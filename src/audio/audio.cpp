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
 * Copyright (c) 2005-2021 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cinttypes>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>

#include "audio/audio.h" 

#include "audio/codec.h"
#include "audio/echo.h" 
#include "audio/audio_capture.h" 
#include "audio/audio_playback.h" 
#include "audio/capture/sdi.h"
#include "audio/playback/sdi.h"
#include "audio/jack.h" 
#include "audio/resampler.hpp"
#include "audio/utils.h"
#include "audio/audio_filter.h"
#include "audio/filter_chain.hpp"
#include "debug.h"
#include "../export.h" // not audio/export.h
#include "host.h"
#include "module.h"
#include "rang.hpp"
#include "rtp/audio_decoders.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "tv.h"
#include "transmit.h"
#include "pdb.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/net.h"
#include "utils/thread.h"
#include "utils/worker.h"
#include "utils/string_view_utils.hpp"

using namespace std;
using rang::fg;
using rang::style;

enum audio_transport_device {
        NET_NATIVE,
        NET_JACK,
        NET_STANDARD
};

#define DEFAULT_AUDIO_RECV_BUF_SIZE (256 * 1024)
#define MOD_NAME "[audio] "

struct audio_network_parameters {
        char *addr = nullptr;
        int recv_port = 0;
        int send_port = 0;
        struct pdb *participants = 0;
        int force_ip_version = 0;
        char *mcast_if = nullptr;
        int ttl = -1;
};

struct state_audio {
        state_audio(struct module *parent, time_ns_t st) :
                mod(MODULE_CLASS_AUDIO, parent, this),
                audio_receiver_module(MODULE_CLASS_RECEIVER, mod.get(), this),
                audio_sender_module(MODULE_CLASS_SENDER,mod.get(), this),
                filter_chain(audio_sender_module.get()),
                start_time(st),
                t0(st)
        {
        }
        ~state_audio() {
                delete fec_state;
        }

        module_raii mod;
        struct state_audio_capture *audio_capture_device = nullptr;
        struct state_audio_playback *audio_playback_device = nullptr;

        module_raii audio_receiver_module;
        module_raii audio_sender_module;

        Filter_chain filter_chain;

        struct audio_codec_state *audio_encoder = nullptr;
        
        struct audio_network_parameters audio_network_parameters{};
        struct rtp *audio_network_device = nullptr;
        struct pdb *audio_participants = nullptr;
        std::string proto_cfg; // audio network protocol options
        void *jack_connection = nullptr;
        enum audio_transport_device sender = NET_NATIVE;
        enum audio_transport_device receiver = NET_NATIVE;
        
        time_ns_t start_time;
        time_ns_t t0; // for statistics

        audio_frame2 captured;

        struct tx *tx_session = nullptr;
        fec       *fec_state = nullptr;
        
        pthread_t audio_sender_thread_id{},
                  audio_receiver_thread_id{};
        bool audio_sender_thread_started = false,
             audio_receiver_thread_started = false;

        char *audio_channel_map = nullptr;
        const char *audio_scale = nullptr;
        echo_cancellation_t *echo_state = nullptr;
        struct exporter *exporter = nullptr;
        int resample_to = 0;

        char *requested_encryption = nullptr;

        volatile bool paused = false; // for CoUniverse...

        int audio_tx_mode = 0;

        double volume = 1.0; // receiver volume scale
        bool muted_receiver = false;
        bool muted_sender = false;

        size_t recv_buf_size = DEFAULT_AUDIO_RECV_BUF_SIZE;
};

/** 
 * Copies one input channel into n output (interlaced).
 * 
 * Input and output data may overlap. 
 */
typedef void (*audio_device_help_t)(void);

static void *audio_sender_thread(void *arg);
static void *audio_receiver_thread(void *arg);
static struct rtp *initialize_audio_network(struct audio_network_parameters *params);
static struct response *audio_receiver_process_message(struct state_audio *s, struct msg_receiver *msg);
static struct response *audio_sender_process_message(struct state_audio *s, struct msg_sender *msg);

static void audio_channel_map_usage(void);
static void audio_scale_usage(void);

static void audio_channel_map_usage(void)
{
        printf("\t--audio-channel-map <mapping>   mapping of input audio channels\n");
        printf("\t                                to output audio channels comma-separated\n");
        printf("\t                                list of channel mapping\n");
        printf("\t                                eg. 0:0,1:0 - mixes first 2 channels\n");
        printf("\t                                    0:0    - play only first channel\n");
        printf("\t                                    0:0,:1 - sets second channel to\n");
        printf("\t                                             a silence, first one is\n");
        printf("\t                                             left as is\n");
        printf("\t                                    0:0,0:1 - splits mono into\n");
        printf("\t                                              2 channels\n");
}

static void audio_scale_usage(void)
{
        col() << "Usage:\n";
        col() << TBOLD(TRED("\t--audio-scale [<factor>|<method>]\n"));
        col() << "\t        Floating point number that tells a static scaling factor for all\n";
        col() << "\t        output channels. Scaling method can be one from these:\n";
        col() << TBOLD("\t          0.0-1.0") << " - factor to scale to (usually 0-1 but can be more)\n";
        col() << TBOLD("\t          mixauto") << " - automatically adjust volume if using channel\n";
        col() << "\t                    mixing/remapping (default)\n";
        col() << TBOLD("\t          auto") << " - automatically adjust volume\n";
        col() << TBOLD("\t          none") << " - no scaling will be performed\n";
}

/**
 * take care that addrs can also be comma-separated list of addresses !
 * @retval  0 state succesfully initialized
 * @retval <0 error occured
 * @retval >0 success but no state was created (eg. help printed)
 */
int audio_init(struct state_audio **ret, struct module *parent,
                struct audio_options *opt,
                const char *encryption,
                int force_ip_version, const char *mcast_iface,
                long long int bitrate, volatile int *audio_delay,
                time_ns_t start_time,
                int mtu, int ttl, struct exporter *exporter)
{
        char *tmp, *unused = NULL;
        UNUSED(unused);
        char *addr;
        int resample_to = get_audio_codec_sample_rate(opt->codec_cfg);
        int retval = -1;
        
        assert(opt->send_cfg != NULL);
        assert(opt->recv_cfg != NULL);

        if (opt->channel_map &&
                     strcmp("help", opt->channel_map) == 0) {
                audio_channel_map_usage();
                return 1;
        }

        if (opt->scale &&
                     strcmp("help", opt->scale) == 0) {
                audio_scale_usage();
                return 1;
        }
        
        struct state_audio *s = new state_audio(parent, start_time);

        s->audio_channel_map = opt->channel_map;
        s->audio_scale = opt->scale;

        s->audio_sender_thread_started = s->audio_receiver_thread_started = false;
        s->resample_to = resample_to;

        s->exporter = exporter;

        if (opt->echo_cancellation) {
#ifdef HAVE_SPEEXDSP
                s->echo_state = echo_cancellation_init();
                fprintf(stderr, "Echo cancellation is currently experimental "
                                "and may not work as expected.");
                goto error;
#else
                fprintf(stderr, "Speex not compiled in. Could not enable echo cancellation.\n");
                delete s;
                goto error;
#endif /* HAVE_SPEEXDSP */
        } else {
                s->echo_state = NULL;
        }

        if(opt->filter_cfg){
                std::string_view cfg_sv = opt->filter_cfg;;
                std::string_view item;
                while(item = tokenize(cfg_sv, '#'), !item.empty()) {
                        if(!s->filter_chain.emplace_new(item)) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to init audio filter\n");
                                goto error;
                        }
                }
        }

        if(encryption) {
                s->requested_encryption = strdup(encryption);
        }
        
        assert(opt->host != nullptr);
        tmp = strdup(opt->host);
        s->audio_participants = pdb_init(audio_delay);
        addr = strtok_r(tmp, ",", &unused);
        assert(addr != nullptr);

        s->audio_network_parameters.addr = strdup(addr);
        s->audio_network_parameters.recv_port = opt->recv_port;
        s->audio_network_parameters.send_port = opt->send_port;
        s->audio_network_parameters.participants = s->audio_participants;
        s->audio_network_parameters.force_ip_version = force_ip_version;
        s->audio_network_parameters.mcast_if = mcast_iface
                ? strdup(mcast_iface) : NULL;
        s->audio_network_parameters.ttl = ttl;
        free(tmp);

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
                        retval = ret;
                        goto error;
                }
                s->tx_session = tx_init(s->audio_sender_module.get(), mtu, TX_MEDIA_AUDIO, opt->fec_cfg, encryption, bitrate);
                if(!s->tx_session) {
                        fprintf(stderr, "Unable to initialize audio transmit.\n");
                        goto error;
                }

                s->audio_tx_mode |= MODE_SENDER;
        } else {
                s->audio_capture_device = audio_capture_init_null_device();
        }
        
        if (strcmp(opt->recv_cfg, "none") != 0) {
                const char *cfg = "";
                char *device = strdup(opt->recv_cfg);
		if(strchr(device, ':')) {
			char *delim = strchr(device, ':');
			*delim = '\0';
			cfg = delim + 1;
		}

                int ret = audio_playback_init(device, cfg, &s->audio_playback_device);
                free(device);
                if (ret != 0) {
                        retval = ret;
                        goto error;
                }
                size_t len = sizeof(struct rtp *);
                audio_playback_ctl(s->audio_playback_device, AUDIO_PLAYBACK_PUT_NETWORK_DEVICE,
                                        &s->audio_network_device, &len);

                s->audio_tx_mode |= MODE_RECEIVER;
        } else {
                s->audio_playback_device = audio_playback_init_null_device();
        }

        if (s->audio_tx_mode != 0) {
                if ((s->audio_network_device = initialize_audio_network(
                                                &s->audio_network_parameters))
                                == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to open audio network\n";
                        goto error;
                }
        }

        if ((s->audio_tx_mode & MODE_SENDER) != 0U || "help"s == opt->codec_cfg) {
                if ((s->audio_encoder = audio_codec_init_cfg(opt->codec_cfg, AUDIO_CODER)) == nullptr) {
                        goto error;
                }
        }

        s->proto_cfg = opt->proto_cfg;

        if (strcasecmp(opt->proto, "ultragrid_rtp") == 0) {
                s->sender = NET_NATIVE;
                s->receiver = NET_NATIVE;
        } else if (strcasecmp(opt->proto, "rtsp") == 0 || strcasecmp(opt->proto, "sdp") == 0) {
                s->receiver = NET_STANDARD;
                s->sender = NET_STANDARD;
        } else if (strcasecmp(opt->proto, "JACK") == 0) {
#ifndef HAVE_JACK_TRANS
                fprintf(stderr, "[Audio] JACK transport requested, "
                                "but JACK support isn't compiled.\n");
                goto error;
#else
                s->sender = NET_JACK;
                s->receiver = NET_JACK;
#endif
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unknow audio protocol: %s\n", opt->proto);
                goto error;
        }

        *ret = s;
        return 0;

error:
        if(s->tx_session)
                module_done(CAST_MODULE(s->tx_session));
        if(s->audio_participants) {
                pdb_destroy(&s->audio_participants);
        }

        audio_codec_done(s->audio_encoder);
        delete s;
        return retval;
}

void audio_start(struct state_audio *s) {
#ifdef HAVE_JACK_TRANS
        if (s->sender == NET_JACK || s->receiver == NET_JACK) {
                s->jack_connection = jack_start(s->proto_cfg.c_str());
                if (s->jack_connection) {
                        if (!is_jack_sender(s->jack_connection)) {
                                s->sender = NET_NATIVE;
                        }
                        if (!is_jack_receiver(s->jack_connection)) {
                                s->receiver = NET_NATIVE;
                        }
                } else {
                        log_msg(LOG_LEVEL_FATAL, "JACK transport initialization failed!\n");
                        exit_uv(EXIT_FAIL_AUDIO);
                }
        }
#endif

        if (s->audio_tx_mode & MODE_SENDER) {
                if (pthread_create
                    (&s->audio_sender_thread_id, NULL, audio_sender_thread, (void *)s) != 0) {
                        log_msg(LOG_LEVEL_FATAL, "Error creating audio thread. Quitting\n");
                        exit_uv(EXIT_FAIL_AUDIO);
                } else {
			s->audio_sender_thread_started = true;
		}
        }

        if (s->audio_tx_mode & MODE_RECEIVER) {
                if (pthread_create
                    (&s->audio_receiver_thread_id, NULL, audio_receiver_thread, (void *)s) != 0) {
                        log_msg(LOG_LEVEL_FATAL, "Error creating audio thread. Quitting\n");
                        exit_uv(EXIT_FAIL_AUDIO);
                } else {
			s->audio_receiver_thread_started = true;
		}
        }
}

void audio_join(struct state_audio *s) {
        if(s) {
                if(s->audio_receiver_thread_started)
                        pthread_join(s->audio_receiver_thread_id, NULL);
                if(s->audio_sender_thread_started)
                        pthread_join(s->audio_sender_thread_id, NULL);
        }
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
                struct response *r = audio_sender_process_message(s, (struct msg_sender *) msg);
                free_message(msg, r);
        }

        module_done(CAST_MODULE(s->tx_session));
        if(s->audio_network_device)
                rtp_done(s->audio_network_device);
        if(s->audio_participants) {
                pdb_destroy(&s->audio_participants);
        }
        free(s->requested_encryption);

        free(s->audio_network_parameters.addr);
        free(s->audio_network_parameters.mcast_if);

        audio_codec_done(s->audio_encoder);

        delete s;
}

static struct rtp *initialize_audio_network(struct audio_network_parameters *params)
{
        struct rtp *r;
        double rtcp_bw = 1024 * 512;    // FIXME:  something about 5% for rtcp is said in rfc

        r = rtp_init_if(params->addr, params->mcast_if, params->recv_port,
                        params->send_port, params->ttl, rtcp_bw,
                        FALSE, rtp_recv_callback,
                        (uint8_t *) params->participants,
                        params->force_ip_version, false);
        if (r != NULL) {
                pdb_add(params->participants, rtp_my_ssrc(r));
                rtp_set_option(r, RTP_OPT_WEAK_VALIDATION, TRUE);
                rtp_set_option(r, RTP_OPT_PROMISC, TRUE);
                rtp_set_option(r, RTP_OPT_RECORD_SOURCE, TRUE);
                rtp_set_sdes(r, rtp_my_ssrc(r), RTCP_SDES_TOOL,
                             PACKAGE_STRING, strlen(PACKAGE_VERSION));
                if (strcmp(params->addr, IN6_BLACKHOLE_STR) == 0) {
                        rtp_set_option(r, RTP_OPT_SEND_BACK, TRUE);
                }
                rtp_set_recv_buf(r, DEFAULT_AUDIO_RECV_BUF_SIZE);
        }

        return r;
}

struct audio_decoder {
        bool enabled;
        struct pbuf_audio_data pbuf_data;
};

static struct response * audio_receiver_process_message(struct state_audio *s, struct msg_receiver *msg)
{

        switch (msg->type) {
        case RECEIVER_MSG_CHANGE_RX_PORT:
                {
                        assert(s->audio_tx_mode == MODE_RECEIVER); // receiver only
                        struct rtp *old_audio_network_device = s->audio_network_device;
                        int old_rx_port = s->audio_network_parameters.recv_port;
                        s->audio_network_parameters.recv_port = msg->new_rx_port;
                        s->audio_network_device = initialize_audio_network(
                                        &s->audio_network_parameters);
                        if (!s->audio_network_device) {
                                s->audio_network_parameters.recv_port = old_rx_port;
                                s->audio_network_device = old_audio_network_device;
                                string err = string("Changing audio RX port to ") +
                                                to_string(msg->new_rx_port) + "  failed!";
                                LOG(LOG_LEVEL_ERROR) << err << "\n";
                                return new_response(RESPONSE_INT_SERV_ERR, err.c_str());
                        } else {
                                rtp_done(old_audio_network_device);
                                LOG(LOG_LEVEL_INFO) << "Successfully changed audio "
                                                "RX port to " << msg->new_rx_port << ".\n";
                        }
                        break;
                }
        case RECEIVER_MSG_GET_AUDIO_STATUS:
                {
                        double ret = s->muted_receiver ? 0.0 : s->volume;
                        char volume_str[128] = "";
                        snprintf(volume_str, sizeof volume_str, "%lf,%d", ret, s->muted_receiver);
                        return new_response(RESPONSE_OK, volume_str);
                        break;
                }
        case RECEIVER_MSG_INCREASE_VOLUME:
        case RECEIVER_MSG_DECREASE_VOLUME:
        case RECEIVER_MSG_MUTE:
                {
                        if (msg->type == RECEIVER_MSG_MUTE) {
                                s->muted_receiver = !s->muted_receiver;
                        } else if (msg->type == RECEIVER_MSG_INCREASE_VOLUME) {
                                s->volume *= 1.1;
                        } else {
                                s->volume /= 1.1;
                        }
                        double new_volume = s->muted_receiver ? 0.0 : s->volume;
                        double db = 20.0 * log10(new_volume);
                        if (msg->type == RECEIVER_MSG_MUTE) {
                                LOG(LOG_LEVEL_NOTICE) << "Audio receiver " << (s->muted_receiver ? "" : "un") << "muted.\n";
                        } else {
                                log_msg(LOG_LEVEL_INFO, "Playback volume: %.2f%% (%+.2f dB)\n", new_volume * 100.0, db);
                        }
                        struct pdb_e *cp;
                        pdb_iter_t it;
                        cp = pdb_iter_init(s->audio_participants, &it);
                        while (cp != NULL) {
                                struct audio_decoder *dec_state = (struct audio_decoder *) cp->decoder_state;
                                if (dec_state) {
                                        audio_decoder_set_volume(dec_state->pbuf_data.decoder, new_volume);
                                }
                                cp = pdb_iter_next(&it);
                        }
                        pdb_iter_done(&it);
                        break;
                }
        default:
                abort();
        }

        return new_response(RESPONSE_OK, NULL);
}

static struct audio_decoder *audio_decoder_state_create(struct state_audio *s) {
        auto *dec_state = (struct audio_decoder *) calloc(1, sizeof(struct audio_decoder));
        assert(dec_state != NULL);
        dec_state->enabled = true;
        dec_state->pbuf_data.decoder = (struct state_audio_decoder *) audio_decoder_init(s->audio_channel_map, s->audio_scale, s->requested_encryption, (audio_playback_ctl_t) audio_playback_ctl, s->audio_playback_device, s->audio_receiver_module.get());
        if (!dec_state->pbuf_data.decoder) {
                free(dec_state);
                return NULL;
        }
        audio_decoder_set_volume(dec_state->pbuf_data.decoder, s->muted_receiver ? 0.0 : s->volume);
        return dec_state;
}

static void audio_decoder_state_deleter(void *state)
{
        struct audio_decoder *s = (struct audio_decoder *) state;

        free(s->pbuf_data.buffer.data);
        audio_decoder_destroy(s->pbuf_data.decoder);

        free(s);
}

static void audio_update_recv_buf(struct state_audio *s, size_t curr_frame_len)
{
        size_t new_size = curr_frame_len * 2;

        if (new_size > s->recv_buf_size) {
                s->recv_buf_size = new_size;
                LOG(LOG_LEVEL_DEBUG) << "[Audio receiver] Recv buffer adjusted to " << new_size << "\n";
                rtp_set_recv_buf(s->audio_network_device, s->recv_buf_size);
        }
}

static void *audio_receiver_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_audio *s = (struct state_audio *) arg;
        // rtp variables
        struct pdb_e *cp;
        struct audio_desc device_desc{};
        bool playback_supports_multiple_streams;

        struct pbuf_audio_data *current_pbuf = NULL;

#ifdef HAVE_JACK_TRANS
        struct pbuf_audio_data jack_pbuf{};
        current_pbuf = &jack_pbuf;
#endif

        size_t len = sizeof playback_supports_multiple_streams;
        if (!audio_playback_ctl(s->audio_playback_device, AUDIO_PLAYBACK_CTL_MULTIPLE_STREAMS,
                        &playback_supports_multiple_streams, &len)) {
                playback_supports_multiple_streams = false;
        }

        printf("Audio receiving started.\n");
        while (!should_exit) {
                struct message *msg;
                while((msg= check_message(s->audio_receiver_module.get()))) {
                        struct response *r = audio_receiver_process_message(s, (struct msg_receiver *) msg);
                        free_message(msg, r);
                }

                bool decoded = false;

                if (s->receiver == NET_NATIVE || s->receiver == NET_STANDARD) {
                        time_ns_t curr_time = get_time_in_ns();
                        uint32_t ts = (curr_time - s->start_time) / 100'000 * 9; // at 90000 Hz
                        rtp_update(s->audio_network_device, curr_time);
                        rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);
                        struct timeval timeout;
                        timeout.tv_sec = 0;
                        // timeout.tv_usec = 999999 / 59.94; // audio goes almost always at the same rate
                                                             // as video frames
                        timeout.tv_usec = 1000; // this stuff really smells !!!
                        rtp_recv_r(s->audio_network_device, &timeout, ts);
                        pdb_iter_t it;
                        cp = pdb_iter_init(s->audio_participants, &it);
                
                        while (cp != NULL) {
                                if (cp->decoder_state == NULL &&
                                                !pbuf_is_empty(cp->playout_buffer)) { // the second check is need ed because we want to assign display to participant that really sends data
                                        // disable all previous sources
                                        if (!playback_supports_multiple_streams) {
                                                pdb_iter_t it;
                                                struct pdb_e *cp = pdb_iter_init(s->audio_participants, &it);
                                                while (cp != NULL) {
                                                        if(cp->decoder_state) {
                                                                ((struct audio_decoder *) cp->decoder_state)->enabled = false;
                                                        }
                                                        cp = pdb_iter_next(&it);
                                                }
                                                pdb_iter_done(&it);
                                        }

                                        if (get_commandline_param("low-latency-audio")) {
                                                pbuf_set_playout_delay(cp->playout_buffer, strcmp(get_commandline_param("low-latency-audio"), "ultra") == 0 ? 0.001 :0.005);
                                        }
                                        cp->decoder_state = audio_decoder_state_create(s);
                                        if (!cp->decoder_state) {
                                                exit_uv(1);
                                                break;
                                        }
                                        cp->decoder_state_deleter = audio_decoder_state_deleter;
                                }

                                struct audio_decoder *dec_state = (struct audio_decoder *) cp->decoder_state;
                                if (dec_state && dec_state->enabled) {
                                        dec_state->pbuf_data.buffer.data_len = 0;
                                        // We iterate in loop since there can be more than one frmae present in
                                        // the playout buffer and it would be discarded by following pbuf_remove()
                                        // call.
                                        while (pbuf_decode(cp->playout_buffer, curr_time, s->receiver == NET_NATIVE ? decode_audio_frame : decode_audio_frame_mulaw, &dec_state->pbuf_data)) {
                                                current_pbuf = &dec_state->pbuf_data;
                                                decoded = true;
                                        }
                                }

                                pbuf_remove(cp->playout_buffer, curr_time);
                                cp = pdb_iter_next(&it);

                                if (decoded && !playback_supports_multiple_streams)
                                        break;
                        }
                        pdb_iter_done(&it);
                }else { /* NET_JACK */
#ifdef HAVE_JACK_TRANS
                        decoded = jack_receive(s->jack_connection, &jack_pbuf);
#endif
                }

                if (decoded) {
                        bool failed = false;

                        struct audio_desc curr_desc;
                        curr_desc = audio_desc_from_frame(&current_pbuf->buffer);

                        if(!audio_desc_eq(device_desc, curr_desc)) {
                                int log_l;
                                string msg;
                                if (audio_playback_reconfigure(s->audio_playback_device, curr_desc.bps * 8,
                                                        curr_desc.ch_count,
                                                        curr_desc.sample_rate) != TRUE) {
                                        log_l = LOG_LEVEL_ERROR;
                                        msg = "Audio reconfiguration failed";
                                        failed = true;
                                }
                                else {
                                        log_l = LOG_LEVEL_INFO;
                                        msg = "Audio reconfiguration succeeded";
                                        device_desc = curr_desc;
                                        rtp_flush_recv_buf(s->audio_network_device);
                                }
                                LOG(log_l) << msg << " (" << curr_desc << ")" << (log_l < LOG_LEVEL_WARNING ? "!" : ".") << "\n";
                                rtp_flush_recv_buf(s->audio_network_device);
                        }

                        if (failed) {
                                continue;
                        }
                        audio_update_recv_buf(s, current_pbuf->frame_size);

                        if(s->echo_state) {
#ifdef HAVE_SPEEXDSP
                                echo_play(s->echo_state, &current_pbuf->buffer);
#endif
                        }

                        if (!playback_supports_multiple_streams) {
                                audio_playback_put_frame(s->audio_playback_device, &current_pbuf->buffer);
                        } else {
                                pdb_iter_t it;
                                cp = pdb_iter_init(s->audio_participants, &it);
                                while (cp != NULL) {
                                        struct audio_decoder *dec_state = (struct audio_decoder *) cp->decoder_state;
                                        if (dec_state && dec_state->enabled && dec_state->pbuf_data.buffer.data_len > 0) {
                                                struct audio_frame *f = &dec_state->pbuf_data.buffer;
                                                f->network_source = &dec_state->pbuf_data.source;
                                                audio_playback_put_frame(s->audio_playback_device, f);
                                        }
                                        cp = pdb_iter_next(&it);
                                }
                                pdb_iter_done(&it);
                        }
                }
        }

#ifdef HAVE_JACK_TRANS
        free(jack_pbuf.buffer.data);
#endif

        return NULL;
}

static struct response *audio_sender_process_message(struct state_audio *s, struct msg_sender *msg)
{
        switch (msg->type) {
                case SENDER_MSG_CHANGE_FEC:
                        {
                                auto old_fec_state = s->fec_state;
                                s->fec_state = NULL;
                                if (strcmp(msg->fec_cfg, "flush") == 0) {
                                        delete old_fec_state;
                                } else {
                                        s->fec_state = fec::create_from_config(msg->fec_cfg);
                                        if (!s->fec_state) {
                                                s->fec_state = old_fec_state;
                                                if (strstr(msg->fec_cfg, "help") != nullptr) { // -f LDGM:help or so + init
                                                        exit_uv(0);
                                                } else {
                                                        LOG(LOG_LEVEL_ERROR) << "[control] Unable to initalize FEC!\n";
                                                }
                                                return new_response(RESPONSE_INT_SERV_ERR, NULL);
                                        } else {
                                                delete old_fec_state;
                                                log_msg(LOG_LEVEL_NOTICE, "[control] Fec changed successfully\n");
                                        }
                                }
                        }
                        break;
                case SENDER_MSG_CHANGE_RECEIVER:
                        {
                                assert(s->audio_tx_mode == MODE_SENDER);
                                auto old_device = s->audio_network_device;
                                auto old_receiver = s->audio_network_parameters.addr;

                                s->audio_network_parameters.addr = strdup(msg->receiver);
                                s->audio_network_device =
                                        initialize_audio_network(&s->audio_network_parameters);
                                if (!s->audio_network_device) {
                                        s->audio_network_device = old_device;
                                        free(s->audio_network_parameters.addr);
                                        s->audio_network_parameters.addr = old_receiver;
                                                return new_response(RESPONSE_INT_SERV_ERR, "Changing receiver failed!");
                                } else {
                                        free(old_receiver);
                                        rtp_done(old_device);
                                }

                                break;
                        }
                case SENDER_MSG_CHANGE_PORT:
                        {
                                assert(s->audio_tx_mode == MODE_SENDER);
                                auto old_device = s->audio_network_device;
                                auto old_port = s->audio_network_parameters.send_port;

                                s->audio_network_parameters.send_port = msg->tx_port;
                                if (msg->rx_port) {
                                        s->audio_network_parameters.recv_port = msg->rx_port;
                                }
                                s->audio_network_device =
                                        initialize_audio_network(&s->audio_network_parameters);
                                if (!s->audio_network_device) {
                                        s->audio_network_device = old_device;
                                        s->audio_network_parameters.send_port = old_port;
                                                return new_response(RESPONSE_INT_SERV_ERR, "Changing receiver failed!");
                                } else {
                                        rtp_done(old_device);
                                }

                                break;
                        }
                        break;
                case SENDER_MSG_GET_STATUS:
                        {
                                char status[128] = "";
                                snprintf(status, sizeof status, "%d", (int) s->muted_sender);
                                return new_response(RESPONSE_OK, status);
                                break;
                        }
                case SENDER_MSG_PAUSE:
                        s->paused = true;
                        break;
                case SENDER_MSG_MUTE:
                        s->muted_sender = !s->muted_sender;
                        log_msg(LOG_LEVEL_NOTICE, "Audio sender %smuted.\n", s->muted_sender ? "" : "un");
                        break;
                case SENDER_MSG_PLAY:
                        s->paused = false;
                        break;
                case SENDER_MSG_QUERY_VIDEO_MODE:
                        return new_response(RESPONSE_BAD_REQUEST, NULL);
                case SENDER_MSG_RESET_SSRC:
                        {
                                assert(s->audio_tx_mode == MODE_SENDER);
                                uint32_t old_ssrc = rtp_my_ssrc(s->audio_network_device);
                                auto old_devices = s->audio_network_device;
                                s->audio_network_device =
                                        initialize_audio_network(&s->audio_network_parameters);
                                if (!s->audio_network_device) {
                                        s->audio_network_device = old_devices;
                                        log_msg(LOG_LEVEL_ERROR, "[control] Audio: Unable to change SSRC!\n");
                                        return new_response(RESPONSE_INT_SERV_ERR, NULL);
                                } else {
                                        rtp_done(old_devices);
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Audio: changed SSRC from 0x%08" PRIx32 " to "
                                                        "0x%08" PRIx32 ".\n", old_ssrc, rtp_my_ssrc(s->audio_network_device));
                                }
                        }
                        break;
        }
        return new_response(RESPONSE_OK, NULL);
}

struct asend_stats_processing_data {
        audio_frame2 frame;
        double seconds;
        bool muted_sender;
};

static void *asend_compute_and_print_stats(void *arg) {
        auto *d = (struct asend_stats_processing_data*) arg;

        log_msg(LOG_LEVEL_INFO, "[Audio sender] Sent %d samples in last %f seconds.\n",
                        d->frame.get_sample_count(),
                        d->seconds);

        ostringstream volume;
        volume << fixed << setprecision(2);
        using namespace std::string_literals;
        for (int i = 0; i < d->frame.get_channel_count(); ++i) {
                double rms = 0.0;
                double peak = 0.0;
                rms = calculate_rms(&d->frame, i, &peak);
                volume << (i > 0 ? ", ["s : "["s) << i << "] " << SBOLD(SGREEN(20.0 * log(rms) / log(10.0) << "/" << 20.0 * log(peak) / log(10.0)));
        }

        LOG(LOG_LEVEL_INFO) << "[Audio sender] Volume: " << volume.str() << " dBFS RMS/peak" << (d->muted_sender ? TBOLD(TRED(" (muted)")) : "") << "\n" ;

        delete d;

        return NULL;
}

static void process_statistics(struct state_audio *s, audio_frame2 *buffer)
{
        if (!s->captured.has_same_prop_as(*buffer)) {
                s->captured.init(buffer->get_channel_count(), buffer->get_codec(),
                                buffer->get_bps(), buffer->get_sample_rate());
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
        struct audio_frame *buffer = NULL;
        unique_ptr<audio_frame2_resampler> resampler_state;
        try {
                resampler_state = unique_ptr<audio_frame2_resampler>(new audio_frame2_resampler);
        } catch (ug_runtime_error &e) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", e.what());
                exit_uv(1);
                return NULL;
        }

        printf("Audio sending started.\n");

        while (!should_exit) {
                struct message *msg;
                while((msg = check_message(s->audio_sender_module.get()))) {
                        struct response *r = audio_sender_process_message(s, (struct msg_sender *) msg);
                        free_message(msg, r);
                }

                if ((s->audio_tx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                        time_ns_t curr_time = get_time_in_ns();
                        uint32_t ts = (curr_time - s->start_time) / 10'0000 * 9; // at 90000 Hz
                        rtp_update(s->audio_network_device, curr_time);
                        rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);

                        // receive RTCP
                        struct timeval timeout;
                        timeout.tv_sec = 0;
                        timeout.tv_usec = 0;
                        rtcp_recv_r(s->audio_network_device, &timeout, ts);
                }

                buffer = audio_capture_read(s->audio_capture_device);
                if(buffer) {
                        if(s->echo_state) {
#ifdef HAVE_SPEEXDSP
                                buffer = echo_cancel(s->echo_state, buffer);
                                if(!buffer)
                                        continue;
#endif
                        }
                        if (s->muted_sender) {
                                memset(buffer->data, 0, buffer->data_len);
                        }
                        export_audio(s->exporter, buffer);
                        if (s->paused) {
                                continue;
                        }

                        s->filter_chain.filter(&buffer);

                        if(!buffer)
                                continue;

                        audio_frame2 bf_n(buffer);

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
                        // COMPRESS
                        process_statistics(s, &bf_n);
                        // SEND
                        if(s->sender == NET_NATIVE) {
                                audio_frame2 *uncompressed = &bf_n;
                                while (audio_frame2 to_send = audio_codec_compress(s->audio_encoder, uncompressed)) {
                                        if (s->fec_state != nullptr) {
                                                to_send = s->fec_state->encode(to_send);
                                        }
                                    if(to_send.get_fec_params(0).type == FEC_RS) {
                                        audio_fec_tx_send(s->tx_session, s->audio_network_device, &to_send);
                                    }
                                    else {
                                        audio_tx_send(s->tx_session, s->audio_network_device, &to_send);
                                    }
                                        uncompressed = NULL;
                                }
                        }else if(s->sender == NET_STANDARD){
                            audio_frame2 *uncompressed = &bf_n;
                            while (audio_frame2 compressed = audio_codec_compress(s->audio_encoder, uncompressed)) {
                                    //TODO to be dynamic as a function of the selected codec, now only accepting mulaw without checking errors
                                    audio_tx_send_standard(s->tx_session, s->audio_network_device, &compressed);
                                    uncompressed = NULL;
                            }
                        }
#ifdef HAVE_JACK_TRANS
                        else
                                jack_send(s->jack_connection, buffer);
#endif
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

void audio_register_display_callbacks(struct state_audio *s, void *udata, void (*putf)(void *, const struct audio_frame *), int (*reconfigure)(void *, int, int, int), int (*get_property)(void *, int, void *, size_t *))
{
        struct state_sdi_playback *sdi_playback;
        if(!audio_playback_get_display_flags(s->audio_playback_device))
                return;
        
        sdi_playback = (struct state_sdi_playback *) audio_playback_get_state_pointer(s->audio_playback_device);
        sdi_register_display_callbacks(sdi_playback, udata, putf, reconfigure, get_property);
}

unsigned int audio_get_display_flags(struct state_audio *s)
{
        return audio_playback_get_display_flags(s->audio_playback_device);
}

std::ostream& operator<<(std::ostream& os, const audio_desc& desc)
{
    os << desc.ch_count << " channel" << (desc.ch_count > 1 ? "s" : "") << ", " << desc.bps << " Bps, " << desc.sample_rate << " Hz, codec: " << get_name_to_audio_codec(desc.codec);
    return os;
}


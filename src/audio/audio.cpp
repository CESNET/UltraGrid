/*
 * FILE:    audio/audio.cpp
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2017 CESNET z.s.p.o.
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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
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
#include "audio/utils.h"
#include "compat/platform_semaphore.h"
#include "compat/platform_semaphore.h"
#include "debug.h"
#include "../export.h" // not audio/export.h
#include "host.h"
#include "module.h"
#include "perf.h"
#include "rang.hpp"
#include "rtp/audio_decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "tv.h"
#include "transmit.h"
#include "pdb.h"
#include "utils/worker.h"

using namespace std;
using rang::fg;
using rang::style;

enum audio_transport_device {
        NET_NATIVE,
        NET_JACK,
        NET_STANDARD
};

struct audio_network_parameters {
        char *addr = nullptr;
        int recv_port = 0;
        int send_port = 0;
        struct pdb *participants = 0;
        int force_ip_version = 0;
        char *mcast_if = nullptr;
};

struct state_audio {
        state_audio(struct module *parent) {
                module_init_default(&mod);
                mod.priv_data = this;
                mod.cls = MODULE_CLASS_AUDIO;
                module_register(&mod, parent);

                module_init_default(&audio_receiver_module);
                audio_receiver_module.cls = MODULE_CLASS_RECEIVER;
                audio_receiver_module.priv_data = this;
                module_register(&audio_receiver_module, &mod);

                module_init_default(&audio_sender_module);
                audio_sender_module.cls = MODULE_CLASS_SENDER;
                audio_sender_module.priv_data = this;
                module_register(&audio_sender_module, &mod);

                gettimeofday(&t0, NULL);
        }
        ~state_audio() {
                module_done(&audio_receiver_module);
                module_done(&audio_sender_module);
                module_done(&mod);
        }

        struct module mod;
        struct state_audio_capture *audio_capture_device = nullptr;
        struct state_audio_playback *audio_playback_device = nullptr;

        struct module audio_receiver_module;
        struct module audio_sender_module;

        struct audio_codec_state *audio_coder = nullptr;
        
        struct audio_network_parameters audio_network_parameters{};
        struct rtp *audio_network_device = nullptr;
        struct pdb *audio_participants = nullptr;
        std::string proto_cfg; // audio network protocol options
        void *jack_connection = nullptr;
        enum audio_transport_device sender = NET_NATIVE;
        enum audio_transport_device receiver = NET_NATIVE;
        
        std::chrono::steady_clock::time_point start_time;

        struct timeval t0; // for statistics
        audio_frame2 captured;

        struct tx *tx_session = nullptr;
        
        pthread_t audio_sender_thread_id,
                  audio_receiver_thread_id;
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
        printf("\t--audio-scale [<factor>|<method>]\n");
        printf("\t                                 Floating point number that tells\n");
        printf("\t                                 a static scaling factor for all\n");
        printf("\t                                 output channels.\n");
        printf("\t                                 Scaling method can be one from these:\n");
        printf("\t                                   mixauto - automatically adjust\n");
        printf("\t                                             volume if using channel\n");
        printf("\t                                             mixing/remapping\n");
        printf("\t                                             (default)\n");
        printf("\t                                   auto - automatically adjust volume\n");
        printf("\t                                   none - no scaling will be performed\n");
}

/**
 * take care that addrs can also be comma-separated list of addresses !
 */
struct state_audio * audio_cfg_init(struct module *parent, const char *addrs, int recv_port, int send_port,
                const char *send_cfg, const char *recv_cfg,
                const char *proto, const char *proto_cfg,
                const char *fec_cfg, const char *encryption,
                char *audio_channel_map, const char *audio_scale,
                bool echo_cancellation, int force_ip_version, const char *mcast_if,
                const char *audio_codec_cfg,
                long long int bitrate, volatile int *audio_delay, const std::chrono::steady_clock::time_point *start_time, int mtu, struct exporter *exporter)
{
        struct state_audio *s = NULL;
        char *tmp, *unused = NULL;
        UNUSED(unused);
        char *addr;
        int resample_to = get_audio_codec_sample_rate(audio_codec_cfg);
        
        assert(send_cfg != NULL);
        assert(recv_cfg != NULL);

        if (!strcmp("help", send_cfg)) {
                audio_capture_print_help();
                exit_uv(0);
                return NULL;
        }
        
        if (!strcmp("help", recv_cfg)) {
                audio_playback_help();
                exit_uv(0);
                return NULL;
        }

        if(audio_channel_map &&
                     strcmp("help", audio_channel_map) == 0) {
                audio_channel_map_usage();
                exit_uv(0);
                return NULL;
        }

        if(audio_scale &&
                     strcmp("help", audio_scale) == 0) {
                audio_scale_usage();
                exit_uv(0);
                return NULL;
        }
        
        s = new state_audio(parent);
        s->start_time = *start_time;

        s->audio_channel_map = audio_channel_map;
        s->audio_scale = audio_scale;

        s->audio_sender_thread_started = s->audio_receiver_thread_started = false;
        s->resample_to = resample_to;

        s->audio_coder = audio_codec_init_cfg(audio_codec_cfg, AUDIO_CODER);
        if(!s->audio_coder) {
                goto error;
        }

        s->exporter = exporter;

        if(echo_cancellation) {
#ifdef HAVE_SPEEX
                //s->echo_state = echo_cancellation_init();
                fprintf(stderr, "Echo cancellation is currently broken "
                                "in UltraGrid.\nPlease write to %s "
                                "if you wish to use this feature.\n",
                                PACKAGE_BUGREPORT);
                goto error;
#else
                fprintf(stderr, "Speex not compiled in. Could not enable echo cancellation.\n");
                delete s;
                goto error;
#endif /* HAVE_SPEEX */
        } else {
                s->echo_state = NULL;
        }

        if(encryption) {
                s->requested_encryption = strdup(encryption);
        }
        
        assert(addrs && strlen(addrs) > 0);
        tmp = strdup(addrs);
        s->audio_participants = pdb_init(audio_delay);
        addr = strtok_r(tmp, ",", &unused);

        s->audio_network_parameters.addr = strdup(addr);
        s->audio_network_parameters.recv_port = recv_port;
        s->audio_network_parameters.send_port = send_port;
        s->audio_network_parameters.participants = s->audio_participants;
        s->audio_network_parameters.force_ip_version = force_ip_version;
        s->audio_network_parameters.mcast_if = mcast_if
                ? strdup(mcast_if) : NULL;

        if ((s->audio_network_device = initialize_audio_network(
                                        &s->audio_network_parameters))
                        == NULL) {
                printf("Unable to open audio network\n");
                free(tmp);
                goto error;
        }
        free(tmp);

        if (strcmp(send_cfg, "none") != 0) {
                char *cfg = NULL;
                char *device = strdup(send_cfg);
		if(strchr(device, ':')) {
			char *delim = strchr(device, ':');
			*delim = '\0';
			cfg = delim + 1;
		}

                int ret = audio_capture_init(device, cfg, &s->audio_capture_device);
                free(device);
                
                if(ret < 0) {
                        fprintf(stderr, "Error initializing audio capture.\n");
                        goto error;
                }
                if(ret > 0) {
                        goto error;
                }
                s->tx_session = tx_init(&s->audio_sender_module, mtu, TX_MEDIA_AUDIO, fec_cfg, encryption, bitrate);
                if(!s->tx_session) {
                        fprintf(stderr, "Unable to initialize audio transmit.\n");
                        goto error;
                }

                s->audio_tx_mode |= MODE_SENDER;
        } else {
                s->audio_capture_device = audio_capture_init_null_device();
        }
        
        if (strcmp(recv_cfg, "none") != 0) {
                char *cfg = NULL;
                char *device = strdup(recv_cfg);
		if(strchr(device, ':')) {
			char *delim = strchr(device, ':');
			*delim = '\0';
			cfg = delim + 1;
		}

                int ret = audio_playback_init(device, cfg, &s->audio_playback_device);
                free(device);
                if(ret < 0) {
                        fprintf(stderr, "Error initializing audio playback.\n");
                        goto error;
                }
                if(ret > 0) {
                        goto error;
                }
                size_t len = sizeof(struct rtp *);
                audio_playback_ctl(s->audio_playback_device, AUDIO_PLAYBACK_PUT_NETWORK_DEVICE,
                                        &s->audio_network_device, &len);

                s->audio_tx_mode |= MODE_RECEIVER;
        } else {
                s->audio_playback_device = audio_playback_init_null_device();
        }

        s->proto_cfg = proto_cfg;

        if (strcasecmp(proto, "ultragrid_rtp") == 0) {
                s->sender = NET_NATIVE;
                s->receiver = NET_NATIVE;
        } else if (strcasecmp(proto, "rtsp") == 0 || strcasecmp(proto, "sdp") == 0) {
                s->receiver = NET_STANDARD;
                s->sender = NET_STANDARD;
        } else if (strcasecmp(proto, "JACK") == 0) {
#ifdef HAVE_JACK_TRANS
                fprintf(stderr, "[Audio] JACK configuration string entered ('-j'), "
                                "but JACK support isn't compiled.\n");
                goto error;
#endif
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unknow audio protocol: %s\n", proto);
                goto error;
        }

        return s;

error:
        if(s->tx_session)
                module_done(CAST_MODULE(s->tx_session));
        if(s->audio_participants) {
                pdb_destroy(&s->audio_participants);
        }

        audio_codec_done(s->audio_coder);
        delete s;
        exit_uv(EXIT_FAIL_AUDIO);
        return NULL;
}

void audio_start(struct state_audio *s) {
#ifdef HAVE_JACK_TRANS
        s->jack_connection = jack_start(s->proto_cfg.c_str());
        if(s->jack_connection) {
                if(is_jack_sender(s->jack_connection))
                        s->sender = NET_JACK;
                if(is_jack_receiver(s->jack_connection))
                        s->receiver = NET_JACK;
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
        while ((msg = check_message(&s->audio_receiver_module))) {
                struct response *r = audio_receiver_process_message(s, (struct msg_receiver *) msg);
                free_message(msg, r);
        }
        while ((msg = check_message(&s->audio_sender_module))) {
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

        audio_codec_done(s->audio_coder);

        delete s;
}

static struct rtp *initialize_audio_network(struct audio_network_parameters *params)
{
        struct rtp *r;
        double rtcp_bw = 1024 * 512;    // FIXME:  something about 5% for rtcp is said in rfc

        r = rtp_init_if(params->addr, params->mcast_if, params->recv_port,
                        params->send_port, 255, rtcp_bw,
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
                rtp_set_recv_buf(r, 256*1024);
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
        case RECEIVER_MSG_GET_VOLUME:
                {
                        double ret = s->volume;
                        char volume_str[128] = "";
                        snprintf(volume_str, sizeof volume_str, "%f", ret);
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
                        log_msg(LOG_LEVEL_INFO, "Playback volume: %.2f%% (%+.2f dB)\n", new_volume * 100.0, db);
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

static void audio_decoder_state_deleter(void *state)
{
        struct audio_decoder *s = (struct audio_decoder *) state;

        free(s->pbuf_data.buffer.data);
        audio_decoder_destroy(s->pbuf_data.decoder);

        free(s);
}

static void *audio_receiver_thread(void *arg)
{
        struct state_audio *s = (struct state_audio *) arg;
        // rtp variables
        struct timeval timeout, curr_time;
        uint32_t ts;
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
                while((msg= check_message(&s->audio_receiver_module))) {
                        struct response *r = audio_receiver_process_message(s, (struct msg_receiver *) msg);
                        free_message(msg, r);
                }

                bool decoded = false;

                if (s->receiver == NET_NATIVE || s->receiver == NET_STANDARD) {
                        gettimeofday(&curr_time, NULL);
                        auto curr_time_hr = std::chrono::high_resolution_clock::now();
                        ts = std::chrono::duration_cast<std::chrono::duration<double>>(s->start_time - std::chrono::steady_clock::now()).count() * 90000;
                        rtp_update(s->audio_network_device, curr_time);
                        rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);
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
                                        struct audio_decoder *dec_state;
                                        dec_state = (struct audio_decoder *) calloc(1, sizeof(struct audio_decoder));

                                        if (get_commandline_param("low-latency-audio")) {
                                                pbuf_set_playout_delay(cp->playout_buffer, 0.005);
                                        }
                                        assert(dec_state != NULL);
                                        cp->decoder_state = dec_state;
                                        dec_state->enabled = true;
                                        dec_state->pbuf_data.decoder = (struct state_audio_decoder *) audio_decoder_init(s->audio_channel_map, s->audio_scale, s->requested_encryption, (audio_playback_ctl_t) audio_playback_ctl, s->audio_playback_device, &s->audio_receiver_module);
                                        audio_decoder_set_volume(dec_state->pbuf_data.decoder, s->muted_receiver ? 0.0 : s->volume);
                                        assert(dec_state->pbuf_data.decoder != NULL);
                                        cp->decoder_state_deleter = audio_decoder_state_deleter;
                                }

                                struct audio_decoder *dec_state = (struct audio_decoder *) cp->decoder_state;
                                if (dec_state && dec_state->enabled) {
                                        dec_state->pbuf_data.buffer.data_len = 0;
                                        // We iterate in loop since there can be more than one frmae present in
                                        // the playout buffer and it would be discarded by following pbuf_remove()
                                        // call.
                                        while (pbuf_decode(cp->playout_buffer, curr_time_hr, s->receiver == NET_NATIVE ? decode_audio_frame : decode_audio_frame_mulaw, &dec_state->pbuf_data)) {

                                                current_pbuf = &dec_state->pbuf_data;
                                                decoded = true;
                                        }
                                }

                                pbuf_remove(cp->playout_buffer, curr_time_hr);
                                cp = pdb_iter_next(&it);

                                if (decoded && !playback_supports_multiple_streams)
                                        break;
                        }
                        pdb_iter_done(&it);
                }else { /* NET_JACK */
#ifdef HAVE_JACK_TRANS
                        decoded = jack_receive(s->jack_connection, &jack_pbuf);
                        audio_playback_put_frame(s->audio_playback_device, &jack_pbuf.buffer);
#endif
                }

                if (decoded) {
                        bool failed = false;
                        if(s->echo_state) {
#ifdef HAVE_SPEEX
                                echo_play(s->echo_state, &current_pbuf->buffer);
#endif
                        }

                        struct audio_desc curr_desc;
                        curr_desc = audio_desc_from_audio_frame(&current_pbuf->buffer);

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

                        if(!failed) {
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
        }

#ifdef HAVE_JACK_TRANS
        free(jack_pbuf.buffer.data);
#endif

        return NULL;
}

static struct response *audio_sender_process_message(struct state_audio *s, struct msg_sender *msg)
{
        switch (msg->type) {
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
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Audio: changed SSRC from 0x%08lx to "
                                                        "0x%08lx.\n", old_ssrc, rtp_my_ssrc(s->audio_network_device));
                                }
                        }
                        break;
                case SENDER_MSG_CHANGE_FEC:
                        if (strcmp(msg->fec_cfg, "flush") != 0) {
                                LOG(LOG_LEVEL_ERROR) << "Not implemented!\n";
                        }
                        return new_response(RESPONSE_NOT_IMPL, NULL);
        }
        return new_response(RESPONSE_OK, NULL);
}

struct asend_stats_processing_data {
        audio_frame2 frame;
        double seconds;
};

static void *asend_compute_and_print_stats(void *arg) {
        auto d = (struct asend_stats_processing_data*) arg;

        log_msg(LOG_LEVEL_INFO, "[Audio sender] Sent %d samples in last %f seconds.\n",
                        d->frame.get_sample_count(),
                        d->seconds);
        for (int i = 0; i < d->frame.get_channel_count(); ++i) {
                double rms, peak;
                rms = calculate_rms(&d->frame, i, &peak);
                LOG(LOG_LEVEL_INFO) << "[Audio sender] Channel " << i << " - volume: " << setprecision(2) << fixed << fg::green << style::bold << 20 * log(rms) / log(10) << style::reset << fg::reset << " dBFS RMS, " << fg::green << style::bold << 20 * log(peak) / log(10) << style::reset << fg::reset << " dBFS peak.\n";
        }

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

        struct timeval t;
        double seconds;
        gettimeofday(&t, 0);
        seconds = tv_diff(t, s->t0);
        if (seconds > 5.0) {
                auto d = new asend_stats_processing_data;
                std::swap(d->frame, s->captured);
                d->seconds = seconds;

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
        struct state_audio *s = (struct state_audio *) arg;
        struct audio_frame *buffer = NULL;
        audio_frame2_resampler resampler_state;

        printf("Audio sending started.\n");
        while (!should_exit) {
                struct message *msg;
                while((msg = check_message(&s->audio_sender_module))) {
                        struct response *r = audio_sender_process_message(s, (struct msg_sender *) msg);
                        free_message(msg, r);
                }

                if ((s->audio_tx_mode & MODE_RECEIVER) == 0) { // otherwise receiver thread does the stuff...
                        struct timeval curr_time;
                        uint32_t ts;
                        gettimeofday(&curr_time, NULL);
                        ts = std::chrono::duration_cast<std::chrono::duration<double>>(s->start_time - std::chrono::steady_clock::now()).count() * 90000;
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
                        if (s->muted_sender) {
                                memset(buffer->data, 0, buffer->data_len);
                        }
                        export_audio(s->exporter, buffer);
                        if(s->echo_state) {
#ifdef HAVE_SPEEX
                                buffer = echo_cancel(s->echo_state, buffer);
                                if(!buffer)
                                        continue;
#endif
                        }
                        if (s->paused) {
                                continue;
                        }

                        audio_frame2 bf_n(buffer);

                        // RESAMPLE
                        int resample_to = s->resample_to;
                        if (resample_to == 0) {
                                const int *supp_sample_rates = audio_codec_get_supported_samplerates(s->audio_coder);
                                resample_to = find_codec_sample_rate(bf_n.get_sample_rate(),
                                                supp_sample_rates);
                        }
                        if (resample_to != 0 && bf_n.get_sample_rate() != s->resample_to) {
                                if (bf_n.get_bps() != 2) {
                                        bf_n.change_bps(2);
                                }

                                bf_n.resample(resampler_state, resample_to);
                        }
                        // COMPRESS
                        process_statistics(s, &bf_n);
                        // SEND
                        if(s->sender == NET_NATIVE) {
                                audio_frame2 *uncompressed = &bf_n;
                                const audio_frame2 *compressed = NULL;
                                while((compressed = audio_codec_compress(s->audio_coder, uncompressed))) {
                                        audio_tx_send(s->tx_session, s->audio_network_device, compressed);
                                        uncompressed = NULL;
                                }
                        }else if(s->sender == NET_STANDARD){
                            audio_frame2 *uncompressed = &bf_n;
                            const audio_frame2 *compressed = NULL;
                            while((compressed = audio_codec_compress(s->audio_coder, uncompressed))) {
                                    //TODO to be dynamic as a function of the selected codec, now only accepting mulaw without checking errors
                                    audio_tx_send_standard(s->tx_session, s->audio_network_device, compressed);
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

void audio_register_display_callbacks(struct state_audio *s, void *udata, void (*putf)(void *, struct audio_frame *), int (*reconfigure)(void *, int, int, int), int (*get_property)(void *, int, void *, size_t *))
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

struct audio_desc audio_desc_from_frame(struct audio_frame *frame)
{
        return (struct audio_desc) { frame->bps, frame->sample_rate,
                frame->ch_count, AC_PCM };
}

std::ostream& operator<<(std::ostream& os, const audio_desc& desc)
{
    os << desc.ch_count << " channel" << (desc.ch_count > 1 ? "s" : "") << ", " << desc.bps << " Bps, " << desc.sample_rate << " Hz, codec: " << get_name_to_audio_codec(desc.codec);
    return os;
}


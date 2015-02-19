/*
 * FILE:    audio/audio.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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

#include <speex/speex_resampler.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "audio/audio.h" 

#include "audio/codec.h"
#include "audio/echo.h" 
#include "audio/export.h" 
#include "audio/audio_capture.h" 
#include "audio/audio_playback.h" 
#include "audio/capture/sdi.h"
#include "audio/playback/sdi.h"
#include "audio/jack.h" 
#include "audio/utils.h"
#include "compat/platform_semaphore.h"
#include "debug.h"
#include "host.h"
#include "module.h"
#include "perf.h"
#include "rtp/audio_decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "tv.h"
#include "transmit.h"
#include "pdb.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static volatile bool should_exit_audio = false;

struct audio_device_t {
        int index;
        void *state;
};

enum audio_transport_device {
        NET_NATIVE,
        NET_JACK,
        NET_STANDARD
};

struct audio_network_parameters {
        char *addr;
        int recv_port;
        int send_port;
        struct pdb *participants;
        bool use_ipv6;
        char *mcast_if;
};

struct state_audio {
        struct module mod;
        struct state_audio_capture *audio_capture_device;
        struct state_audio_playback *audio_playback_device;

        struct module audio_receiver_module;
        struct module audio_sender_module;

        struct audio_codec_state *audio_coder;
        
        struct audio_network_parameters audio_network_parameters;
        struct rtp *audio_network_device;
        struct pdb *audio_participants;
        void *jack_connection;
        enum audio_transport_device sender;
        enum audio_transport_device receiver;
        
        struct timeval start_time;

        struct timeval t0; // for statistics
        audio_frame2 *captured;

        struct tx *tx_session;
        
        pthread_t audio_sender_thread_id,
                  audio_receiver_thread_id;
	bool audio_sender_thread_started,
		audio_receiver_thread_started;

        char *audio_channel_map;
        const char *audio_scale;
        echo_cancellation_t *echo_state;
        struct audio_export *exporter;
        int  resample_to;

        char *requested_encryption;

        volatile bool paused;

        int audio_tx_mode;
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
                char *jack_cfg, const char *fec_cfg, const char *encryption,
                char *audio_channel_map, const char *audio_scale,
                bool echo_cancellation, bool use_ipv6, const char *mcast_if,
                const char *audio_codec_cfg,
                bool isStd, long packet_rate)
{
        struct state_audio *s = NULL;
        char *tmp, *unused = NULL;
        UNUSED(unused);
        char *addr;
        int resample_to = get_audio_codec_sample_rate(audio_codec_cfg);
        
        audio_capture_init_devices();
        audio_playback_init_devices();

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
        
        s = (struct state_audio *) calloc(1, sizeof(struct state_audio));

        if (strcmp("none", send_cfg) == 0 && strcmp("none", recv_cfg) == 0) {
                // nothing to do, return empty state
                return s;
        }

        module_init_default(&s->mod);
        s->mod.priv_data = s;
        s->mod.cls = MODULE_CLASS_AUDIO;
        module_register(&s->mod, parent);

        module_init_default(&s->audio_receiver_module);
        s->audio_receiver_module.cls = MODULE_CLASS_RECEIVER;
        s->audio_receiver_module.priv_data = s;
        module_register(&s->audio_receiver_module, &s->mod);

        module_init_default(&s->audio_sender_module);
        s->audio_sender_module.cls = MODULE_CLASS_SENDER;
        s->audio_sender_module.priv_data = s;
        module_register(&s->audio_sender_module, &s->mod);


        s->audio_participants = NULL;
        s->audio_channel_map = audio_channel_map;
        s->audio_scale = audio_scale;

        s->audio_sender_thread_started = s->audio_receiver_thread_started = false;
        s->resample_to = resample_to;

        s->audio_coder = audio_codec_init_cfg(audio_codec_cfg, AUDIO_CODER);
        if(!s->audio_coder) {
                goto error;
        }

        if(export_dir) {
                char name[512];
                snprintf(name, 512, "%s/sound.wav", export_dir);
                s->exporter = audio_export_init(name);
        } else {
                s->exporter = NULL;
        }

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
                free(s);
                goto error;
#endif /* HAVE_SPEEX */
        } else {
                s->echo_state = NULL;
        }

        if(encryption) {
                s->requested_encryption = strdup(encryption);
        }
        
        s->tx_session = tx_init(&s->mod, 1500, TX_MEDIA_AUDIO, fec_cfg, encryption, packet_rate);
        if(!s->tx_session) {
                fprintf(stderr, "Unable to initialize audio transmit.\n");
                goto error;
        }

        gettimeofday(&s->start_time, NULL);        
        gettimeofday(&s->t0, NULL);
        s->captured = new audio_frame2;
        
        tmp = strdup(addrs);
        s->audio_participants = pdb_init();
        addr = strtok_r(tmp, ",", &unused);

        s->audio_network_parameters.addr = strdup(addr);
        s->audio_network_parameters.recv_port = recv_port;
        s->audio_network_parameters.send_port = send_port;
        s->audio_network_parameters.participants = s->audio_participants;
        s->audio_network_parameters.use_ipv6 = use_ipv6;
        s->audio_network_parameters.mcast_if = mcast_if
                ? strdup(mcast_if) : NULL;

        if (strcmp(recv_cfg, "none") == 0) {
                // do not occupy recv port if we are not receiving (note that this disables communication with
                // our receiver, because RTCP ports are changed as well)
                s->audio_network_parameters.recv_port = 0;
        }

        if ((s->audio_network_device = initialize_audio_network(
                                        &s->audio_network_parameters))
                        == NULL) {
                printf("Unable to open audio network\n");
                free(tmp);
                goto error;
        }
        free(tmp);

        s->audio_tx_mode = 0;

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
                s->audio_tx_mode |= MODE_RECEIVER;
        } else {
                s->audio_playback_device = audio_playback_init_null_device();
        }

        if (strcmp(send_cfg, "none") != 0) {
                if (pthread_create
                    (&s->audio_sender_thread_id, NULL, audio_sender_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        goto error;
                } else {
			s->audio_sender_thread_started = true;
		}
        }

        if (strcmp(recv_cfg, "none") != 0) {
                if (pthread_create
                    (&s->audio_receiver_thread_id, NULL, audio_receiver_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        goto error;
                } else {
			s->audio_receiver_thread_started = true;
		}
        }
        
        s->sender = NET_NATIVE;
        s->receiver = NET_NATIVE;
        if(isStd && strcmp(recv_cfg, "none") != 0) s->receiver = NET_STANDARD;
        if(isStd && strcmp(send_cfg, "none") != 0) s->sender = NET_STANDARD;

#ifdef HAVE_JACK_TRANS
        s->jack_connection = jack_start(jack_cfg);
        if(s->jack_connection) {
                if(is_jack_sender(s->jack_connection))
                        s->sender = NET_JACK;
                if(is_jack_receiver(s->jack_connection))
                        s->receiver = NET_JACK;
        }
#else
        if(jack_cfg) {
                fprintf(stderr, "[Audio] JACK configuration string entered ('-j'), "
                                "but JACK support isn't compiled.\n");
                goto error;
        }
#endif

        return s;

error:
        if(s->tx_session)
                module_done(CAST_MODULE(s->tx_session));
        if(s->audio_participants) {
                pdb_destroy(&s->audio_participants);
        }

        if (s->audio_receiver_module.cls != 0) {
                module_done(&s->audio_receiver_module);
        }
        if (s->audio_sender_module.cls != 0) {
                module_done(&s->audio_sender_module);
        }
        if (s->mod.cls != 0) {
                module_done(&s->mod);
        }

        delete s->captured;

        audio_codec_done(s->audio_coder);
        free(s);
        exit_uv(1);
        return NULL;
}

void audio_join(struct state_audio *s) {
        if(s) {
                if(s->audio_receiver_thread_started)
                        pthread_join(s->audio_receiver_thread_id, NULL);
                if(s->audio_sender_thread_started)
                        pthread_join(s->audio_sender_thread_id, NULL);
        }
}
        
void audio_finish()
{
        should_exit_audio = true;
}

void audio_done(struct state_audio *s)
{
        if(s) {
                audio_playback_done(s->audio_playback_device);
                audio_capture_done(s->audio_capture_device);
                module_done(CAST_MODULE(s->tx_session));
                module_done(&s->audio_receiver_module);
                module_done(&s->audio_sender_module);
                if(s->audio_network_device)
                        rtp_done(s->audio_network_device);
                if(s->audio_participants) {
                        pdb_iter_t it;
                        struct pdb_e *cp = pdb_iter_init(s->audio_participants, &it);
                        while (cp != NULL) {
                                struct pdb_e *item = NULL;
                                pdb_remove(s->audio_participants, cp->ssrc, &item);
                                cp = pdb_iter_next(&it);
                                free(item);
                        }
                        pdb_iter_done(&it);
                        pdb_destroy(&s->audio_participants);
                }
                audio_export_destroy(s->exporter);
                module_done(&s->mod);
                free(s->requested_encryption);

                free(s->audio_network_parameters.addr);
                free(s->audio_network_parameters.mcast_if);

                delete s->captured;

                free(s);
        }
}

static struct rtp *initialize_audio_network(struct audio_network_parameters *params)
{
        struct rtp *r;
        double rtcp_bw = 1024 * 512;    // FIXME:  something about 5% for rtcp is said in rfc

        r = rtp_init_if(params->addr, params->mcast_if, params->recv_port,
                        params->send_port, 255, rtcp_bw,
                        FALSE, rtp_recv_callback,
                        (uint8_t *) params->participants,
                        params->use_ipv6, false);
        if (r != NULL) {
                pdb_add(params->participants, rtp_my_ssrc(r));
                rtp_set_option(r, RTP_OPT_WEAK_VALIDATION, TRUE);
                rtp_set_sdes(r, rtp_my_ssrc(r), RTCP_SDES_TOOL,
                             PACKAGE_STRING, strlen(PACKAGE_VERSION));
        }

        return r;
}

static void audio_receiver_process_message(struct state_audio *s, struct msg_receiver *msg)
{
        switch (msg->type) {
        case RECEIVER_MSG_CHANGE_RX_PORT:
                assert(s->audio_tx_mode == MODE_RECEIVER); // receiver only
                rtp_done(s->audio_network_device);
                s->audio_network_parameters.recv_port = msg->new_rx_port;
                s->audio_network_device = initialize_audio_network(
                                &s->audio_network_parameters);
                if (!s->audio_network_device) {
                        fprintf(stderr, "Changing RX port failed!");
                }
                break;
        default:
                abort();
        }
}

static void *audio_receiver_thread(void *arg)
{
        struct state_audio *s = (struct state_audio *) arg;
        // rtp variables
        struct timeval timeout, curr_time;
        uint32_t ts;
        struct pdb_e *cp;
        struct pbuf_audio_data pbuf_data;
        struct audio_desc device_desc;

        memset(&pbuf_data.buffer, 0, sizeof(struct audio_frame));
        memset(&device_desc, 0, sizeof(struct audio_desc));

        pbuf_data.decoder = (struct state_audio_decoder *) audio_decoder_init(s->audio_channel_map, s->audio_scale, s->requested_encryption);
        assert(pbuf_data.decoder != NULL);
                
        printf("Audio receiving started.\n");
        while (!should_exit_audio) {
                struct message *msg;
                while((msg= check_message(&s->audio_receiver_module))) {
                        audio_receiver_process_message(s, (struct msg_receiver *) msg);
                        free_message(msg);
                }

                bool decoded = false;

                pbuf_data.buffer.data_len = 0;

                if(s->receiver == NET_NATIVE) {
                        gettimeofday(&curr_time, NULL);
                        ts = tv_diff(curr_time, s->start_time) * 90000;
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
                                // We iterate in loop since there can be more than one frmae present in
                                // the playout buffer and it would be discarded by following pbuf_remove()
                                // call.
                                while (pbuf_decode(cp->playout_buffer, curr_time, decode_audio_frame, &pbuf_data)) {
                                        decoded = true;
                                }

                                pbuf_remove(cp->playout_buffer, curr_time);
                                cp = pdb_iter_next(&it);

                                if (decoded)
                                        break;
                        }
                        pdb_iter_done(&it);
                }else if(s->receiver == NET_STANDARD){
                //TODO now expecting to receive mulaw standard RTP (decode frame mulaw callback) , next steps, to be dynamic...
                    gettimeofday(&curr_time, NULL);
                    ts = tv_diff(curr_time, s->start_time) * 90000;
                    rtp_update(s->audio_network_device, curr_time);
                    rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);
                    timeout.tv_sec = 0;
                    timeout.tv_usec = 999999 / 59.94; /* audio goes almost always at the same rate
                                                                                 as video frames */
                    rtp_recv_r(s->audio_network_device, &timeout, ts);
                    pdb_iter_t it;
                    cp = pdb_iter_init(s->audio_participants, &it);

                    while (cp != NULL) {
                        // should be perhaps run iteratively? similarly to NET_NATIVE
                        if (pbuf_decode(cp->playout_buffer, curr_time, decode_audio_frame_mulaw, &pbuf_data)) {
                            bool failed = false;
                            if(s->echo_state) {
#ifdef HAVE_SPEEX
echo_play(s->echo_state, &pbuf_data.buffer);
#endif
                            }

                            struct audio_desc curr_desc;
                            curr_desc = audio_desc_from_audio_frame(&pbuf_data.buffer);

                            if(!audio_desc_eq(device_desc, curr_desc)) {
                                if(audio_reconfigure(s, curr_desc.bps * 8,
                                    curr_desc.ch_count,
                                    curr_desc.sample_rate) != TRUE) {
                                    fprintf(stderr, "Audio reconfiguration failed!");
                                    failed = true;
                                }
                                else {
                                    fprintf(stderr, "Audio reconfiguration succeeded.");
                                    device_desc = curr_desc;
                                    rtp_flush_recv_buf(s->audio_network_device);
                                }
                                fprintf(stderr, " (%d channels, %d bps, %d Hz)\n",
                                    curr_desc.ch_count,
                                    curr_desc.bps, curr_desc.sample_rate);

                            }

                            if(!failed)
                                audio_playback_put_frame(s->audio_playback_device, &pbuf_data.buffer);
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(&it);
                    }
                    pdb_iter_done(&it);
                }else { /* NET_JACK */
#ifdef HAVE_JACK_TRANS
                        decoded = jack_receive(s->jack_connection, &pbuf_data);
                        audio_playback_put_frame(s->audio_playback_device, &pbuf_data.buffer);
#endif
                }

                if (decoded) {
                        bool failed = false;
                        if(s->echo_state) {
#ifdef HAVE_SPEEX
                                echo_play(s->echo_state, &pbuf_data.buffer);
#endif
                        }

                        struct audio_desc curr_desc;
                        curr_desc = audio_desc_from_audio_frame(&pbuf_data.buffer);

                        if(!audio_desc_eq(device_desc, curr_desc)) {
                                if(audio_reconfigure(s, curr_desc.bps * 8,
                                                        curr_desc.ch_count,
                                                        curr_desc.sample_rate) != TRUE) {
                                        fprintf(stderr, "Audio reconfiguration failed!");
                                        failed = true;
                                }
                                else {
                                        fprintf(stderr, "Audio reconfiguration succeeded.");
                                        device_desc = curr_desc;
                                        rtp_flush_recv_buf(s->audio_network_device);
                                }
                                fprintf(stderr, " (%d channels, %d bps, %d Hz)\n",
                                                curr_desc.ch_count,
                                                curr_desc.bps, curr_desc.sample_rate);

                        }

                        if(!failed)
                                audio_playback_put_frame(s->audio_playback_device, &pbuf_data.buffer);
                }
        }

        free(pbuf_data.buffer.data);
        audio_decoder_destroy(pbuf_data.decoder);

        return NULL;
}

struct state_resample {
        struct audio_frame resampled;
        char *resample_buffer;
        SpeexResamplerState *resampler;
        int resample_from, resample_ch_count;
        int resample_to;
        const int *codec_supported_bytes_per_sample;
};

static void resample(struct state_resample *s, struct audio_frame *buffer);
static bool set_contains(const int *vals, int needle);

static bool set_contains(const int *vals, int needle)
{
        if(!vals)
                return true;
        while(*vals != 0) {
                if(*vals == needle) {
                        return true;
                }
                ++vals;
        }
        return false;
}

static void resample(struct state_resample *s, struct audio_frame *buffer)
{
        memcpy(&s->resampled, buffer, sizeof(struct audio_frame));

        if (buffer->sample_rate == s->resample_to &&
                        set_contains(s->codec_supported_bytes_per_sample, buffer->bps)) {
                memcpy(&s->resampled, buffer, sizeof(s->resampled));
                s->resampled.data = (char *) malloc(buffer->data_len);
                memcpy(s->resampled.data, buffer->data, buffer->data_len);
        } else {
                // resampler is able only to resample 16-bit samples
                assert(set_contains(s->codec_supported_bytes_per_sample, 2));
                // expect that we may got as much as 12-times more data (eg. 8 kHz to 96 kHz)
                uint32_t write_frames = 12 * (buffer->data_len / buffer->ch_count / buffer->bps);
                s->resampled.data = (char *) malloc(write_frames * 2 * buffer->ch_count);
                if(s->resample_from != buffer->sample_rate || s->resample_ch_count != buffer->ch_count) {
                        s->resample_from = buffer->sample_rate;
                        s->resample_ch_count = buffer->ch_count;
                        if(s->resampler) {
                                speex_resampler_destroy(s->resampler);
                        }
                        int err;
                        s->resampler = speex_resampler_init(buffer->ch_count, s->resample_from,
                                        s->resample_to, 10, &err);
                        if(err) {
                                abort();
                        }
                }
                char *in_buf;
                int data_len;
                if(buffer->bps != 2) {
                        change_bps(s->resample_buffer, 2, buffer->data, buffer->bps, buffer->data_len);
                        in_buf = s->resample_buffer;
                        data_len = buffer->data_len / buffer->bps * 2;
                } else {
                        in_buf = buffer->data;
                        data_len = buffer->data_len;
                }

                uint32_t in_frames = data_len /  buffer->ch_count / 2;
                uint32_t in_frames_orig = in_frames;
                speex_resampler_process_interleaved_int(s->resampler, (spx_int16_t *)(void *) in_buf, &in_frames,
                                (spx_int16_t *)(void *) s->resampled.data, &write_frames);
                assert (in_frames == in_frames_orig);

                s->resampled.data_len = write_frames * 2 /* bps */ * buffer->ch_count;
                s->resampled.sample_rate = s->resample_to;
                s->resampled.bps = 2;
        }
}

static void audio_sender_process_message(struct state_audio *s, struct msg_sender *msg)
{
        assert(s->audio_tx_mode == MODE_SENDER);

        int ret;
        switch (msg->type) {
                case SENDER_MSG_CHANGE_RECEIVER:
                        ret = rtp_change_dest(s->audio_network_device,
                                        msg->receiver);

                        if (ret == FALSE) {
                                fprintf(stderr, "Changing audio receiver to: %s failed!\n",
                                                msg->receiver);
                        }

                        if (rtcp_change_dest(s->audio_network_device,
                       				msg->receiver) == FALSE){
                        		fprintf(stderr, "Changing rtcp audio receiver to: %s failed!\n",
                        	                    msg->receiver);
                        }

                        break;
                case SENDER_MSG_CHANGE_PORT:
                        rtp_done(s->audio_network_device);
                        s->audio_network_parameters.send_port = msg->port;
                        s->audio_network_device = initialize_audio_network(
                                        &s->audio_network_parameters);
                        break;
                case SENDER_MSG_PAUSE:
                        s->paused = true;
                        break;
                case SENDER_MSG_PLAY:
                        s->paused = false;
                        break;
                case SENDER_MSG_CHANGE_FEC:
                        fprintf(stderr, "Not implemented!\n");
                        abort();

        }
}

static void process_statistics(struct state_audio *s, audio_frame2 *buffer)
{
        if (!s->captured->has_same_prop_as(*buffer)) {
                s->captured->init(buffer->get_channel_count(), buffer->get_codec(),
                                buffer->get_bps(), buffer->get_sample_rate());
        }
        s->captured->append(*buffer);

        struct timeval t;
        double seconds;
        gettimeofday(&t, 0);
        seconds = tv_diff(t, s->t0);
        if (seconds > 5.0) {
                printf("[Audio sender] Sent %d samples in last %f seconds.\n",
                                s->captured->get_sample_count(),
                                seconds);
                for (int i = 0; i < s->captured->get_channel_count(); ++i) {
                        double rms, peak;
                        rms = calculate_rms(s->captured, i, &peak);
                        printf("[Audio sender] Channel %d - volume: %f dBFS RMS, %f dBFS peak.\n",
                                        i, 20 * log(rms) / log(10), 20 * log(peak) / log(10));
                }
                s->t0 = t;
                s->captured->reset();
        }
}

static void *audio_sender_thread(void *arg)
{
        struct state_audio *s = (struct state_audio *) arg;
        struct audio_frame *buffer = NULL;
        struct state_resample resample_state;

        memset(&resample_state, 0, sizeof(resample_state));
        resample_state.resample_to = s->resample_to;
        resample_state.resample_buffer = (char *) malloc(1024 * 1024);
        resample_state.codec_supported_bytes_per_sample =
                audio_codec_get_supported_bps(s->audio_coder);
        
        printf("Audio sending started.\n");
        while (!should_exit_audio) {
                struct message *msg;
                while((msg= check_message(&s->audio_sender_module))) {
                        audio_sender_process_message(s, (struct msg_sender *) msg);
                        free_message(msg);
                }

                buffer = audio_capture_read(s->audio_capture_device);
                if(buffer) {
                        audio_export(s->exporter, buffer);
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
                        audio_frame2 buffer_new;
                        if(s->sender == NET_NATIVE) {
                                // RESAMPLE
                                resample(&resample_state, buffer);
                                // COMPRESS
                                buffer_new = audio_frame2(&resample_state.resampled);
                                process_statistics(s, &buffer_new);
                                free(resample_state.resampled.data);
                                audio_frame2 *uncompressed = &buffer_new;
                                const audio_frame2 *compressed = NULL;
                                while((compressed = audio_codec_compress(s->audio_coder, uncompressed))) {
                                        audio_tx_send(s->tx_session, s->audio_network_device, compressed);
                                        uncompressed = NULL;
                                }
                        }else if(s->sender == NET_STANDARD){
                            // RESAMPLE
                            resample(&resample_state, buffer);
                            // COMPRESS
                            buffer_new = audio_frame2(&resample_state.resampled);
                            free(resample_state.resampled.data);
                            audio_frame2 *uncompressed = &buffer_new;
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

        if(resample_state.resampler) {
                speex_resampler_destroy(resample_state.resampler);
        }
        free(resample_state.resample_buffer);

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

void audio_register_put_callback(struct state_audio *s, void (*callback)(void *, struct audio_frame *),
                void *udata)
{
        struct state_sdi_playback *sdi_playback;
        if(!audio_playback_get_display_flags(s->audio_playback_device))
                return;
        
        sdi_playback = (struct state_sdi_playback *) audio_playback_get_state_pointer(s->audio_playback_device);
        sdi_register_put_callback(sdi_playback, callback, udata);
}

void audio_register_reconfigure_callback(struct state_audio *s, int (*callback)(void *, int, int,
                        int),
                void *udata)
{
        struct state_sdi_playback *sdi_playback;
        if(!audio_playback_get_display_flags(s->audio_playback_device))
                return;
        
        sdi_playback = (struct state_sdi_playback *) audio_playback_get_state_pointer(s->audio_playback_device);
        sdi_register_reconfigure_callback(sdi_playback, callback, udata);
}

unsigned int audio_get_display_flags(struct state_audio *s)
{
        return audio_playback_get_display_flags(s->audio_playback_device);
}

int audio_reconfigure(struct state_audio *s, int quant_samples, int channels,
                int sample_rate)
{
        return audio_playback_reconfigure(s->audio_playback_device, quant_samples,
                        channels, sample_rate);
}

struct audio_desc audio_desc_from_frame(struct audio_frame *frame)
{
        return (struct audio_desc) { frame->bps, frame->sample_rate,
                frame->ch_count, AC_PCM };
}

std::ostream& operator<<(std::ostream& os, const audio_desc& desc)
{
    os << desc.ch_count << " channels, " << desc.bps << " Bps, " << desc.sample_rate << " Hz, codec: " << get_name_to_audio_codec(desc.codec);
    return os;
}


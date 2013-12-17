/*
 * FILE:    main.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          David Cassany    <david.cassany@i2cat.net>
 *          Ignacio Contreras <ignacio.contreras@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "perf.h"
#include "rtp/video_decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "sender.h"
#include "stats.h"
#include "utils/wait_obj.h"
#include "video.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_export.h"
#include "pdb.h"
#include "tv.h"
#include "transmit.h"
#include "tfrc.h"
#include "ihdtv.h"
#include "compat/platform_semaphore.h"
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "rtsp/c_basicRTSPOnlyServer.h"

#if defined DEBUG && defined HAVE_LINUX
#include <mcheck.h>
#endif

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_UI   		2
#define EXIT_FAIL_DISPLAY	3
#define EXIT_FAIL_CAPTURE	4
#define EXIT_FAIL_NETWORK	5
#define EXIT_FAIL_TRANSMIT	6
#define EXIT_FAIL_COMPRESS	7
#define EXIT_FAIL_DECODER	8

#define PORT_BASE               5004
#define PORT_AUDIO              5006

/* please see comments before transmit.c:audio_tx_send() */
/* also note that this actually differs from video */
#define DEFAULT_AUDIO_FEC       "mult:3"

#define OPT_AUDIO_CHANNEL_MAP (('a' << 8) | 'm')
#define OPT_AUDIO_CAPTURE_CHANNELS (('a' << 8) | 'c')
#define OPT_AUDIO_SCALE (('a' << 8) | 's')
#define OPT_ECHO_CANCELLATION (('E' << 8) | 'C')
#define OPT_CUDA_DEVICE (('C' << 8) | 'D')
#define OPT_MCAST_IF (('M' << 8) | 'I')
#define OPT_EXPORT (('E' << 8) | 'X')
#define OPT_IMPORT (('I' << 8) | 'M')
#define OPT_AUDIO_CODEC (('A' << 8) | 'C')
#define OPT_CAPTURE_FILTER (('O' << 8) | 'F')
#define OPT_ENCRYPTION (('E' << 8) | 'N')
#define OPT_CONTROL_PORT (('C' << 8) | 'P')
#define OPT_VERBOSE (('V' << 8) | 'E')

#ifdef HAVE_MACOSX
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  5944320
#else
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100)
#endif

#define MODE_SENDER   1
#define MODE_RECEIVER 2

#define MAX_CAPTURE_COUNT 17

struct state_uv {
        int recv_port_number;
        int send_port_number;
        struct rtp **network_devices; // ULTRAGRID_RTP
        unsigned int connections_count;

        struct rx_tx *rxtx;
        void *rxtx_state;

        int mode; // MODE_SENDER, MODE_RECEIVER or both
        
        struct vidcap *capture_device;
        struct timeval start_time;
        struct pdb *participants;

        enum video_mode decoder_mode;
        char *postprocess;

        uint32_t ts;
        struct display *display_device;
        char *requested_compression;
        const char *requested_display;
        const char *requested_receiver;
        bool ipv6;
        const char *requested_mcast_if;
        unsigned requested_mtu;

        struct state_audio *audio;

        struct video_export *video_exporter;

        struct module *root_module;

        const char *requested_encryption;

        struct module receiver_mod;

        pthread_mutex_t init_lock;
};

static int exit_status = EXIT_SUCCESS;
static volatile bool should_exit_sender = false;

static struct state_uv *uv_state;

//
// prototypes
//
static void list_video_display_devices(void);
static void list_video_capture_devices(void);
static void display_buf_increase_warning(int size);
static void remove_display_from_decoders(struct state_uv *uv);
static void init_root_module(struct module *mod, struct state_uv *uv);

static void signal_handler(int signal)
{
        debug_msg("Caught signal %d\n", signal);
        exit_uv(0);
        return;
}

static void crash_signal_handler(int sig)
{
        fprintf(stderr, "\n%s has crashed", PACKAGE_NAME);
#ifndef WIN32
        fprintf(stderr, " (%s)", strsignal(sig));
#endif
        fprintf(stderr, ".\n\nPlease send a bug report to address %s.\n"
                        , PACKAGE_BUGREPORT);
        fprintf(stderr, "You may find some tips how to report bugs in file REPORTING-BUGS "
                        "distributed with %s.\n", PACKAGE_NAME);

        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        raise(sig);
}

void exit_uv(int status) {
        exit_status = status;

        should_exit_sender = true;
        should_exit_receiver = true;
        audio_finish(uv_state->audio);
}

static void usage(void)
{
        /* TODO -c -p -b are deprecated options */
        printf("\nUsage: uv [-d <display_device>] [-t <capture_device>] [-r <audio_playout>]\n");
        printf("          [-s <audio_caputre>] [-l <limit_bitrate>] "
                        "[-m <mtu>] [-c] [-i] [-6]\n");
        printf("          [-m <video_mode>] [-p <postprocess>] "
                        "[-f <fec_options>] [-P <port>]\n");
        printf("          [--mcast-if <iface>]\n");
        printf("          [--export[=<d>]|--import <d>]\n");
        printf("          address(es)\n\n");
        printf("\t--verbose                \tprint verbose messages\n");
        printf("\n");
        printf
            ("\t-d <display_device>        \tselect display device, use '-d help'\n");
        printf("\t                         \tto get list of supported devices\n");
        printf("\n");
        printf
            ("\t-t <capture_device>        \tselect capture device, use '-t help'\n");
        printf("\t                         \tto get list of supported devices\n");
        printf("\n");
        printf("\t-c <cfg>                 \tcompress video (see '-c help')\n");
        printf("\n");
        printf("\t--h264                 \t\tRTSP server: dynamically serving H264 RTP standard transport\n");
        printf("\n");
        printf("\t-i|--sage[=<opts>]       \tiHDTV compatibility mode / SAGE TX\n");
        printf("\n");
#ifdef HAVE_IPv6
        printf("\t-6                       \tUse IPv6\n");
        printf("\n");
#endif //  HAVE_IPv6
        printf("\t--mcast-if <iface>       \tBind to specified interface for multicast\n");
        printf("\n");
        printf("\t-r <playback_device>     \tAudio playback device (see '-r help')\n");
        printf("\n");
        printf("\t-s <capture_device>      \tAudio capture device (see '-s help')\n");
        printf("\n");
        printf("\t-j <settings>            \tJACK Audio Connection Kit settings\n");
        printf("\n");
        printf("\t-M <video_mode>          \treceived video mode (eg tiled-4K, 3D,\n");
        printf("\t                         \tdual-link)\n");
        printf("\n");
        printf("\t-p <postprocess>         \tpostprocess module\n");
        printf("\n");
        printf("\t-f [A:|V:]<settings>     \tFEC settings (audio or video) - use \"none\"\n"
               "\t                         \t\"mult:<nr>\",\n");
        printf("\t                         \t\"ldgm:<max_expected_loss>%%\" or\n");
        printf("\t                         \t\"ldgm:<k>:<m>:<c>\"\n");
        printf("\n");
        printf("\t-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]\n");
        printf("\t                         \t<port> is base port number, also 3 subsequent\n");
        printf("\t                         \tports can be used for RTCP and audio\n");
        printf("\t                         \tstreams. Default: %d.\n", PORT_BASE);
        printf("\t                         \tYou can also specify all two or four ports\n");
        printf("\t                         \tdirectly.\n");
        printf("\n");
        printf("\t-l <limit_bitrate> | unlimited\tlimit sending bitrate (aggregate)\n");
        printf("\t                         \tto limit_bitrate Mb/s\n");
        printf("\n");
        printf("\t--audio-channel-map      <mapping> | help\n");
        printf("\n");
        printf("\t--audio-scale            <factor> | <method> | help\n");
        printf("\n");
        printf("\t--audio-capture-channels <count> number of input channels that will\n");
        printf("\t                                 be captured (default 2).\n");
        printf("\n");
        printf("\t--echo-cancellation      \tapply acustic echo cancellation to audio\n");
        printf("\n");
        printf("\t--cuda-device <index>|help\tuse specified CUDA device\n");
        printf("\n");
        printf("\t--import <directory>     \timport previous session from directory\n");
        printf("\n");
        printf("\t--export[=<directory>]   \texport captured (and compressed) data\n");
        printf("\n");
        printf("\t-A <address>             \taudio destination address\n");
        printf("\t                         \tIf not specified, will use same as for video\n");
        printf("\t--audio-codec <codec>[:<sample_rate>]|help\taudio codec\n");
        printf("\n");
        printf("\t--capture-filter <filter>\tCapture filter(s), must preceed\n");
        printf("\n");
        printf("\t--encryption <passphrase>\tKey material for encryption\n");
        printf("\n");
        printf("\taddress(es)              \tdestination address\n");
        printf("\n");
        printf("\t                         \tIf comma-separated list of addresses\n");
        printf("\t                         \tis entered, video frames are split\n");
        printf("\t                         \tand chunks are sent/received\n");
        printf("\t                         \tindependently.\n");
        printf("\n");
}

static void list_video_display_devices()
{
        int i;
        display_type_t *dt;

        printf("Available display devices:\n");
        display_init_devices();
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                printf("\t%s\n", dt->name);
        }
        display_free_devices();
}

static void list_video_capture_devices()
{
        int i;
        struct vidcap_type *vt;

        printf("Available capture devices:\n");
        vidcap_init_devices();
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                printf("\t%s\n", vt->name);
        }
        vidcap_free_devices();
}

static void display_buf_increase_warning(int size)
{
        fprintf(stderr, "\n***\n"
                        "Unable to set buffer size to %d B.\n"
                        "Please set net.core.rmem_max value to %d or greater. (see also\n"
                        "https://www.sitola.cz/igrid/index.php/Setup_UltraGrid)\n"
#ifdef HAVE_MACOSX
                        "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
                        "\tsysctl -w net.inet.udp.recvspace=%d\n"
#else
                        "\tsysctl -w net.core.rmem_max=%d\n"
#endif
                        "To make this persistent, add these options (key=value) to /etc/sysctl.conf\n"
                        "\n***\n\n",
                        size, size,
#ifdef HAVE_MACOSX
                        size * 4,
#endif /* HAVE_MACOSX */
                        size);

}

static struct rtp **initialize_network(const char *addrs, int recv_port_base,
                int send_port_base, struct pdb *participants, bool use_ipv6,
                const char *mcast_if)
{
	struct rtp **devices = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
	int ttl = 255;
	char *saveptr = NULL;
	char *addr;
	char *tmp;
	int required_connections, index;
        int recv_port = recv_port_base;
        int send_port = send_port_base;

	tmp = strdup(addrs);
	if(strtok_r(tmp, ",", &saveptr) == NULL) {
		free(tmp);
		return NULL;
	}
	else required_connections = 1;
	while(strtok_r(NULL, ",", &saveptr) != NULL)
		++required_connections;

	free(tmp);
	tmp = strdup(addrs);

	devices = (struct rtp **)
		malloc((required_connections + 1) * sizeof(struct rtp *));

	for(index = 0, addr = strtok_r(tmp, ",", &saveptr);
		index < required_connections;
		++index, addr = strtok_r(NULL, ",", &saveptr), recv_port += 2, send_port += 2)
	{
                /* port + 2 is reserved for audio */
                if (recv_port == recv_port_base + 2)
                        recv_port += 2;
                if (send_port == send_port_base + 2)
                        send_port += 2;

		devices[index] = rtp_init_if(addr, mcast_if, recv_port,
                                send_port, ttl, rtcp_bw, FALSE,
                                rtp_recv_callback, (void *)participants,
                                use_ipv6);
		if (devices[index] != NULL) {
			rtp_set_option(devices[index], RTP_OPT_WEAK_VALIDATION,
				TRUE);
			rtp_set_sdes(devices[index], rtp_my_ssrc(devices[index]),
				RTCP_SDES_TOOL,
				PACKAGE_STRING, strlen(PACKAGE_STRING));

                        int size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
                        int ret = rtp_set_recv_buf(devices[index], INITIAL_VIDEO_RECV_BUFFER_SIZE);
                        if(!ret) {
                                display_buf_increase_warning(size);
                        }

                        rtp_set_send_buf(devices[index], 1024 * 56);

			pdb_add(participants, rtp_my_ssrc(devices[index]));
		}
		else {
			int index_nest;
			for(index_nest = 0; index_nest < index; ++index_nest) {
				rtp_done(devices[index_nest]);
			}
			free(devices);
			devices = NULL;
		}
	}
	if(devices != NULL) devices[index] = NULL;
	free(tmp);

        return devices;
}

void destroy_rtp_devices(struct rtp ** network_devices)
{
	struct rtp ** current = network_devices;
        if(!network_devices)
                return;
	while(*current != NULL) {
		rtp_done(*current++);
	}
	free(network_devices);
}

static struct vcodec_state *new_video_decoder(struct state_uv *uv) {
        struct vcodec_state *state = calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->decoder = video_decoder_init(&uv->receiver_mod, uv->decoder_mode,
                                uv->postprocess, uv->display_device,
                                uv->requested_encryption);

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

static void destroy_video_decoder(void *state) {
        struct vcodec_state *video_decoder_state = state;

        if(!video_decoder_state) {
                return;
        }

        video_decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

/**
 * Removes display from decoders and effectively kills them. They cannot be used
 * until new display assigned.
 */
static void remove_display_from_decoders(struct state_uv *uv) {
        if (uv->participants != NULL) {
                pdb_iter_t it;
                struct pdb_e *cp = pdb_iter_init(uv->participants, &it);
                while (cp != NULL) {
                        if(cp->decoder_state)
                                video_decoder_remove_display(
                                                ((struct vcodec_state*) cp->decoder_state)->decoder);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
}

static void receiver_process_messages(struct state_uv *uv, struct module *receiver_mod)
{
        struct msg_receiver *msg;
        while ((msg = (struct msg_receiver *) check_message(receiver_mod))) {
                switch (msg->type) {
                case RECEIVER_MSG_CHANGE_RX_PORT:
                        assert(uv->mode == MODE_RECEIVER); // receiver only
                        destroy_rtp_devices(uv->network_devices);
                        uv->recv_port_number = msg->new_rx_port;
                        uv->network_devices = initialize_network(uv->requested_receiver, uv->recv_port_number,
                                        uv->send_port_number, uv->participants, uv->ipv6,
                                        uv->requested_mcast_if);
                        if (!uv->network_devices) {
                                fprintf(stderr, "Changing RX port failed!\n");
                                abort();
                        }
                        break;
                case RECEIVER_MSG_VIDEO_PROP_CHANGED:
                        {
                                pdb_iter_t it;
                                /// @todo should be set only to relevant participant, not all
                                struct pdb_e *cp = pdb_iter_init(uv->participants, &it);
                                while (cp) {
                                        pbuf_set_playout_delay(cp->playout_buffer,
                                                        1.0 / msg->new_desc.fps,
                                                        1.0 / msg->new_desc.fps *
                                                        (is_codec_interframe(msg->new_desc.color_spec) ? 2.2 : 1.2)
                                                        );

                                        cp = pdb_iter_next(&it);
                                }
                        }
                        break;
                }

                free_message((struct message *) msg);
        }
}

struct rtp **change_tx_port(struct state_uv *uv, int tx_port)
{
        destroy_rtp_devices(uv->network_devices);
        uv->send_port_number = tx_port;
        uv->network_devices = initialize_network(uv->requested_receiver, uv->recv_port_number,
                        uv->send_port_number, uv->participants, uv->ipv6,
                        uv->requested_mcast_if);
        if (!uv->network_devices) {
                fprintf(stderr, "Changing RX port failed!\n");
                abort();
        }
        return uv->network_devices;
}

void *ultragrid_rtp_receiver_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;

        struct pdb_e *cp;
        struct timeval curr_time;
        int fr;
        int ret;
        unsigned int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_decoder(uv);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER

        initialize_video_decompress();

        fr = 1;

        struct module *control_mod = get_module(get_root_module(uv->root_module), "control");
        unlock_module(control_mod);
        struct stats *stat_loss = stats_new_statistics(
                        (struct control_state *) control_mod,
                        "loss");
        struct stats *stat_received = stats_new_statistics(
                        (struct control_state *) control_mod,
                        "received");
        uint64_t total_received = 0ull;

        while (!should_exit_receiver) {
                struct timeval timeout;
                /* Housekeeping and RTCP... */
                gettimeofday(&curr_time, NULL);
                uv->ts = tv_diff(curr_time, uv->start_time) * 90000;
                rtp_update(uv->network_devices[0], curr_time);
                rtp_send_ctrl(uv->network_devices[0], uv->ts, 0, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&curr_time, NULL);
                        receiver_process_messages(uv, &uv->receiver_mod);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                timeout.tv_usec = 10000;
                ret = rtp_recv_poll_r(uv->network_devices, &timeout, uv->ts);

                // timeout
                if (ret == FALSE) {
                        // processing is needed here in case we are not receiving any data
                        receiver_process_messages(uv, &uv->receiver_mod);
                        //printf("Failed to receive data\n");
                }
                total_received += ret;
                stats_update_int(stat_received, total_received);

                /* Decode and render for each participant in the conference... */
                pdb_iter_t it;
                cp = pdb_iter_init(uv->participants, &it);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               curr_time));
                        }

                        if(cp->decoder_state == NULL) {
#ifdef SHARED_DECODER
                                cp->decoder_state = shared_decoder;
#else
                                cp->decoder_state = new_video_decoder(uv);
                                cp->decoder_state_deleter = destroy_video_decoder;
#endif // SHARED_DECODER
                                if(cp->decoder_state == NULL) {
                                        fprintf(stderr, "Fatal: unable to find decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
                                ((struct vcodec_state*) cp->decoder_state)->display = uv->display_device;
                        }

                        struct vcodec_state *vdecoder_state = cp->decoder_state;

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_video_frame, vdecoder_state)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == uv->connections_count)
                                {
                                        tiles_post = 0;
                                        gettimeofday(&curr_time, NULL);
                                        fr = 1;
#if 0
                                        display_put_frame(uv->display_device,
                                                          cp->video_decoder_state->frame_buffer);
                                        cp->video_decoder_state->frame_buffer =
                                            display_get_frame(uv->display_device);
#endif
                                }
                                last_tile_received = curr_time;
                                uint32_t sender_ssrc = cp->ssrc;
                                stats_update_int(stat_loss,
                                                rtp_compute_fract_lost(uv->network_devices[0],
                                                        sender_ssrc));
                        }

                        /* dual-link TIMEOUT - we won't wait for next tiles */
                        if(tiles_post > 1 && tv_diff(curr_time, last_tile_received) >
                                        999999 / 59.94 / uv->connections_count) {
                                tiles_post = 0;
                                gettimeofday(&curr_time, NULL);
                                fr = 1;
#if 0
                                display_put_frame(uv->display_device,
                                                cp->video_decoder_state->frame_buffer);
                                cp->video_decoder_state->frame_buffer =
                                        display_get_frame(uv->display_device);
#endif
                                last_tile_received = curr_time;
                        }

                        if(vdecoder_state->decoded % 100 == 99) {
                                int new_size = vdecoder_state->max_frame_size * 110ull / 100;
                                if(new_size > last_buf_size) {
                                        struct rtp **device = uv->network_devices;
                                        while(*device) {
                                                int ret = rtp_set_recv_buf(*device, new_size);
                                                if(!ret) {
                                                        display_buf_increase_warning(new_size);
                                                }
                                                debug_msg("Recv buffer adjusted to %d\n", new_size);
                                                device++;
                                        }
                                }
                                last_buf_size = new_size;
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }

        module_done(&uv->receiver_mod);

#ifdef SHARED_DECODER
        destroy_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders(uv);
#endif //  SHARED_DECODER

        // pass posioned pill to display
        display_put_frame(uv->display_device, NULL, PUTF_BLOCKING);

        stats_destroy(stat_loss);
        stats_destroy(stat_received);

        return 0;
}

static void uncompressed_frame_dispose(struct video_frame *frame)
{
        struct wait_obj *wait_obj = (struct wait_obj *) frame->dispose_udata;
        wait_obj_notify(wait_obj);
}

/**
 * This function captures video and possibly compresses it.
 * It then delegates sending to another thread.
 *
 * @param[in] arg pointer to UltraGrid (root) module
 */
static void *capture_thread(void *arg)
{
        struct module *uv_mod = (struct module *)arg;
        struct state_uv *uv = (struct state_uv *) uv_mod->priv_data;
        struct sender_data sender_data;
        memset(&sender_data, 0, sizeof(sender_data));

        struct compress_state *compression = NULL;
        int ret = compress_init(uv_mod, uv->requested_compression, &compression);
        if(ret != 0) {
                if(ret < 0) {
                        fprintf(stderr, "Error initializing compression.\n");
                        exit_uv(1);
                }
                if(ret > 0) {
                        exit_uv(0);
                }
                pthread_mutex_unlock(&uv->init_lock);
                goto compress_done;
        }

        sender_data.parent = uv_mod; /// @todo should be compress thread module
        sender_data.rxtx_protocol = uv->rxtx->protocol;
        sender_data.tx_module_state = uv->rxtx_state;
        sender_data.send_frame = uv->rxtx->send;
        sender_data.uv = uv;
        sender_data.video_exporter = uv->video_exporter;
        sender_data.compression = compression;

        struct wait_obj *wait_obj = wait_obj_init();

        if(!sender_init(&sender_data)) {
                fprintf(stderr, "Error initializing sender.\n");
                exit_uv(1);
                pthread_mutex_unlock(&uv->init_lock);
                goto compress_done;
        }

        pthread_mutex_unlock(&uv->init_lock);

        while (!should_exit_sender) {
                /* Capture and transmit video... */
                struct audio_frame *audio;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);
                void (*old_dispose)(struct video_frame *) = NULL;
                void *old_udata = NULL;
                if (tx_frame != NULL) {
                        if(audio) {
                                audio_sdi_send(uv->audio, audio);
                        }
                        //tx_frame = vf_get_copy(tx_frame);
                        old_dispose = tx_frame->dispose;
                        old_udata = tx_frame->dispose_udata;
                        tx_frame->dispose = uncompressed_frame_dispose;
                        tx_frame->dispose_udata = wait_obj;
                        wait_obj_reset(wait_obj);

                        // Sends frame to compression - this passes it to a sender thread
                        compress_frame(compression, tx_frame);

                        // wait to frame is processed - eg by compress or sender (uncompressed video)
                        wait_obj_wait(wait_obj);
                        tx_frame->dispose = old_dispose;
                        tx_frame->dispose_udata = old_udata;
                }
        }

        compress_frame(compression, NULL); // pass poisoned pill (will go through to the sender)
        sender_done(&sender_data);
        wait_obj_done(wait_obj);

compress_done:
        module_done(CAST_MODULE(compression));

        return NULL;
}

static bool enable_export(const char *dir)
{
        if(!dir) {
                for (int i = 1; i <= 9999; i++) {
                        char name[16];
                        snprintf(name, 16, "export.%04d", i);
                        int ret = platform_mkdir(name);
                        if(ret == -1) {
                                if(errno == EEXIST) {
                                        continue;
                                } else {
                                        fprintf(stderr, "[Export] Directory creation failed: %s\n",
                                                        strerror(errno));
                                        return false;
                                }
                        } else {
                                export_dir = strdup(name);
                                break;
                        }
                }
        } else {
                int ret = platform_mkdir(dir);
                if(ret == -1) {
                                if(errno == EEXIST) {
                                        fprintf(stderr, "[Export] Warning: directory %s exists!\n", dir);
                                } else {
                                        perror("[Export] Directory creation failed");
                                        return false;
                                }
                }

                export_dir = strdup(dir);
        }

        if(export_dir) {
                printf("Using export directory: %s\n", export_dir);
                return true;
        } else {
                return false;
        }
}

static void init_root_module(struct module *mod, struct state_uv *uv)
{
        module_init_default(mod);
        mod->cls = MODULE_CLASS_ROOT;
        mod->parent = NULL;
        mod->deleter = NULL;
        mod->priv_data = uv;
}

int main(int argc, char *argv[])
{
#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        // NULL terminated array of capture devices
        struct vidcap_params *vidcap_params = vidcap_params_allocate();
        struct vidcap_params *vidcap_params_tail = vidcap_params;

        char *display_cfg = NULL;
        const char *audio_recv = "none";
        const char *audio_send = "none";
        char *jack_cfg = NULL;
        char *requested_video_fec = strdup("none");
        char *requested_audio_fec = strdup(DEFAULT_AUDIO_FEC);
        char *audio_channel_map = NULL;
        char *audio_scale = "mixauto";

        bool echo_cancellation = false;

        bool should_export = false;
        char *export_opts = NULL;

        char *sage_opts = NULL;
        int control_port = CONTROL_DEFAULT_PORT;
        struct control_state *control = NULL;
        rtsp_serv_t* rtsp_server = NULL;

        int bitrate = 0;

        const char *audio_host = NULL;
        int audio_rx_port = -1, audio_tx_port = -1;

        struct module root_mod;
        struct state_uv *uv;
        int ch;

        audio_codec_t audio_codec = AC_PCM;

        pthread_t receiver_thread_id,
                  tx_thread_id;
	bool receiver_thread_started = false,
		  tx_thread_started = false;
        unsigned vidcap_flags = 0,
                 display_flags = 0;
        int compressed_audio_sample_rate = 48000;
        int ret;

#if defined DEBUG && defined HAVE_LINUX
        mtrace();
#endif

        vidcap_params_assign_device(vidcap_params, "none");

        if (argc == 1) {
                usage();
                return EXIT_FAIL_USAGE;
        }

        uv_argc = argc;
        uv_argv = argv;

        static struct option getopt_options[] = {
                {"display", required_argument, 0, 'd'},
                {"capture", required_argument, 0, 't'},
                {"mtu", required_argument, 0, 'm'},
                {"ipv6", no_argument, 0, '6'},
                {"mode", required_argument, 0, 'M'},
                {"version", no_argument, 0, 'v'},
                {"compress", required_argument, 0, 'c'},
                {"ihdtv", no_argument, 0, 'i'},
                {"sage", optional_argument, 0, 'S'},
                {"h264", no_argument, 0, 'H'},
                {"receive", required_argument, 0, 'r'},
                {"send", required_argument, 0, 's'},
                {"help", no_argument, 0, 'h'},
                {"jack", required_argument, 0, 'j'},
                {"fec", required_argument, 0, 'f'},
                {"port", required_argument, 0, 'P'},
                {"limit-bitrate", required_argument, 0, 'l'},
                {"audio-channel-map", required_argument, 0, OPT_AUDIO_CHANNEL_MAP},
                {"audio-scale", required_argument, 0, OPT_AUDIO_SCALE},
                {"audio-capture-channels", required_argument, 0, OPT_AUDIO_CAPTURE_CHANNELS},
                {"echo-cancellation", no_argument, 0, OPT_ECHO_CANCELLATION},
                {"cuda-device", required_argument, 0, OPT_CUDA_DEVICE},
                {"mcast-if", required_argument, 0, OPT_MCAST_IF},
                {"export", optional_argument, 0, OPT_EXPORT},
                {"import", required_argument, 0, OPT_IMPORT},
                {"audio-host", required_argument, 0, 'A'},
                {"audio-codec", required_argument, 0, OPT_AUDIO_CODEC},
                {"capture-filter", required_argument, 0, OPT_CAPTURE_FILTER},
                {"control-port", required_argument, 0, OPT_CONTROL_PORT},
                {"encryption", required_argument, 0, OPT_ENCRYPTION},
                {"verbose", no_argument, 0, OPT_VERBOSE},
                {0, 0, 0, 0}
        };
        int option_index = 0;

        //      uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv_state = uv;

        uv->audio = NULL;
        uv->ts = 0;
        uv->capture_device = NULL;
        uv->display_device = NULL;
        uv->requested_display = "none";
        uv->requested_compression = "none";
        uv->decoder_mode = VIDEO_NORMAL;
        uv->postprocess = NULL;
        uv->requested_mtu = 0;
        uv->participants = NULL;
        uv->network_devices = NULL;
        uv->video_exporter = NULL;
        uv->recv_port_number =
                uv->send_port_number =
                PORT_BASE;

        init_root_module(&root_mod, uv);
        uv->root_module = &root_mod;
        uv->rxtx = &ultragrid_rtp; // default

        perf_init();
        perf_record(UVP_INIT, 0);

        while ((ch =
                getopt_long(argc, argv, "d:t:m:r:s:v6c:ihj:M:p:f:P:l:A:", getopt_options,
                            &option_index)) != -1) {
                switch (ch) {
                case 'd':
                        if (!strcmp(optarg, "help")) {
                                list_video_display_devices();
                                return 0;
                        }
                        uv->requested_display = optarg;
			if(strchr(optarg, ':')) {
				char *delim = strchr(optarg, ':');
				*delim = '\0';
				display_cfg = delim + 1;
			}
                        break;
                case 't':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        vidcap_params_assign_device(vidcap_params_tail, optarg);
                        vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        break;
                case 'm':
                        uv->requested_mtu = atoi(optarg);
                        break;
                case 'M':
                        uv->decoder_mode = get_video_mode_from_str(optarg);
                        if (uv->decoder_mode == VIDEO_UNKNOWN) {
                                return strcasecmp(optarg, "help") == 0 ? EXIT_SUCCESS : EXIT_FAIL_USAGE;
                        }
                        break;
                case 'p':
                        uv->postprocess = optarg;
                        break;
                case 'v':
                        printf("%s", PACKAGE_STRING);
#ifdef GIT_VERSION
                        printf(" (rev %s)", GIT_VERSION);
#endif
                        printf("\n");
                        printf("\n" PACKAGE_NAME " was compiled with following features:\n");
                        printf(AUTOCONF_RESULT);
                        return EXIT_SUCCESS;
                case 'c':
                        uv->requested_compression = optarg;
                        break;
                case 'i':
#ifdef HAVE_IHDTV
                        uv->rxtx = &ihdtv_rxtx;
                        printf("setting ihdtv protocol\n");
                        fprintf(stderr, "Warning: iHDTV support may be currently broken.\n"
                                        "Please contact %s if you need this.\n", PACKAGE_BUGREPORT);
#else
                        fprintf(stderr, "iHDTV support isn't compiled in this %s build\n",
                                        PACKAGE_NAME);
#endif
                        break;
                case 'S':
                        uv->rxtx = &sage_rxtx;
                        sage_opts = optarg;
                        break;
                case 'H':
                        uv->rxtx = &h264_rtp;
                        //h264_opts = optarg;
                        break;
                case 'r':
                        audio_recv = optarg;                       
                        break;
                case 's':
                        audio_send = optarg;
                        break;
                case 'j':
                        jack_cfg = optarg;
                        break;
                case 'f':
                        if(strlen(optarg) > 2 && optarg[1] == ':' &&
                                        (toupper(optarg[0]) == 'A' || toupper(optarg[0]) == 'V')) {
                                if(toupper(optarg[0]) == 'A') {
                                        free(requested_audio_fec);
                                        requested_audio_fec = strdup(optarg + 2);
                                } else {
                                        free(requested_audio_fec);
                                        requested_audio_fec = strdup(optarg + 2);
                                }
                        } else {
                                // there should be setting for both audio and video
                                // but we conservativelly expect that the user wants
                                // only vieo and let audio default until explicitly
                                // stated otehrwise
                                free(requested_video_fec);
                                requested_video_fec = strdup(optarg);
                        }
                        break;
		case 'h':
			usage();
			return 0;
                case 'P':
                        if(strchr(optarg, ':')) {
                                char *save_ptr = NULL;
                                char *tok;
                                uv->recv_port_number = atoi(strtok_r(optarg, ":", &save_ptr));
                                uv->send_port_number = atoi(strtok_r(NULL, ":", &save_ptr));
                                if((tok = strtok_r(NULL, ":", &save_ptr))) {
                                        audio_rx_port = atoi(tok);
                                        if((tok = strtok_r(NULL, ":", &save_ptr))) {
                                                audio_tx_port = atoi(tok);
                                        } else {
                                                usage();
                                                return EXIT_FAIL_USAGE;
                                        }
                                }
                        } else {
                                uv->recv_port_number =
                                        uv->send_port_number =
                                        atoi(optarg);
                        }
                        break;
                case 'l':
                        if(strcmp(optarg, "unlimited") == 0) {
                                bitrate = -1;
                        } else {
                                bitrate = atoi(optarg);
                                if(bitrate <= 0) {
                                        usage();
                                        return EXIT_FAIL_USAGE;
                                }
                        }
                        break;
                case '6':
                        uv->ipv6 = true;
                        break;
                case OPT_AUDIO_CHANNEL_MAP:
                        audio_channel_map = optarg;
                        break;
                case OPT_AUDIO_SCALE:
                        audio_scale = optarg;
                        break;
                case OPT_AUDIO_CAPTURE_CHANNELS:
                        audio_capture_channels = atoi(optarg);
                        break;
                case OPT_ECHO_CANCELLATION:
                        echo_cancellation = true;
                        break;
                case OPT_CUDA_DEVICE:
#ifdef HAVE_JPEG
                        if(strcmp("help", optarg) == 0) {
                                struct compress_state *compression;
                                int ret = compress_init(&root_mod, "JPEG:list_devices", &compression);
                                if(ret >= 0) {
                                        if(ret == 0) {
                                                module_done(CAST_MODULE(compression));
                                        }
                                        return EXIT_SUCCESS;
                                } else {
                                        return EXIT_FAILURE;
                                }
                        } else {
                                char *item, *save_ptr = NULL;
                                unsigned int i = 0;
                                while((item = strtok_r(optarg, ",", &save_ptr))) {
                                        if(i >= MAX_CUDA_DEVICES) {
                                                fprintf(stderr, "Maximal number of CUDA device exceeded.\n");
                                                return EXIT_FAILURE;
                                        }
                                        cuda_devices[i] = atoi(item);
                                        optarg = NULL;
                                        ++i;
                                }
                                cuda_devices_count = i;
                        }
                        break;
#else
                        fprintf(stderr, "CUDA support is not enabled!\n");
                        return EXIT_FAIL_USAGE;
#endif // HAVE_CUDA
                case OPT_MCAST_IF:
                        uv->requested_mcast_if = optarg;
                        break;
                case 'A':
                        audio_host = optarg;
                        break;
                case OPT_EXPORT:
                        should_export = true;
                        export_opts = optarg;
                        break;
                case OPT_IMPORT:
                        audio_send = strdup("embedded");
                        {
                                char dev_string[1024];
                                snprintf(dev_string, sizeof(dev_string), "import:%s", optarg);
                                vidcap_params_assign_device(vidcap_params_tail, dev_string);
                                vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        }
                        break;
                case OPT_AUDIO_CODEC:
                        if(strcmp(optarg, "help") == 0) {
                                list_audio_codecs();
                                return EXIT_SUCCESS;
                        }
                        if(strchr(optarg, ':')) {
                                compressed_audio_sample_rate = atoi(strchr(optarg, ':')+1);
                                *strchr(optarg, ':') = '\0';
                        }
                        audio_codec = get_audio_codec_to_name(optarg);
                        if(audio_codec == AC_NONE) {
                                fprintf(stderr, "Unknown audio codec entered: \"%s\"\n",
                                                optarg);
                                return EXIT_FAIL_USAGE;
                        }
                        break;
                case OPT_CAPTURE_FILTER:
                        vidcap_params_assign_capture_filter(vidcap_params_tail, optarg);
                        break;
                case OPT_ENCRYPTION:
                        uv->requested_encryption = optarg;
                        break;
                case OPT_CONTROL_PORT:
                        control_port = atoi(optarg);
                        break;
                case OPT_VERBOSE:
                        verbose = true;
                        break;
                case '?':
                default:
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }

        argc -= optind;
        argv += optind;

        if (uv->requested_mtu == 0)     // mtu wasn't specified on the command line
        {
                uv->requested_mtu = 1500;       // the default value for RTP
        }

        printf("%s", PACKAGE_STRING);
#ifdef GIT_VERSION
        printf(" (rev %s)", GIT_VERSION);
#endif
        printf("\n");
        printf("Display device   : %s\n", uv->requested_display);
        printf("Capture device   : %s\n", vidcap_params_get_driver(vidcap_params));
        printf("Audio capture    : %s\n", audio_send);
        printf("Audio playback   : %s\n", audio_recv);
        printf("MTU              : %d B\n", uv->requested_mtu);
        printf("Video compression: %s\n", uv->requested_compression);
        printf("Audio codec      : %s\n", get_name_to_audio_codec(audio_codec));
        printf("Network protocol : %s\n", uv->rxtx->name);
        printf("Audio FEC        : %s\n", requested_audio_fec);
        printf("Video FEC        : %s\n", requested_video_fec);
        printf("\n");

        if(audio_rx_port == -1) {
                audio_tx_port = uv->send_port_number + 2;
                audio_rx_port = uv->recv_port_number + 2;
        }

        if(should_export) {
                if(!enable_export(export_opts)) {
                        fprintf(stderr, "Export initialization failed.\n");
                        return EXIT_FAILURE;
                }
                uv->video_exporter = video_export_init(export_dir);
        }

        gettimeofday(&uv->start_time, NULL);

        if(uv->requested_mtu > RTP_MAX_PACKET_LEN) {
                fprintf(stderr, "Requested MTU exceeds maximal value allowed by RTP library (%d).\n",
                                RTP_MAX_PACKET_LEN);
                return EXIT_FAIL_USAGE;
        }

        if (argc == 0) {
                uv->requested_receiver = "localhost";
        } else {
                uv->requested_receiver = argv[0];
        }

#ifdef WIN32
	WSADATA wsaData;
	int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if(err != 0) {
		fprintf(stderr, "WSAStartup failed with error %d.", err);
		return EXIT_FAILURE;
	}
	if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
		fprintf(stderr, "Counld not found usable version of Winsock.\n");
		WSACleanup();
		return EXIT_FAILURE;
	}
#endif

        if(control_init(control_port, &control, &root_mod) != 0) {
                fprintf(stderr, "%s Unable to initialize remote control!\n",
                                control_port != CONTROL_DEFAULT_PORT ? "Warning:" : "Error:");
                if(control_port != CONTROL_DEFAULT_PORT) {
                        return EXIT_FAILURE;
                }
        }

        if(!audio_host) {
                audio_host = uv->requested_receiver;
        }
        uv->audio = audio_cfg_init (&root_mod, audio_host, audio_rx_port,
                        audio_tx_port, audio_send, audio_recv,
                        jack_cfg, requested_audio_fec, uv->requested_encryption,
                        audio_channel_map,
                        audio_scale, echo_cancellation, uv->ipv6, uv->requested_mcast_if,
                        audio_codec, compressed_audio_sample_rate);
        free(requested_audio_fec);
        if(!uv->audio)
                goto cleanup;

        vidcap_flags |= audio_get_vidcap_flags(uv->audio);
        display_flags |= audio_get_display_flags(uv->audio);

        uv->participants = pdb_init();

        // Display initialization should be prior to modules that may use graphic card (eg. GLSL) in order
        // to initalize shared resource (X display) first
        ret =
             initialize_video_display(uv->requested_display, display_cfg, display_flags, &uv->display_device);
        if (ret < 0) {
                printf("Unable to open display device: %s\n",
                       uv->requested_display);
                exit_uv(EXIT_FAIL_DISPLAY);
                goto cleanup;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }

        printf("Display initialized-%s\n", uv->requested_display);

        ret = initialize_video_capture(&root_mod, vidcap_params, &uv->capture_device);
        if (ret < 0) {
                printf("Unable to open capture device: %s\n",
                                vidcap_params_get_driver(vidcap_params));
                exit_uv(EXIT_FAIL_CAPTURE);
                goto cleanup;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }
        printf("Video capture initialized-%s\n", vidcap_params_get_driver(vidcap_params));

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
#ifndef WIN32
        signal(SIGHUP, signal_handler);
#endif
        signal(SIGABRT, crash_signal_handler);
        signal(SIGSEGV, crash_signal_handler);

#ifdef USE_RT
#ifdef HAVE_SCHED_SETSCHEDULER
        sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
                printf("WARNING: Unable to set real-time scheduling\n");
        }
#else
        printf("WARNING: System does not support real-time scheduling\n");
#endif /* HAVE_SCHED_SETSCHEDULER */
#endif /* USE_RT */

        if (strcmp("none", uv->requested_display) != 0) {
                uv->mode |= MODE_RECEIVER;
        }
        if (strcmp("none", vidcap_params_get_driver(vidcap_params)) != 0) {
                uv->mode |= MODE_SENDER;
        }

        struct ultragrid_rtp_state ug_rtp;
        struct sage_rxtx_state sage_rxtx;
        struct h264_rtp_state h264_rtp;

        if (uv->rxtx->protocol == IHDTV) {
                struct vidcap *capture_device = NULL;
                struct display *display_device = NULL;
                if (uv->mode & MODE_SENDER)
                        capture_device = uv->capture_device;
                if (uv->mode & MODE_RECEIVER)
                        display_device = uv->display_device;
                uv->rxtx_state = initialize_ihdtv(capture_device,
                                display_device, uv->requested_mtu,
                                argc, argv);
                if(!uv->rxtx_state) {
                        usage();
                        return EXIT_FAILURE;
                }
        }else if (uv->rxtx->protocol == H264_STD) {
                if ((uv->network_devices = initialize_network(uv->requested_receiver,
                    uv->recv_port_number, uv->send_port_number, uv->participants,
                    uv->ipv6, uv->requested_mcast_if)) == NULL)
                {
                        printf("Unable to open network\n");
                        exit_uv(EXIT_FAIL_NETWORK);
                        goto cleanup;
                } else {
                        struct rtp **item;
                        uv->connections_count = 0;
                        /* only count how many connections has initialize_network opened */
                        for(item = uv->network_devices; *item != NULL; ++item){
                                ++uv->connections_count;
                        //#ifdef HAVE_RTSP_SERVER
                                rtsp_server = init_rtsp_server(0, &root_mod); //port, root_module
                                c_start_server(rtsp_server);
                        //#endif
                        }
                }

                if(bitrate == 0) { // else packet_rate defaults to 13600 or so
                        bitrate = 6618;
                }

                if(bitrate != -1) {
                        packet_rate = 1000 * uv->requested_mtu * 8 / bitrate;
                } else {
                        packet_rate = 0;
                }

                if ((h264_rtp.tx = tx_init(&root_mod,
                                                uv->requested_mtu, TX_MEDIA_VIDEO,
                                                NULL,
                                                NULL)) == NULL) {
                        printf("Unable to initialize transmitter.\n");
                        exit_uv(EXIT_FAIL_TRANSMIT);
                        goto cleanup;
                }

                h264_rtp.connections_count = uv->connections_count;
                h264_rtp.network_devices = uv->network_devices;

                uv->rxtx_state = &h264_rtp;
                free(requested_video_fec);
        } else if(uv->rxtx->protocol == ULTRAGRID_RTP) {
                if ((uv->network_devices =
                                        initialize_network(uv->requested_receiver, uv->recv_port_number,
                                                uv->send_port_number, uv->participants, uv->ipv6,
                                                uv->requested_mcast_if))
                                == NULL) {
                        printf("Unable to open network\n");
                        exit_uv(EXIT_FAIL_NETWORK);
                        goto cleanup;
                } else {
                        struct rtp **item;
                        uv->connections_count = 0;
                        /* only count how many connections has initialize_network opened */
                        for(item = uv->network_devices; *item != NULL; ++item)
                                ++uv->connections_count;
                }

                if(bitrate == 0) { // else packet_rate defaults to 13600 or so
                        bitrate = 6618;
                }

                if(bitrate != -1) {
                        packet_rate = 1000 * uv->requested_mtu * 8 / bitrate;
                } else {
                        packet_rate = 0;
                }

                if ((ug_rtp.tx = tx_init(&root_mod,
                                                uv->requested_mtu, TX_MEDIA_VIDEO,
                                                requested_video_fec,
                                                uv->requested_encryption)) == NULL) {
                        printf("Unable to initialize transmitter.\n");
                        exit_uv(EXIT_FAIL_TRANSMIT);
                        goto cleanup;
                }

                ug_rtp.connections_count = uv->connections_count;
                ug_rtp.network_devices = uv->network_devices;

                uv->rxtx_state = &ug_rtp;
                free(requested_video_fec);
        } else { // SAGE
                memset(&sage_rxtx, 0, sizeof(sage_rxtx));
                sage_receiver = uv->requested_receiver;
                ret = initialize_video_display("sage",
                                sage_opts, 0, &sage_rxtx.sage_tx_device);
                if(ret != 0) {
                        fprintf(stderr, "Unable to initialize SAGE TX.\n");
                        exit_uv(EXIT_FAIL_NETWORK);
                        goto cleanup;
                }
                pthread_create(&sage_rxtx.thread_id, NULL, (void * (*)(void *)) display_run,
                                &sage_rxtx.sage_tx_device);
        }

        /* following block only shows help (otherwise initialized in receiver thread */
        if((uv->postprocess && strstr(uv->postprocess, "help") != NULL)) {
                struct state_video_decoder *dec = video_decoder_init(NULL, uv->decoder_mode,
                                uv->postprocess, NULL,
                                uv->requested_encryption);
                video_decoder_destroy(dec);
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }
        /* following block only shows help (otherwise initialized in sender thread */
        if(strstr(uv->requested_compression,"help") != NULL) {
                struct compress_state *compression;
                int ret = compress_init(&root_mod, uv->requested_compression, &compression);

                if(ret >= 0) {
                        if(ret == 0)
                                module_done(CAST_MODULE(compression));
                        exit_uv(EXIT_SUCCESS);
                } else {
                        exit_uv(EXIT_FAILURE);
                }
                goto cleanup;
        }

        if(uv->mode & MODE_RECEIVER) {
                if (uv->rxtx->receiver_thread == NULL) {
                        fprintf(stderr, "Selected RX/TX mode doesn't support receiving.\n");
                        exit_uv(EXIT_FAILURE);
                        goto cleanup;
                }
               // init module here so as it is capable of receiving messages
               module_init_default(&uv->receiver_mod);
               uv->receiver_mod.cls = MODULE_CLASS_RECEIVER;
               module_register(&uv->receiver_mod, uv->root_module);
                if (pthread_create
                                (&receiver_thread_id, NULL, uv->rxtx->receiver_thread,
                                 (void *)uv) != 0) {
                        perror("Unable to create display thread!\n");
                        exit_uv(EXIT_FAILURE);
                        goto cleanup;
                } else {
                        receiver_thread_started = true;
                }
        }

        if(uv->mode & MODE_SENDER) {
                pthread_mutex_lock(&uv->init_lock);
                if (pthread_create
                                (&tx_thread_id, NULL, capture_thread,
                                 (void *) &root_mod) != 0) {
                        perror("Unable to create capture thread!\n");
						pthread_mutex_unlock(&uv->init_lock);
                        exit_uv(EXIT_FAILURE);
                        goto cleanup;
                } else {
                        // wait for sender module initialization
                        pthread_mutex_lock(&uv->init_lock);
                        pthread_mutex_unlock(&uv->init_lock);
                        tx_thread_started = true;
                }
        }

        if(audio_get_display_flags(uv->audio)) {
                audio_register_put_callback(uv->audio, (void (*)(void *, struct audio_frame *)) display_put_audio_frame, uv->display_device);
                audio_register_reconfigure_callback(uv->audio, (int (*)(void *, int, int,
                                                        int)) display_reconfigure_audio, uv->display_device);
        }

        // should be started after requested modules are able to respond after start
        control_start(control);

        if (strcmp("none", uv->requested_display) != 0)
                display_run(uv->display_device);

cleanup:
        if (strcmp("none", uv->requested_display) != 0 &&
                        receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

                ;
        if (uv->mode & MODE_SENDER
                        && tx_thread_started)
                pthread_join(tx_thread_id, NULL);

        /* also wait for audio threads */
        audio_join(uv->audio);

        control_done(control);

        if(uv->audio)
                audio_done(uv->audio);
        if (uv->rxtx_state)
                uv->rxtx->done(uv->rxtx_state);
        if(uv->capture_device)
                vidcap_done(uv->capture_device);
        if(uv->display_device)
                display_done(uv->display_device);
        if (uv->network_devices) {
                destroy_rtp_devices(uv->network_devices);
        }
        if (uv->participants != NULL) {
                pdb_iter_t it;
                struct pdb_e *cp = pdb_iter_init(uv->participants, &it);
                while (cp != NULL) {
                        struct pdb_e *item = NULL;
                        pdb_remove(uv->participants, cp->ssrc, &item);
                        cp = pdb_iter_next(&it);
                        free(item);
                }
                pdb_iter_done(&it);
                pdb_destroy(&uv->participants);
        }

        video_export_destroy(uv->video_exporter);

        free(export_dir);

        while  (vidcap_params) {
                struct vidcap_params *next = vidcap_params_get_next(vidcap_params);
                vidcap_params_free_struct(vidcap_params);
                vidcap_params = next;
        }

        if(rtsp_server) c_stop_server(rtsp_server);

        module_done(&root_mod);
        pthread_mutex_destroy(&uv->init_lock);
        free(uv);

#if defined DEBUG && defined HAVE_LINUX
        muntrace();
#endif

#ifdef WIN32
	WSACleanup();
#endif

        printf("Exit\n");

        return exit_status;
}


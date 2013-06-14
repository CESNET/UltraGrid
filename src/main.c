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
 *
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
#include "capture_filter.h"
#include "control.h"
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "perf.h"
#include "rtp/decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "sender.h"
#include "stats.h"
#include "video_codec.h"
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
#include "ihdtv/ihdtv.h"
#include "lib_common.h"
#include "compat/platform_semaphore.h"
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"

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

#ifdef HAVE_MACOSX
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  5944320
#else
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100)
#endif

struct state_uv {
        int recv_port_number;
        int send_port_number;
        union {
                struct rtp **network_devices; // ULTRAGRID_RTP
                struct display *sage_tx_device; // == SAGE
        };
        unsigned int connections_count;
        
        struct vidcap *capture_device;
        struct capture_filter *capture_filter;
        struct timeval start_time, curr_time;
        struct pdb *participants;
        
        char *decoder_mode;
        char *postprocess;
        
        uint32_t ts;
        struct tx *tx;
        struct display *display_device;
        char *requested_compression;
        const char *requested_display;
        const char *requested_capture;
        unsigned requested_mtu;
        
        enum tx_protocol tx_protocol;

        struct state_audio *audio;

        /* used mainly to serialize initialization */
        pthread_mutex_t master_lock;

        struct video_export *video_exporter;

        struct sender_data sender_data;

        struct module *root_module;

        const char *requested_encryption;
};

static volatile int wait_to_finish = FALSE;
static volatile int threads_joined = FALSE;
static int exit_status = EXIT_SUCCESS;
static bool should_exit_receiver = false;
static bool should_exit_sender = false;

static struct state_uv *uv_state;
#ifdef HAVE_IHDTV
static struct video_frame *frame_buffer = NULL;
static long frame_begin[2];
#endif

//
// prototypes
//
static struct rtp **initialize_network(char *addrs, int recv_port_base,
                int send_port_base, struct pdb *participants, bool use_ipv6,
                char *mcast_if);

static void list_video_display_devices(void);
static void list_video_capture_devices(void);
static void display_buf_increase_warning(int size);
static bool enable_export(char *dir);
static void remove_display_from_decoders(struct state_uv *uv);
static void init_root_module(struct module *mod, struct state_uv *uv);

static void signal_handler(int signal)
{
        debug_msg("Caught signal %d\n", signal);
        exit_uv(0);
        return;
}

static void _exit_uv(int status);

static void _exit_uv(int status) {
        exit_status = status;
        wait_to_finish = TRUE;
        if(!threads_joined) {
                if(uv_state->capture_device) {
                        should_exit_sender = true;
                }
                if(uv_state->display_device) {
                        should_exit_receiver = true;
                }
                if(uv_state->audio)
                        audio_finish(uv_state->audio);
        }
        wait_to_finish = FALSE;
}

void (*exit_uv)(int status) = _exit_uv;

static void usage(void)
{
        /* TODO -c -p -b are deprecated options */
        printf("\nUsage: uv [-d <display_device>] [-t <capture_device>] [-r <audio_playout>]\n");
        printf("          [-s <audio_caputre>] [-l <limit_bitrate>] "
                        "[-m <mtu>] [-c] [-i] [-6]\n");
        printf("          [-m <video_mode>] [-p <postprocess>] "
                        "[-f <fec_options>] [-p <port>]\n");
        printf("          [--mcast-if <iface>]\n");
        printf("          [--export[=<d>]|--import <d>]\n");
        printf("          address(es)\n\n");
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
        printf("\t--capture-filter <filter>\tCapture filter(s)\n");
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

static struct rtp **initialize_network(char *addrs, int recv_port_base,
                int send_port_base, struct pdb *participants, bool use_ipv6,
                char *mcast_if)
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

	for(index = 0, addr = strtok_r(addrs, ",", &saveptr); 
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

static void destroy_devices(struct rtp ** network_devices)
{
	struct rtp ** current = network_devices;
        if(!network_devices)
                return;
	while(*current != NULL) {
		rtp_done(*current++);
	}
	free(network_devices);
}

#ifdef HAVE_IHDTV
static void *ihdtv_receiver_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection *) ((void **)arg)[0];
        struct display *display_device = (struct display *)((void **)arg)[1];

        while (1) {
                if (ihdtv_receive
                    (connection, frame_buffer->tiles[0].data, frame_buffer->tiles[0].data_len))
                        return 0;       // we've got some error. probably empty buffer
                display_put_frame(display_device, frame_buffer);
                frame_buffer = display_get_frame(display_device);
        }
        return 0;
}

static void *ihdtv_sender_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection *) ((void **)arg)[0];
        struct vidcap *capture_device = (struct vidcap *)((void **)arg)[1];
        struct video_frame *tx_frame;
        struct audio_frame *audio;

        while (1) {
                if ((tx_frame = vidcap_grab(capture_device, &audio)) != NULL) {
                        ihdtv_send(connection, tx_frame, 9000000);      // FIXME: fix the use of frame size!!
                        free(tx_frame);
                } else {
                        fprintf(stderr,
                                "Error receiving frame from capture device\n");
                        return 0;
                }
        }

        return 0;
}
#endif // IHDTV

static struct vcodec_state *new_decoder(struct state_uv *uv) {
        struct vcodec_state *state = calloc(1, sizeof(struct vcodec_state));

        if(state) {
                state->messages = simple_linked_list_init();
                state->decoder = decoder_init(uv->decoder_mode, uv->postprocess, uv->display_device,
                                uv->requested_encryption);

                if(!state->decoder) {
                        fprintf(stderr, "Error initializing decoder (incorrect '-M' or '-p' option?).\n");
                        free(state);
                        exit_uv(1);
                        return NULL;
                } else {
                        //decoder_register_video_display(state->decoder, uv->display_device);
                }
        }

        return state;
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
                        if(cp->video_decoder_state)
                                decoder_remove_display(cp->video_decoder_state->decoder);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
}

static void *receiver_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;

        struct pdb_e *cp;
        struct timeval timeout;
        int fr;
        int ret;
        unsigned int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
        struct stats *stat_loss = NULL;
#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_decoder(uv);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER

        initialize_video_decompress();

        pthread_mutex_unlock(&uv->master_lock);

        fr = 1;

        struct module *control_mod = get_module(get_root_module(uv->root_module), "control");
        stat_loss = stats_new_statistics(
                        (struct control_state *) control_mod,
                        "loss");
        unlock_module(control_mod);

        while (!should_exit_receiver) {
                /* Housekeeping and RTCP... */
                gettimeofday(&uv->curr_time, NULL);
                uv->ts = tv_diff(uv->curr_time, uv->start_time) * 90000;
                rtp_update(uv->network_devices[0], uv->curr_time);
                rtp_send_ctrl(uv->network_devices[0], uv->ts, 0, uv->curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&uv->curr_time, NULL);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                timeout.tv_usec = 10000;
                ret = rtp_recv_poll_r(uv->network_devices, &timeout, uv->ts);

		/*
                   if (ret == FALSE) {
                   printf("Failed to receive data\n");
                   }
                 */
                UNUSED(ret);

                /* Decode and render for each participant in the conference... */
                pdb_iter_t it;
                cp = pdb_iter_init(uv->participants, &it);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, uv->curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               uv->curr_time));
                        }

                        if(cp->video_decoder_state == NULL) {
#ifdef SHARED_DECODER
                                cp->video_decoder_state = shared_decoder;
#else
                                cp->video_decoder_state = new_decoder(uv);
#endif // SHARED_DECODER
                                if(cp->video_decoder_state == NULL) {
                                        fprintf(stderr, "Fatal: unable to find decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
                                cp->video_decoder_state->display = uv->display_device;
                        }

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, uv->curr_time, decode_frame, cp->video_decoder_state)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == uv->connections_count) 
                                {
                                        tiles_post = 0;
                                        gettimeofday(&uv->curr_time, NULL);
                                        fr = 1;
#if 0
                                        display_put_frame(uv->display_device,
                                                          cp->video_decoder_state->frame_buffer);
                                        cp->video_decoder_state->frame_buffer =
                                            display_get_frame(uv->display_device);
#endif
                                }
                                last_tile_received = uv->curr_time;
                                uint32_t sender_ssrc = cp->ssrc;
                                stats_update_int(stat_loss,
                                                rtp_compute_fract_lost(uv->network_devices[0],
                                                        sender_ssrc));
                        }

                        /* dual-link TIMEOUT - we won't wait for next tiles */
                        if(tiles_post > 1 && tv_diff(uv->curr_time, last_tile_received) > 
                                        999999 / 59.94 / uv->connections_count) {
                                tiles_post = 0;
                                gettimeofday(&uv->curr_time, NULL);
                                fr = 1;
#if 0
                                display_put_frame(uv->display_device,
                                                cp->video_decoder_state->frame_buffer);
                                cp->video_decoder_state->frame_buffer =
                                        display_get_frame(uv->display_device);
#endif
                                last_tile_received = uv->curr_time;
                        }

                        if(cp->video_decoder_state->decoded % 100 == 99) {
                                int new_size = cp->video_decoder_state->max_frame_size * 110ull / 100;
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

                        while(simple_linked_list_size(cp->video_decoder_state->messages) > 0) {
                                struct vcodec_message *msg =
                                        simple_linked_list_pop(cp->video_decoder_state->messages);

                                assert(msg->type == FPS_CHANGED);
                                struct fps_changed_message *data = msg->data;

                                pbuf_set_playout_delay(cp->playout_buffer,
                                                1.0 / data->val,
                                                1.0 / data->val * (data->interframe_codec ? 2.2 : 1.2)
                                                );
                                free(data);
                                free(msg);
                        }


                        pbuf_remove(cp->playout_buffer, uv->curr_time);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
        
#ifdef SHARED_DECODER
        destroy_decoder(shared_decoder);
#else
        /* Because decoders work asynchronously we need to make sure
         * that display won't be called */
        remove_display_from_decoders(uv);
#endif //  SHARED_DECODER

        display_finish(uv_state->display_device);

        stats_destroy(stat_loss);

        return 0;
}

static void *compress_thread(void *arg)
{
        struct module *uv_mod = (struct module *)arg;
        struct state_uv *uv = (struct state_uv *) uv_mod->priv_data;

        struct video_frame *tx_frame;
        struct audio_frame *audio;
        //struct video_frame *splitted_frames = NULL;
        int i = 0;

        struct compress_state *compression;
        int ret = compress_init(uv_mod, uv->requested_compression, &compression);

        uv->sender_data.parent = uv_mod; /// @todo should be compress thread module
        uv->sender_data.connections_count = uv->connections_count;
        uv->sender_data.tx_protocol = uv->tx_protocol;
        if(uv->tx_protocol == ULTRAGRID_RTP) {
                uv->sender_data.network_devices = uv->network_devices;
        } else {
                uv->sender_data.sage_tx_device = uv->sage_tx_device;
        }
        uv->sender_data.tx = uv->tx;

        if(!sender_init(&uv->sender_data)) {
                fprintf(stderr, "Error initializing sender.\n");
                exit_uv(1);
                pthread_mutex_unlock(&uv->master_lock);
                goto compress_done;
        }

        pthread_mutex_unlock(&uv->master_lock);
        /* NOTE: unlock before propagating possible error */
        if(ret != 0) {
                if(ret < 0) {
                        fprintf(stderr, "Error initializing compression.\n");
                        exit_uv(1);
                }
                if(ret > 0) {
                        exit_uv(0);
                }
                goto compress_done;
        }

        while (!should_exit_sender) {
                /* Capture and transmit video... */
                tx_frame = vidcap_grab(uv->capture_device, &audio);
                if(tx_frame) {
                        tx_frame = capture_filter(uv->capture_filter, tx_frame);

                }
                if (tx_frame != NULL) {
                        if(audio) {
                                audio_sdi_send(uv->audio, audio);
                        }
                        //TODO: Unghetto this
                        tx_frame = compress_frame(compression, tx_frame, i);
                        if(!tx_frame)
                                continue;

                        i = (i + 1) % 2;

                        video_export(uv->video_exporter, tx_frame);

                        bool nonblock = true;
                        /* when sending uncompressed video, we simply post it for send
                         * and wait until done */
                        /* for compressed, we do not need to wait */
                        if(is_compress_none(compression)) {
                                nonblock = false;
                        }

                        sender_post_new_frame(&uv->sender_data, tx_frame, nonblock);
                }
        }

        vidcap_finish(uv_state->capture_device);

        sender_done(&uv->sender_data);

compress_done:
        module_done(CAST_MODULE(compression));

        return NULL;
}

static bool enable_export(char *dir)
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
        char *network_device = NULL;
        char *capture_cfg = NULL;
        char *display_cfg = NULL;
        const char *audio_recv = "none";
        const char *audio_send = "none";
        char *jack_cfg = NULL;
        char *requested_video_fec = strdup("none");
        char *requested_audio_fec = strdup(DEFAULT_AUDIO_FEC);
        char *audio_channel_map = NULL;
        char *audio_scale = "mixauto";

        bool echo_cancellation = false;
        bool use_ipv6 = false;
        char *mcast_if = NULL;

        bool should_export = false;
        char *export_opts = NULL;

        char *sage_opts = NULL;
        struct control_state *control = NULL;

        int bitrate = 0;
        
        char *audio_host = NULL;
        int audio_rx_port = -1, audio_tx_port = -1;

        struct module root_mod;
        struct state_uv *uv;
        int ch;

        char *requested_capture_filter = NULL;

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
                {"encryption", required_argument, 0, OPT_ENCRYPTION},
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
        uv->requested_capture = "none";
        uv->requested_compression = "none";
        uv->decoder_mode = NULL;
        uv->postprocess = NULL;
        uv->requested_mtu = 0;
        uv->tx_protocol = ULTRAGRID_RTP;
        uv->participants = NULL;
        uv->tx = NULL;
        uv->network_devices = NULL;
        uv->video_exporter = NULL;
        uv->recv_port_number =
                uv->send_port_number =
                PORT_BASE;
        uv->sage_tx_device = NULL;

        init_root_module(&root_mod, uv);
        uv->root_module = &root_mod;

        pthread_mutex_init(&uv->master_lock, NULL);

        perf_init();
        perf_record(UVP_INIT, 0);

        init_lib_common();

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
                        uv->requested_capture = optarg;
			if(strchr(optarg, ':')) {
				char *delim = strchr(optarg, ':');
				*delim = '\0';
				capture_cfg = delim + 1;
			}
                        break;
                case 'm':
                        uv->requested_mtu = atoi(optarg);
                        break;
                case 'M':
                        uv->decoder_mode = optarg;
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
                        uv->tx_protocol = IHDTV;
                        printf("setting ihdtv protocol\n");
                        fprintf(stderr, "Warning: iHDTV support may be currently broken.\n"
                                        "Please contact %s if you need this.\n", PACKAGE_BUGREPORT);
#else
                        fprintf(stderr, "iHDTV support isn't compiled in this %s build\n",
                                        PACKAGE_NAME);
#endif
                        break;
                case 'S':
                        uv->tx_protocol = SAGE;
                        sage_opts = optarg;
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
                                }
                                if((tok = strtok_r(NULL, ":", &save_ptr))) {
                                        audio_tx_port = atoi(tok);
                                } else {
                                        usage();
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
                        use_ipv6 = true;
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
#ifdef HAVE_CUDA
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
                        mcast_if = optarg;
                        break;
                case 'A':
                        audio_host = optarg;
                        break;
                case OPT_EXPORT:
                        should_export = true;
                        export_opts = optarg;
                        break;
                case OPT_IMPORT:
                        uv->requested_capture = "import";
                        audio_send = strdup("embedded");
                        capture_cfg = optarg;
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
                        requested_capture_filter = optarg;
                        break;
                case OPT_ENCRYPTION:
                        uv->requested_encryption = optarg;
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
        printf("Capture device   : %s\n", uv->requested_capture);
        printf("Audio capture    : %s\n", audio_send);
        printf("Audio playback   : %s\n", audio_recv);
        printf("MTU              : %d B\n", uv->requested_mtu);
        printf("Video compression: %s\n", uv->requested_compression);
        printf("Audio codec      : %s\n", get_name_to_audio_codec(audio_codec));

        printf("Network protocol : ");
        switch(uv->tx_protocol) {
                case ULTRAGRID_RTP:
                        printf("UltraGrid RTP\n"); break;
                case IHDTV:
                        printf("iHDTV\n"); break;
                case SAGE:
                        printf("SAGE\n"); break;
        }

        printf("Audio FEC        : %s\n", requested_audio_fec);
        printf("Video FEC        : %s\n", requested_video_fec);
        printf("\n");

        if(audio_rx_port == -1) {
                audio_tx_port = uv->send_port_number + 2;
                audio_rx_port = uv->recv_port_number + 2;
        }

        if(control_init(CONTROL_DEFAULT_PORT, &control, &root_mod) != 0) {
                fprintf(stderr, "Warning: Unable to initialize remote control!\n:");
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

        if (uv->tx_protocol == IHDTV) {
                if ((argc != 0) && (argc != 1) && (argc != 2)) {
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        } else {
                if (argc == 0) {
                        network_device = strdup("localhost");
                } else {
                        network_device = (char *) argv[0];
                }
                if(uv->tx_protocol == SAGE) {
                        sage_network_device = network_device;
                }
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

        ret = capture_filter_init(requested_capture_filter, &uv->capture_filter);
        if(ret != 0) {
                goto cleanup;
        }
	
        if(!audio_host) {
                audio_host = network_device;
        }
        uv->audio = audio_cfg_init (&root_mod, audio_host, audio_rx_port,
                        audio_tx_port, audio_send, audio_recv,
                        jack_cfg, requested_audio_fec, uv->requested_encryption,
                        audio_channel_map,
                        audio_scale, echo_cancellation, use_ipv6, mcast_if,
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
                goto cleanup_wait_audio;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup_wait_audio;
        }

        printf("Display initialized-%s\n", uv->requested_display);

        ret = initialize_video_capture(uv->requested_capture, capture_cfg, vidcap_flags, &uv->capture_device);
        if (ret < 0) {
                printf("Unable to open capture device: %s\n",
                       uv->requested_capture);
                exit_uv(EXIT_FAIL_CAPTURE);
                goto cleanup_wait_audio;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup_wait_audio;
        }
        printf("Video capture initialized-%s\n", uv->requested_capture);

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
#ifndef WIN32
        signal(SIGHUP, signal_handler);
#endif
        signal(SIGABRT, signal_handler);

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

        if (uv->tx_protocol == IHDTV) {
#ifdef HAVE_IHDTV
                ihdtv_connection tx_connection, rx_connection;

                printf("Initializing ihdtv protocol\n");

                // we cannot act as both together, because parameter parsing would have to be revamped
                if ((strcmp("none", uv->requested_display) != 0)
                    && (strcmp("none", uv->requested_capture) != 0)) {
                        printf
                            ("Error: cannot act as both sender and receiver together in ihdtv mode\n");
                        return -1;
                }

                void *rx_connection_and_display[] =
                    { (void *)&rx_connection, (void *)uv->display_device };
                void *tx_connection_and_display[] =
                    { (void *)&tx_connection, (void *)uv->capture_device };

                if (uv->requested_mtu == 0)     // mtu not specified on command line
                {
                        uv->requested_mtu = 8112;       // not really a mtu, but a video-data payload per packet
                }

                if (strcmp("none", uv->requested_display) != 0) {
                        if (ihdtv_init_rx_session
                            (&rx_connection, (argc == 0) ? NULL : argv[0],
                             (argc ==
                              0) ? NULL : ((argc == 1) ? argv[0] : argv[1]),
                             3000, 3001, uv->requested_mtu) != 0) {
                                fprintf(stderr,
                                        "Error initializing receiver session\n");
                                return 1;
                        }

                        if (pthread_create
                            (&receiver_thread_id, NULL, ihdtv_receiver_thread,
                             rx_connection_and_display) != 0) {
                                fprintf(stderr,
                                        "Error creating receiver thread. Quitting\n");
                                return 1;
                        } else {
				receiver_thread_started = true;
			}
                }

                if (strcmp("none", uv->requested_capture) != 0) {
                        if (argc == 0) {
                                fprintf(stderr,
                                        "Error: specify the destination address\n");
                                usage();
                                return EXIT_FAIL_USAGE;
                        }

                        if (ihdtv_init_tx_session
                            (&tx_connection, argv[0],
                             (argc == 2) ? argv[1] : argv[0],
                             uv->requested_mtu) != 0) {
                                fprintf(stderr,
                                        "Error initializing sender session\n");
                                return 1;
                        }

                        if (pthread_create
                            (&tx_thread_id, NULL, ihdtv_sender_thread,
                             tx_connection_and_display) != 0) {
                                fprintf(stderr,
                                        "Error creating sender thread. Quitting\n");
                                return 1;
                        } else {
				tx_thread_started = true;
			}
                }

                while (!0) // was 'should_exit'
                        sleep(1);
#endif // HAVE_IHDTV
        } else if(uv->tx_protocol == ULTRAGRID_RTP) {
                if ((uv->network_devices =
                                        initialize_network(network_device, uv->recv_port_number,
                                                uv->send_port_number, uv->participants, use_ipv6, mcast_if))
                                == NULL) {
                        printf("Unable to open network\n");
                        exit_uv(EXIT_FAIL_NETWORK);
                        goto cleanup_wait_display;
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

                if ((uv->tx = tx_init(&root_mod,
                                                uv->requested_mtu, TX_MEDIA_VIDEO,
                                                requested_video_fec,
                                                uv->requested_encryption)) == NULL) {
                        printf("Unable to initialize transmitter.\n");
                        exit_uv(EXIT_FAIL_TRANSMIT);
                        goto cleanup_wait_display;
                }
                free(requested_video_fec);
        } else { // SAGE
                ret = initialize_video_display("sage",
                                sage_opts, 0, &uv->sage_tx_device);
                if(ret != 0) {
                        fprintf(stderr, "Unable to initialize SAGE TX.\n");
                        exit_uv(EXIT_FAIL_NETWORK);
                        goto cleanup_wait_display;
                }
                pthread_create(&tx_thread_id, NULL, (void * (*)(void *)) display_run,
                                uv->sage_tx_device);
                tx_thread_started = true;
        }

        if(uv->tx_protocol == ULTRAGRID_RTP || uv->tx_protocol == SAGE) {
                /* following block only shows help (otherwise initialized in receiver thread */
                if((uv->postprocess && strstr(uv->postprocess, "help") != NULL) || 
                                (uv->decoder_mode && strstr(uv->decoder_mode, "help") != NULL)) {
                        struct state_decoder *dec = decoder_init(uv->decoder_mode, uv->postprocess, NULL,
                                        uv->requested_encryption);
                        decoder_destroy(dec);
                        exit_uv(EXIT_SUCCESS);
                        goto cleanup_wait_display;
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
                        goto cleanup_wait_display;
                }

                if (strcmp("none", uv->requested_display) != 0) {
                        pthread_mutex_lock(&uv->master_lock); 
                        if (pthread_create
                            (&receiver_thread_id, NULL, receiver_thread,
                             (void *)uv) != 0) {
                                perror("Unable to create display thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup_wait_display;
                        } else {
				receiver_thread_started = true;
			}
                }

                if (strcmp("none", uv->requested_capture) != 0) {
                        pthread_mutex_lock(&uv->master_lock); 
                        if (pthread_create
                            (&tx_thread_id, NULL, compress_thread,
                             (void *) &root_mod) != 0) {
                                perror("Unable to create capture thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup_wait_capture;
                        } else {
				tx_thread_started = true;
			}
                }
        }
        
        pthread_mutex_lock(&uv->master_lock); 

        if(audio_get_display_flags(uv->audio)) {
                audio_register_put_callback(uv->audio, (void (*)(void *, struct audio_frame *)) display_put_audio_frame, uv->display_device);
                audio_register_reconfigure_callback(uv->audio, (int (*)(void *, int, int, 
                                                        int)) display_reconfigure_audio, uv->display_device);
        }

        if (strcmp("none", uv->requested_display) != 0)
                display_run(uv->display_device);

cleanup_wait_display:
        if (strcmp("none", uv->requested_display) != 0 && receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

cleanup_wait_capture:
        if(uv->tx_protocol == SAGE && uv->sage_tx_device) {
                display_finish(uv->sage_tx_device);
        }
        if (strcmp("none", uv->requested_capture) != 0 &&
                         tx_thread_started)
                pthread_join(tx_thread_id, NULL);

cleanup_wait_audio:
        /* also wait for audio threads */
        audio_join(uv->audio);

cleanup:
        while(wait_to_finish)
                ;
        threads_joined = TRUE;

        if(uv->tx_protocol == SAGE && uv->sage_tx_device)
                display_done(uv->sage_tx_device);
        if(uv->audio)
                audio_done(uv->audio);
        if(uv->tx)
                module_done(CAST_MODULE(uv->tx));
	if(uv->tx_protocol == ULTRAGRID_RTP && uv->network_devices)
                destroy_devices(uv->network_devices);
        if(uv->capture_device)
                vidcap_done(uv->capture_device);
        if(uv->display_device)
                display_done(uv->display_device);
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

        pthread_mutex_unlock(&uv->master_lock); 

        pthread_mutex_destroy(&uv->master_lock);

        free(uv);
        free(export_dir);
        
        lib_common_done();

#if defined DEBUG && defined HAVE_LINUX
        muntrace();
#endif

#ifdef WIN32
	WSACleanup();
#endif

        control_done(control);

        module_done(&root_mod);

        printf("Exit\n");

        return exit_status;
}

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

#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "rtp/decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "pdb.h"
#include "tv.h"
#include "transmit.h"
#include "tfrc.h"
#include "ihdtv/ihdtv.h"
#include "compat/platform_semaphore.h"
#include "audio/audio.h"

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

struct state_uv {
        int port_number;
        struct rtp **network_devices;
        unsigned int connections_count;
        
        struct vidcap *capture_device;
        struct timeval start_time, curr_time;
        struct pdb *participants;
        
        char *decoder_mode;
        char *postprocess;
        
        uint32_t ts;
        struct tx *tx;
        struct display *display_device;
        char *compress_options;
        int requested_compression;
        const char *requested_display;
        const char *requested_capture;
        unsigned requested_mtu;
        
        int use_ihdtv_protocol;

        struct state_audio *audio;
};

long packet_rate = 13600;
volatile int should_exit = FALSE;
volatile int wait_to_finish = FALSE;
volatile int threads_joined = FALSE;
static int exit_status = EXIT_SUCCESS;

uint32_t RTT = 0;               /* this is computed by handle_rr in rtp_callback */
struct video_frame *frame_buffer = NULL;
uint32_t hd_color_spc = 0;

long frame_begin[2];

int uv_argc;
char **uv_argv;
static struct state_uv *uv_state;

void list_video_display_devices(void);
void list_video_capture_devices(void);
struct display *initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags);
struct vidcap *initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags);

#ifndef WIN32
static void signal_handler(int signal)
{
        debug_msg("Caught signal %d\n", signal);
        exit_uv(0);
        return;
}
#endif                          /* WIN32 */

void _exit_uv(int status);

void _exit_uv(int status) {
        exit_status = status;
        wait_to_finish = TRUE;
        should_exit = TRUE;
        if(!threads_joined) {
                if(uv_state->capture_device)
                        vidcap_finish(uv_state->capture_device);
                if(uv_state->display_device)
                        display_finish(uv_state->display_device);
                if(uv_state->audio)
                        audio_finish(uv_state->audio);
        }
        wait_to_finish = FALSE;
}

void (*exit_uv)(int status) = _exit_uv;

static void usage(void)
{
        /* TODO -c -p -b are deprecated options */
        printf("\nUsage: uv [-d <display_device>] [-t <capture_device>] [-r <audio_playout>] [-s <audio_caputre>] \n");
        printf("          [-m <mtu>] [-c] [-i] [-M <video_mode>] [-p <postprocess>] [-f <FEC_options>] [-P <port>] address(es)\n\n");
        printf
            ("\t-d <display_device>        \tselect display device, use '-d help' to get\n");
        printf("\t                         \tlist of supported devices\n");
        printf("\n");
        printf
            ("\t-t <capture_device>        \tselect capture device, use '-t help' to get\n");
        printf("\t                         \tlist of supported devices\n");
        printf("\n");
        printf("\t-c <cfg>                 \tcompress video (see '-c help')\n");
        printf("\n");
        printf("\t-i                       \tiHDTV compatibility mode\n");
        printf("\n");
        printf("\t-r <playback_device>     \tAudio playback device (see '-r help')\n");
        printf("\n");
        printf("\t-s <capture_device>      \tAudio capture device (see '-s help')\n");
        printf("\n");
        printf("\t-j <settings>            \tJACK Audio Connection Kit settings (see '-j help')\n");
        printf("\n");
        printf("\t-M <video_mode>          \treceived video mode (eg tiled-4K, 3D, dual-link)\n");
        printf("\n");
        printf("\t-p <postprocess>         \tpostprocess module\n");
        printf("\n");
        printf("\t-f <settings>            \tconfig forward error checking, currently \"XOR:<leap>:<nr_streams>\" or \"mult:<nr>\"\n");
        printf("\n");
        printf("\t-P <port>                \tbase port number, also 3 subsequent ports can be used (default: %d)\n", PORT_BASE);
        printf("\n");
        printf("\taddress(es)              \tdestination address\n");
        printf("\n");
        printf("\t                         \tIf comma-separated list of addresses\n");
        printf("\t                         \tis entered, video frames are split\n");
        printf("\t                         \tand chunks are sent/received independently.\n");
        printf("\n");
}

void list_video_display_devices()
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

struct display *initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags)
{
        struct display *d;
        display_type_t *dt;
        display_id_t id = 0;
        int i;
        
        if(!strcmp(requested_display, "none"))
                 id = display_get_null_device_id();

        if (display_init_devices() != 0) {
                printf("Unable to initialise devices\n");
                abort();
        } else {
                debug_msg("Found %d display devices\n",
                          display_get_device_count());
        }
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                if (strcmp(requested_display, dt->name) == 0) {
                        id = dt->id;
                        debug_msg("Found device\n");
                        break;
                } else {
                        debug_msg("Device %s does not match %s\n", dt->name,
                                  requested_display);
                }
        }
        if(i == display_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
                return NULL;
        }
        display_free_devices();

        d = display_init(id, fmt, flags);
        return d;
}

void list_video_capture_devices()
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

struct vidcap *initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags)
{
        struct vidcap_type *vt;
        vidcap_id_t id = 0;
        int i;
        
        if(!strcmp(requested_capture, "none"))
                id = vidcap_get_null_device_id();

        vidcap_init_devices();
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                if (strcmp(vt->name, requested_capture) == 0) {
                        id = vt->id;
                        break;
                }
        }
        if(i == vidcap_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' capture card "
                        "was not found.\n", requested_capture);
                return NULL;
        }
        vidcap_free_devices();

        return vidcap_init(id, fmt, flags);
}

static struct rtp **initialize_network(char *addrs, int port_base, struct pdb *participants)
{
	struct rtp **devices = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
	int ttl = 255;
	char *saveptr = NULL;
	char *addr;
	char *tmp;
	int required_connections, index;
        int port = port_base;

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
		++index, addr = strtok_r(NULL, ",", &saveptr), port += 2)
	{
                if (port == PORT_AUDIO)
                        port += 2;
		devices[index] = rtp_init(addr, port, port, ttl, rtcp_bw, 
                                FALSE, rtp_recv_callback, 
                                (void *)participants);
		if (devices[index] != NULL) {
			rtp_set_option(devices[index], RTP_OPT_WEAK_VALIDATION, 
				TRUE);
			rtp_set_sdes(devices[index], rtp_my_ssrc(devices[index]),
				RTCP_SDES_TOOL,
				PACKAGE_STRING, strlen(PACKAGE_STRING));
#ifdef HAVE_MACOSX
                        rtp_set_recv_buf(devices[index], 5944320);
#else
                        rtp_set_recv_buf(devices[index], 8*1024*1024);
#endif

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

static struct tx *initialize_transmit(unsigned requested_mtu, char *fec)
{
        /* Currently this is trivial. It'll get more complex once we */
        /* have multiple codecs and/or error correction.             */
        return tx_init(requested_mtu, fec);
}

static void *ihdtv_reciever_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection *) ((void **)arg)[0];
        struct display *display_device = (struct display *)((void **)arg)[1];

        while (!should_exit) {
                if (ihdtv_recieve
                    (connection, frame_buffer->tiles[0].data, frame_buffer->tiles[0].data_len))
                        return 0;       // we've got some error. probably empty buffer
                display_put_frame(display_device, frame_buffer->tiles[0].data);
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

        while (!should_exit) {
                if ((tx_frame = vidcap_grab(capture_device, &audio)) != NULL) {
                        ihdtv_send(connection, tx_frame, 9000000);      // FIXME: fix the use of frame size!!
                        free(tx_frame);
                } else {
                        fprintf(stderr,
                                "Error recieving frame from capture device\n");
                        return 0;
                }
        }

        return 0;
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
        struct pbuf_video_data pbuf_data;


        initialize_video_decompress();
        pbuf_data.decoder = decoder_init(uv->decoder_mode, uv->postprocess);
        if(!pbuf_data.decoder) {
                fprintf(stderr, "Error initializing decoder ('-M' option).\n");
                exit_uv(1);
        } else {
                decoder_register_video_display(pbuf_data.decoder, uv->display_device);
        }
        pbuf_data.frame_buffer = frame_buffer;

        fr = 1;

        while (!should_exit) {
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
                timeout.tv_usec = 999999 / 59.94;
                ret = rtp_recv_poll_r(uv->network_devices, &timeout, uv->ts);

                /*
                   if (ret == FALSE) {
                   printf("Failed to receive data\n");
                   }
                 */

                /* Decode and render for each participant in the conference... */
                cp = pdb_iter_init(uv->participants);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, uv->curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               uv->curr_time));
                        }

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, uv->curr_time, decode_frame, &pbuf_data, TRUE)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == uv->connections_count) 
                                {
                                        tiles_post = 0;
                                        gettimeofday(&uv->curr_time, NULL);
                                        fr = 1;
                                        display_put_frame(uv->display_device,
                                                          (char *) pbuf_data.frame_buffer);
                                        pbuf_data.frame_buffer =
                                            display_get_frame(uv->display_device);
                                }
                                last_tile_received = uv->curr_time;
                        }
                        pbuf_remove(cp->playout_buffer, uv->curr_time);
                        cp = pdb_iter_next(uv->participants);
                }
                pdb_iter_done(uv->participants);

                /* dual-link TIMEOUT - we won't wait for next tiles */
                if(tiles_post > 1 && tv_diff(uv->curr_time, last_tile_received) > 
                                999999 / 59.94 / uv->connections_count) {
                        tiles_post = 0;
                        gettimeofday(&uv->curr_time, NULL);
                        fr = 1;
                        display_put_frame(uv->display_device,
                                          pbuf_data.frame_buffer->tiles[0].data);
                        pbuf_data.frame_buffer =
                            display_get_frame(uv->display_device);
                        last_tile_received = uv->curr_time;
                }
        }
        
        decoder_destroy(pbuf_data.decoder);

        return 0;
}

static void *sender_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;

        struct video_frame *tx_frame, *splitted_frames = NULL;
        struct audio_frame *audio;
        //struct video_frame *splitted_frames = NULL;
        int tile_y_count;
        
        struct compress_state *compression; 
        compression = compress_init(uv->compress_options);
        if(uv->requested_compression
                        && compression == NULL) {
                fprintf(stderr, "Error initializing compression.\n");
                exit_uv(0);
        }

        tile_y_count = uv->connections_count;

        /* we have more than one connection */
        if(tile_y_count > 1) {
                /* it is simply stripping frame */
                splitted_frames = vf_alloc(tile_y_count);
        }

        while (!should_exit) {
                /* Capture and transmit video... */
                tx_frame = vidcap_grab(uv->capture_device, &audio);
                if (tx_frame != NULL) {
                        if(audio) {
                                audio_sdi_send(uv->audio, audio);
                        }
                        //TODO: Unghetto this
                        if (uv->requested_compression) {
                                tx_frame = compress_frame(compression, tx_frame);
                        }
                        if(!tx_frame)
                                continue;
                        if(uv->connections_count == 1) { /* normal case - only one connection */
                                tx_send(uv->tx, tx_frame, 
                                                uv->network_devices[0]);
                        } else { /* split */
                                int i;

                                //assert(frame_count == 1);
                                vf_split_horizontal(splitted_frames, tx_frame,
                                               tile_y_count);
                                for (i = 0; i < tile_y_count; ++i) {
                                        tx_send_tile(uv->tx, splitted_frames, i,
                                                        uv->network_devices[i]);
                                }
                        }
                }
        }
        vf_free(splitted_frames);
        
        compress_done(compression);

        return NULL;
}

int main(int argc, char *argv[])
{

#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        char *network_device = NULL;

        char *capture_cfg = NULL;
        char *display_cfg = NULL;
        char *audio_send = NULL;
        char *audio_recv = NULL;
        char *jack_cfg = NULL;
        char *requested_fec = NULL;
        char *save_ptr = NULL;
        
        struct state_uv *uv;
        int ch;
        
        pthread_t receiver_thread_id, sender_thread_id;
        unsigned vidcap_flags = 0,
                 display_flags = 0;

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
                {"mode", required_argument, 0, 'M'},
                {"version", no_argument, 0, 'v'},
                {"compress", required_argument, 0, 'c'},
                {"ihdtv", no_argument, 0, 'i'},
                {"receive", required_argument, 0, 'r'},
                {"send", required_argument, 0, 's'},
                {"help", no_argument, 0, 'h'},
                {"jack", required_argument, 0, 'j'},
                {"fec", required_argument, 0, 'f'},
                {"port", required_argument, 0, 'P'},
                {0, 0, 0, 0}
        };
        int option_index = 0;

        //      uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv = (struct state_uv *)malloc(sizeof(struct state_uv));
        uv_state = uv;

        uv->audio = NULL;
        uv->ts = 0;
        uv->display_device = NULL;
        uv->requested_display = "none";
        uv->requested_capture = "none";
        uv->requested_compression = FALSE;
        uv->compress_options = NULL;
        uv->decoder_mode = NULL;
        uv->postprocess = NULL;
        uv->requested_mtu = 0;
        uv->use_ihdtv_protocol = 0;
        uv->participants = NULL;
        uv->tx = NULL;
        uv->network_devices = NULL;
        uv->port_number = PORT_BASE;

        perf_init();
        perf_record(UVP_INIT, 0);

        while ((ch =
                getopt_long(argc, argv, "d:t:m:r:s:vc:ihj:M:p:f:P:", getopt_options,
                            &option_index)) != -1) {
                switch (ch) {
                case 'd':
                        if (!strcmp(optarg, "help")) {
                                list_video_display_devices();
                                return 0;
                        }
                        uv->requested_display = strtok_r(optarg, ":", &save_ptr);
                        if(save_ptr && strlen(save_ptr) > 0)
                                display_cfg = save_ptr;
                        break;
                case 't':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->requested_capture = strtok_r(optarg, ":", &save_ptr);
                        if(save_ptr && strlen(save_ptr) > 0)
                                capture_cfg = save_ptr;
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
                        printf("%s\n", PACKAGE_STRING);
                        return EXIT_SUCCESS;
                case 'c':
                        uv->requested_compression = TRUE;
                        uv->compress_options = optarg;
                        break;
                case 'i':
                        uv->use_ihdtv_protocol = 1;
                        printf("setting ihdtv protocol\n");
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
                        requested_fec = optarg;
                        break;
		case 'h':
			usage();
			return 0;
                case 'P':
                        uv->port_number = atoi(optarg);
                        break;
                case '?':
                        break;
                default:
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }
        
        argc -= optind;
        argv += optind;


        if (uv->use_ihdtv_protocol) {
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
        }

        uv->audio = audio_cfg_init (network_device, uv->port_number + 2, audio_send, audio_recv, jack_cfg);
        if(!uv->audio)
                goto cleanup;

        if(audio_does_send_sdi(uv->audio))
                vidcap_flags |= VIDCAP_FLAG_ENABLE_AUDIO;
        if(audio_does_receive_sdi(uv->audio))
                display_flags |= DISPLAY_FLAG_ENABLE_AUDIO;

        printf("%s\n", PACKAGE_STRING);
        printf("Display device: %s\n", uv->requested_display);
        printf("Capture device: %s\n", uv->requested_capture);
        printf("MTU           : %d\n", uv->requested_mtu);
        /*printf("Compression   : ");
        if (uv->requested_compression) {
                printf("%s", get_compress_name(uv->compression));
        } else {
                printf("none");
        }
        printf("\n");*/

        if (uv->use_ihdtv_protocol)
                printf("Network protocol: ihdtv\n");
        else
                printf("Network protocol: ultragrid rtp\n");

        gettimeofday(&uv->start_time, NULL);

        uv->participants = pdb_init();

        if ((uv->capture_device =
                        initialize_video_capture(uv->requested_capture, capture_cfg, vidcap_flags)) == NULL) {
                printf("Unable to open capture device: %s\n",
                       uv->requested_capture);
                exit_status = EXIT_FAIL_CAPTURE;
                goto cleanup;
        }
        printf("Video capture initialized-%s\n", uv->requested_capture);

        if ((uv->display_device =
             initialize_video_display(uv->requested_display, display_cfg, display_flags)) == NULL) {
                printf("Unable to open display device: %s\n",
                       uv->requested_display);
                exit_status = EXIT_FAIL_DISPLAY;
                goto cleanup;
        }

        printf("Display initialized-%s\n", uv->requested_display);

#ifndef WIN32
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGHUP, signal_handler);
        signal(SIGABRT, signal_handler);
#endif

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

        if (uv->use_ihdtv_protocol) {
                ihdtv_connection tx_connection, rx_connection;

                printf("Initializing ihdtv protocol\n");

                // we cannot act as both together, because parameter parsing whould have to be revamped
                if ((strcmp("none", uv->requested_display) != 0)
                    && (strcmp("none", uv->requested_capture) != 0)) {
                        printf
                            ("Error: cannot act as both sender and reciever together in ihdtv mode\n");
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
                                        "Error initializing reciever session\n");
                                return 1;
                        }

                        if (pthread_create
                            (&receiver_thread_id, NULL, ihdtv_reciever_thread,
                             rx_connection_and_display) != 0) {
                                fprintf(stderr,
                                        "Error creating reciever thread. Quitting\n");
                                return 1;
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
                            (&sender_thread_id, NULL, ihdtv_sender_thread,
                             tx_connection_and_display) != 0) {
                                fprintf(stderr,
                                        "Error creating sender thread. Quitting\n");
                                return 1;
                        }
                }

                while (!should_exit)
                        sleep(1);

        } else {
                if ((uv->network_devices =
                     initialize_network(network_device, uv->port_number, uv->participants)) == NULL) {
                        printf("Unable to open network\n");
                        exit_status = EXIT_FAIL_NETWORK;
                        goto cleanup;
                } else {
                        struct rtp **item;
                        uv->connections_count = 0;
                        /* only count how many connections has initialize_network opened */
                        for(item = uv->network_devices; *item != NULL; ++item)
                                ++uv->connections_count;
                }

                if (uv->requested_mtu == 0)     // mtu wasn't specified on the command line
                {
                        uv->requested_mtu = 1500;       // the default value for rpt
                }

                if ((uv->tx = initialize_transmit(uv->requested_mtu, requested_fec)) == NULL) {
                        printf("Unable to initialize transmitter\n");
                        exit_status = EXIT_FAIL_TRANSMIT;
                        goto cleanup;
                }

                /* following block only shows help (otherwise initialized in receiver thread */
                if((uv->postprocess && strstr(uv->postprocess, "help") != NULL) || 
                                (uv->decoder_mode && strstr(uv->decoder_mode, "help") != NULL)) {
                        struct state_decoder *dec = decoder_init(uv->decoder_mode, uv->postprocess);
                        decoder_destroy(dec);
                        exit_status = EXIT_SUCCESS;
                        goto cleanup;
                }
                /* following block only shows help (otherwise initialized in sender thread */
                if(uv->requested_compression && strstr(uv->compress_options,"help") != NULL) {
                        struct compress_state *compression = compress_init(uv->compress_options);
                        compress_done(compression);
                        exit_status = EXIT_SUCCESS;
                        goto cleanup;
                }
                        
                if (strcmp("none", uv->requested_display) != 0) {
                        if (pthread_create
                            (&receiver_thread_id, NULL, receiver_thread,
                             (void *)uv) != 0) {
                                perror("Unable to create display thread!\n");
                                exit_status = 1;
                                goto cleanup;
                        }
                }
                if (strcmp("none", uv->requested_capture) != 0) {
                        if (pthread_create
                            (&sender_thread_id, NULL, sender_thread,
                             (void *)uv) != 0) {
                                perror("Unable to create capture thread!\n");
                                exit_status = 1;
                                goto cleanup;
                        }
                }
        }
        
        if(audio_does_receive_sdi(uv->audio)) {
                audio_register_get_callback(uv->audio, (struct audio_frame * (*)(void *)) display_get_audio_frame, uv->display_device);
                audio_register_put_callback(uv->audio, (void (*)(void *, struct audio_frame *)) display_put_audio_frame, uv->display_device);
                audio_register_reconfigure_callback(uv->audio, (int (*)(void *, int, int, 
                                                        int)) display_reconfigure_audio, uv->display_device);
        }

        if (strcmp("none", uv->requested_display) != 0)
                display_run(uv->display_device);

        if (strcmp("none", uv->requested_display) != 0)
                pthread_join(receiver_thread_id, NULL);

        if (strcmp("none", uv->requested_capture) != 0 || uv->requested_compression)
                pthread_join(sender_thread_id, NULL);
        
        /* also wait for audio threads */
        audio_join(uv->audio);

cleanup:
        while(wait_to_finish)
                ;
        threads_joined = TRUE;

        if(uv->audio)
                audio_done(uv->audio);
        if(uv->tx)
                tx_done(uv->tx);
	if(uv->network_devices)
                destroy_devices(uv->network_devices);
        if(uv->capture_device)
                vidcap_done(uv->capture_device);
        if(uv->display_device)
                display_done(uv->display_device);
        if (uv->participants != NULL)
                pdb_destroy(&uv->participants);
        printf("Exit\n");

        return exit_status;
}

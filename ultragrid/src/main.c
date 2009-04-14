/*
 * FILE:    main.c
 * AUTHORS: Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai  <ladan@isi.edu>
 *
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
 *      California Information Sciences Institute.
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
 * $Revision: 1.12 $
 * $Date: 2009/04/14 14:00:36 $
 *
 */

#include <string.h>
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "video_types.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "video_compress.h"
#include "pdb.h"
#include "tv.h"
#include "transmit.h"
#include "tfrc.h"
#include "version.h"
#include "ihdtv/ihdtv.h"
#include "compat/platform_semaphore.h"

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_UI   		2
#define EXIT_FAIL_DISPLAY	3
#define EXIT_FAIL_CAPTURE	4
#define EXIT_FAIL_NETWORK	5
#define EXIT_FAIL_TRANSMIT	6


struct state_uv {
	struct rtp		*network_device;
	struct vidcap		*capture_device;
	struct timeval		 start_time, curr_time;
	struct pdb		*participants;
	uint32_t		 ts;
	int			 fps;
	struct video_tx		*tx;
	struct display		*display_device;
	struct video_compress	*compression;
	const char		*requested_display;
	const char		*requested_capture;
	int			 requested_compression;
	int			 dxt_display;
	unsigned 		 requested_mtu;

	int                      use_ihdtv_protocol;
};

long            packet_rate = 13600;
static int	should_exit = FALSE;
uint32_t  	RTT = 0;    /* this is computed by handle_rr in rtp_callback */
char		*frame_buffer = NULL;
uint32_t        hd_size_x=1920;
uint32_t	hd_size_y=1080;
uint32_t	hd_color_bpp=3;
uint32_t	bitdepth = 10;
uint32_t	progressive = 0;
#ifdef HAVE_HDSTATION
#include <dvs_clib.h>
uint32_t	hd_video_mode=SV_MODE_SMPTE274_29I | SV_MODE_NBIT_10BDVS | SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;
//uint32_t	hd_video_mode=SV_MODE_SMPTE274_25P | SV_MODE_NBIT_10BDVS | SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;
#else
uint32_t	hd_video_mode=0;
#endif

long frame_begin[2];

#ifndef WIN32
static void
signal_handler(int signal)
{
        debug_msg("Caught signal %d\n", signal);
        should_exit = TRUE;
        return;
}
#endif

static void 
usage(void) 
{
	printf("Usage: uv [-d <display_device>] [-t <capture_device>] [-m <mtu>] [-f <framerate>] [-c] [-p] [-i] [-b <8|10>] address\n\t -i  ... ihdtv\n");
}

static void
initialize_video_codecs(void)
{
	int		nc, i;

	vcodec_init();
	nc = vcodec_get_num_codecs();
	for (i = 0; i < nc; i++) {
		if (vcodec_can_encode(i)) {
			printf("Video encoder : %s\n", vcodec_get_description(i));
		}
		if (vcodec_can_decode(i)) {
			printf("Video decoder : %s\n", vcodec_get_description(i));
		}

		/* Map static and "well-known" dynamic payload types */
		//TODO: Is this where I would list DXT payload?
		if (strcmp(vcodec_get_name(i), "uv_yuv") == 0) {
			vcodec_map_payload(96, i);	/*  96 : Uncompressed YUV */
		}
	}
}

static struct display *
initialize_video_display(const char *requested_display)
{
	struct display		*d;
	display_type_t		*dt;
	display_id_t		 id = display_get_null_device_id();
	int			 i;

	if (display_init_devices() != 0) {
		printf("Unable to initialise devices\n");
		abort();
	} else {
		debug_msg("Found %d display devices\n", display_get_device_count());
	}
	for (i = 0; i < display_get_device_count(); i++) {
		dt = display_get_device_details(i);
		if (strcmp(requested_display, dt->name) == 0) {
			id = dt->id;
			debug_msg("Found device\n");
		} else {
			debug_msg("Device %s does not match %s\n", dt->name, requested_display);
		}
	}
	display_free_devices();

	d = display_init(id);
	if (d != NULL) {
		frame_buffer = display_get_frame(d);
	}
	return d;
}

static struct vidcap *
initialize_video_capture(const char *requested_capture, int fps)
{
	struct vidcap_type	*vt;
	vidcap_id_t		 id = vidcap_get_null_device_id();
	int			 i;

	vidcap_init_devices();
	for (i = 0; i < vidcap_get_device_count(); i++) {
		vt = vidcap_get_device_details(i);
		if (strcmp(vt->name, requested_capture) == 0) {
			id = vt->id;
		}
	}
	vidcap_free_devices();

	return vidcap_init(id, fps);
}

static struct rtp *
initialize_network(char *addr, struct pdb *participants)
{
	struct rtp 	*r;
	double		 rtcp_bw = 5 * 1024 * 1024;	/* FIXME */
	
	r = rtp_init(addr, 5004, 5004, 255, rtcp_bw, FALSE, rtp_recv_callback, (void *) participants);
	if (r != NULL) {
		pdb_add(participants, rtp_my_ssrc(r));
		rtp_set_option(r, RTP_OPT_WEAK_VALIDATION,   TRUE);
		rtp_set_sdes(r, rtp_my_ssrc(r), RTCP_SDES_TOOL, ULTRAGRID_VERSION, strlen(ULTRAGRID_VERSION));
	}
	return r;
}

static struct video_tx *
initialize_transmit(unsigned requested_mtu)
{
	/* Currently this is trivial. It'll get more complex once we */
	/* have multiple codecs and/or error correction.             */
	return tx_init(requested_mtu);
}

static void *
ihdtv_reciever_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection*) ((void**)arg)[0];
        struct display *display_device = (struct display*) ((void**)arg)[1];

        while(!should_exit)
        {
                if( ihdtv_recieve(connection, frame_buffer, 1920 * 1080 * 3) )
                        return 0;       // we've got some error. probably empty buffer
                display_put_frame(display_device, frame_buffer);
                frame_buffer = display_get_frame(display_device);
        }
        return 0;
}


static void *
ihdtv_sender_thread(void *arg)
{
        ihdtv_connection *connection = (ihdtv_connection*) ((void**)arg)[0];
        struct vidcap           *capture_device = (struct vidcap*) ((void**)arg)[1];
        struct video_frame *tx_frame;

        while(!should_exit){
                if((tx_frame = vidcap_grab(capture_device)) != NULL)
                {
                        ihdtv_send(connection, tx_frame, 9000000); // FIXME: fix the use of frame size!!
                        free(tx_frame);
                }
                else
                {
                        fprintf(stderr,"Error recieving frame from capture device\n");
                        return 0;
                }
        }

        return 0;
}

static void *
display_thread(void *arg)
{
	struct state_uv *uv = (struct state_uv *) arg;

	struct pdb_e		*cp;
	struct timeval           timeout;
	int                      fr;
	int 			 i = 0;
	int                      ret;

	fr = 1;

	timeout.tv_sec  = 0;
	timeout.tv_usec = 999999 / 59.94;

	while (!should_exit) {
		/* Receive packets from the network... The timeout is adjusted */
		/* to match the video capture rate, so the transmitter works.  */
		if(fr) {
			gettimeofday(&uv->curr_time, NULL);
			frame_begin[i] = uv->curr_time.tv_usec;
			fr=0;
		}

		ret = rtp_recv(uv->network_device, &timeout, uv->ts);

		if (ret == FALSE) {
			printf("Failed to receive data\n");
		}

		/* Decode and render for each participant in the conference... */
		cp = pdb_iter_init(uv->participants);
		while (cp != NULL) { 
			if (tfrc_feedback_is_due(cp->tfrc_state, uv->curr_time)) {
				debug_msg("tfrc rate %f\n", 
						tfrc_feedback_txrate(cp->tfrc_state, uv->curr_time));
			}

			/* Decode and render video... */
			if (pbuf_decode(cp->playout_buffer, uv->curr_time, frame_buffer, i, uv->dxt_display)) {
				gettimeofday(&uv->curr_time, NULL);
				fr = 1;
				display_put_frame(uv->display_device, frame_buffer);
				i = (i + 1) % 2;
				frame_buffer = display_get_frame(uv->display_device);
			}
			pbuf_remove(cp->playout_buffer, uv->curr_time);
			cp = pdb_iter_next(uv->participants);
		}
		pdb_iter_done(uv->participants);
	}

	return 0;
}

static void *
capture_thread(void *arg)
{
	struct state_uv *uv = (struct state_uv *) arg;

	struct video_frame	*tx_frame;

	while (!should_exit) {
		/* Capture and transmit video... */
		tx_frame = vidcap_grab(uv->capture_device);
		if (tx_frame != NULL) {
			//TODO: Unghetto this
				if(uv->requested_compression) {
#ifdef HAVE_FASTDXT
					compress_data(uv->compression,tx_frame);
					dxt_tx_send(uv->tx, tx_frame, uv->network_device);
#endif /* HAVE_FASTDXT */
				}else{
					tx_send(uv->tx, tx_frame, uv->network_device);
				}
			free(tx_frame);
		}
	}

	return 0;
}

int 
main(int argc, char *argv[])
{

#ifdef HAVE_SCHED_SETSCHEDULER
	struct sched_param	 sp;
#endif

	struct state_uv		*uv;

	int			 ch;
	pthread_t		 display_thread_id, capture_thread_id;

	uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));

	uv->ts = 0;
	uv->fps = 60;
	uv->display_device = NULL;
	uv->compression = NULL;
	uv->requested_display = "none";
	uv->requested_capture = "none";
	uv->requested_compression = 0;
	uv->requested_mtu = 0;
	uv->use_ihdtv_protocol = 0;

	while ((ch = getopt(argc, argv, "d:t:m:f:b:vcpi")) != -1) {
		switch (ch) {
		case 'd' :
			uv->requested_display = optarg;
			if(!strcmp(uv->requested_display,"dxt"))
				uv->dxt_display = 1;
			break;
		case 't' :
			uv->requested_capture = optarg;
			break;
		case 'm' :
			uv->requested_mtu = atoi(optarg);
			break;
		case 'f' :
			uv->fps = atoi(optarg);
			break;
		case 'b' :
			bitdepth = atoi(optarg);
			if (bitdepth != 10 && bitdepth != 8) {
				usage();
				return EXIT_FAIL_USAGE;
			}
			if (bitdepth == 8) {
				hd_color_bpp = 2;
#ifdef HAVE_HDSTATION
				hd_video_mode=SV_MODE_SMPTE274_29I | SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;
#endif /* HAVE_HDSTATION */
			}
			break;
		case 'v' :
        		printf("%s\n", ULTRAGRID_VERSION);
			return EXIT_SUCCESS;
		case 'c' :
			uv->requested_compression=1;
			break;
		case 'p' :
			progressive=1;
			break;
		case 'i':
			uv->use_ihdtv_protocol = 1;
			printf("setting ihdtv protocol\n");
			break;
		default :
			usage();
			return EXIT_FAIL_USAGE;
		}
	}
	argc -= optind;
	argv += optind;

        if(uv->use_ihdtv_protocol)
        {
                if((argc != 0) && (argc != 1) && (argc != 2))
                {
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }
        else if (argc != 1) {
		usage();
		return EXIT_FAIL_USAGE;
	}

	printf("%s\n", ULTRAGRID_VERSION);
	printf("Display device: %s\n", uv->requested_display);
	printf("Capture device: %s\n", uv->requested_capture);
	printf("Frame rate    : %d\n", uv->fps);
	printf("MTU           : %d\n", uv->requested_mtu);
	if(uv->requested_compression)
		printf("Compression   : DXT\n");
	else
		printf("Compression   : None\n");

	if(uv->use_ihdtv_protocol)
		printf("Network protocol: ihdtv\n");
	else
		printf("Network protocol: ultragrid rtp\n");

	gettimeofday(&uv->start_time, NULL);

	initialize_video_codecs();
	uv->participants = pdb_init();

	if ((uv->capture_device = initialize_video_capture(uv->requested_capture, uv->fps)) == NULL) {
		printf("Unable to open capture device: %s\n", uv->requested_capture);
		return EXIT_FAIL_CAPTURE;
	}
	printf("Video capture initialized-%s\n", uv->requested_capture);

	if ((uv->display_device = initialize_video_display(uv->requested_display)) == NULL) {
		printf("Unable to open display device: %s\n", uv->requested_display);
		return EXIT_FAIL_DISPLAY;
	}
	printf("Display initialized-%s\n", uv->requested_display);

#ifdef HAVE_FASTDXT
	if (uv->requested_compression) {
		uv->compression = initialize_video_compression();
	}
#endif /* HAVE_FASTDXT */

        if(uv->use_ihdtv_protocol == 0)
        {
                if ((uv->network_device = initialize_network(argv[0], uv->participants)) == NULL) {
                        printf("Unable to open network\n");
                        return EXIT_FAIL_NETWORK;
                }

                if(uv->requested_mtu == 0)  // mtu wasn't specified on the command line
                {
                        uv->requested_mtu = 1500;   // the default value for rpt
                }

                if ((uv->tx = initialize_transmit(uv->requested_mtu)) == NULL) {
                        printf("Unable to initialize transmitter\n");
                        return EXIT_FAIL_TRANSMIT;
                }
        }

#ifndef WIN32
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGQUIT, signal_handler);
	signal(SIGHUP, signal_handler);
	signal(SIGABRT, signal_handler);
#endif

#ifdef HAVE_SCHED_SETSCHEDULER
	sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
	if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
		printf("WARNING: Unable to set real-time scheduling\n");
	}
#else
	printf("WARNING: System does not support real-time scheduling\n");
#endif

	if(uv->use_ihdtv_protocol) {
                ihdtv_connection tx_connection, rx_connection;

                printf("Initializing ihdtv protocol\n");

                // we cannot act as both together, because parameter parsing whould have to be revamped
                if ((strcmp("none", uv->requested_display) != 0) && (strcmp("none", uv->requested_capture) != 0))
                {
                        printf("Error: cannot act as both sender and reciever together in ihdtv mode\n");
                        return -1;
                }

                void *rx_connection_and_display[] = {(void*)&rx_connection, (void*)uv->display_device};
                void *tx_connection_and_display[] = {(void*)&tx_connection, (void*)uv->capture_device};

                pthread_t reciever_thread;
                pthread_t sender_thread;

                if(uv->requested_mtu == 0)  // mtu not specified on command line
                {
                        uv->requested_mtu = 8112;   // not really a mtu, but a video-data payload per packet
                }

                if (strcmp("none", uv->requested_display) != 0)
                {
                        if(ihdtv_init_rx_session(&rx_connection, (argc==0)?NULL:argv[0], (argc==0)?NULL:( (argc==1)?argv[0]:argv[1] ), 3000, 3001, uv->requested_mtu) != 0 )
                        {
                                fprintf(stderr, "Error initializing reciever session\n");
                                return 1;
                        }

                        if (pthread_create(&reciever_thread, NULL, ihdtv_reciever_thread, rx_connection_and_display) != 0)
                        {
                                fprintf(stderr, "Error creating reciever thread. Quitting\n");
                                return 1;
                        }
                }

                if (strcmp("none", uv->requested_capture) != 0)
                {
                        if(argc == 0)
                        {
                                fprintf(stderr, "Error: specify the destination address\n");
                                usage();
                                return EXIT_FAIL_USAGE;
                        }

                        if( ihdtv_init_tx_session(&tx_connection, argv[0], (argc==2)?argv[1]:argv[0], uv->requested_mtu) != 0 )
                        {
                                fprintf(stderr, "Error initializing sender session\n");
                                return 1;
                        }

                        if (pthread_create(&sender_thread, NULL, ihdtv_sender_thread, tx_connection_and_display) != 0)
                        {
                                fprintf(stderr, "Error creating sender thread. Quitting\n");
                                return 1;
                        }
                }

                while(!should_exit)
                        sleep(1);

	}
	else {
		if (strcmp("none", uv->requested_display) != 0) {
			if(pthread_create(&display_thread_id, NULL, display_thread, (void *) uv) != 0) {
				perror("Unable to create display thread!\n");
				should_exit = TRUE;
			}
		}
		if (strcmp("none", uv->requested_capture) != 0) {
			if(pthread_create(&capture_thread_id, NULL, capture_thread, (void *) uv) != 0) {
				perror("Unable to create capture thread!\n");
				should_exit = TRUE;
			}
		}

		while (!should_exit) {
			/* Housekeeping and RTCP... */
			gettimeofday(&uv->curr_time, NULL);
			uv->ts = tv_diff(uv->curr_time, uv->start_time) * 90000;
			rtp_update(uv->network_device, uv->curr_time);
			rtp_send_ctrl(uv->network_device, uv->ts, 0, uv->curr_time);

#ifndef X_DISPLAY_MISSING		
#ifdef HAVE_SDL
			if (strcmp(uv->requested_display, "sdl") == 0) {
				display_sdl_handle_events();
			}
#endif /* HAVE_SDL */
#endif /* X_DISPLAY_MISSING */

			usleep(200000);

		}
	}

	tx_done(uv->tx);
	rtp_done(uv->network_device);
	vidcap_done(uv->capture_device);
	display_done(uv->display_device);
	vcodec_done();
	pdb_destroy(&uv->participants);
	return EXIT_SUCCESS;
}

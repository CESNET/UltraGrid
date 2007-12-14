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
 * $Revision: 1.5 $
 * $Date: 2007/12/14 16:18:29 $
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

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_UI   		2
#define EXIT_FAIL_DISPLAY	3
#define EXIT_FAIL_CAPTURE	4
#define EXIT_FAIL_NETWORK	5
#define EXIT_FAIL_TRANSMIT	6

long            packet_rate = 13600;
static int	should_exit  = FALSE;
uint32_t  	RTT=0;    /* this is computed by handle_rr in rtp_callback */
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
	printf("Usage: uv [-d <display_device>] [-t <capture_device>] [-m <mtu>] [-f <framerate>] [-c] [-b <8|10>] address\n");
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

int 
main(int argc, char *argv[])
{
	struct rtp		*network_device;
	struct vidcap		*capture_device;
	struct timeval		 timeout, start_time, curr_time;
	struct pdb		*participants;
	struct pdb_e		*cp;
	uint32_t		 ts = 0;
	int			 ch;
	int 			 toggel=0;
	int			 fps = 60;
	struct video_frame	*tx_frame;
	struct video_tx		*tx;
	struct display		*display_device = NULL;
	struct video_compress	*compression = NULL;
	const char		*requested_display = "none";
	const char		*requested_capture = "none";
	int			requested_compression=0;
	int			dxt_display=0;
	unsigned 		 requested_mtu     = 1500;
	int                      fr;
	int 			i=0;

#ifdef HAVE_SCHED_SETSCHEDULER
	struct sched_param	 sp;
#endif

	while ((ch = getopt(argc, argv, "d:t:m:f:b:vcp")) != -1) {
		switch (ch) {
		case 'd' :
			requested_display = optarg;
			if(!strcmp(requested_display,"dxt"))
				dxt_display=1;
			break;
		case 't' :
			requested_capture = optarg;
			break;
		case 'm' :
			requested_mtu = atoi(optarg);
			break;
		case 'f' :
			fps = atoi(optarg);
			break;
		case 'b' :
			bitdepth = atoi(optarg);
			if (bitdepth != 10 && bitdepth != 8) {
				usage();
				return EXIT_FAIL_USAGE;
			}
			hd_color_bpp = 2;
#ifdef HAVE_HDSTATION
			hd_video_mode=SV_MODE_SMPTE274_29I | SV_MODE_COLOR_YUV422 | SV_MODE_ACTIVE_STREAMER;
#endif /* HAVE_HDSTATION */
			break;
		case 'v' :
        		printf("%s\n", ULTRAGRID_VERSION);
			return EXIT_SUCCESS;
		case 'c' :
			requested_compression=1;
			break;
		case 'p' :
			progressive=1;
			break;
		default :
			usage();
			return EXIT_FAIL_USAGE;
		}
	}
	argc -= optind;
	argv += optind;

	if (argc != 1) {
		usage();
		return EXIT_FAIL_USAGE;
	}

	printf("%s\n", ULTRAGRID_VERSION);
	printf("Display device: %s\n", requested_display);
	printf("Capture device: %s\n", requested_capture);
	printf("Frame rate    : %d\n", fps);
	printf("MTU           : %d\n", requested_mtu);
	if(requested_compression)
		printf("Compression   : DXT\n");
	else
		printf("Compression   : None\n");

	gettimeofday(&start_time, NULL);

	initialize_video_codecs();
	participants = pdb_init();

	if ((display_device = initialize_video_display(requested_display)) == NULL) {
		printf("Unable to open display device: %s\n", requested_display);
		return EXIT_FAIL_DISPLAY;
	}
	printf("Display initialized-%s\n", requested_display);

	if ((capture_device = initialize_video_capture(requested_capture, fps)) == NULL) {
		printf("Unable to open capture device: %s\n", requested_capture);
		return EXIT_FAIL_CAPTURE;
	}

#ifdef HAVE_FASTDXT
	if (requested_compression) {
		compression=initialize_video_compression();
	}
#endif /* HAVE_FASTDXT */

	if ((network_device = initialize_network(argv[0], participants)) == NULL) {
		printf("Unable to open network\n");
		return EXIT_FAIL_NETWORK;
	}

	if ((tx = initialize_transmit(requested_mtu)) == NULL) {
		printf("Unable to initialize transmitter\n");
		return EXIT_FAIL_TRANSMIT;
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

	fr = 1;
	while (!should_exit) {
		/* Housekeeping and RTCP... */
		gettimeofday(&curr_time, NULL);
		ts = tv_diff(curr_time, start_time) * 90000;
		rtp_update(network_device, curr_time);
		rtp_send_ctrl(network_device, ts, 0, curr_time);

#ifdef HAVE_SDL
		if (strcmp(requested_display, "sdl") == 0) {
			display_sdl_handle_events();
		}
#endif /* HAVE_SDL */

		/* Receive packets from the network... The timeout is adjusted */
		/* to match the video capture rate, so the transmitter works.  */
		if(fr) {
			gettimeofday(&curr_time, NULL);
			frame_begin[i] = curr_time.tv_usec;
			fr=0;
		}
		timeout.tv_sec  = 0;
		timeout.tv_usec = 999999 / 59.94;
		rtp_recv(network_device, &timeout, ts);

		/* Decode and render for each participant in the conference... */
		cp = pdb_iter_init(participants);
		while (cp != NULL) { 
			if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
				debug_msg("tfrc rate %f\n", 
						tfrc_feedback_txrate(cp->tfrc_state, curr_time));
			}

			/* Decode and render video... */
			if (pbuf_decode(cp->playout_buffer, curr_time, frame_buffer, i,dxt_display)) {
				gettimeofday(&curr_time, NULL);
                        	//printf("Frame %lld begin (%d.%d)\n", (long long)frno++, (int)curr_time.tv_sec, (int)curr_time.tv_usec);		
				fr=1;
				display_put_frame(display_device, frame_buffer);
				i = (i + 1) %2;
				frame_buffer = display_get_frame(display_device);
			}
			pbuf_remove(cp->playout_buffer, curr_time);
			cp = pdb_iter_next(participants);
		}
		pdb_iter_done(participants);

		/* Capture and transmit video... */
		tx_frame = vidcap_grab(capture_device);
		if (tx_frame != NULL) {
			/* send every other frame */
			//if (++toggel%2==0)
			toggel=0;
			//TODO: Unghetto this
				if(requested_compression) {
#ifdef HAVE_FASTDXT
					compress_data(compression,tx_frame);
					dxt_tx_send(tx, tx_frame, network_device);
#endif /* HAVE_FASTDXT */
				}else{
					tx_send(tx, tx_frame, network_device);
				}
			free(tx_frame);
		}
#if 0
		if(iter<101)
			iter++;
		else
			exit(1);
#endif
	}

	tx_done(tx);
	rtp_done(network_device);
	vidcap_done(capture_device);
	display_done(display_device);
	vcodec_done();
	pdb_destroy(&participants);
	return EXIT_SUCCESS;
}

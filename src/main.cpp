/*
 * FILE:    main.cpp
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
 *          Martin Pulec     <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2014 CESNET z.s.p.o.
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
#include "rtsp/rtsp_utils.h"
#include "stats.h"
#include "utils/misc.h"
#include "utils/wait_obj.h"
#include "video.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_compress.h"
#include "video_export.h"
#include "video_rxtx/h264_rtp.h"
#include "video_rxtx/ihdtv.h"
#include "video_rxtx/sage.h"
#include "video_rxtx/ultragrid_rtp.h"
#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/codec.h"
#include "audio/utils.h"

#include <iostream>
#include <string>

#ifdef USE_MTRACE
#include <mcheck.h>
#endif

#define EXIT_FAIL_USAGE     1
#define EXIT_FAIL_UI        2
#define EXIT_FAIL_DISPLAY   3
#define EXIT_FAIL_CAPTURE   4
#define EXIT_FAIL_NETWORK   5
#define EXIT_FAIL_TRANSMIT  6
#define EXIT_FAIL_COMPRESS  7
#define EXIT_FAIL_DECODER   8

#define PORT_BASE               5004
#define PORT_AUDIO              5006

/* please see comments before transmit.c:audio_tx_send() */
/* also note that this actually differs from video */
#define DEFAULT_AUDIO_FEC       "mult:3"
#define DEFAULT_BITRATE         (6618ll * 1000 * 1000)

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
#define OPT_LDGM_DEVICE (('L' << 8) | 'D')
#define OPT_WINDOW_TITLE (('W' << 8) | 'T')
#define OPT_CAPABILITIES (('C' << 8) | 'C')

#define MAX_CAPTURE_COUNT 17

using namespace std;

struct state_uv {
        struct vidcap *capture_device;
        struct display *display_device;

        struct state_audio *audio;

        struct module *root_module;

        video_rxtx *state_video_rxtx;
};

static int exit_status = EXIT_SUCCESS;
static volatile bool should_exit_sender = false;

static struct state_uv *uv_state;

//
// prototypes
//
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
        audio_finish();
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
        printf("\t--control-port <port>[:0|1] \tset control port (default port: 5054)\n");
        printf("\t                         \tconnection types: 0- Server (default), 1- Client\n");
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
        printf("\t--rtsp-server            \tRTSP server: dynamically serving H264 RTP standard transport (use '--rtps-server=help' to see usage)\n");
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
        printf("\t--ldgm-device {GPU|CPU}  \tdevice to be used to compute LDGM\n");
        printf("\n");
        printf("\t-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]\n");
        printf("\t                         \t<port> is base port number, also 3 subsequent\n");
        printf("\t                         \tports can be used for RTCP and audio\n");
        printf("\t                         \tstreams. Default: %d.\n", PORT_BASE);
        printf("\t                         \tYou can also specify all two or four ports\n");
        printf("\t                         \tdirectly.\n");
        printf("\n");
        printf("\t-l <limit_bitrate> | unlimited | auto\tlimit sending bitrate\n");
        printf("\t                         \tto <limit_bitrate> (with optional k/M/G suffix)\n");
        printf("\n");
        printf("\t--audio-channel-map      <mapping> | help\n");
        printf("\n");
        printf("\t--audio-scale <factor> | <method> | help\n");
        printf("\t                         \tscales received audio\n");
        printf("\n");
        printf("\t--audio-capture-channels <count> number of input channels that will\n");
        printf("\t                                 be captured (default 2).\n");
        printf("\n");
        printf("\t--echo-cancellation      \tapply acustic echo cancellation to audio\n");
        printf("\n");
        printf("\t--cuda-device <index>|help\tuse specified CUDA device\n");
        printf("\n");
        printf("\t--playback <directory>   \treplays captured recorded\n");
        printf("\n");
        printf("\t--record[=<directory>]   \trecord captured audio and video\n");
        printf("\n");
        printf("\t-A <address>             \taudio destination address\n");
        printf("\t                         \tIf not specified, will use same as for video\n");
        printf("\t--audio-codec <codec>[:sample_rate=<sr>][:bitrate=<br>]|help\taudio codec\n");
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
        struct wait_obj *wait_obj;

        wait_obj = wait_obj_init();

        while (!should_exit_sender) {
                /* Capture and transmit video... */
                struct audio_frame *audio;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);
                if (tx_frame != NULL) {
                        if(audio) {
                                audio_sdi_send(uv->audio, audio);
                        }
                        //tx_frame = vf_get_copy(tx_frame);
                        bool wait_for_cur_uncompressed_frame;
                        if (!tx_frame->dispose) {
                                tx_frame->dispose = uncompressed_frame_dispose;
                                tx_frame->dispose_udata = wait_obj;
                                wait_obj_reset(wait_obj);
                                wait_for_cur_uncompressed_frame = true;
                        } else {
                                wait_for_cur_uncompressed_frame = false;
                        }

                        uv->state_video_rxtx->send(tx_frame);

                        // wait for frame frame to be processed, eg. by compress
                        // or sender (uncompressed video). Grab invalidates previous frame
                        // (if not defined dispose function).
                        if (wait_for_cur_uncompressed_frame) {
                                wait_obj_wait(wait_obj);
                                tx_frame->dispose = NULL;
                                tx_frame->dispose_udata = NULL;
                        }
                }
        }

        wait_obj_done(wait_obj);

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
        struct vidcap_params *vidcap_params_head = vidcap_params_allocate();
        struct vidcap_params *vidcap_params_tail = vidcap_params_head;

        char *display_cfg = NULL;
        const char *audio_recv = "none";
        const char *audio_send = "none";
        char *jack_cfg = NULL;
        const char *requested_video_fec = "none";
        const char *requested_audio_fec = DEFAULT_AUDIO_FEC;
        char *audio_channel_map = NULL;
        const char *audio_scale = "mixauto";
        int rtsp_port = 0;
        bool isStd = FALSE;
        int recv_port_number = PORT_BASE;
        int send_port_number = PORT_BASE;

        bool echo_cancellation = false;

        bool should_export = false;
        char *export_opts = NULL;

        char *sage_opts = NULL;
        int control_port = CONTROL_DEFAULT_PORT;
        int connection_type = 0;
        struct control_state *control = NULL;

        const char *audio_host = NULL;
        int audio_rx_port = -1, audio_tx_port = -1;
        enum video_mode decoder_mode = VIDEO_NORMAL;
        const char *requested_compression = "none";

        bool ipv6 = false;
        struct module root_mod;
        struct state_uv *uv;
        int ch;

        const char *audio_codec = "PCM";

        pthread_t receiver_thread_id,
                  capture_thread_id;
        bool receiver_thread_started = false,
             capture_thread_started = false;
        unsigned display_flags = 0;
        int compressed_audio_sample_rate = 48000;
        int ret;
        struct vidcap_params *audio_cap_dev;
        long packet_rate;
        const char *requested_mcast_if = NULL;

        unsigned requested_mtu = 0;
        const char *postprocess = NULL;
        const char *requested_display = "none";
        const char *requested_receiver = "localhost";
        const char *requested_encryption = NULL;
        struct video_export *video_exporter = NULL;

        int bitrate = RATE_AUTO;

#ifdef USE_MTRACE
        mtrace();
#endif

        vidcap_params_set_device(vidcap_params_head, "none");

        if (argc == 1) {
                usage();
                return EXIT_FAIL_USAGE;
        }

        uv_argc = argc;
        uv_argv = argv;

        video_rxtx_loader loader;

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
                {"rtsp-server", optional_argument, 0, 'H'},
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
                {"record", optional_argument, 0, OPT_EXPORT},
                {"playback", required_argument, 0, OPT_IMPORT},
                {"audio-host", required_argument, 0, 'A'},
                {"audio-codec", required_argument, 0, OPT_AUDIO_CODEC},
                {"capture-filter", required_argument, 0, OPT_CAPTURE_FILTER},
                {"control-port", required_argument, 0, OPT_CONTROL_PORT},
                {"encryption", required_argument, 0, OPT_ENCRYPTION},
                {"verbose", no_argument, 0, OPT_VERBOSE},
                {"ldgm-device", required_argument, 0, OPT_LDGM_DEVICE},
                {"window-title", required_argument, 0, OPT_WINDOW_TITLE},
                {"capabilities", no_argument, 0, OPT_CAPABILITIES},
                {0, 0, 0, 0}
        };
        int option_index = 0;

        //      uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv_state = uv;

        uv->audio = NULL;
        uv->capture_device = NULL;
        uv->display_device = NULL;

        init_root_module(&root_mod, uv);
        uv->root_module = &root_mod;
        enum rxtx_protocol video_protocol = ULTRAGRID_RTP;

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
                        requested_display = optarg;
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
                        vidcap_params_set_device(vidcap_params_tail, optarg);
                        vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        break;
                case 'm':
                        requested_mtu = atoi(optarg);
                        break;
                case 'M':
                        decoder_mode = get_video_mode_from_str(optarg);
                        if (decoder_mode == VIDEO_UNKNOWN) {
                                return strcasecmp(optarg, "help") == 0 ? EXIT_SUCCESS : EXIT_FAIL_USAGE;
                        }
                        break;
                case 'p':
                        postprocess = optarg;
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
                        requested_compression = optarg;
                        break;
                case 'i':
#ifdef HAVE_IHDTV
                        video_protocol = IHDTV;
                        printf("setting ihdtv protocol\n");
                        fprintf(stderr, "Warning: iHDTV support may be currently broken.\n"
                                        "Please contact %s if you need this.\n", PACKAGE_BUGREPORT);
#else
                        fprintf(stderr, "iHDTV support isn't compiled in this %s build\n",
                                        PACKAGE_NAME);
#endif
                        break;
                case 'S':
                        video_protocol = SAGE;
                        sage_opts = optarg;
                        break;
                case 'H':
                        video_protocol = H264_STD;
                        if (optarg == NULL) {
                        	rtsp_port = 0;
                        } else {
                                if (!strcmp(optarg, "help")) {
#ifdef HAVE_RTSP_SERVER
                                        rtps_server_usage();
#endif
                                        return 0;
                                }
#ifdef HAVE_RTSP_SERVER
                                rtsp_port = get_rtsp_server_port(optarg);
#endif
                                if (rtsp_port == -1) return 0;
                        }
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
                                        requested_audio_fec = optarg + 2;
                                } else {
                                        requested_video_fec = optarg + 2;
                                }
                        } else {
                                // there should be setting for both audio and video
                                // but we conservativelly expect that the user wants
                                // only vieo and let audio default until explicitly
                                // stated otehrwise
                                requested_video_fec = optarg;
                        }
                        break;
                case 'h':
                        usage();
                        return 0;
                case 'P':
                        if(strchr(optarg, ':')) {
                                char *save_ptr = NULL;
                                char *tok;
                                recv_port_number = atoi(strtok_r(optarg, ":", &save_ptr));
                                send_port_number = atoi(strtok_r(NULL, ":", &save_ptr));
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
                                recv_port_number =
                                        send_port_number =
                                        atoi(optarg);
                        }
                        break;
                case 'l':
                        if(strcmp(optarg, "unlimited") == 0) {
                                bitrate = 0;
                        } else if(strcmp(optarg, "auto") == 0) {
                                bitrate = RATE_AUTO;
                        } else {
                                bitrate = unit_evaluate(optarg);
                                if(bitrate <= 0) {
                                        usage();
                                        return EXIT_FAIL_USAGE;
                                }
                        }
                        break;
                case '6':
                        ipv6 = true;
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
                        requested_mcast_if = optarg;
                        break;
                case 'A':
                        audio_host = optarg;
                        break;
                case OPT_EXPORT:
                        should_export = true;
                        export_opts = optarg;
                        break;
                case OPT_IMPORT:
                        audio_send = "embedded";
                        {
                                char dev_string[1024];
                                snprintf(dev_string, sizeof(dev_string), "import:%s", optarg);
                                vidcap_params_set_device(vidcap_params_tail, dev_string);
                                vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        }
                        break;
                case OPT_AUDIO_CODEC:
                        if(strcmp(optarg, "help") == 0) {
                                list_audio_codecs();
                                return EXIT_SUCCESS;
                        }
                        audio_codec = optarg;
                        if(get_audio_codec(optarg) == AC_NONE) {
                                fprintf(stderr, "Unknown audio codec entered: \"%s\"\n",
                                                optarg);
                                return EXIT_FAIL_USAGE;
                        }
                        compressed_audio_sample_rate = get_audio_codec_sample_rate(audio_codec);
                        break;
                case OPT_CAPTURE_FILTER:
                        vidcap_params_set_capture_filter(vidcap_params_tail, optarg);
                        break;
                case OPT_ENCRYPTION:
                        requested_encryption = optarg;
                        break;
                case OPT_CONTROL_PORT:
                        if (strchr(optarg, ':')) {
                                char *save_ptr = NULL;
                                char *tok;
                                control_port = atoi(strtok_r(optarg, ":", &save_ptr));
                                connection_type = atoi(strtok_r(NULL, ":", &save_ptr));
                                if(connection_type < 0 || connection_type > 1){
                                        usage();
                                        return EXIT_FAIL_USAGE;
                                }
                                if ((tok = strtok_r(NULL, ":", &save_ptr))) {
                                        usage();
                                        return EXIT_FAIL_USAGE;
                                }
                        } else {
                                control_port = atoi(optarg);
                        }
                        break;
                case OPT_VERBOSE:
                        verbose = true;
                        break;
                case OPT_LDGM_DEVICE:
                        if (strcasecmp(optarg, "GPU") == 0) {
                                ldgm_device_gpu = true;
                        } else {
                                ldgm_device_gpu = false;
                        }
                        break;
                case OPT_WINDOW_TITLE:
                        window_title = optarg;
                        break;
                case OPT_CAPABILITIES:
                        print_capabilities(CAPABILITY_CAPTURE | CAPABILITY_COMPRESS);
                        return EXIT_SUCCESS;
                        break;
                case '?':
                default:
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }

        argc -= optind;
        argv += optind;

        if (requested_mtu == 0)     // mtu wasn't specified on the command line
        {
                requested_mtu = 1500;       // the default value for RTP
        }

        printf("%s", PACKAGE_STRING);
#ifdef GIT_VERSION
        printf(" (rev %s)", GIT_VERSION);
#endif
        printf("\n");
        printf("Display device   : %s\n", requested_display);
        printf("Capture device   : %s\n", vidcap_params_get_driver(vidcap_params_head));
        printf("Audio capture    : %s\n", audio_send);
        printf("Audio playback   : %s\n", audio_recv);
        printf("MTU              : %d B\n", requested_mtu);
        printf("Video compression: %s\n", requested_compression);
        printf("Audio codec      : %s\n", get_name_to_audio_codec(get_audio_codec(audio_codec)));
        printf("Network protocol : %s\n", video_rxtx::get_name(video_protocol));
        printf("Audio FEC        : %s\n", requested_audio_fec);
        printf("Video FEC        : %s\n", requested_video_fec);
        printf("\n");

        if(audio_rx_port == -1) {
                audio_tx_port = send_port_number + 2;
                audio_rx_port = recv_port_number + 2;
        }

        if(should_export) {
                if(!enable_export(export_opts)) {
                        fprintf(stderr, "Export initialization failed.\n");
                        return EXIT_FAILURE;
                }
                video_exporter = video_export_init(export_dir);
        }

        if (bitrate != RATE_AUTO && bitrate != 0) {
                packet_rate = compute_packet_rate(bitrate, requested_mtu);
        } else {
                packet_rate = bitrate;
        }

        if (argc > 0) {
                requested_receiver = argv[0];
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

        if(control_init(control_port, connection_type, &control, &root_mod) != 0) {
                fprintf(stderr, "%s Unable to initialize remote control!\n",
                                control_port != CONTROL_DEFAULT_PORT ? "Warning:" : "Error:");
                if(control_port != CONTROL_DEFAULT_PORT) {
                        return EXIT_FAILURE;
                }
        }

        if(!audio_host) {
                audio_host = requested_receiver;
        }
#ifdef HAVE_RTSP_SERVER
        if((audio_send != NULL || audio_recv != NULL) && video_protocol == H264_STD){
            //TODO: to implement a high level rxtx struct to manage different standards (i.e.:H264_STD, VP8_STD,...)
            isStd = TRUE;
        }
#endif
        uv->audio = audio_cfg_init (&root_mod, audio_host, audio_rx_port,
                        audio_tx_port, audio_send, audio_recv,
                        jack_cfg, requested_audio_fec, requested_encryption,
                        audio_channel_map,
                        audio_scale, echo_cancellation, ipv6, requested_mcast_if,
                        audio_codec, isStd, packet_rate);
        if(!uv->audio)
                goto cleanup;

        display_flags |= audio_get_display_flags(uv->audio);

        // Display initialization should be prior to modules that may use graphic card (eg. GLSL) in order
        // to initalize shared resource (X display) first
        ret =
             initialize_video_display(requested_display, display_cfg, display_flags, &uv->display_device);
        if (ret < 0) {
                printf("Unable to open display device: %s\n",
                       requested_display);
                exit_uv(EXIT_FAIL_DISPLAY);
                goto cleanup;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }

        printf("Display initialized-%s\n", requested_display);

        /* Pass embedded/analog/AESEBU flags to selected vidcap
         * device. */
        if (audio_capture_get_vidcap_flags(audio_send)) {
                audio_cap_dev = vidcap_params_get_nth(
                                vidcap_params_head,
                                audio_capture_get_vidcap_index(audio_send));
                if (audio_cap_dev != NULL) {
                        unsigned int orig_flags =
                                vidcap_params_get_flags(audio_cap_dev);
                        vidcap_params_set_flags(audio_cap_dev, orig_flags
                                        | audio_capture_get_vidcap_flags(audio_send));
                } else {
                        fprintf(stderr, "Entered index for non-existing vidcap (audio).\n");
                        exit_uv(EXIT_FAIL_CAPTURE);
                        goto cleanup;
                }
        }

        ret = initialize_video_capture(&root_mod, vidcap_params_head, &uv->capture_device);
        if (ret < 0) {
                printf("Unable to open capture device: %s\n",
                                vidcap_params_get_driver(vidcap_params_head));
                exit_uv(EXIT_FAIL_CAPTURE);
                goto cleanup;
        }
        if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }
        printf("Video capture initialized-%s\n", vidcap_params_get_driver(vidcap_params_head));

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

        if (strcmp("none", requested_display) != 0) {
                rxtx_mode |= MODE_RECEIVER;
        }
        if (strcmp("none", vidcap_params_get_driver(vidcap_params_head)) != 0) {
                rxtx_mode |= MODE_SENDER;
        }

        if (rxtx_mode == 0) {
                goto after_video_init;
        }

        try {
                map<string, param_u> params;

                // common
                params["parent"].ptr = &root_mod;
                params["exporter"].ptr = video_exporter;
                params["compression"].ptr = (void *) requested_compression;

                // iHDTV
                params["argc"].i = argc;
                params["argv"].ptr = argv;
                params["capture_device"].ptr = NULL;
                params["display_device"].ptr = NULL;
                if (rxtx_mode & MODE_SENDER)
                        params["capture_device"].ptr = uv->capture_device;
                if (rxtx_mode & MODE_RECEIVER)
                        params["display_device"].ptr = uv->display_device;

                //RTP
                params["mtu"].i = requested_mtu;
                params["receiver"].ptr = (void *) requested_receiver;
                params["rx_port"].i = recv_port_number;
                params["tx_port"].i = send_port_number;
                params["use_ipv6"].b = ipv6;
                params["mcast_if"].ptr = (void *) requested_mcast_if;
                params["mtu"].i = requested_mtu;
                params["fec"].ptr = (void *) requested_video_fec;
                params["encryption"].ptr = (void *) requested_encryption;
                params["packet_rate"].i = packet_rate;

                // UltraGrid RTP
                params["postprocess"].ptr = (void *) postprocess;
                params["decoder_mode"].l = decoder_mode;
                params["display_device"].ptr = uv->display_device;

                // SAGE
                params["sage_opts"].ptr = sage_opts;

                // RTSP
                params["rtsp_port"].i = rtsp_port;
                params["audio_codec"].l = get_audio_codec(audio_codec);
                params["audio_sample_rate"].i = compressed_audio_sample_rate;
                params["audio_channels"].i = audio_capture_channels;
                params["audio_bps"].i = 2;
                params["a_rx_port"].i = audio_rx_port;

                if (video_protocol == H264_STD) {
                        rtps_types_t avType;
                        if(strcmp("none", vidcap_params_get_driver(vidcap_params_head)) != 0 && (strcmp("none",audio_send) != 0)) avType = av; //AVStream
                        else if((strcmp("none",audio_send) != 0)) avType = audio; //AStream
                        else if(strcmp("none", vidcap_params_get_driver(vidcap_params_head))) avType = video; //VStream
                        else {
                                printf("[RTSP SERVER CHECK] no stream type... check capture devices input...\n");
                                avType = none;
                        }

                        params["avType"].l = avType;
                }

                uv->state_video_rxtx = video_rxtx::create(video_protocol, params);
                if (!uv->state_video_rxtx) {
                        throw string("Requested RX/TX cannot be created (missing library?)");
                }

                if(rxtx_mode & MODE_RECEIVER) {
                        if (!uv->state_video_rxtx->supports_receiving()) {
                                fprintf(stderr, "Selected RX/TX mode doesn't support receiving.\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        }
                        // init module here so as it is capable of receiving messages
                        if (pthread_create
                                        (&receiver_thread_id, NULL, video_rxtx::receiver_thread,
                                         (void *) uv->state_video_rxtx) != 0) {
                                perror("Unable to create display thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        } else {
                                receiver_thread_started = true;
                        }
                }

                if(rxtx_mode & MODE_SENDER) {
                        if (pthread_create
                                        (&capture_thread_id, NULL, capture_thread,
                                         (void *) &root_mod) != 0) {
                                perror("Unable to create capture thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        } else {
                                capture_thread_started = true;
                        }
                }

                if(audio_get_display_flags(uv->audio)) {
                        audio_register_put_callback(uv->audio, (void (*)(void *, struct audio_frame *)) display_put_audio_frame, uv->display_device);
                        audio_register_reconfigure_callback(uv->audio, (int (*)(void *, int, int,
                                                        int)) display_reconfigure_audio, uv->display_device);
                }

                // should be started after requested modules are able to respond after start
                control_start(control);

                if (strcmp("none", requested_display) != 0)
                        display_run(uv->display_device);
        } catch (string const &str) {
                cerr << str << endl;
                exit_status = EXIT_FAILURE;
        } catch (int i) {
                exit_status = i;
        }

after_video_init:
cleanup:
        if (strcmp("none", requested_display) != 0 &&
                        receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

        if (rxtx_mode & MODE_SENDER
                        && capture_thread_started)
                pthread_join(capture_thread_id, NULL);

        /* also wait for audio threads */
        audio_join(uv->audio);
        if (uv->state_video_rxtx)
                uv->state_video_rxtx->join();

        if(uv->audio)
                audio_done(uv->audio);
        delete uv->state_video_rxtx;

        if (uv->capture_device)
                vidcap_done(uv->capture_device);
        if (uv->display_device)
                display_done(uv->display_device);

        video_export_destroy(video_exporter);

        free(export_dir);

        control_done(control);

        while  (vidcap_params_head) {
                struct vidcap_params *next = vidcap_params_get_next(vidcap_params_head);
                vidcap_params_free_struct(vidcap_params_head);
                vidcap_params_head = next;
        }

        module_done(&root_mod);
        free(uv);

#ifdef USE_MTRACE
        muntrace();
#endif

#ifdef WIN32
        WSACleanup();
#endif

        printf("Exit\n");

        return exit_status;
}


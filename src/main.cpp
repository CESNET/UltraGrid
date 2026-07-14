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
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
 * Copyright (c) 2005-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
#endif // HAVE_CONFIG_H

#include <algorithm>                    // for max, min
#include <cassert>                      // for assert
#include <cctype>                       // for toupper
#include <chrono>
#include <climits>                      // for INT_MIN
#include <csignal>                      // for signal, SIGPIPE, SIG_DFL, SIGHUP
#include <cstdint>                      // for UINT16_MAX, uint32_t
#include <cstdio>                       // for printf, perror, fprintf, stderr
#include <cstdlib>
#include <cstring>                      // for strcmp, strlen, strtok_r, strchr
#include <getopt.h>
#include <initializer_list>             // for initializer_list
#include <iostream>
#include <memory>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>                  // for operator==, basic_string_view
#include <strings.h>                    // for strcasecmp
#include <unistd.h>                     // for optarg, optind, STDERR...
#include <utility>                      // for move

#include "audio/audio.h"                // for audio_options, additional_aud...
#include "audio/audio_capture.h"        // for audio_capture_get_vidcap_flags
#include "audio/audio_playback.h"       // for audio_playback_help
#include "audio/codec.h"                // for audio_codec_params, get_name_...
#include "audio/types.h"                // for AC_NONE, AUDIO_FRAME_DISPOSE
#include "compat/alarm.h"               // IWYU pragma: keep for alarm
#include "control_socket.h"
#include "cuda_wrapper.h"
#include "debug.h"
#include "export.h"                     // for export_destroy, export_init
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "module.h"
#include "playback.h"
#include "rtp/rtp.h"
#include "rxtx.h"
#include "rxtx/rtp_common.h"            // for RTP_PORT_BASE
#include "rxtx/ultragrid_rtp.h"         // for ultragrid_rtp_server_mode_help
#include "types.h"                      // for video_frame, video_frame_call...
#include "utils/color_out.h"
#include "utils/macros.h"               // for snprintf_ch, to_fourcc
#include "utils/misc.h"
#include "utils/nat.h"
#include "utils/net.h"
#include "utils/pthread.h"              // for PTHREAD_NULL
#include "utils/string.h"
#include "utils/string_view_utils.hpp"
#include "utils/thread.h"
#include "utils/udp_holepunch.h"
#include "utils/video.h"
#include "utils/wait_obj.h"             // for wait_obj_done, wait_obj_init
#include "video_capture.h"
#include "video_capture_params.h"       // for vidcap_params_get_driver, vid...
#include "video_display.h"

constexpr char MOD_NAME[] = "[main] ";

constexpr int OPT_AUDIO_CAPTURE_CHANNELS = ('a' << 8) | 'c';
constexpr int OPT_AUDIO_DELAY            = ('A' << 8) | 'D';
constexpr int OPT_AUDIO_HOST             = ('A' << 8) | 'H';
constexpr int OPT_AUDIO_PROTOCOL         = ('A' << 8) | 'P';
constexpr int OPT_AUDIO_SCALE            = ('a' << 8) | 's';
constexpr int OPT_ECHO_CANCELLATION      = ('E' << 8) | 'C';
constexpr int OPT_MCAST_IF               = ('M' << 8) | 'I';
constexpr int OPT_PIX_FMTS               = ('P' << 8) | 'F';
constexpr int OPT_PIXFMT_CONV_POLICY     = ('P' << 8) | 'C';
constexpr int OPT_RTSP_SERVER            = ('R' << 8) | 'S';
constexpr int OPT_VIDEO_CODECS           = ('V' << 8) | 'C';
constexpr int OPT_VIDEO_PROTOCOL         = ('V' << 8) | 'P';
constexpr int OPT_WINDOW_TITLE           = ('W' << 8) | 'T';

using namespace std;
using namespace std::chrono;

struct state_uv {
        uint32_t magic = state_magic;
        state_uv() noexcept {
                init_root_module(&root_module);
                register_should_exit_callback(&root_module, state_uv::should_exit_capture_callback, this);
        }
        ~state_uv() {
                module_done(&root_module);
                destroy_root_module(&root_module);
        }

        struct vidcap *capture_device{};
        struct display *display_device{};

        struct state_audio *audio{};

        struct module root_module;

        struct rxtx *state_rxtx{};

        static void should_exit_capture_callback(void *udata) {
                auto *s = (state_uv *) udata;
                s->should_exit_capture = true;
        }
        bool should_exit_capture = false;
        static constexpr uint32_t state_magic = to_fourcc('U', 'G', 'S', 'T');
};

static void signal_handler(int signum)
{
#ifdef SIGPIPE
        if (signum == SIGPIPE) {
                signal(SIGPIPE, SIG_IGN);
        }
#endif // defined SIGPIPE
        if (log_level >= LOG_LEVEL_VERBOSE) {
                char buf[128];
                char *ptr = buf;
                char *ptr_end = buf + sizeof buf;
                strappend(&ptr, ptr_end, "Caught signal ");
                if (signum / 10 != 0) {
                        *ptr++ = (char) ('0' + signum / 10);
                }
                *ptr++ = (char) ('0' + signum % 10);
                append_sig_desc(&ptr, ptr_end - 1, signum);
                *ptr++ = '\n';
                write_all(STDERR_FILENO, ptr - buf, buf);
        }
        exit_uv(0);
}

static void
print_help_item(const string &name, initializer_list<string> help)
{
        enum {
                HELP_LEFT_OFF  = 40,
                TAB_LEN        = 8,
        };
        int help_lines = 0;

        col() << "\t" << TBOLD(<< name <<);

        if (name.length() >= HELP_LEFT_OFF - TAB_LEN || help.size() == 0) {
                cout << "\n";
                help_lines = 1;
        }

        for (auto const &line : help) {
                unsigned spaces = HELP_LEFT_OFF;
                if (help_lines == 0) {
                        spaces -= name.length() + TAB_LEN;
                }
                for (unsigned i = 0; i < spaces; ++i) {
                        cout << " ";
                }
                cout << line << "\n";
                help_lines += 1;
        }
}

static void
usage(bool full = false)
{
        col() << "Usage: " << SBOLD(SRED((uv_argv[0])) << " [options] address\n\n");
        printf("Options:\n");
        print_help_item("-h | -H, --fullhelp", {"show usage (basic/full)"});
        print_help_item("-d <display_device>", {"select display device, use '-d help'",
                        "to get list of supported devices"});
        print_help_item("-t <capture_device>", {"select capture device, use '-t help'",
                        "to get list of supported devices"});
        print_help_item("-c <cfg>", {"video compression (see '-c help')"});
        print_help_item("-r <playback_device>", {"audio playback device (see '-r help')"});
        print_help_item("-s <capture_device>", {"audio capture device (see '-s help')"});
        if (full) {
                print_help_item(
                    "-V, --verbose[=<level>]",
                    { "print verbose messages (optionally specify level [0-" +
                      to_string(LOG_LEVEL_MAX) + "])" });
                print_help_item("--list-modules", {"prints list of modules"});
                print_help_item("--control-port <port>[:0|1]", {"set control port (default port: " + to_string(DEFAULT_CONTROL_PORT) + ")",
                                "connection types: 0- Server (default), 1- Client"});
                print_help_item("-x, --protocol <proto>", {"transmission protocol to use (see `-x help`)"});
#ifdef HAVE_IPv6
                print_help_item("-4/-6", {"force IPv4/IPv6 resolving"});
#endif //  HAVE_IPv6
                print_help_item("--mcast-if <iface>", {"bind to specified interface for multicast"});
                print_help_item("-m <mtu>", {"set path MTU assumption towards receiver"});
                print_help_item("-M <video_mode>", {"received video mode (eg tiled-4K, 3D,",
                                "dual-link)"});
                print_help_item("-N, --nat-traverse"s, {"try to deploy NAT traversal techniques"s});
                print_help_item("-C/-S"s, {"client/server mode - use '-S help'"s});
                print_help_item("-p <postprocess> | help", {"postprocess module"});
                print_help_item("-T, --ttl <num>", {"Use specified TTL for multicast/unicast (0..255, -1 for default)"});
        }
        print_help_item("-f [A:|V:]<settings>", {"FEC settings (audio or video) - use",
                        "\"none\", \"mult:<nr>\",", "\"ldgm:<max_expected_loss>%\" or", "\"ldgm:<k>:<m>:<c>\"",
                        "\"rs:<k>:<n>\""});
        if (full) {
                print_help_item("-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]", {
                                "<port> is base port number, also 3",
                                "subsequent ports can be used for RTCP",
                                "and audio streams. RTP default: " + to_string(RTP_PORT_BASE)  + ".",
                                "You can also specify all two or four", "ports directly."});
                print_help_item("-l <limit_bitrate> | unlimited | auto", {"limit sending bitrate",
                                "to <limit_bitrate> (with optional k/M/G suffix)"});
                print_help_item("--audio-host <address>", {"audio destination address",
                                "If not specified, will use same as for video"});
        }
        print_help_item("-a, --audio-capture-format <fmt> | help", {"format of captured audio"});
        if (full) {
                print_help_item("--audio-channel-map <mapping> | help", {});
                print_help_item("--audio-filter <filter>[:<config>][#<filter>[:<config>]]...", {});
        }
        print_help_item("-A, --audio-codec "
                        "<codec>[:sample_rate=<sr>][:bitrate=<br>] | help",
                        { "audio codec" });
        if (full) {
                print_help_item("--audio-delay <delay_ms>", {"amount of time audio should be delayed to video",
                                "(may be also negative to delay video)"});
                print_help_item("--audio-scale <factor> | <method> | help",
                                {"scales received audio"});
                print_help_item("--echo-cancellation", {"apply acoustic echo cancellation to audio (experimental)"});
                print_help_item("-D, --cuda-device <index> | help", {"use specified GPU"});
                print_help_item("-e, --encryption <passphrase>", {"key material for encryption"});
                print_help_item("-I, --playback <directory> | help", {"replays recorded audio and video"});
                print_help_item("-E, --record[=<directory>]", {"record captured audio and video"});
                print_help_item("-F, --capture-filter <filter> | help",
                                {"capture filter, must precede -t if more device used"});
                print_help_item("--param <params> | help", {"additional advanced parameters, use help for list"});
                print_help_item("--pix-fmts", {"list of pixel formats"});
                print_help_item("--conv-policy [cds]{3} | help", {"pixel format conversion policy"});
                print_help_item("--video-codecs", {"list of video codecs"});
        }
        printf("\n");
        print_help_item("address", {"destination address"});
        if (full) {
                printf("\n");
                color_printf("Environment variables: "
                             TBOLD("NDILIB_REDIST_FOLDER") ", " TBOLD("ULTRAGRID_ERRORS_FATAL")
#ifdef _WIN32
                              ", " TBOLD("NV_OPTIMUS_ENABLEMENT")
#endif
                             " and others (less important)\n");
        }
        printf("\n");
}

static void
print_fps(const char *prefix, steady_clock::time_point *t0, int *frames,
          double nominal_fps)
{
        enum {
                MIN_FPS_PERC_WARN  = 98,
                MIN_FPS_PERC_WARN2 = 90,
        };
        if (prefix == nullptr) {
                return;
        }
        *frames += 1;
        steady_clock::time_point t1 = steady_clock::now();
        double seconds = duration_cast<duration<double>>(t1 - *t0).count();
        if (seconds < 5.0) {
                return;
        }
        const double fps      = *frames / seconds;
        const char *const fps_col  = get_stat_color(fps / nominal_fps);
        log_msg(LOG_LEVEL_INFO,
                TBOLD(TMAGENTA("%s")) " %d frames in %g seconds = " TBOLD(
                    "%s%g FPS" TERM_FG_RESET) "\n",
                prefix, *frames, seconds, fps_col, fps);
        *t0     = t1;
        *frames = 0;
}

static void
frame_wait_obj_notify(struct video_frame *f)
{
        auto *wait_obj = (struct wait_obj *) f->callbacks.dispose_udata;
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
        set_thread_name(__func__);

        struct state_uv *uv = (struct state_uv *) arg;
        assert(uv->magic == state_uv::state_magic);
        struct wait_obj *wait_obj = wait_obj_init();
        steady_clock::time_point t0 = steady_clock::now();
        int frames = 0;
        const char              *print_fps_prefix =
            vidcap_get_fps_print_prefix(uv->capture_device);

        while (!uv->should_exit_capture) {
                /* Capture and transmit video... */
                struct audio_frame *audio = nullptr;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);

                if (audio != nullptr) {
                        audio_sdi_send(uv->audio, audio);
                        AUDIO_FRAME_DISPOSE(audio);
                }

                if (tx_frame == nullptr) {
                        continue;
                }
                print_fps(print_fps_prefix, &t0, &frames, tx_frame->fps);
                // tx_frame = vf_get_copy(tx_frame);
                bool                    wait_for_cur_uncompressed_frame = false;
                if (tx_frame->callbacks.dispose == nullptr) {
                        wait_obj_reset(wait_obj);
                        wait_for_cur_uncompressed_frame = true;
                        tx_frame->callbacks.dispose = frame_wait_obj_notify;
                        tx_frame->callbacks.dispose_udata = wait_obj;
                }

                rxtx_send_video(uv->state_rxtx, tx_frame);

                // wait for frame frame to be processed, eg. by compress
                // or sender (uncompressed video). Grab invalidates previous
                // frame (if not defined dispose function).
                if (wait_for_cur_uncompressed_frame) {
                        wait_obj_wait(wait_obj);
                        tx_frame->callbacks.dispose       = nullptr;
                        tx_frame->callbacks.dispose_udata = nullptr;
                }
        }

        wait_obj_done(wait_obj);

        return NULL;
}

/// @retval <0 return code whose absolute value will be returned from main()
/// @retval 0 success
/// @retval 1 device list was printed
static int parse_cuda_device(char *optarg) {
        if(strcmp("help", optarg) == 0) {
#ifdef HAVE_CUDA
                cuda_wrapper_print_devices_info(true);
                return 1;
#else
                LOG(LOG_LEVEL_ERROR) << "CUDA support is not enabled!\n";
                return -EXIT_FAILURE;
#endif
        }
        char *item, *save_ptr = NULL;
        unsigned int i = 0;
        while((item = strtok_r(optarg, ",", &save_ptr))) {
                if(i >= MAX_CUDA_DEVICES) {
                        LOG(LOG_LEVEL_ERROR) << "Maximal number of CUDA device exceeded.\n";
                        return -EXIT_FAILURE;
                }
                const int val = parse_number(item, 0, 16);
                if (val == INT_MIN) {
                        return -EXIT_FAIL_USAGE;
                }
                cuda_devices[i] = val;
                optarg = NULL;
                ++i;
        }
        cuda_devices_count = i;
        assert(cuda_devices_count >= 1);
        cuda_devices_explicit = true;
        return 0;
}

template<int N>
static void copy_sv_to_c_buf(char (&dest)[N], std::string_view sv){
        sv.copy(dest, N - 1);
        dest[N - 1] = '\0';
}

[[maybe_unused]] static bool parse_holepunch_conf(char *conf, struct Holepunch_config *punch_c){
        std::string_view sv = conf;
        while(!sv.empty()){
                auto token = tokenize(sv, ':');
                auto key = tokenize(token, '=');
                auto val = tokenize(token, '=');

                if(key == "help"){
                        col() << "Usage:\n" <<
                                "\tuv " << TBOLD("-Nholepunch:room=<room>:(server=<host> | coord_srv=<host:port>:stun_srv=<host:port>)[:client_name=<name>][:bind_ip=<addr>] \n") <<
                                "\twhere\n"
                                "\t\t" << TBOLD("server") << " - used if both stun & coord server are on the same host on standard ports (3478, 12558)\n"
                                "\t\t" << TBOLD("room") << " - name of room to join\n"
                                "\t\t" << TBOLD("client_name") << " - name to identify as to the coord server, if not specified hostname is used\n"
                                "\t\t" << TBOLD("bind_ip") << " - local ip to bind to\n"
                                "\n";
                        return false;
                }

                if(key == "coord_srv"){
                        copy_sv_to_c_buf(punch_c->coord_srv_addr, val);

                        token = tokenize(sv, ':');
                        if(token.empty()){
                                log_msg(LOG_LEVEL_ERROR, "Missing hole punching coord server port.\n");
                                return false;
                        }

                        if(!parse_num(token, punch_c->coord_srv_port)){
                                log_msg(LOG_LEVEL_ERROR, "Failed to parse hole punching coord server port.\n");
                                return false;
                        }
                } else if(key == "stun_srv"){
                        copy_sv_to_c_buf(punch_c->stun_srv_addr, val);

                        token = tokenize(sv, ':');
                        if(token.empty()){
                                log_msg(LOG_LEVEL_ERROR, "Missing hole punching stun server port.\n");
                                return false;
                        }

                        if(!parse_num(token, punch_c->stun_srv_port)){
                                log_msg(LOG_LEVEL_ERROR, "Failed to parse hole punching stun server port.\n");
                                return false;
                        }
                } else if(key == "server"){
                        copy_sv_to_c_buf(punch_c->stun_srv_addr, val);
                        copy_sv_to_c_buf(punch_c->coord_srv_addr, val);

                        punch_c->stun_srv_port = 3478;
                        punch_c->coord_srv_port = 12558;
                } else if(key == "room"){
                        copy_sv_to_c_buf(punch_c->room_name, val);
                } else if(key == "client_name"){
                        copy_sv_to_c_buf(punch_c->client_name, val);
                } else if(key == "bind_ip"){
                        copy_sv_to_c_buf(punch_c->bind_addr, val);
                }
        }

        if(!strlen(punch_c->stun_srv_addr)
                || !strlen(punch_c->coord_srv_addr)
                || !strlen(punch_c->room_name))
        {
                log_msg(LOG_LEVEL_ERROR, "Not all hole punch params provided.\n");
                return false;
        }

        return true;
}

static int
parse_mtu(char *optarg)
{
        enum { IPV4_REQ_MIN_MTU = 576, };
        bool enforce = false;
        if (optarg[0] != '\0' && optarg[strlen(optarg) - 1] == '!') {
                enforce = true;
                optarg[strlen(optarg) - 1] = '\0';
        }
        const int ret = parse_number(optarg, 1, 10);
        if (ret == INT_MIN) {
                return -1;
        }
        if (ret < IPV4_REQ_MIN_MTU && !enforce) {
                MSG(WARNING,
                    "MTU %s seems to be too low, use \"%d!\" to force.\n",
                    optarg, ret);
                return -1;
        }
        return ret;
}

struct ug_options {
        ug_options() {
                vidcap_params_set_device(vidcap_params_head, "none");
                // will be adjusted later
                audio.codec_cfg = nullptr;
        }
        ~ug_options() {
                vidcap_params_free(vidcap_params_head);
        }
        struct rxtx_params rxtx = RXTX_INIT;
        struct audio_options audio = AUDIO_OPTIONS_INIT;
        std::string audio_filter_cfg;
        // NULL terminated array of capture devices
        struct vidcap_params *vidcap_params_head = vidcap_params_allocate();
        struct vidcap_params *vidcap_params_tail = vidcap_params_head;

        const char *display_cfg = "";

        bool should_export = false;
        char *export_opts = NULL;

        int control_port = 0;
        int connection_type = 0;

        const char *postprocess = NULL;
        const char *requested_display = "none";

        bool is_client = false;
        bool is_server = false;

        const char *requested_capabilities = nullptr;

        char net_protocol[128] = "ultragrid_rtp";

        char *nat_traverse_config = nullptr;
};

static int
parse_audio_capture(struct ug_options *opt, const char *optarg)
{
        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                audio_capture_print_help(strcmp(optarg, "full") == 0);
                return 1;
        }
        if (string(opt->audio.send_cfg) != ug_options().audio.send_cfg &&
            (audio_capture_get_vidcap_flags(opt->audio.send_cfg) == 0 ||
             audio_capture_get_vidcap_flags(optarg) == 0)) {
                log_msg(LOG_LEVEL_ERROR, "Multiple audio devices given! Only "
                                         "allowed for video-attached audio "
                                         "connection (AESEBU/analog/embedded).\n");
                return -EXIT_FAIL_USAGE;
        }
        if (audio_capture_get_vidcap_flags(optarg)) {
                vidcap_params_add_flags(opt->vidcap_params_tail,
                                        audio_capture_get_vidcap_flags(optarg));
        }
        opt->audio.send_cfg = optarg;
        return 0;
}

static bool parse_port(char *optarg, struct ug_options *opt) {
        char *save_ptr = nullptr;
        char *first_val = strtok_r(optarg, ":", &save_ptr);
        if (first_val == nullptr || strcmp(first_val, "help") == 0) {
                color_printf("see\n\n    " TBOLD("%s --fullhelp") "\n\nfor port specification usage\n", uv_argv[0]);
                return false;
        }
        struct rxtx_medium_params *audio = &opt->rxtx.medium[TX_MEDIA_AUDIO];
        struct rxtx_medium_params *video = &opt->rxtx.medium[TX_MEDIA_VIDEO];
        if (char *tx_port_str = strtok_r(nullptr, ":", &save_ptr)) {
                video->rx_port = stoi(first_val, nullptr, 0);
                video->tx_port = stoi(tx_port_str, nullptr, 0);
                char *tok = nullptr;
                if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                        audio->rx_port = stoi(tok, nullptr, 0);
                        if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                                audio->tx_port = stoi(tok, nullptr, 0);
                        } else {
                                usage(uv_argv[0]);
                                return false;
                        }
                }
        } else {
                opt->rxtx.port_base = stoi(first_val, nullptr, 0);
        }
        if (audio->rx_port < -1 || audio->tx_port < -1 ||
            video->rx_port < -1 || video->tx_port < -1 ||
            opt->rxtx.port_base < -1 ||
            audio->rx_port > UINT16_MAX ||
            audio->tx_port > UINT16_MAX ||
            video->rx_port > UINT16_MAX ||
            video->tx_port > UINT16_MAX || opt->rxtx.port_base > UINT16_MAX) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Invalid port value, allowed range 1-65535\n";
                return false;
        }
        return true;
}

static bool parse_protocol(int ch, char *optarg, struct ug_options *opt) {
        if ((strlen(optarg) > 2 && optarg[1] == ':') ||
            ch == OPT_AUDIO_PROTOCOL || ch == OPT_VIDEO_PROTOCOL) {
                MSG(ERROR, "Separate audio and video protocol setting no "
                           "longer available!\n");
                return false;
        }

        char *proto = optarg;
        const char *cfg = "";
        char *delim = strchr(optarg, ':');
        if (delim != nullptr) {
                *delim = '\0';
                cfg = delim + 1;
        }
        if (strcmp(optarg, "help") == 0 ||
                strcmp(optarg, "fullhelp") == 0) {
                color_printf("Specify a network transmission protocol.\n\n");
                color_printf("Usage:\n");
                color_printf(TBOLD("\t--protocol/-x proto[:opts]") "\n");
                color_printf("\n");
                rxtx_list_protocols(strcmp(optarg, "fullhelp") == 0);
                return false;
        }
        strcpy_ch(opt->net_protocol, proto);
        strcpy_ch(opt->rxtx.protocol_opts, cfg);

        return true;
}

static bool parse_control_port(char *optarg, struct ug_options *opt) {
        if (strchr(optarg, ':')) {
                char *save_ptr = NULL;
                char *tok;
                opt->control_port = atoi(strtok_r(optarg, ":", &save_ptr));
                opt->connection_type = atoi(strtok_r(NULL, ":", &save_ptr));

                if (opt->connection_type < 0 || opt->connection_type > 1){
                        usage(uv_argv[0]);
                        return false;
                }
                if ((tok = strtok_r(NULL, ":", &save_ptr))) {
                        usage(uv_argv[0]);
                        return false;
                }
        } else {
                opt->control_port = atoi(optarg);
                opt->connection_type = 0;
        }
        return true;
}

/**
 * @retval -error error value that should be returned from main
 * @retval 0 success
 * @retval 1 success (help written)
 */
static int
parse_options_internal(int argc, char *argv[], struct ug_options *opt)
{
        static struct option getopt_options[] = {                // sort by
                {"audio-capture-format",   required_argument, nullptr, 'a'},
                {"capabilities",           optional_argument, nullptr, 'b'},
                {"compress",               required_argument, nullptr, 'c'},
                {"display",                required_argument, nullptr, 'd'},
                {"encryption",             required_argument, nullptr, 'e'},
                {"fec",                    required_argument, nullptr, 'f'},
                {"help",                   no_argument,       nullptr, 'h'},
                {"audio-filter",           required_argument, nullptr, 'i'},
                {"limit-bitrate",          required_argument, nullptr, 'l'},
                {"capture",                required_argument, nullptr, 't'},
                {"mtu",                    required_argument, nullptr, 'm'},
                {"control-port",           required_argument, nullptr, 'n'},
                {"postprocess",            required_argument, nullptr, 'p'},
                {"receive",                required_argument, nullptr, 'r'},
                {"send",                   required_argument, nullptr, 's'},
                {"version",                no_argument,       nullptr, 'v'},
                {"protocol",               required_argument, nullptr, 'x'},
                {"audio-codec",            required_argument, nullptr, 'A'},
                {"client",                 no_argument,       nullptr, 'C'},
                {"cuda-device",            required_argument, nullptr, 'D'},
                {"record",                 optional_argument, nullptr, 'E'},
                {"capture-filter",         required_argument, nullptr, 'F'},
                {"fullhelp",               no_argument,       nullptr, 'H'},
                {"playback",               required_argument, nullptr, 'I'},
                {"list-modules",           no_argument,       nullptr, 'L'},
                {"mode",                   required_argument, nullptr, 'M'},
                {"nat-traverse",           optional_argument, nullptr, 'N'},
                {"param",                  required_argument, nullptr, 'O'},
                {"port",                   required_argument, nullptr, 'P'},
                {"server",                 no_argument,       nullptr, 'S'},
                {"ttl",                    required_argument, nullptr, 'T'},
                {"audio-channel-map",      required_argument, nullptr, 'U'},
                {"verbose",                optional_argument, nullptr, 'V'},
                {"audio-capture-channels", required_argument, 0, OPT_AUDIO_CAPTURE_CHANNELS},
                {"audio-delay",            required_argument, 0, OPT_AUDIO_DELAY},
                {"audio-host",             required_argument, 0, OPT_AUDIO_HOST},
                {"audio-protocol",         required_argument, 0, OPT_AUDIO_PROTOCOL},
                {"audio-scale",            required_argument, 0, OPT_AUDIO_SCALE},
                {"echo-cancellation",      no_argument,       0, OPT_ECHO_CANCELLATION},
                {"mcast-if",               required_argument, 0, OPT_MCAST_IF},
                {"conv-policy",            required_argument, 0, OPT_PIXFMT_CONV_POLICY},
                {"pix-fmts",               no_argument,       0, OPT_PIX_FMTS},
                {"rtsp-server",            optional_argument, 0, OPT_RTSP_SERVER},
                {"video-codecs",           no_argument,       0, OPT_VIDEO_CODECS},
                {"video-protocol",         required_argument, 0, OPT_VIDEO_PROTOCOL},
                {"window-title",           required_argument, 0, OPT_WINDOW_TITLE},
                {0, 0, 0, 0}
        };
        const char *optstring =
            "46A:CD:E::F:HI:LM:N::O:P:ST:U:Va:b::c:e:f:d:hi:l:m:n:p:r:s:t:vx:";

        int ch = 0;
        while ((ch =
                getopt_long(argc, argv, optstring, getopt_options,
                            NULL)) != -1) {
                string tmp = optarg ? optarg : "";
                char *optarg_copy = tmp.data();
                switch (ch) {
                case 'd':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                list_video_display_devices(strcmp(optarg, "fullhelp") == 0);
                                return 1;
                        }
                        if (opt->requested_display && strcmp(opt->requested_display, "none") != 0) {
                                log_msg(LOG_LEVEL_ERROR, "Multiple displays given!\n");
                                return -EXIT_FAIL_USAGE;
                        }
                        opt->requested_display = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                opt->display_cfg = delim + 1;
                        }
                        break;
                case 't':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                list_video_capture_devices(strcmp(optarg, "fullhelp") == 0);
                                return 1;
                        }
                        vidcap_params_set_device(opt->vidcap_params_tail, optarg);
                        opt->vidcap_params_tail = vidcap_params_allocate_next(opt->vidcap_params_tail);
                        break;
                case 'm':
                        opt->rxtx.mtu = parse_mtu(optarg_copy);
                        if (opt->rxtx.mtu == -1) {
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'M':
                        opt->rxtx.decoder_mode = get_video_mode_from_str(optarg);
                        if (opt->rxtx.decoder_mode == VIDEO_UNKNOWN) {
                                return strcasecmp(optarg, "help") == 0 ? 1 : -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'p':
                        opt->postprocess = optarg;
                        break;
                case 'v':
                        print_configuration();
                        return 1;
                case 'c':
                        strcpy_ch(opt->rxtx.video_compression, optarg);
                        break;
                case OPT_RTSP_SERVER:
                        log_msg(LOG_LEVEL_WARNING, "Option \"--rtsp-server[=args]\" "
                                        "is deprecated and will be removed in future.\n"
                                        "Please use \"-x rtsp[:args]\"instead.\n");
                        strcpy_ch(opt->net_protocol, "rtsp");
                        strcpy_ch(opt->rxtx.protocol_opts, optarg ? optarg : "");
                        break;
                case OPT_AUDIO_PROTOCOL:
                case OPT_VIDEO_PROTOCOL:
                case 'x':
                        if (!parse_protocol(ch, optarg_copy, opt)) {
                                return 1;
                        }
                        break;
                case 'r':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                audio_playback_help(strcmp(optarg, "full") == 0);
                                return 1;
                        }
                        if (string(opt->audio.recv_cfg) != ug_options().audio.recv_cfg) {
                                log_msg(LOG_LEVEL_ERROR, "Multiple audio playback devices given!\n");
                                return -EXIT_FAIL_USAGE;
                        }
                        opt->audio.recv_cfg = optarg;
                        break;
                case 's':
                        if (int ret = parse_audio_capture(opt, optarg)) {
                                return ret;
                        }
                        break;
                case 'f':
                        if(strlen(optarg) > 2 && optarg[1] == ':' &&
                                        (toupper(optarg[0]) == 'A' || toupper(optarg[0]) == 'V')) {
                                if(toupper(optarg[0]) == 'A') {
                                        opt->rxtx.medium[TX_MEDIA_AUDIO].fec = optarg + 2;
                                } else {
                                        opt->rxtx.medium[TX_MEDIA_VIDEO].fec = optarg + 2;
                                }
                        } else {
                                // there should be setting for both audio and video
                                // but we conservativelly expect that the user wants
                                // only vieo and let audio default until explicitly
                                // stated otherwise
                                opt->rxtx.medium[TX_MEDIA_VIDEO].fec = optarg;
                        }
                        break;
                case 'h':
                        usage(false);
                        return 1;
                case 'P':
                        if (!parse_port(optarg_copy, opt)) {
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'l':
                        strcpy_ch(opt->rxtx.video_bitrate_limit, optarg);
                        break;
                case '4':
                case '6':
                        opt->rxtx.force_ip_version = ch - '0';
                        break;
                case 'U':
                        opt->audio.channel_map = optarg;
                        break;
                case OPT_AUDIO_SCALE:
                        opt->audio.scale = optarg;
                        break;
                case OPT_AUDIO_CAPTURE_CHANNELS:
                        log_msg(LOG_LEVEL_WARNING, "Parameter --audio-capture-channels is deprecated. "
                                        "Use \"-a channels=<count>\" instead.\n");
                        audio_capture_channels = atoi(optarg);
                        if (audio_capture_channels < 1) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid number of channels %d!\n", audio_capture_channels);
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'a':
                        if (int ret = set_audio_capture_format(optarg)) {
                                return ret < 0 ? -EXIT_FAIL_USAGE : 1;
                        }
                        break;
                case 'i':
                        if(!opt->audio_filter_cfg.empty()){
                                opt->audio_filter_cfg += "#";
                        }
                        opt->audio_filter_cfg += optarg;
                        break;
                case OPT_ECHO_CANCELLATION:
                        opt->audio.echo_cancellation = true;
                        break;
                case 'H':
                        usage(true);
                        return 1;
                case 'D':
                        if (int ret = parse_cuda_device(optarg_copy)) {
                                return ret;
                        }
                        break;
                case OPT_MCAST_IF:
                        snprintf_ch(opt->rxtx.mcast_if, "%s", optarg);
                        break;
                case OPT_AUDIO_HOST:
                        bug_msg(LOG_LEVEL_ERROR,
                                "Separate audio host has been removed. Let us "
                                "if you use this feature. ");
                        return -1;
                case 'E':
                        opt->should_export = true;
                        opt->export_opts = optarg;
                        break;
                case 'I':
                        opt->audio.send_cfg = "embedded";
                        {
                                char dev_string[1024];
                                int ret = playback_set_device(dev_string, sizeof dev_string, optarg);
                                if (ret != 0) {
                                        return ret == 1 ? 1 : -EXIT_FAIL_USAGE;
                                }
                                vidcap_params_set_device(opt->vidcap_params_tail, dev_string);
                                opt->vidcap_params_tail = vidcap_params_allocate_next(opt->vidcap_params_tail);
                        }
                        break;
                case 'A':
                        if(strcmp(optarg, "help") == 0) {
                                list_audio_codecs();
                                return 1;
                        }
                        opt->audio.codec_cfg = optarg;
                        if (parse_audio_codec_params(opt->audio.codec_cfg)
                                .codec == AC_NONE) {
                                return -1;
                        }
                        break;
                case 'F':
                        vidcap_params_add_capture_filter(
                            opt->vidcap_params_tail, optarg);
                        break;
                case 'e':
                        snprintf_ch(opt->rxtx.encryption, "%s", optarg);
                        break;
                case 'n':
                        if (!parse_control_port(optarg_copy, opt)) {
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'V':
                        break; // already handled in common_preinit()
                case OPT_WINDOW_TITLE:
                        log_msg(LOG_LEVEL_WARNING, "Deprecated option used, please use "
                                        "--param window-title=<title>\n");
                        set_commandline_param("window-title", optarg);
                        break;
                case 'b':
                        opt->requested_capabilities = optarg ? optarg : "";
                        break;
                case OPT_AUDIO_DELAY:
                        set_audio_delay(stoi(optarg));
                        break;
                case 'L':
                        return list_all_modules() ? 1 : -EXIT_FAILURE;
                case 'O':
                        if (!parse_params(optarg, false)) {
                                return 1;
                        }
                        break;
                case OPT_PIX_FMTS:
                        print_pixel_formats();
                        return 1;
                case OPT_PIXFMT_CONV_POLICY:
                        if (int ret = set_pixfmt_conv_policy(optarg)) {
                                return ret < 0 ? -EXIT_FAIL_USAGE : 1;
                        }
                        break;
                case OPT_VIDEO_CODECS:
                        print_video_codecs();
                        return 1;
                case 'N':
                        opt->nat_traverse_config = optarg == nullptr ? const_cast<char *>("") : optarg;
                        break;
                case 'C':
                        opt->is_client = true;
                        break;
                case 'S':
                        opt->is_server = true;
                        break;
                case 'T':
                        opt->rxtx.ttl = stoi(optarg);
                        break;
                case '?':
                default:
                        usage(uv_argv[0]);
                        return -EXIT_FAIL_USAGE;
                }
        }

        opt->audio.filter_cfg = opt->audio_filter_cfg.data();

        argc -= optind;
        argv += optind;

        if (argc > 1) {
                log_msg(LOG_LEVEL_ERROR, "Multiple receivers given!\n");
                usage(uv_argv[0]);
                return -EXIT_FAIL_USAGE;
        }

        if (argc > 0) {
                opt->rxtx.receiver = argv[0];
        }

        return 0;
}

/// @copydoc parse_options_internal
static int
parse_options(int argc, char *argv[], struct ug_options *opt)
{
        try {
                return parse_options_internal(argc, argv, opt);
        } catch (logic_error &e) {
                if (!invalid_arg_is_numeric(e.what())) {
                        throw;
                }
                if (dynamic_cast<invalid_argument *>(&e) != nullptr) {
                        LOG(LOG_LEVEL_ERROR)
                            << MOD_NAME
                            << "Non-numeric value passed to option "
                               "expecting a number!\n";
                } else if (dynamic_cast<out_of_range *>(&e) != nullptr) {
                        LOG(LOG_LEVEL_ERROR)
                            << MOD_NAME << "Passed value is out of bounds!\n";
                } else {
                        throw;
                }
                return -1;
        }
}

static int
adjust_params_holepunch(struct ug_options *opt)
{
#ifndef HAVE_LIBJUICE
        (void) opt;
        log_msg(LOG_LEVEL_ERROR, "Ultragrid was compiled without holepunch support\n");
        return EXIT_FAILURE;
#else
        struct rxtx_medium_params *audio = &opt->rxtx.medium[TX_MEDIA_AUDIO];
        struct rxtx_medium_params *video = &opt->rxtx.medium[TX_MEDIA_VIDEO];
        static char punched_host[512];
        Holepunch_config punch_c = {};

        if(!parse_holepunch_conf(opt->nat_traverse_config, &punch_c)){
                return -EXIT_FAILURE;
        }

        set_commandline_param("udp-disable-multi-socket", "");

        if (strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) == 0
                        && strcmp("none", opt->requested_display) != 0)
        {
                vidcap_params_set_device(opt->vidcap_params_tail, "testcard:2:2:1:UYVY");
                opt->vidcap_params_tail = vidcap_params_allocate_next(opt->vidcap_params_tail);
        }

        if (strcmp("none", opt->audio.send_cfg) == 0
                        && strcmp("none", opt->audio.recv_cfg) != 0)
        {
                set_audio_capture_format("sample_rate=5");
                opt->audio.send_cfg = "testcard:frames=1";
        }

        punch_c.video_rx_port = &video->rx_port;
        punch_c.video_tx_port = &video->tx_port;
        punch_c.audio_rx_port = &audio->rx_port;
        punch_c.audio_tx_port = &audio->tx_port;

        punch_c.host_addr = punched_host;
        punch_c.host_addr_len = sizeof(punched_host);

        auto punch_fcn = reinterpret_cast<bool(*)(Holepunch_config *)>(
                        const_cast<void *>(
                                load_library("udp_holepunch", LIBRARY_CLASS_UNDEFINED, HOLEPUNCH_ABI_VERSION)));

        if(!punch_fcn){
                log_msg(LOG_LEVEL_ERROR, "Failed to load holepunching module\n");
                return -EXIT_FAILURE;
        }

        if(!punch_fcn(&punch_c)){
                log_msg(LOG_LEVEL_ERROR, "Hole punching failed.\n");
                return -EXIT_FAILURE;
        }

        log_msg(LOG_LEVEL_INFO, "[holepunch] remote: %s\n rx: %d\n tx: %d\n",
                        punched_host, video->rx_port, video->tx_port);
        opt->rxtx.receiver = punched_host;
        return 0;
#endif //HAVE_LIBJUICE
}

/// @returns 0 - OK; 1 - also ok but exit (help printed); -RC - exit with RC
static int
adjust_params(struct ug_options *opt)
{
        struct rxtx_medium_params *audio = &opt->rxtx.medium[TX_MEDIA_AUDIO];
        struct rxtx_medium_params *video = &opt->rxtx.medium[TX_MEDIA_VIDEO];
        if (opt->is_server) {
                set_commandline_param("udp-disable-multi-socket", "");
                if (opt->rxtx.receiver != nullptr) {
                        if (strcmp(opt->rxtx.receiver, "help") == 0) {
                                ultragrid_rtp_server_mode_help();
                                return 1;
                        }
                        LOG(LOG_LEVEL_ERROR) << "Receiver must not be given in server mode!\n";
                        return -EXIT_FAIL_USAGE;
                }
                opt->rxtx.receiver = IN6_BLACKHOLE_SERVER_MODE_STR;
                if (strcmp(opt->requested_display, "none") == 0 && strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) != 0) {
                        opt->requested_display = "dummy";
                }
                if (strcmp(opt->audio.recv_cfg, "none") == 0 && strcmp(opt->audio.send_cfg, "none") != 0) {
                        opt->audio.recv_cfg = "dummy";
                }
        }
        if (opt->is_client) {
                set_commandline_param("udp-disable-multi-socket", "");
                if (opt->rxtx.receiver == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "Server address required in client mode!\n";
                        return -EXIT_FAIL_USAGE;
                }
                if (strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) == 0 && strcmp(opt->requested_display, "none") != 0) {
                        vidcap_params_set_device(opt->vidcap_params_tail, "testcard:2:1:1:UYVY");
                }
                if (strcmp("none", opt->audio.send_cfg) == 0 && strcmp("none", opt->audio.recv_cfg) != 0) {
                        set_audio_capture_format("sample_rate=1");
                        opt->audio.send_cfg = "testcard:frames=1";
                }
        }

        if (opt->rxtx.receiver == nullptr) {
                opt->rxtx.receiver = "localhost";
        }

        if (!is_ipv6_supported()) {
                LOG(LOG_LEVEL_WARNING) << "IPv6 support missing, setting IPv4-only mode.\n";
                opt->rxtx.force_ip_version = 4;
        }

        if(opt->nat_traverse_config && strncmp(opt->nat_traverse_config, "holepunch", strlen("holepunch")) == 0){
                int rc = adjust_params_holepunch(opt);
                if (rc != 0) {
                        return rc;
                }
        }

        if (strcmp("none", opt->audio.recv_cfg) != 0) {
                audio->rxtx_mode =
                    (enum rxtx_mode)(audio->rxtx_mode | MODE_RECEIVER);
        }

        if (strcmp("none", opt->audio.send_cfg) != 0) {
                audio->rxtx_mode =
                    (enum rxtx_mode)(audio->rxtx_mode | MODE_SENDER);
        }

        if (strcmp("none", opt->requested_display) != 0) {
                video->rxtx_mode =
                    (enum rxtx_mode)(video->rxtx_mode | MODE_RECEIVER);
        }
        if (strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) != 0) {
                video->rxtx_mode =
                    (enum rxtx_mode)(video->rxtx_mode | MODE_SENDER);
        }

        return 0;
}

/// warn about unused options or not recommended options or agruments
static void
validate_params(struct ug_options *opt)
{
        struct rxtx_medium_params *video = &opt->rxtx.medium[TX_MEDIA_VIDEO];
        if (opt->vidcap_params_head == opt->vidcap_params_tail) {
                if (strlen(opt->rxtx.video_compression) == 0) {
                        MSG(WARNING,
                            "Video compression set but no vidcap given!\n");
                }
                if (vidcap_params_get_capture_filter(opt->vidcap_params_tail) != nullptr) {
                        MSG(WARNING, "Video capture filter specified but not capturing!\n");
                }
                if (strcmp(video->fec, "none") != 0) {
                        MSG(WARNING, "Video FEC specified but not capturing!\n");
                }
        }

        if (strcmp(opt->requested_display, "none") == 0) {
                if (opt->postprocess != nullptr) {
                        MSG(WARNING, "Video postprocess specified without a display!\n");
                }
        }

        if (strcmp(opt->audio.send_cfg, ug_options().audio.send_cfg) == 0) {
                if (strcmp(opt->rxtx.medium[TX_MEDIA_AUDIO].fec,
                           ug_options().rxtx.medium[TX_MEDIA_AUDIO].fec) != 0) {
                        MSG(WARNING,
                            "Audio FEC specified but not capturing!\n");
                }
                if (opt->audio.codec_cfg != nullptr) {
                        MSG(WARNING,
                            "Audio compression specified but not capturing!\n");
                }
        }
}

static const char *
mtu_to_str(int mtu, size_t buflen, char *buf)
{
        if (mtu == 0) {
                return "undefined";
        }
        (void) snprintf(buf, buflen, "%d B", mtu);
        return buf;
}

static const char *
port_to_str(int port, size_t buflen, char *buf)
{
        switch (port) {
        case -1:
                return "undefined";
        case 0:
                return "dynamic";
        default:
                (void) snprintf(buf, buflen, "%d", port);
                return buf;
        }
}

#define EXIT(expr) { int rc = expr; common_cleanup(init); return rc; }
#define RET_TO_RC(ret) (ret < 0 ? -ret : EXIT_SUCCESS)

int main(int argc, char *argv[])
{

        struct init_data *init = nullptr;
#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        ug_options opt{};
        struct rxtx_medium_params *audio = &opt.rxtx.medium[TX_MEDIA_AUDIO];
        struct rxtx_medium_params *video = &opt.rxtx.medium[TX_MEDIA_VIDEO];

        pthread_t capture_thread_id  = PTHREAD_NULL;

        unsigned display_flags = 0;
        struct control_state *control = NULL;
        int ret;

        struct ug_nat_traverse *nat_traverse = nullptr;

#ifndef _WIN32
        signal(SIGQUIT, crash_signal_handler);
#endif
        signal(SIGABRT, crash_signal_handler);
        signal(SIGFPE, crash_signal_handler);
        signal(SIGILL, crash_signal_handler);
        signal(SIGSEGV, crash_signal_handler);

        if ((init = common_preinit(argc, argv)) == nullptr) {
                log_msg(LOG_LEVEL_FATAL, "common_preinit() failed!\n");
                EXIT(EXIT_FAILURE);
        }

        struct state_uv uv{};
        keyboard_control kc{&uv.root_module};
        const bool show_help = tok_in_argv(uv_argv, "help");

        print_version();
        printf("\n");

        if (int ret = parse_options(argc, argv, &opt)) {
                EXIT(ret < 0 ? -ret : EXIT_SUCCESS);
        }

        if (!show_help) {
                validate_params(&opt);
        }
        if (int ret = adjust_params(&opt)) {
                EXIT(RET_TO_RC(ret));
        }

        opt.audio.parent = &uv.root_module;
        opt.rxtx.parent = &uv.root_module;

        struct exporter *exporter =
            export_init(&uv.root_module, opt.export_opts, opt.should_export);
        if (exporter == nullptr) {
                log_msg(LOG_LEVEL_ERROR, "Export initialization failed.\n");
                EXIT(EXIT_FAILURE);
        }
        opt.audio.exporter = opt.rxtx.video_exporter = exporter;

        if (control_init(opt.control_port, opt.connection_type, &control,
                         &uv.root_module, opt.rxtx.force_ip_version) != 0) {
                LOG(LOG_LEVEL_FATAL) << "Error: Unable to initialize remote control!\n";
                EXIT(EXIT_FAIL_CONTROL_SOCK);
        }

        if(!opt.nat_traverse_config
                        || strncmp(opt.nat_traverse_config, "holepunch", strlen("holepunch")) != 0){
                nat_traverse = start_nat_traverse(
                    opt.nat_traverse_config, opt.rxtx.receiver,
                    video->rx_port, audio->rx_port);
                if(!nat_traverse){
                        exit_uv(1);
                        goto cleanup;
                }
        }

        display_flags |= audio_get_display_flags(opt.audio.recv_cfg);

        // Display initialization should be prior to modules that may use graphic card (eg. GLSL) in order
        // to initialize shared resource (X display) first
        ret =
             initialize_video_display(&uv.root_module, opt.requested_display, opt.display_cfg, display_flags, opt.postprocess, &uv.display_device);
        if (ret < 0) {
                printf("Unable to open display device: %s\n",
                       opt.requested_display);
                exit_uv(EXIT_FAIL_DISPLAY);
                goto cleanup;
        } else if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }
        log_msg(LOG_LEVEL_DEBUG, "Display initialized-%s\n", opt.requested_display);

        ret = initialize_video_capture(&uv.root_module, opt.vidcap_params_head, &uv.capture_device);
        if (ret < 0) {
                printf("Unable to open capture device: %s\n",
                                vidcap_params_get_driver(opt.vidcap_params_head));
                exit_uv(EXIT_FAIL_CAPTURE);
                goto cleanup;
        } else if(ret > 0) {
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }
        log_msg(LOG_LEVEL_DEBUG, "Video capture initialized-%s\n", vidcap_params_get_driver(opt.vidcap_params_head));

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
#ifndef _WIN32
        signal(SIGHUP, signal_handler);
        signal(SIGPIPE, signal_handler);
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

        opt.rxtx.capture_device = uv.capture_device; // iHDTV
        opt.rxtx.display_device = uv.display_device; // UltraGrid RTP, iHDTV

        ret = rxtx_init(opt.net_protocol, &opt.rxtx, &uv.state_rxtx);
        if (ret != 0) {
                exit_uv(ret < 0 ? EXIT_FAILURE : EXIT_SUCCESS);
                goto cleanup;
        }

        opt.audio.rxtx    = uv.state_rxtx;
        opt.audio.display = uv.display_device;
        if (opt.audio.codec_cfg == nullptr) {
                opt.audio.codec_cfg = strlen(opt.rxtx.audio_compression) > 0
                                          ? opt.rxtx.audio_compression
                                          : "PCM";
        }
        ret               = audio_init(&uv.audio, &opt.audio);
        if (ret != 0) {
                exit_uv(ret < 0 ? EXIT_FAIL_AUDIO : 0);
                goto cleanup;
        }

        color_printf("\n");
        char buf[128];
        color_printf(TBOLD("Display device   :") " %s\n", opt.requested_display);
        color_printf(TBOLD("Capture device   :") " %s\n", vidcap_params_get_driver(opt.vidcap_params_head));
        color_printf(TBOLD("Audio capture    :") " %s\n", opt.audio.send_cfg);
        color_printf(TBOLD("Audio playback   :") " %s\n", opt.audio.recv_cfg);
        color_printf(TBOLD("MTU              :") " %s\n", mtu_to_str(opt.rxtx.mtu, sizeof buf, buf));
        color_printf(TBOLD("Video compression:") " %s\n", opt.rxtx.video_compression);
        color_printf(TBOLD("Audio codec      : ") "%s\n", opt.audio.codec_cfg);
        color_printf(TBOLD("Network port     : ") "%s\n", port_to_str(opt.rxtx.port_base, sizeof buf, buf));
        color_printf(TBOLD("Network protocol : ") "%s\n", rxtx_get_proto_long_name(opt.net_protocol));
        color_printf(TBOLD("Audio FEC        : ") "%s\n", audio->fec);
        color_printf(TBOLD("Video FEC        : ") "%s\n", video->fec);
        color_printf("\n");

        if (video->rxtx_mode & MODE_SENDER) {
                if (pthread_create(&capture_thread_id, NULL, capture_thread,
                                   (void *) &uv) != 0) {
                        perror("Unable to create capture thread!\n");
                        exit_uv(EXIT_FAILURE);
                        goto cleanup;
                }
        }

        if (opt.requested_capabilities != nullptr) {
                print_capabilities(opt.requested_capabilities);
                exit_uv(EXIT_SUCCESS);
                goto cleanup;
        }

        audio_start(uv.audio);

        control_start(control);
        kc.start();

        display_run_mainloop(uv.display_device);

cleanup:
        if (!pthread_equal(capture_thread_id, PTHREAD_NULL)) {
                pthread_join(capture_thread_id, NULL);
        }

        /* also wait for audio threads */
        audio_join(uv.audio);
        if (uv.state_rxtx) {
                rxtx_join(uv.state_rxtx);
        }

        export_destroy(exporter);

        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
#ifndef _WIN32
        signal(SIGHUP, SIG_DFL);
        signal(SIGALRM, hang_signal_handler);
#endif
        alarm(5); // prevent exit hangs

        rxtx_destroy(uv.state_rxtx);
        audio_done(uv.audio);

        if (uv.capture_device)
                vidcap_done(uv.capture_device);
        if (uv.display_device)
                display_done(uv.display_device);

        stop_nat_traverse(nat_traverse);

        kc.stop();
        control_done(control);

        common_cleanup(init);

        printf("Exit\n");

        return get_exit_status(&uv.root_module);
}

/* vim: set expandtab sw=8: */

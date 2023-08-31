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
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <array>
#include <chrono>
#include <cstdlib>
#ifndef _WIN32
#include <execinfo.h>
#endif // defined WIN32
#include <getopt.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string.h>
#include <thread>
#include <tuple>

#include "compat/misc.h"
#include "control_socket.h"
#include "cuda_wrapper.h"
#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "playback.h"
#include "rtp/rtp.h"
#include "rtsp/rtsp_utils.h"
#include "tv.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/nat.h"
#include "utils/net.h"
#include "utils/sdp.h"
#include "utils/string.h"
#include "utils/string_view_utils.hpp"
#include "utils/thread.h"
#include "utils/wait_obj.h"
#include "utils/udp_holepunch.h"
#include "video.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_compress.h"
#include "export.h"
#include "video_rxtx.h"
#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/audio_playback.h"
#include "audio/codec.h"
#include "audio/utils.h"

#define MOD_NAME                "[main] "
#define PORT_BASE               5004

#define DEFAULT_AUDIO_FEC       "none"
static constexpr const char *DEFAULT_VIDEO_COMPRESSION = "none";
static constexpr const char *DEFAULT_AUDIO_CODEC = "PCM";

#define OPT_AUDIO_CAPTURE_CHANNELS (('a' << 8) | 'c')
#define OPT_AUDIO_CHANNEL_MAP (('a' << 8) | 'm')
#define OPT_AUDIO_DELAY (('A' << 8) | 'D')
#define OPT_AUDIO_HOST (('A' << 8) | 'H')
#define OPT_AUDIO_PROTOCOL (('A' << 8) | 'P')
#define OPT_AUDIO_SCALE (('a' << 8) | 's')
#define OPT_AUDIO_FILTER (('a' << 8) | 'f')
#define OPT_CAPABILITIES (('C' << 8) | 'C')
#define OPT_CONTROL_PORT (('C' << 8) | 'P')
#define OPT_ECHO_CANCELLATION (('E' << 8) | 'C')
#define OPT_LIST_MODULES (('L' << 8) | 'M')
#define OPT_MCAST_IF (('M' << 8) | 'I')
#define OPT_PIX_FMTS (('P' << 8) | 'F')
#define OPT_PIXFMT_CONV_POLICY (('P' << 8) | 'C')
#define OPT_RTSP_SERVER (('R' << 8) | 'S')
#define OPT_VIDEO_CODECS (('V' << 8) | 'C')
#define OPT_VIDEO_PROTOCOL (('V' << 8) | 'P')
#define OPT_WINDOW_TITLE (('W' << 8) | 'T')

#define MAX_CAPTURE_COUNT 17

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
        }

        struct vidcap *capture_device{};
        struct display *display_device{};

        struct state_audio *audio{};

        struct module root_module;

        video_rxtx *state_video_rxtx{};

        static void should_exit_capture_callback(void *udata) {
                auto *s = (state_uv *) udata;
                s->should_exit_capture = true;
        }
        bool should_exit_capture = false;
        static constexpr uint32_t state_magic = to_fourcc('U', 'G', 'S', 'T');
};

static void signal_handler(int signal)
{
        if (log_level >= LOG_LEVEL_VERBOSE) {
                char buf[128];
                char *ptr = buf;
                char *ptr_end = buf + sizeof buf;
                strappend(&ptr, ptr_end, "Caught signal ");
                if (signal / 10) {
                        *ptr++ = '0' + signal/10;
                }
                *ptr++ = '0' + signal%10;
                *ptr++ = '\n';
                write_all(ptr - buf, buf);
        }
        exit_uv(0);
}

static void print_help_item(const string &name, const vector<string> &help) {
        int help_lines = 0;

        col() << "\t" << TBOLD(<< name <<);

        for (auto const &line : help) {
                int spaces = help_lines == 0 ? 31 - (int) name.length() : 39;
                for (int i = 0; i < max(spaces, 0) + 1; ++i) {
                        cout << " ";
                }
                cout << line << "\n";
                help_lines += 1;
        }

        if (help_lines == 0) {
                cout << "\n";
        }
        cout << "\n";
}

static void usage(const char *exec_path, bool full = false)
{
        col() << "Usage: " << SBOLD(SRED((exec_path ? exec_path : "<executable_path>")) << " [options] address\n\n");
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
                print_help_item("--verbose[=<level>]", {"print verbose messages (optionally specify level [0-" + to_string(LOG_LEVEL_MAX) + "])"});
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
                print_help_item("-p <postprocess> | help", {"postprocess module"});
                print_help_item("-T, --ttl <num>", {"Use specified TTL for multicast/unicast (0..255, -1 for default)"});
        }
        print_help_item("-f [A:|V:]<settings>", {"FEC settings (audio or video) - use",
                        "\"none\", \"mult:<nr>\",", "\"ldgm:<max_expected_loss>%\" or", "\"ldgm:<k>:<m>:<c>\"",
                        "\"rs:<k>:<n>\""});
        if (full) {
                print_help_item("-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]", { "",
                                "<port> is base port number, also 3",
                                "subsequent ports can be used for RTCP",
                                "and audio streams. Default: " + to_string(PORT_BASE) + ".",
                                "You can also specify all two or four", "ports directly."});
                print_help_item("-l <limit_bitrate> | unlimited | auto", {"limit sending bitrate",
                                "to <limit_bitrate> (with optional k/M/G suffix)"});
                print_help_item("-A <address>", {"audio destination address",
                                "If not specified, will use same as for video"});
        }
        print_help_item("-a, --audio-capture-format <fmt> | help", {"format of captured audio"});
        if (full) {
                print_help_item("--audio-channel-map <mapping> | help", {});
                print_help_item("--audio-filter <filter>[:<config>][#<filter>[:<config>]]...", {});
        }
        print_help_item("--audio-codec <codec>[:sample_rate=<sr>][:bitrate=<br>] | help", {"audio codec"});
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
        print_help_item("address", {"destination address"});
        printf("\n");
}

static void print_fps(const char *prefix, steady_clock::time_point *t0, int *frames) {
        if (prefix == nullptr) {
                return;
        }
        *frames += 1;
        steady_clock::time_point t1 = steady_clock::now();
        double seconds = duration_cast<duration<double>>(t1 - *t0).count();
        if (seconds >= 5.0) {
                double fps = *frames / seconds;
                log_msg(LOG_LEVEL_INFO, TERM_BOLD TERM_BG_BLACK TERM_FG_BRIGHT_GREEN "%s" TERM_RESET " %d frames in %g seconds = " TBOLD("%g FPS") "\n", prefix, *frames, seconds, fps);
                *t0 = t1;
                *frames = 0;
        }
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
        char *print_fps_prefix = vidcap_get_fps_print_prefix(uv->capture_device) ? strdupa(vidcap_get_fps_print_prefix(uv->capture_device)) : NULL;
        if (print_fps_prefix && print_fps_prefix[strlen(print_fps_prefix) - 1] == ' ') { // trim trailing ' '
                print_fps_prefix[strlen(print_fps_prefix) - 1] = '\0';
        }

        while (!uv->should_exit_capture) {
                /* Capture and transmit video... */
                struct audio_frame *audio = nullptr;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);

                if (audio != nullptr) {
                        audio_sdi_send(uv->audio, audio);
                        AUDIO_FRAME_DISPOSE(audio);
                }

                if (tx_frame != NULL) {
                        print_fps(print_fps_prefix, &t0, &frames);
                        //tx_frame = vf_get_copy(tx_frame);
                        bool wait_for_cur_uncompressed_frame;
                        shared_ptr<video_frame> frame;
                        if (!tx_frame->callbacks.dispose) {
                                wait_obj_reset(wait_obj);
                                wait_for_cur_uncompressed_frame = true;
                                frame = shared_ptr<video_frame>(tx_frame, [wait_obj](struct video_frame *) {
                                                        wait_obj_notify(wait_obj);
                                                });
                        } else {
                                wait_for_cur_uncompressed_frame = false;
                                frame = shared_ptr<video_frame>(tx_frame, tx_frame->callbacks.dispose);
                        }

                        uv->state_video_rxtx->send(std::move(frame)); // std::move really important here (!)

                        // wait for frame frame to be processed, eg. by compress
                        // or sender (uncompressed video). Grab invalidates previous frame
                        // (if not defined dispose function).
                        if (wait_for_cur_uncompressed_frame) {
                                wait_obj_wait(wait_obj);
                                tx_frame->callbacks.dispose = NULL;
                                tx_frame->callbacks.dispose_udata = NULL;
                        }
                }
        }

        wait_obj_done(wait_obj);

        return NULL;
}

static bool parse_bitrate(char *optarg, long long int *bitrate) {
        map<string, long long int> bitrate_spec_map = {
                { "auto", RATE_AUTO },
                { "dynamic", RATE_DYNAMIC },
                { "unlimited", RATE_UNLIMITED },
        };

        if (auto it = bitrate_spec_map.find(optarg); it != bitrate_spec_map.end()) {
                *bitrate = it->second;
                return true;
        }
        if (strcmp(optarg, "help") == 0) {
#               define NUMERIC_PATTERN "{1-9}{0-9}*[kMG][!][E]"
                col() << "Usage:\n" <<
                        "\tuv " << TERM_BOLD "-l [auto | dynamic | unlimited | " << NUMERIC_PATTERN << "]\n" TERM_RESET <<
                        "where\n"
                        "\t" << TBOLD("auto") << " - spread packets across frame time\n"
                        "\t" << TBOLD("dynamic") << " - similar to \"auto\" but more relaxed - occasional huge frame can spread 1.5x frame time (default)\n"
                        "\t" << TBOLD("unlimited") << " - send packets at a wire speed (in bursts)\n"
                        "\t" << TBOLD(NUMERIC_PATTERN) << " - send packets at most at specified bitrate\n\n" <<
                        TBOLD("Notes: ") << "Use an exclamation mark to indicate intentionally very low bitrate. 'E' to use the value as a fixed bitrate, not cap /i. e. even the frames that may be sent at lower bitrate are sent at the nominal bitrate)\n" <<
                        "\n";
                return true;
        }
        bool force = false, fixed = false;
        for (int i = 0; i < 2; ++i) {
                if (optarg[strlen(optarg) - 1] == '!' ||
                                optarg[strlen(optarg) - 1] == 'E') {
                        if (optarg[strlen(optarg) - 1] == '!') {
                                force = true;
                                optarg[strlen(optarg) - 1] = '\0';
                        }
                        if (optarg[strlen(optarg) - 1] == 'E') {
                                fixed = true;
                                optarg[strlen(optarg) - 1] = '\0';
                        }
                }
        }
        *bitrate = unit_evaluate(optarg);
        if (*bitrate <= 0) {
                log_msg(LOG_LEVEL_ERROR, "Invalid bitrate %s!\n", optarg);
                return false;
        }
        long long mb5 = 5ll * 1000 * 1000, // it'll take 6.4 sec to send 4 MB frame at 5 Mbps
             gb100 = 100ll * 1000 * 1000 * 1000; // traffic shaping to eg. 40 Gbps may make sense
        if ((*bitrate < mb5 || *bitrate > gb100) && !force) {
                log_msg(LOG_LEVEL_WARNING, "Bitrate %lld bps seems to be too %s, use \"-l %s!\" to force if this is not a mistake.\n", *bitrate, *bitrate < mb5 ? "low" : "high", optarg);
                return false;
        }
        if (fixed) {
                *bitrate |= RATE_FLAG_FIXED_RATE;
        }
        return true;
}

/// @retval <0 return code whose absolute value will be returned from main()
/// @retval 0 success
/// @retval 1 device list was printed
static int parse_cuda_device(char *optarg) {
        if(strcmp("help", optarg) == 0) {
#ifdef HAVE_CUDA
                cuda_wrapper_print_devices_info();
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
                cuda_devices[i] = atoi(item);
                optarg = NULL;
                ++i;
        }
        cuda_devices_count = i;
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
                                "\tuv " << TBOLD("-Nholepunch:room=<room>:(server=<host> | coord_srv=<host:port>:stun_srv=<host:port>)[:client_name=<name>] \n") <<
                                "\twhere\n"
                                "\t\t" << TBOLD("server") << " - used if both stun & coord server are on the same host on standard ports (3478, 12558)\n"
                                "\t\t" << TBOLD("room") << " - name of room to join\n"
                                "\t\t" << TBOLD("client_name") << " - name to identify as to the coord server, if not specified hostname is used\n"
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

struct ug_options {
        ug_options() {
                vidcap_params_set_device(vidcap_params_head, "none");
        }
        ~ug_options() {
                while  (vidcap_params_head != nullptr) {
                        struct vidcap_params *next = vidcap_params_get_next(vidcap_params_head);
                        vidcap_params_free_struct(vidcap_params_head);
                        vidcap_params_head = next;
                }
        }
        struct audio_options audio = {
                .host = nullptr,
                .recv_port = -1,
                .send_port = -1,
                .recv_cfg = "none",
                .send_cfg = "none",
                .proto = "ultragrid_rtp",
                .proto_cfg = "",
                .fec_cfg = DEFAULT_AUDIO_FEC,
                .channel_map = nullptr,
                .scale = "mixauto",
                .echo_cancellation = false,
                .codec_cfg = nullptr,
                .filter_cfg = ""
        };
        // NULL terminated array of capture devices
        struct vidcap_params *vidcap_params_head = vidcap_params_allocate();
        struct vidcap_params *vidcap_params_tail = vidcap_params_head;

        const char *display_cfg = "";
        const char *requested_video_fec = "none";
        int port_base = PORT_BASE;
        int requested_ttl = -1;
        int video_rx_port = -1, video_tx_port = -1;

        bool should_export = false;
        char *export_opts = NULL;

        int control_port = 0;
        int connection_type = 0;

        enum video_mode decoder_mode = VIDEO_NORMAL;
        const char *requested_compression = nullptr;

        int force_ip_version = 0;

        const char *requested_mcast_if = NULL;

        unsigned requested_mtu = 0;
        const char *postprocess = NULL;
        const char *requested_display = "none";
        const char *requested_receiver = nullptr;
        const char *requested_encryption = NULL;

        long long int bitrate = RATE_DEFAULT;
        bool is_client = false;
        bool is_server = false;

        const char *requested_capabilities = nullptr;

        const char *video_protocol = "ultragrid_rtp";
        const char *video_protocol_opts = "";

        char *nat_traverse_config = nullptr;

        unsigned int video_rxtx_mode = 0;
};

static int parse_audio_capture(struct ug_options *opt, char *optarg) {
        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                audio_capture_print_help(strcmp(optarg, "full") == 0);
                return 1;
        }
        if (string(opt->audio.send_cfg) != ug_options().audio.send_cfg &&
            !!audio_capture_get_vidcap_flags(opt->audio.send_cfg) !=
                !!audio_capture_get_vidcap_flags(optarg)) {
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
        if (char *tx_port_str = strtok_r(nullptr, ":", &save_ptr)) {
                opt->port_base = stoi(first_val, nullptr, 0);
                opt->video_tx_port = stoi(tx_port_str, nullptr, 0);
                char *tok = nullptr;
                if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                        opt->audio.recv_port = stoi(tok, nullptr, 0);
                        if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                                opt->audio.send_port = stoi(tok, nullptr, 0);
                        } else {
                                usage(uv_argv[0]);
                                return false;
                        }
                }
        } else {
                opt->port_base = stoi(first_val, nullptr, 0);
        }
        if (opt->audio.recv_port < -1 || opt->audio.send_port < -1 || opt->video_rx_port < -1 || opt->video_tx_port < -1 || opt->port_base < -1 ||
                        opt->audio.recv_port > UINT16_MAX || opt->audio.send_port > UINT16_MAX || opt->video_rx_port > UINT16_MAX || opt->video_tx_port > UINT16_MAX || opt->port_base > UINT16_MAX) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Invalid port value, allowed range 1-65535\n";
                return false;
        }
        return true;
}

static bool parse_protocol(int ch, char *optarg, struct ug_options *opt) {
        bool set_audio = true;
        bool set_video = true;
        if (strlen(optarg) > 2 && optarg[1] == ':') {
                set_audio = toupper(optarg[0]) == 'A';
                set_video = toupper(optarg[0]) == 'V';
                if (!set_audio && !set_video) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Wrong protocol setting: " << optarg << "\n";
                        return false;
                }
                optarg += 2;
        }
        char *proto = optarg;
        const char *cfg = "";
        if (strchr(optarg, ':')) {
                char *delim = strchr(optarg, ':');
                *delim = '\0';
                cfg = delim + 1;
        }
        switch (ch) {
                case OPT_AUDIO_PROTOCOL:
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "--audio-protocol deprecated, use '-x A:proto'\n";
                        set_video = false;
                        break;
                case OPT_VIDEO_PROTOCOL:
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "--video-protocol deprecated, use '-x A:proto'\n";
                        set_video = false;
                        break;
        }
        if (strcmp(optarg, "help") == 0 ||
                strcmp(optarg, "fullhelp") == 0) {
                col() << "Specify a transmission protocol.\nUsage:\n";
                col() << SBOLD("\t-x proto") "   - use common protocol for audio and video\n";
                col() << SBOLD("\t-x A:proto") " - use specified audio protocol\n";
                col() << SBOLD("\t-x V:proto") " - use specified audio protocol\n";
                col() << "\nAudio protocol can be one of: " << TBOLD(AUDIO_PROTOCOLS) " (not all must be available)\n";
                video_rxtx::list(strcmp(optarg, "fullhelp") == 0);
                return false;
        }
        if (set_audio) {
                opt->audio.proto = proto;
                opt->audio.proto_cfg = cfg;

        }
        if (set_video) {
                opt->video_protocol = proto;
                opt->video_protocol_opts = cfg;

        }
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
                {"compress",               required_argument, nullptr, 'c'},
                {"display",                required_argument, nullptr, 'd'},
                {"encryption",             required_argument, nullptr, 'e'},
                {"fec",                    required_argument, nullptr, 'f'},
                {"help",                   no_argument,       nullptr, 'h'},
                {"limit-bitrate",          required_argument, nullptr, 'l'},
                {"capture",                required_argument, nullptr, 't'},
                {"mtu",                    required_argument, nullptr, 'm'},
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
                {"mode",                   required_argument, nullptr, 'M'},
                {"nat-traverse",           optional_argument, nullptr, 'N'},
                {"port",                   required_argument, nullptr, 'P'},
                {"server",                 no_argument,       nullptr, 'S'},
                {"ttl",                    required_argument, nullptr, 'T'},
                {"verbose",                optional_argument, nullptr, 'V'},
                {"audio-channel-map",      required_argument, 0, OPT_AUDIO_CHANNEL_MAP},
                {"audio-capture-channels", required_argument, 0, OPT_AUDIO_CAPTURE_CHANNELS},
                {"audio-delay",            required_argument, 0, OPT_AUDIO_DELAY},
                {"audio-host",             required_argument, 0, OPT_AUDIO_HOST},
                {"audio-protocol",         required_argument, 0, OPT_AUDIO_PROTOCOL},
                {"capabilities",           optional_argument, 0, OPT_CAPABILITIES},
                {"control-port",           required_argument, 0, OPT_CONTROL_PORT},
                {"audio-filter",           required_argument, 0, OPT_AUDIO_FILTER},
                {"audio-scale",            required_argument, 0, OPT_AUDIO_SCALE},
                {"echo-cancellation",      no_argument,       0, OPT_ECHO_CANCELLATION},
                {"list-modules",           no_argument,       0, OPT_LIST_MODULES},
                {"mcast-if",               required_argument, 0, OPT_MCAST_IF},
                {"param",                  required_argument, 0, OPT_PARAM},
                {"conv-policy",            required_argument, 0, OPT_PIXFMT_CONV_POLICY},
                {"pix-fmts",               no_argument,       0, OPT_PIX_FMTS},
                {"rtsp-server",            optional_argument, 0, OPT_RTSP_SERVER},
                {"video-codecs",           no_argument,       0, OPT_VIDEO_CODECS},
                {"video-protocol",         required_argument, 0, OPT_VIDEO_PROTOCOL},
                {"window-title",           required_argument, 0, OPT_WINDOW_TITLE},
                {0, 0, 0, 0}
        };
        const char *optstring = "46A:CD:E::F:HI:M:N::P:ST:Va:c:e:f:d:hl:m:p:r:s:t:vx:";

        int ch = 0;
        while ((ch =
                getopt_long(argc, argv, optstring, getopt_options,
                            NULL)) != -1) {
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
                        opt->requested_mtu = atoi(optarg);
                        if (opt->requested_mtu < 576 && optarg[strlen(optarg) - 1] != '!') {
                                log_msg(LOG_LEVEL_WARNING, "MTU %1$u seems to be too low, use \"%1$u!\" to force.\n", opt->requested_mtu);
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'M':
                        opt->decoder_mode = get_video_mode_from_str(optarg);
                        if (opt->decoder_mode == VIDEO_UNKNOWN) {
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
                        opt->requested_compression = optarg;
                        break;
                case OPT_RTSP_SERVER:
                        log_msg(LOG_LEVEL_WARNING, "Option \"--rtsp-server[=args]\" "
                                        "is deprecated and will be removed in future.\n"
                                        "Please use \"-x rtsp[:args]\"instead.\n");
                        opt->video_protocol = "rtsp";
                        opt->video_protocol_opts = optarg ? optarg : "";
                        break;
                case OPT_AUDIO_PROTOCOL:
                case OPT_VIDEO_PROTOCOL:
                case 'x':
                        if (!parse_protocol(ch, optarg, opt)) {
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
                                        opt->audio.fec_cfg = optarg + 2;
                                } else {
                                        opt->requested_video_fec = optarg + 2;
                                }
                        } else {
                                // there should be setting for both audio and video
                                // but we conservativelly expect that the user wants
                                // only vieo and let audio default until explicitly
                                // stated otehrwise
                                opt->requested_video_fec = optarg;
                        }
                        break;
                case 'h':
                        usage(uv_argv[0], false);
                        return 1;
                case 'P':
                        if (!parse_port(optarg, opt)) {
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'l':
                        if (!parse_bitrate(optarg, &opt->bitrate)) {
                                return -EXIT_FAILURE;
                        }
                        if (opt->bitrate == RATE_DEFAULT) {
                                return 1; // help written
                        }
                        break;
                case '4':
                case '6':
                        opt->force_ip_version = ch - '0';
                        break;
                case OPT_AUDIO_CHANNEL_MAP:
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
                case OPT_AUDIO_FILTER:
                        opt->audio.filter_cfg = optarg;
                        break;
                case OPT_ECHO_CANCELLATION:
                        opt->audio.echo_cancellation = true;
                        break;
                case 'H':
                        usage(uv_argv[0], true);
                        return 1;
                case 'D':
                        if (int ret = parse_cuda_device(optarg)) {
                                return ret;
                        }
                        break;
                case OPT_MCAST_IF:
                        opt->requested_mcast_if = optarg;
                        break;
                case OPT_AUDIO_HOST:
                        opt->audio.host = optarg;
                        break;
                case 'E':
                        opt->should_export = true;
                        opt->export_opts = optarg;
                        break;
                case 'I':
                        opt->audio.send_cfg = "embedded";
                        {
                                char dev_string[1024];
                                int ret;
                                if ((ret = playback_set_device(dev_string, sizeof dev_string, optarg)) <= 0) {
                                        return ret == 0 ? 1 : -EXIT_FAIL_USAGE;
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
                        if (!check_audio_codec(optarg)) {
                                LOG(LOG_LEVEL_WARNING)
                                    << MOD_NAME
                                    << "The original semantics of '-A' "
                                       "parameter has changed,\nplease use "
                                       "'--audio-host' to specify audio "
                                       "receiver address.\n";
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'F':
                        vidcap_params_set_capture_filter(opt->vidcap_params_tail, optarg);
                        break;
                case 'e':
                        opt->requested_encryption = optarg;
                        break;
                case OPT_CONTROL_PORT:
                        if (!parse_control_port(optarg, opt)) {
                                return -EXIT_FAIL_USAGE;
                        }
                        break;
                case 'V':
                        break; // already handled in common_preinit()
                case OPT_WINDOW_TITLE:
                        log_msg(LOG_LEVEL_WARNING, "Deprecated option used, please use "
                                        "--param window-title=<title>\n");
                        commandline_params["window-title"] = optarg;
                        break;
                case OPT_CAPABILITIES:
                        opt->requested_capabilities = optarg ? optarg : "";
                        break;
                case OPT_AUDIO_DELAY:
                        set_audio_delay(stoi(optarg));
                        break;
                case OPT_LIST_MODULES:
                        return list_all_modules() ? 1 : -EXIT_FAILURE;
                case OPT_PARAM:
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
                        opt->requested_ttl = stoi(optarg);
                        break;
                case '?':
                default:
                        usage(uv_argv[0]);
                        return -EXIT_FAIL_USAGE;
                }
        }

        argc -= optind;
        argv += optind;

        if (argc > 1) {
                log_msg(LOG_LEVEL_ERROR, "Multiple receivers given!\n");
                usage(uv_argv[0]);
                return -EXIT_FAIL_USAGE;
        }

        if (argc > 0) {
                opt->requested_receiver = argv[0];
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
                if (strcmp(e.what(), "stoi") != 0 &&
                    strcmp(e.what(), "stod") != 0) {
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

static int adjust_params(struct ug_options *opt) {
        unsigned int audio_rxtx_mode = 0;
        if (opt->is_server) {
                commandline_params["udp-disable-multi-socket"] = string();
                if (opt->requested_receiver != nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "Receiver must not be given in server mode!\n";
                        return EXIT_FAIL_USAGE;
                }
                opt->requested_receiver = IN6_BLACKHOLE_STR;
                if (strcmp(opt->requested_display, "none") == 0 && strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) != 0) {
                        opt->requested_display = "dummy";
                }
                if (strcmp(opt->audio.recv_cfg, "none") == 0 && strcmp(opt->audio.send_cfg, "none") != 0) {
                        opt->audio.recv_cfg = "dummy";
                }
        }
        if (opt->is_client) {
                commandline_params["udp-disable-multi-socket"] = string();
                if (opt->requested_receiver == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "Server address required in client mode!\n";
                        return EXIT_FAIL_USAGE;
                }
                if (strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) == 0 && strcmp(opt->requested_display, "none") != 0) {
                        vidcap_params_set_device(opt->vidcap_params_tail, "testcard:2:1:1:UYVY");
                }
                if (strcmp("none", opt->audio.send_cfg) == 0 && strcmp("none", opt->audio.recv_cfg) != 0) {
                        set_audio_capture_format("sample_rate=1");
                        opt->audio.send_cfg = "testcard:frames=1";
                }
        }

        if (opt->requested_receiver == nullptr) {
                opt->requested_receiver = "localhost";
        }

        if (opt->audio.host == nullptr) {
                opt->audio.host = opt->requested_receiver;
        }

        if (!is_ipv6_supported()) {
                LOG(LOG_LEVEL_WARNING) << "IPv6 support missing, setting IPv4-only mode.\n";
                opt->force_ip_version = 4;
        }

        // default values for different RXTX protocols
        if (strcasecmp(opt->video_protocol, "rtsp") == 0 || strcasecmp(opt->video_protocol, "sdp") == 0) {
                if (opt->requested_compression == nullptr) {
                        if (strcasecmp(opt->video_protocol, "rtsp") == 0) {
                                opt->requested_compression = DEFAULT_SDP_COMPRESSION;
                        } else {
                                opt->requested_compression = "none"; // will be set later by h264_sdp_video_rxtx::send_frame()
                        }
                }
                if (opt->force_ip_version == 0 && strcasecmp(opt->video_protocol, "rtsp") == 0) {
                        opt->force_ip_version = 4;
                }
        } else {
                if (opt->requested_compression == nullptr) {
                        opt->requested_compression = DEFAULT_VIDEO_COMPRESSION;
                }
        }

        if (opt->audio.codec_cfg == nullptr) {
                if (strcasecmp(opt->audio.proto, "rtsp") == 0 || strcasecmp(opt->audio.proto, "sdp") == 0) {
                        opt->audio.codec_cfg = "Opus:sample_rate=48000";
                } else {
                        opt->audio.codec_cfg = DEFAULT_AUDIO_CODEC;
                }
        }

        if(opt->nat_traverse_config && strncmp(opt->nat_traverse_config, "holepunch", strlen("holepunch")) == 0){
#ifndef HAVE_LIBJUICE
                log_msg(LOG_LEVEL_ERROR, "Ultragrid was compiled without holepunch support\n");
                return EXIT_FAILURE;
#else
                static char punched_host[512];
                Holepunch_config punch_c = {};

                if(!parse_holepunch_conf(opt->nat_traverse_config, &punch_c)){
                        return EXIT_FAILURE;
                }

                commandline_params["udp-disable-multi-socket"] = string();

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

                punch_c.video_rx_port = &opt->video_rx_port;
                punch_c.video_tx_port = &opt->video_tx_port;
                punch_c.audio_rx_port = &opt->audio.recv_port;
                punch_c.audio_tx_port = &opt->audio.send_port;

                punch_c.host_addr = punched_host;
                punch_c.host_addr_len = sizeof(punched_host);

                auto punch_fcn = reinterpret_cast<bool(*)(Holepunch_config *)>(
                                const_cast<void *>(
                                        load_library("udp_holepunch", LIBRARY_CLASS_UNDEFINED, HOLEPUNCH_ABI_VERSION)));

                if(!punch_fcn){
                        log_msg(LOG_LEVEL_ERROR, "Failed to load holepunching module\n");
                        return EXIT_FAILURE;
                }

                if(!punch_fcn(&punch_c)){
                        log_msg(LOG_LEVEL_ERROR, "Hole punching failed.\n");
                        return EXIT_FAILURE;
                }

                log_msg(LOG_LEVEL_INFO, "[holepunch] remote: %s\n rx: %d\n tx: %d\n",
                                punched_host, opt->video_rx_port, opt->video_tx_port);
                opt->requested_receiver = punched_host;
                opt->audio.host = punched_host;
#endif //HAVE_LIBJUICE
        }


        if (strcmp("none", opt->audio.recv_cfg) != 0) {
                audio_rxtx_mode |= MODE_RECEIVER;
        }

        if (strcmp("none", opt->audio.send_cfg) != 0) {
                audio_rxtx_mode |= MODE_SENDER;
        }

        if (strcmp("none", opt->requested_display) != 0) {
                opt->video_rxtx_mode |= MODE_RECEIVER;
        }
        if (strcmp("none", vidcap_params_get_driver(opt->vidcap_params_head)) != 0) {
                opt->video_rxtx_mode |= MODE_SENDER;
        }

        // use dyn ports if sending only to ourselves or neither sending nor receiving
        const unsigned mode_both = MODE_RECEIVER | MODE_SENDER;
        if (is_host_loopback(opt->requested_receiver) && ((opt->video_rxtx_mode & mode_both) == mode_both || (opt->video_rxtx_mode & mode_both) == 0)
                        && (opt->video_rx_port == -1 && opt->video_tx_port == -1)) {
                opt->video_rx_port = opt->video_tx_port = 0;
        }
        if (is_host_loopback(opt->requested_receiver) && ((audio_rxtx_mode & mode_both) == mode_both || (audio_rxtx_mode & mode_both) == 0)
                        && (opt->audio.recv_port == -1 && opt->audio.send_port == -1)) {
                opt->audio.recv_port = opt->audio.send_port = 0;
        }

        if (opt->video_rx_port == -1) {
                if ((opt->video_rxtx_mode & MODE_RECEIVER) == 0U) {
                        // do not occupy recv port if we are not receiving (note that this disables communication with
                        // our receiver, because RTCP ports are changed as well)
                        opt->video_rx_port = 0;
                } else {
                        opt->video_rx_port = opt->port_base;
                }
        }

        if (opt->video_tx_port == -1) {
                if ((opt->video_rxtx_mode & MODE_SENDER) == 0U) {
                        opt->video_tx_port = 0; // does not matter, we are receiver
                } else {
                        opt->video_tx_port = opt->port_base;
                }
        }

        if (opt->audio.recv_port == -1) {
                if ((audio_rxtx_mode & MODE_RECEIVER) == 0U) {
                        // do not occupy recv port if we are not receiving (note that this disables communication with
                        // our receiver, because RTCP ports are changed as well)
                        opt->audio.recv_port = 0;
                } else {
                        opt->audio.recv_port = opt->video_rx_port ? opt->video_rx_port + 2 : opt->port_base + 2;
                }
        }

        if (opt->audio.send_port == -1) {
                if ((audio_rxtx_mode & MODE_SENDER) == 0U) {
                        opt->audio.send_port = 0;
                } else {
                        opt->audio.send_port = opt->video_tx_port ? opt->video_tx_port + 2 : opt->port_base + 2;
                }
        }

        // If we are sure that this UltraGrid is sending to itself we can optimize some parameters
        // (aka "-m 9000 -l unlimited"). If ports weren't equal it is possibile that we are sending
        // to a reflector, thats why we require equal ports (we are a receiver as well).
        if (is_host_loopback(opt->requested_receiver)
                        && (opt->video_rx_port == opt->video_tx_port || opt->video_tx_port == 0)
                        && (opt->audio.recv_port == opt->audio.send_port || opt->audio.send_port == 0)) {
                opt->requested_mtu = opt->requested_mtu == 0 ? min(RTP_MAX_MTU, 65535) : opt->requested_mtu;
                opt->bitrate = opt->bitrate == RATE_DEFAULT ? RATE_UNLIMITED : opt->bitrate;
        } else {
                opt->requested_mtu = opt->requested_mtu == 0 ? 1500 : opt->requested_mtu;
                opt->bitrate = opt->bitrate == RATE_DEFAULT ? RATE_DYNAMIC : opt->bitrate;
        }

        if((strcmp("none", opt->audio.send_cfg) != 0 || strcmp("none", opt->audio.recv_cfg) != 0) && strcmp(opt->video_protocol, "rtsp") == 0){
            //TODO: to implement a high level rxtx struct to manage different standards (i.e.:H264_STD, VP8_STD,...)
            if (strcmp(opt->audio.proto, "rtsp") != 0) {
                    LOG(LOG_LEVEL_WARNING) << "Using RTSP for video but not for audio is not recommended. Consider adding '--audio-protocol rtsp'.\n";
            }
        }
        if (strcmp(opt->audio.proto, "rtsp") == 0 && strcmp(opt->video_protocol, "rtsp") != 0) {
                LOG(LOG_LEVEL_WARNING) << "Using RTSP for audio but not for video is not recommended and might not work.\n";
        }

        return 0;
}

static bool help_in_argv(char **argv) {
        while (*argv) {
                if (strstr(*argv, "help")) {
                        return true;
                }
                argv++;
        }
        return false;
}

#define EXIT(expr) { int rc = expr; common_cleanup(init); return rc; }

int main(int argc, char *argv[])
{

        struct init_data *init = nullptr;
#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        ug_options opt{};

        pthread_t receiver_thread_id,
                  capture_thread_id;
        bool receiver_thread_started = false,
             capture_thread_started = false;
        unsigned display_flags = 0;
        struct control_state *control = NULL;
        struct exporter *exporter = NULL;
        int ret;

        time_ns_t start_time = get_time_in_ns();

        struct ug_nat_traverse *nat_traverse = nullptr;

#ifndef WIN32
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
        bool show_help = help_in_argv(uv_argv);

        print_version();
        printf("\n");

        if (int ret = parse_options(argc, argv, &opt)) {
                EXIT(ret < 0 ? -ret : EXIT_SUCCESS);
        }

        if (int ret = adjust_params(&opt)) {
                EXIT(ret);
        }

        if (!show_help) {
                col() << TBOLD("Display device   : ") << opt.requested_display << "\n";
                col() << TBOLD("Capture device   : ") << vidcap_params_get_driver(opt.vidcap_params_head) << "\n";
                col() << TBOLD("Audio capture    : ") << opt.audio.send_cfg << "\n";
                col() << TBOLD("Audio playback   : ") << opt.audio.recv_cfg << "\n";
                col() << TBOLD("MTU              : ") << opt.requested_mtu << " B\n";
                col() << TBOLD("Video compression: ") << opt.requested_compression << "\n";
                col() << TBOLD("Audio codec      : ") << get_name_to_audio_codec(get_audio_codec(opt.audio.codec_cfg)) << "\n";
                col() << TBOLD("Network protocol : ") << video_rxtx::get_long_name(opt.video_protocol) << "\n";
                col() << TBOLD("Audio FEC        : ") << opt.audio.fec_cfg << "\n";
                col() << TBOLD("Video FEC        : ") << opt.requested_video_fec << "\n";
                col() << "\n";
        }

        exporter = export_init(&uv.root_module, opt.export_opts, opt.should_export);
        if (!exporter) {
                log_msg(LOG_LEVEL_ERROR, "Export initialization failed.\n");
                EXIT(EXIT_FAILURE);
        }

        if (control_init(opt.control_port, opt.connection_type, &control, &uv.root_module, opt.force_ip_version) != 0) {
                LOG(LOG_LEVEL_FATAL) << "Error: Unable to initialize remote control!\n";
                EXIT(EXIT_FAIL_CONTROL_SOCK);
        }

        if(!opt.nat_traverse_config
                        || strncmp(opt.nat_traverse_config, "holepunch", strlen("holepunch")) != 0){
                nat_traverse = start_nat_traverse(opt.nat_traverse_config, opt.requested_receiver, opt.video_rx_port, opt.audio.recv_port);
                if(!nat_traverse){
                        exit_uv(1);
                        goto cleanup;
                }
        }

        ret = audio_init (&uv.audio, &uv.root_module, &opt.audio,
                        opt.requested_encryption,
                        opt.force_ip_version, opt.requested_mcast_if,
                        opt.bitrate, &audio_offset, start_time,
                        opt.requested_mtu, opt.requested_ttl, exporter);
        if (ret != 0) {
                exit_uv(ret < 0 ? EXIT_FAIL_AUDIO : 0);
                goto cleanup;
        }

        display_flags |= audio_get_display_flags(uv.audio);

        // Display initialization should be prior to modules that may use graphic card (eg. GLSL) in order
        // to initalize shared resource (X display) first
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
#ifndef WIN32
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

        try {
                map<string, param_u> params;

                // common
                params["parent"].ptr = &uv.root_module;
                params["exporter"].ptr = exporter;
                params["compression"].str = opt.requested_compression;
                params["rxtx_mode"].i = opt.video_rxtx_mode;

                // iHDTV
                params["argc"].i = argc;
                params["argv"].ptr = argv;
                params["capture_device"].ptr = (opt.video_rxtx_mode & MODE_SENDER) != 0U ? uv.capture_device : nullptr;
                params["display_device"].ptr = (opt.video_rxtx_mode & MODE_RECEIVER) != 0U ? uv.display_device : nullptr;

                //RTP
                params["mtu"].i = opt.requested_mtu;
                params["ttl"].i = opt.requested_ttl;
                params["receiver"].str = opt.requested_receiver;
                params["rx_port"].i = opt.video_rx_port;
                params["tx_port"].i = opt.video_tx_port;
                params["force_ip_version"].i = opt.force_ip_version;
                params["mcast_if"].str = opt.requested_mcast_if;
                params["mtu"].i = opt.requested_mtu;
                params["fec"].str = opt.requested_video_fec;
                params["encryption"].str = opt.requested_encryption;
                params["bitrate"].ll = opt.bitrate;
                params["start_time"].ll = start_time;
                params["video_delay"].vptr = (volatile void *) &video_offset;

                // UltraGrid RTP
                params["decoder_mode"].l = (long) opt.decoder_mode;
                params["display_device"].ptr = uv.display_device;

                // SAGE + RTSP
                params["opts"].str = opt.video_protocol_opts;

                // RTSP
                params["audio_codec"].l = get_audio_codec(opt.audio.codec_cfg);
                params["audio_sample_rate"].i = get_audio_codec_sample_rate(opt.audio.codec_cfg) ? get_audio_codec_sample_rate(opt.audio.codec_cfg) : 48000;
                params["audio_channels"].i = audio_capture_channels;
                params["audio_bps"].i = 2;
                params["a_rx_port"].i = opt.audio.recv_port;
                params["a_tx_port"].i = opt.audio.send_port;

                if (strcmp(opt.video_protocol, "rtsp") == 0) {
                        rtps_types_t avType;
                        if(strcmp("none", vidcap_params_get_driver(opt.vidcap_params_head)) != 0 && (strcmp("none",opt.audio.send_cfg) != 0)) avType = av; //AVStream
                        else if((strcmp("none",opt.audio.send_cfg) != 0)) avType = audio; //AStream
                        else if(strcmp("none", vidcap_params_get_driver(opt.vidcap_params_head))) avType = video; //VStream
                        else {
                                printf("[RTSP SERVER CHECK] no stream type... check capture devices input...\n");
                                avType = none;
                        }

                        params["avType"].l = (long) avType;
                }

                sdp_set_properties(opt.requested_receiver, opt.video_rxtx_mode & MODE_SENDER && strcasecmp(opt.video_protocol, "sdp") == 0, opt.audio.send_port != 0 && strcasecmp(opt.audio.proto, "sdp") == 0);

                uv.state_video_rxtx = video_rxtx::create(opt.video_protocol, params);
                if (!uv.state_video_rxtx) {
                        if (strcmp(opt.video_protocol, "help") != 0) {
                                throw string("Requested RX/TX cannot be created (missing library?)");
                        } else {
                                throw 0;
                        }
                }

                if ((opt.video_rxtx_mode & MODE_RECEIVER) != 0U) {
                        if (!uv.state_video_rxtx->supports_receiving()) {
                                fprintf(stderr, "Selected RX/TX mode doesn't support receiving.\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        }
                        // init module here so as it is capable of receiving messages
                        if (pthread_create
                                        (&receiver_thread_id, NULL, video_rxtx::receiver_thread,
                                         (void *) uv.state_video_rxtx) != 0) {
                                perror("Unable to create display thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        } else {
                                receiver_thread_started = true;
                        }
                }

                if ((opt.video_rxtx_mode & MODE_SENDER) != 0U) {
                        if (pthread_create
                                        (&capture_thread_id, NULL, capture_thread,
                                         (void *) &uv) != 0) {
                                perror("Unable to create capture thread!\n");
                                exit_uv(EXIT_FAILURE);
                                goto cleanup;
                        } else {
                                capture_thread_started = true;
                        }
                }

                if(audio_get_display_flags(uv.audio)) {
                        audio_register_display_callbacks(uv.audio,
                                       uv.display_device,
                                       (void (*)(void *, const struct audio_frame *)) display_put_audio_frame,
                                       (bool (*)(void *, int, int, int)) display_reconfigure_audio,
                                       (bool (*)(void *, int, void *, size_t *)) display_ctl_property);
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

        } catch (ug_no_error const &e) {
                exit_uv(0);
        } catch (ug_runtime_error const &e) {
                cerr << e.what() << endl;
                exit_uv(e.get_code());
        } catch (runtime_error const &e) {
                cerr << e.what() << endl;
                exit_uv(EXIT_FAILURE);
        } catch (exception const &e) {
                cerr << e.what() << endl;
                exit_uv(EXIT_FAILURE);
        } catch (string const &str) {
                cerr << str << endl;
                exit_uv(EXIT_FAILURE);
        } catch (int i) {
                exit_uv(i);
        }

cleanup:
        if (strcmp("none", opt.requested_display) != 0 &&
                        receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

        if ((opt.video_rxtx_mode & MODE_SENDER) != 0U
                        && capture_thread_started) {
                pthread_join(capture_thread_id, NULL);
        }

        /* also wait for audio threads */
        audio_join(uv.audio);
        if (uv.state_video_rxtx)
                uv.state_video_rxtx->join();

        export_destroy(exporter);

        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
#ifndef WIN32
        signal(SIGHUP, SIG_DFL);
        signal(SIGALRM, hang_signal_handler);
#endif
        alarm(5); // prevent exit hangs

        if(uv.audio)
                audio_done(uv.audio);
        delete uv.state_video_rxtx;

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

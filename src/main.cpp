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
 * Copyright (c) 2005-2019 CESNET z.s.p.o.
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

#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <string.h>
#include <thread>
#include <tuple>

#include "compat/platform_pipe.h"
#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "playback.h"
#include "rtp/rtp.h"
#include "rtsp/rtsp_utils.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/net.h"
#include "utils/thread.h"
#include "utils/wait_obj.h"
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
#define OPT_AUDIO_CAPTURE_FORMAT (('C' << 8) | 'F')
#define OPT_AUDIO_CHANNEL_MAP (('a' << 8) | 'm')
#define OPT_AUDIO_CODEC (('A' << 8) | 'C')
#define OPT_AUDIO_DELAY (('A' << 8) | 'D')
#define OPT_AUDIO_PROTOCOL (('A' << 8) | 'P')
#define OPT_AUDIO_SCALE (('a' << 8) | 's')
#define OPT_CAPABILITIES (('C' << 8) | 'C')
#define OPT_CAPTURE_FILTER (('O' << 8) | 'F')
#define OPT_CONTROL_PORT (('C' << 8) | 'P')
#define OPT_CUDA_DEVICE (('C' << 8) | 'D')
#define OPT_ECHO_CANCELLATION (('E' << 8) | 'C')
#define OPT_ENCRYPTION (('E' << 8) | 'N')
#define OPT_FULLHELP (('F' << 8u) | 'H')
#define OPT_EXPORT (('E' << 8) | 'X')
#define OPT_IMPORT (('I' << 8) | 'M')
#define OPT_LIST_MODULES (('L' << 8) | 'M')
#define OPT_MCAST_IF (('M' << 8) | 'I')
#define OPT_PARAM (('O' << 8) | 'P')
#define OPT_PIX_FMTS (('P' << 8) | 'F')
#define OPT_PROTOCOL (('P' << 8) | 'R')
#define OPT_START_PAUSED (('S' << 8) | 'P')
#define OPT_VERBOSE (('V' << 8) | 'E')
#define OPT_VIDEO_CODECS (('V' << 8) | 'C')
#define OPT_VIDEO_PROTOCOL (('V' << 8) | 'P')
#define OPT_WINDOW_TITLE (('W' << 8) | 'T')

#define MAX_CAPTURE_COUNT 17

using namespace std;
using namespace std::chrono;

struct state_uv {
        state_uv() : capture_device{}, display_device{}, audio{}, state_video_rxtx{} {
                if (platform_pipe_init(should_exit_pipe) != 0) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot create pipe!\n";
                        abort();
                }
                module_init_default(&root_module);
                root_module.cls = MODULE_CLASS_ROOT;
                root_module.new_message = state_uv::new_message;
                root_module.priv_data = this;
                should_exit_thread = thread(should_exit_watcher, this);
        }
        ~state_uv() {
                stop();
        }
        void stop() {
                if (exited) {
                        return;
                }
                broadcast_should_exit();
                should_exit_thread.join();
                module_done(&root_module);
                for (int i = 0; i < 2; ++i) {
                        platform_pipe_close(should_exit_pipe[0]);
                }
                exited = true;
        }
        static void should_exit_watcher(state_uv *s) {
                set_thread_name(__func__);
                char c;
                while (recv(s->should_exit_pipe[0], &c, 1, 0) != 1) perror("recv");
                unique_lock<mutex> lk(s->lock);
                for (auto c : s->should_exit_callbacks) {
                        get<0>(c)(get<1>(c));
                }
        }
        void broadcast_should_exit() {
                char c = 0;
                unique_lock<mutex> lk(lock);
                if (exited || should_exit_thread_notified) {
                        return;
                }
                should_exit_thread_notified = true;
                while (send(should_exit_pipe[1], &c, 1, 0) != 1) perror("send");
        }
        static void new_message(struct module *mod)
        {
                auto s = (state_uv *) mod->priv_data;
                struct msg_root *m;
                while ((m = (struct msg_root *) check_message(mod))) {
                        if (m->type != ROOT_MSG_REGISTER_SHOULD_EXIT) {
                                free_message((struct message *) m, new_response(RESPONSE_BAD_REQUEST, NULL));
                                continue;
                        }
                        unique_lock<mutex> lk(s->lock);
                        s->should_exit_callbacks.push_back(make_tuple(m->should_exit_callback, m->udata));
                        lk.unlock();
                        free_message((struct message *) m, new_response(RESPONSE_OK, NULL));
                }
        }

        string capture_device_name;
        struct vidcap *capture_device;
        struct display *display_device;

        struct state_audio *audio;

        struct module root_module;

        video_rxtx *state_video_rxtx;

private:
        mutex lock;
        fd_t should_exit_pipe[2];
        thread should_exit_thread;
        bool should_exit_thread_notified{false};
        bool exited{false};
        list<tuple<void (*)(void *), void *>> should_exit_callbacks;
};


static volatile int exit_status = EXIT_SUCCESS;
static struct state_uv * volatile uv_state;

static void write_all(size_t len, const char *msg) {
        const char *ptr = msg;
        do {
                ssize_t written = write(STDERR_FILENO, ptr, len);
                if (written < 0) {
                        break;
                }
                len -= written;
                ptr += written;
        } while (len > 0);
}

static void signal_handler(int signal)
{
        if (log_level >= LOG_LEVEL_DEBUG) {
                char msg[] = "Caught signal ";
                char buf[128];
                char *ptr = buf;
                for (size_t i = 0; i < sizeof msg - 1; ++i) {
                        *ptr++ = msg[i];
                }
                if (signal / 10) {
                        *ptr++ = '0' + signal/10;
                }
                *ptr++ = '0' + signal%10;
                *ptr++ = '\n';
                write_all(ptr - buf, buf);
        }
        exit_uv(0);
}

static void crash_signal_handler(int sig)
{
        char buf[1024];
        char *ptr = buf;
        *ptr++ = '\n';
        const char message1[] = " has crashed";
        for (size_t i = 0; i < sizeof PACKAGE_NAME - 1; ++i) {
                *ptr++ = PACKAGE_NAME[i];
        }
        for (size_t i = 0; i < sizeof message1 - 1; ++i) {
                *ptr++ = message1[i];
        }
#ifndef WIN32
        *ptr++ = ' '; *ptr++ = '(';
        for (size_t i = 0; i < sizeof sys_siglist[sig] - 1; ++i) {
                if (sys_siglist[sig][i] == '\0') {
                        break;
                }
                *ptr++ = sys_siglist[sig][i];
        }
        *ptr++ = ')';
#endif
        const char message2[] = ".\n\nPlease send a bug report to address ";
        for (size_t i = 0; i < sizeof message2 - 1; ++i) {
                *ptr++ = message2[i];
        }
        for (size_t i = 0; i < sizeof PACKAGE_BUGREPORT - 1; ++i) {
                *ptr++ = PACKAGE_BUGREPORT[i];
        }
        *ptr++ = '.'; *ptr++ = '\n';
        const char message3[] = "You may find some tips how to report bugs in file REPORTING-BUGS distributed with ";
        for (size_t i = 0; i < sizeof message3 - 1; ++i) {
                *ptr++ = message3[i];
        }
        for (size_t i = 0; i < sizeof PACKAGE_NAME - 1; ++i) {
                *ptr++ = PACKAGE_NAME[i];
        }
        *ptr++ = '.'; *ptr++ = '\n';

        write_all(ptr - buf, buf);

        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        raise(sig);
}

#ifndef WIN32
static void hang_signal_handler(int sig)
{
        assert(sig == SIGALRM);
        char msg[] = "Hang detected - you may continue waiting or kill UltraGrid. Please report if UltraGrid doesn't exit after reasonable amount of time.\n";
        write_all(sizeof msg - 1, msg);
        signal(SIGALRM, SIG_DFL);
}
#endif // ! defined WIN32

void exit_uv(int status) {
        exit_status = status;
        should_exit = true;
        uv_state->broadcast_should_exit();
}

static void print_help_item(const string &name, const vector<string> &help) {
        int help_lines = 0;

        cout << BOLD("\t" << name);

        for (auto line : help) {
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
        cout << "Usage: " << BOLD(RED((exec_path ? exec_path : "<executable_path>")) << " [options] address\n\n");
        printf("Options:\n");
        print_help_item("-h | --fullhelp", {"show usage (basic/full)"});
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
                print_help_item("--video-protocol <proto>", {"transmission protocol, see '--video-protocol help'",
                                "for list. Use --video-protocol rtsp for RTSP server",
                                "(see --video-protocol rtsp:help for usage)"});
                print_help_item("--audio-protocol <proto>[:<settings>]", {"<proto> can be " AUDIO_PROTOCOLS});
                print_help_item("--protocol <proto>", {"shortcut for '--audio-protocol <proto> --video-protocol <proto>'"});
#ifdef HAVE_IPv6
                print_help_item("-4/-6", {"force IPv4/IPv6 resolving"});
#endif //  HAVE_IPv6
                print_help_item("--mcast-if <iface>", {"bind to specified interface for multicast"});
                print_help_item("-m <mtu>", {"set path MTU assumption towards receiver"});
                print_help_item("-M <video_mode>", {"received video mode (eg tiled-4K, 3D,",
                                "dual-link)"});
                print_help_item("-p <postprocess> | help", {"postprocess module"});
        }
        print_help_item("-f [A:|V:]<settings>", {"FEC settings (audio or video) - use",
                        "\"none\", \"mult:<nr>\",", "\"ldgm:<max_expected_loss>%%\" or", "\"ldgm:<k>:<m>:<c>\"",
                        "\"rs:<k>:<n>\""});
        print_help_item("-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]", { "",
                        "<port> is base port number, also 3",
                        "subsequent ports can be used for RTCP",
                        "and audio streams. Default: " + to_string(PORT_BASE) + ".",
                        "You can also specify all two or four", "ports directly."});
        print_help_item("-l <limit_bitrate> | unlimited | auto", {"limit sending bitrate",
                        "to <limit_bitrate> (with optional k/M/G suffix)"});
        if (full) {
                print_help_item("-A <address>", {"audio destination address",
                                "If not specified, will use same as for video"});
        }
        print_help_item("--audio-capture-format <fmt> | help", {"format of captured audio"});
        if (full) {
                print_help_item("--audio-channel-map <mapping> | help", {});
        }
        print_help_item("--audio-codec <codec>[:sample_rate=<sr>][:bitrate=<br>] | help", {"audio codec"});
        if (full) {
                print_help_item("--audio-delay <delay_ms>", {"amount of time audio should be delayed to video",
                                "(may be also negative to delay video)"});
                print_help_item("--audio-scale <factor> | <method> | help",
                                {"scales received audio"});
        }
#if 0
        printf("\t--echo-cancellation      \tapply acoustic echo cancellation to audio\n");
        printf("\n");
#endif
        print_help_item("--cuda-device <index> | help", {"use specified CUDA device"});
        if (full) {
                print_help_item("--encryption <passphrase>", {"key material for encryption"});
                print_help_item("--playback <directory> | help", {"replays recorded audio and video"});
                print_help_item("--record[=<directory>]", {"record captured audio and video"});
                print_help_item("--capture-filter <filter> | help",
                                {"capture filter(s), must be given before capture device"});
                print_help_item("--param <params> | help", {"additional advanced parameters, use help for list"});
                print_help_item("--pix-fmts", {"list of pixel formats"});
                print_help_item("--video-codecs", {"list of video codecs"});
        }
        print_help_item("address", {"destination address"});
        printf("\n");
}

static void print_fps(steady_clock::time_point *t0, int *frames, const string &name) {
        *frames += 1;
        steady_clock::time_point t1 = steady_clock::now();
        double seconds = duration_cast<duration<double>>(t1 - *t0).count();
        if (seconds >= 5.0) {
                double fps = *frames / seconds;
                LOG(LOG_LEVEL_INFO) << "[" << name << "] " << *frames << " frames in " << seconds << " seconds = " << BOLD(fps << " FPS\n");
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

        struct module *uv_mod = (struct module *)arg;
        struct state_uv *uv = (struct state_uv *) uv_mod->priv_data;
        struct wait_obj *wait_obj = wait_obj_init();
        steady_clock::time_point t0 = steady_clock::now();
        int frames = 0;
        bool should_print_fps = vidcap_generic_fps(uv->capture_device);

        while (!should_exit) {
                /* Capture and transmit video... */
                struct audio_frame *audio = nullptr;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);

                if (audio != nullptr) {
                        audio_sdi_send(uv->audio, audio);
                        AUDIO_FRAME_DISPOSE(audio);
                }

                if (tx_frame != NULL) {
                        if (should_print_fps) {
                                print_fps(&t0, &frames, uv->capture_device_name);
                        }
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

                        uv->state_video_rxtx->send(move(frame)); // std::move really important here (!)

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

static bool parse_audio_capture_format(const char *optarg)
{
        if (strcmp(optarg, "help") == 0) {
                printf("Usage:\n");
                printf("\t--audio-capture-format {channels=<num>|bps=<bits_per_sample>|sample_rate=<rate>}*\n");
                printf("\t\tmultiple options can be separated by a colon\n");
                return false;
        }

        unique_ptr<char[]> arg_copy(new char[strlen(optarg) + 1]);
        char *arg = arg_copy.get();
        strcpy(arg, optarg);

        char *item, *save_ptr, *tmp;
        tmp = arg;
        char *endptr;

        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                if (strncmp(item, "channels=", strlen("channels=")) == 0) {
                        item += strlen("channels=");
                        audio_capture_channels = strtol(item, &endptr, 10);
                        if (audio_capture_channels < 1 || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid number of channels %s!\n", item);
                                return false;
                        }
                } else if (strncmp(item, "bps=", strlen("bps=")) == 0) {
                        item += strlen("bps=");
                        int bps = strtol(item, &endptr, 10);
                        if (bps % 8 != 0 || (bps != 8 && bps != 16 && bps != 24 && bps != 32) || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid bps %s!\n", item);
                                log_msg(LOG_LEVEL_ERROR, "Supported values are 8, 16, 24, or 32 bits.\n");
                                return false;

                        }
                        audio_capture_bps = bps / 8;
                } else if (strncmp(item, "sample_rate=", strlen("sample_rate=")) == 0) {
                        long long val = unit_evaluate(item + strlen("sample_rate="));
                        assert(val > 0 && val <= numeric_limits<decltype(audio_capture_sample_rate)>::max());
                        audio_capture_sample_rate = val;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "Unkonwn format for --audio-capture-format!\n");
                        return false;
                }

                tmp = NULL;
        }

        return true;
}

static bool parse_bitrate(char *optarg, long long int *bitrate) {
        if (strcmp(optarg, "auto") == 0) {
                *bitrate = RATE_AUTO;
                return true;
        }
        if (strcmp(optarg, "help") == 0) {
                const char numeric_pattern[] = "{1-9}{0-9}*[kMG][!][E]";
                cout << "Usage:\n" <<
                        "\tuv " << BOLD("-l [auto | unlimited | " << numeric_pattern << "]\n") <<
                        "\twhere\n"
                        "\t\t" << BOLD("auto") << " - spread packets across frame time\n"
                        "\t\t" << BOLD("unlimited") << " - send packets at a wire speed (in bursts)\n"
                        "\t\t" << BOLD(numeric_pattern) << " - send packets at most at specified bitrate\n\n" <<
                        BOLD("Notes: ") << "Use an exclamation mark to indicate intentionally very low bitrate. 'E' to use the value as a fixed bitrate, not cap /i. e. even the frames that may be sent at lower bitrate are sent at the nominal bitrate)\n" <<
                        "\n";
                return true;
        }
        if (strcmp(optarg, "unlimited") == 0) {
                *bitrate = RATE_UNLIMITED;
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

static bool parse_params(char *optarg)
{
        if (optarg && strcmp(optarg, "help") == 0) {
                puts("Params can be one or more (separated by comma) of following:");
                print_param_doc();
                return false;
        }
        char *item, *save_ptr;
        while ((item = strtok_r(optarg, ",", &save_ptr))) {
                char *key_cstr = item;
                if (strchr(item, '=')) {
                        char *val_cstr = strchr(item, '=') + 1;
                        *strchr(item, '=') = '\0';
                        commandline_params[key_cstr] = val_cstr;
                } else {
                        commandline_params[key_cstr] = string();
                }
                if (!validate_param(key_cstr)) {
                        log_msg(LOG_LEVEL_ERROR, "Unknown parameter: %s\n", key_cstr);
                        log_msg(LOG_LEVEL_INFO, "Type '%s --param help' for list.\n", uv_argv[0]);
                        return false;
                }
                optarg = NULL;
        }
        return true;
}

#define EXIT(retval) { common_cleanup(init); return retval; }

int main(int argc, char *argv[])
{
        struct init_data *init = nullptr;
#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        // NULL terminated array of capture devices
        struct vidcap_params *vidcap_params_head = vidcap_params_allocate();
        struct vidcap_params *vidcap_params_tail = vidcap_params_head;

        const char *display_cfg = "";
        const char *audio_recv = "none";
        const char *audio_send = "none";
        const char *requested_video_fec = "none";
        const char *requested_audio_fec = DEFAULT_AUDIO_FEC;
        char *audio_channel_map = NULL;
        const char *audio_scale = "mixauto";
        int port_base = PORT_BASE;
        int video_rx_port = -1, video_tx_port = -1, audio_rx_port = -1, audio_tx_port = -1;

        bool echo_cancellation = false;

        bool should_export = false;
        char *export_opts = NULL;

        int control_port = 0;
        int connection_type = 0;
        struct control_state *control = NULL;

        const char *audio_host = NULL;
        enum video_mode decoder_mode = VIDEO_NORMAL;
        const char *requested_compression = nullptr;

        int force_ip_version = 0;
        int ch;

        const char *audio_codec = nullptr;

        pthread_t receiver_thread_id,
                  capture_thread_id;
        bool receiver_thread_started = false,
             capture_thread_started = false;
        unsigned display_flags = 0;
        int ret;
        struct vidcap_params *audio_cap_dev;
        const char *requested_mcast_if = NULL;

        unsigned requested_mtu = 0;
        const char *postprocess = NULL;
        const char *requested_display = "none";
        const char *requested_receiver = "localhost";
        const char *requested_encryption = NULL;
        struct exporter *exporter = NULL;

        long long int bitrate = RATE_DEFAULT;

        int audio_rxtx_mode = 0, video_rxtx_mode = 0;

        const chrono::steady_clock::time_point start_time(chrono::steady_clock::now());

        bool print_capabilities_req = false;
        bool start_paused = false;

        static struct option getopt_options[] = {
                {"display", required_argument, 0, 'd'},
                {"capture", required_argument, 0, 't'},
                {"mtu", required_argument, 0, 'm'},
                {"mode", required_argument, 0, 'M'},
                {"version", no_argument, 0, 'v'},
                {"compress", required_argument, 0, 'c'},
                {"receive", required_argument, 0, 'r'},
                {"send", required_argument, 0, 's'},
                {"help", no_argument, 0, 'h'},
                {"fec", required_argument, 0, 'f'},
                {"port", required_argument, 0, 'P'},
                {"limit-bitrate", required_argument, 0, 'l'},
                {"audio-channel-map", required_argument, 0, OPT_AUDIO_CHANNEL_MAP},
                {"audio-scale", required_argument, 0, OPT_AUDIO_SCALE},
                {"audio-capture-channels", required_argument, 0, OPT_AUDIO_CAPTURE_CHANNELS},
                {"audio-capture-format", required_argument, 0, OPT_AUDIO_CAPTURE_FORMAT},
                {"echo-cancellation", no_argument, 0, OPT_ECHO_CANCELLATION},
                {"fullhelp", no_argument, 0, OPT_FULLHELP},
                {"cuda-device", required_argument, 0, OPT_CUDA_DEVICE},
                {"mcast-if", required_argument, 0, OPT_MCAST_IF},
                {"record", optional_argument, 0, OPT_EXPORT},
                {"playback", required_argument, 0, OPT_IMPORT},
                {"audio-host", required_argument, 0, 'A'},
                {"audio-codec", required_argument, 0, OPT_AUDIO_CODEC},
                {"capture-filter", required_argument, 0, OPT_CAPTURE_FILTER},
                {"control-port", required_argument, 0, OPT_CONTROL_PORT},
                {"encryption", required_argument, 0, OPT_ENCRYPTION},
                {"verbose", optional_argument, 0, OPT_VERBOSE},
                {"window-title", required_argument, 0, OPT_WINDOW_TITLE},
                {"capabilities", no_argument, 0, OPT_CAPABILITIES},
                {"audio-delay", required_argument, 0, OPT_AUDIO_DELAY},
                {"list-modules", no_argument, 0, OPT_LIST_MODULES},
                {"start-paused", no_argument, 0, OPT_START_PAUSED},
                {"audio-protocol", required_argument, 0, OPT_AUDIO_PROTOCOL},
                {"video-protocol", required_argument, 0, OPT_VIDEO_PROTOCOL},
                {"protocol", required_argument, 0, OPT_PROTOCOL},
                {"rtsp-server", optional_argument, 0, 'H'},
                {"param", required_argument, 0, OPT_PARAM},
                {"pix-fmts", no_argument, 0, OPT_PIX_FMTS},
                {"video-codecs", no_argument, 0, OPT_VIDEO_CODECS},
                {0, 0, 0, 0}
        };
        const char optstring[] = "d:t:m:r:s:v46c:hM:p:f:P:l:A:";

        const char *audio_protocol = "ultragrid_rtp";
        const char *audio_protocol_opts = "";

        const char *video_protocol = "ultragrid_rtp";
        const char *video_protocol_opts = "";

        // First we need to set verbosity level prior to everything else.
        // common_preinit() uses the verbosity level.
        while ((ch =
                getopt_long(argc, argv, optstring, getopt_options,
                            NULL)) != -1) {
                switch (ch) {
                case OPT_VERBOSE:
                        if (optarg) {
                                log_level = atoi(optarg);
                        } else {
                                log_level = LOG_LEVEL_VERBOSE;
                        }
                        break;
                default:
                        break;
                }
        }
        optind = 1;

        if ((init = common_preinit(argc, argv)) == nullptr) {
                log_msg(LOG_LEVEL_FATAL, "common_preinit() failed!\n");
                EXIT(EXIT_FAILURE);
        }

        struct state_uv uv{};
        uv_state = &uv;
        keyboard_control kc{&uv.root_module};

        vidcap_params_set_device(vidcap_params_head, "none");

        print_version();
        printf("\n");

        while ((ch =
                getopt_long(argc, argv, optstring, getopt_options,
                            NULL)) != -1) {
                switch (ch) {
                case 'd':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                list_video_display_devices(strcmp(optarg, "fullhelp") == 0);
                                EXIT(0);
                        }
                        requested_display = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                display_cfg = delim + 1;
                        }
                        break;
                case 't':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                list_video_capture_devices(strcmp(optarg, "fullhelp") == 0);
                                EXIT(0);
                        }
                        vidcap_params_set_device(vidcap_params_tail, optarg);
                        vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        break;
                case 'm':
                        requested_mtu = atoi(optarg);
                        if (requested_mtu < 576 && optarg[strlen(optarg) - 1] != '!') {
                                log_msg(LOG_LEVEL_WARNING, "MTU %1$u seems to be too low, use \"%1$u!\" to force.\n", requested_mtu);
                                EXIT(EXIT_FAIL_USAGE);
                        }
                        break;
                case 'M':
                        decoder_mode = get_video_mode_from_str(optarg);
                        if (decoder_mode == VIDEO_UNKNOWN) {
                                EXIT(strcasecmp(optarg, "help") == 0 ? EXIT_SUCCESS : EXIT_FAIL_USAGE);
                        }
                        break;
                case 'p':
                        postprocess = optarg;
                        break;
                case 'v':
                        print_configuration();
                        EXIT(EXIT_SUCCESS);
                case 'c':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                show_compress_help(strcmp(optarg, "fullhelp") == 0);
                                EXIT(EXIT_SUCCESS);
                        }
                        requested_compression = optarg;
                        break;
                case 'H':
                        log_msg(LOG_LEVEL_WARNING, "Option \"--rtsp-server[=args]\" "
                                        "is deprecated and will be removed in future.\n"
                                        "Please use \"--video-protocol rtsp[:args]\"instead.\n");
                        video_protocol = "rtsp";
                        video_protocol_opts = optarg ? optarg : "";
                        break;
                case OPT_AUDIO_PROTOCOL:
                        audio_protocol = optarg;
                        if (strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                audio_protocol_opts = delim + 1;
                        }
                        if (strcmp(audio_protocol, "help") == 0) {
                                printf("Audio protocol can be one of: " AUDIO_PROTOCOLS "\n");
                                EXIT(EXIT_SUCCESS);
                        }
                        break;
                case OPT_VIDEO_PROTOCOL:
                        video_protocol = optarg;
                        if (strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                video_protocol_opts = delim + 1;
                        }
                        if (strcmp(video_protocol, "help") == 0) {
                                video_rxtx::list(strcmp(optarg, "fullhelp") == 0);
                                EXIT(EXIT_SUCCESS);
                        }
                        break;
                case OPT_PROTOCOL:
                        if (strcmp(optarg, "help") == 0 ||
                                        strcmp(optarg, "fullhelp") == 0) {
                                cout << "Specify a " << BOLD("common") << " protocol for both audio and video.\n";
                                cout << "Audio protocol can be one of: " << BOLD(AUDIO_PROTOCOLS "\n");
                                video_rxtx::list(strcmp(optarg, "fullhelp") == 0);
                                EXIT(EXIT_SUCCESS);
                        }
                        audio_protocol = video_protocol = optarg;
                        if (strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                audio_protocol_opts = video_protocol_opts = delim + 1;
                        }
                        break;
                case 'r':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                audio_playback_help(strcmp(optarg, "full") == 0);
                                EXIT(EXIT_SUCCESS);
                        }
                        audio_recv = optarg;                       
                        break;
                case 's':
                        if (strcmp(optarg, "help") == 0 || strcmp(optarg, "fullhelp") == 0) {
                                audio_capture_print_help(strcmp(optarg, "full") == 0);
                                EXIT(EXIT_SUCCESS);
                        }
                        audio_send = optarg;
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
                        usage(uv_argv[0], false);
                        EXIT(0);
                case 'P':
                        try {
                                if (strchr(optarg, ':') != nullptr) {
                                        char *save_ptr = nullptr;
                                        video_rx_port = stoi(strtok_r(optarg, ":", &save_ptr), nullptr, 0);
                                        video_tx_port = stoi(strtok_r(nullptr, ":", &save_ptr), nullptr, 0);
                                        char *tok = nullptr;
                                        if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                                                audio_rx_port = stoi(tok, nullptr, 0);
                                                if ((tok = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                                                        audio_tx_port = stoi(tok, nullptr, 0);
                                                } else {
                                                        usage(uv_argv[0]);
                                                        EXIT(EXIT_FAIL_USAGE);
                                                }
                                        }
                                } else {
                                        port_base = stoi(optarg, nullptr, 0);
                                }
                                if (audio_rx_port < -1 || audio_tx_port < -1 || video_rx_port < -1 || video_tx_port < -1 || port_base < -1 ||
                                                audio_rx_port > UINT16_MAX || audio_tx_port > UINT16_MAX || video_rx_port > UINT16_MAX || video_tx_port > UINT16_MAX || port_base > UINT16_MAX) {
                                        throw ug_runtime_error("Invalid port value, allowed range 1-65535", EXIT_FAIL_USAGE);
                                }
                        } catch (exception const &e) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong port specification: " << e.what() << "\n";
                                EXIT(EXIT_FAIL_USAGE);
                        }
                        break;
                case 'l':
                        if (!parse_bitrate(optarg, &bitrate)) {
                                EXIT(EXIT_FAILURE);
                        }
                        if (bitrate == RATE_DEFAULT) {
                                EXIT(EXIT_SUCCESS); // help written
                        }
                        break;
                case '4':
                case '6':
                        force_ip_version = ch - '0';
                        break;
                case OPT_AUDIO_CHANNEL_MAP:
                        audio_channel_map = optarg;
                        break;
                case OPT_AUDIO_SCALE:
                        audio_scale = optarg;
                        break;
                case OPT_AUDIO_CAPTURE_CHANNELS:
                        log_msg(LOG_LEVEL_WARNING, "Parameter --audio-capture-channels is deprecated. "
                                        "Use \"--audio-capture-format channels=<count>\" instead.\n");
                        audio_capture_channels = atoi(optarg);
                        if (audio_capture_channels < 1) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid number of channels %d!\n", audio_capture_channels);
                                EXIT(EXIT_FAIL_USAGE);
                        }
                        break;
                case OPT_AUDIO_CAPTURE_FORMAT:
                        if (!parse_audio_capture_format(optarg)) {
                                EXIT(EXIT_FAIL_USAGE);
                        }
                        break;
                case OPT_ECHO_CANCELLATION:
                        echo_cancellation = true;
                        break;
                case OPT_FULLHELP:
                        usage(uv_argv[0], true);
                        EXIT(EXIT_SUCCESS);
                case OPT_CUDA_DEVICE:
#ifdef HAVE_GPUJPEG
                        if(strcmp("help", optarg) == 0) {
                                struct compress_state *compression;
                                int ret = compress_init(&uv.root_module, "GPUJPEG:list_devices", &compression);
                                if(ret >= 0) {
                                        if(ret == 0) {
                                                module_done(CAST_MODULE(compression));
                                        }
                                        EXIT(EXIT_SUCCESS);
                                } else {
                                        EXIT(EXIT_FAILURE);
                                }
                        } else {
                                char *item, *save_ptr = NULL;
                                unsigned int i = 0;
                                while((item = strtok_r(optarg, ",", &save_ptr))) {
                                        if(i >= MAX_CUDA_DEVICES) {
                                                fprintf(stderr, "Maximal number of CUDA device exceeded.\n");
                                                EXIT(EXIT_FAILURE);
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
                        EXIT(EXIT_FAIL_USAGE);
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
                                int ret;
                                if ((ret = playback_set_device(dev_string, sizeof dev_string, optarg)) <= 0) {
                                        EXIT(ret == 0 ? EXIT_SUCCESS : EXIT_FAIL_USAGE);
                                }
                                vidcap_params_set_device(vidcap_params_tail, dev_string);
                                vidcap_params_tail = vidcap_params_allocate_next(vidcap_params_tail);
                        }
                        break;
                case OPT_AUDIO_CODEC:
                        if(strcmp(optarg, "help") == 0) {
                                list_audio_codecs();
                                EXIT(EXIT_SUCCESS);
                        }
                        audio_codec = optarg;
                        if (!check_audio_codec(optarg)) {
                                EXIT(EXIT_FAIL_USAGE);
                        }
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
                                        usage(uv_argv[0]);
                                        EXIT(EXIT_FAIL_USAGE);
                                }
                                if ((tok = strtok_r(NULL, ":", &save_ptr))) {
                                        usage(uv_argv[0]);
                                        EXIT(EXIT_FAIL_USAGE);
                                }
                        } else {
                                control_port = atoi(optarg);
                                connection_type = 0;
                        }
                        break;
                case OPT_VERBOSE:
                        break; // already handled earlier
                case OPT_WINDOW_TITLE:
                        log_msg(LOG_LEVEL_WARNING, "Deprecated option used, please use "
                                        "--param window-title=<title>\n");
                        commandline_params["window-title"] = optarg;
                        break;
                case OPT_CAPABILITIES:
                        print_capabilities_req = true;
                        break;
                case OPT_AUDIO_DELAY:
                        audio_offset = max(atoi(optarg), 0);
                        video_offset = atoi(optarg) < 0 ? abs(atoi(optarg)) : 0;
                        break;
                case OPT_LIST_MODULES:
                        list_all_modules();
                        EXIT(EXIT_SUCCESS);
                case OPT_START_PAUSED:
                        start_paused = true;
                        break;
                case OPT_PARAM:
                        if (!parse_params(optarg)) {
                                EXIT(EXIT_SUCCESS);
                        }
                        break;
                case OPT_PIX_FMTS:
                        print_pixel_formats();
                        EXIT(EXIT_SUCCESS);
                case OPT_VIDEO_CODECS:
                        print_video_codecs();
                        EXIT(EXIT_SUCCESS);
                case '?':
                default:
                        usage(uv_argv[0]);
                        EXIT(EXIT_FAIL_USAGE);
                }
        }

        argc -= optind;
        argv += optind;

        if (argc > 1) {
                log_msg(LOG_LEVEL_ERROR, "Multiple receivers given!\n");
                usage(uv_argv[0]);
                EXIT(EXIT_FAIL_USAGE);
        }

        if (argc > 0) {
                requested_receiver = argv[0];
        }

        if (!audio_host) {
                audio_host = requested_receiver;
        }

        if (!is_ipv6_supported()) {
                log_msg(LOG_LEVEL_WARNING, "IPv6 support missing, setting IPv4-only mode.\n");
                force_ip_version = 4;
        }

        if (!set_output_buffering()) {
                log_msg(LOG_LEVEL_WARNING, "Cannot set console output buffering!\n");
        }

        // default values for different RXTX protocols
        if (strcasecmp(video_protocol, "rtsp") == 0 || strcasecmp(video_protocol, "sdp") == 0) {
                if (audio_codec == nullptr) {
                        audio_codec = "OPUS:sample_rate=48000";
                }
                if (requested_compression == nullptr) {
                        requested_compression = "none"; // will be set later
                }
                if (force_ip_version == 0 && strcasecmp(video_protocol, "rtsp") == 0) {
                        force_ip_version = 4;
                }
        } else {
                if (requested_compression == nullptr) {
                        requested_compression = DEFAULT_VIDEO_COMPRESSION;
                }
                if (audio_codec == nullptr) {
                        audio_codec = DEFAULT_AUDIO_CODEC;
                }
        }

        if (strcmp("none", audio_recv) != 0) {
                audio_rxtx_mode |= MODE_RECEIVER;
        }

        if (strcmp("none", audio_send) != 0) {
                audio_rxtx_mode |= MODE_SENDER;
        }

        if (strcmp("none", requested_display) != 0) {
                video_rxtx_mode |= MODE_RECEIVER;
        }
        if (strcmp("none", vidcap_params_get_driver(vidcap_params_head)) != 0) {
                video_rxtx_mode |= MODE_SENDER;
        }

        if (video_rx_port == -1) {
                if ((video_rxtx_mode & MODE_RECEIVER) == 0) {
                        // do not occupy recv port if we are not receiving (note that this disables communication with
                        // our receiver, because RTCP ports are changed as well)
                        video_rx_port = 0;
                } else {
                        video_rx_port = port_base;
                }
        }

        if (video_tx_port == -1) {
                if ((video_rxtx_mode & MODE_SENDER) == 0) {
                        video_tx_port = 0; // does not matter, we are receiver
                } else {
                        video_tx_port = port_base;
                }
        }

        if (audio_rx_port == -1) {
                if ((audio_rxtx_mode & MODE_RECEIVER) == 0) {
                        // do not occupy recv port if we are not receiving (note that this disables communication with
                        // our receiver, because RTCP ports are changed as well)
                        audio_rx_port = 0;
                } else {
                        audio_rx_port = video_rx_port ? video_rx_port + 2 : port_base + 2;
                }
        }

        if (audio_tx_port == -1) {
                if ((audio_rxtx_mode & MODE_SENDER) == 0) {
                        audio_tx_port = 0;
                } else {
                        audio_tx_port = video_tx_port ? video_tx_port + 2 : port_base + 2;
                }
        }

        // If we are sure that this UltraGrid is sending to itself we can optimize some parameters
        // (aka "-m 9000 -l unlimited"). If ports weren't equal it is possibile that we are sending
        // to a reflector, thats why we require equal ports (we are a receiver as well).
        if (is_host_loopback(requested_receiver) && video_rx_port == video_tx_port &&
                        audio_rx_port == audio_tx_port) {
                requested_mtu = requested_mtu == 0 ? min(RTP_MAX_MTU, 65535) : requested_mtu;
                bitrate = bitrate == RATE_DEFAULT ? RATE_UNLIMITED : bitrate;
        } else {
                requested_mtu = requested_mtu == 0 ? 1500 : requested_mtu;
                bitrate = bitrate == RATE_DEFAULT ? RATE_AUTO : bitrate;
        }

        if((strcmp("none", audio_send) != 0 || strcmp("none", audio_recv) != 0) && strcmp(video_protocol, "rtsp") == 0){
            //TODO: to implement a high level rxtx struct to manage different standards (i.e.:H264_STD, VP8_STD,...)
            if (strcmp(audio_protocol, "rtsp") != 0) {
		log_msg(LOG_LEVEL_WARNING, "Using RTSP for video but not for audio is not recommended. Consider adding '--audio-protocol rtsp'.\n");
            }
        }
        if (strcmp(audio_protocol, "rtsp") == 0 && strcmp(video_protocol, "rtsp") != 0) {
                log_msg(LOG_LEVEL_WARNING, "Using RTSP for audio but not for video is not recommended and might not work.\n");
        }

        cout << BOLD("Display device   : ") << requested_display << "\n";
        cout << BOLD("Capture device   : ") << vidcap_params_get_driver(vidcap_params_head) << "\n";
        cout << BOLD("Audio capture    : ") << audio_send << "\n";
        cout << BOLD("Audio playback   : ") << audio_recv << "\n";
        cout << BOLD("MTU              : ") << requested_mtu << " B\n";
        cout << BOLD("Video compression: ") << requested_compression << "\n";
        cout << BOLD("Audio codec      : ") << get_name_to_audio_codec(get_audio_codec(audio_codec)) << "\n";
        cout << BOLD("Network protocol : ") << video_rxtx::get_long_name(video_protocol) << "\n";
        cout << BOLD("Audio FEC        : ") << requested_audio_fec << "\n";
        cout << BOLD("Video FEC        : ") << requested_video_fec << "\n";
        cout << "\n";

        exporter = export_init(&uv.root_module, export_opts, should_export);
        if (!exporter) {
                log_msg(LOG_LEVEL_ERROR, "Export initialization failed.\n");
                EXIT(EXIT_FAILURE);
        }

        if (control_init(control_port, connection_type, &control, &uv.root_module, force_ip_version) != 0) {
                LOG(LOG_LEVEL_FATAL) << "Error: Unable to initialize remote control!\n";
                EXIT(EXIT_FAIL_CONTROL_SOCK);
        }

        uv.audio = audio_cfg_init (&uv.root_module, audio_host, audio_rx_port,
                        audio_tx_port, audio_send, audio_recv,
                        audio_protocol, audio_protocol_opts,
                        requested_audio_fec, requested_encryption,
                        audio_channel_map,
                        audio_scale, echo_cancellation, force_ip_version, requested_mcast_if,
                        audio_codec, bitrate, &audio_offset, &start_time,
                        requested_mtu, exporter);
        if(!uv.audio) {
                exit_uv(EXIT_FAIL_AUDIO);
                goto cleanup;
        }

        display_flags |= audio_get_display_flags(uv.audio);

        // Display initialization should be prior to modules that may use graphic card (eg. GLSL) in order
        // to initalize shared resource (X display) first
        ret =
             initialize_video_display(&uv.root_module, requested_display, display_cfg, display_flags, postprocess, &uv.display_device);
        if (ret < 0) {
                printf("Unable to open display device: %s\n",
                       requested_display);
                exit_uv(EXIT_FAIL_DISPLAY);
                goto cleanup;
        } else if(ret > 0) {
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

        ret = initialize_video_capture(&uv.root_module, vidcap_params_head, &uv.capture_device);
        if (ret < 0) {
                printf("Unable to open capture device: %s\n",
                                vidcap_params_get_driver(vidcap_params_head));
                exit_uv(EXIT_FAIL_CAPTURE);
                goto cleanup;
        } else if(ret > 0) {
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

        control_start(control);
        kc.start();

        try {
                map<string, param_u> params;

                // common
                params["parent"].ptr = &uv.root_module;
                params["exporter"].ptr = exporter;
                params["compression"].str = requested_compression;
                params["rxtx_mode"].i = video_rxtx_mode;
                params["paused"].b = start_paused;

                // iHDTV
                params["argc"].i = argc;
                params["argv"].ptr = argv;
                params["capture_device"].ptr = NULL;
                params["display_device"].ptr = NULL;
                if (video_rxtx_mode & MODE_SENDER)
                        params["capture_device"].ptr = uv.capture_device;
                if (video_rxtx_mode & MODE_RECEIVER)
                        params["display_device"].ptr = uv.display_device;

                //RTP
                params["mtu"].i = requested_mtu;
                params["receiver"].str = requested_receiver;
                params["rx_port"].i = video_rx_port;
                params["tx_port"].i = video_tx_port;
                params["force_ip_version"].i = force_ip_version;
                params["mcast_if"].str = requested_mcast_if;
                params["mtu"].i = requested_mtu;
                params["fec"].str = requested_video_fec;
                params["encryption"].str = requested_encryption;
                params["bitrate"].ll = bitrate;
                params["start_time"].cptr = (const void *) &start_time;
                params["video_delay"].vptr = (volatile void *) &video_offset;

                // UltraGrid RTP
                params["decoder_mode"].l = (long) decoder_mode;
                params["display_device"].ptr = uv.display_device;

                // SAGE + RTSP
                params["opts"].str = video_protocol_opts;

                // RTSP/SDP
                params["audio_codec"].l = get_audio_codec(audio_codec);
                params["audio_sample_rate"].i = get_audio_codec_sample_rate(audio_codec) ? get_audio_codec_sample_rate(audio_codec) : 48000;
                params["audio_channels"].i = audio_capture_channels;
                params["audio_bps"].i = 2;
                params["a_rx_port"].i = audio_rx_port;
                params["a_tx_port"].i = audio_tx_port;

                if (strcmp(video_protocol, "rtsp") == 0) {
                        rtps_types_t avType;
                        if(strcmp("none", vidcap_params_get_driver(vidcap_params_head)) != 0 && (strcmp("none",audio_send) != 0)) avType = av; //AVStream
                        else if((strcmp("none",audio_send) != 0)) avType = audio; //AStream
                        else if(strcmp("none", vidcap_params_get_driver(vidcap_params_head))) avType = video; //VStream
                        else {
                                printf("[RTSP SERVER CHECK] no stream type... check capture devices input...\n");
                                avType = none;
                        }

                        params["avType"].l = (long) avType;
                }

                uv.state_video_rxtx = video_rxtx::create(video_protocol, params);
                if (!uv.state_video_rxtx) {
                        if (strcmp(video_protocol, "help") != 0) {
                                throw string("Requested RX/TX cannot be created (missing library?)");
                        } else {
                                throw 0;
                        }
                }

                if (video_rxtx_mode & MODE_RECEIVER) {
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

                if (video_rxtx_mode & MODE_SENDER) {
                        uv.capture_device_name = vidcap_params_get_driver(vidcap_params_head);
                        if (pthread_create
                                        (&capture_thread_id, NULL, capture_thread,
                                         (void *) &uv.root_module) != 0) {
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
                                       (void (*)(void *, struct audio_frame *)) display_put_audio_frame,
                                       (int (*)(void *, int, int, int)) display_reconfigure_audio,
                                       (int (*)(void *, int, void *, size_t *)) display_ctl_property);
                }

                audio_start(uv.audio);

                // This has to be run after start of capture thread since it may request
                // captured video format information.
                if (print_capabilities_req) {
                        print_capabilities(&uv.root_module, strcmp("none", vidcap_params_get_driver(vidcap_params_head)) != 0);
                        exit_uv(EXIT_SUCCESS);
                        goto cleanup;
                }

                if (display_needs_mainloop(uv.display_device) && mainloop) {
                        throw string("Cannot run display when "
                                        "another mainloop registered!\n");
                }
                display_run(uv.display_device);
                if (mainloop) {
                        mainloop(mainloop_udata);
                }
                display_join(uv.display_device);
        } catch (ug_no_error const &e) {
                exit_uv(0);
        } catch (ug_runtime_error const &e) {
                cerr << e.what() << endl;
                exit_uv(e.get_code());
        } catch (runtime_error const &e) {
                cerr << e.what() << endl;
                exit_uv(EXIT_FAILURE);
        } catch (string const &str) {
                cerr << str << endl;
                exit_uv(EXIT_FAILURE);
        } catch (int i) {
                exit_uv(i);
        }

cleanup:
        if (strcmp("none", requested_display) != 0 &&
                        receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

        if (video_rxtx_mode & MODE_SENDER
                        && capture_thread_started)
                pthread_join(capture_thread_id, NULL);

        /* also wait for audio threads */
        audio_join(uv.audio);
        if (uv.state_video_rxtx)
                uv.state_video_rxtx->join();

        export_destroy(exporter);

        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
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

        kc.stop();
        control_done(control);

        while  (vidcap_params_head) {
                struct vidcap_params *next = vidcap_params_get_next(vidcap_params_head);
                vidcap_params_free_struct(vidcap_params_head);
                vidcap_params_head = next;
        }

        uv.stop();
        common_cleanup(init);

        printf("Exit\n");

        return exit_status;
}

/* vim: set expandtab sw=8: */

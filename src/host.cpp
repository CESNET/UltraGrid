/**
 * @file   host.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains common external definitions.
 */
/*
 * Copyright (c) 2013-2021 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifndef _WIN32
#include <execinfo.h>
#endif // defined WIN32

#include "host.h"

#include "audio/audio_capture.h"
#include "audio/audio_playback.h"
#include "audio/codec.h"
#include "compat/misc.h"
#include "debug.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "perf.h"
#include "rang.hpp"
#include "utils/color_out.h" // unit_evaluate
#include "utils/misc.h" // unit_evaluate
#include "video_capture.h"
#include "video_compress.h"
#include "video_display.h"
#include "capture_filter.h"
#include "video.h"

#include <array>
#include <chrono>
#include <functional>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <utility>

#if defined HAVE_X || defined BUILD_LIBRARIES
#include <dlfcn.h>
#endif
#if defined HAVE_X
#include <X11/Xlib.h>
/// @todo
/// The actual SONAME should be actually figured in configure.
#define X11_LIB_NAME "libX11.so.6"
#endif

#ifdef __linux__
#include <mcheck.h>
#endif

using rang::style;
using namespace std;

unsigned int audio_capture_channels = 0;
unsigned int audio_capture_bps = 0;
unsigned int audio_capture_sample_rate = 0;

unsigned int cuda_devices[MAX_CUDA_DEVICES] = { 0 };
unsigned int cuda_devices_count = 1;

int audio_init_state_ok;

uint32_t RTT = 0;               /*  this is computed by handle_rr in rtp_callback */

int uv_argc;
char **uv_argv;

char *export_dir = NULL;
volatile bool should_exit = false;

volatile int audio_offset; ///< added audio delay in ms (non-negative), can be used to tune AV sync
volatile int video_offset; ///< added video delay in ms (non-negative), can be used to tune AV sync

/// 0->1 - call glfwInit, 1->0 call glfwTerminate; glfw{Init,Terminate} should be called from main thr, thus no need to synchronize
int glfw_init_count;

std::unordered_map<std::string, std::string> commandline_params;

mainloop_t mainloop;
void *mainloop_udata;

struct init_data {
        list <void *> opened_libs;
};

static void print_param_doc(void);
static bool validate_param(const char *param);

void common_cleanup(struct init_data *init)
{
        if (init) {
#if defined BUILD_LIBRARIES
                for (auto a : init->opened_libs) {
                        dlclose(a);
                }
#endif
        }
        delete init;

#ifdef __linux__
        muntrace();
#endif

#ifdef WIN32
        WSACleanup();
#endif
}

ADD_TO_PARAM("stdout-buf",
         "* stdout-buf={no|line|full}\n"
         "  Buffering for stdout\n");
ADD_TO_PARAM("stderr-buf",
         "* stderr-buf={no|line|full}\n"
         "  Buffering for stderr\n");
static bool set_output_buffering() {
        const unordered_map<const char *, pair<FILE *, function<int(void)> >> outs = { // pair<output, default mode>
                { "stdout-buf", pair{stdout, [](){ return isMsysPty(fileno(stdout)) ? _IONBF : _IOLBF; }} },
                { "stderr-buf", pair{stderr, [](){ return _IONBF; }} }
        };
        for (auto outp : outs) {
                int mode = outp.second.second(); // default

                if (get_commandline_param(outp.first)) {
                        const unordered_map<string, int> buf_map {
                                { "no", _IONBF }, { "line", _IOLBF }, { "full", _IOFBF }
                        };

                        if (string("help") == get_commandline_param(outp.first)) {
                                printf("Available values for buffering are \"no\", \"line\" and \"full\"\n");
                                return false;
                        }
                        auto it = buf_map.find(get_commandline_param(outp.first));
                        if (it == buf_map.end()) {
                                log_msg(LOG_LEVEL_ERROR, "Wrong buffer type: %s\n", get_commandline_param(outp.first));
                                return false;
                        }
                        mode = it->second;
                }
                if (setvbuf(outp.second.first, NULL, mode, BUFSIZ) != 0) {
                        log_msg(LOG_LEVEL_WARNING, "setvbuf: %s\n", ug_strerror(errno));
                }
        }
        return true;
}

#ifdef HAVE_X
/**
 * Custom X11 error handler to catch errors and handle them more reasonably
 * than the default handler which exits the program immediately, which, however,
 * doesn't produce a stacktrace.
 */
static int x11_error_handler(Display *d, XErrorEvent *e) {
        //char msg[1024] = "";
        //XGetErrorText(d, e->error_code, msg, sizeof msg - 1);
        UNUSED(d);
        log_msg(LOG_LEVEL_ERROR, "X11 error - code: %d, serial: %d, error: %d, request: %d, minor: %d\n",
                        e->error_code, e->serial, e->error_code, e->request_code, e->minor_code);
        fprintf(stderr, "Backtrace:\n");
        array<void *, 256> addresses{};
        int num_symbols = backtrace(addresses.data(), addresses.size());
        backtrace_symbols_fd(addresses.data(), num_symbols, 2);
        return 0;
}
#endif

/**
 * dummy load of libgcc for backtrace() to be signal safe within crash_signal_handler() (see backtrace(3))
 */
static void load_libgcc()
{
#ifndef WIN32
        array<void *, 1> addresses{};
        backtrace(addresses.data(), addresses.size());
#endif
}

bool parse_audio_capture_format(const char *optarg)
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

        char *item = nullptr;
        char *save_ptr = nullptr;
        char *endptr = nullptr;
        char *tmp = arg;

        while ((item = strtok_r(tmp, ",:", &save_ptr))) {
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
                                if (bps % 8 != 0) {
                                        LOG(LOG_LEVEL_WARNING) << "bps is in bits per sample but a value not divisible by 8 was given.\n";
                                }
                                log_msg(LOG_LEVEL_ERROR, "Supported values are 8, 16, 24, or 32 bits.\n");
                                return false;

                        }
                        audio_capture_bps = bps / 8;
                } else if (strncmp(item, "sample_rate=", strlen("sample_rate=")) == 0) {
                        const char *sample_rate_str = item + strlen("sample_rate=");
                        long long val = unit_evaluate(sample_rate_str);
                        if (val <= 0 || val > numeric_limits<decltype(audio_capture_sample_rate)>::max()) {
                                LOG(LOG_LEVEL_ERROR) << "Invalid sample_rate " << sample_rate_str << "!\n";
                                return false;
                        }
                        audio_capture_sample_rate = val;
                } else {
                        LOG(LOG_LEVEL_ERROR) << "Unkonwn option \"" << item << "\" for --audio-capture-format!\n";
                        return false;
                }

                tmp = nullptr;
        }

        return true;
}

/**
 * Sets things that must be set before anything else (logging and params)
 *
 * (params because "std{out,err}-buf" param used by set_output_buffering())
 */
static bool parse_opts_set_logging(int argc, char *argv[])
{
        char *log_opt = nullptr;
        static struct option getopt_options[] = {
                {"param", no_argument, nullptr, OPT_PARAM}, // no_argument -- sic (!), see below
                {"verbose", optional_argument, nullptr, 'V'},
                { nullptr, 0, nullptr, 0 }
        };
        int saved_opterr = opterr;
        opterr = 0; // options are further handled in main.cpp
        // modified getopt - process all "argv[0] argv[n]" pairs to avoid permutation
        // of argv arguments - we do not have the whole option set in optstring, so it
        // would put optargs to the end ("uv -t testcard -V" -> "uv -t -V testcard")

        int logging_lvl = LOG_LEVEL_INFO;
        bool logger_skip_repeats = true;
        log_timestamp_mode logger_show_timestamps = LOG_TIMESTAMP_AUTO;

        for (int i = 1; i < argc; ++i) {
                char *my_argv[] = { argv[0], argv[i] };

                int ch = 0;
                while ((ch = getopt_long(2, my_argv, "V", getopt_options,
                            NULL)) != -1) {
                        switch (ch) {
                        case 'V':
                                if (optarg) {
                                        log_opt = optarg;
                                } else {
                                        logging_lvl += 1;
                                }
                                break;
                        case OPT_PARAM:
                                if (i == argc - 1) {
                                        fprintf(stderr, "Missing argument to \"--param\"!\n");
                                        return false;
                                }
                                if (!parse_params(argv[i + 1], true)) {
                                        return false;
                                }
                                break;
                        default: // will be handled in main
                                break;
                        }
                }
                optind = 1;
        }
        opterr = saved_opterr;

        Logger::preinit();

        if (log_opt != nullptr && !parse_log_cfg(log_opt, &logging_lvl, &logger_skip_repeats, &logger_show_timestamps)) {
                return false;
        }
        log_level = logging_lvl;
        get_log_output().set_skip_repeats(logger_skip_repeats);
        get_log_output().set_timestamp_mode(logger_show_timestamps);
        return true;
}

struct init_data *common_preinit(int argc, char *argv[])
{
        struct init_data *init;

        uv_argc = argc;
        uv_argv = argv;

        if (!parse_opts_set_logging(argc, argv)) {
                return nullptr;
        }

        if (!set_output_buffering()) {
                LOG(LOG_LEVEL_WARNING) << "Cannot set console output buffering!\n";
                return nullptr;
        }
        std::clog.rdbuf(std::cout.rdbuf()); // use stdout for logs by default
        color_output_init();

#ifdef HAVE_X
        void *handle = dlopen(X11_LIB_NAME, RTLD_NOW);

        if (handle) {
                Status (*XInitThreadsProc)();
                XInitThreadsProc = (Status (*)()) dlsym(handle, "XInitThreads");
                if (XInitThreadsProc) {
                        Status s = XInitThreadsProc();
                        if (s != True) {
                                log_msg(LOG_LEVEL_WARNING, "XInitThreads failed.\n");
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, "Unable to load symbol XInitThreads: %s\n", dlerror());
                }

                typedef int (*XSetErrorHandler_t(int (*handler)(Display *, XErrorEvent *)))();
                XSetErrorHandler_t *XSetErrorHandlerProc;
                XSetErrorHandlerProc = (XSetErrorHandler_t *) dlsym(handle, "XSetErrorHandler");
                if (XSetErrorHandlerProc) {
                        XSetErrorHandlerProc(x11_error_handler);
                } else {
                        log_msg(LOG_LEVEL_WARNING, "Unable to load symbol XSetErrorHandler: %s\n", dlerror());
                }

                dlclose(handle);
        } else {
                log_msg(LOG_LEVEL_WARNING, "Unable open " X11_LIB_NAME " library: %s\n", dlerror());
        }
#endif

#ifdef WIN32
        WSADATA wsaData;
        int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if(err != 0) {
                fprintf(stderr, "WSAStartup failed with error %d.", err);
                return nullptr;
        }
        if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
                fprintf(stderr, "Counld not found usable version of Winsock.\n");
                WSACleanup();
                return nullptr;
        }

        SetConsoleOutputCP(CP_UTF8); // see also https://stackoverflow.com/questions/1660492/utf-8-output-on-windows-console
#endif

        init = new init_data{};
        if (strstr(argv[0], "run_tests") == nullptr) {
                open_all("ultragrid_*.so", init->opened_libs); // load modules
        }

#ifdef __linux__
        mtrace();
#endif

        perf_init();
        perf_record(UVP_INIT, 0);

        load_libgcc();

        return init;
}

void print_capabilities(struct module *root, bool use_vidcap)
{
        auto flags = cout.flags();
        auto precision = cout.precision();

        // try to figure out actual input video format
        struct video_desc desc{};
        if (use_vidcap && root) {
                // try 30x in 100 ms intervals
                for (int attempt = 0; attempt < 30; ++attempt) {
                        auto t0 = std::chrono::steady_clock::now();
                        struct msg_sender *m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                        m->type = SENDER_MSG_QUERY_VIDEO_MODE;
                        struct response *r = send_message_sync(root, "sender", (struct message *) m, 100, 0);
                        if (response_get_status(r) == RESPONSE_OK) {
                                const char *text = response_get_text(r);
                                istringstream iss(text);
                                iss >> desc;
                                free_response(r);
                                break;
                        }
                        free_response(r);
                        auto query_lasted = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
                        // wait some time (100 ms - time of the last query) in order to
                        // the signal to "settle"
                        usleep(max<int>(100,query_lasted.count())*1000);
                }
        }
        cout << "[capability][start] version 4" << endl;
        // compressions
        cout << "[cap] Compressions:" << endl;
        auto compressions = get_libraries_for_class(LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

        for (auto it : compressions) {
                auto vci = static_cast<const struct video_compress_info *>(it.second);
                auto presets = vci->get_presets();
                cout << "[cap][compress] " << it.first << std::endl;
                for (auto const & it : presets) {
                        cout << "[cap] (" << vci->name << (it.name.empty() ? "" : ":") <<
                                it.name << ";" << it.quality << ";" << setiosflags(ios_base::fixed) << setprecision(2) << it.compute_bitrate(&desc) << ";" <<
                                it.enc_prop.latency << ";" << it.enc_prop.cpu_cores << ";" << it.enc_prop.gpu_gflops << ";" <<
                                it.dec_prop.latency << ";" << it.dec_prop.cpu_cores << ";" << it.dec_prop.gpu_gflops <<
                                ")\n";
                }

                if(vci->get_module_info){
                        auto module_info = vci->get_module_info();
                        cout << "[capability][video_compress] {"
                                "\"name\":" << std::quoted(it.first) << ", "
                                "\"options\": [";

                        int i = 0;
                        for(const auto& opt : module_info.opts){
                                if(i++ > 0)
                                        cout << ", ";

                                cout << "{"
                                        "\"display_name\":" << std::quoted(opt.display_name) << ", "
                                        "\"display_desc\":" << std::quoted(opt.display_desc) << ", "
                                        "\"key\":" << std::quoted(opt.key) << ", "
                                        "\"opt_str\":" << std::quoted(opt.opt_str) << ", "
                                        "\"is_boolean\":\"" << (opt.is_boolean ? "t" : "f") << "\"}";
                        }

                        cout << "], "
                                "\"codecs\": [";

                        int j = 0;
                        for(const auto& c : module_info.codecs){
                                if(j++ > 0)
                                        cout << ", ";

                                cout << "{\"name\":" << std::quoted(c.name) << ", "
                                        "\"priority\": " << c.priority << ", "
                                        "\"encoders\":[";

                                int z = 0;
                                for(const auto& e : c.encoders){
                                        if(z++ > 0)
                                                cout << ", ";

                                        cout << "{\"name\":" << std::quoted(e.name) << ", "
                                                "\"opt_str\":" << std::quoted(e.opt_str) << "}";
                                }
                                cout << "]}";
                        }

                        cout << "]}" << std::endl;

                }
        }

        // capture filters
        cout << "[cap] Capture filters:" << endl;
        auto cap_filters = get_libraries_for_class(LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

        for (auto it : cap_filters) {
                cout << "[cap][capture_filter] " << it.first << std::endl;
        }

        // capturers
        cout << "[cap] Capturers:" << endl;
        print_available_capturers();

        // displays
        cout << "[cap] Displays:" << endl;
        auto const & display_capabilities =
                get_libraries_for_class(LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
        for (auto const & it : display_capabilities) {
                auto vdi = static_cast<const struct video_display_info *>(it.second);
                int count = 0;
                struct device_info *devices;
                void (*deleter)(void *) = nullptr;
                vdi->probe(&devices, &count, &deleter);
                cout << "[cap][display] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[capability][device] {"
                                "\"purpose\":\"video_disp\", "
                                "\"module\":" << std::quoted(it.first) << ", "
                                "\"device\":" << std::quoted(devices[i].dev) << ", "
                                "\"name\":" << std::quoted(devices[i].name) << ", "
                                "\"extra\": {" << devices[i].extra << "}, "
                                "\"repeatable\":\"" << devices[i].repeatable << "\"}\n";
                }
                deleter ? deleter(devices) : free(devices);
        }

        cout << "[cap] Audio capturers:" << endl;
        auto const & audio_cap_capabilities =
                get_libraries_for_class(LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
        for (auto const & it : audio_cap_capabilities) {
                auto aci = static_cast<const struct audio_capture_info *>(it.second);
                int count = 0;
                struct device_info *devices;
                aci->probe(&devices, &count);
                cout << "[cap][audio_cap] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[capability][device] {"
                                "\"purpose\":\"audio_cap\", "
                                "\"module\":" << std::quoted(it.first) << ", "
                                "\"device\":" << std::quoted(devices[i].dev) << ", "
                                "\"extra\": {" << devices[i].extra << "}, "
                                "\"name\":" << std::quoted(devices[i].name) << "}\n";
                }
                free(devices);
        }

        cout << "[cap] Audio playback:" << endl;
        auto const & audio_play_capabilities =
                get_libraries_for_class(LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);
        for (auto const & it : audio_play_capabilities) {
                auto api = static_cast<const struct audio_playback_info *>(it.second);
                int count = 0;
                struct device_info *devices;
                api->probe(&devices, &count);
                cout << "[cap][audio_play] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[capability][device] {"
                                "\"purpose\":\"audio_play\", "
                                "\"module\":" << std::quoted(it.first) << ", "
                                "\"device\":" << std::quoted(devices[i].dev) << ", "
                                "\"extra\": {" << devices[i].extra << "}, "
                                "\"name\":" << std::quoted(devices[i].name) << "}\n";
                }
                free(devices);
        }

        // audio compressions
        auto codecs = get_audio_codec_list();
        for(const auto& codec : codecs){
                cout << "[cap][audio_compress] " << codec.first << std::endl;
        }

        cout << "[capability][end]" << endl;

        cout.flags(flags);
        cout.precision(precision);
}

const char *get_version_details()
{
        return
#ifdef GIT_BRANCH
                GIT_BRANCH " "
#endif
#ifdef GIT_REV
                "rev " GIT_REV " "
#endif
                "built " __DATE__ " " __TIME__;
}

void print_version()
{
        bool is_release = true;
#ifdef GIT_BRANCH
        if (strstr(GIT_BRANCH, "release") == nullptr &&
                        strstr(GIT_BRANCH, "tags/v") == nullptr) {
                is_release = false;
        }
#endif
        cout << rang::fg::bright_blue << rang::style::bold << PACKAGE_STRING <<
                (is_release ? "" : "+") <<
                rang::fg::reset << rang::style::reset << " (" <<
                get_version_details() << ")\n";
}

void print_configuration()
{
        const char *config_flags = CONFIG_FLAGS;
        if (strlen(config_flags) == 0) {
                config_flags = "(none)";
        }
        printf("configuration flags: %s", config_flags);
        printf("\n\n");
        printf(PACKAGE_NAME " was compiled with following features:\n");
        printf(AUTOCONF_RESULT);
}

const char *get_commandline_param(const char *key)
{
	auto it = commandline_params.find(key);
	if (it != commandline_params.end()) {
                return it->second.c_str();
	} else {
		return NULL;
        }
}

int get_audio_delay(void)
{
        return audio_offset > 0 ? audio_offset : -video_offset;
}

void set_audio_delay(int audio_delay)
{
	audio_offset = max(audio_delay, 0);
	video_offset = audio_delay < 0 ? abs(audio_delay) : 0;
}

static struct {
        const char *param;
        const char *doc;
} params[100];

/**
 * Registers param to param database ("--param"). If param given multiple times, only
 * first value is stored.
 */
void register_param(const char *param, const char *doc)
{
        assert(param != NULL && doc != NULL);
        for (unsigned int i = 0; i < sizeof params / sizeof params[0]; ++i) {
                if (params[i].param == NULL) {
                        params[i].param = param;
                        params[i].doc = doc;
                        return;
                }
                if (strcmp(params[i].param, param) == 0) {
                        if (strcmp(params[i].doc, doc) != 0) {
                                log_msg(LOG_LEVEL_WARNING, "Param \"%s\" as it is already registered but with different documentation.\n", param);
                        }
                        return;
                }
        }
        log_msg(LOG_LEVEL_WARNING, "Cannot register param \"%s\", maxmimum number of parameters reached.\n", param);
}

bool validate_param(const char *param)
{
        for (unsigned int i = 0; i < sizeof params / sizeof params[0]; ++i) {
                if (params[i].param == NULL) {
                        return false;
                }
                if (strcmp(params[i].param, param) == 0) {
                        return true;
                }
        }
        return false;
}


/**
 * Parses command-line parameters given as "--param <key>=<val>[...".
 *
 * @param optarg   command-line arguments given to "--param", must not be NULL
 * @param preinit  true  - only parse/set known parameters (notably output buffer setting), ignore help
 *                 false - set also remaining parameters including full check and "help" output
 *
 * @note
 * This function will be usually called twice - first with preinit=true and then with false
 */
bool parse_params(const char *optarg, bool preinit)
{
        if (!preinit && strcmp(optarg, "help") == 0) {
                puts("Use of params below is experimental and should be used with a caution and a knowledge of consequences and affected functionality!\n");
                puts("Params can be one or more (separated by comma) of following:");
                print_param_doc();
                return false;
        }
        char *tmp = strdupa(optarg);
        char *item = nullptr;
        char *save_ptr = nullptr;
        while ((item = strtok_r(tmp, ",", &save_ptr)) != nullptr) {
                tmp = nullptr;
                char *key_cstr = item;
                const char *val_cstr = "";
                if (char *delim = strchr(item, '=')) {
                        val_cstr = delim + 1;
                        *delim = '\0';
                }
                if (!validate_param(key_cstr)) {
                        if (preinit) {
                                continue;
                        }
                        LOG(LOG_LEVEL_ERROR) << "Unknown parameter: " << key_cstr << "\n";
                        LOG(LOG_LEVEL_INFO) << "Type '" << uv_argv[0] << " --param help' for list.\n";
                        return false;
                }
                commandline_params[key_cstr] = val_cstr;
        }
        return true;
}

void print_param_doc()
{
        for (unsigned int i = 0; i < sizeof params / sizeof params[0]; ++i) {
                if (params[i].doc != NULL) {
                        puts(params[i].doc);
                } else {
                        break;
                }
        }
}

void print_pixel_formats(void) {
        cout << "codec RGB/YCbCr depth description\n";
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = static_cast<codec_t>(static_cast<int>(c) + 1)) {
                char tag;
                if (is_codec_opaque(c)) {
                        continue;
                }

                tag = codec_is_a_rgb(c) ? 'R' : 'Y';
                auto width = cout.width();
                auto flags = cout.flags();
                cout << " " << style::bold << left << setw(12) << get_codec_name(c) << style::reset << setw(0) << " " << tag << " " << setw(2) << get_bits_per_component(c) << setw(0) << "   " << get_codec_name_long(c) << setw(width) << "\n";
                cout.flags(flags);
        }
}

void print_video_codecs(void) {
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = static_cast<codec_t>(static_cast<int>(c) + 1)) {
                char tag;
                if (!is_codec_opaque(c)) {
                        continue;
                }

                tag = is_codec_interframe(c) ? 'I' : '.';
                auto width = cout.width();
                auto flags = cout.flags();
                cout << " " << style::bold << left << setw(12) << get_codec_name(c) << style::reset << setw(0) << " " << tag << " " << "   " << get_codec_name_long(c) << setw(width) << "\n";
                cout.flags(flags);
        }
        cout << "\nLegend:\n" << " I - interframe codec\n";
}

bool register_mainloop(mainloop_t m, void *u)
{
        if (mainloop) {
                return false;
        }

        mainloop = m;
        mainloop_udata = u;

        return true;
}

void register_should_exit_callback(struct module *mod, void (*callback)(void *), void *udata)
{
        auto m = (struct msg_root *) new_message(sizeof(struct msg_root));
        m->type = ROOT_MSG_REGISTER_SHOULD_EXIT;
        m->should_exit_callback = callback;
        m->udata = udata;

        struct response *r = send_message_sync(get_root_module(mod), "root", (struct message *) m, -1, SEND_MESSAGE_FLAG_NO_STORE);
        assert(response_get_status(r) == RESPONSE_OK);
        free_response(r);
}

ADD_TO_PARAM("errors-fatal", "* errors-fatal\n"
                "  Treats every error as a fatal (exits " PACKAGE_NAME ")\n");
/**
 * Soft version of exit_uv() checks errors-fatal command-line parameters and
 * if set, exit UltraGrid. Otherwise error is ignored.
 *
 * Caller code normally continues after this function so the error must not
 * have been fatal and UltraGrid must remain in a consistent state.
 */
void handle_error(int status) {
        if (get_commandline_param("errors-fatal")) {
                exit_uv(status);
        }
}

// some common parameters used within multiple modules
ADD_TO_PARAM("audio-buffer-len", "* audio-buffer-len=<ms>\n"
                "  Sets length of software audio playback buffer (in ms, ALSA/Coreaudio/Portaudio/WASAPI)\n");
ADD_TO_PARAM("audio-cap-frames", "* audio-cap-frames=<f>\n"
                "  Sets number of audio frames captured at once (CoreAudio)\n");
ADD_TO_PARAM("audio-disable-adaptive-buffer", "* audio-disable-adaptive-buffer\n"
                "  Disables audio adaptive playback buffer (CoreAudio/JACK)\n");
ADD_TO_PARAM("color", "* color=CT\n"
                "  [experimental] Color space to use, C - colorimetry: 0 - undefined, 1 - BT.709, 2 - BT.2020/2100, 3 - P3; T - transfer fn: 0 - undefined, 1 - 709, 2 - HLG; 3 - PQ (signalized to GLFW on mac, NDI receiver)\n");
#ifdef DEBUG
ADD_TO_PARAM("debug-dump", "* debug-dump=<module>[=<n>][,<module2>[=<n>]\n"
                "  Dumps specified buffer for debugging, n-th buffer may be selected, name is <module>.dump.\n"
                "  Avaiable modules: lavd-uncompressed\n");
#endif
ADD_TO_PARAM("low-latency-audio", "* low-latency-audio[=ultra]\n"
                "  Try to reduce audio latency at the expense of worse reliability\n"
                "  Add ultra for even more aggressive setting.\n");
ADD_TO_PARAM("window-title", "* window-title=<title>\n"
                "  Use alternative window title (SDL/GL only)\n");


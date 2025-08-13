/**
 * @file   host.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains common external definitions.
 */
/*
 * Copyright (c) 2013-2025 CESNET
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
#endif

#include "host.h"

#ifdef _WIN32
#include <io.h>
#elif defined(__APPLE__) || defined(__GLIBC__)
#include <execinfo.h>
#include <fcntl.h>
#endif // !defined _WIN32

#ifdef __gnu_linux__
#include <features.h>                   // for __GLIBC__, __GLIBC_MINOR__
#include <sys/mman.h>                   // for memfd_create, MFD_CLOEXEC
#endif

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 27
#include <sys/syscall.h>
#endif

#ifdef HAVE_LIBBACKTRACE
#include <backtrace.h>
#else
struct backtrace_state {};
#endif

#include <algorithm>                    // for max
#include <array>
#include <cassert>                      // for assert
#include <cerrno>                       // for errno
#include <cmath>                        // for abs
#include <csignal>                      // for signal, SIGALRM, SIG_DFL, raise
#include <cstdint>                      // for uint32_t
#include <cstdio>                       // for printf, puts, perror, _IONBF
#include <cstdlib>                      // for abort, getenv, free, abs, EXI...
#include <cstring>                      // for strlen, strcmp, NULL, strchr
#include <getopt.h>                     // for getopt, optarg, opterr
#include <iterator>                     // for size
#include <map>                          // for map, _Rb_tree_iterator, opera...
#include <mutex>                        // for mutex, unique_lock
#include <string_view>                  // for operator<<, operator==, string...
#include <sys/types.h>                  // for ssize_t
#include <tuple>                        // for tuple, get, make_tuple
#include <unistd.h>                     // for STDERR_FILENO

#include "audio/audio_capture.h"
#include "audio/audio_filter.h"
#include "audio/audio_playback.h"
#include "audio/codec.h"
#include "audio/types.h"                // for audio_desc
#include "audio/utils.h"
#include "capture_filter.h"
#include "compat/platform_pipe.h"
#include "compat/net.h"                 // for fd_t
#include "cuda_wrapper.h"               // for cudaDeviceReset
#include "debug.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "module.h"
#include "types.h"                      // for device_info, device_option
#include "utils/color_out.h"
#include "utils/fs.h"                   // for MAX_PATH_SIZE
#include "utils/misc.h" // ug_strerror
#include "utils/random.h"
#include "utils/string.h"
#include "utils/string_view_utils.hpp"
#include "utils/text.h"
#include "utils/thread.h"
#include "utils/windows.h"
#include "video_capture.h"
#include "video_codec.h"                // for get_codec_name, get_codec_nam...
#include "video_compress.h"
#include "video_display.h"

#include <iomanip>
#include <iostream>
#include <list>
#include <fstream>
#include <string>
#include <thread>
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

#ifdef HAVE_FEC_INIT
#define restrict __restrict // not a C++ keyword
extern "C" {
#include <fec.h>
}
#endif

#ifdef __gnu_linux__
#include <mcheck.h>
#endif

#define MOD_NAME "[host] "

using std::array;
using std::cout;
using std::endl;
using std::get;
using std::ifstream;
using std::left;
using std::list;
using std::make_tuple;
using std::max;
using std::mutex;
using std::pair;
using std::setw;
using std::string;
using std::thread;
using std::to_string;
using std::tuple;
using std::unordered_map;
using std::unique_lock;

unsigned int audio_capture_channels = 0;
unsigned int audio_capture_bps = 0;
unsigned int audio_capture_sample_rate = 0;

unsigned int cuda_devices[MAX_CUDA_DEVICES] = { 0 };
unsigned int cuda_devices_count = 1;
bool cuda_devices_explicit = false;

uint32_t RTT = 0;               /*  this is computed by handle_rr in rtp_callback */

int uv_argc;
char **uv_argv;

char *export_dir = NULL;

volatile int audio_offset; ///< added audio delay in ms (non-negative), can be used to tune AV sync
volatile int video_offset; ///< added video delay in ms (non-negative), can be used to tune AV sync

/// 0->1 - call glfwInit, 1->0 call glfwTerminate; glfw{Init,Terminate} should be called from main thr, thus no need to synchronize
int glfw_init_count;

std::unordered_map<std::string, std::string> commandline_params;

mainloop_t mainloop;
void *mainloop_udata;

#if defined HAVE_CUDA && defined _WIN32
// required for NVCC+MSVC compiled objs if /nodefaultlib is used
extern "C" int _fltused = 0;
#endif

static struct backtrace_state *bt;

struct init_data {
        bool com_initialized = false;
        list <void *> opened_libs;
};

static void print_backtrace();
static void print_param_doc(void);
static bool validate_param(const char *param);

static bool unexpected_exit_called = true; // check for unexpected exit()
void common_cleanup(struct init_data *init)
{
        if (init) {
#if defined BUILD_LIBRARIES
                for (auto a : init->opened_libs) {
                        dlclose(a);
                }
#endif
                com_uninitialize(&init->com_initialized);
        }
        delete init;

#ifdef __gnu_linux__
        muntrace();
#endif

#ifdef _WIN32
        WSACleanup();
#endif

#if defined CUDA_DEVICE_RESET
        // to allow "cuda-memcheck --leak-check full"
        cuda_wrapper_device_reset();
#endif

        unexpected_exit_called = false;
}

ADD_TO_PARAM("stdout-buf",
         "* stdout-buf={no|line|full}\n"
         "  Buffering for stdout\n");
ADD_TO_PARAM("stderr-buf",
         "* stderr-buf={no|line|full}\n"
         "  Buffering for stderr\n");
static bool set_output_buffering() {
        const unordered_map<const char *, pair<FILE *, int>> outs = { // pair<output, default mode>
#ifdef _WIN32
                { "stdout-buf", pair{stdout, _IONBF} },
#else
                { "stdout-buf", pair{stdout, _IOLBF} },
#endif
                { "stderr-buf", pair{stderr, _IONBF} }
        };
        for (const auto& outp : outs) {
                int mode = outp.second.second; // default

                if(running_in_debugger()){
                        mode = _IONBF;
                        log_msg(LOG_LEVEL_WARNING, "Running inside debugger - disabling output buffering.\n");
                }

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
        print_backtrace();
        return 0;
}
#endif

/**
 * dummy load of libgcc for backtrace() to be signal safe within crash_signal_handler() (see backtrace(3))
 */
static void load_libgcc()
{
#if !defined(_WIN32) && defined(__GLIBC__)
        array<void *, 1> addresses{};
        backtrace(addresses.data(), addresses.size());
#endif
}

/**
 * @retval -1 invalid usage
 * @retval  0 success
 * @retval  1 help was printed
 */
int set_audio_capture_format(const char *optarg)
{
        struct audio_desc desc = {};
        if (int ret = parse_audio_format(optarg, &desc)) {
                return ret;
        }

        audio_capture_bps = IF_NOT_NULL_ELSE(desc.bps, audio_capture_bps);
        audio_capture_channels = IF_NOT_NULL_ELSE(desc.ch_count, audio_capture_channels);
        audio_capture_sample_rate = IF_NOT_NULL_ELSE(desc.sample_rate, audio_capture_sample_rate);

        return 0;
}

int set_pixfmt_conv_policy(const char *optarg) {
        if (strcmp(optarg, "help") == 0) {
                char desc[] =
                        TBOLD("--conv-policy") " specifies the order in which various pixfmt properties are to be evaluated "
                        "if some pixel format needs conversion to another suitable pixel format.";
                 color_printf("%s\n\n", wrap_paragraph(desc));
                 color_printf("\t" TBOLD("c") " - color space\n");
                 color_printf("\t" TBOLD("d") " - bit depth\n");
                 color_printf("\t" TBOLD("s") " - subsampling\n");
                 color_printf("\nDefault: \"" TBOLD("dsc") "\" - first is respected bit-depth, then subsampling and finally color space\n\n"
                        "Permute the above letters to change the default order, eg. \"" TBOLD("cds") "\" to attempt to keep colorspace.\n");
                 return 1;
        }
        if (strlen(optarg) != strlen(pixfmt_conv_pref)) {
                log_msg(LOG_LEVEL_ERROR,  "Wrong pixfmt conversion policy length (exactly 3 letters need to be used)!\n");
                return -1;
        }
        if (strchr(optarg, 'd') == NULL || strchr(optarg, 's') == NULL || strchr(optarg, 'c') == NULL) {
                log_msg(LOG_LEVEL_ERROR,  "Wrong pixfmt conversion policy - use exactly the set 'dsc'!\n");
                return -1;
        }
        memcpy(pixfmt_conv_pref, optarg, strlen(pixfmt_conv_pref));
        return 0;
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
                {"param", required_argument, nullptr, 'O'},
                {"verbose", optional_argument, nullptr, 'V'},
                { nullptr, 0, nullptr, 0 }
        };
        const char *const optstring = "+O:V";
        int saved_opterr = opterr;
        opterr = 0; // options are further handled in main.cpp. skip unknown

        int logging_lvl = 0;
        bool logger_skip_repeats = true;

        log_timestamp_mode logger_show_timestamps = LOG_TIMESTAMP_AUTO;

        while (optind < argc) {
                const int ch =
                    getopt_long(argc, argv, optstring, getopt_options, nullptr);
                switch (ch) {
                case 'V':
                        if (optarg != nullptr) {
                                log_opt = optarg;
                        } else {
                                logging_lvl += 1;
                        }
                        break;
                case 'O':
                        if (!parse_params(optarg, true)) {
                                return false;
                        }
                        break;
                case -1: // skip a non-option (or an argument to an option that
                         // is not recognized by this getopt)
                        optind += 1;
                        break;
                case '?': // other option that will be handled in main
                        break;
                default:
                        abort(); // shouldn't reach here
                }
        }
        optind = 0;
        opterr = saved_opterr;
        if (logging_lvl == 0) {
                logging_lvl = getenv("ULTRAGRID_VERBOSE") != nullptr && strlen(getenv("ULTRAGRID_VERBOSE")) > 0 ? LOG_LEVEL_VERBOSE : log_level;
        } else {
                logging_lvl += LOG_LEVEL_INFO;
        }
        if (log_opt != nullptr && !parse_log_cfg(log_opt, &logging_lvl, &logger_skip_repeats, &logger_show_timestamps)) {
                return false;
        }
        log_level = logging_lvl;
        get_log_output().set_skip_repeats(logger_skip_repeats);
        get_log_output().set_timestamp_mode(logger_show_timestamps);
        return true;
}

bool
tok_in_argv(char **argv, const char *tok)
{
        while (*argv != nullptr) {
                if (strstr(*argv, tok) != nullptr) {
                        return true;
                }
                argv++;
        }
        return false;
}

static void echeck_unexpected_exit(void ) {
        if (!unexpected_exit_called) {
                return;
        }
        fprintf(stderr, "exit() called unexpectedly! Maybe by some library?\n");
}

#ifdef HAVE_LIBBACKTRACE
static void
libbt_error_callback(void *data, const char *msg, int errnum)
{
        int fd = *reinterpret_cast<int*>(data);
        char buf[STR_LEN];
        char *start = buf;
        const char *const end = buf + sizeof buf;

        //fprintf(stderr, "libbacktrace error: %s (%d)\n", msg, errnum);

        strappend(&start, end, "libbacktrace error: ");
        strappend(&start, end, msg);
        strappend(&start, end, " (");
        append_number(&start, end, errnum);

        write_all(fd, start - buf, buf);
}

static int
libbt_full_callback(void *data, uintptr_t pc, const char *filename, int lineno,
              const char *function)
{
        int fd = *reinterpret_cast<int*>(data);
        char buf[STR_LEN];
        char *start = buf;
        const char *const end = buf + sizeof buf;

        // printf("  %s at %s:%d [pc=%p]\n", function ? function : "??",
        //        filename ? filename : "??", lineno, (void *) pc);

        strappend(&start, end, "  ");
        if (function == nullptr) {
                function = "??";
        }
        strappend(&start, end, function);
        strappend(&start, end, " at ");
        if (filename == nullptr) {
                filename = "??";
        }
        strappend(&start, end, filename);
        strappend(&start, end, ":");
        append_number(&start, end, lineno);
        strappend(&start, end, " [pc=0x");
        append_number(&start, end, (uintmax_t) pc);
        strappend(&start, end, "]\n");

        write_all(fd, start - buf, buf);

        return 0; // continue
}
#endif // defined HAVE_LIBBACKTRACE

struct init_data *common_preinit(int argc, char *argv[])
{
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

        struct init_data init{};
#ifdef _WIN32
        WSADATA wsaData;
        int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if(err != 0) {
                fprintf(stderr, "WSAStartup failed with error %d.", err);
                return nullptr;
        }
        if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
                fprintf(stderr, "Could not found usable version of Winsock.\n");
                WSACleanup();
                return nullptr;
        }

        SetConsoleOutputCP(CP_UTF8); // see also https://stackoverflow.com/questions/1660492/utf-8-output-on-windows-console

        // Initialize COM on main thread - otherwise Portaudio would initialize it as COINIT_APARTMENTTHREADED but MULTITHREADED
        // is perhaps better variant (Portaudio would accept that).
        const bool init_com = !tok_in_argv(argv, "screen:unregister_elevated");
        if (init_com) {
                com_initialize(&init.com_initialized, nullptr);
        }

        // warn in W10 "legacy" terminal emulators
        if (getenv("TERM") == nullptr &&
            _isatty(fileno(stdout)) &&
            get_windows_build() < BUILD_WINDOWS_11_OR_LATER &&
            (win_has_ancestor_process("powershell.exe") ||
             win_has_ancestor_process("cmd.exe")) &&
            !win_has_ancestor_process("WindowsTerminal.exe")) {
                MSG(WARNING, "Running inside PS/cmd terminal is not recommended "
                             "because scrolling the output freezes the process, "
                             "consider using Windows Terminal instead!\n");
                Sleep(1000);
        }
#endif

        if (strstr(argv[0], "run_tests") == nullptr) {
                open_all("ultragrid_*.so", init.opened_libs); // load modules
        }

        ug_rand_init();

#ifdef __gnu_linux__
        mtrace();
#endif

        load_libgcc();

#ifdef HAVE_FEC_INIT
        fec_init();
#endif

#ifdef HAVE_LIBBACKTRACE
        int fd = STDERR_FILENO;
        bt = backtrace_create_state(uv_argv[0], 1 /*thread safe*/,
                                    libbt_error_callback, &fd);
#endif

        atexit(echeck_unexpected_exit);

        return new init_data{ std::move(init) };
}

struct state_root {
        state_root() noexcept {
                if (platform_pipe_init(should_exit_pipe) != 0) {
                        LOG(LOG_LEVEL_FATAL) << "FATAL: Cannot create pipe!\n";
                        abort();
                }
                should_exit_thread = thread(should_exit_watcher, this);
        }
        ~state_root() {
                unique_lock<mutex> lk(lock);
                should_exit_callbacks.clear();
                lk.unlock();
                broadcast_should_exit(true); // here just exit the should exit thr
                should_exit_thread.join();
                for (int i = 0; i < 2; ++i) {
                        platform_pipe_close(should_exit_pipe[0]);
                }
        }

        static void should_exit_watcher(state_root *s) {
                set_thread_name(__func__);
                char q = 0;
                bool should_exit_thread_notified = false;
                while (q != QUIT_WATCHER_FLAG) {
                        while (PLATFORM_PIPE_READ(s->should_exit_pipe[0], &q,
                                                  1) != 1) {
                                perror("PLATFORM_PIPE_READ");
                        }
                        if (!should_exit_thread_notified) {
                                unique_lock<mutex> lk(s->lock);
                                for (auto const &c : s->should_exit_callbacks) {
                                        get<0>(c)(get<1>(c));
                                }
                                should_exit_thread_notified = true;
                        }
                }
        }
        void broadcast_should_exit(bool quit_watcher = false)
        {
                const char q = quit_watcher ? QUIT_WATCHER_FLAG : 0;
                while (PLATFORM_PIPE_WRITE(should_exit_pipe[1], &q, 1) != 1) {
                        perror("PLATFORM_PIPE_WRITE");
                }
        }

        volatile int exit_status = EXIT_SUCCESS;
private:
        static constexpr char QUIT_WATCHER_FLAG = 1;
        mutex lock;
        fd_t should_exit_pipe[2];
        thread should_exit_thread;
        list<tuple<void (*)(void *), void *>> should_exit_callbacks;
        friend void register_should_exit_callback(struct module *mod,
                                                  void (*callback)(void *),
                                                  void *udata);
        friend void unregister_should_exit_callback(struct module *mod,
                                                    void (*callback)(void *),
                                                    void *udata);
};

static state_root * volatile state_root_static; ///< used by exit_uv() called from signal handler

/**
 * Initializes root module
 *
 * This the root module is also responsible for regiestering and broadcasting
 * should_exit events (see register_should_exit_callback) called by exit_uv().
 */
void init_root_module(struct module *root_mod) {
        module_init_default(root_mod);
        root_mod->cls = MODULE_CLASS_ROOT;
        root_mod->new_message = nullptr; // note that the root mod messages
                                         // processes also the reflector
        state_root_static = new state_root();
        root_mod->priv_data = state_root_static;
        module_register(root_mod, nullptr);
}

void destroy_root_module(struct module *root_mod) {
        delete (state_root *) root_mod->priv_data;
}

/**
 * Exit function that sets return value and brodcasts registered modules should_exit.
 *
 * Should be called after init_root_module() is called.
 */
void exit_uv(int status) {
        if (!state_root_static) {
                log_msg(LOG_LEVEL_WARNING, "%s called without state registered.\n", __func__);
        }
        state_root_static->exit_status = status;
        state_root_static->broadcast_should_exit();
}

int get_exit_status(struct module *root_mod) {
        assert(root_mod->cls == MODULE_CLASS_ROOT);
        return static_cast<state_root *>(root_mod->priv_data)->exit_status;
}

using module_info_map = std::map<std::string, const void *>;

static void print_device(std::string purpose, std::string const & mod, const device_info& device){
        cout << "[capability][device] {"
                "\"purpose\":" << std::quoted(purpose) << ", "
                "\"module\":" << std::quoted(mod) << ", "
                "\"device\":" << std::quoted(device.dev) << ", "
                "\"name\":" << std::quoted(device.name) << ", "
                "\"extra\": {" << device.extra << "}, "
                "\"repeatable\":\"" << device.repeatable << "\", "
                "\"modes\": [";

        for(unsigned int j = 0; j < std::size(device.modes); j++) {
                if (device.modes[j].id[0] == '\0') { // last item
                        break;
                }
                if (j > 0) {
                        printf(", ");
                }
                std::cout << "{\"name\":" << std::quoted(device.modes[j].name) << ", "
                        "\"opts\":" << device.modes[j].id << "}";
        }
        std::cout << "]";

        std::cout << ", \"options\": [";
        for(unsigned int j = 0; j < std::size(device.options); j++) {
                if (device.options[j].key[0] == '\0') { // last item
                        break;
                }
                if (j > 0) {
                        printf(", ");
                }
                cout << "{"
                    "\"display_name\":" << std::quoted(device.options[j].display_name) << ", "
                    "\"display_desc\":" << std::quoted(device.options[j].display_desc) << ", "
                    "\"key\":" << std::quoted(device.options[j].key) << ", "
                    "\"opt_str\":" << std::quoted(device.options[j].opt_str) << ", "
                    "\"is_boolean\":\"" << (device.options[j].is_boolean ? "t" : "f") << "\"}";
        }
        std::cout << "]";

        std::cout << "}\n";
}

template<typename T>
static void probe_device(std::string_view cap_str, std::string const & name, const void *mod){
        auto vdi = static_cast<T>(mod);
        int count = 0;
        struct device_info *devices = nullptr;
        void (*deleter)(void *) = nullptr;
        vdi->probe(&devices, &count, &deleter);
        for (int i = 0; i < count; ++i) {
                print_device(std::string(cap_str), name, devices[i]);
        }
        deleter ? deleter(devices) : free(devices);
}

static void probe_compress(std::string const & name, const void *mod) noexcept {
        auto vci = static_cast<const struct video_compress_info *>(mod);

        if(vci->get_module_info){
                auto module_info = vci->get_module_info();
                cout << "[capability][video_compress] {"
                        "\"name\":" << std::quoted(name) << ", "
                        "\"options\": [";

                int i = 0;
                for(const auto& opt : module_info.opts){
                        if(i++ > 0)
                                cout << ", ";

                        cout << "{"
                                "\"display_name\":" << std::quoted(opt.display_name) << ", "
                                "\"display_desc\":" << std::quoted(opt.display_desc) << ", "
                                "\"placeholder_text\":" << std::quoted(opt.placeholder_text) << ", "
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

const static struct {
        std::string_view desc;
        std::string_view cap_str;
        enum library_class cls;
        int abi_ver;
        void (*probe_print)(std::string name, const void *);
} mod_classes[] = {
        {"Compressions", "compress",
                LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION,
                [](std::string name, const void *m) { probe_compress(name, m); }},
        {"Capture filters", "capture_filter",
                LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION,
                nullptr},
        {"Capturers", "capture",
                LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION,
                [](std::string name, const void *m){ probe_device<const video_capture_info *>("capture", name, m); }},
        {"Displays", "display",
                LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION,
                [](std::string name, const void *m){ probe_device<const video_display_info *>("video_disp", name, m); }},
        {"Audio capturers", "audio_cap",
                LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION,
                [](std::string name, const void *m){ probe_device<const audio_capture_info *>("audio_cap", name, m); }},
        {"Audio filters", "audio_filter",
                LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION,
                nullptr},
        {"Audio compress", "audio_compress",
                LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION,
                nullptr},
        {"Audio playback", "audio_play",
                LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION,
                [](std::string name, const void *m){ probe_device<const audio_playback_info *>("audio_play", name, m); }},
};


static void probe_all(std::map<enum library_class, module_info_map>& class_mod_map)
{
        for(const auto& mod_class : mod_classes){
                for(const auto& mod : class_mod_map[mod_class.cls]){
                        if(!mod_class.probe_print)
                                continue;
                        mod_class.probe_print(mod.first, mod.second);
                }
        }
}

static void print_modules(std::map<enum library_class, module_info_map>& class_mod_map)
{
        for(const auto& mod_class : mod_classes){
                std::cout << "[cap] " << mod_class.desc << ":\n";
                for(const auto& mod : class_mod_map[mod_class.cls]){
                        cout << "[cap][" << mod_class.cap_str <<"] " << mod.first << "\n";
                }
        }
}

void print_capabilities(const char *cfg)
{
        std::string_view conf(cfg);

        std::cout << "[capability][start] version 4" << endl;

        std::map<enum library_class, module_info_map> class_mod_map;
        for(const auto& mod_class: mod_classes){
                class_mod_map.emplace(mod_class.cls,
                                get_libraries_for_class(mod_class.cls, mod_class.abi_ver));
        }
        auto codecs = get_audio_codec_list();
        for(const auto& codec : codecs){
                class_mod_map[LIBRARY_CLASS_AUDIO_COMPRESS].emplace(get<0>(codec).name, nullptr);
        }

        if(conf == "noprobe"){
                print_modules(class_mod_map);
        } else if(conf.empty()){
                print_modules(class_mod_map);
                probe_all(class_mod_map);
        } else {
                auto class_sv = tokenize(conf, ':');
                auto mod_sv = tokenize(conf, ':');

                enum library_class cls = LIBRARY_CLASS_UNDEFINED;
                void (*probe_print)(std::string name, const void *) = nullptr;
                for(const auto& i : mod_classes){
                        if(i.cap_str == class_sv){
                                cls = i.cls;
                                probe_print = i.probe_print;
                        }
                }

                if(cls == LIBRARY_CLASS_UNDEFINED){
                        log_msg(LOG_LEVEL_FATAL, "Unknown library class\n");
                        return;
                }

                auto& modmap = class_mod_map[cls];

                auto modinfo = modmap.find(std::string(mod_sv));
                if(modinfo == modmap.end()){
                        log_msg(LOG_LEVEL_FATAL, "Module not found\n");
                        return;
                }

                if(probe_print)
                        probe_print(std::string(mod_sv), modinfo->second);

        }

        cout << "[capability][end]" << endl;
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

        col() << SBOLD(S256_FG(T_TOMATO, PACKAGE_STRING <<
                (is_release ? "" : "+"))) <<
                " (" << get_version_details() << ")\n";
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

void set_commandline_param(const char *key, const char *val)
{
        commandline_params[key] = val;
}

int get_audio_delay(void)
{
        return audio_offset > 0 ? audio_offset : -video_offset;
}

void set_audio_delay(int audio_delay)
{
        log_msg(LOG_LEVEL_NOTICE, "Setting A/V delay: %d\n", audio_delay);
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
        log_msg(LOG_LEVEL_WARNING, "Cannot register param \"%s\", maximum number of parameters reached.\n", param);
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
 * This function will be usually called twice - first with preinit=true and then with false.
 * It is because the buffering parameter need to be set early (prior to any output). On the
 * other hand not all params can be set immediately -- modules are not yet registered, so it
 * is called once more the preinit is done.
 */
bool parse_params(const char *optarg, bool preinit)
{
        if (!preinit && strcmp(optarg, "help") == 0) {
                puts("Use of params below is experimental and should be used with a caution and a knowledge of consequences and affected functionality!\n");
                puts("Params can be one or more (separated by comma) of following:");
                print_param_doc();
                return false;
        }
        std::string_view sv = optarg;
        while (!sv.empty()) {
                std::string key = std::string(tokenize(sv, ','));
                std::string val;
                std::string::size_type delim_pos = key.find('=');
                if (delim_pos != std::string::npos) {
                        val = key.substr(delim_pos + 1);
                        key.resize(delim_pos);
                }
                if (!validate_param(key.c_str())) {
                        if (preinit) {
                                continue;
                        }
                        LOG(LOG_LEVEL_ERROR) << "Unknown parameter: " << key << "\n";
                        LOG(LOG_LEVEL_INFO) << "Type '" << uv_argv[0] << " --param help' for list.\n";
                        if (get_commandline_param("allow-unknown-params") ==
                            nullptr) {
                                return false;
                        }
                }
                commandline_params[key] = std::move(val);
        }
        return true;
}

bool
parse_bitrate(char *optarg, long long int *bitrate)
{
        unordered_map<string, long long int> bitrate_spec_map = {
                {"auto",       RATE_AUTO     },
                { "dynamic",   RATE_DYNAMIC  },
                { "unlimited", RATE_UNLIMITED},
        };

        if (auto it = bitrate_spec_map.find(optarg);
            it != bitrate_spec_map.end()) {
                *bitrate = it->second;
                return true;
        }
        if (strcmp(optarg, "help") == 0) {
                constexpr char const *NUMERIC_PATTERN = "[1-9][0-9]*[kMG][!][E]";
                col()
                    << "Usage:\n"
                    << "\tuv " << TERM_BOLD "-l [auto | dynamic | unlimited | "
                    << NUMERIC_PATTERN << "]\n" TERM_RESET
                    << "where\n"
                       "\t"
                    << TBOLD("auto")
                    << " - spread packets across frame time\n"
                       "\t"
                    << TBOLD("dynamic")
                    << " - similar to \"auto\" but more relaxed - occasional "
                       "huge frame can spread 1.5x frame time (default)\n"
                       "\t"
                    << TBOLD("unlimited")
                    << " - send packets at a wire speed (in bursts)\n"
                       "\t"
                    << SBOLD(NUMERIC_PATTERN)
                    << " - send packets at most at specified bitrate\n\n"
                    << TBOLD("Notes: ")
                    << "Use an exclamation mark to indicate intentionally very "
                       "low bitrate. 'E' to use the value as a fixed bitrate, "
                       "not cap /i. e. even the frames that may be sent at "
                       "lower bitrate are sent at the nominal bitrate)\n"
                    << "\n";
                return true;
        }
        bool force = false;
        bool fixed = false;
        for (int i = 0; i < 2; ++i) {
                if (optarg[strlen(optarg) - 1] == '!' ||
                    optarg[strlen(optarg) - 1] == 'E') {
                        if (optarg[strlen(optarg) - 1] == '!') {
                                force                      = true;
                                optarg[strlen(optarg) - 1] = '\0';
                        }
                        if (optarg[strlen(optarg) - 1] == 'E') {
                                fixed                      = true;
                                optarg[strlen(optarg) - 1] = '\0';
                        }
                }
        }
        *bitrate = unit_evaluate(optarg, nullptr);
        if (*bitrate <= 0) {
                log_msg(LOG_LEVEL_ERROR, "Invalid bitrate %s!\n", optarg);
                return false;
        }
        constexpr long long mb5 =
            5ll * 1000 *
            1000; // it'll take 6.4 sec to send 4 MB frame at 5 Mbps
        constexpr long long gb100 =
            100ll * 1000 * 1000 *
            1000; // traffic shaping to eg. 40 Gbps may make sense
        if ((*bitrate < mb5 || *bitrate > gb100) && !force) {
                log_msg(LOG_LEVEL_WARNING,
                        "Bitrate %lld bps seems to be too %s, use \"-l %s!\" "
                        "to force if this is not a mistake.\n",
                        *bitrate, *bitrate < mb5 ? "low" : "high", optarg);
                return false;
        }
        if (fixed) {
                *bitrate |= RATE_FLAG_FIXED_RATE;
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
                if (is_codec_opaque(c)) {
                        continue;
                }

                char tag = codec_is_a_rgb(c) ? 'R' : 'Y';
                col() << " " << left << setw(12) << SBOLD(get_codec_name(c)) << setw(0) << " " << tag << " " << setw(2) << get_bits_per_component(c) << setw(0) << "   " << get_codec_name_long(c) << "\n";
        }
}

void print_video_codecs(void) {
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = static_cast<codec_t>(static_cast<int>(c) + 1)) {
                if (!is_codec_opaque(c)) {
                        continue;
                }

                char tag = is_codec_interframe(c) ? 'I' : '.';
                col() << " " << left << setw(12) << SBOLD(get_codec_name(c)) << setw(0) << " " << tag << " " << "   " << get_codec_name_long(c) << "\n";
        }
        cout << "\nLegend:\n" << " I - interframe codec\n";
}

/**
 * Registers mainloop that will be run if display-owned mainloop isn't run.
 * Currently this is only used by Syphon, that either connects to display event
 * loop and if there isn't any, it runs its own.
 */
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
        auto              *s = (state_root *) get_root_module(mod)->priv_data;
        unique_lock<mutex> lk(s->lock);
        s->should_exit_callbacks.emplace_back(callback, udata);
}

/**
 * Unregisters should_exit callback registered with @ref register_should_exit_callback
 *
 * The callback must have been registered (and not unregistered),
 * otherwise abort is (currently) called.
 */
void
unregister_should_exit_callback(struct module *mod, void (*callback)(void *),
                                void          *udata)
{
        auto              *s = (state_root *) get_root_module(mod)->priv_data;
        unique_lock<mutex> lk(s->lock);
        for (auto it = s->should_exit_callbacks.begin();
             it != s->should_exit_callbacks.end(); ++it) {
                if (get<0>(*it) == callback && get<1>(*it) == udata) {
                        s->should_exit_callbacks.erase(it);
                        return;
                }
        }
        abort();
}

ADD_TO_PARAM("errors-fatal", "* errors-fatal\n"
                "  Treats some errors as fatal and exit even though " PACKAGE_NAME " could continue otherwise.\n"
                "  This allows less severe errors to be caught (which should not occur under normal circumstances).\n"
                "  An environment variable ULTRAGRID_ERRORS_FATAL with the same effect can also be used.\n");
/**
 * Soft version of exit_uv() checks errors-fatal command-line parameters and
 * if set, exit UltraGrid. Otherwise error is ignored.
 *
 * Caller code normally continues after this function so the error must not
 * have been fatal and UltraGrid must remain in a consistent state.
 */
void handle_error(int status) {
        if (get_commandline_param("errors-fatal") || getenv("ULTRAGRID_ERRORS_FATAL")) {
                exit_uv(status);
        }
}

bool running_in_debugger(){
#ifdef __linux__
        ifstream ppid_f("/proc/" + to_string(getppid()) + "/comm", ifstream::in);
        if (ppid_f.is_open()) {
                string comm;
                ppid_f >> comm;
                if (comm == "gdb") {
                        return true;
                }
        }
#endif
        return false;
}

#if defined(__APPLE__) || defined(__GLIBC__)
/// dumps output of fd (from start_off offset) to stderr
/// and keep the pointer at the end of the file
/// @retval size of the file pointed by fd (current pos)
static off_t
st_glibc_flush_output(int fd, off_t start_off)
{
        if (fd == STDERR_FILENO) {
                return 0;
        }

        lseek(fd, start_off, SEEK_SET);
        char    buf[STR_LEN];
        ssize_t rbytes = 0;
        while ((rbytes = read(fd, buf, sizeof buf)) > 0) {
                ssize_t written = 0;
                ssize_t wbytes  = 0;
                while (written < rbytes &&
                       (wbytes = write(STDERR_FILENO, buf + written,
                                       rbytes - written)) > 0) {
                        written += wbytes;
                }
        }
        return lseek(fd, 0, SEEK_CUR);
}
/**
 * print stacktrace with backtrace_symbols_fd() (glibc or macOS)
 *
 * ideally all functions should be async-signal-safe as defined by POSIX
 * (glibc deviates sligntly, see also signal-safety(7))
 */
static void
print_stacktrace_glibc()
{
        // print to a temporary file to avoid interleaving from multiple
        // threads
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 27)
        int fd = memfd_create("ultragrid_backtrace", MFD_CLOEXEC);
#else
        char path[MAX_PATH_SIZE];
#ifdef __APPLE__
        unsigned long tid = pthread_mach_thread_np(pthread_self());
#else
        unsigned long tid = syscall(__NR_gettid);
#endif
        // snprintf(path, sizeof path, "%s/ug-%lu", get_temp_dir(), tid);
        char *start = path;
        path[sizeof path - 1] = '\0';
        const char *const end = path + sizeof path - 1;
        strappend(&start, end, get_temp_dir());
        strappend(&start, end, "/ug-bt-");
        append_number(&start, end, tid);
        *start = '\0';
        int fd = open(path, O_CLOEXEC | O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        unlink(path);
#endif
        if (fd == -1) {
                fd = STDERR_FILENO;
        }
        char backtrace_msg[] = "Backtrace:\n";
        write_all(fd, sizeof backtrace_msg - 1, backtrace_msg);
        array<void *, 256> addresses{};
        const int num_symbols = backtrace(addresses.data(), addresses.size());
        backtrace_symbols_fd(addresses.data(), num_symbols, fd);

        // in case that the below fails, try write at least something
        off_t last_pos = st_glibc_flush_output(fd, 0);

#ifdef HAVE_LIBBACKTRACE
        char backtrace2_msg[] = "\nBacktrace symbolic:\n";
        write_all(fd, sizeof backtrace2_msg - 1, backtrace2_msg);
        for (int i = 0; i < num_symbols; i++) {
                // printf("%2d: ", i);
                enum { NDIGITS = 2 };
                char sym_nr[] = { 'X', 'X', ':', ' ' };
                int num_tmp = i;
                for (int i = 0; i < NDIGITS; ++i) {
                        if (num_tmp == 0 && i != 0) {
                                sym_nr[NDIGITS - 1 - i] = ' ';
                        } else {
                                sym_nr[NDIGITS - 1 - i] = '0' + (num_tmp % 10);
                                num_tmp /= 10;
                        }
                }
                write_all(fd, sizeof sym_nr, sym_nr);
                // backtrace_pcinfo may not be async-signal-safe
                backtrace_pcinfo(bt, (uintptr_t) addresses[i], libbt_full_callback,
                                 libbt_error_callback, &fd);
        }
        st_glibc_flush_output(fd, last_pos);
#endif

        if (fd != STDERR_FILENO) {
                close(fd);
        }
}
#endif // defined(__GLIBC__)

static void
print_backtrace()
{
#ifdef _WIN32
        print_stacktrace_win();
#elif defined(__APPLE__) || defined(__GLIBC__)
        print_stacktrace_glibc();
#else
        const char *msg = "Stacktrace printout not supported!\n";
        write(STDERR_FILENO, msg, strlen(msg));
#endif
}

void crash_signal_handler(int sig)
{
        print_backtrace();

        char buf[1024];
        char *ptr = buf;
        char *ptr_end = buf + sizeof buf;
        strappend(&ptr, ptr_end, "\n" PACKAGE_NAME " has crashed");

        append_sig_desc(&ptr, ptr_end, sig);

        strappend(&ptr, ptr_end, ".\n\nPlease send a bug report to address " PACKAGE_BUGREPORT ".\n");
        strappend(&ptr, ptr_end, "You may find some tips how to report bugs in file doc/REPORTING_BUGS.md distributed with " PACKAGE_NAME "\n");
        strappend(&ptr, ptr_end, "(or available online at https://github.com/CESNET/UltraGrid/blob/master/doc/REPORTING-BUGS.md).\n");

        write_all(STDERR_FILENO, ptr - buf, buf);

        restore_old_tio();

        signal(sig, SIG_DFL);
        raise(sig);
}

void hang_signal_handler(int sig)
{
        UNUSED(sig);
#ifndef _WIN32
        assert(sig == SIGALRM);
        char msg[] = "Hang detected - you may continue waiting or kill UltraGrid. Please report if UltraGrid doesn't exit after reasonable amount of time.\n";
        write_all(STDERR_FILENO, sizeof msg - 1, msg);
        signal(SIGALRM, SIG_DFL);
#endif // ! defined _WIN32
}

// some common parameters used within multiple modules
ADD_TO_PARAM("allow-unknown-params", "* allow-unknown-params\n"
                "  Do not exit on unknown parameter.\n");
ADD_TO_PARAM("audio-buffer-len", "* audio-buffer-len=<ms>\n"
                "  Sets length of software audio playback buffer (in ms, ALSA/Coreaudio/Portaudio/WASAPI)\n");
ADD_TO_PARAM("audio-cap-frames", "* audio-cap-frames=<f>\n"
                "  Sets number of audio frames captured at once (CoreAudio)\n");
ADD_TO_PARAM("audio-disable-adaptive-buffer", "* audio-disable-adaptive-buffer\n"
                "  Disables audio adaptive playback buffer (CoreAudio/JACK)\n");
ADD_TO_PARAM("color", "* color=CT\n"
                "  [experimental] Color space to use, C - colorimetry: 0 - undefined, 1 - BT.709, 2 - BT.2020/2100, 3 - P3; T - transfer fn: 0 - undefined, 1 - 709, 2 - HLG; 3 - PQ (signalized to GLFW on mac, NDI receiver)\n");
ADD_TO_PARAM("low-latency-audio", "* low-latency-audio[=ultra]\n"
                "  Try to reduce audio latency at the expense of worse reliability\n"
                "  Add ultra for even more aggressive setting.\n");
ADD_TO_PARAM("window-title", "* window-title=<title>\n"
                "  Use alternative window title (SDL/GL only)\n");


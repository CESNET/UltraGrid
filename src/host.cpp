/*
 * This file contains common external definitions
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "host.h"

#include "audio/audio_capture.h"
#include "audio/audio_playback.h"
#include "debug.h"
#include "lib_common.h"
#include "messaging.h"
#include "perf.h"
#include "rang.hpp"
#include "video_capture.h"
#include "video_compress.h"
#include "video_display.h"
#include "capture_filter.h"
#include "video.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#ifdef HAVE_X
#include <dlfcn.h>
#include <X11/Xlib.h>
/// @todo
/// The actual SONAME should be actually figured in configure.
#define X11_LIB_NAME "libX11.so.6"
#endif

#ifdef USE_MTRACE
#include <mcheck.h>
#endif

using namespace std;

unsigned int audio_capture_channels = DEFAULT_AUDIO_CAPTURE_CHANNELS;
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

volatile int log_level = LOG_LEVEL_INFO;

volatile int audio_offset;
volatile int video_offset;

std::unordered_map<std::string, std::string> commandline_params;

mainloop_t mainloop;
void *mainloop_udata;

static void common_cleanup()
{
#ifdef USE_MTRACE
        muntrace();
#endif

#ifdef WIN32
        WSACleanup();
#endif
}

ADD_TO_PARAM(stdout_buf, "stdout-buf",
         "* stdout-buf={no|line|full}\n"
         "  Buffering for stdout\n");
ADD_TO_PARAM(stderr_buf, "stderr-buf",
         "* stderr-buf={no|line|full}\n"
         "  Buffering for stderr\n");
bool set_output_buffering() {
        const unordered_map<const char *, FILE *> outs = {
                { "stdout-buf", stdout },
                { "stderr-buf", stderr }
        };
        for (auto outp : outs) {
                if (get_commandline_param(outp.first)) {
                        const unordered_map<string, int> buf_map {
                                { "no", _IONBF }, { "line", _IOLBF }, { "full", _IOFBF }
                        };

                        auto it = buf_map.find(get_commandline_param(outp.first));
                        if (it == buf_map.end()) {
                                log_msg(LOG_LEVEL_ERROR, "Wrong buffer type: %s\n", get_commandline_param(outp.first));
                                return false;
                        } else {
                                setvbuf(outp.second, NULL, it->second, 0);
                        }
                }
        }
        return true;
}

#ifdef HAVE_X
/**
 * Custom X11 error handler to catch errors and handle them more reasonably
 * than the default handler which exits the program immediately, which, however
 * is not correct in multithreaded program.
 */
static int x11_error_handler(Display *d, XErrorEvent *e) {
        //char msg[1024] = "";
        //XGetErrorText(d, e->error_code, msg, sizeof msg - 1);
        UNUSED(d);
        log_msg(LOG_LEVEL_ERROR, "X11 error - code: %d, serial: %d, error: %d, request: %d, minor: %d\n",
                        e->error_code, e->serial, e->error_code, e->request_code, e->minor_code);

        return 0;
}
#endif

bool common_preinit(int argc, char *argv[])
{
        uv_argc = argc;
        uv_argv = argv;

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
                return false;
        }
        if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
                fprintf(stderr, "Counld not found usable version of Winsock.\n");
                WSACleanup();
                return false;
        }
#endif

        open_all("ultragrid_*.so"); // load modules

#ifdef USE_MTRACE
        mtrace();
#endif

        perf_init();
        perf_record(UVP_INIT, 0);

        atexit(common_cleanup);

        return true;
}

#include <sstream>

void print_capabilities(struct module *root, bool use_vidcap)
{
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
                int count;
                struct device_info *devices;
                vdi->probe(&devices, &count);
                cout << "[cap][display] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[cap] (" << devices[i].id << ";" << devices[i].name << ";" <<
                                devices[i].repeatable << ")\n";
                }
                free(devices);
        }

        cout << "[cap] Audio capturers:" << endl;
        auto const & audio_cap_capabilities =
                get_libraries_for_class(LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
        for (auto const & it : audio_cap_capabilities) {
                auto aci = static_cast<const struct audio_capture_info *>(it.second);
                int count;
                struct device_info *devices;
                aci->probe(&devices, &count);
                cout << "[cap][audio_cap] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[cap] (" << devices[i].id << ";" << devices[i].name << ")\n";
                }
                free(devices);
        }

        cout << "[cap] Audio playback:" << endl;
        auto const & audio_play_capabilities =
                get_libraries_for_class(LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);
        for (auto const & it : audio_play_capabilities) {
                auto api = static_cast<const struct audio_playback_info *>(it.second);
                int count;
                struct device_info *devices;
                api->probe(&devices, &count);
                cout << "[cap][audio_play] " << it.first << std::endl;
                for (int i = 0; i < count; ++i) {
                        cout << "[cap] (" << devices[i].id << ";" << devices[i].name << ")\n";
                }
                free(devices);
        }
}

void print_version()
{
        cout << rang::fg::cyan << rang::style::bold << PACKAGE_STRING <<
                rang::fg::reset << rang::style::reset << " (" <<
#ifdef GIT_BRANCH
                GIT_BRANCH << " "
#endif
#ifdef GIT_REV
                "rev " GIT_REV " " <<
#endif
                "built " __DATE__ " " __TIME__ ")\n";
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

void register_param(const char *param, const char *doc)
{
        assert(param != NULL && doc != NULL);
        for (unsigned int i = 0; i < sizeof params / sizeof params[0]; ++i) {
                if (params[i].param == NULL) {
                        params[i].param = param;
                        params[i].doc = doc;
                        break;
                }
        }
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

bool register_mainloop(mainloop_t m, void *u)
{
        if (mainloop) {
                return false;
        }

        mainloop = m;
        mainloop_udata = u;

        return true;
}

// some common parameters used within multiple modules
ADD_TO_PARAM(audio_buffer_len, "audio-buffer-len", "* audio-buffer-len=<ms>\n"
                "  Sets length of software audio playback buffer (in ms, ALSA/Portaudio)\n");
ADD_TO_PARAM(low_latency_audio, "low-latency-audio", "* low-latency-audio\n"
                "  Try to reduce audio latency at the expense of worse reliability\n");
ADD_TO_PARAM(window_title, "window-title", "* window-title=<title>\n"
                "  Use alternative window title (SDL/GL only)\n");


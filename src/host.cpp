/*
 * This file contains common external definitions
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "host.h"

#include "messaging.h"
#include "video_capture.h"
#include "video_compress.h"
#include "video.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

unsigned int cuda_device = 0;
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
volatile bool should_exit_receiver = false;

volatile int log_level = LOG_LEVEL_INFO;
bool color_term = (getenv("TERM") && (strcmp(getenv("TERM"), "xterm") == 0 || strcmp(getenv("TERM"), "xterm-256color") == 0 || strcmp(getenv("TERM"), "screen") == 0)) && isatty(1) && isatty(2);

bool ldgm_device_gpu = false;

const char *window_title = NULL;
#include <sstream>

void print_capabilities(int mask, struct module *root, bool use_vidcap)
{
        if (mask & CAPABILITY_COMPRESS) {
                // try to figure out actual video format
                struct video_desc desc{};
                if (use_vidcap && root) {
                        for (int attempt = 0; attempt < 2; ++attempt) {
                                struct msg_sender *m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                                m->type = SENDER_MSG_QUERY_VIDEO_MODE;
                                struct response *r = send_message_sync(root, "sender", (struct message *) m, 1000);
                                if (response_get_status(r) == RESPONSE_OK) {
                                        const char *text = response_get_text(r);
                                        istringstream iss(text);
                                        iss >> desc;
                                        free_response(r);
                                        break;
                                }
                                free_response(r);
                                sleep(1);
                        }
                }

                cout << "[cap] Compressions:" << endl;
                auto const & compress_capabilities = get_compress_capabilities();
                for (auto const & it : compress_capabilities) {
                        cout << "[cap] (" << it.name << ";" << it.quality << ";" << setiosflags(ios_base::fixed) << setprecision(2) << it.compute_bitrate(&desc) << ";" <<
                                it.enc_prop.latency << ";" << it.enc_prop.cpu_cores << ";" << it.enc_prop.gpu_gflops << ";" <<
                                it.dec_prop.latency << ";" << it.dec_prop.cpu_cores << ";" << it.dec_prop.gpu_gflops <<
                                ")\n";
                }
        }
        if (mask & CAPABILITY_CAPTURE) {
                cout << "[cap] Capturers:" << endl;
                print_available_capturers();
        }
}


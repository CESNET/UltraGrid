#include <cassert>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <memory>

#include "config_unix.h"
#include "config_win32.h"

#include "libug.h"

#include "capture_filter.h"
#include "debug.h"
#include "host.h"
#include "src/host.h"
#include "utils/wait_obj.h"
#include "video.h"
#include "video_display.h"
#include "video_display/pipe.hpp"
#include "video_rxtx.h"

using namespace std;

static char app_name[] = "(undefined)";
static char *argv[] = { app_name , nullptr };
static int argc = 1;

extern "C" void exit_uv(int status);

void exit_uv(int status) {
        should_exit = true;
        LOG(LOG_LEVEL_WARNING) << "Requested exit with code " << status << ".\n";
}

static_assert(static_cast<int>(UG_RGBA) == static_cast<int>(RGBA));
static_assert(static_cast<int>(UG_I420) == static_cast<int>(I420));
static_assert(static_cast<int>(UG_CUDA_RGBA) == static_cast<int>(CUDA_RGBA));
static_assert(static_cast<int>(UG_CUDA_I420) == static_cast<int>(CUDA_I420));

////////////////////////////////////
//             SENDER
////////////////////////////////////
struct ug_sender {
        struct video_rxtx *video_rxtx{};
        struct wait_obj *wait_obj = wait_obj_init();
        struct module root_module;
        struct init_data *common;
        struct capture_filter *stripe = nullptr;

        ug_sender() {
                common = common_preinit(argc, argv, nullptr);
                module_init_default(&root_module);
                root_module.cls = MODULE_CLASS_ROOT;
                //root_module.new_message = state_uv::new_message;
                root_module.priv_data = this;
        }

        ~ug_sender() {
                wait_obj_done(wait_obj);
                if (video_rxtx != nullptr) {
                        video_rxtx->join();
                        delete video_rxtx;
                }
                common_cleanup(common);
        }
};

struct ug_sender *ug_sender_init(const struct ug_sender_parameters *init_params)
{
        assert(init_params != nullptr);

        if (init_params->receiver == nullptr) {
                LOG(LOG_LEVEL_ERROR) << "Receiver must be set!\n";
                return nullptr;
        }

        log_level += init_params->verbose;

        struct ug_sender *s = new ug_sender();

        chrono::steady_clock::time_point start_time(chrono::steady_clock::now());
        map<string, param_u> params;

        // common
        params["parent"].ptr = &s->root_module;
        params["exporter"].ptr = NULL;
        params["compression"].str = init_params->compression == UG_UNCOMPRESSED ? "none" : "libavcodec:codec=mjpeg";
        params["rxtx_mode"].i = MODE_SENDER;
        params["paused"].b = false;

        //RTP
        params["mtu"].i = init_params->mtu != 0 ? init_params->mtu : 1500;
        params["receiver"].str = init_params->receiver;
        params["rx_port"].i = 0;
        params["tx_port"].i = init_params->port != 0 ? init_params->port : DEFAULT_UG_PORT;
        params["force_ip_version"].i = 0;
        params["mcast_if"].str = NULL;
        params["fec"].str = "none";
        params["encryption"].str = NULL;
        params["bitrate"].ll = RATE_UNLIMITED;
        params["start_time"].cptr = (const void *) &start_time;
        params["video_delay"].vptr = 0;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = NULL;

        params["shm_state"].ptr = nullptr;

        try {
                s->video_rxtx = video_rxtx::create("ultragrid_rtp", params);
                if (!s->video_rxtx) {
                        delete s;
                        return nullptr;
                }
        } catch (exception const &err) {
                cerr << err.what() << endl;
                delete s;
                return nullptr;
        }

        if (init_params->disable_strips == 0) {
                if (capture_filter_init(nullptr, "stripe", &s->stripe) != 0) {
                        abort();
                }
        }

        render_packet_received_callback = init_params->rprc;
        render_packet_received_callback_udata = init_params->rprc_udata;

        return s;
}

void ug_send_frame(struct ug_sender *s, const char *data, libug_pixfmt_t codec, int width, int height, uint32_t seq)
{
        struct video_frame *f = vf_alloc(1);
        f->color_spec = (codec_t) codec;
        f->tiles[0].width = width;
        f->tiles[0].height = height;
        f->fps = 120;
        f->interlacing = PROGRESSIVE;
        struct RenderPacket render_packet{};
        render_packet.frame = seq;
        f->render_packet = render_packet;

        f->tiles[0].data = const_cast<char *>(data);
        f->tiles[0].data_len = vc_get_datalen(width, height, f->color_spec);

        if (s->stripe) {
                struct video_frame *strips = capture_filter(s->stripe, f);
                auto frame = shared_ptr<video_frame>(strips, strips->callbacks.dispose);
                s->video_rxtx->send(move(frame));
        } else {
                wait_obj_reset(s->wait_obj);
                auto frame = shared_ptr<video_frame>(f, [&](struct video_frame *) {
                                wait_obj_notify(s->wait_obj);
                                });
                s->video_rxtx->send(move(frame));

                wait_obj_wait(s->wait_obj);
        }

        vf_free(f);
}

void ug_sender_done(struct ug_sender *s)
{
        render_packet_received_callback = nullptr;
        render_packet_received_callback_udata = nullptr;
        if (s->stripe) {
                capture_filter_destroy(s->stripe);
        }
        delete s;
}

////////////////////////////////////
//             RECEIVER
////////////////////////////////////
struct ug_receiver {
        struct video_rxtx *video_rxtx{};
        struct display *display{};
        struct module root_module;
        struct init_data *common;

        ug_receiver() {
                common = common_preinit(argc, argv, nullptr);
                module_init_default(&root_module);
                root_module.cls = MODULE_CLASS_ROOT;
                //root_module.new_message = state_uv::new_message;
                root_module.priv_data = this;
        }

        virtual ~ug_receiver() {
                if (video_rxtx != nullptr) {
                        video_rxtx->join();
                        delete video_rxtx;
                }
                if (display != nullptr) {
                        display_done(display);
                }
                common_cleanup(common);
        }

        pthread_t display_thread;
        pthread_t receiver_thread;
};

struct ug_receiver *ug_receiver_start(struct ug_receiver_parameters *init_params)
{
        struct ug_receiver *s = new ug_receiver();

        chrono::steady_clock::time_point start_time(chrono::steady_clock::now());
        map<string, param_u> params;

        log_level += init_params->verbose;

        // common
        params["parent"].ptr = &s->root_module;
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_RECEIVER;
        params["paused"].b = false;

        //RTP
        params["mtu"].i = 9000; // doesn't matter
        params["receiver"].str = init_params->sender ? init_params->sender : "localhost";
        params["rx_port"].i = init_params->port ? init_params->port : DEFAULT_UG_PORT;
        params["tx_port"].i = 0;
        params["force_ip_version"].i = 0;
        params["mcast_if"].str = NULL;
        params["fec"].str = "none";
        params["encryption"].str = NULL;
        params["bitrate"].ll = RATE_UNLIMITED;
        params["start_time"].cptr = (const void *) &start_time;
        params["video_delay"].vptr = 0;

        char display[128] = "vrg";
        const char *display_cfg = "";

        if (init_params->display != nullptr) {
                strncpy(display, init_params->display, sizeof display);
                if (strchr(display, ':') != nullptr) {
                        display_cfg = strchr(display, ':') + 1;
                        *strchr(display, ':') = '\0';
                }
        }

        if (init_params->disable_strips == 0) {
                commandline_params["unstripe"] = string();
        }

        if (init_params->decompress_to != 0) {
                commandline_params["decoder-use-codec"] = get_codec_name((codec_t) init_params->decompress_to);
        }

        if (init_params->force_gpu_decoding) {
                commandline_params["decompress"] = "gpujpeg";
        }

        if (initialize_video_display(&s->root_module, display, display_cfg, 0, nullptr, &s->display) != 0) {
                LOG(LOG_LEVEL_ERROR) << "Unable to initialize VRG display!\n";
                delete s;
                return nullptr;
        }

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = s->display;

        params["shm_state"].ptr = nullptr;

        try {
                s->video_rxtx = video_rxtx::create("ultragrid_rtp", params);
                if (s->video_rxtx == nullptr) {
                        delete s;
                        return nullptr;
                }
        } catch (exception const &err) {
                LOG(LOG_LEVEL_ERROR) << err.what() << endl;
                delete s;
                return nullptr;
        }

        pthread_create(&s->receiver_thread, NULL, video_rxtx::receiver_thread,
                                         (void *) s->video_rxtx);
        pthread_create(&s->display_thread, nullptr, (void* (*)(void*))(void *) display_run, s->display);


        return s;
}

void ug_receiver_done(struct ug_receiver *s)
{
        exit_uv(0);
        pthread_join(s->receiver_thread, NULL);
        pthread_join(s->display_thread, NULL);

        delete s;
}


#include <cassert>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <memory>

#include "config_unix.h"
#include "config_win32.h"

#include "libug.h"

#include "debug.h"
#include "host.h"
#include "src/host.h"
#include "utils/wait_obj.h"
#include "video.h"
#include "video_rxtx.h"

using namespace std;

extern "C" void exit_uv(int status);

void exit_uv(int status) {
        LOG(LOG_LEVEL_WARNING) << "Requested exit with code " << status << " but continuing anyway "
                "(we are not UltraGrid).\n";
}

struct ug_sender {
        struct video_rxtx *video_rxtx{};
        struct wait_obj *wait_obj = wait_obj_init();
        struct module root_module;

        ug_sender() {
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
        }
};

struct ug_sender *ug_sender_init(const struct ug_sender_parameters *init_params)
{
        assert(init_params != nullptr);

        if (init_params->receiver == nullptr) {
                LOG(LOG_LEVEL_ERROR) << "Receiver must be set!\n";
                return nullptr;
        }

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
        params["rx_port"].i = init_params->rx_port != 0 ? init_params->rx_port : DEFAULT_UG_PORT;
        params["tx_port"].i = init_params->tx_port != 0 ? init_params->tx_port : DEFAULT_UG_PORT;
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

        render_packet_received_callback = init_params->rprc;
        render_packet_received_callback_udata = init_params->rprc_udata;

        return s;
}

void ug_send_frame(struct ug_sender *s, const char *data, libug_pixfmt_t codec, int width, int height)
{
        struct video_frame *f = vf_alloc(1);
        f->color_spec = (codec_t) codec;
        f->tiles[0].width = width;
        f->tiles[0].height = height;
        f->fps = 120;
        f->interlacing = PROGRESSIVE;

        f->tiles[0].data = const_cast<char *>(data);
        f->tiles[0].data_len = vc_get_datalen(width, height, f->color_spec);

        wait_obj_reset(s->wait_obj);
        auto frame = shared_ptr<video_frame>(f, [&](struct video_frame *) {
                        wait_obj_notify(s->wait_obj);
                        });
        s->video_rxtx->send(move(frame));

        wait_obj_wait(s->wait_obj);

        vf_free(f);
}

void ug_sender_done(struct ug_sender *s)
{
        render_packet_received_callback = nullptr;
        render_packet_received_callback_udata = nullptr;
        delete s;
}


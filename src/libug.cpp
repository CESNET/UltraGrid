#include <chrono>
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

#define PORT_BASE 5004

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
                video_rxtx->join();
                delete video_rxtx;
        }
};

struct ug_sender *ug_sender_init(const char *receiver, int mtu)
{
        struct ug_sender *s = new ug_sender();

        chrono::steady_clock::time_point start_time(chrono::steady_clock::now());
        map<string, param_u> params;

        // common
        params["parent"].ptr = &s->root_module;
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_SENDER;
        params["paused"].b = false;

        //RTP
        params["mtu"].i = mtu;
        params["receiver"].str = receiver;
        params["rx_port"].i = 0;
        params["tx_port"].i = PORT_BASE;
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

        try {
                s->video_rxtx = video_rxtx::create("ultragrid_rtp", params);
                if (!s->video_rxtx) {
                        delete s;
                        return nullptr;
                }
        } catch (string const &err) {
                cerr << err << endl;
                delete s;
                return nullptr;
        }

        return s;
}

void ug_send_frame(struct ug_sender *s, const char *data, size_t len, ug_codec_t codec, int width, int height)
{
        struct video_frame *f = vf_alloc(1);
        f->color_spec = (codec_t) codec;
        f->tiles[0].width = width;
        f->tiles[0].height = height;
        f->fps = 120;
        f->interlacing = PROGRESSIVE;

        f->tiles[0].data = const_cast<char *>(data);
        f->tiles[0].data_len = len;

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
        delete s;
}


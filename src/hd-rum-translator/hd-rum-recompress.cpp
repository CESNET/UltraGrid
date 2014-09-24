#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hd-rum-translator/hd-rum-recompress.h"

#include "debug.h"
#include "host.h"
#include "rtp/rtp.h"

#include "video_rxtx/ultragrid_rtp.h"

#include <chrono>
#include <memory>

using namespace std;

struct state_recompress {
        unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;
        string host;

        chrono::steady_clock::time_point t0;
        int frames;
        int tx_port;
};

void *recompress_init(struct module *parent,
                const char *host, const char *compress, unsigned short rx_port,
                unsigned short tx_port, int mtu, char *fec, long packet_rate)
{
        bool use_ipv6 = false;

        map<string, param_u> params;

        // common
        params["parent"].ptr = parent;
        params["exporter"].ptr = NULL;
        params["compression"].ptr = (void *) compress;
        params["rxtx_mode"].i = MODE_SENDER;

        //RTP
        params["mtu"].i = mtu;
        params["receiver"].ptr = (void *) host;
        params["rx_port"].i = rx_port;
        params["tx_port"].i = tx_port;
        params["use_ipv6"].b = use_ipv6;
        params["mcast_if"].ptr = (void *) NULL;
        params["fec"].ptr = (void *) fec;
        params["encryption"].ptr = (void *) NULL;
        params["packet_rate"].i = packet_rate;

        // UltraGrid RTP
        params["postprocess"].ptr = (void *) NULL;
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = NULL;

        return new state_recompress{
                decltype(state_recompress::video_rxtx)(dynamic_cast<ultragrid_rtp_video_rxtx *>(video_rxtx::create(ULTRAGRID_RTP, params))),
                host,
                chrono::steady_clock::now(),
                0,
                tx_port
        };
}

void recompress_process_async(void *state, shared_ptr<video_frame> frame)
{
        auto s = static_cast<state_recompress *>(state);

        s->frames += 1;

        chrono::steady_clock::time_point now = chrono::steady_clock::now();
        double seconds = chrono::duration_cast<chrono::microseconds>(now - s->t0).count() / 1000000.0;
        if(seconds > 5) {
                double fps = s->frames / seconds;
                fprintf(stdout, "[%.4x->%s:%d] %d frames in %g seconds = %g FPS\n",
                                rtp_my_ssrc(s->video_rxtx->m_network_devices[0]),
                                s->host.c_str(), s->tx_port,
                                s->frames, seconds, fps);
                s->t0 = now;
                s->frames = 0;
        }

        s->video_rxtx->send(frame);
}

void recompress_assign_ssrc(void *state, uint32_t ssrc)
{
        // UNIMPLEMENTED NOW
        UNUSED(state);
        UNUSED(ssrc);
}

void recompress_done(void *state)
{
        auto s = static_cast<state_recompress *>(state);

        s->video_rxtx->join();

        delete s;
}


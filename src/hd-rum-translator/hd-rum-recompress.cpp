/**
 * @file   hd-rum-translator/hd-rum-recompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 *
 * Component of the transcoding reflector that takes an uncompressed frame,
 * recompresses it to another compression and sends it to destination
 * (therefore it wraps the whole sending part of UltraGrid).
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
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cinttypes>
#include <chrono>
#include <memory>
#include <thread>
#include <string>


#include "hd-rum-translator/hd-rum-recompress.h"

#include "debug.h"
#include "host.h"
#include "utils/misc.h"
#include "rtp/rtp.h"
#include "tv.h"
#include "video_compress.h"

#include "video_rxtx/ultragrid_rtp.hpp"
#include "utils/profile_timer.hpp"

namespace {
struct compress_state_deleter{
        void operator()(struct compress_state *s){ compress_done(s); }
};
}

using namespace std;

struct recompress_output_port {
        recompress_output_port() = default;
        recompress_output_port(
                std::string host, unsigned short rx_port,
                unsigned short tx_port, const struct common_opts *common,
                const char *fec, long long bitrate);

        std::unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;
        std::string host;
        int tx_port;

        std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
        int frames;

        bool active;
};

struct recompress_worker_ctx {
        std::string compress_cfg;
        std::unique_ptr<compress_state, compress_state_deleter> compress;

        std::mutex ports_mut;
        std::vector<recompress_output_port> ports;

        std::thread thread;
};

struct state_recompress {
        struct module *parent;
        std::mutex mut;
        std::map<std::string, recompress_worker_ctx> workers;
        std::vector<std::pair<std::string, int>> index_to_port;
};

recompress_output_port::recompress_output_port(
                std::string host, unsigned short rx_port,
                unsigned short tx_port, const struct common_opts *common,
                const char *fec, long long bitrate) :
        host(std::move(host)),
        tx_port(tx_port),
        frames(0),
        active(true)
{
        std::map<std::string, param_u> params;

        // common
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_SENDER;

        //RTP
        params["common"].cptr = common;
        params["receiver"].str = this->host.c_str();
        params["rx_port"].i = rx_port;
        params["tx_port"].i = tx_port;
        params["fec"].str = fec;
        params["bitrate"].ll = bitrate;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = NULL;

        auto rxtx = video_rxtx::create("ultragrid_rtp", params);
        if (this->host.find(':') != std::string::npos) {
                rxtx->m_port_id = "[" + this->host + "]:" + to_string(tx_port);
        } else {
                rxtx->m_port_id = this->host + ":" + to_string(tx_port);
        }

        video_rxtx.reset(dynamic_cast<ultragrid_rtp_video_rxtx *>(rxtx));
}

static void recompress_port_write(recompress_output_port& port, shared_ptr<video_frame> frame)
{
        PROFILE_FUNC;

        port.frames += 1;

        auto now = chrono::steady_clock::now();

        double seconds = chrono::duration_cast<chrono::duration<double>>(now - port.t0).count();
        if(seconds > 5) {
                double fps = port.frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[0x%08" PRIx32 "->%s:%d:0x%08" PRIx32 "] %d frames in %g seconds = %g FPS\n",
                                frame->ssrc,
                                port.host.c_str(), port.tx_port,
                                port.video_rxtx->get_ssrc(),
                                port.frames, seconds, fps);
                port.t0 = now;
                port.frames = 0;
        }

        port.video_rxtx->send(std::move(frame));
}

static void recompress_worker(struct recompress_worker_ctx *ctx){
        PROFILE_FUNC;
        assert(ctx->compress);

        while(auto frame = compress_pop(ctx->compress.get())){
                std::lock_guard<std::mutex> lock(ctx->ports_mut);
                for(auto& port : ctx->ports){
                        if(port.active)
                                recompress_port_write(port, frame);
                }
                PROFILE_DETAIL("compress_pop");
        }
}

static int move_port_to_worker(struct state_recompress *s, const char *compress,
                recompress_output_port&& port)
{
        auto& worker = s->workers[compress];
        if(!worker.compress){
                worker.compress_cfg = compress;
                int ret = compress_init(s->parent, compress, out_ptr(worker.compress));
                if (ret != 0) {
                        s->workers.erase(compress);
                        return -1;
                }

                worker.thread = std::thread(recompress_worker, &worker);
        }

        std::lock_guard<std::mutex> lock(worker.ports_mut);
        int index_in_worker = worker.ports.size();
        worker.ports.push_back(std::move(port));

        return index_in_worker;
}

int recompress_add_port(struct state_recompress *s,
		const char *host, const char *compress, unsigned short rx_port,
                unsigned short tx_port, const struct common_opts *common,
                const char *fec, long long bitrate)
{
        recompress_output_port port;

        try{
                port = recompress_output_port(host, rx_port, tx_port,
                                common, fec, bitrate);
        } catch(...) {
                return -1;
        }

        std::lock_guard<std::mutex> lock(s->mut);
        int index_in_worker = move_port_to_worker(s, compress, std::move(port));
        if(index_in_worker < 0)
                return -1;

        int index_of_port = s->index_to_port.size();
        s->index_to_port.emplace_back(compress, index_in_worker);

        return index_of_port;
}

static void extract_port(struct state_recompress *s,
                const std::string& compress_cfg, int i,
                recompress_output_port *move_to = nullptr)
{
        auto& worker = s->workers[compress_cfg];
        {
                std::unique_lock<std::mutex> lock(worker.ports_mut);
                if(move_to)
                        *move_to = std::move(worker.ports[i]);
                worker.ports.erase(worker.ports.begin() + i);

                if(worker.ports.empty()){
                        //poison compress
                        compress_frame(worker.compress.get(), nullptr);
                        worker.thread.join();
                        s->workers.erase(compress_cfg);
                }
        }

        for(auto& p : s->index_to_port){
                if(p.first == compress_cfg && p.second > i)
                        p.second--;
        }
}

void recompress_remove_port(struct state_recompress *s, int index){
        std::lock_guard<std::mutex> lock(s->mut);
        auto [compress_cfg, i] = s->index_to_port[index];

        extract_port(s, compress_cfg, i);
        s->index_to_port.erase(s->index_to_port.begin() + index);
}

uint32_t recompress_get_port_ssrc(struct state_recompress *s, int idx){
        std::lock_guard<std::mutex> lock(s->mut);
        auto [compress_cfg, i] = s->index_to_port[idx];

        std::lock_guard<std::mutex> work_lock(s->workers[compress_cfg].ports_mut);
        return s->workers[compress_cfg].ports[i].video_rxtx->get_ssrc();
}

void recompress_port_set_active(struct state_recompress *s,
                int index, bool active)
{
        std::lock_guard<std::mutex> lock(s->mut);
        auto [compress_cfg, i] = s->index_to_port[index];

        std::unique_lock<std::mutex> worker_lock(s->workers[compress_cfg].ports_mut);
        s->workers[compress_cfg].ports[i].active = active;
}

bool recompress_port_change_compress(struct state_recompress *s, int index,
                const char *new_compress)
{
        std::lock_guard<std::mutex> lock(s->mut);
        auto [old_compress, i] = s->index_to_port[index];

        if(old_compress == new_compress)
                return true;

        recompress_output_port port;
        extract_port(s, old_compress, i, &port);
        int index_in_worker = move_port_to_worker(s, new_compress, std::move(port));

        if(index_in_worker < 0){
                s->index_to_port.erase(s->index_to_port.begin() + index);
                return false;
        }

        s->index_to_port[index] = {new_compress, index_in_worker};

        return true;
}

static int worker_get_num_active_ports(const recompress_worker_ctx& worker){
        int ret = 0;
        for(const auto& port : worker.ports){
                if(port.active)
                        ret++;
        }
        return ret;
}

int recompress_get_num_active_ports(struct state_recompress *s){
        std::lock_guard<std::mutex> lock(s->mut);
        int ret = 0;
        for(const auto& worker : s->workers){
                ret += worker_get_num_active_ports(worker.second);
        }

        return ret;
}

struct state_recompress *recompress_init(struct module *parent) {
        auto state = new state_recompress();
        if(!state)
                return nullptr;

        state->parent = parent;

        return state;
}

void recompress_process_async(state_recompress *s, std::shared_ptr<video_frame> frame){
        PROFILE_FUNC;
        std::lock_guard<std::mutex> lock(s->mut);
        for(const auto& worker : s->workers){
                if(worker_get_num_active_ports(worker.second) > 0)
                        compress_frame(worker.second.compress.get(), frame);
        }
}

void recompress_done(struct state_recompress *s) {
        {
                std::lock_guard<std::mutex> lock(s->mut);
                for(auto& worker : s->workers){
                        //poison compress
                        compress_frame(worker.second.compress.get(), nullptr);

                        worker.second.thread.join();
                        // compress_done(worker.second.compress.get());
                }
        }
        delete s;
}


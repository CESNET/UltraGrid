/**
 * @file   video_compress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup video_compress
 *
 * @brief Video compress functions.
 */
/*
 * Copyright (c) 2011-2025 CESNET
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

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>                     // for move
#include <vector>

#include "messaging.h"
#include "module.h"
#include "tv.h"
#include "utils/misc.h"                // for format_number_with_delim
#include "utils/synchronized_queue.h"
#include "utils/thread.h"
#include "utils/vf_split.h"
#include "utils/worker.h"
#include "video.h"
#include "video_compress.h"
#include "lib_common.h"
#include "debug.h"

#define MOD_NAME "[vcompress] "

using namespace std;

struct compress_state;

namespace {
/**
 * @brief This structure represents real internal compress state
 */
struct compress_state_real {
private:
        compress_state_real(struct module *parent, const char *config_string);
        void          start(struct compress_state *proxy);
        void          async_consumer(struct compress_state *s);
        void          async_tile_consumer(struct compress_state *s);
        thread        asynch_consumer_thread;
public:
        static compress_state_real *create(struct module *parent, const char *config_string,
                        struct compress_state *proxy) {
                compress_state_real *s = new compress_state_real(parent, config_string);
                s->start(proxy);
                return s;
        }
        ~compress_state_real();
        const video_compress_info    *funcs;            ///< handle for the driver
        vector<void*> state;                  ///< driver internal states
        string              compress_options; ///< compress options (for reconfiguration)
        volatile bool       discard_frames;   ///< this class is no longer active
};
}

/**
 * @brief Video compress state.
 *
 * This structure represents external video compress state. This is basically a proxy for real
 * state. The point of doing this is to allow dynamic reconfiguration of the real state.
 */
struct compress_state {
        explicit compress_state(struct module *parent)
        {
                module_init_default(&mod);
                mod.cls       = MODULE_CLASS_COMPRESS;
                mod.priv_data = this;
                module_register(&mod, parent);
        }
        ~compress_state() { module_done(&mod); }

        struct module mod;               ///< compress module data
        struct compress_state_real *ptr{}; ///< pointer to real compress state
        synchronized_queue<shared_ptr<video_frame>, 1> queue;
        bool poisoned = false;
};

static shared_ptr<video_frame> compress_frame_tiles(struct compress_state *proxy,
                shared_ptr<video_frame> frame);

/// @brief Displays list of available compressions.
void show_compress_help(bool full)
{
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        list_modules(LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION, full);
}

static void async_poison(struct compress_state_real *s){
        if (s->funcs->compress_frame_async_push_func) {
                s->funcs->compress_frame_async_push_func(s->state[0], {}); // poison
        } else if (s->funcs->compress_tile_async_push_func){
                for(size_t i = 0; i < s->state.size(); i++){
                        s->funcs->compress_tile_async_push_func(s->state[i], {}); // poison
                }
        }
}

/**
 * @brief Processes message.
 *
 * This function is a callback called from control thread to change some parameters of
 * compression.
 *
 * @param[in] receiver pointer to the compress module
 * @param[in] msg      message to process
 */
static void compress_process_message(struct compress_state *proxy, struct msg_change_compress_data *data)
{
        struct response *r = NULL;
        /* In this case we are only changing some parameter of compression.
         * This means that we pass the parameter to compress driver. */
        if(data->what == CHANGE_PARAMS) {
                for(unsigned int i = 0; i < proxy->ptr->state.size(); ++i) {
                        struct msg_change_compress_data *tmp_data =
                                (struct msg_change_compress_data *)
                                new_message(sizeof(struct msg_change_compress_data));
                        tmp_data->what = data->what;
                        strncpy(tmp_data->config_string, data->config_string,
                                        sizeof(tmp_data->config_string));
                        char receiver[100];
                        snprintf(receiver, sizeof receiver, "data[%u]", i);
                        struct response *resp = send_message(&proxy->mod, receiver,
                                        (struct message *) tmp_data);
                        /// @todo
                        /// Handle responses more intelligently (eg. aggregate).
                        free_response(r); // frees previous response
                        r = resp;
                }

        } else {
                struct compress_state_real *new_state;
                char config[1024];
                strncpy(config, data->config_string, sizeof(config));

                try {
                        new_state = compress_state_real::create(&proxy->mod, config, proxy);
                } catch (int i) {
                        free_message((struct message *) data,
                                        new_response(RESPONSE_INT_SERV_ERR, NULL));
                        return;
                }

                struct compress_state_real *old = proxy->ptr;
                // let the async processing finish
                old->discard_frames = true;
                async_poison(old);
                delete old;
                proxy->ptr = new_state;
                r = new_response(RESPONSE_OK, NULL);
        }

        free_message((struct message *) data, r);
}

/**
 * @brief This function initializes video compression.
 *
 * This function wraps the call of compress_init_real().
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @param[out] state        created state
 * @retval     0            if state created successfully
 * @retval    <0            if error occurred
 * @retval    >0            finished successfully, no state created (eg. displayed help)
 */
int compress_init(struct module *parent, const char *config_string, struct compress_state **state) {
        struct compress_state *proxy;
        proxy = new struct compress_state(parent);

        try {
                proxy->ptr = compress_state_real::create(&proxy->mod, config_string, proxy);
        } catch (int i) {
                delete proxy;
                return i;
        }


        *state = proxy;
        return 0;
}

/**
 * @brief Constructor for compress_state_real
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @throws    -1            if error occurred
 * @retval     1            finished successfully, no state created (eg. displayed help)
 */
compress_state_real::compress_state_real(struct module *parent, const char *config_string) :
        funcs(nullptr), discard_frames(false)
{
        string compress_name;

        if (!config_string)
                throw -1;

        if (strcmp(config_string, "help") == 0 || strcmp(config_string, "fullhelp") == 0) {
                show_compress_help(strcmp(config_string, "fullhelp") == 0);
                throw 1;
        }

        char *tmp = strdup(config_string);
        if (strchr(tmp, ':')) {
                char *opts = strchr(tmp, ':') + 1;
                *strchr(tmp, ':') = '\0';
                compress_options = opts;
        }
        compress_name = tmp;
        free(tmp);

        auto vci = static_cast<const struct video_compress_info *>(load_library(compress_name.c_str(), LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION));
        if(!vci) {
                LOG(LOG_LEVEL_ERROR)
                    << MOD_NAME "Unknown or unavailable compression: "
                    << config_string << "\n";
                throw -1;
        }

        funcs = vci;

        if (funcs->init_func) {
                state.resize(1);
                state[0] = funcs->init_func(parent, compress_options.c_str());
                if(!state[0]) {
                        LOG(LOG_LEVEL_ERROR)
                            << MOD_NAME "Compression initialization failed: "
                            << config_string << "\n";
                        throw -1;
                }
                if(state[0] == INIT_NOERR) {
                        throw 1;
                }
        } else {
                throw -1;
        }
}

void compress_state_real::start(struct compress_state *proxy)
{
        if (funcs->compress_frame_async_push_func) {
                asynch_consumer_thread = thread(&compress_state_real::async_consumer, this, proxy);
        } else if (funcs->compress_tile_async_push_func){
                asynch_consumer_thread = thread(&compress_state_real::async_tile_consumer, this, proxy);
        }
}

/**
 * Checks if there are at least as many states as there are tiles.
 * If there are not enough states it initializes new ones. 
 *
 * @param         proxy         compress state
 * @param[in]     frame         uncompressed frame
 * @return                      false in case of failure
 */
static bool check_state_count(unsigned tile_count, struct compress_state *proxy)
{
        struct compress_state_real *s = proxy->ptr;

        if(tile_count != s->state.size()) {
                size_t old_size = s->state.size();
                s->state.resize(tile_count);
                for (unsigned int i = old_size; i < s->state.size(); ++i) {
                        s->state[i] = s->funcs->init_func(&proxy->mod, s->compress_options.c_str());
                        if(!s->state[i]) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME
                                    "Compression initialization failed\n";
                                return false;
                        }
                }
        }
        return true;
}

/**
 * Puts frame for compression to queue and returns, result must be queried by
 * compress_pop().
 *
 * In case of error, no frame is returned.
 *
 * Accepts poison pill (shared_ptr<video_frame>{nullptr}) and passes it over the queue
 * to compress_pop().
 *
 * @param proxy        compress state
 * @param frame        uncompressed frame to be compressed
 */
void compress_frame(struct compress_state *proxy, shared_ptr<video_frame> frame)
{
        if (!proxy)
                abort();

        struct msg_change_compress_data *msg = NULL;
        while ((msg = (struct msg_change_compress_data *) check_message(&proxy->mod))) {
                compress_process_message(proxy, msg);
        }

        struct compress_state_real *s = proxy->ptr;

        if (!frame) {
                proxy->poisoned = true;
        }
        if (frame) {
                frame->compress_start = get_time_in_ns();
        }

        if (s->funcs->compress_frame_async_push_func) {
                assert(s->funcs->compress_frame_async_pop_func);
                s->funcs->compress_frame_async_push_func(s->state[0],
                                                         std::move(frame));
                return;
        }
        if (s->funcs->compress_tile_async_push_func) {
                assert(s->funcs->compress_tile_async_pop_func);
                if (!frame) {
                        async_poison(s);
                        return;
                }

                if(!check_state_count(frame->tile_count, proxy)){
                        return;
                }

                vector<shared_ptr<video_frame>> separate_tiles =
                    vf_separate_tiles(std::move(frame));

                for(unsigned i = 0; i < separate_tiles.size(); i++){
                        s->funcs->compress_tile_async_push_func(s->state[i], separate_tiles[i]);
                }
                return;
        }

        // sync APIs - pass poisoned pill to the queue but not to compressions,
        if (!frame) { // which doesn't need that but use NULL frame differently
                proxy->queue.push(shared_ptr<video_frame>());
                return;
        }

        shared_ptr<video_frame> sync_api_frame;
        do {
                if (s->funcs->compress_frame_func) {
                        sync_api_frame = s->funcs->compress_frame_func(s->state[0], frame);
                } else if(s->funcs->compress_tile_func) {
                        sync_api_frame = compress_frame_tiles(proxy, frame);
                } else {
                        assert(!"No eligible compress API found");
                }

                // empty return value here represents error, but we don't want to pass it to queue, since it would
                // be interpreted as poisoned pill
                if (!sync_api_frame) {
                        return;
                }
                sync_api_frame->compress_end = get_time_in_ns();
                proxy->queue.push(sync_api_frame);
                frame = nullptr;
        } while (true);
}

/**
 * @name Tile API Routines
 * The worker callbacks here are optimization - all tiles are processed concurrently.
 * @{
 */
/**
 * @brief Auxiliary structure passed to worker thread.
 */
struct compress_worker_data {
        void *state;      ///< compress driver status
        shared_ptr<video_frame> frame; ///< uncompressed tile to be compressed

        compress_tile_t callback;  ///< tile compress callback
        shared_ptr<video_frame> ret; ///< OUT - returned compressed tile, NULL if failed
};

/**
 * @brief This function is callback passed to a "thread pool"
 * @param arg @ref compress_worker_data
 * @return @ref compress_worker_data (same as input)
 */
static void *compress_tile_callback(void *arg) {
        compress_worker_data *s = (compress_worker_data *) arg;

        s->ret = s->callback(s->state, s->frame);

        return s;
}

/**
 * Compresses video frame with tiles API
 *
 * @param         proxy         compress state
 * @param[in]     frame         uncompressed frame
 * @return                      compressed video frame, may be NULL if compression failed
 */
static shared_ptr<video_frame> compress_frame_tiles(struct compress_state *proxy,
                shared_ptr<video_frame> frame)
{
        struct compress_state_real *s = proxy->ptr;
        vector<shared_ptr<video_frame>> separate_tiles;
        if (frame) {
                if (!check_state_count(frame->tile_count, proxy)) {
                        return nullptr;
                }
                separate_tiles = vf_separate_tiles(frame);
        } else {
                separate_tiles.resize(proxy->ptr->state.size());
        }

        // frame pointer may no longer be valid
        frame = NULL;

        const int tile_cnt = (int) proxy->ptr->state.size();
        vector<task_result_handle_t> task_handle(tile_cnt);

        vector <compress_worker_data> data_tile(tile_cnt);
        for (int i = 0; i < tile_cnt; ++i) {
                struct compress_worker_data *data = &data_tile[i];
                data->state = s->state[i];
                data->frame = separate_tiles[i];
                data->callback = s->funcs->compress_tile_func;

                task_handle[i] = task_run_async(compress_tile_callback, data);
        }

        vector<shared_ptr<video_frame>> compressed_tiles(separate_tiles.size());

        bool failed = false;
        for (int i = 0; i < tile_cnt; ++i) {
                struct compress_worker_data *data = (struct compress_worker_data *)
                        wait_task(task_handle[i]);

                if(!data->ret) {
                        failed = true;
                }

                compressed_tiles[i] = data->ret;
        }

        if (failed) {
                return NULL;
        }

        return vf_merge_tiles(compressed_tiles);
}
/**
 * @}
 */

/**
 * @brief Video compression cleanup function.
 * @param mod video compress module
 */
void
compress_done(struct compress_state *proxy)
{
        if (proxy == nullptr) {
                return;
        }

        struct compress_state_real *s = proxy->ptr;
        if (!proxy->poisoned) { // pass poisoned pill if it wasn't
                compress_frame(proxy, {});
        }

        delete s;
        delete proxy;
}

compress_state_real::~compress_state_real()
{
        if (asynch_consumer_thread.joinable()) {
                asynch_consumer_thread.join();
        }

        for(unsigned int i = 0; i < state.size(); ++i) {
                funcs->done(state[i]);
        }
}

namespace {
void compress_state_real::async_tile_consumer(struct compress_state *s)
{
        set_thread_name(__func__);
        vector<shared_ptr<video_frame>> compressed_tiles;
        unsigned expected_seq = 0;
        while (true) {
                bool fail = false;
                for(unsigned i = 0; i < state.size(); i++){
                        std::shared_ptr<video_frame> ret = nullptr;
                        //discard frames with seq lower than expected
                        do {
                                ret = funcs->compress_tile_async_pop_func(state[i]);
                        } while(ret && ret->seq < expected_seq);

                        if (!ret) {
                                if(!discard_frames)
                                        s->queue.push(nullptr); //poison
                                return;
                        }

                        if(ret->seq > expected_seq){
                                log_msg(LOG_LEVEL_ERROR,
                                                "Expected sequence number %u but got %u!\n",
                                                expected_seq,
                                                ret->seq);
                                if(i == 0){
                                        //If this is first tile we can continue
                                        expected_seq = ret->seq;
                                } else {
                                        expected_seq = ret->seq + 1;
                                        fail = true;
                                        break;
                                }
                        }

                        ret->compress_end = get_time_in_ns();
                        compressed_tiles.resize(state.size(), nullptr);
                        compressed_tiles[i] = std::move(ret);
                }

                if(fail)
                        continue;

                if (!discard_frames) {
                        s->queue.push(vf_merge_tiles(compressed_tiles));
                }
                //If frames are not numbered they always have seq = 0
                if(expected_seq > 0) expected_seq++;
        }
}

void compress_state_real::async_consumer(struct compress_state *s)
{
        set_thread_name(__func__);
        while (true) {
                auto frame = funcs->compress_frame_async_pop_func(state[0]);
                if (!discard_frames) {
                        s->queue.push(frame);

                }
                if (!frame) {
                        return;

                }

        }

}
} // end of anonymous namespace

/**
 * @returns compressed frame previously enqueued by compress_frame(). If an error
 * occurs function doesn't return.
 * @retval shared_ptr<video_frame>{} poison pill passed previously to compress_frame()
 */
shared_ptr<video_frame> compress_pop(struct compress_state *proxy)
{
        if(!proxy)
                return NULL;

        auto f = proxy->queue.pop();
        if (f) {
                char sz_str[FORMAT_NUM_MAX_SZ];
                MSG(DEBUG, "Compressed frame size: %s B; duration: %7.3f ms\n",
                    format_number_with_delim(vf_get_data_len(f.get()), sz_str,
                                             sizeof sz_str),
                    (f->compress_end - f->compress_start) / MS_IN_NS_DBL);
        }
        return f;
}


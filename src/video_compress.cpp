/**
 * @file   video_compress.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup video_compress
 *
 * @brief Video compress functions.
 */
/*
 * Copyright (c) 2011-2013 CESNET z.s.p.o.
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
#endif // HAVE_CONFIG_H

#include <memory>
#include <stdio.h>
#include <string>
#include <string.h>
#include <thread>
#include <vector>

#include "compat/platform_time.h"
#include "messaging.h"
#include "module.h"
#include "utils/synchronized_queue.h"
#include "utils/thread.h"
#include "utils/vf_split.h"
#include "utils/worker.h"
#include "video.h"
#include "video_compress.h"
#include "lib_common.h"
#include "debug.h"

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
        vector<struct module *> state;                  ///< driver internal states
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
        struct module mod;               ///< compress module data
        struct compress_state_real *ptr; ///< pointer to real compress state
        synchronized_queue<shared_ptr<video_frame>, 1> queue;
};

/**
 * This is placeholder state returned by compression module meaning that the initialization was
 * successful but no state was create. This is the case eg. when the module only displayed help.
 */
struct module compress_init_noerr;

static shared_ptr<video_frame> compress_frame_tiles(struct compress_state *proxy,
                shared_ptr<video_frame> frame);
static void compress_done(struct module *mod);

/// @brief Displays list of available compressions.
void show_compress_help()
{
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        list_modules(LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
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
                        struct response *resp = send_message_to_receiver(proxy->ptr->state[i],
                                        (struct message *) tmp_data);
                        /// @todo
                        /// Handle responses more inteligently (eg. aggregate).
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
 * This function wrapps the call of compress_init_real().
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @param[out] state        created state
 * @retval     0            if state created sucessfully
 * @retval    <0            if error occured
 * @retval    >0            finished successfully, no state created (eg. displayed help)
 */
int compress_init(struct module *parent, const char *config_string, struct compress_state **state) {
        struct compress_state *proxy;
        proxy = new struct compress_state();

        module_init_default(&proxy->mod);
        proxy->mod.cls = MODULE_CLASS_COMPRESS;
        proxy->mod.priv_data = proxy;
        proxy->mod.deleter = compress_done;

        try {
                proxy->ptr = compress_state_real::create(&proxy->mod, config_string, proxy);
        } catch (int i) {
                delete proxy;
                return i;
        }

        module_register(&proxy->mod, parent);

        *state = proxy;
        return 0;
}

/**
 * @brief Constructor for compress_state_real
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @throws    -1            if error occured
 * @retval     1            finished successfully, no state created (eg. displayed help)
 */
compress_state_real::compress_state_real(struct module *parent, const char *config_string) :
        funcs(nullptr), discard_frames(false)
{
        string compress_name;

        if (!config_string)
                throw -1;

        if (strcmp(config_string, "help") == 0)
        {
                show_compress_help();
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
                fprintf(stderr, "Unknown compression: %s\n", config_string);
                throw -1;
        }

        funcs = vci;

        if (funcs->init_func) {
                state.resize(1);
                state[0] = funcs->init_func(parent, compress_options.c_str());
                if(!state[0]) {
                        fprintf(stderr, "Compression initialization failed: %s\n", config_string);
                        throw -1;
                }
                if(state[0] == &compress_init_noerr) {
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
 * @brief Returns name of compression module
 *
 * @param proxy compress state
 * @returns     compress name
 */
const char *get_compress_name(struct compress_state *proxy)
{
        if(proxy)
                return proxy->ptr->funcs->name;
        else
                return NULL;
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
                                fprintf(stderr, "Compression initialization failed\n");
                                return false;
                        }
                }
        }
        return true;
}

/**
 * @brief Compressses frame
 *
 * @param proxy        compress state
 * @param frame        uncompressed frame to be compressed
 * @return             compressed frame, may be NULL if compression failed
 */
void compress_frame(struct compress_state *proxy, shared_ptr<video_frame> frame)
{
        if (!proxy)
                abort();

        uint64_t t0 = time_since_epoch_in_ms();

        struct msg_change_compress_data *msg = NULL;
        while ((msg = (struct msg_change_compress_data *) check_message(&proxy->mod))) {
                compress_process_message(proxy, msg);
        }

        struct compress_state_real *s = proxy->ptr;

        if (s->funcs->compress_frame_async_push_func) {
                assert(s->funcs->compress_frame_async_pop_func);
                if (frame) {
                        frame->compress_start = t0;
                }
                s->funcs->compress_frame_async_push_func(s->state[0], frame);
        } else if (s->funcs->compress_tile_async_push_func) {
                assert(s->funcs->compress_tile_async_pop_func);
                if (!frame) {
                        async_poison(s);
                        return;
                }

                frame->compress_start = t0;

                if(!check_state_count(frame->tile_count, proxy)){
                        return;
                }

                vector<shared_ptr<video_frame>> separate_tiles = vf_separate_tiles(frame);
                // frame pointer may no longer be valid
                frame = NULL;

                for(unsigned i = 0; i < separate_tiles.size(); i++){
                        s->funcs->compress_tile_async_push_func(s->state[i], separate_tiles[i]);
                }

        } else {
                if (!frame) { // pass poisoned pill
                        proxy->queue.push(shared_ptr<video_frame>());
                        return;
                }

                shared_ptr<video_frame> sync_api_frame;
                if (s->funcs->compress_frame_func) {
                        sync_api_frame = s->funcs->compress_frame_func(s->state[0], frame);
                } else if(s->funcs->compress_tile_func) {
                        sync_api_frame = compress_frame_tiles(proxy, frame);
                } else {
                        assert(!"No egliable compress API found");
                }

                // empty return value here represents error, but we don't want to pass it to queue, since it would
                // be interpreted as poisoned pill
                if (!sync_api_frame) {
                        return;
                }

                sync_api_frame->compress_start = t0;
                sync_api_frame->compress_end = time_since_epoch_in_ms();

                proxy->queue.push(sync_api_frame);
        }
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
        struct module *state;      ///< compress driver status
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

        if(!check_state_count(frame->tile_count, proxy)){
                return NULL;
        }

        vector<shared_ptr<video_frame>> separate_tiles = vf_separate_tiles(frame);
        // frame pointer may no longer be valid
        frame = NULL;

        vector<task_result_handle_t> task_handle(separate_tiles.size());

        vector <compress_worker_data> data_tile(separate_tiles.size());
        for(unsigned int i = 0; i < separate_tiles.size(); ++i) {
                struct compress_worker_data *data = &data_tile[i];
                data->state = s->state[i];
                data->frame = separate_tiles[i];
                data->callback = s->funcs->compress_tile_func;

                task_handle[i] = task_run_async(compress_tile_callback, data);
        }

        vector<shared_ptr<video_frame>> compressed_tiles(separate_tiles.size(), nullptr);

        bool failed = false;
        for(unsigned int i = 0; i < separate_tiles.size(); ++i) {
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
static void compress_done(struct module *mod)
{
        if(!mod)
                return;

        struct compress_state *proxy = (struct compress_state *) mod->priv_data;
        struct compress_state_real *s = proxy->ptr;
        delete s;

        delete proxy;
}

compress_state_real::~compress_state_real()
{
        if (asynch_consumer_thread.joinable()) {
                asynch_consumer_thread.join();
        }

        for(unsigned int i = 0; i < state.size(); ++i) {
                module_done(state[i]);
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

                        ret->compress_end = time_since_epoch_in_ms();
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

shared_ptr<video_frame> compress_pop(struct compress_state *proxy)
{
        if(!proxy)
                return NULL;

        return proxy->queue.pop();
}


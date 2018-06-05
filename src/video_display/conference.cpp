/**
 * @file   video_display/conference.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2014-2015 CESNET, z. s. p. o.
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
/**
 * @todo
 * Rewrite the code to have more clear state machine!
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "video_codec.h"

#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <sys/time.h>

#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_CUDA
#ifdef HAVE_OPENCV_2
#include <opencv2/gpu/gpu.hpp>
using GpuMat = cv::gpu::GpuMat;
using Stream = cv::gpu::Stream;
#define gpu_resize(...) cv::gpu::resize(__VA_ARGS__)
#elif defined HAVE_OPENCV_3
#include <opencv2/cudawarping.hpp>
using GpuMat = cv::cuda::GpuMat;
using Stream = cv::cuda::Stream;
#define gpu_resize(...) cv::cuda::resize(__VA_ARGS__)
#endif
#endif


struct Tile{
        cv::Mat img;
#ifdef HAVE_OPENCV_CUDA
        GpuMat gpuImg;
#endif
        int width, height;
        int posX, posY;

        uint32_t ssrc;

        //If true, it idicates that the tile image was changed
        //and needs to be scaled and/or uploaded to the gpu again
        bool dirty;
};

class TiledImage{
        public:
                TiledImage(int width, int height);
                virtual ~TiledImage() {}

                void computeLayout();
                virtual cv::Mat getImg() = 0;

                void addTile(unsigned width, unsigned height, uint32_t ssrc);
                void removeTile(uint32_t ssrc);
                struct Tile *getTile(uint32_t ssrc);
                cv::Mat *getMat(uint32_t ssrc);
                void updateTile(uint32_t ssrc, bool scaleNow = true);
                virtual void processTile(struct Tile *) = 0;

                virtual void resetImg() = 0;

                int width, height;

                //In Normal layout every tile covers equal space
                //In OneBig layout the primary tile takes 2/3 of space
                enum Layout{
                        Normal, OneBig
                };

                Layout layout;
                unsigned primary = 0; //The biggest tile in the OneBig layout

        protected:
                std::vector<struct Tile> imgs;
};

#ifdef HAVE_OPENCV_CUDA
class TiledImageGpu : public TiledImage{
        public:
                TiledImageGpu(int width, int height) : TiledImage(width, height){
                        image.create(height, width, CV_8UC3);
                }
                cv::Mat getImg() override;
                void processTile(struct Tile *) override;
                void resetImg() override { image.setTo(cv::Scalar::all(0)); }
        private:
                GpuMat image;
                Stream stream;
};
#endif

class TiledImageCpu : public TiledImage{
        public:
                TiledImageCpu(int width, int height) : TiledImage(width, height){
                        image.create(height, width, CV_8UC3);
                }
                cv::Mat getImg() override;
                void processTile(struct Tile *) override;
                void resetImg() override { image.setTo(cv::Scalar::all(0)); }
        private:
                cv::Mat image;
};

TiledImage::TiledImage(int width, int height) : width(width), height(height){
        layout = Normal;
}

void TiledImage::addTile(unsigned width, unsigned height, uint32_t ssrc){
        struct Tile t;
        t.width = width;
        t.height = height;
        t.posX = 0;
        t.posY = 0;
        t.ssrc = ssrc;
        t.dirty = 1;

        t.img.create(height, width, CV_8UC3);

        imgs.push_back(std::move(t));
}

void TiledImage::removeTile(uint32_t ssrc){
        auto it = imgs.begin();

        while(it != imgs.end()){
                if(it->ssrc == ssrc){
                        it = imgs.erase(it);
                } else {
                        it++;
                }
        }
}

#ifdef HAVE_OPENCV_CUDA
//Uploads the tile to the gpu, resizes it and puts it in the correct position
void TiledImageGpu::processTile(struct Tile *t){
#ifdef HAVE_OPENCV_2
        stream.enqueueUpload(t->img, t->gpuImg);
#elif defined HAVE_OPENCV_3
        t->gpuImg.upload(t->img, stream);
#endif
        GpuMat rect = image(cv::Rect(t->posX, t->posY, t->width, t->height));
        gpu_resize(t->gpuImg, rect, cv::Size(t->width, t->height), 0, 0, cv::INTER_NEAREST, stream);
        t->dirty = 0;
}
#endif

void TiledImageCpu::processTile(struct Tile *t){
        cv::Mat rect = image(cv::Rect(t->posX, t->posY, t->width, t->height));
        cv::resize(t->img, rect, cv::Size(t->width, t->height), 0, 0, cv::INTER_NEAREST);
        t->dirty = 0;
}

void TiledImage::updateTile(uint32_t ssrc, bool scaleNow){
        struct Tile *t = getTile(ssrc);
        if(!scaleNow){
                t->dirty = true;
                return;
        }
        processTile(t);
}

struct Tile *TiledImage::getTile(uint32_t ssrc){
        for(auto& t : imgs){
                if(t.ssrc == ssrc){
                        return &t;
                }
        }

        return nullptr;
}

cv::Mat *TiledImage::getMat(uint32_t ssrc){
        return &getTile(ssrc)->img;
}

void setTileSize(struct Tile *t, unsigned width, unsigned height){
        float scaleFactor = std::min((float) width / t->img.size().width, (float) height / t->img.size().height);

        t->width = t->img.size().width * scaleFactor;
        t->height = t->img.size().height * scaleFactor;

        //Center tile
        t->posX += (width - t->width) / 2;
        t->posY += (height - t->height) / 2;
} 

void TiledImage::computeLayout(){
        unsigned tileW;
        unsigned tileH;

        if(imgs.size() == 1){
                Tile& t = imgs[0];
                t.posX = 0;
                t.posY = 0;
                setTileSize(&t, width, height);
                t.dirty = 1;
                resetImg();
                return;
        }

        unsigned smallH = height / 3;
        unsigned bigH = smallH * 2;

        unsigned rows;
        switch(layout){
                case OneBig:
                        rows = imgs.size() - 1;
                        if(rows < 1) rows = 1;
                        tileW = width / rows;
                        tileH = smallH;
                        break;
                case Normal:
                default:
                        rows = (unsigned) ceil(sqrt(imgs.size()));
                        tileW = width / rows;
                        tileH = height / rows;
                        break;
        }

        unsigned i = 0;
        int pos = 0;
        for(struct Tile& t: imgs){
                unsigned curW = tileW;
                unsigned curH = tileH;
                t.posX = (pos % rows) * curW;
                t.posY = (pos / rows) * curH;
                if(layout == OneBig){
                        t.posY += bigH;
                        if(i == primary){
                                t.posX = 0;
                                t.posY = 0;
                                curW = width;
                                curH = bigH;
                                pos--;
                        }
                }

                setTileSize(&t, curW, curH);
                t.dirty = 1;

                i++;
                pos++;
        }

        resetImg();
}

#ifdef HAVE_OPENCV_CUDA
cv::Mat TiledImageGpu::getImg(){
        stream.waitForCompletion();

        for(struct Tile& t: imgs){
                if(t.dirty){
                        processTile(&t);
                }
        }

        cv::Mat result;
        stream.waitForCompletion();
        image.download(result);
        return result;
}
#endif

cv::Mat TiledImageCpu::getImg(){
        for(struct Tile& t: imgs){
                if(t.dirty){
                        processTile(&t);
                }
        }

        return image;
}

using namespace std;

static constexpr chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_conference_common {
        state_conference_common(int width, int height, int fps, bool gpu) : width(width), 
                                                              height(height),
                                                              fps(fps),
                                                              gpu(gpu)
        {
                autofps = fps <= 0;
#ifdef HAVE_OPENCV_CUDA
                if(gpu){
                        output = std::unique_ptr<TiledImage>(new TiledImageGpu(width, height));
                }else{
                        output = std::unique_ptr<TiledImage>(new TiledImageCpu(width, height));
                }
#else
                output = std::unique_ptr<TiledImage>(new TiledImageCpu(width, height));
#endif
        }

        ~state_conference_common() {
                display_done(real_display);
        }
        struct display *real_display = {};
        struct video_desc display_desc = {};

        queue<struct video_frame *> incoming_queue;
        condition_variable in_queue_decremented_cv;
        unordered_map<uint32_t, chrono::system_clock::time_point> ssrc_list;

        int width = 0;
        int height = 0;
        double fps = 0.0;
        bool autofps;
        bool gpu;
        std::unique_ptr<TiledImage> output;

        chrono::system_clock::time_point next_frame;

        pthread_t thread_id = {};

        mutex lock;
        condition_variable cv;

        struct module *parent = NULL;
};

struct state_conference {
        shared_ptr<struct state_conference_common> common;
        struct video_desc desc;
};

static struct display *display_conference_fork(void *state)
{
        shared_ptr<struct state_conference_common> s = ((struct state_conference *)state)->common;
        struct display *out;
        char fmt[2 + sizeof(void *) * 2 + 1] = "";
        snprintf(fmt, sizeof fmt, "%p", state);

        int rc = initialize_video_display(s->parent,
                        "conference", fmt, 0, NULL, &out);
        if (rc == 0) return out; else return NULL;

        return out;
}

static void show_help(){
        printf("Conference display\n");
        printf("Usage:\n");
#ifdef HAVE_OPENCV_CUDA
        printf("\t-d conference:<display_config>#<width>:<height>:[fps]:[{CPU|GPU}]\n");
#else 
        printf("\t-d conference:<display_config>#<width>:<height>:[fps]:[{CPU}]\n");
#endif
}

static void *display_conference_init(struct module *parent, const char *fmt, unsigned int flags)
{
        struct state_conference *s;
        char *fmt_copy = NULL;
        const char *requested_display = "gl";
        const char *cfg = NULL;

        printf("FMT: %s\n", fmt);

        s = new state_conference();

        int width, height;
        double fps = 0;
#ifdef HAVE_OPENCV_CUDA
        bool gpu = true;
#else
        bool gpu = false;
#endif


        if (fmt && strlen(fmt) > 0) {
                if (isdigit(fmt[0])) { // fork
                        struct state_conference *orig;
                        sscanf(fmt, "%p", &orig);
                        s->common = orig->common;
                        return s;
                } else {
                        char *tmp = strdup(fmt);
                        char *save_ptr = NULL;
                        char *item;

                        item = strtok_r(tmp, "#", &save_ptr);
                        if(!item || strlen(item) == 0){
                                show_help();
                                free(tmp);
                                delete s;
                                return &display_init_noerr;
                        }
                        //Display configuration
                        fmt_copy = strdup(item);
                        requested_display = fmt_copy;
                        char *delim = strchr(fmt_copy, ':');
                        if (delim) {
                                *delim = '\0';
                                cfg = delim + 1;
                        }
                        item = strtok_r(NULL, "#", &save_ptr);
                        //Conference configuration
                        if(!item || strlen(item) == 0){
                                show_help();
                                free(fmt_copy);
                                free(tmp);
                                delete s;
                                return &display_init_noerr;
                        }
                        width = atoi(item);
                        item = strchr(item, ':');
                        if(!item || strlen(item + 1) == 0){
                                show_help();
                                free(fmt_copy);
                                free(tmp);
                                delete s;
                                return &display_init_noerr;
                        }
                        height = atoi(++item);
                        if((item = strchr(item, ':'))){
                                fps = atoi(++item);
                        }
                        if((item && (item = strchr(item, ':')))){
                                ++item;
                                if(strcasecmp(item, "cpu") == 0){
                                        gpu = false;
#ifndef HAVE_OPENCV_CUDA
                                } else if(strcasecmp(item, "gpu") == 0){
                                        gpu = false;
                                        fprintf(stderr, "GPU videomixing requested, but ultragrid was built without CUDA support for OpenCV; CPU implementation in service\n");
#endif
                                }
                        }
                        free(tmp);
                }
        } else {
                show_help();
                delete s;
                return &display_init_noerr;
        }
        s->common = shared_ptr<state_conference_common>(new state_conference_common(width, height, fps, gpu));
        assert (initialize_video_display(parent, requested_display, cfg, flags, NULL, &s->common->real_display) == 0);
        free(fmt_copy);

        int ret = pthread_create(&s->common->thread_id, NULL, (void *(*)(void *)) display_run,
                        s->common->real_display);
        assert (ret == 0);

        s->common->parent = parent;

        return s;
}

static void check_reconf(struct state_conference_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                fprintf(stderr, "RECONFIGURED\n");
                display_reconfigure(s->real_display, s->display_desc, VIDEO_NORMAL);
        }
}

static video_desc get_video_desc(std::shared_ptr<struct state_conference_common> s){
        video_desc desc;
        desc.width = s->width;
        desc.height = s->height;
        desc.fps = s->fps;
        desc.color_spec = UYVY;
        desc.interlacing = PROGRESSIVE;
        desc.tile_count = 1;

        return desc;
}

static void display_conference_run(void *state)
{
        shared_ptr<struct state_conference_common> s = ((struct state_conference *)state)->common;
        uint32_t last_ssrc = 0;

        while (1) {
                //get frame
                struct video_frame *frame;
                {
                        unique_lock<mutex> lg(s->lock);
                        s->cv.wait(lg, [s]{return s->incoming_queue.size() > 0;});
                        frame = s->incoming_queue.front();
                        s->incoming_queue.pop();
                        s->in_queue_decremented_cv.notify_one();
                }

                if (!frame) {
                        display_put_frame(s->real_display, NULL, PUTF_BLOCKING);
                        break;
                }

                chrono::system_clock::time_point now = chrono::system_clock::now();

                //Check if any source timed out
                auto it = s->ssrc_list.begin();
                while (it != s->ssrc_list.end()) {
                        if (chrono::duration_cast<chrono::milliseconds>(now - it->second) > SOURCE_TIMEOUT) {
                                verbose_msg("Source 0x%08lx timeout. Deleting from conference display.\n", it->first);

                                s->output->removeTile(it->first);
                                it = s->ssrc_list.erase(it);

                                if(!s->ssrc_list.empty()){
                                        s->output->computeLayout();
                                }

                        } else {
                                ++it;
                        }
                }

                it = s->ssrc_list.find(frame->ssrc);
                if (it == s->ssrc_list.end()) {
                        //Received frame from new source
                        s->output->addTile(frame->tiles[0].width, frame->tiles[0].height, frame->ssrc);
                        s->output->computeLayout();
                        last_ssrc = frame->ssrc;
                        if(frame->fps > s->fps && s->autofps){
                                s->fps = frame->fps;
                                s->next_frame = now;
                        }
                        //If this is the first source we need to schedule
                        //the next output frame now
                        if(s->ssrc_list.empty()){
                                s->next_frame = now;
                        }
                }
                s->ssrc_list[frame->ssrc] = now;

                //Convert the tile to RGB and then upload to gpu
                for(unsigned i = 0; i < frame->tiles[0].height; i++){
                        int width = frame->tiles[0].width;
                        int elemSize = s->output->getMat(frame->ssrc)->elemSize();
                        vc_copylineUYVYtoRGB_SSE(s->output->getMat(frame->ssrc)->data + i*width*elemSize, (const unsigned char*)frame->tiles[0].data + i*width*2, width*elemSize);
                }
                s->output->updateTile(frame->ssrc);

                vf_free(frame);

                now = chrono::high_resolution_clock::now();

                //If it's time to send next frame downlaod it from gpu,
                //convert to UYVY and send
                if (now >= s->next_frame){
                        cv::Mat result = s->output->getImg();
                        check_reconf(s.get(), get_video_desc(s));
                        struct video_frame *outFrame = display_get_frame(s->real_display);
                        for(int i = 0; i < result.size().height; i++){
                                int width = result.size().width;
                                int elemSize = result.elemSize();
                                vc_copylineRGBtoUYVY_SSE((unsigned char*)outFrame->tiles[0].data + i*width*2, result.data + i*width*elemSize, width*2);
                        }
                        outFrame->ssrc = last_ssrc;

                        display_put_frame(s->real_display, outFrame, PUTF_BLOCKING);
                        s->next_frame += chrono::milliseconds((int) (1000 / s->fps));
                }
        }

        pthread_join(s->thread_id, NULL);
}

static void display_conference_done(void *state)
{
        struct state_conference *s = (struct state_conference *)state;
        delete s;
}

static struct video_frame *display_conference_getf(void *state)
{
        struct state_conference *s = (struct state_conference *)state;

        return vf_alloc_desc_data(s->desc);
}

static int display_conference_putf(void *state, struct video_frame *frame, int flags)
{
        shared_ptr<struct state_conference_common> s = ((struct state_conference *)state)->common;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                unique_lock<mutex> lg(s->lock);
                if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        fprintf(stderr, "Conference: queue full!\n");
                        //vf_free(frame);
                }
                if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        vf_free(frame);
                        return 1;
                }
                s->in_queue_decremented_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
                s->incoming_queue.push(frame);
                lg.unlock();
                s->cv.notify_one();
        }

        return 0;
}

static int display_conference_get_property(void *state, int property, void *val, size_t *len)
{
        shared_ptr<struct state_conference_common> s = ((struct state_conference *)state)->common;
        if (property == DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES) {
                ((struct multi_sources_supp_info *) val)->val = true;
                ((struct multi_sources_supp_info *) val)->fork_display = display_conference_fork;
                ((struct multi_sources_supp_info *) val)->state = state;
                *len = sizeof(struct multi_sources_supp_info);
                return TRUE;

        } else {
                return display_get_property(s->real_display, property, val, len);
        }
}

static int display_conference_reconfigure(void *state, struct video_desc desc)
{
        struct state_conference *s = (struct state_conference *) state;

        s->desc = desc;

        return 1;
}

static void display_conference_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_conference_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_conference_info = {
        [](struct device_info **available_cards, int *count) {
                *available_cards = nullptr;
                *count = 0;
        },
        display_conference_init,
        display_conference_run,
        display_conference_done,
        display_conference_getf,
        display_conference_putf,
        display_conference_reconfigure,
        display_conference_get_property,
        display_conference_put_audio_frame,
        display_conference_reconfigure_audio,
};

REGISTER_MODULE(conference, &display_conference_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


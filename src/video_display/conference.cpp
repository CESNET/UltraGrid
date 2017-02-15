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
#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION <= 2
        #include <opencv2/gpu/gpu.hpp>
        namespace cvgpu = cv::gpu;
        #define CVGPU_STREAM_ENQ_UPLOAD(cpuSrc, gpuDst, stream) do { (stream).enqueueUpload((cpuSrc), (gpuDst)); } while (0)
#else
        #include <opencv2/core/cuda.hpp>
        namespace cvgpu = cv::cuda;
        #define CVGPU_STREAM_ENQ_UPLOAD(cpuSrc, gpuDst, stream) do { (gpuDst).upload((cpuSrc), (stream)); } while (0)
#endif
#include <sys/time.h>

struct Tile{
        std::shared_ptr<cv::Mat> img;
        std::shared_ptr<cvgpu::GpuMat> gpuImg;
        int width, height;
        int posX, posY;

        //If true, it idicates that the tile image was changed
        //and needs to be uploaded to the gpu again
        bool dirty;
};

class TiledImage{
        public:
                TiledImage(int width, int height);
                virtual ~TiledImage() {}

                void computeLayout();
                cv::Mat getImg();

                void addTile(unsigned width, unsigned height, uint32_t ssrc);
                void removeTile(uint32_t ssrc);
                std::shared_ptr<cv::Mat> getMat(uint32_t ssrc);
                void uploadTile(uint32_t ssrc);

                //In Normal layout every tile covers equal space
                //In OneBig layout the primary tile takes 2/3 of space
                enum Layout{
                        Normal, OneBig
                };

                Layout layout;
                unsigned primary = 0; //The biggest tile in the OneBig layout

        private:
                std::vector<std::shared_ptr<struct Tile>> imgs;
                std::unordered_map<uint32_t, std::shared_ptr<struct Tile>> tiles;
                cvgpu::GpuMat image;
                cvgpu::Stream stream;
};

TiledImage::TiledImage(int width, int height){
        image = cvgpu::GpuMat(height, width, CV_8UC3);
        layout = Normal;
}

void TiledImage::addTile(unsigned width, unsigned height, uint32_t ssrc){
        std::shared_ptr<struct Tile> t = std::make_shared<struct Tile>();
        t->width = width;
        t->height = height;
        t->posX = 0;
        t->posY = 0;
        t->dirty = 1;

        t->img = std::make_shared<cv::Mat> (height, width, CV_8UC3);
        t->gpuImg = std::make_shared<cvgpu::GpuMat> (height, width, CV_8UC3);

        imgs.push_back(t);
        tiles[ssrc] = t;
}

void TiledImage::removeTile(uint32_t ssrc){
        std::vector<std::shared_ptr<struct Tile>>::iterator it = imgs.begin();

        while(it != imgs.end()){
                if(*it == tiles[ssrc]){
                        it = imgs.erase(it);
                } else {
                        it++;
                }
        }

        tiles.erase(ssrc);
}

//Uploads the tile to the gpu, resizes it and puts it in the correct position
void TiledImage::uploadTile(uint32_t ssrc){
        std::shared_ptr<struct Tile> t = tiles[ssrc];
        CVGPU_STREAM_ENQ_UPLOAD(*(t->img), *(t->gpuImg), stream);
        cvgpu::GpuMat rect = image(cv::Rect(t->posX, t->posY, t->width, t->height));
        cvgpu::resize(*t->gpuImg, rect, cv::Size(t->width, t->height), 0, 0, cv::INTER_NEAREST, stream);
        t->dirty = 0;
}

std::shared_ptr<cv::Mat> TiledImage::getMat(uint32_t ssrc){
        return tiles[ssrc]->img;
}

void setTileSize(struct Tile *t, unsigned width, unsigned height){
        float scaleFactor = std::min((float) width / t->img->size().width, (float) height / t->img->size().height);

        t->width = t->img->size().width * scaleFactor;
        t->height = t->img->size().height * scaleFactor;

        //Center tile
        t->posX += (width - t->width) / 2;
        t->posY += (height - t->height) / 2;
} 

void TiledImage::computeLayout(){
        unsigned tileW;
        unsigned tileH;
        unsigned width = image.size().width;
        unsigned height = image.size().height;

        if(imgs.size() == 1){
                Tile *t = imgs[0].get();
                t->posX = 0;
                t->posY = 0;
                setTileSize(t, width, height);
                t->dirty = 1;
                image.setTo(cv::Scalar::all(0));
                return;
        }

        unsigned smallH = image.size().height / 3;
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
        for(std::shared_ptr<struct Tile> t: imgs){
                unsigned curW = tileW;
                unsigned curH = tileH;
                t->posX = (pos % rows) * curW;
                t->posY = (pos / rows) * curH;
                if(layout == OneBig){
                        t->posY += bigH;
                        if(i == primary){
                                t->posX = 0;
                                t->posY = 0;
                                curW = width;
                                curH = bigH;
                                pos--;
                        }
                }

                setTileSize(t.get(), curW, curH);
                t->dirty = 1;

                i++;
                pos++;
        }

        image.setTo(cv::Scalar::all(0));
}

cv::Mat TiledImage::getImg(){
        cvgpu::GpuMat rect;
        cvgpu::GpuMat tileImg;

        stream.waitForCompletion();

        for(std::shared_ptr<struct Tile> t: imgs){
                if(t->dirty){
                        //If the tile changed we need to upload it to the gpu
                        CVGPU_STREAM_ENQ_UPLOAD(*(t->img), *(t->gpuImg), stream);
                        rect = image(cv::Rect(t->posX, t->posY, t->width, t->height));
                        cvgpu::resize(*t->gpuImg, rect, cv::Size(t->width, t->height), 0, 0, cv::INTER_NEAREST, stream);
                        t->dirty = 0;
                }
        }

        cv::Mat result;
        stream.waitForCompletion();
        image.download(result);
        return result;
}

using namespace std;

static constexpr int BUFFER_LEN = 5;
static constexpr chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_conference_common {
        state_conference_common(int width, int height, int fps) : width(width), 
                                                              height(height),
                                                              fps(fps)
        {
                autofps = fps <= 0;
        }

        ~state_conference_common() {
                display_done(real_display);
        }
        struct display *real_display;
        struct video_desc display_desc;

        queue<struct video_frame *> incoming_queue;
        condition_variable in_queue_decremented_cv;
        unordered_map<uint32_t, chrono::system_clock::time_point> ssrc_list;

        int width;
        int height;
        double fps;
        bool autofps;
        TiledImage output = TiledImage(width, height);

        chrono::system_clock::time_point next_frame;

        pthread_t thread_id;

        mutex lock;
        condition_variable cv;

        struct module *parent;
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
        printf("Mosaic display\n");
        printf("Usage:\n");
        printf("\t-d conference:<display_config>#<width>:<height>:[fps]\n");
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
                        //Display configuration
                        fmt_copy = strdup(item);
                        requested_display = fmt_copy;
                        char *delim = strchr(fmt_copy, ':');
                        if (delim) {
                                *delim = '\0';
                                cfg = delim + 1;
                        }
                        item = strtok_r(NULL, "#", &save_ptr);
                        //Mosaic configuration
                        if(!item || strlen(item) == 0){
                                show_help();
                                return &display_init_noerr;
                        }
                        width = atoi(item);
                        item = strchr(item, ':');
                        if(!item || strlen(item + 1) == 0){
                                show_help();
                                return &display_init_noerr;
                        }
                        height = atoi(++item);
                        if((item = strchr(item, ':'))){
                                fps = atoi(++item);
                        }
                }
        } else {
                show_help();
                return &display_init_noerr;
        }
        s->common = shared_ptr<state_conference_common>(new state_conference_common(width, height, fps));
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

                                s->output.removeTile(it->first);
                                it = s->ssrc_list.erase(it);

                                if(!s->ssrc_list.empty()){
                                        s->output.computeLayout();
                                }

                        } else {
                                ++it;
                        }
                }

                it = s->ssrc_list.find(frame->ssrc);
                if (it == s->ssrc_list.end()) {
                        //Received frame from new source
                        s->output.addTile(frame->tiles[0].width, frame->tiles[0].height, frame->ssrc);
                        s->output.computeLayout();
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
                        int elemSize = s->output.getMat(frame->ssrc)->elemSize();
                        vc_copylineUYVYtoRGB_SSE(s->output.getMat(frame->ssrc)->data + i*width*elemSize, (const unsigned char*)frame->tiles[0].data + i*width*2, width*elemSize);
                }
                s->output.uploadTile(frame->ssrc);

                vf_free(frame);

                now = chrono::high_resolution_clock::now();

                //If it's time to send next frame downlaod it from gpu,
                //convert to UYVY and send
                if (now >= s->next_frame){
                        cv::Mat result = s->output.getImg();
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
                        fprintf(stderr, "Mosaic: queue full!\n");
                        //vf_free(frame);
                }
                if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
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


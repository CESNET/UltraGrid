/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "anonymize_utils.h"
#include "pixelization.h"

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"

#include "video.h"
#include "video_codec.h"
#include "utils/lock_guard.h"

#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <condition_variable>
#include <time.h>
#include <stdio.h>

//may change to defoult values and use 
#define NUMBER_OF_THREADS_USING 8    //in future will be probably changed
#define NUMBER_OF_ROWS 2                    //number of threads have to be dividible by number of rows, or big problems will ocure
                                            //and 1 / (thread / rows) should'n be with infinite decimal pleace (example 1 / (6/2))
                                            //best threads, rows and thread / rows should be power of 2
#define OVERLAPING_RECOGNIZE 30             //how many pixels are overlaping in recognition

using namespace std;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_anonymize{
public:
    state_anonymize(const char *);
    ~state_anonymize();
    typedef vector<word_type> word_ret;
    struct video_frame *filter(struct video_frame *in);
    void pixelate(std::vector<word_type>* words, struct video_frame *in);
private:
    double compare_value_to_be_same;
    screen_package screens;
    video_frame_work_queue m_wqueue;
    video_frame_process_worker working_thread;
    thread video_frame_worker_thread;
};

static int init(struct module *parent, const char *cfg, void **state)
{
    if (cfg && strcasecmp(cfg, "help") == 0) {
        printf("anonymize private information, still only testing version:\n\n");
        printf("anonymize usage:\n");

        return 1;
    }

    struct state_anonymize *s = new state_anonymize(cfg);
    assert(s);
    

    if (cfg) {
        char *tmp = strdup(cfg);
        bool ret = true;//parse_configuration(s, tmp);
        free(tmp);
        if (!ret) {
            delete s;
            return -1;
        }
    }
    /*
    module_init_default(&s->mod);
    s->mod.cls = MODULE_CLASS_DATA;
    module_register(&s->mod, parent);
    */
    *state = s;
    return 0;
}

static void done(void *){
    
}

/*static struct video_frame *filter(void *, struct video_frame *in)
{
    if (in->color_spec != UYVY) {
            log_msg(LOG_LEVEL_WARNING, "Cannot create anonymize from other codec than UYVY!\n");
            return in;
    }
    struct video_frame *out = vf_alloc_desc_data(video_desc_from_frame(in));
    out->dispose = vf_free;

    unsigned char *in_data = (unsigned char *) in->tiles[0].data;
    unsigned char *out_data = (unsigned char *) out->tiles[0].data;

    for (unsigned int i = 0; i < in->tiles[0].width * in->tiles[0].height; ++i) {
            *out_data++ = 127;
            in_data++;
            *out_data++ = *in_data++;
    }

    VIDEO_FRAME_DISPOSE(in);

    return out;
}*/

state_anonymize::state_anonymize(const char *cfg):screens(0.97),m_wqueue(screens),working_thread(m_wqueue, screens)
{  
    compare_value_to_be_same = 0.97;    //first shot to this value

    video_frame_worker_thread = thread(&video_frame_process_worker::thread_function, &working_thread);
}

state_anonymize::~state_anonymize()
{
    working_thread.stop();
}

struct video_frame * state_anonymize::filter(struct video_frame *in){
    clock_t time1, time2, time3, time4;
    bool frame_is_uyvy = false;
    bool frame_is_rgb = false;
    time1 = clock();
    
    struct video_frame * in_rgb = new struct video_frame();
    in_rgb->tiles = new tile();
    in_rgb->tiles[0].height = in->tiles[0].height;
    in_rgb->tiles[0].width = in->tiles[0].width;
    in_rgb->tiles[0].offset = in->tiles[0].offset;
    in_rgb->color_spec = codec_t::RGB;
    in_rgb->fps = in->fps;
    
    if(in->color_spec == UYVY){
        frame_is_uyvy = true;
        
        in_rgb->tiles[0].data = (char *) malloc((in->tiles[0].data_len / 2)*3);
        
        time3 = clock();
        vc_copylineUYVYtoRGB_SSE((unsigned char *) in_rgb->tiles[0].data, (unsigned char *) in->tiles[0].data, (in->tiles[0].data_len / 2)*3);
        time4 = clock();
        //cout << "UYVY -> RGB " << (double)(time4 - time3)/CLOCKS_PER_SEC << " seconds" << endl;
    }else if(in->color_spec == RGB){
        frame_is_rgb = true;
        
    in_rgb->tiles[0].data = (char *) malloc((in->tiles[0].data_len / 2)*3);
        time3 = clock();
        vc_copylineRGB((unsigned char *) in_rgb->tiles[0].data, (unsigned char *) in->tiles[0].data, in->tiles[0].data_len, 0, 8, 16);
        time4 = clock();
        //cout << "UYVY -> RGB " << (double)(time4 - time3)/CLOCKS_PER_SEC << " seconds" << endl;
    }else{
        log_msg(LOG_LEVEL_WARNING, "Cannot create anonymize from other codec than UYVY!\n");
        string out_msg =  "This codec is ";
        out_msg.append(to_string(in->color_spec)).append("\n");
        log_msg(LOG_LEVEL_WARNING,out_msg.c_str());
        return in;
    }
    
    time3 = clock();
    cv::Mat new_histogram = screen::create_histogram(in_rgb);
    time4 = clock();
    //cout << "time to COMPUTE histograms " << (double)(time4 - time3)/CLOCKS_PER_SEC << " seconds" << endl;
    
    
    vector<word_type>* words = screens.get_boxes(new_histogram);
    if(words != NULL){      //
        //cout << "should pixelate something" << endl;
        pixelate(words, in);
        screen::free_video_frame(in_rgb);
    }else{
        m_wqueue.add(in_rgb, new_histogram);
        //cout << "nothing to pixelate" << endl;
    }
    
    time2 = clock();
    //cout << "time for 1 frame: " << (double)(time2 - time1)/CLOCKS_PER_SEC << " seconds" << endl;
    //cout << "----------------------------------------------------------------" << endl;
    return in;
}

void state_anonymize::pixelate(std::vector<word_type>* words, video_frame* in){
    if(words == NULL){
        return;
    }
    for(int i=0;i<words->size();i++){
        state_pixelization pixelization;
        pixelization.x = words->at(i).m_x;
        pixelization.y = words->at(i).m_y;
        pixelization.width = words->at(i).m_width;
        pixelization.height = words->at(i).m_height;
        pixelization.pixelSize = 30;
        pixelization.in_relative_units = false;
        in = pixelization.filter(in);
    }
    return;
}

static bool parse_configuration(struct state_blank *s, char *cfg)
{
    return true;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    return ((state_anonymize *) state)->filter(in);
}

static const struct capture_filter_info capture_filter_anonymize = {
    .init = init,
    .done = done,
    .filter = filter,
};

REGISTER_MODULE(anonymize, &capture_filter_anonymize, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

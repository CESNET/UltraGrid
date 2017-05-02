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

#include <map>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <stdio.h>

using namespace std;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_anonymize{
public:
    state_anonymize();
    ~state_anonymize();
    bool set_configuration(const char *);
    bool start();
    typedef vector<word_type> word_ret;
    struct video_frame *filter(struct video_frame *in);
    void pixelate(std::vector<word_type>* words, struct video_frame *in);
    static std::vector<std::string> split_string(const std::string s, std::string delim);
    bool process_command(const std::string setting, const std::string value);
    void set_defoult_setting();
    bool str_to_bool(std::string, bool &);
    bool str_to_int(std::string, int &);
    bool str_to_float(std::string, float &);
    bool save();
    bool load();
private:
    screen_package * m_screens;
    video_frame_work_queue * m_wqueue;        
    video_frame_process_worker * m_working_thread;
    thread m_video_frame_worker_thread;
    
    configuration_values m_config;
};



static int init(struct module *parent, const char *cfg, void **state)
{
    if (cfg && strcasecmp(cfg, "help") == 0) {
        printf("anonymize private information, still only testing version:\n\n");
        printf("anonymize usage:\n");
        printf("anonymize:[options=value]\n\n");
        printf("options:\n");
        printf("save; value:bool\n");
        printf("load; value:bool\n");
        printf("pixelization_size or ps; value:int\n");
        printf("scale_size or ssr; value:float\n");
        printf("shrink_size or ssh; value:float\n");
        printf("save_path; value:string\n");
        printf("load_path; value:string\n");
        printf("image_path; value:string\n");
        printf("recognizing_threads or rt or threads; value:int\n");
        printf("debug_info or d; value:int\n");
        printf("screen_compare_value or scp; value:float\n");
        printf("recognize_new_screen or rs; value:bool\n");
        printf("save_new_screen_as_image or ssi; value:bool\n");
        printf("error_messages; value:bool\n");
        printf("tesseract_language_name; value:string\n");
        printf("anonymize_unrecognized or au; value:bool");
        return 1;
    }

    struct state_anonymize *s = new state_anonymize();
    assert(s);
    bool config_ok = s->set_configuration(cfg);
    if(!config_ok){
        return 1;
    }
    config_ok = s->start();
    if(!config_ok){
        return 1;
    }
    s->load();
    
    *state = s;
    return 0;
}

static void done(void * state){
    struct state_anonymize *s = (struct state_anonymize *) state;
    s->save();
    delete s;
}

state_anonymize::state_anonymize(){

}

bool state_anonymize::set_configuration(const char *cfg_in){
    //set default values, change it if configuration say so 
    set_defoult_setting();
    if(cfg_in != NULL){//configuration is not empty, parse configuration
        string cfg = std::string(cfg_in);
        
        vector<std::string> config_commands = split_string(cfg, ":");
        for(int i=0;i<config_commands.size();i++){
            string command = config_commands[i];
            vector<std::string> command_split = split_string(command, "=");
            if(command_split.size() == 2){
                bool ret = process_command(command_split[0], command_split[1]);
                if(!ret){
                    if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                        cerr << "couldn't process command: \"" << command_split[0] << "\" with value: \"" << command_split[1] << "\"" << endl;
                    }
                }
            }else{
                if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                    cerr << "wrong anonymize setting command \"" << command << "\"" << endl;
                    cerr << "command should look like: \"setting=value\"" << endl;
                }
                return false;
            }
        }
    }else{//configuration is empty, default setting is set et the beginning
        cout << "defoult configuration" << endl;
    }
    return true;
}

bool state_anonymize::start(){
    m_screens = new screen_package(m_config.compare_value_to_be_same);
    if(m_screens == NULL){
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            cerr << "anonymize couldn't create screen_package class" << endl;
        }
        return false;
    }
    m_wqueue = new video_frame_work_queue(m_screens);
    if(m_wqueue == NULL){
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            cerr << "anonymize couldn't create video_frame_work_queue class" << endl;
        }
        return false;
    }
    m_working_thread = new video_frame_process_worker(m_wqueue, m_screens, m_config);
    if(m_working_thread == NULL){
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            cerr << "anonymize couldn't create video_frame_process_worker class" << endl;
        }
        return false;
    }
    
    m_video_frame_worker_thread = thread(&video_frame_process_worker::thread_function, m_working_thread);
    return true;
}

bool state_anonymize::process_command(const std::string setting, const std::string value){
    if(setting.compare("save") == 0){           //if should save to file learned screens and boxes on them
        bool correct_value = str_to_bool(value, m_config.save);
        if(!correct_value){
            return false;
        }
    }else if(setting.compare("load") == 0){     //if should load from file screens and boxes on them so don't have to process screens again
        bool correct_value = str_to_bool(value, m_config.load);
        if(!correct_value){
            return false;
        }
    }else if((setting.compare("pixelization_size") == 0) || (setting.compare("ps") == 0)){  //size of pixels in boxes after pixelization
        bool correct_value = str_to_int(value, m_config.pixelization_size);
        if(!correct_value){
            return false;
        }
        if(m_config.pixelization_size > MAX_PIXELIZATION_SIZE){
            m_config.pixelization_size = MAX_PIXELIZATION_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "pixelization_size value too big, set to maximum value " << MAX_PIXELIZATION_SIZE << endl;
            }
        }else if(m_config.pixelization_size < MIN_PIXELIZATION_SIZE){
            m_config.pixelization_size = MIN_PIXELIZATION_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "pixelization_size value too small, set to minimum value " << MIN_PIXELIZATION_SIZE << endl;
            }
        }
    }else if((setting.compare("scale_size") == 0) || (setting.compare( "ssr") == 0)){ //how much to scale image before recognizing
        bool correct_value = str_to_float(value, m_config.scale_size);
        if(!correct_value){
            return false;
        }
        if(m_config.scale_size > MAX_SCALE_SIZE){
            m_config.scale_size = MAX_SCALE_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "scale_size value too big, set to maximum value " << MAX_SCALE_SIZE << endl;
            }
        }else if(m_config.scale_size < MIN_SCALE_SIZE){
            m_config.scale_size = MIN_SCALE_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "scale_size value too small, set to minimum value " << MIN_SCALE_SIZE << endl;
            }
        }
    }else if((setting.compare("shrink_size") == 0) || (setting.compare( "ssh") == 0)){ //how much to shrink image before creating 
        bool correct_value = str_to_float(value, m_config.shrink_size);   //TO DO: rename variabile
        if(!correct_value){
            return false;
        }
        if(m_config.shrink_size > MAX_SHRINK_SIZE){
            m_config.shrink_size = MAX_SHRINK_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "shrink_size value too big, set to maximum value " << MAX_SHRINK_SIZE << endl;
            }
        }else if(m_config.shrink_size < MIN_SHRINK_SIZE){
            m_config.shrink_size = MIN_SHRINK_SIZE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "shrink_size value too small, set to minimum value " << MIN_SHRINK_SIZE << endl;
            }
        }
    }else if(setting.compare("save_path") == 0){    //path to file where to save screens
        m_config.save_path = value;
        return helping_functions::can_file_exists_test(value.c_str());
    }else if(setting.compare("load_path") == 0){    //path to file from program should load screens
        m_config.load_path = value;
        return helping_functions::file_exists_test(value);
    }else if(setting.compare("image_path") == 0){   //path to folder where image of saved screens should be saved
        m_config.image_path = value;
        return helping_functions::directory_exists_test(value);
    }else if((setting.compare("recognizing_threads") == 0) || (setting.compare("rt") == 0) || (setting.compare("threads") == 0)){   //number of threads for recognizing
        bool correct_value = str_to_int(value, m_config.num_recognizing_threads);
        if(!correct_value){
            return false;
        }
        if(m_config.num_recognizing_threads > MAX_RECOGNIZING_THREAD){
            m_config.num_recognizing_threads = MAX_RECOGNIZING_THREAD;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "recognizing_threads value too big, set to maximum value " << MAX_RECOGNIZING_THREAD << endl;
            }
        }else if(m_config.num_recognizing_threads < MIN_RECOGNIZING_THREAD){
            m_config.num_recognizing_threads = MIN_RECOGNIZING_THREAD;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "recognizing_threads value too small, set to minimum value " << MIN_RECOGNIZING_THREAD << endl;
            }
        }
    }else if((setting.compare("debug_info") == 0) || (setting.compare( "d") == 0)){  //if debug messages should be printed out, bigger number more of them, or different values print different things
        int debug_info;
        bool correct_value = str_to_int(value, debug_info);
        if(!correct_value){
            return false;
        }
        if(debug_info > MAX_DEBUG_INFO){
            return false;
        }else if(debug_info < MIN_DEBUG_INFO){
            return false;
        }
        print_message_struct::set_messages_setting(debug_info);
    }else if((setting.compare("screen_compare_value") == 0) || (setting.compare("scp") == 0)){//print big message about if someone change this and save should it change when load
        float tmp_value;
        bool correct_value = str_to_float(value, tmp_value);
        if(!correct_value){
            return false;
        }
        m_config.compare_value_to_be_same = tmp_value;
        if(m_config.compare_value_to_be_same >= MAX_COMPARE_VALUE){
            m_config.compare_value_to_be_same = MAX_COMPARE_VALUE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "screen_compare_value value too big, set to maximum value " << MAX_COMPARE_VALUE << endl;
            }
        }else if(m_config.compare_value_to_be_same <= MIN_COMPARE_VALUE){
            m_config.compare_value_to_be_same = MIN_COMPARE_VALUE;
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "screen_compare_value value too small, set to minimum value " << MIN_COMPARE_VALUE << endl;
            }
        }
    }else if((setting.compare("recognize_new_screen") == 0) || (setting.compare("rs") == 0)){
        bool correct_value = str_to_bool(value, m_config.recognize_new_screens);
        if(!correct_value){
            return false;
        }
    }else if((setting.compare("save_new_screen_as_image") == 0) || (setting.compare("ssi") == 0)){
        bool correct_value = str_to_bool(value, m_config.save_new_screen);
        if(!correct_value){
            return false;
        }
    }else if(setting.compare("error_messages") == 0){   //option if error messages should show
        bool error_message;
        bool correct_value = str_to_bool(value, error_message);
        if(!correct_value){
            return false;
        }
        print_message_struct::set_error_messages_setting(error_message);
    }else if((setting.compare("tesseract_language_name") == 0) || (setting.compare("language") == 0) || (setting.compare("l") == 0)){
        m_config.tesseract_language_name = value;
        return true;
    }else if((setting.compare("anonymize_unrecognized") == 0) || (setting.compare("au") == 0)){
        bool correct_value = str_to_bool(value, m_config.anonymize_unrecognized);
        if(!correct_value){
            return false;
        }
    }else if(setting.compare("") == 0){//explicit show/ explicit deny ???????????????
        
    }else if(setting.compare("") == 0){//explicit show/ explicit deny ???????????????
        
    }
}

void state_anonymize::set_defoult_setting(){
    m_config.save = false;
    m_config.load = false;   
    m_config.pixelization_size = 20;
    m_config.scale_size = 2.0;
    m_config.shrink_size = 2.0;
    m_config.save_path = "./ultragrid_anonymize_data.json";
    m_config.load_path = "./ultragrid_anonymize_data.json";
    m_config.image_path = "./";
    m_config.num_recognizing_threads = 8;
    m_config.compare_value_to_be_same = 0.97;
    m_config.recognize_new_screens = true;
    m_config.save_new_screen = false;
    print_message_struct::set_error_messages_setting(true);
    m_config.tesseract_language_name = "uga";
    m_config.anonymize_unrecognized = false;
}


bool state_anonymize::str_to_bool(std::string str, bool & ret){
    if((str.compare("0") == 0) || (str.compare("false") == 0)){
        ret = false;
        return true;
    }
    if((str.compare("1") == 0) || (str.compare("true") == 0)){
        ret = true;
        return true;
    }
    return false;
}

bool state_anonymize::str_to_int(std::string str, int & ret){
    ret = atoi(str.c_str());
    if(ret == 0){   //error could happened
        if(str.compare("0") != 0){  //error occured
            return false;
        }
    }
    return true;
}

bool state_anonymize::str_to_float(std::string str, float & ret){
    ret = atof(str.c_str());
    if(ret == 0.0){   //error could happened
        if(str.find("0.0") == str.npos){  //error occured
            return false;
        }
    }
    return true;
}

bool state_anonymize::save(){
    if(m_config.save){
        boost::property_tree::ptree main_tree;
        boost::property_tree::ptree screens_tree;

        for(int k=0;k<m_screens->size();k++){
            std::pair<cv::Mat, word_vector> screen = m_screens->get_screen(k);
            string screen_name = m_screens->get_screen_name(k);
            cv::Mat histogram = screen.first;
            word_vector lists_of_boxes = screen.second;
            cv::Mat histogram_int = histogram_coding::integer_coding(histogram);
            std::string histogram_string = histogram_coding::mat_to_string_coding(histogram_int);

            boost::property_tree::ptree screen_tree;
            boost::property_tree::ptree boxes_tree;
            boost::property_tree::ptree box_tree;

            for(int j=0;j<lists_of_boxes.size();j++){
                box_tree.put("confidence", lists_of_boxes.at(j).m_confidence);
                box_tree.put("height", lists_of_boxes.at(j).m_height);
                box_tree.put("width", lists_of_boxes.at(j).m_width);
                box_tree.put("x", lists_of_boxes.at(j).m_x);
                box_tree.put("y", lists_of_boxes.at(j).m_y);
                box_tree.put("text", lists_of_boxes.at(j).m_str);
                box_tree.put("index", lists_of_boxes.at(j).m_index);
                box_tree.put("type", lists_of_boxes.at(j).m_type);
                boxes_tree.push_back(std::make_pair("", box_tree));
            }

            screen_tree.put("screen_name", screen_name);
            screen_tree.add_child("boxes", boxes_tree);
            screen_tree.put("histogram", histogram_string);

            screens_tree.push_back(std::make_pair("", screen_tree));
        }
        main_tree.add_child("screens", screens_tree);

        boost::property_tree::write_json(m_config.save_path, main_tree);
    }
}

bool state_anonymize::load(){
    if(m_config.load){
        boost::property_tree::ptree main_tree;
        boost::property_tree::read_json(m_config.load_path, main_tree);
        
        boost::property_tree::ptree screens_tree;
        
        for(auto& screen: main_tree.get_child("screens")){
            word_vector lists_of_boxes;
            cv::Mat histogram, histogram_int;
            std::string screen_name;
            
            string histogram_string = screen.second.get_child("histogram").get_value<std::string>();
            histogram_int = histogram_coding::string_to_mat_decoding(histogram_string);
            histogram = histogram_coding::integer_decoding(histogram_int);
            screen_name = screen.second.get_child("screen_name").get_value<std::string>();
            for(auto& box_tree: screen.second.get_child("boxes")){                

                int confidence = box_tree.second.get_child("confidence").get_value<int>();
                int height = box_tree.second.get_child("height").get_value<int>();
                int width = box_tree.second.get_child("width").get_value<int>();
                int x = box_tree.second.get_child("x").get_value<int>();
                int y = box_tree.second.get_child("y").get_value<int>();
                std::string str = box_tree.second.get_child("text").get_value<std::string>();
                int index = box_tree.second.get_child("index").get_value<int>();
                string type = box_tree.second.get_child("type").get_value<std::string>();
                word_type box(x, y, width, height, index, confidence, string(str));
                box.set_type(type);
                lists_of_boxes.push_back(box);
            }
            m_screens->save_screen(histogram, lists_of_boxes, screen_name);
        }
    }
}

std::vector<std::string> state_anonymize::split_string(const std::string str, std::string delim) {
    std::vector<std::string> elems;
        char* cstr=const_cast<char*>(str.c_str());
        char* current;
        current=strtok(cstr,delim.c_str());
        while(current!=NULL){
            elems.push_back(current);
            current=strtok(NULL,delim.c_str());
        }
    return elems;
}

state_anonymize::~state_anonymize(){
    m_working_thread->stop();
    m_video_frame_worker_thread.join();
    delete(m_working_thread);
    delete(m_wqueue);
    delete(m_screens);
}

struct video_frame * state_anonymize::filter(struct video_frame *in){
    //clock_t time1, time2, time3, time4;
    //clock_t t_frame_start, t_frame_end;
    chrono::time_point<std::chrono::system_clock> ttime1, ttime2;
    chrono::time_point<std::chrono::system_clock> tt_frame_start, tt_frame_end;
    std::chrono::duration<double> elapsed_seconds;
    bool frame_is_uyvy = false;
    bool frame_is_rgb = false;
    //t_frame_start = clock();
    tt_frame_start = chrono::system_clock::now();

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
        
        //time3 = clock();
        ttime1 = chrono::system_clock::now();
        vc_copylineUYVYtoRGB_SSE((unsigned char *) in_rgb->tiles[0].data, (unsigned char *) in->tiles[0].data, (in->tiles[0].data_len / 2)*3);
        //time4 = clock();
        ttime2 = chrono::system_clock::now();
        if(print_message_struct::print_message(print_message_struct::Message_type::time_frame_to_RGB)){
            elapsed_seconds = ttime2 - ttime1;
            cout << "frame preprocess: UYVY -> RGB  " << elapsed_seconds.count() << " seconds" << endl;
        }
    }else if(in->color_spec == RGB){
        frame_is_rgb = true;
        
        in_rgb->tiles[0].data = (char *) malloc((in->tiles[0].data_len / 2)*3);
        //time3 = clock();
        ttime1 = chrono::system_clock::now();
        vc_copylineRGB((unsigned char *) in_rgb->tiles[0].data, (unsigned char *) in->tiles[0].data, in->tiles[0].data_len, 0, 8, 16);
        //time4 = clock();
        ttime2 = chrono::system_clock::now();
        if(print_message_struct::print_message(print_message_struct::Message_type::time_frame_to_RGB)){
            elapsed_seconds = ttime2 - ttime1;
            cout << "frame preprocess: RGB -> RGB " << elapsed_seconds.count() << " seconds" << endl;
        }
    }else{
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            log_msg(LOG_LEVEL_WARNING, "Cannot create anonymize from other codec than UYVY!\n");
            string out_msg =  "This codec is ";
            out_msg.append(to_string(in->color_spec)).append("\n");
            log_msg(LOG_LEVEL_WARNING,out_msg.c_str());
        }
        return in;
    }
    
    if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_more)){
        cout << "calculating histogram" << endl;
    }
    //time1 = clock();
    ttime1 = chrono::system_clock::now();
    cv::Mat new_histogram = screen::create_histogram(in_rgb, m_config.shrink_size);
    //time2 = clock();
    ttime2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::histogram_time)){
        elapsed_seconds = ttime2 - ttime1;
        cout << "time to COMPUTE histograms " << elapsed_seconds.count() << " seconds" << endl;
    }
    
    //time1 = clock();
    ttime1 = chrono::system_clock::now();
    vector<word_type>* words = m_screens->get_boxes(new_histogram);
    //time2 = clock();
    ttime2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::histogram_time)){
        elapsed_seconds = ttime2 - ttime1;
        cout << "time to FIND words associated with histogram: " << elapsed_seconds.count() << " seconds, number of screens saved: " << m_screens->size() << endl;
    }
    
    if(words != NULL){
        ttime1 = chrono::system_clock::now();
        pixelate(words, in);
        ttime2 = chrono::system_clock::now();
        if(print_message_struct::print_message(print_message_struct::Message_type::time_pixelate)){
            elapsed_seconds = ttime2 - ttime1;
            cout << "time to pixelate " << words->size() << "words: " << elapsed_seconds.count() << " seconds" << endl;
        }
        screen::free_video_frame(in_rgb);
    }else{
        if(m_config.recognize_new_screens){
            if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_more)){
                cout << "unknown screen, adding it for process" << endl;
            }
            m_wqueue->add(in_rgb, new_histogram);
        }
        if(m_config.anonymize_unrecognized){
            vector<word_type> tmp_word;
            tmp_word.push_back(word_type(0, 0, in->tiles->width, in->tiles->height, 100, ""));
            ttime1 = chrono::system_clock::now();
            pixelate(&tmp_word, in);
            ttime2 = chrono::system_clock::now();
            if(print_message_struct::print_message(print_message_struct::Message_type::time_pixelate)){
                elapsed_seconds = ttime2 - ttime1;
                cout << "time to pixelate unrecognized screen: " << elapsed_seconds.count() << " seconds" << endl;
            }
        }
    }
    
    //t_frame_end = clock();
    tt_frame_end = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::frame_time)){
        elapsed_seconds = tt_frame_end - tt_frame_start;
        cout << "time for frame: " << elapsed_seconds.count() << " seconds" << endl;
        cout << "----------------------------------------------------------------" << endl;
    }
    return in;
}

void state_anonymize::pixelate(std::vector<word_type>* words, video_frame* in){
    if(words == NULL){
        return;
    }
    int pixelated_boxes = 0;
    for(int i=0;i<words->size();i++){
        state_pixelization pixelization;
        pixelization.x = words->at(i).m_x;
        pixelization.y = words->at(i).m_y;
        pixelization.width = words->at(i).m_width;
        pixelization.height = words->at(i).m_height;
        pixelization.pixelSize = m_config.pixelization_size;
        pixelization.in_relative_units = false;
        in = pixelization.filter(in);
        pixelated_boxes++;
    }
    if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_some)){
        cout << "pixelated " <<  pixelated_boxes << " boxes" << endl;
    }
    return;
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

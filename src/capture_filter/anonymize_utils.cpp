/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


#include <iostream>
#include <c++/5/mutex>
#include <c++/5/condition_variable>
#include <c++/5/thread>
#include <time.h>
#include "anonymize_utils.h"

using namespace std;

set<int> print_message_struct::messages_setting;
bool print_message_struct::error_messages;

//TO DO: delete before release
//print dimensions of matrix
void debug_mat(cv::Mat mat){
      string r;

    uchar depth = mat.type() & CV_MAT_DEPTH_MASK;
    int chans = 1 + (mat.type() >> CV_CN_SHIFT);

    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }


    cout << r << "C" << chans << " --- " << mat.cols << "x" << mat.rows << endl;
}

bool print_message_struct::print_message(Message_type message_type){
    
    if(message_type == Message_type::forced){
        return true;
    }else if(message_type == Message_type::error){
        return error_messages;
    }else if(message_type == Message_type::frame_time){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 1){
                return true;
            }
        }
    }else if(message_type == Message_type::histogram_time){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 2){
                return true;
            }
        }
    }else if(message_type == Message_type::time_pixelate){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 3){
                return true;
            }
        }
    }else if(message_type == Message_type::time_frame_preprocess){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 4){
                return true;
            }
        }
    }else if(message_type == Message_type::time_tesseract_OCR){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 5){
                return true;
            }
        }
    }else if(message_type == Message_type::time_semantic_analysis){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 6){
                return true;
            }
        }
    }else if(message_type == Message_type::time_frame_to_RGB){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 7){
                return true;
            }
        }
    }else if(message_type == Message_type::histogram_comparation){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 8){
                return true;
            }
        }
    }else if(message_type == Message_type::order_of_action_some){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if((current_message_setting == 9) || (current_message_setting == 10)){
                return true;
            }
        }
    }else if(message_type == Message_type::order_of_action_more){
        for(auto message_setting=messages_setting.begin(); message_setting!=messages_setting.end();message_setting++){
            int current_message_setting = (*message_setting);
            if(current_message_setting == 10){
                return true;
            }
        }            
    }
    return false;
}

//bool print_message_struct::print_message(Message_type message_type, int thread){
//    return false;
//}

double screen::compare(cv::Mat obj1, cv::Mat obj2, int compare_method){
    return cv::compareHist(obj1, obj2, compare_method);
}

double screen::compare(cv::Mat obj1, cv::Mat obj2){
    return compare(obj1, obj2, 0);
}

cv::Mat screen::create_histogram(struct video_frame *in, float shrink_coeficient){
    
    int width  = in->tiles[0].width;
    int height = in->tiles[0].height;
    cv::Mat picture = cv::Mat(height, width, CV_8UC3,  in->tiles[0].data);
    cv::Mat resize_picture;
    cv::resize(picture, resize_picture, cv::Size(width/shrink_coeficient, height/shrink_coeficient), 0, 0, cv::INTER_LINEAR);
    
    const int histSize[] = {256};
    int channel_red[] = { 0 };
    int channel_grean[] = { 1 };
    int channel_blue[] = { 2 };
    float range[] = {0,256};
    const float *histRange = {range};

    cv::Mat histogram_red;
    cv::Mat histogram_green;
    cv::Mat histogram_blue;
    cv::calcHist(&resize_picture, 1, channel_red, cv::Mat(), histogram_red, 1, histSize, &histRange, true, false);
    cv::calcHist(&resize_picture, 1, channel_grean, cv::Mat(), histogram_green, 1, histSize, &histRange, true, false);
    cv::calcHist(&resize_picture, 1, channel_blue, cv::Mat(), histogram_blue, 1, histSize, &histRange, true, false);
    
    
    vector<cv::Mat> combinator;
    combinator.push_back(histogram_red);
    combinator.push_back(histogram_green);
    combinator.push_back(histogram_blue);
    
    cv::Mat combined_histogramtogramtogram;
    
    hconcat( combinator, combined_histogramtogramtogram );
    //cv::vconcat( combinator, combined_histogramtogramtogram );
    
    cv::normalize(combined_histogramtogramtogram, combined_histogramtogramtogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    
    return combined_histogramtogramtogram;
}

void screen::free_video_frame(struct video_frame *in){
        free(in->tiles->data);
        delete in->tiles;
        delete in;
}

screen_package::screen_package(double threshold):m_histogram_compare_threshold(threshold){
    
}

void screen_package::save_screen(cv::Mat hist, std::vector<word_type> words){
    m_mutex.lock();
    m_histograms.push_back(hist);
    m_lists_of_boxes.push_back(words);
    string name = "screen_";
    name.append(to_string(m_histograms.size()));
    m_screen_names.push_back(name);
    m_mutex.unlock();
}

void screen_package::save_screen(cv::Mat hist, std::vector<word_type> words, std::string name){
    m_mutex.lock();
    m_histograms.push_back(hist);
    m_lists_of_boxes.push_back(words);
    m_screen_names.push_back(name);
    m_mutex.unlock();
}

std::vector<word_type>* screen_package::get_boxes(cv::Mat hist){
    std::vector<word_type>* ret = NULL;
    m_mutex.lock();
    for(int i=m_histograms.size()-1;i>=0;i--){
        double comparation_value = screen::compare(hist, m_histograms.at(i));
        if(print_message_struct::print_message(print_message_struct::Message_type::histogram_comparation)){
            cout << "histogram diff " << comparation_value << " compare to " << m_histogram_compare_threshold << " was comparing to screen " << i << endl;
        }
        if(comparation_value > m_histogram_compare_threshold){
            ret = &(m_lists_of_boxes.at(i));
            break;
        }
    }
    m_mutex.unlock();
    return ret;
}

unsigned int screen_package::size(){
    m_mutex.lock();
    int size = m_histograms.size();
    m_mutex.unlock();
    return size;
}

void screen_package::set_histogram_compare_threshold(double new_value){
    m_mutex.lock();
    m_histogram_compare_threshold = new_value;
    m_mutex.unlock();
}

double screen_package::get_histogram_compare_threshold(){
    m_mutex.lock();
    double histogram_threshold = m_histogram_compare_threshold;
    m_mutex.unlock();
    return histogram_threshold;
}

std::pair<cv::Mat, word_vector> screen_package::get_screen(int position){
    std::pair<cv::Mat, word_vector> ret;
    m_mutex.lock();
    ret.first = m_histograms.at(position);
    ret.second = m_lists_of_boxes.at(position);
    m_mutex.unlock();
    return ret;
}

std::string screen_package::get_screen_name(int position){
    string ret;
    m_mutex.lock();
    ret = m_screen_names.at(position);
    m_mutex.unlock();
    return ret;
}

video_frame_work_queue::video_frame_work_queue(screen_package * screens):m_screens(screens) {
    still_running = true;
}

video_frame_work_queue::~video_frame_work_queue(){
    
}

void video_frame_work_queue::set_screen_package(screen_package * screens){
    m_screens = screens;
}

int video_frame_work_queue::size(){
    m_mutex.lock();
    int size;
    size = m_queue_frame.size();
    m_mutex.unlock();
    return size;
}

void video_frame_work_queue::add(struct video_frame * item, cv::Mat histogram){
    m_mutex.lock();
    for(cv::Mat & saved_hisogram : m_queue_histogram){
        double treshlold = m_screens->get_histogram_compare_threshold();
        if(screen::compare(saved_hisogram, histogram) >  treshlold){
            screen::free_video_frame(item);
            m_mutex.unlock();
            return ;        //if already have simular screen (screen with simular histogram) don't add this screen to process queue and save memory
        }
    }
    m_queue_frame.push_back(item);
    m_queue_histogram.push_back(histogram);
    m_condv.notify_one();
    m_mutex.unlock();
}

std::pair<struct video_frame *, cv::Mat> video_frame_work_queue::remove(){
    unique_lock<mutex> m_lock(m_mutex);
    while((m_queue_frame.size() == 0) && still_running){
        m_condv.wait(m_lock);
    }
    if(still_running){
        struct video_frame * item = m_queue_frame.front();
        cv::Mat histogram = m_queue_histogram.front();
        m_queue_frame.pop_front();
        m_queue_histogram.pop_front();
        return std::pair<struct video_frame *, cv::Mat>(item, histogram);
    }else{
        return std::pair<struct video_frame *, cv::Mat>(NULL, cv::Mat());
    }
}

void video_frame_work_queue::end_queue(){
    still_running = false;
    m_condv.notify_all();
}

video_frame_process_worker::video_frame_process_worker(video_frame_work_queue * queue, screen_package * screens, configuration_values config)
                                        :m_queue(queue), m_screens(screens){
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);
    m_still_running = true;
   
    m_config = config;
    m_tesseract_utils.set_configuration(config);
    m_tesseract_utils.init();
    
    m_tesseract_utils.set_year(aTime->tm_year + 1900);
    m_tesseract_utils.set_month(aTime->tm_mon + 1);
    m_tesseract_utils.set_day(aTime->tm_mday);
    
}

video_frame_process_worker::~video_frame_process_worker(){
    
}

//should be only one thread, more threads could calculate simular picture twice unnecessarily
void video_frame_process_worker::thread_function(){
    // Remove 1 item at a time and process it. Blocks if no items are available to process.
    while(m_still_running) {
        std::pair<struct video_frame *, cv::Mat> saved_information_about_screen = (std::pair<struct video_frame *, cv::Mat>)m_queue->remove();
        struct video_frame * frame = saved_information_about_screen.first;
        if(frame == NULL){
            continue;
        }
        cv::Mat histogram = saved_information_about_screen.second;
        if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_some)){
            cout << "processing screen" << endl;
        }
        
        vector<word_type>* words = m_screens->get_boxes(histogram);
        if(words != NULL){  //check if this or simular frame was calculated before, can happened regularly
            continue;
        }
        //calculate tesseract boxes
        if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_some)){
            cout << "generating boxes" << endl;
        }
        word_vector tesseract_boxes = m_tesseract_utils.find_boxes(frame);
        
        /*for(int i=0;i<tesseract_boxes.size()/10;i++){
            cout << tesseract_boxes.at(i).m_index << " " << tesseract_boxes.at(i).m_width << "x" << tesseract_boxes.at(i).m_height << " " << tesseract_boxes.at(i).m_x << "x" << tesseract_boxes.at(i).m_y << " " << tesseract_boxes.at(i).m_str << endl;
        }*/
        if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_some)){
            cout << "saving #" << tesseract_boxes.size() << " boxes" << endl;
        }
        m_screens->save_screen(histogram, tesseract_boxes);
        if(m_config.save_new_screen){
            if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_some)){
                cout << "saving image" << endl;
            }
            cv::Mat picture_mat = cv::Mat(frame->tiles[0].height, frame->tiles[0].width, CV_8UC3,  frame->tiles[0].data);
            cv::cvtColor(picture_mat, picture_mat, CV_BGR2RGB); //change to rgb picture
            std::string path = m_config.image_path;
            path.append(m_screens->get_screen_name(m_screens->size() - 1)).append(".jpg");
            cv::imwrite(path, picture_mat);
        }
        screen::free_video_frame(frame);
    }
}

void video_frame_process_worker::stop(){
    m_still_running = false;
    m_queue->end_queue();
}

cv::Mat histogram_coding::integer_coding(cv::Mat mat){
    cv::Mat multiply_mat = mat * MULTIPLY_HISTOGRAM_CODING;
    cv::Mat integer_mat;    
    multiply_mat.convertTo(integer_mat, CV_32SC1);
    return integer_mat;
}

cv::Mat histogram_coding::integer_decoding(cv::Mat mat){
    cv::Mat float_mat;    
    mat.convertTo(float_mat, CV_32FC1);
    cv::Mat ret = float_mat / MULTIPLY_HISTOGRAM_CODING;
    return ret;
}

std::string histogram_coding::mat_to_string_coding(cv::Mat mat){
    std::string ret = "";
    ret.append(to_string(mat.rows)).append("x").append(to_string(mat.cols));
    for(int i=0;i<mat.rows;i++){
        for(int j=0;j<mat.cols;j++){
            ret.append(histogram_coding::int_to_character_coding(mat.at<int>(i,j)));
        }
    }
    return ret;
}

cv::Mat histogram_coding::string_to_mat_decoding(std::string str){
    boost::regex regex_num_rows_cols("([0-9]*)x([0-9]*)(.*)");
    boost::smatch result;
    int rows, cols;
    string string_mat_data;
    if(boost::regex_search(str, result, regex_num_rows_cols)){
        rows = stoi(string(result[1]));
        cols = stoi(string(result[2]));
        string_mat_data = string(result[3]);
    }
    vector<std::string> strings = split_string_code(string_mat_data);
    cout << strings.size() << "=" << rows << "x" << cols << endl;
    cv::Mat ret = cv::Mat(rows,cols,CV_32SC1);
    if(strings.size() == rows * cols){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                ret.at<int>(i, j) = char_to_int_coding(strings.at(i*cols+j));
            }
        }
    }
    return ret;
}

vector<std::string> histogram_coding::split_string_code(std::string str){
    vector<std::string> ret;
    string part_str = "";
    for(int i=0;i<str.size();i++){
        part_str += str.at(i);
        if(islower(str.at(i))){
            ret.push_back(part_str);
            part_str.clear();
        }
    }
    return ret;
}

bool histogram_coding::delta_encoding(){
    
}

//TO DO: probably use this one
//with delta coding little smaller, but maybe use some compression, without delta casting is better for compression
//const for multiplying, 1000 good for length, need to test if sufficient for rounding
std::string histogram_coding::int_to_character_coding(unsigned int num){

    int process_num = num;
    std::string ret = "";
    do{
        int tmp = process_num % 25;
        process_num /= 25;
        if(process_num > 0){
            ret += 'A' + tmp;
        }else{
            ret += 'a' + tmp;
        }
    }while(process_num != 0);
    return ret;
}


unsigned int histogram_coding::char_to_int_coding(std::string str){
    unsigned int ret = 0;
    for(int i=0;i<str.size();i++){
        int char_num = 0;
        if(str.at(i) < 'a'){
            char_num = str.at(i) - 'A';
        }else{
            char_num = str.at(i) - 'a';
        }
        ret += char_num * pow(25,i);
    }
    return ret;
}


bool helping_functions::file_exists_test (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
};
    

bool helping_functions::directory_exists_test (const std::string& name){
    DIR* dir = opendir(name.c_str());
    if (dir){/* Directory exists. */
        closedir(dir);
        return true;
    }else{
        return false;
    }
}

bool helping_functions::can_file_exists_test (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "a")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
};
std::vector<std::string> helping_functions::split_string(const std::string str, std::string delim) {
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
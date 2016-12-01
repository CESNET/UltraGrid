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

double screen::compare(cv::Mat obj1, cv::Mat obj2, int compare_method){
    return cv::compareHist(obj1, obj2, compare_method);
}

double screen::compare(cv::Mat obj1, cv::Mat obj2){
    return compare(obj1, obj2, 0);
}

cv::Mat screen::create_histogram(struct video_frame *in){
    int width  = in->tiles[0].width;
    int height = in->tiles[0].height;
    cv::Mat picture = cv::Mat(height, width, CV_8UC3,  in->tiles[0].data);
    cv::Mat resize_picture;
    cv::resize(picture, resize_picture, cv::Size(width/SHRINK_COEFICIENT, height/SHRINK_COEFICIENT), 0, 0, cv::INTER_LINEAR);

    cv::Mat hsv_input(width, height, CV_8UC3,0.0);
    cv::cvtColor(resize_picture, hsv_input, cv::COLOR_RGB2HSV);
    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    const float h_ranges[] = { 0, 180 };
    const float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1, 2 };

    /// Histograms
    cv::Mat hist_input;
    /// Calculate the histograms for the HSV images
    cv::calcHist(&hsv_input, 1, channels, cv::Mat(), hist_input, 2, histSize, ranges, true, false);
    cv::normalize(hist_input, hist_input, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return hist_input;
}

void screen::free_video_frame(struct video_frame *in){
        free(in->tiles->data);
        delete in->tiles;
        delete in;
}

screen_package::screen_package(){
    screen_package(0.97);
}

screen_package::screen_package(double threshold):m_histogram_compare_threshold(threshold){
    
}

void screen_package::save_screen(cv::Mat hist, std::vector<word_type> words){
    m_mutex.lock();
    m_histograms.push_back(hist);
    m_lists_of_boxes.push_back(words);
    m_mutex.unlock();
}

std::vector<word_type>* screen_package::get_boxes(cv::Mat hist){
    std::vector<word_type>* ret = NULL;
    m_mutex.lock();
    for(int i=m_histograms.size()-1;i>=0;i--){
        cout << "histogram diff " << screen::compare(hist, m_histograms.at(i)) << " compare to " << m_histogram_compare_threshold << endl;
        if(screen::compare(hist, m_histograms.at(i)) > m_histogram_compare_threshold){
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

video_frame_work_queue::video_frame_work_queue(screen_package & screens):m_screens(screens) {
    
}

video_frame_work_queue::~video_frame_work_queue(){
    
}

int video_frame_work_queue::size(){
    m_mutex.lock();
    int size = m_queue_frame.size();
    m_mutex.unlock();
    return size;
}

void video_frame_work_queue::add(struct video_frame * item, cv::Mat histogram){
    m_mutex.lock();
    for(cv::Mat & saved_hisogram : m_queue_histogram){
        double treshlold = m_screens.get_histogram_compare_threshold();
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
    m_condv.wait(m_lock, [this](){ return m_queue_frame.size() != 0;});
    //cout << "taking frame out" << endl;
    struct video_frame * item = m_queue_frame.front();
    cv::Mat histogram = m_queue_histogram.front();
    m_queue_frame.pop_front();
    m_queue_histogram.pop_front();
    return std::pair<struct video_frame *, cv::Mat>(item, histogram);
}

video_frame_process_worker::video_frame_process_worker(video_frame_work_queue & queue, screen_package & screens):m_queue(queue), m_screens(screens){
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);

    m_tesseract_utils.set_year(aTime->tm_year + 1900);
    m_tesseract_utils.set_month(aTime->tm_mon + 1);
    m_tesseract_utils.set_day(aTime->tm_mday);
    
}

//should be only one thread, more threads could calculate simular picture twice unnecessarily
void video_frame_process_worker::thread_function(){
    // Remove 1 item at a time and process it. Blocks if no items are available to process.
    while(true) {
        std::pair<struct video_frame *, cv::Mat> saved_information_about_screen = (std::pair<struct video_frame *, cv::Mat>)m_queue.remove();
        struct video_frame * frame = saved_information_about_screen.first;
        cv::Mat histogram = saved_information_about_screen.second;
        cout << "processing screen" << endl;
        
        vector<word_type>* words = m_screens.get_boxes(histogram);
        if(words != NULL){  //check if this or simular frame was calculated before, can happened regularly
            continue;
        }
        //calculate tesseract boxes
        cout << "generating boxes" << endl;
        word_vector tesseract_boxes = m_tesseract_utils.find_boxes(frame);
        
        for(int i=0;i<tesseract_boxes.size()/10;i++){
            cout << tesseract_boxes.at(i).m_index << " " << tesseract_boxes.at(i).m_width << "x" << tesseract_boxes.at(i).m_height << " " << tesseract_boxes.at(i).m_x << "x" << tesseract_boxes.at(i).m_y << " " << tesseract_boxes.at(i).m_str << endl;
        }
        cout << "saving #" << tesseract_boxes.size() << " boxes" << endl;
        m_screens.save_screen(histogram, tesseract_boxes);
        screen::free_video_frame(frame);
    }
}

void video_frame_process_worker::stop(){
    //stop tesseract
    m_tesseract_utils.~tesseract_utils();
}

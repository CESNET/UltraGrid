/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   anonymize_utils.h
 * Author: xminarik
 *
 * Created on September 9, 2016, 6:00 PM
 */

#ifndef ANONYMIZE_UTILS_H
#define ANONYMIZE_UTILS_H

//shrink coeficient to calculate histogram 
#define SHRINK_COEFICIENT 4

//upscale coeficient to calculate tessereact
#define UPSCALE_COEFICIENT 2

//number of threads for tesseract
#define NUMBER_OF_THREADS_USING 8 

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <boost/regex.hpp>

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"

#include "video.h"
#include "video_codec.h"
#include "utils/lock_guard.h"

#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <regex>
#include <iostream>

using namespace std;

struct word_type {
    word_type(int x, int y, int width, int height, int index, int conf, string &&s) :
            m_x(x), m_y(y), m_width(width), m_height(height), m_index(index), m_confidence(conf), m_str(s)
    {
    }
    word_type(int x, int y, int width, int height, int conf, string &&s) :
            m_x(x), m_y(y), m_width(width), m_height(height), m_confidence(conf), m_str(s)
    {
        m_index = 0;
    }
    void set_index(int index){
        m_index = index;
    }
    int m_x;
    int m_y;
    int m_width;
    int m_height;
    int m_index;
    int m_confidence;
    list<* word_type> m_close_words_left;
    list<* word_type> m_close_words_right;
    list<* word_type> m_close_words_up;
    list<* word_type> m_close_words_down;
    string m_str;
};

typedef vector<word_type> word_vector;

class screen{
public:
    static double compare(cv::Mat obj1, cv::Mat obj2, int compare_method);
    static double compare(cv::Mat obj1, cv::Mat obj2);
    static cv::Mat create_histogram(struct video_frame *in);
    static void free_video_frame(struct video_frame *in);
};

class screen_package{
private:
    std::vector<cv::Mat> m_histograms;
    std::vector<word_vector> m_lists_of_boxes;
    double m_histogram_compare_threshold;     //compare need to be bigger then this number (compare result is <1,0>)
    mutex m_mutex;
public:
    screen_package();
    screen_package(double compare_result_threshold);
    /**
     * save histogram and words connected to it
     * @param hist
     * @param words
     */
    void save_screen(cv::Mat hist, std::vector<word_type> words);
    /**
     * 
     * @param hist histogram to find boxes to
     * @return pointer to internal vector word type, don't delete it, if it is NULL 
     * there is no simular histogram
     */
    std::vector<word_type>* get_boxes(cv::Mat hist);
    unsigned int size();
    /**
     * set threshold to comparing histograms 
     * (higher threshold mean histograms need to be more simular to be equal)
     * @param new_value
     */
    void set_histogram_compare_threshold(double new_value);
    /**
     * get threshold to comparing histograms 
     * (higher threshold mean histograms need to be more simular to be equal)
     * @param new_value
     */
    double get_histogram_compare_threshold();
};

class tesseract_utils{
public:
    /**
     * class representing queue of BOXes
     */
    class boxes_queue{
        list<BOX *> m_internal_queue;  
        mutex m_mutex;
        condition_variable m_condv;
    public:
        boxes_queue();
        ~boxes_queue();
        int size();
        void add(Boxa * item);
        BOX * remove();
    };
    
    /**
     * class representing queue of words
     */
    class words_queue{
        word_vector m_internal_queue;  
        mutex m_mutex;
        condition_variable m_condv;
        int m_input_length;
    public:
        words_queue();
        ~words_queue();
        int size();
        void add(word_type item);
        word_vector return_all();
        void set_input_length(int input_length);
    };
    
    /**
     * class for thread working on processing boxes
     */ 
    class boxes_worker{
        boxes_queue & m_boxes_queue;
        words_queue & m_words_queue;
        int m_num_worker;
        tesseract::TessBaseAPI m_private_api;
    public:
        boxes_worker(boxes_queue & queue, words_queue & w_queue, int num_worker);
        void init_api(const char* datapath, string language);
        void set_image(const unsigned char* imagedata, int width, int height, int bytes_per_pixel, int bytes_per_line);
        void end_api();
        void thread_function();
    };
    
    /**
     * main function to get recognized boxes
     * @param in
     * @return 
     */
    tesseract_utils();
    ~tesseract_utils();
    word_vector find_boxes(struct video_frame *in);
    void init();
    void finalize();
    void destroy_word(word_type);
    static void destroy_box(BOX*);
    /**
     * don't use if you don't know what you doing, use find_boxes
     * @param in
     * @return 
     */
    Boxa* find_all_boxes();
    void remove_small_boxes(Boxa* boxes);
    word_vector recognize_boxes(Boxa* boxes);  //paralael
    word_vector  keep_boxes_to_anonymize(word_vector); //propably paralel
    static bool compare_index_word_type(word_type a, word_type b){return a.m_index < b.m_index;}
    void thread_recognize_boxes(Boxa* boxes, int& number_of_processed_boxes, int number_of_this_thread, word_vector& ret);   //thread to recognize boxes
    void unscale_position(word_vector &);
    
    void set_year(int year);
    void set_month(int month);
    void set_day(int day);
    string trim_and_reduce_spaces( const string& s );
private:
    int m_actual_year;
    int m_actual_day;
    int m_actual_month;
    mutex mx_cv_recognize_thread;
    mutex m_threads_on;
    mutex mx_gret;
    mutex mx_num_recognize_done;
    mutex mx_boxes;
    mutex mx_ret;
    mutex m_mx_tesseract;
    int m_num_recognize_done;
    condition_variable cv_recognize_done;
    condition_variable cv_recognize_thread;
    tesseract::TessBaseAPI m_api_name_search;
    tesseract::TessBaseAPI m_api;
    //tesseract::TessBaseAPI m_paralel_api[NUMBER_OF_THREADS_USING];
    //vector<tesseract::TessBaseAPI> m_paralel_api;
    boxes_queue m_boxes_queue;
    words_queue m_words_queue;
    
    vector<thread> m_threads;
    vector<boxes_worker*> m_workers;
};

/**
 * class representing queue of (video frame *)
 * as bonus is saving histograms for convenience of next processing(remove need to calculate it more than once)
 */
class video_frame_work_queue{
    list<struct video_frame *> m_queue_frame;
    list<cv::Mat> m_queue_histogram;
    mutex m_mutex;
    condition_variable m_condv;
    screen_package & m_screens;
    
public:
    video_frame_work_queue(screen_package & screens);
    ~video_frame_work_queue();
    int size();
    void add(struct video_frame * item, cv::Mat histogram);
    std::pair<struct video_frame *, cv::Mat> remove();
};
 
/**
 * class for thread processing video frames
 * should run only one at a time, more simultaneity is not define
 */
class video_frame_process_worker{
    video_frame_work_queue & m_queue; 
    screen_package & m_screens;
    tesseract_utils m_tesseract_utils;
public:
    video_frame_process_worker(video_frame_work_queue & queue, screen_package & screens);
    void thread_function();
    //quit tesseract inside, can't start after
    void stop();
};

#endif /* ANONYMIZE_UTILS_H */


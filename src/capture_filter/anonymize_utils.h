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

#define MULTIPLY_HISTOGRAM_CODING 1000
#define RECOGNIZING_OFFSET 10
#define MAX_PIXELIZATION_SIZE 200
#define MIN_PIXELIZATION_SIZE 5
#define MAX_SCALE_SIZE 5
#define MIN_SCALE_SIZE 1
#define MAX_SHRINK_SIZE 5
#define MIN_SHRINK_SIZE 1
#define MAX_RECOGNIZING_THREAD 30
#define MIN_RECOGNIZING_THREAD 1
#define MAX_DEBUG_INFO 10
#define MIN_DEBUG_INFO 0
#define MAX_COMPARE_VALUE 1
#define MIN_COMPARE_VALUE 0

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <boost/regex.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"

#include "video.h"
#include "video_codec.h"
#include "utils/lock_guard.h"

#include <vector>
#include <set>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <regex>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <chrono>

using namespace std;

struct configuration_values{
public:
    bool save;
    bool load;
    int pixelization_size;
    float scale_size;
    float shrink_size;
    string save_path;
    string load_path;
    string image_path;
    int num_recognizing_threads;
    double compare_value_to_be_same;
    bool recognize_new_screens;
    bool save_new_screen;
    string tesseract_language_name;
    bool anonymize_unrecognized;
}; 

struct print_message_struct{
private:
    static set<int> messages_setting;
    static bool error_messages;
public:
    enum Message_type{
        forced,
        error,
        frame_time,
        histogram_time,
        time_pixelate,
        time_frame_preprocess,
        time_tesseract_OCR,
        time_semantic_analysis,
        time_frame_to_RGB,
        histogram_comparation,
        order_of_action_some,
        order_of_action_more 
    };
    /**
     * 1 - time that take to process each individual frame
     * 2 - histogram_time
     * 3 - time pixelate
     * 4 - frame preprocess
     * 5 - tesseract OCR
     * 6 - semantic analysis
     * 7 - time frame to RGB
     * 8 - histogram comparation
     * 9 - order of actions (some)
     * 10- order of actions (lot)
     * 
     */
    static void set_messages_setting(int setting){
        messages_setting.insert(setting);
    }
    static void set_error_messages_setting(bool setting){
        error_messages = error;
    }
    /**
     * function that determinate if message should be printed or not
     * @param message_type determinate by type of message
     * @return true if message should be printed
    */
    static bool print_message(Message_type message_type);
    //static bool print_message(Message_type message_type, int thread);
    
};

struct helping_functions{
    static bool file_exists_test (const std::string& name);
    static bool directory_exists_test (const std::string& name);
    static bool can_file_exists_test (const std::string& name);
    static std::vector<std::string> split_string(const std::string str, std::string delim);
    
};




struct histogram_coding{
    static cv::Mat integer_coding(cv::Mat mat); 
    static cv::Mat integer_decoding(cv::Mat mat); 
    static std::string mat_to_string_coding(cv::Mat);
    static cv::Mat string_to_mat_decoding(std::string);    
    static vector<std::string> split_string_code(std::string);
    static std::string int_to_character_coding(unsigned int num);
    static unsigned int char_to_int_coding(std::string num);
};

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
    void set_type(string type){
        m_type = type;
    }
    int m_x;
    int m_y;
    int m_width;
    int m_height;
    int m_index;
    int m_confidence;
    string m_str;
    string m_type;
};

typedef vector<word_type> word_vector;

class screen{
public:
    static double compare(cv::Mat obj1, cv::Mat obj2, int compare_method);
    static double compare(cv::Mat obj1, cv::Mat obj2);
    static cv::Mat create_histogram(struct video_frame *in, float shrink_coeficient);
    static void free_video_frame(struct video_frame *in);
};

class screen_package{
private:
    std::vector<cv::Mat> m_histograms;
    std::vector<word_vector> m_lists_of_boxes;
    std::vector<std::string> m_screen_names; 
    double m_histogram_compare_threshold;     //compare need to be bigger then this number (compare result is <1,0>)
    mutex m_mutex;
    bool still_running;
public:
    screen_package(double compare_result_threshold);
    /**
     * save histogram and words connected to it
     * @param hist
     * @param words
     */
    void save_screen(cv::Mat hist, std::vector<word_type> words);
    /**
     * save histogram and words connected to it
     * @param hist histogram
     * @param words boxes to anonymize
     * @param name name of screens
     */
    void save_screen(cv::Mat hist, std::vector<word_type> words, std::string name);
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
    
    std::pair<cv::Mat, word_vector> get_screen(int position);
    std::string get_screen_name(int position);
};

class tesseract_utils{
public:
    /**
     * class representing trigger
     */
    class tesseract_worker_trigger{ 
        mutex m_mutex;
        condition_variable m_condv;
    public:
        tesseract_worker_trigger();
        ~tesseract_worker_trigger();
        void triger();
        void can_go();
    };
    
    /**
     * class representing queue of words
     * return words after all threads called function thread_done()
     */
    class tesseract_worker_data_queue{
        word_vector m_internal_queue;  
        mutex m_mutex;
        condition_variable m_condv;
        int m_threads;
        int m_threads_done;
    public:
        tesseract_worker_data_queue();
        ~tesseract_worker_data_queue();
        int size();
        void add(word_vector items);
        word_vector return_all();
        void clear();
        void set_number_threads(int threads);
        void thread_done();
    };
    
    /**
     * class for thread working on processing boxes
     */ 
    class tesseract_worker{
        tesseract_worker_trigger & m_worker_trigger;
        tesseract_worker_data_queue & m_output_data_queue;
        int m_num_worker;
        int m_sum_all_workers;
        tesseract::TessBaseAPI m_private_api;
        bool still_running;
        int m_width;
        int m_height;
    public:
        tesseract_worker(tesseract_worker_trigger & queue, tesseract_worker_data_queue & w_queue, int num_worker, int all_workers);
        /**
         * inicializate Tesseract engine inside this thread
         * @param datapath tesseract thing
         * @param language language to be set
         */
        void init_api(const char* datapath, string language);
        /**
         * set image to tesseract engine
         */
        void set_image(const unsigned char* imagedata, int width, int height, int bytes_per_pixel, int bytes_per_line);
        /**
         * end tesseract api
         */
        void end_api();
        /**
         * stop this thread
         */
        void stop_thread();
        /**
         * function inside tesseract threads
         */
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
    /**
     * inicialization function
     */
    void init();
    /**
     * ending function, close Tesseract engine
     */
    void finalize();
    static void destroy_box(BOX*);
    /**
     * semantic processing
     * @param input words
     * @return  processed words
     */
    word_vector  keep_boxes_to_anonymize(word_vector); 
    static bool compare_index_word_type(word_type a, word_type b){return a.m_index < b.m_index;}
    static bool sort_by_hight_words_function(word_type a, word_type b){return a.m_y < b.m_y;}
    void unscale_position(word_vector &);
    
    void set_year(int year);
    void set_month(int month);
    void set_day(int day);
    void set_configuration(configuration_values);
    string trim_and_reduce_spaces( const string& s );
    
    bool is_name(string checking_name);
    
private:
    configuration_values m_config;
    bool m_initialized;
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
    condition_variable cv_recognize_done;
    condition_variable cv_recognize_thread;
    tesseract::TessBaseAPI m_api_name_search;
    tesseract_worker_trigger m_tesseract_worker_trigger;
    tesseract_worker_data_queue m_words_queue;
    
    vector<thread> m_threads;
    vector<tesseract_worker*> m_workers;
    
    bool is_name_internal(string checking_name);
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
    screen_package * m_screens;
    bool still_running;
    
public:
    video_frame_work_queue(screen_package * screens);
    ~video_frame_work_queue();
    void set_screen_package(screen_package * screens);
    int size();
    void add(struct video_frame * item, cv::Mat histogram);
    std::pair<struct video_frame *, cv::Mat> remove();
    void end_queue();
};

/**
 * class for thread processing video frames
 * should run only one at a time, more simultaneity is not define
 */
class video_frame_process_worker{
    configuration_values m_config;
    video_frame_work_queue * m_queue; 
    screen_package * m_screens;
    tesseract_utils m_tesseract_utils;
    bool m_still_running;
public:
    video_frame_process_worker(video_frame_work_queue * queue, screen_package * screens, configuration_values config);
    ~video_frame_process_worker();
    /**
     * function for frame recognition thread
     */
    void thread_function();
    //quit tesseract inside, can't start after
    void stop();
};

class levenshtein_distance{
public:
    /*
    * weight of substitution of character type in word 
    * example:
    * regex: dd.dd.dddd
    * word : 11.1i.1988 <- all characters exept 'i' return 0
    *                      'i' is alpha character but should be digit, this function return weight for this substitution
    */
    static int levenshtein_distance_substitution_weight(char letter, char regex);
    /*
    * weight of missing character type in word 
    * example:
    * regex: dd.dd.dddd
    * word : 1111.1988 <- '.' is missing, value of '.' (value for punct) should be added in calculation  
    */
    static int levenshtein_distance_addition_weight(char regex);
    
    /*
     * weight of extra character in word 
     * example:
     * regex: dd.dd.dddd
     * word : 11.111.1988 <- '1' is extra, value of '1' (value for digit) should be added in calculation  
     */
    static int levenshtein_distance_deletition_weight(unsigned wchar_t letter);
    /**
    * 
    * @param word_to_compare
    * @param regex can contain only "aAdD. "
    * a - alpha optional
    * A - aplha required
    * d - digit optional
    * D - digit required
    * . - punct //to be changed
    * ' '(space) - blank
    * @return levenshtein weighted distance, weights are taken from functions about
    */
    static int calculate_levenshtein_distance(std::string word_to_compare, std::string regex);
    
};

#endif /* ANONYMIZE_UTILS_H */


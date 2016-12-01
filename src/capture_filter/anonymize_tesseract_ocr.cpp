/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


#include "anonymize_utils.h"
#include <tesseract/genericvector.h>
#include <cstring>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <cmath>
#include <iostream>
#include <c++/5/condition_variable>
#include <boost/algorithm/string.hpp> 


tesseract_utils::tesseract_utils(){    
    for(int i=0;i<NUMBER_OF_THREADS_USING;i++){
        boxes_worker *  worker = new boxes_worker(m_boxes_queue, m_words_queue, i);
        m_workers.push_back(worker);
        m_threads.push_back(thread(&tesseract_utils::boxes_worker::thread_function, worker));
    }
    init();
}

tesseract_utils::~tesseract_utils() {
    for(std::thread &t : m_threads){
        t.join();
    }
    //free workers
    for (int i=0;i<m_workers.size();i++){
        delete m_workers[i];
    }
}


void tesseract_utils::init(){
     //not currently used
   /*GenericVector<STRING> pars_vec;
    pars_vec.push_back("language_model_penalty_non_freq_dict_word");
    pars_vec.push_back("language_model_penalty_non_dict_word");
    pars_vec.push_back("segment_penalty_dict_nonword");
    pars_vec.push_back("segment_penalty_garbage");
    pars_vec.push_back("stopper_nondict_certainty_base");
    pars_vec.push_back("language_model_penalty_spacing");

    GenericVector<STRING> pars_values;
    pars_values.push_back("0.50");
    pars_values.push_back("0.50");
    pars_values.push_back("1.0");
    pars_values.push_back("1.0");
    pars_values.push_back("-10");
    pars_values.push_back("1.0");*/
    if (m_api.Init(NULL, "uga", tesseract::OEM_TESSERACT_ONLY)) {
        throw string("Could not initialize tesseract api");
    }
    
    if (m_api_name_search.Init(NULL, "ugt", tesseract::OEM_TESSERACT_ONLY)) {
        throw string("Could not initialize tesseract api to name search");
    }
    
    for(int i=0; i<NUMBER_OF_THREADS_USING ; i++){
        m_workers[i]->init_api(NULL, "uga");
    }
}

void tesseract_utils::finalize(){
    m_api.End();
    m_api_name_search.End();
    for(int i=0; i<NUMBER_OF_THREADS_USING; i++){
        m_workers[i]->end_api();
    }
}

/**
 * 
 * @param in expect RGB format
 * @return 
 */
std::vector<word_type> tesseract_utils::find_boxes(struct video_frame *in){ 
    
    if(in->color_spec != RGB){
        log_msg(LOG_LEVEL_WARNING, "Cannot \"find boxes\" from other format then RGB\n");
        log_msg(LOG_LEVEL_ERROR,   "internal error in \"find boxes\"\n");
    }
    
    int width  = in->tiles[0].width;
    int height = in->tiles[0].height;
    cv::Mat picture_mat = cv::Mat(height, width, CV_8UC3,  in->tiles[0].data);
    for (int i=0;i<500;i++){
        //cout << (int)in->tiles[0].data << " ";
    }
    cout << endl;
    cv::Mat resize_picture;
    cv::Mat unsharped_picture;
    
    //resize
    cv::resize(picture_mat, resize_picture, cv::Size(width*2, height*2), 0, 0, cv::INTER_LINEAR);
    
    //unsharp
    float sharpening_amount = 0.5;
    cv::GaussianBlur(resize_picture, unsharped_picture, cv::Size(0, 0), 9);
    //Sharpened = Original + ( Original - Blurred ) * Amount
    cv::addWeighted(resize_picture, 1 + sharpening_amount, unsharped_picture, -1 * sharpening_amount, 0, unsharped_picture); 

    //prepare format for tesseract
    Pix *image = pixCreateNoInit(resize_picture.cols, resize_picture.rows, 8);
    //Pix *image = pixCreateNoInit(unsharped_picture.cols, unsharped_picture.rows, 8);
    
    //cv::namedWindow( "unsharped_picture window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "unsharped_picture window", unsharped_picture );                   // Show our image inside it.
    //cv::namedWindow( "resize_picture window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "resize_picture window", resize_picture ); 
    //cv::waitKey(0);
    
    
    vc_copylineRGBtoGrayscale_SSE((unsigned char *) image->data, (unsigned char *) resize_picture.data, (resize_picture.rows * resize_picture.cols));
    //vc_copylineRGBtoGrayscale_SSE((unsigned char *) image->data, (unsigned char *) unsharped_picture.data, (unsharped_picture.rows * unsharped_picture.cols));
    
    m_api.SetImage((unsigned char *) image->data, image->w, image->h, 1, image->w);
    for(int i = 0; i < NUMBER_OF_THREADS_USING; i++){
        m_workers[i]->set_image((unsigned char *) image->data, image->w, image->h, 1, image->w);
    }
    
    Boxa* boxes = find_all_boxes();
    
    //remove small boxes
    remove_small_boxes(boxes);
    
    //recognize boxes
    cout << "get length " << boxes->n << endl;
    m_words_queue.set_input_length(boxes->n);
    m_boxes_queue.add(boxes);
    
    
    word_vector words = m_words_queue.return_all();
   // vector<word_type> words = recognize_boxes(boxes);
    
    //keep boxes to anonymize
    words = keep_boxes_to_anonymize(words);
    
    unscale_position(words);
    pixDestroy(&image);
    return words;
}

void tesseract_utils::destroy_box(BOX *b) {
    boxDestroy(&b);
}

void tesseract_utils::destroy_word(word_type word) {
    //boxDestroy(&b);
}

void tesseract_utils::remove_small_boxes(Boxa* boxes){
    for(int i = 0; i < boxaGetCount(boxes);){
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        if((box->w < 5) || (box->h < 14) ){//|| (box->w > 1000) || (box->h > 1000)){
            boxaRemoveBox(boxes, i);
        }else{
            i++;
        }
    }
}


word_vector tesseract_utils::recognize_boxes(Boxa* boxes){
    word_vector ret;
    int number_of_processed_boxes = 0;
    std::vector<std::thread> threads;
    /*for(int i = 0; i < boxes->n; i++){
        cout << boxes->box[i]->w << "x" << boxes->box[i]->h << endl;
    }*/

    for(int i = 0; i < NUMBER_OF_THREADS_USING; i++){
    //    cout << "thread " << i << " started" << endl;
        threads.push_back(std::thread(&tesseract_utils::thread_recognize_boxes, this,  boxes, std::ref(number_of_processed_boxes), i, std::ref(ret)));
    }

    for(std::thread &t : threads){
        t.join();
    }
    threads.clear();
    sort(ret.begin(), ret.end(), compare_index_word_type);
    return ret;
}

word_vector tesseract_utils::keep_boxes_to_anonymize(word_vector words){
    int new_order = 0;
    word_vector ret;
    ret.clear();
    //clean spaces
    for(int i=0; i < words.size(); i++){
         //newline -> space; no space on the end of word
        string word_text = words.at(i).m_str;
        for(int j=0; j < word_text.size(); j++){
            if(word_text[j] == '\n'){
                word_text[j] = ' ';
            }
        }
        for(int j=0; j < word_text.size(); j++){
            trim_and_reduce_spaces(word_text);
        }
        while((word_text.size() - 1 == word_text.rfind(' ')) && (word_text.size() > 2)){
            word_text.erase(word_text.size() -1);
        }
        //cout << i << "-box " << words.at(i).m_str << " -> " << word_text << endl;
        //save text cleaned from unnecessary spaces
        words.at(i).m_str = word_text;
    }
    
    //TO DO: connect words
    //probably think of something better
    /*for(int i=0; i < words.size();i++){
        word_type word_first = words.at(i);
        for(int j=i+1;j < words.size();j++){
            word_type word_second = words.at(j);
            int delta_x = word_first.m_x - word_second.m_x;
            int delta_y = word_first.m_y - word_second.m_y;
            if(abs(delta_y) < 10){   //TO DO: make treashold changeble and possible to set from outside
                if(delta_x < 0){//first word start more left than second
                    if(delta_x + word_first.m_width > -5 && delta_x + word_first.m_width < 10){ //TO DO: make treashold changeble and possible to set from outside
                        //is close enough
                        word_first.m_close_words_right.
                    }
                }else{//first word start more right than second
                    if(delta_x + word_second.m_width > -5 && delta_x + word_second.m_width < 10){ //TO DO: make treashold changeble and possible to set from outside
                        //is close enough
                        
                    }
                }
            }
            if(abs(delta_x) < 10){  //TO DO: make treashold changeble and possible to set from outside
                if(delta_y < 0){//first word start more up than second
                    
                }else{//first word start more down than second
                    
                }
            }
        }
    }*/
    
    //TO DO: split words 
    
    //find names
    for(int i=0; i < words.size(); i++){
        if(m_api_name_search.IsValidWord(words.at(i).m_str.c_str())){
            ret.push_back(word_type(words.at(i).m_x, words.at(i).m_y, words.at(i).m_width, words.at(i).m_height, new_order, words.at(i).m_confidence, string(words.at(i).m_str.c_str())));
            new_order++;
        }
    }
    
    //find birth dates
    for(int i=0; i < words.size(); i++){
        if(words.at(i).m_str.find_first_of("0123456789") != std::string::npos){
            //first posisibility format
            boost::regex rx1("([0-9]{1,2})[-/.,]([0-9]{1,2})[-/.,]([0-9]{2,4})");//(space is not good, can recognize some random number, maybe in second format could be good)
            //second possibility format (if separator are not recognized(may change when use to recognize boxes or when generate boxes to saved))
            boost::regex rx2("([0-9]{1,2})([0-9]{1,2})([0-9]{4})");
            boost::smatch res;
            if(boost::regex_search(words.at(i).m_str, res, rx1)){
                //first posisibility format
                string year_text = string(res[3]);
                int year = stoi(year_text, nullptr);
                if(string(res[3]).size() == 2){
                    if(m_actual_year % 100 < year){
                        year += 1900;
                    }else{
                        year += 2000;
                    }
                }
                if(year < m_actual_year - 2){
                    //cout << words.at(i).m_str << endl;
                    ret.push_back(word_type(words.at(i).m_x, words.at(i).m_y, words.at(i).m_width, words.at(i).m_height, new_order, words.at(i).m_confidence, string(words.at(i).m_str.c_str())));
                    new_order++;
                }
            }else if(boost::regex_search(words.at(i).m_str, res, rx2)){
                //second possibility format
                string year_text = string(res[3]);
                int year = stoi(year_text, nullptr);
                if((year > m_actual_year - 100) && (year < m_actual_year)){
                    //cout << words.at(i).m_str << endl;
                    ret.push_back(word_type(words.at(i).m_x, words.at(i).m_y, words.at(i).m_width, words.at(i).m_height, new_order, words.at(i).m_confidence, string(words.at(i).m_str.c_str())));
                    new_order++;
                }
            }
        }
    }
    
    //find personal numbers
    for(int i=0; i < words.size(); i++){
        cout << "word " << words.at(i).m_str;
        if(words.at(i).m_str.find_first_of("0123456789") != std::string::npos){
            boost::regex rx("[0-9]{6}/[0-9]{2,4}", std::regex_constants::basic);
            if(boost::regex_search(words.at(i).m_str.c_str(), rx)){
                //cout << words.at(i).m_str;
                ret.push_back(word_type(words.at(i).m_x, words.at(i).m_y, words.at(i).m_width, words.at(i).m_height, new_order, words.at(i).m_confidence, string(words.at(i).m_str.c_str())));
                new_order++;
            }
        }
        cout << endl;
    }
    
    return ret;
}

void tesseract_utils::unscale_position(word_vector & words){
    for(int i=0;i<words.size();i++){
        words.at(i).m_x /= UPSCALE_COEFICIENT;
        words.at(i).m_y /= UPSCALE_COEFICIENT; 
        words.at(i).m_width /= UPSCALE_COEFICIENT;
        words.at(i).m_height /= UPSCALE_COEFICIENT; 
    }
}

void tesseract_utils::thread_recognize_boxes(Boxa* boxes, int& number_of_processed_boxes, int number_of_this_thread, word_vector& ret){
    int actual_box;

    //mx_boxes locking have following schema { - lock, } - unlock
    //{             -start
    //}{}{}{        -cycle
    //}             -end
    mx_boxes.lock();  
    /*while(number_of_processed_boxes < boxes->n){   //if all boxes where processed        
        BOX* box = boxaGetBox(boxes, number_of_processed_boxes, L_CLONE);
        //cout << box->w << "x" << box->h << endl;
        actual_box = number_of_processed_boxes;
        number_of_processed_boxes++;
        mx_boxes.unlock();
        
        //std::cout << box->x << " " << box->y << " " << box->w << " " << box->h << std::endl
        //not suported !!!!!!!!!!!!!!!!!!!!! (by me)
        //m_paralel_api[number_of_this_thread].SetRectangle(box->x, box->y, box->w, box->h);
        //char* ocrResult = m_paralel_api[number_of_this_thread].GetUTF8Text();
        //int conf = m_paralel_api[number_of_this_thread].MeanTextConf();
        
        
        mx_ret.lock();
        ret.push_back(word_type(box->x, box->y, box->w, box->h, actual_box, conf, string(ocrResult)));
        mx_ret.unlock();

        delete [] ocrResult; 
        destroy_box(box);
        
        mx_boxes.lock(); 
    }*/
    mx_boxes.unlock();  
}

Boxa* tesseract_utils::find_all_boxes(){
    m_api.Recognize(NULL);
    return m_api.GetComponentImages(tesseract::RIL_WORD, true, NULL, NULL);
}

void tesseract_utils::set_year(int year){
    m_actual_year = year;
}

void tesseract_utils::set_month(int month){
    m_actual_month = month;
}

void tesseract_utils::set_day(int day){
    m_actual_day = day;
}

string tesseract_utils::trim_and_reduce_spaces( const string& s )
  {
  string result = boost::trim_copy( s );
  while (result.find( "  " ) != result.npos)
    boost::replace_all( result, "  ", " " );
  return result;
  }

tesseract_utils::boxes_queue::boxes_queue(){
    
}

tesseract_utils::boxes_queue::~boxes_queue(){

}

int tesseract_utils::boxes_queue::size(){
    m_mutex.lock();
    int size = m_internal_queue.size();
    m_mutex.unlock();
    return size;
}

void tesseract_utils::boxes_queue::add(Boxa * item){
    m_mutex.lock();
    for (int i=0;i<item->n;i++){
        m_internal_queue.push_back(boxaGetBox(item, i, L_CLONE));
    }
    m_condv.notify_all();
    m_mutex.unlock();
    boxaDestroy(&item);
}

BOX * tesseract_utils::boxes_queue::remove(){
    unique_lock<mutex> m_lock(m_mutex);
    m_condv.wait(m_lock, [this](){return m_internal_queue.size() != 0; });      //wait until something is there
    
    BOX * item = m_internal_queue.front();
    m_internal_queue.pop_front();
    return item;
}

tesseract_utils::boxes_worker::boxes_worker(boxes_queue & b_queue, words_queue & w_queue, int num_worker)
                            :m_boxes_queue(b_queue), m_words_queue(w_queue), m_num_worker(num_worker){
    
}

void tesseract_utils::boxes_worker::init_api(const char* datapath, string language){
    if (m_private_api.Init(datapath, language.c_str(), tesseract::OEM_TESSERACT_ONLY)) {
        cout << "Could not initialize tesseract api in thread" << m_num_worker << endl;
        throw string("Could not initialize tesseract api in thread " + to_string(m_num_worker));
    }
}

void tesseract_utils::boxes_worker::set_image(const unsigned char* imagedata, int width, int height, int bytes_per_pixel, int bytes_per_line){
    m_private_api.SetImage(imagedata, width, height, bytes_per_pixel, bytes_per_line);
}

void tesseract_utils::boxes_worker::end_api(){
    m_private_api.End();
}

void tesseract_utils::boxes_worker::thread_function(){
    while(true){
        BOX * box = m_boxes_queue.remove();
        
        m_private_api.SetRectangle(box->x, box->y, box->w, box->h);
        char* ocrResult = m_private_api.GetUTF8Text();
        int conf = m_private_api.MeanTextConf();
        m_words_queue.add(word_type(box->x, box->y, box->w, box->h, conf, string(ocrResult)));

        delete [] ocrResult;
        destroy_box(box);
    }
}


tesseract_utils::words_queue::words_queue(){
}

tesseract_utils::words_queue::~words_queue(){
    
}

int tesseract_utils::words_queue::size(){
    m_mutex.lock();
    int size = m_internal_queue.size();
    m_mutex.unlock();
    return size;
}

void tesseract_utils::words_queue::add(word_type item){
    unique_lock<mutex> m_lock(m_mutex);
    item.set_index(m_internal_queue.size());    //set index
    m_internal_queue.push_back(item);
    if(m_internal_queue.size() == m_input_length){
        m_condv.notify_one();
    }
}

/**
 * 
 * @return whole saved vector
 */
word_vector tesseract_utils::words_queue::return_all(){
    unique_lock<mutex> m_lock(m_mutex);
    cout << "returning all words" << endl;
    m_condv.wait(m_lock, [this](){return m_internal_queue.size() >= m_input_length;});
    word_vector return_vector(m_internal_queue);
    m_internal_queue.clear();
    cout << "return " << return_vector.size() << "words" << endl;
    return return_vector;
}

void tesseract_utils::words_queue::set_input_length(int input_length){
    unique_lock<mutex> m_lock(m_mutex);
    m_input_length = input_length;
}
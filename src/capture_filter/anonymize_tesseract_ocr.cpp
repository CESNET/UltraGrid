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
#include <c++/5/bits/stl_list.h> 
#include <algorithm>



#include <time.h>

tesseract_utils::tesseract_utils(){   
    m_initialized = false; 
}

tesseract_utils::~tesseract_utils() {
    for (int i=0;i<m_workers.size();i++){
        m_workers[i]->stop_thread();
    }
    m_tesseract_worker_trigger.triger();
    int k = 0;
    for(std::thread &t : m_threads){
        t.join();
        k++;
    }
    //free workers
    for (int i=0;i<m_workers.size();i++){
        m_workers[i]->end_api();
        delete m_workers[i];
    }
}


void tesseract_utils::init(){
    if(!m_initialized){
        if(m_config.num_recognizing_threads > 0){
            for(int i=0;i<m_config.num_recognizing_threads;i++){
                tesseract_worker *  worker = new tesseract_worker(m_tesseract_worker_trigger, m_words_queue, i, m_config.num_recognizing_threads);
                m_workers.push_back(worker);
                m_threads.push_back(thread(&tesseract_utils::tesseract_worker::thread_function, worker));
            }
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
           
           if (m_api_name_search.Init(NULL, m_config.tesseract_language_name.c_str(), tesseract::OEM_TESSERACT_ONLY)) {
               string error = "Could not initialize tesseract api for name search, trying tesseract language \"";
               error.append(m_config.tesseract_language_name).append("\"");
               throw string(error);
           }

           for(int i=0; i<m_config.num_recognizing_threads; i++){
               m_workers[i]->init_api(NULL, m_config.tesseract_language_name.c_str());
           }
           m_initialized = true;
        }else{
            if(print_message_struct::print_message(print_message_struct::Message_type::error)){
                cerr << "number of recognized threads should be > 0" << endl;
            }
        }
    }else{
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            cerr << "already initialized tesseract utils" << endl;
        }
    }
}


void tesseract_utils::set_configuration(configuration_values config){
    if(!m_initialized){
        m_config = config;
    }else{
        if(print_message_struct::print_message(print_message_struct::Message_type::error)){
            cerr << "cannot set configuration after initialization" << endl;
        }
    }
}

void tesseract_utils::finalize(){
    m_api_name_search.End();
    for(int i=0; i<m_config.num_recognizing_threads; i++){
        m_workers[i]->end_api();
    }
}

/**
 * 
 * @param in expect RGB format
 * @return 
 */
std::vector<word_type> tesseract_utils::find_boxes(struct video_frame *in){ 
    chrono::time_point<std::chrono::system_clock> time1, time2, time3, time4;
    std::chrono::duration<double> elapsed_seconds;
    if(in->color_spec != RGB){
        log_msg(LOG_LEVEL_WARNING, "Cannot \"find boxes\" from other format then RGB\n");
        log_msg(LOG_LEVEL_ERROR,   "internal error in \"find boxes\"\n");
    }
    
    time3 = chrono::system_clock::now();
    int width  = in->tiles[0].width;
    int height = in->tiles[0].height;
    cv::Mat picture_mat = cv::Mat(height, width, CV_8UC3,  in->tiles[0].data);

    cv::Mat resize_picture;
    cv::Mat unsharped_picture;
    
    //resize
    time1 = chrono::system_clock::now();
    cv::resize(picture_mat, resize_picture, cv::Size(width*m_config.scale_size, height*m_config.scale_size), 0, 0, cv::INTER_LINEAR);
    time2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::time_frame_preprocess)){
        elapsed_seconds = time2 - time1;
        cout << "time to preprocess; resize frame: " << elapsed_seconds.count() << " seconds; resize scale: " << m_config.scale_size << endl;
    }
    
    //unsharp
    float sharpening_amount = 0.5;
    //cv::GaussianBlur(resize_picture, unsharped_picture, cv::Size(0, 0), 9);
    //Sharpened = Original + ( Original - Blurred ) * Amount
    //cv::addWeighted(resize_picture, 1 + sharpening_amount, unsharped_picture, -1 * sharpening_amount, 0, unsharped_picture); 

    //prepare format for tesseract
    Pix *image = pixCreateNoInit(resize_picture.cols, resize_picture.rows, 8);
    //Pix *image = pixCreateNoInit(unsharped_picture.cols, unsharped_picture.rows, 8);
   
    time1 = chrono::system_clock::now();
    vc_copylineRGBtoGrayscale_SSE((unsigned char *) image->data, (unsigned char *) resize_picture.data, (resize_picture.rows * resize_picture.cols));
    time2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::time_frame_preprocess)){
        elapsed_seconds = time2 - time1;
        cout << "time to preprocess; to grayscale frame: " << elapsed_seconds.count() << " seconds" << endl;
    }
    //vc_copylineRGBtoGrayscale_SSE((unsigned char *) image->data, (unsigned char *) unsharped_picture.data, (unsharped_picture.rows * unsharped_picture.cols));
    time4 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::time_frame_preprocess)){
        elapsed_seconds = time4 - time3;
        cout << "time to complete preprocess: " << elapsed_seconds.count() << " seconds" << endl;
    }
    
    time1 = chrono::system_clock::now();
    for(int i = 0; i < m_config.num_recognizing_threads; i++){
        m_workers[i]->set_image((unsigned char *) image->data, image->w, image->h, 1, image->w);
    }
    
    /*Boxa* boxes = find_all_boxes();
    
    //remove small boxes
    remove_small_boxes(boxes);
    
    //recognize boxes    
    m_words_queue.set_input_length(boxes->n);
    m_boxes_queue.add(boxes);
    
    word_vector words = m_words_queue.return_all();
    */
    m_words_queue.clear();
    m_words_queue.set_number_threads(m_config.num_recognizing_threads);
    m_tesseract_worker_trigger.triger();
    
    word_vector words = m_words_queue.return_all();
    time2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::time_tesseract_OCR)){
        elapsed_seconds = time2 - time1;
        cout << "time to recognize words from frame using " << m_config.num_recognizing_threads << " threads: " << elapsed_seconds.count() << " seconds" << endl;
    }
    
    unscale_position(words);
    
    sort(words.begin(), words.end(), tmp_sort_words_function);
    
    //keep boxes to anonymize
    time1 = chrono::system_clock::now();
    words = keep_boxes_to_anonymize(words);
    time2 = chrono::system_clock::now();
    if(print_message_struct::print_message(print_message_struct::Message_type::time_semantic_analysis)){
        elapsed_seconds = time2 - time1;
        cout << "time to semantic analysis: " << elapsed_seconds.count() << " seconds" << endl;
    }
    
    pixDestroy(&image);
    return words;
}

void tesseract_utils::destroy_box(BOX *b) {
    boxDestroy(&b);
}

void tesseract_utils::destroy_word(word_type word) {
    //boxDestroy(&b);
}

/*void tesseract_utils::remove_small_boxes(Boxa* boxes){
    for(int i = 0; i < boxaGetCount(boxes);){
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        if((box->w < 5) || (box->h < 14) ){//|| (box->w > 1000) || (box->h > 1000)){
            boxaRemoveBox(boxes, i);
        }else{
            i++;
        }
    }
}*/

int levenshtein_distance::levenshtein_distance_substitution_weight(char letter, char regex){
    int letter_type;
    int regex_type;
    int weight_mat[4][4] = {    //matrix of weights to be search in
      // d| a| .|' '| <- regex type/ ↓ letter type
        {0, 2, 2, 3},//d
        {2, 0, 2, 3},//a
        {2, 2, 0, 2},//.
        {3, 3, 2, 0} //' '
    };
    //find letter type
    if(iswdigit(letter)){    
        letter_type = 0;
    }else if(iswalpha(letter)){
        letter_type = 1;
    }else if(iswpunct(letter)){
        letter_type = 2;
    }else if(iswblank(letter)){
        letter_type = 3;
    }else{
        letter_type = 1;    //probably is some kind of letter
    }
    //find regex type
    regex = towlower(regex);    //while substitution don't care if is digit required or optional
    switch(regex){
        case 'd':
            regex_type = 0;
            break;
        case 'a':
            regex_type = 1;
            break;
        case '.':
            regex_type = 2;
            break;
        case ' ':
            regex_type = 3;
            break;
    }
    return weight_mat[letter_type][regex_type];
}


int levenshtein_distance::levenshtein_distance_addition_weight(char regex){
    int regex_type;
    int weight_mat[6] = {    //vector of weights to be search in
      // D| A| .|' '| d| a
         3, 3, 2, 1,  0, 0
    };
    //find regex type
    switch(regex){
        case 'D':
            regex_type = 0;
            break;
        case 'A':
            regex_type = 1;
            break;
        case '.':
            regex_type = 2;
            break;
        case ' ':
            regex_type = 3;
            break;
        case 'd':
            regex_type = 4;
            break;
        case 'a':
            regex_type = 5;
            break;
    }
    //else{
    //    cerr << "using wrong character " << letter << " in levenshtein distance, can only use digit-\"d\" punctional-\".\" alpha-\"a\" blank-\" \"" << endl;
    //}
    return weight_mat[regex_type];
}

int levenshtein_distance::levenshtein_distance_deletition_weight(unsigned wchar_t letter){
    int letter_type;
    int weight_mat[4] = {    //vector of weights to be search in
      // d| a| .|' '|
         3, 3, 2, 1
    };
    //find letter type
    if(isdigit(letter)){    
        letter_type = 0;
    }else if(isalpha(letter)){
        letter_type = 1;
    }else if(ispunct(letter)){
        letter_type = 2;
    }else if(isblank(letter)){
        letter_type = 3;
    }else{
        letter_type = 1;    //probably is some kind of letter
    }
    return weight_mat[letter_type];
}

int levenshtein_distance::calculate_levenshtein_distance(string word_to_compare, string regex){
    int mat[word_to_compare.size()+1][regex.size()+1];
    mat[0][0] = 0;
    for(int i=1;i<word_to_compare.size()+1;i++){
        mat[i][0] = mat[i-1][0] + levenshtein_distance_deletition_weight(word_to_compare.at(i-1));
    }
    for(int i=1;i<regex.size()+1;i++){
        mat[0][i] = mat[0][i-1] + levenshtein_distance_addition_weight(regex.at(i-1));
    }
 
    for(int j=1;j<word_to_compare.size()+1;j++){
        for(int i=1;i<regex.size()+1;i++){                        
            //minimum from 3 values
            mat[j][i] = min(min(    mat[j-1][i] + levenshtein_distance_deletition_weight(word_to_compare.at(j-1)),      // deletion
                                    mat[j][i-1] + levenshtein_distance_addition_weight(regex.at(i-1))    ),             // insertion
                                    mat[j-1][i-1] + levenshtein_distance_substitution_weight(word_to_compare.at(j-1),
                                                                                            regex.at(i-1)));            // substitution
        }
    }
    return mat[word_to_compare.size()][regex.size()];
}

string set_correct_name_case(string name){
    if(name.length() == 0){
        return name;
    }
    string ret;
    for(int i=0;i<name.length();i++){
        unsigned char letter = name.at(i);
        if(i>0){//all letters except first
            ret.push_back(tolower(letter));
        }else{//first letter
            ret.push_back(toupper(letter));
        }
    }
    return ret;
}

bool tesseract_utils::is_name(string checking_name){
    //split by space check every par separatly
    vector<std::string> name_parts = helping_functions::split_string(checking_name, " ");
    for(int i=0;i<name_parts.size();i++){
        if(!is_name_internal(name_parts.at(i))){
                return false;
        }
    }
    return true;
}

bool tesseract_utils::is_name_internal(string checking_name){
    if(checking_name.length() < 2){
        return false;
    }
    //split by space check every par separatly
    int non_letter_chars = 0;
    for(int i=0;i<checking_name.length();i++){
        if(!isalpha(checking_name.at(i))){
            non_letter_chars++;
        }
    }
    if(non_letter_chars * 2 < checking_name.length()){//checking_name should have less than halve non alpha characters
        string checking_name_corect_case = set_correct_name_case(checking_name);
        if(m_api_name_search.IsValidWord(checking_name_corect_case.c_str()) != 0){
            return true;
        }else{//try to split name to two parts and match them separately
            if(checking_name.length() > 7){//try to split words with length 8 and more (shorter can be fluke)
                for(int i=3;i<checking_name.length()-2;i++){//make the length of the words at least 3 each
                    string checking_name_first_halve = set_correct_name_case(checking_name.substr(0, i));
                    string checking_name_second_halve = set_correct_name_case(checking_name.substr(i));
                    if(m_api_name_search.IsValidWord(checking_name_first_halve.c_str())){
                        if(m_api_name_search.IsValidWord(checking_name_second_halve.c_str())){
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

word_vector tesseract_utils::keep_boxes_to_anonymize(word_vector words){
    struct process_words_package{
        int levenshtein_distance_date;
        int levenshtein_distance_personal_num;
        word_type * word;
        bool is_name;
    };
    //vector that will be returned
    word_vector ret;
    //few vectors to remember what parts are important and base on this maybe find some less 
    word_vector names;
    vector<process_words_package> process_words_vector;
    ret.clear();
    
    //clean spaces
    for(int i=0; i < words.size(); i++){
         //newline -> space
        string word_text = words.at(i).m_str;
        for(int j=0; j < word_text.size(); j++){
            if(word_text[j] == '\n'){
                word_text[j] = ' ';
            }
        }
        //reduce and trim spaces
        trim_and_reduce_spaces(word_text);
        //remove trailing spaces, important tesseract don't recognize names if spaces are there
        while((word_text.size() - 1 == word_text.rfind(' ')) && (word_text.size() > 2)){
            word_text.erase(word_text.size() -1);
        }
        //save text cleaned from unnecessary spaces
        words.at(i).m_str = word_text;
    }
    
    //calculate levenshtein distance
    string regex_date = "dD.dD.DDDD";
    string regex_personalnum = "DDDDDD.ddDD";
    for(int i=0; i < words.size(); i++){
        
        process_words_package word_package;
        word_package.levenshtein_distance_date = levenshtein_distance::calculate_levenshtein_distance(words.at(i).m_str, regex_date);
        word_package.levenshtein_distance_personal_num = levenshtein_distance::calculate_levenshtein_distance(words.at(i).m_str, regex_personalnum);
        word_package.is_name = is_name(words.at(i).m_str);
        word_package.word = &(words.at(i));
        process_words_vector.push_back(word_package);
        
    }
    
    int new_order = 0;
    for(int i=0; i < process_words_vector.size(); i++){
        
        if(process_words_vector.at(i).word->m_confidence > 50){
            bool return_this_word = false;
            string type;
            if(process_words_vector.at(i).levenshtein_distance_personal_num <= 3){
                return_this_word = true;
                type = "personal number";
            }
            if(process_words_vector.at(i).levenshtein_distance_date <= 3){
                boost::regex regex_dates_basic("([0-9]{0,2})([-/.,]*)([0-9]{0,2})([-/.,]*)(19[0-9]{2}|20[0-9]{2})");
                boost::smatch res;
                if(boost::regex_search((*(process_words_vector.at(i).word)).m_str, res, regex_dates_basic)){
                    int year = stoi(string(res[5]));
                    if(year < m_actual_year- 20){
                        return_this_word = true;
                    }else{
                        return_this_word = false;
                    }
                }else{
                    return_this_word = true;
                }
                type = "date";
            }
            if(process_words_vector.at(i).is_name){
                return_this_word = true;
                type = "name";
            }
            
            if(return_this_word){
                word_type word = word_type( process_words_vector.at(i).word->m_x,
                                            process_words_vector.at(i).word->m_y,
                                            process_words_vector.at(i).word->m_width,
                                            process_words_vector.at(i).word->m_height,
                                            new_order,
                                            process_words_vector.at(i).word->m_confidence,
                                            string(process_words_vector.at(i).word->m_str.c_str()));
                word.set_type(type);
                ret.push_back(word);
                new_order++;
            }

        }
        
    }
    
    return ret;
}

void tesseract_utils::unscale_position(word_vector & words){
    for(int i=0;i<words.size();i++){
        words.at(i).m_x /= m_config.scale_size;
        words.at(i).m_y /= m_config.scale_size; 
        words.at(i).m_width /= m_config.scale_size;
        words.at(i).m_height /= m_config.scale_size; 
    }
}
/*
Boxa* tesseract_utils::find_all_boxes(){
    std::chrono::time_point<std::chrono::system_clock> time1, time2,time3, time4;
    std::chrono::duration<double> elapsed_seconds;
    cout << "start -----------====================================" <<endl;
    time1 = std::chrono::system_clock::now();
    m_api.Recognize(NULL);
    time2 = std::chrono::system_clock::now();
    elapsed_seconds = time2-time1;
    cout << "correct time??? " << elapsed_seconds.count() << endl;
    cout << "words iterator -------------------------------------" << endl;
    time3 = std::chrono::system_clock::now();
    tesseract::ResultIterator* ri = m_api.GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if(ri != 0){
        do{
            const char* word = ri->GetUTF8Text(level);
            float conf = ri->Confidence(level);
            int x1,y1,x2,y2;
            ri->BoundingBox(level, &x1, &y1, &x2, &y2);
            cout << "word--: " << word << " ; conf: " << conf << " ;box:" << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
            delete[] word;
        }while(ri->Next(level));
    }
    time4 = std::chrono::system_clock::now();
    elapsed_seconds = time4-time3;
    cout << "iteration time!!!!!!!!!! " << elapsed_seconds.count() << " seconds -------------------" << endl;
    elapsed_seconds = time4-time1;
    cout << "all time!!!!!!!!!! " << elapsed_seconds.count()  << " seconds -------------------" << endl;
    return m_api.GetComponentImages(tesseract::RIL_WORD, true, NULL, NULL);
}*/

void tesseract_utils::set_year(int year){
    m_actual_year = year;
}

void tesseract_utils::set_month(int month){
    m_actual_month = month;
}

void tesseract_utils::set_day(int day){
    m_actual_day = day;
}

string tesseract_utils::trim_and_reduce_spaces( const string& s ){
    string result = boost::trim_copy( s );
    while (result.find( "  " ) != result.npos)
        boost::replace_all( result, "  ", " " );
    return result;
}

tesseract_utils::tesseract_worker_trigger::tesseract_worker_trigger(){
}

tesseract_utils::tesseract_worker_trigger::~tesseract_worker_trigger(){

}
/*
int tesseract_utils::tesseract_worker_trigger::size(){
    m_mutex.lock();
    int size;
    size = m_internal_queue.size();
    m_mutex.unlock();
    return size;
}

void tesseract_utils::tesseract_worker_trigger::add(Boxa * item){
    m_mutex.lock();
    for (int i=0;i<item->n;i++){
        m_internal_queue.push_back(boxaGetBox(item, i, L_CLONE));
    }
    m_condv.notify_all();
    m_mutex.unlock();
    boxaDestroy(&item);
}

BOX * tesseract_utils::tesseract_worker_trigger::remove(){
    unique_lock<mutex> m_lock(m_mutex);
    while((m_internal_queue.size() == 0) && still_running){
        m_condv.wait(m_lock);      //wait until something is there
    }
    if(still_running){
        BOX * item = m_internal_queue.front();
        m_internal_queue.pop_front();
        return item;
    }else{
        for(BOX * item : m_internal_queue){
            boxDestroy(&item);
        }
        return NULL;
    }
}
void tesseract_utils::tesseract_worker_trigger::end_queue(){
    still_running = false;
    m_condv.notify_all();
}
*/

void tesseract_utils::tesseract_worker_trigger::triger(){
    m_mutex.lock();
    m_condv.notify_all();
    m_mutex.unlock();
}

void tesseract_utils::tesseract_worker_trigger::can_go(){
    unique_lock<mutex> m_lock(m_mutex);
    m_condv.wait(m_lock);
}

tesseract_utils::tesseract_worker::tesseract_worker(tesseract_worker_trigger & trigger, tesseract_worker_data_queue & out_queue, int num_worker,int all_workers)
                            :m_worker_trigger(trigger), m_output_data_queue(out_queue), m_num_worker(num_worker), m_sum_all_workers(all_workers){
    still_running = true;
}

void tesseract_utils::tesseract_worker::init_api(const char* datapath, string language){
    if (m_private_api.Init(datapath, language.c_str(), tesseract::OEM_TESSERACT_ONLY)) {
        throw string("Could not initialize tesseract api in thread " + to_string(m_num_worker));
    }
}

void tesseract_utils::tesseract_worker::set_image(const unsigned char* imagedata, int width, int height, int bytes_per_pixel, int bytes_per_line){
    m_width = width;
    m_height = height;
    m_private_api.SetImage(imagedata, width, height, bytes_per_pixel, bytes_per_line);
}

void tesseract_utils::tesseract_worker::end_api(){
    m_private_api.End();
}

void tesseract_utils::tesseract_worker::stop_thread(){
    still_running = false;
}

void tesseract_utils::tesseract_worker::thread_function(){
    while(still_running){
        m_worker_trigger.can_go();
        
        word_vector return_words;
        float recognizing_height_base = (m_height / (float)m_sum_all_workers);
        int recognizing_offset = RECOGNIZING_OFFSET * 2;
        int recognizing_start_y_position = (m_num_worker * recognizing_height_base);
        if(m_num_worker == 0){                          //first
            recognizing_offset -= RECOGNIZING_OFFSET;
        }else{
            recognizing_start_y_position -= RECOGNIZING_OFFSET;
        }
        if(m_num_worker + 1 == m_sum_all_workers){      //last
            recognizing_offset -= RECOGNIZING_OFFSET;
            recognizing_start_y_position = m_height - (int(recognizing_height_base) + recognizing_offset);
        }
        
        if(print_message_struct::print_message(print_message_struct::Message_type::order_of_action_more)){
            cout << "thread #" << m_num_worker << "; set recognizing position x=0 y=" <<  recognizing_start_y_position << " width=" << m_width << " height=" << int(recognizing_height_base) + recognizing_offset << endl; 
        }
        
        m_private_api.SetRectangle(0, recognizing_start_y_position, m_width, int(recognizing_height_base) + recognizing_offset);
        
        m_private_api.Recognize(NULL);
        
        tesseract::ResultIterator* ri = m_private_api.GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
        if((ri != 0) && still_running){
            do{
                const char* text = ri->GetUTF8Text(level);
                if(text != NULL){
                    float conf = ri->Confidence(level);
                    int x1,y1,x2,y2;
                    ri->BoundingBox(level, &x1, &y1, &x2, &y2);

                    return_words.push_back(word_type(x1, y1, x2-x1, y2-y1, int(conf), string(text)));

                    delete[] text;
                }
            }while(ri->Next(level) && still_running);
        }
        delete ri;
        m_output_data_queue.add(return_words);
        m_output_data_queue.thread_done();        
    }
}
        

tesseract_utils::tesseract_worker_data_queue::tesseract_worker_data_queue(){
}

tesseract_utils::tesseract_worker_data_queue::~tesseract_worker_data_queue(){
    
}

int tesseract_utils::tesseract_worker_data_queue::size(){
    m_mutex.lock();
    int size = m_internal_queue.size();
    m_mutex.unlock();
    return size;
}

void tesseract_utils::tesseract_worker_data_queue::add(word_vector items){
    unique_lock<mutex> m_lock(m_mutex);
    for(int i=0;i<items.size();i++){
        items.at(i).set_index(m_internal_queue.size());     //set index
        m_internal_queue.push_back(items.at(i));
    }
}

/**
 * 
 * @return whole saved vector
 */
word_vector tesseract_utils::tesseract_worker_data_queue::return_all(){
    unique_lock<mutex> m_lock(m_mutex);
    cout << "returning all words" << endl;
    m_condv.wait(m_lock, [this](){return m_threads_done >= m_threads;});
    word_vector return_vector(m_internal_queue);
    m_internal_queue.clear();
    cout << "return " << return_vector.size() << "words" << endl;
    return return_vector;
}

void tesseract_utils::tesseract_worker_data_queue::clear(){
    unique_lock<mutex> m_lock(m_mutex);
    m_internal_queue.clear();
    m_threads_done = 0;
}

void tesseract_utils::tesseract_worker_data_queue::set_number_threads(int threads){
    unique_lock<mutex> m_lock(m_mutex);
    m_threads = threads;
}

void tesseract_utils::tesseract_worker_data_queue::thread_done(){
    unique_lock<mutex> m_lock(m_mutex);
    m_threads_done++;
    if(m_threads_done >= m_threads){
        m_condv.notify_all();
    }
}
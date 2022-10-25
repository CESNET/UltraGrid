#include "string_view_utils.hpp"

std::string_view tokenize(std::string_view& str, char delim, char quot){
        if(str.empty())
                return {};

        bool escaped = false;

        auto token_begin = str.begin();
        while(token_begin != str.end()){
                if(*token_begin == quot)
                        escaped = !escaped;
                else if(*token_begin != delim)
                        break;

                token_begin++;
        }

        auto token_end = token_begin;
        while(token_end != str.end()){
                if(*token_end == quot){
                        str = std::string_view(token_end, str.end() - token_end);
                        str.remove_prefix(1); //remove the end quote
                        return std::string_view(token_begin, token_end - token_begin);
                }
                else if(*token_end == delim && !escaped)
                        break;

                token_end++;
        }

        str = std::string_view(token_end, str.end() - token_end);

        return std::string_view(token_begin, token_end - token_begin);
}

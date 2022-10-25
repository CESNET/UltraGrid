/**
 * @file   utils/string_view_utils.hpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET z.s.p.o.
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

#ifndef STRING_VIEW_UTILS_183e2da27fe9
#define STRING_VIEW_UTILS_183e2da27fe9

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <stdlib.h>

/**
 * @brief Tokenizer for string_view
 *
 * Useful for non-destructive tokenization of strings. Skips empty tokens.
 * str is modified to view the not yet processed remainder.
 *
 * Typical usage (prints lines "Hello", "World", "!"):
 * ~~~~~~~~~~~~~~~~~~~{.cpp}
 * std::string_view sv = ":::Hello:World::!::";
 * while(!sv.empty()){
 *     cout << tokenize(sv, ':') << "\n";
 * }
 * ~~~~~~~~~~~~~~~~~~~
 *
 * The 'quot' param allows for a basic quotation based deilimiter escape,
 * however the whole token must be escaped.
 * i.e. sync:"opts=a=1:b=2":fs is fine, but sync:opts="a=1:b=2":fs is not.
 */
std::string_view tokenize(std::string_view& str, char delim, char quot = '\0');

inline bool sv_contains(std::string_view haystack, std::string_view needle){
	return haystack.find(needle) != std::string_view::npos;
}

/* std::from_chars is c++17, but on major compilers the feature was missing or 
 * partial (no floating point support). This header provides safe fallbacks
 * if the particular implementation lacks std::from_chars for a given type.
 *
 * std::from_chars support:
 * GCC - partial since v8, floating since v11
 * Clang - partial since 7
 */
#if __has_include(<charconv>)
#include <charconv>

template<typename T, typename = void>
struct from_chars_available : std::false_type {  };

template<typename T>
struct from_chars_available<T,
        std::void_t<decltype(std::from_chars(std::declval<const char *>(), std::declval<const char *>(), std::declval<T&>()))>>
        : std::true_type {  };

template<typename T>
inline constexpr bool from_chars_available_v = from_chars_available<T>::value;

template<typename T, std::enable_if_t<from_chars_available_v<T>, bool> = true,
        std::enable_if_t<!std::is_floating_point_v<T>, bool> = true>
bool parse_num(std::string_view sv, T& res, int base = 10){
        return !sv.empty() && std::from_chars(sv.begin(), sv.end(), res, base).ec == std::errc();
}

template<typename T, std::enable_if_t<from_chars_available_v<T>, bool> = true,
        std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
bool parse_num(std::string_view sv, T& res){
        return !sv.empty() && std::from_chars(sv.begin(), sv.end(), res).ec == std::errc();
}

#else

template<typename T>
inline constexpr bool from_chars_available_v = false;

#endif // has_include(<charconv>)

template<typename T, auto convert, typename... Args>
bool parse_num_fallback_impl(std::string_view sv, T& res, Args... args){
        if(sv.empty())
                return false;

        std::string tmp(sv);
        const char *c_str_ptr = tmp.c_str();
        char *endptr = nullptr;
        T num;

        num = convert(c_str_ptr, &endptr, args...);

        if(c_str_ptr == endptr)
                return false;

        res = num;
        return true;
}

template<typename T,
        std::enable_if_t<!from_chars_available_v<T>, bool> = true,
        std::enable_if_t<std::is_integral_v<T>, bool> = true>
bool parse_num(std::string_view sv, T& res, int base = 10){
        return parse_num_fallback_impl<T, strtol>(sv, res, base);
}

template<typename T,
        std::enable_if_t<!from_chars_available_v<T>, bool> = true,
        std::enable_if_t<std::is_same_v<T, double>, bool> = true>
bool parse_num(std::string_view sv, T& res){
        return parse_num_fallback_impl<T, strtod>(sv, res);
}

template<typename T,
        std::enable_if_t<!from_chars_available_v<T>, bool> = true,
        std::enable_if_t<std::is_same_v<T, float>, bool> = true>
bool parse_num(std::string_view sv, T& res){
        return parse_num_fallback_impl<T, strtof>(sv, res);
}

#endif

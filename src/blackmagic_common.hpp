/**
 * @file   blackmagic_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2020 CESNET, z. s. p. o.
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

#ifndef BLACKMAGIC_COMMON_HPP
#define BLACKMAGIC_COMMON_HPP

#ifdef WIN32
#include "DeckLinkAPI_h.h" /*  From DeckLink SDK */
#else
#include "DeckLinkAPI.h" /*  From DeckLink SDK */
#endif

#include <cstdbool>
#include <cstdint>
#include <map>
#include <vector>
#include <string>
#include <utility>

#include "utils/macros.h"
#include "video.h"

std::string bmd_hresult_to_string(HRESULT res);

// Order of codecs is important because it is used as a preference list (upper
// codecs are favored) returned by DISPLAY_PROPERTY_CODECS property (display)
static std::vector<std::pair<codec_t, BMDPixelFormat>> uv_to_bmd_codec_map = {
                  { R12L, bmdFormat12BitRGBLE },
                  { R10k, bmdFormat10BitRGBX },
                  { v210, bmdFormat10BitYUV },
                  { RGBA, bmdFormat8BitBGRA },
                  { UYVY, bmdFormat8BitYUV },
};

#ifdef WIN32
#define BMD_BOOL BOOL
#define BMD_TRUE TRUE
#define BMD_FALSE FALSE
#else
#define BMD_BOOL bool
#define BMD_TRUE true
#define BMD_FALSE false
#endif

#define BMD_OPT_DEFAULT 0 ///< default is set to 0 to allow zero-initialization
// below are special values not known to BMD that must be interpreted by UG
// and *must*not*be*passed*to*BMD*API !
#define BMD_OPT_FALSE   to_fourcc('T', 'R', 'U', 'E')
#define BMD_OPT_TRUE    to_fourcc('F', 'A', 'L', 'S')
#define BMD_OPT_KEEP    to_fourcc('K', 'E', 'E', 'P')

#ifdef HAVE_MACOSX
#define BMD_STR CFStringRef
#elif defined WIN32
#define BMD_STR BSTR
#else
#define BMD_STR const char *
#endif
char *get_cstr_from_bmd_api_str(BMD_STR string);
BMD_STR get_bmd_api_str_from_cstr(const char *cstr);
void release_bmd_api_str(BMD_STR string);
#ifdef __cplusplus
#include <string>
std::string get_str_from_bmd_api_str(BMD_STR string);
#endif

IDeckLinkIterator *create_decklink_iterator(bool verbose = true, bool coinit = true);
bool decklink_initialize();
void decklink_uninitialize();
bool blackmagic_api_version_check();
void print_decklink_version(void);

bool bmd_check_stereo_profile(IDeckLink *deckLink);
bool decklink_set_duplex(IDeckLink *decklink, uint32_t profileID);
std::string bmd_get_device_name(IDeckLink *decklink);
std::string bmd_get_audio_connection_name(BMDAudioOutputAnalogAESSwitch audioConnection);
uint32_t bmd_read_fourcc(const char *);
int parse_bmd_flag(const char *val);
int invert_bmd_flag(int val);
void r10k_limited_to_full(const char *in, char *out, size_t len);
void r10k_full_to_limited(const char *in, char *out, size_t len);

std::ostream &operator<<(std::ostream &output, REFIID iid);

/// @param type - Int of Flag
/// @param fatal - exit if failed to set (usually set to true for explicit settings, false implicit)
#define BMD_CONFIG_SET(type, key, val, fatal) do {\
        if (val != (decltype(val)) BMD_OPT_DEFAULT && val != (decltype(val)) BMD_OPT_KEEP) {\
                HRESULT result = deckLinkConfiguration->Set##type(key, val);\
                if (result != S_OK) {\
                        LOG(fatal ? LOG_LEVEL_ERROR : LOG_LEVEL_WARNING) << MOD_NAME << "Unable to set " #key ": " << bmd_hresult_to_string(result) << "\n";\
                        if (fatal) { \
                                goto error;\
                        } \
                } else { \
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << #key << " set to: " << val << "\n";\
                } \
        }\
} while (0)

/// action to BMD_CONFIG_SET_ACTION if no operation should be performed if error
#define BMD_NO_ACTION
/// same as BMD_CONFIG_SET, uses action instead of fatal
#define BMD_CONFIG_SET_ACTION(type, key, val, action) do {\
        if (val != (decltype(val)) BMD_OPT_DEFAULT && val != (decltype(val)) BMD_OPT_KEEP) {\
                HRESULT result = deckLinkConfiguration->Set##type(key, val);\
                if (result != S_OK) {\
                        LOG(strcmp(#action, "BMD_NO_ACTION") == 0 ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR) << MOD_NAME << "Unable to set " #key ": " << bmd_hresult_to_string(result) << "\n";\
                        action; \
                } else { \
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << #key << " set to: " << val << "\n";\
                } \
        }\
} while (0)

#define BMD_CHECK(cmd, name, action) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                        action;\
                }\
        } while (0)


#define R10K_FULL_OPT "bmd-r10k-full-range"

#endif // defined BLACKMAGIC_COMMON_HPP


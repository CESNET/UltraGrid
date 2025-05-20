/**
 * @file   blackmagic_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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

#ifdef _WIN32
#include <winsock2.h> // needs to be included prior to windows.h which will be
                      // included by foolowing include; wsock2.h needed for
                      // htonl used by vcap/vdisp
#endif

#include "DeckLinkAPI.h" /*  From DeckLink SDK */

#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <type_traits>
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

#ifdef _WIN32
#define BMD_BOOL BOOL
#define BMD_TRUE TRUE
#define BMD_FALSE FALSE
#else
#define BMD_BOOL bool
#define BMD_TRUE true
#define BMD_FALSE false
#endif

enum {
        BMD_MAX_AUD_CH = 64,
};

struct bmd_option {
private:
        enum class type_tag : int { t_default, t_keep, t_flag, t_int, t_float, t_string } m_type = type_tag::t_default;
        union {
                bool b;
                int64_t i;
                double f;
                char s[128];
        } m_val{};
        bool m_user_specified = true; ///< ignore errors if set to false
        friend std::ostream &operator<<(std::ostream &output, const bmd_option &b);
public:
        bmd_option() = default;
        explicit bmd_option(int64_t val, bool user_spec = true);
        explicit bmd_option(bool val, bool user_spec = true);
        bool is_default() const;
        [[nodiscard]] bool is_help() const;
        bool is_user_set() const;
        void set_keep();
        bool keep() const;
        bool get_flag() const;
        int64_t get_int() const;
        [[nodiscard]] const char *get_string() const;
        bool parse(const char *);
        void set_flag(bool val_);
        void set_int(int64_t val_);
        void set_float(double val_);
        void set_string(const char *val_);
        bool device_write(IDeckLinkConfiguration *deckLinkConfiguration, BMDDeckLinkConfigurationID opt, std::string const &log_prefix = "[DeckLink] ") const;
};

std::ostream &operator<<(std::ostream &output, const bmd_option &b);

#ifdef __APPLE__
#define BMD_STR CFStringRef
#elif defined _WIN32
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
std::string bmd_get_flags_str(BMDDisplayModeFlags flags);
#endif

IDeckLinkIterator *create_decklink_iterator(bool *com_initialized, bool verbose = true);
/// @param com_initialized - pointer passed to create_decklink_iterator
void decklink_uninitialize(bool *com_initialized);
bool blackmagic_api_version_check();
void print_decklink_version(void);

bool bmd_check_stereo_profile(IDeckLink *deckLink);
bool decklink_set_profile(IDeckLink *deckLink, bmd_option const &req_profile, bool stereo);
std::string bmd_get_device_name(IDeckLink *decklink);
std::string bmd_get_audio_connection_name(BMDAudioOutputAnalogAESSwitch audioConnection);
uint32_t bmd_read_fourcc(const char *);
void r10k_limited_to_full(const char *in, char *out, size_t len);
void r10k_full_to_limited(const char *in, char *out, size_t len);
void print_bmd_device_profiles(const char *line_prefix);
const std::map<BMDVideoConnection, std::string> &get_connection_string_map();

std::ostream &operator<<(std::ostream &output, REFIID iid);

#define BMD_CHECK(cmd, name, action) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                        action;\
                }\
        } while (0)

#define BMD_NOOP ///<  can be passed to @ref BMD_CHECK

#define R10K_FULL_OPT "bmd-r10k-full-range"
#define BMD_NAT_SORT "bmd-sort-natural"

template <typename T>
bool decklink_supports_codec(T *deckLink, BMDPixelFormat pf);
extern template bool
decklink_supports_codec<IDeckLinkOutput>(IDeckLinkOutput *deckLink,
                                         BMDPixelFormat   pf);
extern template bool
decklink_supports_codec<IDeckLinkInput>(IDeckLinkInput *deckLink,
                                        BMDPixelFormat  pf);
bool bmd_parse_audio_levels(const char *opt) noexcept(false);
void print_bmd_attribute(IDeckLinkProfileAttributes *deckLinkAttributes,
                         const char                 *query_prop_fcc);
void print_bmd_connections(IDeckLinkProfileAttributes *deckLinkAttributes,
                           BMDDeckLinkAttributeID      id,
                           const char                 *module_prefix);
BMDVideoConnection bmd_get_connection_by_name(const char *connection);

/**
 * @details parameters:
 * - first - IDeckLinkInstance
 * - unsigned - topological ID
 * - int - natural idx (the old, order as given by IDeckLinkIterator)
 * - char - new index ('a', 'b', 'c'...); sorted by topological ID
 */
using bmd_dev = std::tuple<std::unique_ptr<IDeckLink, void (*)(IDeckLink *)>,
                           unsigned, int, char>;
std::vector<bmd_dev> bmd_get_sorted_devices(bool *com_initialized,
                                            bool  verbose      = true,
                                            bool  natural_sort = false);
class BMDNotificationCallback;
BMDNotificationCallback *
bmd_print_status_subscribe_notify(IDeckLink *deckLink, const char *log_prefix,
                                  bool capture);
void bmd_unsubscribe_notify(BMDNotificationCallback *notificationCallback);

bool bmd_parse_fourcc_arg(
    std::map<BMDDeckLinkConfigurationID, bmd_option> &device_options,
    const char                                       *arg);
void bmd_options_validate(
    std::map<BMDDeckLinkConfigurationID, bmd_option> &device_options);

#endif // defined BLACKMAGIC_COMMON_HPP


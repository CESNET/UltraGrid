/**
 * @file   blackmagic_common.cpp
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <algorithm>
#include <condition_variable>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "DeckLinkAPIVersion.h"
#include "blackmagic_common.hpp"
#include "debug.h"
#include "host.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/windows.h"
#include "utils/worker.h"

#define MOD_NAME "[DeckLink] "

using std::clamp;
using std::fixed;
using std::hex;
using std::invalid_argument;
using std::map;
using std::min;
using std::pair;
using std::ostringstream;
using std::setfill;
using std::setw;
using std::stod;
using std::stoi;
using std::string;
using std::uppercase;
using std::unordered_map;
using std::vector;

string bmd_hresult_to_string(HRESULT res)
{
        const char *errptr = nullptr;
#ifdef _WIN32
        errptr = hresult_to_str(res);
#else
        HRESULT_GET_ERROR_COMMON(res, errptr)
#endif
        ostringstream oss;
        if (errptr) {
                oss << errptr;
        }
        oss << " " << "(0x" << hex << setfill('0') << setw(8) << res << ")";
        return oss.str();
}

/**
 * returned c-sring needs to be freed when not used
 */
char *get_cstr_from_bmd_api_str(BMD_STR bmd_string)
{
        if (!bmd_string) {
                return strdup("(NULL!)");
        }
       char *cstr;
#ifdef HAVE_MACOSX
       size_t len = CFStringGetMaximumSizeForEncoding(CFStringGetLength(bmd_string), kCFStringEncodingUTF8) + 1;
       cstr = (char *) malloc(len);
       CFStringGetCString(bmd_string, (char *) cstr, len, kCFStringEncodingUTF8);
#elif defined WIN32
       size_t len = SysStringLen(bmd_string) * 4 + 1;
       cstr = (char *) malloc(len);
       wcstombs((char *) cstr, bmd_string, len);
#else // Linux
       cstr = strdup(bmd_string);
#endif

       return cstr;
}

BMD_STR get_bmd_api_str_from_cstr(const char *cstr)
{
#ifdef __APPLE__
        return CFStringCreateWithCString(kCFAllocatorMalloc, cstr, kCFStringEncodingUTF8);
#elif defined _WIN32
        mbstate_t mbstate{};
        const char *tmp = cstr;
        size_t required_size = mbsrtowcs(NULL, &tmp, 0, &mbstate) + 1;
        BMD_STR out = (wchar_t *) malloc(required_size * sizeof(wchar_t));
        mbsrtowcs(out, &tmp, required_size, &mbstate);
	return out;
#else
        return strdup(cstr);
#endif
}

void release_bmd_api_str(BMD_STR string)
{
        if (!string) {
                return;
        }
#ifdef HAVE_MACOSX
        CFRelease(string);
#elif defined WIN32
        SysFreeString(string);
#else
        free(const_cast<char *>(string));
#endif
}

std::string get_str_from_bmd_api_str(BMD_STR string)
{
        char *displayModeCString = get_cstr_from_bmd_api_str(string);
        std::string out = displayModeCString;
        free(displayModeCString);
        return out;
}

/**
 * @note
 * Each successful call (returning non-null pointer) of this function with coinit == true
 * should be followed by com_uninitialize() when done with DeckLink (not when releasing
 * IDeckLinkIterator!), typically on application shutdown.
 */
IDeckLinkIterator *create_decklink_iterator(bool *com_initialized, bool verbose, bool coinit)
{
        IDeckLinkIterator *deckLinkIterator = nullptr;
#ifdef WIN32
        if (coinit) {
                com_initialize(com_initialized, "[BMD] ");
        }
        HRESULT result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
                        IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result)) {
                decklink_uninitialize(com_initialized);
                deckLinkIterator = nullptr;
        }
#else
        UNUSED(coinit);
        *com_initialized = false;
        deckLinkIterator = CreateDeckLinkIteratorInstance();
#endif

        if (!deckLinkIterator && verbose) {
                log_msg(LOG_LEVEL_ERROR, "A DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
                log_msg(LOG_LEVEL_INFO, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);

        }

        return deckLinkIterator;
}

void decklink_uninitialize(bool *com_initialized)
{
        com_uninitialize(com_initialized);
}

bool blackmagic_api_version_check()
{
        bool ret = false;
        IDeckLinkAPIInformation *APIInformation = NULL;
        HRESULT result;
        bool com_initialized = false;

        if (!com_initialize(&com_initialized, "[BMD] ")) {
                goto cleanup;
        }
#ifdef WIN32
        result = CoCreateInstance(CLSID_CDeckLinkAPIInformation, NULL, CLSCTX_ALL,
                IID_IDeckLinkAPIInformation, (void **) &APIInformation);
        if(FAILED(result)) {
#else
        APIInformation = CreateDeckLinkAPIInformationInstance();
        if(APIInformation == NULL) {
#endif
                log_msg(LOG_LEVEL_ERROR, "Cannot get API information! Perhaps drivers not installed.\n");
                goto cleanup;
        }
        int64_t value;
        result = APIInformation->GetInt(BMDDeckLinkAPIVersion, &value);
        if(result != S_OK) {
                log_msg(LOG_LEVEL_ERROR, "Cannot get API version!\n");
                goto cleanup;
        }

        if (BLACKMAGIC_DECKLINK_API_VERSION > value) { // this is safe comparision, for internal structure please see SDK documentation
                log_msg(LOG_LEVEL_ERROR, "The DeckLink drivers may not be installed or are outdated.\n");
                log_msg(LOG_LEVEL_ERROR, "You should have at least the version UltraGrid has been linked with.\n");
                log_msg(LOG_LEVEL_ERROR, "Vendor download page is http://www.blackmagic-design.com/support\n");
                print_decklink_version();
                ret = false;
        } else {
                ret = true;
        }

cleanup:
        if (APIInformation) {
                APIInformation->Release();
        }
        decklink_uninitialize(&com_initialized);

        return ret;
}

void print_decklink_version()
{
        BMD_STR current_version = NULL;
        IDeckLinkAPIInformation *APIInformation = NULL;
        HRESULT result;

#ifdef WIN32
        bool com_initialized = false;
        if (!com_initialize(&com_initialized, "[BMD] ")) {
                goto cleanup;
        }

        result = CoCreateInstance(CLSID_CDeckLinkAPIInformation, NULL, CLSCTX_ALL,
                IID_IDeckLinkAPIInformation, (void **) &APIInformation);
        if(FAILED(result)) {
#else
        APIInformation = CreateDeckLinkAPIInformationInstance();
        if(APIInformation == NULL) {
#endif
                fprintf(stderr, "Cannot get API information! Perhaps drivers not installed.\n");
                goto cleanup;
        }

        result = APIInformation->GetString(BMDDeckLinkAPIVersion, &current_version);
        if (result != S_OK) {
                fprintf(stderr, "Cannot get API version string!\n");
                goto cleanup;
        } else {
                fprintf(stderr, "This UltraGrid version was compiled against DeckLink SDK %s. ", BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                const char *currentVersionCString = get_cstr_from_bmd_api_str(current_version);
                fprintf(stderr, "System version is %s.\n", currentVersionCString);
                release_bmd_api_str(current_version);
                free(const_cast<char *>(currentVersionCString));
        }

cleanup:
        if (APIInformation) {
                APIInformation->Release();
        }
#ifdef WIN32
        com_uninitialize(&com_initialized);
#endif
}

// Profile description map
static const map<BMDProfileID, pair<const char *, const char *>> kDeviceProfiles =
{
        { bmdProfileOneSubDeviceFullDuplex,   { "1 sub-device full-duplex", "8K Pro, Duo 2, Quad 2" } },
        { bmdProfileOneSubDeviceHalfDuplex,   { "1 sub-device half-duplex", "8K Pro" } },
        { bmdProfileTwoSubDevicesFullDuplex,  { "2 sub-devices full-duplex", "8K Pro" } },
        { bmdProfileTwoSubDevicesHalfDuplex,  { "2 sub-devices half-duplex", "Duo 2, Quad 2" } },
        { bmdProfileFourSubDevicesHalfDuplex, { "4 sub-devices half-duplex", "8K Pro" } },
};

#define EXIT_IF_FAILED(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
			ret = {};\
			goto cleanup;\
                }\
        } while (0)

#define RELEASE_IF_NOT_NULL(x) if (x != nullptr) { x->Release(); x = nullptr; }

static BMDProfileID GetDeckLinkProfileID(IDeckLinkProfile* profile)
{
        IDeckLinkProfileAttributes*             profileAttributes = nullptr;
        if (HRESULT result = profile->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&profileAttributes); FAILED(result)) {
                return {};
        }

        int64_t profileIDInt = 0;
        // Get Profile ID attribute
        const HRESULT res =
            profileAttributes->GetInt(BMDDeckLinkProfileID, &profileIDInt);
        if (SUCCEEDED(res)) {
                profileAttributes->Release();
        } else {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "BMDDeckLinkProfileID: "
                                     << bmd_hresult_to_string(res) << "\n";
        }
        return (BMDProfileID) profileIDInt;
}

class ProfileCallback : public IDeckLinkProfileCallback
{
        public:
                ProfileCallback(IDeckLinkProfile *requestedProfile) : m_requestedProfile(requestedProfile) {
                        m_requestedProfile->AddRef();
                }

                HRESULT ProfileChanging (/* in */ [[maybe_unused]] IDeckLinkProfile* profileToBeActivated, /* in */ [[maybe_unused]] BMD_BOOL streamsWillBeForcedToStop) override { return S_OK; }
                HRESULT ProfileActivated (/* in */ [[maybe_unused]] IDeckLinkProfile* activatedProfile) override {
                        {
                                std::lock_guard<std::mutex> lock(m_profileActivatedMutex);
                                m_requestedProfileActivated = true;
                        }
                        m_profileActivatedCondition.notify_one();
                        return S_OK;
                }
                HRESULT STDMETHODCALLTYPE QueryInterface([[maybe_unused]] REFIID iid, [[maybe_unused]] LPVOID *ppv) override
                {
                        *ppv = nullptr;
                        return E_NOINTERFACE;
                }
                ULONG   STDMETHODCALLTYPE       AddRef() override { return ++m_refCount; }
                ULONG   STDMETHODCALLTYPE       Release() override {
                        ULONG refCount = --m_refCount;
                        if (refCount == 0)
                                delete this;

                        return refCount;
                }

                bool WaitForProfileActivation(void) {
                        BMD_BOOL isActiveProfile = BMD_FALSE;
                        const char *profileName = kDeviceProfiles.find(GetDeckLinkProfileID(m_requestedProfile)) != kDeviceProfiles.end() ?
                                kDeviceProfiles.at(GetDeckLinkProfileID(m_requestedProfile)).first : "(unknown)";
                        if ((m_requestedProfile->IsActive(&isActiveProfile) == S_OK) && isActiveProfile) {
                                LOG(LOG_LEVEL_INFO) << "[DeckLink] Profile " << profileName << " already active.\n";
                                return true;
                        }

                        LOG(LOG_LEVEL_INFO) << "[DeckLink] Waiting for profile activation... (this may take few seconds)\n";
                        std::unique_lock<std::mutex> lock(m_profileActivatedMutex);
                        bool ret =  m_profileActivatedCondition.wait_for(lock, std::chrono::seconds{5}, [&]{ return m_requestedProfileActivated; });
                        if (ret) {
                                LOG(LOG_LEVEL_NOTICE) << "[DeckLink] Profile " << profileName << " activated succesfully.\n";
                        } else {
                                LOG(LOG_LEVEL_ERROR) << "[DeckLink] Profile " << profileName << " activation timeouted!\n";
                        }
                        return ret;
                }
                virtual ~ProfileCallback() {
                        m_requestedProfile->Release();
                }

        private:
                IDeckLinkProfile *m_requestedProfile;
                int m_refCount = 1;
                std::condition_variable m_profileActivatedCondition;
                std::mutex              m_profileActivatedMutex;
                bool m_requestedProfileActivated = false;
};

/**
 * @param a value from BMDProfileID or bmdDuplexHalf (maximize number of IOs)
 */
bool decklink_set_profile(IDeckLink *deckLink, bmd_option const &req_profile, bool stereo) {
        if (req_profile.is_default() && !stereo) {
                return true;
        }

        bool ret = true;
        IDeckLinkProfileManager *manager = nullptr;
        IDeckLinkProfileIterator *it = nullptr;
        IDeckLinkProfile *profile = nullptr;
        bool found = false;
        ProfileCallback *p = nullptr;

        if (HRESULT res = deckLink->QueryInterface(IID_IDeckLinkProfileManager, (void**)&manager)) {
                const bool error = !(req_profile.is_default() && res == E_NOINTERFACE);
                LOG(error ? LOG_LEVEL_ERROR : LOG_LEVEL_VERBOSE) << MOD_NAME << "Cannot set duplex - query profile manager: " << bmd_hresult_to_string(res) << "\n";
                return error;
        }

        const uint32_t profileID = req_profile.is_default() ? (int64_t) bmdProfileOneSubDeviceFullDuplex : req_profile.get_int();

        EXIT_IF_FAILED(manager->GetProfiles(&it), "Cannot set duplex - get profiles");

        while (it->Next(&profile) == S_OK) {
                IDeckLinkProfileAttributes *attributes;
                int64_t id;
                if (profile->QueryInterface(IID_IDeckLinkProfileAttributes,
                                        (void**)&attributes) != S_OK) {
                        LOG(LOG_LEVEL_WARNING) << "[DeckLink] Cannot get profile attributes!\n";
                        continue;
                }
                if (attributes->GetInt(BMDDeckLinkProfileID, &id) == S_OK) {
                        if (profileID == bmdDuplexHalf) {
                                if (id == bmdProfileTwoSubDevicesHalfDuplex || id == bmdProfileFourSubDevicesHalfDuplex) {
                                        found = true;
                                }
                        } else if (profileID == id) {
                                found = true;
                        }
                        if (found) {
                                p = new ProfileCallback(profile);
                                BMD_CHECK(manager->SetCallback(p), "IDeckLinkProfileManager::SetCallback", goto cleanup);
                                if (profile->SetActive() != S_OK) {
                                        LOG(LOG_LEVEL_ERROR) << "[DeckLink] Cannot set profile!\n";
                                        ret = false;
                                }
                                if (!p->WaitForProfileActivation()) {
                                        ret = false;
                                }
                        }
                } else {
                        LOG(LOG_LEVEL_WARNING) << "[DeckLink] Cannot get profile ID!\n";
                }
                attributes->Release();
                profile->Release();
                if (found) {
                        break;
                }
        }

        if (!found && ret) { // no err but not found
                LOG(LOG_LEVEL_WARNING) << "[DeckLink] did not find suitable duplex profile!\n";
                ret = false;
        }

cleanup:
        RELEASE_IF_NOT_NULL(p);
        RELEASE_IF_NOT_NULL(it);
        RELEASE_IF_NOT_NULL(manager);
	return ret;
}

static BMDProfileID decklink_get_active_profile_id(IDeckLink *decklink)
{
        BMDProfileID ret{};
        IDeckLinkProfileManager *manager = nullptr;

        if (HRESULT result = decklink->QueryInterface(IID_IDeckLinkProfileManager, (void**)&manager); FAILED(result)) {
                if (result != E_NOINTERFACE) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot get IDeckLinkProfileManager: " << bmd_hresult_to_string(result) << "\n";
                }
                return {};
        }

        IDeckLinkProfileIterator *it = nullptr;
        IDeckLinkProfile *profile = nullptr;
        EXIT_IF_FAILED(manager->GetProfiles(&it), "Cannot get profiles iterator");
        while (it->Next(&profile) == S_OK) {
                BMD_BOOL isActiveProfile = BMD_FALSE;
                if ((profile->IsActive(&isActiveProfile) == S_OK) && isActiveProfile) {
                        ret = GetDeckLinkProfileID(profile);
                        profile->Release();
                        break;
                }
                profile->Release();
        }

cleanup:
        RELEASE_IF_NOT_NULL(it);
        RELEASE_IF_NOT_NULL(manager);
        return ret;
}

bool bmd_check_stereo_profile(IDeckLink *deckLink) {
        if (BMDProfileID profile_active = decklink_get_active_profile_id(deckLink)) {
                if (profile_active != bmdProfileOneSubDeviceHalfDuplex &&
                                profile_active != bmdProfileOneSubDeviceFullDuplex) {
                        uint32_t profile_fcc_host = ntohl(profile_active);
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Active profile '%.4s' may not be compatible with stereo mode.\n", (char *) &profile_fcc_host);
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Use 'profile=' parameter to set 1-subdevice mode in either '1dhd' (half) or '1dfd' (full) duplex.\n");
                }
                return false;
        }
        return true;
}

string bmd_get_device_name(IDeckLink *decklink) {
	BMD_STR         deviceNameString = NULL;
	char *          deviceNameCString = NULL;
	string 		ret;

	if (decklink->GetDisplayName((BMD_STR *) &deviceNameString) == S_OK) {
		deviceNameCString = get_cstr_from_bmd_api_str(deviceNameString);
		ret = deviceNameCString;
		release_bmd_api_str(deviceNameString);
		free(deviceNameCString);
	}

        return ret;
}

uint32_t bmd_read_fourcc(const char *str) {
        union {
                uint32_t fourcc;
                char c4[4];
        } u;
        memset(u.c4, ' ', 4);
        memcpy(u.c4, str, min(strlen(str), sizeof u.c4));
        return htonl(u.fourcc);
}

std::ostream &operator<<(std::ostream &output, REFIID iid)
{
#ifdef _WIN32
        OLECHAR* guidString;
        StringFromCLSID(iid, &guidString);
        char buffer[128];
        int ret = wcstombs(buffer, guidString, sizeof buffer);
        if (ret == sizeof buffer) {
                buffer[sizeof buffer - 1] = '\0';
        }
        output << buffer;
        ::CoTaskMemFree(guidString);
#else
        auto flags = output.flags();
        output << hex << uppercase << setfill('0') <<
                setw(2) << static_cast<int>(iid.byte0) << setw(2) << static_cast<int>(iid.byte1) <<
                setw(2) << static_cast<int>(iid.byte2) << setw(2) << static_cast<int>(iid.byte3) << "-" <<
                setw(2) << static_cast<int>(iid.byte4) << setw(2) << static_cast<int>(iid.byte5) << "-" <<
                setw(2) << static_cast<int>(iid.byte6) << setw(2) << static_cast<int>(iid.byte7) << "-" <<
                setw(2) << static_cast<int>(iid.byte8) << setw(2) << static_cast<int>(iid.byte9) << "-" <<
                setw(2) << static_cast<int>(iid.byte10) << setw(2) << static_cast<int>(iid.byte11) <<
                setw(2) << static_cast<int>(iid.byte12) << setw(2) << static_cast<int>(iid.byte13) <<
                setw(2) << static_cast<int>(iid.byte14) << setw(2) << static_cast<int>(iid.byte15);
        output.flags(flags);
#endif
        return output;
}

static string fcc_to_string(uint32_t fourcc) {
#define BMDFCC(x) {x,#x}
        static const unordered_map<uint32_t, const char *> conf_name_map = {
                BMDFCC(bmdVideo3DPackingSidebySideHalf), BMDFCC(bmdVideo3DPackingLinebyLine), BMDFCC(bmdVideo3DPackingTopAndBottom), BMDFCC(bmdVideo3DPackingFramePacking), BMDFCC(bmdVideo3DPackingRightOnly), BMDFCC(bmdVideo3DPackingLeftOnly),
                BMDFCC(bmdDeckLinkCapturePassthroughModeDisabled),
                BMDFCC(bmdDeckLinkCapturePassthroughModeCleanSwitch),
                BMDFCC(bmdDeckLinkConfig444SDIVideoOutput),
                BMDFCC(bmdDeckLinkConfigCapture1080pAsPsF),
                BMDFCC(bmdDeckLinkConfigCapturePassThroughMode),
                BMDFCC(bmdDeckLinkConfigFieldFlickerRemoval),
                BMDFCC(bmdDeckLinkConfigLowLatencyVideoOutput),
                BMDFCC(bmdDeckLinkConfigHDMI3DPackingFormat),
                BMDFCC(bmdDeckLinkConfigOutput1080pAsPsF),
                BMDFCC(bmdDeckLinkConfigQuadLinkSDIVideoOutputSquareDivisionSplit),
                BMDFCC(bmdDeckLinkConfigSDIOutputLinkConfiguration),
                BMDFCC(bmdDeckLinkConfigSMPTELevelAOutput),
                BMDFCC(bmdDeckLinkConfigVideoInputConnection),
                BMDFCC(bmdDeckLinkConfigVideoInputConversionMode),
                BMDFCC(bmdDeckLinkConfigVideoOutputConversionMode),
                BMDFCC(bmdDeckLinkConfigVideoOutputIdleOperation),
                BMDFCC(bmdIdleVideoOutputLastFrame),
                BMDFCC(bmdLinkConfigurationSingleLink), BMDFCC(bmdLinkConfigurationDualLink), BMDFCC(bmdLinkConfigurationQuadLink),
        };
#undef BMDFCC
        if (auto it = conf_name_map.find(fourcc); it != conf_name_map.end()) {
                return it->second;
        }
        union {
                char c[5];
                uint32_t i;
        } fcc{};
        fcc.i = htonl(fourcc);
        return string("'") + fcc.c + "'";
}

bmd_option::bmd_option(bool val, bool user_spec) : m_type(type_tag::t_flag), m_user_specified(user_spec) {
        m_val.b = val;
}

bmd_option::bmd_option(int64_t val, bool user_spec) : m_type(type_tag::t_int), m_user_specified(user_spec) {
        m_val.i = val;
}

std::ostream &operator<<(std::ostream &output, const bmd_option &b) {
        switch (b.m_type) {
                case bmd_option::type_tag::t_default:
                        output << "(default)";
                        break;
                case bmd_option::type_tag::t_keep:
                        output << "(keep)";
                        break;
                case bmd_option::type_tag::t_flag:
                        output << (b.get_flag() ? "true" : "false");
                        break;
                case bmd_option::type_tag::t_int:
                        if (IS_FCC(b.get_int())) {
                                output << fcc_to_string(b.get_int());
                        } else if (b.get_int() >= 0) {
                                auto flags = output.flags();
                                output << b.get_int() << " (0x" << hex
                                       << b.get_int() << ")";
                                output.flags(flags);
                        } else {
                                output << b.get_int();
                        }
                        break;
                case bmd_option::type_tag::t_float: {
                        auto flags = output.flags();
                        output << fixed << b.m_val.f;
                        output.flags(flags);
                        break;
                }
                case bmd_option::type_tag::t_string:
                        output << b.m_val.s;
                        break;
        }
        return output;
}

void bmd_option::set_flag(bool val_) {
        m_val.b = val_;
        m_type = type_tag::t_flag;
}
void bmd_option::set_int(int64_t val_) {
        m_val.i = val_;
        m_type = type_tag::t_int;
        m_user_specified = true;
}
void bmd_option::set_float(double val_) {
        m_val.f = val_;
        m_type = type_tag::t_float;
        m_user_specified = true;
}
void bmd_option::set_string(const char *val_) {
        strncpy(m_val.s, val_, sizeof m_val.s - 1);
        m_type = type_tag::t_string;
        m_user_specified = true;
}
void bmd_option::set_keep() {
        m_type = type_tag::t_keep;
}
bool bmd_option::keep() const {
        return m_type == type_tag::t_keep;
}
bool bmd_option::get_flag() const {
        if (m_type != type_tag::t_flag) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Option is not set to a flag but get_flag() called! Current type tag: %d\n", (int) m_type);
                return {};
        }
        return m_val.b;
}
int64_t bmd_option::get_int() const {
        if (m_type != type_tag::t_int) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Option is not set to an int but get_int() called! Current type tag: %d\n", (int) m_type);
                return {};
        }
        return m_val.i;
}
bool bmd_option::is_default() const {
        return m_type == type_tag::t_default;
}
bool bmd_option::is_user_set() const {
        return m_user_specified;
}
/**
 * @note
 * Returns true also for empty/NULL val - this allow specifying the flag without explicit value
 */
void bmd_option::parse(const char *val)
{
        // check flag
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-pointer-arithmetic"
#endif // defined __clang__
        if (val == nullptr || val == static_cast<char *>(nullptr) + 1 // allow constructions like parse_bmd_flag(strstr(opt, '=') + 1)
                        || strlen(val) == 0 || strcasecmp(val, "true") == 0 || strcasecmp(val, "on") == 0  || strcasecmp(val, "yes") == 0) {
                set_flag(true);
                return;
        }
#ifdef __clang__
#pragma clang diagnostic pop
#endif // defined __clang
        if (strcasecmp(val, "false") == 0 || strcasecmp(val, "off") == 0  || strcasecmp(val, "no") == 0) {
                set_flag(false);
                return;
        }

        if (strcasecmp(val, "keep") == 0) {
                set_keep();
                return;
        }

        // explicitly typed (either "str" or 'fcc')
        if ((val[0] == '"' && val[strlen(val) - 1] == '"') || (val[0] == '\'' && val[strlen(val) - 1] == '\'')) {
                string raw_val(val + 1);
                raw_val.erase(raw_val.length() - 1, 1);
                if (val[0] == '"') {
                        set_string(raw_val.c_str());
                } else {
                        set_int(bmd_read_fourcc(raw_val.c_str()));
                }
                return;
        }

        // check number
        bool is_number = true;
        bool decimal_point = false;
        for (size_t i = 0; i < strlen(val); ++i) {
                if (val[i] == '.') {
                        if (decimal_point) { // there was already decimal point
                                is_number = false;
                                break;
                        }
                        decimal_point = true;
                        continue;
                }
                is_number = is_number && isxdigit(val[i]);
                if (i == 0 && (val[i] == '-' || val[i] == '+')) {
                        is_number = true;
                }
        }
        if (is_number) {
                if (decimal_point) {
                        set_float(stod(val));
                } else {
                        set_int(stoi(val, nullptr, 0));
                }
                return;
        }
        if (strlen(val) <= 4) {
                set_int(bmd_read_fourcc(val));
                return;
        }

        set_string(val);
}

bool bmd_option::device_write(IDeckLinkConfiguration *deckLinkConfiguration, BMDDeckLinkConfigurationID opt, string const &log_prefix) const {
        HRESULT res = E_FAIL;
        switch (m_type) {
                case type_tag::t_flag:
                        res = deckLinkConfiguration->SetFlag(opt, get_flag());
                        break;
                case type_tag::t_int:
                        res = deckLinkConfiguration->SetInt(opt, get_int());
                        break;
                case type_tag::t_float:
                        res = deckLinkConfiguration->SetFloat(opt, m_val.f);
                        break;
                case type_tag::t_string: {
                        BMD_STR s = get_bmd_api_str_from_cstr(m_val.s);
                        res = deckLinkConfiguration->SetString(opt, s);
                        release_bmd_api_str(s);
                        break;
                }
                case type_tag::t_keep:
                case type_tag::t_default:
                        return true;
        }
        ostringstream value_oss;
        if (opt == bmdDeckLinkConfigVideoInputConnection && get_connection_string_map().find((BMDVideoConnection) get_int()) != get_connection_string_map().end()) {
                value_oss << get_connection_string_map().at((BMDVideoConnection) get_int());
        } else {
                value_oss << *this;
        }
        if (res != S_OK) {
                const int lvl = m_user_specified   ? LOG_LEVEL_ERROR
                                : res == E_NOTIMPL ? LOG_LEVEL_INFO
                                                   : LOG_LEVEL_WARNING;
                LOG(lvl) << log_prefix << "Unable to set key "
                         << fcc_to_string(opt) << " to " << value_oss.str()
                         << ": " << bmd_hresult_to_string(res) << "\n";
                return !m_user_specified;
        }
        LOG(LOG_LEVEL_INFO) << log_prefix << fcc_to_string(opt) << " set to: " << value_oss.str() << "\n";
        return true;
}

static void apply_r10k_lut(void *i, void *o, size_t len, void *udata)
{
        auto lut = (const unsigned int * __restrict) udata;
        auto *in = (const unsigned char *) i;
        auto *out = (unsigned char *) o;
        const unsigned char *in_end = in + len;
        while (in < in_end) {
                unsigned r = in[0] << 2U | in[1] >> 6U;
                unsigned g = (in[1] & 0x3FU) << 4U | in[2] >> 4U;
                unsigned b = (in[2] & 0xFU) << 6U | in[3] >> 2U;
                r = lut[r];
                g = lut[g];
                b = lut[b];
                out[0] = r >> 2U;
                out[1] = (r & 0x3U) << 6U | (g >> 4U);
                out[2] = (g & 0xFU) << 4U | (b >> 6U);
                out[3] = (b & 0x3FU) << 2U;
                in += 4;
                out += 4;
        }
}

static void fill_limited_to_full_lut(unsigned int *lut) {
        for (int i = 0; i < 1024; ++i) {
                int val = clamp(i, 64, 960);
                val = 4 + (val - 64) * 1015 / 896;
                lut[i] = val;
        }
}
/**
 * converts from range 64-960 to 4-1019
 *
 * in and out pointers can point to the same address
 */
void r10k_limited_to_full(const char *in, char *out, size_t len)
{
        static unsigned int lut[1024];
        if (lut[1023] == 0) {
                fill_limited_to_full_lut(lut);
        }
        DEBUG_TIMER_START(r10k_limited_to_full);
        respawn_parallel(const_cast<char *>(in), out, len / 4, 4, apply_r10k_lut, lut);
        DEBUG_TIMER_STOP(r10k_limited_to_full);
}

static void fill_full_to_limited_lut(unsigned int *lut) {
        for (int i = 0; i < 1024; ++i) {
                int val = clamp(i, 4, 1019);
                val = 64 + (val - 4) * 896 / 1015;
                lut[i] = val;
        }
}
/**
 * converts from full range (4-1019) to  64-960
 *
 * in and out pointers can point to the same address
 */
void r10k_full_to_limited(const char *in, char *out, size_t len)
{
        static unsigned int lut[1024];
        if (lut[1023] == 0) {
                fill_full_to_limited_lut(lut);
        }
        DEBUG_TIMER_START(r10k_limited_to_full);
        respawn_parallel(const_cast<char *>(in), out, len / 4, 4, apply_r10k_lut, lut);
        DEBUG_TIMER_STOP(r10k_limited_to_full);
}

string bmd_get_audio_connection_name(BMDAudioOutputAnalogAESSwitch audioConnection) {
        switch(audioConnection) {
                case bmdAudioOutputSwitchAESEBU:
                        return "AES/EBU";
                case bmdAudioOutputSwitchAnalog:
                        return "analog";
                default:
                        return "default";
        }
}

string bmd_get_flags_str(BMDDisplayModeFlags flags) {
        bool first = true;
        ostringstream oss;
        vector<pair<uint32_t, const char *>> map {
                { bmdDisplayModeColorspaceRec601, "Rec601" },
                { bmdDisplayModeColorspaceRec709, "Rec709" },
                { bmdDisplayModeColorspaceRec2020, "Rec2020" },
                { bmdDisplayModeSupports3D, "3D" },
        };

        for (auto &f : map ) {
                if (flags & f.first) {
                        oss << (!first ? ", " : "") << f.second;
                        first = false;
                }
        }
        if (flags == 0) {
                oss << "(none)";
        }
        if (flags >= bmdDisplayModeColorspaceRec2020 << 1) {
                oss << ", (unknown flags)";
        }
        return oss.str();
}

void print_bmd_device_profiles(const char *line_prefix)
{
        for (const auto &p : kDeviceProfiles) {
                const uint32_t fcc = htonl(p.first);
                color_printf("%s" TBOLD("%.4s") " - %s (%s)\n", line_prefix, (const char *) &fcc, p.second.first, p.second.second);
        }
        color_printf("%s" TBOLD("keep") " - keep device setting\n", line_prefix);
}

const map<BMDVideoConnection, string> &get_connection_string_map() {
        static const map<BMDVideoConnection, string> m = {
                { bmdVideoConnectionSDI, "SDI" },
                { bmdVideoConnectionHDMI, "HDMI"},
                { bmdVideoConnectionOpticalSDI, "OpticalSDI"},
                { bmdVideoConnectionComponent, "Component"},
                { bmdVideoConnectionComposite, "Composite"},
                { bmdVideoConnectionSVideo, "SVideo"}
        };
        return m;
}

template <typename T> struct bmd_no_conv {
};
template <> struct bmd_no_conv<IDeckLinkInput> {
        static constexpr BMDVideoInputConversionMode value =
            bmdNoVideoInputConversion;
};
template <> struct bmd_no_conv<IDeckLinkOutput> {
        static constexpr BMDVideoOutputConversionMode value =
            bmdNoVideoOutputConversion;
};
/**
 * This function returns true if any display mode and any output supports the
 * codec. The codec, however, may not be supported with actual video mode.
 *
 * @todo For UltraStudio Pro DoesSupportVideoMode returns E_FAIL on not
 * supported pixel formats instead of setting supported to false.
 */
template <typename T>
bool
decklink_supports_codec(T *deckLink, BMDPixelFormat pf)
{
        IDeckLinkDisplayModeIterator *displayModeIterator = nullptr;
        IDeckLinkDisplayMode         *deckLinkDisplayMode = nullptr;

        if (FAILED(
                deckLink->GetDisplayModeIterator(&displayModeIterator))) {
                MSG(ERROR, "Fatal: cannot create display mode iterator.\n");
                return false;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK) {
                BMD_BOOL supported = false;
                const HRESULT res       = deckLink->DoesSupportVideoMode(
                    bmdVideoConnectionUnspecified,
                    deckLinkDisplayMode->GetDisplayMode(), pf,
                    bmd_no_conv<T>::value, bmdSupportedVideoModeDefault,
                    nullptr, &supported);
                deckLinkDisplayMode->Release();
                if (res != S_OK) {
                        MSG(WARNING, "DoesSupportVideoMode: %s\n",
                            bmd_hresult_to_string(res).c_str());
                        continue;
                }
                if (supported) {
                        displayModeIterator->Release();
                        return true;
                }
        }
        displayModeIterator->Release();

        return false;
}
template bool
decklink_supports_codec<IDeckLinkOutput>(IDeckLinkOutput *deckLink,
                                         BMDPixelFormat   pf);
template bool decklink_supports_codec<IDeckLinkInput>(IDeckLinkInput *deckLink,
                                                      BMDPixelFormat  pf);

bool
bmd_parse_audio_levels(const char *opt) noexcept(false)
{
        if (strcasecmp(opt, "false") == 0 || strcasecmp(opt, "off") == 0 ||
            strcasecmp(opt, "line") == 0) {
                return false;
        }
        if (strcasecmp(opt, "true") == 0 || strcasecmp(opt, "on") == 0 ||
            strcasecmp(opt, "mic") == 0) {
                return true;
        }
        throw invalid_argument(string("invalid BMD audio level ") + opt);
}

ADD_TO_PARAM(R10K_FULL_OPT, "* " R10K_FULL_OPT "\n"
                "  Do not do conversion from/to limited range on in/out for R10k on BMD devs.\n");


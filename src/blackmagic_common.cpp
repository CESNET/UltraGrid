/**
 * @file   blackmagic_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET, zájmové sdružení právnických osob
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

#include <algorithm>
#include <atomic>                // for atomic
#include <cassert>
#include <cctype>                // for isxdigit
#include <chrono>                // for seconds
#include <climits>               // for UINT_MAX
#include <cinttypes>             // for PRId64
#include <condition_variable>
#include <csignal>
#include <cstdio>                // for fprintf, stderr
#include <cstdint>               // for int64_t, uint32_t
#include <cstdlib>               // for free
#include <cstring>               // for strlen, NULL, strdup, memcpy, size_t
#include <iomanip>
#include <iterator>              // for pair
#include <map>
#include <mutex>                 // for mutex, lock_guard, unique_lock
#include <sstream>
#include <stdexcept>
#include <utility>

#include "compat/misc.h"         // for strncasecmp
#include "compat/net.h"          // for htonl, ntohl
#include "compat/strings.h"      // for strncasecmp
#include "DeckLinkAPIVersion.h"
#include "blackmagic_common.hpp"
#include "debug.h"
#include "host.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/debug.h"         // for DEBUG_TIMER_*
#include "utils/macros.h"
#include "utils/string.h"        // for DELDEL
#include "utils/windows.h"
#include "utils/worker.h"

// BMD sometimes do a ABI bump that entirely breaks compatibility (eg. changing
// GUIDs), this can be inspected by checking the "versioned" DeckLinkAPI here:
// <https://github.com/MartinPulec/desktopvideo_sdk-api/tree/main/Linux/include>
#define BMD_LAST_INCOMPATIBLE_ABI 0x0b050100 // 11.5.1

#if BLACKMAGIC_DECKLINK_API_VERSION > 0x0c080000
#warning \
    "Increased BMD API - enum diffs recheck recommends (or just increase the compared API version)"
#endif

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
#ifdef __APPLE__
       size_t len = CFStringGetMaximumSizeForEncoding(CFStringGetLength(bmd_string), kCFStringEncodingUTF8) + 1;
       cstr = (char *) malloc(len);
       CFStringGetCString(bmd_string, (char *) cstr, len, kCFStringEncodingUTF8);
#elif defined _WIN32
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
#ifdef __APPLE__
        CFRelease(string);
#elif defined _WIN32
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
 * @param[out] com_initialized  pointer to be passed to decklnk_uninitialize
 (keeps information if COM needs to be uninitialized)
 * @note
 * Each successful call (returning non-null pointer) of this function
 * should be followed by com_uninitialize() when done with DeckLink (not when releasing
 * IDeckLinkIterator!), typically on application shutdown.
 */
IDeckLinkIterator *create_decklink_iterator(bool *com_initialized, bool verbose)
{
        IDeckLinkIterator *deckLinkIterator = nullptr;
#ifdef _WIN32
        com_initialize(com_initialized, "[BMD] ");
        HRESULT result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
                        IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result)) {
                decklink_uninitialize(com_initialized);
                deckLinkIterator = nullptr;
        }
#else
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
#ifdef _WIN32
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

        // this is safe comparison, for internal structure please see SDK
        // documentation
        if (value <= BMD_LAST_INCOMPATIBLE_ABI) {
                MSG(ERROR, "The DeckLink drivers are be outdated.\n");
                MSG(ERROR, "You must have drivers newer than %d.%d.%d.\n",
                    BMD_LAST_INCOMPATIBLE_ABI >> 24,
                    (BMD_LAST_INCOMPATIBLE_ABI >> 16) & 0xFF,
                    (BMD_LAST_INCOMPATIBLE_ABI >> 8) & 0xFF);
                MSG(ERROR, "Vendor download page is "
                           "http://www.blackmagic-design.com/support\n");
                print_decklink_version();
        } else {
                ret = true;
                if (BLACKMAGIC_DECKLINK_API_VERSION > value) {
                        MSG(WARNING, "The DeckLink drivers are be outdated.\n");
                        MSG(WARNING,
                            "Although it will likely work, it is recommended "
                            "to use drivers at least as the API that "
                            "UltraGrid is linked with.\n");
                        print_decklink_version();
                        MSG(WARNING,
                            "Vendor download page is "
                            "http://www.blackmagic-design.com/support\n\n");
                }
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

#ifdef _WIN32
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
#ifdef _WIN32
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
                                LOG(LOG_LEVEL_NOTICE) << "[DeckLink] Profile " << profileName << " activated successfully.\n";
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

        if (req_profile.is_help()) {
                printf("Available profiles:\n");
                print_bmd_device_profiles("\t");
                return false;
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

        // set '1dfd' for stereo if profile is not set explicitly
        assert(!req_profile.is_default() || stereo);
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

#define BMDFCC(x) {x,#x}
static const struct {
        uint32_t    fourcc;
        const char *name;
} opt_name_map[] = {

        { 0, "Serial port Flags"           },

        BMDFCC(bmdDeckLinkConfigSwapSerialRxTx),

        { 0, "Video Input/Output Integers" },

        BMDFCC(bmdDeckLinkConfigHDMI3DPackingFormat),
        BMDFCC(bmdDeckLinkConfigBypass),
        BMDFCC(bmdDeckLinkConfigClockTimingAdjustment),

        { 0, "Audio Input/Output Flags"    },

        BMDFCC(bmdDeckLinkConfigAnalogAudioConsumerLevels),
        BMDFCC(bmdDeckLinkConfigSwapHDMICh3AndCh4OnInput),
        BMDFCC(bmdDeckLinkConfigSwapHDMICh3AndCh4OnOutput),

        { 0, "Video Output Flags"          },

        BMDFCC(bmdDeckLinkConfigFieldFlickerRemoval),
        BMDFCC(bmdDeckLinkConfigHD1080p24ToHD1080i5994Conversion),
        BMDFCC(bmdDeckLinkConfig444SDIVideoOutput),
        BMDFCC(bmdDeckLinkConfigBlackVideoOutputDuringCapture),
        BMDFCC(bmdDeckLinkConfigLowLatencyVideoOutput),
        BMDFCC(bmdDeckLinkConfigDownConversionOnAllAnalogOutput),
        BMDFCC(bmdDeckLinkConfigSMPTELevelAOutput),
        BMDFCC(bmdDeckLinkConfigRec2020Output),
        BMDFCC(bmdDeckLinkConfigQuadLinkSDIVideoOutputSquareDivisionSplit),
        BMDFCC(bmdDeckLinkConfigOutput1080pAsPsF),

        { 0, "Video Output Integers"       },

        BMDFCC(bmdDeckLinkConfigVideoOutputConnection),
        BMDFCC(bmdDeckLinkConfigVideoOutputConversionMode),
        BMDFCC(bmdDeckLinkConfigAnalogVideoOutputFlags),
        BMDFCC(bmdDeckLinkConfigReferenceInputTimingOffset),
        BMDFCC(bmdDeckLinkConfigReferenceOutputMode),
        BMDFCC(bmdDeckLinkConfigVideoOutputIdleOperation),
        BMDFCC(bmdDeckLinkConfigDefaultVideoOutputMode),
        BMDFCC(bmdDeckLinkConfigDefaultVideoOutputModeFlags),
        BMDFCC(bmdDeckLinkConfigSDIOutputLinkConfiguration),
        BMDFCC(bmdDeckLinkConfigHDMITimecodePacking),
        BMDFCC(bmdDeckLinkConfigPlaybackGroup),

        { 0, "Video Output Floats"         },

        BMDFCC(bmdDeckLinkConfigVideoOutputComponentLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputComponentChromaBlueGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputComponentChromaRedGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputCompositeLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputCompositeChromaGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputSVideoLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoOutputSVideoChromaGain),

        { 0, "Video Input Flags"           },

        BMDFCC(bmdDeckLinkConfigVideoInputScanning),
        BMDFCC(bmdDeckLinkConfigUseDedicatedLTCInput),
        BMDFCC(bmdDeckLinkConfigSDIInput3DPayloadOverride),
        BMDFCC(bmdDeckLinkConfigCapture1080pAsPsF),

        { 0, "Video Input Integers"        },

        BMDFCC(bmdDeckLinkConfigVideoInputConnection),
        BMDFCC(bmdDeckLinkConfigAnalogVideoInputFlags),
        BMDFCC(bmdDeckLinkConfigVideoInputConversionMode),
        BMDFCC(bmdDeckLinkConfig32PulldownSequenceInitialTimecodeFrame),
        BMDFCC(bmdDeckLinkConfigVANCSourceLine1Mapping),
        BMDFCC(bmdDeckLinkConfigVANCSourceLine2Mapping),
        BMDFCC(bmdDeckLinkConfigVANCSourceLine3Mapping),
        BMDFCC(bmdDeckLinkConfigCapturePassThroughMode),
        BMDFCC(bmdDeckLinkConfigCaptureGroup),

        { 0, "Video Input Floats"          },

        BMDFCC(bmdDeckLinkConfigVideoInputComponentLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoInputComponentChromaBlueGain),
        BMDFCC(bmdDeckLinkConfigVideoInputComponentChromaRedGain),
        BMDFCC(bmdDeckLinkConfigVideoInputCompositeLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoInputCompositeChromaGain),
        BMDFCC(bmdDeckLinkConfigVideoInputSVideoLumaGain),
        BMDFCC(bmdDeckLinkConfigVideoInputSVideoChromaGain),

        { 0, "Keying Integers"             },

        BMDFCC(bmdDeckLinkConfigInternalKeyingAncillaryDataSource),

        { 0, "Audio Input Flags"           },

        BMDFCC(bmdDeckLinkConfigMicrophonePhantomPower),

        { 0, "Audio Input Integers"        },

        BMDFCC(bmdDeckLinkConfigAudioInputConnection),

        { 0, "Audio Input Floats"          },

        BMDFCC(bmdDeckLinkConfigAnalogAudioInputScaleChannel1),
        BMDFCC(bmdDeckLinkConfigAnalogAudioInputScaleChannel2),
        BMDFCC(bmdDeckLinkConfigAnalogAudioInputScaleChannel3),
        BMDFCC(bmdDeckLinkConfigAnalogAudioInputScaleChannel4),
        BMDFCC(bmdDeckLinkConfigDigitalAudioInputScale),
        BMDFCC(bmdDeckLinkConfigMicrophoneInputGain),

        { 0, "Audio Output Integers"       },

        BMDFCC(bmdDeckLinkConfigAudioOutputAESAnalogSwitch),

        { 0, "Audio Output Floats"         },

        BMDFCC(bmdDeckLinkConfigAnalogAudioOutputScaleChannel1),
        BMDFCC(bmdDeckLinkConfigAnalogAudioOutputScaleChannel2),
        BMDFCC(bmdDeckLinkConfigAnalogAudioOutputScaleChannel3),
        BMDFCC(bmdDeckLinkConfigAnalogAudioOutputScaleChannel4),
        BMDFCC(bmdDeckLinkConfigDigitalAudioOutputScale),
        BMDFCC(bmdDeckLinkConfigHeadphoneVolume),

        { 0, "Network Flags"               },

        BMDFCC(bmdDeckLinkConfigEthernetUseDHCP),
        BMDFCC(bmdDeckLinkConfigEthernetPTPFollowerOnly),
        BMDFCC(bmdDeckLinkConfigEthernetPTPUseUDPEncapsulation),

        { 0, "Network Integers"            },

        BMDFCC(bmdDeckLinkConfigEthernetPTPPriority1),
        BMDFCC(bmdDeckLinkConfigEthernetPTPPriority2),
        BMDFCC(bmdDeckLinkConfigEthernetPTPDomain),

        { 0, "Network Strings"             },

        BMDFCC(bmdDeckLinkConfigEthernetStaticLocalIPAddress),
        BMDFCC(bmdDeckLinkConfigEthernetStaticSubnetMask),
        BMDFCC(bmdDeckLinkConfigEthernetStaticGatewayIPAddress),
        BMDFCC(bmdDeckLinkConfigEthernetStaticPrimaryDNS),
        BMDFCC(bmdDeckLinkConfigEthernetStaticSecondaryDNS),
        BMDFCC(bmdDeckLinkConfigEthernetVideoOutputAddress),
        BMDFCC(bmdDeckLinkConfigEthernetAudioOutputAddress),
        BMDFCC(bmdDeckLinkConfigEthernetAncillaryOutputAddress),
        BMDFCC(bmdDeckLinkConfigEthernetAudioOutputChannelOrder),

        { 0, "Device Information Strings"  },

        BMDFCC(bmdDeckLinkConfigDeviceInformationLabel),
        BMDFCC(bmdDeckLinkConfigDeviceInformationSerialNumber),
        BMDFCC(bmdDeckLinkConfigDeviceInformationCompany),
        BMDFCC(bmdDeckLinkConfigDeviceInformationPhone),
        BMDFCC(bmdDeckLinkConfigDeviceInformationEmail),
        BMDFCC(bmdDeckLinkConfigDeviceInformationDate),

        { 0, "Deck Control Integers"       },

        BMDFCC(bmdDeckLinkConfigDeckControlConnection),
};

static const struct {
        uint32_t    fourcc;
        const char *name;
} val_name_map[] = {
        BMDFCC(bmdVideo3DPackingSidebySideHalf),
        BMDFCC(bmdVideo3DPackingLinebyLine),
        BMDFCC(bmdVideo3DPackingTopAndBottom),
        BMDFCC(bmdVideo3DPackingFramePacking),
        BMDFCC(bmdVideo3DPackingRightOnly),
        BMDFCC(bmdVideo3DPackingLeftOnly),
        BMDFCC(bmdDeckLinkCapturePassthroughModeDisabled),
        BMDFCC(bmdDeckLinkCapturePassthroughModeCleanSwitch),
        BMDFCC(bmdIdleVideoOutputBlack),
        BMDFCC(bmdIdleVideoOutputLastFrame),
        BMDFCC(bmdLinkConfigurationSingleLink),
        BMDFCC(bmdLinkConfigurationDualLink),
        BMDFCC(bmdLinkConfigurationQuadLink),
};
#undef BMDFCC

static string fcc_to_string(uint32_t fourcc) {
        for (unsigned i = 0; i < ARR_COUNT(opt_name_map); ++i) {
                if (opt_name_map[i].fourcc == fourcc) {
                        return opt_name_map[i].name;
                }
        }
        for (unsigned i = 0; i < ARR_COUNT(val_name_map); ++i) {
                if (val_name_map[i].fourcc == fourcc) {
                        return val_name_map[i].name;
                }
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
const char *bmd_option::get_string() const {
        if (m_type != type_tag::t_string) {
                MSG(WARNING,
                    "Option is not set to a string but get_string() called! Current "
                    "type tag: %d\n",
                    (int) m_type);
                return {};
        }
        return m_val.s;
}
bool bmd_option::is_default() const {
        return m_type == type_tag::t_default;
}
bool bmd_option::is_help() const {
        return m_type == type_tag::t_string &&
               strcmp(get_string(), "help") == 0;
}
bool bmd_option::is_user_set() const {
        return m_user_specified;
}

static void
bmd_opt_help()
{
        color_printf(TBOLD("BMD") " option syntax:\n");
        color_printf("\t" TBOLD("<FourCC>=<val>") "\n\n");
        color_printf(
            TBOLD("<FourCC>") " must have exactly " TBOLD("4 characters") "\n");
        color_printf("\n");
        color_printf("The value must corresponding type is deduced accordingly:\n");
        color_printf("- " TBOLD("flag") " - values on/of, true/false, yes/no\n");
        color_printf("- literal " TBOLD("keep") " - keep the preset value\n");
        color_printf("- literal " TBOLD(
            "help") " - show help (applicable to profile only)\n");
        color_printf("- " TBOLD(
            "int") " - any number without decimal point\n");
        color_printf("- " TBOLD(
            "float") " - a number with a decimal point\n");
        color_printf(
            "- " TBOLD("FourCC") " - a value with len <= 4 not listed above\n");
        color_printf("\n");
        color_printf("If the type deduction is not working, you can use also "
                     "following syntax:\n");
        color_printf("- \"value\" - assume the value is string\n");
        color_printf("- 'vlue' - assume the value is FourCC\n");
        color_printf("\n");

        color_printf("List of keys:\n");
        for (unsigned i = 0; i < ARR_COUNT(opt_name_map); ++i) {
                if (opt_name_map[i].fourcc == 0) {
                        color_printf("\n%s:\n", opt_name_map[0].name);
                } else {
                        uint32_t val = htonl(opt_name_map[i].fourcc);
                        color_printf("- " TBOLD("%.4s") " - %s\n",
                                     (char *) &val, opt_name_map[i].name);
                }
        }
        color_printf("\n");
        color_printf("See also\n" TUNDERLINE(
            "https://github.com/CESNET/UltraGrid/blob/master/ext-deps/"
            "DeckLink/Linux/DeckLinkAPIConfiguration.h") "\nfor details.\n");
        color_printf("\n");
        color_printf("Incomplete " TBOLD("(!)") " list of values:\n");
        color_printf("(note that the value belongs to its appropriate key)\n");
        for (unsigned i = 0; i < ARR_COUNT(val_name_map); ++i) {
                uint32_t val = htonl(val_name_map[i].fourcc);
                color_printf("- " TBOLD("%.4s") " - %s\n", (char *) &val,
                             val_name_map[i].name);
        }
        color_printf("\n");
        color_printf("Available values can be found here:\n" TUNDERLINE(
            "https://github.com/CESNET/UltraGrid/blob/master/ext-deps/"
            "DeckLink/Linux/DeckLinkAPI.h") "\n");
        color_printf("\n");
        color_printf("The actual key type and possible values must be, however "
                     "consutlted with:\n" TUNDERLINE(
                         "https://documents.blackmagicdesign.com/UserManuals/"
                         "DeckLinkSDKManual.pdf") "\n");

        color_printf("\n");
        color_printf("Examples:\n");
        color_printf(TBOLD("aacl=on") " - set audio consumer levels (flag)\n");
        color_printf(TBOLD("voio=blac") " - display black when no output\n");
        color_printf(TBOLD("DHCP=yes") " - use DHCP config for DeckLink IP\n");
        color_printf(TBOLD("DHCP=no:"
            "nsip=10.0.0.3:nssm=255.255.255.0:nsgw=10.0.0.1") " - use static "
                                                              "net config for "
                                                              "DeckLink IP\n");
        color_printf(TBOLD(
            "noaa=239.255.194.26\\:16384:noav=239.255.194.26\\:"
            "163888") " - set output "
                      "audio/video address\n(note that the shell will remove "
                      "backslash if not quoted, so you may use eg.:\nuv -t "
                      "'decklink:noaa=239.255.194.26\\:16384')\n");
        color_printf("\n");
}

/**
 * @param val  can be empty or NULL - this allow specifying the flag without explicit value
 * @retval true  value was set
 * @retval false help for FourCC syntas was print
 */
bool
bmd_option::parse(const char *val)
{
        // check flag
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-pointer-arithmetic"
#endif // defined __clang__
        if (val == nullptr || val == static_cast<char *>(nullptr) + 1 // allow constructions like parse_bmd_flag(strstr(opt, '=') + 1)
                        || strlen(val) == 0 || strcasecmp(val, "true") == 0 || strcasecmp(val, "on") == 0  || strcasecmp(val, "yes") == 0) {
                set_flag(true);
                return true;
        }
#ifdef __clang__
#pragma clang diagnostic pop
#endif // defined __clang
        if (strcasecmp(val, "false") == 0 || strcasecmp(val, "off") == 0  || strcasecmp(val, "no") == 0) {
                set_flag(false);
                return true;
        }

        if (strcasecmp(val, "keep") == 0) {
                set_keep();
                return true;
        }

        if (strcmp(val, "help") == 0) {
                set_string(val);
                return true;
        }

        if (strcmp(val, "FourCC") == 0) { // help=FourCC
                bmd_opt_help();
                return false;
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
                return true;
        }

        // check number
        bool is_number = true;
        bool decimal_point = false;
        for (size_t i = 0; i < strlen(val); ++i) {
                if (i == 0 && strncasecmp(val, "0x", 2) == 0) {
                        i += 1;
                        continue;
                }
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
                return true;
        }
        if (strlen(val) <= 4) {
                set_int(bmd_read_fourcc(val));
                return true;
        }

        set_string(val);
        return true;
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
        if ((opt == bmdDeckLinkConfigVideoInputConnection ||
             opt == bmdDeckLinkConfigVideoOutputConnection) &&
            get_connection_string_map().find((BMDVideoConnection) get_int()) !=
                get_connection_string_map().end()) {
                value_oss << get_connection_string_map().at(
                    (BMDVideoConnection) get_int());
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
                { bmdVideoConnectionSVideo, "SVideo"},
                { bmdVideoConnectionEthernet, "Ethernet"},
                { bmdVideoConnectionOpticalEthernet, "OpticalEthernet"},
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

/// TOREMOVE (implicit "aacl" option is preferred)
bool
bmd_parse_audio_levels(const char *opt) noexcept(false)
{
        MSG(WARNING, "audio_level option is deprecated, use "
                     "\"aacl[=true|false]\" instead\n");
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

void
print_bmd_attribute(IDeckLinkProfileAttributes *deckLinkAttributes,
                    const char                 *query_prop_fcc)
{
        if (strcmp(query_prop_fcc, "help") == 0) {
                col{} << "Query usage:\n"
                      << SBOLD("\tq[uery]=<fcc>")
                      << " - gets DeckLink attribute value\n";
                return;
        }
        union {
                char fcc[sizeof(BMDDeckLinkAttributeID)] = { ' ', ' ', ' ',
                                                             ' ' };
                BMDDeckLinkAttributeID key;
        };
        memcpy(fcc, query_prop_fcc, min(strlen(query_prop_fcc), sizeof fcc));
        key = (BMDDeckLinkAttributeID) htonl(key);
        BMD_BOOL      bool_val{};
        int64_t       int_val{};
        double        float_val{};
        BMD_STR       string_val{};
        ostringstream oss;
        HRESULT result = deckLinkAttributes->GetFlag(key, &bool_val);
        if (result == S_OK) {
                oss << (bool_val ? "true" : "false");
        }
        if (result == E_INVALIDARG) {
                result = deckLinkAttributes->GetInt(key, &int_val);
                if (result == S_OK) {
                        oss << int_val << " (0x" << hex << int_val << ")";
                }
        }
        if (result == E_INVALIDARG) {
                result = deckLinkAttributes->GetFloat(key, &float_val);
                if (result == S_OK) {
                        oss << float_val;
                }
        }
        if (result == E_INVALIDARG) {
                result = deckLinkAttributes->GetString(key, &string_val);
                if (result == S_OK) {
                        string str = get_str_from_bmd_api_str(string_val);
                        release_bmd_api_str(string_val);
                        oss << str;
                }
        }
        if (result != S_OK) {
                LOG(LOG_LEVEL_ERROR)
                    << MOD_NAME << "Cannot get " << query_prop_fcc << ": "
                    << bmd_hresult_to_string(result) << "\n";
                return;
        }
        col() << "\t" << hex << "Value of " << SBOLD(query_prop_fcc)
              << " attribute for this device is " << SBOLD(oss.str()) << "\n";
}

/**
 * @returns list of DeckLink devices sorted (indexed) by topological ID
 *          If no topological ID is reported, use UINT_MAX (and lower vals)
 * @param verbose - print errors
 * @param natural_sort - use the (old) natural sort as given by the iterator
 * @note
 Each call of this function should be followed by com_uninitialize() when done
 with DeckLink. No BMD stuff originating from this function call
 (IDeckLinks) can be used after that and new call must be made.
 * @note
 * The returned IDeckLink instances are managed with the unique_ptr, so
 * take care about the returned object lifetime (particularly not to
 * be destroyed after decklink_uninitialize()).
 */
std::vector<bmd_dev>
bmd_get_sorted_devices(bool *com_initialized, bool verbose, bool natural_sort)
{
        IDeckLinkIterator *deckLinkIterator =
            create_decklink_iterator(com_initialized, verbose);
        if (deckLinkIterator == nullptr) {
                return {};
        }

        IDeckLink *deckLink = nullptr;
        std::vector<bmd_dev> out;
        int                  idx = 0;
        while (deckLinkIterator->Next(&deckLink) == S_OK) {
                IDeckLinkProfileAttributes *deckLinkAttributes = nullptr;
                HRESULT                     result             = E_FAIL;
                result =
                    deckLink->QueryInterface(IID_IDeckLinkProfileAttributes,
                                             (void **) &deckLinkAttributes);
                assert(result == S_OK);
                int64_t id = 0;
                result =
                    deckLinkAttributes->GetInt(BMDDeckLinkTopologicalID, &id);
                if (result != S_OK) {
                        id = UINT_MAX - idx;
                }
                assert(id >= 0 && id <= UINT_MAX);
                deckLinkAttributes->Release();

                auto release = [](IDeckLink *d) { d->Release(); };
                auto &it      = out.emplace_back(
                    std::unique_ptr<IDeckLink, void (*)(IDeckLink *)>{
                        deckLink, release },
                    0, 0, 0);
                std::get<unsigned>(it) = id;
                std::get<int>(it) = idx++;
        }
        deckLinkIterator->Release();
        if (!natural_sort) {
                std::sort(out.begin(), out.end(), [](bmd_dev &a, bmd_dev &b) {
                        return std::get<unsigned>(a) < std::get<unsigned>(b);
                });
        }
        // assign new indices
        char new_idx = 'a';
        for (auto &d : out) {
                std::get<char>(d) = new_idx++;
        }
        return out;
}

void
print_bmd_connections(IDeckLinkProfileAttributes *deckLinkAttributes,
                      BMDDeckLinkAttributeID id, const char *module_prefix)
{
        printf("\tsupported %s connetions:",
               id == BMDDeckLinkVideoInputConnections ? "input" : "output");
        int64_t connections = 0;
        if (deckLinkAttributes->GetInt(id, &connections) != S_OK) {
                log_msg(LOG_LEVEL_ERROR, "\n%sCould not get connections.\n\n",
                        module_prefix);
                return;
        }
        for (auto const &it : get_connection_string_map()) {
                if ((connections & it.first) != 0) {
                        col() << " " << SBOLD(it.second);
                }
        }
        col() << "\n";
}

BMDVideoConnection
bmd_get_connection_by_name(const char *connection)
{
        for (auto const &it : get_connection_string_map()) {
                if (strcasecmp(connection, it.second.c_str()) == 0) {
                        return it.first;
                }
        }
        return bmdVideoConnectionUnspecified;
}

/*   ____            _    _     _       _     ____  _        _             
 *  |  _ \  ___  ___| | _| |   (_)_ __ | | __/ ___|| |_ __ _| |_ _   _ ___ 
 *  | | | |/ _ \/ __| |/ / |   | | '_ \| |/ /\___ \| __/ _` | __| | | / __|
 *  | |_| |  __/ (__|   <| |___| | | | |   <  ___) | || (_| | |_| |_| \__ \
 *  |____/ \___|\___|_|\_\_____|_|_| |_|_|\_\|____/ \__\__,_|\__|\__,_|___/
 */

/// value map, needs to be zero-terminated
/// if status_type == ST_BIT_FIELD, val==0 must be set
struct bmd_status_val_map {
        uint32_t    val;
        const char *name;
};
/// FourCC based values (can be all in one array)
static const struct bmd_status_val_map status_val_map_dfl[] = {
        { bmdEthernetLinkStateDisconnected,     "disconnected"        },
        { bmdEthernetLinkStateConnectedUnbound, "connected (unbound)" },
        { bmdEthernetLinkStateConnectedBound,   "connected (bound)"   },
        { 0,                                    nullptr               },
};
static const struct bmd_status_val_map bmd_busy_state_bit_field_map[] = {
        { 0,                       "inactive"    }, // default val if no bit set
        { bmdDeviceCaptureBusy,    "capture"     },
        { bmdDevicePlaybackBusy,   "playback"    },
        { bmdDeviceSerialPortBusy, "serial-port" },
        { 0,                       nullptr       },
};
static const struct bmd_status_val_map bmd_dyn_range_map[] = {
        { bmdDynamicRangeSDR,          "SDR"     },
        { bmdDynamicRangeHDRStaticPQ,  "HDR PQ"  },
        { bmdDynamicRangeHDRStaticHLG, "HDR HLG" },
        { 0,                           nullptr}
};
static const struct bmd_status_val_map bmd_cs_map[] = {
        { bmdColorspaceRec601,  "Rec.601"  },
        { bmdColorspaceRec709,  "Rec.709"  },
        { bmdColorspaceRec2020, "Rec.2020" },
        { 0,                    nullptr    }
};
enum status_type {
        ST_ENUM,      // set type_data.map
        ST_BIT_FIELD, // set type_data.map
        ST_INT,       // set type_data.int_fmt_str
        ST_STRING,
};
static const struct status_property {
        BMDDeckLinkStatusID prop;
        const char         *prop_name;
        enum status_type    type;
        union type_data {
                const char                      *int_fmt_str;
                const struct bmd_status_val_map *map;
        } type_data;
        bool playback_only; ///< relevant only for playback;
        int  req_log_level;
} status_map[] = {
        { bmdDeckLinkStatusBusy,
         "Busy",                          ST_BIT_FIELD,
         { .map = bmd_busy_state_bit_field_map },
         false, LOG_LEVEL_VERBOSE },
        { bmdDeckLinkStatusPCIExpressLinkWidth,
         "PCIe Link Width",               ST_INT,
         { .int_fmt_str = "%" PRIu64 "x" },
         false, LOG_LEVEL_VERBOSE },
        { bmdDeckLinkStatusPCIExpressLinkSpeed,
         "PCIe Link Speed",               ST_INT,
         { .int_fmt_str = "Gen. %" PRIu64 },
         false, LOG_LEVEL_VERBOSE },
        { bmdDeckLinkStatusDeviceTemperature,
         "Temperature",                   ST_INT,
         { .int_fmt_str = "%" PRIu64 " °C" },
         false, LOG_LEVEL_VERBOSE }, // temperature info is rate-limited
        { bmdDeckLinkStatusDetectedVideoInputColorspace,
         "Video Colorspace",              ST_ENUM,
         { .map = bmd_cs_map },
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusDetectedVideoInputDynamicRange,
         "Video Dynamic Range",           ST_ENUM,
         { .map = bmd_dyn_range_map },
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetLink,
         "Ethernet state",                ST_ENUM,
         { .map = status_val_map_dfl },
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetLinkMbps,
         "Ethernet link speed",           ST_INT,
         { .int_fmt_str = "%" PRIu64 " Mbps" },
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetLocalIPAddress,
         "Ethernet IP address",           ST_STRING,
         {},
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetSubnetMask,
         "Ethernet subnet mask",          ST_STRING,
         {},
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetGatewayIPAddress,
         "Ethernet gateway IP",           ST_STRING,
         {},
         false, LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetVideoOutputAddress,
         "Ethernet video output address", ST_STRING,
         {},
         true,  LOG_LEVEL_INFO    },
        { bmdDeckLinkStatusEthernetAudioOutputAddress,
         "Ethernet audio output address", ST_STRING,
         {},
         true,  LOG_LEVEL_INFO    },
};
static void
print_status_item(IDeckLinkStatus *deckLinkStatus, BMDDeckLinkStatusID prop,
                  const char *log_prefix)
{
        const struct status_property *s_prop = nullptr;

        for (unsigned i = 0; i < ARR_COUNT(status_map); ++i) {
                if (status_map[i].prop == prop) {
                        s_prop = &status_map[i];
                        break;
                }
        }
        if (s_prop == nullptr) { // not found
                return;
        }
        if (log_level < s_prop->req_log_level) {
                return;
        }

        int64_t int_val = 0;
        BMD_STR string_val{};
        HRESULT rc = s_prop->type == ST_STRING
                         ? deckLinkStatus->GetString(s_prop->prop, &string_val)
                         : deckLinkStatus->GetInt(s_prop->prop, &int_val);
        if (!SUCCEEDED(rc)) {
                if (FAILED(rc) && rc != E_NOTIMPL) {
                        log_msg(LOG_LEVEL_WARNING,
                                "%sObtain property %s (0x%08x) value: %s\n",
                                log_prefix, s_prop->prop_name, (unsigned) prop,
                                bmd_hresult_to_string(rc).c_str());
                }
                return;
        }

        switch (s_prop->type) {
        case ST_STRING: {
                string str = get_str_from_bmd_api_str(string_val);
                release_bmd_api_str(string_val);
                log_msg(LOG_LEVEL_INFO, "%s%s: %s\n", log_prefix,
                        s_prop->prop_name, str.c_str());
                break;
        }
        case ST_INT: {
                char buf[STR_LEN];
                snprintf_ch(buf, s_prop->type_data.int_fmt_str, int_val);
                log_msg(LOG_LEVEL_INFO, "%s%s: %s\n",
                        log_prefix, s_prop->prop_name, buf);
                break;
        }
        case ST_BIT_FIELD: {
                char val[STR_LEN];
                val[0] = '\0';
                for (unsigned j = 0; s_prop->type_data.map[j].name != nullptr;
                     ++j) {
                        if ((int_val & s_prop->type_data.map[j].val) == 0) {
                                continue;
                        }
                        snprintf(val + strlen(val), sizeof val - strlen(val),
                                 "%s%s", val[0] != '\0' ? ", " : "",
                                 s_prop->type_data.map[j].name);
                }
                if (val[0] == '\0') {
                        snprintf_ch(val, "%s", s_prop->type_data.map[0].name);
                }

                log_msg(LOG_LEVEL_INFO, "%s%s: %s\n", log_prefix,
                        s_prop->prop_name, val);
                break;
        }
        case ST_ENUM: {
                const char *val = "unknown";
                for (unsigned j = 0; s_prop->type_data.map[j].name != nullptr;
                     ++j) {
                        if (s_prop->type_data.map[j].val == int_val) {
                                val = s_prop->type_data.map[j].name;
                                break;
                        }
                }

                log_msg(LOG_LEVEL_INFO, "%s%s: %s\n", log_prefix,
                        s_prop->prop_name, val);
                break;
        }
        }
}

// from BMD SDK sample StatusMonitor.cpp
class BMDNotificationCallback : public IDeckLinkNotificationCallback
{
      public:
        explicit BMDNotificationCallback(
            IDeckLinkStatus       *deckLinkStatus,
            IDeckLinkNotification *deckLinkNotification, const char *log_prefix)
            : m_deckLinkStatus(deckLinkStatus),
              m_deckLinkNotification(deckLinkNotification),
              m_logPrefix(log_prefix), m_refCount(1)

        {
                m_deckLinkStatus->AddRef();
        }

        // Implement the IDeckLinkNotificationCallback interface
        HRESULT STDMETHODCALLTYPE Notify(BMDNotifications topic,
                                         uint64_t         param1,
                                         uint64_t /* param2 */) override
        {
                // Check whether the notification we received is a status
                // notification
                if (topic != bmdStatusChanged) {
                        return S_OK;
                }

                // Print the updated status value
                auto statusId = (BMDDeckLinkStatusID) param1;
                if (statusId == bmdDeckLinkStatusDeviceTemperature) {
                        HandleTemperature();
                        return S_OK;
                }
                print_status_item(m_deckLinkStatus, statusId,
                                  m_logPrefix.c_str());

                return S_OK;
        }

        // IUnknown needs only a dummy implementation
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID /* iid */,
                                                 LPVOID * /* ppv */) override
        {
                return E_NOINTERFACE;
        }

        ULONG STDMETHODCALLTYPE AddRef() override { return ++m_refCount; }

        ULONG STDMETHODCALLTYPE Release() override
        {
                ULONG newRefValue = --m_refCount;

                if (newRefValue == 0) {
                        delete this;
                }

                return newRefValue;
        }

        void HandleTemperature() {
                int64_t         cur_temp = 0;
                m_deckLinkStatus->GetInt(bmdDeckLinkStatusDeviceTemperature,
                                         &cur_temp);
                // check overheating
                if (cur_temp >= m_tempThresholdErr) {
                        log_msg(LOG_LEVEL_ERROR,
                                "%sDevice is overheating! The temperature is "
                                "%" PRId64 " °C.\n",
                                m_logPrefix.c_str(), cur_temp);
                        return;
                }
                if (cur_temp < m_tempThresholdWarn &&
                    log_level < LOG_LEVEL_VERBOSE) {
                        return;
                }
                const time_ns_t now = get_time_in_ns();
                if (cur_temp >= m_tempThresholdWarn &&
                    now - m_tempWarnLastShown > m_tempShowIntervalWarn) {
                        log_msg(
                            LOG_LEVEL_WARNING,
                            "%sDevice temperature is %" PRId64 " °C (>= %d °C).\n",
                            m_logPrefix.c_str(), cur_temp, m_tempThresholdWarn);
                        m_tempWarnLastShown = now;
                        return;
                }

                // normal behavior - print once a minute in verbose
                if (now - m_tempLastShown < m_tempShowInterval) {
                        return;
                }
                print_status_item(m_deckLinkStatus,
                                  bmdDeckLinkStatusDeviceTemperature,
                                  m_logPrefix.c_str());
                m_tempLastShown = now;
        }

      private:
        IDeckLinkStatus       *m_deckLinkStatus;
        IDeckLinkNotification *m_deckLinkNotification;
        string                 m_logPrefix;
        std::atomic<ULONG>     m_refCount;

        // temperature check
        static constexpr time_ns_t m_tempShowInterval     = SEC_TO_NS(60);
        static constexpr time_ns_t m_tempShowIntervalWarn = SEC_TO_NS(20);
        static constexpr int       m_tempThresholdWarn    = 77;
        static constexpr int       m_tempThresholdErr     = 82;
        time_ns_t                  m_tempLastShown        = 0;
        time_ns_t                  m_tempWarnLastShown    = 0;

        virtual ~BMDNotificationCallback()
        {
                BMD_CHECK(
                    m_deckLinkNotification->Unsubscribe(bmdStatusChanged, this),
                    "BMD device notification unsubscribe", BMD_NOOP);
                m_deckLinkStatus->Release();
                m_deckLinkNotification->Release();
        }

      public:
};

/**
 * @brief dump DeckLink status + register to notifications
 *
 * Currently only Ethernet status for DeckLink IP devices are printed.
 * @param capture   if true, skip irrelevant values like output addresses (A/V)
 *
 * @note
 * Subscribing for notification is needed because if user sets eg. an IP
 * address, the change isn't performed immediately so that the old value is
 * actually printed while the new value is set later and it can be observed by
 * the notification observer.
 *
 * @returns a pointer representing the notification callback, must be passed to
 * destroy with bmd_unsubscribe_notify()
 */
BMDNotificationCallback *
bmd_print_status_subscribe_notify(IDeckLink *deckLink, const char *log_prefix,
                                  bool capture)
{
        IDeckLinkProfileAttributes *deckLinkAttributes = nullptr;
        HRESULT                     result = deckLink->QueryInterface(
            IID_IDeckLinkProfileAttributes, (void **) &deckLinkAttributes);
        if (SUCCEEDED(result)) {
                BMD_STR string_val{};
                if (SUCCEEDED(deckLinkAttributes->GetString(
                        BMDDeckLinkEthernetMACAddress, &string_val))) {
                        string mac_addr = get_str_from_bmd_api_str(string_val);
                        release_bmd_api_str(string_val);
                        log_msg(LOG_LEVEL_INFO, "%sEthernet MAC address: %s\n",
                                log_prefix, mac_addr.c_str());
                }
                deckLinkAttributes->Release();
        } else {
                log_msg(LOG_LEVEL_ERROR,
                        "%sCannot obtain IID_IDeckLinkProfileAttributes from "
                        "DeckLink: %s\n",
                        log_prefix, bmd_hresult_to_string(result).c_str());
        }

        IDeckLinkStatus *deckLinkStatus = nullptr;
        BMD_CHECK(deckLink->QueryInterface(IID_IDeckLinkStatus,
                                           (void **) &deckLinkStatus),
                  "Cannot obtain IID_IDeckLinkStatus from DeckLink",
                  return nullptr);
        // print status_map values now
        for (unsigned u = 0; u < ARR_COUNT(status_map); ++u) {
                if (capture && status_map[u].playback_only) {
                        continue;
                }
                print_status_item(deckLinkStatus, status_map[u].prop,
                                  log_prefix);
        }

        // Obtain the notification interface
        IDeckLinkNotification *deckLinkNotification = nullptr;
        BMD_CHECK(deckLink->QueryInterface(IID_IDeckLinkNotification,
                                           (void **) &deckLinkNotification),
                  "Could not obtain the IDeckLinkNotification interface",
                  deckLinkStatus->Release();
                  return nullptr);

        auto *notificationCallback = new BMDNotificationCallback(
            deckLinkStatus, deckLinkNotification, log_prefix);
        assert(notificationCallback != nullptr);

        BMD_CHECK(deckLinkNotification->Subscribe(bmdStatusChanged,
                                                  notificationCallback),
                  "Could not subscribe to the status "
                  "change notification",
                  notificationCallback->Release();
                  return nullptr);

        return notificationCallback;
}

/**
 * @param notificationCallback the pointer returned by
 * bmd_print_status_subscribe_notify(); may be nullptr
 */
void
bmd_unsubscribe_notify(BMDNotificationCallback *notificationCallback)
{
        if (notificationCallback == nullptr) {
                return;
        }

        notificationCallback->Release();
}

/// parse bmd_option from given arg in format FourCC[=val]
bool
bmd_parse_fourcc_arg(
    map<BMDDeckLinkConfigurationID, bmd_option> &device_options,
    const char                                  *arg)
{
        const char *val = nullptr;

        char tmp[STR_LEN];
        if (strchr(arg, '=') != nullptr) {
                snprintf_ch(tmp, "%s", strchr(arg, '=') + 1);
                replace_all(tmp, DELDEL, ":");
                val = tmp;
        }
        return device_options[(BMDDeckLinkConfigurationID) bmd_read_fourcc(arg)]
            .parse(val);
}

/**
 * validates bmd_option parameter combination, only issue warnings
 *
 * currently it just warns if IP address for DeckLink IP is given but no
 * DHCP (disabled) because if not disabled, DHCP overrides the address
 * (link-local IPv4 addr used if DHCP serv not present).
 */
void
bmd_options_validate(
    map<BMDDeckLinkConfigurationID, bmd_option> &device_options)
{
        if (device_options.find(
                bmdDeckLinkConfigEthernetStaticLocalIPAddress) !=
                device_options.end() &&
            device_options.find(bmdDeckLinkConfigEthernetUseDHCP) ==
                device_options.end()) {
                MSG(WARNING,
                    "IP address set but DHCP not disabled via command-line "
                    "(but may be disabled in settings), consider adding ':DHCP=no'.\n");
        }
}

ADD_TO_PARAM(R10K_FULL_OPT, "* " R10K_FULL_OPT "\n"
                "  Do not do conversion from/to limited range on in/out for R10k on BMD devs.\n");
ADD_TO_PARAM(BMD_NAT_SORT, "* " BMD_NAT_SORT "\n"
                "  Use the old BMD device sorting.\n");


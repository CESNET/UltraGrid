/**
 * @file   blackmagic_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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

#include "debug.h"

#include "blackmagic_common.h"
#include "DeckLinkAPIVersion.h"
#include <unordered_map>

using namespace std;

static unordered_map<HRESULT, string> bmd_hresult_to_string_map = {
        {S_OK, "success"},
        {S_FALSE, "false"},
        {E_UNEXPECTED, "unexpected value"},
        {E_NOTIMPL, "not implemented"},
        {E_OUTOFMEMORY, "out of memory"},
        {E_INVALIDARG, "invalid argument"},
        {E_NOINTERFACE, "interface was not found"},
        {E_POINTER, "invalid pointer"},
        {E_HANDLE, "invalid handle"},
        {E_ABORT, "operation aborted"},
        {E_FAIL, "failure"},
        {E_ACCESSDENIED, "access denied"},
};

string bmd_hresult_to_string(HRESULT res)
{
        auto it = bmd_hresult_to_string_map.find(res);
        if (it != bmd_hresult_to_string_map.end()) {
                return it->second;
        }
        return {};
}

/**
 * returned c-sring needs to be freed when not used
 */
const char *get_cstr_from_bmd_api_str(BMD_STR bmd_string)
{
       const char *cstr;
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

void release_bmd_api_str(BMD_STR string)
{
#ifdef HAVE_MACOSX
        CFRelease(string);
#elif defined WIN32
        SysFreeString(string);
#else
        free((void *) string);
#endif
}

/**
 * @note
 * Each successful call (returning non-null pointer) of this function with coinit == true
 * should be followed by decklink_uninitialize() when done with DeckLink (not when releasing
 * IDeckLinkIterator!), typically on application shutdown.
 */
IDeckLinkIterator *create_decklink_iterator(bool verbose, bool coinit)
{
        IDeckLinkIterator *deckLinkIterator = nullptr;
#ifdef WIN32
        if (coinit) {
                // Initialize COM on this thread
                HRESULT result = CoInitialize(NULL);
                if(FAILED(result)) {
                        fprintf(stderr, "Initialize of COM failed - result = "
                                        "%08lx.\n", result);
                        if (result == S_FALSE) {
                                CoUninitialize();
                        }
                        return NULL;
                }
        }
        HRESULT result = CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL,
                        IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result)) {
                deckLinkIterator = nullptr;
        }
#else
        UNUSED(coinit);
        deckLinkIterator = CreateDeckLinkIteratorInstance();
#endif

        if (!deckLinkIterator && verbose) {
                fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
                fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);

        }

        return deckLinkIterator;
}

void decklink_uninitialize()
{
#ifdef WIN32
        CoUninitialize();
#endif
}

bool blackmagic_api_version_check()
{
        bool ret = false;
        IDeckLinkAPIInformation *APIInformation = NULL;
        HRESULT result;

#ifdef WIN32
        // Initialize COM on this thread
        result = CoInitialize(NULL);
        if(FAILED(result)) {
                fprintf(stderr, "Initialize of COM failed - result = "
                                "%08lx.\n", result);
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
        int64_t value;
        result = APIInformation->GetInt(BMDDeckLinkAPIVersion, &value);
        if(result != S_OK) {
                fprintf(stderr, "Cannot get API version!\n");
                goto cleanup;
        }

        if (BLACKMAGIC_DECKLINK_API_VERSION > value) { // this is safe comparision, for internal structure please see SDK documentation
                fprintf(stderr, "The DeckLink drivers may not be installed or are outdated.\n");
                fprintf(stderr, "You should have at least the version UltraGrid has been linked with.\n");
                fprintf(stderr, "Vendor download page is http://http://www.blackmagic-design.com/support\n");
                print_decklink_version();
                ret = false;
        } else {
                ret = true;
        }

cleanup:
        if (APIInformation) {
                APIInformation->Release();
        }
#ifdef WIN32
        CoUninitialize();
#endif

        return ret;
}

void print_decklink_version()
{
        BMD_STR current_version = NULL;
        IDeckLinkAPIInformation *APIInformation = NULL;
        HRESULT result;

#ifdef WIN32
        // Initialize COM on this thread
        result = CoInitialize(NULL);
        if(FAILED(result)) {
                fprintf(stderr, "Initialize of COM failed - result = "
                                "%08lx.\n", result);
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
                free((void *)currentVersionCString);
        }

cleanup:
        if (APIInformation) {
                APIInformation->Release();
        }
#ifdef WIN32
        CoUninitialize();
#endif
}


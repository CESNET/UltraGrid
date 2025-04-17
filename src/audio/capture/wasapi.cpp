/**
 * @file   audio/capture/wasapi.cpp
 * @author Martin Pulec <martin.pulec@cesnet.cz>
 */
/*
 *  Copyright (c) 2019-2023 CESNET, z. s. p. o.
 *  All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * @file
 * @todo
 * Probe and help are almost similar - consolide common code.
 */


#include <assert.h>               // for assert
#include <audiosessiontypes.h>    // for AUDCLNT_STREAMFLAGS_LOOPBACK, _AUDC...
#include <audioclient.h>
#include <basetsd.h>              // for HRESULT, UINT, LPWSTR, GUID, UINT32
#include <cctype>                 // for isdigit
#include <cstdio>                 // for mesnprintf
#include <cstdlib>                // for realloc, atoi, malloc, free
#include <cstring>                // for NULL, memset, strcmp, strlen, wcslen
#include <combaseapi.h>           // for CoTaskMemFree, CoCreateInstance
#include <cwchar>                 // for mbsrtowcs
#include <guiddef.h>              // for IsEqualGUID
#include <iomanip>
#include <iostream>
#include <ksmedia.h>              // for KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
#include <mediaobj.h>             // for REFERENCE_TIME
#include <mmeapi.h>               // for WAVEFORMATEX, WAVE_FORMAT_PCM
#include <mmdeviceapi.h>
#include <mmreg.h>                // for PWAVEFORMATEXTENSIBLE, WAVE_FORMAT_...
#include <objbase.h>              // for STGM_READ
#include <propidl.h>              // for PropVariantClear, PropVariantInit
#include <propsys.h>              // for IPropertyStore
#include <sstream>
#include <string>
#include <synchapi.h>             // for Sleep
#include <winerror.h>             // for SUCCEEDED, S_OK, FAILED

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "debug.h"
#include "host.h"                 // for audio_capture_channels, audio_captu...
#include "lib_common.h"
#include "types.h"                // for device_info
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/macros.h"         // for snprintf_ch
#include "utils/windows.h"

#define MOD_NAME "[WASAPI cap.] "
#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000

//const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
//const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
//const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);
const static GUID IDevice_FriendlyName = { 0xa45c254e, 0xdf1c, 0x4efd, { 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0 } };
const static PROPERTYKEY PKEY_Device_FriendlyName = { IDevice_FriendlyName, 14 };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces" // not our issue - defined by Mingw-w64
const GUID KSDATAFORMAT_SUBTYPE_PCM = { STATIC_KSDATAFORMAT_SUBTYPE_PCM };
const GUID KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = { STATIC_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT };
#pragma GCC diagnostic pop

using std::cout;
using std::ostringstream;
using std::setfill;
using std::setw;
using std::string;
using std::wstring;

struct state_acap_wasapi {
        bool com_initialized;
        struct audio_frame frame;
        IMMDevice *pDevice;
        IAudioClient *pAudioClient;
        IAudioCaptureClient *pCaptureClient;
        UINT32 bufferSize;
};

enum { IDX_LOOP = -2, IDX_DFL = -1 };

string wasapi_get_name(IMMDevice *pDevice);
static void show_help();
string wasapi_get_default_device_id(EDataFlow dataFlow, IMMDeviceEnumerator *enumerator);

#define SAFE_RELEASE(u) \
    do { if ((u) != NULL) (u)->Release(); (u) = NULL; } while(0)
#undef THROW_IF_FAILED
#define THROW_IF_FAILED(cmd) do { HRESULT hr = cmd; if (!SUCCEEDED(hr)) { ostringstream oss; oss << #cmd << ": " << hresult_to_str(hr); throw ug_runtime_error(oss.str()); } } while(0)
static void audio_cap_wasapi_probe(struct device_info **available_devices, int *dev_count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = nullptr;
        *dev_count = 0;
        bool com_initialized = false;

        if (!com_initialize(&com_initialized, MOD_NAME)) {
                return;
        }
        IMMDeviceEnumerator *enumerator = nullptr;
        IMMDeviceCollection *pEndpoints = nullptr;

        try {
                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));

                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATEMASK_ALL, &pEndpoints));
                UINT count;
                THROW_IF_FAILED(pEndpoints->GetCount(&count));
                for (UINT i = 0; i < count; ++i) {
                        IMMDevice *pDevice = nullptr;
                        LPWSTR pwszID = NULL;
                        try {
                                THROW_IF_FAILED(pEndpoints->Item(i, &pDevice));
                                THROW_IF_FAILED(pDevice->GetId(&pwszID));
                                *available_devices = (struct device_info *) realloc(*available_devices, (*dev_count + 1) * sizeof(struct device_info));
                                memset(&(*available_devices)[*dev_count], 0, sizeof(struct device_info));
                                snprintf((*available_devices)[*dev_count].dev, sizeof (*available_devices)[*dev_count].dev, ":%u", i); ///< @todo This may be rather id than index
                                snprintf(
                                    (*available_devices)[*dev_count].name,
                                    sizeof(*available_devices)[*dev_count].name,
                                    "WASAPI %s", wasapi_get_name(pDevice).c_str());

                                ++*dev_count;
                        } catch (ug_runtime_error &e) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Device " << i << ": " << e.what() << "\n";
                        }
                        SAFE_RELEASE(pDevice);
                        CoTaskMemFree(pwszID);
                }
        } catch (...) {
        }
        SAFE_RELEASE(enumerator);
        SAFE_RELEASE(pEndpoints);
        com_uninitialize(&com_initialized);
        // add looopback
        *available_devices = (struct device_info *)realloc(
            *available_devices, (*dev_count + 1) * sizeof(struct device_info));
        memset(&(*available_devices)[*dev_count], 0,
               sizeof(struct device_info));
        snprintf_ch((*available_devices)[*dev_count].dev, ":loopback");
        snprintf_ch((*available_devices)[*dev_count].name,
                 "WASAPI computer audio output");
        *dev_count += 1;
}

string wasapi_get_name(IMMDevice *pDevice) {
        wstring out;
        IPropertyStore *pProps = NULL;
        PROPVARIANT varName;
        LPWSTR pwszID = NULL;
        // Initialize container for property value.
        PropVariantInit(&varName);
        HRESULT hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
        if (!FAILED(hr)) {
                hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
                if (!FAILED(hr)) {
                        out = varName.pwszVal;
                }
        }
        SAFE_RELEASE(pProps);
        CoTaskMemFree(pwszID);
        PropVariantClear(&varName);
        return win_wstr_to_str(out.c_str());
}

string wasapi_get_default_device_id(EDataFlow dataFlow, IMMDeviceEnumerator *enumerator) {
        IMMDevice *pDevice = nullptr;
        if (!SUCCEEDED(enumerator->GetDefaultAudioEndpoint(dataFlow, eConsole, &pDevice))) {
                return {};
        }
        LPWSTR pwszID = nullptr;
        if (!SUCCEEDED(pDevice->GetId(&pwszID))) {
                SAFE_RELEASE(pDevice);
                return {};
        }
        string ret = win_wstr_to_str(pwszID);
        SAFE_RELEASE(pDevice);

        return ret;
}

static void show_help() {
        col() << "Usage:\n"
              << SBOLD(SRED("\t-s wasapi") << "[:d[evice]=<index>|<ID>|<name>]")
              << "\n\nAvailable devices:\n";

        IMMDeviceEnumerator *enumerator = nullptr;
        IMMDeviceCollection *pEndpoints = nullptr;
        bool com_initialized = false;
        if (!com_initialize(&com_initialized, MOD_NAME)) {
                return;
        }

        try {
                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));
                string default_dev_id = wasapi_get_default_device_id(eCapture, enumerator);
                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATEMASK_ALL, &pEndpoints));
                UINT count;
                THROW_IF_FAILED(pEndpoints->GetCount(&count));
                for (UINT i = 0; i < count; ++i) {
                        IMMDevice *pDevice = nullptr;
                        LPWSTR pwszID = NULL;
                        try {
                                THROW_IF_FAILED(pEndpoints->Item(i, &pDevice));
                                THROW_IF_FAILED(pDevice->GetId(&pwszID));
                                string dev_id = win_wstr_to_str(pwszID);
                                color_printf(
                                    "%s\t" TBOLD("%2d") ") " TBOLD(
                                        "%s") " (ID: %s)\n",
                                    (dev_id == default_dev_id ? "(*)" : ""), i,
                                    wasapi_get_name(pDevice).c_str(),
                                    dev_id.c_str());
                        } catch (ug_runtime_error &e) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Device " << i << ": " << e.what() << "\n";
                        }
                        SAFE_RELEASE(pDevice);
                        CoTaskMemFree(pwszID);
                }
        } catch (...) {
        }
        SAFE_RELEASE(enumerator);
        SAFE_RELEASE(pEndpoints);
        com_uninitialize(&com_initialized);
        col() << "  " << SBOLD("loopback") << ") " << SBOLD("computer audio output") << " (ID: loopback)\n";
        printf("\nDevice " TBOLD("name") " can be a substring (selects first matching device).\n");
}

static void
parse_fmt(const char *cfg, int *req_index, char *req_dev_name,
          size_t req_dev_name_sz, wchar_t *req_deviceID, size_t req_deviceID_sz)
{
        if (strlen(cfg) == 0) {
                return;
        }

        if (IS_KEY_PREFIX(cfg, "device")) {
                cfg = strchr(cfg, '=') + 1;
        }

        if (isdigit(cfg[0])) {
                *req_index = atoi(cfg);
        } else if (strcmp(cfg, "loopback") == 0) {
                *req_index = IDX_LOOP;
        } else if (cfg[0] == '{') { // ID
                const char *uuid = cfg;
                mbstate_t   state{};
                mbsrtowcs(req_deviceID, &uuid, req_deviceID_sz - 1, &state);
                assert(uuid == NULL);
        } else { // name
                snprintf(req_dev_name, req_dev_name_sz, "%s", cfg);
        }
}

static void * audio_cap_wasapi_init(struct module *parent, const char *cfg)
{
        if (strcmp(cfg, "help") == 0) {
                show_help();
                return INIT_NOERR;
        }
        UNUSED(parent);
        WAVEFORMATEX *pwfx = NULL;

        int index = IDX_DFL;          // or
        char req_dev_name[1024] = ""; // or
        wchar_t deviceID[1024] = L"";

        parse_fmt(cfg, &index, req_dev_name, sizeof req_dev_name, deviceID,
                  sizeof deviceID);

        auto s = new state_acap_wasapi();
        if (!com_initialize(&s->com_initialized, MOD_NAME)) {
                delete s;
                return nullptr;
        }
        IMMDeviceEnumerator *enumerator = nullptr;
        try {

                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));
                if (wcslen(deviceID) > 0) {
                        THROW_IF_FAILED(enumerator->GetDevice(deviceID,  &s->pDevice));
                } else if (index >= 0 || strlen(req_dev_name) > 0)  {
                        IMMDeviceCollection *pEndpoints = nullptr;
                        try {
                                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATEMASK_ALL, &pEndpoints));
                                UINT count;
                                THROW_IF_FAILED(pEndpoints->GetCount(&count));
                                for (UINT i = 0; i < count; ++i) {
                                        if (i == (UINT) index) {
                                                THROW_IF_FAILED(pEndpoints->Item(i, &s->pDevice));
                                                break;
                                        }
                                        if (strlen(req_dev_name) > 0) {
                                                IMMDevice *pDevice = nullptr;
                                                pEndpoints->Item(i, &pDevice);
                                                if (pDevice != nullptr &&
                                                    wasapi_get_name(pDevice)
                                                            .find(
                                                                req_dev_name) !=
                                                        std::string::npos) {
                                                        s->pDevice = pDevice;
                                                        break;
                                                }
                                                SAFE_RELEASE(pDevice);
                                        }
                                }
                        } catch (ug_runtime_error &e) { // just continue with the next
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << e.what() << "\n";
                        }
                        SAFE_RELEASE(pEndpoints);
                } else { // default or loopback device
                        THROW_IF_FAILED(enumerator->GetDefaultAudioEndpoint(
                            index == IDX_DFL ? eCapture : eRender, eConsole,
                            &s->pDevice));
                }
                if (!s->pDevice) {
                        throw ug_runtime_error("Device not found!");
                }
                THROW_IF_FAILED(s->pDevice->Activate(IID_IAudioClient, CLSCTX_ALL, NULL,
                                (void **)&s->pAudioClient));

                auto friendlyName = wasapi_get_name(s->pDevice);
                if (!friendlyName.empty()) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Using device: "
                                << friendlyName << "\n";
                }

                REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
                // get the mixer format
                THROW_IF_FAILED(s->pAudioClient->GetMixFormat(&pwfx));
                // set our preferences
                if (audio_capture_channels != 0) {
                        pwfx->nChannels = audio_capture_channels;
                }
                if (audio_capture_sample_rate) {
                        pwfx->nSamplesPerSec = audio_capture_sample_rate;
                }
                pwfx->wBitsPerSample = 16;
                if (audio_capture_bps) {
                        pwfx->wBitsPerSample = audio_capture_bps * 8;
                }
                // coerce the format
                // taken from https://github.com/Riateche/SoundStreamer/blob/master/SoundStreamer/SoundStreamer.cpp
                switch (pwfx->wFormatTag) {
                        case WAVE_FORMAT_IEEE_FLOAT:
                                pwfx->wFormatTag = WAVE_FORMAT_PCM;
                                pwfx->nBlockAlign = pwfx->nChannels * pwfx->wBitsPerSample / 8;
                                pwfx->nAvgBytesPerSec = pwfx->nBlockAlign * pwfx->nSamplesPerSec;
                                break;

                        case WAVE_FORMAT_EXTENSIBLE:
                                {
                                        // naked scope for case-local variable
                                        PWAVEFORMATEXTENSIBLE waveFormatEx = reinterpret_cast<PWAVEFORMATEXTENSIBLE>(pwfx);
                                        if (IsEqualGUID(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, waveFormatEx->SubFormat)) {
                                                waveFormatEx->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;
                                                waveFormatEx->Samples.wValidBitsPerSample =
                                                        pwfx->wBitsPerSample;
                                                pwfx->nBlockAlign = pwfx->nChannels * pwfx->wBitsPerSample / 8;
                                                pwfx->nAvgBytesPerSec = pwfx->nBlockAlign * pwfx->nSamplesPerSec;
                                        } else {
                                                throw ug_runtime_error("Don't know how to coerce mix format to int-16");
                                        }
                                }
                                break;

                        default:
                                ostringstream oss;
                                oss << "Don't know how to coerce WAVEFORMATEX with wFormatTag = 0x" << setw(8) << setfill('0') <<  " to PCM";
                                throw ug_runtime_error(oss.str());
                }

                THROW_IF_FAILED(s->pAudioClient->Initialize(
                         AUDCLNT_SHAREMODE_SHARED,
                         index == IDX_LOOP ? AUDCLNT_STREAMFLAGS_LOOPBACK : 0,
                         hnsRequestedDuration,
                         0,
                         pwfx,
                         NULL));

                UINT32 bufferFrameCount;
                THROW_IF_FAILED(s->pAudioClient->GetBufferSize(&bufferFrameCount));
                s->frame.bps = pwfx->wBitsPerSample / 8;
                s->frame.sample_rate = pwfx->nSamplesPerSec;
                s->frame.ch_count = pwfx->nChannels;
                s->frame.max_size = bufferFrameCount * s->frame.bps * s->frame.ch_count;
                s->frame.data = (char *) malloc(s->frame.max_size);

                THROW_IF_FAILED(s->pAudioClient->GetService(IID_IAudioCaptureClient, (void **) &s->pCaptureClient));
                THROW_IF_FAILED(s->pAudioClient->Start());

        } catch (ug_runtime_error &e) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
                if (audio_capture_channels != 0) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Maybe wrong number of channels? Try using default.";
                }
                CoUninitialize();
                delete s;
                s = nullptr;
        }

        SAFE_RELEASE(enumerator);
        CoTaskMemFree(pwfx);
        return s;
}

static void audio_cap_wasapi_done(void *state)
{
        CoUninitialize();
        delete static_cast<state_acap_wasapi *>(state);
}

#define FAIL_IF_NOT(cmd) do {HRESULT hr = cmd; if (hr != S_OK) { LOG(LOG_LEVEL_ERROR) << MOD_NAME << #cmd << ": " << hresult_to_str(hr) << "\n"; return nullptr;}} while(0)
static struct audio_frame *audio_cap_wasapi_read(void *state)
{
        auto s = static_cast<state_acap_wasapi *>(state);

        UINT32 packetLength = 0;
        FAIL_IF_NOT(s->pCaptureClient->GetNextPacketSize(&packetLength));
        if (packetLength == 0) {
                Sleep(10);
        }

        FAIL_IF_NOT(s->pCaptureClient->GetNextPacketSize(&packetLength));

        s->frame.data_len = 0;

        while (packetLength != 0) {
		UINT32 numFramesAvailable;
		BYTE *pData;
		DWORD flags;
		// Get the available data in the shared buffer.
		FAIL_IF_NOT(s->pCaptureClient->GetBuffer(
				&pData,
				&numFramesAvailable,
				&flags, NULL, NULL));

		//if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
		//{
		//	pData = NULL;  // Tell CopyData to write silence.
		//}

		// Copy the available capture data to the audio sink.
		int count = numFramesAvailable * s->frame.bps * s->frame.ch_count;
		if (s->frame.data_len + count > s->frame.max_size) {
			count = s->frame.max_size - s->frame.data_len;
		}
		memcpy(s->frame.data + s->frame.data_len, pData, count);
                //fprintf(stderr, "%d\n", count);
                s->frame.data_len += count;
		FAIL_IF_NOT(s->pCaptureClient->ReleaseBuffer(count / s->frame.ch_count / s->frame.bps));
		if (s->frame.data_len == s->frame.max_size) {
			return &s->frame;
		}

		FAIL_IF_NOT(s->pCaptureClient->GetNextPacketSize(&packetLength));
        }
	return &s->frame;
}

static const struct audio_capture_info acap_wasapi_info = {
        audio_cap_wasapi_probe,
        audio_cap_wasapi_init,
        audio_cap_wasapi_read,
        audio_cap_wasapi_done
};

REGISTER_MODULE(wasapi, &acap_wasapi_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */


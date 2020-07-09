/**
 * @file   audio/capture/wasapi.cpp
 * @author Martin Pulec <martin.pulec@cesnet.cz>
 */
/*
 *  Copyright (c) 2019 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <audioclient.h>
#include <iomanip>
#include <iostream>
#include <mfapi.h>
#include <mmdeviceapi.h>
#include <sstream>
#include <string>
#include <windows.h>

#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include "rang.hpp"
#include "ug_runtime_error.hpp"
#include "utils/hresult.h"

#define MOD_NAME "[WASAPI cap.] "
#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000

//const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
//const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
//const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);
const static GUID IDevice_FriendlyName = { 0xa45c254e, 0xdf1c, 0x4efd, { 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0 } };
const static PROPERTYKEY PKEY_Device_FriendlyName = { IDevice_FriendlyName, 14 };
const GUID KSDATAFORMAT_SUBTYPE_PCM = { STATIC_KSDATAFORMAT_SUBTYPE_PCM };
const GUID KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = { STATIC_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT };

using rang::fg;
using rang::style;
using std::cout;
using std::ostringstream;
using std::setfill;
using std::setw;
using std::string;
using std::wstring;

struct state_acap_wasapi {
        struct audio_frame frame;
        IMMDevice *pDevice;
        IAudioClient *pAudioClient;
        IAudioCaptureClient *pCaptureClient;
        UINT32 bufferSize;
};

static string get_name(IMMDevice *pDevice);
static void show_help();

#define SAFE_RELEASE(u) \
    do { if ((u) != NULL) (u)->Release(); (u) = NULL; } while(0)
#undef THROW_IF_FAILED
#define THROW_IF_FAILED(cmd) do { HRESULT hr = cmd; if (!SUCCEEDED(hr)) { ostringstream oss; oss << #cmd << ": " << hresult_to_str(hr); throw ug_runtime_error(oss.str()); } } while(0)
static void audio_cap_wasapi_probe(struct device_info **available_devices, int *dev_count)
{
        *available_devices = (struct device_info *) malloc(0);
        *dev_count = 0;

        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        IMMDeviceEnumerator *enumerator = nullptr;
        IMMDeviceCollection *pEndpoints = nullptr;
        if (hr != S_OK && hr != S_FALSE) {
                return;
        }

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
                                sprintf((*available_devices)[*dev_count].id, "wasapi:%u", i); ///< @todo This may be rather id than index
                                sprintf((*available_devices)[*dev_count].name, "WASAPI %s", get_name(pDevice).c_str());
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
        CoUninitialize();
}

static void audio_cap_wasapi_help(const char *driver_name)
{
        UNUSED(driver_name);
        show_help();
}

static string wstring_to_string(wstring const & wstr) {
        size_t len = wstr.length() * 4 + 1;
        auto str = (char *) calloc(len, 1);
        wcstombs(str, wstr.c_str(), len - 1);
        string out = str;
        free(str);
        return out;
}

static string get_name(IMMDevice *pDevice) {
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
        return wstring_to_string(out);
}

static void show_help() {
        cout << "Usage:\n" <<
                style::bold << fg::red << "\t-s wasapi" << fg::reset << "[:<index>|:<ID>]\n" << style::reset <<
                "\nAvailable devices:\n";

        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        IMMDeviceEnumerator *enumerator = nullptr;
        IMMDeviceCollection *pEndpoints = nullptr;
        if (hr != S_OK && hr != S_FALSE) {
                return;
        }

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
                                cout << "\t" << style::bold << i << style::reset << ") " << style::bold << get_name(pDevice) << style::reset << " (ID: " << wstring_to_string(pwszID) << ")\n";
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
        CoUninitialize();
}

static void * audio_cap_wasapi_init(const char *cfg)
{
        wchar_t deviceID[1024] = L"";
        WAVEFORMATEX *pwfx = NULL;
        int index = -1;
        if (cfg && strlen(cfg) > 0) {
                if (strcmp(cfg, "help") == 0) {
                        show_help();
                        return &audio_init_state_ok;
                }
                if (isdigit(cfg[0])) {
                        index = atoi(cfg);
                } else {
                        mbtowc(deviceID, cfg, (sizeof deviceID / 2) - 1);
                }
        }
        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (hr != S_OK && hr != S_FALSE) {
                return nullptr;
        }
        auto s = new state_acap_wasapi();
        IMMDeviceEnumerator *enumerator = nullptr;
        try {

                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));
                if (wcslen(deviceID) > 0) {
                        THROW_IF_FAILED(enumerator->GetDevice(deviceID,  &s->pDevice));
                } else if (index != -1)  {
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
                                }
                        } catch (ug_runtime_error &e) { // just continue with the next
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << e.what() << "\n";
                        }
                        SAFE_RELEASE(pEndpoints);
                } else { // default device
                        THROW_IF_FAILED(enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &s->pDevice));
                }
                if (!s->pDevice) {
                        throw ug_runtime_error("Device not found!");
                }
                THROW_IF_FAILED(s->pDevice->Activate(IID_IAudioClient, CLSCTX_ALL, NULL,
                                (void **)&s->pAudioClient));

                auto friendlyName = get_name(s->pDevice);
                if (!friendlyName.empty()) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Using device: "
                                << friendlyName << "\n";
                }

                REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
                // get the mixer format
                THROW_IF_FAILED(s->pAudioClient->GetMixFormat(&pwfx));
                // set our preferences
                if (audio_capture_channels) {
                        pwfx->nChannels = audio_capture_channels;
                }
                if (audio_capture_sample_rate) {
                        pwfx->nSamplesPerSec = pwfx->nSamplesPerSec;
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
                         0,
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
                usleep(10000);
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
        audio_cap_wasapi_help,
        audio_cap_wasapi_init,
        audio_cap_wasapi_read,
        audio_cap_wasapi_done
};

REGISTER_MODULE(wasapi, &acap_wasapi_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */


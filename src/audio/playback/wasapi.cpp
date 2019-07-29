/**
 * @file   audio/playback/wasapi.cpp
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
 * @todo
 * Code for probe and help is very similar - consolidate.
 * @todo
 * Audio buffer length should be rather adaptive to length of incoming
 * frames.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <audioclient.h>
#include <iostream>
#include <mfapi.h>
#include <mmdeviceapi.h>
#include <sstream>
#include <string>
#include <windows.h>

#include "audio/audio_playback.h"
#include "debug.h"
#include "lib_common.h"
#include "rang.hpp"
#include "ug_runtime_error.h"
#include "utils/hresult.h"

#define DEFAULT_WASAPI_BUFLEN_MS 67
#define MOD_NAME "[WASAPI play.] "
#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000

const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioRenderClient = __uuidof(IAudioRenderClient);
const static GUID IDevice_FriendlyName = { 0xa45c254e, 0xdf1c, 0x4efd, { 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0 } };
const static PROPERTYKEY PKEY_Device_FriendlyName = { IDevice_FriendlyName, 14 };
const static GUID UG_KSDATAFORMAT_SUBTYPE_PCM = { STATIC_KSDATAFORMAT_SUBTYPE_PCM };


using rang::fg;
using rang::style;
using std::cout;
using std::ostringstream;
using std::string;
using std::wcout;
using std::wstring;

struct state_aplay_wasapi {
        struct audio_desc desc;
        IMMDevice *pDevice;
        IAudioClient *pAudioClient;
        IAudioRenderClient *pRenderClient;
        UINT32 bufferSize;
};

static void show_help();

#define SAFE_RELEASE(u) \
    do { if ((u) != NULL) (u)->Release(); (u) = NULL; } while(0)
#undef THROW_IF_FAILED
#define THROW_IF_FAILED(cmd) do { HRESULT hr = cmd; if (!SUCCEEDED(hr)) { ostringstream oss; oss << #cmd << ": " << hresult_to_str(hr); throw ug_runtime_error(oss.str()); } } while(0)
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

static void audio_play_wasapi_probe(struct device_info **available_devices, int *dev_count)
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

                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATEMASK_ALL, &pEndpoints));
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

static void audio_play_wasapi_help(const char *driver_name)
{
        UNUSED(driver_name);
        show_help();
}

static void show_help() {
        cout << "Usage:\n" <<
                style::bold << fg::red << "\t-r wasapi" << fg::reset << "[:<index>|:<ID>] --param audio-buffer-len=<ms>\n" << style::reset <<
                "\nAvailable devices:\n";

        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (hr != S_OK && hr != S_FALSE) {
                return;
        }
        IMMDeviceEnumerator *enumerator = nullptr;
        IMMDeviceCollection *pEndpoints = nullptr;

        try {
                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));

                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATEMASK_ALL, &pEndpoints));
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

#undef THROW_IF_FAILED
#define THROW_IF_FAILED(cmd) do { HRESULT hr = cmd; if (!SUCCEEDED(hr)) { LOG(LOG_LEVEL_ERROR) << MOD_NAME << #cmd << ": " << hresult_to_str(hr) << "\n"; delete s; s = nullptr; break; } } while(0)
static void * audio_play_wasapi_init(const char *cfg)
{
        wchar_t deviceID[1024] = L"";
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
        auto s = new state_aplay_wasapi();
        IMMDeviceEnumerator *enumerator = nullptr;
        try {

                THROW_IF_FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
                                        (void **) &enumerator));
                if (wcslen(deviceID) > 0) {
                        THROW_IF_FAILED(enumerator->GetDevice(deviceID,  &s->pDevice));
                } else if (index != -1)  {
                        IMMDeviceCollection *pEndpoints = nullptr;
                        try {
                                THROW_IF_FAILED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATEMASK_ALL, &pEndpoints));
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
                        THROW_IF_FAILED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &s->pDevice));
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
        } catch (ug_runtime_error &e) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
                CoUninitialize();
                delete s;
                s = nullptr;
        }
        // print friendly name
        SAFE_RELEASE(enumerator);
        return s;
}

static void audio_play_wasapi_done(void *state)
{
        CoUninitialize();
        delete static_cast<state_aplay_wasapi *>(state);
}

static DWORD get_channel_mask(int *count) {
        assert(*count > 0);
        switch (*count) {
                case 1: return KSAUDIO_SPEAKER_MONO;
                case 2: return KSAUDIO_SPEAKER_STEREO;
                case 3: *count = 4; return KSAUDIO_SPEAKER_SURROUND;
                case 4: return KSAUDIO_SPEAKER_QUAD; // or SURROUND
                case 5: *count = 6; return KSAUDIO_SPEAKER_5POINT1_SURROUND;
                case 6: return KSAUDIO_SPEAKER_5POINT1; // ditto
                case 7: *count = 7; return KSAUDIO_SPEAKER_5POINT1_SURROUND;
                case 8: return KSAUDIO_SPEAKER_7POINT1; // ditto
                default: *count = 8; return KSAUDIO_SPEAKER_7POINT1;
        }
}

static WAVEFORMATEXTENSIBLE audio_format_to_waveformatex(struct audio_desc *desc) {
        assert(desc->codec == AC_PCM);
        WAVEFORMATEXTENSIBLE fmt;
        fmt.SubFormat = UG_KSDATAFORMAT_SUBTYPE_PCM;
        fmt.dwChannelMask = get_channel_mask(&desc->ch_count);
        fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        fmt.Format.nChannels = desc->ch_count;
        fmt.Format.wBitsPerSample = desc->bps * 8;
        fmt.Format.nSamplesPerSec = desc->sample_rate;
        fmt.Format.nBlockAlign = fmt.Format.nChannels * fmt.Format.wBitsPerSample / 8;
        fmt.Format.nAvgBytesPerSec = fmt.Format.nSamplesPerSec * fmt.Format.nBlockAlign;
        fmt.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX); // at least 22 for extensible
        fmt.Samples.wValidBitsPerSample = fmt.Format.wBitsPerSample;
        return fmt;
}

static bool audio_play_wasapi_ctl(void *state, int request, void *data, size_t *len)
{
        auto s = static_cast<struct state_aplay_wasapi *>(state);
        HRESULT hr;
        WAVEFORMATEXTENSIBLE fmt;
        WAVEFORMATEX *closestMatch;
        switch (request) {
                case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                        if (*len < sizeof(struct audio_desc)) {
                                return false;
                        }
                        struct audio_desc desc;
                        memcpy(&desc, data, sizeof desc);
                        desc.codec = AC_PCM;

                        fmt = audio_format_to_waveformatex(&desc);
                        hr = s->pAudioClient->IsFormatSupported(
                                        AUDCLNT_SHAREMODE_SHARED,
                                        (WAVEFORMATEX *) &fmt,
                                        &closestMatch);
                        if (hr != S_OK && hr != S_FALSE) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unable to get format: " << hresult_to_str(hr) << "\n";
                                return false;
                        }
                        if (hr == S_FALSE) {
                                desc.ch_count = closestMatch->nChannels;
                                desc.bps = closestMatch->wBitsPerSample / 8;
                                desc.sample_rate = closestMatch->nSamplesPerSec;
                                CoTaskMemFree(closestMatch);
                        }

                        memcpy(data, &desc, sizeof desc);
                        *len = sizeof desc;
                        return true;
                default:
                        return false;
        }
}

#define FAIL_IF_NOT(cmd) do {HRESULT hr = cmd; if (hr != S_OK) { LOG(LOG_LEVEL_ERROR) << MOD_NAME << #cmd << ": " << hresult_to_str(hr) << "\n"; return FALSE;}} while(0)
static int audio_play_wasapi_reconfigure(void *state, struct audio_desc desc)
{
        auto s = static_cast<struct state_aplay_wasapi *>(state);
        int buflen_ms = DEFAULT_WASAPI_BUFLEN_MS;
        if (get_commandline_param("audio-buffer-len") != nullptr) {
                buflen_ms = atoi(get_commandline_param("audio-buffer-len"));
                assert(buflen_ms > 0);
        }
        REFERENCE_TIME bufferDuration = buflen_ms * REFTIMES_PER_MILLISEC;
        WAVEFORMATEXTENSIBLE fmt = audio_format_to_waveformatex(&desc);
        FAIL_IF_NOT(s->pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, bufferDuration, 0,
                        (WAVEFORMATEX *) &fmt, NULL));
        FAIL_IF_NOT(s->pAudioClient->GetService(IID_IAudioRenderClient, (void **) &s->pRenderClient));

        FAIL_IF_NOT(s->pAudioClient->GetBufferSize(&s->bufferSize));
        LOG(LOG_LEVEL_INFO) << MOD_NAME "Buffer size: " << s->bufferSize << " frames\n";

        FAIL_IF_NOT(s->pAudioClient->Start());

        return TRUE;
}

#define CHECK(cmd) do {HRESULT hr = cmd; if (hr != S_OK) { LOG(LOG_LEVEL_ERROR) << MOD_NAME << #cmd ": " << hresult_to_str(hr) << "\n"; return;}} while(0)
static void audio_play_wasapi_put_frame(void *state, struct audio_frame *buffer)
{
        auto s = static_cast<struct state_aplay_wasapi *>(state);
        UINT32 numFramesPadding;

        CHECK(s->pAudioClient->GetCurrentPadding(&numFramesPadding));

        UINT32 NumFramesRequested = buffer->data_len / buffer->ch_count / buffer->bps;
        if (s->bufferSize - numFramesPadding < NumFramesRequested) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Buffer overflow!\n";
                NumFramesRequested = s->bufferSize - numFramesPadding;
        }
        BYTE *data;
        CHECK(s->pRenderClient->GetBuffer(NumFramesRequested, &data));
        memcpy(data, buffer->data, NumFramesRequested * buffer->ch_count * buffer->bps);
        CHECK(s->pRenderClient->ReleaseBuffer(NumFramesRequested, 0));

}

static const struct audio_playback_info aplay_wasapi_info = {
        audio_play_wasapi_probe,
        audio_play_wasapi_help,
        audio_play_wasapi_init,
        audio_play_wasapi_put_frame,
        audio_play_wasapi_ctl,
        audio_play_wasapi_reconfigure,
        audio_play_wasapi_done
};

REGISTER_MODULE(wasapi, &aplay_wasapi_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

/* vim: set expandtab sw=8: */


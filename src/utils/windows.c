/**
 * @file   utils/windows.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2025 CESNET
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

#ifdef _WIN32
#include <audioclient.h>
#include <ntstatus.h>
#include <objbase.h>
#include <tlhelp32.h>
#include <vfwmsgs.h>
#include <dbghelp.h>
#include <io.h>

#include "utils/macros.h"
#endif // defined _WIN32

#include "debug.h"
#include "utils/windows.h"

#define MOD_NAME "[utils/win] "

/**
 * @param[out] com_initialize a pointer to a bool that will be passed to
 * com_uninintialize(). The value should be initialized to false - the current
 * logic can guard only one init/uninit so true is assumed bo be a repeated call.
 * @param[in] err_prefix optional error prefix to be used for eventual error
 * messages (may be NULL)
 * @retval true  if either COM already initialized for this thread or this call
 * initializes COM successfully
 * @retval false  COM could not have been initialized
 */
bool com_initialize(bool *com_initialized, const char *err_prefix)
{
        if (*com_initialized) {
                MSG(WARNING, "com_initialized should be initialized to false "
                             "upon the call!\n");
        }
#ifdef _WIN32
        if (err_prefix == NULL) {
                err_prefix = "";
        }
        *com_initialized = false;
        // Initialize COM on this thread
        HRESULT result = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        if (SUCCEEDED(result)) {
                *com_initialized = true;
                return true;
        }
        if (result == RPC_E_CHANGED_MODE) {
                log_msg(LOG_LEVEL_WARNING, "%sCOM already initialized with a different mode!\n", err_prefix);
                return true;
        }
        log_msg(LOG_LEVEL_ERROR, "%sInitialize of COM failed - %s\n", err_prefix, hresult_to_str(result));
        return false;
#else
        (void) com_initialized, (void) err_prefix;
        return true;
#endif
}

void com_uninitialize(bool *com_initialized)
{
#ifdef _WIN32
        if (!*com_initialized) {
                return;
        }
        *com_initialized = false;
        CoUninitialize();
#else
        (void) com_initialized;
#endif
}

#ifdef _WIN32
const char *hresult_to_str(HRESULT res) {
        _Thread_local static char unknown[128];
        const char *errptr = NULL;

        if (HRESULT_FACILITY(res) == FACILITY_WIN32) {
                return get_win32_error(HRESULT_CODE(res));
        }

        HRESULT_GET_ERROR_COMMON(res, errptr)
        if (errptr) {
                return errptr;
        }

        switch (res) {
                case RPC_E_CHANGED_MODE: return "A previous call to CoInitializeEx specified different concurrency model for this thread."; break; \
                case CO_E_NOTINITIALIZED: return "CoInitialize has not been called.";
		case AUDCLNT_E_ALREADY_INITIALIZED: return "The IAudioClient object is already initialized.";
		case AUDCLNT_E_WRONG_ENDPOINT_TYPE: return "The AUDCLNT_STREAMFLAGS_LOOPBACK flag is set but the endpoint device is a capture device, not a rendering device.";
		case AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED: return "The requested buffer size is not aligned. This code can be returned for a render or a capture device if the caller specified AUDCLNT_SHAREMODE_EXCLUSIVE and the AUDCLNT_STREAMFLAGS_EVENTCALLBACK flags. The caller must call Initialize again with the aligned buffer size.";
		case AUDCLNT_E_BUFFER_SIZE_ERROR: return "Indicates that the buffer duration value requested by an exclusive-mode client is out of range. The requested duration value for pull mode must not be greater than 500 milliseconds; for push mode the duration value must not be greater than 2 seconds.";
		case AUDCLNT_E_CPUUSAGE_EXCEEDED: return "Indicates that the process-pass duration exceeded the maximum CPU usage. The audio engine keeps track of CPU usage by maintaining the number of times the process-pass duration exceeds the maximum CPU usage. The maximum CPU usage is calculated as a percent of the engine's periodicity. The percentage value is the system's CPU throttle value (within the range of 10%% and 90%%). If this value is not found, then the default value of 40%% is used to calculate the maximum CPU usage.)";
		case AUDCLNT_E_DEVICE_INVALIDATED: return "The audio endpoint device has been unplugged, or the audio hardware or associated hardware resources have been reconfigured, disabled, removed, or otherwise made unavailable for use.";
		case AUDCLNT_E_DEVICE_IN_USE: return "The endpoint device is already in use. Either the device is being used in exclusive mode, or the device is being used in shared mode and the caller asked to use the device in exclusive mode.";
		case AUDCLNT_E_ENDPOINT_CREATE_FAILED: return "The method failed to create the audio endpoint for the render or the capture device. This can occur if the audio endpoint device has been unplugged, or the audio hardware or associated hardware resources have been reconfigured, disabled, removed, or otherwise made unavailable for use.";
		case AUDCLNT_E_INVALID_DEVICE_PERIOD: return "Indicates that the device period requested by an exclusive-mode client is greater than 500 milliseconds.";
		case AUDCLNT_E_UNSUPPORTED_FORMAT: return "The audio engine (shared mode) or audio endpoint device (exclusive mode) does not support the specified format.";
		case AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED: return "The caller is requesting exclusive-mode use of the endpoint device, but the user has disabled exclusive-mode use of the device.";
		case AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL: return "The AUDCLNT_STREAMFLAGS_EVENTCALLBACK flag is set but parameters hnsBufferDuration and hnsPeriodicity are not equal.";
		case AUDCLNT_E_SERVICE_NOT_RUNNING: return "The Windows audio service is not running.";
                case VFW_E_CANNOT_CONNECT:
                        return "DirectShow: No combination of intermediate filters "
                               "could be found to make the connection.";
                case VFW_E_NOT_CONNECTED:
                        return "DirectShow: The operation cannot be performed "
                               "because the pins are not connected.";
                default:
                       snprintf(unknown, sizeof unknown, "(unknown: %ld S=%lu,R=%lu,C=%lu,facility=%lu,code=%lu)", res, ((unsigned long int) res) >> 31u, (res >> 30u) & 0x1, (res >> 29u) & 0x1, (res >> 16u) & 0x7ff, res & 0xff);
                       return unknown;
        }
}

/**
 * formats Win32 return codes to string
 *
 * @param error   error code (typically GetLastError())
 *
 * @note returned error string is _not_ <NL>-terminated (stripped from FormatMessage)
 *
 * To achieve localized output, use languageid `MAKELANGID (LANG_NEUTRAL, SUBLANG_NEUTRAL)` and
 * FormatMessageW (with wchar_t), converted to UTF-8 by win_wstr_to_str.
*/
const char *
get_win32_error(DWORD error)
{
        _Thread_local static char buf[1024] = "(unknown)";
        if (FormatMessageA (FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,   // flags
                                NULL,                // lpsource
                                error,                   // message id
                                MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),    // languageid
                                buf, // output buffer
                                sizeof buf, // size of msgbuf, bytes
                                NULL)               // va_list of arguments
           ) {
                char *end = buf + strlen(buf) - 1;
                if (end > buf && *end == '\n') { // skip LF
                        *end-- = '\0';
                }
                if (end > buf && *end == '\r') { // skip CR
                        *end-- = '\0';
                }
                return buf;
        }

        snprintf(buf, sizeof buf, "[Could not find a description for error # %#lx.]", error);

        return buf;
}

/**
 * requires `SetConsoleOutputCP(CP_UTF8);` (done in common_preinit()) or passing the code page as terminal is set to
 * (command chcp)
 */
const char *win_wstr_to_str(const wchar_t *wstr) {
        _Thread_local static char res[1024];
        int ret = WideCharToMultiByte(CP_UTF8, 0, wstr, -1 /* NULL-terminated */, res, sizeof res - 1, NULL, NULL);
        if (ret == 0) {
                if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                        bug_msg(
                            LOG_LEVEL_ERROR,
                            "win_wstr_to_str: Insufficient buffer length %zd. ",
                            sizeof res);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "win_wstr_to_str: %s\n", get_win32_error(GetLastError()));
                }
        }
        return res;
}

static bool
GetProcessInfo(HANDLE hSnapshot, DWORD processId, PROCESSENTRY32 *pe32)
{
        if (!Process32First(hSnapshot, pe32)) {
                return false;
        }

        do {
                if (pe32->th32ProcessID == processId) {
                        return true;
                }
        } while (Process32Next(hSnapshot, pe32));

        return false;
}

/**
 * @returns true if there is a process "name" among current process ancestors
 * @param name the searched process name (case insensitive)
 */
bool
win_has_ancestor_process(const char *name)
{
        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (hSnapshot == INVALID_HANDLE_VALUE) {
                return false;
        }
        DWORD          pid  = GetCurrentProcessId();
        PROCESSENTRY32 pe32 = { .dwSize = sizeof(PROCESSENTRY32) };
        DWORD          visited_ps[32];
        unsigned       visited_ps_cnt = 0;
        bool           cycle          = false;
        while (!cycle && GetProcessInfo(hSnapshot, pid, &pe32)) {
                MSG(DEBUG, "%s: %s PID: %ld, PPID: %ld\n", __func__,
                    pe32.szExeFile, pid, pe32.th32ParentProcessID);
                if (_stricmp(pe32.szExeFile, name) == 0) {
                        CloseHandle(hSnapshot);
                        return true;
                }

                for (unsigned i = 0; i < visited_ps_cnt; i++) {
                        if (visited_ps[i] == pe32.th32ParentProcessID) {
                                cycle = true;
                        }
                }
                if (visited_ps_cnt ==
                    sizeof visited_ps / sizeof visited_ps[0]) {
                        MSG(WARNING, "Visited PS exhausted!\n");
                        break;
                }
                visited_ps[visited_ps_cnt++] = pid;

                pid = pe32.th32ParentProcessID;
        }
        CloseHandle(hSnapshot);
        return false;
}

unsigned long
get_windows_build()
{
        RTL_OSVERSIONINFOW osVersionInfo = { .dwOSVersionInfoSize =
                                                 sizeof(RTL_OSVERSIONINFOW) };

        HMODULE hNtDll = GetModuleHandleW(L"ntdll.dll");
        if (hNtDll == NULL) {
                MSG(VERBOSE, "Cannot load ntdll.dll!\n");
                return 0;
        }

        typedef NTSTATUS(WINAPI * RtlGetVersionFunc)(
            PRTL_OSVERSIONINFOW lpVersionInformation);
        RtlGetVersionFunc pRtlGetVersion =
            (RtlGetVersionFunc) GetProcAddress(hNtDll, "RtlGetVersion");
        if (pRtlGetVersion == NULL) {
                MSG(VERBOSE, "Cannot get RtlGetVersion from ntdll.dll!\n");
                return 0;
        }

        if (pRtlGetVersion(&osVersionInfo) != STATUS_SUCCESS) {
                MSG(VERBOSE, "Cannot get Windows version nfo!\n");
                return 0;
        }
        MSG(DEBUG, "Windows version: %lu.%lu (build %lu)\n",
            osVersionInfo.dwMajorVersion, osVersionInfo.dwMinorVersion,
            osVersionInfo.dwBuildNumber);
        return osVersionInfo.dwBuildNumber;
}

void
print_stacktrace_win()
{
        void          *stack[100];
        unsigned short frames;
        SYMBOL_INFO   *symbol;
        HANDLE         process;

        process = GetCurrentProcess();
        SymInitialize(process, NULL, TRUE);

        frames = CaptureStackBackTrace(0, 100, stack, NULL);
        symbol =
            (SYMBOL_INFO *) malloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char));
        symbol->MaxNameLen   = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        char backtrace_msg[] = "Backtrace:\n";
        _write(STDERR_FILENO, backtrace_msg, strlen(backtrace_msg));
        for (unsigned short i = 0; i < frames; i++) {
                BOOL ret =  SymFromAddr(process, (DWORD64) (stack[i]), 0, symbol);
                char buf[STR_LEN];
                snprintf_ch(buf, "%i (%p): ", frames - i - 1, stack[i]);
                if (ret == TRUE) {
                        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                                 "%s - 0x%0llX\n", symbol->Name,
                                 symbol->Address);

                        IMAGEHLP_LINE64 line = { .SizeOfStruct = sizeof line };
                        DWORD           displacementLine = 0;
                        if (SymGetLineFromAddr64(process, (DWORD64) (stack[i]),
                                                 &displacementLine, &line)) {
                                snprintf(buf + strlen(buf),
                                         sizeof buf - strlen(buf),
                                         "\tFile: %s, line: %lu, displacement: "
                                         "%lu\n",
                                         line.FileName, line.LineNumber,
                                         displacementLine);
                        }
                } else {
                        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                                 "(cannot resolve)\n");
                }
                _write(STDERR_FILENO, buf, strlen(buf));
        }

        free(symbol);
}

#endif // defined _WIN32

/* vim: set expandtab sw=8 tw=120: */

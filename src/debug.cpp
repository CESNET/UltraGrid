/*
 * FILE:    debug.cpp
 * PROGRAM: RAT
 * AUTHORS: Isidor Kouvelas
 *          Colin Perkins
 *          Mark Handley
 *          Orion Hodson
 *          Jerry Isdale
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 * Copyright (c) 1995-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <array>
#include <cstdint>

#include "compat/platform_time.h"
#include "debug.h"
#include "host.h"
#include "rang.hpp"

volatile int log_level = LOG_LEVEL_INFO;

static void _dprintf(const char *format, ...)
{
        if (log_level < LOG_LEVEL_DEBUG) {
                return;
        }

#ifdef WIN32
        char msg[65535];
        va_list ap;

        va_start(ap, format);
        _vsnprintf(msg, 65535, format, ap);
        va_end(ap);
        OutputDebugString(msg);
#else
        va_list ap;

        va_start(ap, format);
        vfprintf(stderr, format, ap);
        va_end(ap);
#endif                          /* WIN32 */
}

void log_msg(int level, const char *format, ...)
{
        va_list ap;

        if (log_level < level) {
                return;
        }

#if 0 // WIN32
        if (log_level == LOG_LEVEL_DEBUG) {
                char msg[65535];
                va_list ap;

                va_start(ap, format);
                _vsnprintf(msg, 65535, format, ap);
                va_end(ap);
                OutputDebugString(msg);
                return;
        }
#endif                          /* WIN32 */

        // get number of required bytes
        va_start(ap, format);
        int size = vsnprintf(NULL, 0, format, ap);
        va_end(ap);

        // format the string
        char *buffer = (char *) alloca(size + 1);
        va_start(ap, format);
        if (vsprintf(buffer, format, ap) != size) {
                va_end(ap);
                return;
        }
        va_end(ap);

        LOG(level) << buffer;
}

/**
 * This function is analogous to perror(). The message is printed using the logger.
 */
void log_perror(int level, const char *msg)
{
        std::array<char, 1024> strerror_buf;
        const char *errstring;
#ifdef _WIN32
        strerror_s(strerror_buf.data(), strerror_buf.size(), errno); // C11 Annex K (bounds-checking interfaces)
        errstring = strerror_buf.data();
#elif ! defined _POSIX_C_SOURCE || (_POSIX_C_SOURCE >= 200112L && !  _GNU_SOURCE)
        strerror_r(errno, strerror_buf.data(), strerror_buf.size()); // XSI version
        errstring = strerror_buf.data();
#else // GNU strerror_r version
        errstring = strerror_r(errno, strerror_buf.data(), strerror_buf.size());
#endif
        log_msg(level, "%s: %s\n", msg, errstring);
}

/**
 * debug_dump:
 * @lp: pointer to memory region.
 * @len: length of memory region in bytes.
 *
 * Writes a dump of a memory region to stdout.  The dump contains a
 * hexadecimal and an ascii representation of the memory region.
 *
 **/
void debug_dump(void *lp, int len)
{
        char *p;
        int i, j, start;
        char Buff[81];
        char stuffBuff[10];
        char tmpBuf[10];

        _dprintf("Dump of %d=%x bytes\n", len, len);
        start = 0L;
        while (start < len) {
                /* start line with pointer position key */
                p = (char *)lp + start;
                sprintf(Buff, "%p: ", p);

                /* display each character as hex value */
                for (i = start, j = 0; j < 16; p++, i++, j++) {
                        if (i < len) {
                                sprintf(tmpBuf, "%X", ((int)(*p) & 0xFF));

                                if (strlen((char *)tmpBuf) < 2) {
                                        stuffBuff[0] = '0';
                                        stuffBuff[1] = tmpBuf[0];
                                        stuffBuff[2] = ' ';
                                        stuffBuff[3] = '\0';
                                } else {
                                        stuffBuff[0] = tmpBuf[0];
                                        stuffBuff[1] = tmpBuf[1];
                                        stuffBuff[2] = ' ';
                                        stuffBuff[3] = '\0';
                                }
                                strcat(Buff, stuffBuff);
                        } else
                                strcat(Buff, " ");
                        if (j == 7)     /* space between groups of 8 */
                                strcat(Buff, " ");
                }

                strcat(Buff, "  ");

                /* display each character as character value */
                for (i = start, j = 0, p = (char *)lp + start;
                     (i < len && j < 16); p++, i++, j++) {
                        if (((*p) >= ' ') && ((*p) <= '~'))     /* test displayable */
                                sprintf(tmpBuf, "%c", *p);
                        else
                                sprintf(tmpBuf, "%c", '.');
                        strcat(Buff, tmpBuf);
                        if (j == 7)     /* space between groups of 8 */
                                strcat(Buff, " ");
                }
                _dprintf("%s\n", Buff);
                start = i;      /* next line starting byte */
        }
}

bool set_log_level(const char *optarg, bool *logger_repeat_msgs) {
        assert(optarg != nullptr);
        assert(logger_repeat_msgs != nullptr);

        using namespace std::string_literals;
        using std::clog;
        using std::cout;

        static const struct { const char *name; int level; } mapping[] = {
                { "quiet", LOG_LEVEL_QUIET },
                { "fatal", LOG_LEVEL_FATAL },
                { "error", LOG_LEVEL_ERROR },
                { "warning", LOG_LEVEL_WARNING},
                { "notice", LOG_LEVEL_NOTICE},
                { "info", LOG_LEVEL_INFO  },
                { "verbose", LOG_LEVEL_VERBOSE},
                { "debug", LOG_LEVEL_DEBUG },
                { "debug2", LOG_LEVEL_DEBUG2 },
        };

        if ("help"s == optarg) {
                cout << "log level: [0-" << LOG_LEVEL_MAX;
                for (auto m : mapping) {
                        cout << "|" << m.name;
                }
                cout << "][+repeat]\n";
                cout << "\trepeat - print repeating log messages\n";
                return false;
        }

        if (strstr(optarg, "+repeat") != nullptr) {
                *logger_repeat_msgs = true;
        }

        if (optarg[0] == '+') {
                return true;
        }

        if (isdigit(optarg[0])) {
                long val = strtol(optarg, nullptr, 0);
                if (val < 0 || val > LOG_LEVEL_MAX) {
                        clog << "Log: wrong value: " << log_level << "\n";
                        return false;
                }
                log_level = val;
                return true;
        }

        for (auto m : mapping) {
                if (strstr(optarg, m.name) == optarg) {
                        log_level = m.level;
                        return true;
                }
        }

        LOG(LOG_LEVEL_ERROR) << "Wrong log level specification: " << optarg << "\n";
        return false;
}

void Logger::preinit(bool skip_repeated)
{
        Logger::skip_repeated = skip_repeated;
        if (rang::rang_implementation::supportsColor()
                        && rang::rang_implementation::isTerminal(std::cout.rdbuf())
                        && rang::rang_implementation::isTerminal(std::cerr.rdbuf())) {
                // force ANSI sequences even when written to ostringstream
                rang::setControlMode(rang::control::Force);
#ifdef _WIN32
                // ANSI control sequences need to be explicitly set in Windows
                if ((rang::rang_implementation::setWinTermAnsiColors(std::cout.rdbuf()) || rang::rang_implementation::isMsysPty(_fileno(stdout))) &&
                                (rang::rang_implementation::setWinTermAnsiColors(std::cerr.rdbuf()) || rang::rang_implementation::isMsysPty(_fileno(stderr)))) {
                        rang::setWinTermMode(rang::winTerm::Ansi);
                }
#endif
        }
}

std::atomic<Logger::last_message *> Logger::last_msg{};
std::atomic<bool> Logger::skip_repeated{true};


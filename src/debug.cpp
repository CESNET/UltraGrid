/*
 * FILE:    debug.cpp
 * PROGRAM: RAT
 * AUTHORS: Isidor Kouvelas
 *          Colin Perkins
 *          Mark Handley
 *          Orion Hodson
 *          Jerry Isdale
 *
 * Copyright (c) 1995-2000 University College London
 * Copyright (c) 2005-2023 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // defined HAVE_CONFIG_H
#include "config_unix.h"
#include "config_win32.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "debug.h"
#include "host.h"
#include "utils/color_out.h"
#include "utils/string_view_utils.hpp"
#include "utils/misc.h" // ug_strerror

using std::atomic;
using std::string;
using std::unordered_map;

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

static void
log_vprintf(int level, const char *format, va_list ap)
{
        va_list aq;

        if (log_level < level) {
                return;
        }

        // get number of required bytes
        va_copy(aq, ap);
        int size = vsnprintf(NULL, 0, format, aq);
        va_end(aq);

        // format the string
        auto buf = get_log_output().get_buffer();
        const auto& style = get_log_output().get_level_style(level);

        buf.append(style);
        buf.append(size + 1, '\0');

        va_copy(aq, ap);
        vsnprintf(buf.data() + style.length(), size + 1, format, ap);
        va_end(aq);
        auto & str = buf.get();
        str.resize(str.size() - 1); // drop '\0' written by vsnprintf

        if (!get_log_output().is_interactive()) {
                prune_ansi_sequences_inplace(buf.get());
        } else if (style.size() > 0) {
                if (str.at(str.size() - 1) == '\n') { // put TERM_RESET before '\n'
                        str.erase(str.size() - 1);
                        buf.append(TERM_RESET "\n");
                } else {
                        buf.append(TERM_RESET);
                }
        }
        buf.submit();
}

void log_msg(int level, const char *format, ...) {
        va_list ap;
        va_start(ap, format);
        log_vprintf(level, format, ap);
        va_end(ap);
}

void log_msg_once(int level, uint32_t id, const char *msg, ...) {
        if (log_level < level) {
                return;
        }
        static volatile uint32_t oneshot_messages[10];
        static atomic<size_t> oneshot_messages_cnt;
        for (size_t i = 0; i < oneshot_messages_cnt; ++i) {
                if (oneshot_messages[i] == id) {
                        return;
                }
        }

        va_list aq;
        va_start(aq, msg);
        log_vprintf(level, msg, aq);
        va_end(aq);

        size_t last_cnt = oneshot_messages_cnt.fetch_add(1);
        if (last_cnt < sizeof oneshot_messages / sizeof oneshot_messages[0]) {
                oneshot_messages[last_cnt] = id;
        } else {
                log_msg(LOG_LEVEL_WARNING, "oneshot_messages full!\n");
        }
}

/**
 * This function is analogous to perror(). The message is printed using the logger.
 */
void log_perror(int level, const char *msg)
{
        log_msg(level, "%s: %s\n", msg, ug_strerror(errno));
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
void debug_dump(const void *lp, int len)
{
        const char *p;
        int i, j, start;
        char Buff[81];
        char tmpBuf[10];

        _dprintf("Dump of %d=%x bytes\n", len, len);
        start = 0L;
        while (start < len) {
                /* start line with pointer position key */
                p = (const char *)lp + start;
                snprintf(Buff, sizeof Buff, "%p: ", p);

                /* display each character as hex value */
                for (i = start, j = 0; j < 16; p++, i++, j++) {
                        if (i < len) {
                                snprintf(tmpBuf, sizeof tmpBuf, "%02X", ((int)(*p) & 0xFF));
                                strcat(Buff, tmpBuf);
                        } else
                                strcat(Buff, " ");
                        if (j % 2 == 1) /* space between groups of 2 */
                                strcat(Buff, " ");
                        if (j == 7)     /* space between groups of 8 */
                                strcat(Buff, " ");
                }

                strcat(Buff, "  ");

                /* display each character as character value */
                for (i = start, j = 0, p = (const char *)lp + start;
                     (i < len && j < 16); p++, i++, j++) {
                        if (((*p) >= ' ') && ((*p) <= '~'))     /* test displayable */
                                snprintf(tmpBuf, sizeof tmpBuf, "%c", *p);
                        else
                                snprintf(tmpBuf, sizeof tmpBuf, "%c", '.');
                        strcat(Buff, tmpBuf);
                        if (j == 7)     /* space between groups of 8 */
                                strcat(Buff, " ");
                }
                _dprintf("%s\n", Buff);
                start = i;      /* next line starting byte */
        }
}

bool parse_log_cfg(const char *conf_str,
                int *log_lvl,
                bool *logger_skip_repeats,
                log_timestamp_mode *show_timestamps)
{
        assert(conf_str != nullptr);
        assert(log_lvl != nullptr);
        assert(logger_skip_repeats != nullptr);
        assert(show_timestamps != nullptr);

        using std::clog;

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

        std::string_view cfg = conf_str;

        if (cfg == "help") {
                col() << "log level: [0-" << LOG_LEVEL_MAX;
                for (auto m : mapping) {
                        col() << "|" << m.name;
                }
                col() << "][+repeat][+/-timestamps]\n";
                col() << TBOLD("\trepeat") << " - print repeating log messages\n";
                col() << TBOLD("\ttimestamps") << " - enable/disable timestamps\n";
                col() << "environment variable " << SBOLD("ULTRAGRID_VERBOSE") << " - use verbose output (same as `-V` on command-line)\n";
                return false;
        }

        if (sv_contains(cfg, "+repeat")) {
                *logger_skip_repeats = false;
        }

        if(auto ts_pos = cfg.find("timestamps"); ts_pos != cfg.npos && ts_pos > 0){
                if(cfg[ts_pos - 1] == '+')
                        *show_timestamps = LOG_TIMESTAMP_ENABLED;
                else
                        *show_timestamps = LOG_TIMESTAMP_DISABLED;
        }

        if (cfg.empty() || cfg[0] == '+' || cfg[0] == '-') { // only flags, no log level
                return true;
        }

        if (isdigit(cfg[0])) {
                long val = -1;
                parse_num(cfg, val);
                if (val < 0 || val > LOG_LEVEL_MAX) {
                        clog << "Log: wrong value: " << cfg << " (allowed range [0.." << LOG_LEVEL_MAX << "])\n";
                        return false;
                }
                *log_lvl = val;
                return true;
        }

        auto log_lvl_sv = cfg;
        auto sign_pos = log_lvl_sv.find_first_of("+-");
        if(sign_pos != log_lvl_sv.npos)
                log_lvl_sv.remove_suffix(log_lvl_sv.size() - sign_pos);

        for (auto m : mapping) {
                if (log_lvl_sv == m.name) {
                        *log_lvl = m.level;
                        return true;
                }
        }

        LOG(LOG_LEVEL_ERROR) << "Wrong log level specification: " << cfg << "\n";
        return false;
}

#ifdef DEBUG
#define DUMP_FILE_USAGE \
        "* debug-dump=<module>[=<n>][,<module2>[=<n>]\n" \
        "  Dumps specified buffer to a file for debugging, n-th buffer may " \
        "be selected, name is <module>.dump.\n" \
        "  Avaiable modules: lavc-avframe, lavd-avframe\n"
ADD_TO_PARAM("debug-dump", DUMP_FILE_USAGE);
void debug_file_dump(const char *key, void (*serialize)(const void *data, FILE *), void *data) {
        const char *dump_file_val = get_commandline_param("debug-dump");
        static thread_local unordered_map<string, int> skip_map;
        if (dump_file_val == nullptr) {
                return;
        }

        if (strcmp(dump_file_val, "help") == 0) {
                color_printf("\n" TBOLD("debug-dump") " usage:\n");
                printf("%s\n", DUMP_FILE_USAGE);
                exit_uv(0);
                return;
        }

        // check if key is contained
        string not_first = string(",") + key;
        int skip_n = 0;
        if (strstr(dump_file_val, key) == dump_file_val) {
                const char *val = dump_file_val + strlen(key);
                if (val[0] == '=') {
                        skip_n = atoi(val + 1);
                }
        } else if (strstr(dump_file_val, not_first.c_str()) != NULL) {
                const char *val = strstr(dump_file_val, not_first.c_str()) + strlen(key);
                if (val[0] == '=') {
                        skip_n = atoi(val + 1);
                }
        } else {
                return;
        }

        if (skip_map.find(key) == skip_map.end()) {
                skip_map[key] = skip_n;
        }

        if (skip_map[key] == -1) { // already exported
                return;
        }

        if (skip_map[key] > 0) {
                skip_map[key]--;
                return;
        }

        // export
        string name = string(key) + ".dump";
        FILE *out = fopen(name.c_str(), "wb");
        if (out == nullptr) {
                perror("debug_file_dump fopen");
                return;
        }
        serialize(data, out);
        fclose(out);

        skip_map[key] = -1;
}
#endif

Log_output::Log_output(){
        last_msg.reserve(initial_buf_size);
        interactive = color_output_init();
}

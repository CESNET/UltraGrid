/**
 * @file   keyboard_control.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * With code taken from Olivier Mehani (set_tio()).
 */
/*
 * Copyright (c) 2015-2025 CESNET, zájmové sdružení právnických osob
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
/**
 * @file
 * @todo
 * Consider using some abstraction over terminal control mainly because
 * differences between Linux and Windows (ANSI mode or terminfo?)
 *
 * @todo
 * - key should not be overridden (eg. preset and custom)
 * - preset keys and customs ones should be unified (in key mapping)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"            // for HAVE_TERMIOS_H
#endif // HAVE_CONFIG_H

#include <cassert>             // for assert
#include <cctype>              // for isdigit, islower, isprint
#include <chrono>              // for milliseconds
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>             // for sigaction, sigemptyset, sa_handle
#include <cstdio>              // for printf, sscanf, EOF, fgets, perror
#include <cstdlib>             // for free, atexit, strtoll
#include <cstring>             // for strlen, strchr, strcpy, strstr, memset
#include <ctime>               // for strftime, time, time_t, tm
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>          // for FD_ISSET, FD_SET, select, FD_ZERO, fd_set
#else
#include <sys/select.h>        // for FD_ISSET, FD_SET, select, FD_ZERO, fd_set
#endif

#include "compat/time.h"       // for gmtime_s, localtime_s
#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "messaging.h"
#include "module.h"
#include "types.h"             // for video_desc
#include "utils/color_out.h"
#include "utils/misc.h"        // for ug_strerror
#include "utils/thread.h"
#include "video.h"

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#include <unistd.h>
#else
#include <conio.h>
#endif

static bool set_tio();

#define CONFIG_FILE "ug-key-map.txt"
#define MOD_NAME "[key control] "

using std::chrono::milliseconds; // _WIN32 only
using std::cout;
using std::dec;                // _WIN32 only
using std::endl;
using std::hex;
using std::istringstream;
using std::lock_guard;
using std::make_pair;
using std::mutex;
using std::setw;
using std::setfill;            // _WIN32 only
using std::string;
using std::stringstream;
using std::thread;
using std::unique_lock;

#ifdef HAVE_TERMIOS_H
static bool signal_catched = false;

static void catch_signal(int)
{
        signal_catched = true;
}
#endif

#ifdef HAVE_TERMIOS_H
static struct termios old_tio;
#endif

class keyboard_control::impl {
public:
        explicit impl(struct module *parent);
        ~impl();
        void start();
        void stop();

        void msg_received();
        static void msg_received_func(struct module *);

private:
        void read_command(bool multiple);
        void execute_command(const char *command);
        bool exec_local_command(const char *command);
        void exec_external_commands(const char *commands);

        void load_config_map();

        int64_t get_next_key();
        void info();
        void run();
        void signal();
        void usage();

        struct module m_mod{};

        std::thread m_keyboard_thread;
        struct module *m_root              {nullptr};
        struct module *m_parent;
#ifdef HAVE_TERMIOS_H
        int m_event_pipe[2] = {0, 0};
#else
        std::condition_variable m_cv;
#endif
        bool m_should_exit                 {false};
        std::queue<int64_t> m_pressed_keys;
        bool m_started                     {false};
        bool m_locked_against_changes      {true};
        std::time_t m_start_time;

        std::set<int64_t> guarded_keys    { '*', '/', '9', '0', 'c', 'C', 'm', 'M', '>', '<', 'e', 's', 'S', 'v', 'V', 'r' };
        std::map<int64_t, std::pair<std::string, std::string> > key_mapping; // user defined - key, command, name
        std::mutex m_lock;
};

keyboard_control::keyboard_control(struct module *parent) : m_impl{std::make_unique<impl>(parent)} {}
keyboard_control::~keyboard_control() = default;
void keyboard_control::start() {
        m_impl->start();
}
void keyboard_control::stop() {
        m_impl->stop();
}

void restore_old_tio(void)
{
#ifdef HAVE_TERMIOS_H
        struct sigaction sa, sa_old;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = SIG_IGN;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTTOU, &sa, &sa_old);
        /* set the new settings immediately */
        /* restore the former settings */
        tcsetattr(STDIN_FILENO,TCSANOW,&old_tio);
        sigaction(SIGTTOU, &sa_old, NULL);
#endif
}

keyboard_control::impl::impl(struct module *parent) :
        m_parent(parent)
{
        m_start_time = time(NULL);

        m_root = get_root_module(parent);

#ifdef HAVE_TERMIOS_H
        /* get the terminal settings for stdin */
        /* we want to keep the old setting to restore them a the end */
        tcgetattr(STDIN_FILENO,&old_tio);
#endif

        module_init_default(&m_mod);
        m_mod.cls = MODULE_CLASS_KEYCONTROL;
        m_mod.priv_data = this;
        m_mod.new_message = keyboard_control::impl::msg_received_func;
        module_register(&m_mod, m_parent);
}

keyboard_control::impl::~impl() {
        stop(); // in case that it was not called explicitly
}

ADD_TO_PARAM("disable-keyboard-control", "* disable-keyboard-control[=no]\n"
                "  disables keyboard control (usable mainly for non-interactive runs)\n");
void keyboard_control::impl::start()
{
        bool force_key_control = false;
        if (get_commandline_param("disable-keyboard-control")) {
                if (strcmp(get_commandline_param("disable-keyboard-control"),
                           "no") != 0) {
                        return;
                }
                force_key_control = true;
        }

        if (!force_key_control) {
                if (running_in_debugger()) {
                        LOG(LOG_LEVEL_WARNING)
                            << MOD_NAME "Running inside gdb - disabling "
                                        "interactive keyboard control\n";
                        return;
                }
#ifdef HAVE_TERMIOS_H
                if (!isatty(STDIN_FILENO)) {
                        log_msg(LOG_LEVEL_WARNING,
                                "[key control] Stdin is not a TTY - disabling "
                                "keyboard control.\n");
                        return;
                }
#endif // defined HAVE_TERMIOS_H
        }

#ifdef HAVE_TERMIOS_H
        if (pipe(m_event_pipe) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[key control] Cannot create control pipe!\n");
                return;
        }

        if (!set_tio()) {
                log_msg(LOG_LEVEL_WARNING, "[key control] Background task - disabling keyboard control.\n");
                return;
        }
        atexit(restore_old_tio);
#endif

        load_config_map();

        m_keyboard_thread = thread(&keyboard_control::impl::run, this);
        m_started = true;
}

void keyboard_control::impl::signal()
{
#ifdef HAVE_TERMIOS_H
        char c = 0;
        if (write(m_event_pipe[1], &c, 1) != 1) {
                perror(MOD_NAME "m_event_pipe write");
        }
#else
        m_cv.notify_one();
#endif
}

void keyboard_control::impl::stop()
{
        module_done(&m_mod);
        unique_lock<mutex> lk(m_lock);
        if (!m_started) {
                return;
        }
        m_should_exit = true;
        m_started = false;
        lk.unlock();
        signal();
        m_keyboard_thread.join();

#ifdef HAVE_TERMIOS_H
        close(m_event_pipe[0]);
        close(m_event_pipe[1]);
#endif
}

#ifdef HAVE_TERMIOS_H
static int
GETCH()
{
        unsigned char ch = 0;
        const ssize_t ret = read(0, &ch, sizeof ch);
        if (ret > 0) {
                return ch;
        }
        if (ret == 0) {
                MSG(WARNING, "read returned 0!\n");
        } else {
                MSG(WARNING, "read error: %s\n", ug_strerror(errno));
        }
        return -1;
}
#else
#define GETCH getch
#endif

#define CHECK_EOF(x) \
        do { \
                if ((x) < 0) { \
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME \
                            "Unexpected " << ((x) == -1 ? "EOF" : "error") \
                                               << " detected!\n"; \
                        return EOF; \
                } \
        } while (0)

/**
 * Tries to parse at least some small subset of ANSI control sequences not to
 * be ArrowUp interpreted as '\033', '[' and 'A' individually.
 *
 * @retval <0 on error
 *
 * @todo
 * * improve coverage and/or find some suitable implemnation (ideally one file,
 *   not ncurses)
 * * the function can be moved to separate util file
 */
static int64_t get_ansi_code() {
        int64_t c = GETCH();
        debug_msg(MOD_NAME "Pressed %" PRId64 "\n", c);
        if (c == '[') { // CSI
                c = '\033' << 8 | '[';
                while (true) {
                        if ((c >> 56u) != 0) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Long control sequence detected!\n";
                                return INT64_MIN;
                        }
                        c <<= 8;
                        int tmp = GETCH();
                        assert(tmp < 0x80); // ANSI esc seq should use only 7-bit encoding
                        debug_msg(MOD_NAME "Pressed %d\n", tmp);
                        CHECK_EOF(tmp);
                        c |= tmp;
                        if (tmp >= 0x40 && tmp <= 0xee) { // final byte
                                break;
                        }
                }
        } else if (c == 'N' // is this even used?
                        || c == 'O') { // eg. \EOP - F1-F4 (the rest of Fn is CSI)
                int tmp = GETCH();
                debug_msg(MOD_NAME "Pressed %d\n", tmp);
                CHECK_EOF(tmp);
                c = '\033' << 16 | c << 8 | tmp;
        } else if (c >= 'a' && c <= 'z') {
                return K_ALT(c);
        } else {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unknown control sequence!\n";
                return INT64_MIN;
        }

        return c;
}

#ifndef _WIN32
/*
 * Input must be 0x80-0xff
 */
static int count_utf8_bytes(unsigned int i) {
        int count = 0;
        while ((i & 0x80) != 0u) {
                count++;
                i <<= 1;
        }
        return count;
}

/// @retval <0 on error
static int64_t get_utf8_code(int c) {
        if (c < 0xc0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Wrong UTF sequence!\n";
                return INT64_MIN;
        }
        int ones = count_utf8_bytes(c);
        for (int i = 1; i < ones; ++i) {
                c = (c & 0x7fffff) << 8;
                int tmp = GETCH();
                debug_msg(MOD_NAME "Pressed %d\n", tmp);
                CHECK_EOF(tmp);
                c |= tmp;
        }
        if (ones > 7) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unsupported UTF sequence length!\n";
                return INT64_MIN;
        }
        return c;
}
#endif

#ifdef _WIN32
static int64_t convert_win_to_ansi_keycode(int c) {
        switch (c) {
        case 0xe048: return K_UP;
        case 0xe04b: return K_LEFT;
        case 0xe04d: return K_RIGHT;
        case 0xe050: return K_DOWN;
        default:
                   LOG(LOG_LEVEL_WARNING) << MOD_NAME "Cannot map Win keycode 0x"
                           << hex << setfill('0') << setw(2) << c << setw(0) << dec << " to ANSI.\n";
                   return -1;
        }
}
#endif

static bool is_utf8(int64_t ch) {
#ifdef _WIN32
        return false;
#endif

        if (ch <= 0) {
                return false;
        }

        unsigned char first_byte = 0U;
        while (ch != 0) {
                first_byte = ch & 0xff;
                ch >>= 8;
        }
        return first_byte & 0x80; /// @todo UTF-8 validity check would be nice
}

static string get_utf8_representation(int64_t ch) {
        char str[9] = "";
        while (ch != 0) {
                memmove(str + 1, str, 8);
                str[0] = ch & 0xff;
                ch >>= 8;
        }
        return str;
}

static string get_keycode_representation(int64_t ch) {
        switch (ch) {
        case K_UP: return "Up";
        case K_DOWN: return "Down";
        case K_LEFT: return "Left";
        case K_RIGHT: return "Right";
        case K_PGUP: return "Page-Up";
        case K_PGDOWN: return "Page-Dn";
        case K_CTRL_UP: return "Ctrl-Up";
        case K_CTRL_DOWN: return "Ctrl-Dn";
        }

        if (ch >= 1 && ch <= 'z' - 'a' + 1) {
                return string("Ctrl-") + string(1, 'a' + ch - 1);
        }

        if (ch >> 16 == 0 && ch >> 8 == '\033') {
                return string("Alt-") + string(1, ch & 0xff);
        }

        if (ch <= UCHAR_MAX && isprint(ch)) {
                return string("'") + string(1, ch) + "'";
        }

        if (is_utf8(ch)) {
                return string("'") + get_utf8_representation(ch) + "'";
        }

        stringstream oss;
        oss << "0x" << hex << ch;
        return oss.str();
}

/**
 * @returns next key either from keyboard or received via control socket.
 * @retval  0 if m_should_exit is set
 * @retval -1 on EOF
 * @retval -2 on error
 */
int64_t keyboard_control::impl::get_next_key()
{
        int64_t c;
        while (1) {
                {
                        unique_lock<mutex> lk(m_lock);
                        if (m_should_exit) {
                                return 0;
                        }

                        if (!m_pressed_keys.empty()) {
                                c = m_pressed_keys.front();
                                m_pressed_keys.pop();
                                return c;
                        }
                }
#ifdef HAVE_TERMIOS_H
                fd_set set;
                FD_ZERO(&set);
                FD_SET(0, &set);
                FD_SET(m_event_pipe[0], &set);
                select(m_event_pipe[0] + 1, &set, NULL, NULL, NULL);

                if (FD_ISSET(m_event_pipe[0], &set)) {
                        char c;
                        int bytes = read(m_event_pipe[0], &c, 1);
                        assert(bytes == 1);
                        continue; // skip to event processing
                }

                if (FD_ISSET(0, &set)) {
#else
                if (kbhit()) {
#endif
                        int64_t c = GETCH();
                        debug_msg(MOD_NAME "Pressed %" PRId64 "\n", c);
                        if (c == '\033') {
                                if ((c = get_ansi_code()) < 0) {
                                        continue;
                                }
#ifdef _WIN32
                        } else if (c == 0x0 || c == 0xe0) { // Win keycodes
                                int tmp = GETCH();
                                debug_msg(MOD_NAME "Pressed %d\n", tmp);
				CHECK_EOF(tmp);
                                if ((c = convert_win_to_ansi_keycode(c << 8 | tmp)) == -1) {
                                        continue;
                                }
#else
                        } else if (c > 0x80) {
                                if ((c = get_utf8_code(c)) < 0) {
                                        continue;
                                }
#endif
                        }
                        return c;
                }
#if !defined HAVE_TERMIOS_H
                // kbhit is non-blockinkg - we don't want to busy-wait
                unique_lock<mutex> lk(m_lock);
                m_cv.wait_for(lk, milliseconds(200));
                lk.unlock();
#endif
        }
}

void keyboard_control::impl::run()
{
        set_thread_name("keyboard-control");
        int saved_log_level = -1;
        int64_t c;

        while ((c = get_next_key())) {
                if (c == -1) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME
                            "EOF detected! Exiting keyboard control.\n";
                        break;
                }
                if (c == -2) {
                        perror(MOD_NAME "Error detected");
                        continue;
                }
                if (c == K_CTRL('X')) {
                        m_locked_against_changes = !m_locked_against_changes; // ctrl-x pressed
                        col() << TGREEN("Keyboard control: " << (m_locked_against_changes ? "" : "un") << "locked against changes\n");
                        continue;
                }
                if (m_locked_against_changes && guarded_keys.find(c) != guarded_keys.end()) {
                        col() << TGREEN("Keyboard control: locked against changes, press 'Ctrl-x' to unlock or 'h' for help.\n");
                        continue;
                }

                m_lock.lock();
                if (key_mapping.find(c) != key_mapping.end()) { // user defined mapping exists
                        string cmd = key_mapping.at(c).first;
                        exec_external_commands(cmd.c_str());
                        m_lock.unlock();
                        continue;
                }
                m_lock.unlock();

                switch (c) {
                case '*':
                case '/':
                case '9':
                case '0':
                case 'm':
                {
                        char path[] = "audio.receiver";
                        auto m = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                        switch (c) {
                        case '0':
                        case '*': m->type = RECEIVER_MSG_INCREASE_VOLUME; break;
                        case '9':
                        case '/': m->type = RECEIVER_MSG_DECREASE_VOLUME; break;
                        case 'm': m->type = RECEIVER_MSG_MUTE_TOGGLE; break;
                        }

                        auto resp = send_message(m_root, path, (struct message *) m);
                        free_response(resp);
                        break;
                }
                case 'M':
                {
                        char path[] = "audio.sender";
                        auto m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                        m->type = SENDER_MSG_MUTE_TOGGLE;

                        auto resp = send_message(m_root, path, (struct message *) m);
                        free_response(resp);
                        break;
                }
                case '>':
                case '<':
                {
                        int audio_delay = get_audio_delay();
                        audio_delay += c == '>' ? 10 : -10;
                        log_msg(LOG_LEVEL_INFO, "New audio delay: %d ms.\n", audio_delay);
                        set_audio_delay(audio_delay);
                        break;
                }
                case 'e':
                {
                        char path[] = "exporter";
                        auto m = (struct message *) new_message(sizeof(struct msg_universal));
                        strcpy(((struct msg_universal *) m)->text, "toggle");
                        auto resp = send_message(m_root, path, (struct message *) m);
                        free_response(resp);
                        break;
                }
                case 'c':
                case 'C':
                        read_command(c == 'C');
                        break;
                case 'v':
                case 'V':
                {
                        if (islower(c)) {
                                log_level = (log_level + 1) % (LOG_LEVEL_MAX + 1);
                        } else {
                                log_level = (log_level - 1 + (LOG_LEVEL_MAX + 1)) % (LOG_LEVEL_MAX + 1);
                        }
                        cout << "Log level: " << log_level << "\n";
                        break;
                }
                case 'r':
                {
                        bool skip = !get_log_output().get_skip_repeats();
                        get_log_output().set_skip_repeats(skip);
                        cout << "Skip repeated messages: " << std::boolalpha << skip << std::noboolalpha << "\n";
                        break;
                }
                case 's':
                        if (saved_log_level == -1) {
                                saved_log_level = log_level;
                                col() << TGREEN("Output suspended, press 'q' to continue.\n");
                                log_level = LOG_LEVEL_QUIET;
                        }
                        break;
                case 'S':
                        if (saved_log_level != -1) {
                                log_level = saved_log_level;
                                saved_log_level = -1;
                                col() << TGREEN( "Output resumed.\n");
                        }
                        break;
                case 'h':
                        usage();
                        break;
                case 'i':
                        cout << "\n";
                        info();
                        break;
                case '\n':
                case '\r':
                        cout << endl;
                        break;
                default:
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unsupported key " << get_keycode_representation(c) << " pressed. Press 'h' to help.\n";
                }
        }
}

static void
print_start_elapsed_time(time_t start_time)
{
        enum {
                ELAPS_LEN = 9,
                START_LEN = 25,
        };
        struct tm tm_buf = {};

        localtime_s(&start_time, &tm_buf);
        char start[START_LEN] = "";
        strftime(start, sizeof start, "%c", &tm_buf);

        const time_t diff = time(nullptr) - start_time;
        gmtime_s(&diff, &tm_buf);
        char elapsed[ELAPS_LEN] = "";
        strftime(elapsed, sizeof elapsed, "%T", &tm_buf);

        col() << TBOLD("Start time: ") << start << " (elapsed " << elapsed
              << ")\n";
}

void keyboard_control::impl::info()
{
        col() << TBOLD("UltraGrid version: ") << get_version_details() << "\n";
        print_start_elapsed_time(m_start_time);
        col() << TBOLD("Verbosity level: ") << log_level << (log_level == LOG_LEVEL_INFO ? " (default)" : "") << "\n";
        col() << TBOLD("Locked against changes: ") << (m_locked_against_changes ? "true" : "false") << "\n";
        col() << TBOLD("Audio playback delay: ") << get_audio_delay() << " ms\n";

        {
                int muted_sender = -1;
                int muted_receiver = -1;
                const char *path = "audio.receiver";
                auto *m_recv = reinterpret_cast<struct msg_receiver *>(new_message(sizeof(struct msg_receiver)));
                m_recv->type = RECEIVER_MSG_GET_AUDIO_STATUS;
                auto *resp = send_message_sync(m_root, path, reinterpret_cast<struct message *>(m_recv), 100, SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(resp) == 200) {
                        double vol = 0;
                        sscanf(response_get_text(resp), "%lf,%d", &vol, &muted_receiver);
                        double db = 20.0 * log10(vol);
                        color_printf(TBOLD("Playback volume: ") "%.2f%% (%s%.2f dB)\n", vol * 100.0, (db >= 0.0 ? "+" : ""), db);
                }
                free_response(resp);
                path = "audio.sender";
                auto *m_send = reinterpret_cast<struct msg_sender *>(new_message(sizeof(struct msg_sender)));
                m_send->type = SENDER_MSG_GET_STATUS;
                resp = send_message_sync(m_root, path, reinterpret_cast<struct message *>(m_send), 100, SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(resp) == 200) {
                        sscanf(response_get_text(resp), "%d", &muted_sender);
                }
                free_response(resp);
                auto muted_to_string = [](int val) {
                        switch (val) {
                                case 0: return "false";
                                case 1: return "true";
                                default: return "(unknown)";
                        }
                };
                col() << TBOLD("Audio muted - sender: ") << muted_to_string(muted_sender) << ", " << TBOLD("receiver: ") << muted_to_string(muted_receiver) << "\n";
        }

	{
                struct msg_sender *m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                m->type = SENDER_MSG_QUERY_VIDEO_MODE;
                struct response *r = send_message_sync(m_root, "sender", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        const char *text = response_get_text(r);
                        col() << TBOLD("Transmitted video format: ") << text << "\n";
                }
                free_response(r);
	}

	{
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "get_format");
                struct response *r = send_message_sync(m_root, "receiver.decoder", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        col() << TBOLD("Received video format: ")
                              << response_get_text(r) << "\n";
                }
                free_response(r);
	}

	{
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "get_port");
                struct response *r = send_message_sync(m_root, "control", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        col() << TBOLD("Control port: ") << response_get_text(r) << "\n";
                }
                free_response(r);
	}

	{
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "status");
                struct response *r = send_message_sync(m_root, "exporter", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        col() << TBOLD("Exporting: ") << response_get_text(r) << "\n";
                }
                free_response(r);
	}

        cout << "\n";
}

#define G(key) (guarded_keys.find(key) != guarded_keys.end() ? " [g]" : "")

void keyboard_control::impl::usage()
{
        col() << "\nAvailable keybindings:\n" <<
                TBOLD("\t  * 0  ") << "- increase receiver volume" << G('*') << "\n" <<
                TBOLD("\t  / 9  ") << "- decrease receiver volume" << G('/') << "\n" <<
                TBOLD("\t   >   ") << "- increase receiver audio delay by 10 ms" << G('>') << "\n" <<
                TBOLD("\t   <   ") << "- decrease receiver audio delay by 10 ms" << G('<') << "\n" <<
                TBOLD("\t   m   ") << "- mute/unmute receiver" << G('m') << "\n" <<
                TBOLD("\t   M   ") << "- mute/unmute sender" << G('M') << "\n" <<
                TBOLD("\t   v   ") << "- increase verbosity level" << G('v') << "\n" <<
                TBOLD("\t   V   ") << "- decrease verbosity level" << G('V') << "\n" <<
                TBOLD("\t   r   ") << "- skip repeated messages (toggle) " << G('r') << "\n" <<
                TBOLD("\t   e   ") << "- record captured content (toggle)" << G('e') << "\n" <<
                TBOLD("\t   h   ") << "- show help" << G('h') << "\n" <<
                TBOLD("\t   i   ") << "- show various information" << G('i') << "\n" <<
                TBOLD("\t  s S  ") << "- suspend/resume output" << G('s') << G('S') << "\n" <<
                TBOLD("\t  c C  ") << "- execute command through control socket (capital for multiple)" << G('c') << G('C') << "\n" <<
                TBOLD("\tCtrl-x ") << "- unlock/lock against changes" << G(K_CTRL('X')) << "\n" <<
                TBOLD("\tCtrl-c ") << "- exit " << G(K_CTRL('c')) << "\n" <<
                "\n";

        if (key_mapping.size() > 0) {
                cout << "Custom keybindings:\n";
                for (auto const &it : key_mapping) {
                        col() << TBOLD("\t" << setw(10) << get_keycode_representation(it.first) << setw(0) <<) << " - " << (it.second.second.empty() ? it.second.first : it.second.second) << G(it.first) << "\n";
                }
                cout << "\n";
        }
        col() << TBOLD("[g]") << " indicates that that option is guarded (Ctrl-x needs to be pressed first)\n";
}

void keyboard_control::impl::load_config_map() {
        FILE *in = fopen(CONFIG_FILE, "r");
        if (!in) {
                return;
        }
        char buf[1024];
        while (fgets(buf, sizeof buf - 1, in)) {
                if (strlen(buf) > 0 && buf[strlen(buf) - 1] == '\n') { // trim NL
                        buf[strlen(buf) - 1] = '\0';
                }
                if (strlen(buf) == 0) {
                        continue;
                }
                if (!exec_local_command(buf)) {
                        LOG(LOG_LEVEL_ERROR) << "Wrong config: " << buf << "\n";
                        continue;
                }
        }
        fclose(in);
}

bool keyboard_control::impl::exec_local_command(const char *command)
{
        if (strcmp(command, "localhelp") == 0) {
                printf("Local help:\n"
                                "\tmap <X> cmd1[;cmd2..][#name]\n"
                                "\t\tmaps key to command(s) for control socket (with an optional\n\t\tidentifier)\n"
                                "\tmap #<X> cmd1[;cmd2..][#name]\n"
                                "\t\tmaps key number (ASCII or multi-byte for non-ASCII) to\n\t\tcommand(s) for control socket (with an optional identifier)\n"
                                "\tunmap <X>\n"
                                "\tpress <num>\n"
                                "\t\tEmulate pressing key <num>\n\n"
                                "Mappings can be stored in " CONFIG_FILE "\n\n");
                return true;
        } else if (strstr(command, "map ") == command) {
                char *ccpy = strdup(command + strlen("map "));
                if (strlen(ccpy) > 3) {
                        int64_t key = ccpy[0];
                        char *command = ccpy + 2;
                        if (key == '#' && isdigit(ccpy[1])) {
                                key = strtoll(ccpy + 1, &command, 10);
                                command += 1; // skip ' '
                        }
                        const char *name = "";
                        if (strchr(command, '#')) {
                                name = strchr(command, '#') + 1;
                                *strchr(command, '#') = '\0';
                        }
                        if (key_mapping.find(key) == key_mapping.end()) {
                                key_mapping.insert({key, make_pair(command, name)});
                        } else {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Trying to register key shortcut " << get_keycode_representation(key) << ", which is already registered, ignoring.\n";
                        }
                }
                free(ccpy);
                return true;
        } else if (strstr(command, "unmap ") == command) {
                command += strlen("unmap ");
                if (strlen(command) >= 1) {
                        key_mapping.erase(command[0]);
                }
                return true;
        } else if (strstr(command, "press ") == command) {
                const char *key = command + strlen("press ");
                int64_t k;
                sscanf(key, "%" SCNi64, &k);
                m_pressed_keys.push(k);
                signal();
                return true;
        }
        return false;
}

void keyboard_control::impl::exec_external_commands(const char *commands)
{
        char *cpy = strdup(commands);
        char *item, *save_ptr, *tmp = cpy;
        while ((item = strtok_r(tmp, ";", &save_ptr))) {
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "execute ");
                strncat(m->text, item, sizeof m->text - strlen(m->text) - 1);
                struct response *r = send_message_sync(m_root, "control", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) != RESPONSE_OK) {
                        LOG(LOG_LEVEL_ERROR) << "Could not send a message!\n";
                }
                free_response(r);

                tmp = nullptr;
        }
        free(cpy);
}

void keyboard_control::impl::execute_command(const char *command) {
        if (!exec_local_command(command)) {
                exec_external_commands(command);
        }
}

void keyboard_control::impl::read_command(bool multiple)
{
        int saved_log_level = log_level;
        log_level = LOG_LEVEL_QUIET;
        restore_old_tio();
        if (multiple) {
                printf("Enter commands for control (try \"help\" or \"localhelp\").\n");
                printf("Exit the mode with blank line.\n");
        } else {
                printf("Enter a command for control (try \"help\" or \"localhelp\"):\n");
        }
        while (1) {
                log_level = LOG_LEVEL_QUIET;
                printf("control> ");
                char buf[sizeof msg_universal::text];
                if (!fgets(buf, sizeof buf - 1, stdin)) {
                        log_level = saved_log_level;
                        break;
                }
                log_level = saved_log_level;

                if (strlen(buf) == 0 || (strlen(buf) == 1 && buf[0] == '\n')) { // empty input
                        break;
                }
                if (buf[strlen(buf) - 1] == '\n') { // strip newline
                        buf[strlen(buf) - 1] = '\0';
                }

                m_lock.lock();
                execute_command(buf);
                m_lock.unlock();

                if (!multiple) {
                        break;
                }
        }

        set_tio();
}

/**
 * From http://shtrom.ssji.net/skb/getc.html
 */
static bool set_tio()
{
#ifdef HAVE_TERMIOS_H
        struct termios new_tio = old_tio;
        /* disable canonical mode (buffered i/o) and local echo */
        new_tio.c_lflag &=(~ICANON & ~ECHO);
        // Wrap calling of tcsetattr() by handling SIGTTOU. SIGTTOU can be raised if task is
        // run in background and trying to call tcsetattr(). If so, we disable keyboard
        // control.
        struct sigaction sa, sa_old;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = catch_signal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTTOU, &sa, &sa_old);
        /* set the new settings immediately */
        tcsetattr(STDIN_FILENO,TCSANOW,&new_tio);
        sigaction(SIGTTOU, &sa_old, NULL);

        if (!signal_catched) {
                return true;
        }
#endif
        return false;
}

void keyboard_control::impl::msg_received_func(struct module *m) {
        auto s = static_cast<keyboard_control::impl *>(m->priv_data);

        s->msg_received();
}

void keyboard_control::impl::msg_received() {
        struct message *msg;
        while ((msg = check_message(&m_mod))) {
                auto m = reinterpret_cast<struct msg_universal *>(msg);
                lock_guard<mutex> lk(m_lock);
                execute_command(m->text);
                struct response *r = new_response(RESPONSE_OK, nullptr);
                free_message((struct message *) m, r);
        }
}

void keycontrol_send_key(struct module *root, int64_t key) {
        struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
        snprintf(m->text, sizeof m->text, "press %" PRId64, key);
        struct response *r = send_message_sync(root, "keycontrol", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
        if (response_get_status(r) != RESPONSE_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot set key to keycontrol (error %d)!\n", response_get_status(r));
        }
        free_response(r);
}

/** @brief Registers callback message for given key
 *
 * @ref message_universal message type is always sent
 * @param[in] sender_mod  module that should receive the message for key
 * @param[in] message     text that will be passed back to receiver_mod if key was pressed
 * @param[in] description optional description that will be displayed in keyboard control help (may be NULL)
 */
bool keycontrol_register_key(struct module *receiver_mod, int64_t key, const char *message, const char *description) {
        assert(strchr(message, '#') == nullptr && (description == nullptr || strchr(description, '#') == nullptr));
        char receiver_path[1024];
        if (!module_get_path_str(receiver_mod, receiver_path, sizeof receiver_path)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot format format path for sender!\n");
                return false;
        }
        struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
        if (description == nullptr) {
                description = message;
        }
        snprintf(m->text, sizeof m->text, "map #%" PRId64 " %s %s#%s", key, receiver_path, message, description);
        struct response *r = send_message_sync(get_root_module(receiver_mod), "keycontrol", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
        if (response_get_status(r) != RESPONSE_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot register keyboard control (error %d)!\n", response_get_status(r));
                free_response(r);
                return false;
        }
        free_response(r);
        return true;
}

void get_keycode_name(int64_t ch, char *buf, size_t buflen) {
        if (buflen == 0) {
                return;
        }
        string name = get_keycode_representation(ch);
        strncpy(buf, name.c_str(), buflen - 1);
        buf[buflen - 1] = '\0';
}


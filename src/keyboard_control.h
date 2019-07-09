/**
 * @file   keyboard_control.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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

#ifndef keyboard_control_h_
#define keyboard_control_h_

#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#endif

#include "module.h"

// values are byte represenations of escape sequences (without ESC itself)
#define K_UP    0x1b5b41
#define K_DOWN  0x1b5b42
#define K_RIGHT 0x1b5b43
#define K_LEFT  0x1b5b44

class keyboard_control {
public:
        keyboard_control(struct module *parent);
        ~keyboard_control();
        void start();
        void stop();

        static void msg_received_func(struct module *);
        void msg_received();

private:
        void read_command(bool multiple);
        void execute_command(const char *command);
        bool exec_local_command(const char *commnad);
        void exec_external_commands(const char *commnads);

        void load_config_map();

        void info();
        void run();
        void usage();

        struct module m_mod;

        std::thread m_keyboard_thread;
        struct module *m_root;
#ifdef HAVE_TERMIOS_H
        int m_should_exit_pipe[2];
#else
        bool m_should_exit;
#endif
        bool m_started;
        bool m_locked_against_changes;
        std::time_t m_start_time;

        /// @todo add a flag to determine which keys should be guarded against
        ///       unintentional pressing
        std::map<int, std::pair<std::string, std::string> > key_mapping; // user defined - key, command, name
        std::mutex m_lock;
};

#endif // keyboard_control_h_


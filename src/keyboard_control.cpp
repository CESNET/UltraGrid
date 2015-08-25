/**
 * @file   keyboard_control.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * With code taken from Olivier Mehani from http://shtrom.ssji.net/skb/getc.html
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "messaging.h"

#include <iostream>

#ifdef HAVE_TERMIOS_H
#include <unistd.h>
#include <termios.h>
#else
#include <conio.h>
#endif

using namespace std;

void keyboard_control::start(struct module *root)
{
        m_root = root;
#ifdef HAVE_TERMIOS_H
        assert(pipe(m_should_exit_pipe) == 0);
#endif
        m_keyboard_thread = thread(&keyboard_control::run, this);
        m_started = true;
}

void keyboard_control::stop()
{
        if (!m_started) {
                return;
        }
#ifdef HAVE_TERMIOS_H
        char c;
        assert(write(m_should_exit_pipe[1], &c, 1) == 1);
        close(m_should_exit_pipe[1]);
#else
        m_should_exit = true;
#endif
        m_keyboard_thread.join();
        m_started = false;
}

void keyboard_control::run()
{
#ifdef HAVE_TERMIOS_H
        struct termios old_tio, new_tio;

        /* get the terminal settings for stdin */
        tcgetattr(STDIN_FILENO,&old_tio);

        /* we want to keep the old setting to restore them a the end */
        new_tio=old_tio;

        /* disable canonical mode (buffered i/o) and local echo */
        new_tio.c_lflag &=(~ICANON & ~ECHO);

        /* set the new settings immediately */
        tcsetattr(STDIN_FILENO,TCSANOW,&new_tio);
#endif

        while(1) {
#ifdef HAVE_TERMIOS_H
                fd_set set;
                FD_ZERO(&set);
                FD_SET(0, &set);
                FD_SET(m_should_exit_pipe[0], &set);
                select(m_should_exit_pipe[0] + 1, &set, NULL, NULL, NULL);
                if (FD_ISSET(0, &set)) {
                        char c = getchar();
#else
                usleep(200000);
                while (kbhit()) {
                        char c = getch();
#endif
                        debug_msg("Key %c pressed\n", c);
                        switch (c) {
                        case '*':
                        case '/':
                        case '9':
                        case '0':
                        case 'm': {
                                char path[] = "audio.receiver";
                                auto m = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                                switch (c) {
                                case '0':
                                case '*': m->type = RECEIVER_MSG_INCREASE_VOLUME; break;
                                case '9':
                                case '/': m->type = RECEIVER_MSG_DECREASE_VOLUME; break;
                                case 'm': m->type = RECEIVER_MSG_MUTE; break;
                                }

                                auto resp = send_message(m_root, path, (struct message *) m);
                                free_response(resp);

                                break;
                                }
                        case 'h':
                                usage();
                                break;
                        case 'v':
                                log_level = (log_level + 1) % (LOG_LEVEL_MAX + 1);
                                cout << "Log level: " << log_level << "\n";
                                break;
                        }
                }
#ifdef HAVE_TERMIOS_H
                if (FD_ISSET(m_should_exit_pipe[0], &set)) {
#else
                if (m_should_exit) {
#endif
                        break;
                }
        }

#ifdef HAVE_TERMIOS_H
        /* restore the former settings */
        tcsetattr(STDIN_FILENO,TCSANOW,&old_tio);

        close(m_should_exit_pipe[0]);
#endif
}


void keyboard_control::usage()
{
        cout << "\nAvailable keybindings:\n" <<
                "\t  * 0  - increase volume\n" <<
                "\t  / 9  - decrease volume\n" <<
                "\t   m   - mute/unmute\n" <<
                "\t   v   - increase verbosity level\n" <<
                "\t   h   - show help\n" <<
                "\tCtrl-c - exit\n" <<
                "\n";
        cout << "Verbosity level: " << log_level << (log_level == LOG_LEVEL_INFO ? " (default)" : "") << "\n\n";
}


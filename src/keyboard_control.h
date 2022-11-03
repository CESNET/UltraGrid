/**
 * @file   keyboard_control.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2019 CESNET, z. s. p. o.
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

#include "host.h"
#include "module.h"

#ifdef __cplusplus

#include <cinttypes>
#include <memory>

class keyboard_control {
public:
        keyboard_control(struct module *parent);
        ~keyboard_control();
        keyboard_control(keyboard_control&&) = delete;
        keyboard_control& operator=(keyboard_control&&) = delete;

        void start();
        void stop();

private:
        class impl;
        std::unique_ptr<impl> m_impl;
};

#endif // __cplusplus

// values are byte represenations of escape sequences (without ESC itself)
#define K_UP     0x1b5b41
#define K_DOWN   0x1b5b42
#define K_RIGHT  0x1b5b43
#define K_LEFT   0x1b5b44
#define K_PGUP   0x1b5b357e
#define K_PGDOWN 0x1b5b367e
#define K_CTRL(x) (1 + ((x) >= 'A' && (x) <= 'Z' ? (x) - 'A' + 'a' : (x)) - 'a')
#define K_CTRL_UP 0x1b5b313b3541LL
#define K_CTRL_DOWN 0x1b5b313b3542LL
#define K_ALT(x) ('\033' << 8 | (x)) /// exception - include ESC, otherwise Alt-a would become 'a'

#define MAX_KEYCODE_NAME_LEN 8

// attribute used guaranties that the symbol is present even if not referenced (modular build)
/// @todo all UG core functions should have the 'used' attribute
EXTERN_C void keycontrol_send_key(struct module *root, int64_t key) ATTRIBUTE(used);
EXTERN_C bool keycontrol_register_key(struct module *sender_mod, int64_t key, const char *message, const char *description) ATTRIBUTE(used);
EXTERN_C void get_keycode_name(int64_t ch, char *buf, size_t buflen) ATTRIBUTE(used);
EXTERN_C void restore_old_tio(void);

#endif // keyboard_control_h_


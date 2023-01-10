/**
 * @file   spout_sender.cpp
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017-2021 CESNET, z. s. p. o.
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
#include <SpoutLibrary.h>

#include "debug.h"
#include "spout_sender.h"

void *spout_sender_register(const char *name) {

	SPOUTHANDLE spout = GetSpout();
	spout->SetSenderName(name);
        spout_set_log_level(spout, log_level);

	return spout;
}

void spout_sender_sendframe(void *s, int width, int height, unsigned int id) {
	((SPOUTHANDLE)s)->SendTexture(id, GL_TEXTURE_2D, width, height, false); // default is flip
	}

void spout_sender_unregister(void *s) {
        auto *spout = static_cast<SPOUTHANDLE>(s);
        spout->ReleaseSender();
        spout->Release();
}

void spout_set_log_level(void *s, int ug_level) {
        auto *spout = static_cast<SPOUTHANDLE>(s);
        enum SpoutLibLogLevel l;
        switch (ug_level) {
                case LOG_LEVEL_QUIET: l = SPOUT_LOG_SILENT; break;
                case LOG_LEVEL_FATAL:
                case LOG_LEVEL_ERROR:
                case LOG_LEVEL_WARNING: l = (enum SpoutLibLogLevel) (7 - ug_level); break;
                case LOG_LEVEL_NOTICE:
                case LOG_LEVEL_INFO: l = SPOUT_LOG_NOTICE; break;
                default: l = SPOUT_LOG_VERBOSE;
        }

        spout->SetSpoutLogLevel(l);
}

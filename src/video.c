/**
 * @file   video.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This file defines some common video functions.
 *
 * These function are neither video frame nor video codec related.
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#include "video.h"

/**
 * This function matches string representation of video mode with its
 * respective enumeration value.
 *
 * @param reuqested_mode textual representation of video mode
 * @return valid member of enum video mode
 */
enum video_mode get_video_mode_from_str(const char *requested_mode) {
        if(strcasecmp(requested_mode, "help") == 0) {
                printf("Video mode options\n\n");
                printf("-M {tiled-4K | 3D | dual-link }\n");
                return VIDEO_UNKNOWN;
        } else if(strcasecmp(requested_mode, "tiled-4K") == 0) {
                return VIDEO_4K;
        } else if(strcasecmp(requested_mode, "3D") == 0) {
                return VIDEO_STEREO;
        } else if(strcasecmp(requested_mode, "dual-link") == 0) {
                return VIDEO_DUAL;
        } else {
                fprintf(stderr, "Unknown video mode (see -M help)\n");
                return VIDEO_UNKNOWN;
        }
}

int get_video_mode_tiles_x(enum video_mode video_type)
{
        int ret = 0;
        switch(video_type) {
                case VIDEO_NORMAL:
                case VIDEO_DUAL:
                        ret = 1;
                        break;
                case VIDEO_4K:
                case VIDEO_STEREO:
                        ret = 2;
                        break;
                case VIDEO_UNKNOWN:
                        abort();
        }
        return ret;
}

int get_video_mode_tiles_y(enum video_mode video_type)
{
        int ret = 0;
        switch(video_type) {
                case VIDEO_NORMAL:
                case VIDEO_STEREO:
                        ret = 1;
                        break;
                case VIDEO_4K:
                case VIDEO_DUAL:
                        ret = 2;
                        break;
                case VIDEO_UNKNOWN:
                        abort();
        }
        return ret;
}

const char *get_video_mode_description(enum video_mode video_mode)
{
        switch (video_mode) {
                case VIDEO_NORMAL:
                        return "normal";
                case VIDEO_STEREO:
                        return "3D";
                case VIDEO_4K:
                        return "tiled 4K";
                case VIDEO_DUAL:
                        return "dual-link";
                case VIDEO_UNKNOWN:
                        abort();
        }
        return NULL;
}


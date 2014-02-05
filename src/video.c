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

typedef struct {
        const char *name;
        int x;
        int y;
} video_mode_info_t;

const video_mode_info_t video_mode_info[]  = {
        [VIDEO_UNKNOWN] = { "(unknown)", 0, 0 },
        [VIDEO_NORMAL] = { "normal", 1, 1 },
        [VIDEO_DUAL] = { "dual-link", 1, 2 },
        [VIDEO_STEREO] = { "3D", 2, 1 },
        [VIDEO_4K] = { "tiled-4k", 2, 2 },
        [VIDEO_3X1] = { "3x1", 3, 1 },
};
const unsigned int video_mode_info_count = sizeof(video_mode_info) / sizeof(video_mode_info_t);

/**
 * This function matches string representation of video mode with its
 * respective enumeration value.
 *
 * @param reuqested_mode textual representation of video mode
 * @return valid member of enum video mode
 */
enum video_mode get_video_mode_from_str(const char *requested_mode) {
        if(strcasecmp(requested_mode, "help") == 0) {
                printf("Video mode options:\n\t-M {");
                for (unsigned int i = 1 /* omit unknown */; i < video_mode_info_count;
                                ++i) {
                        printf(" %s ", video_mode_info[i].name);
                        if (i < video_mode_info_count - 1) {
                                printf("| ");
                        }
                }
                printf("}\n");
                return VIDEO_UNKNOWN;
        } else {
                for (unsigned int i = 0; i < video_mode_info_count;
                                ++i) {
                        if (strcasecmp(requested_mode, video_mode_info[i].name) == 0) {
                                return i;
                        }
                }
                fprintf(stderr, "Unknown video mode (see -M help)\n");
                return VIDEO_UNKNOWN;
        }
}

int get_video_mode_tiles_x(enum video_mode video_mode)
{
        return video_mode_info[video_mode].x;
}

int get_video_mode_tiles_y(enum video_mode video_mode)
{
        return video_mode_info[video_mode].y;
}

const char *get_video_mode_description(enum video_mode video_mode)
{
        return video_mode_info[video_mode].name;
}

enum video_mode guess_video_mode(int num_substreams)
{
        assert(num_substreams > 0);

        switch (num_substreams) {
                case 1:
                        return VIDEO_NORMAL;
                case 2:
                        return VIDEO_STEREO;
                case 3:
                        return VIDEO_3X1;
                case 4:
                        return VIDEO_4K;
                default:
                        return VIDEO_UNKNOWN;
        }
}


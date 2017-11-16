/**
 * @file video_capture/aja_win32_utils.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This source is used only with AJA DLL build to supply dependencies that
 * AJA module depends upon, however the actual sources cannot currently be
 * built with MSVC (further dependencies, parts that cannot be compiled with
 * the compiler or depend upon them).
 */
/*
 * Copyright (c) 2017-2018 CESNET z.s.p.o.
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

#define WIN32
#include "config_win32.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "video.h"
#include "audio/audio.h"
#include "audio/utils.h"

void remux_channel(char *out, char *in, int bps, int in_len, int in_stream_channels, int out_stream_channels, int pos_in_stream, int pos_out_stream)
{
        int samples = in_len / (in_stream_channels * bps);
        int i;

        assert (bps <= 4);

        in += pos_in_stream * bps;
        out += pos_out_stream * bps;

        for (i = 0; i < samples; ++i) {
                memcpy(out, in, bps);

                out += bps * out_stream_channels;
                in += bps * in_stream_channels;

        }
}

int vc_get_linesize(unsigned int width, codec_t codec)
{
	int bpp;
	switch (codec) {
        case v210:
                width = (width + 47) / 48 * 48;
                return width * 8 / 3;
        case UYVY:
        case YUYV:
                bpp = 2;
                break;
        case RGB:
        case BGR:
                bpp = 3;
                break;
        case R10k:
        case RGBA:
                bpp = 4;
                break;
        default:
                fprintf(stderr, "[AJA] Unimplemented codec. Please report a bug.");
                abort();
	}
        return width * bpp;
}

struct video_frame * vf_alloc(int count)
{
        struct video_frame *buf;
        assert(count > 0);

        buf = (struct video_frame *) calloc(1, sizeof(struct video_frame));
        assert(buf != NULL);

        buf->tiles = (struct tile *)
                        calloc(1, sizeof(struct tile) * count);
        assert(buf->tiles != NULL);
        buf->tile_count = count;

        return buf;
}

struct video_frame * vf_alloc_desc(struct video_desc desc)
{
        struct video_frame *buf;
        assert(desc.tile_count > 0);

        buf = vf_alloc(desc.tile_count);
        if(!buf) return NULL;

        buf->color_spec = desc.color_spec;
        buf->interlacing = desc.interlacing;
        buf->fps = desc.fps;
        // tile_count already filled
        for(unsigned int i = 0u; i < desc.tile_count; ++i) {
                memset(&buf->tiles[i], 0, sizeof(buf->tiles[i]));
                buf->tiles[i].width = desc.width;
                buf->tiles[i].height = desc.height;
                buf->tiles[i].data_len = vc_get_linesize(desc.width, desc.color_spec) * desc.height;
        }

        return buf;
}

void vf_free(struct video_frame *buf)
{
        if(!buf)
                return;
        if (buf->data_deleter) {
                buf->data_deleter(buf);
        }
        free(buf->tiles);
        free(buf);
}


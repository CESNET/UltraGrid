/*
 * FILE:    video_codec.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "debug.h"

#include <stdio.h>
#include <string.h>
#include "video.h"
#include "video_codec.h"

struct video_frame * vf_alloc(int count)
{
        struct video_frame *buf;
        assert(count > 0);
        
        buf = (struct video_frame *) calloc(1, sizeof(struct video_frame));
        
        buf->tiles = (struct tile *) 
                        calloc(1, sizeof(struct tile) * count);
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
        }

        return buf;
}

struct video_frame * vf_alloc_desc_data(struct video_desc desc)
{
        struct video_frame *buf;

        buf = vf_alloc_desc(desc);

        if(buf) {
                for(unsigned int i = 0; i < desc.tile_count; ++i) {
                        buf->tiles[i].linesize = vc_get_linesize(desc.width,
                                        desc.color_spec);
                        buf->tiles[i].data_len = buf->tiles[i].linesize *
                                desc.height;
                        buf->tiles[i].data = (char *) malloc(buf->tiles[i].data_len);
                }
        }

        return buf;
}

void vf_free(struct video_frame *buf)
{
        if(!buf)
                return;
        free(buf->tiles);
        free(buf);
}

void vf_free_data(struct video_frame *buf)
{
        if(!buf)
                return;

        for(unsigned int i = 0u; i < buf->tile_count; ++i) {
                free(buf->tiles[i].data);
        }
        vf_free(buf);
}

struct tile *tile_alloc() {
        return calloc(1, sizeof(struct tile));
}

struct tile *tile_alloc_desc(struct video_desc desc) {
        struct tile *res = tile_alloc();
        res->width = desc.width;
        res->height = desc.height;
        return res;
}

void tile_free(struct tile *tile) {
        free(tile);
}

void tile_free_data(struct tile *tile) {
        if(tile) {
                free(tile->data);
                tile_free(tile);
        }
}

void vf_write_desc(struct video_frame *buf, struct video_desc desc)
{
        assert(desc.tile_count == buf->tile_count);

        buf->color_spec = desc.color_spec;
        buf->fps = desc.fps;
        buf->interlacing = desc.interlacing;
        for(unsigned int i = 0; i < buf->tile_count; ++i) {
                buf->tiles[0].width = desc.width;
                buf->tiles[0].height = desc.height;
        }
}

struct tile * vf_get_tile(struct video_frame *buf, int pos)
{
        assert ((unsigned int) pos < buf->tile_count);

        return &buf->tiles[pos];
}

int video_desc_eq(struct video_desc a, struct video_desc b)
{
        return video_desc_eq_excl_param(a, b, PARAM_TILE_COUNT); // TILE_COUNT is excluded because it
                                                                 // was omitted so not to break 
                                                                 // compatibility...
}

int video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params)
{
        return ((excluded_params & PARAM_WIDTH) || a.width == b.width) &&
                ((excluded_params & PARAM_HEIGHT) || a.height == b.height) &&
                ((excluded_params & PARAM_CODEC) || a.color_spec == b.color_spec) &&
                ((excluded_params & PARAM_INTERLACING) || a.interlacing == b.interlacing) &&
                ((excluded_params & PARAM_TILE_COUNT) || a.tile_count == b.tile_count) &&
                ((excluded_params & PARAM_FPS) || fabs(a.fps - b.fps) < 0.01);// &&
               // TODO: remove these obsolete constants
               //(a.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit)) == (b.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit));
}

struct video_desc video_desc_from_frame(struct video_frame *frame)
{
        struct video_desc desc;

        assert(frame != NULL);

        desc.width = frame->tiles[0].width;
        desc.height = frame->tiles[0].height;
        desc.color_spec = frame->color_spec;
        desc.fps = frame->fps;
        desc.interlacing = frame->interlacing;
        desc.tile_count = frame->tile_count;

        return desc;
}

int get_video_mode_tiles_x(int video_type)
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
        }
        return ret;
}

int get_video_mode_tiles_y(int video_type)
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
        }
        return ret;
}

const char *get_interlacing_description(enum interlacing_t interlacing)
{
        switch (interlacing) {
                case PROGRESSIVE:
                        return "progressive";
                case UPPER_FIELD_FIRST:
                        return "interlaced (upper field first)";
                case LOWER_FIELD_FIRST:
                        return "interlaced (lower field first)";
                case INTERLACED_MERGED:
                        return "interlaced merged";
                case SEGMENTED_FRAME:
                        return "progressive segmented";
        }

        return NULL;
}

const char *get_interlacing_suffix(enum interlacing_t interlacing)
{
        switch (interlacing) {
                case PROGRESSIVE:
                        return "p";
                case UPPER_FIELD_FIRST:
                        return "tff";
                case LOWER_FIELD_FIRST:
                        return "bff";
                case INTERLACED_MERGED:
                        return "i";
                case SEGMENTED_FRAME:
                        return "psf";
        }

        return NULL;
}

const char *get_video_mode_description(int video_mode)
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
        }
        return NULL;
}

/* TODO: rewrite following 2 functions in more efficient way */
void il_upper_to_merged(char *dst, char *src, int linesize, int height)
{
        int y;
        char *tmp = malloc(linesize * height);
        char *line1, *line2;

        line1 = tmp;
        line2 = src;
        for(y = 0; y < (height + 1) / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }

        line1 = tmp + linesize;
        line2 = src + linesize * ((height + 1) / 2);
        for(y = 0; y < height / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

void il_merged_to_upper(char *dst, char *src, int linesize, int height)
{
        int y;
        char *tmp = malloc(linesize * height);
        char *line1, *line2;

        line1 = tmp;
        line2 = src;
        for(y = 0; y < (height + 1) / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize;
                line2 += linesize * 2;
        }

        line1 = tmp + linesize * ((height + 1) / 2);
        line2 = src + linesize;
        for(y = 0; y < height / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize;
                line2 += linesize * 2;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

double compute_fps(int fps, int fpsd, int fd, int fi)
{
        double res; 

        res = fps;
        if(fd)
                res /= 1.001;
        res /= fpsd;

        if(fi) {
                res = 1.0 / res;
        }

        return res;
}

struct video_frame *vf_get_copy(struct video_frame *original) {
        struct video_frame *frame_copy;

        frame_copy = (struct video_frame *) malloc(sizeof(struct video_frame));
        memcpy(frame_copy, original, sizeof(struct video_frame));

        frame_copy->tiles = (struct tile *) malloc(sizeof(struct tile) * frame_copy->tile_count);
        memcpy(frame_copy->tiles, original->tiles, sizeof(struct tile) * frame_copy->tile_count);

        for(int i = 0; i < (int) frame_copy->tile_count; ++i) {
                frame_copy->tiles[i].data = (char *) malloc(frame_copy->tiles[i].data_len);
                memcpy(frame_copy->tiles[i].data, original->tiles[i].data,
                                frame_copy->tiles[i].data_len);
        }

        return frame_copy;
}


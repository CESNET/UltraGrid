/**
 * @file   video_frame.c
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @brief This file contains video frame manipulation functions.
 */
/*
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
#include "video_codec.h"
#include "video_frame.h"

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

struct video_frame * vf_alloc_desc_data(struct video_desc desc)
{
        struct video_frame *buf;

        buf = vf_alloc_desc(desc);

        if(buf) {
                for(unsigned int i = 0; i < desc.tile_count; ++i) {
                        buf->tiles[i].data_len = vc_get_linesize(desc.width,
                                        desc.color_spec) *
                                desc.height;
                        buf->tiles[i].data = (char *) malloc(buf->tiles[i].data_len);
                        assert(buf->tiles[i].data != NULL);
                }
        }

        buf->data_deleter = vf_data_deleter;

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

void vf_data_deleter(struct video_frame *buf)
{
        if(!buf)
                return;

        for(unsigned int i = 0u; i < buf->tile_count; ++i) {
                free(buf->tiles[i].data);
        }
}

struct tile * vf_get_tile(struct video_frame *buf, int pos)
{
        assert ((unsigned int) pos < buf->tile_count);

        return &buf->tiles[pos];
}

int video_desc_eq(struct video_desc a, struct video_desc b)
{
        return video_desc_eq_excl_param(a, b, 0);
}

int video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params)
{
        return ((excluded_params & PARAM_WIDTH) || a.width == b.width) &&
                ((excluded_params & PARAM_HEIGHT) || a.height == b.height) &&
                ((excluded_params & PARAM_CODEC) || a.color_spec == b.color_spec) &&
                ((excluded_params & PARAM_INTERLACING) || a.interlacing == b.interlacing) &&
                ((excluded_params & PARAM_TILE_COUNT) || a.tile_count == b.tile_count) &&
                ((excluded_params & PARAM_FPS) || fabs(a.fps - b.fps) < 0.01);// &&
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

static const char *interlacing_suffixes[] = {
                [PROGRESSIVE] = "p",
                [UPPER_FIELD_FIRST] = "tff",
                [LOWER_FIELD_FIRST] = "bff",
                [INTERLACED_MERGED] = "i",
                [SEGMENTED_FRAME] = "psf",
};

const char *get_interlacing_suffix(enum interlacing_t interlacing)
{
        if (interlacing < sizeof interlacing_suffixes / sizeof interlacing_suffixes[0])
                return interlacing_suffixes[interlacing];
        else
                return NULL;
}

enum interlacing_t get_interlacing_from_suffix(const char *suffix)
{
        for (size_t i = 0; i < sizeof interlacing_suffixes / sizeof interlacing_suffixes[0]; ++i) {
                if (interlacing_suffixes[i] && strcmp(suffix, interlacing_suffixes[i]) == 0) {
                        return i;
                }
        }

        return PROGRESSIVE;
}


/**
 * @todo
 * Needs to be more efficient
 */
void il_lower_to_merged(char *dst, char *src, int linesize, int height, void **stored_state)
{
        struct il_lower_to_merged_state {
                size_t field_len;
                char field[];
        };
        struct il_lower_to_merged_state *last_field = (struct il_lower_to_merged_state *) *stored_state;

        char *tmp = malloc(linesize * height);
        char *line1, *line2;

        // upper field
        line1 = tmp;
        int upper_field_len = linesize * ((height + 1) / 2);
        // first check if we have field from last frame
        if (last_field == NULL) {
                last_field = (struct il_lower_to_merged_state *)
                        malloc(sizeof(struct il_lower_to_merged_state) + upper_field_len);
                last_field->field_len = upper_field_len;
                *stored_state = last_field;
                // if no, use current one
                line2 = src + linesize * (height / 2);
        } else {
                // otherwise use field from last "frame"
                line2 = last_field->field;
        }
        for (int y = 0; y < (height + 1) / 2; y++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        // store
        assert ((int) last_field->field_len == upper_field_len);
        memcpy(last_field->field, src + linesize * (height / 2), upper_field_len);

        // lower field
        line1 = tmp + linesize;
        line2 = src;
        for (int y = 0; y < height / 2; y++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

/* TODO: rewrite following 2 functions in more efficient way */
void il_upper_to_merged(char *dst, char *src, int linesize, int height, void **state)
{
        UNUSED(state);
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

void il_merged_to_upper(char *dst, char *src, int linesize, int height, void **state)
{
        UNUSED(state);
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

        frame_copy->data_deleter = vf_data_deleter;

        return frame_copy;
}

bool save_video_frame_as_pnm(struct video_frame *frame, const char *name)
{
        unsigned char *data = NULL, *tmp_data = NULL;
        struct tile *tile = &frame->tiles[0];
        int len = tile->width * tile->height * 3;
        if (frame->color_spec == RGB) {
                data = (unsigned char *) tile->data;
        } else {
                data = tmp_data = (unsigned char *) malloc(len);
                if (frame->color_spec == UYVY) {
                        vc_copylineUYVYtoRGB(data, (const unsigned char *)
                                        tile->data, len);
                } else if (frame->color_spec == RGBA) {
                        vc_copylineRGBAtoRGB(data, (const unsigned char *)
                                        tile->data, len, 0, 8, 16);
                } else {
                        free(tmp_data);
                        return false;
                }
        }

        if (!data) {
                return false;
        }

        FILE *out = fopen(name, "w");
        if(out) {
                fprintf(out, "P6\n%d %d\n255\n", tile->width, tile->height);
                if (fwrite(data, len, 1, out) != 1) {
                        perror("fwrite");
                }
                fclose(out);
        }
        free(tmp_data);

        return true;
}

void vf_store_metadata(struct video_frame *f, void *s)
{
        memcpy(s, (char *) f + offsetof(struct video_frame, fec_params), VF_METADATA_SIZE);
}

void vf_restore_metadata(struct video_frame *f, void *s)
{
        memcpy((char *) f + offsetof(struct video_frame, fec_params), s, VF_METADATA_SIZE);
}


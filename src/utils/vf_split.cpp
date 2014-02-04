/**
 * @file   src/utils/vf_split.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2014 CESNET z.s.p.o.
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

#include <stdio.h>
#include <string.h>
#include "utils/vf_split.h"
#include "video.h"
#include "video_codec.h"

void vf_split(struct video_frame *out, struct video_frame *src,
              unsigned int x_count, unsigned int y_count, int preallocate)
{
        unsigned int        tile_idx, line_idx;
        struct tile        *cur_tiles;
        unsigned int        tile_line = 0;
        int                 out_linesize;
        int                 src_linesize;

        out->color_spec = src->color_spec;
        out->fps = src->fps;
        //out->aux = src->aux | AUX_TILED;

        assert(vf_get_tile(src, 0)->width % x_count == 0u && vf_get_tile(src, 0)->height % y_count == 0u);

        src_linesize = vc_get_linesize(src->tiles[0].width,
                        src->color_spec);

        for(tile_idx = 0u; tile_idx < x_count * y_count; ++tile_idx) {
                out->tiles[tile_idx].width = vf_get_tile(src, 0)->width / x_count;
                out->tiles[tile_idx].height = vf_get_tile(src, 0)->height / y_count;

                out_linesize = vc_get_linesize(out->tiles[tile_idx].width,
                                src->color_spec);
                out->tiles[tile_idx].data_len = out_linesize * out->tiles[tile_idx].height;
        }

        cur_tiles = &out->tiles[0];
        for(line_idx = 0u; line_idx < vf_get_tile(src, 0)->height; ++line_idx, ++tile_line) {
                unsigned int cur_tile_idx;
                unsigned int byte = 0u;

                if (line_idx % (vf_get_tile(src, 0)->height / y_count) == 0u) /* next tiles*/
                {
                        tile_line = 0u;
                        if (line_idx != 0u)
                                cur_tiles += x_count;
                        if (preallocate) {
                                for (cur_tile_idx = 0u; cur_tile_idx < x_count;
                                               ++cur_tile_idx) {
                                        cur_tiles[cur_tile_idx].data = (char *)
                                                malloc(cur_tiles[cur_tile_idx].
                                                                data_len);
                                }
                        }
                }

                for(cur_tile_idx = 0u; cur_tile_idx < x_count; ++cur_tile_idx) {
                        memcpy((void *) &cur_tiles[cur_tile_idx].data[
                                        tile_line *
                                        out_linesize],
                                        (void *) &src->tiles[0].data[line_idx *
                                        src_linesize + byte],
                                        cur_tiles[cur_tile_idx].width *
                                        get_bpp(src->color_spec));
                        byte += cur_tiles[cur_tile_idx].width * get_bpp(src->color_spec);
                }
        }
}

void vf_split_horizontal(struct video_frame *out, struct video_frame *src,
              unsigned int y_count)
{
        unsigned int i;

        for(i = 0u; i < y_count; ++i) {
                //out->aux = src->aux | AUX_TILED;
                out->fps = src->fps;
                out->color_spec = src->color_spec;
                out->tiles[i].width = src->tiles[0].width;
                out->tiles[i].height = src->tiles[0].height / y_count;

                int linesize = vc_get_linesize(out->tiles[i].width,
                                out->color_spec);
                out->tiles[i].data_len = linesize *
                        out->tiles[i].height;
                out->tiles[i].data = src->tiles[0].data + i * out->tiles[i].height
                        * linesize;
        }
}

#include "utils/wait_obj.h"

using namespace std;

namespace {
struct dispose_original_frame_udata {
        dispose_original_frame_udata(struct video_frame *original_frame) :
                        m_disposed(0u),
                        m_original_frame(original_frame) {
                pthread_mutex_init(&m_lock, NULL);
        }

        ~dispose_original_frame_udata() {
                pthread_mutex_destroy(&m_lock);
        }

        static void dispose_tile(struct video_frame *frame) {
                struct dispose_original_frame_udata *inst =
                        (struct dispose_original_frame_udata *) frame->dispose_udata;
                pthread_mutex_lock(&inst->m_lock);
                inst->m_disposed++;
                if (inst->m_disposed == inst->m_original_frame->tile_count) {
                        VIDEO_FRAME_DISPOSE(inst->m_original_frame);
                        pthread_mutex_unlock(&inst->m_lock);
                        delete inst;
                }
                pthread_mutex_unlock(&inst->m_lock);
        }

        pthread_mutex_t m_lock;
        unsigned int m_disposed;
        struct video_frame *m_original_frame;
};

} // end of anonymous namespace

vector<struct video_frame *> vf_separate_tiles(struct video_frame *frame)
{
        vector<struct video_frame *> ret(frame->tile_count, 0);
        struct video_desc desc = video_desc_from_frame(frame);
        desc.tile_count = 1;

        struct dispose_original_frame_udata *udata =
                new dispose_original_frame_udata(frame);

        for (unsigned int i = 0; i < frame->tile_count; ++i) {
                ret[i] = vf_alloc_desc(desc);
                ret[i]->tiles[0].data_len = frame->tiles[i].data_len;
                ret[i]->tiles[0].data = frame->tiles[i].data;
                ret[i]->dispose = dispose_original_frame_udata::dispose_tile;
                ret[i]->dispose_udata = udata;
        }

        return ret;
}

namespace {

struct separate_tiles_dispose_udata {
        separate_tiles_dispose_udata(size_t max_count) : m_count(0) {
                m_dispose = (void (**)(struct video_frame *frame)) calloc(max_count,
                                        sizeof(void (*)(struct video_frame *frame)));
                m_frame = (struct video_frame **) calloc(max_count, sizeof(struct video_frame *));
        }
        separate_tiles_dispose_udata() {
                free((void *) m_dispose);
                free(m_frame);
        }
        void add(struct video_frame *frame) {
                m_dispose[m_count] = frame->dispose;
                m_frame[m_count] = frame;
                m_count++;
        }
        void (**m_dispose)(struct video_frame *frame);
        struct video_frame **m_frame;
        size_t m_count;
};

void separate_tiles_dispose(struct video_frame *frame) {
        struct separate_tiles_dispose_udata *dispose_udata =
                (struct separate_tiles_dispose_udata *) frame->dispose_udata;

        for (size_t i = 0; i < dispose_udata->m_count; i++) {
                dispose_udata->m_dispose[i](dispose_udata->m_frame[i]);
        }
        delete dispose_udata;
        vf_free(frame);
}
} // end of anonymous namespace

struct video_frame * vf_merge_tiles(std::vector<struct video_frame *> const & tiles)
{
        struct video_desc desc = video_desc_from_frame(tiles[0]);
        desc.tile_count = tiles.size();
        struct video_frame *ret = vf_alloc_desc(desc);

        struct separate_tiles_dispose_udata *udata =
                new separate_tiles_dispose_udata(tiles.size());

        for (unsigned int i = 0; i < tiles.size(); ++i) {
                ret->tiles[i].data = tiles[i]->tiles[0].data;
                ret->tiles[i].data_len = tiles[i]->tiles[0].data_len;
                if (tiles[i]->dispose) {
                        udata->add(tiles[i]);
                }
        }

        ret->dispose = separate_tiles_dispose;
        ret->dispose_udata = udata;

        return ret;
}


/**
 * @file   tools/ipc_frame_ug.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2026 CESNET, zájmové sdružení právnických osob
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>

#include "ipc_frame_ug.h"
#include "ipc_frame.h"
#include "pixfmt_conv.h"
#include "types.h"
#include "video_codec.h"
#include "debug.h"

namespace {
void scale_frame(char *dst, char *src,
                int src_w, int src_h,
                int f, codec_t codec)
{
        int src_line_len = vc_get_linesize(src_w, codec);
        int block_size = get_pf_block_bytes(codec);
        assert(block_size > 0);
        int blocks_per_line = src_line_len / block_size / f;
        int dst_line_len = vc_get_linesize(blocks_per_line * get_pf_block_pixels(codec), codec);
        int out_lines = 0;
        for(int y = 0; y + f <= src_h; y += f){
                int written = 0;
                for(int x = 0; x + f * block_size <= src_line_len; x += f * block_size){
                        memcpy(dst + out_lines * dst_line_len + written, src + y*src_line_len + x, block_size);
                        written += block_size;
                }
                out_lines++;
        }
}

}//anon namespace

bool ipc_frame_from_ug_frame_hq(struct Ipc_frame *dst,
                const struct video_frame *src,
                codec_t codec,
                unsigned scale_factor)
{
        assert(codec == RGB);

        if(!src)
                return false;

        decoder_t dec = get_decoder_from_to(src->color_spec, codec);
        if(!dec){
                return false;
        }

        dst->header.width = src->tiles[0].width;
        dst->header.height = src->tiles[0].height;
        dst->header.color_spec = static_cast<Ipc_frame_color_spec>(codec);

        if(scale_factor != 0){
                int block_size_px = get_pf_block_pixels(codec);
                int block_count = (dst->header.width + block_size_px - 1) / block_size_px;
                dst->header.width = (block_count / scale_factor) * block_size_px;
                dst->header.height /= scale_factor;
        }

        int dst_frame_size = get_bpp(codec) * dst->header.width * dst->header.height;
        if(!ipc_frame_reserve(dst, dst_frame_size))
                return false;

        dst->header.data_len = dst_frame_size;

        char *scale_src = nullptr;
        std::vector<unsigned char> rgb_frame;

        if(dec == vc_memcpy)
                scale_src = src->tiles[0].data;
        else{
                auto rgb_line_len = vc_get_linesize(src->tiles[0].width, codec);
                unsigned char *dec_dst = nullptr;
                if(scale_factor != 0){
                        rgb_frame.resize(rgb_line_len * src->tiles[0].height);
                        scale_src = (char *) rgb_frame.data();
                        dec_dst = (unsigned char *) scale_src;
                } else {
                        dec_dst = (unsigned char *) dst->data;
                }

                for(unsigned i = 0; i < src->tiles[0].height; i++){
                        dec(dec_dst + rgb_line_len * i,
                                        (unsigned char *) src->tiles[0].data + vc_get_linesize(src->tiles[0].width, src->color_spec) * i,
                                        rgb_line_len,
                                        0, 8, 16);
                }
        }

        if(scale_factor == 0)
                return true;

        scale_frame(dst->data, scale_src,
                        src->tiles[0].width, src->tiles[0].height,
                        scale_factor, codec);

        return true;
}

Ipc_frame_color_spec ipc_frame_color_spec_from_ug(codec_t codec){
        switch(codec){
        case RGBA: return IPC_FRAME_COLOR_RGBA;
        case RGB: return IPC_FRAME_COLOR_RGB;
        case UYVY: return IPC_FRAME_COLOR_UYVY;
        default: return IPC_FRAME_COLOR_NONE;
        }
}

bool ipc_frame_from_ug_frame(struct Ipc_frame *dst,
                const struct video_frame *src,
                codec_t codec,
                unsigned scale_factor)
{
        if(!src)
                return false;

        decoder_t dec = nullptr;
        if(codec != VIDEO_CODEC_NONE){
                dec = get_decoder_from_to(src->color_spec, codec);
                if(!dec){
                        return false;
                }
        } else {
                codec = src->color_spec;
                dec = vc_memcpy;
        }

        dst->header.width = src->tiles[0].width;
        dst->header.height = src->tiles[0].height;
        dst->header.color_spec = ipc_frame_color_spec_from_ug(codec);

        int dst_frame_to_allocate = 0;

        if(scale_factor != 0){
                int block_size_px = get_pf_block_pixels(src->color_spec);
                int block_count = (dst->header.width + block_size_px - 1) / block_size_px;
                dst->header.width = (block_count / scale_factor) * block_size_px;
                dst->header.height /= scale_factor;

                if(dec != vc_memcpy){
                        //When both scaling and converting we need a tmp space - allocate extra
                        dst_frame_to_allocate += vc_get_linesize(dst->header.width, src->color_spec) * dst->header.height;
                }
        }

        int dst_frame_size = get_bpp(codec) * dst->header.width * dst->header.height;
        dst_frame_to_allocate += dst_frame_size;

        if(!ipc_frame_reserve(dst, dst_frame_to_allocate))
                return false;

        dst->header.data_len = dst_frame_size;

        char *scale_dst = dst->data;
        unsigned char *dec_dst = (unsigned char *) dst->data;
        unsigned char *dec_src = (unsigned char *) src->tiles[0].data;

        if(scale_factor != 0){
                if(dec != vc_memcpy){
                        scale_dst = dst->data + dst_frame_size;
                }
                dec_src = (unsigned char *) scale_dst;

                scale_frame(scale_dst, src->tiles[0].data,
                                src->tiles[0].width, src->tiles[0].height,
                                scale_factor, src->color_spec);

        }


        if(dec_src == dec_dst){
                assert(dec == vc_memcpy);
                return true;
        }

        int dst_line_len = vc_get_linesize(dst->header.width, codec);
        int src_line_len = vc_get_linesize(dst->header.width, src->color_spec);
        for(int i = 0; i < dst->header.height; i++){
                dec(dec_dst + dst_line_len * i,
                                dec_src + src_line_len * i,
                                dst_line_len,
                                0, 8, 16);
        }

        return true;
}

int ipc_frame_get_scale_factor(int src_w, int src_h, int target_w, int target_h){
        if(target_w== -1 || target_h== -1)
                return 0;

        float scale = std::sqrt((static_cast<float>(src_w) * src_h)
                        / (static_cast<float>(target_w) * target_h));

        if(scale < 1)
                scale = 1;

        return std::round(scale);
}

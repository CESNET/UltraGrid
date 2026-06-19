/**
 * @file   openapv/openapv_common.cpp
 * @author Juraj Zemančík    <550535@mail.muni.cz>
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právických osob
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

#include "openapv_common.hpp"

#include <cstring>
#include <memory>
#include <oapv/oapv.h>

#include "debug.h"

#define MOD_NAME "[oapv] "

void ug_oapv_imgb_free(oapv_imgb_t *imgb){
        if(!imgb)
                return;

        for (int i = 0; i < imgb->np; ++i){
                free(imgb->baddr[i]);
        }

        delete imgb;
}

const char *oapv_err_str(int err) {
        switch (err) {
        case OAPV_OK:                         return "ok";
        case OAPV_ERR_INVALID_ARGUMENT:       return "invalid argument";
        case OAPV_ERR_OUT_OF_MEMORY:          return "out of memory";
        case OAPV_ERR_REACHED_MAX:            return "reached max";
        case OAPV_ERR_UNSUPPORTED:            return "unsupported";
        case OAPV_ERR_UNEXPECTED:             return "unexpected";
        case OAPV_ERR_UNSUPPORTED_COLORSPACE: return "unsupported color space";
        case OAPV_ERR_MALFORMED_BITSTREAM:    return "malformed bitstream";
        case OAPV_ERR_OUT_OF_BS_BUF:          return "bitstream buffer too small";
        case OAPV_ERR_NOT_FOUND:              return "not found";
        case OAPV_ERR_FAILED_SYSCALL:         return "failed syscall";
        case OAPV_ERR_INVALID_PROFILE:        return "invalid profile";
        case OAPV_ERR_INVALID_LEVEL:          return "invalid level";
        case OAPV_ERR_INVALID_WIDTH:          return "invalid width (odd width is not allowed for 4:2:2)";
        case OAPV_ERR_INVALID_HEIGHT:         return "invalid height";
        case OAPV_ERR_INVALID_QP:             return "invalid QP (encoder accepts 0~63 only, regardless of bit depth)";
        case OAPV_ERR_INVALID_FAMILY:         return "invalid family";
        default:                              return "unknown error";
        }
}

oapv_imgb_t *create_oapv_imgb(int width, int height, int colorspace){
        auto ret = std::make_unique<oapv_imgb_t>();
        *ret = {};

        ret->cs = colorspace;
        ret->w[0] = width;
        ret->h[0] = height;

        int cf = OAPV_CS_GET_FORMAT(colorspace);
        switch(cf) {
        case OAPV_CF_YCBCR422:
                ret->w[1] = ret->w[2] = (width + 1) / 2;
                ret->h[1] = ret->h[2] = height;
                ret->np = 3;
                break;
        case OAPV_CF_YCBCR444:
                ret->w[1] = ret->w[2] = width;
                ret->h[1] = ret->h[2] = height;
                ret->np = 3;
                break;
        case OAPV_CF_YCBCR4444:
                ret->w[1] = ret->w[2] = ret->w[3] = width;
                ret->h[1] = ret->h[2] = ret->h[3] = height;
                ret->np = 4;
                break;
        default:
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported color format for input buffer: %d\n", cf);
                return nullptr;
        }

        for (int i = 0; i < ret->np; ++i) {
                ret->aw[i] = ((ret->w[i] + OAPV_MB_W - 1) / OAPV_MB_W) * OAPV_MB_W;
                ret->ah[i] = ((ret->h[i] + OAPV_MB_H - 1) / OAPV_MB_H) * OAPV_MB_H;
                ret->s[i] = ret->aw[i] * OAPV_CS_GET_BYTE_DEPTH(ret->cs);
                ret->e[i] = ret->ah[i];
                ret->bsize[i] = ret->s[i] * ret->e[i];
                ret->baddr[i] = malloc(ret->bsize[i]);
                if (ret->baddr[i] == nullptr) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate plane %d (%d bytes).\n",
                                i, ret->bsize[i]);
                        for (int j = 0; j < i; ++j) {
                                free(ret->baddr[j]);
                        }
                        return nullptr;
                }
                memset(ret->baddr[i], 0, ret->bsize[i]);
                ret->a[i] = ret->baddr[i];
        }

        return ret.release();
}

bool Oapv_Frames::configure_with(int width, int height, int colorspace){
        imgb.reset(create_oapv_imgb(width, height, colorspace));
        if(!imgb){
                frms.num_frms = 0;
                return false;
        }

        frms.num_frms = 1;
        frms.frm[0].group_id = 1;
        frms.frm[0].pbu_type = OAPV_PBU_TYPE_PRIMARY_FRAME;
        frms.frm[0].imgb = imgb.get();
        return true;
}


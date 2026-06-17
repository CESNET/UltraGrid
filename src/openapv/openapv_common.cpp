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

#include <oapv/oapv.h>

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


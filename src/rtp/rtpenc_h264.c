/**
 * @file
 * @author Gerard Castillo <gerard.castillo@i2cat.net>
 * @author David Cassany   <david.cassany@i2cat.net>
 * @author Martin Pulec    <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2014-2022 CESNET
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
 *      California Information Sciences Institute.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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

#include <stddef.h>
#include <stdint.h>

#include "debug.h"
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpenc_h264.h"

static uint32_t get4Bytes(const unsigned char *ptr) {
        return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}

/**
 * Returns pointer to next NAL unit in stream.
 *
 * @param with_start_code returned pointer will point to start code preceeding NAL unit, otherwise it will point
 *                        to NAL unit beginning (skipping the start code)
 */
static const unsigned char *get_next_nal(const unsigned char *start, long len, _Bool with_start_code) {
        const unsigned char * const stop = start + len;
        while (stop - start >= 4) {
                uint32_t next4Bytes = get4Bytes(start);
                if (next4Bytes == 0x00000001) {
                        return start + (with_start_code ? 0 : 4);
                }
                if ((next4Bytes & 0xFFFFFF00) == 0x00000100) {
                        return start + (with_start_code ? 0 : 3);
                }
                // We save at least some of "next4Bytes".
                if ((unsigned) (next4Bytes & 0xFF) > 1) {
                        // Common case: 0x00000001 or 0x000001 definitely doesn't begin anywhere in "next4Bytes", so we save all of it:
                        start += 4;
                } else {
                        // Save the first byte, and continue testing the rest:
                        start += 1;
                }
        }
        return NULL;
}

/**
 * Returns pointer to next NAL unit in stream (excluding start code).
 *
 * @param start  start of the buffer
 * @param len    length of the buffer
 * @param endptr pointer to store the end of the NAL unit; may be NULL
 * @returns      NAL unit beginning or NULL if no further NAL unit was found
 */
const unsigned char *rtpenc_h264_get_next_nal(const unsigned char *start, long len, const unsigned char **endptr) {
        const unsigned char *nal = get_next_nal(start, len, 0);
        if (endptr == NULL) {
                return nal;
        }
        if (nal == NULL) {
                return NULL;
        }
        const unsigned char *end = get_next_nal(nal, len - (nal - start), 1);
        *endptr = end ? end : start + len;
        return nal;
}

/// @returns first NAL that is not AUD
const unsigned char *
rtpenc_get_first_nal(const unsigned char *src, long src_len, bool hevc)
{
        const unsigned char *nal       = src;
        int                  nalu_type = 0;
        do {
                nal =
                    rtpenc_h264_get_next_nal(nal, src_len - (nal - src), NULL);
                if (!nal) {
                        return NULL;
                }
                nalu_type = NALU_HDR_GET_TYPE(nal[0], hevc);
                debug_msg("Received %s NALU.\n", get_nalu_name(nalu_type));
        } while (nalu_type == NAL_H264_AUD || nalu_type == NAL_HEVC_AUD);
        return nal;
}

/// @returns name of NAL unit
const char *
get_nalu_name(int type)
{
        switch ((enum nal_type) type) {
        case NAL_H264_IDR:
                return "H264 IDR";
        case NAL_H264_SEI:
                return "H264 SEI";
        case NAL_H264_SPS:
                return "H264 SPS";
        case NAL_H264_PPS:
                return "H264 PPS";
        case NAL_H264_AUD:
                return "H264 AUD";
        case NAL_HEVC_VPS:
                return "HEVC VPS";
        case NAL_HEVC_SPS:
                return "HEVC SPS";
        case NAL_HEVC_PPS:
                return "HEVC PPS";
        case NAL_HEVC_AUD:
                return "HEVC AUD";
        }
        _Thread_local static char buf[32];
        snprintf(buf, sizeof buf, "(NALU %d)", type);
        return buf;
}

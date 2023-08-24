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

#ifndef _RTP_ENC_H264_H
#define _RTP_ENC_H264_H

#define START_CODE_3B 0x00, 0x00, 0x01
#define START_CODE_4B 0x00, 0x00, 0x00, 0x01
#define H264_NAL_SEI_PREFIX 0x06, 0x05 // SEI, category
#define HEVC_NAL_SEI_PREFIX 0x4E, 0x01, 0x05 // SEI, category
/// custom orig format byte syntax - highest 1 bit zero, next 2 bits (depth-8)/2, next 3 bits subsampling (Y-1) from X:Y:Z, next 1 bit - vertical is subsampled, last bit RGB
#define UG_ORIG_FORMAT_ISO_IEC_11578_GUID 0xDB, 0x69, 0xDA, 0x43, 0x42, 0x11, 0x40, 0xEC, 0xA2, 0xF1, 0x45, 0x96, 0x64, 0xFA, 0x14, 0x63

#ifndef __cplusplus
#include <stdbool.h>
#else
extern "C" {
#endif

// function documented at definition
const unsigned char *rtpenc_get_next_nal(const unsigned char *start, long len,
                                         const unsigned char **endptr);
const unsigned char *rtpenc_get_first_nal(const unsigned char *src,
                                          long src_len, bool hevc);
const char          *get_nalu_name(int type);

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file   rtp/fec.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET z.s.p.o.
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

#ifndef FEC_H_
#define FEC_H_

#include "types.h"

#ifdef __cplusplus
#include <map>
#include <memory>

struct video_frame;

struct fec {
        virtual std::shared_ptr<video_frame> encode(std::shared_ptr<video_frame>) = 0;
        /**
         * @retval true  if whole frame was reconstructed
         * @retval false if decoder was unable to reconstruct the frame
         *               However, if it was reconstructed at least partially
         *               (or the code is a systematic one) and length
         *               can be read, set len to a non-zero value.
         */
        virtual bool decode(char *in, int in_len, char **out, int *len,
                        const std::map<int, int> &) = 0;
        virtual ~fec() {}

        static fec *create_from_config(const char *str) noexcept;
        static fec *create_from_desc(struct fec_desc) noexcept;
        static int pt_from_fec_type(enum tx_media_type media_type, enum fec_type fec_type, bool encrypted) throw();
        static enum fec_type fec_type_from_pt(int pt) throw();
};
#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif
int fec_pt_from_fec_type(enum tx_media_type media_type, enum fec_type fec_type, bool encrypted);
#ifdef __cplusplus
}
#endif


#endif // FEC_H_


/**
 * @file   fec.cpp
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <string>

#include "rtp/fec.h"
#include "rtp/ldgm.h"
#include "rtp/rs.h"
#include "rtp/rtp_callback.h"

using namespace std;

fec *fec::create_from_config(const char *c_str)
{
        if (strncmp(c_str, "LDGM percents ", strlen("LDGM percents ")) == 0) {
                char *str = strdup(c_str);
                int mtu_len, data_len;
                double loss_pct;
                char *ptr = str + strlen("LDGM percents ");
                char *save_ptr, *item;
                item = strtok_r(ptr, " ", &save_ptr);
                assert (item != NULL);
                mtu_len = atoi(item);
                assert(mtu_len > 0);
                item = strtok_r(NULL, " ", &save_ptr);
                assert (item != NULL);
                data_len = atoi(item);
                assert(data_len > 0);
                item = strtok_r(NULL, " ", &save_ptr);
                assert (item != NULL);
                loss_pct = atof(item);
                assert(loss_pct > 0.0);
                fec *ret = new ldgm(mtu_len, data_len, loss_pct);
                free(str);
                return ret;
        } else if (strncmp(c_str, "LDGM cfg ", strlen("LDGM cfg ")) == 0) {
                return new ldgm(c_str + strlen("LDGM cfg "));
        } else if (strncmp(c_str, "RS cfg ", strlen("RS cfg ")) == 0) {
                return new rs(c_str + strlen("rs cfg "));
        } else {
                throw string("Unrecognized FEC configuration!");
        }
        return NULL;
}

fec *fec::create_from_desc(struct fec_desc desc)
{
        switch (desc.type) {
        case FEC_LDGM:
                return new ldgm(desc.k, desc.m, desc.c, desc.seed);
        case FEC_RS:
                return new rs(desc.k, desc.k + desc.m);
        default:
                abort();
        }

}

int fec::pt_from_fec_type(enum fec_type type, bool encrypted) throw()
{
        switch (type) {
        case FEC_NONE:
                return encrypted ? PT_VIDEO_LDGM : PT_VIDEO;
        case FEC_LDGM:
                return encrypted ? PT_ENCRYPT_VIDEO_LDGM : PT_VIDEO_LDGM;
        case FEC_RS:
                return encrypted ? PT_ENCRYPT_VIDEO_RS : PT_VIDEO_RS;
        default:
                abort();
        }
}

enum fec_type fec::fec_type_from_pt(int pt) throw()
{
        switch (pt) {
        case PT_VIDEO:
        case PT_ENCRYPT_VIDEO:
                return FEC_NONE;
        case PT_VIDEO_LDGM:
        case PT_ENCRYPT_VIDEO_LDGM:
                return FEC_LDGM;
        case PT_VIDEO_RS:
        case PT_ENCRYPT_VIDEO_RS:
                return FEC_RS;
        default:
                abort();
        }
}


int fec_pt_from_fec_type(enum fec_type type, bool encrypted)
{
        return fec::pt_from_fec_type(type, encrypted);
}


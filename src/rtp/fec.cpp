/**
 * @file   rtp/fec.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Common ancesor of FEC modules.
 */
/*
 * Copyright (c) 2014-2021 CESNET, z. s. p. o.
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

#include "debug.h"
#include "rtp/fec.h"
#include "rtp/ldgm.h"
#include "rtp/rs.h"
#include "rtp/rtp_callback.h"
#include "ug_runtime_error.hpp"
#include "utils/macros.h"

using namespace std;

fec *fec::create_from_config(const char *c_str) noexcept
{
        try {
                if (strncmp(c_str, "LDGM percents ", strlen("LDGM percents ")) == 0) {
                        char *str = strdup(c_str);
                        char *ptr = str + strlen("LDGM percents ");
                        char *save_ptr = nullptr;
                        char *item = nullptr;
                        item = strtok_r(ptr, " ", &save_ptr);
                        assert (item != nullptr);
                        int mtu_len = stoi(item);
                        assert(mtu_len > 0);
                        item = strtok_r(nullptr, " ", &save_ptr);
                        assert (item != nullptr);
                        int data_len = stoi(item);
                        assert(data_len > 0);
                        item = strtok_r(nullptr, " ", &save_ptr);
                        assert (item != nullptr);
                        double loss_pct = stof(item);
                        assert(loss_pct > 0.0);
                        fec *ret = new ldgm(mtu_len, data_len, loss_pct);
                        free(str);
                        return ret;
                }
                if (strncmp(c_str, "LDGM cfg ", strlen("LDGM cfg ")) == 0) {
                        return new ldgm(c_str + strlen("LDGM cfg "));
                }
                if (strncmp(c_str, "RS cfg ", strlen("RS cfg ")) == 0) {
                        return new rs(c_str + strlen("rs cfg "));
                }
                throw ug_runtime_error("Unrecognized FEC configuration!");
        } catch (string const &s) {
                LOG(LOG_LEVEL_ERROR) << s << "\n";
        } catch (exception const &e) {
                LOG(LOG_LEVEL_ERROR) << e.what() << "\n";
        } catch (int i) {
                if (i != 0) {
                        LOG(LOG_LEVEL_ERROR) << "FEC initialization returned " << i << "\n";
                }
        } catch (...) {
                LOG(LOG_LEVEL_ERROR) << "Unknown FEC error!\n";
        }
        return nullptr;
}

fec *fec::create_from_desc(struct fec_desc desc) noexcept
{
        try {
                switch (desc.type) {
                        case FEC_LDGM:
                                return new ldgm(desc.k, desc.m, desc.c, desc.seed);
                        case FEC_RS:
                                return new rs(desc.k, desc.k + desc.m, desc.mult);
                        default:
                                abort();
                }
        } catch (string const &s) {
                LOG(LOG_LEVEL_ERROR) << s << "\n";
        } catch (exception const &e) {
                LOG(LOG_LEVEL_ERROR) << e.what() << "\n";
        } catch (int i) {
                if (i != 0) {
                        LOG(LOG_LEVEL_ERROR) << "FEC initialization returned " << i << "\n";
                }
        } catch (...) {
                LOG(LOG_LEVEL_ERROR) << "Unknown FEC error!\n";
        }
        return nullptr;

}

int fec::pt_from_fec_type(enum tx_media_type media_type, enum fec_type fec_type, bool encrypted) throw()
{
        if (media_type == TX_MEDIA_VIDEO) {
                switch (fec_type) {
                case FEC_NONE:
                        return encrypted ? PT_ENCRYPT_VIDEO : PT_VIDEO;
                case FEC_LDGM:
                        return encrypted ? PT_ENCRYPT_VIDEO_LDGM : PT_VIDEO_LDGM;
                case FEC_RS:
                        return encrypted ? PT_ENCRYPT_VIDEO_RS : PT_VIDEO_RS;
                default: break;
                }
        } else {
                switch (fec_type) {
                case FEC_NONE:
                        return encrypted ? PT_ENCRYPT_AUDIO : PT_AUDIO;
                case FEC_RS:
                        return encrypted ? PT_ENCRYPT_AUDIO_RS : PT_AUDIO_RS;
                default: break;
                }
        }
        UG_ASSERT(0 && "Unsupported media/FEC type combination");
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


int fec_pt_from_fec_type(enum tx_media_type media_type, enum fec_type fec_type, bool encrypted)
{
        return fec::pt_from_fec_type(media_type, fec_type, encrypted);
}


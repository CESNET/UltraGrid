/**
 * @file utils/sdp_parser.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#include "sdp_parser.hpp"
#include "utils/string_view_utils.hpp"

Sap_packet_view Sap_packet_view::from_buffer(const void *buf, size_t size){
        Sap_packet_view ret;

        std::string_view sap(static_cast<const char *>(buf), size);

        if(sap.size() < 8){
                return ret;
        }

        ret.flags = sap[0];
        sap.remove_prefix(1);

        ret.version = ret.flags >> 6;

        uint8_t auth_len = 0;
        if(sap.empty())
                return ret;

        auth_len = sap[0];
        sap.remove_prefix(1);
        
        ret.hash = sap[0] << 8 | sap[1];
        sap.remove_prefix(2);

        ret.source = sap.substr(0, 4);
        sap.remove_prefix(4);

        if(sap.size() < auth_len)
                return ret;

        sap.remove_prefix(auth_len);

        if(sv_is_prefix(sap, "v=0")){
                //RFC says that the payload type is optional if it's sdp and
                //that it should be detected by the presence of v=0
                ret.payload_type = "application/sdp";
                ret.payload = sap;
        } else {
                auto null_idx = sap.find('\0');
                if(null_idx == sap.npos){
                        return ret;
                }
                ret.payload_type = sap.substr(0, null_idx);
                ret.payload = sap.substr(null_idx + 1);
        }

        ret.valid = true;
        return ret;
}

Sdp_view Sdp_view::from_buffer(const void *buf, size_t size){
        Sdp_view ret;

        std::string_view sdp(static_cast<const char *>(buf), size);

        if(sdp.empty()){
                return ret;
        }

        while(!sdp.empty()){
                auto line = tokenize(sdp, '\n');
                if(line.empty())
                        continue;

                if(line.back() == '\r')
                        line.remove_suffix(1);

                auto eq_idx = line.find('=');
                if(eq_idx == line.npos){
                        continue;
                }
                auto key = line.substr(0, eq_idx);
                auto val = line.substr(eq_idx + 1);

                if(key == "c"){
                        if(ret.media.empty()){
                                ret.connection = val;
                        } else {
                                ret.media.back().connection = val;
                        }
                } else if(key == "a"){
                        auto attrib = tokenize(val, ':');
                        auto attrib_val = tokenize(val, ':');

                        if(ret.media.empty()){
                                ret.session_attributes.push_back({attrib, attrib_val});
                        } else {
                                ret.media.back().attributes.push_back({attrib, attrib_val});
                        }

                } else if(key =="s"){
                        ret.session_name = val;
                } else if(key == "o"){
                        ret.origin = val;
                } else if(key == "m"){
                        ret.media.emplace_back();
                        auto& medium = ret.media.back();

                        medium.media_desc = val;
                }

        }

        ret.valid = true;

        return ret;
}

Rtp_pkt_view Rtp_pkt_view::from_buffer(void *buf, size_t size){
        Rtp_pkt_view ret{};

        if(size < 12)
                return ret;

        auto charbuf = static_cast<unsigned char*>(buf);

        [[maybe_unused]] uint8_t version = charbuf[0] >> 6;
        bool padding = charbuf[0] & (1 << 5);
        bool extension = charbuf[0] & (1 << 4);
        ret.csrc_count = charbuf[0] & 0x0F;
        ret.marker = charbuf[1] & 0x80;
        ret.payload_type = charbuf[1] & 0x7F;
        ret.seq = charbuf[2] << 8 | charbuf[3];
        ret.timestamp = charbuf[4] << 24 | charbuf[5] << 16 | charbuf[6] << 8 | charbuf[7];
        ret.ssrc = charbuf[8] << 24 | charbuf[9] << 16 | charbuf[10] << 8 | charbuf[11];


        size_t data_offset = 12 + ret.csrc_count * 4;
        ret.data = &charbuf[data_offset];
        ret.data_len = size - data_offset;

        if(padding){
                if(ret.data_len == 0)
                        return ret;

                uint8_t padding_bytes = ((uint8_t *)(ret.data))[ret.data_len - 1];
                if(padding_bytes < ret.data_len)
                        return ret;

                ret.data_len -= padding_bytes;
        }

        ret.valid = true;
        return ret;
}

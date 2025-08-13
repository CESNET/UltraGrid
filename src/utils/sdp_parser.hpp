/**
 * @file utils/sdp_parser.hpp
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

#ifndef SDP_PARSER_HPP_cccb52235120
#define SDP_PARSER_HPP_cccb52235120

#include <string_view>
#include <vector>
#include <cstdint>

#define SAP_FLAG_COMPRESSED (1 << 0)
#define SAP_FLAG_ENCRYPTED (1 << 1)
#define SAP_FLAG_DELETION (1 << 2)
#define SAP_FLAG_IPV6 (1 << 4)


struct Sap_packet_view{
        uint8_t version;
        uint8_t flags;
        uint16_t hash;

        std::string_view source;
        std::string_view payload_type;
        std::string_view payload;

        static Sap_packet_view from_buffer(const void *buf, size_t size);

        bool isValid() const { return valid; }
        bool isCompressed() const { return flags & SAP_FLAG_COMPRESSED; }
        bool isEncrypted() const { return flags & SAP_FLAG_ENCRYPTED; }
        bool isDeletion() const { return flags & SAP_FLAG_DELETION; }
        bool isIpv6() const { return flags & SAP_FLAG_IPV6; }

        bool valid = false;

};

struct Sdp_attribute{
        std::string_view key;
        std::string_view val;
};

struct Sdp_media_desc{
        std::string_view media_desc;
        std::string_view title;
        std::string_view connection;
        std::vector<Sdp_attribute> attributes;
};

struct Sdp_view{
        std::string_view username;
        uint64_t sess_id;
        uint64_t sess_version;
        std::string_view nettype;
        std::string_view addrtype;
        std::string_view unicast_addr;

        std::string_view session_name;
        std::string_view session_info;
        std::string_view connection;

        std::string_view session_time;
        std::vector<Sdp_attribute> session_attributes;

        std::vector<Sdp_media_desc> media;

        static Sdp_view from_buffer(const void *buf, size_t size);
        bool isValid() const { return valid; }
        
        bool valid = false;
};

struct Rtp_pkt_view{
        bool marker;
        uint8_t payload_type;
        uint16_t seq;
        uint32_t timestamp;
        uint32_t ssrc;
        //uint32_t *csrcs;
        size_t csrc_count;
        void *data;
        size_t data_len;

        static Rtp_pkt_view from_buffer(void *buf, size_t size);
        bool isValid() const { return valid; }
        
        bool valid = false;
};


#endif

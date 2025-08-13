#include "sdp_parser.hpp"
#include "utils/string_view_utils.hpp"

Sap_packet_view Sap_packet_view::from_buffer(const void *buf, size_t size){
        Sap_packet_view ret;

        std::string_view sap(static_cast<const char *>(buf), size);

        if(sap.empty()){
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
        
        //TODO error check length
        ret.hash = sap[0] << 8 | sap[1];
        sap.remove_prefix(2);

        //TODO error check length
        ret.source = sap.substr(0, 4);
        sap.remove_prefix(4);

        //TODO error check length
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

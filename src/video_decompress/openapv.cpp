#include <oapv/oapv.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"

namespace {

struct state_video_decompress_openapv {
        
        oapvd_t decoder_handle{};
        oapvm_t metadata_handle{};
        oapvd_cdesc_t cdesc{};

        oapv_frms_t decoded_frames{};
        oapv_bitb_t input_buffer{};
};

static void *openapv_decompress_init(void) {
        // TODO
}

static int openapv_decompress_reconfigure(void *state, struct video_desc desc,
        int rshift, int gshift, int bshift, int pitch, codec_t out_codec) {
        // TODO        
}

static decompress_status openapv_decompress(void *state, unsigned char *dst, unsigned char *buffer, 
        unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop)
{
        // TODO
}

static int openapv_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        // TODO
}

static void openapv_decompress_done(void *state) {
       // TODO
}

static int openapv_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        // TODO
}

static const struct video_decompress_info openapv_info = {
        openapv_decompress_init,
        openapv_decompress_reconfigure,
        openapv_decompress,
        openapv_decompress_get_property,
        openapv_decompress_done,
        openapv_decompress_get_priority,
};

REGISTER_MODULE(openapv, &openapv_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

}

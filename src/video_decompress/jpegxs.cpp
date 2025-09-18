#include <svt-jpegxs/SvtJpegxsDec.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"

struct state_decompress_jpegxs {
        struct svt_jpeg_xs_decoder_api *decoder;
        ~state_decompress_jpegxs() {
                svt_jpeg_xs_decoder_close(decoder);
        }

        struct video_desc desc;
        codec_t out_codec;
};

static void *jpegxs_decompress_init(void) {
        struct state_decompress_jpegxs *s = new state_decompress_jpegxs();

        return s;
}

static const struct video_decompress_info jpegxs_info = {
        jpegxs_decompress_init, // jpegxs_decompress_init
        NULL, // jpegxs_decompress_reconfigure
        NULL, // jpegxs_decompress
        NULL, // jpegxs_decompress_get_property
        NULL, // jpegxs_decompress_done
        NULL, // jpegxs_decompress_get_priority
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
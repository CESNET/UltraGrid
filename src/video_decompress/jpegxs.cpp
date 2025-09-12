#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"
#include <svt-jpegxs/SvtJpegxsDec.h>

struct state_decompress_gpujpeg {
        struct svt_jpeg_xs_decoder_api *decoder;

        struct video_desc desc;
        codec_t out_codec;
};

static const struct video_decompress_info jpegxs_info = {
        NULL, // jpegxs_decompress_init
        NULL, // jpegxs_decompress_reconfigure
        NULL, // jpegxs_decompress
        NULL, // jpegxs_decompress_get_property
        NULL, // jpegxs_decompress_done
        NULL, // jpegxs_decompress_get_priority
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
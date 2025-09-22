#include <svt-jpegxs/SvtJpegxsDec.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"

struct state_decompress_jpegxs {
        struct svt_jpeg_xs_decoder_api *decoder;
        ~state_decompress_jpegxs() {
                free(bitstream.buffer);
                for (uint8_t i = 0; i < image_config.components_num; ++i) {
                        free(image_buffer.data_yuv[i]);
                }
                svt_jpeg_xs_decoder_close(decoder);
        }

        svt_jpeg_xs_image_config_t image_config;

        svt_jpeg_xs_bitstream_buffer_t bitstream;
        svt_jpeg_xs_image_buffer_t image_buffer;

        struct video_desc desc;
        codec_t out_codec;
};

static void *jpegxs_decompress_init(void) {
        struct state_decompress_jpegxs *s = new state_decompress_jpegxs();

        return s;
}

static int jpegxs_decompress_reconfigure(void *state, struct video_desc desc,
        int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpegxs *s = (struct state_decompress_jpegxs *) state;

        // TODO
        
}

static decompress_status jpegxs_decompress(void *state, unsigned char *dst, unsigned char *buffer, 
        unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop)
{
        // TODO
}

static int jpegxs_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = false;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = false;
                                *len = sizeof(int);
                                ret = true;
                        }
                        break;
                default:
                        ret = false;
        }

        return ret;
}

static void jpegxs_decompress_done(void *state) {
       delete (struct state_decompress_jpegxs *) state;
}

static int jpegxs_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        if (compression != JPEG_XS) {
                return VDEC_PRIO_NA;
        }

        // TODO

        return VDEC_PRIO_PREFERRED;
}

static const struct video_decompress_info jpegxs_info = {
        jpegxs_decompress_init, // jpegxs_decompress_init
        jpegxs_decompress_reconfigure, // jpegxs_decompress_reconfigure
        jpegxs_decompress, // jpegxs_decompress
        jpegxs_decompress_get_property, // jpegxs_decompress_get_property
        jpegxs_decompress_done, // jpegxs_decompress_done
        jpegxs_decompress_get_priority, // jpegxs_decompress_get_priority
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
#include <memory>

#include "lib_common.h"
#include "video.h"
#include "video_compress.h"
#include "utils/video_frame_pool.h"
#include <svt-jpegxs/SvtJpegxsEnc.h>

using std::shared_ptr;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
private:
        state_video_compress_jpegxs(struct module *parent, const char *opts);
public:
        svt_jpeg_xs_encoder_api_t encoder;
        unsigned int configured:1;
        svt_jpeg_xs_image_buffer_t in_buf;
        svt_jpeg_xs_bitstream_buffer_t out_buf;
        video_frame_pool pool;

        static state_video_compress_jpegxs *create(struct module *parent, const char *opts);
        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();
};

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;
}

state_video_compress_jpegxs *state_video_compress_jpegxs::create(struct module *parent, const char *opts) {
        auto ret = new state_video_compress_jpegxs(parent, opts);
        
        return ret;
}

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc) {
        svt_jpeg_xs_encoder_api_t enc = {};

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
        if (err != SvtJxsErrorNone) {
                return false;
        }

        // TODO parameters
        enc.source_width = desc.width;
        enc.source_height = desc.height;
        enc.input_bit_depth = 8;
        enc.colour_format = COLOUR_FORMAT_PLANAR_YUV422;
        enc.bpp_numerator = 3;

        err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
        if (err != SvtJxsErrorNone) {
                return false;
        }

        uint32_t pixel_size = enc.input_bit_depth <= 8 ? 1 : 2;
        svt_jpeg_xs_image_buffer_t in_buf;
        in_buf.stride[0] = enc.source_width;
        in_buf.stride[1] = enc.source_width / 2;
        in_buf.stride[2] = enc.source_width / 2;
        for (uint8_t i = 0; i < 3; ++i) {
                in_buf.alloc_size[i] = in_buf.stride[i] * enc.source_height * pixel_size;
                in_buf.data_yuv[i] = malloc(in_buf.alloc_size[i]);
                if (!in_buf.data_yuv[i]) {
                        return false;
                }
        }

        svt_jpeg_xs_bitstream_buffer_t out_buf;
        uint32_t bitstream_size = (uint32_t)(
                ((uint64_t)enc.source_width * enc.source_height * enc.bpp_numerator / enc.bpp_denominator + 7) / +8);
        out_buf.allocation_size = bitstream_size;
        out_buf.used_size = 0;
        out_buf.buffer = (uint8_t *) malloc(out_buf.allocation_size);
        if (!out_buf.buffer) {
                return false;
        }

        s->encoder = enc;
        s->in_buf = in_buf;
        s->out_buf = out_buf;
        s->configured = true;

        s->pool.reconfigure(desc, bitstream_size);

        return true;
}

void *
jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if (opts && strcmp(opts, "help") == 0) {
                printf("JPEG XS help message\n");
                return INIT_NOERR;
        }

        s = state_video_compress_jpegxs::create(parent, opts);

        return s;
}

shared_ptr<video_frame> jpegxs_compress(void *state, shared_ptr<video_frame> frame) {
        auto *s = (struct state_video_compress_jpegxs *) state;

        if (!s->configured) {
                struct video_desc desc = video_desc_from_frame(frame.get());
                if (!configure_with(s, desc)) {
                        return NULL;
                }
        }
        
        return frame;
}

// void state_video_compress_jpegxs::push(std::shared_ptr<video_frame> in_frame)
// {
//         // TODO
// }

// std::shared_ptr<video_frame> state_video_compress_jpegxs::pop()
// {
//         // TODO
// }

// static auto jpegxs_compress_push(void *state, std::shared_ptr<video_frame> in_frame) {
//         static_cast<struct state_video_compress_jpegxs *>(state)->push(std::move(in_frame));
// }

// static auto jpegxs_compress_pull(void *state) {
//         return static_cast<struct state_video_compress_jpegxs *>(state)->pop();
// }

static void
jpegxs_compress_done(void *state) {
        delete (struct state_video_compress_jpegxs *) state;
}

static compress_module_info get_jpegxs_module_info() {
        compress_module_info module_info;
        module_info.name = "jpegxs";
        return module_info;
}

const struct video_compress_info jpegxs_info = {
        jpegxs_compress_init, // jpegxs_compress_init
        jpegxs_compress_done, // jpegxs_compress_done
        jpegxs_compress, // jpegxs_compress (synchronous)
        NULL,
        NULL, // jpegxs_compress_push (asynchronous)
        NULL, //  jpegxs_compress_pull
        NULL,
        NULL,
        get_jpegxs_module_info // get_jpegxs_module_info
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
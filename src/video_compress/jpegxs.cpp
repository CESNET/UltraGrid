#include <memory>

#include "lib_common.h"
#include "video_compress.h"
#include <svt-jpegxs/SvtJpegxsEnc.h>

using std::shared_ptr;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
private:
        state_video_compress_jpegxs(struct module *parent, const char *opts);
public:
        svt_jpeg_xs_encoder_api_t m_encoder;

        static state_video_compress_jpegxs *create(struct module *parent, const char *opts);
        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();
};

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;

        m_encoder.source_width = 1920;
        m_encoder.source_height = 1080;
        m_encoder.input_bit_depth = 8;
        m_encoder.colour_format = COLOUR_FORMAT_PLANAR_YUV422;
        m_encoder.bpp_numerator = 3;

        svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &m_encoder);
}

state_video_compress_jpegxs *state_video_compress_jpegxs::create(struct module *parent, const char *opts) {
        auto ret = new state_video_compress_jpegxs(parent, opts);
        
        return ret;
}

void *
jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG XS help message\n");
                return INIT_NOERR;
        }

        s = state_video_compress_jpegxs::create(parent, opts);

        return s;
}

shared_ptr<video_frame> jpegxs_compress(void *state, shared_ptr<video_frame> frame) {
        // TODO
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
        NULL, // jpegxs_compress (synchronous)
        NULL,
        NULL, // jpegxs_compress_push (asynchronous)
        NULL, //  jpegxs_compress_pull
        NULL,
        NULL,
        get_jpegxs_module_info // get_jpegxs_module_info
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
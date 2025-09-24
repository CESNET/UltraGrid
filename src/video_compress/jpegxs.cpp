#include <memory>
#include <iostream>
#include <string>
#include <svt-jpegxs/SvtJpegxsEnc.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_compress.h"
#include "utils/video_frame_pool.h"

#define MOD_NAME "[JPEG XS enc.] "

using std::shared_ptr;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
private:
        state_video_compress_jpegxs(struct module *parent, const char *opts);
public:
        ~state_video_compress_jpegxs() {
                svt_jpeg_xs_encoder_close(&encoder);
                free(out_buf.buffer);
                free(in_buf.data_yuv[0]);
                free(in_buf.data_yuv[1]);
                free(in_buf.data_yuv[2]);
        }
        svt_jpeg_xs_encoder_api_t encoder;
        bool configured = 0;
        svt_jpeg_xs_image_buffer_t in_buf;
        svt_jpeg_xs_bitstream_buffer_t out_buf;
        video_frame_pool pool;

        bool parse_fmt(char *fmt);
        static state_video_compress_jpegxs *create(struct module *parent, const char *opts);
        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();
};

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to load JPEG XS default parameters\n");
                throw 1;
        }

        encoder.input_bit_depth = 8;
        encoder.bpp_numerator = 3;

        if(opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(fmt)) {
                        free(fmt);
                        throw 1;
                }
                free(fmt);
        }
}

state_video_compress_jpegxs *state_video_compress_jpegxs::create(struct module *parent, const char *opts) {
        auto ret = new state_video_compress_jpegxs(parent, opts);
        
        return ret;
}

ColourFormat subsampling_to_jpegxs(int ug_subs) {
        switch (ug_subs) {
        case 444:
                return COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
        case 422:
                return COLOUR_FORMAT_PLANAR_YUV422;
        case 420:
                return COLOUR_FORMAT_PLANAR_YUV420;
        default:
                abort();
        }
}

static bool setup_image_input_buffer(svt_jpeg_xs_image_buffer_t *in_buf, const svt_jpeg_xs_encoder_api_t *enc) {

        uint32_t pixel_size = enc->input_bit_depth <= 8 ? 1 : 2;
        uint32_t w = enc->source_width;
        uint32_t h = enc->source_height;

        uint32_t w_factor = 1;
        uint32_t h_factor = 1;

        switch (enc->colour_format) {
        case COLOUR_FORMAT_PLANAR_YUV444_OR_RGB: // no subsampling
                break;
        case COLOUR_FORMAT_PLANAR_YUV422: // half horizontal
                w_factor = 2;
                break;
        case COLOUR_FORMAT_PLANAR_YUV420: // half horizontal + vertical
                w_factor = 2;
                h_factor = 2;
                break;
        default:
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported colour format\n");
                return false;
        }

        in_buf->stride[0] = w;
        in_buf->stride[1] = w / w_factor;
        in_buf->stride[2] = w / w_factor;

        for (uint8_t i = 0; i < 3; ++i) {
                in_buf->alloc_size[i] = in_buf->stride[i] * (h / h_factor) * pixel_size;
                in_buf->data_yuv[i] = malloc(in_buf->alloc_size[i]);
                if (!in_buf->data_yuv[i]) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate plane %d\n", i);
                        return false;
                }
        }

        return true;
}

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc) {
        s->encoder.source_width = desc.width;
        s->encoder.source_height = desc.height;
        s->encoder.colour_format = subsampling_to_jpegxs(get_subsampling(desc.color_spec) / 10);

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize JPEG XS encoder\n");
                return false;
        }

        if (!setup_image_input_buffer(&s->in_buf, &s->encoder)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize input image buffer\n");
                return false;
        }

        uint32_t bitstream_size = (uint32_t)(
                ((uint64_t)s->encoder.source_width * s->encoder.source_height * 
                s->encoder.bpp_numerator / s->encoder.bpp_denominator + 7) / 8);

        s->out_buf.allocation_size = bitstream_size;
        s->out_buf.used_size = 0;
        s->out_buf.buffer = (uint8_t *) malloc(s->out_buf.allocation_size);
        if (!s->out_buf.buffer) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize output image buffer\n");
                return false;
        }

        struct video_desc compressed_desc;
        compressed_desc = desc;
        compressed_desc.color_spec = JPEG_XS;
        s->pool.reconfigure(compressed_desc, bitstream_size);
        s->configured = true;

        return true;
}

bool state_video_compress_jpegxs::parse_fmt(char *fmt) {
        char *tok, *save_ptr = NULL;

        while ((tok = strtok_r(fmt, ":", &save_ptr)) != nullptr) {
                if (IS_KEY_PREFIX(tok, "bpp")) {
                        const char *bpp = strchr(tok, '=') + 1;
                        int num = 0, den = 1;
                        if (sscanf(bpp, "%d/%d", &num, &den) == 2 || sscanf(bpp, "%d", &num) == 1) {
                                encoder.bpp_numerator = num;
                                encoder.bpp_denominator = den;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "WARNING: Wrong bpp format: %s\n", tok);
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WARNING: Trailing configuration parameter: %s\n", tok);
                }
                fmt = nullptr;
        }

        return true;
}

void *
jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if (opts && strcmp(opts, "help") == 0) {
                col() << "JPEG XS compression usage:\n";
                return INIT_NOERR;
        }

        s = state_video_compress_jpegxs::create(parent, opts);

        return s;
}

// unpack UYVY to YUV422 planar
static void uyvy_to_yuv422p(const uint8_t *uyvy, int width, int height, svt_jpeg_xs_image_buffer *in_buf) {
        uint8_t *dst_y = (uint8_t *) in_buf->data_yuv[0];
        uint8_t *dst_u = (uint8_t *) in_buf->data_yuv[1];
        uint8_t *dst_v = (uint8_t *) in_buf->data_yuv[2];

        for (int y = 0; y < height; ++y) {
                const uint8_t *src_line = uyvy + y * width * 2;
                uint8_t *dst_y_line = dst_y + y * width;
                uint8_t *dst_u_line = dst_u + y * (width / 2);
                uint8_t *dst_v_line = dst_v + y * (width / 2);

                for (int x = 0; x < width; x += 2) {
                        int i = x * 2;
                        uint8_t u = src_line[i + 0];
                        uint8_t y0 = src_line[i + 1];
                        uint8_t v = src_line[i + 2];
                        uint8_t y1 = src_line[i + 3];

                        dst_y_line[x + 0] = y0;
                        dst_y_line[x + 1] = y1;

                        int chroma_index = x / 2;
                        dst_u_line[chroma_index] = u;
                        dst_v_line[chroma_index] = v;
                }
        }
}

// unpack YUYV to YUV422 planar
static void yuyv_to_yuv422p(const uint8_t *yuyv, int width, int height, svt_jpeg_xs_image_buffer *in_buf) {
        uint8_t *dst_y = (uint8_t *) in_buf->data_yuv[0];
        uint8_t *dst_u = (uint8_t *) in_buf->data_yuv[1];
        uint8_t *dst_v = (uint8_t *) in_buf->data_yuv[2];

        for (int y = 0; y < height; ++y) {
                const uint8_t *src_line = yuyv + y * width * 2;
                uint8_t *dst_y_line = dst_y + y * width;
                uint8_t *dst_u_line = dst_u + y * (width / 2);
                uint8_t *dst_v_line = dst_v + y * (width / 2);

                for (int x = 0; x < width; x += 2) {
                        int i = x * 2;
                        uint8_t y0 = src_line[i + 0];
                        uint8_t u = src_line[i + 1];
                        uint8_t y1 = src_line[i + 2];
                        uint8_t v = src_line[i + 3];

                        dst_y_line[x + 0] = y0;
                        dst_y_line[x + 1] = y1;

                        int chroma_index = x / 2;
                        dst_u_line[chroma_index] = u;
                        dst_v_line[chroma_index] = v;
                }
        }
}

shared_ptr<video_frame> jpegxs_compress(void *state, shared_ptr<video_frame> frame) {
        if (!frame) {
                return {};
        }

        auto *s = (struct state_video_compress_jpegxs *) state;

        if (!s->configured) {
                struct video_desc desc = video_desc_from_frame(frame.get());
                if (!configure_with(s, desc)) {
                        return NULL;
                }
        }

        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        int width = in_tile->width;
        int height = in_tile->height;
        uyvy_to_yuv422p((const uint8_t *) in_tile->data, width, height, &s->in_buf);

        svt_jpeg_xs_frame_t enc_input;
        enc_input.bitstream = s->out_buf;
        enc_input.image = s->in_buf;
        enc_input.user_prv_ctx_ptr = NULL;

        SvtJxsErrorType_t err;
        err = svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to encoder\n");
                return NULL;
        }

        svt_jpeg_xs_frame_t enc_output;
        err = svt_jpeg_xs_encoder_get_packet(&s->encoder, &enc_output, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded packet\n");
                return NULL;
        }

        shared_ptr<video_frame> out_frame = s->pool.get_frame();
        out_frame->color_spec = JPEG_XS;
        out_frame->fps = frame->fps;
        out_frame->interlacing = frame->interlacing;
        out_frame->frame_type = frame->frame_type;
        out_frame->tile_count = 1;
        
        struct tile *out_tile = vf_get_tile(out_frame.get(), 0);
        out_tile->width = frame->tiles[0].width;
        out_tile->height = frame->tiles[0].height;
        size_t enc_size = enc_output.bitstream.used_size;
        if (enc_size > out_tile->data_len) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Encoded frame too big (%zu > %u)\n", enc_size, out_tile->data_len);
                return {};
        }
        
        out_tile->data_len = enc_size;
        memcpy(out_tile->data, enc_output.bitstream.buffer, enc_size);

        return out_frame;
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
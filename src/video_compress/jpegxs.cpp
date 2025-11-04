#include <memory>
#include <iostream>
#include <string>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <svt-jpegxs/SvtJpegxsEnc.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_compress.h"
#include "utils/video_frame_pool.h"
#include "utils/synchronized_queue.h"
#include "jpegxs/jpegxs_conv.h"

#define MOD_NAME "[JPEG XS enc.] "

using std::shared_ptr;
using std::condition_variable;
using std::mutex;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using std::vector;
using std::queue;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
        state_video_compress_jpegxs(struct module *parent, const char *opts);

        ~state_video_compress_jpegxs() {
                if (worker.joinable()) {
                        worker.join();
                }

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

        void (*convert_to_planar)(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer *dst);
        bool parse_fmt(char *fmt);
        void push(shared_ptr<video_frame> in_frame);
        shared_ptr<video_frame> pop();

        synchronized_queue<shared_ptr<struct video_frame>, -1> in_queue;
        synchronized_queue<shared_ptr<struct video_frame>, -1> out_queue;
        thread worker;
};

shared_ptr<video_frame> jpegxs_compress(void *state, shared_ptr<video_frame> frame);
static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc);
static void jpegxs_worker(state_video_compress_jpegxs *s);

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to load JPEG XS default parameters, error code: %x\n", err);
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

        worker = thread(jpegxs_worker, this);
}

static void jpegxs_worker(state_video_compress_jpegxs *s) {
    while (true) {

        auto frame = s->in_queue.pop();

        if (!frame) {
            s->out_queue.push(frame);
            break;
        }

        if (!s->configured) {
                struct video_desc desc = video_desc_from_frame(frame.get());
                if (!configure_with(s, desc)) {
                        break;;
                }
        }

        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        int width = in_tile->width;
        int height = in_tile->height;

        s->convert_to_planar((const uint8_t *) in_tile->data, width, height, &s->in_buf);

        svt_jpeg_xs_frame_t enc_input;
        enc_input.bitstream = s->out_buf;
        enc_input.image = s->in_buf;
        enc_input.user_prv_ctx_ptr = NULL;

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to encoder, error code: %x\n", err);
                continue;
        }

        svt_jpeg_xs_frame_t enc_output;
        err = svt_jpeg_xs_encoder_get_packet(&s->encoder, &enc_output, 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded packet, error code: %x\n", err);
                continue;
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
                continue;
        }
        
        out_tile->data_len = enc_size;
        memcpy(out_tile->data, enc_output.bitstream.buffer, enc_size);

        s->out_queue.push(out_frame);
    }
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
                in_buf->alloc_size[i] = in_buf->stride[i] * (h / (i == 0 ? 1 : h_factor)) * pixel_size;
                in_buf->data_yuv[i] = malloc(in_buf->alloc_size[i]);
                if (!in_buf->data_yuv[i]) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate plane %d\n", i);
                        return false;
                }
        }

        return true;
}

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc)
{
        s->encoder.verbose = VERBOSE_SYSTEM_INFO;
        s->encoder.source_width = desc.width;
        s->encoder.source_height = desc.height;
        s->encoder.colour_format = subsampling_to_jpegxs(get_subsampling(desc.color_spec) / 10);

        const struct uv_to_jpegxs_conversion *conv = get_uv_to_jpegxs_conversion(desc.color_spec);
        if (!conv || !conv->convert) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec: %s\n", get_codec_name(desc.color_spec));
                return false;
        }
        s->convert_to_planar = conv->convert;

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
                        if (strspn(bpp, "0123456789/") != strlen(bpp) && (sscanf(bpp, "%d/%d", &num, &den) == 2 || sscanf(bpp, "%d", &num)) && num > 0 && den > 0) {
                                encoder.bpp_numerator = num;
                                encoder.bpp_denominator = den;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid bits per pixel value '%s' (must be a positive integer or fraction, e.g., 2 or 3/4). Using default 3.\n", tok);
                        }
                } else if (IS_KEY_PREFIX(tok, "decomp_v")) {
                        const int v = atoi(strchr(tok, '=') + 1);
                        if (0 <= v && v <= 2) {
                                encoder.ndecomp_v = v;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid vertical decomposition value '%s' (must be 0, 1 or 2). Using default 2.\n", tok);
                        }
                } else if (IS_KEY_PREFIX(tok, "decomp_h")) {
                        const int h = atoi(strchr(tok, '=') + 1);
                        if (1 <= h && h <= 5) {
                                encoder.ndecomp_h = h;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid horizontal decomposition value '%s' (must be 1, 2, 3, 4 or 5). Using default 5.\n", tok);
                        } 
                } else if (IS_KEY_PREFIX(tok, "quantization")) {
                        const int q = atoi(strchr(tok, '=') + 1);
                        if (q == 0 || q == 1) {
                                encoder.quantization = q;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid quantization method '%s' (must be 0 - deadzone, or 1 - uniform). Using default 0.\n", tok);
                        }
                } else if (IS_KEY_PREFIX(tok, "slice_height")) {
                        const int sh = atoi(strchr(tok, '=') + 1);
                        if (sh > 0 && (sh & ( (1 << encoder.ndecomp_v) - 1 )) == 0) {
                                encoder.slice_height = sh;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid slice height value '%s' (must be a multiple of 2^decomp_v). Using default 16.\n", tok);
                        }
                } else if (IS_KEY_PREFIX(tok, "rc")) {
                        const int rc = atoi(strchr(tok, '=') + 1);
                        if (0 <= rc && rc <= 3) {
                                encoder.rate_control_mode = rc;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid rate control mode '%s' (must be 0 - CBR budget per precinct, 1 - CBR budget per precinct with padding movement, 2 - CBR budget per slice, or 3 - CBR budget per slice with max rate size). Using default 0.\n", tok);
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WARNING: Trailing configuration parameter: %s\n", tok);
                }
                fmt = nullptr;
        }

        return true;
}

static void *jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if (opts && strcmp(opts, "help") == 0) {
                col() << "JPEG XS compression usage:\n";
                return INIT_NOERR;
        }

        s = new state_video_compress_jpegxs(parent, opts);

        return s;
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

        s->convert_to_planar((const uint8_t *) in_tile->data, width, height, &s->in_buf);

        svt_jpeg_xs_frame_t enc_input;
        enc_input.bitstream = s->out_buf;
        enc_input.image = s->in_buf;
        enc_input.user_prv_ctx_ptr = NULL;

        SvtJxsErrorType_t err;
        err = svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to encoder, error code: %x\n", err);
                return NULL;
        }

        svt_jpeg_xs_frame_t enc_output;
        err = svt_jpeg_xs_encoder_get_packet(&s->encoder, &enc_output, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded packet, error code: %x\n", err);
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


void state_video_compress_jpegxs::push(shared_ptr<video_frame> frame)
{
        in_queue.push(frame);
}

shared_ptr<video_frame> state_video_compress_jpegxs::pop()
{
        auto frame = out_queue.pop();

        return frame;
}

static void jpegxs_compress_push(void *state, shared_ptr<video_frame> frame) {
        static_cast<struct state_video_compress_jpegxs *>(state)->push(std::move(frame));
}

static shared_ptr<video_frame> jpegxs_compress_pop(void *state) {
        return static_cast<struct state_video_compress_jpegxs *>(state)->pop();
}

static void jpegxs_compress_done(void *state) {
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
        jpegxs_compress_push, // jpegxs_compress_push (asynchronous)
        jpegxs_compress_pop, //  jpegxs_compress_pop
        NULL,
        NULL,
        get_jpegxs_module_info // get_jpegxs_module_info
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
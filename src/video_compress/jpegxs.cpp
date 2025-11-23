#include <memory>
#include <iostream>
#include <string>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <svt-jpegxs/SvtJpegxsEnc.h>
#include <svt-jpegxs/SvtJpegxsImageBufferTools.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_compress.h"
#include "utils/video_frame_pool.h"
#include "utils/synchronized_queue.h"
#include "jpegxs/jpegxs_conv.h"

#define DEFAULT_POOL_SIZE 5
#define MOD_NAME "[JPEG XS enc.] "

using std::shared_ptr;
using std::condition_variable;
using std::mutex;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using std::vector;
using std::queue;
using std::atomic;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
        state_video_compress_jpegxs(struct module *parent, const char *opts);

        ~state_video_compress_jpegxs() {
                if (worker_send.joinable()) {
                        worker_send.join();
                }
                if (worker_get.joinable()) {
                        worker_get.join();
                }
                if (frame_pool) {
                        svt_jpeg_xs_frame_pool_free(frame_pool);
                }
                svt_jpeg_xs_encoder_close(&encoder);
        }

        svt_jpeg_xs_encoder_api_t encoder;
        svt_jpeg_xs_image_config_t image_config;
        svt_jpeg_xs_frame_pool_t *frame_pool;
        int pool_size = DEFAULT_POOL_SIZE;

        bool configured = 0;
        bool stop = 0;

        video_frame_pool pool;

        void (*convert_to_planar)(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer *dst);
        
        bool parse_fmt(char *fmt);

        synchronized_queue<shared_ptr<struct video_frame>, -1> in_queue;
        synchronized_queue<shared_ptr<struct video_frame>, -1> out_queue;

        thread worker_send;
        thread worker_get;

        mutex mtx;
        condition_variable cv_configured;

        atomic<uint64_t> frames_sent{0};
        atomic<uint64_t> frames_received{0};
};

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc);
static void jpegxs_worker_send(state_video_compress_jpegxs *s);
static void jpegxs_worker_get(state_video_compress_jpegxs *s);

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to load JPEG XS default parameters, error code: %x\n", err);
                throw 1;
        }

        encoder.bpp_numerator = 3;

        if(opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(fmt)) {
                        free(fmt);
                        throw 1;
                }
                free(fmt);
        }

        worker_send = thread(jpegxs_worker_send, this);
        worker_get = thread(jpegxs_worker_get, this); 
}

static void jpegxs_worker_send(state_video_compress_jpegxs *s) {
while (true) {
        auto frame = s->in_queue.pop();

        if (!frame) {
                s->out_queue.push(frame);
        {
                unique_lock<mutex> lock(s->mtx);
                s->stop = true;
        }
                s->cv_configured.notify_one();
                break;
        }

        if (!s->configured) {
                struct video_desc desc = video_desc_from_frame(frame.get());
                if (!configure_with(s, desc)) {
                        break;
                }
        }

        svt_jpeg_xs_frame_t enc_input;
        SvtJxsErrorType_t err = svt_jpeg_xs_frame_pool_get(s->frame_pool, &enc_input, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to get frame from JPEG XS pool, error code: %x\n", err);
                continue;
        }

        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        s->convert_to_planar((const uint8_t *) in_tile->data, in_tile->width, in_tile->height, &enc_input.image);

        err = svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to encoder, error code: %x\n", err);
                continue;
        }

        s->frames_sent++;
}
}

static void jpegxs_worker_get(state_video_compress_jpegxs *s) {
{
        unique_lock<mutex> lock(s->mtx);
        s->cv_configured.wait(lock, [&]{
                return s->configured || s->stop;
        });

        if (!s->configured && s->stop) {
                return;
        }
}
while (true) {
{
        unique_lock<mutex> lock(s->mtx);
        if (s->stop && s->frames_received == s->frames_sent) {
                return;
        }
}

        svt_jpeg_xs_frame_t enc_output;
        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_get_packet(&s->encoder, &enc_output, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded packet, error code: %x\n", err);
                continue;
        }
        s->frames_received++;

        shared_ptr<video_frame> out_frame = s->pool.get_frame();
        
        struct tile *out_tile = vf_get_tile(out_frame.get(), 0);
        size_t enc_size = enc_output.bitstream.used_size;
        if (enc_size > out_tile->data_len) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Encoded frame too big (%zu > %u)\n", enc_size, out_tile->data_len);
                continue;
        }
        
        out_tile->data_len = enc_size;
        memcpy(out_tile->data, enc_output.bitstream.buffer, enc_size);

        s->out_queue.push(out_frame);
        svt_jpeg_xs_frame_pool_release(s->frame_pool, &enc_output);
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

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc)
{
        const struct uv_to_jpegxs_conversion *conv = get_uv_to_jpegxs_conversion(desc.color_spec);
        if (!conv || !conv->convert) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec: %s\n", get_codec_name(desc.color_spec));
                return false;
        }
        s->convert_to_planar = conv->convert;

        s->encoder.verbose = VERBOSE_SYSTEM_INFO;
        s->encoder.source_width = desc.width;
        s->encoder.source_height = desc.height;
        s->encoder.input_bit_depth = get_bits_per_component(desc.color_spec);
        s->encoder.colour_format = subsampling_to_jpegxs(get_subsampling(desc.color_spec) / 10);

        SvtJxsErrorType_t err = SvtJxsErrorNone;
        uint32_t bitstream_size = 0;
        
        err = svt_jpeg_xs_encoder_get_image_config(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->encoder, &s->image_config, &bitstream_size);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get image config from JPEG XS encoder parameters: %x\n", err);
                return false;
        }

        s->frame_pool = svt_jpeg_xs_frame_pool_alloc(&s->image_config, bitstream_size, s->pool_size);
        if (!s->frame_pool) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate JPEG XS frame pool\n");
                return false;
        }

        err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize JPEG XS encoder: %x\n", err);
                return false;
        }

        struct video_desc compressed_desc;
        compressed_desc = desc;
        compressed_desc.color_spec = JPEG_XS;
        s->pool.reconfigure(compressed_desc, bitstream_size); 
{
        unique_lock<mutex> lock(s->mtx);
        s->configured = true;
}
        s->cv_configured.notify_one();

        return true;
}

bool state_video_compress_jpegxs::parse_fmt(char *fmt) {
        char *tok, *save_ptr = NULL;

        while ((tok = strtok_r(fmt, ":", &save_ptr)) != nullptr) {
                if (IS_KEY_PREFIX(tok, "bpp")) {
                        const char *bpp = strchr(tok, '=') + 1;
                        int num = 0, den = 1;
                        if (strspn(bpp, "0123456789/") == strlen(bpp) && (sscanf(bpp, "%d/%d", &num, &den) == 2 || sscanf(bpp, "%d", &num)) && num > 0 && den > 0) {
                                encoder.bpp_numerator = num;
                                encoder.bpp_denominator = den;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid bits per pixel value '%s' (must be a positive integer or fraction, e.g., 2 or 3/4). Using default 3.\n", tok);
                        }
                } else if (IS_KEY_PREFIX(tok, "pool_size")) {
                        const int ps = atoi(strchr(tok, '=') + 1);
                        if (0 <= ps) {
                                pool_size = ps;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid pool size value '%s' (must be a positive integer). Using default 5.\n", tok);
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
                } else if (IS_KEY_PREFIX(tok, "threads")) {
                        const int threads = atoi(strchr(tok, '=') + 1);
                        int max_threads = thread::hardware_concurrency();
                        if (0 <= threads && threads <= max_threads) {
                                encoder.threads_num = threads;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid number of threads '%s' (must be between 0 and %d). Using default 0, which means lowest possible number of threads is created.\n", tok, max_threads);
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WARNING: Trailing configuration parameter: %s\n", tok);
                }
                fmt = nullptr;
        }

        return true;
}

static const struct {
        const char *label;
        const char *key;
        const char *help_name;
        const char *description;
        const char *opt_str;
        bool is_boolean;
        const char *placeholder;
} usage_opts[] = {
        {"Bits per pixel", "bpp", "bpp",
                "\t\tTarget bits-per-pixel ratio for the encoder. May be given as an\n"
                "\t\tinteger (e.g., 2) or as a fraction (e.g., 3/4). Controls the\n"
                "\t\toutput bitrate indirectly. Must be a positive value.\n",
                ":bpp=", false, "3"
        },
        {"Vertical decomposition", "decomp_v", "decomp_v",
                "\t\tNumber of vertical wavelet decompositions. Allowed values are\n"
                "\t\t0, 1, or 2.\n",
                ":decomp_v=", false, "2"
        },
        {"Horizontal decomposition", "decomp_h", "decomp_h",
                "\t\tNumber of horizontal wavelet decompositions. Allowed values\n"
                "\t\tare between 1 and 5.\n",
                ":decomp_h=", false, "5"
        },
        {"Quantization algorithm", "quantization", "quantization",
                "\t\tSelects the quantization algorithm: 0 = deadzone, 1 = uniform.\n",
                ":quantization=", false, "0"
        },
        {"Slice height", "slice_height", "slice_height",
                "\t\tHeight of a slice in lines. Must be a positive integer and a\n"
                "\t\tmultiple of 2^decomp_v.\n",
                ":slice_height=", false, "16"
        },
        {"Rate control mode", "rc", "rc",
                "\t\tRate control mode:\n"
                "\t\t 0 = CBR budget per precinct\n"
                "\t\t 1 = CBR budget per precinct with padding movement\n"
                "\t\t 2 = CBR budget per slice\n"
                "\t\t 3 = CBR budget per slice with max rate size\n",
                ":rc=", false, "0"
        },
        {"Threads scaling parameter", "threads", "threads",
                "\t\tNumber of encoder threads. Must be between 0 and the number of\n"
                "\t\tavailable CPU cores. Value 0 means the lowest possible number\n"
                "\t\tof threads is created by the encoder.\n",
                ":threads=", false, "0"
        },
};

static void *jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if (opts && strcmp(opts, "help") == 0) {
                color_printf(TBOLD("JPEG XS") " compression usage:\n");
                color_printf("\t" TBOLD(
                        TRED("-c jpegxs") "[:bpp=<ratio>][:decomp_v=<0-2>][:decomp_h=<1-5>]"
                                          "[:quantization=<0-1>][:slice_height=<n>][:rc=<mode>]"
                                          "[:threads=<num_threads>][:pool_size=<n>]") "\n");
                color_printf("\t" TBOLD(TRED("-c jpegxs") ":help") "\n");

                color_printf("\nwhere:\n");
                for (const auto &opt : usage_opts) {
                        color_printf("\t" TBOLD("<%s>") "\n%s\n",
                                opt.key, opt.description);
                }
                printf("\n");
                return INIT_NOERR;
        }

        s = new state_video_compress_jpegxs(parent, opts);

        return s;
}

static void jpegxs_compress_push(void *state, shared_ptr<video_frame> frame) {
        static_cast<struct state_video_compress_jpegxs *>(state)->in_queue.push(std::move(frame));
}

static shared_ptr<video_frame> jpegxs_compress_pop(void *state) {
        return static_cast<struct state_video_compress_jpegxs *>(state)->out_queue.pop();
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
        NULL,
        NULL,
        jpegxs_compress_push, // jpegxs_compress_push
        jpegxs_compress_pop, //  jpegxs_compress_pop
        NULL,
        NULL,
        get_jpegxs_module_info // get_jpegxs_module_info
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
/**
 * @file video_decompress/gstreamer_va.cpp
 * @brief GStreamer VA-API HEVC 10-bit 4:4:4 decoder for UltraGrid
 */

#include <arpa/inet.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "video_codec.h"
#include "video_decompress.h"

#define MOD_NAME "[gstreamer-va] "

namespace {

struct state {
        GstElement *pipeline = nullptr;
        GstAppSrc *source = nullptr;
        GstAppSink *sink = nullptr;
        video_desc desc{};
        codec_t out_codec = VIDEO_CODEC_NONE;
        int pitch = 0;
        uint64_t inputs = 0;
        uint64_t outputs = 0;
        uint64_t timeouts = 0;
        uint64_t pull_us = 0;
        uint64_t convert_us = 0;
        std::chrono::steady_clock::time_point stats_since =
            std::chrono::steady_clock::now();
};

std::once_flag gst_init_flag;

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
void bgr10a2_to_r10k_avx2(const uint32_t *src, uint32_t *dst, size_t pixels)
{
        const __m256i mask = _mm256_set1_epi32(0x3ff);
        const __m256i shuffle = _mm256_setr_epi8(
             3,  2,  1,  0,  7,  6,  5,  4,
            11, 10,  9,  8, 15, 14, 13, 12,
             3,  2,  1,  0,  7,  6,  5,  4,
            11, 10,  9,  8, 15, 14, 13, 12);
        size_t i = 0;
        for (; i + 8 <= pixels; i += 8) {
                const __m256i p =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
                const __m256i b = _mm256_and_si256(p, mask);
                const __m256i g =
                    _mm256_and_si256(_mm256_srli_epi32(p, 10), mask);
                const __m256i r =
                    _mm256_and_si256(_mm256_srli_epi32(p, 20), mask);
                __m256i out = _mm256_or_si256(
                    _mm256_slli_epi32(r, 22),
                    _mm256_or_si256(_mm256_slli_epi32(g, 12),
                                    _mm256_slli_epi32(b, 2)));
                out = _mm256_shuffle_epi8(out, shuffle);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), out);
        }
        for (; i < pixels; ++i) {
                const uint32_t p = src[i];
                const uint32_t out =
                    (((p >> 20U) & 0x3ffU) << 22U) |
                    (((p >> 10U) & 0x3ffU) << 12U) |
                    ((p & 0x3ffU) << 2U);
                dst[i] = __builtin_bswap32(out);
        }
}
#endif

void bgr10a2_to_r10k(const uint32_t *src, uint32_t *dst, size_t pixels)
{
#if defined(__x86_64__) || defined(_M_X64)
        if (__builtin_cpu_supports("avx2")) {
                bgr10a2_to_r10k_avx2(src, dst, pixels);
                return;
        }
#endif
        for (size_t i = 0; i < pixels; ++i) {
                const uint32_t p = src[i];
                const uint32_t out =
                    (((p >> 20U) & 0x3ffU) << 22U) |
                    (((p >> 10U) & 0x3ffU) << 12U) |
                    ((p & 0x3ffU) << 2U);
                dst[i] = __builtin_bswap32(out);
        }
}

void log_bus_error(GstElement *pipeline)
{
        GstBus *bus = gst_element_get_bus(pipeline);
        GstMessage *msg = gst_bus_pop_filtered(
            bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
        if (msg != nullptr) {
                GError *err = nullptr;
                gchar *detail = nullptr;
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                        gst_message_parse_error(msg, &err, &detail);
                        MSG(ERROR, "%s (%s)\n", err->message, detail ? detail : "");
                } else {
                        gst_message_parse_warning(msg, &err, &detail);
                        MSG(WARNING, "%s (%s)\n", err->message, detail ? detail : "");
                }
                g_clear_error(&err);
                g_free(detail);
                gst_message_unref(msg);
        }
        gst_object_unref(bus);
}

void destroy_pipeline(state *s)
{
        if (s->pipeline != nullptr) {
                gst_element_set_state(s->pipeline, GST_STATE_NULL);
                gst_object_unref(s->pipeline);
        }
        s->pipeline = nullptr;
        s->source = nullptr;
        s->sink = nullptr;
}

bool create_pipeline(state *s)
{
        destroy_pipeline(s);

        char pipeline_desc[1024];
        snprintf(pipeline_desc, sizeof pipeline_desc,
                 "appsrc name=ugsrc is-live=true format=time block=true "
                 "caps=\"video/x-h265,stream-format=byte-stream,alignment=au,"
                 "width=%u,height=%u,framerate=%d/1000\" ! "
                 "h265parse config-interval=-1 ! "
                 "vah265dec discard-corrupted-frames=true ! "
                 "vapostproc disable-passthrough=true ! "
                 "video/x-raw,format=BGR10A2_LE,width=%u,height=%u ! "
                 "appsink name=ugsink sync=false max-buffers=2 drop=true",
                 s->desc.width, s->desc.height,
                 static_cast<int>(s->desc.fps * 1000.0),
                 s->desc.width, s->desc.height);

        GError *error = nullptr;
        s->pipeline = gst_parse_launch(pipeline_desc, &error);
        if (s->pipeline == nullptr || error != nullptr) {
                MSG(ERROR, "Cannot construct VA pipeline: %s\n",
                    error ? error->message : "unknown error");
                g_clear_error(&error);
                destroy_pipeline(s);
                return false;
        }

        s->source = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(s->pipeline), "ugsrc"));
        s->sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(s->pipeline), "ugsink"));
        if (s->source == nullptr || s->sink == nullptr) {
                MSG(ERROR, "Cannot find appsrc/appsink in VA pipeline\n");
                destroy_pipeline(s);
                return false;
        }
        // The pipeline owns both elements; retain only non-owning pointers.
        gst_object_unref(s->source);
        gst_object_unref(s->sink);

        if (gst_element_set_state(s->pipeline, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
                MSG(ERROR, "Cannot start VA decode pipeline\n");
                log_bus_error(s->pipeline);
                destroy_pipeline(s);
                return false;
        }
        MSG(NOTICE, "Using vah265dec with VA postprocessing to BGR10A2_LE\n");
        return true;
}

void *init()
{
        std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });
        return new state;
}

int reconfigure(void *opaque, video_desc desc, int, int, int, int pitch,
                codec_t out_codec)
{
        auto *s = static_cast<state *>(opaque);
        if (out_codec == VIDEO_CODEC_NONE) {
                s->desc = desc;
                s->out_codec = out_codec;
                return true;
        }
        if (out_codec != R10k || pitch < static_cast<int>(desc.width * 4U)) {
                MSG(WARNING, "Only packed R10k output is supported\n");
                return false;
        }
        s->desc = desc;
        s->out_codec = out_codec;
        s->pitch = pitch;
        return create_pipeline(s);
}

decompress_status decompress(void *opaque, unsigned char *dst,
                             unsigned char *buffer, unsigned int src_len,
                             int frame_seq, video_frame_callbacks *,
                             pixfmt_desc *internal)
{
        auto *s = static_cast<state *>(opaque);
        if (s->out_codec == VIDEO_CODEC_NONE) {
                *internal = {};
                internal->depth = 10;
                internal->subsampling = SUBS_444;
                internal->rgb = true;
                return DECODER_GOT_CODEC;
        }
        if (s->pipeline == nullptr) {
                return DECODER_NO_FRAME;
        }
        ++s->inputs;

        GstBuffer *input = gst_buffer_new_allocate(nullptr, src_len, nullptr);
        gst_buffer_fill(input, 0, buffer, src_len);
        const double fps = s->desc.fps > 0.0 ? s->desc.fps : 24.0;
        GST_BUFFER_PTS(input) =
            gst_util_uint64_scale(frame_seq, GST_SECOND, static_cast<guint64>(fps));
        GST_BUFFER_DURATION(input) =
            gst_util_uint64_scale(1, GST_SECOND, static_cast<guint64>(fps));
        if (gst_app_src_push_buffer(s->source, input) != GST_FLOW_OK) {
                log_bus_error(s->pipeline);
                return DECODER_NO_FRAME;
        }

        const auto pull_start = std::chrono::steady_clock::now();
        GstSample *sample = gst_app_sink_try_pull_sample(s->sink, 100 * GST_MSECOND);
        s->pull_us += std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - pull_start)
                          .count();
        if (sample == nullptr) {
                ++s->timeouts;
                log_bus_error(s->pipeline);
                return DECODER_NO_FRAME;
        }
        ++s->outputs;
        GstBuffer *output = gst_sample_get_buffer(sample);
        GstMapInfo map{};
        if (!gst_buffer_map(output, &map, GST_MAP_READ)) {
                gst_sample_unref(sample);
                return DECODER_NO_FRAME;
        }

        const size_t pixels = static_cast<size_t>(s->desc.width) * s->desc.height;
        if (map.size < pixels * 4U) {
                MSG(ERROR, "Short BGR10A2 frame (%zu, expected at least %zu)\n",
                    map.size, pixels * 4U);
                gst_buffer_unmap(output, &map);
                gst_sample_unref(sample);
                return DECODER_NO_FRAME;
        }
        const auto convert_start = std::chrono::steady_clock::now();
        const auto *src = reinterpret_cast<const uint32_t *>(map.data);
        for (unsigned y = 0; y < s->desc.height; ++y) {
                auto *row = reinterpret_cast<uint32_t *>(dst + y * s->pitch);
                bgr10a2_to_r10k(src + static_cast<size_t>(y) * s->desc.width,
                               row, s->desc.width);
        }
        s->convert_us += std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - convert_start)
                             .count();
        gst_buffer_unmap(output, &map);
        gst_sample_unref(sample);
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - s->stats_since);
        if (elapsed.count() >= 5000) {
                MSG(NOTICE,
                    "stats: inputs=%llu outputs=%llu timeouts=%llu "
                    "avg-pull=%.2fms avg-convert=%.2fms over %.2fs\n",
                    static_cast<unsigned long long>(s->inputs),
                    static_cast<unsigned long long>(s->outputs),
                    static_cast<unsigned long long>(s->timeouts),
                    s->inputs ? s->pull_us / 1000.0 / s->inputs : 0.0,
                    s->outputs ? s->convert_us / 1000.0 / s->outputs : 0.0,
                    elapsed.count() / 1000.0);
                s->inputs = s->outputs = s->timeouts = 0;
                s->pull_us = s->convert_us = 0;
                s->stats_since = now;
        }
        return DECODER_GOT_FRAME;
}

int get_property(void *, int property, void *val, size_t *len)
{
        if (property == DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME &&
            *len >= sizeof(int)) {
                *static_cast<int *>(val) = false;
                *len = sizeof(int);
                return true;
        }
        return false;
}

void done(void *opaque)
{
        auto *s = static_cast<state *>(opaque);
        destroy_pipeline(s);
        delete s;
}

int priority(codec_t compression, pixfmt_desc, codec_t output)
{
        if (compression != H265) {
                return VDEC_PRIO_NA;
        }
        if (output == VC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }
        return output == R10k ? VDEC_PRIO_PREFERRED : VDEC_PRIO_NA;
}

const video_decompress_info info = {
    init, reconfigure, decompress, get_property, done, priority,
};

} // namespace

REGISTER_MODULE(gstreamer_va, &info, LIBRARY_CLASS_VIDEO_DECOMPRESS,
                VIDEO_DECOMPRESS_ABI_VERSION);

/**
 * @file video_compress/gstreamer_va.cpp
 * @brief Intel VA-API HEVC 10-bit 4:4:4 encoder through GStreamer
 */

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <cstring>
#include <condition_variable>
#include <atomic>
#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "video_codec.h"
#include "video_compress.h"
#include "video_frame.h"

#define MOD_NAME "[gstreamer-va-enc] "

namespace {

struct state {
        GstElement *pipeline = nullptr;
        GstAppSrc *source = nullptr;
        GstAppSink *sink = nullptr;
        video_desc input_desc{};
        int bitrate_kbps = 60000;
        int gop = 24;
        uint64_t frame_number = 0;
        std::mutex mutex;
        std::condition_variable ready;
        bool eos = false;
        std::atomic<uint64_t> pushed{0};
        std::atomic<uint64_t> popped{0};
        std::atomic<uint64_t> encoded_bytes{0};
        std::chrono::steady_clock::time_point stats_since =
            std::chrono::steady_clock::now();
        std::deque<std::array<char, VF_METADATA_SIZE>> metadata;
};

std::once_flag gst_init_flag;

void destroy_pipeline(state *s)
{
        if (s->pipeline) {
                gst_element_set_state(s->pipeline, GST_STATE_NULL);
                gst_object_unref(s->pipeline);
        }
        s->pipeline = nullptr;
        s->source = nullptr;
        s->sink = nullptr;
}

void log_bus_error(state *s)
{
        GstBus *bus = gst_element_get_bus(s->pipeline);
        GstMessage *msg = gst_bus_pop_filtered(
            bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
        if (msg) {
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

bool create_pipeline(state *s, const video_desc &desc)
{
        destroy_pipeline(s);
        char spec[1600];
        const int fps_num = static_cast<int>(desc.fps * 1000.0 + 0.5);
        snprintf(
            spec, sizeof spec,
            "appsrc name=ugsrc is-live=true format=time block=true "
            "caps=\"video/x-decklink-r12l,width=%u,height=%u,"
            "row-stride=%u,framerate=%d/1000,interlace-mode=progressive,"
            "color-range=full\" ! "
            "r12lconvert ! "
            "video/x-raw,format=BGR10A2_LE,width=%u,height=%u ! "
            "vapostproc disable-passthrough=true ! "
            "video/x-raw(memory:VAMemory),format=Y410,width=%u,height=%u ! "
            "vah265lpenc bitrate=%d key-int-max=%d b-frames=0 ref-frames=1 "
            "target-usage=7 rate-control=cbr aud=true ! "
            "h265parse config-interval=-1 ! "
            "video/x-h265,stream-format=byte-stream,alignment=au ! "
            "appsink name=ugsink sync=false max-buffers=2 drop=false",
            desc.width, desc.height, vc_get_linesize(desc.width, R12L), fps_num,
            desc.width, desc.height, desc.width, desc.height,
            s->bitrate_kbps, s->gop);
        GError *error = nullptr;
        s->pipeline = gst_parse_launch(spec, &error);
        if (!s->pipeline || error) {
                MSG(ERROR, "Cannot construct encoder pipeline: %s\n",
                    error ? error->message : "unknown error");
                g_clear_error(&error);
                destroy_pipeline(s);
                return false;
        }
        s->source = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(s->pipeline), "ugsrc"));
        s->sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(s->pipeline), "ugsink"));
        if (!s->source || !s->sink) {
                destroy_pipeline(s);
                return false;
        }
        gst_object_unref(s->source);
        gst_object_unref(s->sink);
        if (gst_element_set_state(s->pipeline, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
                log_bus_error(s);
                destroy_pipeline(s);
                return false;
        }
        s->input_desc = desc;
        s->frame_number = 0;
        MSG(NOTICE, "Using r12lconvert + vapostproc + vah265lpenc at %d kb/s\n",
            s->bitrate_kbps);
        return true;
}

void *init(module *, const char *cfg)
{
        std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });
        auto *s = new state;
        if (cfg && *cfg) {
                std::string options(cfg);
                size_t pos = options.find("bitrate=");
                if (pos != std::string::npos) {
                        s->bitrate_kbps = std::stoi(options.substr(pos + 8)) / 1000;
                }
                pos = options.find("gop=");
                if (pos != std::string::npos) {
                        s->gop = std::stoi(options.substr(pos + 4));
                }
        }
        return s;
}

void push(void *opaque, std::shared_ptr<video_frame> input)
{
        auto *s = static_cast<state *>(opaque);
        if (!input) {
                std::lock_guard<std::mutex> lock(s->mutex);
                s->eos = true;
                if (s->source) {
                        gst_app_src_end_of_stream(s->source);
                }
                s->ready.notify_all();
                return;
        }
        const video_desc desc = video_desc_from_frame(input.get());
        if (desc.color_spec != R12L || !video_desc_eq(desc, s->input_desc)) {
                std::lock_guard<std::mutex> lock(s->mutex);
                if (desc.color_spec != R12L || !create_pipeline(s, desc)) {
                        return;
                }
                s->ready.notify_all();
        }
        const tile *in = vf_get_tile(input.get(), 0);
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, in->data_len, nullptr);
        gst_buffer_fill(buffer, 0, in->data, in->data_len);
        const int fps_num = static_cast<int>(desc.fps * 1000.0 + 0.5);
        GST_BUFFER_PTS(buffer) =
            gst_util_uint64_scale(s->frame_number, 1000ULL * GST_SECOND, fps_num);
        GST_BUFFER_DURATION(buffer) =
            gst_util_uint64_scale(1, 1000ULL * GST_SECOND, fps_num);
        ++s->frame_number;
        ++s->pushed;
        {
                std::lock_guard<std::mutex> lock(s->mutex);
                s->metadata.emplace_back();
                vf_store_metadata(input.get(), s->metadata.back().data());
        }
        if (gst_app_src_push_buffer(s->source, buffer) != GST_FLOW_OK) {
                log_bus_error(s);
        }
}

std::shared_ptr<video_frame> pop(void *opaque)
{
        auto *s = static_cast<state *>(opaque);
        {
                std::unique_lock<std::mutex> lock(s->mutex);
                s->ready.wait(lock, [s] { return s->pipeline != nullptr || s->eos; });
                if (!s->pipeline && s->eos) {
                        return {};
                }
        }
        GstSample *sample =
            gst_app_sink_try_pull_sample(s->sink, GST_SECOND);
        if (!sample) {
                if (s->eos && gst_app_sink_is_eos(s->sink)) {
                        return {};
                }
                log_bus_error(s);
                return vcomp_pop_retry;
        }
        GstBuffer *encoded = gst_sample_get_buffer(sample);
        GstMapInfo map{};
        if (!gst_buffer_map(encoded, &map, GST_MAP_READ)) {
                gst_sample_unref(sample);
                return {};
        }
        video_desc out_desc = s->input_desc;
        out_desc.color_spec = H265;
        auto output = std::shared_ptr<video_frame>(
            vf_alloc_desc(out_desc), [](video_frame *f) {
                    free(f->tiles[0].data);
                    vf_free(f);
            });
        output->tiles[0].data_len = map.size;
        output->tiles[0].data = static_cast<char *>(malloc(map.size));
        memcpy(output->tiles[0].data, map.data, map.size);
        {
                std::lock_guard<std::mutex> lock(s->mutex);
                if (!s->metadata.empty()) {
                        vf_restore_metadata(output.get(), s->metadata.front().data());
                        s->metadata.pop_front();
                }
        }
        gst_buffer_unmap(encoded, &map);
        gst_sample_unref(sample);
        ++s->popped;
        s->encoded_bytes += map.size;
        const auto now = std::chrono::steady_clock::now();
        if (now - s->stats_since >= std::chrono::seconds(5)) {
                const double seconds =
                    std::chrono::duration<double>(now - s->stats_since).count();
                const uint64_t bytes = s->encoded_bytes.exchange(0);
                MSG(NOTICE,
                    "stats: pushed=%llu encoded=%llu bitrate=%.2fMb/s over %.2fs\n",
                    static_cast<unsigned long long>(s->pushed.exchange(0)),
                    static_cast<unsigned long long>(s->popped.exchange(0)),
                    bytes * 8.0 / seconds / 1000000.0, seconds);
                s->stats_since = now;
        }
        return output;
}

void done(void *opaque)
{
        auto *s = static_cast<state *>(opaque);
        destroy_pipeline(s);
        delete s;
}

compress_module_info module_info()
{
        return {"GStreamer Intel VA HEVC 4:4:4",
                {{"Bitrate", "Target bitrate", "60000000", "bitrate",
                  ":bitrate=", false}},
                {{"H.265", {{"Intel VA low-power 4:4:4", ""}}, 100}}};
}

const video_compress_info info = {
    init, done, nullptr, nullptr, push, pop, nullptr, nullptr, module_info,
};

} // namespace

REGISTER_MODULE(gstreamer_va, &info, LIBRARY_CLASS_VIDEO_COMPRESS,
                VIDEO_COMPRESS_ABI_VERSION);

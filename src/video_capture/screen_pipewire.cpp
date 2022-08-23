#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>
#include <chrono>

#include <pipewire/pipewire.h>
#include <gio/gio.h>
#include <gio/gunixfdlist.h>
#include <spa/utils/result.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/props.h>
#include <spa/debug/format.h>

#include "utils/synchronized_queue.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"
#include "concurrent_queue/readerwritercircularbuffer.h"
#include "concurrent_queue/readerwriterqueue.h"

#define MAX_BUFFERS 2
static constexpr int SENDING_FRAMES_QUEUE_SIZE = 4;
static constexpr int BLANK_FRAMES_QUEUE_SIZE = 6;
static constexpr int BLANK_FRAMES_COUNT = 6;
static_assert(BLANK_FRAMES_COUNT <= BLANK_FRAMES_QUEUE_SIZE);


struct request_path_t {
        std::string token;
        std::string path;

        static request_path_t create(const std::string &name) {
                ++token_counter;

                auto token = std::string("m") + std::to_string(token_counter);
                request_path_t result = {
                        .token = token,
                        .path = std::string("/org/freedesktop/portal/desktop/request/") + name + "/" + token
                };

                std::cout << "new request: '" << result.path << "'" << std::endl;
                return result;
        }

private:
        static unsigned int token_counter;
};

unsigned int request_path_t::token_counter = 0;

struct session_path_t {
        std::string token;
        std::string path;

        static session_path_t create(const std::string &name) {
                ++token_counter;

                auto token = std::string("m") + std::to_string(token_counter);
                return {
                        .token = token,
                        .path = std::string("/org/freedesktop/portal/desktop/session/") + name + "/" + token
                };
        }

private:
        static unsigned int token_counter;
};

unsigned int session_path_t::token_counter = 0;


using PortalCallCallback = std::function<void(GVariant *parameters)>;

static void portal_call(GDBusConnection *connection, GDBusProxy *screencast_proxy, const char *object_path,
                                 const char *method_name,
                                 std::initializer_list<std::pair<const char *, GVariant * >> options,
                                 PortalCallCallback on_response) {
        assert(screencast_proxy != nullptr);
        assert(method_name != nullptr);

        std::string sender_name = g_dbus_connection_get_unique_name(connection) + 1;
        std::replace(sender_name.begin(), sender_name.end(), '.', '_');
        request_path_t request_path = request_path_t::create(sender_name);


        auto callback = [](GDBusConnection *connection, const gchar *sender_name, const gchar *object_path,
                                           const gchar *interface_name, const gchar *signal_name, GVariant *parameters,
                                           gpointer user_data) {
                (void) sender_name;
                (void) interface_name;
                (void) signal_name;
                
                static_cast<PortalCallCallback *> (user_data)->operator()(parameters);
                
                //TODO: check if this actually works
                g_dbus_connection_call(
                                connection, "org.freedesktop.portal.Desktop",
                                object_path, "org.freedesktop.portal.Request", "Close",
                                nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, nullptr, nullptr);
        };

        g_dbus_connection_signal_subscribe(connection, "org.freedesktop.portal.Desktop",
                                                                           "org.freedesktop.portal.Request",
                                                                           "Response",
                                                                           request_path.path.c_str(),
                                                                           nullptr,
                                                                           G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE,
                                                                           callback,
                                                                           new PortalCallCallback{std::move(on_response)},
                                                                           [](gpointer user_data) { delete static_cast< PortalCallCallback * >(user_data); });

        auto call_finished = [](GObject *source_object, GAsyncResult *result, gpointer user_data) {
                (void) user_data;
                GError *error = nullptr;
                GVariant *result_finished = g_dbus_proxy_call_finish(G_DBUS_PROXY(source_object), result, &error);
                g_assert_no_error(error);
                const char *path = nullptr;
                g_variant_get(result_finished, "(o)", &path);
                g_variant_unref(result_finished);
                std::cout << "call finished: " << path << std::endl;
        };

        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);

        for (auto &[key, value] : options) {
                g_variant_builder_add(&builder, "{sv}", key, value);
        }

        g_variant_builder_add(&builder, "{sv}", "handle_token", g_variant_new_string(request_path.token.c_str()));

        GVariant *args = nullptr;
        if (object_path == nullptr) {
                args = g_variant_new("(a{sv})", &builder);
        } else if (strcmp(method_name, "Start") == 0) { // FIXME: horrible hack
                assert(g_variant_is_object_path(object_path));
                args = g_variant_new("(osa{sv})", object_path, "", &builder);
        } else {
                assert(g_variant_is_object_path(object_path));
                args = g_variant_new("(oa{sv})", object_path, &builder);
        }
        g_variant_builder_clear(&builder);
        g_dbus_proxy_call(screencast_proxy, method_name, args, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, call_finished, screencast_proxy);
}

struct screen_cast_session {
        GMainLoop *dbus_loop = nullptr;

        int pipewire_fd = -1;
        uint32_t pipewire_node = -1;
        
        struct pw_thread_loop *loop = nullptr;
        struct pw_core *core = nullptr;

        struct pw_stream *stream = nullptr;
        struct spa_hook stream_listener = {};
        struct spa_hook core_listener = {};

        struct spa_io_position *position = nullptr;

        struct spa_video_info format = {};
        //int32_t output_line_size = 0;
        struct spa_rectangle size = {};

        char padding1[1000];
        // used exlusively by ultragrid thread
        struct video_frame *in_flight_frame = nullptr;
        char padding2[1000];

        // used exclusively by pipewire thread
        struct video_frame *dequed_blank_frame = nullptr;
        char padding3[1000];

        moodycamel::BlockingReaderWriterCircularBuffer<video_frame*> blank_frames {BLANK_FRAMES_QUEUE_SIZE};
        moodycamel::BlockingReaderWriterCircularBuffer<video_frame*> sending_frames {SENDING_FRAMES_QUEUE_SIZE};
        
        std::promise<void> screen_cast_is_ready;

        struct video_frame *new_blank_frame()
        {
                struct video_frame *frame = vf_alloc(1);
                frame->color_spec = RGBA;
                frame->interlacing = PROGRESSIVE;
                frame->fps = 60;
                frame->callbacks.data_deleter = vf_data_deleter;
                
                struct tile* tile = vf_get_tile(frame, 0);
                assert(tile != nullptr);
                tile->width = size.width; //TODO
                tile->height = size.height; //TODO

                tile->data_len = vc_get_linesize(tile->width, frame->color_spec) * tile->height;
                tile->data = (char *) malloc(tile->data_len);
                return frame;
        }

        ~screen_cast_session() {
                pw_thread_loop_stop(loop);
                g_main_loop_quit(dbus_loop);
                pw_stream_disconnect(stream);
                //pw_thread_loop_destroy(loop);
                //pw_core_disconnect(core);
                std::cout<<"screen_cast_session destructor finished"<<std::endl;
        }
};

static void on_stream_state_changed(void *session_ptr, enum pw_stream_state old, enum pw_stream_state state, const char *error) {
        (void) session_ptr;

        printf("stream state change : \"%s\" -> \"%s\"\n", pw_stream_state_as_string(old), pw_stream_state_as_string(state));
        printf("error: %s\n", error ? error : "(nullptr)");
        switch (state) {
                case PW_STREAM_STATE_UNCONNECTED:
                        //pw_thread_loop_stop(data->loop);
                        break;
                case PW_STREAM_STATE_PAUSED:
                        //pw_main_loop_quit(data->loop);
                        //pw_stream_set_active(data->stream, true);
                        break;
                default:
                        break;
        }
}

static void on_stream_io_changed(void *_data, uint32_t id, void *area, uint32_t size) {
        auto *data = static_cast<screen_cast_session*>(_data);
        printf("stream changed: id=%d size=%d\n", id, size);
        switch (id) {
                case SPA_IO_Position:
                        data->position = static_cast<spa_io_position *>(area);
                        break;
        }
}

static void on_stream_param_changed(void *session_ptr, uint32_t id, const struct spa_pod *param) {
        auto &session = *static_cast<screen_cast_session*>(session_ptr);
        (void) id;
        std::cout << "[cap_pipewire] param changed:\n" << std::endl;
        spa_debug_format(2, nullptr, param);

        // from example code, not sure what this is
        if (param == NULL || id != SPA_PARAM_Format)
                return;

        int parse_format_ret = spa_format_parse(param, &session.format.media_type, &session.format.media_subtype);
        assert(parse_format_ret > 0);

        assert(session.format.media_type == SPA_MEDIA_TYPE_video);
        assert(session.format.media_subtype == SPA_MEDIA_SUBTYPE_raw);

        spa_format_video_raw_parse(param, &session.format.info.raw);
        session.size = SPA_RECTANGLE(session.format.info.raw.size.width,
                                                                session.format.info.raw.size.height);
        std::cout << "size: " << session.format.info.raw.size.width << " x " << session.format.info.raw.size.height
                                << std::endl;
        int mult = 1;

        int linesize = vc_get_linesize(session.size.width, RGBA);
        int32_t size = linesize * session.size.height;

        uint8_t params_buffer[1024];

        struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));
        const struct spa_pod *params[5];

        params[0] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                SPA_PARAM_BUFFERS_buffers,
                SPA_POD_CHOICE_RANGE_Int(8, 2, MAX_BUFFERS),
                SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
                SPA_PARAM_BUFFERS_size, SPA_POD_Int(size * mult),
                SPA_PARAM_BUFFERS_stride,
                SPA_POD_Int(linesize * mult),
                SPA_PARAM_BUFFERS_align, SPA_POD_Int(16),
                SPA_PARAM_BUFFERS_dataType,
                SPA_POD_CHOICE_FLAGS_Int((1 << SPA_DATA_MemPtr)))
        );

        /* a header metadata with timing information */
        params[1] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
                SPA_PARAM_META_size,
                SPA_POD_Int(sizeof(struct spa_meta_header)))
        );
        /* video cropping information */
        params[2] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_VideoCrop),
                SPA_PARAM_META_size,
                SPA_POD_Int(sizeof(struct spa_meta_region)))
        );
        
        #define CURSOR_META_SIZE(w, h)   (sizeof(struct spa_meta_cursor) + sizeof(struct spa_meta_bitmap) + (w) * (h) * 4)
        /* cursor information */
        params[3] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Cursor),
                SPA_PARAM_META_size, SPA_POD_CHOICE_RANGE_Int(
                        CURSOR_META_SIZE(64, 64),
                        CURSOR_META_SIZE(1, 1),
                        CURSOR_META_SIZE(256, 256))
                )
        );

        pw_stream_update_params(session.stream, params, 4);

        for(int i = 0; i < BLANK_FRAMES_COUNT; ++i)
                session.blank_frames.wait_enqueue(session.new_blank_frame());
        
        session.screen_cast_is_ready.set_value();
}

static void copy_bgra_to_rgba(char *dest, char *src, int width, int height) {
        int linesize = 4*width;
        for (int line_offset = 0; line_offset < height * linesize; line_offset += linesize) {
                for(int x = 0; x < 4*width; x += 4) {
                        // rgba <- bgra
                        dest[line_offset + x    ] = src[line_offset + x + 2];
                        dest[line_offset + x + 1] = src[line_offset + x + 1];
                        dest[line_offset + x + 2] = src[line_offset + x    ];
                        dest[line_offset + x + 3] = src[line_offset + x + 3];
                }
        }
}

static void on_process(void *session_ptr) {
        screen_cast_session &session = *static_cast<screen_cast_session*>(session_ptr);
        //std::cout<<"on process"<<std::endl;
        static int frame_count = 0;
        static uint64_t begin_time = time_since_epoch_in_ms();


        pw_buffer *buffer;
        int n_buffers_from_pw = 0;
        while((buffer = pw_stream_dequeue_buffer(session.stream)) != nullptr){    
                ++n_buffers_from_pw;

                
                if(session.dequed_blank_frame == nullptr && !session.blank_frames.try_dequeue(session.dequed_blank_frame))
                {
                        //std::cout << "dropping - no blank frame" << std::endl;
                        pw_stream_queue_buffer(session.stream, buffer);
                        continue;
                }
                        

                if(buffer == nullptr){
                        std::cout<<"pipewire is out of buffers"<<std::endl;
                        return;
                }

                assert(buffer->buffer != nullptr);
                assert(buffer->buffer->datas != nullptr);
                assert(buffer->buffer->datas[0].data != nullptr);
                //memcpy(session.dequed_blank_frame->tiles[0].data, static_cast<char*>(buffer->buffer->datas[0].data), session.size.height * vc_get_linesize(session.size.width, RGBA));
                copy_bgra_to_rgba(session.dequed_blank_frame->tiles[0].data, static_cast<char*>(buffer->buffer->datas[0].data), session.size.width, session.size.height);
                
                session.sending_frames.wait_enqueue(session.dequed_blank_frame);
                
                session.dequed_blank_frame = nullptr;
                pw_stream_queue_buffer(session.stream, buffer);
                
                ++frame_count;
                uint64_t time_now = time_since_epoch_in_ms();

                uint64_t delta = time_now - begin_time;
                if(delta >= 5000) {
                        std::cout<<"on process: average fps in last 5 seconds: " <<  frame_count / (delta / 1000.0) << std::endl;
                        frame_count = 0;
                        begin_time = time_since_epoch_in_ms();
                }
        }

        static uint8_t counter = 0;
        if( (++counter)%40 == 0)
                std::cout<<"from pw: "<< n_buffers_from_pw << "\t sending: "<<session.sending_frames.size_approx() << "\t blank: " << session.blank_frames.size_approx() << std::endl;
}

static void on_drained(void*)
{
        std::cout<<"drained\n"<<std::endl;
}

static void on_add_buffer(void *session_ptr, struct pw_buffer *)
{
        (void) session_ptr;

        std::cout<<"add_buffer\n"<<std::endl;
}

static void on_remove_buffer(void *session_ptr, struct pw_buffer *)
{
        (void) session_ptr;
        std::cout<<"remove_buffer\n"<<std::endl;
}

static const struct pw_stream_events stream_events = {
                PW_VERSION_STREAM_EVENTS,
                .destroy = nullptr,
                .state_changed = on_stream_state_changed,
                .control_info = nullptr,
                .io_changed = on_stream_io_changed,
                .param_changed = on_stream_param_changed,
                .add_buffer = on_add_buffer,
                .remove_buffer = on_remove_buffer,
                .process = on_process,
                .drained = on_drained,
                .command = nullptr,
                .trigger_done = nullptr,
};

static void on_core_error_cb(void *session_ptr, uint32_t id, int seq, int res,
                                                         const char *message) {
        (void) session_ptr;
        printf("[on_core_error_cb] Error id:%u seq:%d res:%d (%s): %s", id,
                   seq, res, strerror(res), message);
}

static void on_core_done_cb(void *session_ptr, uint32_t id, int seq) {
        (void) session_ptr;
        printf("[on_core_done_cb] id=%d seq=%d", id, seq);
}

static const struct pw_core_events core_events = {
                PW_VERSION_CORE_EVENTS,
                .info = nullptr,
                .done = on_core_done_cb,
                .ping = nullptr,
                .error = on_core_error_cb,
                .remove_id = nullptr,
                .bound_id = nullptr,
                .add_mem = nullptr,
                .remove_mem = nullptr
};

static int start_pipewire(screen_cast_session &session)
{
        const struct spa_pod *params[2] = {};
        uint8_t params_buffer[1024];
        struct spa_pod_builder pod_builder = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));

        session.loop = pw_thread_loop_new("pipewire_thread_loop", nullptr);
        assert(session.loop != nullptr);
        pw_thread_loop_lock(session.loop);
        pw_context *context = pw_context_new(pw_thread_loop_get_loop(session.loop), nullptr, 0);
        assert(context != nullptr);

        if (pw_thread_loop_start(session.loop) != 0) {
                assert(false && "error starting pipewire thread loop");
        }


        int new_pipewire_fd = fcntl(session.pipewire_fd, F_DUPFD_CLOEXEC, 5);
        std::cout << "duplicating fd " << session.pipewire_fd << " -> " << new_pipewire_fd << std::endl;
        pw_core *core = pw_context_connect_fd(context, new_pipewire_fd, nullptr,
                                                                                  0); //why does obs dup the fd?
        assert(core != nullptr);

        pw_core_add_listener(core, &session.core_listener, &core_events, &session);

        session.stream = pw_stream_new(core, "my_screencast", pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Video",
                        PW_KEY_MEDIA_CATEGORY, "Capture",
                        PW_KEY_MEDIA_ROLE, "Screen",
                        nullptr));
        assert(session.stream != nullptr);
        pw_stream_add_listener(session.stream, &session.stream_listener, &stream_events, &session);

        auto size_rect_def = SPA_RECTANGLE(1920, 1080);
        auto size_rect_min = SPA_RECTANGLE(1, 1);
        auto size_rect_max = SPA_RECTANGLE(3840, 2160);

        auto framerate_def = SPA_FRACTION(25, 1);
        auto framerate_min = SPA_FRACTION(0, 1);
        auto framerate_max = SPA_FRACTION(60, 1);


        const int n_params = 1;
        params[0] = static_cast<spa_pod *> (spa_pod_builder_add_object(
                        &pod_builder, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
                        SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
                        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
                        SPA_FORMAT_VIDEO_format,
                        SPA_POD_CHOICE_ENUM_Id(4, 
                                SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_BGRx,
			        SPA_VIDEO_FORMAT_RGBA, SPA_VIDEO_FORMAT_RGBx),
                        SPA_FORMAT_VIDEO_size,
                        SPA_POD_CHOICE_RANGE_Rectangle(
                                        &size_rect_def,
                                        &size_rect_min,
                                        &size_rect_max),
                        SPA_FORMAT_VIDEO_framerate,
                        SPA_POD_CHOICE_RANGE_Fraction(
                                        &framerate_def,
                                        &framerate_min,
                                        &framerate_max)
        ));

        int res = pw_stream_connect(session.stream,
                                        PW_DIRECTION_INPUT,
                                        session.pipewire_node,
                                        static_cast<pw_stream_flags>(
                                                PW_STREAM_FLAG_AUTOCONNECT |
                                                PW_STREAM_FLAG_MAP_BUFFERS
                                        ),
                                        params, n_params);
        if (res < 0) {
                fprintf(stderr, "can't connect: %s\n", spa_strerror(res));
                return -1;
        }
        
        std::cout<<"thread loop unlock"<<std::endl;
        pw_thread_loop_unlock(session.loop);
        std::cout<<"thread loop unlocked"<<std::endl;
        return 0;
}

static void run_screencast(screen_cast_session *session_ptr) {
        auto& session = *session_ptr;

        session.pipewire_fd = -1;
        session.pipewire_node = UINT32_MAX;
        session.dbus_loop = g_main_loop_new(nullptr, false);

        GError *error = nullptr;

        GDBusConnection *connection = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error);
        g_assert_no_error(error);
        assert(connection != nullptr);

        std::string sender_name = g_dbus_connection_get_unique_name(connection) + 1;
        std::replace(sender_name.begin(), sender_name.end(), '.', '_');
        GDBusProxy *screencast_proxy = g_dbus_proxy_new_sync(
                        connection, G_DBUS_PROXY_FLAGS_NONE, nullptr,
                        "org.freedesktop.portal.Desktop",
                        "/org/freedesktop/portal/desktop",
                        "org.freedesktop.portal.ScreenCast", nullptr, &error);
        g_assert_no_error(error);
        assert(screencast_proxy != nullptr);

        session_path_t session_path = session_path_t::create(sender_name);
        std::cout << "session path: '" << session_path.path << "'" << "token: '" << session_path.token << "'\n";

        auto pipewire_opened = [](GObject *source, GAsyncResult *res, void *user_data) {
                auto session = static_cast<screen_cast_session*>(user_data);
                GError *error = nullptr;
                GUnixFDList *fd_list = nullptr;

                GVariant *result = g_dbus_proxy_call_with_unix_fd_list_finish(G_DBUS_PROXY(source), &fd_list, res, &error);
                g_assert_no_error(error);

                gint32 handle;
                g_variant_get(result, "(h)", &handle);
                assert(handle == 0); //it should always be the first index

                session->pipewire_fd = g_unix_fd_list_get(fd_list, handle, &error);
                g_assert_no_error(error);

                assert(session->pipewire_fd != -1);
                assert(session->pipewire_node != UINT32_MAX);
                
                std::cout<<"starting pipewire"<<std::endl;
                start_pipewire(*session);
        };

        auto started = [&](GVariant *parameters) {
                std::cout << "started: " << g_variant_print(parameters, true) << std::endl;
                GVariant *result;
                uint32_t response;
                g_variant_get(parameters, "(u@a{sv})", &response, &result);
                assert(response == 0 && "failed to start");
                GVariant *streams = g_variant_lookup_value(result, "streams", G_VARIANT_TYPE_ARRAY);
                GVariant *stream_properties;
                GVariantIter iter;
                g_variant_iter_init(&iter, streams);
                assert(g_variant_iter_n_children(&iter) == 1);
                bool got_item = g_variant_iter_loop(&iter, "(u@a{sv})", &session.pipewire_node, &stream_properties);
                assert(got_item);
                g_variant_unref(stream_properties);
                g_variant_unref(result);
                GVariantBuilder builder;
                g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
                g_dbus_proxy_call_with_unix_fd_list(screencast_proxy, "OpenPipeWireRemote",
                                                                                        g_variant_new("(oa{sv})", session_path.path.c_str(), &builder),
                                                                                        G_DBUS_CALL_FLAGS_NONE, -1,
                                                                                        nullptr, nullptr, pipewire_opened, &session);
                g_variant_builder_clear(&builder);
        };

        auto sources_selected = [&](GVariant *parameters) {
                gchar *pretty = g_variant_print(parameters, true);
                std::cout << "selected sources: " << pretty << std::endl;
                g_free((gpointer) pretty);

                uint32_t result;
                GVariant *response;
                g_variant_get(parameters, "(u@a{sv})", &result, &response);
                assert(result == 0 && "Failed to select sources");
                g_variant_unref(response);
                portal_call(connection, screencast_proxy, session_path.path.c_str(), "Start", {}, started);
        };

        auto session_created = [&](GVariant *parameters) {
                guint32 response;
                const char *session_handle;
                
                GVariant *result;
                g_variant_get(parameters, "(u@a{sv})", &response, &result);
                assert(response == 0 && "Failed to create session");
                g_variant_lookup(result, "session_handle", "s", &session_handle);
                g_variant_unref(result);

                std::cout << "session created with handle: " << session_handle << std::endl;
                assert(session_path.path == session_handle);

                portal_call(connection, screencast_proxy, session_handle, "SelectSources",
                                        {
                                                        {"types",    g_variant_new_uint32(3)}, // 1 full screen, 2 - a window, 3 - both
                                                        {"multiple", g_variant_new_boolean(false)}
                                        },
                                        sources_selected
                );
        };

        portal_call(connection, screencast_proxy, nullptr, "CreateSession",
                                {
                                                {"session_handle_token", g_variant_new_string(session_path.token.c_str())}
                                },
                                session_created
        );

        g_dbus_connection_flush(connection, nullptr, nullptr, nullptr);
        std::cout << "running dbus loop" << std::endl;
        g_main_loop_run(session.dbus_loop);
        g_object_unref(screencast_proxy);
        g_object_unref(connection);
        //g_main_loop_unref(session.dbus_loop); //FIXME
        std::cout << "finished dbus loop " << std::endl;
}

static struct vidcap_type * vidcap_screen_pipewire_probe(bool verbose, void (**deleter)(void *))
{
        (void) verbose;
        (void) deleter;
        
        std::cout<<"[cap_pipewire] probe\n";
        exit(0);
        return nullptr;
}


static int vidcap_screen_pipewire_init(struct vidcap_params *params, void **state)
{
        (void) params;
        
        screen_cast_session *session = new screen_cast_session();
        *state = session;
        
        std::cout<<"[cap_pipewire] init\n";
        pw_init(&uv_argc, &uv_argv);

        std::future<void> ready = session->screen_cast_is_ready.get_future();
        std::thread dbus_thread(run_screencast, session);
        ready.wait();
        dbus_thread.detach();
        std::cout<<"ready"<<std::endl;
        return VIDCAP_INIT_OK;
}

static void vidcap_screen_pipewire_done(void *session_ptr)
{
        std::cout<<"[cap_pipewire] done\n";   
        delete static_cast<screen_cast_session*>(session_ptr);
}

static struct video_frame *vidcap_screen_pipewire_grab(void *session_ptr, struct audio_frame **audio)
{    
        using namespace std::chrono_literals;
        assert(session_ptr != nullptr);
        auto &session = *static_cast<screen_cast_session*>(session_ptr);
        *audio = nullptr;
   
        if(session.in_flight_frame != nullptr){
                session.blank_frames.wait_enqueue(session.in_flight_frame);
        }
        session.in_flight_frame = nullptr;

        {
                session.sending_frames.wait_dequeue(session.in_flight_frame);
        }
        return session.in_flight_frame;
}

static const struct video_capture_info vidcap_screen_pipewire_info = {
        vidcap_screen_pipewire_probe,
        vidcap_screen_pipewire_init,
        vidcap_screen_pipewire_done,
        vidcap_screen_pipewire_grab,
        true,
};

REGISTER_MODULE(screen_pipewire, &vidcap_screen_pipewire_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
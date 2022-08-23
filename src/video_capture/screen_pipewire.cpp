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
#include <fstream>

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
#include "concurrent_queue/readerwriterqueue.h"

#define MAX_BUFFERS 5
static constexpr int QUEUE_SIZE = 10;

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

                LOG(LOG_LEVEL_DEBUG) << "new request: '" << result.path << "'\n";
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

class ScreenCastPortal {
private:
        GMainLoop *dbus_loop;
        GDBusConnection *connection;
        GDBusProxy *screencast_proxy;
        std::string sender_name_;
public:
        // see https://flatpak.github.io/xdg-desktop-portal/#gdbus-signal-org-freedesktop-portal-Request.Response
        static constexpr uint32_t REQUEST_RESPONSE_OK = 0;
        static constexpr uint32_t REQUEST_RESPONSE_CANCELLED_BY_USER = 1;
        static constexpr uint32_t REQUEST_RESPONSE_OTHER_ERROR = 2;

        ScreenCastPortal() 
        {
                GError *error = nullptr;
                
                dbus_loop = g_main_loop_new(nullptr, false);
                connection = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error);
                g_assert_no_error(error);
                assert(connection != nullptr);

                sender_name_ = g_dbus_connection_get_unique_name(connection) + 1;
                std::replace(sender_name_.begin(), sender_name_.end(), '.', '_');
                screencast_proxy = g_dbus_proxy_new_sync(
                                connection, G_DBUS_PROXY_FLAGS_NONE, nullptr,
                                "org.freedesktop.portal.Desktop",
                                "/org/freedesktop/portal/desktop",
                                "org.freedesktop.portal.ScreenCast", nullptr, &error);
                g_assert_no_error(error); 
                assert(screencast_proxy != nullptr);
        }
        
        void call_with_request(const char* method_name, std::initializer_list<GVariant*> arguments, GVariantBuilder &params_builder, PortalCallCallback on_response)
        {
                assert(method_name != nullptr);
                request_path_t request_path = request_path_t::create(sender_name());
                //std::cout<<"DEBUG::: call " << request_path.path << "\n";
                LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: call_with_request: '" << method_name << "' request: '" << request_path.path << "'\n";
                auto callback = [](GDBusConnection *connection, const gchar *sender_name, const gchar *object_path,
                                        const gchar *interface_name, const gchar *signal_name, GVariant *parameters,
                                        gpointer user_data) {
                        (void) sender_name;
                        (void) interface_name;
                        (void) signal_name;
                        
                        static_cast<PortalCallCallback *> (user_data)->operator()(parameters);
                        //TODO: delete callback
                        //TODO: check if this actually works
                        g_dbus_connection_call(connection, "org.freedesktop.portal.Desktop",
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
                        g_assert_no_error(error); //TODO: this should set init_error
                        const char *path = nullptr;
                        g_variant_get(result_finished, "(o)", &path);
                        g_variant_unref(result_finished);
                        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: call_with_request finished: '" << path << "'\n";
                };


                g_variant_builder_add(&params_builder, "{sv}", "handle_token", g_variant_new_string(request_path.token.c_str()));
                
                GVariantBuilder args_builder;
                g_variant_builder_init(&args_builder, G_VARIANT_TYPE_TUPLE);
                for(GVariant* arg : arguments){
                        g_variant_builder_add_value(&args_builder, arg);
                }
                g_variant_builder_add_value(&args_builder, g_variant_builder_end(&params_builder));

                g_dbus_proxy_call(screencast_proxy, method_name, g_variant_builder_end(&args_builder), G_DBUS_CALL_FLAGS_NONE, -1, nullptr, call_finished, screencast_proxy);     
        }

        void run_loop() {
                g_main_loop_run(dbus_loop);
                LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: finished dbus loop \n";
        }

        void quit_loop() {
                g_main_loop_quit(dbus_loop);
        }

        //TODO: remove
        GDBusProxy *proxy() {
                return screencast_proxy;
        }

        GDBusConnection *dbus_connection() const {
                return connection;
        }

        const std::string& sender_name() const
        {
                return sender_name_;
        }

        ~ScreenCastPortal() {
                g_main_loop_quit(dbus_loop);
                //g_main_loop_unref(session.dbus_loop);
                g_object_unref(screencast_proxy);
                g_object_unref(connection);
        } 
};

class video_frame_wrapper
{
private:
        video_frame* frame;
public:
        explicit video_frame_wrapper(video_frame* frame = nullptr)
                :frame(frame)
        {}

        video_frame_wrapper(video_frame_wrapper&) = delete;
        video_frame_wrapper& operator= (video_frame_wrapper&) = delete;
        
        video_frame_wrapper(video_frame_wrapper&& other) noexcept
                : frame(std::exchange(other.frame, nullptr))
        {}

        video_frame_wrapper& operator=(video_frame_wrapper&& other) noexcept{
                vf_free(frame);
                frame = std::exchange(other.frame, nullptr);
                return *this;
        }

        ~video_frame_wrapper(){
                vf_free(frame);
        }

        video_frame* get() {
                return frame;
        }

        video_frame* operator->(){
                return get();
        }
};


struct screen_cast_session { 

        struct {
                bool show_cursor = false;
                std::string persistence_filename = "";
                int target_fps = 60;
        } user_options;

        std::unique_ptr<ScreenCastPortal> portal;

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
        int pw_frame_count = 0;
        uint64_t pw_begin_time = time_since_epoch_in_ms();
        uint64_t pw_approx_average_fps = user_options.target_fps;

        // used exlusively by ultragrid thread
        video_frame_wrapper in_flight_frame;

        // used exclusively by pipewire thread
        moodycamel::BlockingReaderWriterQueue<video_frame_wrapper> blank_frames {QUEUE_SIZE};
        moodycamel::BlockingReaderWriterQueue<video_frame_wrapper> sending_frames {QUEUE_SIZE};

        // empty string if no error occured, or an error message
        std::promise<std::string> init_error;

        video_frame_wrapper new_blank_frame()
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
                return video_frame_wrapper(frame);
        }

        std::chrono::duration<int64_t, std::milli> expected_frame_time_ms(){
                using namespace std::chrono_literals;
                return 1000ms / format.info.raw.framerate.num * format.info.raw.framerate.denom;
        }

        ~screen_cast_session() {
                LOG(LOG_LEVEL_INFO) << "[screen_pw]: screen_cast_session begin desructor\n";
                //pw_thread_loop_unlock();
                pw_thread_loop_stop(loop);
                //pw_stream_disconnect(stream);
                pw_stream_destroy(stream);
                //pw_thread_loop_destroy(loop);
                //pw_core_disconnect(core);
                LOG(LOG_LEVEL_INFO) << "[screen_pw]: screen_cast_session destroyed\n";
        }
};

static void on_stream_state_changed(void *session_ptr, enum pw_stream_state old, enum pw_stream_state state, const char *error) {
        (void) session_ptr;
        LOG(LOG_LEVEL_INFO) << "[screen_pw] stream state changed \"" << pw_stream_state_as_string(old) 
                                                << "\" -> \""<<pw_stream_state_as_string(state)<<"\"\n";
        
        if(error != nullptr) {
                LOG(LOG_LEVEL_ERROR) << "[screen_pw] stream error: '"<< error << "'\n";
        }
        
        switch (state) {
                case PW_STREAM_STATE_UNCONNECTED:
                        LOG(LOG_LEVEL_INFO) << "[screen_pw] stream disconected\n"; 
                        //assert(false && "disconected");
                        //pw_thread_loop_stop(session);
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
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: [cap_pipewire] param changed:\n";
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
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: size: " << session.format.info.raw.size.width << " x " << session.format.info.raw.size.height
                                << "\n";
        int mult = 1;

        int linesize = vc_get_linesize(session.size.width, RGBA);
        int32_t size = linesize * session.size.height;

        uint8_t params_buffer[1024];

        struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));
        const struct spa_pod *params[5];

        params[0] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                SPA_PARAM_BUFFERS_buffers,
                SPA_POD_CHOICE_RANGE_Int(20, 10, 50), //FIXME
                SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
                SPA_PARAM_BUFFERS_size, SPA_POD_Int(size * mult),
                SPA_PARAM_BUFFERS_stride,
                SPA_POD_Int(linesize * mult),
                SPA_PARAM_BUFFERS_align, SPA_POD_Int(16),
                SPA_PARAM_BUFFERS_dataType,
                SPA_POD_CHOICE_FLAGS_Int((1 << SPA_DATA_MemPtr)))
        );
        /*
        // a header metadata with timing information
        params[1] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
                SPA_PARAM_META_size,
                SPA_POD_Int(sizeof(struct spa_meta_header)))
        );
        // video cropping information
        params[2] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_VideoCrop),
                SPA_PARAM_META_size,
                SPA_POD_Int(sizeof(struct spa_meta_region)))
        );
        
        #define CURSOR_META_SIZE(w, h)   (sizeof(struct spa_meta_cursor) + sizeof(struct spa_meta_bitmap) + (w) * (h) * 4)
        // cursor
        params[3] = static_cast<spa_pod *>(spa_pod_builder_add_object(&builder,
                SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
                SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Cursor),
                SPA_PARAM_META_size, SPA_POD_CHOICE_RANGE_Int(
                        CURSOR_META_SIZE(64, 64),
                        CURSOR_META_SIZE(1, 1),
                        CURSOR_META_SIZE(256, 256))
                )
        );*/

        pw_stream_update_params(session.stream, params, 1);

        for(int i = 0; i < QUEUE_SIZE; ++i)
                session.blank_frames.enqueue(session.new_blank_frame());
        
        session.init_error.set_value("");
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

        pw_buffer *buffer;
        int n_buffers_from_pw = 0;
        while((buffer = pw_stream_dequeue_buffer(session.stream)) != nullptr){    
                ++n_buffers_from_pw;

                /*if( == nullptr && !session.blank_frames.try_dequeue(session.dequed_blank_frame))
                {
                        LOG(LOG_LEVEL_INFO) << "[screen_pw]: dropping - no blank frame\n";
                        pw_stream_queue_buffer(session.stream, buffer);
                        continue;
                }*/
           
                //;
                video_frame_wrapper next_frame;
                
                assert(buffer->buffer != nullptr);
                assert(buffer->buffer->datas != nullptr);
                assert(buffer->buffer->n_datas == 1);
                assert(buffer->buffer->datas[0].data != nullptr);
                // assert(buffer->size != 0);
                if(buffer->buffer->datas[0].chunk == nullptr || buffer->buffer->datas[0].chunk->size == 0) {
                        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: dropping - empty pw frame " << "\n";
                        pw_stream_queue_buffer(session.stream, buffer);
                        continue;
                }

                using namespace std::chrono_literals;
                if(!session.blank_frames.wait_dequeue_timed(next_frame, 1000ms / session.pw_approx_average_fps * 3 / 4)) {
                        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: dropping frame (blank frame dequeue timed out)\n";
                        pw_stream_queue_buffer(session.stream, buffer);
                        continue;
                }


                //memcpy(next_frame->tiles[0].data, static_cast<char*>(buffer->buffer->datas[0].data), session.size.height * vc_get_linesize(session.size.width, RGBA));
                copy_bgra_to_rgba(next_frame->tiles[0].data, static_cast<char*>(buffer->buffer->datas[0].data), session.size.width, session.size.height);
                
                /*
                struct spa_meta_cursor *cursor_metadata = static_cast<spa_meta_cursor *>(spa_buffer_find_meta_data(buffer->buffer, SPA_META_Cursor, sizeof(spa_meta_cursor)));
                //std::cout<<"cursor meta" << cursor_metadata << std::endl;
                
                if(cursor_metadata != nullptr && spa_meta_cursor_is_valid(cursor_metadata)){
                        std::cout << "cursor: "<< cursor_metadata->position.x << " " << cursor_metadata->position.y << "";
                }*/

                session.sending_frames.enqueue(std::move(next_frame));
                
                pw_stream_queue_buffer(session.stream, buffer);
                
                ++session.pw_frame_count;
                uint64_t time_now = time_since_epoch_in_ms();

                uint64_t delta = time_now - session.pw_begin_time;
                if(delta >= 5000) {
                        double average_fps = session.pw_frame_count / (delta / 1000.0);
                        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: on process: average fps in last 5 seconds: " << average_fps << "\n";
                        session.pw_approx_average_fps = average_fps;
                        session.pw_frame_count = 0;
                        session.pw_begin_time = time_since_epoch_in_ms();
                }
        }
        
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: from pw: "<< n_buffers_from_pw << "\t sending: "<<session.sending_frames.size_approx() << "\t blank: " << session.blank_frames.size_approx() << "\n";
        
}

static void on_drained(void*)
{
        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: pipewire: drained\n";
}

static void on_add_buffer(void *session_ptr, struct pw_buffer *)
{
        (void) session_ptr;

        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: pipewire: add_buffer\n";
}

static void on_remove_buffer(void *session_ptr, struct pw_buffer *)
{
        (void) session_ptr;
        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: pipewire: remove_buffer\n";
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
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: duplicating fd " << session.pipewire_fd << " -> " << new_pipewire_fd << "\n";
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

        pw_thread_loop_unlock(session.loop);
        return 0;
}

static void on_portal_session_closed(GDBusConnection *connection, const gchar *sender_name, const gchar *object_path,
                                                                        const gchar *interface_name, const gchar *signal_name, GVariant *parameters, gpointer user_data)
{
        (void) connection;
        (void) sender_name;
        (void) object_path;
        (void) interface_name;
        (void) signal_name;
        (void) parameters;
        auto &session = *static_cast<screen_cast_session*>(user_data);
        //TODO: check if this is fired by newer Gnome 
        LOG(LOG_LEVEL_INFO) << "[screen_pw] session closed by compositor\n";
        pw_thread_loop_stop(session.loop);
}

static void run_screencast(screen_cast_session *session_ptr) {
        auto& session = *session_ptr;
        session.portal = std::make_unique<ScreenCastPortal>();

        session.pipewire_fd = -1;
        session.pipewire_node = UINT32_MAX;
        
        session_path_t session_path = session_path_t::create(session.portal->sender_name());
        LOG(LOG_LEVEL_VERBOSE) << "[screen_pw]: session path: '" << session_path.path << "'" << " token: '" << session_path.token << "'\n";

        g_dbus_connection_signal_subscribe(session.portal->dbus_connection(), 
                                                                           nullptr, // sender
                                                                           "org.freedesktop.portal.Session", // interface_name
                                                                           "closed", //signal name
                                                                           session_path.path.c_str(), // object path
                                                                           nullptr, // arg0
                                                                           G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE,
                                                                           on_portal_session_closed,
                                                                           session_ptr,
                                                                           nullptr);

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
                
                LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: starting pipewire\n";
                start_pipewire(*session);
        };

        auto started = [&](GVariant *parameters) {
                LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: started: " << g_variant_print(parameters, true) << "\n";
                
                uint32_t response;
                GVariant *results;
                g_variant_get(parameters, "(u@a{sv})", &response, &results);
                if(response == ScreenCastPortal::REQUEST_RESPONSE_CANCELLED_BY_USER) {
                        session.init_error.set_value("failed to start (dialog cancelled by user)");
                        return;
                } else if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                        session.init_error.set_value("failed to start (unknown reason)");
                        return;
                }

                const char *restore_token = nullptr;
                if (g_variant_lookup(results, "restore_token", "s", &restore_token)){
                        if(session.user_options.persistence_filename.empty()){
                                LOG(LOG_LEVEL_WARNING) << "[screen_pw]: got unexpected restore_token from ScreenCast portal, ignoring it\n";
                        }else{
                                std::ofstream file(session.user_options.persistence_filename);
                                file<<restore_token;
                        }
                }

                GVariant *streams = g_variant_lookup_value(results, "streams", G_VARIANT_TYPE_ARRAY);
                GVariant *stream_properties;
                GVariantIter iter;
                g_variant_iter_init(&iter, streams);
                assert(g_variant_iter_n_children(&iter) == 1); //TODO: do I need the KDE work-around like in OBS for this bug?
                bool got_item = g_variant_iter_loop(&iter, "(u@a{sv})", &session.pipewire_node, &stream_properties);
                assert(got_item);
                g_variant_unref(stream_properties);
                g_variant_unref(results);
                GVariantBuilder builder;
                g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
                g_dbus_proxy_call_with_unix_fd_list(session.portal->proxy(), "OpenPipeWireRemote",
                                                                                        g_variant_new("(oa{sv})", session_path.path.c_str(), &builder),
                                                                                        G_DBUS_CALL_FLAGS_NONE, -1,
                                                                                        nullptr, nullptr, pipewire_opened, &session);
                g_variant_builder_clear(&builder);
        };

        auto sources_selected = [&](GVariant *parameters) {
                gchar *pretty = g_variant_print(parameters, true);
                LOG(LOG_LEVEL_INFO) << "[screen_pw]: selected sources: " << pretty << "\n";
                g_free((gpointer) pretty);

                uint32_t response;
                GVariant *results;
                g_variant_get(parameters, "(u@a{sv})", &response, &results);
                if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                        session.init_error.set_value("Failed to select sources");
                        return;
                }
                g_variant_unref(results);
                //portal_call(connection, screencast_proxy, session_path.path.c_str(), "Start", {}, started);
                
                {
                        GVariantBuilder params;
                        g_variant_builder_init(&params, G_VARIANT_TYPE_VARDICT);
                        session.portal->call_with_request("Start", {g_variant_new_object_path(session_path.path.c_str()),  /*parent window: */ g_variant_new_string("")}, params, started);
                }
        };

        auto session_created = [&](GVariant *parameters) {
                uint32_t response; 
                GVariant *results;

                g_variant_get(parameters, "(u@a{sv})", &response, &results);
                if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                        session.init_error.set_value("Failed to create session");
                        return;
                }
                const char *session_handle = nullptr;
                g_variant_lookup(results, "session_handle", "s", &session_handle);
                g_variant_unref(results);

                LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: session created with handle: " << session_handle << "\n";
                assert(session_path.path == session_handle);

                
                {
                        GVariantBuilder params;
                        g_variant_builder_init(&params, G_VARIANT_TYPE_VARDICT);
                        g_variant_builder_add(&params, "{sv}", "types", g_variant_new_uint32(3)); // 1 full screen, 2 - a window, 3 - both
                        g_variant_builder_add(&params, "{sv}", "multiple", g_variant_new_boolean(false));
                        if(session.user_options.show_cursor)
                                g_variant_builder_add(&params, "{sv}", "cursor_mode", g_variant_new_uint32(2));
                        
                        if(!session.user_options.persistence_filename.empty()){
                                std::string token;
                                std::ifstream file(session.user_options.persistence_filename);

                                if(file.is_open()) {
                                        std::ostringstream ss;
                                        ss << file.rdbuf();
                                        token = ss.str();
                                }
                                
                                //  0: Do not persist (default), 1: Permissions persist as long as the application is running, 2: Permissions persist until explicitly revoked
                                g_variant_builder_add(&params, "{sv}", "persist_mode", g_variant_new_uint32(2)); 
                                if(!token.empty())
                                        g_variant_builder_add(&params, "{sv}", "restore_token", g_variant_new_string(token.c_str())); 
                        }

                        // {"cursor_mode", g_variant_new_uint32(4)}
                        session.portal->call_with_request("SelectSources", {g_variant_new_object_path(session_path.path.c_str())}, params, sources_selected);
                }
        };


        {
                GVariantBuilder params;
                g_variant_builder_init(&params, G_VARIANT_TYPE_VARDICT);
                g_variant_builder_add(&params, "{sv}", "session_handle_token", g_variant_new_string(session_path.token.c_str()));
                
                session.portal->call_with_request("CreateSession", {}, params, session_created);
        }

        session.portal->run_loop();
        //g_dbus_connection_flush(connection, nullptr, nullptr, nullptr); //TODO: needed?
}

static struct vidcap_type * vidcap_screen_pipewire_probe(bool verbose, void (**deleter)(void *))
{
        (void) verbose;
        (void) deleter;
        
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: [cap_pipewire] probe\n";
        exit(0);
        return nullptr;
}


static int vidcap_screen_pipewire_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        screen_cast_session *session = new screen_cast_session();
        *state = session;

        if(vidcap_params_get_fmt(params)) {
                std::cout<<"fmt: '" << vidcap_params_get_fmt(params)<<"'" << "\n";
                
                std::string params_string = vidcap_params_get_fmt(params);
                
                if (params_string == "help") {
                                //TODO
                                //show_help();
                                //free(s);
                                //TODO
                                return VIDCAP_INIT_NOERR;
                } else if (params_string == "showcursor") {d
                        session->user_options.show_cursor = true;
                } else if (params_string == "persistent") {
                        session->user_options.persistence_filename = "screen-pw.token";
                }
        }
        
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: [cap_pipewire] init\n";
        pw_init(&uv_argc, &uv_argv);

        std::future<std::string> future_error = session->init_error.get_future();
        std::thread dbus_thread(run_screencast, session);
        future_error.wait();
        
        if (std::string error_msg = future_error.get(); !error_msg.empty()) {
                LOG(LOG_LEVEL_FATAL) << "[screen_pw]: " << error_msg << "\n";
                dbus_thread.join();
                session->portal->quit_loop();
                return VIDCAP_INIT_FAIL;
        }

        dbus_thread.detach();
        LOG(LOG_LEVEL_DEBUG) << "[screen_pw]: init ok\n";
        return VIDCAP_INIT_OK;
}

static void vidcap_screen_pipewire_done(void *session_ptr)
{
        LOG(LOG_LEVEL_DEBUG) <<"[cap_pipewire] done\n";   
        delete static_cast<screen_cast_session*>(session_ptr);
}

static struct video_frame *vidcap_screen_pipewire_grab(void *session_ptr, struct audio_frame **audio)
{    
        assert(session_ptr != nullptr);
        auto &session = *static_cast<screen_cast_session*>(session_ptr);
        *audio = nullptr;
   
        if(session.in_flight_frame.get() != nullptr){
                session.blank_frames.enqueue(std::move(session.in_flight_frame));
        }

        using namespace std::chrono_literals;
        session.sending_frames.wait_dequeue_timed(session.in_flight_frame, 500ms);
        //session.sending_frames.try_deque(session.in_flight_frame);
        return session.in_flight_frame.get();
}

static const struct video_capture_info vidcap_screen_pipewire_info = {
        vidcap_screen_pipewire_probe,
        vidcap_screen_pipewire_init,
        vidcap_screen_pipewire_done,
        vidcap_screen_pipewire_grab,
        true,
};

REGISTER_MODULE(screen_pipewire, &vidcap_screen_pipewire_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
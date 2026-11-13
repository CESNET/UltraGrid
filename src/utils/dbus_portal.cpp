/**
 * @file   utils/dbus_portal.cpp
 * @author Matej Hrica       <492778@mail.muni.cz>
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET z.s.p.o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <algorithm>
#include <string>
#include <memory>
#include <cassert>
#include <future>
#include <fstream>
#include <gio/gio.h>
#include <gio/gunixfdlist.h>
#include <glib-object.h>      // for GObject, g_object_unref
#include <glib.h>             // for g_variant_builder_add, GVariant, g_vari...
#include <glib/gtypes.h>      // for gint32
#include <initializer_list>   // for initializer_list

#include "debug.h"
#include "misc.h"
#include "dbus_portal.hpp"

#include <random>

#define MOD_NAME "[dbus] "

namespace {

std::string generate_token(){
        std::random_device random_device;
        auto random_engine = std::default_random_engine(random_device());
        auto dist = std::uniform_int_distribution<uint64_t>();
        const auto rand_val = dist(random_engine);

        return "uv_" + std::to_string(rand_val);
}

struct Path_with_token{
        std::string token;
        std::string path;

        Path_with_token() = default;

        explicit Path_with_token(std::string_view path, std::string_view name)
                : token(generate_token()), path(std::string(path) + std::string(name) + "/" + token) { }
};

constexpr std::string_view base_request_path = "/org/freedesktop/portal/desktop/request/";
constexpr std::string_view base_session_path = "/org/freedesktop/portal/desktop/session/";

using GVariant_uniq = std::unique_ptr<GVariant, deleter_from_fcn<g_variant_unref>>;
using GMainLoop_uniq = std::unique_ptr<GMainLoop, deleter_from_fcn<g_main_loop_unref>>;
using GDBusConnection_uniq = std::unique_ptr<GDBusConnection, deleter_from_fcn<g_object_unref>>;
using GDBusProxy_uniq = std::unique_ptr<GDBusProxy, deleter_from_fcn<g_object_unref>>;

class ScreenCastPortal final : public PipewirePortal{
private:
        template<auto callback>
        void call_with_request(const char* method_name,
                        std::initializer_list<GVariant*> arguments,
                        GVariantBuilder &params_builder);

        static void pipewire_opened(GObject *source,
                        GAsyncResult *res,
                        void *user_data);

        static void started(uint32_t response, GVariant *results, void *user_data);
        static void sources_selected(uint32_t response, GVariant *results, ScreenCastPortal *s);
        static void session_created(uint32_t response, GVariant *results, ScreenCastPortal *s);

        GMainLoop_uniq dbus_loop;
        GDBusConnection_uniq connection;
        GDBusProxy_uniq screencast_proxy;
        std::string unique_name;
        Path_with_token session;

        std::string restore_file;
        bool show_cursor = false;

        int pw_fd = -1;
        uint32_t pw_node = UINT32_MAX;

        std::promise<std::string> promise;
        std::thread thread;

public:
        // see https://flatpak.github.io/xdg-desktop-portal/#gdbus-signal-org-freedesktop-portal-Request.Response
        static constexpr uint32_t REQUEST_RESPONSE_OK = 0;
        static constexpr uint32_t REQUEST_RESPONSE_CANCELLED_BY_USER = 1;
        static constexpr uint32_t REQUEST_RESPONSE_OTHER_ERROR = 2;

        ScreenCastPortal(std::string restore_file, bool show_cursor);
        ~ScreenCastPortal() override;

        void loop_worker();

        PipewirePortalResult run() override;

        int get_pw_fd(){ return pw_fd; }
        uint32_t get_pw_node(){ return pw_node; }
        
        GDBusProxy *proxy() {
                return screencast_proxy.get();
        }

        GDBusConnection *dbus_connection() const {
                return connection.get();
        }

        const std::string& sender_name() const
        {
                return unique_name;
        }

        const std::string& session_path() const
        {
                return session.path;
        }

        const std::string& session_token() const
        {
                return session.token;
        }
};

ScreenCastPortal::ScreenCastPortal(std::string restore_file, bool show_cursor)
        : restore_file(std::move(restore_file)),
        show_cursor(show_cursor)
{
        GError *error = nullptr;

        dbus_loop.reset(g_main_loop_new(nullptr, false));
        connection.reset(g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error));
        g_assert_no_error(error);
        assert(connection != nullptr);

        unique_name = g_dbus_connection_get_unique_name(connection.get()) + 1;
        std::replace(unique_name.begin(), unique_name.end(), '.', '_');
        screencast_proxy.reset(g_dbus_proxy_new_sync(
                        connection.get(), G_DBUS_PROXY_FLAGS_NONE, nullptr,
                        "org.freedesktop.portal.Desktop",
                        "/org/freedesktop/portal/desktop",
                        "org.freedesktop.portal.ScreenCast", nullptr, &error));
        g_assert_no_error(error); 
        assert(screencast_proxy != nullptr);

        session = Path_with_token(base_session_path, unique_name);
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "session path: '" << session.path << "'" << " token: '" << session.token << "'\n";
}

ScreenCastPortal::~ScreenCastPortal() {
        g_dbus_connection_call(dbus_connection(),
                        "org.freedesktop.portal.Desktop",
                        session.path.c_str(),
                        "org.freedesktop.portal.Session",
                        "Close", nullptr, nullptr,
                        G_DBUS_CALL_FLAGS_NONE, -1, nullptr, nullptr,
                        nullptr);
        g_main_loop_quit(dbus_loop.get());

        if(thread.joinable())
                thread.join();
}

template<auto func>
void CallbackWrapper(GDBusConnection *connection,
                const gchar *sender_name,
                const gchar *object_path,
                const gchar *interface_name,
                const gchar *signal_name,
                GVariant *parameters,
                gpointer user_data)
{
                (void) sender_name;
                (void) interface_name;
                (void) signal_name;

                uint32_t response;
                GVariant_uniq results;
                g_variant_get(parameters, "(u@a{sv})", &response, out_ptr(results).operator GVariant**());

                func(response, results.get(), static_cast<ScreenCastPortal *>(user_data));

                g_dbus_connection_call(connection, "org.freedesktop.portal.Desktop",
                                object_path, "org.freedesktop.portal.Request", "Close",
                                nullptr, nullptr, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, nullptr, nullptr);

}

template<auto callback>
void ScreenCastPortal::call_with_request(const char* method_name,
                std::initializer_list<GVariant*> arguments,
                GVariantBuilder &params_builder)
{
        assert(method_name != nullptr);
        Path_with_token request_path = Path_with_token(base_request_path, sender_name());
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "call_with_request: '" << method_name << "' request: '" << request_path.path << "'\n";

        g_dbus_connection_signal_subscribe(connection.get(), "org.freedesktop.portal.Desktop",
                        "org.freedesktop.portal.Request",
                        "Response",
                        request_path.path.c_str(),
                        nullptr,
                        G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE,
                        CallbackWrapper<callback>,
                        this,
                        nullptr);

        auto call_finished = [](GObject *source_object, GAsyncResult *result, gpointer user_data) {
                auto s = static_cast<ScreenCastPortal *>(user_data);
                GError *error = nullptr;
                GVariant_uniq result_finished(g_dbus_proxy_call_finish(G_DBUS_PROXY(source_object), result, &error));

                if(error != nullptr){
                        s->promise.set_value(error->message);
                        return;
                }

                const char *path = nullptr;
                g_variant_get(result_finished.get(), "(o)", &path);
                LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "call_with_request finished: '" << path << "'\n";
        };


        g_variant_builder_add(&params_builder, "{sv}", "handle_token", g_variant_new_string(request_path.token.c_str()));

        GVariantBuilder args_builder;
        g_variant_builder_init(&args_builder, G_VARIANT_TYPE_TUPLE);
        for(GVariant* arg : arguments){
                g_variant_builder_add_value(&args_builder, arg);
        }
        g_variant_builder_add_value(&args_builder, g_variant_builder_end(&params_builder));

        g_dbus_proxy_call(screencast_proxy.get(), method_name, g_variant_builder_end(&args_builder), G_DBUS_CALL_FLAGS_NONE, -1, nullptr, call_finished, this);
}

static void on_portal_session_closed([[maybe_unused]] GDBusConnection *connection,
                [[maybe_unused]] const gchar *sender_name,
                [[maybe_unused]] const gchar *object_path,
                [[maybe_unused]] const gchar *interface_name,
                [[maybe_unused]] const gchar *signal_name,
                [[maybe_unused]] GVariant *parameters,
                [[maybe_unused]] gpointer user_data)
{
        //Doesn't seem to be called by GNOME
        LOG(LOG_LEVEL_INFO) << MOD_NAME "session closed by compositor\n";
}

void ScreenCastPortal::pipewire_opened(GObject *source,
                GAsyncResult *res,
                void *user_data)
{
        auto s = static_cast<ScreenCastPortal *>(user_data);
        GError *error = nullptr;
        GUnixFDList *fd_list = nullptr;

        GVariant_uniq result(g_dbus_proxy_call_with_unix_fd_list_finish(G_DBUS_PROXY(source), &fd_list, res, &error));
        g_assert_no_error(error);

        gint32 handle;
        g_variant_get(result.get(), "(h)", &handle);
        assert(handle == 0); //it should always be the first index

        s->pw_fd = g_unix_fd_list_get(fd_list, handle, &error);
        g_assert_no_error(error);

        s->promise.set_value("");
}

void ScreenCastPortal::started(uint32_t response, GVariant *results, void *user_data) {
        auto s = static_cast<ScreenCastPortal *>(user_data);
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "started: " << g_variant_print(results, true) << "\n";

        if(response == ScreenCastPortal::REQUEST_RESPONSE_CANCELLED_BY_USER) {
                s->promise.set_value("failed to start (dialog cancelled by user)");
                return;
        } else if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                s->promise.set_value("failed to start (unknown reason)");
                return;
        }

        const char *restore_token = nullptr;
        if (g_variant_lookup(results, "restore_token", "s", &restore_token)){
                std::ofstream file(s->restore_file);
                file<<restore_token;
        }

        GVariant *streams = g_variant_lookup_value(results, "streams", G_VARIANT_TYPE_ARRAY);
        GVariant *stream_properties;
        GVariantIter iter;
        g_variant_iter_init(&iter, streams);
        assert(g_variant_iter_n_children(&iter) == 1);
        bool got_item = g_variant_iter_loop(&iter, "(u@a{sv})", &s->pw_node, &stream_properties);
        assert(got_item);

        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
        g_dbus_proxy_call_with_unix_fd_list(s->proxy(), "OpenPipeWireRemote",
                        g_variant_new("(oa{sv})", s->session_path().c_str(), &builder),
                        G_DBUS_CALL_FLAGS_NONE, -1,
                        nullptr, nullptr, ScreenCastPortal::pipewire_opened, s);
        g_variant_builder_clear(&builder);
}

void ScreenCastPortal::sources_selected(uint32_t response, GVariant *results, ScreenCastPortal *s) {
        gchar *pretty = g_variant_print(results, true);
        LOG(LOG_LEVEL_INFO) << MOD_NAME "selected sources: " << pretty << "\n";
        g_free((gpointer) pretty);

        if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                s->promise.set_value("Failed to select sources");
                return;
        }

        {
                GVariantBuilder options;
                g_variant_builder_init(&options, G_VARIANT_TYPE_VARDICT);
                s->call_with_request<started>("Start", {g_variant_new_object_path(s->session_path().c_str()),  /*parent window: */ g_variant_new_string("")}, options);
        }
};

void ScreenCastPortal::session_created(uint32_t response, GVariant *results, ScreenCastPortal *s) {
        if(response != ScreenCastPortal::REQUEST_RESPONSE_OK) {
                s->promise.set_value("Failed to create session");
                return;
        }

        const char *session_handle = nullptr;
        g_variant_lookup(results, "session_handle", "s", &session_handle);

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "session created with handle: " << session_handle << "\n";
        assert(s->session_path() == session_handle);

        {
                GVariantBuilder params;
                g_variant_builder_init(&params, G_VARIANT_TYPE_VARDICT);
                g_variant_builder_add(&params, "{sv}", "types", g_variant_new_uint32(3)); // 1 full screen, 2 - a window, 3 - both
                g_variant_builder_add(&params, "{sv}", "multiple", g_variant_new_boolean(false));
                if(s->show_cursor)
                        g_variant_builder_add(&params, "{sv}", "cursor_mode", g_variant_new_uint32(2));

                if(!s->restore_file.empty()){
                        std::string token;
                        std::ifstream file(s->restore_file);

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

                s->call_with_request<sources_selected>("SelectSources", {g_variant_new_object_path(s->session_path().c_str())}, params);
        }
};

void ScreenCastPortal::loop_worker(){
        g_dbus_connection_signal_subscribe(dbus_connection(),
                        nullptr, // sender
                        "org.freedesktop.portal.Session", // interface_name
                        "closed", //signal name
                        session_path().c_str(), // object path
                        nullptr, // arg0
                        G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE,
                        on_portal_session_closed,
                        nullptr,
                        nullptr);

        GVariantBuilder params;
        g_variant_builder_init(&params, G_VARIANT_TYPE_VARDICT);
        g_variant_builder_add(&params, "{sv}", "session_handle_token", g_variant_new_string(session_token().c_str()));

        call_with_request<session_created>("CreateSession", {}, params);

        g_main_loop_run(dbus_loop.get());
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "finished dbus loop \n";
}

PipewirePortalResult ScreenCastPortal::run(){
        thread = std::thread(&ScreenCastPortal::loop_worker, this);

        auto error = promise.get_future().get();
        if(!error.empty()){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", error.c_str());
                return {-1, UINT32_MAX};
        }

        return {get_pw_fd(), get_pw_node()};
}
} //anon namespace


std::unique_ptr<PipewirePortal> CreateScreenCastPortal(const std::string& restore_file, bool show_cursor){
        return std::make_unique<ScreenCastPortal>(restore_file, show_cursor);
}

/**
 * @file   video_display/openxr_gl.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2021 CESNET, z. s. p. o.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <chrono>
#include <thread>

#include <assert.h>
//#include <GL/glew.h>
#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/glx.h>
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

#include "opengl_utils.hpp"
#include "utils/profile_timer.hpp"

#define MAX_BUFFER_SIZE   3

class Openxr_session{
public:
        Openxr_session(XrInstance instance,
                        XrSystemId systemId,
                        Display *xDisplay,
                        GLXContext glxContext,
                        GLXDrawable glxDrawable)
        {
                XrGraphicsBindingOpenGLXlibKHR graphics_binding_gl = {};
                graphics_binding_gl.type = XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR;
                graphics_binding_gl.xDisplay = xDisplay;
                graphics_binding_gl.glxContext = glxContext;
                graphics_binding_gl.glxDrawable = glxDrawable;

                XrSessionCreateInfo session_create_info = {};
                session_create_info.type = XR_TYPE_SESSION_CREATE_INFO;
                session_create_info.next = &graphics_binding_gl;
                session_create_info.systemId = systemId;

                XrResult result;

                PFN_xrGetOpenGLGraphicsRequirementsKHR pfnGetOpenGLGraphicsRequirementsKHR = nullptr;
                result = xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR",
                                        reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLGraphicsRequirementsKHR));

                XrGraphicsRequirementsOpenGLKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
                result = pfnGetOpenGLGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements);

                result = xrCreateSession(instance, &session_create_info, &session);
                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to create OpenXR session!");
                }
        }

        ~Openxr_session(){
                xrDestroySession(session);
        }

        XrSession get(){ return session; }

        void begin(){
                XrSessionBeginInfo session_begin_info;
                session_begin_info.type = XR_TYPE_SESSION_BEGIN_INFO;
                session_begin_info.next = NULL;
                session_begin_info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                XrResult result = xrBeginSession(session, &session_begin_info);
                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to begin OpenXR session!");
                }
        }
private:
        XrSession session;
};

class Openxr_instance{
public:
        Openxr_instance(){
                uint32_t properties_count = 0;
                XrResult result = xrEnumerateInstanceExtensionProperties(nullptr,
                                properties_count,
                                &properties_count,
                                nullptr);
                if(!XR_SUCCEEDED(result)){
                        log_msg(LOG_LEVEL_WARNING, "Failed to check XR_KHR_OPENGL availability!\n");
                } else {
                        std::vector<XrExtensionProperties> props;
                        props.resize(properties_count);
                        for(auto& prop : props){
                                prop.type = XR_TYPE_EXTENSION_PROPERTIES;
                                prop.next = nullptr;
                        }
                        result = xrEnumerateInstanceExtensionProperties(nullptr,
                                        properties_count,
                                        &properties_count,
                                        props.data());

                        bool found = false;
                        for(const auto& prop : props){
                                if(strcmp(prop.extensionName, XR_KHR_OPENGL_ENABLE_EXTENSION_NAME) == 0){
                                        found = true;
                                        break;
                                }
                        }
                        if(!found){
                                throw std::runtime_error("OpenXR runtime does not support OpenGL interop!");
                        }
                }
                const char* const enabledExtensions[] = {XR_KHR_OPENGL_ENABLE_EXTENSION_NAME};

                XrInstanceCreateInfo instanceCreateInfo;
                instanceCreateInfo.type = XR_TYPE_INSTANCE_CREATE_INFO;
                instanceCreateInfo.next = NULL;
                instanceCreateInfo.createFlags = 0;
                instanceCreateInfo.enabledExtensionCount = 1;
                instanceCreateInfo.enabledExtensionNames = enabledExtensions;
                instanceCreateInfo.enabledApiLayerCount = 0;
                strcpy(instanceCreateInfo.applicationInfo.applicationName, "UltraGrid OpenXR gl display");
                strcpy(instanceCreateInfo.applicationInfo.engineName, "");
                instanceCreateInfo.applicationInfo.applicationVersion = 1;
                instanceCreateInfo.applicationInfo.engineVersion = 0;
                instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

                result = xrCreateInstance(&instanceCreateInfo, &instance);
                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to create OpenXR instance!");
                }

                XrInstanceProperties instanceProperties;
                instanceProperties.type = XR_TYPE_INSTANCE_PROPERTIES;
                instanceProperties.next = NULL;

                result = xrGetInstanceProperties(instance, &instanceProperties);

                if(XR_SUCCEEDED(result)){
                        printf("Runtime Name: %s\n", instanceProperties.runtimeName);
                        printf("Runtime Version: %d.%d.%d\n",
                                        XR_VERSION_MAJOR(instanceProperties.runtimeVersion),
                                        XR_VERSION_MINOR(instanceProperties.runtimeVersion),
                                        XR_VERSION_PATCH(instanceProperties.runtimeVersion));
                }


        }

        ~Openxr_instance(){
                xrDestroyInstance(instance);
        }

        Openxr_instance(const Openxr_instance&) = delete;
        Openxr_instance(Openxr_instance&&) = delete;
        Openxr_instance& operator=(const Openxr_instance&) = delete;
        Openxr_instance& operator=(Openxr_instance&&) = delete;

        XrInstance get() { return instance; }
private:
        XrInstance instance;
};

class Openxr_swapchain{
public:
        Openxr_swapchain() = default;

        Openxr_swapchain(XrSession session, const XrSwapchainCreateInfo *info) :
                session(session)
        {
                xrCreateSwapchain(session, info, &swapchain);
        }

        Openxr_swapchain(XrSession session,
                        int64_t swapchain_format,
                        uint32_t w,
                        uint32_t h) :
                session(session)
        {
                XrSwapchainCreateInfo swapchain_create_info;
                swapchain_create_info.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
                swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
                        XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
                swapchain_create_info.createFlags = 0;
                swapchain_create_info.format = swapchain_format;
                swapchain_create_info.sampleCount = 1;
                swapchain_create_info.width = w;
                swapchain_create_info.height = h;
                swapchain_create_info.faceCount = 1;
                swapchain_create_info.arraySize = 1;
                swapchain_create_info.mipCount = 1;
                swapchain_create_info.next = nullptr;

                XrResult result = xrCreateSwapchain(session, &swapchain_create_info, &swapchain);

                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to create OpenXR swapchain!");
                }
        }

        ~Openxr_swapchain(){
                if(swapchain != XR_NULL_HANDLE){
                        xrDestroySwapchain(swapchain);
                }
        }

        Openxr_swapchain(const Openxr_swapchain&) = delete;
        Openxr_swapchain(Openxr_swapchain&& o) noexcept : Openxr_swapchain() {
                swap(o);
        }
        Openxr_swapchain& operator=(const Openxr_swapchain&) = delete;
        Openxr_swapchain& operator=(Openxr_swapchain&& o) { swap(o); return *this; }

        uint32_t get_length(){
                uint32_t len = 0;
                XrResult result = xrEnumerateSwapchainImages(swapchain, 0, &len, nullptr);
                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to enumerate swapchain images\n");
                }

                return len;
        }

        void swap(Openxr_swapchain& o) noexcept{
                std::swap(swapchain, o.swapchain);
                std::swap(session, o.session);
        }

        XrSwapchain get(){ return swapchain; }

private:
        XrSwapchain swapchain = XR_NULL_HANDLE;
        XrSession session = XR_NULL_HANDLE;
};

class Gl_interop_swapchain{
public:
        Gl_interop_swapchain(XrSession session, const XrSwapchainCreateInfo *info) :
                xr_swapchain(session, info)
        {
                init();
        }

        Gl_interop_swapchain(XrSession session,
                        int64_t swapchain_format,
                        uint32_t w,
                        uint32_t h) :
                xr_swapchain(session, swapchain_format, w, h)
        {
                init();
        }

        GLuint get_texture(size_t idx){
                return images[idx].image;
        }

        Framebuffer& get_framebuffer(size_t idx){
                return framebuffers[idx];
        }

        XrSwapchain get() { return xr_swapchain.get(); }

        Gl_interop_swapchain(const Gl_interop_swapchain&) = delete;
        Gl_interop_swapchain(Gl_interop_swapchain&&) = default;
        Gl_interop_swapchain& operator=(const Gl_interop_swapchain&) = delete;
        Gl_interop_swapchain& operator=(Gl_interop_swapchain&&) = default;

private:
        void init(){
                uint32_t length = xr_swapchain.get_length();
                images.resize(length);
                framebuffers.resize(length);

                for(auto& image : images){
                        image.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
                        image.next = nullptr;
                }

                XrResult result = xrEnumerateSwapchainImages(xr_swapchain.get(),
                                length, &length,
                                (XrSwapchainImageBaseHeader *)(images.data()));

                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to enumerate swapchain images");
                }

                for(size_t i = 0; i < length; i++){
                        framebuffers[i].attach_texture(get_texture(i));
                }
        }

        Openxr_swapchain xr_swapchain;
        std::vector<Framebuffer> framebuffers;
        std::vector<XrSwapchainImageOpenGLKHR> images;
};

class Openxr_local_space {
public:
        Openxr_local_space(XrSession session){
                XrPosef origin{};
                origin.orientation.x = 0.0;
                origin.orientation.y = 0.0;
                origin.orientation.z = 0.0;
                origin.orientation.w = 1.0;

                XrReferenceSpaceCreateInfo space_create_info;
                space_create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
                space_create_info.next = NULL;
                space_create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
                space_create_info.poseInReferenceSpace = origin;

                XrResult result = xrCreateReferenceSpace(session, &space_create_info, &space);
                if(!XR_SUCCEEDED(result)){
                        throw std::runtime_error("Failed to create OpenXR reference space!");
                }
        }

        ~Openxr_local_space(){
                xrDestroySpace(space);
        }

        Openxr_local_space(const Openxr_local_space&) = delete;
        Openxr_local_space(Openxr_local_space&&) = delete;
        Openxr_local_space& operator=(const Openxr_local_space&) = delete;
        Openxr_local_space& operator=(Openxr_local_space&&) = delete;

        XrSpace get() { return space; }

private:
        XrSpace space;
};

struct Openxr_state{
        Openxr_instance instance;
        XrSystemId system_id;
        //Openxr_session session;
};

struct state_xrgl{
        video_desc current_desc;
        int buffered_frames_count;

        Sdl_window window;
        Openxr_state xr_state;

        Scene scene;

        std::chrono::steady_clock::time_point last_frame;

        std::mutex lock;
        std::condition_variable frame_consumed_cv;
        std::condition_variable new_frame_ready_cv;
        std::queue<video_frame *> frame_queue;

        std::vector<video_frame *> free_frame_pool;
        std::condition_variable free_frame_ready_cv;

        std::vector<video_frame *> dispose_frame_pool;
};

static std::vector<XrViewConfigurationView> get_views(Openxr_state& xr_state){
        unsigned view_count;
        XrResult result = xrEnumerateViewConfigurationViews(xr_state.instance.get(),
                        xr_state.system_id,
                        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                        0,
                        &view_count,
                        nullptr);

        if(!XR_SUCCEEDED(result)){
                throw std::runtime_error("Failed to enumerate view configuration views!");
        }

        std::vector<XrViewConfigurationView> config_views(view_count);
        for(auto& view : config_views) view.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;

        result = xrEnumerateViewConfigurationViews(xr_state.instance.get(),
                        xr_state.system_id,
                        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                        view_count,
                        &view_count,
                        config_views.data());

        if(!XR_SUCCEEDED(result)){
                throw std::runtime_error("Failed to enumerate view configuration views!");
        }

        return config_views;
}

static std::vector<int64_t> get_swapchain_formats(XrSession session){
        XrResult result;
        unsigned swapchain_format_count;
        result = xrEnumerateSwapchainFormats(session,
                        0,
                        &swapchain_format_count,
                        nullptr);

        printf("Runtime supports %d swapchain formats\n", swapchain_format_count);
        std::vector<int64_t> swapchain_formats(swapchain_format_count);
        result = xrEnumerateSwapchainFormats(session,
                        swapchain_format_count,
                        &swapchain_format_count,
                        swapchain_formats.data());

        if(!XR_SUCCEEDED(result)){
                throw std::runtime_error("Failed to enumerate swapchain formats!");
        }

        return swapchain_formats;
}

static glm::mat4 get_proj_mat(const XrFovf& fov, float zNear, float zFar){
        glm::mat4 res{};

        float tanLeft = glm::tan(fov.angleLeft);
        float tanRight = glm::tan(fov.angleRight);
        float tanUp = glm::tan(fov.angleUp);
        float tanDown = glm::tan(fov.angleDown);

        res[0][0] = 2.f / (tanRight - tanLeft);
        res[1][1] = 2.f / (tanUp - tanDown);
        res[2][0] = (tanRight + tanLeft) / (tanRight - tanLeft); 
        res[2][1] = (tanUp + tanDown) / (tanUp - tanDown);
        res[2][2] = -(zFar + zNear) / (zFar - zNear);
        res[2][3] = -1;
        res[3][2] = -(2.f * zFar * zNear) / (zFar - zNear);

        return res;
}

static int64_t select_swapchain_fmt(const std::vector<int64_t>& swapchain_formats){
        const int64_t preferred_formats[] = {
                GL_RGBA8_EXT,
                GL_SRGB8_ALPHA8_EXT
        };

        for(auto pref : preferred_formats){
                for(auto supported_fmt : swapchain_formats){
                        if(pref == supported_fmt){
                                return pref;
                        }
                }
        }

        return 0;
}

static void map_new_buffer(struct video_frame *f){
        glBufferData(GL_PIXEL_UNPACK_BUFFER, f->tiles[0].data_len, 0, GL_STREAM_DRAW);
        f->tiles[0].data = (char *) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        GLenum ret = glGetError();
        if(ret != GL_NO_ERROR){
                std::cerr << "Error mapping: " << ret << std::endl;
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void recycle_frame(video_frame *f){
        GlBuffer *pbo = static_cast<GlBuffer *>(f->callbacks.dispose_udata);
        if(!pbo){
                return;
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->get());
        if(f->tiles[0].data){
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        map_new_buffer(f);
}

static void delete_frame(video_frame *f){
        GlBuffer *pbo = static_cast<GlBuffer *>(f->callbacks.dispose_udata);
        vf_free(f);

        if(!pbo){
                return;
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->get());
        if(f->tiles[0].data){
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
         
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        delete pbo;
}

static video_frame *allocate_frame(state_xrgl *s){
        video_frame *buffer = vf_alloc_desc(s->current_desc);
        GlBuffer *pbo = new GlBuffer();
        buffer->callbacks.dispose_udata = pbo;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->get());

        map_new_buffer(buffer);

        return buffer;
}

static void worker(state_xrgl *s){
        PROFILE_FUNC;
        s->window.make_worker_context_current();
        std::unique_lock<std::mutex> lk(s->lock);
        for(size_t i = 0; i < MAX_BUFFER_SIZE; i++){
                video_frame *buf = vf_alloc(1);
                GlBuffer *pbo = new GlBuffer();
                buf->callbacks.dispose_udata = pbo;
                s->free_frame_pool.push_back(buf);
        }
        lk.unlock();
        s->free_frame_ready_cv.notify_all();

        bool running = true;
        lk.lock();
        while(running){
                PROFILE_DETAIL("Wait for frame");
                s->new_frame_ready_cv.wait(lk,
                                [s]{
                                return !s->frame_queue.empty() || !s->dispose_frame_pool.empty();
                                });
                if(!s->dispose_frame_pool.empty()){
                        PROFILE_DETAIL("Process disposed frames");
                        for(video_frame *f : s->dispose_frame_pool){
                                if (video_desc_eq(video_desc_from_frame(f), s->current_desc)) {
                                        recycle_frame(f);
                                        s->free_frame_pool.push_back(f);
                                } else {
                                        delete_frame(f);

                                        struct video_frame *buffer = allocate_frame(s);
                                        s->free_frame_pool.push_back(buffer);
                                }
                        }
                        s->dispose_frame_pool.clear();
                        s->free_frame_ready_cv.notify_all();
                }

                if(!s->frame_queue.empty()){
                        PROFILE_DETAIL("put_frame");
                        video_frame *frame = s->frame_queue.front();
                        s->frame_queue.pop();
                        lk.unlock();
                        if(!frame){
                                SDL_Event event;
                                event.type = SDL_QUIT;
                                SDL_PushEvent(&event);
                                running = false;
                        } else {
                                s->frame_consumed_cv.notify_one();
                                s->scene.put_frame(frame, frame->callbacks.dispose_udata != nullptr);

                                lk.lock();
                                s->dispose_frame_pool.push_back(frame);
                        }
                }
        }
}

static void display_xrgl_run(void *state){
        PROFILE_FUNC;

        state_xrgl *s = static_cast<state_xrgl *>(state);

        //Make sure the worker context is initialized before starting worker
        s->window.make_worker_context_current();
        s->window.make_render_context_current();

        std::thread worker_thread(worker, s);

        Display *xDisplay = nullptr;
        GLXContext glxContext;
        GLXDrawable glxDrawable;

        s->window.getXlibHandles(&xDisplay, &glxContext, &glxDrawable);

        Openxr_session session(s->xr_state.instance.get(),
                        s->xr_state.system_id,
                        xDisplay,
                        glxContext,
                        glxDrawable);

        std::vector<XrViewConfigurationView> config_views = get_views(s->xr_state);

        Openxr_local_space space(session.get());

        session.begin();

        std::vector<int64_t> swapchain_fmts = get_swapchain_formats(session.get());

        int64_t selected_swapchain_fmt = select_swapchain_fmt(swapchain_fmts);

        std::vector<Gl_interop_swapchain> swapchains;
        for(const auto& view : config_views){
                swapchains.emplace_back(session.get(),
                                selected_swapchain_fmt,
                                view.recommendedImageRectWidth,
                                view.recommendedImageRectHeight);
        }

        size_t view_count = config_views.size();
        std::vector<XrCompositionLayerProjectionView> projection_views(view_count);


        XrCompositionLayerProjection projection_layer;
        projection_layer.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
        projection_layer.next = nullptr;
        projection_layer.layerFlags = 0;
        projection_layer.space = space.get();
        projection_layer.viewCount = view_count;
        projection_layer.views = projection_views.data();

        std::vector<XrView> views(view_count);

        for(auto& view : views){
                view.type = XR_TYPE_VIEW;
                view.next = nullptr;
        }

        for(unsigned i = 0; i < view_count; i++){
                projection_views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
                projection_views[i].next = nullptr;
                projection_views[i].subImage.swapchain = swapchains[i].get();
                projection_views[i].subImage.imageArrayIndex = 0;
                projection_views[i].subImage.imageRect.offset.x = 0;
                projection_views[i].subImage.imageRect.offset.y = 0;
                projection_views[i].subImage.imageRect.extent.width =
                        config_views[i].recommendedImageRectWidth;
                projection_views[i].subImage.imageRect.extent.height =
                        config_views[i].recommendedImageRectHeight;

        }

        bool running = true;
        if(selected_swapchain_fmt == GL_SRGB8_ALPHA8_EXT){
                /* Convert to sRGB to correctly display on the HMD.
                 * This however breaks the preview window colors, because
                 * the framebuffer is directly copied into preview window
                 * using glBlitNamedFramebuffer
                 */
                glEnable(GL_FRAMEBUFFER_SRGB);
        }

        glm::mat4 view_reset_rot = glm::mat4(1.0f);

        while(running){
                XrResult result;

                XrFrameState frame_state{};
                frame_state.type = XR_TYPE_FRAME_STATE;
                frame_state.next = nullptr;

                XrFrameWaitInfo frame_wait_info{};
                frame_wait_info.type = XR_TYPE_FRAME_WAIT_INFO;
                frame_wait_info.next = nullptr;

                PROFILE_DETAIL("wait frame");
                result = xrWaitFrame(session.get(), &frame_wait_info, &frame_state);
                if (!XR_SUCCEEDED(result)){
                        log_msg(LOG_LEVEL_ERROR, "Failed to xrWaitFrame\n");
                        break;
                }

                bool reset_view = false;

                SDL_Event event;
                while(SDL_PollEvent(&event)){
                        switch(event.type){
                        case SDL_QUIT:
                                running = false;
                                break;
                        case SDL_WINDOWEVENT:
                                if(event.window.event == SDL_WINDOWEVENT_RESIZED){
                                        s->window.width = event.window.data1;
                                        s->window.height = event.window.data2;
                                }
                                break;
                        case SDL_KEYDOWN:
                        case SDL_KEYUP:
                                switch(event.key.keysym.sym){
                                case SDLK_q:
                                        running = false;
                                        break;
                                case SDLK_v:
                                        if(event.type == SDL_KEYUP){
                                                reset_view = true;
                                        }
                                        break;
                                }
                                break;
                        default:
                                break;
                        }
                }

                while(true){
                        XrEventDataBuffer xr_event{};
                        xr_event.type = XR_TYPE_EVENT_DATA_BUFFER;
                        xr_event.next = nullptr;

                        result = xrPollEvent(s->xr_state.instance.get(), &xr_event);
                        if(result != XR_SUCCESS){
                                break;
                        }

                        switch(xr_event.type){
                        case XR_TYPE_EVENT_DATA_EVENTS_LOST:
                                log_msg(LOG_LEVEL_WARNING, "OpenXR events lost!\n");
                                break;
                        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                                log_msg(LOG_LEVEL_WARNING, "OpenXR instance loss pending!\n");
                                running = false;
                                break;
                        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
                                {
                                        XrEventDataSessionStateChanged* event = (XrEventDataSessionStateChanged*) &xr_event;
                                        if(event->state >= XR_SESSION_STATE_STOPPING){
                                                log_msg(LOG_LEVEL_NOTICE, "Received event requesting stop\n");
                                                running = false;
                                        }
                                        break;
                                }
                        default:
                                log_msg(LOG_LEVEL_NOTICE, "Unhandled event %d\n", xr_event.type);
                                break;
                        }

                }

                XrViewLocateInfo view_locate_info;
                view_locate_info.type = XR_TYPE_VIEW_LOCATE_INFO;
                view_locate_info.displayTime = frame_state.predictedDisplayTime;
                view_locate_info.space = space.get();

                XrViewState view_state;
                view_state.type = XR_TYPE_VIEW_STATE;
                view_state.next = nullptr;

                uint32_t located_views = 0;
                result = xrLocateViews(session.get(),
                                &view_locate_info,
                                &view_state,
                                view_count,
                                &located_views,
                                views.data());

                if (!XR_SUCCEEDED(result)){
                        log_msg(LOG_LEVEL_ERROR, "Failed to locate views!\n");
                        break;
                }

                /*
                   printf("View: %f %f %f %f, %f %f %f, fov = %f %f %f %f\n",
                   views[1].pose.orientation.x,
                   views[1].pose.orientation.y,
                   views[1].pose.orientation.z,
                   views[1].pose.orientation.w,
                   views[1].pose.position.x,
                   views[1].pose.position.y,
                   views[1].pose.position.z,
                   views[1].fov.angleLeft,
                   views[1].fov.angleRight,
                   views[1].fov.angleUp,
                   views[1].fov.angleDown);
                   */

                XrFrameBeginInfo frame_begin_info;
                frame_begin_info.type = XR_TYPE_FRAME_BEGIN_INFO;
                frame_begin_info.next = nullptr;

                PROFILE_DETAIL("begin frame");
                result = xrBeginFrame(session.get(), &frame_begin_info);
                if (!XR_SUCCEEDED(result)){
                        log_msg(LOG_LEVEL_ERROR, "Failed to begin frame!\n");
                        break;
                }

                for(unsigned i = 0; i < view_count; i++){
                        PROFILE_DETAIL("render view");

                        XrSwapchainImageAcquireInfo swapchain_image_acquire_info;
                        swapchain_image_acquire_info.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
                        swapchain_image_acquire_info.next = nullptr;

                        uint32_t buf_idx;
                        result = xrAcquireSwapchainImage(
                                        swapchains[i].get(),
                                        &swapchain_image_acquire_info,
                                        &buf_idx);

                        if(!XR_SUCCEEDED(result)){
                                log_msg(LOG_LEVEL_ERROR, "Failed to acquire swapchain image!\n");
                                break;
                        }

                        XrSwapchainImageWaitInfo swapchain_image_wait_info;
                        swapchain_image_wait_info.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
                        swapchain_image_wait_info.next = nullptr;
                        swapchain_image_wait_info.timeout = 1000;

                        result = xrWaitSwapchainImage(swapchains[i].get(), &swapchain_image_wait_info);
                        if(!XR_SUCCEEDED(result)){
                                log_msg(LOG_LEVEL_ERROR, "failed to wait for swapchain image!\n");
                                break;
                        }

                        projection_views[i].pose = views[i].pose;
                        projection_views[i].fov = views[i].fov;
                        unsigned w = config_views[i].recommendedImageRectWidth;
                        unsigned h = config_views[i].recommendedImageRectHeight;

                        glm::mat4 projMat = get_proj_mat(views[i].fov, 0.05f, 100.f);
                        //glm::mat4 projMat = glm::perspective(glm::radians(70.f), (float) w /h, 0.1f, 300.f);
                        const auto& rot = views[i].pose.orientation;
                        glm::mat4 viewMat = glm::mat4_cast(glm::quat(rot.w, rot.x, rot.y, rot.z));
                        if(reset_view){
                                reset_view = false;
                                view_reset_rot = viewMat;
                        }
                        glm::mat4 pvMat = projMat * glm::inverse(viewMat) * view_reset_rot;

                        auto framebuffer = swapchains[i].get_framebuffer(buf_idx).get();
                        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                        glClear(GL_COLOR_BUFFER_BIT);

                        s->scene.render(w, h, pvMat);

                        glBindFramebuffer(GL_FRAMEBUFFER, 0);

                        if(i % 2){
                                int dstX0;
                                int dstY0;
                                int dstX1;
                                int dstY1;
                                if(w * s->window.height > s->window.width * h){
                                        dstX0 = 0;
                                        dstX1 = s->window.width;

                                        int height = (s->window.width * h) / w;
                                        dstY0 = (s->window.height - height) / 2;
                                        dstY1 = dstY0 + height;
                                } else {
                                        int width = (s->window.height * w) / h;
                                        dstX0 = (s->window.width - width) / 2;
                                        dstX1 = dstX0 + width;

                                        dstY0 = 0;
                                        dstY1 = s->window.height;

                                }
                                glClear(GL_COLOR_BUFFER_BIT);
                                glBlitNamedFramebuffer((GLuint)framebuffer, // readFramebuffer
                                                (GLuint)0,    // backbuffer     // drawFramebuffer
                                                (GLint)0,     // srcX0
                                                (GLint)0,     // srcY0
                                                (GLint)w,     // srcX1
                                                (GLint)h,     // srcY1
                                                (GLint)dstX0,     // dstX0
                                                (GLint)dstY0,     // dstY0
                                                (GLint)dstX1, // dstX1
                                                (GLint)dstY1, // dstY1
                                                (GLbitfield)GL_COLOR_BUFFER_BIT, // mask
                                                (GLenum)GL_LINEAR);              // filter

                                SDL_GL_SwapWindow(s->window.sdl_window);
                        }

                        PROFILE_DETAIL("glFinish");
                        glFinish();

                        PROFILE_DETAIL("release swapchain");
                        XrSwapchainImageReleaseInfo swapchain_image_release_info;
                        swapchain_image_release_info.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;
                        swapchain_image_release_info.next = nullptr;

                        result = xrReleaseSwapchainImage(
                                        swapchains[i].get(),
                                        &swapchain_image_release_info);

                        if (!XR_SUCCEEDED(result)){
                                log_msg(LOG_LEVEL_ERROR, "Failed to release swapchain image!\n");
                                break;
                        }
                }

                const XrCompositionLayerBaseHeader *composition_layers = (const XrCompositionLayerBaseHeader *) &projection_layer; 
                XrFrameEndInfo frame_end_info;
                frame_end_info.type = XR_TYPE_FRAME_END_INFO;
                frame_end_info.displayTime = frame_state.predictedDisplayTime;
                frame_end_info.layerCount = 1;
                frame_end_info.layers = &composition_layers;
                frame_end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
                frame_end_info.next = nullptr;

                PROFILE_DETAIL("End Frame");
                result = xrEndFrame(session.get(), &frame_end_info);
                if (!XR_SUCCEEDED(result)){
                        log_msg(LOG_LEVEL_ERROR, "Failed to end frame!\n");
                        break;
                }

        }

        exit_uv(0);
        worker_thread.join();
}

static void * display_xrgl_init(struct module *parent, const char *fmt, unsigned int flags) {
        UNUSED(parent);
        UNUSED(fmt);
        UNUSED(flags);
        state_xrgl *s = new state_xrgl();

        XrSystemGetInfo system_get_info;
        system_get_info.type = XR_TYPE_SYSTEM_GET_INFO;
        system_get_info.next = nullptr;
        system_get_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

        XrResult result = xrGetSystem(s->xr_state.instance.get(),
                        &system_get_info,
                        &s->xr_state.system_id);

        if(!XR_SUCCEEDED(result)){
                throw std::runtime_error("No available device found!");
        }

        s->free_frame_pool.reserve(MAX_BUFFER_SIZE);
        s->dispose_frame_pool.reserve(MAX_BUFFER_SIZE);

        return s;
}

static void display_xrgl_done(void *state) {
        state_xrgl *s = static_cast<state_xrgl *>(state);

        delete s;
}

static struct video_frame * display_xrgl_getf(void *state) {
        struct state_xrgl *s = static_cast<state_xrgl *>(state);

        std::unique_lock<std::mutex> lock(s->lock);

        while (true) {
                s->free_frame_ready_cv.wait(lock, [s]{return s->free_frame_pool.size() > 0;});
                struct video_frame *buffer = s->free_frame_pool.back();
                s->free_frame_pool.pop_back();
                if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
                        return buffer;
                } else {
                        s->dispose_frame_pool.push_back(buffer);
                        s->new_frame_ready_cv.notify_one();
                }
        }
}

static int display_xrgl_putf(void *state, struct video_frame *frame, int nonblock) {
        struct state_xrgl *s = static_cast<state_xrgl *>(state);

        std::unique_lock<std::mutex> lk(s->lock);

        if(!frame) {
                s->frame_queue.push(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return 0;
        }

        if (nonblock == PUTF_DISCARD) {
                s->dispose_frame_pool.push_back(frame);
                s->new_frame_ready_cv.notify_one();
                return 0;
        }
        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK) {
                s->dispose_frame_pool.push_back(frame);
                s->new_frame_ready_cv.notify_one();
                return 1;
        }
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.push(frame);

        lk.unlock();
        s->new_frame_ready_cv.notify_one();

        return 0;
}

static int display_xrgl_reconfigure(void *state, struct video_desc desc) {
        state_xrgl *s = static_cast<state_xrgl *>(state);

        s->current_desc = desc;
        return 1;
}

static int display_xrgl_get_property(void *state, int property, void *val, size_t *len) {
        UNUSED(state);
        codec_t codecs[] = {
                RGBA,
                RGB,
                UYVY,
        };
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static void display_xrgl_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)){
        UNUSED(deleter);
        *count = 0;
        *available_cards = nullptr;

        Openxr_instance instance;

        XrSystemGetInfo systemGetInfo;
        systemGetInfo.type = XR_TYPE_SYSTEM_GET_INFO;
        systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        systemGetInfo.next = NULL;

        XrSystemId systemId;
        XrResult result = xrGetSystem(instance.get(), &systemGetInfo, &systemId);
        if(!XR_SUCCEEDED(result)){
                return;
        }

        XrSystemProperties systemProperties;
        systemProperties.type = XR_TYPE_SYSTEM_PROPERTIES;
        systemProperties.next = NULL;
        systemProperties.graphicsProperties = {};
        systemProperties.trackingProperties = {};

        result = xrGetSystemProperties(instance.get(), systemId, &systemProperties);
        if(!XR_SUCCEEDED(result)){
                return;
        }

        *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
        *count = 1;
        snprintf((*available_cards)[0].id, sizeof((*available_cards)[0].id), "openxr_gl:system=%lu", systemId);
        snprintf((*available_cards)[0].name, sizeof((*available_cards)[0].name), "OpenXr: %s", systemProperties.systemName);
        (*available_cards)[0].repeatable = false;
}


static const struct video_display_info openxr_gl_info = {
        display_xrgl_probe,
        display_xrgl_init,
        display_xrgl_run,
        display_xrgl_done,
        display_xrgl_getf,
        display_xrgl_putf,
        display_xrgl_reconfigure,
        display_xrgl_get_property,
        NULL,
        NULL,
        DISPLAY_NEEDS_MAINLOOP,
};

REGISTER_MODULE(openxr_gl, &openxr_gl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

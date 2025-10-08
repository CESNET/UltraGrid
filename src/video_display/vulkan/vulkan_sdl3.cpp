/**
 * @file   video_display/vulkan_sdl3.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Bela      <492789@mail.muni.cz>
 */
 /*
  * Copyright (c) 2018-2025 CESNET
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
  /**
   * @file
   * @todo
   * Missing from SDL1:
   * * audio (would be perhaps better as an audio playback device)
   */

#include <cctype>                                          // for toupper
#include <cmath>                                           // for sqrt
#include <cstdio>                                          // for sscanf
#include <cstdlib>                                         // for calloc
#include <cstring>                                         // for strchr, strcmp

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "utils/color_out.h"
#include "utils/macros.h"                                  // for IS_KEY_PREFIX
#include "video_display.h"
#include "video_display/splashscreen.h"
#include "video.h"
//remove leaking macros
#undef min
#undef max

#include "vulkan_display.hpp" // vulkan.h must be before GLFW/SDL

// @todo remove the defines when no longer needed
#ifdef __arm64__
#define SDL_DISABLE_MMINTRIN_H 1
#define SDL_DISABLE_IMMINTRIN_H 1
#endif // defined __arm64__

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility> // pair

namespace {

namespace vkd = vulkan_display;
namespace chrono = std::chrono;
using namespace std::literals;

constexpr int magic_vulkan_sdl3 = 0x33536b56; // 'VkS3'
constexpr int initial_frame_count = 0;
#define MOD_NAME "[Vulkan SDL3] "

#define SDL_CHECK(cmd, ...) \
        do { \
                if (!(cmd)) { \
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error (%s): %s\n", \
                                #cmd, SDL_GetError()); \
                        __VA_ARGS__; \
                } \
        } while (0)

void display_vulkan_new_message(module*);
video_frame* display_vulkan_getf(void* state);

class WindowCallback final : public vkd::WindowChangedCallback {
        SDL_Window* window = nullptr;
public:
        explicit WindowCallback(SDL_Window* window):
                window{window} { }
        
        vkd::WindowParameters get_window_parameters() override {
                assert(window);
                int width = 0;
                int height = 0;
                SDL_CHECK(SDL_GetWindowSizeInPixels(window, &width, &height));
                if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
                        width = 0;
                        height = 0;
                }
                return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
        }
};

struct FrameMappings{
        struct Mapping{
                video_frame* frame;
                vkd::TransferImage image;
        };

        std::vector<Mapping> mappings;

        video_frame* create_frame(vkd::TransferImage image){
                for(auto& pair : mappings){
                        if(pair.image == nullptr){
                                pair.image = image;
                                return pair.frame;
                        }
                }
                auto new_frame = vf_alloc(1);
                mappings.emplace_back(Mapping{new_frame, image});
                return new_frame;
        }

        vkd::TransferImage get_image(video_frame* frame){
                for(auto& pair : mappings){
                        if(pair.frame == frame){
                                return pair.image;
                        }
                }
                assert(false);
                return {};
        }

        ~FrameMappings(){
                for(auto& pair : mappings){
                        vf_free(pair.frame);
                }
        }
};

struct state_vulkan_sdl3 {
        const uint32_t magic = magic_vulkan_sdl3;
        module mod{};

        Uint32 sdl_user_new_message_event;

        chrono::steady_clock::time_point time{};
        uint64_t frames = 0;

        bool deinterlace = false;
        bool fullscreen = false;
        bool keep_aspect = false;

        int width = 0;
        int height = 0;
        
        SDL_Window* window = nullptr;

        // Use raw pointers because std::unique_ptr might not have standard layout
        vkd::VulkanDisplay* vulkan = nullptr;
        WindowCallback* window_callback = nullptr;
        FrameMappings* frame_mappings = new FrameMappings();

        std::atomic<bool> should_exit = false;
        video_desc current_desc{};

        explicit state_vulkan_sdl3(module* parent) {
                module_init_default(&mod);
                mod.new_message = display_vulkan_new_message;
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);

                sdl_user_new_message_event = SDL_RegisterEvents(1);
                assert(sdl_user_new_message_event != static_cast<Uint32>(-1));
        }

        state_vulkan_sdl3(const state_vulkan_sdl3& other) = delete;
        state_vulkan_sdl3& operator=(const state_vulkan_sdl3& other) = delete;
        state_vulkan_sdl3(state_vulkan_sdl3&& other) = delete;
        state_vulkan_sdl3& operator=(state_vulkan_sdl3&& other) = delete;

        ~state_vulkan_sdl3() {
                delete frame_mappings;
                module_done(&mod);
        }
};

// make sure that state_vulkan_sdl3 is C compatible
static_assert(std::is_standard_layout_v<state_vulkan_sdl3>);

//todo C++20 : change to to_array
constexpr std::array<std::pair<char, std::string_view>, 3> display_vulkan_keybindings{{
        {'d', "toggle deinterlace"},
        {'f', "toggle fullscreen"},
        {'q', "quit"}
}};

constexpr void update_description(const video_desc& video_desc, video_frame& frame) {
        frame.color_spec = video_desc.color_spec;
        frame.fps = video_desc.fps;
        frame.interlacing = video_desc.interlacing;
        frame.tile_count = video_desc.tile_count;
        for (unsigned i = 0; i < video_desc.tile_count; i++) {
                frame.tiles[i].width = video_desc.width;
                frame.tiles[i].height = video_desc.height;
        }
}

constexpr int64_t translate_sdl_key_to_ug(SDL_KeyboardEvent keyev) {
        keyev.mod &=
            ~(SDL_KMOD_NUM | SDL_KMOD_CAPS); // remove num+caps lock modifiers

        // ctrl alone -> do not interpret
        if (keyev.key == SDLK_LCTRL || keyev.key == SDLK_RCTRL) {
                return 0;
        }

        bool ctrl = false;
        bool shift = false;
        if (keyev.mod & SDL_KMOD_CTRL) {
                ctrl = true;
        }
        keyev.mod &= ~SDL_KMOD_CTRL;

        if (keyev.mod & SDL_KMOD_SHIFT) {
                shift = true;
        }
        keyev.mod &= ~SDL_KMOD_SHIFT;

        if (keyev.mod != 0) {
                return -1;
        }

        if ((keyev.key & SDLK_SCANCODE_MASK) == 0) {
                if (shift) {
                        keyev.key = toupper(keyev.key);
                }
                return ctrl ? K_CTRL(keyev.key) : keyev.key;
        }
        switch (keyev.key) {
                case SDLK_RIGHT: return K_RIGHT;
                case SDLK_LEFT:  return K_LEFT;
                case SDLK_DOWN:  return K_DOWN;
                case SDLK_UP:    return K_UP;
                case SDLK_PAGEDOWN:    return K_PGDOWN;
                case SDLK_PAGEUP:    return K_PGUP;
        }
        return -1;
}

constexpr bool display_vulkan_process_key(state_vulkan_sdl3& s, int64_t key) {
        switch (key) {
                case 'd':
                        s.deinterlace = !s.deinterlace;
                        log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n",
                                s.deinterlace ? "ON" : "OFF");
                        return true;
                case 'f': {
                        s.fullscreen = !s.fullscreen;
                        float mouse_x = 0, mouse_y = 0;
                        SDL_GetGlobalMouseState(&mouse_x, &mouse_y);
                        SDL_CHECK(SDL_SetWindowFullscreen(s.window, s.fullscreen ? SDL_WINDOW_FULLSCREEN : 0));
                        SDL_CHECK(SDL_WarpMouseGlobal(mouse_x, mouse_y));
                        return true;
                }
                case 'q':
                        exit_uv(0);
                        return true;
                default:
                        return false;
        }
}

void log_and_exit_uv(std::exception& e) {
        LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << std::endl;
        exit_uv(EXIT_FAILURE);
}

void process_user_messages(state_vulkan_sdl3& s) {
        msg_universal* msg = nullptr;
        while ((msg = reinterpret_cast<msg_universal*>(check_message(&s.mod)))) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received message: %s\n", msg->text);
                response* r = nullptr;
                int key;
                if (strstr(msg->text, "win-title ") == msg->text) {
                        SDL_CHECK(SDL_SetWindowTitle(s.window, msg->text + strlen("win-title ")));
                        r = new_response(RESPONSE_OK, NULL);
                } else if (sscanf(msg->text, "%d", &key) == 1) {
                        if (!display_vulkan_process_key(s, key)) {
                                r = new_response(RESPONSE_BAD_REQUEST, "Unsupported key for SDL");
                        } else {
                                r = new_response(RESPONSE_OK, NULL);
                        }
                } else {
                        r = new_response(RESPONSE_BAD_REQUEST, "Wrong command");
                }
                free_message(reinterpret_cast<message*>(msg), r);
        }
}

void process_events(state_vulkan_sdl3& s) {
        SDL_Event sdl_event;
        while (SDL_PollEvent(&sdl_event)) {
                if (sdl_event.type == s.sdl_user_new_message_event) {
                        process_user_messages(s);
                } else if (sdl_event.type == SDL_EVENT_KEY_DOWN && sdl_event.key.repeat == 0) {
                        auto& keysym = sdl_event.key;
                        log_msg(LOG_LEVEL_VERBOSE, 
                                MOD_NAME "Pressed key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                SDL_GetKeyName(keysym.key), 
                                keysym.scancode, 
                                keysym.key, 
                                keysym.mod);
                        int64_t sym = translate_sdl_key_to_ug(keysym);
                        if (sym > 0) {
                                if (!display_vulkan_process_key(s, sym)) {
                                        // unknown key -> pass to control
                                        keycontrol_send_key(get_root_module(&s.mod), sym);
                                }
                        } else if (sym == -1) {
                                log_msg(LOG_LEVEL_WARNING, 
                                        MOD_NAME "Cannot translate key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                        SDL_GetKeyName(keysym.key), 
                                        keysym.scancode, 
                                        keysym.key, 
                                        keysym.mod);
                        }

                } else if (s.keep_aspect && sdl_event.type == SDL_EVENT_WINDOW_RESIZED) {
                        // https://forums.libsdl.org/viewtopic.php?p=38342
                        int old_width = s.width, old_height = s.height;
                        auto width = sdl_event.window.data1;
                        auto height = sdl_event.window.data2;
                        if (old_width == width) {
                                width = old_width * height / old_height;
                        } else if (old_height == height) {
                                height = old_height * width / old_width;
                        } else {
                                auto area = int64_t{ width } * height;
                                width = sqrt(area * old_width / old_height);
                                height = sqrt(area * old_height / old_width);
                        }
                        s.width = width;
                        s.height = height;
                        SDL_CHECK(SDL_SetWindowSize(s.window, width, height));
                        debug_msg(MOD_NAME "resizing to %d x %d\n", width, height);
                } else if (sdl_event.type == SDL_EVENT_WINDOW_RESIZED)
                {
                        s.vulkan->window_parameters_changed();

                } else if (sdl_event.type == SDL_EVENT_QUIT) {
                        exit_uv(0);
                }
        }
}

void display_vulkan_run(void* state) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);
        
        s->time = chrono::steady_clock::now();
        while (!s->should_exit) {
                process_events(*s);
                try {
                        bool displayed = s->vulkan->display_queued_image();
                        if (displayed) {
                                s->frames++;
                        }
                } 
                catch (std::exception& e) { log_and_exit_uv(e); break; }
                auto now = chrono::steady_clock::now();
                double seconds = chrono::duration<double>{ now - s->time }.count();
                if (seconds > 5) {
                        display_print_fps(MOD_NAME, seconds, (int) s->frames,
                                          s->current_desc.fps);

                        s->time = now;
                        s->frames = 0;
                }
        }
        SDL_CHECK(SDL_HideWindow(s->window));

}

void sdl3_print_displays() {
        int            count    = 0;
        SDL_DisplayID *displays = SDL_GetDisplays(&count);
        for (int i = 0; i < count; ++i) {
                if (i > 0) {
                        std::cout << ", ";
                }
                const char* dname = SDL_GetDisplayName(displays[i]);
                if (dname == nullptr) {
                        dname = SDL_GetError();
                }
                col() << SBOLD(i) << " - " << dname << " (ID: " << displays[i] << ")" ;
        }
        std::cout << "\n";
}

void print_gpus() {
        vkd::VulkanInstance instance;
        std::vector<const char*>required_extensions{};
        std::vector<std::pair<std::string, bool>> gpus{};
        try {
                instance.init(required_extensions, false);
                instance.get_available_gpus(gpus);
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return; }
        
        col() << "\n\tVulkan GPUs:\n";
        uint32_t counter = 0;
        for (const auto& gpu : gpus) {
                col() << (gpu.second ? "" : TERM_FG_RED);
                col() << "\t\t " << counter++ << "\t - " << gpu.first;
                col() << (gpu.second ? "" : " - unsuitable") << TERM_RESET << '\n';
        }
}

void show_help() {
        SDL_CHECK(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));

        auto print_drivers = []() {
                for (int i = 0; i < SDL_GetNumVideoDrivers(); ++i) {
                        col() << (i == 0 ? "" : ", ") << SBOLD(SDL_GetVideoDriver(i));
                }
                col() << '\n';
        };
        
        col() << "VULKAN_SDL3 options:\n";
        col() << SBOLD(SRED("\t-d vulkan")
                        << "[:d|:fs|:keep-aspect|:nocursor|:nodecorate|:novsync|:tearing|:validation|:display=<d>|"
                        ":driver=<drv>|:gpu=<gpu_id>|:pos=<x>,<y>|:size=<W>x<H>|:window_flags=<f>|:help])") << "\n";

        col() << ("\twhere:\n");

        col() << SBOLD("\t               d") << " - deinterlace\n";
        col() << SBOLD("\t              fs") << " - fullscreen\n";
        
        col() << SBOLD("\t     keep-aspect") << " - keep window aspect ratio respective to the video\n";
        col() << SBOLD("\t        nocursor") << " - hides cursor\n";
        col() << SBOLD("\t      nodecorate") << " - disable window border\n";
        col() << SBOLD("\t         novsync") << " - disable vsync\n";
        col() << SBOLD("\t         tearing") << " - permits screen tearing\n";
        col() << SBOLD("\t      validation") << " - enable vulkan validation layers\n";

        col() << SBOLD("\t     display=<d>") << " - display index or ID, available indices: ";
        sdl3_print_displays();
        col() << SBOLD("\t    driver=<drv>") << " - available drivers: ";
        print_drivers();
        col() << SBOLD("\t    gpu=<gpu_id>") << " - gpu index selected from the following list or keywords " << SBOLD("integrated") " or " << SBOLD("discrete") "\n";
        col() << SBOLD("\t     pos=<x>,<y>") << " - set window position\n";
        col() << SBOLD("\t    size=<W>x<H>") << " - set window size\n";
        col() << SBOLD("\twindow_flags=<f>") << " - flags to be passed to SDL_CreateWindow (use prefix 0x for hex)\n";
        
        print_gpus();

        col() << "\n\tKeyboard shortcuts:\n";
        for (auto& binding : display_vulkan_keybindings) {
                col() << SBOLD("\t\t'" << binding.first) << "'\t - " << binding.second << "\n";
        }
        SDL_Quit();
}


struct CodecToVulkanFormat{
        codec_t ug_codec;
        vulkan_display::Format vulkan_format;
};


// Ultragrid to VulkanDisplay Format mapping
const std::vector<CodecToVulkanFormat>& get_ug_to_vkd_format_mapping(state_vulkan_sdl3& s){
        //the backup vkd::Format must follow the corresponding native vkd::Format 
        constexpr std::array<CodecToVulkanFormat, 10> format_mapping {{
                {RGBA, vkd::Format::RGBA8},
                {RGB,  vkd::Format::RGB8},
                {UYVY, vkd::Format::UYVY8_422},
                {UYVY, vkd::Format::UYVY8_422_conv},
                {YUYV, vkd::Format::YUYV8_422},
                {VUYA, vkd::Format::VUYA8_4444_conv},
                {Y216, vkd::Format::YUYV16_422},
                {Y416, vkd::Format::UYVA16_4444_conv},
                {R10k, vkd::Format::RGB10A2_conv},
                {RG48, vkd::Format::RGB16},
        }};

        static std::vector<CodecToVulkanFormat> supported_formats_mapping{};
        if(!supported_formats_mapping.empty()){
                return supported_formats_mapping;
        }

        for(auto& pair: format_mapping){
                try {
                        if (!supported_formats_mapping.empty() && supported_formats_mapping.back().ug_codec == pair.ug_codec){
                                continue;
                        }
                        bool format_supported = s.vulkan->is_image_description_supported({1920, 1080, pair.vulkan_format });
                        if (format_supported) {
                                supported_formats_mapping.push_back(pair);
                        }
                }
                catch (std::exception& e) { log_and_exit_uv(e); }
        }
        return supported_formats_mapping;
}


vkd::ImageDescription to_vkd_image_desc(const video_desc& ultragrid_desc, state_vulkan_sdl3& s) {
        auto& mapping = get_ug_to_vkd_format_mapping(s);
        codec_t searched_codec = ultragrid_desc.color_spec;
        auto iter = std::find_if(mapping.begin(), mapping.end(),
                [searched_codec](auto pair) { return pair.ug_codec == searched_codec; });
        assert(iter != mapping.end());
        return { ultragrid_desc.width, ultragrid_desc.height, iter->vulkan_format };
}


bool display_vulkan_reconfigure(void* state, video_desc desc) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);

        assert(desc.tile_count == 1);
        s->current_desc = desc;
        if (!s->vulkan->is_image_description_supported(to_vkd_image_desc(desc, *s))){
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Video description is not supported - frames are too big or format isn't supported.");
                return false;
        }
        return true;
}

/**
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
void draw_splashscreen(state_vulkan_sdl3& s) {
        vkd::TransferImage image;
        try {
                image = s.vulkan->acquire_image({splash_width, splash_height, vkd::Format::RGBA8});
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return; }
        const char* source = splash_data;
        char* dest = reinterpret_cast<char*>(image.get_memory_ptr());
        auto padding = image.get_row_pitch() - splash_width * 4;

        for (unsigned row = 0; row < splash_height; row++) {
                for (unsigned int col = 0; col < splash_width; col++) {
                        HEADER_PIXEL(source, dest);
                        dest += 4;
                }
                dest += padding;
        }
        try {
                s.vulkan->queue_image(image, true);
                s.vulkan->display_queued_image();
        } 
        catch (std::exception& e) { log_and_exit_uv(e); }
}

struct command_line_arguments {
        bool cursor = true;
        bool help = false;
        bool vsync = true;
        bool tearing_permitted = false;
        bool validation = false;

        int display_idx = -1;
        int x = SDL_WINDOWPOS_UNDEFINED;
        int y = SDL_WINDOWPOS_UNDEFINED;

        uint32_t window_flags = 0 ; ///< user requested flags
        uint32_t gpu_idx = vkd::no_gpu_selected;
        std::string driver{};
};

bool parse_command_line_arguments(command_line_arguments& args, state_vulkan_sdl3& s, char *fmt) {
        constexpr std::string_view wrong_option_msg = MOD_NAME "Wrong option: ";

        char *token = nullptr;
        char *saveptr = nullptr;
        while ((token = strtok_r(fmt, ":", &saveptr))) try {
                fmt = nullptr;
                //cstoi = cstring to int (checked)
                auto cstoi = [token](const char *cstr,
                                     size_t      exp_size = 0) -> int {
                        std::size_t endpos = 0;
                        int result = std::stoi(cstr, &endpos, 0);
                        if (exp_size == 0) {
                                exp_size = strlen(cstr);
                        }
                        if (endpos != exp_size) {
                                throw std::runtime_error{
                                        std::string(token) +
                                        " (error parsing number)"
                                };
                        }
                        return result;
                };
                if (strcmp(token, "help") == 0) {
                        show_help();
                        args.help = true;
                } else if (strcmp(token, "d") == 0) {
                        s.deinterlace = true;
                } else if (strcmp(token, "fs") == 0) {
                        s.fullscreen = true;
                } else if (strcmp(token, "keep-aspect") == 0) {
                        s.keep_aspect = true;
                } else if (strcmp(token, "nocursor") == 0) {
                        args.cursor = false;
                } else if (strcmp(token, "nodecorate") == 0) {
                        args.window_flags |= SDL_WINDOW_BORDERLESS;
                } else if (strcmp(token, "novsync") == 0) {
                        args.vsync = false;
                } else if (strcmp(token, "tearing") == 0) {
                        args.tearing_permitted = true;
                } else if (strcmp(token, "validation") == 0) {
                        args.validation = true;
                } else if (IS_KEY_PREFIX(token, "display")) {
                        args.display_idx = cstoi(strchr(token, '=') + 1);
                } else if (IS_KEY_PREFIX(token, "driver")) {
                        args.driver = std::string{ strchr(token, '=') + 1};
                } else if (IS_KEY_PREFIX(token, "gpu")) {
                        char *val = strchr(token, '=') + 1;
                        if (strcmp(val, "integrated") == 0) {
                                args.gpu_idx = vulkan_display::gpu_integrated;
                        } else if (strcmp(val, "discrete") == 0) {
                                args.gpu_idx = vulkan_display::gpu_discrete;
                        } else {
                                args.gpu_idx = cstoi(val);
                        }
                } else if (IS_KEY_PREFIX(token, "pos")) {
                        if (strchr(token, ',') == nullptr) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Missing colon in option:" 
                                        << token << '\n';
                                return false;
                        }
                        token = strchr(token, '=') + 1;
                        args.x = cstoi(token, strchr(token, ',') - token);
                        token = strchr(token, ',') + 1;
                        args.y = cstoi(token);
                } else if (IS_KEY_PREFIX(token, "size")) {
                        if (strchr(token, 'x') == nullptr) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Missing deliminer 'x' in option:" 
                                        << token << '\n';
                                return false;
                        }
                        token = strchr(token, '=') + 1;
                        s.width = cstoi(token, strchr(token, 'x') - token);
                        token = strchr(token, 'x') + 1;
                        s.height = cstoi(token);
                } else if (IS_KEY_PREFIX(token, "window_flags")) {
                        token = strchr(token, '=') + 1;
                        args.window_flags |= cstoi(strchr(token, '=') + 1);
                } else {
                        LOG(LOG_LEVEL_ERROR) << wrong_option_msg << token << '\n';
                        return false;
                }
        }
        catch (std::exception& e) {
                LOG(LOG_LEVEL_ERROR) << wrong_option_msg << e.what() << '\n';
                return false;
        }
        return true;
}

void
sdl_set_log_level()
{
        if (log_level <= LOG_LEVEL_INFO) {
                return;
        }
        // SDL_INFO corresponds rather to UG verbose and SDL_VERBOSE is actually
        // more detailed than SDL_DEBUG so map this to UG debug
        SDL_SetLogPriorities(log_level == LOG_LEVEL_VERBOSE
                                 ? SDL_LOG_PRIORITY_INFO
                                 : SDL_LOG_PRIORITY_VERBOSE);
}

SDL_DisplayID
get_display_id_from_idx(int idx)
{
        int            count    = 0;
        SDL_DisplayID *displays = SDL_GetDisplays(&count);
        if (idx < count) {
                return displays[idx];
        }
        for (int i = 0; i < count; ++i) {
                if (displays[i] == (unsigned) idx) {
                        return idx;
                }
        }
        MSG(ERROR, "Display index %d out of range or ID invalid!\n", idx);
        return 0;
}

bool
vulkan_sdl3_set_window_position(state_vulkan_sdl3            *s,
                                const command_line_arguments *args)
{
        if (args->display_idx == -1 && args->x == SDL_WINDOWPOS_UNDEFINED &&
            args->y == SDL_WINDOWPOS_UNDEFINED) {
                return true; // nothing to set
        }
        int x = args->x;
        int y = args->y;
        if (args->display_idx != -1) {
                if (x != SDL_WINDOWPOS_UNDEFINED || y != SDL_WINDOWPOS_UNDEFINED) {
                        MSG(ERROR, "Do not set window positon and display at "
                                   "the same time!\n");
                        return false;
                }
                const SDL_DisplayID display_id =
                    get_display_id_from_idx(args->display_idx);
                if (display_id == 0) {
                        return false;
                }
                x = SDL_WINDOWPOS_CENTERED_DISPLAY(display_id);
                y = SDL_WINDOWPOS_CENTERED_DISPLAY(display_id);
        }
        if (SDL_SetWindowPosition(s->window, x, y)) {
                return true;
        }
        const bool is_wayland =
            strcmp(SDL_GetCurrentVideoDriver(), "wayland") == 0;
        if (is_wayland && args->display_idx != -1 && !s->fullscreen) {
                MSG(ERROR,
                    "In Wayland, display specification is possible only with "
                    "fullscreen flag ':fs' (%s)\n",
                    SDL_GetError());
        } else {
                MSG(ERROR, "Error (SDL_SetWindowPosition): %s\n",
                    SDL_GetError());
        }
        return false;
}

void* display_vulkan_init(module* parent, const char* fmt, unsigned int flags) {
        sdl_set_log_level();
        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                log_msg(LOG_LEVEL_ERROR, "UltraGrid VULKAN_SDL3 module currently doesn't support audio!\n");
                return nullptr;
        }

        auto s = std::make_unique<state_vulkan_sdl3>(parent);

        command_line_arguments args{};
        if (fmt) {
                std::string cpy = fmt;
                if (!parse_command_line_arguments(args, *s, cpy.data())) {
                        return nullptr;
                }
                if (args.help) {
                        return INIT_NOERR;
                }
        }

        if (!args.driver.empty()) {
                SDL_CHECK(SDL_SetHint(SDL_HINT_VIDEO_DRIVER, args.driver.c_str()));
        }
        bool ret = SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
        if (!ret) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL3: %s\n", SDL_GetError());
                return nullptr;
        }
        MSG(NOTICE, "Using driver: %s\n", SDL_GetCurrentVideoDriver());

        if (!args.cursor) {
                SDL_CHECK(SDL_HideCursor());
        }
        SDL_CHECK(SDL_DisableScreenSaver());

        for (auto& binding : display_vulkan_keybindings) {
                std::string msg = std::to_string(static_cast<int>(binding.first));
                keycontrol_register_key(&s->mod, binding.first, msg.c_str(), binding.second.data());
        }

        log_msg(LOG_LEVEL_NOTICE, "SDL3 initialized successfully.\n");

        const char* window_title = "UltraGrid - Vulkan Display";
        if (get_commandline_param("window-title")) {
                window_title = get_commandline_param("window-title");
        }

        if(s->width == -1 && s->height == -1){
                int display_index = 0;
                const SDL_DisplayMode *mode = SDL_GetDesktopDisplayMode(display_index);
                s->width = 0;
                s->height = 0;
                if(mode == nullptr){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get display size\n");
                } else {
                        s->width = mode->w;
                        s->height = mode->h;
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Detected display size: %dx%d\n", mode->w, mode->h);
                }
        }
        if (s->width == 0) s->width = 960;
        if (s->height == 0) s->height = 540;

        int window_flags = args.window_flags | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
        if (s->fullscreen) {
                window_flags |= SDL_WINDOW_FULLSCREEN;
        }

        s->window = SDL_CreateWindow(window_title, s->width, s->height, window_flags);
        if (!s->window) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to create window : %s\n", SDL_GetError());
                return nullptr;
        }
        s->window_callback = new WindowCallback(s->window);
        if (!vulkan_sdl3_set_window_position(s.get(), &args)) {
                return nullptr;
        }

        uint32_t extension_count = 0;
        const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
        std::vector<const char*> required_extensions(extensions, extensions + extension_count);
        assert(extension_count > 0);
        
        std::string path_to_shaders = vkd::get_shader_path();
        LOG(LOG_LEVEL_INFO) << MOD_NAME "Path to shaders: " << path_to_shaders << '\n';
        try {
                vkd::VulkanInstance instance;
                instance.init(required_extensions, args.validation);
                VkSurfaceKHR surface{};
                /// @todo check if not to provide the allocator
                if (!SDL_Vulkan_CreateSurface(s->window, (VkInstance) instance.get_instance(), nullptr, &surface)) {
                        std::cout << SDL_GetError() << std::endl;
                        throw std::runtime_error("SDL cannot create surface.");
                }
                s->vulkan = new vkd::VulkanDisplay{};
                s->vulkan->init(std::move(instance), vk::SurfaceKHR(surface),
                                initial_frame_count, *s->window_callback,
                                args.gpu_idx, std::move(path_to_shaders),
                                args.vsync, args.tearing_permitted);
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Vulkan display initialised." << std::endl;
        }
        catch (std::exception& e) { log_and_exit_uv(e); return nullptr; }

        draw_splashscreen(*s);
        return static_cast<void*>(s.release());
}

void display_vulkan_done(void* state) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);

        SDL_CHECK(SDL_ShowCursor());

        try {
                s->vulkan->destroy();
        }
        catch (std::exception& e) { log_and_exit_uv(e); }

        delete s->vulkan;
        s->vulkan = nullptr;
        delete s->window_callback;
        s->window_callback = nullptr;

        if (s->window) {
                SDL_DestroyWindow(s->window);
                s->window = nullptr;
        }

        SDL_Quit();

        delete s;
}

video_frame* display_vulkan_getf(void* state) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);
        
        const auto& desc = s->current_desc;
        vkd::TransferImage image;
        try {
                image = s->vulkan->acquire_image(to_vkd_image_desc(desc, *s));
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return nullptr; }
        video_frame& frame = *s->frame_mappings->create_frame(image);
        
        update_description(desc, frame);
        frame.tiles[0].data_len = image.get_row_pitch() * image.get_size().height;
        frame.tiles[0].data = reinterpret_cast<char*>(image.get_memory_ptr());
        return &frame;
}

bool display_vulkan_putf(void* state, video_frame* frame, long long timeout_ns) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);

        if (!frame) {
                s->should_exit = true;
                return true;
        }

        vkd::TransferImage image = s->frame_mappings->get_image(frame);

        if (timeout_ns == PUTF_DISCARD) {
                try {
                        s->vulkan->discard_image(image);
                } 
                catch (std::exception& e) { log_and_exit_uv(e); return false; }
                return true;
        }
        
        if (s->deinterlace) {
                image.set_process_function([](vkd::TransferImage& image) {
                        vc_deinterlace(reinterpret_cast<unsigned char*>(image.get_memory_ptr()),
                                image.get_row_pitch(),
                                image.get_size().height);
                });
        }
        
        try {
                return !s->vulkan->queue_image(image, timeout_ns != PUTF_BLOCKING);
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return 1; }
        return true;
}

bool display_vulkan_get_property(void* state, int property, void* val, size_t* len) {
        auto* s = static_cast<state_vulkan_sdl3*>(state);
        assert(s->magic == magic_vulkan_sdl3);

        switch (property) {
                case DISPLAY_PROPERTY_CODECS: {
                        auto& mapping = get_ug_to_vkd_format_mapping(*s);
                        std::vector<codec_t> codecs{};
                        codecs.reserve(mapping.size());
                        LOG(LOG_LEVEL_INFO) << MOD_NAME "Supported codecs are: ";
                        for (auto&[codec, vkd_fromat] : mapping) {
                                codecs.push_back(codec);
                                LOG(LOG_LEVEL_INFO) << get_codec_name(codec) << " ";
                        }
                        LOG(LOG_LEVEL_INFO) << std::endl;
                        size_t codecs_len = codecs.size() * sizeof(codec_t);
                        if (codecs_len > *len) {
                                return false;
                        }
                        memcpy(val, codecs.data(), codecs_len);
                        *len = codecs_len;
                        break;
                }
                case DISPLAY_PROPERTY_BUF_PITCH: {
                        if (sizeof(int) > *len) {
                                return false;
                        }
                        *len = sizeof(int);

                        vkd::TransferImage image;
                        const auto& desc = s->current_desc;
                        assert(s->current_desc.width != 0);
                        try {
                                image = s->vulkan->acquire_image(to_vkd_image_desc(desc, *s));
                                auto value = static_cast<int>(image.get_row_pitch());
                                memcpy(val, &value, sizeof(value));
                                s->vulkan->discard_image(image);
                        } 
                        catch (std::exception& e) { log_and_exit_uv(e); return false; }
                        break;
                }
                default:
                        return false;
        }
        return true;
}

void display_vulkan_new_message(module* mod) {
        auto s = reinterpret_cast<state_vulkan_sdl3*>(mod);
        assert(s->magic == magic_vulkan_sdl3);

        SDL_Event event{};
        event.type = s->sdl_user_new_message_event;
        SDL_CHECK(SDL_PushEvent(&event));
}


static void display_vulkan_put_audio_frame([[maybe_unused]] void* state, [[maybe_unused]] const struct audio_frame* frame)
{
}


bool display_vulkan_reconfigure_audio([[maybe_unused]] void* state, [[maybe_unused]] int quant_samples,
        [[maybe_unused]] int channels, [[maybe_unused]] int sample_rate)
{
        return false;
}

const video_display_info display_vulkan_info = {
        [](device_info** available_cards, int* count, [[maybe_unused]] void (**deleter)(void*)) {
                *count = 1;
                *available_cards = static_cast<device_info*>(calloc(1, sizeof(device_info)));
                auto& info = **available_cards;
                strcpy(info.dev, "");
                strcpy(info.name, "Vulkan SDL3 SW display");
                info.repeatable = true;
        },
        display_vulkan_init,
        display_vulkan_run,
        display_vulkan_done,
        display_vulkan_getf,
        display_vulkan_putf,
        display_vulkan_reconfigure,
        display_vulkan_get_property,
        display_vulkan_put_audio_frame,
        display_vulkan_reconfigure_audio,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

} // namespace


REGISTER_MODULE(vulkan, &display_vulkan_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
REGISTER_MODULE_WITH_FLAG(vulkan_sdl3, &display_vulkan_info,
                          LIBRARY_CLASS_VIDEO_DISPLAY,
                          VIDEO_DISPLAY_ABI_VERSION, MODULE_FLAG_ALIAS);

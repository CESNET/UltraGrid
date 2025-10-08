/**
 * @file   video_display/gl.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2025 CESNET
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

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#else
#define GLEW_NO_GLU
#include <GL/glew.h>
#endif /* __APPLE__ */
#include <GLFW/glfw3.h>

#include "spout_sender.h"
#include "syphon_server.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>

#include "color_space.h"
#ifdef HAVE_CONFIG_H
#include "config.h"    // for HAVE_SPOUT, HAVE_SYPHON
#endif
#include "compat/strings.h"      // for strcasecmp
#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/debug.h"         // for DEBUG_TIMER_*
#include "utils/macros.h" // OPTIMIZED_FOR
#include "utils/ref_count.hpp"
#include "video.h"
#include "video_display.h"

#define MAGIC_GL         0x1331018e
#define MOD_NAME         "[GL] "
#define DEFAULT_WIN_NAME "Ultragrid - OpenGL Display"
#define GL_DEINTERLACE_IMPOSSIBLE_MSG_ID 0x38e52705
#define GL_DISABLE_10B_OPT_PARAM_NAME "gl-disable-10b"
#define GL_WINDOW_HINT_OPT_PARAM_NAME "glfw-window-hint" ///< @todo TOREMOVE
#define MAX_BUFFER_SIZE 1
#define ADAPTIVE_VSYNC -1
#define SYSTEM_VSYNC 0xFE
#define SINGLE_BUF 0xFF // use single buffering instead of double

#include "gl_vdpau.hpp"

using std::begin;
using std::condition_variable;
using std::copy;
using std::copy_if;
using std::cout;
using std::end;
using std::find;
using std::lock_guard;
using std::map;
using std::min;
using std::mutex;
using std::ostringstream;
using std::pair;
using std::queue;
using std::stof;
using std::stoi;
using std::stol;
using std::string;
using std::string_view;
using std::swap;
using std::unique_lock;
using std::unordered_map;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::steady_clock;
using namespace std::chrono_literals;

static const char * deinterlace_fp = R"raw(
#version 110
uniform sampler2D image;
uniform float lineOff;
void main()
{
        vec4 pix;
        vec4 pix_down;
        pix = texture2D(image, gl_TexCoord[0].xy);
        pix_down = texture2D(image, vec2(gl_TexCoord[0].x, gl_TexCoord[0].y + lineOff));
        gl_FragColor = (pix + pix_down) / 2.0;
}
)raw";

static const char * uyvy_to_rgb_fp = R"raw(
#version 110
uniform sampler2D image;
uniform float imageWidth;
void main()
{
        float imageWidthRaw;
        imageWidthRaw = float((int(imageWidth) + 1) / 2 * 2);
        vec4 yuv;
        yuv.rgba  = texture2D(image, vec2(gl_TexCoord[0].x / imageWidthRaw * imageWidth, gl_TexCoord[0].y)).grba;
        if(gl_TexCoord[0].x * imageWidth / 2.0 - floor(gl_TexCoord[0].x * imageWidth / 2.0) > 0.5)
                yuv.r = yuv.a;
        yuv.r = Y_SCALED_PLACEHOLDER * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        gl_FragColor.r = yuv.r + R_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.g = yuv.r + G_CB_PLACEHOLDER * yuv.g + G_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.b = yuv.r + B_CB_PLACEHOLDER * yuv.g;
        gl_FragColor.a = 1.0;
}
)raw";

static const char * yuva_to_rgb_fp = R"raw(
#version 110
uniform sampler2D image;
void main()
{
        vec4 yuv;
        yuv.rgba  = texture2D(image, gl_TexCoord[0].xy).grba;
        yuv.r = Y_SCALED_PLACEHOLDER * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        gl_FragColor.r = yuv.r + R_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.g = yuv.r + G_CB_PLACEHOLDER * yuv.g + G_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.b = yuv.r + B_CB_PLACEHOLDER * yuv.g;
        gl_FragColor.a = yuv.a;
}
)raw";

static const char * const vuya_to_rgb_fp = R"raw(
#version 110
uniform sampler2D image;
void main()
{
        vec4 yuv;
        yuv.bgra  = texture2D(image, gl_TexCoord[0].xy).rgba;
        yuv.r = Y_SCALED_PLACEHOLDER * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        gl_FragColor.r = yuv.r + R_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.g = yuv.r + G_CB_PLACEHOLDER * yuv.g + G_CR_PLACEHOLDER * yuv.b;
        gl_FragColor.b = yuv.r + B_CB_PLACEHOLDER * yuv.g;
        gl_FragColor.a = yuv.a;
}
)raw";

/// with courtesy of https://stackoverflow.com/questions/20317882/how-can-i-correctly-unpack-a-v210-video-frame-using-glsl
/// adapted to GLSL 1.1 with help of https://stackoverflow.com/questions/5879403/opengl-texture-coordinates-in-pixel-space/5879551#5879551
static const char * v210_to_rgb_fp = R"raw(
#version 110
#extension GL_EXT_gpu_shader4 : enable
uniform sampler2D image;
uniform float imageWidth;

// YUV offset
const vec3 yuvOffset = vec3(-0.0625, -0.5, -0.5);

// RGB coefficients
const vec3 Rcoeff = vec3(Y_SCALED_PLACEHOLDER, 0.0, R_CR_PLACEHOLDER);
const vec3 Gcoeff = vec3(Y_SCALED_PLACEHOLDER, G_CB_PLACEHOLDER, G_CR_PLACEHOLDER);
const vec3 Bcoeff = vec3(Y_SCALED_PLACEHOLDER, B_CB_PLACEHOLDER, 0.0);

// U Y V A | Y U Y A | V Y U A | Y V Y A

int GROUP_FOR_INDEX(int i) {
  return i / 4;
}

int SUBINDEX_FOR_INDEX(int i) {
  return i % 4;
}

int _y(int i) {
  return 2 * i + 1;
}

int _u(int i) {
  return 4 * (i/2);
}

int _v(int i) {
  return 4 * (i / 2) + 2;
}

int offset(int i) {
  return i + (i / 3);
}

vec3 ycbcr2rgb(vec3 yuvToConvert) {
  vec3 pix;
  yuvToConvert += yuvOffset;
  pix.r = dot(yuvToConvert, Rcoeff);
  pix.g = dot(yuvToConvert, Gcoeff);
  pix.b = dot(yuvToConvert, Bcoeff);
  return pix;
}

void main(void) {
  float imageWidthRaw; // v210 texture size
  imageWidthRaw = float((int(imageWidth) + 47) / 48 * 32); // 720->480

  // interpolate (0,1) texcoords to [0,719]
  int texcoordDenormX;
  texcoordDenormX = int(round(gl_TexCoord[0].x * imageWidth - .5));

  // 0 1 1 2 3 3 4 5 5 6 7 7 etc.
  int yOffset;
  yOffset = offset(_y(texcoordDenormX));
  int sourceColumnIndexY;
  sourceColumnIndexY = GROUP_FOR_INDEX(yOffset);

  // 0 0 1 1 2 2 4 4 5 5 6 6 etc.
  int uOffset;
  uOffset = offset(_u(texcoordDenormX));
  int sourceColumnIndexU;
  sourceColumnIndexU = GROUP_FOR_INDEX(uOffset);

  // 0 0 2 2 3 3 4 4 6 6 7 7 etc.
  int vOffset;
  vOffset = offset(_v(texcoordDenormX));
  int sourceColumnIndexV;
  sourceColumnIndexV = GROUP_FOR_INDEX(vOffset);

  // 1 0 2 1 0 2 1 0 2 etc.
  int compY;
  compY = SUBINDEX_FOR_INDEX(yOffset);

  // 0 0 1 1 2 2 0 0 1 1 2 2 etc.
  int compU;
  compU = SUBINDEX_FOR_INDEX(uOffset);

  // 2 2 0 0 1 1 2 2 0 0 1 1 etc.
  int compV;
  compV = SUBINDEX_FOR_INDEX(vOffset);

  vec4 y;
  vec4 u;
  vec4 v;
  y = texture2D(image, vec2((float(sourceColumnIndexY) + .5) / imageWidthRaw, gl_TexCoord[0].y));
  u = texture2D(image, vec2((float(sourceColumnIndexU) + .5) / imageWidthRaw, gl_TexCoord[0].y));
  v = texture2D(image, vec2((float(sourceColumnIndexV) + .5) / imageWidthRaw, gl_TexCoord[0].y));

  vec3 outColor = ycbcr2rgb(vec3(y[compY], u[compU], v[compV]));

  gl_FragColor = vec4(outColor, 1.0);
}
)raw";

/* DXT YUV (FastDXT) related */
static const char *fp_display_dxt1_yuv = R"raw(
#version 110
uniform sampler2D image;

void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);

        float Y = 1.1643 * (col[0] - 0.0625);
        float U = (col[1] - 0.5);
        float V = (col[2] - 0.5);

        float R = Y + 1.7926 * V;
        float G = Y - 0.2132 * U - 0.5328 * V;
        float B = Y + 2.1124 * U;

        gl_FragColor=vec4(R,G,B,1.0);
}
)raw";

static const char * vert = R"raw(
#version 110
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();
}
)raw";

static const char fp_display_dxt5ycocg[] = R"raw(
#version 110
uniform sampler2D image;
void main()
{
        vec4 color;
        float Co;
        float Cg;
        float Y;
        float scale;
        color = texture2D(image, gl_TexCoord[0].xy);
        scale = (color.z * ( 255.0 / 8.0 )) + 1.0;
        Co = (color.x - (0.5 * 256.0 / 255.0)) / scale;
        Cg = (color.y - (0.5 * 256.0 / 255.0)) / scale;
        Y = color.w;
        gl_FragColor = vec4(Y + Co - Cg, Y + Cg, Y - Co - Cg, 1.0);
} // main end
)raw";

unordered_map<codec_t, const char *> glsl_programs = {
        { UYVY, uyvy_to_rgb_fp },
        { Y416, yuva_to_rgb_fp },
        { VUYA, vuya_to_rgb_fp },
        { v210, v210_to_rgb_fp },
        { DXT1_YUV, fp_display_dxt1_yuv },
        { DXT5, fp_display_dxt5ycocg },
};

static constexpr pair<int64_t, string_view> keybindings[] = {
        pair<int64_t, string_view>{'f', "toggle fullscreen"},
        pair<int64_t, string_view>{'q', "quit"},
        pair<int64_t, string_view>{K_ALT('d'), "toggle deinterlace"},
        pair<int64_t, string_view>{K_ALT('p'), "pause video"},
        pair<int64_t, string_view>{K_ALT('s'), "screenshot"},
        pair<int64_t, string_view>{K_ALT('c'), "show/hide cursor"},
        pair<int64_t, string_view>{K_CTRL_DOWN, "make window 10% smaller"},
        pair<int64_t, string_view>{K_CTRL_UP, "make window 10% bigger"}
};

#ifdef GLFW_PLATFORM
const static struct {
        int platform_id;
        const char *name;
} platform_map[] = {
        { GLFW_PLATFORM_WIN32,   "Win32"   },
        { GLFW_PLATFORM_COCOA,   "Cocoa"   },
        { GLFW_PLATFORM_WAYLAND, "Wayland" },
        { GLFW_PLATFORM_X11,     "X11"     },
};
#endif // defined GLFW_PLATFORM

/* Prototyping */
static bool check_display_gl_version(bool print_ver);
static bool display_gl_init_opengl(struct state_gl *s);
static bool display_gl_putf(void *state, struct video_frame *frame, long long timeout);
static bool display_gl_process_key(struct state_gl *s, long long int key);
static bool display_gl_reconfigure(void *state, struct video_desc desc);
static void gl_draw(double ratio, double bottom_offset, bool double_buf);
static void gl_change_aspect(struct state_gl *s, int width, int height);
static void gl_resize(GLFWwindow *win, int width, int height);
static void gl_render_glsl(struct state_gl *s, char *data);
static void gl_reconfigure_screen(struct state_gl *s, struct video_desc desc);
static void gl_process_frames(struct state_gl *s);
static void glfw_key_callback(GLFWwindow* win, int key, int scancode, int action, int mods);
static void glfw_mouse_callback(GLFWwindow *win, double x, double y);
static void glfw_close_callback(GLFWwindow *win);
static void glfw_print_error(int error_code, const char* description);
static void glfw_print_video_mode(struct state_gl *s);
static void display_gl_set_sync_on_vblank(int value);
static void screenshot(struct video_frame *frame);
static void upload_compressed_texture(struct state_gl *s, char *data);
static void upload_texture(struct state_gl *s, char *data);
static bool check_rpi_pbo_quirks();
static void set_gamma(struct state_gl *s);

struct state_gl {
        unordered_map<codec_t, GLuint> PHandles;
        GLuint          PHandle_deint = 0;
        GLuint          current_program = 0;

        // Framebuffer
        GLuint fbo_id = 0;
        GLuint texture_display = 0;
        GLuint texture_raw = 0;
        GLuint pbo_id = 0;

        /* For debugging... */
        uint32_t        magic = MAGIC_GL;

        GLFWmonitor    *monitor = nullptr;
        GLFWwindow     *window = nullptr;

        bool            fs = false;
        enum class deint { off, on, force } deinterlace = deint::off;

        struct video_frame *current_frame = nullptr;

        queue<struct video_frame *> frame_queue;
        queue<struct video_frame *> free_frame_queue;
        struct video_desc current_desc = {};
        struct video_desc current_display_desc {};
        mutex           lock;
        condition_variable new_frame_ready_cv;
        condition_variable frame_consumed_cv;

        double          aspect = 0.0;
        double          video_aspect = 0.0;
        double          gamma = 0.0;

        int             dxt_height = 0;

        int             vsync = 1;
        bool            noresizable = false;
        volatile bool   paused = false;
        enum show_cursor_t { SC_TRUE, SC_FALSE, SC_AUTOHIDE } show_cursor = SC_AUTOHIDE;
        steady_clock::time_point                      cursor_shown_from{}; ///< indicates time point from which is cursor show if show_cursor == SC_AUTOHIDE, timepoint() means cursor is not currently shown
        string          syphon_spout_srv_name;

        double          window_size_factor = 1.0;

        struct module   mod;

        void *syphon_spout = nullptr;

        bool fixed_size = false;
        int fixed_w = 0;
        int fixed_h = 0;
        int pos_x = INT_MIN;
        int pos_y = INT_MIN;

        enum modeset_t { MODESET = -2, MODESET_SIZE_ONLY = GLFW_DONT_CARE, NOMODESET = 0 } modeset = NOMODESET; ///< positive vals force framerate
        map<int, int> init_hints;
        map<int, int> window_hints = {
                {GLFW_AUTO_ICONIFY, GLFW_FALSE}
        };
        int use_pbo = -1;
        int req_monitor_idx = -1;
#ifdef HWACC_VDPAU
        struct state_vdpau vdp;
#endif
        bool         vdp_interop = false;
        vector<char> scratchpad; ///< scratchpad sized WxHx8

        state_gl(struct module *parent) {
                module_init_default(&mod);
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);
        }

        ~state_gl() {
                ref_count_terminate_last()(glfwTerminate, glfw_init_count);

                module_done(&mod);
        }

        static const char *deint_to_string(state_gl::deint val) {
                switch (val) {
                        case state_gl::deint::off: return "OFF";
                        case state_gl::deint::on: return "ON";
                        case state_gl::deint::force: return "FORCE";
                }
                return NULL;
        }
};

static constexpr codec_t gl_supp_codecs[] = {
#ifdef HWACC_VDPAU
        HW_VDPAU,
#endif
        UYVY,
        VUYA,
        v210,
        R10k,
        RGBA,
        RGB,
        RG48,
        Y416,
        DXT1,
        DXT1_YUV,
        DXT5
};

static void gl_print_monitors(bool fullhelp) {
        printf("\nmonitors:\n");
        int count = 0;
        GLFWmonitor **mon = glfwGetMonitors(&count);
        if (count <= 0) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "No monitors found!\n";
        }
        GLFWmonitor *primary = glfwGetPrimaryMonitor();
        for (int i = 0; i < count; ++i) {
                col() << "\t" << (mon[i] == primary ? "*" : " ") << TBOLD(<< i <<) << ") " << glfwGetMonitorName(mon[i]);
#if GLFW_VERSION_MAJOR > 3 || (GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 3)
                int xpos, ypos, width, height;
                glfwGetMonitorWorkarea(mon[i], &xpos, &ypos, &width, &height);
                cout << " - " << width << "x" << height << "+" << xpos << "+" << ypos << "\n";
#else
                cout << "\n";
#endif
                if (!fullhelp) {
                        continue;
                }
                int mod_count = 0;
                const GLFWvidmode *modes = glfwGetVideoModes(mon[i], &mod_count);
                for (int mod = 0; mod < mod_count; ++mod) {
                        cout << "\t\t- " << modes[mod].width << "x" << modes[mod].height << "@" << modes[mod].refreshRate << ", bits: " <<
                                modes[mod].redBits << ", " <<
                                modes[mod].greenBits << ", " <<
                                modes[mod].blueBits << "\n";
                }
        }
        if (!fullhelp) {
                cout << "(use \"fullhelp\" to see modes)\n";
        }
        printf("\n");
}

#ifdef GLFW_PLATFORM
static void
gl_print_platforms()
{
        printf("available platforms:\n");
        for (unsigned i = 0; i < ARR_COUNT(platform_map); ++i) {
                if (glfwPlatformSupported(platform_map[i].platform_id)) {
                        color_printf("\t- " TBOLD("%s") "\n", platform_map[i].name);
                }
        }
        color_printf("\n");
}

static void
gl_print_current_platform()
{
        const int platform = glfwGetPlatform();
        const char *name = "UNKNOWN/ERROR";
        for (unsigned i = 0; i < ARR_COUNT(platform_map); ++i) {
                if (platform_map[i].platform_id == platform) {
                        name = platform_map[i].name;
                        break;
                }
        }
#ifdef __linux__
        int ll = LOG_LEVEL_NOTICE;
#else
        int ll = LOG_LEVEL_VERBOSE;
#endif
        log_msg(ll, MOD_NAME "Using platform: %s\n", name);
}
#else // mot defined GLFW_PLATFORM
static void
gl_print_platforms()
{
        MSG(ERROR, "platforms unsupported (old GLFW)\n");
}
// NOOP
static void
gl_print_current_platform()
{
}
#endif // not defined GLFW_PLATFORM

#define FEATURE_PRESENT(x) (strcmp(STRINGIFY(x), "1") == 0 ? "on" : "off")

/**
 * Show help
 */
static void gl_show_help(bool full) {
        col() << "usage:\n";
        col() << SBOLD(SRED("\t-d gl[:<options>]")
                       << (full ? " [--param " GL_DISABLE_10B_OPT_PARAM_NAME "]"
                                : ""))
              << "\n";
        col() << SBOLD("\t-d gl:[full]help") "\n\n";
        col() << "options:\n";
        col() << TBOLD("\taspect=<w>/<h>") << "\trequested video aspect (eg. 16/9). Leave unset if PAR = 1.\n";
        col() << TBOLD("\tcursor")      << "\t\tshow visible cursor\n";
        col() << TBOLD("\td[force]")    << "\tdeinterlace (optionally forcing deinterlace of progressive video)\n";
        col() << TBOLD("\tfs[=<monitor>]") << "\tfullscreen with optional display specification\n";
        col() << TBOLD("\tgamma[=<val>]")
              << "\tgamma value to be added _in addition_ to the hardware "
                 "gamma correction\n";
        col() << TBOLD("\thide-window") << "\tdo not show OpenGL window (useful with Syphon/SPOUT)\n";
        col() << TBOLD("\tmodeset[=<fps>]")
              << "\tset received video mode as display mode (in fullscreen)\n"
                 "\t\t\t"
              << SUNDERLINE("fps") " - override FPS; "
              << SUNDERLINE("modeset=size") << " - set only size\n";
        col() << TBOLD("\tnodecorate") << "\tdisable window decorations\n";
        col() << TBOLD("\tnovsync")     << "\t\tdo not turn sync on VBlank\n";
        col() << TBOLD("\t[no]pbo")     << "\t\tWhether or not use PBO (ignore if not sure)\n";
        col() << TBOLD("\tsingle")      << "\t\tuse single buffer (instead of double-buffering)\n";
        col() << TBOLD("\tsize=<ratio>%")
              << "\tspecifies desired size of window relative\n"
                 "\t\t\tto native resolution (in percents)\n";
        col() << TBOLD("\tsize=<W>x<H>")
              << "\twindow size in pixels, with optional position; full\n"
              << "\t\t\tsyntax: " TBOLD("[<W>x<H>][{+-}<X>[{+-}<Y>]]")
              << (full ? " [1]" : "") << "\n";
        if (full) {
                col() << TBOLD("\tsize=<mode_name>")
                      << " mode name (eg. VGA), use \"help\" to show\n";
        }
        col() << TBOLD("\tfixed_size") << "\tdo not resize window on new stream\n";
        col() << TBOLD("\tnoresizable") << "\twindow won't be resizable (useful with size=)\n";
#ifdef SPOUT
        col() << TBOLD("\tspout")       << "\t\tuse Spout (optionally with name)\n";
#endif
#ifdef SYPHON
        col() << TBOLD("\tsyphon")      << "\t\tuse Syphon (optionally with name)\n";
#endif
        col() << TBOLD("\tvsync=<x>")   << "\tsets vsync to: 0 - disable; 1 - enable; -1 - adaptive vsync; D - leaves system default\n";
        if (full) {
                col() << TBOLD("\tinit_hint=<k>=<v>[,<k2>=<v2>]")" set GLFW "
                        "init hint key\n"
                         "\t\t\t<k> to value <v> (" TBOLD("list_hints")" to enumerate)\n";
                col() << TBOLD("\twindow_hint=<k>=<v>[,<k2>=<v2>]")" set GLFW "
                        "window hint key\n"
                         "\t\t\t<k> to value <v>,  (" TBOLD("list_hints")" to enumerate)\n";
                col() << TBOLD("\tplatform=<p>")   << "\tuse platform (usable only in Linux)\n";
                col() << TBOLD("\t--param " GL_DISABLE_10B_OPT_PARAM_NAME)     << "\tdo not set 10-bit framebuffer (performance issues)\n";
                col() << "\n" TBOLD(
                    "[1]") " position doesn't work in Wayland\n";
        } else {
                color_printf(
                    "\t(use \"" TBOLD("fullhelp") "\" to see options)\n");
        }

        printf("\nkeyboard shortcuts:\n");
        for (auto const &i : keybindings) {
                char keyname[50];
                get_keycode_name(i.first, keyname, sizeof keyname);
                col() << "\t" << TBOLD(<< keyname <<) << "\t\t" << i.second << "\n";
        }

        if (ref_count_init_once<int>()(glfwInit, glfw_init_count).value_or(GLFW_TRUE) == GLFW_FALSE) {
                LOG(LOG_LEVEL_ERROR) << "Cannot initialize GLFW!\n";
                return;
        }
        gl_print_monitors(full);
        if (full) {
                gl_print_platforms();
        }
        GLFWwindow *window = glfwCreateWindow(32, 32, DEFAULT_WIN_NAME, nullptr, nullptr);
        if (window != nullptr) {
                glfwMakeContextCurrent(window);
                check_display_gl_version(true);
                glfwDestroyWindow(window);
        }
        ref_count_terminate_last()(glfwTerminate, glfw_init_count);

        col() << "Compiled " << SBOLD("features: ") << "SPOUT - "
              << FEATURE_PRESENT(SPOUT) << ", Syphon - "
              << FEATURE_PRESENT(SYPHON) << ", VDPAU - "
              << FEATURE_PRESENT(HWACC_VDPAU) << "\n";
}

static void gl_load_splashscreen(struct state_gl *s)
{
        struct video_frame *frame = get_splashscreen();
        display_gl_reconfigure(s, video_desc_from_frame(frame));
        s->frame_queue.push(frame);
}

static bool set_size(struct state_gl *s, const char *tok)
{
        if (strstr(tok, "fixed_size=") == tok) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "fixed_size with dimensions is"
                                 " deprecated, use size"
                                 " instead\n");
        }
        tok = strchr(tok, '=') + 1;
        if (strchr(tok, '%') != NULL) {
                s->window_size_factor = atof(tok) / 100.0;
        } else if (strpbrk(tok, "x+-") == NULL) {
                struct video_desc desc = get_video_desc_from_string(tok);
                if (desc.width != 0) {
                        s->fixed_size = true;
                        s->fixed_w = desc.width;
                        s->fixed_h = desc.height;
                        return true;
                }
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong size spec: %s\n", tok);
                return false;
        }
        if (strchr(tok, 'x') != NULL) {
                s->fixed_size = true;
                s->fixed_w = atoi(tok);
                s->fixed_h = atoi(strchr(tok, 'x') + 1);
        }
        tok = strpbrk(tok, "+-");
        if (tok != NULL) {
                s->pos_x = atoi(tok);
                tok = strpbrk(tok + 1, "+-");
        }
        if (tok != NULL) {
                s->pos_y = atoi(tok);
        }
        return true;
}

static const map<string, int> hint_map = {
        {"autoiconify", 0x20006},
        { "platform",   0x50003},
        { "x11",        0x60003},
        { "wayland",    0x60004},
};

/// @param window_hints true for window hints, otherwise init hints
static void
parse_hints(struct state_gl *s, bool window_hints, char *hints)
{
        char *tok      = nullptr;
        char *save_ptr = nullptr;
        while ((tok = strtok_r(hints, ",", &save_ptr)) != nullptr) {
                hints = nullptr;
                if (strchr(tok, '=') == nullptr) {
                        MSG(ERROR, "Malformed window hint - missing value "
                                   "(expected <k>=<v>).\n");
                        continue;
                }
                const char *key_s = tok;
                const char *val_s = strchr(tok, '=') + 1;
                *strchr(tok, '=') = '\0';
                const int key     = hint_map.find(key_s) != hint_map.end()
                                        ? hint_map.at(key_s)
                                        : stoi(key_s, nullptr, 0);
                const int val     = hint_map.find(val_s) != hint_map.end()
                                        ? hint_map.at(val_s)
                                        : stoi(val_s, nullptr, 0);
                if (window_hints) {
                        s->window_hints[key] = val;
                } else {
                        s->init_hints[key] = val;
                }
        }
}

static bool
set_platform(struct state_gl *s, const char *platform)
{
#ifdef GLFW_PLATFORM
        for (unsigned i = 0; i < ARR_COUNT(platform_map); ++i) {
                if (strcasecmp(platform_map[i].name, platform) == 0) {
                        s->init_hints[GLFW_PLATFORM] = platform_map[i].platform_id;
                        return true;
                }
        }
        MSG(ERROR, "Unknown platform: %s\n", platform);
        return false;
#else
        (void) s, (void)  platform;
        MSG(ERROR, "platforms unsupported (old GLFW)\n");
        return false;
#endif
}

static void
list_hints()
{
        color_printf(
            "Very " TBOLD("incomplete") " list of hints and its values.\n");
        color_printf("Keywords listed below can be passed to the options "
                     "instead of the numeric value.\n\n");

        color_printf("The list contains both keys and values for init and "
                     "window hints.\n");
        color_printf("Window hints start with 0x20000, init with 0x50000, the "
                     "others are values.\n\n");

        color_printf("For complete list refer to glfw3.h (eg. online "
                     "<https://github.com/glfw/glfw/blob/master/include/GLFW/"
                     "glfw3.h>).\n\n");

        color_printf("Some of hints keys and values:\n");
        for (const auto &h : hint_map) {
                color_printf("\t" TBOLD("%s") " - %#x\n", h.first.c_str(),
                             h.second);
        }
        color_printf("\n");

        color_printf(
            "Example usage: " TBOLD("-d gl:init_hint=platform=x11") "\n\n");
}

static bool
display_gl_parse_fmt(struct state_gl *s, char *ptr)
{
        bool ret = true;
        char *tok, *save_ptr = NULL;

        while((tok = strtok_r(ptr, ":", &save_ptr)) != NULL) {
                if (strcmp(tok, "help") == 0 || strcmp(tok, "fullhelp") == 0) {
                        gl_show_help(strcmp(tok, "fullhelp") == 0);
                        return false;
                }
                if (!strcmp(tok, "d") || !strcmp(tok, "dforce")) {
                        s->deinterlace = !strcmp(tok, "d") ? state_gl::deint::on : state_gl::deint::force;
                } else if(!strncmp(tok, "fs", 2)) {
                        s->fs = true;
                        if (char *val = strchr(tok, '=')) {
                                s->req_monitor_idx = stoi(val + 1);
                        }
                } else if(strstr(tok, "modeset") != nullptr) {
                        if (strcmp(tok, "nomodeset") != 0) {
                                if (char *val = strchr(tok, '=')) {
                                        val += 1;
                                        s->modeset = strcmp(val, "size") == 0 ? state_gl::MODESET_SIZE_ONLY : (enum state_gl::modeset_t) stoi(val);
                                } else {
                                        s->modeset = state_gl::MODESET;
                                }
                        }
                } else if(!strncmp(tok, "aspect=", strlen("aspect="))) {
                        s->video_aspect = atof(tok + strlen("aspect="));
                        char *pos = strchr(tok,'/');
                        if(pos) s->video_aspect /= atof(pos + 1);
                } else if(!strcasecmp(tok, "nodecorate")) {
                        s->window_hints[GLFW_DECORATED] =  GLFW_FALSE;
                } else if(!strcasecmp(tok, "novsync")) {
                        s->vsync = 0;
                } else if(!strcasecmp(tok, "single")) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Single-buffering is not recommended and may not work, also GLFW discourages its use.\n";
                        s->vsync = SINGLE_BUF;
                } else if (!strncmp(tok, "vsync=", strlen("vsync="))) {
                        if (toupper((tok + strlen("vsync="))[0]) == 'D') {
                                s->vsync = SYSTEM_VSYNC;
                        } else {
                                s->vsync = atoi(tok + strlen("vsync="));
                        }
                } else if (!strcasecmp(tok, "cursor")) {
                        s->show_cursor = state_gl::SC_TRUE;
                } else if (strstr(tok, "syphon") == tok || strstr(tok, "spout") == tok) {
#if defined HAVE_SYPHON || defined HAVE_SPOUT
                        if (strchr(tok, '=')) {
                                s->syphon_spout_srv_name = strchr(tok, '=') + 1;
                        } else {
                                s->syphon_spout_srv_name = "UltraGrid";
                        }
#else
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Syphon/Spout support not compiled in.\n");
                        return false;
#endif
                } else if (strstr(tok, "gamma=") == tok) {
                        s->gamma = stof(strchr(tok, '=') + 1);
                        if (strchr(tok, '/')) {
                                s->gamma /= stof(strchr(tok, '/') + 1);
                        }
                } else if (!strcasecmp(tok, "hide-window")) {
                        s->window_hints[GLFW_VISIBLE] = GLFW_FALSE;
                } else if (strcasecmp(tok, "pbo") == 0 || strcasecmp(tok, "nopbo") == 0) {
                        s->use_pbo = strcasecmp(tok, "pbo") == 0 ? 1 : 0;
                } else if (strstr(tok, "size=") == tok ||
                           strstr(tok, "fixed_size=") == tok) {
                        ret = ret && set_size(s, tok);
                } else if (strcmp(tok, "fixed_size") == 0) {
                        s->fixed_size = true;
                } else if (strcmp(tok, "noresizable") == 0) {
                        s->noresizable = true;
                } else if (strstr(tok, "window_hint=") == tok) {
                        parse_hints(s, true, strchr(tok, '=') + 1);
                } else if (strstr(tok, "init_hint=") == tok) {
                        parse_hints(s, false, strchr(tok, '=') + 1);
                } else if (IS_KEY_PREFIX(tok, "platform")) {
                        ret = ret && set_platform(s, strchr(tok, '=') + 1);
                } else if (strcmp(tok, "list_hints") == 0) {
                        list_hints();
                        return false;
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", tok);
                        return false;
                }
                ptr = NULL;
        }

        return ret;
}

static void * display_gl_init(struct module *parent, const char *fmt, unsigned int flags) {
        UNUSED(flags);

	struct state_gl *s = nullptr;
        try {
                s = new state_gl(parent);
        } catch (...) {
                return nullptr;
        }

        if (fmt != NULL) {
                char *tmp = strdup(fmt);
                bool ret = false;
                try {
                        ret = display_gl_parse_fmt(s, tmp);
                } catch (std::invalid_argument &e) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Invalid numeric value for an option!\n";
                }
                free(tmp);
                if (!ret) {
                        delete s;
                        return strstr(fmt, "help") == nullptr ? nullptr : INIT_NOERR;
                }
        }

        s->use_pbo = s->use_pbo == -1 ? !check_rpi_pbo_quirks() : s->use_pbo; // don't use PBO for Raspberry Pi (better performance)

        log_msg(LOG_LEVEL_INFO,"GL setup: fullscreen: %s, deinterlace: %s\n",
                        s->fs ? "ON" : "OFF", state_gl::deint_to_string(s->deinterlace));

        gl_load_splashscreen(s);
        for (auto const &i : keybindings) {
                if (i.first == 'q') { // don't report 'q' to avoid accidental close - user can use Ctrl-c there
                        continue;
                }
                char msg[18];
                snprintf(msg, sizeof msg, "%" PRIx64, i.first);
                keycontrol_register_key(&s->mod, i.first, msg, i.second.data());
        }

#if GLFW_VERSION_MAJOR < 3 || (GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR < 3)
        if(!s->init_hints.empty()){
                log_msg(LOG_LEVEL_WARNING, "GLFW version < 3.3 doesn't support init hints. Ignoring them.\n");
        }
#else
        for (auto const &hint : s->init_hints) {
                glfwInitHint(hint.first, hint.second);
        }
#endif

        if (!display_gl_init_opengl(s)) {
                delete s;
                return nullptr;
        }

        return (void*)s;
}

#if defined HAVE_SPOUT || HAVE_SYPHON
#ifdef HAVE_SPOUT
#define NAME_SPOUT_SYPHON_RAW spout
#else
#define NAME_SPOUT_SYPHON_RAW syphon
#endif
#define NAME_SPOUT_SYPHON TOSTRING(NAME_SPOUT_SYPHON_RAW) " disp."
static void * display_spout_syphon_init(struct module *parent, const char *fmt, unsigned int flags) {
        char config_str[1024] = "hide-window:" NAME_SPOUT_SYPHON;
        if (strcmp(fmt, "help") == 0) {
                color_printf("Display " TBOLD(NAME_SPOUT_SYPHON) " is a tiny wrapper over OpenGL display (that may be used directly as well).\n\n");
                color_printf("Usage:\n");
                color_printf(TBOLD(TRED("\t-d " NAME_SPOUT_SYPHON) "[:name=<server_name]\n\n"));
                color_printf("where:\n"
                                TBOLD("\tname") " - name of the created server\n");
                return INIT_NOERR;
        }
        if (strstr(fmt, "name=") == fmt) {
                snprintf(config_str + strlen(config_str), sizeof config_str - strlen(config_str), "=%s", strchr(fmt, '=') + 1);
        } else if (strlen(fmt) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[" NAME_SPOUT_SYPHON "] Wrong option \"%s\"!\n", fmt);
                return NULL;
        }
        return display_gl_init(parent, config_str, flags);
}
#endif // defined HAVE_SPOUT || HAVE_SYPHON

/**
 * This function just sets new video description.
 */
static bool
display_gl_reconfigure(void *state, struct video_desc desc)
{
        struct state_gl	*s = (struct state_gl *) state;

        assert(find(begin(gl_supp_codecs), end(gl_supp_codecs),
                    desc.color_spec) != end(gl_supp_codecs));
        if (get_bits_per_component(desc.color_spec) > 8) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Displaying 10+ bits - performance degradation may occur, consider '--param " GL_DISABLE_10B_OPT_PARAM_NAME "'\n";
        }
        if (desc.interlacing == INTERLACED_MERGED && s->deinterlace == state_gl::deint::off) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Receiving interlaced video but deinterlacing is off - suggesting toggling it on (press 'd' or pass cmdline option)\n";
        }

        s->current_desc = desc;

        return true;
}

static void glfw_print_video_mode(struct state_gl *s) {
        if (!s->fs || !s->modeset) {
                return;
        }
        const GLFWvidmode* mode = glfwGetVideoMode(s->monitor);
        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Display mode set to: " << mode->width << "x" << mode->height << "@" << mode->refreshRate << "\n";
}

static int get_refresh_rate(enum state_gl::modeset_t modeset, GLFWmonitor *mon, double video_refresh_rate) {
        if (!mon) {
                return GLFW_DONT_CARE;
        }
        switch (modeset) {
                case state_gl::modeset_t::MODESET:
                        return round(video_refresh_rate);
                case state_gl::modeset_t::MODESET_SIZE_ONLY:
                        return GLFW_DONT_CARE;
                case state_gl::modeset_t::NOMODESET: {
                        const GLFWvidmode* mode = glfwGetVideoMode(mon);
                        return mode->refreshRate;
                }
                default:
                        return static_cast<int>(modeset);
        }
}

static void glfw_resize_window(GLFWwindow *win, bool fs, int height, double aspect, double fps, double window_size_factor)
{
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "glfw - fullscreen: %d, aspect: %lf, factor %lf\n",
                        (int) fs, aspect, window_size_factor);
        auto *s = (struct state_gl *) glfwGetWindowUserPointer(win);
        if (fs && s->monitor != nullptr) {
                int width = round(height * aspect);
                GLFWmonitor *mon = s->monitor;
                int refresh_rate = get_refresh_rate(s->modeset, mon, fps);
                if (s->modeset == state_gl::modeset_t::NOMODESET) {
                        const GLFWvidmode* mode = glfwGetVideoMode(mon);
                        width = mode->width;
                        height = mode->height;
                }
                glfwSetWindowMonitor(win, mon, GLFW_DONT_CARE, GLFW_DONT_CARE, width, height, refresh_rate);
        } else {
                glfwSetWindowSize(win, round(window_size_factor * height * aspect), window_size_factor * height);
        }
        glfwSwapBuffers(win);
}

/*
 * Please note, that setting the value of 0 to GLX function is invalid according to
 * documentation. However. reportedly NVidia driver does unset VSync.
 */
static void display_gl_set_sync_on_vblank(int value) {
        if (value == SYSTEM_VSYNC) {
                return;
        }
        glfwSwapInterval(value);
}

static void screenshot(struct video_frame *frame)
{
        char name[128];
        time_t t;
        t = time(NULL);
        struct tm *time_ptr;

#ifdef _WIN32
        time_ptr = localtime(&t);
#else
        struct tm time_tmp;
        localtime_r(&t, &time_tmp);
        time_ptr = &time_tmp;
#endif

        strftime(name, sizeof(name), "screenshot-%a, %d %b %Y %H:%M:%S %z.pnm",
                                               time_ptr);
        if (save_video_frame_as_pnm(frame, name)) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Generated screenshot \"%s\".\n", name);
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to generate screenshot!\n");
        }
}

/**
 * This function does the actual reconfiguration of GL state.
 *
 * This function must be called only from GL thread.
 */
static void gl_reconfigure_screen(struct state_gl *s, struct video_desc desc)
{
        assert(s->magic == MAGIC_GL);

        s->dxt_height = desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV || desc.color_spec == DXT5 ?
                (desc.height + 3) / 4 * 4 : desc.height;

        s->aspect = s->video_aspect ? s->video_aspect : (double) desc.width / desc.height;

        log_msg(LOG_LEVEL_INFO, "Setting GL size %dx%d (%dx%d).\n", (int) round(s->aspect * desc.height),
                        desc.height, desc.width, desc.height);

        s->current_program = 0;

        gl_check_error();

        if(desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV) {
                if (desc.color_spec == DXT1) {
                        glActiveTexture(GL_TEXTURE0 + 0);
                        glBindTexture(GL_TEXTURE_2D,s->texture_display);
                } else {
                        glActiveTexture(GL_TEXTURE0 + 2);
                        glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                }
                size_t data_len = ((desc.width + 3) / 4 * 4* s->dxt_height)/2;
                char *buffer = (char *) malloc(data_len);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                (desc.width + 3) / 4 * 4, s->dxt_height, 0, data_len,
                                /* passing NULL here isn't allowed and some rigid implementations
                                 * will crash here. We just pass some egliable buffer to fulfil
                                 * this requirement. glCompressedSubTexImage2D works as expected.
                                 */
                                buffer);
                free(buffer);
                if(desc.color_spec == DXT1_YUV) {
                        glActiveTexture(GL_TEXTURE0 + 0);
                        glBindTexture(GL_TEXTURE_2D,s->texture_display);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                        desc.width, desc.height, 0,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        NULL);
                        s->current_program = s->PHandles.at(DXT1_YUV);
                }
        } else if (desc.color_spec == UYVY) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                (desc.width + 1) / 2, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                s->current_program = s->PHandles.at(UYVY);
        } else if (desc.color_spec == v210) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2,
                                vc_get_linesize(desc.width, v210) / 4, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,
                                NULL);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_SHORT,
                                NULL);
                s->current_program = s->PHandles.at(v210);
        } else if (desc.color_spec == Y416) {
                s->current_program = s->PHandles.at(Y416);
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_SHORT,
                                NULL);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_SHORT,
                                NULL);
        } else if (desc.color_spec == RGBA) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
        } else if (desc.color_spec == VUYA) {
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                s->current_program = s->PHandles.at(VUYA);
        } else if (desc.color_spec == RGB) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                                desc.width, desc.height, 0,
                                GL_RGB, GL_UNSIGNED_BYTE,
                                NULL);
        } else if (desc.color_spec == R10k) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,
                                nullptr);
        } else if (desc.color_spec == RG48) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                                desc.width, desc.height, 0,
                                GL_RGB, GL_UNSIGNED_SHORT,
                                nullptr);
        } else if (desc.color_spec == DXT5) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                (desc.width + 3) / 4 * 4, s->dxt_height, 0,
                                (desc.width + 3) / 4 * 4 * s->dxt_height,
                                NULL);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                s->current_program = s->PHandles.at(DXT5);
        }
#ifdef HWACC_VDPAU
        else if (desc.color_spec == HW_VDPAU) {
                s->vdp.init();
        }
#endif
        if (s->current_program) {
                glUseProgram(s->current_program);
                if (GLint l = glGetUniformLocation(s->current_program, "image"); l != -1) {
                        glUniform1i(l, 2);
                }
                if (GLint l = glGetUniformLocation(s->current_program, "imageWidth"); l != -1) {
                        glUniform1f(l, (GLfloat) desc.width);
                }
                glUseProgram(0);
        }
        if (s->PHandle_deint) {
                glUseProgram(s->PHandle_deint);
                glUniform1i(glGetUniformLocation(s->PHandle_deint, "image"), 0);
                glUniform1f(glGetUniformLocation(s->PHandle_deint, "lineOff"), 1.0f / desc.height);
                glUseProgram(0);
        }
        gl_check_error();

        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, vc_get_linesize(desc.width, desc.color_spec) * desc.height, 0, GL_STREAM_DRAW_ARB);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        gl_check_error();

        if (!s->fixed_size) {
                glfw_resize_window(s->window, s->fs, desc.height, s->aspect, desc.fps, s->window_size_factor);
                //gl_resize(s->window, desc.width, desc.height);
        }
        int width, height;
        glfwGetFramebufferSize(s->window, &width, &height);
        gl_change_aspect(s, width, height);

        gl_check_error();

        display_gl_set_sync_on_vblank(s->vsync);
        gl_check_error();

#ifdef HAVE_SYPHON
        if (!s->syphon_spout && !s->syphon_spout_srv_name.empty()) {
                s->syphon_spout = syphon_server_register(CGLGetCurrentContext(), s->syphon_spout_srv_name.c_str());
        }
#endif
#ifdef HAVE_SPOUT
        if (!s->syphon_spout_srv_name.empty()) {
                if (!s->syphon_spout) {
                        s->syphon_spout = spout_sender_register(s->syphon_spout_srv_name.c_str());
                }
	}
#endif

        s->scratchpad.resize(desc.width * desc.height * 8);
        s->current_display_desc = desc;
}

static void gl_render(struct state_gl *s, char *data)
{
        gl_check_error();

        if (s->current_program) {
                gl_render_glsl(s, data);
        } else {
                upload_texture(s, data);
        }

        gl_check_error();
}

/// @note lk will be unlocked!
static void pop_frame(struct state_gl *s, unique_lock<mutex> &lk)
{
        s->frame_queue.pop();
        lk.unlock();
        s->frame_consumed_cv.notify_one();
}

/// draw pause symbol (2 vertical bars) to the lefttop part of the frame
static void
draw_pause()
{
#define BAR_WIDTH       (.01)
#define LEFT_BAR_LEFT   (-.9)
#define LEFT_BAR_RIGHT  (LEFT_BAR_LEFT + BAR_WIDTH)
#define RIGHT_BAR_LEFT  (LEFT_BAR_RIGHT + BAR_WIDTH)
#define RIGHT_BAR_RIGHT (RIGHT_BAR_LEFT + BAR_WIDTH)
#define BARS_TOP        (.9)
#define BARS_BOTTOM     (.8)
#define BAR_GREY_COL    (0.9)
        glBindTexture(GL_TEXTURE_2D, 0);
        glColor3f(BAR_GREY_COL, BAR_GREY_COL, BAR_GREY_COL);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glBegin(GL_QUADS);
        glVertex2f(LEFT_BAR_LEFT, BARS_BOTTOM);  // bottom left
        glVertex2f(LEFT_BAR_RIGHT, BARS_BOTTOM); // bottom right
        glVertex2f(LEFT_BAR_RIGHT, BARS_TOP);    // top right
        glVertex2f(LEFT_BAR_LEFT, BARS_TOP);     // top left
        glEnd();
        glBegin(GL_QUADS);
        glVertex2f(RIGHT_BAR_LEFT, BARS_BOTTOM);
        glVertex2f(RIGHT_BAR_RIGHT, BARS_BOTTOM);
        glVertex2f(RIGHT_BAR_RIGHT, BARS_TOP);
        glVertex2f(RIGHT_BAR_LEFT, BARS_TOP);
        glEnd();
        glColor4f(1, 1, 1, 1);
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        gl_check_error();
}

static void gl_process_frames(struct state_gl *s)
{
        struct video_frame *frame;

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                auto msg_univ = reinterpret_cast<struct msg_universal *>(msg);
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received message: %s\n", msg_univ->text);
                struct response *r;
                if (strncasecmp(msg_univ->text, "win-title ", strlen("win-title ")) == 0) {
                        glfwSetWindowTitle(s->window, msg_univ->text + strlen("win-title "));
                        r = new_response(RESPONSE_OK, NULL);
                } else {
                        if (strlen(msg_univ->text) == 0) {
                                r = new_response(RESPONSE_BAD_REQUEST, "Wrong message - neither win-title nor key");
                        } else {
                                int64_t key;
                                int ret = sscanf(msg_univ->text, "%" SCNx64, &key);
                                if (ret != 1 || !display_gl_process_key(s, key)) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unknown key received: %s\n", msg_univ->text);
                                        r = new_response(RESPONSE_BAD_REQUEST, "Wrong key received");
                                } else {
                                        r = new_response(RESPONSE_OK, NULL);
                                }
                        }
                }
                free_message(msg, r);
        }

        if (s->show_cursor == state_gl::SC_AUTOHIDE) {
                if (s->cursor_shown_from != steady_clock::time_point()) {
                        const auto now = steady_clock::now();
                        if (duration_cast<seconds>(now - s->cursor_shown_from).count() > 2) {
                                glfwSetInputMode(s->window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                                s->cursor_shown_from = steady_clock::time_point();
                        }
                }
        }

        {
                unique_lock<mutex> lk(s->lock);
                double timeout = min(2.0 / s->current_display_desc.fps, 0.1);
                s->new_frame_ready_cv.wait_for(lk, duration<double>(timeout), [s] {
                                return s->frame_queue.size() > 0;});
                if (s->frame_queue.size() == 0) {
                        return;
                }
                frame = s->frame_queue.front();
                if (!frame) {
                        pop_frame(s, lk);
                        return;
                }

                glfwMakeContextCurrent(s->window);

                if (s->current_frame) {
                        if (s->paused) {
                                swap(frame, s->current_frame);
                        }
                        vf_recycle(s->current_frame);
                        s->free_frame_queue.push(s->current_frame);
                }
                s->current_frame = frame;
        }

        if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
                gl_reconfigure_screen(s, video_desc_from_frame(frame));
        }
        glBindTexture(GL_TEXTURE_2D, s->texture_display);

        gl_render(s, frame->tiles[0].data);
        if (s->deinterlace == state_gl::deint::force || (s->deinterlace == state_gl::deint::on && s->current_display_desc.interlacing == INTERLACED_MERGED)) {
                glUseProgram(s->PHandle_deint);
        }
        gl_draw(s->aspect, (s->dxt_height - s->current_display_desc.height) / (float) s->dxt_height * 2, s->vsync != SINGLE_BUF);
        glUseProgram(0);
        if (s->paused) {
                draw_pause();
        }

        // publish to Syphon/Spout
        if (s->syphon_spout) {
#ifdef HAVE_SYPHON
                syphon_server_publish(s->syphon_spout, frame->tiles[0].width, frame->tiles[0].height, s->texture_display);
#elif defined HAVE_SPOUT
                spout_sender_sendframe(s->syphon_spout, frame->tiles[0].width, frame->tiles[0].height, s->texture_display);
                glBindTexture(GL_TEXTURE_2D, s->texture_display);
#endif // HAVE_SPOUT
        }

        if (s->vsync == SINGLE_BUF) {
                glFlush();
        } else {
                glfwSwapBuffers(s->window);
        }
        log_msg(LOG_LEVEL_DEBUG, "Render buffer %dx%d\n", frame->tiles[0].width, frame->tiles[0].height);
        {
                unique_lock<mutex> lk(s->lock);
                pop_frame(s, lk);
        }
}

static int64_t translate_glfw_to_ug(int key, int mods) {
        key = tolower(key);
        if (mods == GLFW_MOD_CONTROL) {
                switch (key) {
                case GLFW_KEY_UP: return K_CTRL_UP;
                case GLFW_KEY_DOWN: return K_CTRL_DOWN;
                default:
                    return isalpha(key) ? K_CTRL(key) : -1;
                }
        } else if (mods == GLFW_MOD_ALT) {
                if (isalpha(key)) {
                        return K_ALT(key);
                }
                return -1;
        } else if (mods == 0) {
                switch (key) {
                case GLFW_KEY_LEFT: return K_LEFT;
                case GLFW_KEY_UP: return K_UP;
                case GLFW_KEY_RIGHT: return K_RIGHT;
                case GLFW_KEY_DOWN: return K_DOWN;
                case GLFW_KEY_PAGE_UP: return K_PGUP;
                case GLFW_KEY_PAGE_DOWN: return K_PGDOWN;
                default: return key;
                }
        }
        return -1;
}

static bool display_gl_process_key(struct state_gl *s, long long int key)
{
        verbose_msg(MOD_NAME "Key %lld pressed\n", key);
        switch (key) {
                case 'f':
                        {
                                s->fs = !s->fs;
                                int width = s->current_display_desc.width;
                                int height = s->current_display_desc.height;
                                GLFWmonitor *mon = s->fs ? s->monitor : nullptr;
                                if (mon && s->modeset == state_gl::modeset_t::NOMODESET) {
                                        const GLFWvidmode* mode = glfwGetVideoMode(mon);
                                        width = mode->width;
                                        height = mode->height;
                                }
                                int refresh_rate = get_refresh_rate(s->modeset, mon, s->current_display_desc.fps);
                                glfwSetWindowMonitor(s->window, mon, GLFW_DONT_CARE, GLFW_DONT_CARE, width, height, refresh_rate);
                                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Setting fullscreen: " << (s->fs ? "ON" : "OFF") << "\n";
                                set_gamma(s);
                                glfw_print_video_mode(s);
                                break;
                        }
                case 'q':
                        exit_uv(0);
                        break;
                case K_ALT('d'):
                        s->deinterlace = s->deinterlace == state_gl::deint::off ? state_gl::deint::on : state_gl::deint::off;
                        MSG(NOTICE, "Deinterlacing: %s\n",
                            state_gl::deint_to_string(s->deinterlace));
                        break;
                case K_ALT('p'):
                        s->paused = !s->paused;
                        MSG(NOTICE, "%s\n",
                            s->paused ? "Paused (press 'Alt-p' to unpause)"
                                      : "Unpaused");
                        break;
                case K_ALT('s'):
                        screenshot(s->current_frame);
                        break;
                case K_ALT('m'):
                        s->show_cursor = (state_gl::show_cursor_t) (((int) s->show_cursor + 1) % 3);
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Show cursor (0 - on, 1 - off, 2 - autohide): " << s->show_cursor << "\n";
                        glfwSetInputMode(s->window, GLFW_CURSOR, s->show_cursor == state_gl::SC_TRUE ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
                        break;
                case K_CTRL_UP:
                        s->window_size_factor *= 1.1;
                        glfw_resize_window(s->window, s->fs, s->current_display_desc.height, s->aspect, s->current_display_desc.fps, s->window_size_factor);
                        break;
                case K_CTRL_DOWN:
                        s->window_size_factor /= 1.1;
                        glfw_resize_window(s->window, s->fs, s->current_display_desc.height, s->aspect, s->current_display_desc.fps, s->window_size_factor);
                        break;
                default:
                        return false;
        }

        return true;
}

static void glfw_key_callback(GLFWwindow* win, int key, int /* scancode */, int action, int mods)
{
        if (action != GLFW_PRESS) {
                return;
        }
        auto *s = (struct state_gl *) glfwGetWindowUserPointer(win);
        char name[MAX_KEYCODE_NAME_LEN];
        int64_t ugk = translate_glfw_to_ug(key, mods);
        get_keycode_name(ugk, name, sizeof name);
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "%d pressed, modifiers: %d (UG name: %s)\n",
                        key, mods,
                        ugk > 0 ? name : "unknown");
        if (ugk == -1) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Cannot translate key %d (modifiers: %d)!\n", key, mods);
        }
        if (ugk <= 0) {
                return;
        }
        if (!display_gl_process_key(s, ugk)) { // keybinding not found -> pass to control
                keycontrol_send_key(get_root_module(&s->mod), ugk);
        }
}

static void glfw_mouse_callback(GLFWwindow *win, double /* x */, double /* y */)
{
        auto *s = (struct state_gl *) glfwGetWindowUserPointer(win);
        if (s->show_cursor == state_gl::SC_AUTOHIDE) {
                if (s->cursor_shown_from == steady_clock::time_point()) {
                        glfwSetInputMode(s->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                s->cursor_shown_from = steady_clock::now();
        }
}

static bool
check_display_gl_version(bool print_ver)
{
        auto version = (const char *) glGetString(GL_VERSION);
        if (!version) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get OpenGL version!\n");
                return false;
        }
        if (atof(version) < 2.0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "ERROR: OpenGL 2.0 is not supported, try updating your drivers...\n");
                return false;
        }
        if (print_ver) { // from help
                color_printf(TBOLD("OpenGL version:") " %s\n", version);
        } else {
                MSG(INFO, "OpenGL 2.0 is supported...\n");
                MSG(VERBOSE, "Supported OpenGL version is %s.\n", version);
        }
        return true;
}

static void display_gl_print_depth() {
        int bits[3];
        glGetIntegerv(GL_RED_BITS, &bits[0]);
        glGetIntegerv(GL_GREEN_BITS, &bits[1]);
        glGetIntegerv(GL_BLUE_BITS, &bits[2]);
        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Buffer depth - R: " << bits[0] << "b, G: " << bits[1] << "b, B: " << bits[2] << "b\n";
}

static void display_gl_render_last(GLFWwindow *win) {
        auto *s = (struct state_gl *) glfwGetWindowUserPointer(win);
        unique_lock<mutex> lk(s->lock);
        auto *f = s->current_frame;
        s->current_frame = nullptr;
        lk.unlock();
        if (!f) {
                return;
        }
        // redraw last frame
        display_gl_putf(s, f, PUTF_NONBLOCK);
}

#if defined GLEW_VERSION
#ifndef GLEW_ERROR_NO_GLX_DISPLAY
#define GLEW_ERROR_NO_GLX_DISPLAY 4
#endif
static void printGlewError(GLenum err) {
        const char *err_str = nullptr;
        switch (err) {
        case GLEW_ERROR_NO_GL_VERSION:
                err_str = "missing GL version";
                break;
        case GLEW_ERROR_GL_VERSION_10_ONLY:
                err_str = "Need at least OpenGL 1.1";
                break;
        case GLEW_ERROR_GLX_VERSION_11_ONLY:
                err_str = "Need at least GLX 1.2";
                break;
        case GLEW_ERROR_NO_GLX_DISPLAY:
                err_str = "Need GLX display for GLX support";
                break;
        default:
                err_str = (const char *) glewGetErrorString(err);
                break;
        }
        log_msg(err == GLEW_ERROR_NO_GLX_DISPLAY ? LOG_LEVEL_VERBOSE
                                                 : LOG_LEVEL_ERROR,
                MOD_NAME "GLEW Error: %s (err %d)\n", err_str, err);
}
#endif // defined GLEW_VERSION

static void glfw_print_error(int error_code, const char* description) {
        LOG(LOG_LEVEL_ERROR) << "GLFW error " << error_code << ": " << description << "\n";
}

/**
 * [mac specific] Sets user-specified color space.
 * @note patched (by us) GLFW supporting GLFW_COCOA_NS_COLOR_SPACE required
 */
static void set_mac_color_space(void) {
#ifdef GLFW_COCOA_NS_COLOR_SPACE
        const char *col = get_commandline_param("color");
        if (!col) {
                return;
        }
        glfwWindowHint(GLFW_COCOA_NS_COLOR_SPACE, stoi(col, nullptr, 16));
#endif // defined GLFW_COCOA_NS_COLOR_SPACE
}

static GLuint gl_substitute_compile_link(const char *vprogram, const char *fprogram)
{
        char *fp = strdup(fprogram);
        int index = 1;
        const char *col = get_commandline_param("color");
        if (col) {
                int color = stol(col, nullptr, 16) >> 4; // first nibble
                if (color > 0 && color <= 3) {
                        index = color;
                } else {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Wrong chromicities index " << color << "\n";
                }
        }
        if (get_commandline_param("color-601") != nullptr) {
                index = 0;
        }
        const double cs_coeffs[2 * 4] = { KR_601,  KB_601,  KR_709, KB_709,
                                          KR_2020, KB_2020, KR_P3,  KB_P3 };
        double kr = cs_coeffs[2 * index];
        double kb = cs_coeffs[2 * index + 1];
        const char *placeholders[] = { "Y_SCALED_PLACEHOLDER", "R_CR_PLACEHOLDER", "G_CB_PLACEHOLDER", "G_CR_PLACEHOLDER", "B_CB_PLACEHOLDER" };
        const struct color_coeffs cfs = compute_color_coeffs(kr, kb, 8);
        const double values[] = { (double) cfs.y_scale / (1 << COMP_BASE),
                                  (double) cfs.r_cr / (1 << COMP_BASE),
                                  (double) cfs.g_cb / (1 << COMP_BASE),
                                  (double) cfs.g_cr / (1 << COMP_BASE),
                                  (double) cfs.b_cb / (1 << COMP_BASE) };
        for (size_t i = 0; i < sizeof placeholders / sizeof placeholders[0]; ++i) {
                char *tok = fp;
                while ((tok = strstr(fp, placeholders[i])) != nullptr) {
                        memset(tok, ' ', strlen(placeholders[i]));
                        int written = snprintf(tok, sizeof placeholders - 1, "%.6f", values[i]);
                        tok[written] = ' ';
                }
        }
        GLuint ret = glsl_compile_link(vprogram, fp);
        free(fp);
        return ret;
}

static void
display_gl_set_window_hints(struct state_gl *s)
{
        for (auto const &hint : s->window_hints) {
                glfwWindowHint(hint.first, hint.second);
        }
#ifndef _WIN32
        if (s->noresizable) {
                glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        }
#endif

        /// @todo TOREMOVE
        const char *hints = get_commandline_param(GL_WINDOW_HINT_OPT_PARAM_NAME);
        if (hints == nullptr) {
                return;
        }
        vector<char> data(strlen(hints) + 1);
        copy(hints, hints + strlen(hints) + 1, data.begin());
        char *hints_c = data.data();
        char *tmp = hints_c;
        char *tok = nullptr;
        char *save_ptr = nullptr;
        while ((tok = strtok_r(tmp, ":", &save_ptr)) != nullptr) {
                tmp = nullptr;
                if (strchr(tok, '=') == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Malformed window hint - missing value (expected <k>=<v>).\n";
                        continue;
                }
                int key = stoi(tok, 0, 0);
                int val = stoi(strchr(tok, '=') + 1, 0, 0);
                glfwWindowHint(key, val);
        }
}

static void print_gamma_ramp(GLFWmonitor *monitor) {
        const struct GLFWgammaramp *ramp = glfwGetGammaRamp(monitor);
        if (ramp == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot get gamma ramp!\n";
                return;
        }
        ostringstream oss;
        oss << "Gamma ramp:\n";
        for (unsigned int i = 0; i < ramp->size; ++i) {
                oss << "r[" << i << "]=" << ramp->red[i] <<
                        ", g[" << i << "]=" << ramp->green[i] <<
                        ", b[" << i << "]=" << ramp->blue[i] << "\n";

        }
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << oss.str();
}

static void set_gamma(struct state_gl *s) {
        if (!s->gamma || !s->monitor) {
                return;
        }
        glfwSetGamma(s->monitor, s->gamma);
        if (log_level >= LOG_LEVEL_VERBOSE) {
                print_gamma_ramp(s->monitor);
        }
}

static bool
vdp_interop_supported()
{
#ifdef HWACC_VDPAU
        const char *hwacc_param = get_commandline_param("use-hw-accel");
        if (hwacc_param != nullptr && strcmp(hwacc_param, "vdpau-copy") == 0) {
                return false;
        }
        if (strstr((const char *) glGetString(GL_EXTENSIONS),
                   "GL_NV_vdpau_interop") != nullptr) {
                return true;
        }
        log_msg(hwacc_param == nullptr ? LOG_LEVEL_VERBOSE : LOG_LEVEL_WARNING,
                MOD_NAME "VDPAU interop NOT supported!\n");
        MSG(DEBUG, "Available extensions:%s\n", glGetString(GL_EXTENSIONS));
#endif
        return false;
}

ADD_TO_PARAM(GL_DISABLE_10B_OPT_PARAM_NAME ,
         "* " GL_DISABLE_10B_OPT_PARAM_NAME "\n"
         "  Disable 10 bit codec processing to improve performance\n");
ADD_TO_PARAM(GL_WINDOW_HINT_OPT_PARAM_NAME ,
         "* " GL_WINDOW_HINT_OPT_PARAM_NAME "=<k>=<v>[:<k2>=<v2>...]\n"
         "  Set window hint <k> to value <v> (deprecated, use option)\n");
/**
 * Initializes OpenGL stuff. If this function succeeds, display_gl_cleanup_opengl() needs
 * to be called to release resources.
 */
static bool display_gl_init_opengl(struct state_gl *s)
{
        glfwSetErrorCallback(glfw_print_error);

        if (ref_count_init_once<int>()(glfwInit, glfw_init_count)
                .value_or(GLFW_TRUE) == GLFW_FALSE) {
                LOG(LOG_LEVEL_ERROR) << "Cannot initialize GLFW!\n";
                return false;
        }

        gl_print_current_platform();

        if (s->req_monitor_idx != -1) {
                int           count = 0;
                GLFWmonitor **mon   = glfwGetMonitors(&count);
                if (s->req_monitor_idx < 0 || s->req_monitor_idx >= count) {
                        MSG(ERROR, "Wrong monitor index: %d (max %d)\n",
                            s->req_monitor_idx, count - 1);
                        return false;
                }
                s->monitor = mon[s->req_monitor_idx];
        }

        if (s->monitor == nullptr) {
                s->monitor = glfwGetPrimaryMonitor();
                if (s->monitor == nullptr) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "No monitor found! Continuing but full-screen will be disabled.\n";
                }
        }

        if (commandline_params.find(GL_DISABLE_10B_OPT_PARAM_NAME) == commandline_params.end()) {
                for (auto const & bits : {GLFW_RED_BITS, GLFW_GREEN_BITS, GLFW_BLUE_BITS}) {
                        glfwWindowHint(bits, 10);
                }
        }

        set_mac_color_space();
        glfwWindowHint(GLFW_DOUBLEBUFFER, s->vsync == SINGLE_BUF ? GLFW_FALSE : GLFW_TRUE);
        struct video_frame *splash = get_splashscreen();
        int width = splash->tiles[0].width;
        int height = splash->tiles[0].height;
        vf_free(splash);
        GLFWmonitor *mon = s->fs ? s->monitor : nullptr;
        if (s->fixed_size && s->fixed_w && s->fixed_h) {
                width = s->fixed_w;
                height = s->fixed_h;
        } else if (mon != nullptr && s->modeset == state_gl::modeset_t::NOMODESET) {
                const GLFWvidmode* mode = glfwGetVideoMode(mon);
                width = mode->width;
                height = mode->height;
                glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        }
        display_gl_set_window_hints(s);
        if ((s->window = glfwCreateWindow(width, height, IF_NOT_NULL_ELSE(get_commandline_param("window-title"), DEFAULT_WIN_NAME), nullptr, nullptr)) == nullptr) {
                return false;
        }
        if (s->pos_x != INT_MIN) {
                const int y = s->pos_y == INT_MIN ? 0 : s->pos_y;
                glfwSetWindowPos(s->window, s->pos_x, y);
        }
        if (s->noresizable) {
                glfwSetWindowSizeLimits(s->window, width, height, width,
                                        height);
        }
        if (mon != nullptr) { /// @todo remove/revert when no needed (see particular commit message
                glfwSetWindowMonitor(s->window, mon, GLFW_DONT_CARE, GLFW_DONT_CARE, width, height, get_refresh_rate(s->modeset, mon, GLFW_DONT_CARE));
        }
        glfw_print_video_mode(s);
        glfwSetWindowUserPointer(s->window, s);
        set_gamma(s);
        glfwSetInputMode(s->window, GLFW_CURSOR, s->show_cursor == state_gl::SC_TRUE ?  GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
        glfwMakeContextCurrent(s->window);
        glfwSetKeyCallback(s->window, glfw_key_callback);
        glfwSetCursorPosCallback(s->window, glfw_mouse_callback);
        glfwSetWindowCloseCallback(s->window, glfw_close_callback);
        glfwSetFramebufferSizeCallback(s->window, gl_resize);
        glfwSetWindowRefreshCallback(s->window, display_gl_render_last);

#if defined GLEW_VERSION
        if (GLenum err = glewInit()) {
                printGlewError(err);
                if (err != GLEW_ERROR_NO_GLX_DISPLAY) { // do not fail on error 4 (on Wayland), which can be suppressed
                        return false;
                }
        }
#endif
        if (!check_display_gl_version(false)) {
                glfwDestroyWindow(s->window);
                s->window = nullptr;
                return false;
        }
        display_gl_print_depth();

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
        glEnable( GL_TEXTURE_2D );

        glGenTextures(1, &s->texture_display);
        glBindTexture(GL_TEXTURE_2D, s->texture_display);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &s->texture_raw);
        glBindTexture(GL_TEXTURE_2D, s->texture_raw);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        for (auto &it : glsl_programs) {
                GLuint prog = gl_substitute_compile_link(vert, it.second);
                if (prog == 0U) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to link program for %s!\n", get_codec_name(it.first));
                        handle_error(1);
                        continue;
                }
                s->PHandles[it.first] = prog;
        }

        if ((s->PHandle_deint = gl_substitute_compile_link(vert, deinterlace_fp)) == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to compile deinterlace program!\n");
                handle_error(1);
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // set row alignment to 1 byte instead of default
                                               // 4 bytes which won't work on row-unaligned RGB

        // Create fbo
        glGenFramebuffersEXT(1, &s->fbo_id);

        glGenBuffersARB(1, &s->pbo_id);

        s->vdp_interop = vdp_interop_supported();

        glfwMakeContextCurrent(nullptr);

        return true;
}

/// Releases resources allocated with display_gl_init_opengl. After this function,
/// the resource references are invalid.
static void display_gl_cleanup_opengl(struct state_gl *s){
        glfwMakeContextCurrent(s->window);

        for (auto &it : s->PHandles) {
                glDeleteProgram(it.second);
        }
        glDeleteTextures(1, &s->texture_display);
        glDeleteTextures(1, &s->texture_raw);
        glDeleteFramebuffersEXT(1, &s->fbo_id);
        glDeleteBuffersARB(1, &s->pbo_id);
        glfwDestroyWindow(s->window);

        if (s->syphon_spout) {
#ifdef HAVE_SYPHON
                syphon_server_unregister(s->syphon_spout);
#elif defined HAVE_SPOUT
                spout_sender_unregister(s->syphon_spout);
#endif
        }
}

static void display_gl_run(void *arg)
{
        struct state_gl *s = 
                (struct state_gl *) arg;

        glfwMakeContextCurrent(s->window);
        while (!glfwWindowShouldClose(s->window)) {
                glfwPollEvents();
                gl_process_frames(s);
        }
        glfwMakeContextCurrent(nullptr);
}

static void gl_change_aspect(struct state_gl *s, int width, int height)
{
        double x = 1.0,
               y = 1.0;

        glViewport( 0, 0, ( GLint )width, ( GLint )height );

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity( );

        double screen_ratio = (double) width / height;
        if(screen_ratio > s->aspect) {
                x = (double) height * s->aspect / width;
        } else {
                y = (double) width / (height * s->aspect);
        }
        glScalef(x, y, 1);

        glOrtho(-1,1,-1/s->aspect,1/s->aspect,10,-10);
}

static void gl_resize(GLFWwindow *win, int width, int height)
{
        auto *s = (struct state_gl *) glfwGetWindowUserPointer(win);
        debug_msg("Resized to: %dx%d\n", width, height);

        gl_change_aspect(s, width, height);

        if (s->vsync == SINGLE_BUF) {
                glDrawBuffer(GL_FRONT);
                /* Clear the screen */
                glClear(GL_COLOR_BUFFER_BIT);
        }
}

static void upload_compressed_texture(struct state_gl *s, char *data) {
        switch (s->current_display_desc.color_spec) {
                case DXT1:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        (s->current_display_desc.width + 3) / 4 * 4, s->dxt_height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        ((s->current_display_desc.width + 3) / 4 * 4 * s->dxt_height)/2,
                                        data);
                        break;
                case DXT1_YUV:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->current_display_desc.width, s->current_display_desc.height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        (s->current_display_desc.width * s->current_display_desc.height/16)*8,
                                        data);
                        break;
                case DXT5:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        (s->current_display_desc.width + 3) / 4 * 4, s->dxt_height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                        (s->current_display_desc.width + 3) / 4 * 4 * s->dxt_height,
                                        data);
                        break;
                default:
                        abort();
        }
}

static void upload_texture(struct state_gl *s, char *data)
{
#ifdef HWACC_VDPAU
        if (s->current_display_desc.color_spec == HW_VDPAU) {
                s->vdp.loadFrame(reinterpret_cast<hw_vdpau_frame *>(data));
                return;
        }
#endif
        if (s->current_display_desc.color_spec == DXT1 || s->current_display_desc.color_spec == DXT1_YUV || s->current_display_desc.color_spec == DXT5) {
                upload_compressed_texture(s, data);
                return;
        }
        GLuint format = s->current_display_desc.color_spec == RGB || s->current_display_desc.color_spec == RG48 ? GL_RGB : GL_RGBA;
        GLenum type = GL_UNSIGNED_BYTE;
        if (s->current_display_desc.color_spec == R10k || s->current_display_desc.color_spec == v210) {
                type = GL_UNSIGNED_INT_2_10_10_10_REV;
        } else if (s->current_display_desc.color_spec == Y416 || s->current_display_desc.color_spec == RG48) {
                type = GL_UNSIGNED_SHORT;
        }
        GLint width = s->current_display_desc.width;
        if (s->current_display_desc.color_spec == UYVY || s->current_display_desc.color_spec == v210) {
                width = vc_get_linesize(width, s->current_display_desc.color_spec) / 4;
        }
        /// swaps bytes and removes 256B padding
        auto process_r10k = [](uint32_t * __restrict out, const uint32_t *__restrict in, long width, long height) {
                DEBUG_TIMER_START(process_r10k);
                long line_padding_b = vc_get_linesize(width, R10k) - 4 * width;
                for (long i = 0; i < height; i += 1) {
                        OPTIMIZED_FOR (long j = 0; j < width; j += 1) {
                                uint32_t x = *in++;
                                *out++ = /* output is x2b8g8r8 little-endian */
                                        (x & 0xFFU) << 2U | (x & 0xC0'00U) >> 14U | // R
                                        (x & 0x3F'00U) << 6U | (x & 0xF0'00'00) >> 10U | // G
                                        (x & 0x0F'00'00U) << 10U | (x & 0xFC'00'00'00U) >> 6U; // B
                        }
                        in += line_padding_b / sizeof(uint32_t);
                }
                DEBUG_TIMER_STOP(process_r10k);
        };
        int data_size = vc_get_linesize(s->current_display_desc.width, s->current_display_desc.color_spec) * s->current_display_desc.height;
        if (s->use_pbo) {
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id); // current pbo
                glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
                if (void *ptr = glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB)) {
                        // update data directly on the mapped buffer
                        if (s->current_display_desc.color_spec == R10k) { // perform byte swap
                                process_r10k(static_cast<uint32_t *>(ptr), reinterpret_cast<uint32_t *>(data), s->current_display_desc.width, s->current_display_desc.height);
                        } else {
                                memcpy(ptr, data, data_size);
                        }
                        glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
                }
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, s->current_display_desc.height, format, type, nullptr);

                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        } else {
                if (s->current_display_desc.color_spec == R10k) { // perform byte swap
                        process_r10k(reinterpret_cast<uint32_t *>(s->scratchpad.data()), reinterpret_cast<uint32_t *>(data), s->current_display_desc.width, s->current_display_desc.height);
                        data = s->scratchpad.data();
                }
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, s->current_display_desc.height, format, type, data);
        }
}

static bool check_rpi_pbo_quirks()
{
#if ! defined __linux__
        return false;
#else
        std::ifstream cpuinfo("/proc/cpuinfo");
        if (!cpuinfo)
                return false;

        bool detected_rpi = false;
        bool detected_bcm2835 = false;
        std::string line;
        while (std::getline(cpuinfo, line) && !detected_rpi) {
                detected_bcm2835 |= line.find("BCM2835") != std::string::npos;
                detected_rpi |= line.find("Raspberry Pi") != std::string::npos;
        }

        return detected_rpi || detected_bcm2835;
#endif
}

static void gl_render_glsl(struct state_gl *s, char *data)
{
        int status;
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, s->fbo_id);
        gl_check_error();
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture_display, 0);
        status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
        assert(status == GL_FRAMEBUFFER_COMPLETE_EXT);
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, s->texture_raw);

        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
        glLoadIdentity( );

        glMatrixMode( GL_MODELVIEW );
        glPushMatrix();
        glLoadIdentity( );

        glPushAttrib(GL_VIEWPORT_BIT);

        glViewport( 0, 0, s->current_display_desc.width, s->current_display_desc.height);

        upload_texture(s, data);
        gl_check_error();

        glUseProgram(s->current_program);

        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
        gl_check_error();

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();

        glPopAttrib();

        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );
        glPopMatrix();

        glUseProgram(0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, s->texture_display);
}    

static void gl_draw(double ratio, double bottom_offset, bool double_buf)
{
        float bottom;
        gl_check_error();

        glDrawBuffer(double_buf ? GL_BACK : GL_FRONT);
        if (double_buf) {
                glClear(GL_COLOR_BUFFER_BIT);
        }

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity( );
        glTranslatef( 0.0f, 0.0f, -1.35f );

        /* Reflect that we may have higher texture than actual data
         * if we use DXT and source height was not divisible by 4 
         * In normal case, there would be 1.0 */
        bottom = 1.0f - bottom_offset;

        gl_check_error();
        glBegin(GL_QUADS);
        /* Front Face */
        /* Bottom Left Of The Texture and Quad */
        glTexCoord2f( 0.0f, bottom ); glVertex2f( -1.0f, -1/ratio);
        /* Bottom Right Of The Texture and Quad */
        glTexCoord2f( 1.0f, bottom ); glVertex2f(  1.0f, -1/ratio);
        /* Top Right Of The Texture and Quad */
        glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  1/ratio);
        /* Top Left Of The Texture and Quad */
        glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  1/ratio);
        glEnd( );

        gl_check_error();
}

static void glfw_close_callback(GLFWwindow *win)
{
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        exit_uv(0);
}

static bool
display_gl_get_property(void *state, int property, void *val, size_t *len)
{
        auto *s = (struct state_gl *) state;
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
        case DISPLAY_PROPERTY_CODECS: {
                UG_ASSERT(*len >= sizeof gl_supp_codecs);
                auto filter_codecs = [s](codec_t c) {
                        if (get_bits_per_component(c) > DEPTH8 &&
                            commandline_params.find(
                                GL_DISABLE_10B_OPT_PARAM_NAME) !=
                                commandline_params.end()) { // option to disable
                                                            // 10-bit processing
                                return false;
                        }
                        if (glsl_programs.find(c) != glsl_programs.end() &&
                            s->PHandles.find(c) ==
                                s->PHandles.end()) { // GLSL shader needed but
                                                     // compilation failed
                                return false;
                        }
                        if (c == HW_VDPAU && !s->vdp_interop) {
                                return false;
                        }
                        return true;
                };
                const codec_t *endptr =
                    copy_if(begin(gl_supp_codecs), end(gl_supp_codecs),
                            (codec_t *) val, filter_codecs);
                *len = (const char *) endptr - (char *) val;
                return true;
        }
        case DISPLAY_PROPERTY_RGB_SHIFT:
                UG_ASSERT(*len >= sizeof rgb_shift);
                memcpy(val, rgb_shift, sizeof(rgb_shift));
                *len = sizeof(rgb_shift);
                return true;
        case DISPLAY_PROPERTY_BUF_PITCH:
                UG_ASSERT(*len >= sizeof(int));
                *(int *) val = PITCH_DEFAULT;
                *len         = sizeof(int);
                return true;
        case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                UG_ASSERT(*len >= sizeof supported_il_modes);
                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                *len = sizeof(supported_il_modes);
                return true;
        default:
                return false;
        }
}

static void display_gl_done(void *state)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        display_gl_cleanup_opengl(s);

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                vf_free(buffer);
        }

        while (s->frame_queue.size() > 0) {
                struct video_frame *buffer = s->frame_queue.front();
                s->frame_queue.pop();
                vf_free(buffer);
        }

        vf_free(s->current_frame);

        delete s;
}

static struct video_frame * display_gl_getf(void *state)
{
        struct state_gl *s = (struct state_gl *) state;
        assert(s->magic == MAGIC_GL);

        lock_guard<mutex> lock(s->lock);

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
                        return buffer;
                }
                vf_free(buffer);
        }

        struct video_frame *buffer = vf_alloc_desc_data(s->current_desc);
        vf_clear(buffer);
        return buffer;
}

static bool display_gl_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        unique_lock<mutex> lk(s->lock);

        if(!frame) {
                glfwSetWindowShouldClose(s->window, GLFW_TRUE);
                s->frame_queue.push(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return true;
        }

        switch (timeout_ns) {
                case PUTF_DISCARD:
                        vf_recycle(frame);
                        s->free_frame_queue.push(frame);
                        return 0;
                case PUTF_BLOCKING:
                        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
                        break;
                case PUTF_NONBLOCK:
                        break;
                default:
                        s->frame_consumed_cv.wait_for(lk, timeout_ns * 1ns, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
                        break;
        }
        if (s->frame_queue.size() >= MAX_BUFFER_SIZE) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "1 frame(s) dropped!\n";
                vf_recycle(frame);
                s->free_frame_queue.push(frame);
                return false;
        }
        s->frame_queue.push(frame);

        lk.unlock();
        s->new_frame_ready_cv.notify_one();

        return !s->paused;
}

static const struct video_display_info display_gl_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].dev, "");
                strcpy((*available_cards)[0].name, "OpenGL SW display");

                dev_add_option(&(*available_cards)[0], "Deinterlace", "Deinterlace", "deinterlace", ":d", true);
                dev_add_option(&(*available_cards)[0], "Fullscreen", "Launch as fullscreen", "fullscreen", ":fs", true);
                dev_add_option(&(*available_cards)[0], "No decorate", "Disable window decorations", "nodecorate", ":nodecorate", true);
                dev_add_option(&(*available_cards)[0], "Show cursor", "Show visible cursor", "cursor", ":cursor", true);
                dev_add_option(&(*available_cards)[0], "Disable vsync", "Disable vsync", "novsync", ":novsync", true);
                dev_add_option(&(*available_cards)[0], "Aspect", "Requested video aspect <w>/<h>", "aspect", ":aspect=", false);

                (*available_cards)[0].repeatable = true;
        },
        display_gl_init,
        display_gl_run,
        display_gl_done,
        display_gl_getf,
        display_gl_putf,
        display_gl_reconfigure,
        display_gl_get_property,
        NULL,
        NULL,
        MOD_NAME,
};

REGISTER_MODULE(gl, &display_gl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

#if defined HAVE_SPOUT || HAVE_SYPHON
#if defined HAVE_SPOUT && HAVE_SYPHON
#error "Both SPOUT and Syphon defined, please fix!"
#endif // defined HAVE_SPOUT && HAVE_SYPHON
static const struct video_display_info display_spout_syphon_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                *deleter = free;
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
#ifdef HAVE_SPOUT
                strcpy((*available_cards)[0].dev, "syphon");
                strcpy((*available_cards)[0].name, "Syphon display");
#else
                strcpy((*available_cards)[0].dev, "spout");
                strcpy((*available_cards)[0].name, "SPOUT display");
#endif
        },
        display_spout_syphon_init,
        display_gl_run,
        display_gl_done,
        display_gl_getf,
        display_gl_putf,
        display_gl_reconfigure,
        display_gl_get_property,
        NULL,
        NULL,
        "[" NAME_SPOUT_SYPHON "] ",
};
REGISTER_MODULE(NAME_SPOUT_SYPHON_RAW, &display_spout_syphon_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
#endif // defined HAVE_SPOUT || HAVE_SYPHON

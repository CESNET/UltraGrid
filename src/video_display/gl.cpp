/**
 * @file   video_display/gl.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2022 CESNET, z. s. p. o.
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

#include <assert.h>
#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#else
#include <GL/glew.h>
#endif /* HAVE_MACOSX */
#include <GLFW/glfw3.h>

#include "spout_sender.h"
#include "syphon_server.h"

#include <array>
#include <algorithm>
#include <condition_variable>
#include <csetjmp>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>

#include "color.h"
#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rang.hpp"
#include "utils/color_out.h"
#include "utils/macros.h" // OPTIMIZED_FOR
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"
#include "tv.h"

#define MAGIC_GL         0x1331018e
#define MOD_NAME         "[GL] "
#define DEFAULT_WIN_NAME "Ultragrid - OpenGL Display"
#define GL_DISABLE_10B_OPT_PARAM_NAME "gl-disable-10b"
#define GL_WINDOW_HINT_OPT_PARAM_NAME "glfw-window-hint"
#define MAX_BUFFER_SIZE 1
#define ADAPTIVE_VSYNC -1
#define SYSTEM_VSYNC 0xFE
#define SINGLE_BUF 0xFF // use single buffering instead of double

#include "gl_vdpau.hpp"

using namespace std;

static const char * uyvy_to_rgb_fp = R"raw(
#version 110
uniform sampler2D image;
uniform float imageWidth;
void main()
{
        vec4 yuv;
        yuv.rgba  = texture2D(image, gl_TexCoord[0].xy).grba;
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
  float imageWidthOrig;
  imageWidthOrig = float((int(imageWidth) + 47) / 48 * 32);

  // interpolate 0,0 -> 1,1 texcoords to 0,0 -> 720,486
  int texcoordDenormX;
  texcoordDenormX = int((2. * gl_TexCoord[0].x * imageWidth - 1.) / 2.);

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
  y = texture2D(image, vec2(float(2 * sourceColumnIndexY + 1) / (2. * imageWidthOrig), gl_TexCoord[0].y));
  u = texture2D(image, vec2(float(2 * sourceColumnIndexU + 1) / (2. * imageWidthOrig), gl_TexCoord[0].y));
  v = texture2D(image, vec2(float(2 * sourceColumnIndexV + 1) / (2. * imageWidthOrig), gl_TexCoord[0].y));


  vec3 outColor = ycbcr2rgb(vec3(y[compY], u[compU], v[compV]));

  gl_FragColor = vec4(outColor, 1.0);
}
)raw";

/* DXT YUV (FastDXT) related */
static const char *fp_display_dxt1 = R"raw(
#version 110
uniform sampler2D yuvtex;

void main(void) {
        vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);

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

static constexpr array keybindings{
        pair<int64_t, string_view>{'f', "toggle fullscreen"},
        pair<int64_t, string_view>{'q', "quit"},
        pair<int64_t, string_view>{'d', "toggle deinterlace"},
        pair<int64_t, string_view>{'p', "pause video"},
        pair<int64_t, string_view>{K_ALT('s'), "screenshot"},
        pair<int64_t, string_view>{K_ALT('c'), "show/hide cursor"},
        pair<int64_t, string_view>{K_CTRL_DOWN, "make window 10% smaller"},
        pair<int64_t, string_view>{K_CTRL_UP, "make window 10% bigger"}
};

struct state_gl {
        GLuint          PHandle_uyvy = 0;
        GLuint          PHandle_yuva = 0;
        GLuint          PHandle_v210 = 0;
        GLuint          PHandle_dxt = 0;
        GLuint          PHandle_dxt5 = 0;
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
        bool            deinterlace  = false;

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

        int             dxt_height = 0;

        int             vsync = 1;
        bool            paused = false;
        enum show_cursor_t { SC_TRUE, SC_FALSE, SC_AUTOHIDE } show_cursor = SC_AUTOHIDE;
        chrono::steady_clock::time_point                      cursor_shown_from{}; ///< indicates time point from which is cursor show if show_cursor == SC_AUTOHIDE, timepoint() means cursor is not currently shown
        string          syphon_spout_srv_name;

        double          window_size_factor = 1.0;

        struct module   mod;

        void *syphon_spout = nullptr;
        bool hide_window = false;

        bool fixed_size = false;
        int fixed_w = 0;
        int fixed_h = 0;

        enum modeset_t { MODESET = -2, MODESET_SIZE_ONLY = GLFW_DONT_CARE, NOMODESET = 0 } modeset = NOMODESET; ///< positive vals force framerate
        bool nodecorate = false;
        int use_pbo = -1;

#ifdef HWACC_VDPAU
        struct state_vdpau vdp;
#endif

        state_gl(struct module *parent) {
                if (ref_count_init_once<int>()(glfwInit, glfw_init_count).value_or(GLFW_TRUE) == GLFW_FALSE) {
                        LOG(LOG_LEVEL_ERROR) << "Cannot initialize GLFW!\n";
                        throw false;
                }
                module_init_default(&mod);
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);
        }

        ~state_gl() {
                ref_count_terminate_last()(glfwTerminate, glfw_init_count);

                module_done(&mod);
        }

        vector<char> scratchpad; ///< scratchpad sized WxHx8
};

static constexpr array gl_supp_codecs = {
#ifdef HWACC_VDPAU
        HW_VDPAU,
#endif
        UYVY,
        v210,
        R10k,
        RGBA,
        RGB,
        Y416,
        DXT1,
        DXT1_YUV,
        DXT5
};

/* Prototyping */
static bool display_gl_init_opengl(struct state_gl *s);
static int display_gl_putf(void *state, struct video_frame *frame, int nonblock);
static bool display_gl_process_key(struct state_gl *s, long long int key);
static int display_gl_reconfigure(void *state, struct video_desc desc);

static void gl_draw(double ratio, double bottom_offset, bool double_buf);

static void gl_change_aspect(struct state_gl *s, int width, int height);
static void gl_resize(GLFWwindow *win, int width, int height);
static void gl_render_glsl(struct state_gl *s, char *data);
static void gl_reconfigure_screen(struct state_gl *s, struct video_desc desc);
static void gl_process_frames(struct state_gl *s);
static void glfw_key_callback(GLFWwindow* win, int key, int scancode, int action, int mods);
static void glfw_mouse_callback(GLFWwindow *win, double x, double y);
static void glfw_close_callback(GLFWwindow *win);
static void display_gl_set_sync_on_vblank(int value);
static void screenshot(struct video_frame *frame);
static void upload_texture(struct state_gl *s, char *data);
static bool check_rpi_pbo_quirks();

static void gl_print_monitors(bool fullhelp) {
        if (ref_count_init_once<int>()(glfwInit, glfw_init_count).value_or(GLFW_TRUE) == GLFW_FALSE) {
                LOG(LOG_LEVEL_ERROR) << "Cannot initialize GLFW!\n";
                return;
        }
        printf("\nmonitors:\n");
        int count = 0;
        GLFWmonitor **mon = glfwGetMonitors(&count);
        if (count <= 0) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "No monitors found!\n";
        }
        GLFWmonitor *primary = glfwGetPrimaryMonitor();
        for (int i = 0; i < count; ++i) {
                cout << "\t" << (mon[i] == primary ? "*" : " ") << BOLD(i) << ") " << glfwGetMonitorName(mon[i]);
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

        ref_count_terminate_last()(glfwTerminate, glfw_init_count);
}

/**
 * Show help
 */
static void gl_show_help(bool full) {
        cout << "usage:\n";
        cout << rang::style::bold << rang::fg::red << "\t-d gl" << rang::fg::reset << "[:d|:fs[=<monitor>]|:aspect=<v>/<h>|:cursor|:size=X%%|:syphon[=<name>]|:spout[=<name>]|:modeset[=<fps>]|:nodecorate|:fixed_size[=WxH]|:vsync[=<x>|single]]* | gl:[full]help"
                << (full ? " [--param " GL_DISABLE_10B_OPT_PARAM_NAME "|" GL_WINDOW_HINT_OPT_PARAM_NAME "=<k>=<v>]" : "")
                << "\n\n" << rang::style::reset;
        cout << "options:\n";
        cout << BOLD("\td")           << "\t\tdeinterlace\n";
        cout << BOLD("\tfs[=<monitor>]") << "\tfullscreen with optional display specification\n";
        cout << BOLD("\tmodeset[=<fps>]")<< "\tset received video mode as display mode (in fullscreen); modeset=<fps>|size - set specified FPS or only size\n";
        cout << BOLD("\tnodecorate")  << "\tdisable window decorations\n";
        cout << BOLD("\tnovsync")     << "\t\tdo not turn sync on VBlank\n";
        cout << BOLD("\tvsync=<x>")   << "\tsets vsync to: 0 - disable; 1 - enable; -1 - adaptive vsync; D - leaves system default\n";
        cout << BOLD("\tsingle")      << "\t\tuse single buffer (instead of double-buffering)\n";
        cout << BOLD("\taspect=<w>/<h>") << "\trequested video aspect (eg. 16/9). Leave unset if PAR = 1.\n";
        cout << BOLD("\tcursor")      << "\t\tshow visible cursor\n";
        cout << BOLD("\tsize")        << "\t\tspecifies desired size of window compared "
                "to native resolution (in percents)\n";
        cout << BOLD("\tsyphon")      << "\t\tuse Syphon (optionally with name)\n";
        cout << BOLD("\tspout")       << "\t\tuse Spout (optionally with name)\n";
        cout << BOLD("\thide-window") << "\tdo not show OpenGL window (useful with Syphon/SPOUT)\n";
        cout << BOLD("\t[no]pbo")     << "\t\tWhether or not use PBO (ignore if not sure)\n";
        if (full) {
                cout << BOLD("\t--param " GL_DISABLE_10B_OPT_PARAM_NAME)     << "\tdo not set 10-bit framebuffer (performance issues)\n";
                cout << BOLD("\t--param " GL_WINDOW_HINT_OPT_PARAM_NAME) << "<k>=<v>[:<k2>=<v2>] set GLFW window hint key <k> to value <v>, eg. 0x20006=1 to autoiconify (experts only)\n";
        }

        printf("\nkeyboard shortcuts:\n");
        for (auto i : keybindings) {
                char keyname[50];
                get_keycode_name(i.first, keyname, sizeof keyname);
                cout << "\t" << BOLD(keyname) << "\t\t" << i.second << "\n";
        }

        gl_print_monitors(full);
}

static void gl_load_splashscreen(struct state_gl *s)
{
        struct video_desc desc;

        desc.width = 512;
        desc.height = 512;
        desc.color_spec = RGBA;
        desc.interlacing = PROGRESSIVE;
        desc.fps = 1;
        desc.tile_count = 1;

        display_gl_reconfigure(s, desc);

        struct video_frame *frame = vf_alloc_desc_data(desc);

        const char *data = splash_data;
        memset(frame->tiles[0].data, 0, frame->tiles[0].data_len);
        for (unsigned int y = 0; y < splash_height; ++y) {
                char *line = frame->tiles[0].data;
                line += vc_get_linesize(frame->tiles[0].width,
                                frame->color_spec) *
                        (((frame->tiles[0].height - splash_height) / 2) + y);
                line += vc_get_linesize(
                                (frame->tiles[0].width - splash_width)/2,
                                frame->color_spec);
                for (unsigned int x = 0; x < splash_width; ++x) {
                        HEADER_PIXEL(data,line);
                        line += 4;
                }
        }

        s->frame_queue.push(frame);
}

static void *display_gl_parse_fmt(struct state_gl *s, char *ptr) {
        if (strstr(ptr, "help") != 0) {
                gl_show_help(strcmp(ptr, "fullhelp") == 0);
                return &display_init_noerr;
        }

        char *tok, *save_ptr = NULL;

        while((tok = strtok_r(ptr, ":", &save_ptr)) != NULL) {
                if(!strcmp(tok, "d")) {
                        s->deinterlace = true;
                } else if(!strncmp(tok, "fs", 2)) {
                        s->fs = true;
                        if (char *val = strchr(tok, '=')) {
                                val += 1;
                                int idx = stoi(val);
                                int count = 0;
                                GLFWmonitor **mon = glfwGetMonitors(&count);
                                if (idx >= count) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Wrong monitor index: " << idx << " (max " << count - 1 << ")\n";
                                        return nullptr;
                                }
                                s->monitor = mon[idx];
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
                        s->nodecorate = true;
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
                        return nullptr;
#endif
                } else if (!strcasecmp(tok, "hide-window")) {
                        s->hide_window = true;
                } else if (strcasecmp(tok, "pbo") == 0 || strcasecmp(tok, "nopbo") == 0) {
                        s->use_pbo = strcasecmp(tok, "pbo") == 0 ? 1 : 0;
                } else if(!strncmp(tok, "size=",
                                        strlen("size="))) {
                        s->window_size_factor =
                                atof(tok + strlen("size=")) / 100.0;
                } else if (strncmp(tok, "fixed_size", strlen("fixed_size")) == 0) {
                        s->fixed_size = true;
                        if (strncmp(tok, "fixed_size=", strlen("fixed_size=")) == 0) {
                                char *size = tok + strlen("fixed_size=");
                                if (strchr(size, 'x')) {
                                        s->fixed_w = atoi(size);
                                        s->fixed_h = atoi(strchr(size, 'x') + 1);
                                }
                        }
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", tok);
                        return nullptr;
                }
                ptr = NULL;
        }

        return s;
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
                void *ret = nullptr;
                try {
                        ret = display_gl_parse_fmt(s, tmp);
                } catch (std::invalid_argument &e) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Invalid numeric value for an option!\n";
                }
                free(tmp);
                if (ret != s) { // ret is either nullptr or &display_init_noerr (help requested)
                        delete s;
                        return ret;
                }
        }

        s->use_pbo = s->use_pbo == -1 ? !check_rpi_pbo_quirks() : s->use_pbo; // don't use PBO for Raspberry Pi (better performance)

        log_msg(LOG_LEVEL_INFO,"GL setup: fullscreen: %s, deinterlace: %s\n",
                        s->fs ? "ON" : "OFF", s->deinterlace ? "ON" : "OFF");

        if (s->fixed_size && s->fixed_w && s->fixed_h) {
                struct video_desc desc;
                desc.width = s->fixed_w;
                desc.height = s->fixed_h;
                desc.color_spec = RGBA;
                desc.interlacing = PROGRESSIVE;
                desc.fps = 1;
                desc.tile_count = 1;

                s->frame_queue.push(vf_alloc_desc_data(desc));
        }

        gl_load_splashscreen(s);
        for (auto i : keybindings) {
                if (i.first == 'q') { // don't report 'q' to avoid accidental close - user can use Ctrl-c there
                        continue;
                }
                char msg[18];
                sprintf(msg, "%" PRIx64, i.first);
                keycontrol_register_key(&s->mod, i.first, msg, i.second.data());
        }

        if (!display_gl_init_opengl(s)) {
                delete s;
                return nullptr;
        }

        return (void*)s;
}

/**
 * This function just sets new video description.
 */
static int display_gl_reconfigure(void *state, struct video_desc desc)
{
        struct state_gl	*s = (struct state_gl *) state;

        assert (find(gl_supp_codecs.begin(), gl_supp_codecs.end(), desc.color_spec) != gl_supp_codecs.end());
        if (get_bits_per_component(desc.color_spec) > 8) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Displaying 10+ bits - performance degradation may occur, consider '--param " GL_DISABLE_10B_OPT_PARAM_NAME "'\n";
        }

        s->current_desc = desc;

        return TRUE;
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
                int width = height * aspect;
                GLFWmonitor *mon = s->monitor;
                int refresh_rate = get_refresh_rate(s->modeset, mon, fps);
                if (s->modeset == state_gl::modeset_t::NOMODESET) {
                        const GLFWvidmode* mode = glfwGetVideoMode(mon);
                        width = mode->width;
                        height = mode->height;
                }
                glfwSetWindowMonitor(win, mon, GLFW_DONT_CARE, GLFW_DONT_CARE, width, height, refresh_rate);
        } else {
                glfwSetWindowSize(win, window_size_factor * height * aspect, window_size_factor * height);
        }
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

#ifdef WIN32
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

        if(desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV || desc.color_spec == DXT5) {
                s->dxt_height = (desc.height + 3) / 4 * 4;
        } else {
                s->dxt_height = desc.height;
        }

        if(!s->video_aspect)
                s->aspect = (double) desc.width / desc.height;
        else
                s->aspect = s->video_aspect;

        log_msg(LOG_LEVEL_INFO, "Setting GL size %dx%d (%dx%d).\n", (int)(s->aspect * desc.height),
                        desc.height, desc.width, desc.height);
        if (!s->hide_window)
                glfwShowWindow(s->window);

        glUseProgram(0);

        gl_check_error();

        if(desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
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
                        glBindTexture(GL_TEXTURE_2D,s->texture_display);
                        glUseProgram(s->PHandle_dxt);
                }
        } else if (desc.color_spec == UYVY) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_raw);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width / 2, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                s->current_program = s->PHandle_uyvy;
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
                s->current_program = s->PHandle_v210;
        } else if (desc.color_spec == Y416) {
                s->current_program = s->PHandle_yuva;
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
        } else if (desc.color_spec == DXT5) {
                glUseProgram(s->PHandle_dxt5);

                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                (desc.width + 3) / 4 * 4, s->dxt_height, 0,
                                (desc.width + 3) / 4 * 4 * s->dxt_height,
                                NULL);
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

        gl_check_error();

        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, vc_get_linesize(desc.width, desc.color_spec) * desc.height, 0, GL_STREAM_DRAW_ARB);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        gl_check_error();

        if (!s->fixed_size) {
                glfw_resize_window(s->window, s->fs, desc.height, s->aspect, desc.fps, s->window_size_factor);
                gl_resize(s->window, desc.width, desc.height);
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
                if (s->syphon_spout) {
                         spout_sender_unregister(s->syphon_spout);
                }
                s->syphon_spout = spout_sender_register(s->syphon_spout_srv_name.c_str(), desc.width, desc.height);
	}
#endif

        s->scratchpad.resize(desc.width * desc.height * 8);
        s->current_display_desc = desc;
}

static void gl_render(struct state_gl *s, char *data)
{
        /* for DXT, deinterlacing doesn't make sense since it is
         * always deinterlaced before comrpression */
        if(s->deinterlace && (s->current_display_desc.color_spec == RGBA || s->current_display_desc.color_spec == UYVY))
                vc_deinterlace((unsigned char *) data,
                                vc_get_linesize(s->current_display_desc.width, s->current_display_desc.color_spec),
                                s->current_display_desc.height);

        gl_check_error();

        switch(s->current_display_desc.color_spec) {
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
                case UYVY:
                case v210:
                case Y416:
                        gl_render_glsl(s, data);
                        break;
                case R10k:
                case RGB:
                case RGBA:
                        upload_texture(s, data);
                        break;
                case DXT5:                        
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        (s->current_display_desc.width + 3) / 4 * 4, s->dxt_height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                        (s->current_display_desc.width + 3) / 4 * 4 * s->dxt_height,
                                        data);
                        break;
#ifdef HWACC_VDPAU
                case HW_VDPAU:
                        s->vdp.loadFrame(reinterpret_cast<hw_vdpau_frame *>(data));
                        break;
#endif
                default:
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "error: received unsupported codec %s.\n",
                                        get_codec_name(s->current_display_desc.color_spec));
                        exit_uv(EXIT_FAILURE);
                        return;

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
                if (s->cursor_shown_from != chrono::steady_clock::time_point()) {
                        auto now = chrono::steady_clock::now();
                        if (chrono::duration_cast<chrono::seconds>(now - s->cursor_shown_from).count() > 2) {
                                glfwSetInputMode(s->window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                                s->cursor_shown_from = chrono::steady_clock::time_point();
                        }
                }
        }

        {
                unique_lock<mutex> lk(s->lock);
                double timeout = min(2.0 / s->current_display_desc.fps, 0.1);
                s->new_frame_ready_cv.wait_for(lk, chrono::duration<double>(timeout), [s] {
                                return s->frame_queue.size() > 0;});
                if (s->frame_queue.size() == 0) {
                        return;
                }
                frame = s->frame_queue.front();
                if (!frame) {
                        pop_frame(s, lk);
                        return;
                }
                if (s->paused) {
                        vf_recycle(frame);
                        s->free_frame_queue.push(frame);
                        pop_frame(s, lk);
                        return;
                }
                if (s->current_frame) {
                        vf_recycle(s->current_frame);
                        s->free_frame_queue.push(s->current_frame);
                }
                s->current_frame = frame;
        }

        if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
                gl_reconfigure_screen(s, video_desc_from_frame(frame));
        }

        gl_render(s, frame->tiles[0].data);
        gl_draw(s->aspect, (s->dxt_height - s->current_display_desc.height) / (float) s->dxt_height * 2, s->vsync != SINGLE_BUF);

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
                                glfw_print_video_mode(s);
                                break;
                        }
                case 'q':
                        exit_uv(0);
                        break;
                case 'd':
                        s->deinterlace = !s->deinterlace;
                        log_msg(LOG_LEVEL_NOTICE, "Deinterlacing: %s\n", s->deinterlace ? "ON" : "OFF");
                        break;
                case 'p':
                        s->paused = !s->paused;
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME << (s->paused ? "Paused (press 'p' to unpause)" : "Unpaused") << "\n";
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
                if (s->cursor_shown_from == chrono::steady_clock::time_point()) {
                        glfwSetInputMode(s->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                s->cursor_shown_from = chrono::steady_clock::now();
        }
}

static bool display_gl_check_gl_version() {
        auto version = (const char *) glGetString(GL_VERSION);
        if (!version) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get OpenGL version!\n");
                return false;
        }
        if (atof(version) < 2.0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "ERROR: OpenGL 2.0 is not supported, try updating your drivers...\n");
                return false;
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "OpenGL 2.0 is supported...\n");
        return true;
}

static void display_gl_print_depth() {
        array<int, 3> bits = {};
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

#if defined HAVE_LINUX || defined WIN32
static const char *glewGetError(GLenum err) {
        switch (err) {
                case GLEW_ERROR_NO_GL_VERSION: return "missing GL version";
                case GLEW_ERROR_GL_VERSION_10_ONLY: return "Need at least OpenGL 1.1";
                case GLEW_ERROR_GLX_VERSION_11_ONLY: return "Need at least GLX 1.2";
#ifdef GLEW_ERROR_NO_GLX_DISPLAY
                case GLEW_ERROR_NO_GLX_DISPLAY: return "Need GLX display for GLX support";
#endif
                default: return (const char *) glewGetErrorString(err);
        }
}
#endif // defined HAVE_LINUX || defined WIN32

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
        double cs_coeffs[2*4] = { 0, 0, KR_709, KB_709, KR_2020, KB_2020, KR_P3, KB_P3 };
        double kr = cs_coeffs[2 * index];
        double kb = cs_coeffs[2 * index + 1];
        const char *placeholders[] = { "Y_SCALED_PLACEHOLDER", "R_CR_PLACEHOLDER", "G_CB_PLACEHOLDER", "G_CR_PLACEHOLDER", "B_CB_PLACEHOLDER" };
        double values[] =            {  Y_LIMIT_INV,            R_CR(kr,kb),        G_CB(kr,kb),        G_CR(kr,kb),        B_CB(kr,kb)};

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

static void display_gl_set_user_window_hints() {
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

ADD_TO_PARAM(GL_DISABLE_10B_OPT_PARAM_NAME ,
         "* " GL_DISABLE_10B_OPT_PARAM_NAME "\n"
         "  Disable 10 bit codec processing to improve performance\n");
ADD_TO_PARAM(GL_WINDOW_HINT_OPT_PARAM_NAME ,
         "* " GL_WINDOW_HINT_OPT_PARAM_NAME "=<k>=<v>[:<k2>=<v2>...]\n"
         "  Set window hint <k> to value <v>\n");
/**
 * Initializes OpenGL stuff. If this function succeeds, display_gl_cleanup_opengl() needs
 * to be called to release resources.
 */
static bool display_gl_init_opengl(struct state_gl *s)
{
        glfwSetErrorCallback(glfw_print_error);

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
        if (s->nodecorate) {
                glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        }

        set_mac_color_space();
        glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, s->vsync == SINGLE_BUF ? GLFW_FALSE : GLFW_TRUE);
        int width = splash_width;
        int height = splash_height;
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
        display_gl_set_user_window_hints();
        if ((s->window = glfwCreateWindow(width, height, IF_NOT_NULL_ELSE(get_commandline_param("window-title"), DEFAULT_WIN_NAME), nullptr, nullptr)) == nullptr) {
                return false;
        }
        if (mon != nullptr) { /// @todo remove/revert when no needed (see particular commit message
                glfwSetWindowMonitor(s->window, mon, GLFW_DONT_CARE, GLFW_DONT_CARE, width, height, get_refresh_rate(s->modeset, mon, GLFW_DONT_CARE));
        }
        glfw_print_video_mode(s);
        glfwSetWindowUserPointer(s->window, s);
        if (s->hide_window)
                glfwHideWindow(s->window);
        glfwSetInputMode(s->window, GLFW_CURSOR, s->show_cursor == state_gl::SC_TRUE ?  GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
        glfwMakeContextCurrent(s->window);
        glfwSetKeyCallback(s->window, glfw_key_callback);
        glfwSetCursorPosCallback(s->window, glfw_mouse_callback);
        glfwSetWindowCloseCallback(s->window, glfw_close_callback);
        glfwSetFramebufferSizeCallback(s->window, gl_resize);
        glfwSetWindowRefreshCallback(s->window, display_gl_render_last);

#if defined HAVE_LINUX || defined WIN32
        if (GLenum err = glewInit()) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "GLEW Error: %s (err %d)\n", glewGetError(err), err);
                //return false; // do not fail - error 4 (on Wayland) can be suppressed
        }
#endif /* HAVE_LINUX */
        if (!display_gl_check_gl_version()) {
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

        s->PHandle_uyvy = gl_substitute_compile_link(vert, uyvy_to_rgb_fp);
        s->PHandle_yuva = gl_substitute_compile_link(vert, yuva_to_rgb_fp);
        s->PHandle_v210 = gl_substitute_compile_link(vert, v210_to_rgb_fp);
        // Create fbo
        glGenFramebuffersEXT(1, &s->fbo_id);
        s->PHandle_dxt = glsl_compile_link(vert, fp_display_dxt1);
        glUseProgram(s->PHandle_dxt);
        glUniform1i(glGetUniformLocation(s->PHandle_dxt,"yuvtex"),0);
        glUseProgram(0);
        s->PHandle_dxt5 = glsl_compile_link(vert, fp_display_dxt5ycocg);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // set row alignment to 1 byte instead of default
                                               // 4 bytes which won't work on row-unaligned RGB

        glGenBuffersARB(1, &s->pbo_id);
        glfwMakeContextCurrent(nullptr);

        return true;
}

/// Releases resources allocated with display_gl_init_opengl. After this function,
/// the resource references are invalid.
static void display_gl_cleanup_opengl(struct state_gl *s){
        glfwMakeContextCurrent(s->window);

        glDeleteProgram(s->PHandle_uyvy);
        glDeleteProgram(s->PHandle_v210);
        glDeleteProgram(s->PHandle_yuva);
        glDeleteProgram(s->PHandle_dxt);
        glDeleteProgram(s->PHandle_dxt5);
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

static void upload_texture(struct state_gl *s, char *data)
{
        GLuint format = s->current_display_desc.color_spec == RGB ? GL_RGB : GL_RGBA;
        GLenum type = GL_UNSIGNED_BYTE;
        if (s->current_display_desc.color_spec == R10k || s->current_display_desc.color_spec == v210) {
                type = GL_UNSIGNED_INT_2_10_10_10_REV;
        } else if (s->current_display_desc.color_spec == Y416) {
                type = GL_UNSIGNED_SHORT;
        }
        GLint width = s->current_display_desc.width;
        if (s->current_display_desc.color_spec == UYVY || s->current_display_desc.color_spec == v210) {
                width = vc_get_linesize(width, s->current_display_desc.color_spec) / 4;
        }
        auto byte_swap_r10k = [](uint32_t * __restrict out, const uint32_t *__restrict in, long data_len) {
                DEBUG_TIMER_START(byte_swap_r10k);
                OPTIMIZED_FOR (int i = 0; i < data_len / 4; i += 1) {
                        uint32_t x = *in++;
                        *out++ = /* output is x2b8g8r8 little-endian */
                                (x & 0xFFU) << 2U | (x & 0xC0'00U) >> 14U | // R
                                (x & 0x3F'00U) << 6U | (x & 0xF0'00'00) >> 10U | // G
                                (x & 0x0F'00'00U) << 10U | (x & 0xFC'00'00'00U) >> 6U; // B
                }
                DEBUG_TIMER_STOP(byte_swap_r10k);
        };
        int data_size = vc_get_linesize(s->current_display_desc.width, s->current_display_desc.color_spec) * s->current_display_desc.height;
        if (s->use_pbo) {
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id); // current pbo
                glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
                if (void *ptr = glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB)) {
                        // update data directly on the mapped buffer
                        if (s->current_display_desc.color_spec == R10k) { // perform byte swap
                                byte_swap_r10k(static_cast<uint32_t *>(ptr), reinterpret_cast<uint32_t *>(data), data_size);
                        } else {
                                memcpy(ptr, data, data_size);
                        }
                        glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
                }
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, s->current_display_desc.height, format, type, nullptr);

                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        } else {
                if (s->current_display_desc.color_spec == R10k) { // perform byte swap
                        byte_swap_r10k(reinterpret_cast<uint32_t *>(s->scratchpad.data()), reinterpret_cast<uint32_t *>(data), data_size);
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

static int display_gl_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof gl_supp_codecs <= *len) {
                                auto filter_codecs = [](codec_t c) {
                                        return get_bits_per_component(c) <= 8 || commandline_params.find(GL_DISABLE_10B_OPT_PARAM_NAME) == commandline_params.end(); // option to disable 10-bit processing
                                };
                                copy_if(gl_supp_codecs.begin(), gl_supp_codecs.end(), (codec_t *) val, filter_codecs);
                        } else {
                                return FALSE;
                        }

                        *len = sizeof gl_supp_codecs;
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
                } else {
                        vf_free(buffer);
                }
        }

        struct video_frame *buffer = vf_alloc_desc_data(s->current_desc);
        clear_video_buffer(reinterpret_cast<unsigned char *>(buffer->tiles[0].data),
                        vc_get_linesize(buffer->tiles[0].width, buffer->color_spec),
                        vc_get_linesize(buffer->tiles[0].width, buffer->color_spec),
                        buffer->tiles[0].height,
                        buffer->color_spec);
        return buffer;
}

static int display_gl_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        unique_lock<mutex> lk(s->lock);

        if(!frame) {
                glfwSetWindowShouldClose(s->window, GLFW_TRUE);
                s->frame_queue.push(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return 0;
        }

        if (nonblock == PUTF_DISCARD) {
                vf_recycle(frame);
                s->free_frame_queue.push(frame);
                return 0;
        }
        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK) {
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "1 frame(s) dropped!\n";
                vf_recycle(frame);
                s->free_frame_queue.push(frame);
                return 1;
        }
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.push(frame);

        lk.unlock();
        s->new_frame_ready_cv.notify_one();

        return 0;
}

static void display_gl_put_audio_frame(void *state, const struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_gl_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_gl_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].dev, "");
                strcpy((*available_cards)[0].name, "OpenGL SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_gl_init,
        display_gl_run,
        display_gl_done,
        display_gl_getf,
        display_gl_putf,
        display_gl_reconfigure,
        display_gl_get_property,
        display_gl_put_audio_frame,
        display_gl_reconfigure_audio,
        DISPLAY_NEEDS_MAINLOOP, // many GLFW functions must be called from main thread (notably glfwPollEvents())
        true,
};

REGISTER_MODULE(gl, &display_gl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


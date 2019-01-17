/**
 * @file   video_display/gl.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2016 CESNET, z. s. p. o.
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
#include <GLUT/glut.h>
#elif defined HAVE_LINUX
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glut.h>
#include "x11_common.h"
#else // WIN32
#include <GL/glew.h>
#include <GL/glut.h>
#endif /* HAVE_MACOSX */

#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif /* FREEGLUT */

#include "spout_sender.h"
#include "syphon_server.h"

#include <algorithm>
#include <condition_variable>
#include <csetjmp>
#include <iostream>
#include <mutex>
#include <queue>

#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rang.hpp"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"
#include "tv.h"

#define MAGIC_GL         0x1331018e
#define MODULE_NAME      "[GL] "
#define DEFAULT_WIN_NAME "Ultragrid - OpenGL Display"

#define STRINGIFY(A) #A

#define MAX_BUFFER_SIZE 1
#define SYSTEM_VSYNC 0xFE
#define SINGLE_BUF 0xFF // use single buffering instead of double

#ifdef HWACC_VDPAU
#include "hwaccel_vdpau.h"
typedef GLintptr vdpauSurfaceNV;
#define NV_CAST(x) ((void *)(uintptr_t)(x))
#endif

using namespace std;

static const char * yuv422_to_rgb_fp = STRINGIFY(
uniform sampler2D image;
uniform float imageWidth;
void main()
{
        vec4 yuv;
        yuv.rgba  = texture2D(image, gl_TexCoord[0].xy).grba;
        if(gl_TexCoord[0].x * imageWidth / 2.0 - floor(gl_TexCoord[0].x * imageWidth / 2.0) > 0.5)
                yuv.r = yuv.a;
        yuv.r = 1.1643 * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        float tmp; // this is a workaround over broken Gallium3D with Nouveau in U14.04 (and perhaps others)
        tmp = -0.2664 * yuv.b;
        tmp = 2.0 * tmp;
        gl_FragColor.r = yuv.r + 1.7926 * yuv.b;
        gl_FragColor.g = yuv.r - 0.2132 * yuv.g + tmp;
        gl_FragColor.b = yuv.r + 2.1124 * yuv.g;
        gl_FragColor.a = 1.0;
});

/* DXT YUV (FastDXT) related */
static const char *fp_display_dxt1 = STRINGIFY(
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
);

static const char * vert = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();}
);

static const char fp_display_dxt5ycocg[] = STRINGIFY(
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
);

#ifdef HWACC_VDPAU
struct state_vdpau {
        bool initialized = false;
        GLuint textures[4] = {0};
        hw_vdpau_frame lastFrame;

        bool interopInitialized = false;
        bool mixerInitialized = false;
        VdpDevice device = VDP_INVALID_HANDLE;
        VdpGetProcAddress *get_proc_address = nullptr;
        vdpauSurfaceNV vdpgl_surf = 0;

        VdpOutputSurface out_surf = VDP_INVALID_HANDLE;
        VdpVideoMixer mixer = VDP_INVALID_HANDLE;

        uint32_t surf_width = 0;
        uint32_t surf_height = 0;
        VdpChromaType surf_ct = 0;

        void (*VDPAUInitNV)(const void * /*vdpDevice*/,
                        const void * /*getProcAddress*/);

        void (*VDPAUFiniNV)(void);

        vdpauSurfaceNV (*VDPAURegisterVideoSurfaceNV)(const void * /*vdpSurface*/,
                        GLenum /*target*/,
                        GLsizei /*numTextureNames*/,
                        const uint * /*textureNames*/);

        vdpauSurfaceNV (*VDPAURegisterOutputSurfaceNV)(const void * /*vdpSurface*/,
                        GLenum /*target*/,
                        GLsizei /*numTextureNames*/,
                        const uint * /*textureNames*/);

        GLboolean (*VDPAUIsSurfaceNV)(vdpauSurfaceNV /*surface*/);
        void (*VDPAUUnregisterSurfaceNV)(vdpauSurfaceNV /*surface*/);

        void (*VDPAUGetSurfaceivNV)(vdpauSurfaceNV /*surface*/,
                        GLenum /*pname*/,
                        GLsizei /*bufSize*/,
                        GLsizei * /*length*/,
                        int * /*values*/);

        void (*VDPAUSurfaceAccessNV)(vdpauSurfaceNV /*surface*/,
                        GLenum /*access*/);

        void (*VDPAUMapSurfacesNV)(GLsizei /*numSurfaces*/,
                        const vdpauSurfaceNV * /*surfaces*/);

        void (*VDPAUUnmapSurfacesNV)(GLsizei /*numSurfaces*/,
                        const vdpauSurfaceNV * /*surfaces*/);

        bool loadVdpGlFuncs();
        bool init();
        void checkInterop(VdpDevice device, VdpGetProcAddress *get_proc_address);
        void initInterop(VdpDevice device, VdpGetProcAddress *get_proc_address);
        void uninitInterop();

        void initMixer(uint32_t w, uint32_t h, VdpChromaType ct);
        void mixerRender(VdpVideoSurface f);
        void uninitMixer();

        void uninit();

        vdp_funcs funcs;

};
#endif

struct state_gl {
        GLuint          PHandle_uyvy, PHandle_dxt, PHandle_dxt5;

        // Framebuffer
        GLuint fbo_id;
        GLuint texture_display;
        GLuint texture_uyvy;
        GLuint pbo_id;

        /* For debugging... */
        uint32_t	magic;

        int             window;

	bool            fs;
        bool            deinterlace;

        struct video_frame *current_frame;

        queue<struct video_frame *> frame_queue;
        queue<struct video_frame *> free_frame_queue;
        struct video_desc current_desc;
        struct video_desc current_display_desc;
        mutex           lock;
        condition_variable new_frame_ready_cv;
        condition_variable frame_consumed_cv;

        double          aspect;
        double          video_aspect;
        unsigned long int frames;
        
        int             dxt_height;

        struct timeval  tv;

        int             vsync;
        bool            paused;
        enum show_cursor_t { SC_TRUE, SC_FALSE, SC_AUTOHIDE } show_cursor;
        chrono::steady_clock::time_point                      cursor_shown_from; ///< indicates time point from which is cursor show if show_cursor == SC_AUTOHIDE, timepoint() means cursor is not currently shown
        string          syphon_spout_srv_name;

        bool should_exit_main_loop; // used only for GLUT (not freeglut)

        double          window_size_factor;

        struct module   mod;

        void *syphon_spout;
        bool hide_window;

        bool fixed_size, first_run;
        int fixed_w, fixed_h;

        bool nodecorate = false;

#ifdef HWACC_VDPAU
        struct state_vdpau vdp;
#endif

        state_gl(struct module *parent) : PHandle_uyvy(0), PHandle_dxt(0), PHandle_dxt5(0),
                fbo_id(0), texture_display(0), texture_uyvy(0), pbo_id(0),
                magic(MAGIC_GL), window(-1), fs(false), deinterlace(false), current_frame(nullptr),
                aspect(0.0), video_aspect(0.0), frames(0ul), dxt_height(0),
                vsync(1), paused(false), show_cursor(SC_AUTOHIDE),
                should_exit_main_loop(false), window_size_factor(1.0),
                syphon_spout(nullptr), hide_window(false), fixed_size(false), first_run(true),
                fixed_w(0), fixed_h(0)
        {
                gettimeofday(&tv, NULL);
                memset(&current_desc, 0, sizeof(current_desc));
                memset(&current_display_desc, 0, sizeof(current_display_desc));

                module_init_default(&mod);
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);
        }

        ~state_gl() {
                module_done(&mod);
        }
};

static struct state_gl *gl;

/* Prototyping */
static int display_gl_putf(void *state, struct video_frame *frame, int nonblock);
static int display_gl_reconfigure(void *state, struct video_desc desc);

static void gl_draw(double ratio, double bottom_offset, bool double_buf);
static void gl_show_help(void);

static void gl_change_aspect(struct state_gl *s, int width, int height);
static void gl_resize(int width, int height);
static void gl_render_uyvy(struct state_gl *s, char *data);
static void gl_reconfigure_screen(struct state_gl *s, struct video_desc desc);
static void glut_idle_callback(void);
static void glut_key_callback(unsigned char key, int x, int y);
static void glut_mouse_callback(int x, int y);
static void glut_close_callback(void);
static void glut_resize_window(bool fs, int height, double aspect, double window_size_factor);
static void display_gl_set_sync_on_vblank(int value);
static void screenshot(struct video_frame *frame);
static void upload_texture(struct state_gl *s, char *data);

static void gl_render_vdpau(struct state_gl *s, char *data);

#ifdef HAVE_MACOSX
extern "C" void NSApplicationLoad(void);
#endif

/**
 * Show help
 */
static void gl_show_help(void) {
        cout << "GL options:\n";
        cout << rang::style::bold << rang::fg::red << "\t-d gl" << rang::fg::reset << "[:d|:fs|:aspect=<v>/<h>|:cursor|:size=X%%|:syphon[=<name>]|:spout[=<name>]|:nodecorate|:fixed_size[=WxH]|:vsync[<x>|single]]* | help\n\n" << rang::style::reset;
        printf("\t\td\t\tdeinterlace\n");
        printf("\t\tfs\t\tfullscreen\n");
        printf("\t\tnodecorate\tdisable window decorations (works with X11 only)\n");
        printf("\t\tnovsync\t\tdo not turn sync on VBlank\n");
        printf("\t\tvsync=<x>\tsets vsync to: 0 - disable; 1 - enable; -1 - adaptive vsync; D - leaves system default\n");
        printf("\t\tsingle\t\tuse single buffer (instead of double-buffering\n");
        printf("\t\taspect=<w>/<h>\trequested video aspect (eg. 16/9). Leave unset if PAR = 1.\n");
        printf("\t\tcursor\t\tshow visible cursor\n");
        printf("\t\tsize\t\tspecifies desired size of window compared "
                        "to native resolution (in percents)\n");
        printf("\t\tsyphon\t\tuse Syphon (optionally with name)\n");
        printf("\t\tspout\t\tuse Spout (optionally with name)\n");
        printf("\t\thide-window\tdo not show OpenGL window (useful with Syphon)\n");

        printf("\n\nKeyboard shortcuts:\n");
        printf("\t\t'f'\t\ttoggle fullscreen\n");
        printf("\t\t'q'\t\tquit\n");
        printf("\t\t'd'\t\ttoggle deinterlace\n");
        printf("\t\t' '\t\tpause video\n");
        printf("\t\t's'\t\tscreenshot\n");
        printf("\t\t'm'\t\tshow/hide cursor\n");
        printf("\t\t'+'\t\tmake window smaller by factor 50%%\n");
        printf("\t\t'-'\t\tmake window twice as bigger\n");
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

static void * display_gl_init(struct module *parent, const char *fmt, unsigned int flags) {
        UNUSED(flags);
        if (gl) {
                LOG(LOG_LEVEL_ERROR) << "Multiple instances of GL display is disallowed!\n";
                return nullptr;
        }

	struct state_gl *s = new state_gl(parent);
        
	// parse parameters
	if (fmt != NULL) {
		if (strcmp(fmt, "help") == 0) {
			gl_show_help();
			delete s;
			return &display_init_noerr;
		}

		char *tmp, *ptr;
                tmp = ptr = strdup(fmt);
		char *tok, *save_ptr = NULL;
		
		while((tok = strtok_r(ptr, ":", &save_ptr)) != NULL) {
                        if(!strcmp(tok, "d")) {
                                s->deinterlace = true;
                        } else if(!strcmp(tok, "fs")) {
                                s->fs = true;
                        } else if(!strncmp(tok, "aspect=", strlen("aspect="))) {
                                s->video_aspect = atof(tok + strlen("aspect="));
                                char *pos = strchr(tok,'/');
                                if(pos) s->video_aspect /= atof(pos + 1);
                        } else if(!strcasecmp(tok, "nodecorate")) {
#ifdef HAVE_LINUX
                                s->nodecorate = true;
#else
				log_msg(LOG_LEVEL_WARNING, "[GL] Nodecorate not supported for current platform!\n");
#endif
                        } else if(!strcasecmp(tok, "novsync")) {
                                s->vsync = 0;
                        } else if(!strcasecmp(tok, "single")) {
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
                                log_msg(LOG_LEVEL_ERROR, "[GL] Syphon/Spout support not compiled in.\n");
                                free(tmp);
                                delete s;
                                return NULL;
#endif
                        } else if (!strcasecmp(tok, "hide-window")) {
                                s->hide_window = true;
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
                                log_msg(LOG_LEVEL_ERROR, "[GL] Unknown option: %s\n", tok);
                                free(tmp);
                                delete s;
                                return NULL;
                        }
                        ptr = NULL;
                }

		free(tmp);
	}

        fprintf(stdout,"GL setup: fullscreen: %s, deinterlace: %s\n",
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

        /* GLUT callbacks take only some arguments so we need static variable */
        gl = s;

        return (void*)s;
}

/**
 * This function just sets new video description.
 */
static int display_gl_reconfigure(void *state, struct video_desc desc)
{
        struct state_gl	*s = (struct state_gl *) state;

        assert (desc.color_spec == RGBA ||
                        desc.color_spec == RGB  ||
                        desc.color_spec == UYVY ||
                        desc.color_spec == DXT1 ||
                        desc.color_spec == DXT1_YUV ||
                        desc.color_spec == DXT5
#ifdef HWACC_VDPAU
                        || desc.color_spec == HW_VDPAU
#endif
                        );

        s->current_desc = desc;

        return TRUE;
}

static void glut_resize_window(bool fs, int height, double aspect, double window_size_factor)
{
        if (fs) {
                glutReshapeWindow(glutGet(GLUT_SCREEN_WIDTH),
                               glutGet(GLUT_SCREEN_HEIGHT));
                glutFullScreen();
        } else {
                glutReshapeWindow(window_size_factor *
                                height * aspect,
                                window_size_factor *
                                height);
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
        bool have_ext_swap_control_tear = false;
#ifdef HAVE_LINUX
        if (strstr(glXQueryExtensionsString(glXGetCurrentDisplay(), 0),
                                "GLX_EXT_swap_control_tear")) {
                have_ext_swap_control_tear = true;
        }
#elif defined WIN32
        const char * (*wglGetExtensionsStringARBProc)(HDC hdc) = (const char *(*)(HDC))
                wglGetProcAddress("wglGetExtensionsStringARB");
        if (strstr(wglGetExtensionsStringARBProc(wglGetCurrentDC()),
                                "WGL_EXT_swap_control_tear")) {
                have_ext_swap_control_tear = true;
        }
#endif
        if (value == -1) {
                if (!have_ext_swap_control_tear) {
                        log_msg(LOG_LEVEL_WARNING, "WGL/GLX_EXT_swap_control_tear not detected, using normal vsync.\n");
                        value = 1;
                } else {
                        log_msg(LOG_LEVEL_VERBOSE, "WGL/GLX_EXT_swap_control_tear detected, using adaptive vsync\n");
                }
        }

#ifdef HAVE_MACOSX
        int swap_interval = value;
        CGLContextObj cgl_context = CGLGetCurrentContext();
        CGLSetParameter(cgl_context, kCGLCPSwapInterval, &swap_interval);
#elif HAVE_LINUX
        /* using GLX_SGI_swap_control
         *
         * Also it is worth considering to use GLX_EXT_swap_control (instead?).
         * But we would need both Display and GLXDrawable variables which we do not currently have
         */
        int (*glXSwapIntervalSGIProc)(int interval) = 0;

        glXSwapIntervalSGIProc = (int (*)(int))
                glXGetProcAddressARB( (const GLubyte *) "glXSwapIntervalSGI");

        if(glXSwapIntervalSGIProc) {
                glXSwapIntervalSGIProc(value);
        } else {
                log_msg(LOG_LEVEL_WARNING, "[GL display] GLX_SGI_swap_control is presumably not supported. Unable to set sync-on-VBlank.\n");
        }
#elif WIN32
        BOOL (*wglSwapIntervalEXTProc)(int interval) = 0;

        wglSwapIntervalEXTProc = (BOOL (*)(int))
                wglGetProcAddress("wglSwapIntervalEXT");

        if (wglSwapIntervalEXTProc) {
                BOOL ret = wglSwapIntervalEXTProc(value);
                if (!ret) {
                        log_msg(LOG_LEVEL_WARNING, "[GL display] Unable to set sync-on-VBlank.\n");
                }
        } else {
                log_msg(LOG_LEVEL_WARNING, "[GL display] WGL_EXT_swap_control is presumably not supported. Unable to set sync-on-VBlank.\n");
        }
#endif
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

        strftime(name, sizeof(name), "screenshot-%a, %d %b %Y %T %z.pnm",
                                               time_ptr);
        if (save_video_frame_as_pnm(frame, name)) {
                log_msg(LOG_LEVEL_NOTICE, "[GL] Generated screenshot \"%s\".\n", name);
        } else {
                log_msg(LOG_LEVEL_ERROR, "[GL] Unable to generate screenshot!\n");
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

        fprintf(stdout,"Setting GL window size %dx%d (%dx%d).\n", (int)(s->aspect * desc.height),
                        desc.height, desc.width, desc.height);
        if (!s->hide_window)
                glutShowWindow();

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
                glBindTexture(GL_TEXTURE_2D,s->texture_uyvy);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width / 2, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glUseProgram(s->PHandle_uyvy);
                glUniform1i(glGetUniformLocation(s->PHandle_uyvy, "image"), 2);
                glUniform1f(glGetUniformLocation(s->PHandle_uyvy, "imageWidth"),
                                (GLfloat) desc.width);
                glUseProgram(0);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                desc.width, desc.height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
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

        gl_check_error();

        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, vc_get_linesize(desc.width, desc.color_spec) * desc.height, 0, GL_STREAM_DRAW_ARB);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        gl_check_error();

        if (!s->fixed_size || s->first_run) {
                glut_resize_window(s->fs, desc.height, s->aspect, s->window_size_factor);
                s->first_run = false;
        } else {
                // s->aspect might have changed so we may want to run this to reflect it
                gl_change_aspect(s, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
        }

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

	if (s->nodecorate) {
#ifdef HAVE_LINUX
                // https://stackoverflow.com/questions/12343390/how-to-get-x-window-id-of-a-window-created-by-glut
		GLXDrawable d = glXGetCurrentDrawable();
		x11_unset_window_decorations(glXGetCurrentDisplay(), d);
#endif
	}

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
                        gl_render_uyvy(s, data);
                        break;
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
                        gl_render_vdpau(s, data);
                        break;
#endif
                default:
                        fprintf(stderr, "[GL] Fatal error - received unsupported codec.\n");
                        exit_uv(EXIT_FAILURE);
                        return;

        }

        gl_check_error();
}

static void pop_frame(struct state_gl *s)
{
        unique_lock<mutex> lk(s->lock);
        s->frame_queue.pop();
        lk.unlock();
        s->frame_consumed_cv.notify_one();
}

static void glut_idle_callback(void)
{
        struct state_gl *s = gl;
        struct timeval tv;
        double seconds;
        struct video_frame *frame;

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                auto msg_univ = reinterpret_cast<struct msg_universal *>(msg);
                struct response *r;
                if (strncasecmp(msg_univ->text, "win-title ", strlen("win_title ")) == 0) {
                        glutSetWindowTitle(msg_univ->text + strlen("win_title "));
                        r = new_response(RESPONSE_OK, NULL);
                } else {
                        fprintf(stderr, "[GL] Unknown command received: %s\n", msg_univ->text);
                        r = new_response(RESPONSE_BAD_REQUEST, NULL);
                }
                free_message(msg, r);
        }


        if (gl->show_cursor == state_gl::SC_AUTOHIDE) {
                if (gl->cursor_shown_from != chrono::steady_clock::time_point()) {
                        auto now = chrono::steady_clock::now();
                        if (chrono::duration_cast<chrono::seconds>(now - gl->cursor_shown_from).count() > 2) {
                                glutSetCursor(GLUT_CURSOR_NONE);
                                gl->cursor_shown_from = chrono::steady_clock::time_point();
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
        }

        if (!frame) {
#ifdef FREEGLUT
                glutLeaveMainLoop();
#endif
                pop_frame(s);
                return;
        }

        if (s->paused) {
                pop_frame(s);
                unique_lock<mutex> lk(s->lock);
                vf_recycle(frame);
                s->free_frame_queue.push(frame);
                return;
        }

        if (s->current_frame) {
                s->lock.lock();
                vf_recycle(s->current_frame);
                s->free_frame_queue.push(s->current_frame);
                s->lock.unlock();
        }
        s->current_frame = frame;

        if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
                gl_reconfigure_screen(s, video_desc_from_frame(frame));
        }

        gl_render(s, frame->tiles[0].data);
        gl_draw(s->aspect, (gl->dxt_height - gl->current_display_desc.height) / (float) gl->dxt_height * 2, gl->vsync != SINGLE_BUF);

        // publish to Syphon/Spout
        if (s->syphon_spout) {
#ifdef HAVE_SYPHON
                syphon_server_publish(s->syphon_spout, frame->tiles[0].width, frame->tiles[0].height, s->texture_display);
#elif defined HAVE_SPOUT
                spout_sender_sendframe(s->syphon_spout, frame->tiles[0].width, frame->tiles[0].height, s->texture_display);
                glBindTexture(GL_TEXTURE_2D, s->texture_display);
#endif // HAVE_SPOUT
        }

        glutSwapBuffers();
        pop_frame(s);

        /* FPS Data, this is pretty ghetto though.... */
        s->frames++;
        gettimeofday(&tv, NULL);
        seconds = tv_diff(tv, s->tv);

        if (seconds > 5) {
                double fps = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[GL] %lu frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->frames = 0;
                s->tv = tv;
        }
}

static void glut_key_callback(unsigned char key, int /* x */, int /* y */)
{
        switch(key) {
                case 'f':
                        gl->fs = !gl->fs;
                        glut_resize_window(gl->fs, gl->current_display_desc.height, gl->aspect,
                                        gl->window_size_factor);
                        break;
                case 'q':
#if defined FREEGLUT || defined HAVE_MACOSX
                        exit_uv(0);
#else
                        /// @todo
                        /// This shouldn't happen (?). We have either freeglut (Linux, MSW) or
                        /// original GLUT on OS X
			glutDestroyWindow(gl->window);
			exit(1);
#endif
                        break;
                case 'd':
                        gl->deinterlace = !gl->deinterlace;
                        printf("Deinterlacing: %s\n", gl->deinterlace ? "ON" : "OFF");
                        break;
                case ' ':
                        gl->paused = !gl->paused;
                        LOG(LOG_LEVEL_NOTICE) << "[GL] " << (gl->paused ? "Paused (press <space> to unpause)" : "Unpaused") << "\n";
                        break;
                case 's':
                        screenshot(gl->current_frame);
                        break;
                case 'm':
                        gl->show_cursor = (state_gl::show_cursor_t) (((int) gl->show_cursor + 1) % 3);
                        glutSetCursor(gl->show_cursor == state_gl::SC_TRUE ? GLUT_CURSOR_INHERIT : GLUT_CURSOR_NONE);
                        break;
                case '+':
                        gl->window_size_factor *= 2;
                        glut_resize_window(gl->fs, gl->current_display_desc.height, gl->aspect,
                                        gl->window_size_factor);
                        break;
                case '-':
                        gl->window_size_factor /= 2;
                        glut_resize_window(gl->fs, gl->current_display_desc.height, gl->aspect,
                                        gl->window_size_factor);
                        break;
        }
}

static void glut_mouse_callback(int /* x */, int /* y */)
{
        if (gl->show_cursor == state_gl::SC_AUTOHIDE) {
                if (gl->cursor_shown_from == chrono::steady_clock::time_point()) {
                        glutSetCursor(GLUT_CURSOR_INHERIT);
                }
                gl->cursor_shown_from = chrono::steady_clock::now();
        }
}

#ifdef FREEGLUT
static void glut_init_error_callback(const char *fmt, va_list ap)
{
        va_list aq;
        va_copy(aq, ap);
        // get number of required bytes
        int size = vsnprintf(NULL, 0, fmt, ap);
        va_end(aq);

        // format the string
        auto buffer = (char *) alloca(size + 1);
        va_copy(aq, ap);
        if (vsprintf(buffer, fmt, ap) == size) {
                LOG(LOG_LEVEL_ERROR) << "[GL] " << buffer << "\n";
        }
        va_end(aq);

        // This is required - if there is no noexit function call, glutInit()
        // continues normally until it crashes. This solution (passing
        // exception through freeglut) works with GCC (see https://stackoverflow.com/questions/490773/how-is-the-c-exception-handling-runtime-implemented). Other possibilty is
        // to use longjmp() (but saved environment needs to be passed here then).
        throw EXIT_FAIL_DISPLAY;
}
#endif

static bool display_gl_check_gl_version() {
        auto version = (const char *) glGetString(GL_VERSION);
        if (!version) {
                log_msg(LOG_LEVEL_ERROR, MODULE_NAME "Unable to get OpenGL version!\n");
        }
        if (atof(version) < 2.0) {
                log_msg(LOG_LEVEL_ERROR, MODULE_NAME "ERROR: OpenGL 2.0 is not supported, try updating your drivers...\n");
                return false;
        }
        log_msg(LOG_LEVEL_INFO, MODULE_NAME "OpenGL 2.0 is supported...\n");
        return true;
}

static bool display_gl_init_opengl(struct state_gl *s)
{
#if defined HAVE_LINUX || defined WIN32
        GLenum err;
#endif // HAVE_LINUX

#ifdef FREEGLUT
        glutInitErrorFunc(glut_init_error_callback);
#endif
        try {
                glutInit(&uv_argc, uv_argv);
        } catch (...) {
                exit_uv(EXIT_FAIL_DISPLAY);
                return false;
        }
        glutInitDisplayMode(GLUT_RGBA | (s->vsync == SINGLE_BUF ? GLUT_SINGLE : GLUT_DOUBLE));

#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
        /* Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif

#ifdef FREEGLUT
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif
        glutIdleFunc(glut_idle_callback);
	s->window = glutCreateWindow(get_commandline_param("window-title") ? get_commandline_param("window-title") : DEFAULT_WIN_NAME);
        if (s->hide_window)
                glutHideWindow();
        glutSetCursor(s->show_cursor == state_gl::SC_TRUE ?  GLUT_CURSOR_INHERIT : GLUT_CURSOR_NONE);
        //glutHideWindow();
	glutKeyboardFunc(glut_key_callback);
	glutDisplayFunc((void (*)())glutSwapBuffers); // cast is needed because glutSwapBuffers is stdcall on MSW
        glutMotionFunc(glut_mouse_callback);
        glutPassiveMotionFunc(glut_mouse_callback);
#ifdef HAVE_MACOSX
        glutWMCloseFunc(glut_close_callback);
#elif FREEGLUT
        glutCloseFunc(glut_close_callback);
#endif
	glutReshapeFunc(gl_resize);

        if (!display_gl_check_gl_version()) {
                return false;
        }

#if defined HAVE_LINUX || defined WIN32
        err = glewInit();
        if (GLEW_OK != err)
        {
                /* Problem: glewInit failed, something is seriously wrong. */
                fprintf(stderr, "GLEW Error: %d\n", err);
                return false;
        }
#endif /* HAVE_LINUX */

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
        glEnable( GL_TEXTURE_2D );

        glGenTextures(1, &s->texture_display);
        glBindTexture(GL_TEXTURE_2D, s->texture_display);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &s->texture_uyvy);
        glBindTexture(GL_TEXTURE_2D, s->texture_uyvy);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        s->PHandle_uyvy = glsl_compile_link(vert, yuv422_to_rgb_fp);
        // Create fbo
        glGenFramebuffersEXT(1, &s->fbo_id);
        s->PHandle_dxt = glsl_compile_link(vert, fp_display_dxt1);
        glUseProgram(s->PHandle_dxt);
        glUniform1i(glGetUniformLocation(s->PHandle_dxt,"yuvtex"),0);
        glUseProgram(0);
        s->PHandle_dxt5 = glsl_compile_link(vert, fp_display_dxt5ycocg);
        /*if (pthread_create(&(s->thread_id), NULL, display_thread_gl, (void *) s) != 0) {
          perror("Unable to create display thread\n");
          return NULL;
          }*/

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // set row alignment to 1 byte instead of default
                                               // 4 bytes which won't work on row-unaligned RGB

        glGenBuffersARB(1, &s->pbo_id);

        return true;
}

static void display_gl_run(void *arg)
{
        struct state_gl *s = 
                (struct state_gl *) arg;

        if (!display_gl_init_opengl(s)) {
                exit_uv(1);
                return;
        }

#if defined FREEGLUT
	glutMainLoop();
#else
        while (!s->should_exit_main_loop) {
                glut_idle_callback();
                glutCheckLoop();
        }
#endif
}

static void gl_change_aspect(struct state_gl *s, int width, int height)
{
        double screen_ratio;
        double x = 1.0,
               y = 1.0;

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity( );

        screen_ratio = (double) width / height;
        if(screen_ratio > s->aspect) {
                x = (double) height * s->aspect / width;
        } else {
                y = (double) width / (height * s->aspect);
        }
        glScalef(x, y, 1);

        glOrtho(-1,1,-1/s->aspect,1/s->aspect,10,-10);
}

static void gl_resize(int width, int height)
{
        debug_msg("Resized to: %dx%d\n", width, height);

        glViewport( 0, 0, ( GLint )width, ( GLint )height );

        gl_change_aspect(gl, width, height);

        glMatrixMode( GL_MODELVIEW );

        glLoadIdentity( );

        if (gl->vsync == SINGLE_BUF) {
                glDrawBuffer(GL_FRONT);
                /* Clear the screen */
                glClear(GL_COLOR_BUFFER_BIT);
        }

        if (gl->current_frame) {
                // redraw last frame
                for (int i = 0; i < 2; ++i) {
                        gl_render(gl, gl->current_frame->tiles[0].data);
                        gl_draw(gl->aspect, (gl->dxt_height - gl->current_display_desc.height) / (float) gl->dxt_height * 2, gl->vsync != SINGLE_BUF);
                        glutSwapBuffers();
                }
                glutPostRedisplay();
        }
}

static void upload_texture(struct state_gl *s, char *data)
{
#if ! defined GL_NO_PBO
        GLuint format = s->current_display_desc.color_spec == RGB ? GL_RGB : GL_RGBA;
        GLint width = s->current_display_desc.width;
        if (s->current_display_desc.color_spec == UYVY) {
                width /= 2;
        }
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s->pbo_id); // current pbo
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, s->current_display_desc.height, format, GL_UNSIGNED_BYTE, 0);
        int data_size = vc_get_linesize(s->current_display_desc.width, s->current_display_desc.color_spec) * s->current_display_desc.height;
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
        void *ptr = (GLubyte*) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
        if (ptr)
        {
                // update data directly on the mapped buffer
                memcpy(ptr, data, data_size);
                glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
        }

        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, s->current_display_desc.height,  format, GL_UNSIGNED_BYTE, data);
#endif
}

static void gl_render_uyvy(struct state_gl *s, char *data)
{
        int status;
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, s->fbo_id);
        gl_check_error();
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture_display, 0);
        status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
        assert(status == GL_FRAMEBUFFER_COMPLETE_EXT);
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, s->texture_uyvy);

        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
        glLoadIdentity( );

        glMatrixMode( GL_MODELVIEW );
        glPushMatrix();
        glLoadIdentity( );

        glPushAttrib(GL_VIEWPORT_BIT);

        glViewport( 0, 0, s->current_display_desc.width, s->current_display_desc.height);

        upload_texture(s, data);

        glUseProgram(s->PHandle_uyvy);

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

#ifdef HWACC_VDPAU
/**
 * @brief Uninitializes VdpMixer
 */
void state_vdpau::uninitMixer(){
        VdpStatus st;

        if (mixer != VDP_INVALID_HANDLE){
                st = funcs.videoMixerDestroy(mixer);
                mixer = VDP_INVALID_HANDLE;
                if(st != VDP_STATUS_OK){
                        log_msg(LOG_LEVEL_ERROR, "Failed to destroy VdpVideoMixer: %s\n", funcs.getErrorString(st));
                }
        } 

        if (out_surf != VDP_INVALID_HANDLE){
                st = funcs.outputSurfaceDestroy(out_surf);
                if(st != VDP_STATUS_OK){
                        log_msg(LOG_LEVEL_ERROR, "Failed to destroy VdpOutputSurface: %s\n", funcs.getErrorString(st));
                }
                out_surf = VDP_INVALID_HANDLE;
                surf_width = 0;
                surf_height = 0;
                surf_ct = 0;
        }

        VDPAUUnregisterSurfaceNV(vdpgl_surf);
        mixerInitialized = false;
}

/**
 * @brief Initializes VdpMixer
 */
void state_vdpau::initMixer(uint32_t w, uint32_t h, VdpChromaType ct){
        uninitMixer();

        surf_ct = ct;
        surf_width = w;
        surf_height = h;

        VdpStatus st;

        VdpRGBAFormat rgbaFormat = VDP_RGBA_FORMAT_B8G8R8A8;

        st = funcs.outputSurfaceCreate(device,
                        rgbaFormat,
                        surf_width,
                        surf_height,
                        &out_surf);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to create VdpOutputSurface: %s\n", funcs.getErrorString(st));
        }

        VdpVideoMixerParameter params[] = {
                VDP_VIDEO_MIXER_PARAMETER_CHROMA_TYPE,
                VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_WIDTH,
                VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_HEIGHT
        };

        void *param_vals[] = {
                &surf_ct,
                &surf_width,
                &surf_height
        };

        st = funcs.videoMixerCreate(device,
                        0,
                        NULL,
                        3,
                        params,
                        param_vals,
                        &mixer);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to create VdpVideoMixer: %s\n", funcs.getErrorString(st));
        }

        vdpgl_surf = VDPAURegisterOutputSurfaceNV(NV_CAST(out_surf),
                        GL_TEXTURE_2D,
                        1,
                        textures);

        VDPAUSurfaceAccessNV(vdpgl_surf, GL_WRITE_DISCARD_NV);

        mixerInitialized = true;
}

/**
 * @brief Renders VdpVideoSurface into a VdpOutputSurface
 */
void state_vdpau::mixerRender(VdpVideoSurface f){
        VdpStatus st = funcs.videoMixerRender(mixer,
                        VDP_INVALID_HANDLE,
                        NULL,
                        VDP_VIDEO_MIXER_PICTURE_STRUCTURE_FRAME,
                        0,
                        NULL,
                        f,
                        0,
                        NULL,
                        NULL,
                        out_surf,
                        NULL,
                        NULL,
                        0,
                        NULL);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to render: %s\n", funcs.getErrorString(st));
        }
}

/**
 * @brief Checks VdpMixer parameters and reinitializes it if they don't match the video parameters
 */
static void check_mixer(struct state_gl *s, hw_vdpau_frame *frame){
        uint32_t frame_w;
        uint32_t frame_h;
        VdpChromaType ct;

        VdpStatus st = s->vdp.funcs.videoSurfaceGetParameters(frame->surface,
                        &ct,
                        &frame_w,
                        &frame_h);

        if(st != VDP_STATUS_OK){
                log_msg(LOG_LEVEL_ERROR, "Failed to get surface parameters: %s\n", s->vdp.funcs.getErrorString(st));
        }

        if(s->vdp.surf_width != frame_w ||
                        s->vdp.surf_height != frame_h ||
                        s->vdp.surf_ct != ct ||
                        !s->vdp.mixerInitialized)
        {
                s->vdp.initMixer(frame_w, frame_h, ct);
        }
}

/**
 * @brief Renders a vdpau frame
 */
static void gl_render_vdpau(struct state_gl *s, char *data)
{
        assert(s->vdp.initialized);
        hw_vdpau_frame * frame = (hw_vdpau_frame *) data;

        glBindTexture(GL_TEXTURE_2D, 0);

        int state = 0;
        int len = 0;
        if(s->vdp.vdpgl_surf){
                s->vdp.VDPAUGetSurfaceivNV(s->vdp.vdpgl_surf,
                                GL_SURFACE_STATE_NV,
                                1,
                                &len,
                                &state
                                );
        }

        if(state == GL_SURFACE_MAPPED_NV)
                s->vdp.VDPAUUnmapSurfacesNV(1, &s->vdp.vdpgl_surf);

        s->vdp.checkInterop(frame->hwctx.device, frame->hwctx.get_proc_address);
        check_mixer(s, frame);

        s->vdp.mixerRender(frame->surface);

        s->vdp.VDPAUMapSurfacesNV(1, &s->vdp.vdpgl_surf);

        glBindTexture(GL_TEXTURE_2D, s->vdp.textures[0]);

        hw_vdpau_frame_unref(&s->vdp.lastFrame);
        s->vdp.lastFrame = hw_vdpau_frame_copy(frame);
}

/**
 * @brief Checks if the vdpau-GL interoperability is initialized with the same hw contexts
 * and reinitializes it if needed
 */
void state_vdpau::checkInterop(VdpDevice device, VdpGetProcAddress *get_proc_address){
        if(this->device != device || this->get_proc_address != get_proc_address){
                uninitInterop();
                initInterop(device, get_proc_address);
        }
}

/**
 * @brief Initializes vdpau-GL interoperability
 */
void state_vdpau::initInterop(VdpDevice device, VdpGetProcAddress *get_proc_address){
        if(interopInitialized)
                uninitInterop();

        VDPAUInitNV(NV_CAST(device), (void *) get_proc_address);
        this->device = device;
        this->get_proc_address = get_proc_address;

        vdp_funcs_load(&funcs, device, get_proc_address);
        interopInitialized = true;
}

/**
 * @brief Uninitializes vdpau-GL interoperability
 */
void state_vdpau::uninitInterop(){
        if(!interopInitialized)
                return;

        VDPAUFiniNV();
        device = 0;
        get_proc_address = nullptr;
        interopInitialized = false;
        mixerInitialized = false;

        //VDPAUFiniNV() unmaps and unregisters all surfaces automatically
        vdpgl_surf = 0;
}

/**
 * @brief Initializes state_vdpau
 */
bool state_vdpau::init(){
        loadVdpGlFuncs();
        initialized = true;
        glGenTextures(4, textures);
        hw_vdpau_frame_init(&lastFrame);


        return true;
}

/**
 * @brief Uninitializes state_vdpau
 */
void state_vdpau::uninit(){
        glDeleteTextures(4, textures);
        for(int i = 0; i < 4; i++){
                textures[i] = 0;
        }
}

/**
 * @brief Loads GL functions for the GL_NV_vdpau_interop extension
 */
bool state_vdpau::loadVdpGlFuncs(){
        if (!strstr((const char *) glGetString(GL_EXTENSIONS),
                                "GL_NV_vdpau_interop"))
        {
                log_msg(LOG_LEVEL_ERROR, "VDPAU interop NOT supported!\n");
                log_msg(LOG_LEVEL_DEBUG, "Available extensions:%s\n", glGetString(GL_EXTENSIONS));
                return false;
        }

        VDPAUInitNV = (void (*)(const void *, const void *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUInitNV");

        VDPAUFiniNV = (void (*)(void))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUFiniNV");

        VDPAURegisterVideoSurfaceNV = (vdpauSurfaceNV (*)(const void *,
                                GLenum,
                                GLsizei,
                                const uint *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAURegisterVideoSurfaceNV");

        VDPAURegisterOutputSurfaceNV = (vdpauSurfaceNV (*)(const void *,
                                GLenum,
                                GLsizei,
                                const uint *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAURegisterOutputSurfaceNV");

        VDPAUIsSurfaceNV = (GLboolean (*)(vdpauSurfaceNV))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUIsSurfaceNV");

        VDPAUUnregisterSurfaceNV = (void (*)(vdpauSurfaceNV))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUUnregisterSurfaceNV");

        VDPAUGetSurfaceivNV = (void (*)(vdpauSurfaceNV,
                                GLenum,
                                GLsizei,
                                GLsizei *,
                                int *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUGetSurfaceivNV");

        VDPAUSurfaceAccessNV = (void (*)(vdpauSurfaceNV, GLenum))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUSurfaceAccessNV");

        VDPAUMapSurfacesNV = (void (*)(GLsizei, const vdpauSurfaceNV *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUMapSurfacesNV");

        VDPAUUnmapSurfacesNV = (void (*)(GLsizei, const vdpauSurfaceNV *))
                glXGetProcAddressARB( (const GLubyte *) "glVDPAUUnmapSurfacesNV");

        return true;
}
#endif

static void gl_draw(double ratio, double bottom_offset, bool double_buf)
{
        float bottom;
        gl_check_error();

        glDrawBuffer(double_buf ? GL_BACK : GL_FRONT);
        if (double_buf) {
                glClear(GL_COLOR_BUFFER_BIT);
        }

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

static void glut_close_callback(void)
{
        gl->should_exit_main_loop = true;
        gl->window = -1;
        exit_uv(0);
}

static int display_gl_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {
#ifdef HWACC_VDPAU
                HW_VDPAU,
#endif
                UYVY,
                RGBA,
                RGB,
                DXT1,
                DXT1_YUV,
                DXT5
        };
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
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

static void display_gl_done(void *state)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        //pthread_join(s->thread_id, NULL);
        if(s->window != -1) {
                glutDestroyWindow(s->window);
        }
        if (s->PHandle_uyvy)
                glDeleteProgram(s->PHandle_uyvy);
        if (s->PHandle_dxt)
                glDeleteProgram(s->PHandle_dxt);
        if (s->PHandle_dxt5)
                glDeleteProgram(s->PHandle_dxt5);
        if (s->texture_display)
                glDeleteTextures(1, &s->texture_display);
        if (s->texture_uyvy)
                glDeleteTextures(1, &s->texture_uyvy);
        if (s->fbo_id)
                glDeleteFramebuffersEXT(1, &s->fbo_id);
        if (s->pbo_id)
                glDeleteBuffersARB(1, &s->pbo_id);

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

        if (s->syphon_spout) {
#ifdef HAVE_SYPHON
                syphon_server_unregister(s->syphon_spout);
#elif defined HAVE_SPOUT
                spout_sender_unregister(s->syphon_spout);
#endif
        }

        gl = nullptr;
        
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
                s->should_exit_main_loop = true; // used only for GLUT (not freeglut)
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

static void display_gl_put_audio_frame(void *state, struct audio_frame *frame)
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
        [](struct device_info **available_cards, int *count) {
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "gl");
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
};

REGISTER_MODULE(gl, &display_gl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


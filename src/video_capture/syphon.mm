/**
 * @file   video_capture/syphon.mm
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017-2020 CESNET, z. s. p. o.
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
 * @todo
 * get rid of GLUT (currently it is needed because we need to operate from within
 * mainloop for which is GLUT useful, better solution would be to deploy native
 * Apple mainloop)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <condition_variable>
#include <chrono>
#include <GLUT/glut.h>
#include <iostream>
#include <mutex>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#include <queue>
#include <Syphon/Syphon.h>

#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "lib_common.h"
#include "mac_gl_common.h"
#include "rang.hpp"
#include "video.h"
#include "video_capture.h"

using std::condition_variable;
using std::cout;
using std::mutex;
using std::queue;
using std::unique_lock;

#define FPS 60.0
#define MOD_NAME "[Syphon capture] "
#define DEFAULT_MAX_QUEUE_SIZE 1

static void usage(bool full);

static const char fp_display_rgba_to_yuv422_legacy[] =
"#define LEGACY 1\n"
    "#if LEGACY\n"
    "#define TEXCOORD gl_TexCoord[0]\n"
    "#else\n"
    "#define TEXCOORD TEX0\n"
    "#define texture2D texture\n"
    "#endif\n"
    "\n"
    "#if LEGACY\n"
    "#define colorOut gl_FragColor\n"
    "#else\n"
    "out vec4 colorOut;\n"
    "#endif\n"
    "\n"
    "#if ! LEGACY\n"
    "in vec4 TEX0;\n"
    "#endif\n"
    "\n"
    "uniform sampler2DRect image;\n"
    "uniform float imageWidth; // is original image width, it means twice as wide as ours\n"
    "\n"
    "void main()\n"
    "{\n"
    "        vec4 rgba1, rgba2;\n"
    "        vec4 yuv1, yuv2;\n"
    "        vec2 coor1, coor2;\n"
    "        float U, V;\n"
    "\n"
    "        coor1 = TEXCOORD.xy - vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "        coor2 = TEXCOORD.xy + vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "\n"
    "        rgba1  = texture2DRect(image, coor1);\n"
    "        rgba2  = texture2DRect(image, coor2);\n"
    "        \n"
    "        yuv1.x = 1.0/16.0 + (rgba1.r * 0.2126 + rgba1.g * 0.7152 + rgba1.b * 0.0722) * 0.8588; // Y\n"
    "        yuv1.y = 0.5 + (-rgba1.r * 0.1145 - rgba1.g * 0.3854 + rgba1.b * 0.5) * 0.8784;\n"
    "        yuv1.z = 0.5 + (rgba1.r * 0.5 - rgba1.g * 0.4541 - rgba1.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        yuv2.x = 1.0/16.0 + (rgba2.r * 0.2126 + rgba2.g * 0.7152 + rgba2.b * 0.0722) * 0.8588; // Y\n"
    "        yuv2.y = 0.5 + (-rgba2.r * 0.1145 - rgba2.g * 0.3854 + rgba2.b * 0.5) * 0.8784;\n"
    "        yuv2.z = 0.5 + (rgba2.r * 0.5 - rgba2.g * 0.4541 - rgba2.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        U = mix(yuv1.y, yuv2.y, 0.5);\n"
    "        V = mix(yuv1.z, yuv2.z, 0.5);\n"
    "        \n"
    "        colorOut = vec4(U,yuv1.x, V, yuv2.x);\n"
    "}\n"
;

/**
 * Class state_vidcap_syphon must be value-initialized
 */
struct state_vidcap_syphon {
        struct video_desc saved_desc;
        SyphonClient *client;
        int window = -1;
        mutex lock;
        condition_variable frame_ready_cv;
        queue<video_frame *> q;
        int max_queue_size = DEFAULT_MAX_QUEUE_SIZE;

        GLuint fbo_id;
        GLuint tex_id;

        NSString *appName;
        NSString *serverName;

        double override_fps;
        bool use_rgb;

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        int frames;

        GLuint program_to_yuv422;

        int show_help;     ///< only show help and exit - 1 - standard; 2 - full
        bool probe_devices; ///< devices probed
        bool should_exit_main_loop = false; ///< events (probe/help) processed

        int probed_devices_count; ///< used only if state_vidcap_syphon::probe_devices is true
        struct device_info *probed_devices; ///< used only if state_vidcap_syphon::probe_devices is true

        ~state_vidcap_syphon() {
                [appName release];
                [serverName release];
                [client release];

                while (q.size() > 0) {
                        video_frame *f = q.front();
                        q.pop();
                        vf_free(f);
                }
        }
};

static void probe_devices_callback(state_vidcap_syphon *s);
static void vidcap_syphon_done(void *state);

static struct state_vidcap_syphon *state_global;

static void reconfigure(state_vidcap_syphon *s, struct video_desc desc) {
        glBindTexture(GL_TEXTURE_2D, s->tex_id);
        if (s->use_rgb) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, desc.width, desc.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        } else {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width / 2, desc.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        }

        glBindFramebufferEXT(GL_FRAMEBUFFER, s->fbo_id);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_id, 0);

        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

        glMatrixMode( GL_MODELVIEW );

        glLoadIdentity( );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity( );
        glViewport( 0, 0, ( GLint ) desc.width / (s->use_rgb ? 1 : 2), ( GLint ) desc.height );

        glOrtho(0, desc.width, 0, desc.height, 10, -10);

        if (!s->use_rgb) {
                glUniform1f(glGetUniformLocation(s->program_to_yuv422, "imageWidth"),
                                (GLfloat) desc.width);
        }

}

static void oneshot_init(int value [[gnu::unused]])
{
        state_vidcap_syphon *s = state_global;

        // Although initialization happens only once, it is important that
        // timer is periodically triggered because only then is glutCheckLoop()
        // left (without that, glutCheckLoop would block infinitely).
        glutTimerFunc(100, oneshot_init, 0);

        if (should_exit || s->should_exit_main_loop) {
                return;
        }

        if (s->show_help != 0) {
                usage(s->show_help == 2);
                s->should_exit_main_loop = true;
                return;
        }

        if (s->probe_devices) {
                probe_devices_callback(s);
                s->should_exit_main_loop = true;
                return;
        }

        // if we are already initialized, exit
        if (s->client)
                return;

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_TEXTURE_RECTANGLE_ARB);

        glGenFramebuffersEXT(1, &s->fbo_id);
        glGenTextures(1, &s->tex_id);

        if (!s->use_rgb) {
                s->program_to_yuv422 = glsl_compile_link(NULL, fp_display_rgba_to_yuv422_legacy);
                glUseProgram(s->program_to_yuv422);
                glUniform1i(glGetUniformLocation(s->program_to_yuv422, "image"), 0);
        }

        NSArray *descriptions;
        if (s->appName || s->serverName) {
                descriptions = [[SyphonServerDirectory sharedDirectory] serversMatchingName:s->serverName appName:s->appName];
        } else {
                descriptions = [[SyphonServerDirectory sharedDirectory] servers];
        }

        if ([descriptions count] == 0) {
                LOG(LOG_LEVEL_ERROR) << "[Syphon capture] No server(s) found!\n";
                return;
        }

        if (!s->override_fps) {
                LOG(LOG_LEVEL_WARNING) << "[Syphon capture] FPS set to " << FPS << ". Use override_fps to override if you know FPS of the server.\n";
        }

        s->client = [[SyphonClient alloc] initWithServerDescription:[descriptions lastObject] context:CGLGetCurrentContext() options:nil newFrameHandler:^(SyphonClient *client) {
                if ([client hasNewFrame] == NO)
                        return;

                SyphonImage *img = [client newFrameImage];
                unsigned int width = [img textureSize].width;
                unsigned int height = [img textureSize].height;

                struct video_desc d{width, height, s->use_rgb ? RGB : UYVY, s->override_fps ? s->override_fps : FPS, PROGRESSIVE, 1};
                if (!video_desc_eq(s->saved_desc, d)) {
                        reconfigure(s, d);
                        s->saved_desc = d;
                }

                struct video_frame *f = vf_alloc_desc_data(d);

                glBindTexture(GL_TEXTURE_RECTANGLE_ARB, [img textureName]);
                glBegin(GL_QUADS);
                glTexCoord2i(0, height);     glVertex2i(0, 0);
                glTexCoord2i(0, 0);          glVertex2i(0, height);
                glTexCoord2i(width, 0);      glVertex2i(width, height);
                glTexCoord2i(width, height); glVertex2i(width, 0);
                glEnd();

                glReadPixels(0, 0, width / (s->use_rgb ? 1 : 2), height, s->use_rgb ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, f->tiles[0].data);
                gl_check_error();

                [img release];

                unique_lock<mutex> lk(s->lock);
                bool pushed = false;
                if (s->q.size() < s->max_queue_size) {
                        s->q.push(f);
                        pushed = true;
                } else {
                        LOG(LOG_LEVEL_WARNING) << "[Syphon capture] Skipping frame.\n";
                        vf_free(f);
                }
                lk.unlock();

                if (pushed) {
                        s->frame_ready_cv.notify_one();
                        debug_msg("[Syphon capture] Frame acquired.\n");
                }
        }];

        if (!s->client) {
                LOG(LOG_LEVEL_ERROR) << "[Syphon capture] Client could have not been created!\n";
        } else {
                NSDictionary *dict = [s->client serverDescription];
                LOG(LOG_LEVEL_NOTICE) << "[Syphon capture] Using server - app: " <<
                        [[dict objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String] << " name: " <<
                        [[dict objectForKey:@"SyphonServerDescriptionNameKey"] UTF8String] << "\n";
        }
}

static void noop()
{
}

static void syphon_mainloop(void *state)
{
        state_global = (struct state_vidcap_syphon *) state;
        struct state_vidcap_syphon *s = state_global;

        macGlutInit(&uv_argc, uv_argv);
        glutInitDisplayMode(GLUT_RGB);
        s->window = glutCreateWindow("dummy Syphon client window");
        glutHideWindow();

        glutDisplayFunc(noop);
        glutTimerFunc(100, oneshot_init, 0);

        while (!should_exit && !s->should_exit_main_loop) {
                glutCheckLoop();
        }
}

/**
 * Prints usage
 *
 * Because it enumerates available servers it must be run from within
 * application main loop, not directly from vidcap_syphon_init().
 */
static void usage(bool full)
{
        cout << "Usage:\n";
        cout << rang::style::bold << rang::fg::red << "\t-t syphon" << rang::fg::reset << "[:name=<server_name>][:app=<app_name>][:override_fps=<fps>][:RGB]" << (full ? "[:queue_size=<len>]" : "[:fullhelp]") << "\n" << rang::style::reset;
        cout << "\nwhere:\n";
        cout << rang::style::bold << "\tname\n" << rang::style::reset << "\t\tSyphon server name\n";
        cout << rang::style::bold << "\tapp\n" << rang::style::reset << "\t\tSyphon server application name\n";
        cout << rang::style::bold << "\toverride_fps\n" << rang::style::reset << "\t\toverrides FPS in metadata (but not the actual rate captured)\n";
        cout << rang::style::bold << "\tRGB\n" << rang::style::reset << "\t\tuse RGB as an output codec instead of default UYVY\n";
        if (full) {
                cout << rang::style::bold << "\tqueue_size=<len>\n" << rang::style::reset << "\t\tsize of internal frame queue\n";
        }
        cout << "\n";
        cout << "Available servers:\n";

        NSArray *descriptions = [[SyphonServerDirectory sharedDirectory] servers];
        for (id item in descriptions) {
                cout << rang::style::bold << "\tapp: " << rang::style::reset << [[item objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String] << rang::style::bold << " name: " << rang::style::reset << [[item objectForKey:@"SyphonServerDescriptionNameKey"] UTF8String] << "\n";
                //...do something useful with myArrayElement
        }
        if ([descriptions count] == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "\tNo Syphon servers found.\n");
        }
        cout << "\n";
}

static int vidcap_syphon_init(struct vidcap_params *params, void **state)
{
        state_vidcap_syphon *s = new state_vidcap_syphon();

        char *opts = strdup(vidcap_params_get_fmt(params));
        char *item, *save_ptr;
        int ret = VIDCAP_INIT_OK;

        item = strtok_r(opts, ":", &save_ptr);
        while (item) {
                if (strcmp(item, "help") == 0 || strcmp(item, "fullhelp") == 0) {
                        s->show_help = strcmp(item, "help") == 0 ? 1 : 2;
                        syphon_mainloop(s);
                        vidcap_syphon_done(s);
                        return VIDCAP_INIT_NOERR;
                } else if (strstr(item, "app=") == item) {
                        s->appName = [NSString stringWithCString: item + strlen("app=") encoding:NSASCIIStringEncoding];
                } else if (strstr(item, "name=") == item) {
                        s->serverName = [NSString stringWithCString: item + strlen("name=") encoding:NSASCIIStringEncoding];
                } else if (strstr(item, "override_fps=") == item) {
                        s->override_fps = atof(item + strlen("override_fps="));
                } else if (strstr(item, "queue_size=") == item) {
                        s->max_queue_size = atoi(strchr(item, '=') + 1);
                } else if (strcasecmp(item, "RGB") == 0) {
                        s->use_rgb = true;
                } else {
                        LOG(LOG_LEVEL_ERROR) << "Syphon: Unknown argument - " << item << "\n";
                        ret = VIDCAP_INIT_FAIL;
                        break;
                }
                item = strtok_r(NULL, ":", &save_ptr);
        }

        free(opts);

        if (ret != VIDCAP_INIT_OK) {
                delete s;
                return ret;
        }

        register_mainloop(syphon_mainloop, s);

        *state = s;

        return VIDCAP_INIT_OK;
}

static void vidcap_syphon_done(void *state)
{
        state_vidcap_syphon *s = (state_vidcap_syphon *) state;

        if (s->window != -1) {
                glutDestroyWindow(s->window);
        }

        if (s->tex_id != 0) {
                glDeleteTextures(1, &s->tex_id);
        }

        if (s->fbo_id != 0) {
                glDeleteFramebuffersEXT(1, &s->fbo_id);
        }

        if (s->program_to_yuv422) {
                glDeleteProgram(s->program_to_yuv422);
        }

        delete s;
}

static struct video_frame *vidcap_syphon_grab(void *state, struct audio_frame **audio)
{
        state_vidcap_syphon *s = (state_vidcap_syphon *) state;
        struct video_frame *ret = NULL;

        unique_lock<mutex> lk(s->lock);
        s->frame_ready_cv.wait_for(lk, std::chrono::milliseconds(100), [s]{return s->q.size() > 0;});
        if (s->q.size() > 0) {
                ret = s->q.front();
                s->q.pop();
                ret->callbacks.dispose = vf_free;

                // statistics
                s->frames++;
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                double seconds = std::chrono::duration_cast<std::chrono::microseconds>(now - s->t0).count() / 1000000.0;
                if (seconds >= 5) {
                        LOG(LOG_LEVEL_INFO) << "[Syphon capture] " << s->frames << " frames in "
                                << seconds << " seconds = " <<  s->frames / seconds << " FPS\n";
                        s->t0 = now;
                        s->frames = 0;
                }
        }

        *audio = NULL;
        return ret;
}

static void probe_devices_callback(state_vidcap_syphon *s)
{
        NSArray *descriptions = [[SyphonServerDirectory sharedDirectory] servers];
        for (id item in descriptions) {
                s->probed_devices_count += 1;
                s->probed_devices = (struct device_info *) realloc(s->probed_devices, s->probed_devices_count * sizeof(struct device_info));
                memset(&s->probed_devices[s->probed_devices_count - 1], 0, sizeof(struct device_info));
                snprintf(s->probed_devices[s->probed_devices_count - 1].dev, sizeof s->probed_devices[s->probed_devices_count - 1].dev,
                                ":app=%s", [[item objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String]);
                snprintf(s->probed_devices[s->probed_devices_count - 1].name, sizeof s->probed_devices[s->probed_devices_count - 1].name,
                                "Syphon %s", [[item objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String]);
        }
}

static struct vidcap_type *vidcap_syphon_probe(bool verbose, void (**deleter)(void *))
{
        *deleter = free;
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }
        vt->name = "syphon";
        vt->description = "Syphon capture client";
        if (!verbose) {
                return vt;
        }


        state_vidcap_syphon *s = new state_vidcap_syphon();
        s->probe_devices = true;
        syphon_mainloop(s);
        vt->card_count = s->probed_devices_count;
        vt->cards = s->probed_devices;
        s->probed_devices = NULL;
        vidcap_syphon_done(s);

        return vt;
}

static const struct video_capture_info vidcap_syphon_info = {
        vidcap_syphon_probe,
        vidcap_syphon_init,
        vidcap_syphon_done,
        vidcap_syphon_grab,
        false
};

REGISTER_MODULE(syphon, &vidcap_syphon_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

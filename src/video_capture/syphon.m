/**
 * @file   video_capture/syphon.mm
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017-2023 CESNET, z. s. p. o.
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
 * Syphon module needs GUI main event loop to be run. However, in macOS, only
 * one event loop can be run (in main thread). If SW display is run at the same
 * time, it uses that, so is needed to handle 2 modes of operandi:
 *
 * 1. SW display is run - in this case we run its event loop
 * 2. no other main event loop is run, in which case UG runs ours registered
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#include <Syphon/Syphon.h>

#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/list.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"

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

        CFRunLoopTimerRef timer;

        struct gl_context gl_context;
        SyphonClient *client;
        pthread_mutex_t lock;
        pthread_cond_t frame_ready_cv;
        struct simple_linked_list *q;
        size_t max_queue_size;

        GLuint fbo_id;
        GLuint tex_id;

        NSString *appName;
        NSString *serverName;

        double override_fps;
        bool use_rgb;

        GLuint program_to_yuv422;

        int show_help;     ///< only show help and exit - 1 - standard; 2 - full
        bool probe_devices; ///< devices probed
        bool mainloop_started; ///< our mainloop is started (if display is GL/SDL, it won't be started)
        bool should_exit_triggered; ///< should_exit callback called just before starting mainloop

        int probed_devices_count; ///< used only if state_vidcap_syphon::probe_devices is true
        struct device_info *probed_devices; ///< used only if state_vidcap_syphon::probe_devices is true
};

static void probe_devices_callback(struct state_vidcap_syphon *s);
static void vidcap_syphon_done(void *state);

static void reconfigure(struct state_vidcap_syphon *s, struct video_desc desc) {
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

static const char *get_syphon_description(SyphonClient *client) {
        static char ret[1024];
        NSDictionary *dict = [client serverDescription];
        snprintf(ret, sizeof ret, "app: %s name: %s",
                [[dict objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String],
                [[dict objectForKey:@"SyphonServerDescriptionNameKey"] UTF8String]);
        return ret;
}

static void stop_application(void) {
        [[NSApplication sharedApplication] stop : nil];
        NSEvent* event = [NSEvent otherEventWithType:NSApplicationDefined
                location:NSMakePoint(0, 0)
                modifierFlags:0
                timestamp:0
                windowNumber:0
                context:nil
                subtype:0
                data1:0
                data2:0];
        [[NSApplication sharedApplication] postEvent:event atStart:YES];
}

static void oneshot_init(CFRunLoopTimerRef timer, void *context);

static void schedule_next_event(struct state_vidcap_syphon *s) {
        CFAbsoluteTime fireTime = CFAbsoluteTimeGetCurrent() + 0.1; // 100 ms
        CFRunLoopTimerSetNextFireDate(s->timer, fireTime);
        CFRunLoopAddTimer(CFRunLoopGetCurrent(), s->timer, kCFRunLoopCommonModes);
}

static void oneshot_init(CFRunLoopTimerRef timer, void *context)
{
        CFRunLoopRemoveTimer(CFRunLoopGetCurrent(), timer, kCFRunLoopCommonModes);
        struct state_vidcap_syphon *s = context;

        if (s->show_help != 0) {
                usage(s->show_help == 2);
                stop_application();
                return;
        }

        if (s->probe_devices) {
                probe_devices_callback(s);
                stop_application();
                return;
        }

        // keepalive - we need to check if s->client is still valid, otherwise
        // no more events arrive and we won't be able to reconfigure to
        // eventual restarted sender
        schedule_next_event(s);

        if (s->client && [s->client isValid] == NO) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Server %s is no longer valid, releasing.\n", get_syphon_description(s->client));
                [s->client release];
                s->client = nil;
        }

        // if we are already initialized, exit
        if (s->client)
                return;

        NSArray *descriptions;
        if (s->appName || s->serverName) {
                descriptions = [[SyphonServerDirectory sharedDirectory] serversMatchingName:s->serverName appName:s->appName];
        } else {
                descriptions = [[SyphonServerDirectory sharedDirectory] servers];
        }
        if ([descriptions count] == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "No server(s) found!\n");
                return;
        }

        gl_context_make_current(&s->gl_context);

        if (!s->override_fps) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "FPS set to %f. Use override_fps to override if you know FPS of the server.\n", FPS);
        }

        s->client = [[SyphonClient alloc] initWithServerDescription:[descriptions lastObject] context:CGLGetCurrentContext() options:nil newFrameHandler:^(SyphonClient *client) {
                if ([client hasNewFrame] == NO)
                        return;

                SyphonImage *img = [client newFrameImage];
                unsigned int width = [img textureSize].width;
                unsigned int height = [img textureSize].height;

                gl_context_make_current(&s->gl_context);

                struct video_desc d = {width, height, s->use_rgb ? RGB : UYVY, s->override_fps ? s->override_fps : FPS, PROGRESSIVE, 1};
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

                pthread_mutex_lock(&s->lock);
                bool pushed = false;
                if (simple_linked_list_append_if_less(s->q, f, s->max_queue_size)) {
                        pushed = true;
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Skipping frame.\n");
                        vf_free(f);
                }
                pthread_mutex_unlock(&s->lock);

                if (pushed) {
                        pthread_cond_signal(&s->frame_ready_cv);
                        debug_msg(MOD_NAME "Frame acquired.\n");
                }
        }];

        if (!s->client) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Client could have not been created!\n");
        } else {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using server - %s\n", get_syphon_description(s->client));
        }
}

static void should_exit_syphon(void *state) {
        struct state_vidcap_syphon *s = (struct state_vidcap_syphon *) state;
        pthread_mutex_lock(&s->lock);
        s->should_exit_triggered = true;
        if (s->mainloop_started) {
                stop_application();
        }
        pthread_mutex_unlock(&s->lock);
}

static void syphon_mainloop(void *state)
{
        struct state_vidcap_syphon *s = (struct state_vidcap_syphon *) state;
        pthread_mutex_lock(&s->lock);
        if (!s->should_exit_triggered) {
                s->mainloop_started = true;
                pthread_mutex_unlock(&s->lock);
                [[NSApplication sharedApplication] run];
        } else {
                pthread_mutex_unlock(&s->lock);
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
        struct key_val options[] = {
                { "name=<name>", "Syphon server name" },
                { "app=<appname>", "Syphon server application name" },
                { "override_fps", "overrides FPS in metadata (but not the actual rate captured)" },
                { "RGB", "use RGB as an output codec instead of default UYVY" },
                { NULL, NULL }
        };
        struct key_val options_full[] = {
                { "queue_size=<qlen>", "size of internal frame queue" },
                { NULL, NULL }
        };
        print_module_usage("-t syphon", options, options_full, full);
        printf("\n");

        printf("Available servers:\n");
        int i = 1;
        NSArray *descriptions = [[SyphonServerDirectory sharedDirectory] servers];
        for (id item in descriptions) {
                color_printf("\t%d) " TBOLD("app:") " %s " TBOLD("name:") " %s\n", i++, [[item objectForKey:@"SyphonServerDescriptionAppNameKey"] UTF8String], [[item objectForKey:@"SyphonServerDescriptionNameKey"] UTF8String]);
                //...do something useful with myArrayElement
        }
        if ([descriptions count] == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "\tNo Syphon servers found.\n");
        }
        color_printf("\n");
}

static int vidcap_syphon_init_common(char *opts, struct state_vidcap_syphon **out)
{
        struct state_vidcap_syphon *s = calloc(1, sizeof *s);
        s->max_queue_size = DEFAULT_MAX_QUEUE_SIZE;
        s->q = simple_linked_list_init();
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->frame_ready_cv, NULL);
        init_gl_context(&s->gl_context, GL_CONTEXT_LEGACY);
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        CFRunLoopTimerContext timerCtxt = { .info = s };
        s->timer = CFRunLoopTimerCreate(NULL, 0, 1.0, 0, 0, oneshot_init, &timerCtxt);
        schedule_next_event(s);

        char *item, *save_ptr;
        item = opts == NULL ? NULL : strtok_r(opts, ":", &save_ptr);
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
                        log_msg(LOG_LEVEL_ERROR, "Syphon: Unknown argument - %s\n", item);
                        vidcap_syphon_done(s);
                        return VIDCAP_INIT_FAIL;
                }
                item = strtok_r(NULL, ":", &save_ptr);
        }

        glGenFramebuffersEXT(1, &s->fbo_id);
        glGenTextures(1, &s->tex_id);

        if (!s->use_rgb) {
                s->program_to_yuv422 = glsl_compile_link(NULL, fp_display_rgba_to_yuv422_legacy);
                assert(s->program_to_yuv422 != 0);
                glUseProgram(s->program_to_yuv422);
                glUniform1i(glGetUniformLocation(s->program_to_yuv422, "image"), 0);
        }

        *out = s;
        return VIDCAP_INIT_OK;
}

static int vidcap_syphon_init(struct vidcap_params *params, void **state)
{
        char *opts = strdup(vidcap_params_get_fmt(params));
        struct state_vidcap_syphon *s = NULL;
        int ret = vidcap_syphon_init_common(opts, &s);
        free(opts);

        if (ret != VIDCAP_INIT_OK) {
                return ret;
        }

        register_should_exit_callback(vidcap_params_get_parent(params), should_exit_syphon, s);
        register_mainloop(syphon_mainloop, s);

        *state = s;

        return VIDCAP_INIT_OK;
}

static void vidcap_syphon_done(void *state)
{
        struct state_vidcap_syphon *s = state;

        if (s->timer) {
                CFRunLoopRemoveTimer(CFRunLoopGetCurrent(), s->timer, kCFRunLoopCommonModes);
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

        destroy_gl_context(&s->gl_context);

        [s->appName release];
        [s->serverName release];
        [s->client release];
        CFRelease(s->timer);

        struct video_frame *f = NULL;
        while ((f = simple_linked_list_pop(s->q)) != NULL) {
                vf_free(f);
        }
        simple_linked_list_destroy(s->q);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_ready_cv);
        free(s);
}

static struct video_frame *vidcap_syphon_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_syphon *s = state;

        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        ts_add_nsec(&ts, 100 * NS_IN_MS);
        pthread_mutex_lock(&s->lock);
        while (simple_linked_list_size(s->q) == 0) {
                if (pthread_cond_timedwait(&s->frame_ready_cv, &s->lock, &ts) != 0) {
                        break;
                }
        }
        struct video_frame *ret = simple_linked_list_pop(s->q);
        pthread_mutex_unlock(&s->lock);
        if (ret != NULL) {
                ret->callbacks.dispose = vf_free;
        }

        *audio = NULL;
        return ret;
}

static void probe_devices_callback(struct state_vidcap_syphon *s)
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

static void vidcap_syphon_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 0;
        *available_devices = NULL;

        struct state_vidcap_syphon *s = NULL;
        if (vidcap_syphon_init_common(NULL, &s) != VIDCAP_INIT_OK) {
                return;
        }
        s->probe_devices = true;
        syphon_mainloop(s);
        *count = s->probed_devices_count;
        *available_devices = s->probed_devices;
        s->probed_devices = NULL;
        vidcap_syphon_done(s);
}

static const struct video_capture_info vidcap_syphon_info = {
        vidcap_syphon_probe,
        vidcap_syphon_init,
        vidcap_syphon_done,
        vidcap_syphon_grab,
        MOD_NAME,
};

REGISTER_MODULE(syphon, &vidcap_syphon_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

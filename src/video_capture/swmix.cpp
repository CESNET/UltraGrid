/*
 * FILE:    swmix.c
 * AUTHOR:  Martin Pulec     <pulec@cesnet.cz>
 *
 * Copyright (c) 2012 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/swmix.h"
#include "audio/audio.h"

#include <queue>
#include <stdio.h>
#include <stdlib.h>

#include "video_display.h"
#include "video.h"

using namespace std;

/* prototypes of functions defined in this module */
static void show_help(void);
static void *master_worker(void *arg);
static void *slave_worker(void *arg);
static char *get_config_name(void);
static bool parse(struct vidcap_swmix_state *s, struct video_desc *desc, char *fmt,
                FILE **config_file);
static bool get_slave_param_from_file(FILE* config, char *slave_name, int *x, int *y,
                                        int *width, int *height);

static char *get_config_name()
{
        static char buf[PATH_MAX];
        if(!getenv("HOME")) {
                return NULL;
        }

        strncpy(buf, getenv("HOME"), sizeof(buf));
        strncat(buf, "/.ug-swmix.rc", sizeof(buf));
        return buf;
}

static void show_help()
{
        printf("SW Mix capture\n");
        printf("Usage\n");
        printf("\t-t swmix:<width>:<height>:<fps>[:<codec>]#<dev1_config>#"
                        "<dev2_config>[#....]\n");
        printf("\tor\n");
        printf("\t-t swmix:file#<dev1_name>@<dev1_config>#"
                        "<dev2_name>@<dev2_config>[#....]\n");
        printf("\t\twhere <devn_config> is a complete configuration string of device "
                        "involved in an SW mix device\n");
        printf("\t\t<devn_name> is an input name (to be matched against config file %s)\n",
                        get_config_name());
        printf("\t\t<width> widht of resulting video\n");
        printf("\t\t<height> height of resulting video\n");
        printf("\t\t<fps> FPS of resulting video\n");
        printf("\t\t<codec> codec of resulting video, may be one of RGBA, "
                        "RGB or UYVY (optional, default RGBA)\n");
}

struct state_slave {
        pthread_t           thread_id;
        bool                should_exit;
        struct vidcap      *device;
        pthread_mutex_t     lock;
        char               *name;

        struct video_frame *captured_frame;
        struct video_frame *done_frame;
};

struct vidcap_swmix_state {
        struct state_slave *slaves;
        int                 devices_cnt;
        struct gl_context   gl_context;

        GLuint              tex_output;
        GLuint              tex_output_uyvy;
        GLuint              fbo;
        GLuint              fbo_uyvy;

        struct video_frame *frame;
        char               *network_buffer;
        char               *completed_buffer;
        queue<char *>       free_buffer_queue;
        pthread_cond_t      free_buffer_queue_not_empty_cv;

        int                 frames;
        struct              timeval t, t0;

        pthread_mutex_t     lock;
        pthread_cond_t      frame_ready_cv;
        pthread_cond_t      frame_sent_cv;

        pthread_t           master_thread_id;
        bool                should_exit;

        struct slave_data  *slaves_data;
        bool                use_config_file;
};


struct vidcap_type *
vidcap_swmix_probe(void)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_SWMIX_ID;
		vt->name        = "swmix";
		vt->description = "SW mix video capture";
	}
	return vt;
}

struct slave_data {
        struct video_frame *current_frame;
        struct video_desc   saved_desc;
        float               posX[4];
        float               posY[4];
        GLuint              texture;
        double              x, y, width, height; // in 1x1 unit space
        double              fb_aspect;
};

static struct slave_data *init_slave_data(vidcap_swmix_state *s, FILE *config) {
        struct slave_data *slaves_data = (struct slave_data *)
                calloc(s->devices_cnt, sizeof(struct slave_data));

        // arrangement
        // we want to have least MxN, where N <= M + 1
        double m = (-1.0 + sqrt(1.0 + 4.0 * s->devices_cnt)) / 2.0;
        m = ceil(m);
        int n = (s->devices_cnt + m - 1) / ((int) m);

        for(int i = 0; i < s->devices_cnt; ++i) {
                glGenTextures(1, &(slaves_data[i].texture));
                glBindTexture(GL_TEXTURE_2D, slaves_data[i].texture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                slaves_data[i].fb_aspect = (double) s->frame->tiles[0].width /
                        s->frame->tiles[0].height;

                if(!s->use_config_file) {
                        slaves_data[i].width = 1.0 / (int) m;
                        slaves_data[i].height = 1.0 / n;
                        slaves_data[i].x = (i % (int) m) * slaves_data[i].width;
                        slaves_data[i].y = (i / (int) m) * slaves_data[i].height;
                } else {
                        int x, y, width, height;
                        if(!get_slave_param_from_file(config, s->slaves[i].name,
                                        &x, &y, &width, &height)) {
                                fprintf(stderr, "Cannot find config for device "
                                                "\"%s\"\n", s->slaves[i].name);
                                free(slaves_data);
                                return NULL;
                        }
                        slaves_data[i].width = (double) width /
                                s->frame->tiles[0].width;
                        slaves_data[i].height = (double) height /
                                s->frame->tiles[0].height;
                        slaves_data[i].x = (double) x /
                                s->frame->tiles[0].width;
                        slaves_data[i].y = (double) y /
                                s->frame->tiles[0].height;
                }
        }

        return slaves_data;
}

static void destroy_slave_data(struct slave_data *data, int count) {
        for(int i = 0; i < count; ++i) {
                glDeleteTextures(1, &data[i].texture);
        }
        free(data);
}

static void reconfigure_slave_rendering(struct slave_data *s, struct video_desc desc)
{
        glBindTexture(GL_TEXTURE_2D, s->texture);
        switch (desc.color_spec) {
                case RGBA:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width, desc.height,
                                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        break;
                case RGB:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width, desc.height,
                                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        break;
                case UYVY:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width / 2, desc.height,
                                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        break;
                default:
                        fprintf(stderr, "SW mix: Unsupported color spec.\n");
                        exit_uv(1);
        }

        double video_aspect = (double) desc.width / desc.height;
        double fb_aspect = (double) s->fb_aspect * s->width / s->height;
        double width = s->width;
        double height = s->height;
        double x = s->x;
        double y = s->y;

        if(video_aspect > fb_aspect) {
                height = width / video_aspect * s->fb_aspect;
                y += (s->height - height) / 2;
        } else {
                width = height * video_aspect / s->fb_aspect;
                x += (s->width - width) / 2;
        }

        // left top
        s->posX[0] = -1.0 + 2.0 * x;
        s->posY[0] = -1.0 + 2.0 * y;

        // right top
        s->posX[1] = -1.0 + 2.0 * x + 2.0 * width;
        s->posY[1] = -1.0 + 2.0 * y;

        // right bottom
        s->posX[2] = -1.0 + 2.0 * x + 2.0 * width;
        s->posY[2] = -1.0 + 2.0 * y + 2.0 * height;

        // left bottom
        s->posX[3] = -1.0 + 2.0 * x;
        s->posY[3] = -1.0 + 2.0 * y + 2.0 * height;
}

// source code for a shader unit
#define STRINGIFY(A) #A
static const char * fprogram_from_uyvy = STRINGIFY(
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
        gl_FragColor.r = yuv.r + 1.7926 * yuv.b;
        gl_FragColor.g = yuv.r - 0.2132 * yuv.g - 0.5328 * yuv.b;
        gl_FragColor.b = yuv.r + 2.1124 * yuv.g;
});

static const char * fprogram_to_uyvy = STRINGIFY(
uniform sampler2D image;
uniform float imageWidth;
void main()
{
        vec4 rgba1;
        vec4 rgba2;
        vec4 yuv1;
        vec4 yuv2;
        vec2 coor1;
        vec2 coor2;
        float U;
        float V;
        coor1 = gl_TexCoord[0].xy - vec2(1.0 / (imageWidth * 2.0), 0.0);
        coor2 = gl_TexCoord[0].xy + vec2(1.0 / (imageWidth * 2.0), 0.0);
        rgba1  = texture2D(image, coor1);
        rgba2  = texture2D(image, coor2);
        yuv1.x = 1.0/16.0 + (rgba1.r * 0.2126 + rgba1.g * 0.7152 + rgba1.b * 0.0722) * 0.8588;
        yuv1.y = 0.5 + (-rgba1.r * 0.1145 - rgba1.g * 0.3854 + rgba1.b * 0.5) * 0.8784;
        yuv1.z = 0.5 + (rgba1.r * 0.5 - rgba1.g * 0.4541 - rgba1.b * 0.0458) * 0.8784;
        yuv2.x = 1.0/16.0 + (rgba2.r * 0.2126 + rgba2.g * 0.7152 + rgba2.b * 0.0722) * 0.8588;
        yuv2.y = 0.5 + (-rgba2.r * 0.1145 - rgba2.g * 0.3854 + rgba2.b * 0.5) * 0.8784;
        yuv2.z = 0.5 + (rgba2.r * 0.5 - rgba2.g * 0.4541 - rgba2.b * 0.0458) * 0.8784;
        U = mix(yuv1.y, yuv2.y, 0.5);
        V = mix(yuv1.z, yuv2.z, 0.5);
        gl_FragColor = vec4(U,yuv1.x, V, yuv2.x);
});

static const char * vprogram = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();
});

static GLuint get_shader(const char *vprogram, const char *fprogram) {
        char log[32768];
        GLuint vhandle, fhandle;
        GLuint phandle;

        phandle = glCreateProgram();
        vhandle = glCreateShader(GL_VERTEX_SHADER);
        fhandle = glCreateShader(GL_FRAGMENT_SHADER);

        /* compile */
        /* fragmemt */
        glShaderSource(fhandle, 1, &fprogram, NULL);
        glCompileShader(fhandle);
        /* Print compile log */
        glGetShaderInfoLog(fhandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);
        /* vertex */
        glShaderSource(vhandle, 1, &vprogram, NULL);
        glCompileShader(vhandle);
        /* Print compile log */
        glGetShaderInfoLog(vhandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        /* attach and link */
        glAttachShader(phandle, vhandle);
        glAttachShader(phandle, fhandle);
        glLinkProgram(phandle);

        printf("Program compilation/link status: ");
        gl_check_error();

        glGetProgramInfoLog(phandle, 32768, NULL, (GLchar*)log);
        if ( strlen(log) > 0 )
                printf("Link Log: %s\n", log);

        // mark shaders for deletion when program is deleted
        glDeleteShader(vhandle);
        glDeleteShader(fhandle);

        return phandle;
}

static void *master_worker(void *arg)
{
        struct vidcap_swmix_state *s = (struct vidcap_swmix_state *) arg;
        struct timeval t0;
        GLuint from_uyvy, to_uyvy;

        gettimeofday(&t0, NULL);

        gl_context_make_current(&s->gl_context);
        glEnable(GL_TEXTURE_2D);
        from_uyvy = get_shader(vprogram, fprogram_from_uyvy);
        to_uyvy = get_shader(vprogram, fprogram_to_uyvy);
        assert(from_uyvy != 0);
        assert(to_uyvy != 0);

        glUseProgram(to_uyvy);
        glUniform1i(glGetUniformLocation(from_uyvy, "image"), 0);
        glUniform1f(glGetUniformLocation(from_uyvy, "imageWidth"),
                        (GLfloat) s->frame->tiles[0].width);
        glUseProgram(0);

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                char *current_buffer = NULL;

                pthread_mutex_lock(&s->lock);
                while(s->free_buffer_queue.empty()) {
                        pthread_cond_wait(&s->free_buffer_queue_not_empty_cv,
                                        &s->lock);
                }
                current_buffer = s->free_buffer_queue.front();
                s->free_buffer_queue.pop();
                pthread_mutex_unlock(&s->lock);

                // "capture" frames
                for(int i = 0; i < s->devices_cnt; ++i) {
                        pthread_mutex_lock(&s->slaves[i].lock);
                        vf_free_data(s->slaves[i].done_frame);
                        s->slaves[i].done_frame = s->slaves_data[i].current_frame;
                        s->slaves_data[i].current_frame = NULL;
                        if(s->slaves[i].captured_frame) {
                                s->slaves_data[i].current_frame =
                                        s->slaves[i].captured_frame;
                                s->slaves[i].captured_frame = NULL;
                        } else if(s->slaves[i].done_frame) {
                                s->slaves_data[i].current_frame =
                                        s->slaves[i].done_frame;
                                s->slaves[i].done_frame = NULL;
                        }
                        pthread_mutex_unlock(&s->slaves[i].lock);
                }

                // check for mode change
                for(int i = 0; i < s->devices_cnt; ++i) {
                        if(s->slaves_data[i].current_frame) {
                                struct video_desc desc = video_desc_from_frame(s->slaves_data[i].current_frame);
                                if(!video_desc_eq(desc, s->slaves_data[i].saved_desc)) {
                                        reconfigure_slave_rendering(&s->slaves_data[i], desc);
                                        s->slaves_data[i].saved_desc = desc;
                                }
                        }
                }

                // draw
                glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT,
                                GL_TEXTURE_2D, s->tex_output, 0);
                glClearColor(0, 0, 0, 1);
                glClear(GL_COLOR_BUFFER_BIT);

                glViewport(0, 0, s->frame->tiles[0].width, s->frame->tiles[0].height);

                for(int i = 0; i < s->devices_cnt; ++i) {
                        if(s->slaves_data[i].current_frame) {
                                glBindTexture(GL_TEXTURE_2D, s->slaves_data[i].texture);
                                int width = s->slaves_data[i].current_frame->tiles[0].width;
                                GLenum format;
                                switch(s->slaves_data[i].current_frame->color_spec) {
                                        case UYVY:
                                                width /= 2;
                                                format = GL_RGBA;
                                                break;
                                        case RGB:
                                                format = GL_RGB;
                                                break;
                                        case RGBA:
                                                format = GL_RGBA;
                                                break;
                                }
                                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                                width,
                                                s->slaves_data[i].current_frame->tiles[0].height,
                                                format, GL_UNSIGNED_BYTE,
                                                s->slaves_data[i].current_frame->tiles[0].data);

                                if(s->slaves_data[i].current_frame->color_spec == UYVY) {
                                        glUseProgram(from_uyvy);
                                        GLuint image_id = glGetUniformLocation(from_uyvy, "image");
                                        glUniform1i(glGetUniformLocation(from_uyvy, "image"), 0);
                                        glUniform1f(glGetUniformLocation(from_uyvy, "imageWidth"),
                                                        (GLfloat) s->slaves_data[i].current_frame->tiles[0].width);
                                }

                                glBegin(GL_QUADS);
                                glTexCoord2f(0.0, 0.0); glVertex2f(s->slaves_data[i].posX[0],
                                                s->slaves_data[i].posY[0]);
                                glTexCoord2f(1.0, 0.0); glVertex2f(s->slaves_data[i].posX[1],
                                                s->slaves_data[i].posY[1]);
                                glTexCoord2f(1.0, 1.0); glVertex2f(s->slaves_data[i].posX[2],
                                                s->slaves_data[i].posY[2]);
                                glTexCoord2f(0.0, 1.0); glVertex2f(s->slaves_data[i].posX[3],
                                                s->slaves_data[i].posY[3]);
                                glEnd();
                                glUseProgram(0);
                        }
                }

                // read back
                glBindTexture(GL_TEXTURE_2D, s->tex_output);
                int width = s->frame->tiles[0].width;
                GLenum format = GL_RGBA;
                if(s->frame->color_spec == UYVY) {
                        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo_uyvy);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT,
                                        GL_TEXTURE_2D, s->tex_output_uyvy, 0);
                        glViewport(0, 0, s->frame->tiles[0].width / 2, s->frame->tiles[0].height);
                        glUseProgram(to_uyvy);
                        glBegin(GL_QUADS);
                        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                        glEnd();
                        glUseProgram(0);
                        width /= 2;
                        glBindTexture(GL_TEXTURE_2D, s->tex_output_uyvy);
                } else if (s->frame->color_spec == RGB) {
                        format = GL_RGB;
                }

                glReadPixels(0, 0, width,
                                s->frame->tiles[0].height,
                                format, GL_UNSIGNED_BYTE,
                                current_buffer);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);

                // wait until next frame time is due
                double sec;
                struct timeval t;
                do {
                        gettimeofday(&t, NULL);
                        sec = tv_diff(t, t0);
                } while(sec < 1.0 / s->frame->fps);
                t0 = t;

                pthread_mutex_lock(&s->lock);
                while(s->completed_buffer != NULL) {
                        pthread_cond_wait(&s->frame_sent_cv, &s->lock);
                }
                s->completed_buffer = current_buffer;
                pthread_cond_signal(&s->frame_ready_cv);
                pthread_mutex_unlock(&s->lock);

        }

        glDeleteProgram(from_uyvy);
        glDeleteProgram(to_uyvy);
        glDisable(GL_TEXTURE_2D);
        gl_context_make_current(NULL);

        return NULL;
}

static void *slave_worker(void *arg)
{
        struct state_slave *s = (struct state_slave *) arg;

        while(!s->should_exit) {
                struct video_frame *frame;
                struct audio_frame *unused_af;

                frame = vidcap_grab(s->device, &unused_af);
                if(frame) {
                        struct video_frame *frame_copy = vf_get_copy(frame);
                        pthread_mutex_lock(&s->lock);
                        if(s->captured_frame) {
                                vf_free_data(s->captured_frame);
                        }
                        s->captured_frame = frame_copy;
                        pthread_mutex_unlock(&s->lock);
                }
        }

        return NULL;
}

bool read_params_from_config_file(struct video_desc *desc, FILE *config_file) {
        char line[1024];
        char codec[5];
        int width, height;
        double fps;
        bool found  = false;

        if(!fgets(line, sizeof(line), config_file))
                return false;
        int items_read = sscanf(line, "%d:%d:%lf:%4s", &width, &height, &fps, codec);
        if(items_read != 3 && items_read != 4) {
                fprintf(stderr, "Not enough arguments in config file. Was read: \n"
                               " %s\nExpecting <width>:<height>:<fps>[:<codec>]\n", line);
        }
        desc->width = width;
        desc->height = height;
        desc->fps = fps;

        if(items_read == 4) {
                for (int i = 0; codec_info[i].name != NULL; i++) {
                        if (strcmp(codec, codec_info[i].name) == 0) {
                                desc->color_spec = codec_info[i].codec;
                                found = true;
                        }
                }
                if(!found) {
                        fprintf(stderr, "Unrecognized color spec string: %s\n", codec);
                        return false;
                }
        }

        return true;
}

static bool get_slave_param_from_file(FILE* config_file, char *slave_name, int *x, int *y,
                                        int *width, int *height)
{
        char *ret;
        char line[1024];
        fseek(config_file, 0, SEEK_SET); // rewind
        ret = fgets(line, sizeof(line), config_file);  // skip first line
        if(!ret) return false;
        while (fgets(line, sizeof(line), config_file)) {
                char name[128];
                int x_, y_, width_, height_;
                if(sscanf(line, "%128s %d %d %d %d", name, &x_, &y_, &width_, &height_) != 5)
                        continue;
                if(strcasecmp(name, slave_name) == 0) {
                        *x = x_;
                        *y = y_;
                        *width = width_;
                        *height = height_;
                        return true;
                }
        }
        return false;
}

static bool parse(struct vidcap_swmix_state *s, struct video_desc *desc, char *fmt,
                FILE **config_file)
{
        char *save_ptr = NULL;
        char *item;
        char *parse_string;
        char *tmp;
        int token_nr = 0;
        *config_file = NULL;
        tmp = parse_string = strdup(fmt);
        if(strchr(parse_string, '#')) *strchr(parse_string, '#') = '\0';
        while((item = strtok_r(tmp, ":", &save_ptr))) {
                bool found = false;
                switch (token_nr) {
                        case 0:
                                desc->width = atoi(item);
                                break;
                        case 1:
                                desc->height = atoi(item);
                                break;
                        case 2:
                                desc->fps = atof(item);
                                break;
                        case 3:
                                for (int i = 0; codec_info[i].name != NULL; i++) {
                                        if (strcmp(item, codec_info[i].name) == 0) {
                                                desc->color_spec = codec_info[i].codec;
                                                found = true;
                                        }
                                }
                                if(!found) {
                                        fprintf(stderr, "Unrecognized color spec string: %s\n", item);
                                        return false;
                                }
                                break;
                }
                tmp = NULL;
                token_nr += 1;
        }
        free(parse_string);

        if(desc->width == 0 && desc->height == 0 && desc->fps == 0.0) {
                s->use_config_file = true;

                *config_file = fopen(get_config_name(), "r");
                if(!*config_file) {
                        fprintf(stderr, "Params not set and config file %s not found.\n",
                                        get_config_name());
                        return false;
                }
                if(!read_params_from_config_file(desc, *config_file)) {
                        return false;
                }
        } else {
                if(desc->width * desc->height * desc->fps == 0.0) {
                        show_help();
                        return false;
                }
        }

        if(desc->color_spec != RGBA && desc->color_spec != RGB && desc->color_spec != UYVY) {
                fprintf(stderr, "Unsupported output codec.\n");
                return false;
        }

        s->devices_cnt = -1;
        tmp = parse_string = strdup(fmt);
        while(strtok_r(tmp, "#", &save_ptr)) {
                s->devices_cnt++;
                tmp = NULL;
        }
        free(parse_string);

        s->slaves = (struct state_slave *) calloc(s->devices_cnt, sizeof(struct state_slave));
        int i = 0;
        parse_string = strdup(fmt);
        strtok_r(parse_string, "#", &save_ptr); // drop first part
        while((item = strtok_r(NULL, "#", &save_ptr))) {
                char *copy = strdup(item);
                char *device;
                char *config = copy;
                char *device_cfg = NULL;
                char *name = NULL;

                // we have device name to be matched against config file
                if(strchr(config, '@')) {
			char *delim = strchr(config, '@');
			*delim = '\0';
                        name = config;
			config = delim + 1;
                }
                device = config;
		if(strchr(config, ':')) {
			char *delim = strchr(config, ':');
			*delim = '\0';
			device_cfg = delim + 1;
		}

                s->slaves[i].device = initialize_video_capture(device,
                                               device_cfg, 0);
                if(name) {
                        s->slaves[i].name = strdup(name);
                } else {
                        s->slaves[i].name = NULL;
                }

                free(copy);
                if(!s->slaves[i].device) {
                        fprintf(stderr, "[swmix] Unable to initialize device %d.\n", i);
                        return false;
                }
                ++i;
        }
        free(parse_string);

        return true;
}

void *
vidcap_swmix_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_swmix_state *s;
        struct video_desc desc;
        GLenum format;

	printf("vidcap_swmix_init\n");

        s = new vidcap_swmix_state;
	if(s == NULL) {
		printf("Unable to allocate swmix capture state\n");
		return NULL;
	}

        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(!init_fmt || strcmp(init_fmt, "help") == 0) {
                show_help();
                return NULL;
        }

        memset(&desc, 0, sizeof(desc));
        desc.tile_count = 1;
        desc.color_spec = RGBA;

        FILE *config_file = NULL;

        if(!parse(s, &desc, init_fmt, &config_file)) {
                goto error;
        }

        s->frame = vf_alloc_desc(desc);

        s->should_exit = false;
        s->completed_buffer = NULL;
        s->network_buffer = NULL;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->frame_ready_cv, NULL);
        pthread_cond_init(&s->frame_sent_cv, NULL);
        pthread_cond_init(&s->free_buffer_queue_not_empty_cv, NULL);

        if(!init_gl_context(&s->gl_context, GL_CONTEXT_LEGACY)) {
                fprintf(stderr, "[swmix] Unable to initialize OpenGL context.\n");
                goto error;
        }

        gl_context_make_current(&s->gl_context);

        s->slaves_data = init_slave_data(s, config_file);
        if(!s->slaves_data) {
                return NULL;
        }

        if(config_file)
                fclose(config_file);

        format = GL_RGBA;
        if(desc.color_spec == RGB) {
                format = GL_RGB;
        }
        glGenTextures(1, &s->tex_output);
        glBindTexture(GL_TEXTURE_2D, s->tex_output);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, format, desc.width, desc.height,
                        0, format, GL_UNSIGNED_BYTE, NULL);

        glGenTextures(1, &s->tex_output_uyvy);
        glBindTexture(GL_TEXTURE_2D, s->tex_output_uyvy);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width / 2, desc.height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glGenFramebuffers(1, &s->fbo);
        glGenFramebuffers(1, &s->fbo_uyvy);

        gl_context_make_current(NULL);

        for(int i = 0; i < s->devices_cnt; ++i) {
                pthread_mutex_init(&(s->slaves[i].lock), NULL);
        }

        s->frame->tiles[0].data_len = vc_get_linesize(s->frame->tiles[0].width,
                                s->frame->color_spec) * s->frame->tiles[0].height;
        for(int i = 0; i < 3; ++i) {
                char *buffer = (char *) malloc(s->frame->tiles[0].data_len);
                s->free_buffer_queue.push(buffer);
        }

        pthread_create(&s->master_thread_id, NULL, master_worker, (void *) s);

        for(int i = 0; i < s->devices_cnt; ++i) {
                pthread_create(&(s->slaves[i].thread_id), NULL, slave_worker, (void *) &s->slaves[i]);
        }

	return s;

error:
        if(s->slaves) {
                int i;
                for (i = 0u; i < s->devices_cnt; ++i) {
                        if(s->slaves[i].device) {
                                 vidcap_done(s->slaves[i].device);
                        }
                }
                free(s->slaves);
        }
        delete s;
        return NULL;
}

void
vidcap_swmix_finish(void *state)
{
	struct vidcap_swmix_state *s = (struct vidcap_swmix_state *) state;

}

void
vidcap_swmix_done(void *state)
{
	struct vidcap_swmix_state *s = (struct vidcap_swmix_state *) state;

	assert(s != NULL);

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->slaves[i].should_exit = true;
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                pthread_join(s->slaves[i].thread_id, NULL);
        }

        // wait for master thread to finish
        pthread_mutex_lock(&s->lock);
        s->should_exit = true;
        if(s->network_buffer) {
                s->free_buffer_queue.push(s->network_buffer);
                s->network_buffer = NULL;
                pthread_cond_signal(&s->free_buffer_queue_not_empty_cv);
        }
        if(s->completed_buffer) {
                s->free_buffer_queue.push(s->completed_buffer);
                s->completed_buffer = NULL;
                pthread_cond_signal(&s->frame_sent_cv);
        }
        pthread_mutex_unlock(&s->lock);
        pthread_join(s->master_thread_id, NULL);

        if(s->completed_buffer)
                free(s->completed_buffer);
        while(!s->free_buffer_queue.empty()) {
                free(s->free_buffer_queue.front());
                s->free_buffer_queue.pop();
        }

       for (int i = 0; i < s->devices_cnt; ++i) {
                vidcap_finish(s->slaves[i].device);
                vidcap_done(s->slaves[i].device);
                pthread_mutex_destroy(&s->slaves[i].lock);
                vf_free_data(s->slaves[i].captured_frame);
                vf_free_data(s->slaves[i].done_frame);
        }
        free(s->slaves);

        vf_free(s->frame);

        gl_context_make_current(&s->gl_context);

        destroy_slave_data(s->slaves_data, s->devices_cnt);

        glDeleteTextures(1, &s->tex_output);
        glDeleteTextures(1, &s->tex_output_uyvy);
        glDeleteFramebuffers(1, &s->fbo);
        glDeleteFramebuffers(1, &s->fbo_uyvy);

        gl_context_make_current(NULL);
        destroy_gl_context(&s->gl_context);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_ready_cv);
        pthread_cond_destroy(&s->frame_sent_cv);
        pthread_cond_destroy(&s->free_buffer_queue_not_empty_cv);

        delete s;
}

struct video_frame *
vidcap_swmix_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_swmix_state *s = (struct vidcap_swmix_state *) state;

        *audio = NULL;

        pthread_mutex_lock(&s->lock);
        while(s->completed_buffer == NULL) {
                pthread_cond_wait(&s->frame_ready_cv, &s->lock);
        }
        if(s->network_buffer) {
                s->free_buffer_queue.push(s->network_buffer);
                pthread_cond_signal(&s->free_buffer_queue_not_empty_cv);
        }
        s->network_buffer = s->completed_buffer;
        s->completed_buffer = NULL;
        pthread_cond_signal(&s->frame_sent_cv);
        pthread_mutex_unlock(&s->lock);

        s->frame->tiles[0].data = s->network_buffer;

        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "[swmix cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }

	return s->frame;
}


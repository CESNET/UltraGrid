/*
 * Copyright (c) 2012-2013 CESNET z.s.p.o.
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
 * @file   video_capture/swmix.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief SW video mix is a virtual video mixer.
 *
 * @todo
 * Reenable configuration file position matching.
 *
 * @todo
 * Refactor to use also different scalers than OpenGL (eg. libswscale)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "gl_context.h"
#include "host.h"
#include "lib_common.h"
#include "utils/config_file.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

#include <queue>
#include <stdio.h>
#include <stdlib.h>

#define MAX_AUDIO_LEN (1024*1024)

using namespace std;

typedef enum {
        BICUBIC,
        BILINEAR
} interpolation_t;

/*
 * Bicubic interpolation taken from:
 * http://www.codeproject.com/Articles/236394/Bi-Cubic-and-Bi-Linear-Interpolation-with-GLSL
 */
#define STRINGIFY(A) #A
static const char * bicubic_template = STRINGIFY(
uniform sampler2D textureSampler;
uniform float fWidth;
uniform float fHeight;
float CatMullRom( float x )
{
    const float B = 0.0;
    const float C = 0.5;
    float f = x;
    if( f < 0.0 )
    {
        f = -f;
    }
    if( f < 1.0 )
    {
        return ( ( 12.0 - 9.0 * B - 6.0 * C ) * ( f * f * f ) +
            ( -18.0 + 12.0 * B + 6.0 *C ) * ( f * f ) +
            ( 6.0 - 2.0 * B ) ) / 6.0;
    }
    else if( f >= 1.0 && f < 2.0 )
    {
        return ( ( -B - 6.0 * C ) * ( f * f * f )
            + ( 6.0 * B + 30.0 * C ) * ( f *f ) +
            ( - ( 12.0 * B ) - 48.0 * C  ) * f +
            8.0 * B + 24.0 * C)/ 6.0;
    }
    else
    {
        return 0.0;
    }
}
float BSpline( float x )
{
        float f = x;
        if( f < 0.0 )
        {
                f = -f;
        }

        if( f >= 0.0 && f <= 1.0 )
        {
                return ( 2.0 / 3.0 ) + ( 0.5 ) * ( f* f * f ) - (f*f);
        }
        else if( f > 1.0 && f <= 2.0 )
        {
                return 1.0 / 6.0 * pow( ( 2.0 - f  ), 3.0 );
        }
        return 1.0;
}
float Triangular( float f )
{
        f = f / 2.0;
        if( f < 0.0 )
        {
                return ( f + 1.0 );
        }
        else
        {
                return ( 1.0 - f );
        }
        return 0.0;
}
void main()
{
    float texelSizeX = 1.0 / fWidth;
    float texelSizeY = 1.0 / fHeight;
    vec4 nSum = vec4( 0.0, 0.0, 0.0, 0.0 );
    vec4 nDenom = vec4( 0.0, 0.0, 0.0, 0.0 );
    float a = fract( gl_TexCoord[0].x * fWidth );
    float b = fract( gl_TexCoord[0].y * fHeight );
    for( int m = -1; m <=2; m++ )
    {
        for( int n =-1; n<= 2; n++)
        {
                        vec4 vecData = texture2D(textureSampler,
                               gl_TexCoord[0].xy + vec2(texelSizeX * float( m ),
                                        texelSizeY * float( n )));
                        float f  = INTERP_ALGORITHM_PLACEHOLDER( float( m ) - a );
                        vec4 vecCooef1 = vec4( f,f,f,f );
                        float f1 = INTERP_ALGORITHM_PLACEHOLDER( -( float( n ) - b ) );
                        vec4 vecCoeef2 = vec4( f1, f1, f1, f1 );
            nSum = nSum + ( vecData * vecCoeef2 * vecCooef1  );
            nDenom = nDenom + (( vecCoeef2 * vecCooef1 ));
        }
    }
    gl_FragColor = nSum / nDenom;
});

/* prototypes of functions defined in this module */
static void show_help(void);
static void *master_worker(void *arg);
static void *slave_worker(void *arg);
static char *get_config_name(void);
static bool get_slave_param_from_file(FILE* config, const char *slave_name, int *x, int *y,
                                        int *width, int *height);
static bool get_device_config_from_file(FILE* config_file, char *slave_name,
                char *device_name_config) __attribute__((unused));

static char *get_config_name()
{
        const char *rc_suffix = "/.ug-swmix.rc";
        static char buf[PATH_MAX];
        if(!getenv("HOME")) {
                return NULL;
        }

        strncpy(buf, getenv("HOME"), sizeof(buf) - 1);
        strncat(buf, rc_suffix, sizeof(buf) - 1);
        return buf;
}

static void show_help()
{
        printf("SW Mix capture\n");
        printf("Usage\n");
        printf("\t-t swmix:<width>:<height>:<fps>[:<codec>[:interpolation=<i_type>[,<algo>]][:layout=<X>x<Y>]] "
                        "-t <dev1_config> -t <dev2_config>\n");
        printf("\tor\n");
        printf("\t-t swmix:file -t <dev1_config> -t <dev2_config> ...\n");
        printf("\t\twhere <devn_config> is a configuration string of device as usual -\n"
                        "\t\t\tdevice config string or alias from UG config file");
        printf("\t\t<width> widht of resulting video\n");
        printf("\t\t<height> height of resulting video\n");
        printf("\t\t<fps> FPS of resulting video, may be eg. 25 or 50i\n");
        printf("\t\t<codec> codec of resulting video, may be one of RGBA, "
                        "RGB or UYVY (optional, default RGBA)\n");
        printf("\t\t<i_type> can be one of 'bilinear' or 'bicubic' (default)\n");
        printf("\t\t\t<algo> bicubic interpolation algorithm: CatMullRom, BSpline (default) or Triangular\n");
        printf("\n");
        printf("\t\tIn first variant, individual inputs are arranged automatically.\n");
        printf("\t\tWith the second variant, you provide overall layout and layout for \n"
                        "\t\tindividual inputs in SW mix config file (%s).\n", get_config_name());
}

struct state_slave {
        pthread_t           thread_id;
        bool                should_exit;
        pthread_mutex_t     lock;
        struct vidcap_params *device_params;

        struct video_frame *captured_frame;
        struct video_frame *done_frame;

        struct audio_frame  audio_frame;
        bool                audio_captured;
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
        char               *network_audio_buffer;
        char               *completed_audio_buffer;
        int                 completed_audio_buffer_len;
        struct audio_frame  audio;
        int                 audio_device_index; ///< index of video device from which to take audio
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

        char               *bicubic_algo;
        GLuint              bicubic_program;
        interpolation_t     interpolation;
        int                 grid_x, grid_y;
};


static struct vidcap_type *
vidcap_swmix_probe(bool verbose)
{
        UNUSED(verbose);
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
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
        GLuint              texture[2]; // RGB(A), (UYVY)
        GLuint              fbo; // RGB(A)
        double              x, y, width, height; // in 1x1 unit space
        double              fb_aspect;

        decoder_t           decoder;
        codec_t             decoder_from, decoder_to;
};

static struct slave_data *init_slave_data(vidcap_swmix_state *s, FILE *config) {
        struct slave_data *slaves_data = (struct slave_data *)
                calloc(s->devices_cnt, sizeof(struct slave_data));
        double m;
        int n;

        if (s->grid_x != 0) {
                m = s->grid_x;
                n = s->grid_y;
        } else {
                // arrangement
                // we want to have least MxN, where N <= M + 1
                m = (-1.0 + sqrt(1.0 + 4.0 * s->devices_cnt)) / 2.0;
                m = ceil(m);
                n = (s->devices_cnt + m - 1) / ((int) m);
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                glGenTextures(2, slaves_data[i].texture);
                for(int j = 0; j < 2; ++j) {
                        glBindTexture(GL_TEXTURE_2D, slaves_data[i].texture[j]);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                }

                glGenFramebuffers(1, &slaves_data[i].fbo);

                slaves_data[i].fb_aspect = (double) s->frame->tiles[0].width /
                        s->frame->tiles[0].height;

                if(!s->use_config_file) {
                        slaves_data[i].width = 1.0 / (int) m;
                        slaves_data[i].height = 1.0 / n;
                        slaves_data[i].x = (i % (int) m) * slaves_data[i].width;
                        slaves_data[i].y = (i / (int) m) * slaves_data[i].height;
                } else {
                        int x, y, width, height;
                        if(!get_slave_param_from_file(config,
                                                vidcap_params_get_name(s->slaves[i].device_params),
                                        &x, &y, &width, &height)) {
                                fprintf(stderr, "Cannot find config for device \"%s\"\n",
                                                vidcap_params_get_name(s->slaves[i].device_params));
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
                glDeleteTextures(2, data[i].texture);
                glDeleteFramebuffers(1, &data[i].fbo);
        }
        free(data);
}

static void reconfigure_slave_rendering(struct slave_data *s, struct video_desc desc)
{
        glBindTexture(GL_TEXTURE_2D, s->texture[0]);
        switch (desc.color_spec) {
                case UYVY:
                case RGBA:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width, desc.height,
                                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        break;
                case RGB:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, desc.width, desc.height,
                                        0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
                        break;
                case BGR:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, desc.width, desc.height,
                                        0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
                        break;
                default:
                        fprintf(stderr, "SW mix: Unsupported color spec.\n");
                        exit_uv(1);
        }

        if(desc.color_spec == UYVY) {
                glBindTexture(GL_TEXTURE_2D, s->texture[1]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, desc.width / 2, desc.height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        }
        glBindTexture(GL_TEXTURE_2D, 0);

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

static void check_for_slave_format_change(struct slave_data *s)
{
        struct video_desc desc = video_desc_from_frame(s->current_frame);

        if(!video_desc_eq(desc, s->saved_desc)) {
                codec_t in_codec = s->current_frame->color_spec;
                codec_t out_codec = in_codec;
                codec_t natively_supported[] = { RGB, BGR, RGBA, UYVY, VIDEO_CODEC_NONE };

                // prepare decoder
                if (!codec_is_in_set(desc.color_spec, natively_supported)) {
                        int j = 0;
                        while (natively_supported[j] != VIDEO_CODEC_NONE) {
                                if ((s->decoder = get_decoder_from_to(out_codec, natively_supported[j], false))) {
                                        s->decoder_from = desc.color_spec;
                                        desc.color_spec = s->decoder_to = natively_supported[j];
                                        break;
                                }
                                j++;
                        }
                }

                reconfigure_slave_rendering(s, desc);
                desc.color_spec = s->decoder_from;
                s->saved_desc = desc;
        }
}

static void load_texture(struct slave_data *s, GLuint from_uyvy)
{
        glBindTexture(GL_TEXTURE_2D, s->texture[0]);
        int width = s->current_frame->tiles[0].width;
        GLenum format;

        codec_t in_codec = s->current_frame->color_spec;
        codec_t out_codec = in_codec;
        codec_t natively_supported[] = { RGB, BGR, RGBA, UYVY, VIDEO_CODEC_NONE };
        decoder_t decoder = NULL;
        if (!codec_is_in_set(out_codec, natively_supported)) {
                if (s->decoder_from == in_codec) {
                        out_codec = s->decoder_to;
                        decoder = s->decoder;
                }
        }

        switch (out_codec) {
                case UYVY:
                        width /= 2;
                        format = GL_RGBA;
                        break;
                case RGB:
                        format = GL_RGB;
                        break;
                case BGR:
                        format = GL_BGR;
                        break;
                case RGBA:
                        format = GL_RGBA;
                        break;
                default:
                        error_msg("Unexpected color space %s!",
                                        get_codec_name(out_codec));
                        abort();
        }
        int src_height = s->current_frame->tiles[0].height;
        int src_width = s->current_frame->tiles[0].width;
        unsigned char *data = (unsigned char *) s->current_frame->tiles[0].data;
        unsigned char *tmp = NULL, *in_gl_buffer = data;
        if (decoder) {
                tmp = in_gl_buffer = (unsigned char *) malloc(src_height *
                                vc_get_linesize(src_width, out_codec));
                for (int i = 0; i < src_height; ++i) {
                        decoder(tmp + i * vc_get_linesize(src_width, out_codec),
                                        data + i * vc_get_linesize(src_width, in_codec),
                                        vc_get_linesize(src_width, out_codec), 0, 8, 16);
                }
        }

        if(out_codec == UYVY) {
                glBindTexture(GL_TEXTURE_2D, s->texture[1]);
        } else {
                glBindTexture(GL_TEXTURE_2D, s->texture[0]);
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        width, src_height, format, GL_UNSIGNED_BYTE, in_gl_buffer);
        free(tmp);

        if(out_codec == UYVY) {
                glUseProgram(from_uyvy);
                glUniform1i(glGetUniformLocation(from_uyvy, "image"), 0);
                glUniform1f(glGetUniformLocation(from_uyvy, "imageWidth"),
                                (GLfloat) s->current_frame->tiles[0].width);

                glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT,
                                GL_TEXTURE_2D, s->texture[0], 0);

                glViewport(0, 0, s->current_frame->tiles[0].width,
                                s->current_frame->tiles[0].height);
                glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                glEnd();
                glUseProgram(0);
        }
}

static void render_slave(struct slave_data *s, interpolation_t interpolation, GLuint bicubic_program)
{
        if(interpolation == BICUBIC) {
                glUniform1f(glGetUniformLocation(bicubic_program, "fWidth"),
                                (GLfloat) s->current_frame->tiles[0].width);
                glUniform1f(glGetUniformLocation(bicubic_program, "fHeight"),
                                (GLfloat) s->current_frame->tiles[0].height);
        }
        glBindTexture(GL_TEXTURE_2D, s->texture[0]);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(s->posX[0],
                        s->posY[0]);
        glTexCoord2f(1.0, 0.0); glVertex2f(s->posX[1],
                        s->posY[1]);
        glTexCoord2f(1.0, 1.0); glVertex2f(s->posX[2],
                        s->posY[2]);
        glTexCoord2f(0.0, 1.0); glVertex2f(s->posX[3],
                        s->posY[3]);
        glEnd();
}

static void *master_worker(void *arg)
{
        struct vidcap_swmix_state *s = (struct vidcap_swmix_state *) arg;
        struct timeval t0;
        GLuint from_uyvy, to_uyvy;

        gettimeofday(&t0, NULL);

        gl_context_make_current(&s->gl_context);
        glEnable(GL_TEXTURE_2D);
        from_uyvy = glsl_compile_link(vprogram, fprogram_from_uyvy);
        to_uyvy = glsl_compile_link(vprogram, fprogram_to_uyvy);
        assert(from_uyvy != 0);
        assert(to_uyvy != 0);

        glUseProgram(to_uyvy);
        glUniform1i(glGetUniformLocation(to_uyvy, "image"), 0);
        glUniform1f(glGetUniformLocation(to_uyvy, "imageWidth"),
                        (GLfloat) s->frame->tiles[0].width);
        glUseProgram(0);

        int field = 0;
        char *tmp_buffer = (char *) malloc(s->frame->tiles[0].data_len);

        char *current_buffer = NULL;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                if(field == 0) {
                        pthread_mutex_lock(&s->lock);
                        while(s->free_buffer_queue.empty()) {
                                pthread_cond_wait(&s->free_buffer_queue_not_empty_cv,
                                                &s->lock);
                        }
                        current_buffer = s->free_buffer_queue.front();
                        s->free_buffer_queue.pop();
                        pthread_mutex_unlock(&s->lock);
                }

                // "capture" frames
                for(int i = 0; i < s->devices_cnt; ++i) {
                        pthread_mutex_lock(&s->slaves[i].lock);
                        VIDEO_FRAME_DISPOSE(s->slaves[i].done_frame);
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
                                check_for_slave_format_change(&s->slaves_data[i]);
                        }
                }

                char *audio_data = NULL;
                int audio_len = 0;

                // load data
                for(int i = 0; i < s->devices_cnt; ++i) {
                        if(s->slaves_data[i].current_frame) {
                                if(s->slaves[i].audio_captured && s->audio_device_index == -1) {
                                        fprintf(stderr, "[swmix] Locking device #%d as an audio source.\n",
                                                        i);
                                        s->audio_device_index = i;
                                }

                                if ((s->frame->interlacing != INTERLACED_MERGED || field == 1)
							&& s->audio_device_index == i) {
                                        s->audio.bps = s->slaves[i].audio_frame.bps;
                                        s->audio.ch_count = s->slaves[i].audio_frame.ch_count;
                                        s->audio.sample_rate = s->slaves[i].audio_frame.sample_rate;
                                        if(s->slaves[i].audio_frame.data_len) {
                                                audio_data = (char *) malloc(s->slaves[i].audio_frame.data_len);
                                                audio_len = s->slaves[i].audio_frame.data_len;
                                                memcpy(audio_data, s->slaves[i].audio_frame.data,
                                                                audio_len);
                                                s->slaves[i].audio_frame.data_len = 0;
                                        }
                                }

                                load_texture(&s->slaves_data[i], from_uyvy);

                        }
                }

                // draw
                glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT,
                                GL_TEXTURE_2D, s->tex_output, 0);
                glClearColor(0, 0, 0, 1);
                glClear(GL_COLOR_BUFFER_BIT);

                glViewport(0, 0, s->frame->tiles[0].width, s->frame->tiles[0].height);

                if(s->interpolation == BICUBIC) {
                        glUseProgram(s->bicubic_program);
                        glUniform1i(glGetUniformLocation(s->bicubic_program, "image"), 0);
                }

                for(int i = 0; i < s->devices_cnt; ++i) {
                        if(s->slaves_data[i].current_frame) {
                                render_slave(&s->slaves_data[i], s->interpolation, s->bicubic_program);
                        }
                }
                glUseProgram(0);

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

                char *read_buf;
                if(s->frame->interlacing == PROGRESSIVE) {
                        read_buf = current_buffer;
                } else {
                        read_buf = tmp_buffer;
                }
                glReadPixels(0, 0, width,
                                s->frame->tiles[0].height,
                                format, GL_UNSIGNED_BYTE,
                                read_buf);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);

                if(s->frame->interlacing == INTERLACED_MERGED) {
                        int linesize =
                                vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec);
                        for(unsigned int i = field; i < s->frame->tiles[0].height; i += 2) {
                                memcpy(current_buffer + i * linesize, tmp_buffer + i * linesize,
                                                linesize);
                        }
                        field = (field + 1) % 2;
                }

                if(field == 0) {
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
                        s->completed_audio_buffer = audio_data;
                        s->completed_audio_buffer_len = audio_len;
                        current_buffer = NULL;
                        pthread_cond_signal(&s->frame_ready_cv);
                        pthread_mutex_unlock(&s->lock);
                        field = 0;
                }

        }

        free(tmp_buffer);

        glDeleteProgram(from_uyvy);
        glDeleteProgram(to_uyvy);
        glDisable(GL_TEXTURE_2D);
        gl_context_make_current(NULL);

        return NULL;
}

static void *slave_worker(void *arg)
{
        struct state_slave *s = (struct state_slave *) arg;

        struct vidcap *device;
        int ret =
                initialize_video_capture(NULL, s->device_params, &device);
        if(ret != 0) {
                fprintf(stderr, "[swmix] Unable to initialize device %s (%s:%s).\n",
                                vidcap_params_get_name(s->device_params),
                                vidcap_params_get_driver(s->device_params),
                                vidcap_params_get_fmt(s->device_params));
                exit_uv(1);
                return NULL;
        }

        while(!s->should_exit) {
                struct video_frame *frame;
                struct audio_frame *audio;

                frame = vidcap_grab(device, &audio);
                if (frame) {
                        struct video_frame *frame_local;
                        if (frame->callbacks.dispose) {
                                frame_local = frame;
                        } else {
                                frame_local = vf_get_copy(frame);
                                frame_local->callbacks.dispose = vf_free;
                                frame_local->callbacks.dispose_udata = NULL; // not needed
                        }
                        if (frame_local->interlacing == INTERLACED_MERGED) {
                                vc_deinterlace((unsigned char *) frame_local->tiles[0].data,
                                                vc_get_linesize(frame_local->tiles[0].width, frame_local->color_spec),
                                                frame_local->tiles[0].height);
                        }
                        pthread_mutex_lock(&s->lock);
                        // video frame was not processed, simply replace it
                        VIDEO_FRAME_DISPOSE(s->captured_frame);
                        s->captured_frame = frame_local;
                        if(audio) {
                                s->audio_captured = true;
                                int len = audio->data_len;
                                if(len + s->audio_frame.data_len > (int) s->audio_frame.max_size) {
                                        len = s->audio_frame.max_size - s->audio_frame.data_len;
                                        fprintf(stderr, "[SW Mix] Audio buffer overflow!\n");
                                }
                                memcpy(s->audio_frame.data + s->audio_frame.data_len, audio->data,
                                                len);
                                s->audio_frame.data_len += len;
                                s->audio_frame.ch_count = audio->ch_count;
                                s->audio_frame.bps = audio->bps;
                                s->audio_frame.sample_rate = audio->sample_rate;
                        }
                        pthread_mutex_unlock(&s->lock);
                }
        }

        vidcap_done(device);

        return NULL;
}

static bool get_slave_param_from_file(FILE* config_file, const char *slave_name, int *x, int *y,
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

static bool get_device_config_from_file(FILE* config_file, char *slave_name,
                char *device_name_config)
{
        char *ret;
        char line[1024];
        fseek(config_file, 0, SEEK_SET); // rewind
        ret = fgets(line, sizeof(line), config_file);  // skip first line
        if(!ret) return false;
        while (fgets(line, sizeof(line), config_file)) {
                char name[128];
                char dev_config[128];
                int x_, y_, width_, height_;
                if(sscanf(line, "%128s %d %d %d %d %128s", name, &x_, &y_, &width_, &height_, dev_config) != 6)
                        continue;
                if(strcasecmp(name, slave_name) == 0) {
                        strcpy(device_name_config, dev_config);
                        return true;
                }
        }
        return false;
}

#define PARSE_OK 0
#define PARSE_ERROR 1
#define PARSE_FILE 2
static int parse_config_string(const char *fmt, unsigned int *width,
                unsigned int *height, double *fps,
        codec_t *color_spec, interpolation_t *interpolation, char **bicubic_algo, interlacing_t *interl, int *grid_x, int *grid_y)
{
        char *save_ptr = NULL;
        char *item;
        char *parse_string;
        char *tmp;
        int token_nr = 0;

        *interl = PROGRESSIVE;

        parse_string = (char *) alloca(strlen(fmt) + 1);
        strcpy(parse_string, fmt);
        tmp = parse_string;

        while((item = strtok_r(tmp, ":", &save_ptr))) {
                switch (token_nr) {
                        case 0:
                                if(strcasecmp(item, "file") == 0) {
                                        return PARSE_FILE;
                                }
                                *width = atoi(item);
                                break;
                        case 1:
                                *height = atoi(item);
                                break;
                        case 2:
                                {
                                        char *endptr;
                                        *fps = strtod(item, &endptr);
                                        if(tolower(*endptr) == 'i') {
                                                *fps /= 2;
                                                *interl = INTERLACED_MERGED;
                                        }
                                }
                                break;
                        case 3:
                                *color_spec = get_codec_from_name(item);
                                if (*color_spec == VIDEO_CODEC_NONE) {
                                        fprintf(stderr, "Unrecognized color spec string: %s\n", item);
                                        return PARSE_ERROR;
                                }
                                break;
                        default:
                                if(strncasecmp(item, "interpolation=", strlen("interpolation=")) == 0) {
                                        if(strcasecmp(item + strlen("interpolation="), "bilinear") == 0) {
                                                *interpolation = BILINEAR;
                                        } else {
                                                *interpolation = BICUBIC;
                                                if(strchr(item, ',')) {
                                                        *bicubic_algo = strdup(strchr(item, ',') + 1);
                                                }
                                        }
                                } else if (strncasecmp(item, "layout=", strlen("layout=")) == 0) {
                                        const char *l = item + strlen("layout=");
                                        if (!strchr(l, 'x')) {
                                                log_msg(LOG_LEVEL_ERROR, "Error parsing layout!\n");
                                                return PARSE_ERROR;
                                        }
                                        *grid_x = atoi(l);
                                        *grid_y = atoi(strchr(l, 'x') + 1);
                                        if (*grid_x <= 0 || *grid_y <= 0) {
                                                log_msg(LOG_LEVEL_ERROR, "Error parsing layout!\n");
                                                return PARSE_ERROR;
                                        }
                                } else {
                                        log_msg(LOG_LEVEL_ERROR, "Unknown option: %s\n", item);
                                        return PARSE_ERROR;
                                }
                }
                tmp = NULL;
                token_nr += 1;
        }

        if(token_nr < 3)
                return PARSE_ERROR;

        return PARSE_OK;
}

static bool parse(struct vidcap_swmix_state *s, struct video_desc *desc, char *fmt,
                FILE **config_file, interpolation_t *interpolation,
                const struct vidcap_params *params)
{
        *config_file = NULL;
        int ret;

        ret = parse_config_string(fmt, &desc->width, &desc->height, &desc->fps, &desc->color_spec,
                        interpolation, &s->bicubic_algo, &desc->interlacing, &s->grid_x, &s->grid_y);
        if(ret == PARSE_ERROR) {
                show_help();
                return false;
        }

        if(ret == PARSE_FILE) {
                s->use_config_file = true;

                *config_file = fopen(get_config_name(), "r");
                if(!*config_file) {
                        fprintf(stderr, "Params not set and config file %s not found.\n",
                                        get_config_name());
                        return false;
                }
                char line[1024];
                if(!fgets(line, sizeof(line), *config_file)) {
                        fprintf(stderr, "Input file is empty!\n");
                        return false;
                }
                while(isspace(line[strlen(line) - 1])) line[strlen(line) - 1] = '\0'; // trim trailing spaces
                ret = parse_config_string(line, &desc->width, &desc->height, &desc->fps, &desc->color_spec,
                                interpolation, &s->bicubic_algo, &desc->interlacing, &s->grid_x, &s->grid_y);
                if(ret != PARSE_OK) {
                        fprintf(stderr, "Malformed input file! First line should contain config "
                                        "string same as for cmdline use (between first ':' and '#' "
                                        "exclusive):\n");
                        show_help();
                        return false;
                }
        } else {
                s->use_config_file = false;
        }


        if(desc->color_spec != RGBA && desc->color_spec != RGB && desc->color_spec != UYVY) {
                fprintf(stderr, "Unsupported output codec.\n");
                return false;
        }

        s->devices_cnt = 0;
        const struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
                if (vidcap_params_get_driver(tmp) != NULL)
                        s->devices_cnt++;
                else
                        break;
        }

        if (s->grid_x != 0 && s->devices_cnt > s->grid_x * s->grid_y) {
                log_msg(LOG_LEVEL_ERROR, "[swmix] Invalid layout! More devices given than layout size.\n");
                return false;
        }

        s->slaves = (struct state_slave *) calloc(s->devices_cnt, sizeof(struct state_slave));

        for (int i = 0; i < s->devices_cnt; ++i) {
                s->slaves[i].audio_frame.max_size = MAX_AUDIO_LEN;
                s->slaves[i].audio_frame.data = (char *)
                        malloc(s->slaves[i].audio_frame.max_size);
                s->slaves[i].audio_frame.data_len = 0;
        }

        tmp = params;
        for (int i = 0; i < s->devices_cnt; ++i) {
                tmp = vidcap_params_get_next(tmp);
                s->slaves[i].device_params = vidcap_params_copy(tmp);
        }

        return true;
}

static int
vidcap_swmix_init(const struct vidcap_params *params, void **state)
{
	struct vidcap_swmix_state *s;
        struct video_desc desc;
        GLenum format;

	printf("vidcap_swmix_init\n");

        if(!vidcap_params_get_fmt(params) ||
                        strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        s = new vidcap_swmix_state();
	if(s == NULL) {
		printf("Unable to allocate swmix capture state\n");
		return VIDCAP_INIT_FAIL;
	}

        s->completed_audio_buffer = NULL;
        s->network_audio_buffer = NULL;

        s->frames = 0;
        s->slaves = NULL;
        s->audio_device_index = -1;
        s->bicubic_algo = strdup("BSpline");
        gettimeofday(&s->t0, NULL);

        memset(&desc, 0, sizeof(desc));
        desc.tile_count = 1;
        desc.color_spec = RGBA;

        s->interpolation = BICUBIC;
        FILE *config_file = NULL;

        char *init_fmt = strdup(vidcap_params_get_fmt(params));
        if(!parse(s, &desc, init_fmt, &config_file, &s->interpolation, params)) {
                free(init_fmt);
                goto error;
        }
        free(init_fmt);

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

        if (s->gl_context.gl_major < 2) {
                fprintf(stderr, "[swmix] Unsufficient OpenGL version to run SWMix.\n");
                goto error;
        }

        gl_context_make_current(&s->gl_context);

        {
                char *bicubic = strdup(bicubic_template);
                char *algo_pos;
                while((algo_pos = strstr(bicubic, "INTERP_ALGORITHM_PLACEHOLDER"))) {
                        memset(algo_pos, ' ', strlen("INTERP_ALGORITHM_PLACEHOLDER"));
                        memcpy(algo_pos, s->bicubic_algo, strlen(s->bicubic_algo));
                }
                printf("Using bicubic algorithm: %s\n", s->bicubic_algo);
                s->bicubic_program = glsl_compile_link(vprogram, bicubic);
                free(bicubic);
        }

        s->slaves_data = init_slave_data(s, config_file);
        if(!s->slaves_data) {
                free(config_file);
                delete s;
                return VIDCAP_INIT_FAIL;
        }

        if (config_file) {
                fclose(config_file);
                config_file = nullptr;
        }

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

        *state = s;
        return VIDCAP_INIT_OK;

error:
        if (config_file) {
                fclose(config_file);
        }
        if(s->slaves) {
                free(s->slaves);
        }
        delete s;
        return VIDCAP_INIT_FAIL;
}

static void
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
                pthread_mutex_destroy(&s->slaves[i].lock);
                VIDEO_FRAME_DISPOSE(s->slaves[i].captured_frame);
                VIDEO_FRAME_DISPOSE(s->slaves[i].done_frame);
                free(s->slaves[i].audio_frame.data);
                vidcap_params_free_struct(s->slaves[i].device_params);
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

        free(s->bicubic_algo);

        delete s;
}

static struct video_frame *
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
        if(s->network_audio_buffer) {
                free(s->network_audio_buffer);
                s->network_audio_buffer = NULL;
        }
        s->network_buffer = s->completed_buffer;
        s->completed_buffer = NULL;
        s->network_audio_buffer = s->completed_audio_buffer;
        s->completed_audio_buffer = NULL;
        s->audio.data_len = s->completed_audio_buffer_len;
        pthread_cond_signal(&s->frame_sent_cv);
        pthread_mutex_unlock(&s->lock);

        s->frame->tiles[0].data = s->network_buffer;

        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            log_msg(LOG_LEVEL_INFO, "[swmix cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }

        if(s->network_audio_buffer) {
                s->audio.data = s->network_audio_buffer;
                *audio = &s->audio;
        }

	return s->frame;
}

static const struct video_capture_info vidcap_swmix_info = {
        vidcap_swmix_probe,
        vidcap_swmix_init,
        vidcap_swmix_done,
        vidcap_swmix_grab,
};

REGISTER_MODULE(swmix, &vidcap_swmix_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


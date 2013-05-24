/*
 * FILE:    video_display/gl.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "host.h"

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

#include "gl_context.h"

#include <signal.h>
#include <assert.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "compat/platform_semaphore.h"
#include <unistd.h>
#include "debug.h"
#include "video_display.h"
#include "video_display/gl.h"
#include "tv.h"

#define MAGIC_GL	DISPLAY_GL_ID
#define WIN_NAME        "Ultragrid - OpenGL Display"

#define STRINGIFY(A) #A

static volatile bool should_exit_main_loop = false;

// source code for a shader unit (xsedmik)
static char * yuv422_to_rgb_fp = STRINGIFY(
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

/* DXT YUV (FastDXT) related */
static char *fp_display_dxt1 = STRINGIFY(
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

static char * vert = STRINGIFY(
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

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

struct state_gl {
        GLuint          PHandle_uyvy, PHandle_dxt, PHandle_dxt5;

        // Framebuffer
        GLuint fbo_id;
	GLuint		texture_display;
	GLuint		texture_uyvy;

        /* Thread related information follows... */
        pthread_t	thread_id;
	volatile int    new_frame;

        /* For debugging... */
        uint32_t	magic;

        int             window;

	unsigned	fs:1;
        unsigned        deinterlace:1;

        struct video_frame *frame;
        struct tile     *tile;
        char            *buffers[2];
        volatile int              image_display;
        volatile unsigned    needs_reconfigure:1;
        pthread_mutex_t lock;
        pthread_cond_t  reconf_cv;

	bool            processed;
        pthread_cond_t  processed_cv;

        double          aspect;
        double          video_aspect;
        unsigned long int frames;
        
        int             dxt_height;
        unsigned int    double_buf;

        struct timeval  tv;

        bool            sync_on_vblank;
        bool            paused;
};

static struct state_gl *gl;

/* Prototyping */
static void gl_draw(double ratio);
static void gl_show_help(void);

static void gl_resize(int width, int height);
static void gl_bind_texture(void *args);
static void gl_reconfigure_screen(struct state_gl *s);
static void glut_idle_callback(void);
static void glut_key_callback(unsigned char key, int x, int y);
static void glut_close_callback(void);
static void glut_resize_window(struct state_gl *s);
static void display_gl_enable_sync_on_vblank(void);
static void screenshot(struct state_gl *s);

#ifdef HAVE_MACOSX
void NSApplicationLoad(void);
#endif

/**
 * Show help
 * @since 23-03-2010, xsedmik
 */
static void gl_show_help(void) {
        printf("GL options:\n");
        printf("\t-d gl[:d|:fs|:aspect=<v>/<h>|:single]* | help\n\n");
        printf("\t\td\t\tdeinterlace\n");
        printf("\t\tfs\t\tfullscreen\n");
        printf("\t\nnovsync\t\tdo not turn sync on VBlank\n");
        printf("\t\taspect=<w>/<h>\trequested video aspect (eg. 16/9). Leave unset if PAR = 1.\n");
}

void * display_gl_init(char *fmt, unsigned int flags) {
        UNUSED(flags);
	struct state_gl        *s;
#if defined HAVE_LINUX || defined WIN32
        GLenum err;
#endif // HAVE_LINUX
        
	s = (struct state_gl *) calloc(1,sizeof(struct state_gl));
	s->magic   = MAGIC_GL;
        
        /* GLUT callbacks take only some arguments so we need static variable */
        gl = s;
        s->window = -1;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->reconf_cv, NULL);

        s->paused = false;
        s->fs = FALSE;
        s->deinterlace = FALSE;
        s->video_aspect = 0.0;
        s->image_display = 0;

        pthread_cond_init(&s->processed_cv, NULL);
        s->processed  = false;
        s->double_buf = TRUE;

        s->sync_on_vblank = true;

	// parse parameters
	if (fmt != NULL) {
		if (strcmp(fmt, "help") == 0) {
			gl_show_help();
			free(s);
			return &display_init_noerr;
		}

		char *tmp = strdup(fmt);
		char *tok, *save_ptr = NULL;
		
		while((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                        if(!strcmp(tok, "d")) {
                                s->deinterlace = TRUE;
                        } else if(!strcmp(tok, "fs")) {
                                s->fs = TRUE;
                        } else if(!strncmp(tok, "aspect=", strlen("aspect="))) {
                                s->video_aspect = atof(tok + strlen("aspect="));
                                char *pos = strchr(tok,'/');
                                if(pos) s->video_aspect /= atof(pos + 1);
                        } else if(!strcmp(tok, "single")) {
                                s->double_buf = FALSE;
                        } else if(!strcasecmp(tok, "novsync")) {
                                s->sync_on_vblank = false;
                        } else {
                                fprintf(stderr, "[GL] Unknown option: %s\n", tok);
                        }
                        tmp = NULL;
                }

		free(tmp);
	}

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->buffers[0] = NULL;
        s->buffers[1] = NULL;
        
        s->frames = 0ul;
        gettimeofday(&s->tv, NULL);

        fprintf(stdout,"GL setup: fullscreen: %s, deinterlace: %s\n",
                        s->fs ? "ON" : "OFF", s->deinterlace ? "ON" : "OFF");

	s->new_frame = 0;

        char *tmp, *gl_ver_major;
        char *save_ptr = NULL;

#ifdef HAVE_LINUX
        x11_enter_thread();
#endif

        glutInit(&uv_argc, uv_argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
        /* Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif

#ifdef HAVE_LINUX
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif
        glutIdleFunc(glut_idle_callback);
	s->window = glutCreateWindow(WIN_NAME);
        glutSetCursor(GLUT_CURSOR_NONE);
        glutHideWindow();
	glutKeyboardFunc(glut_key_callback);
	glutDisplayFunc(glutSwapBuffers);
#ifdef HAVE_MACOSX
        glutWMCloseFunc(glut_close_callback);
#elif HAVE_LINUX
        glutCloseFunc(glut_close_callback);
#endif
	glutReshapeFunc(gl_resize);

        tmp = strdup((const char *)glGetString(GL_VERSION));
        gl_ver_major = strtok_r(tmp, ".", &save_ptr);
        if(atoi(gl_ver_major) >= 2) {
                fprintf(stderr, "OpenGL 2.0 is supported...\n");
        } else {
                fprintf(stderr, "ERROR: OpenGL 2.0 is not supported, try updating your drivers...\n");
                free(tmp);
                goto error; 
        }
        free(tmp);

#if defined HAVE_LINUX || defined WIN32
        err = glewInit();
        if (GLEW_OK != err)
        {
                /* Problem: glewInit failed, something is seriously wrong. */
                fprintf(stderr, "GLEW Error: %d\n", err);
                goto error;
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

        return (void*)s;

error:
        free(s);
        return NULL;
}

/*
 * This function will be probably runned from another thread than GL-thread so
 * we cannot reconfigure directly there. Instead, we post a request to do it
 * inside appropriate thread and make changes we can do. The rest does
 * gl_reconfigure_screen.
 */
int display_gl_reconfigure(void *state, struct video_desc desc)
{
        struct state_gl	*s = (struct state_gl *) state;

        assert (desc.color_spec == RGBA ||
                        desc.color_spec == RGB  ||
                        desc.color_spec == UYVY ||
                        desc.color_spec == DXT1 ||
                        desc.color_spec == DXT1_YUV ||
                        desc.color_spec == DXT5);

        s->tile->width = desc.width;
        s->tile->height = desc.height;
        s->frame->fps = desc.fps;
        s->frame->interlacing = desc.interlacing;
        s->frame->color_spec = desc.color_spec;

        pthread_mutex_lock(&s->lock);
        s->needs_reconfigure = TRUE;
        s->new_frame = 1;

        while(s->needs_reconfigure)
                pthread_cond_wait(&s->reconf_cv, &s->lock);
        s->processed = TRUE;
        pthread_cond_signal(&s->processed_cv);
        pthread_mutex_unlock(&s->lock);

        return TRUE;
}

static void glut_resize_window(struct state_gl *s)
{
        if (!s->fs) {
                glutReshapeWindow(s->tile->height * s->aspect, s->tile->height);
        } else {
                glutFullScreen();
        }
}

/*
 * Please note, that setting the value of 0 to GLX function is invalid according to
 * documentation. However. reportedly NVidia driver does unset VSync.
 */
static void display_gl_enable_sync_on_vblank() {
#ifdef HAVE_MACOSX
        int swap_interval = 1;
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
                glXSwapIntervalSGIProc(1);
        } else {
                fprintf(stderr, "[GL display] GLX_SGI_swap_control is presumably not supported. Unable to set sync-on-VBlank.\n");
        }
#endif
}

static void screenshot(struct state_gl *s)
{
        unsigned char *data = NULL, *tmp = NULL;
        int len = s->tile->width * s->tile->height * 3;
        if (s->frame->color_spec == RGB) {
                data = (unsigned char *) s->buffers[s->image_display];
        } else {
                data = tmp = (unsigned char *) malloc(len);
                if (s->frame->color_spec == UYVY) {
                        vc_copylineUYVYtoRGB(data, (const unsigned char *)
                                        s->buffers[s->image_display], len);
                } else if (s->frame->color_spec == RGBA) {
                        vc_copylineRGBAtoRGB(data, (const unsigned char *)
                                        s->buffers[s->image_display], len);
                }
        }

        if(!data) {
                return;
        }

        char name[128];
        time_t t;
        struct tm time_tmp;

        t = time(NULL);
        localtime_r(&t, &time_tmp);

        strftime(name, sizeof(name), "screenshot-%a, %d %b %Y %T %z.pnm",
                                               &time_tmp);
        FILE *out = fopen(name, "w");
        if(out) {
                fprintf(out, "P6\n%d %d\n255\n", s->tile->width, s->tile->height);
                fwrite(data, 1, len, out);
                fclose(out);
        }
        free(tmp);
}

/**
 * This function must be called only from GL thread 
 * (display_thread_gl) !!!
 */
void gl_reconfigure_screen(struct state_gl *s)
{
        assert(s->magic == MAGIC_GL);

        free(s->buffers[0]);
        free(s->buffers[1]);

        if(s->frame->color_spec == DXT1 || s->frame->color_spec == DXT1_YUV || s->frame->color_spec == DXT5) {
                s->dxt_height = (s->tile->height + 3) / 4 * 4;
                s->tile->data_len = vc_get_linesize((s->tile->width + 3) / 4 * 4, s->frame->color_spec)
                        * s->dxt_height;
        } else {
                s->dxt_height = s->tile->height;
                s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec)
                        * s->tile->height;
        }

        s->buffers[0] = (char *) malloc(s->tile->data_len);
        s->buffers[1] = (char *) malloc(s->tile->data_len);

        asm("emms\n");
        if(!s->video_aspect)
                s->aspect = (double) s->tile->width / s->tile->height;
        else
                s->aspect = s->video_aspect;

        fprintf(stdout,"Setting GL window size %dx%d (%dx%d).\n", (int)(s->aspect * s->tile->height),
                        s->tile->height, s->tile->width, s->tile->height);
        glutShowWindow();
        glut_resize_window(s);

        glUseProgram(0);

        gl_check_error();

        if(s->frame->color_spec == DXT1 || s->frame->color_spec == DXT1_YUV) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                (s->tile->width + 3) / 4 * 4, s->dxt_height, 0,
                                ((s->tile->width + 3) / 4 * 4* s->dxt_height)/2,
                                /* passing NULL here isn't allowed and some rigid implementations
                                 * will crash here. We just pass some egliable buffer to fulfil
                                 * this requirement. glCompressedSubTexImage2D works as expected.
                                 */
                                s->buffers[0]);
                if(s->frame->color_spec == DXT1_YUV) {
                        glBindTexture(GL_TEXTURE_2D,s->texture_display);
                        glUseProgram(s->PHandle_dxt);
                }
        } else if (s->frame->color_spec == UYVY) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_uyvy);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                s->tile->width / 2, s->tile->height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glUseProgram(s->PHandle_uyvy);
                glUniform1i(glGetUniformLocationARB(s->PHandle_uyvy, "image"), 2);
                glUniform1f(glGetUniformLocationARB(s->PHandle_uyvy, "imageWidth"),
                                (GLfloat) s->tile->width);
                glUseProgram(0);
                glActiveTexture(GL_TEXTURE0 + 0);
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                s->tile->width, s->tile->height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
        } else if (s->frame->color_spec == RGBA) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                s->tile->width, s->tile->height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
        } else if (s->frame->color_spec == RGB) {
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                                s->tile->width, s->tile->height, 0,
                                GL_RGB, GL_UNSIGNED_BYTE,
                                NULL);
        } else if (s->frame->color_spec == DXT5) {
                glUseProgram(s->PHandle_dxt5);

                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                (s->tile->width + 3) / 4 * 4, s->dxt_height, 0,
                                (s->tile->width + 3) / 4 * 4 * s->dxt_height,
                                NULL);
        }

        gl_check_error();

        if(s->sync_on_vblank) {
                display_gl_enable_sync_on_vblank();
        }
        gl_check_error();
}

static void glut_idle_callback(void)
{
        struct state_gl *s = gl;
        struct timeval tv;
        double seconds;

        pthread_mutex_lock(&s->lock);
        if(!s->new_frame) {
                pthread_mutex_unlock(&s->lock);
                return;
        }
        s->new_frame = 0;

        if (s->needs_reconfigure) {
                /* there has been scheduled request for win reconfiguration */
                gl_reconfigure_screen(s);
                s->needs_reconfigure = FALSE;
                pthread_cond_signal(&s->reconf_cv);
                pthread_mutex_unlock(&s->lock);
                return; /* return after reconfiguration */
        }
        pthread_mutex_unlock(&s->lock);

        /* for DXT, deinterlacing doesn't make sense since it is
         * always deinterlaced before comrpression */
        if(s->deinterlace && (s->frame->color_spec == RGBA || s->frame->color_spec == UYVY))
                vc_deinterlace((unsigned char *) s->buffers[s->image_display],
                                vc_get_linesize(s->tile->width, s->frame->color_spec),
                                s->tile->height);

        gl_check_error();

        if(s->paused)
                goto processed;

        switch(s->frame->color_spec) {
                case DXT1:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        (s->tile->width + 3) / 4 * 4, s->dxt_height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        ((s->tile->width + 3) / 4 * 4 * s->dxt_height)/2,
                                        s->buffers[s->image_display]);
                        break;
                case DXT1_YUV:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        (s->tile->width * s->tile->height/16)*8,
                                        s->buffers[s->image_display]);
                        break;
                case UYVY:
                        gl_bind_texture(s);
                        break;
                case RGBA:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        s->buffers[s->image_display]);
                        break;
                case RGB:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_RGB, GL_UNSIGNED_BYTE,
                                        s->buffers[s->image_display]);
                        break;
                case DXT5:                        
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        (s->tile->width + 3) / 4 * 4, s->dxt_height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                        (s->tile->width + 3) / 4 * 4 * s->dxt_height,
                                        s->buffers[s->image_display]);
                        break;
                default:
                        fprintf(stderr, "[GL] Fatal error - received unsupported codec.\n");
                        exit_uv(128);
                        return;

        }

        gl_check_error();

        /* FPS Data, this is pretty ghetto though.... */
        s->frames++;
        gettimeofday(&tv, NULL);
        seconds = tv_diff(tv, s->tv);

        if (seconds > 5) {
                double fps = s->frames / seconds;
                fprintf(stderr, "[GL] %lu frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->frames = 0;
                s->tv = tv;
        }

        gl_draw(s->aspect);
        glutPostRedisplay();

processed:
        pthread_mutex_lock(&s->lock);
        pthread_cond_signal(&s->processed_cv);
        s->processed = TRUE;
        pthread_mutex_unlock(&s->lock);
}

static void glut_key_callback(unsigned char key, int x, int y)
{
        UNUSED(x);
        UNUSED(y);

        switch(key) {
                case 'f':
                        if(!gl->fs) {
                                gl->fs = TRUE;
                        } else {
                                gl->fs = FALSE;
                        }
                        glut_resize_window(gl);
                        break;
                case 'q':
#if defined FREEGLUT || defined HAVE_MACOSX
                        exit_uv(0);
#else
			glutDestroyWindow(gl->window);
			exit(1);
#endif
                        break;
                case 'd':
                        gl->deinterlace = gl->deinterlace ? FALSE : TRUE;
                        printf("Deinterlacing: %s\n", gl->deinterlace ? "ON" : "OFF");
                        break;
                case ' ':
                        gl->paused = !gl->paused;
                        break;
                case 's':
                        screenshot(gl);
                        break;
        }
}

void display_gl_run(void *arg)
{
        UNUSED(arg);

#if defined HAVE_MACOSX || defined FREEGLUT
        while(!should_exit_main_loop) {
                usleep(1000);
                glut_idle_callback();
#ifndef HAVE_MACOSX
                glutMainLoopEvent();
#else
                glutCheckLoop();
#endif
        }
#else /* defined HAVE_MACOSX || defined FREEGLUT */
	glutMainLoop();
#endif
}


static void gl_resize(int width,int height)
{
        glViewport( 0, 0, ( GLint )width, ( GLint )height );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity( );

        double screen_ratio;
        double x = 1.0,
               y = 1.0;

        debug_msg("Resized to: %dx%d\n", width, height);

        screen_ratio = (double) width / height;
        if(screen_ratio > gl->aspect) {
                x = (double) height * gl->aspect / width;
        } else {
                y = (double) width / (height * gl->aspect);
        }
        glScalef(x, y, 1);

        glOrtho(-1,1,-1/gl->aspect,1/gl->aspect,10,-10);

        glMatrixMode( GL_MODELVIEW );

        glLoadIdentity( );
}

static void gl_bind_texture(void *arg)
{
        struct state_gl        *s = (struct state_gl *) arg;

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

        glViewport( 0, 0, s->tile->width, s->tile->height);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s->tile->width / 2, s->tile->height,  GL_RGBA, GL_UNSIGNED_BYTE, s->buffers[s->image_display]);
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

static void gl_draw(double ratio)
{
        float bottom;
        gl_check_error();

        /* Clear the screen */
        glClear(GL_COLOR_BUFFER_BIT);

        glLoadIdentity( );
        glTranslatef( 0.0f, 0.0f, -1.35f );

        /* Reflect that we may have higher texture than actual data
         * if we use DXT and source height was not divisible by 4 
         * In normal case, there would be 1.0 */
        bottom = 1.0f - (gl->dxt_height - gl->tile->height) / (float) gl->dxt_height * 2;

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
        exit_uv(0);
}

display_type_t *display_gl_probe(void)
{
        display_type_t          *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id          = DISPLAY_GL_ID;
                dt->name        = "gl";
                dt->description = "OpenGL";
        }
        return dt;
}

int display_gl_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1, DXT1_YUV, DXT5};
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
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

void display_gl_done(void *state)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        //pthread_join(s->thread_id, NULL);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->reconf_cv);
        pthread_cond_destroy(&s->processed_cv);
        if(s->window != -1) {
                glutDestroyWindow(s->window);
        }
        glDeleteProgram(s->PHandle_uyvy);
        glDeleteProgram(s->PHandle_dxt);
        glDeleteProgram(s->PHandle_dxt5);
        free(s->buffers[0]);
        free(s->buffers[1]);
        vf_free(s->frame);
        free(s);
}

void display_gl_finish(void *state)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        should_exit_main_loop = true;
        pthread_mutex_lock(&s->lock);
        s->processed = true;
        pthread_cond_signal(&s->processed_cv);
        pthread_mutex_unlock(&s->lock);
}

struct video_frame * display_gl_getf(void *state)
{
        struct state_gl *s = (struct state_gl *) state;
        assert(s->magic == MAGIC_GL);

        if(s->double_buf) {
                s->tile->data = s->buffers[(s->image_display + 1) % 2];
        } else {
                s->tile->data = s->buffers[s->image_display];
        }


        return s->frame;
}

int display_gl_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);
        UNUSED(frame);

        pthread_mutex_lock(&s->lock);
        if(s->double_buf) {
                if(!s->processed && nonblock == PUTF_NONBLOCK) {
                        pthread_mutex_unlock(&s->lock);
                        return EWOULDBLOCK;
                }
                while(!s->processed) 
                        pthread_cond_wait(&s->processed_cv, &s->lock);
                s->processed = false;

                /* ...and give it more to do... */
                s->image_display = (s->image_display + 1) % 2;
        }

        /* ...and signal the worker */
        s->new_frame = 1;
        pthread_mutex_unlock(&s->lock);
        return 0;
}

void display_gl_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

int display_gl_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}



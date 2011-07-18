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

#include "host.h"
#include "config.h"
#include <GL/glew.h>
#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else /* HAVE_MACOSX */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <GL/glut.h>
#endif /* HAVE_MACOSX */
#include <signal.h>
#include <assert.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "compat/platform_semaphore.h"
#include <unistd.h>
#include "debug.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_display.h"
#include "video_display/gl.h"
#include "tv.h"
#include <X11/Xutil.h>
#include <X11/Xatom.h>


#define MAGIC_GL	DISPLAY_GL_ID
#define WIN_NAME        "Ultragrid - OpenGL Display"

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

struct state_gl {
        Display         *display;
        Window          xwindow;
	unsigned int	x_width;
	unsigned int	x_height;

        GLubyte         *buffers[2];
	GLubyte		*y, *u, *v;	//Guess what this might be...
	GLhandleARB     VSHandle,FSHandle,PHandle;

        int             image_display, image_network;
	GLuint		texture[4];

        /* Thread related information follows... */
        pthread_t	thread_id;
        sem_t		semaphore;

        /* For debugging... */
        uint32_t	magic;

        int             window;

	char 		*VProgram,*FProgram;

	unsigned	fs:1;
	unsigned	rgb:1;
        unsigned        dxt:1;
        unsigned        deinterlace:1;

        struct video_frame frame;
        volatile int    needs_reconfigure:1;
        pthread_mutex_t reconf_lock;
        pthread_cond_t  reconf_cv;

        double          raspect;

        unsigned long int frames;
        unsigned        win_initialized:1;

        struct timeval  tv;
};

static struct state_gl *gl;

/* Prototyping */
void gl_draw(double ratio);
void gl_show_help(void);

void display_gl_run(void *arg);
void gl_check_error(void);
inline void getY(GLubyte *input,GLubyte *y, GLubyte *u,GLubyte *v, int w, int h);
void gl_resize_window(int width, int height);
void gl_bind_texture(void *args);
void glsl_gl_init(void *arg);
void glsl_arb_init(void *arg);
void dxt_arb_init(void *arg);
void dxt_bind_texture(void *arg);
void * display_gl_init(char *fmt);

void gl_reconfigure_screen_stub(void *s, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux, struct tile_info tile_info);
void gl_reconfigure_screen_real(struct state_gl *s);
static void cleanup_data(struct state_gl *s);
static void cleanup_gl(struct state_gl *s);
void glut_idle_callback(void);
void glut_redisplay_callback(void);
void glut_key_callback(unsigned char key, int x, int y);

static void get_sub_frame(void *s, int x, int y, int w, int h, struct video_frame *out);

#ifndef HAVE_MACOSX
static void update_fullscreen_state(struct state_gl *s);
/*from wmctrl */
static Window get_window(Display *disp, const char *name);
#endif

/**
 * Show help
 * @since 23-03-2010, xsedmik
 */
void gl_show_help(void) {
        printf("GL options:\n");
        printf("\t[d][fs] | help\n");
        printf("\td - deinterlace\n");
        printf("\tfs - fullscreen\n");
}

void gl_check_error()
{
	GLenum msg;
	int flag=0;
	msg=glGetError();
	while(msg!=GL_NO_ERROR) {
		flag=1;
		switch(msg){
			case GL_INVALID_ENUM:
				fprintf(stderr, "GL_INVALID_ENUM\n");
				break;
			case GL_INVALID_VALUE:
				fprintf(stderr, "GL_INVALID_VALUE\n");
				break;
			case GL_INVALID_OPERATION:
				fprintf(stderr, "GL_INVALID_OPERATION\n");
				break;
			case GL_STACK_OVERFLOW:
				fprintf(stderr, "GL_STACK_OVERFLOW\n");
				break;
			case GL_STACK_UNDERFLOW:
				fprintf(stderr, "GL_STACK_UNDERFLOW\n");
				break;
			case GL_OUT_OF_MEMORY:
				fprintf(stderr, "GL_OUT_OF_MEMORY\n");
				break;
			default:
				fprintf(stderr, "wft mate? Unknown GL ERROR: %d\n", msg);
				break;
		}
		msg=glGetError();
	}
	if(flag)
		exit(1);
}

void * display_gl_init(char *fmt) {

	struct state_gl        *s;

        glutInit(&uv_argc, uv_argv);
	s = (struct state_gl *) calloc(1,sizeof(struct state_gl));
	s->magic   = MAGIC_GL;
        /* GLUT callback don't take any arguments */
        gl = s;

        s->window = -1;

        pthread_mutex_init(&s->reconf_lock, NULL);
        pthread_cond_init(&s->reconf_cv, NULL);

        s->fs = FALSE;
        s->deinterlace = FALSE;

	// parse parameters
	if (fmt != NULL) {
		if (strcmp(fmt, "help") == 0) {
			gl_show_help();
			free(s);
			return NULL;
		}

		char *tmp = strdup(fmt);
		char *tok;
		
		tok = strtok(tmp, ":");
		if ((tok != NULL) && (tok[0] == 'd')){
                        s->deinterlace = TRUE;
                        tok = strtok(NULL,":");
                }
		if ((tok != NULL) && (tok[0] == 'f') && (tok[1] == 's')) {
			s->fs=TRUE;
		}

		free(tmp);
	}

        s->frame.reconfigure = (reconfigure_t) gl_reconfigure_screen_stub;
        s->frame.get_sub_frame = (get_sub_frame_t) get_sub_frame;
        s->frame.state = s;
        s->win_initialized = FALSE;

        s->frames = 0ul;
        gettimeofday(&s->tv, NULL);

        fprintf(stdout,"GL setup: %dx%d, fullscreen: %s, deinterlace: %s\n", s->frame.width, s->frame.height,
                        s->fs ? "ON" : "OFF", s->deinterlace ? "ON" : "OFF");

        platform_sem_init(&s->semaphore, 0, 0);
	/*if (pthread_create(&(s->thread_id), NULL, display_thread_gl, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}*/

#ifndef HAVE_MACOSX
        s->display = XOpenDisplay(NULL);
#endif

	return (void*)s;
}

void glsl_arb_init(void *arg)
{
    struct state_gl	*s = (struct state_gl *) arg;
    char 		*log;

    /* Set up program objects. */
    s->PHandle=glCreateProgramObjectARB();
    // s->VSHandle=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    s->FSHandle=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

    /* Compile Shader */
    assert(s->FProgram!=NULL);
    // assert(s->VProgram!=NULL);
    glShaderSourceARB(s->FSHandle,1,(const GLcharARB**) &s->FProgram,NULL);
    glCompileShaderARB(s->FSHandle);

    /* Print compile log */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->FSHandle,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);
#if 0
    glShaderSourceARB(s->VSHandle,1,&(s->VProgram),NULL);
    glCompileShaderARB(s->VSHandle);
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->VSHandle,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);
#endif
    /* Attach and link our program */
    glAttachObjectARB(s->PHandle,s->FSHandle);
    // glAttachObjectARB(s->PHandle,s->VSHandle);
    glLinkProgramARB(s->PHandle);

    /* Print link log. */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->PHandle,32768,NULL,log);
    printf("Link Log: %s\n", log);
    free(log);

    /* Finally, use the program. */
    glUseProgramObjectARB(s->PHandle);
}

void dxt_arb_init(void *arg)
{
    struct state_gl        *s = (struct state_gl *) arg;
    char *log;
    /* Set up program objects. */
    s->PHandle=glCreateProgramObjectARB();
    s->FSHandle=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    s->VSHandle=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);

    /* Compile Shader */
    assert(s->FProgram!=NULL);
    glShaderSourceARB(s->FSHandle,1,(const GLcharARB**)&(s->FProgram),NULL);
    glCompileShaderARB(s->FSHandle);
    glShaderSourceARB(s->VSHandle,1,(const GLcharARB**)&(s->VProgram),NULL);
    glCompileShaderARB(s->VSHandle);

    /* Print compile log */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->FSHandle,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->VSHandle,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);

    /* Attach and link our program */
    glAttachObjectARB(s->PHandle,s->FSHandle);
    glAttachObjectARB(s->PHandle,s->VSHandle);
    glLinkProgramARB(s->PHandle);

    /* Print link log. */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->PHandle,32768,NULL,log);
    printf("Link Log: %s\n", log);
    free(log);

    /* Finally, use the program. */
    glUseProgramObjectARB(s->PHandle);
}

void glsl_gl_init(void *arg) {

	//TODO: Add log
	struct state_gl	*s = (struct state_gl *) arg;

	s->PHandle=glCreateProgram();
	s->FSHandle=glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(s->FSHandle,1,(const GLcharARB**)&(s->FProgram),NULL);
	glCompileShader(s->FSHandle);

	glAttachShader(s->PHandle,s->FSHandle);

	glLinkProgram(s->PHandle);
	glUseProgram(s->PHandle);
}


inline void getY(GLubyte *input,GLubyte *y, GLubyte *u,GLubyte *v, int w, int h)
{
	//TODO: This should be re-written in assembly
	//Note: We assume 1920x1080 UYVY (YUV 4:2:2)
	//See http://www.fourcc.org/indexyuv.htm for more info
	//0x59565955 - UYVY - 16 bits per pixel (2 bytes per plane)
	register int x;
	for(x = 0; x < w * h * 2; x += 4) {
		*u++=input[x];		//1	5	9	13
		*y++=input[x+1];	//2	6	10	14
		*v++=input[x+2];	//3	7	11	15
		*y++=input[x+3];	//0	4	8	12
	}
}

/*
 * This function will be probably runned from another thread than GL-thread so
 * we cannot reconfigure directly there. Instead, we post a request to do it
 * inside appropriate thread and make changes we can do. The rest does *_real.
 */
void gl_reconfigure_screen_stub(void *arg, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux, struct tile_info tile_info)
{
        struct state_gl	*s = (struct state_gl *) arg;
        int i;

        UNUSED(tile_info);

        assert(s->magic == MAGIC_GL);

        pthread_mutex_lock(&s->reconf_lock);
        if(s->win_initialized)
                cleanup_data(s);

        s->frame.width = width;
        s->frame.height = height;
        s->frame.fps = fps;
        s->frame.aux = aux;
        s->frame.color_spec = codec;
        s->frame.dst_x_offset = 0;

        for(i = 0; codec_info[i].name != NULL; ++i) {
                if(codec == codec_info[i].codec) {
                        s->rgb = codec_info[i].rgb;
                        s->frame.src_bpp = codec_info[i].bpp;
                }
        }
        s->dxt = FALSE;

        switch (codec) {
                case R10k:
                        s->frame.decoder = (decoder_t)vc_copyliner10k;
                        s->frame.dst_bpp = get_bpp(RGBA);
                        break;
                case RGBA:
                        s->frame.decoder = (decoder_t)memcpy; /* or vc_copylineRGBA?
                                                                 but we have default
                                                                 {r,g,b}shift */
                        
                        s->frame.dst_bpp = get_bpp(RGBA);
                        break;
                case v210:
                        s->frame.decoder = (decoder_t)vc_copylinev210;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        break;
                case DVS10:
                        s->frame.decoder = (decoder_t)vc_copylineDVS10;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        break;
                case Vuy2:
                case DVS8:
                case UYVY:
                        s->frame.decoder = (decoder_t)memcpy;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        break;
                case DXT1:
                        s->dxt = TRUE;
                        if(s->frame.aux & AUX_RGB)
                                s->rgb = TRUE;
                        else
                                s->rgb = FALSE;
                        s->frame.decoder = (decoder_t)memcpy;
                        s->frame.dst_bpp = get_bpp(DXT1);
                        break;
        }

        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;
        s->frame.data_len = s->frame.dst_linesize * s->frame.height;

        s->buffers[0] = malloc(s->frame.data_len);
        s->buffers[1] = malloc(s->frame.data_len);
	s->image_network = 0;
	s->image_display = 1;
	s->y=malloc(s->frame.width * s->frame.height);
	s->u=malloc(s->frame.width * s->frame.height);
	s->v=malloc(s->frame.width * s->frame.height);

        s->frame.data = (char *) s->buffers[s->image_network];

	asm("emms\n");

        s->needs_reconfigure = TRUE;
        platform_sem_post(&s->semaphore);

        while(s->needs_reconfigure)
                pthread_cond_wait(&s->reconf_cv, &s->reconf_lock);
        pthread_mutex_unlock(&s->reconf_lock);
}

/**
 * This function must be called only from GL thread 
 * (display_thread_gl) !!!
 */
void gl_reconfigure_screen_real(struct state_gl *s)
{
        int old_win;

        if(s->win_initialized)
                cleanup_gl(s);
        s->raspect = (double) s->frame.height / s->frame.width;

	fprintf(stdout,"Setting GL window size %dx%d.\n", s->frame.width, s->frame.height);

        glutInitWindowSize(s->frame.width, s->frame.height);
        old_win = s->window;
        s->xwindow = 0; /* we do not have X identificator yet */

        s->window = glutCreateWindow(WIN_NAME);
        glutKeyboardFunc(glut_key_callback);
        glutDisplayFunc(glut_redisplay_callback);

        if(old_win != -1)
                glutDestroyWindow(old_win);

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

        if(s->dxt) {
                if(s->rgb) {
                        glEnable( GL_TEXTURE_2D );
                        glGenTextures(1, s->texture);
                        // Bind the texture object
                        glBindTexture(GL_TEXTURE_2D, s->texture[0]);
                        // Set the texture's stretching properties
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glewInit();
                } else {
                        glEnable( GL_TEXTURE_2D );
                        s->FProgram = frag;
                        s->VProgram = vert;

                        glGenTextures(1, s->texture);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                        glewInit();
                        if(glewIsSupported("GL_VERSION_2_0")){
                                fprintf(stderr, "OpenGL 2.0 is supported...\n");
                                //TODO: Re-enable gl_init!
                                //glsl_gl_init(s);
                                dxt_arb_init(s);
                        }else if(GLEW_ARB_fragment_shader){
                                fprintf(stderr, "OpenGL 2.0 not supported, using ARB extension...\n");
                                dxt_arb_init(s);
                        }else{
                                fprintf(stderr, "ERROR: Neither OpenGL 1.0 nor ARB_fragment_shader are supported, try updating your drivers...\n");
                                exit(65);
                        }
                }
        } else {
                if(s->rgb) {
                        glEnable( GL_TEXTURE_2D );
                        glGenTextures(1, s->texture);
                        // Bind the texture object
                        glBindTexture(GL_TEXTURE_2D, s->texture[0]);
                        // Set the texture's stretching properties
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                } else {
                        glEnable( GL_TEXTURE_2D );
                        /* OpenGL Setup */
                        glGenTextures(4, s->texture);	//TODO: Is this necessary?

                        //load shader (xsedmik, 13-02-2010)
                        s->FProgram = glsl;
                        /* Check to see if OpenGL 2.0 is supported, if not use ARB (if supported) */
                        glewInit();
                        if(glewIsSupported("GL_VERSION_2_0")){
                                fprintf(stderr, "OpenGL 2.0 is supported...\n");
                                //TODO: Re-enable gl_init!
                                //glsl_gl_init(s);
                                glsl_arb_init(s);
                        }else if(GLEW_ARB_fragment_shader){
                                fprintf(stderr, "OpenGL 2.0 not supported, using ARB extension...\n");
                                glsl_arb_init(s);
                        }else{
                                fprintf(stderr, "ERROR: Neither OpenGL 1.0 nor ARB_fragment_shader are supported, try updating your drivers...\n");
                                exit(65);
                        }
                }
        }

        if (s->fs) {
                glutFullScreen();
                gl_resize_window(s->x_width, s->x_height);
        }
        else {
                glutReshapeWindow(gl->frame.width, gl->frame.height);
                gl_resize_window(s->frame.width, s->frame.height);
        }

#ifndef HAVE_MACOSX
        glutSwapBuffers();
        while ((s->xwindow = get_window(s->display, WIN_NAME)) == 0) {
                usleep(1000); /* wait for window init */
}

        update_fullscreen_state(s);
#endif
        s->win_initialized = TRUE;
}

void glut_idle_callback(void)
{
        struct state_gl *s = gl;
        struct timeval tv;
        double seconds;
        //display_gl_handle_events(s);
        
        if(should_exit) return;
        platform_sem_wait(&s->semaphore);
        if(should_exit) return;

        pthread_mutex_lock(&s->reconf_lock);
        if (s->needs_reconfigure) {
                gl_reconfigure_screen_real(s);
                s->needs_reconfigure = FALSE;
                pthread_cond_signal(&s->reconf_cv);
        }
        pthread_mutex_unlock(&s->reconf_lock);

        /* for DXT deinterlacing doesn't make sense since it is
         * always deinterlaced before comrpression */
        if(s->deinterlace && !s->dxt)
                vc_deinterlace(s->buffers[s->image_display],
                                s->frame.dst_linesize, s->frame.height);

        if(s->dxt) {
                if(s->rgb) {
                        glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        s->frame.width, s->frame.height, 0,
                                        (s->frame.width*s->frame.height/16)*8,
                                        s->buffers[s->image_display]);
                } else {
                        dxt_bind_texture(s);
                }
        } else {
                if(!s->rgb)
                {
                        getY(s->buffers[s->image_display], s->y, s->u, s->v,
                                        s->frame.width, s->frame.height);
                        gl_bind_texture(s);
                } else {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                        s->frame.width, s->frame.height, 0,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        s->buffers[s->image_display]);
                }

        }
        /* FPS Data, this is pretty ghetto though.... */
        s->frames++;
        gettimeofday(&tv, NULL);
        seconds = tv_diff(tv, s->tv);

        if (seconds > 5) {
                double fps = s->frames / seconds;
                fprintf(stderr, "%lu frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->frames = 0;
                s->tv = tv;
        }

        gl_draw(s->raspect);
        glutPostRedisplay();
}

void glut_key_callback(unsigned char key, int x, int y) 
{
        UNUSED(x);
        UNUSED(y);

        switch(key) {
                case 'f':
                        if(!gl->fs) {
                                glutFullScreen();
                                gl->fs = TRUE;
                                gl_resize_window(gl->x_width, gl->x_height);
#ifndef HAVE_MACOSX
                                update_fullscreen_state(gl);
#endif
                        } else {
                                glutReshapeWindow(gl->frame.width, gl->frame.height);
                                gl->fs = FALSE;
                                gl_resize_window(gl->frame.width, gl->frame.height);
#ifndef HAVE_MACOSX
                                update_fullscreen_state(gl);
#endif
                        }
                        break;
                case 'q':
                        should_exit = TRUE;
                        platform_sem_post(&gl->semaphore);
                        cleanup_gl(gl);
                        if(gl->window != -1)
                                glutDestroyWindow(gl->window);
                        exit(0);
                        break;
                case 'd':
                        gl->deinterlace = gl->deinterlace ? FALSE : TRUE;
                        printf("Deinterlacing: %s\n", gl->deinterlace ? "ON" : "OFF");
                        break;
        }
}

void display_gl_run(void *arg)
{
        struct state_gl        *s = (struct state_gl *) arg;

#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
        /* Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutIdleFunc(glut_idle_callback);

        s->x_width = glutGet(GLUT_SCREEN_WIDTH);
        s->x_height = glutGet(GLUT_SCREEN_HEIGHT);

        /* Wait until we have some data and thus created window 
         * otherwise the mainLoop would exit immediately */
        while(!s->win_initialized) {
                pthread_mutex_lock(&s->reconf_lock);
                if (s->needs_reconfigure) {
                        gl_reconfigure_screen_real(s);
                        s->needs_reconfigure = FALSE;
                        pthread_cond_signal(&s->reconf_cv);
                }
                pthread_mutex_unlock(&s->reconf_lock);
                usleep(1000);
        }

        glutMainLoop();
}


void gl_resize_window(int width,int height)
{
    glViewport( 0, 0, ( GLint )width, ( GLint )height );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity( );

    if(gl->fs) {
            double screen_ratio;
            double x = 1.0,
                   y = 1.0;

            screen_ratio = (double) gl->x_width / gl->x_height;
            if(screen_ratio > 1.0 / gl->raspect) {
                    x = (double) gl->x_height / (gl->x_width * gl->raspect);
            } else {
                    y = (double) gl->x_width / (gl->x_height / gl->raspect);
            }
            glScalef(x, y, 1);
    } 

    glOrtho(-1,1,-gl->raspect,gl->raspect,10,-10);

    glMatrixMode( GL_MODELVIEW );

    glLoadIdentity( );
}

void gl_bind_texture(void *arg)
{
    struct state_gl        *s = (struct state_gl *) arg;
	int i;

    glActiveTexture(GL_TEXTURE1);
    i=glGetUniformLocationARB(s->PHandle,"Utex");
    glUniform1iARB(i,1); 
    glBindTexture(GL_TEXTURE_2D,1);
    glTexImage2D(GL_TEXTURE_2D,0,1,s->frame.width/2,s->frame.height,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->u);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glActiveTexture(GL_TEXTURE2);
    i=glGetUniformLocationARB(s->PHandle,"Vtex");
    glUniform1iARB(i,2); 
    glBindTexture(GL_TEXTURE_2D,2);
    glTexImage2D(GL_TEXTURE_2D,0,1,s->frame.width/2,s->frame.height,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->v);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glActiveTexture(GL_TEXTURE0);
    i=glGetUniformLocationARB(s->PHandle,"Ytex");
    glUniform1iARB(i,0); 
    glBindTexture(GL_TEXTURE_2D,0);
    glTexImage2D(GL_TEXTURE_2D,0,1,s->frame.width,s->frame.height,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->y);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
#if 0
    //TODO: does OpenGL use different stuff here?
    glActiveTexture(GL_TEXTURE0);
    i=glGetUniformLocationARB(s->PHandle,"yuvtex");
    glUniform1iARB(i,0); 
    glBindTexture(GL_TEXTURE_2D,0);
    glTexImage2D(GL_TEXTURE_2D,0,1,HD_WIDTH,HD_HEIGHT,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->outbuffer);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    gl_check_error();

    glActiveTexture(GL_TEXTURE1);
    i=glGetUniformLocationARB(s->PHandle,"rawtex");
    glUniform1iARB(i,1); 
    glBindTexture(GL_TEXTURE_2D,1);
    glTexImage2D(GL_TEXTURE_2D,0,1,HD_WIDTH,HD_HEIGHT,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->outbuffer+1920*1080);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    gl_check_error();
#endif
}    

void dxt_bind_texture(void *arg)
{
        struct state_gl        *s = (struct state_gl *) arg;
        static int i=0;

        //TODO: does OpenGL use different stuff here?
        glActiveTexture(GL_TEXTURE0);
        i=glGetUniformLocationARB(s->PHandle,"yuvtex");
        glUniform1iARB(i,0); 
        glBindTexture(GL_TEXTURE_2D,0);
        glCompressedTexImage2D(GL_TEXTURE_2D, 0,GL_COMPRESSED_RGB_S3TC_DXT1_EXT,s->frame.width,s->frame.height, 0,(s->frame.width*s->frame.height/16)*8, s->buffers[s->image_display]);

}    

void gl_draw(double ratio)
{
    /* Clear the screen */
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity( );
    glTranslatef( 0.0f, 0.0f, -1.35f );

    gl_check_error();
    glBegin(GL_QUADS);
      /* Front Face */
      /* Bottom Left Of The Texture and Quad */
      glTexCoord2f( 0.0f, 1.0f ); glVertex2f( -1.0f, -ratio);
      /* Bottom Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 1.0f ); glVertex2f(  1.0f, -ratio);
      /* Top Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  ratio);
      /* Top Left Of The Texture and Quad */
      glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  ratio);
    glEnd( );

    gl_check_error();
    /* Draw it to the screen */
}

void glut_redisplay_callback(void)
{
    glFlush();
    glutSwapBuffers();
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

void display_gl_done(void *state)
{
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);

        //pthread_join(s->thread_id, NULL);
        pthread_mutex_destroy(&s->reconf_lock);
        pthread_cond_destroy(&s->reconf_cv);
        cleanup_data(s);
        if(s->window != -1)
                glutDestroyWindow(s->window);
}

struct video_frame * display_gl_getf(void *state)
{
        struct state_gl *s = (struct state_gl *) state;
        assert(s->magic == MAGIC_GL);

        s->frame.data = (char *) s->buffers[s->image_network];
        return &s->frame;
}

int display_gl_putf(void *state, char *frame)
{
        int tmp;
        struct state_gl *s = (struct state_gl *) state;

        assert(s->magic == MAGIC_GL);
        UNUSED(frame);

        /* ...and give it more to do... */
        tmp = s->image_display;
        s->image_display = s->image_network;
                s->image_network = tmp;

                /* ...and signal the worker */
                platform_sem_post(&s->semaphore);
#ifndef HAVE_MACOSX
                /* isn't implemented on Macs */
                sem_getvalue(&s->semaphore, &tmp);
                if(tmp > 1)
                        printf("frame drop!\n");
#endif
                return 0;
}

static void cleanup_gl(struct state_gl *s)
{
        glDisable( GL_TEXTURE_2D );

        if(!s->dxt && s->rgb) {
                glDeleteTextures(4, s->texture);	//TODO: Is this necessary?
        } else { /* other cases */
                glDeleteTextures(1, s->texture);
        }
        if(!s->rgb)
        {
                glDeleteObjectARB(s->FSHandle);
                glDeleteObjectARB(s->VSHandle);
                glDeleteObjectARB(s->PHandle);
        }
}

static void cleanup_data(struct state_gl *s)
{
        //glutHideWindow();
        /* glutDestroyWindow(s->window); cannot be here because mainloop would
         * return without any active window, so wait until new is created and let
         * it destroy this window. */

        free(s->y);
        free(s->u);
        free(s->v);
        free(s->buffers[0]);
        free(s->buffers[1]);
}

#ifndef HAVE_MACOSX
/*
 * Xinerama fullscreen functions
 * parts of code are taken from wmctrl
 */
#define MAX_PROPERTY_VALUE_LEN 4096

static Window *get_client_list (Display *disp, unsigned long *size);
static char *get_window_title (Display *disp, Window win);
static char *get_property (Display *disp, Window win,
Atom xa_prop_type, char *prop_name, unsigned long *size);

static Window get_window(Display *disp, const char *name)
{
        Window activate = 0;
        Window *client_list;
        unsigned long client_list_size;
        int i;

        if ((client_list = get_client_list(disp, &client_list_size)) == NULL) {
            return EXIT_FAILURE; 
        }

        for (i = 0; (unsigned long) i < client_list_size / sizeof(Window); i++) {
                char *match_utf8;
                match_utf8 = get_window_title(disp, client_list[i]); /* UTF8 */
                if (match_utf8) {
                        if (strcmp(name, match_utf8) == 0) {
                            activate = client_list[i];
                            break;
                        }
                        free(match_utf8);
                }
        }
        return activate;
}

static Window *get_client_list (Display *disp, unsigned long *size) {
        Window *client_list;

        if ((client_list = (Window *)get_property(disp, DefaultRootWindow(disp), 
                    XA_WINDOW, "_NET_CLIENT_LIST", size)) == NULL) {
                if ((client_list = (Window *)get_property(disp, DefaultRootWindow(disp), 
                                XA_CARDINAL, "_WIN_CLIENT_LIST", size)) == NULL) {
                    fputs("Cannot get client list properties. \n"
                          "(_NET_CLIENT_LIST or _WIN_CLIENT_LIST)"
                          "\n", stderr);
                    return NULL;
                }
        }

        return client_list;
}

static char *get_window_title (Display *disp, Window win) {
        char *wm_name;

        wm_name = get_property(disp, win, XA_STRING, "WM_NAME", NULL);

        return wm_name;
}

static char *get_property (Display *disp, Window win,
                Atom xa_prop_type, char *prop_name, unsigned long *size) {
        Atom xa_prop_name;
        Atom xa_ret_type;
        int ret_format;
        unsigned long ret_nitems;
        unsigned long ret_bytes_after;
        unsigned long tmp_size;
        unsigned char *ret_prop;
        char *ret;

        xa_prop_name = XInternAtom(disp, prop_name, False);

        /* MAX_PROPERTY_VALUE_LEN / 4 explanation (XGetWindowProperty manpage):
        *
        * long_length = Specifies the length in 32-bit multiples of the
        *               data to be retrieved.
        *
        * NOTE:  see 
        * http://mail.gnome.org/archives/wm-spec-list/2003-March/msg00067.html
        * In particular:
        *
        *  When the X window system was ported to 64-bit architectures, a
        * rather peculiar design decision was made. 32-bit quantities such
        * as Window IDs, atoms, etc, were kept as longs in the client side
        * APIs, even when long was changed to 64 bits.
        *
        */
        if (XGetWindowProperty(disp, win, xa_prop_name, 0, MAX_PROPERTY_VALUE_LEN / 4, False,
            xa_prop_type, &xa_ret_type, &ret_format,     
            &ret_nitems, &ret_bytes_after, &ret_prop) != Success) {
                debug_msg("Cannot get %s property.\n", prop_name);
                return NULL;
        }

        if (xa_ret_type != xa_prop_type) {
                debug_msg("Invalid type of %s property.\n", prop_name);
                XFree(ret_prop);
                return NULL;
        }

        /* null terminate the result to make string handling easier */
        tmp_size = (ret_format / 8) * ret_nitems;
        /* Correct 64 Architecture implementation of 32 bit data */
        if(ret_format==32) tmp_size *= sizeof(long)/4;
        ret = malloc(tmp_size + 1);
        memcpy(ret, ret_prop, tmp_size);
        ret[tmp_size] = '\0';

        if (size) {
                *size = tmp_size;
        }

        XFree(ret_prop);
        return ret;
}

static void update_fullscreen_state(struct state_gl *s)
{
        XEvent xev;
        XSizeHints *size_hints;

        size_hints = XAllocSizeHints();
        if(s->fs) {
                size_hints->flags =  PMinSize | PMaxSize | PWinGravity | PAspect | PBaseSize;
                size_hints->min_width =
                        size_hints->max_width=
                        size_hints->base_width=
                        size_hints->min_aspect.x=
                        size_hints->max_aspect.x=
                        s->x_width;
                size_hints->min_height =
                        size_hints->max_height=
                        size_hints->base_height=
                        size_hints->min_aspect.y=
                        size_hints->max_aspect.y=
                        s->x_height;
                size_hints->win_gravity=StaticGravity;
        } else {
                /* (re)set to defaults */
                size_hints->flags = PBaseSize;
                size_hints->base_height=
                        s->frame.height;
                size_hints->base_width=
                        s->frame.width;
        }

        memset(&xev, 0, sizeof(xev));
        xev.type = ClientMessage;
        xev.xclient.serial = 0;
        xev.xclient.send_event=True;
        xev.xclient.window = s->xwindow;
        xev.xclient.message_type = XInternAtom(s->display, "_NET_WM_STATE", False);
        xev.xclient.format = 32;
        xev.xclient.data.l[0] = s->fs ? 1 : 0;
        xev.xclient.data.l[1] = XInternAtom(s->display, "_NET_WM_STATE_FULLSCREEN", False);
        xev.xclient.data.l[2] = 0;

        XUnmapWindow(s->display, s->xwindow);
        XSendEvent(s->display, DefaultRootWindow(s->display), False,
                       SubstructureRedirectMask|SubstructureNotifyMask, &xev);
        XSetWMNormalHints(s->display, s->xwindow, size_hints);
        XMoveWindow(s->display, s->xwindow, 0, 0);
        XFree(size_hints);

        /* shouldn't be needed */
        if (s->fs) {
                XMoveResizeWindow(s->display, s->xwindow, 0, 0, s->x_width, s->x_height);
        } else {
                XMoveResizeWindow(s->display, s->xwindow, 0, 0, s->frame.width, s->frame.height);
        }

        XMapRaised(s->display, s->xwindow);
        XRaiseWindow(s->display, s->xwindow);
        XSendEvent(s->display, DefaultRootWindow(s->display), False,
                       SubstructureRedirectMask|SubstructureNotifyMask, &xev);
        XMoveWindow(s->display, s->xwindow, 0, 0);
        XFlush(s->display);
}
#endif

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_gl *s = (struct state_sdl *)state;

        memcpy(out, &s->frame, sizeof(struct video_frame));
        out->width = w;
        out->height = h;
        out->dst_x_offset +=
                x * get_bpp(s->frame.dst_bpp);
        out->data +=
                y * s->frame.dst_pitch;
        out->src_linesize =
                vc_getsrc_linesize(w, out->color_spec);
        out->dst_linesize =
                vc_getsrc_linesize(x + w, out->color_spec);

}

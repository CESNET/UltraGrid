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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <GL/glew.h>

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else /* HAVE_MACOSX */
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif /* FREEGLUT */
#endif /* HAVE_MACOSX */

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

/* defined in main.c */
extern int uv_argc;
extern int should_exit;
extern char **uv_argv;

struct state_gl {
	GLubyte		*y, *u, *v;	//Guess what this might be...

	GLhandleARB     VSHandle,FSHandle,PHandle;
	char 		*VProgram,*FProgram;
	/* TODO: make same shaders process YUVs for DXT as for
	 * uncompressed data */
	GLhandleARB     VSHandle_dxt,FSHandle_dxt,PHandle_dxt;
	char 		*VProgram_dxt,*FProgram_dxt;

	GLuint		texture[4];

        /* Thread related information follows... */
        pthread_t	thread_id;

        sem_t		semaphore;
        pthread_mutex_t newframe_lock;
        pthread_cond_t  newframe_cv;

        /* For debugging... */
        uint32_t	magic;

        int             window;

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
void gl_resize(int width, int height);
void glsl_arb_init(void *arg);
void dxt_arb_init(void *arg);
void gl_bind_texture(void *args);
void dxt_bind_texture(void *arg);
void * display_gl_init(char *fmt);

int gl_semaphore_timedwait(sem_t * semaphore, pthread_mutex_t * lock,
                pthread_cond_t * cv, int ms);
void gl_semaphore_post(sem_t * semaphore, pthread_mutex_t * lock,
                pthread_cond_t * cv);
void gl_reconfigure_screen_post(void *s, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux);
void gl_reconfigure_screen(struct state_gl *s);
static void cleanup_data(struct state_gl *s);
void glut_idle_callback(void);
void glut_redisplay_callback(void);
void glut_key_callback(unsigned char key, int x, int y);
void glut_close_callback(void);
void glut_resize_window(struct state_gl *s);

static void get_sub_frame(void *s, int x, int y, int w, int h, struct video_frame *out);

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

        /* GLUT callbacks take only some arguments so we need static variable */
        gl = s;

        s->window = -1;

        pthread_mutex_init(&s->reconf_lock, NULL);
        pthread_cond_init(&s->reconf_cv, NULL);
        pthread_mutex_init(&s->newframe_lock, NULL);
        pthread_cond_init(&s->newframe_cv, NULL);

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

        s->frame.reconfigure = (reconfigure_t) gl_reconfigure_screen_post;
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

	return (void*)s;
}

/*
 * TODO: join the two shaders/functions (?)
 * Consider that the latter stores alpha (has to) while this shader doesn't,
 * which is perhaps a little bit better.
 */
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
}

void dxt_arb_init(void *arg)
{
    struct state_gl        *s = (struct state_gl *) arg;
    char *log;
    /* Set up program objects. */
    s->PHandle_dxt=glCreateProgramObjectARB();
    s->FSHandle_dxt=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    s->VSHandle_dxt=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);

    /* Compile Shader */
    assert(s->FProgram_dxt!=NULL);
    glShaderSourceARB(s->FSHandle_dxt,1,(const GLcharARB**)&(s->FProgram_dxt),NULL);
    glCompileShaderARB(s->FSHandle_dxt);
    glShaderSourceARB(s->VSHandle_dxt,1,(const GLcharARB**)&(s->VProgram_dxt),NULL);
    glCompileShaderARB(s->VSHandle_dxt);

    /* Print compile log */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->FSHandle_dxt,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->VSHandle_dxt,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);

    /* Attach and link our program */
    glAttachObjectARB(s->PHandle_dxt,s->FSHandle_dxt);
    glAttachObjectARB(s->PHandle_dxt,s->VSHandle_dxt);
    glLinkProgramARB(s->PHandle_dxt);

    /* Print link log. */
    log=calloc(32768,sizeof(char));
    glGetInfoLogARB(s->PHandle_dxt,32768,NULL,log);
    printf("Link Log: %s\n", log);
    free(log);
}

/*
 * NOTE: UNUSED - can (should?) be removed when we use ARB
 */
/*void glsl_gl_init(void *arg) {

	//TODO: Add log
	struct state_gl	*s = (struct state_gl *) arg;

	s->PHandle=glCreateProgram();
	s->FSHandle=glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(s->FSHandle,1,(const GLcharARB**)&(s->FProgram),NULL);
	glCompileShader(s->FSHandle);

	glAttachShader(s->PHandle,s->FSHandle);

	glLinkProgram(s->PHandle);
	glUseProgram(s->PHandle);
}*/


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
 * inside appropriate thread and make changes we can do. The rest does
 * gl_reconfigure_screen.
 */
void gl_reconfigure_screen_post(void *arg, unsigned int width, unsigned int height,
                codec_t codec, double fps, int aux)
{
        struct state_gl	*s = (struct state_gl *) arg;

        s->frame.width = width;
        s->frame.height = height;
        s->frame.fps = fps;
        s->frame.aux = aux;
        s->frame.color_spec = codec;

        pthread_mutex_lock(&s->reconf_lock);
        s->needs_reconfigure = TRUE;

        gl_semaphore_post(&s->semaphore, &s->newframe_lock,
                &s->newframe_cv);

        while(s->needs_reconfigure)
                pthread_cond_wait(&s->reconf_cv, &s->reconf_lock);
        pthread_mutex_unlock(&s->reconf_lock);
}

void glut_resize_window(struct state_gl *s)
{
        if (!s->fs) {
                glutReshapeWindow(s->frame.width, s->frame.height);
        } else {
                glutFullScreen();
        }
}

/**
 * This function must be called only from GL thread 
 * (display_thread_gl) !!!
 */
void gl_reconfigure_screen(struct state_gl *s)
{
        int i;
	int h_align = 0;

        assert(s->magic == MAGIC_GL);

        if(s->win_initialized)
                cleanup_data(s);

        s->frame.dst_x_offset = 0;

        for(i = 0; codec_info[i].name != NULL; ++i) {
                if(s->frame.color_spec == codec_info[i].codec) {
                        s->rgb = codec_info[i].rgb;
                        s->frame.src_bpp = codec_info[i].bpp;
                        h_align = codec_info[i].h_align;
                }
        }
        assert(h_align != 0);

        s->dxt = FALSE;

        s->frame.rshift = 0;
        s->frame.gshift = 8;
        s->frame.bshift = 16;

        switch (s->frame.color_spec) {
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

        s->frame.src_linesize = s->frame.width;
	s->frame.src_linesize = ((s->frame.src_linesize + h_align - 1) / h_align) * h_align;
        s->frame.src_linesize *= s->frame.src_bpp;

        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;
        s->frame.data_len = s->frame.dst_linesize * s->frame.height;

        s->frame.data = (char *) malloc(s->frame.data_len);
	s->y=malloc(s->frame.width * s->frame.height);
	s->u=malloc(s->frame.width * s->frame.height);
	s->v=malloc(s->frame.width * s->frame.height);

	asm("emms\n");
        s->raspect = (double) s->frame.height / s->frame.width;

	fprintf(stdout,"Setting GL window size %dx%d.\n", s->frame.width, s->frame.height);
	glut_resize_window(s);

	glUseProgramObjectARB(0);

        if(s->dxt) {
		glBindTexture(GL_TEXTURE_2D,s->texture[0]);
		glCompressedTexImage2D(GL_TEXTURE_2D, 0,
				GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
				s->frame.width, s->frame.height, 0,
				(s->frame.width*s->frame.height/16)*8,
				s->frame.data);
		if(!s->rgb) {
			glBindTexture(GL_TEXTURE_2D,s->texture[0]);
			glUseProgramObjectARB(s->PHandle_dxt);
		}
        } else if(!s->dxt) {
		if (!s->rgb) {
			glUseProgramObjectARB(s->PHandle);

			glBindTexture(GL_TEXTURE_2D,s->texture[0]);
			glTexImage2D(GL_TEXTURE_2D, 0, 1,
				s->frame.width/2, s->frame.height, 0,
				GL_LUMINANCE, GL_UNSIGNED_BYTE, s->u);

			glBindTexture(GL_TEXTURE_2D,s->texture[1]);
			glTexImage2D(GL_TEXTURE_2D, 0, 1,
				s->frame.width/2, s->frame.height, 0,
				GL_LUMINANCE, GL_UNSIGNED_BYTE, s->v);

			glBindTexture(GL_TEXTURE_2D,s->texture[2]);
			glTexImage2D(GL_TEXTURE_2D, 0, 1,
				s->frame.width, s->frame.height, 0,
				GL_LUMINANCE, GL_UNSIGNED_BYTE, s->y);
		} else {
			glBindTexture(GL_TEXTURE_2D,s->texture[0]);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                        s->frame.width, s->frame.height, 0,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        s->frame.data);
		}
        }

        s->win_initialized = TRUE;
}

/* Macs doesn't support timedwaits so this is just workaround
 * using sem + mutex + cv
 */
int gl_semaphore_timedwait(sem_t * semaphore, pthread_mutex_t * lock,
                pthread_cond_t * cv, int ms)
{
        pthread_mutex_lock(lock);
        /* first of all - try to decrement, if failed wait with tmout */
        if(sem_trywait(semaphore) != 0) {
                struct timeval tp;
                struct timespec ts;

                gettimeofday(&tp, NULL);
                /* Convert from timeval to timespec */
                ts.tv_sec  = tp.tv_sec;
                ts.tv_nsec = tp.tv_usec * 1000;

                ts.tv_nsec += ms * 1000 * 1000; /* 0.2 sec */
                // make it correct
                ts.tv_sec += ts.tv_nsec / (1000 * 1000 * 1000);
                ts.tv_nsec = ts.tv_nsec % (1000 * 1000 * 1000);

                if(pthread_cond_timedwait(cv, lock,
                                &ts) != 0) {
                        pthread_mutex_unlock(lock);
                        return 1;
                } else {
                        /* just decrement - we know that it has 'value > 0'
                         * in this case */
                        platform_sem_wait(semaphore);
                }
        }
        pthread_mutex_unlock(lock);

        return 0;
}

void gl_semaphore_post(sem_t * semaphore, pthread_mutex_t * lock,
                pthread_cond_t * cv)
{
        pthread_mutex_lock(lock);
        platform_sem_post(semaphore);
        pthread_cond_signal(cv);
        pthread_mutex_unlock(lock);
}

void glut_idle_callback(void)
{
        struct state_gl *s = gl;
        struct timeval tv;
        double seconds;


        if(gl_semaphore_timedwait(&s->semaphore, &s->newframe_lock,
                                &s->newframe_cv, 200) != 0) /* timeout */
                return;

        pthread_mutex_lock(&s->reconf_lock);
        if (s->needs_reconfigure) {
                /* there has been scheduled request for win reconfiguration */
                gl_reconfigure_screen(s);
                s->needs_reconfigure = FALSE;
                pthread_cond_signal(&s->reconf_cv);
                pthread_mutex_unlock(&s->reconf_lock);
                return; /* return after reconfiguration */
        }
        pthread_mutex_unlock(&s->reconf_lock);

        /* for DXT, deinterlacing doesn't make sense since it is
         * always deinterlaced before comrpression */
        if(s->deinterlace && !s->dxt)
                vc_deinterlace((unsigned char *) s->frame.data,
                                s->frame.dst_linesize, s->frame.height);

        if(s->dxt) {
                if(s->rgb) {
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->frame.width, s->frame.height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        (s->frame.width*s->frame.height/16)*8,
                                        s->frame.data);
                } else {
                        dxt_bind_texture(s);
                }
        } else {
                if(!s->rgb)
                {
                        getY((GLubyte *) s->frame.data, s->y, s->u, s->v,
                                        s->frame.width, s->frame.height);
                        gl_bind_texture(s);
                } else {
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->frame.width, s->frame.height,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        s->frame.data);
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
                                gl->fs = TRUE;
                        } else {
                                gl->fs = FALSE;
                        }
			glut_resize_window(gl);
                        break;
                case 'q':
                        platform_sem_post(&gl->semaphore);
                        if(gl->window != -1)
                                glutDestroyWindow(gl->window);
			should_exit = 1;
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

	s->window = glutCreateWindow(WIN_NAME);
	glutKeyboardFunc(glut_key_callback);
	glutDisplayFunc(glut_redisplay_callback);
	glutWMCloseFunc(glut_close_callback);
	glutReshapeFunc(gl_resize);


	s->FProgram_dxt = frag;
	s->VProgram_dxt = vert;
	s->FProgram = glsl;

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
        glewInit();
        glEnable( GL_TEXTURE_2D );
	glGenTextures(4, s->texture);

	glBindTexture(GL_TEXTURE_2D, s->texture[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, s->texture[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, s->texture[2]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(glewIsSupported("GL_VERSION_2_0")){
		fprintf(stderr, "OpenGL 2.0 is supported...\n");
		//TODO: Re-enable gl_init!
		//glsl_gl_init(s);
	}else if(GLEW_ARB_fragment_shader){
		fprintf(stderr, "OpenGL 2.0 not supported, using ARB extension...\n");
	}else{
		fprintf(stderr, "ERROR: Neither OpenGL 2.0 nor ARB_fragment_shader are supported, try updating your drivers...\n");
		exit(65);
	}
	glsl_arb_init(s);
	dxt_arb_init(s);

        /* Wait until we have some data and thus created window 
         * otherwise the mainLoop would exit immediately */
        while(!s->win_initialized) {
                pthread_mutex_lock(&s->reconf_lock);
                if (s->needs_reconfigure) {
                        gl_reconfigure_screen(s);
                        s->needs_reconfigure = FALSE;
                        pthread_cond_signal(&s->reconf_cv);
                }
                pthread_mutex_unlock(&s->reconf_lock);
                usleep(1000);
        }

        glutMainLoop();
}


void gl_resize(int width,int height)
{
	glViewport( 0, 0, ( GLint )width, ( GLint )height );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity( );

	double screen_ratio;
	double x = 1.0,
	   y = 1.0;

	debug_msg("Resized to: %dx%d\n", width, height);

	screen_ratio = (double) width / height;
	if(screen_ratio > 1.0 / gl->raspect) {
	    x = (double) height / (width * gl->raspect);
	} else {
	    y = (double) width / (height / gl->raspect);
	}
	glScalef(x, y, 1);

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
	glBindTexture(GL_TEXTURE_2D,s->texture[0]);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,s->frame.width/2,s->frame.height,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->u);

	glActiveTexture(GL_TEXTURE2);
	i=glGetUniformLocationARB(s->PHandle,"Vtex");
	glUniform1iARB(i,2); 
	glBindTexture(GL_TEXTURE_2D,s->texture[1]);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,s->frame.width/2,s->frame.height,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->v);

	glActiveTexture(GL_TEXTURE0);
	i=glGetUniformLocationARB(s->PHandle,"Ytex");
	glUniform1iARB(i,0); 
	glBindTexture(GL_TEXTURE_2D,s->texture[2]);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,s->frame.width,s->frame.height,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->y);
}    

void dxt_bind_texture(void *arg)
{
        struct state_gl        *s = (struct state_gl *) arg;
        static int i=0;

        //TODO: does OpenGL use different stuff here?
        glActiveTexture(GL_TEXTURE0);
        i=glGetUniformLocationARB(s->PHandle,"yuvtex");
        glUniform1iARB(i,0); 
        glBindTexture(GL_TEXTURE_2D,gl->texture[0]);
	glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
			s->frame.width, s->frame.height,
			GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
			(s->frame.width*s->frame.height/16)*8,
			s->frame.data);
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

void glut_close_callback(void)
{
	exit(0);
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
        pthread_mutex_destroy(&s->newframe_lock);
        pthread_cond_destroy(&s->newframe_cv);
        pthread_mutex_destroy(&s->reconf_lock);
        pthread_cond_destroy(&s->reconf_cv);
        if(s->win_initialized) {
                glutDestroyWindow(s->window);
                cleanup_data(s);
        }
}

struct video_frame * display_gl_getf(void *state)
{
        struct state_gl *s = (struct state_gl *) state;
        assert(s->magic == MAGIC_GL);

        return &s->frame;
}

int display_gl_putf(void *state, char *frame)
{
        struct state_gl *s = (struct state_gl *) state;
	int tmp;

        assert(s->magic == MAGIC_GL);
        UNUSED(frame);

        /* ...and signal the worker */
        gl_semaphore_post(&s->semaphore, &s->newframe_lock,
                &s->newframe_cv);

#ifndef HAVE_MACOSX
        /* isn't implemented on Macs */
        sem_getvalue(&s->semaphore, &tmp);
        if(tmp > 1) {
                printf("frame drop!\n");
                sem_trywait(&s->semaphore); /* decrement then */
        }
#endif
        return 0;
}

static void cleanup_data(struct state_gl *s)
{
        free(s->y);
        free(s->u);
        free(s->v);
        free(s->frame.data);
}

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_gl *s = (struct state_gl *)state;
        UNUSED(h);

        memcpy(out, &s->frame, sizeof(struct video_frame));
        out->data +=
                y * s->frame.dst_pitch +
                (size_t) (x * s->frame.dst_bpp);
        out->src_linesize =
                vc_getsrc_linesize(w, out->color_spec);
        out->dst_linesize =
                w * out->dst_bpp;

}


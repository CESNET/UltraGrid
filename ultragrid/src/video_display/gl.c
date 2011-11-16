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

#define GL_GLEXT_PROTOTYPES 1

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else /* HAVE_MACOSX */
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include "x11_common.h"
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

#define STRINGIFY(A) #A

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
        yuv.r = 1.1643*(yuv.r-0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        gl_FragColor.r = yuv.r + 1.5958 * yuv.b;
        gl_FragColor.g = yuv.r - 0.39173* yuv.g - 0.81290 * yuv.b;
        gl_FragColor.b = yuv.r + 2.017 * yuv.g;
});

static char * yuv422_to_rgb_vp = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();
});

/* DXT YUV (FastDXT) related */
static char * frag = "\
uniform sampler2D yuvtex;\
\
void main(void) {\
vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);\
\
float Y = col[0];\
float U = col[1]-0.5;\
float V = col[2]-0.5;\
Y=1.1643*(Y-0.0625);\
\
float G = Y-0.39173*U-0.81290*V;\
float B = Y+2.017*U;\
float R = Y+1.5958*V;\
\
gl_FragColor=vec4(R,G,B,1.0);}";

static char * vert = "\
void main() {\
gl_TexCoord[0] = gl_MultiTexCoord0;\
gl_Position = ftransform();}";

static const char fp_display_dxt5ycocg[] = 
    "#extension GL_EXT_gpu_shader4 : enable\n"
    "uniform sampler2D _image;\n"
    "void main()\n"
    "{\n"
    "    vec4 _rgba;\n"
    "    float _scale;\n"
    "    float _Co;\n"
    "    float _Cg;\n"
    "    float _R;\n"
    "    float _G;\n"
    "    float _B;\n"
    "    _rgba = texture2D(_image, gl_TexCoord[0].xy);\n"
    "    _scale = 1.00000000E+00/(3.18750000E+01*_rgba.z + 1.00000000E+00);\n"
    "    _Co = (_rgba.x - 5.01960814E-01)*_scale;\n"
    "    _Cg = (_rgba.y - 5.01960814E-01)*_scale;\n"
    "    _R = (_rgba.w + _Co) - _Cg;\n"
    "    _G = _rgba.w + _Cg;\n"
    "    _B = (_rgba.w - _Co) - _Cg;\n"
    "    _rgba = vec4(_R, _G, _B, 1.00000000E+00);\n"
    "    gl_FragColor = _rgba;\n"
    "    return;\n"
    "} // main end\n"
;

/* defined in main.c */
extern int uv_argc;
extern int should_exit;
extern char **uv_argv;

struct state_gl {
        GLhandleARB     VSHandle,FSHandle,PHandle;
	/* TODO: make same shaders process YUVs for DXT as for
	 * uncompressed data */
	GLhandleARB     VSHandle_dxt,FSHandle_dxt,PHandle_dxt;
        GLhandleARB     FSHandle_dxt5, PHandle_dxt5;

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
        volatile unsigned    needs_reconfigure:1;
        pthread_mutex_t lock;
        pthread_cond_t  reconf_cv;

        double          aspect;
        double          video_aspect;
        unsigned long int frames;

        struct timeval  tv;
};

static struct state_gl *gl;

/* Prototyping */
void gl_draw(double ratio);
void gl_show_help(void);

void gl_check_error(void);
void gl_resize(int width, int height);
void glsl_arb_init(void *arg);
void dxt_arb_init(void *arg);
void gl_bind_texture(void *args);
void dxt_bind_texture(void *arg);
void dxt5_arb_init(struct state_gl *s);
void gl_reconfigure_screen(struct state_gl *s);
void glut_idle_callback(void);
void glut_key_callback(unsigned char key, int x, int y);
void glut_close_callback(void);
void glut_resize_window(struct state_gl *s);

/**
 * Show help
 * @since 23-03-2010, xsedmik
 */
void gl_show_help(void) {
        printf("GL options:\n");
        printf("\t-d gl:{d|fs|aspect=<v>/<h>}(,{d|fs|aspect=<v>/<h>})* | help\n\n");
        printf("\t\td\t\tdeinterlace\n");
        printf("\t\tfs\t\tfullscreen\n");
        printf("\t\taspect=<w>/<h>\trequested video aspect (eg. 16/9). Leave unset if PAR = 1.\n");
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

void * display_gl_init(char *fmt, unsigned int flags) {
        UNUSED(flags);
	struct state_gl        *s;
        
#ifndef HAVE_MACOSX
        x11_enter_thread();
#endif

        glutInit(&uv_argc, uv_argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	s = (struct state_gl *) calloc(1,sizeof(struct state_gl));
	s->magic   = MAGIC_GL;
        
        /* GLUT callbacks take only some arguments so we need static variable */
        gl = s;
        s->window = -1;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->reconf_cv, NULL);

        s->fs = FALSE;
        s->deinterlace = FALSE;
        s->video_aspect = 0.0;

	// parse parameters
	if (fmt != NULL) {
		if (strcmp(fmt, "help") == 0) {
			gl_show_help();
			free(s);
			return NULL;
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
                        } else {
                                fprintf(stderr, "[GL] Unknown option: %s\n", tok);
                        }
                        tmp = NULL;
                }

		free(tmp);
	}

        s->frame = vf_alloc(1, 1);
        s->tile = tile_get(s->frame, 0, 0);
        s->tile->data = NULL;
        
        s->frames = 0ul;
        gettimeofday(&s->tv, NULL);

        fprintf(stdout,"GL setup: fullscreen: %s, deinterlace: %s\n",
                        s->fs ? "ON" : "OFF", s->deinterlace ? "ON" : "OFF");

	s->new_frame = 0;
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
        const GLcharARB	*VProgram, *FProgram;
        
        FProgram = (const GLcharARB*) yuv422_to_rgb_fp;
	VProgram = (const GLcharARB*) yuv422_to_rgb_vp;
        /* Set up program objects. */
        s->PHandle=glCreateProgramObjectARB();
        s->VSHandle=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
        s->FSHandle=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
        
        /* Compile Shader */
        glShaderSourceARB(s->FSHandle,1, &FProgram,NULL);
        glCompileShaderARB(s->FSHandle);
        
        /* Print compile log */
        log=calloc(32768,sizeof(char));
        glGetInfoLogARB(s->FSHandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        glShaderSourceARB(s->VSHandle,1, &VProgram,NULL);
        glCompileShaderARB(s->VSHandle);
        memset(log, 0, 32768);
        glGetInfoLogARB(s->VSHandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        /* Attach and link our program */
        glAttachObjectARB(s->PHandle,s->FSHandle);
        glAttachObjectARB(s->PHandle,s->VSHandle);
        glLinkProgramARB(s->PHandle);
        
        /* Print link log. */
        memset(log, 0, 32768);
        glGetInfoLogARB(s->PHandle,32768,NULL,log);
        printf("Link Log: %s\n", log);
        free(log);

        // Create fbo    
        glGenFramebuffersEXT(1, &s->fbo_id);
}

void dxt_arb_init(void *arg)
{
    struct state_gl        *s = (struct state_gl *) arg;
    char *log;
    const GLcharARB *FProgram, *VProgram;
    
    FProgram = (const GLcharARB *) frag;
    VProgram = (const GLcharARB *) vert;
    /* Set up program objects. */
    s->PHandle_dxt=glCreateProgramObjectARB();
    s->FSHandle_dxt=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    s->VSHandle_dxt=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);

    /* Compile Shader */
    glShaderSourceARB(s->FSHandle_dxt,1,&FProgram,NULL);
    glCompileShaderARB(s->FSHandle_dxt);
    glShaderSourceARB(s->VSHandle_dxt,1,&VProgram,NULL);
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

void dxt5_arb_init(struct state_gl *s)
{
        char *log;
        const GLcharARB *FProgram;
    
        FProgram = (const GLcharARB *) fp_display_dxt5ycocg;
        
        /* Set up program objects. */
        s->PHandle_dxt5=glCreateProgramObjectARB();
        s->FSHandle_dxt5=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
        
        /* Compile Shader */
        glShaderSourceARB(s->FSHandle_dxt5, 1, &FProgram, NULL);
        glCompileShaderARB(s->FSHandle_dxt5);
        
        /* Print compile log */
        log=calloc(32768, sizeof(char));
        glGetInfoLogARB(s->FSHandle_dxt5, 32768, NULL, log);
        printf("Compile Log: %s\n", log);
        free(log);
        
        /* Attach and link our program */
        glAttachObjectARB(s->PHandle_dxt5, s->FSHandle_dxt5);
        glLinkProgramARB(s->PHandle_dxt5);
        
        /* Print link log. */
        log=calloc(32768, sizeof(char));
        glGetInfoLogARB(s->PHandle_dxt5, 32768, NULL, log);
        printf("Link Log: %s\n", log);
        free(log);
}

/*
 * This function will be probably runned from another thread than GL-thread so
 * we cannot reconfigure directly there. Instead, we post a request to do it
 * inside appropriate thread and make changes we can do. The rest does
 * gl_reconfigure_screen.
 */
void display_gl_reconfigure(void *state, struct video_desc desc)
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
        s->frame->aux = desc.aux;
        s->frame->color_spec = desc.color_spec;

        pthread_mutex_lock(&s->lock);
        s->needs_reconfigure = TRUE;
	s->new_frame = 1;

        while(s->needs_reconfigure)
                pthread_cond_wait(&s->reconf_cv, &s->lock);
        pthread_mutex_unlock(&s->lock);
}

void glut_resize_window(struct state_gl *s)
{
        if (!s->fs) {
                glutReshapeWindow(s->tile->height * s->aspect, s->tile->height);
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
        assert(s->magic == MAGIC_GL);

        free(s->tile->data);
        
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec)
                        * s->tile->height;
        s->tile->data = (char *) malloc(s->tile->data_len);

	asm("emms\n");
        if(!s->video_aspect)
                s->aspect = (double) s->tile->width / s->tile->height;
        else
                s->aspect = s->video_aspect;

	fprintf(stdout,"Setting GL window size %dx%d (%dx%d).\n", (int)(s->aspect * s->tile->height),
                        s->tile->height, s->tile->width, s->tile->height);
	glutShowWindow();
        glut_resize_window(s);

	glUseProgramObjectARB(0);

        if(s->frame->color_spec == DXT1 || s->frame->color_spec == DXT1_YUV) {
		glBindTexture(GL_TEXTURE_2D,s->texture_display);
		glCompressedTexImage2D(GL_TEXTURE_2D, 0,
				GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
				s->tile->width, s->tile->height, 0,
				(s->tile->width * s->tile->height/16)*8,
				NULL);
		if(s->frame->color_spec == DXT1_YUV) {
			glBindTexture(GL_TEXTURE_2D,s->texture_display);
			glUseProgramObjectARB(s->PHandle_dxt);
		}
        } else if (s->frame->color_spec == UYVY) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D,s->texture_uyvy);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                s->tile->width / 2, s->tile->height, 0,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                NULL);
                glUseProgramObjectARB(s->PHandle);
                glUniform1i(glGetUniformLocation(s->PHandle, "image"), 2);
                glUniform1f(glGetUniformLocation(s->PHandle, "imageWidth"),
                        (GLfloat) s->tile->width);
                glUseProgramObjectARB(0);
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
                glUseProgramObjectARB(s->PHandle_dxt5);
                
                glBindTexture(GL_TEXTURE_2D,s->texture_display);
                glCompressedTexImage2D(GL_TEXTURE_2D, 0,
				GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
				s->tile->width, s->tile->height, 0,
				s->tile->width * s->tile->height,
				NULL);
        }
}

void glut_idle_callback(void)
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
                vc_deinterlace((unsigned char *) s->tile->data,
                                vc_get_linesize(s->tile->width, s->frame->color_spec),
                                s->tile->height);

        switch(s->frame->color_spec) {
                case DXT1:
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                        (s->tile->width * s->tile->height/16)*8,
                                        s->tile->data);
                        break;
                case DXT1_YUV:
                                dxt_bind_texture(s);
                        break;
                case UYVY:
                        gl_bind_texture(s);
                        break;
                case RGBA:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_RGBA, GL_UNSIGNED_BYTE,
                                        s->tile->data);
                        break;
                case RGB:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_RGB, GL_UNSIGNED_BYTE,
                                        s->tile->data);
                        break;
                case DXT5:                        
                        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        s->tile->width, s->tile->height,
                                        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                                        s->tile->width * s->tile->height,
                                        s->tile->data);
                        break;
                default:
                        error_with_code_msg(128, "[GL] Fatal error - received unsupported codec. "
                                                 "Please report a bug.");
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

        gl_draw(s->aspect);
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
                        if(gl->window != -1)
                                glutDestroyWindow(gl->window);
			should_exit = 1;
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
        char *tmp, *gl_ver_major;
        char *save_ptr = NULL;

#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
        /* Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

        glutIdleFunc(glut_idle_callback);
	s->window = glutCreateWindow(WIN_NAME);
        glutHideWindow();
	glutKeyboardFunc(glut_key_callback);
	glutDisplayFunc(glutSwapBuffers);
	glutWMCloseFunc(glut_close_callback);
	glutReshapeFunc(gl_resize);

        tmp = strdup((const char *)glGetString(GL_VERSION));
        gl_ver_major = strtok_r(tmp, ".", &save_ptr);
        if(atoi(gl_ver_major) >= 2) {
                fprintf(stderr, "OpenGL 2.0 is supported...\n");
        } else {
                fprintf(stderr, "ERROR: OpenGL 2.0 is not supported, try updating your drivers...\n");
                free(tmp);
                exit(65);
        }
        free(tmp);

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

	glsl_arb_init(s);
	dxt_arb_init(s);
        dxt5_arb_init(s);
        
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

void gl_bind_texture(void *arg)
{
	struct state_gl        *s = (struct state_gl *) arg;
        
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, s->fbo_id);
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture_display, 0);
        //assert(GL_FRAMEBUFFER_COMPLETE_EXT == glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT));
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, s->texture_uyvy);
        
        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
	glLoadIdentity( );
	glOrtho(-1,1,-1/gl->aspect,1/gl->aspect,10,-10);
        
	glMatrixMode( GL_MODELVIEW );
        glPushMatrix();
	glLoadIdentity( );
        
        glPushAttrib(GL_VIEWPORT_BIT);
        
        glViewport( 0, 0, s->tile->width, s->tile->height);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s->tile->width / 2, s->tile->height,  GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);
        glUseProgramObjectARB(s->PHandle);
        
        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 
        
        float aspect = (double) s->tile->width / s->tile->height;
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0/aspect);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0/aspect);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0/aspect);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0/aspect);
        glEnd();
        
        glPopAttrib();
        
	
        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );
        glPopMatrix();
        
        glUseProgramObjectARB(0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, s->texture_display);
}    

void dxt_bind_texture(void *arg)
{
        struct state_gl        *s = (struct state_gl *) arg;
        static int i=0;

        //TODO: does OpenGL use different stuff here?
        glActiveTexture(GL_TEXTURE0);
        i=glGetUniformLocationARB(s->PHandle,"yuvtex");
        glUniform1iARB(i,0); 
        glBindTexture(GL_TEXTURE_2D,gl->texture_display);
	glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
			s->tile->width, s->tile->height,
			GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
			(s->tile->width * s->tile->height/16)*8,
			s->tile->data);
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
      glTexCoord2f( 0.0f, 1.0f ); glVertex2f( -1.0f, -1/ratio);
      /* Bottom Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 1.0f ); glVertex2f(  1.0f, -1/ratio);
      /* Top Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  1/ratio);
      /* Top Left Of The Texture and Quad */
      glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  1/ratio);
    glEnd( );

    gl_check_error();
}

void glut_close_callback(void)
{
        should_exit = TRUE;
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

int display_gl_get_property(void *state, int property, void *val, int *len)
{
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1, DXT1_YUV, DXT5};
        
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
        if(s->window != -1) {
                glutDestroyWindow(s->window);
        }
        free(s->tile->data);
        vf_free(s->frame);
        free(s);
}

struct video_frame * display_gl_getf(void *state)
{
        struct state_gl *s = (struct state_gl *) state;
        assert(s->magic == MAGIC_GL);

        return s->frame;
}

int display_gl_putf(void *state, char *frame)
{
        struct state_gl *s = (struct state_gl *) state;
	int tmp;

        assert(s->magic == MAGIC_GL);
        UNUSED(frame);

        /* ...and signal the worker */
        pthread_mutex_lock(&s->lock);
        if(s->new_frame != 0) {
                printf("frame drop!\n");
        } else {
		s->new_frame++;
	}

        pthread_mutex_unlock(&s->lock);
        return 0;
}

/*
 * FILE:    video_display/gl_sdl.c
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
#else /* HAVE_MACOSX */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif /* HAVE_MACOSX */
#include <SDL/SDL.h>
#include "compat/platform_semaphore.h"
#include <signal.h>
#include <assert.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "debug.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_display.h"
#include "video_display/gl_sdl.h"
#include "video_codec.h"

#define HD_WIDTH        1920
#define HD_HEIGHT       1080
#define MAGIC_GL       DISPLAY_GL_ID


struct state_sdl {
        Display         *display;
	unsigned int	x_res_x;
	unsigned int	x_res_y;

        int             vw_depth;
        SDL_Overlay     *vw_image;
        GLubyte         *buffers[2];
		GLubyte			*outbuffer;
		GLubyte			*y, *u, *v;	//Guess what this might be...
		char*			 VSHandle,FSHandle,PHandle;
        int             image_display, image_network;
		GLuint  			texture[4];
        /* Thread related information follows... */
        pthread_t                thread_id;
        sem_t                    semaphore;
        /* For debugging... */
        uint32_t                 magic;

        SDL_Surface             *sdl_screen;
        SDL_Rect                rect;

		char *VProgram,*FProgram;
};

/* Prototyping */
static void * display_thread_gl(void *arg);
void gl_deinterlace(GLubyte *buffer);//unsigned
void extrapolate(GLubyte *input, GLubyte *output);
inline void getY(GLubyte *input,GLubyte *y, GLubyte *u,GLubyte *v);
void gl_resize_window(int width, int height);
void gl_bind_texture(void *args);
void gl_draw();
void loadShader(void *arg, char *filename);
void glsl_gl_init(void *arg);
void glsl_arb_init(void *arg);
inline void gl_copyline64(GLubyte *dst, GLubyte *src, int len);
inline void gl_copyline128(GLubyte *d, GLubyte *s, int len);//unsigned
void * display_gl_init(void);

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
				fprintf(stderr, "wft mate? Unknown GL ERROR: %p\n", (void *)msg);
				break;
		}
		msg=glGetError();
	}
	if(flag)
		exit(1);
}

void * display_gl_init(void)
{
    struct state_sdl        *s;

    int			ret;
    int			itemp;
    unsigned int	utemp;
    Window		wtemp;

    s = (struct state_sdl *) calloc(1,sizeof(struct state_sdl));
    s->magic   = MAGIC_GL;

    if (!(s->display = XOpenDisplay(NULL))) {
	printf("Unable to open display GL: XOpenDisplay.\n");
	return NULL;
    }
    
    /* Get XWindows resolution */
    ret = XGetGeometry(s->display, DefaultRootWindow(s->display), &wtemp, &itemp, &itemp, &(s->x_res_x), &(s->x_res_y), &utemp, &utemp);

    s->rect.w = HD_WIDTH;
    s->rect.h = HD_HEIGHT;
    if ((s->x_res_x - HD_WIDTH) > 0) {
	s->rect.x = (s->x_res_x - HD_WIDTH) / 2;
    } else {
	s->rect.x = 0;
    }
    if ((s->x_res_y - HD_HEIGHT) > 0) {
	s->rect.y = (s->x_res_y - HD_HEIGHT) / 2;
    } else {
	s->rect.y = 0;
    }

    s->buffers[0]=malloc(HD_WIDTH*HD_HEIGHT*3);
    s->buffers[1]=malloc(HD_WIDTH*HD_HEIGHT*3);
    s->outbuffer=malloc(HD_WIDTH*HD_HEIGHT*4);
    s->image_network=0;
    s->image_display=1;
    s->y=malloc(HD_WIDTH*HD_HEIGHT);
    s->u=malloc(HD_WIDTH*HD_HEIGHT);
    s->v=malloc(HD_WIDTH*HD_HEIGHT);

    asm("emms\n");

    platform_sem_init(&s->semaphore, 0, 0);
    if (pthread_create(&(s->thread_id), NULL, display_thread_gl, (void *) s) != 0) {
        perror("Unable to create display thread\n");
        return NULL;
    }

    return (void*)s;
}

void loadShader(void *arg, char *filename)
{
	struct state_sdl        *s = (struct state_sdl *) arg;
	struct stat file;
	
	s = (struct state_sdl *) calloc(1,sizeof(struct state_sdl));

	stat(filename,&file);
	s->FProgram=calloc(file.st_size+1,sizeof(char));
	FILE *fh;
	fh=fopen(filename, "r");
	if(!fh){
		perror(filename);
		exit(113);
	}
	fread(s->FProgram,sizeof(char),file.st_size,fh);
	fclose(fh);
}

void glsl_arb_init(void *arg)
{
    struct state_sdl        *s = (struct state_sdl *) arg;
    char *log;

    /* Set up program objects. */
    s->PHandle=glCreateProgramObjectARB();
    // s->VSHandle=glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    s->FSHandle=glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

    /* Compile Shader */
    assert(s->FProgram!=NULL);
    // assert(s->VProgram!=NULL);
    glShaderSourceARB(s->FSHandle,1,(const GLcharARB**)&(s->FProgram),NULL);
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

void glsl_gl_init(void *arg)
{
	//TODO: Add log
        struct state_sdl        *s = (struct state_sdl *) arg;

	s->PHandle=glCreateProgram();
	s->FSHandle=glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(s->FSHandle,1,(const GLcharARB**)&(s->FProgram),NULL);
	glCompileShader(s->FSHandle);

	glAttachShader(s->PHandle,s->FSHandle);

	glLinkProgram(s->PHandle);
	glUseProgram(s->PHandle);
}

void extrapolate(GLubyte *input, GLubyte *output)
{
	/* A linear non-blending method, kida dumb, but it seems to work, and is somewhat fast :-) */
	register int x;
	for(x=0;x<960*1080;x++) {
		output[2*x]=input[x];
		output[2*x+1]=input[x];
	}
}

inline void getY(GLubyte *input,GLubyte *y, GLubyte *u,GLubyte *v)
{
	//TODO: This should be re-written in assembly
	//Note: We assume 1920x1080 UYVY (YUV 4:2:2)
	//See http://www.fourcc.org/indexyuv.htm for more info
	//0x59565955 - UYVY - 16 bits per pixel (2 bytes per plane)
	register int x;
	for(x=0;x<1920*1080*2;x+=4) {
		*u++=input[x];		//1	5	9	13
		*y++=input[x+1];	//2	6	10	14
		*v++=input[x+2];	//3	7	11	15
		*y++=input[x+3];	//0	4	8	12
	}
}

static void * display_thread_gl(void *arg)
{
    struct state_sdl        *s = (struct state_sdl *) arg;
    int j, i;

    const SDL_VideoInfo *videoInfo;
    int videoFlags;
    /* FPS */
    static GLint T0     = 0;
    static GLint Frames = 0;
    double bpp;

#ifdef HAVE_MACOSX
            /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
    	    /* Avoids uncaught exception (1002)  when creating CGSWindow */
            NSApplicationLoad();
#endif

    /* initialize SDL */
    if ( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE ) < 0 ) {
        fprintf( stderr, "Video initialization failed: %s\n",SDL_GetError());
        exit(1);
    }

    /* Fetch the video info */
    videoInfo = SDL_GetVideoInfo( );

    if ( !videoInfo ) {
        fprintf( stderr, "Video query failed: %s\n",SDL_GetError());
        exit(1);
    }

    /* the flags to pass to SDL_SetVideoMode */
    videoFlags  = SDL_OPENGL;          /* Enable OpenGL in SDL */
    videoFlags |= SDL_GL_DOUBLEBUFFER; /* Enable double buffering */
    videoFlags |= SDL_HWPALETTE;       /* Store the palette in hardware */
    videoFlags |= SDL_FULLSCREEN;      /* Fullscreen */

    /* This checks to see if surfaces can be stored in memory */
    if ( videoInfo->hw_available )
        videoFlags |= SDL_HWSURFACE;
    else
        videoFlags |= SDL_SWSURFACE;

    /* This checks if hardware blits can be done */
    if ( videoInfo->blit_hw )
        videoFlags |= SDL_HWACCEL;

    /* Sets up OpenGL double buffering */
    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );	//TODO: Is this necessary?
#ifdef HAVE_SDL_1210
    SDL_GL_SetAttribute(SDL_GL_SWAP_CONTROL, 1); 
#endif /* HAVE_SDL_1210 */

    /* get a SDL surface */
    s->sdl_screen = SDL_SetVideoMode(s->x_res_x, s->x_res_y, 32, videoFlags);
    if(!s->sdl_screen){
        fprintf(stderr,"Error setting video mode %dx%d!\n", s->x_res_x, s->x_res_y);
        exit(128);
    }

    SDL_WM_SetCaption("Ultragrid - OpenGL Display", "Ultragrid");

    SDL_ShowCursor(SDL_DISABLE);

    /* OpenGL Setup */
    glEnable( GL_TEXTURE_2D );
    glClearColor( 1.0f, 1.0f, 1.0f, 0.1f );
    gl_resize_window(s->x_res_x, s->x_res_y);
    glGenTextures(4, s->texture);	//TODO: Is this necessary?

    /* Display splash screen */
    SDL_Surface		*temp;
    GLuint texture;			// This is a handle to our texture object
    temp = SDL_LoadBMP("/usr/share/uv-0.3.1/uv_startup.bmp");
    if (temp == NULL) {
        temp = SDL_LoadBMP("/usr/local/share/uv-0.3.1/uv_startup.bmp");
        if (temp == NULL) {
            temp = SDL_LoadBMP("uv_startup.bmp");
            if (temp == NULL) {
                printf("Unable to load splash bitmap: uv_startup.bmp.\n");
            }
        }
    }

    if (temp != NULL) {
	/* Display the SDL_surface as a OpenGL texture */
        
	// Have OpenGL generate a texture object handle for us
	glGenTextures(1, &texture);
 
	// Bind the texture object
	glBindTexture(GL_TEXTURE_2D, texture);
 
	// Set the texture's stretching properties
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
 
	// Edit the texture object's image data using the information SDL_Surface gives us
	glTexImage2D(GL_TEXTURE_2D, 0, 3, temp->w, temp->h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, temp->pixels);

	gl_draw();

        glDeleteTextures( 1, &texture );
    }

    /* Load shader */
    //TODO: Need a less breaky way to do this...
    struct stat file;
    char *filename=strdup("../src/video_display/gl_sdl.glsl");
    if ((stat(filename,&file)) == -1) {
	filename=strdup("/usr/share/uv-0.3.1/gl_sdl.glsl");
	if ((stat(filename,&file)) == -1) {
	    filename=strdup("/usr/local/share/uv-0.3.1/gl_sdl.glsl");
	    if ((stat(filename,&file)) == -1) {
		fprintf(stderr, "gl_sdl.glsl not found. Giving up!\n");
		exit(113);
	    }
	}
    }
    s->FProgram=calloc(file.st_size+1,sizeof(char));
    FILE *fh;
    fh=fopen(filename, "r");
    if(!fh){
    	perror(filename);
	exit(113);
    }
    fread(s->FProgram,sizeof(char),file.st_size,fh);
    fclose(fh);
#if 0
    char filename2[]="../src/video_display/gl.vert";
    stat(filename2,&file);
    s->VProgram=calloc(file.st_size+1,sizeof(char));
    fh=fopen(filename2, "r");
    if(!fh){
    	perror(filename2);
 	exit(113);
    }
    fread(s->VProgram,sizeof(char),file.st_size,fh);
    fclose(fh);
#endif

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
        fprintf(stderr, "ERROR: Neither OpenGL 2.0 nor ARB_fragment_shader are supported, try updating your drivers...\n");
        exit(65);
    }

    bpp = get_bpp(hd_color_spc);

    /* Check to see if we have data yet, if not, just chillax */
    /* TODO: we need some solution (TM) for sem_getvalue on MacOS X */
#ifndef HAVE_MACOSX
    sem_getvalue(&s->semaphore,&i);
    while(i<1) {
  	display_gl_handle_events(s);
	usleep(1000);
	sem_getvalue(&s->semaphore,&i);
    }
#endif /* HAVE_MACOSX */

    while(1) {
        GLubyte *line1, *line2;
        display_gl_handle_events(s);
        platform_sem_wait(&s->semaphore);

	
	/* 10-bit YUV ->8 bit YUV [I think...] */
	line1 = s->buffers[s->image_display];
	line2 = s->outbuffer;
	if (bitdepth == 10) {	
		for(j=0;j<HD_HEIGHT;j+=2){
#if (HAVE_MACOSX || HAVE_32B_LINUX)
		    gl_copyline64(line2, line1, 5120/32);
		    gl_copyline64(line2+3840, line1+5120*540, 5120/32);
#else /* (HAVE_MACOSX || HAVE_32B_LINUX) */			
		    gl_copyline128(line2, line1, 5120/32);
		    gl_copyline128(line2+3840, line1+5120*540, 5120/32);
#endif /* HAVE_MACOSX || HAVE_32B_LINUX) */ 		    
		    line1 += 5120;
		    line2 += 2*3840;
		}
	} else {
		if (progressive == 1) {
			memcpy(line2, line1, hd_size_x*hd_size_y*bpp);
		} else {
			for(i=0; i<1080; i+=2) {       
				memcpy(line2, line1, (int)hd_size_x*bpp);
				memcpy(line2+(int)(hd_size_x*bpp), line1+(int)(hd_size_x*bpp*540), (int)(hd_size_x*bpp));
				line1 += (int)(hd_size_x*bpp);
				line2 += (int)(2*hd_size_x*bpp);
			}
		}
	}

        // gl_deinterlace(s->outbuffer);
	getY(s->outbuffer,s->y,s->u,s->v);
        gl_bind_texture(s);
        gl_draw(s);

		/* FPS Data, this is pretty ghetto though.... */
		Frames++;
		{
			GLint t = SDL_GetTicks();
			if (t - T0 >= 5000) {
			GLfloat seconds = (t - T0) / 1000.0;
			GLfloat fps = Frames / seconds;
			fprintf(stderr, "%d frames in %g seconds = %g FPS\n", (int)Frames, seconds, fps);
			T0 = t;
			Frames = 0;
			}
		}
    }
    return NULL;

}

/* linear blend deinterlace */
void gl_deinterlace(GLubyte *buffer)
{
        int i,j;
        long pitch = HD_WIDTH*2;
        register long pitch2 = pitch*2;
        GLubyte *bline1, *bline2, *bline3;
        register GLubyte *line1, *line2, *line3;

        bline1 = buffer;
        bline2 = buffer + pitch;
        bline3 = buffer + 3*pitch;
        for(i=0; i < HD_WIDTH*2; i+=16) {
                /* preload first two lines */
                asm volatile(
                             "movdqa (%0), %%xmm0\n"
                             "movdqa (%1), %%xmm1\n"
                             :
                             : "r" ((unsigned long *)bline1),
                               "r" ((unsigned long *)bline2));
                line1 = bline2;
                line2 = bline2 + pitch;
                line3 = bline3;
                for(j=0; j < 1076; j+=2) {
                        asm  volatile(
                              "movdqa (%1), %%xmm2\n"
                              "pavgb %%xmm2, %%xmm0\n"
                              "pavgb %%xmm1, %%xmm0\n"
                              "movdqa (%2), %%xmm1\n"
                              "movdqa %%xmm0, (%0)\n"
                              "pavgb %%xmm1, %%xmm0\n"
                              "pavgb %%xmm2, %%xmm0\n"
                              "movdqa %%xmm0, (%1)\n"
                              :
                              :"r" ((unsigned long *)line1),
                               "r" ((unsigned long *)line2),
                               "r" ((unsigned long *)line3)
                              );
                        line1 += pitch2;
                        line2 += pitch2;
                        line3 += pitch2;
                }
                bline1 += 16;
                bline2 += 16;
                bline3 += 16;
        }
}

void gl_resize_window(int width,int height)
{
    /* Height / width ration */
    GLfloat ratio;
    GLint   y = 0;

    /* Protect against a divide by zero */
    if ( height == 0 )
        height = 1;

    if (height > HD_HEIGHT) {
      y = (height - HD_HEIGHT) / 2;
      height = HD_HEIGHT;
    }
    ratio = ( GLfloat )width / ( GLfloat )(((float)(width * HD_HEIGHT))/((float)HD_WIDTH));

    glViewport( 0, y, ( GLint )width, ( GLint )height );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity( );


    glScalef(1, (((float)(width * HD_HEIGHT))/((float)HD_WIDTH))/((float)height), 1);
    gluPerspective( 45.0f, ratio, 0.1f, 100.0f );

    glMatrixMode( GL_MODELVIEW );

    glLoadIdentity( );
}

void gl_bind_texture(void *arg)
{
    struct state_sdl        *s = (struct state_sdl *) arg;
	int i;

    glActiveTexture(GL_TEXTURE1);
    i=glGetUniformLocationARB(s->PHandle,"Utex");
    glUniform1iARB(i,1); 
    glBindTexture(GL_TEXTURE_2D,1);
    glTexImage2D(GL_TEXTURE_2D,0,1,HD_WIDTH/2,HD_HEIGHT,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->u);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glActiveTexture(GL_TEXTURE2);
    i=glGetUniformLocationARB(s->PHandle,"Vtex");
    glUniform1iARB(i,2); 
    glBindTexture(GL_TEXTURE_2D,2);
    glTexImage2D(GL_TEXTURE_2D,0,1,HD_WIDTH/2,HD_HEIGHT,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->v);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glActiveTexture(GL_TEXTURE0);
    i=glGetUniformLocationARB(s->PHandle,"Ytex");
    glUniform1iARB(i,0); 
    glBindTexture(GL_TEXTURE_2D,0);
    glTexImage2D(GL_TEXTURE_2D,0,1,HD_WIDTH,HD_HEIGHT,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,s->y);
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

void gl_draw()
{
    /* Clear the screen */
    //glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity( );
    glTranslatef( 0.0f, 0.0f, -1.35f );

    gl_check_error();
    glBegin(GL_QUADS);
      /* Front Face */
      /* Bottom Left Of The Texture and Quad */
      glTexCoord2f( 0.0f, 1.0f ); glVertex2f( -1.0f, -0.5625f);
      /* Bottom Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 1.0f ); glVertex2f(  1.0f, -0.5625f);
      /* Top Right Of The Texture and Quad */
      glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  0.5625f);
      /* Top Left Of The Texture and Quad */
      glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  0.5625f);
    glEnd( );

    gl_check_error();
    /* Draw it to the screen */
    SDL_GL_SwapBuffers( );
}

inline void gl_copyline64(GLubyte *dst, GLubyte *src, int len)
{
        register uint64_t *d, *s;

        register uint64_t a1,a2,a3,a4;

        d = (uint64_t *)dst;
        s = (uint64_t *)src;

        while(len-- > 0) {
		a1 = *(s++);
                a2 = *(s++);
                a3 = *(s++);
                a4 = *(s++);

                a1 = (a1 & 0xffffff) | ((a1 >> 8) & 0xffffff000000);
                a2 = (a2 & 0xffffff) | ((a2 >> 8) & 0xffffff000000);
                a3 = (a3 & 0xffffff) | ((a3 >> 8) & 0xffffff000000);
                a4 = (a4 & 0xffffff) | ((a4 >> 8) & 0xffffff000000);

                *(d++) = a1 | (a2 << 48);       /* 0xa2|a2|a1|a1|a1|a1|a1|a1 */
                *(d++) = (a2 >> 16)|(a3 << 32); /* 0xa3|a3|a3|a3|a2|a2|a2|a2 */
                *(d++) = (a3 >> 32)|(a4 << 16); /* 0xa4|a4|a4|a4|a4|a4|a3|a3 */
	}
}

#if !(HAVE_MACOSX || HAVE_32B_LINUX)

inline void gl_copyline128(GLubyte *d, GLubyte *s, int len)
{
        register GLubyte *_d=d,*_s=s;

        while(--len >= 0) {

                asm ("movd %0, %%xmm4\n": : "r" (0xffffff));

                asm volatile ("movdqa (%0), %%xmm0\n"
                        "movdqa 16(%0), %%xmm5\n"
                        "movdqa %%xmm0, %%xmm1\n"
                        "movdqa %%xmm0, %%xmm2\n"
                        "movdqa %%xmm0, %%xmm3\n"
                        "pand  %%xmm4, %%xmm0\n"
                        "movdqa %%xmm5, %%xmm6\n"
                        "movdqa %%xmm5, %%xmm7\n"
                        "movdqa %%xmm5, %%xmm8\n"
                        "pand  %%xmm4, %%xmm5\n"
                        "pslldq $4, %%xmm4\n"
                        "pand  %%xmm4, %%xmm1\n"
                        "pand  %%xmm4, %%xmm6\n"
                        "pslldq $4, %%xmm4\n"
                        "psrldq $1, %%xmm1\n"
                        "psrldq $1, %%xmm6\n"
                        "pand  %%xmm4, %%xmm2\n"
                        "pand  %%xmm4, %%xmm7\n"
                        "pslldq $4, %%xmm4\n"
                        "psrldq $2, %%xmm2\n"
                        "psrldq $2, %%xmm7\n"
                        "pand  %%xmm4, %%xmm3\n"
                        "pand  %%xmm4, %%xmm8\n"
                        "por %%xmm1, %%xmm0\n"
                        "psrldq $3, %%xmm3\n"
                        "psrldq $3, %%xmm8\n"
                        "por %%xmm2, %%xmm0\n"
                        "por %%xmm6, %%xmm5\n"
                        "por %%xmm3, %%xmm0\n"
                        "por %%xmm7, %%xmm5\n"
                        "movdq2q %%xmm0, %%mm0\n"
                        "por %%xmm8, %%xmm5\n"
                        "movdqa %%xmm5, %%xmm1\n"
                        "pslldq $12, %%xmm5\n"
                        "psrldq $4, %%xmm1\n"
                        "por %%xmm5, %%xmm0\n"
                        "psrldq $8, %%xmm0\n"
                        "movq %%mm0, (%1)\n"
                        "movdq2q %%xmm0, %%mm1\n"
                        "movdq2q %%xmm1, %%mm2\n"
                        "movq %%mm1, 8(%1)\n"
                        "movq %%mm2, 16(%1)\n"
                        :
                        : "r" (_s), "r" (_d));
                _s += 32;
                _d += 24;
        }
}

#endif /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

display_type_t *display_gl_probe(void)
{
        display_type_t          *dt;
        display_format_t        *dformat;


        dformat = malloc(4 * sizeof(display_format_t));
        dformat[0].size        = DS_176x144;
        dformat[0].colour_mode = DC_YUV;
        dformat[0].num_images  = 1;
        dformat[1].size        = DS_352x288;
        dformat[1].colour_mode = DC_YUV;
        dformat[1].num_images  = 1;
        dformat[2].size        = DS_702x576;
        dformat[2].colour_mode = DC_YUV;
        dformat[2].num_images  = 1;
        dformat[3].size        = DS_1280x720;
        dformat[3].colour_mode = DC_YUV;
        dformat[3].num_images  = 1;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id          = DISPLAY_GL_ID;
                dt->name        = "gl";
                dt->description = "OpenGL using SDL";
                dt->formats     = dformat;
                dt->num_formats = 4;
        }
        return dt;
}

void display_gl_done(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;

        assert(s->magic == MAGIC_GL);

        SDL_ShowCursor(SDL_ENABLE);

        SDL_Quit();
}
	
char* display_gl_getf(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;
        assert(s->magic == MAGIC_GL);
        return (char *)s->buffers[s->image_network];
}

int display_gl_putf(void *state, char *frame)
{
        int tmp;
        struct state_sdl *s = (struct state_sdl *) state;

        assert(s->magic == MAGIC_GL);
        UNUSED(frame);

        /* ...and give it more to do... */
        tmp = s->image_display;
        s->image_display = s->image_network;
        s->image_network = tmp;

        /* ...and signal the worker */
        platform_sem_post(&s->semaphore);
        sem_getvalue(&s->semaphore, &tmp);
        if(tmp > 1)
                printf("frame drop!\n");
        return 0;
}

display_colour_t display_gl_colour(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;
        assert(s->magic == MAGIC_GL);
        return DC_YUV;
}

int display_gl_handle_events(void *state)
{
        SDL_Event       sdl_event;
	
	UNUSED(state);
	
        while (SDL_PollEvent(&sdl_event)) {
                switch (sdl_event.type) {
                        case SDL_KEYDOWN:
                        case SDL_KEYUP:
                                if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
                                        kill(0, SIGINT);
								}
                                break;

                        default:
                                break;
                }
        }

        return 0;

}

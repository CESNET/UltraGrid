//Standard warning crap
//
//
//
//

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
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
#include <math.h>
#include "debug.h"
#include "video_display.h"
#include "video_display/dxt.h"
// #define glGetProcAddress(n) glXGetProcAddressARB((GLubyte *) n)


#define degree_to_radian(x) ( M_PI * x / 180.0f )
#define radian_to_degree(x) ( x * (180.0f / M_PI) )

#define HD_WIDTH        1920
#define HD_HEIGHT       1080
#define MAGIC_DXT       DISPLAY_DXT_ID
#define SIZE		1036800


struct state_sdl {
        Display         *display;
	unsigned int    x_res_x;
	unsigned int    x_res_y;

        int             vw_depth;
        SDL_Overlay     *vw_image;
        GLubyte         *buffers[2];
	GLubyte		*outbuffer;
	GLubyte		*y, *u, *v;	//Guess what this might be...
	GLhandleARB	VSHandle,FSHandle,PHandle;
        int             image_display, image_network;
	GLuint  	texture[4];
        /* Thread related information follows... */
        pthread_t       thread_id;
        sem_t           semaphore;
        /* For debugging... */
        uint32_t        magic;

        SDL_Surface     *sdl_screen;
        SDL_Rect        rect;

	char		*FProgram,*VProgram;
};

/* Prototyping */
static void * display_thread_dxt(void *arg);
void dxt_resize_window(int width, int height);
void dxt_bind_texture(void *args);
void dxt_draw();
void * display_dxt_init(void);
void dxt_arb_init(void *arg);
void glsl_dxt_init(void *arg);
void dxt_loadShader(void *arg, char *filename);
void dxt_draw();
	
#ifdef DEBUG
void dxt_check_error()
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
				fprintf(stderr, "wft mate? Unknown GL ERROR: %p\n",msg);
				break;
		}
		msg=glGetError();
	}
	if(flag)
		exit(1);
}
#endif /* DEBUG */

void * display_dxt_init(void)
{
    struct state_sdl        *s;

    int                 ret;
    int                 itemp;
    unsigned int        utemp;
    Window              wtemp;

    s = (struct state_sdl *) calloc(1,sizeof(struct state_sdl));
    s->magic   = MAGIC_DXT;

    if (!(s->display = XOpenDisplay(NULL))) {
	    printf("Unable to open display DXT: XOpenDisplay.\n");
	    return NULL;
    }

    /* Get XWindows resolution */
    ret = XGetGeometry(s->display, DefaultRootWindow(s->display), &wtemp, &itemp, &itemp, &(s->x_res_x), &(s->x_res_y), &utemp, &utemp);
    
    s->rect.w = HD_WIDTH;
    s->rect.h =HD_HEIGHT;
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

    asm("emms\n");

    platform_sem_init(&s->semaphore, 0, 0);
    if (pthread_create(&(s->thread_id), NULL, display_thread_dxt, (void *) s) != 0) {
        perror("Unable to create display thread\n");
        return NULL;
    }

    return (void*)s;
}

void dxt_loadShader(void *arg, char *filename)
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

void dxt_arb_init(void *arg)
{
    struct state_sdl        *s = (struct state_sdl *) arg;
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

void glsl_dxt_init(void *arg)
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

void dxt_resize_window(int width, int height)
{
    /* Height / width ration */
    GLfloat ratio;
    GLint   y = 0;

    /* Protect against a divide by zero */
    if ( height == 0 )
        height = 1;


    if ((height > HD_HEIGHT) && (width >= HD_WIDTH)) {
        y = (height - HD_HEIGHT) / 2;
        height = HD_HEIGHT;
    }
    ratio = ( GLfloat )width / ( GLfloat )(((float)(width * HD_HEIGHT))/((float)HD_WIDTH));

    glViewport( 0, y, ( GLint )width, (GLint)height);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity( );

    glScalef(1, (((float)(width * HD_HEIGHT))/((float)HD_WIDTH))/((float)height), 1);
    gluPerspective(45.0f, ratio, 0.1f, 100.0f);

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity( );
}

void dxt_bind_texture(void *arg)
{
	struct state_sdl        *s = (struct state_sdl *) arg;
	static int i=0;

	//TODO: does OpenGL use different stuff here?
	glActiveTexture(GL_TEXTURE0);
	i=glGetUniformLocationARB(s->PHandle,"yuvtex");
	glUniform1iARB(i,0); 
	glBindTexture(GL_TEXTURE_2D,0);
	glCompressedTexImage2D(GL_TEXTURE_2D, 0,GL_COMPRESSED_RGB_S3TC_DXT1_EXT,1920,1080, 0,(1920*1080/16)*8, s->buffers[s->image_display]);

}    

void dxt_draw()
{
    glLoadIdentity( );
    glTranslatef( 0.0f, 0.0f, -1.35f );

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

    SDL_GL_SwapBuffers( );
}

static void * display_thread_dxt(void *arg)
{
    struct state_sdl        *s = (struct state_sdl *) arg;
#ifndef HAVE_MACOSX
    int i;
#endif /* HAVE_MACOSX */
    const SDL_VideoInfo *videoInfo;
    int videoFlags;
    /* FPS */
    static GLint T0     = 0;
    static GLint Frames = 0;

#ifdef HAVE_MACOSX
            /* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
    	    /* Avoids uncaught exception (1002)  when creating CGSWindow */
            NSApplicationLoad();
#endif

    /* initialize SDL */
    if ( SDL_Init( SDL_INIT_VIDEO ) < 0 ) {
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
//    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );      //TODO: Is this necessary?

    //    SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
      //  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
        //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
    //    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,  16);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    /* get a SDL surface */
    s->sdl_screen = SDL_SetVideoMode(s->x_res_x, s->x_res_y, 32, videoFlags);
    if(!s->sdl_screen){
        fprintf(stderr,"Error setting video mode %dx%d!\n", s->x_res_x, s->x_res_y);
        exit(128);
    }

    SDL_WM_SetCaption("Ultragrid - Form Of DXT!", "Ultragrid");

    SDL_ShowCursor(SDL_DISABLE);

        /* OpenGL Setup */
    glEnable( GL_TEXTURE_2D );
    glGenTextures(1, s->texture);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    //glClearColor( 0, 1, 0, 0 );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    dxt_resize_window(s->x_res_x, s->x_res_y);
    glewInit();
    if(glewIsSupported("GL_VERSION_2_0")){
        fprintf(stderr, "OpenGL 2.0 is supported...\n");
	}
    /* Load shader */
    //TODO: Need a less breaky way to do this...
	struct stat file;
	char *filename=strdup("../src/video_display/dxt.frag");
	if (stat(filename,&file) != 0) {
	    filename=strdup("/usr/share/uv-0.3.1/dxt.frag");
	    if (stat(filename,&file) != 0) {
		filename=strdup("/usr/local/share/uv-0.3.1/dxt.frag");
		if (stat(filename,&file) != 0) {
		    fprintf(stderr, "dxt.frag not found. Giving up!\n");
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

	char *filename2=strdup("../src/video_display/dxt.vert");
        if (stat(filename2,&file) != 0) {
	    filename2=strdup("/usr/share/uv-0.3.1/dxt.vert");
	    if (stat(filename2,&file) != 0) {
		filename2=strdup("/usr/local/share/uv-0.3.1/dxt.vert");
		if (stat(filename2,&file) != 0) {
		    fprintf(stderr, "dxt.vert not found. Giving up!\n");
		    exit(113);
		}
	    }
	    
	}
        s->VProgram=calloc(file.st_size+1,sizeof(char));
	
        fh=fopen(filename2, "r");
        if(!fh){
                perror(filename2);
                exit(113);
        }
        fread(s->VProgram,sizeof(char),file.st_size,fh);
        fclose(fh);

    /* Check to see if OpenGL 2.0 is supported, if not use ARB (if supported) */
    glewInit();
    if(glewIsSupported("GL_VERSION_2_0")){
        fprintf(stderr, "OpenGL 2.0 is supported...\n");
                //TODO: Re-enable dxt_init!
                //glsl_dxt_init(s);
	dxt_arb_init(s);
    }else if(GLEW_ARB_fragment_shader){
        fprintf(stderr, "OpenGL 2.0 not supported, using ARB extension...\n");
        dxt_arb_init(s);
    }else{
        fprintf(stderr, "ERROR: Neither OpenGL 2.0 nor ARB_fragment_shader are supported, try updating your drivers...\n");
        exit(65);
    }

    /* Check to see if we have data yet, if not, just chillax */
    /* TODO: we need some solution (TM) for sem_getvalue on MacOS X */

#ifndef HAVE_MACOSX
    sem_getvalue(&s->semaphore,&i);
    while(i<1) {
        display_dxt_handle_events(s);
        usleep(1000);
        sem_getvalue(&s->semaphore,&i);
    }
#endif /* HAVE_MACOSX */

    while(1) {
        display_dxt_handle_events(s);
        platform_sem_wait(&s->semaphore);

        dxt_bind_texture(s);
        dxt_draw(s);

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


display_type_t *display_dxt_probe(void)
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
                dt->id          = DISPLAY_DXT_ID;
                dt->name        = "dxt";
                dt->description = "OpenGL With DXT Compression";
                dt->formats     = dformat;
                dt->num_formats = 4;
        }
        return dt;
}

void display_dxt_done(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;

        assert(s->magic == MAGIC_DXT);

        SDL_ShowCursor(SDL_ENABLE);

        SDL_Quit();
}
	
char* display_dxt_getf(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;
        assert(s->magic == MAGIC_DXT);
        return (char *)s->buffers[s->image_network];
}

int display_dxt_putf(void *state, char *frame)
{
        int tmp;
        struct state_sdl *s = (struct state_sdl *) state;

        assert(s->magic == MAGIC_DXT);
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

display_colour_t display_dxt_colour(void *state)
{
        struct state_sdl *s = (struct state_sdl *) state;
        assert(s->magic == MAGIC_DXT);
        return DC_YUV;
}

int display_dxt_handle_events(void *state)
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

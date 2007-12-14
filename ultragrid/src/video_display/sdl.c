/*
 * FILE:   video_display/sdl.c
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 * $Revision: 1.6 $
 * $Date: 2007/12/14 16:18:29 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef X_DISPLAY_MISSING /* Don't try to compile if X is not present */

#include "debug.h"
#include "video_display.h"
#include "video_display/sdl.h"

/* For X shared memory... */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/Xv.h>
#include <X11/extensions/Xvlib.h>
#include <X11/extensions/XShm.h>
#include <host.h>
#ifdef HAVE_MACOSX
#include <architecture/i386/io.h>
#else /* HAVE_MACOSX */
#include <sys/io.h>
#include <AppKit/AppKit.h>
#endif /* HAVE_MACOSX */
#include <sys/time.h>
#include <semaphore.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>

inline void copyline64(unsigned char *dst, unsigned char *src, int len);
inline void copyline128(unsigned char *dst, unsigned char *src, int len);
void deinterlace(unsigned char *buffer);
extern int      XShmQueryExtension(Display*);
extern int      XShmGetEventBase(Display*);
extern XvImage  *XvShmCreateImage(Display*, XvPortID, int, char*, int, int, XShmSegmentInfo*);

#define MAGIC_SDL	DISPLAY_SDL_ID

extern long frame_begin[2];

struct state_sdl {
	Display			*display;
	Window			 window;
	GC			 gc;
	int			 vw_depth;
	Visual			*vw_visual;
	SDL_Overlay		*vw_image;
	char			*buffers[2];
	XShmSegmentInfo		 vw_shm_segment[2];
	int			 image_display, image_network;
	XvAdaptorInfo		*ai;
	int			 xv_port;
	/* Thread related information follows... */
	pthread_t		 thread_id;
	pthread_mutex_t		 lock;
	pthread_cond_t		 boss_cv;
	pthread_cond_t		 worker_cv;
	sem_t			 semaphore;
	int			 work_to_do;
	int			 boss_waiting;
	int			 worker_waiting;
	/* For debugging... */
	uint32_t		 magic;	

	SDL_Surface		*sdl_screen;
	SDL_Rect		rect;
};

int
display_sdl_handle_events(void)
{
	SDL_Event	sdl_event;
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


/* linear blend deinterlace */
void
deinterlace(unsigned char *buffer)
{
	int i,j;
        long pitch = 1920*2;
	register long pitch2 = pitch*2;
	unsigned char *bline1, *bline2, *bline3;
	register unsigned char *line1, *line2, *line3;

	bline1 = buffer;
	bline2 = buffer + pitch;
	bline3 = buffer + 3*pitch; 
        for(i=0; i < 1920*2; i+=16) {
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

inline void
copyline64(unsigned char *dst, unsigned char *src, int len)
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

/* convert 10bits Cb Y Cr A Y Cb Y A to 8bits Cb Y Cr Y Cb Y */

#ifndef HAVE_MACOSX

inline void
copyline128(unsigned char *d, unsigned char *s, int len)
{
	register unsigned char *_d=d,*_s=s;

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

#endif /* HAVE_MACOSX */


static void*
display_thread_sdl(void *arg)
{
	struct state_sdl        *s = (struct state_sdl *) arg;
	struct timeval tv, tv1;
	int tv2;
	int i;

	while (1) {
		char *line1, *line2;
		display_sdl_handle_events();

		sem_wait(&s->semaphore);

		assert(s->vw_image != NULL);

		line1 = s->buffers[s->image_display];
		line2 = *s->vw_image->pixels;
	
		gettimeofday(&tv1, NULL);
		if (bitdepth == 10) {	
			for(i=0; i<1080; i+=2) {
#ifdef HAVE_MACOSX
				copyline64(line2, line1, 5120/32);
				copyline64(line2+3840, line1+5120*540, 5120/32);
#else /* HAVE_MACOSX */
				copyline128(line2, line1, 5120/32);
				copyline128(line2+3840, line1+5120*540, 5120/32);
#endif /* HAVE_MACOSX */
				line1 += 5120;
				line2 += 2*3840;
			}
		} else {
			if (progressive == 1) {
				memcpy(line2, line1, hd_size_x*hd_size_y*hd_color_bpp);
			} else {
				for(i=0; i<1080; i+=2) {
					memcpy(line2, line1, hd_size_x*hd_color_bpp);	
					memcpy(line2+hd_size_x*hd_color_bpp, line1+hd_size_x*hd_color_bpp*540, hd_size_x*hd_color_bpp);
					line1 += hd_size_x*hd_color_bpp;
					line2 += 2*hd_size_x*hd_color_bpp;
				}
			}
		}

		deinterlace(*s->vw_image->pixels);

		SDL_UnlockYUVOverlay(s->vw_image);

		SDL_DisplayYUVOverlay(s->vw_image, &(s->rect));

		SDL_LockYUVOverlay(s->vw_image);
               
		gettimeofday(&tv, NULL);

	       	tv2 = tv.tv_usec - tv1.tv_usec;
                if(tv2 < 0)
                        tv2 += 1000000;

                // printf("FPS: %f\n", 1000000.0/tv2);

	}
	return NULL;
}


void *
display_sdl_init(void)
{
	struct state_sdl	*s;
	int			ret;

        SDL_Surface             *image;
        SDL_Surface             *temp;
        SDL_Rect                splash_src;
        SDL_Rect                splash_dest;

	int			itemp;
	unsigned int		utemp;
	Window			wtemp;

	unsigned int		x_res_x;
	unsigned int		x_res_y;

	s = (struct state_sdl *) malloc(sizeof(struct state_sdl));
	s->magic   = MAGIC_SDL;

	//iopl(3);

	asm("emms\n");

	pthread_mutex_init(&s->lock, NULL);
	pthread_cond_init(&s->boss_cv, NULL);
	pthread_cond_init(&s->worker_cv, NULL);
	sem_init(&s->semaphore, 0, 0);
	s->work_to_do     = FALSE;
	s->boss_waiting   = FALSE;
	s->worker_waiting = TRUE;

	debug_msg("Window initialized %p\n", s);

	if (!(s->display = XOpenDisplay(NULL))) {
                printf("Unable to open display.\n");
                return NULL;
	}

#ifdef HAVE_MACOSX
	/* Startup function to call when running Cocoa code from a Carbon application. Whatever the fuck that means. */
	/* Avoids uncaught exception (1002)  when creating CGSWindow */
	NSApplicationLoad();
#endif

	ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE);
	if (ret < 0) {
		printf("Unable to initialize SDL.\n");
		return NULL;
	}

	/* Get XWindows resolution */
	ret = XGetGeometry(s->display, DefaultRootWindow(s->display), &wtemp, &itemp, &itemp, &x_res_x, &x_res_y, &utemp, &utemp);

	
        fprintf(stdout,"Setting video mode %dx%d.\n", x_res_x, x_res_y);
	s->sdl_screen = SDL_SetVideoMode(x_res_x, x_res_y, 0, SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_FULLSCREEN);
        if(s->sdl_screen == NULL){
            fprintf(stderr,"Error setting video mode %dx%d!\n", x_res_x, x_res_y);
            exit(128);
        }

	SDL_WM_SetCaption("Ultragrid - SDL Display", "Ultragrid");

	SDL_ShowCursor(SDL_DISABLE);

#define FOURCC_UYVY  0x59565955

	s->vw_image = SDL_CreateYUVOverlay(hd_size_x, hd_size_y, FOURCC_UYVY, s->sdl_screen);
	if (s->vw_image == NULL) {
		printf("SDL_overlay initialization failed.\n");
		exit(127);
	}

	s->buffers[0] = malloc(hd_size_x*hd_size_y*hd_color_bpp);
	s->buffers[1] = malloc(hd_size_x*hd_size_y*hd_color_bpp);

	s->rect.w = hd_size_x;
	s->rect.h = hd_size_y;
	if (x_res_x > hd_size_x) {
		s->rect.x = (x_res_x - hd_size_x) / 2;
	} else {
		s->rect.x = 0;
	}
	if (x_res_y > hd_size_y) {
		s->rect.y = (x_res_y - hd_size_y) / 2;
	} else {
		s->rect.y = 0;
	}
        fprintf(stdout,"Setting SDL rect %dx%d - %d,%d.\n", s->rect.w, s->rect.h, s->rect.x, s->rect.y);

	s->image_network = 0;
        s->image_display = 1;

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
                image = SDL_DisplayFormat(temp);
                SDL_FreeSurface(temp);
 
                splash_src.x = 0;
                splash_src.y = 0;
                splash_src.w = image->w;
                splash_src.h = image->h;
 
                splash_dest.x = (int)((x_res_x - splash_src.w) / 2);
                splash_dest.y = (int)((x_res_y - splash_src.h) / 2) + 60;
                splash_dest.w = image->w;
                splash_dest.h = image->h;
 
                SDL_BlitSurface(image, &splash_src, s->sdl_screen, &splash_dest);
                SDL_Flip(s->sdl_screen);
        }

	if (pthread_create(&(s->thread_id), NULL, display_thread_sdl, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	return (void *)s;

}

void
display_sdl_done(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;

	assert(s->magic == MAGIC_SDL);

	SDL_ShowCursor(SDL_ENABLE);

	SDL_Quit();
}

char *
display_sdl_getf(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;
	assert(s->magic == MAGIC_SDL);
	assert(s->buffers[s->image_network] != NULL);
	return (char *)s->buffers[s->image_network];
}

int
display_sdl_putf(void *state, char *frame)
{
	int		 tmp;
	struct state_sdl *s = (struct state_sdl *) state;

	assert(s->magic == MAGIC_SDL);
	assert(frame != NULL);

	/* ...and give it more to do... */
	tmp = s->image_display;
	s->image_display = s->image_network;
	s->image_network = tmp;
	s->work_to_do    = TRUE;

	/* ...and signal the worker */
	sem_post(&s->semaphore);
	sem_getvalue(&s->semaphore, &tmp);
	if(tmp > 1) 
		printf("frame drop!\n");
	return 0;
}

display_colour_t
display_sdl_colour(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;
	assert(s->magic == MAGIC_SDL);
	return DC_YUV;
}

display_type_t *
display_sdl_probe(void)
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
		dt->id	        = DISPLAY_SDL_ID;
		dt->name        = "sdl";
		dt->description = "SDL with Xvideo extension";
                dt->formats     = dformat;
                dt->num_formats = 4;
        }
        return dt;
}

#endif /* X_DISPLAY_MISSING */

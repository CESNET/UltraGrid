/*
 * FILE:    video_display/sdl.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
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
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "tv.h"
#include "video_codec.h"

#ifdef HAVE_MACOSX
#include <architecture/i386/io.h>
void NSApplicationLoad();
#else                           /* HAVE_MACOSX */
#include <sys/io.h>
#endif                          /* HAVE_MACOSX */
#include <sys/time.h>

#include <math.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>
#include <SDL/SDL_mutex.h>

/* splashscreen (xsedmik) */
#include "video_display/splashscreen.h"

#define MAGIC_SDL   DISPLAY_SDL_ID
#define FOURCC_UYVY  0x59565955

struct state_sdl {
        struct video_frame      frame;

        pthread_t               thread_id;
        SDL_sem                 *semaphore;

        uint32_t                magic;

        struct timeval          tv;
        int                     frames;

        SDL_Overlay  * volatile yuv_image;
        SDL_Surface  * volatile sdl_screen;
        SDL_Rect                dst_rect;

        const struct codec_info_t *codec_info;

        unsigned                rgb:1;
        unsigned                deinterlace:1;
        unsigned                fs:1;

        volatile int            buffer_writable;
        SDL_cond                *buffer_writable_cond;
        SDL_mutex               *buffer_writable_lock;
};

extern int should_exit;

static void toggleFullscreen(struct state_sdl *s);
static void loadSplashscreen(struct state_sdl *s);
inline void copyline64(unsigned char *dst, unsigned char *src, int len);
inline void copyline128(unsigned char *dst, unsigned char *src, int len);
inline void copylinev210(unsigned char *dst, unsigned char *src, int len);
inline void copyliner10k(struct state_sdl *s, unsigned char *dst, unsigned char *src, int len);
void copylineRGBA(struct state_sdl *s, unsigned char *dst, unsigned char *src, int len);
void deinterlace(struct state_sdl *s, unsigned char *buffer);
static void show_help(void);
void cleanup_screen(struct state_sdl *s);
void reconfigure_screen(void *s, unsigned int width, unsigned int height,
                        codec_t codec, double fps, int aux);
static void get_sub_frame(void *s, int x, int y, int w, int h, struct video_frame *out);

extern int should_exit;

/** 
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format. Thereafter are data written into
 * the temporary SDL_Surface. At the end of the function are displayed on 
 * the screen. 
 * 
 * @since 18-02-2010, xsedmik
 * @param s Structure contains the current settings
 */
static void loadSplashscreen(struct state_sdl *s) {

	unsigned int 	x_coord;
	unsigned int 	y_coord;
	char 		pixel[3];
	SDL_Surface*	image;
	SDL_Rect 	splash_src;
	SDL_Rect	splash_dest;
        unsigned int x_res_x, x_res_y;
        
        const SDL_VideoInfo *video_info;
        
	video_info = SDL_GetVideoInfo();
        x_res_x = video_info->current_w;
        x_res_y = video_info->current_h;

        if(splash_height > s->frame.height || splash_width > s->frame.width)
                return;

	// create a temporary SDL_Surface with the settings of displaying surface
	image = SDL_DisplayFormat(s->sdl_screen);
#ifndef HAVE_MACOSX
	SDL_LockSurface(image);
#endif

	// load splash data
	for (y_coord = 0; y_coord < splash_height; y_coord++) {
		for (x_coord = 0; x_coord < splash_width; x_coord++) {

			HEADER_PIXEL(splash_data,pixel);
			Uint32 color = SDL_MapRGB(image->format, pixel[0], pixel[1], pixel[2]);

			switch(image->format->BytesPerPixel) {
				case 1: // Assuming 8-bpp 
				{
					Uint8 *bufp;
					bufp = (Uint8 *)image->pixels + y_coord*image->pitch + x_coord;
					*bufp = color;
				}
				break;

				case 2: // Probably 15-bpp or 16-bpp 
				{
					Uint16 *bufp;
					bufp = (Uint16 *)image->pixels + y_coord*image->pitch/2 + x_coord;
					*bufp = color;
				}
				break;

				case 3: // Slow 24-bpp mode, usually not used 
				{
					Uint8 *bufp;
					bufp = (Uint8 *)image->pixels + y_coord*image->pitch +
						x_coord*image->format->BytesPerPixel;
					*(bufp+image->format->Rshift/8) = pixel[0];
					*(bufp+image->format->Gshift/8) = pixel[1];
					*(bufp+image->format->Bshift/8) = pixel[2];
				}
				break;

				case 4: // Probably 32-bpp 
				{
					Uint32 *bufp;
					bufp = (Uint32 *)image->pixels + y_coord*image->pitch/4 + x_coord;
					*bufp = color;
				}
				break;
			}
		}
	}

#ifndef HAVE_MACOSX
	SDL_UnlockSurface(image);
#endif

	// place loaded splash on the right position (center of screen)
	splash_src.x = 0;
	splash_src.y = 0;
	splash_src.w = splash_width;
	splash_src.h = splash_height;

	if (s->fs) {
		splash_dest.x = ((int) x_res_x - splash_src.w) / 2;
		splash_dest.y = ((int) x_res_y - splash_src.h) / 2;
	
	}
	else {
		splash_dest.x = ((int) s->frame.width - splash_src.w) / 2;
		splash_dest.y = ((int) s->frame.height - splash_src.h) / 2;
	
	}
	splash_dest.w = splash_width;
	splash_dest.h = splash_height;

#ifndef HAVE_MACOSX
        SDL_UnlockSurface(s->sdl_screen);
#endif
        SDL_BlitSurface(image, &splash_src, s->sdl_screen, &splash_dest);
        SDL_Flip(s->sdl_screen);
#ifndef HAVE_MACOSX
        SDL_LockSurface(s->sdl_screen);
#endif
	SDL_FreeSurface(image);
}


/**
 * Function toggles between fullscreen and window display mode
 *
 * @since 23-03-2010, xsedmik
 * @param s Structure contains the current settings
 * @return zero value everytime
 */
static void toggleFullscreen(struct state_sdl *s) {
#ifndef HAVE_MACOSX
	if(s->fs) {
		s->fs = 0;
        }
        else {
		s->fs = 1;
        }
	/* and post for reconfiguration */
	s->frame.width = 0;
#endif
}

/**
 * Handles outer events like a keyboard press
 * Responds to key:<br/>
 * <table>
 * <td><tr>q</tr><tr>terminates program</tr></td>
 * <td><tr>f</tr><tr>toggles between fullscreen and windowed display mode</tr></td>
 * </table>
 *
 * @since 08-04-2010, xsedmik
 * @param arg Structure (state_sdl) contains the current settings
 * @return zero value everytime
 */
int display_sdl_handle_events(void *arg, int post)
{
        SDL_Event sdl_event;
        struct state_sdl *s = arg;
        while (SDL_PollEvent(&sdl_event)) {
                switch (sdl_event.type) {
                case SDL_KEYDOWN:
                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "d")) {
                                s->deinterlace = s->deinterlace ? FALSE : TRUE;
                                printf("Deinterlacing: %s\n", s->deinterlace ? "ON"
                                                : "OFF");
                                if(post)
                                        SDL_SemPost(s->semaphore);
                                return 1;
                        }

                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
                                should_exit = 1;
                                if(post)
                                        SDL_SemPost(s->semaphore);
                                exit(0);
                        }

                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "f")) {
				toggleFullscreen(s);
                                        if(post)
                                                SDL_SemPost(s->semaphore);
                                return 1;
                        }
                        break;
                case SDL_QUIT:
                        should_exit = 1;
                        if(post)
                                SDL_SemPost(s->semaphore);
                        exit(0);
                }
        }

        return 0;

}

void display_sdl_run(void *arg)
{
        struct state_sdl *s = (struct state_sdl *)arg;
        struct timeval tv;

        gettimeofday(&s->tv, NULL);

        while (!should_exit) {
                display_sdl_handle_events(s, 0);
#ifndef HAVE_MACOSX
                /* set flag to prevent dangerous actions */
                if(SDL_SemWaitTimeout(s->semaphore, 200) == SDL_MUTEX_TIMEDOUT) {
                        continue;
                }
#else
                if(SDL_SemTryWait(s->semaphore) == SDL_MUTEX_TIMEDOUT) {
                        usleep(1000);
                        continue;
                }
#endif

                if (s->deinterlace) {
                        if (s->rgb) {
                                /*FIXME: this will not work! Should not deinterlace whole screen, just subwindow */
                                vc_deinterlace(s->sdl_screen->pixels,
                                               s->frame.dst_linesize, s->frame.height);
                        } else {
                                vc_deinterlace(*s->yuv_image->pixels,
                                               s->frame.dst_linesize, s->frame.height);
                        }
                }

                if (s->rgb) {
#ifndef HAVE_MACOSX
                        SDL_UnlockSurface(s->sdl_screen);
#endif
                        SDL_Flip(s->sdl_screen);
#ifndef HAVE_MACOSX
                        SDL_LockSurface(s->sdl_screen);
#endif
                        s->frame.data = s->sdl_screen->pixels +
                            s->sdl_screen->pitch * s->dst_rect.y +
                            s->dst_rect.x *
                            s->sdl_screen->format->BytesPerPixel;
                } else {
#ifndef HAVE_MACOSX
                        SDL_UnlockYUVOverlay(s->yuv_image);
#endif
                        SDL_DisplayYUVOverlay(s->yuv_image, &(s->dst_rect));
#ifndef HAVE_MACOSX
			SDL_LockYUVOverlay(s->yuv_image);
#endif
		}

                SDL_mutexP(s->buffer_writable_lock);
                s->buffer_writable = 1;
                SDL_CondSignal(s->buffer_writable_cond);
                SDL_mutexV(s->buffer_writable_lock);

		s->frames++;
		gettimeofday(&tv, NULL);
		double seconds = tv_diff(tv, s->tv);
		if (seconds > 5) {
			double fps = s->frames / seconds;
			fprintf(stdout, "%d frames in %g seconds = %g FPS\n",
				s->frames, seconds, fps);
			s->tv = tv;
			s->frames = 0;
		}
	}
}

static void show_help(void)
{
        printf("SDL options:\n");
        printf("\twidth:height:codec[:fs][:i][:d][:f:filename] | help\n");
        printf("\tfs - fullscreen\n");
        printf("\td - deinterlace\n");
        printf("\tf filename - read frame content from the filename\n");
        show_codec_help("sdl");
}

void cleanup_screen(struct state_sdl *s)
{
        if (s->rgb == 0) {
                if (s->yuv_image != NULL) {
                        SDL_FreeYUVOverlay(s->yuv_image);
                        s->yuv_image = NULL;
                }
        }
        if (s->sdl_screen != NULL) {
                SDL_FreeSurface(s->sdl_screen);
                s->sdl_screen = NULL;
        }
}

void
reconfigure_screen(void *state, unsigned int width, unsigned int height,
	   codec_t color_spec, double fps, int aux)
{
	struct state_sdl *s = (struct state_sdl *)state;
	const SDL_VideoInfo *video_info;
	int h_align = 0;

	unsigned int x_res_x, x_res_y;
	unsigned int screen_x, screen_y;

	int i;

	/* wait until thread finishes displaying */
        SDL_mutexP(s->buffer_writable_lock);
        while (!s->buffer_writable)
                SDL_CondWait(s->buffer_writable_cond,
                                s->buffer_writable_lock);
        SDL_mutexV(s->buffer_writable_lock);

	cleanup_screen(s);

        s->frame.width = width;
        s->frame.height = height;

	s->frame.fps = fps;
	s->frame.aux = aux;

	fprintf(stdout, "Reconfigure to size %dx%d\n", s->frame.width,
			s->frame.height);

	video_info = SDL_GetVideoInfo();
        x_res_x = video_info->current_w;
        x_res_y = video_info->current_h;
        
	screen_x = x_res_x;
	screen_y = x_res_y;

	fprintf(stdout, "Setting video mode %dx%d.\n", x_res_x, x_res_y);
	if (s->fs)
        {
		s->sdl_screen =
		    SDL_SetVideoMode(x_res_x, x_res_y, 0,
				     SDL_FULLSCREEN | SDL_HWSURFACE |
				     SDL_DOUBLEBUF);
        } else {
		x_res_x = s->frame.width;
		x_res_y = s->frame.height;
		s->sdl_screen =
		    SDL_SetVideoMode(x_res_x, x_res_y, 0,
				     SDL_HWSURFACE | SDL_DOUBLEBUF);
	}
	if (s->sdl_screen == NULL) {
		fprintf(stderr, "Error setting video mode %dx%d!\n", x_res_x,
			x_res_y);
		free(s);
		exit(128);
	}
	SDL_WM_SetCaption("Ultragrid - SDL Display", "Ultragrid");

	SDL_ShowCursor(SDL_DISABLE);

	for (i = 0; codec_info[i].name != NULL; i++) {
		if (color_spec == codec_info[i].codec) {
			s->codec_info = &codec_info[i];
			s->rgb = codec_info[i].rgb;
			s->frame.src_bpp = codec_info[i].bpp;
			h_align = codec_info[i].h_align;
		}
	}
        assert(h_align != 0);

	s->frame.src_linesize = s->frame.width;
        s->frame.src_linesize = ((s->frame.src_linesize + h_align - 1) / h_align) * h_align;
        s->frame.src_linesize *= s->frame.src_bpp;

	if (s->rgb == 0) {
		s->yuv_image =
		    SDL_CreateYUVOverlay(s->frame.width, s->frame.height, FOURCC_UYVY,
						 s->sdl_screen);
                if (s->yuv_image == NULL) {
                        printf("SDL_overlay initialization failed.\n");
                        free(s);
                        exit(127);
                }
#ifndef HAVE_MACOSX
                SDL_LockYUVOverlay(s->yuv_image);
        } else {
                SDL_LockSurface(s->sdl_screen);
#endif
        }

        int w = s->frame.width;

        if (s->codec_info->h_align) {
                w = ((w + s->codec_info->h_align -
                      1) / s->codec_info->h_align) * s->codec_info->h_align;
        }

        if (s->rgb)
                s->frame.dst_linesize = s->frame.width * 4;
        else
                s->frame.dst_linesize = s->frame.width * 2;

        s->dst_rect.x = 0;
        s->dst_rect.y = 0;
        s->dst_rect.w = s->frame.width;
        s->dst_rect.h = s->frame.height;

	if(s->rgb) {
		if (x_res_x > s->frame.width) {
			s->dst_rect.x = ((int) x_res_x - s->frame.width) / 2;
		} else if (x_res_x < s->frame.width) {
			s->dst_rect.w = x_res_x;
		}
		if (x_res_y > s->frame.height) {
			s->dst_rect.y = ((int) x_res_y - s->frame.height) / 2;
		} else if (x_res_y < s->frame.height) {
			s->dst_rect.h = x_res_y;
		}
	} else if(!s->rgb && s->fs && (s->frame.width != x_res_x || s->frame.height != x_res_y)) {
		double frame_aspect = (double) s->frame.width / s->frame.height;
		double screen_aspect = (double) screen_x / screen_y;
		if(screen_aspect > frame_aspect) {
			s->dst_rect.h = screen_y;
			s->dst_rect.w = screen_y * frame_aspect;
			s->dst_rect.x = ((int) screen_x - s->dst_rect.w) / 2;
		} else {
			s->dst_rect.w = screen_x;
			s->dst_rect.h = screen_x / frame_aspect;
			s->dst_rect.y = ((int) screen_y - s->dst_rect.h) / 2;
		}
	}

        fprintf(stdout, "Setting SDL rect %dx%d - %d,%d.\n", s->dst_rect.w,
                s->dst_rect.h, s->dst_rect.x, s->dst_rect.y);

        s->frame.rshift = s->sdl_screen->format->Rshift;
        s->frame.gshift = s->sdl_screen->format->Gshift;
        s->frame.bshift = s->sdl_screen->format->Bshift;
        s->frame.color_spec = s->codec_info->codec;

        if (s->rgb) {
                s->frame.data = s->sdl_screen->pixels +
                    s->sdl_screen->pitch * s->dst_rect.y +
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
                s->frame.data_len =
                    (int) s->sdl_screen->pitch * x_res_y -
                    s->sdl_screen->pitch * s->dst_rect.y +
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
		s->frame.dst_bpp = s->sdl_screen->format->BytesPerPixel;
		s->frame.dst_pitch = s->sdl_screen->pitch;
                /* THIS SHOULDN'T work with current decoder !!! */
		/*s->frame.dst_x_offset =
		    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;*/
		s->frame.dst_x_offset = 0;
        } else {
                s->frame.data = (char *)*s->yuv_image->pixels;
                s->frame.data_len = s->frame.width * s->frame.height * 2;
                s->frame.dst_bpp = 2;
                s->frame.dst_pitch = s->frame.dst_linesize;
		s->frame.dst_x_offset = 0;
        }

        switch (color_spec) {
                case R10k:
                        s->frame.decoder = (decoder_t)vc_copyliner10k;
                        break;
                case v210:
                        s->frame.decoder = (decoder_t)vc_copylinev210;
                        break;
                case DVS10:
                        s->frame.decoder = (decoder_t)vc_copylineDVS10;
                        break;
                case DVS8:
                case UYVY:
                case Vuy2:
                        s->frame.decoder = (decoder_t)memcpy;
                        break;
                case RGBA:
                        s->frame.decoder = (decoder_t)vc_copylineRGBA;
                        break;
                case DXT1:
                        fprintf(stderr, "DXT1 isn't supported for SDL output.\n");
                        exit(EXIT_FAILURE);
        }
}

void *display_sdl_init(char *fmt)
{
        struct state_sdl *s;
        int ret;

        unsigned int i;

        s = (struct state_sdl *)calloc(1, sizeof(struct state_sdl));
        s->magic = MAGIC_SDL;

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
                        show_help();
                        free(s);
                        return NULL;
                }

                if (strcmp(fmt, "fs") == 0) {
                        s->fs = 1;
                        fmt = NULL;
                } else {

                        char *tmp = strdup(fmt);
                        char *tok;

                        tok = strtok(tmp, ":");
                        if (tok == NULL) {
                                show_help();
                                free(s);
                                free(tmp);
                                return NULL;
                        }
                        s->frame.width = atol(tok);
                        tok = strtok(NULL, ":");
                        if (tok == NULL) {
                                show_help();
                                free(s);
                                free(tmp);
                                return NULL;
                        }
                        s->frame.height = atol(tok);
                        tok = strtok(NULL, ":");
                        if (tok == NULL) {
                                show_help();
                                free(s);
                                free(tmp);
                                return NULL;
                        }
                        for (i = 0; codec_info[i].name != NULL; i++) {
                                if (strcmp(tok, codec_info[i].name) == 0) {
                                        s->codec_info = &codec_info[i];
                                        s->rgb = codec_info[i].rgb;
                                }
                        }
                        if (s->codec_info == NULL) {
                                fprintf(stderr, "SDL: unknown codec: %s\n",
                                        tok);
                                free(s);
                                free(tmp);
                                return NULL;
                        }
                        tok = strtok(NULL, ":");
                        while (tok != NULL) {
                                if (tok[0] == 'f' && tok[1] == 's') {
                                        s->fs = 1;
                                } else if (tok[0] == 'd') {
                                        s->deinterlace = 1;
                                }
                                tok = strtok(NULL, ":");
                        }
                        free(tmp);

                        if (s->frame.width <= 0 || s->frame.height <= 0) {
                                printf
                                    ("SDL: failed to parse config options: '%s'\n",
                                     fmt);
                                free(s);
                                return NULL;
                        }
                        s->frame.src_bpp = s->codec_info->bpp;
                        printf("SDL setup: %dx%d codec %s\n", s->frame.width,
                               s->frame.height, s->codec_info->name);
                }
        }

        asm("emms\n");

        s->semaphore = SDL_CreateSemaphore(0);
        s->buffer_writable = 1;
        s->buffer_writable_lock = SDL_CreateMutex();
        s->buffer_writable_cond = SDL_CreateCond();

#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. 
         * Whatever the fuck that means. 
         * Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif

        s->yuv_image = NULL;
        s->sdl_screen = NULL;

        ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE);
        if (ret < 0) {
                printf("Unable to initialize SDL.\n");
                return NULL;
        }

        if (fmt != NULL) {
                reconfigure_screen(s, s->frame.width, s->frame.height,
                                s->codec_info->codec, s->frame.fps,
                                s->frame.aux);
                loadSplashscreen(s);	
        } else {
                reconfigure_screen(s, 500, 500, RGBA, 30.0, 0);
                loadSplashscreen(s);	
        }

        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)reconfigure_screen;
        s->frame.get_sub_frame = (get_sub_frame_t) get_sub_frame;

        /*if (pthread_create(&(s->thread_id), NULL, 
                           display_thread_sdl, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/

        return (void *)s;
}

void display_sdl_done(void *state)
{
        struct state_sdl *s = (struct state_sdl *)state;

        assert(s->magic == MAGIC_SDL);

        SDL_DestroyCond(s->buffer_writable_cond);
        SDL_DestroyMutex(s->buffer_writable_lock);
	cleanup_screen(s);

        /*FIXME: free all the stuff */
        SDL_ShowCursor(SDL_ENABLE);

        SDL_Quit();
}

struct video_frame *display_sdl_getf(void *state)
{
        struct state_sdl *s = (struct state_sdl *)state;
        assert(s->magic == MAGIC_SDL);

        return &s->frame;
}

int display_sdl_putf(void *state, char *frame)
{
        int tmp;
        struct state_sdl *s = (struct state_sdl *)state;

        assert(s->magic == MAGIC_SDL);
        assert(frame != NULL);

        SDL_mutexP(s->buffer_writable_lock);
        s->buffer_writable = 0;
        SDL_mutexV(s->buffer_writable_lock);

        SDL_SemPost(s->semaphore);
        tmp = SDL_SemValue(s->semaphore);
        if (tmp > 1) {
                printf("%d frame(s) dropped!\n", tmp);
                SDL_SemTryWait(s->semaphore); /* decrement then */
        }
        return 0;
}

display_type_t *display_sdl_probe(void)
{
        display_type_t *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_SDL_ID;
                dt->name = "sdl";
                dt->description = "SDL";
        }
        return dt;
}

static void get_sub_frame(void *state, int x, int y, int w, int h, struct video_frame *out) 
{
        struct state_sdl *s = (struct state_sdl *)state;
        UNUSED(h);

        memcpy(out, &s->frame, sizeof(struct video_frame));
        out->data +=
                y * s->frame.dst_pitch;
        out->data_len -=
                y * s->frame.dst_pitch;
	out->dst_x_offset =
                (size_t) (x * (s->rgb ? 4 : 2));
        out->src_linesize =
                vc_getsrc_linesize(w, out->color_spec);
        out->dst_linesize =
                (x + w) * out->dst_bpp;
}


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
#include "audio/audio.h"
#include "audio/utils.h"
#include "utils/ring_buffer.h"
#include "video_codec.h"

#ifdef HAVE_MACOSX
#include <architecture/i386/io.h>
void NSApplicationLoad();
#else                           /* HAVE_MACOSX */
#include <sys/io.h>
#include "x11_common.h"
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
        struct video_frame     *frame;
        struct tile            *tile;
        
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
        int                     screen_w, screen_h;
        
        struct ring_buffer      *audio_buffer;
        struct audio_frame      audio_frame;
        unsigned int            play_audio:1;
        
        unsigned int            self_reconfigure:1;
        
        int                     rshift, gshift, bshift;
        int                     pitch;
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
static void configure_audio(struct state_sdl *s);
void display_sdl_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate);
                
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

        if(splash_height > tile_get(s->frame, 0, 0)->height || splash_width > tile_get(s->frame, 0, 0)->width)
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
		splash_dest.x = ((int) s->tile->width - splash_src.w) / 2;
		splash_dest.y = ((int) s->tile->height - splash_src.h) / 2;
	
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
        s->self_reconfigure = TRUE;
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
                                               vc_get_linesize(s->tile->width, s->frame->color_spec), s->tile->height);
                        } else {
                                vc_deinterlace(*s->yuv_image->pixels,
                                               vc_get_linesize(s->tile->width, s->frame->color_spec), s->tile->height);
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
                        s->tile->data = s->sdl_screen->pixels +
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
        printf("\t-d sdl[:fs][:d] | help\n");
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

void display_sdl_reconfigure(void *state, struct video_desc desc)
{
	struct state_sdl *s = (struct state_sdl *)state;

	unsigned int x_res_x, x_res_y;

	/* wait until thread finishes displaying */
        SDL_mutexP(s->buffer_writable_lock);
        while (!s->buffer_writable)
                SDL_CondWait(s->buffer_writable_cond,
                                s->buffer_writable_lock);
        SDL_mutexV(s->buffer_writable_lock);

	cleanup_screen(s);

        s->tile->width = desc.width;
        s->tile->height = desc.height;

	s->frame->fps = desc.fps;
	s->frame->aux = desc.aux;
	s->frame->color_spec = desc.color_spec;

	fprintf(stdout, "Reconfigure to size %dx%d\n", desc.width,
			desc.height);

	x_res_x = s->screen_w;
	x_res_y = s->screen_h;

	fprintf(stdout, "Setting video mode %dx%d.\n", x_res_x, x_res_y);
        int bpp;
        if(desc.color_spec == RGB) {
                bpp = 24;
        } else {
                bpp = 0; /* screen defautl */
        }
	if (s->fs)
        {
		s->sdl_screen =
		    SDL_SetVideoMode(x_res_x, x_res_y, bpp,
				     SDL_FULLSCREEN | SDL_HWSURFACE |
				     SDL_DOUBLEBUF);
        } else {
		x_res_x = s->tile->width;
		x_res_y = s->tile->height;
		s->sdl_screen =
		    SDL_SetVideoMode(x_res_x, x_res_y, bpp,
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

        
        s->rgb = codec_is_a_rgb(desc.color_spec);

	if (s->rgb == 0) {
		s->yuv_image =
		    SDL_CreateYUVOverlay(s->tile->width, s->tile->height, FOURCC_UYVY,
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

        if (s->rgb)
                s->tile->linesize = s->tile->width * 4;
        else
                s->tile->linesize = s->tile->width * 2;

        s->dst_rect.x = 0;
        s->dst_rect.y = 0;
        s->dst_rect.w = s->tile->width;
        s->dst_rect.h = s->tile->height;

	if(s->rgb) {
		if (x_res_x > s->tile->width) {
			s->dst_rect.x = ((int) x_res_x - s->tile->width) / 2;
		} else if (x_res_x < s->tile->width) {
			s->dst_rect.w = x_res_x;
		}
		if (x_res_y > s->tile->height) {
			s->dst_rect.y = ((int) x_res_y - s->tile->height) / 2;
		} else if (x_res_y < s->tile->height) {
			s->dst_rect.h = x_res_y;
		}
	} else if(!s->rgb && s->fs && (s->tile->width != x_res_x || s->tile->height != x_res_y)) {
		double frame_aspect = (double) s->tile->width / s->tile->height;
		double screen_aspect = (double) s->screen_w / s->screen_h;
		if(screen_aspect > frame_aspect) {
			s->dst_rect.h = s->screen_h;
			s->dst_rect.w = s->screen_h * frame_aspect;
			s->dst_rect.x = ((int) s->screen_w - s->dst_rect.w) / 2;
		} else {
			s->dst_rect.w = s->screen_w;
			s->dst_rect.h = s->screen_w / frame_aspect;
			s->dst_rect.y = ((int) s->screen_h - s->dst_rect.h) / 2;
		}
	}

        fprintf(stdout, "Setting SDL rect %dx%d - %d,%d.\n", s->dst_rect.w,
                s->dst_rect.h, s->dst_rect.x, s->dst_rect.y);
        
        if (s->rgb) {
                s->tile->data = s->sdl_screen->pixels +
                    s->sdl_screen->pitch * s->dst_rect.y +
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
                s->tile->data_len =
                    (int) s->sdl_screen->pitch * x_res_y -
                    s->sdl_screen->pitch * s->dst_rect.y +
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
		s->pitch = s->sdl_screen->pitch;
        } else {
                s->tile->data = (char *)*s->yuv_image->pixels;
                s->tile->data_len = s->tile->width * s->tile->height * 2;
                s->pitch = PITCH_DEFAULT;
        }
        
        s->rshift = s->sdl_screen->format->Rshift;
        s->gshift = s->sdl_screen->format->Gshift;
        s->bshift = s->sdl_screen->format->Bshift;
}

void *display_sdl_init(char *fmt, unsigned int flags)
{
        struct state_sdl *s;
        int ret;
	const SDL_VideoInfo *video_info;

        s = (struct state_sdl *)calloc(1, sizeof(struct state_sdl));
        s->magic = MAGIC_SDL;

#ifndef HAVE_MACOSX
        x11_enter_thread();
#endif

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
                        show_help();
                        free(s);
                        return NULL;
                }
                
                char *tmp = strdup(fmt);
                char *ptr = tmp;
                char *tok;
                
                while((tok = strtok(ptr, ":")))
                {
                        if (strcmp(fmt, "fs") == 0) {
                                s->fs = 1;
                                fmt = NULL;
                        } else if (strcmp(fmt, "d") == 0) {
                                s->deinterlace = 1;
                                fmt = NULL;
                        }
                        ptr = NULL;
                }
                
                free (tmp);
        }

        asm("emms\n");
        
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
        
        s->semaphore = SDL_CreateSemaphore(0);
        s->buffer_writable = 1;
        s->buffer_writable_lock = SDL_CreateMutex();
        s->buffer_writable_cond = SDL_CreateCond();

        s->frame = vf_alloc(1, 1);
        s->tile = tile_get(s->frame, 0, 0);

	video_info = SDL_GetVideoInfo();
        s->screen_w = video_info->current_w;
        s->screen_h = video_info->current_h;
        
        struct video_desc desc = {500, 500, RGBA, 0, 30.0};
        display_sdl_reconfigure(s, desc);
        loadSplashscreen(s);	

        SDL_SysWMinfo info;
        memset(&info, 0, sizeof(SDL_SysWMinfo));
        ret = SDL_GetWMInfo(&info);
#ifndef HAVE_MACOS_X
        if (ret == 1) {
                x11_set_display(info.info.x11.display);
        } else if (ret == 0) {
                fprintf(stderr, "[SDL] Warning: SDL_GetWMInfo unimplemented\n");
        } else if (ret == -1) {
                fprintf(stderr, "[SDL] Warning: SDL_GetWMInfo failure: %s\n", SDL_GetError());
        }
        
#endif


        /*if (pthread_create(&(s->thread_id), NULL, 
                           display_thread_sdl, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }*/
        
        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                s->play_audio = TRUE;
                configure_audio(s);
        } else {
                s->play_audio = FALSE;
        }

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

        return s->frame;
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

int display_sdl_get_property(void *state, int property, void *val, int *len)
{
        struct state_sdl *s = (struct state_sdl *) state;
        
        codec_t codecs[] = {UYVY, RGBA, RGB};
        
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
                        *(int *) val = s->rshift;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = s->gshift;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = s->bshift;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = s->pitch;
                        *len = sizeof(int);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

void sdl_audio_callback(void *userdata, Uint8 *stream, int len) {
        struct state_sdl *s = (struct state_sdl *)userdata;
        if (ring_buffer_read(s->audio_buffer, stream, len) != len)
        {
                fprintf(stderr, "[SDL] audio buffer underflow!!!\n");
                usleep(500);
        }
}

static void configure_audio(struct state_sdl *s)
{
        s->audio_frame.data = NULL;
        s->audio_frame.reconfigure_audio = display_sdl_reconfigure_audio;
        s->audio_frame.state = s;
        
        SDL_Init(SDL_INIT_AUDIO);
        
        if(SDL_GetAudioStatus() !=  SDL_AUDIO_STOPPED) {
                s->play_audio = FALSE;
                fprintf(stderr, "[SDL] Audio init failed - driver is already used (testcard?)\n");
                return;
        }
        
        s->audio_buffer = ring_buffer_init(1<<20);
}

void display_sdl_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_sdl *s = (struct state_sdl *)state;
        SDL_AudioSpec desired, obtained;
        int sample_type;

        s->audio_frame.bps = quant_samples / 8;
        s->audio_frame.sample_rate = sample_rate;
        s->audio_frame.ch_count = channels;
        
        if(s->audio_frame.data != NULL) {
                free(s->audio_frame.data);
                SDL_CloseAudio();
        }                
        
        if(quant_samples % 8 != 0) {
                fprintf(stderr, "[SDL] audio format isn't supported: "
                        "channels: %d, samples: %d, sample rate: %d\n",
                        channels, quant_samples, sample_rate);
                goto error;
        }
        switch(quant_samples) {
                case 8:
                        sample_type = AUDIO_S8;
                        break;
                default:
                        sample_type = AUDIO_S16LSB;
                        break;
                /* TO enable in sdl 1.3
                 * case 32:
                        sample_type = AUDIO_S32;
                        break; */
        }
        
        desired.freq=sample_rate;
        desired.format=sample_type;
        desired.channels=channels;
        
        /* Large audio buffer reduces risk of dropouts but increases response time */
        desired.samples=1024;
        
        /* Our callback function */
        desired.callback=sdl_audio_callback;
        desired.userdata=s;
        
        
        /* Open the audio device */
        if ( SDL_OpenAudio(&desired, &obtained) < 0 ){
          fprintf(stderr, "Couldn't open audio: %s\n", SDL_GetError());
          goto error;
        }
        
        s->audio_frame.max_size = 5 * (quant_samples / 8) * channels *
                        sample_rate;                
        s->audio_frame.data = (char *) malloc (s->audio_frame.max_size);

        /* Start playing */
        SDL_PauseAudio(0);

        return;
error:
        s->play_audio = FALSE;
        s->audio_frame.max_size = 0;
        s->audio_frame.data = NULL;
}


struct audio_frame * display_sdl_get_audio_frame(void *state) {
        struct state_sdl *s = (struct state_sdl *)state;
        if(s->play_audio)
                return &s->audio_frame;
}

void display_sdl_put_audio_frame(void *state, const struct audio_frame *frame) {
        struct state_sdl *s = (struct state_sdl *)state;
        char *tmp;

        if(frame->bps == 4 || frame->bps == 3) {
                tmp = (char *) malloc(frame->data_len / frame->bps * 2);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len);
                ring_buffer_write(s->audio_buffer, tmp, frame->bps * 2);
                free(tmp);
        } else {
                ring_buffer_write(s->audio_buffer, frame->data, frame->data_len);
        }
}


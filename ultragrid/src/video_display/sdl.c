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
 * $Revision: 1.15.2.10 $
 * $Date: 2010/02/05 13:56:49 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef X_DISPLAY_MISSING       /* Don't try to compile if X is not present */

#include "debug.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "tv.h"
#include "video_codec.h"

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
void NSApplicationLoad();
#else                           /* HAVE_MACOSX */
#include <sys/io.h>
#endif                          /* HAVE_MACOSX */
#include <sys/time.h>

#include <math.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>
#include <SDL/SDL_mutex.h>

#define MAGIC_SDL   DISPLAY_SDL_ID
#define FOURCC_UYVY  0x59565955

struct state_sdl {
        Display                 *display;
        SDL_Overlay             *yuv_image;
        SDL_Surface             *rgb_image;
        struct video_frame      frame;

        pthread_t               thread_id;
        SDL_sem                 *semaphore;
        uint32_t                magic;

        struct timeval          tv;
        int                     frames;

        SDL_Surface             *sdl_screen;
        SDL_Rect                dst_rect;

        const struct codec_info_t *codec_info;

        unsigned                rgb:1;
        unsigned                deinterlace:1;
        unsigned                fs:1;
};

void show_help(void);
void cleanup_screen(struct state_sdl *s);
void reconfigure_screen(void *s, unsigned int width, unsigned int height,
                        codec_t codec);

extern int should_exit;

int display_sdl_handle_events(void *arg, int post)
{
        SDL_Event sdl_event;
        struct state_sdl *s = arg;
        while (SDL_PollEvent(&sdl_event)) {
                switch (sdl_event.type) {
                case SDL_KEYDOWN:
                case SDL_KEYUP:
                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
                                should_exit = 1;
                                if(post)
                                        SDL_SemPost(s->semaphore);
                                return 1;
                        }
                        break;

                default:
                        break;
                }
        }

        return 0;

}

static void *display_thread_sdl(void *arg)
{
        struct state_sdl *s = (struct state_sdl *)arg;
        struct timeval tv;

        gettimeofday(&s->tv, NULL);

        while (!should_exit) {
                if(display_sdl_handle_events(s, 0))
                        break;
                SDL_SemWait(s->semaphore);

                if (s->deinterlace) {
                        if (s->rgb) {
                                /*FIXME: this will not work! Should not deinterlace whole screen, just subwindow */
                                vc_deinterlace(s->rgb_image->pixels,
                                               s->frame.dst_linesize, s->frame.height);
                        } else {
                                vc_deinterlace(*s->yuv_image->pixels,
                                               s->frame.dst_linesize, s->frame.height);
                        }
                }

                if (s->rgb) {
                        SDL_Flip(s->sdl_screen);
                        s->frame.data = s->sdl_screen->pixels +
                            s->sdl_screen->pitch * s->dst_rect.y +
                            s->dst_rect.x *
                            s->sdl_screen->format->BytesPerPixel;
                } else {
                        SDL_UnlockYUVOverlay(s->yuv_image);
                        SDL_DisplayYUVOverlay(s->yuv_image, &(s->dst_rect));
                        s->frame.data = (char *)*s->yuv_image->pixels;
                }

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
        return NULL;
}

void show_help(void)
{
        printf("SDL options:\n");
        printf("\twidth:height:codec[:fs][:i][:d][:f:filename] | help\n");
        printf("\tfs - fullscreen\n");
        printf("\td - deinterlace\n");
        printf("\tf filename - read frame content from the filename\n");
        show_codec_help();
}

void cleanup_screen(struct state_sdl *s)
{
        if (s->rgb == 0) {
                SDL_FreeYUVOverlay(s->yuv_image);
        }
}

void
reconfigure_screen(void *state, unsigned int width, unsigned int height,
                   codec_t color_spec)
{
        struct state_sdl *s = (struct state_sdl *)state;
        int itemp;
        unsigned int utemp;
        Window wtemp;

        unsigned int x_res_x;
        unsigned int x_res_y;

        int ret, i;

        cleanup_screen(s);

        fprintf(stdout, "Reconfigure to size %dx%d\n", width, height);

        s->frame.width = width;
        s->frame.height = height;

        ret =
            XGetGeometry(s->display, DefaultRootWindow(s->display), &wtemp,
                         &itemp, &itemp, &x_res_x, &x_res_y, &utemp, &utemp);

        fprintf(stdout, "Setting video mode %dx%d.\n", x_res_x, x_res_y);
        if (s->fs)
                s->sdl_screen =
                    SDL_SetVideoMode(x_res_x, x_res_y, 0,
                                     SDL_FULLSCREEN | SDL_HWSURFACE |
                                     SDL_DOUBLEBUF);
        else {
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
                }
        }

        if (s->rgb == 0) {
                s->yuv_image =
                    SDL_CreateYUVOverlay(s->frame.width, s->frame.height, FOURCC_UYVY,
                                         s->sdl_screen);
                if (s->yuv_image == NULL) {
                        printf("SDL_overlay initialization failed.\n");
                        free(s);
                        exit(127);
                }
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

        s->dst_rect.w = s->frame.width;
        s->dst_rect.h = s->frame.height;

        if (x_res_x > s->frame.width) {
                s->dst_rect.x = (x_res_x - s->frame.width) / 2;
        } else if (x_res_x < s->frame.width) {
                s->dst_rect.w = x_res_x;
        }
        if (x_res_y > s->frame.height) {
                s->dst_rect.y = (x_res_y - s->frame.height) / 2;
        } else if (x_res_y < s->frame.height) {
                s->dst_rect.h = x_res_y;
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
                    s->sdl_screen->pitch * x_res_y -
                    s->sdl_screen->pitch * s->dst_rect.y +
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
                s->frame.dst_x_offset =
                    s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
                s->frame.dst_bpp = s->sdl_screen->format->BytesPerPixel;
                s->frame.dst_pitch = s->sdl_screen->pitch;
        } else {
                s->frame.data = (char *)*s->yuv_image->pixels;
                s->frame.data_len = s->frame.width * s->frame.height * 2;
                s->frame.dst_bpp = 2;
                s->frame.dst_pitch = s->frame.dst_linesize;
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

        }
}

void *display_sdl_init(char *fmt)
{
        struct state_sdl *s;
        int ret;

        SDL_Surface *image;
        SDL_Surface *temp;
        SDL_Rect splash_src;
        SDL_Rect splash_dest;

        unsigned int i;

        s = (struct state_sdl *)calloc(sizeof(struct state_sdl), 1);
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

        if (!(s->display = XOpenDisplay(NULL))) {
                printf("Unable to open display.\n");
                return NULL;
        }
#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. 
         * Whatever the fuck that means. 
         * Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
#endif

        ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE);
        if (ret < 0) {
                printf("Unable to initialize SDL.\n");
                return NULL;
        }

        if (fmt != NULL) {
                reconfigure_screen(s, s->frame.width, s->frame.height, s->codec_info->codec);
                temp = SDL_LoadBMP("/usr/share/uv-0.3.1/uv_startup.bmp");
                if (temp == NULL) {
                        temp =
                            SDL_LoadBMP
                            ("/usr/local/share/uv-0.3.1/uv_startup.bmp");
                        if (temp == NULL) {
                                temp = SDL_LoadBMP("uv_startup.bmp");
                                if (temp == NULL) {
                                        printf
                                            ("Unable to load splash bitmap: uv_startup.bmp.\n");
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

                        splash_dest.x = (int)((s->frame.width - splash_src.w) / 2);
                        splash_dest.y =
                            (int)((s->frame.height - splash_src.h) / 2) + 60;
                        splash_dest.w = image->w;
                        splash_dest.h = image->h;

                        SDL_BlitSurface(image, &splash_src, s->sdl_screen,
                                        &splash_dest);
                        SDL_Flip(s->sdl_screen);
                }
        } 

        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t)reconfigure_screen;

        if (pthread_create(&(s->thread_id), NULL, 
                           display_thread_sdl, (void *)s) != 0) {
                perror("Unable to create display thread\n");
                return NULL;
        }

        return (void *)s;
}

void display_sdl_done(void *state)
{
        struct state_sdl *s = (struct state_sdl *)state;

        assert(s->magic == MAGIC_SDL);

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

        SDL_SemPost(s->semaphore);
        tmp = SDL_SemValue(s->semaphore);
        if (tmp > 1)
                printf("%d frame(s) dropped!\n", tmp);
        return 0;
}

display_type_t *display_sdl_probe(void)
{
        display_type_t *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_SDL_ID;
                dt->name = "sdl";
                dt->description = "SDL with Xvideo extension";
        }
        return dt;
}

#endif                          /* X_DISPLAY_MISSING */

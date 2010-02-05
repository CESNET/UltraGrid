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

#ifndef X_DISPLAY_MISSING /* Don't try to compile if X is not present */

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
#else /* HAVE_MACOSX */
#include <sys/io.h>
#endif /* HAVE_MACOSX */
#include <sys/time.h>

#include <math.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>
#include <SDL/SDL_mutex.h>

#define MAGIC_SDL   DISPLAY_SDL_ID
#define FOURCC_UYVY  0x59565955

struct state_sdl {
    Display             *display;
    Window              window;
    GC                  gc;
    int                 vw_depth;
    Visual              *vw_visual;
    SDL_Overlay         *yuv_image;
    SDL_Surface         *rgb_image;
    struct video_frame  frame;
    XShmSegmentInfo     vw_shm_segment[2];
    int                 image_display, image_network;
    XvAdaptorInfo       *ai;
    int                 xv_port;
    /* Thread related information follows... */
    pthread_t           thread_id;
    SDL_sem             *semaphore;
    /* For debugging... */
    uint32_t            magic; 

    struct timeval      tv;
    int                 frames;

    SDL_Surface         *sdl_screen;
    SDL_Rect            src_rect;
    SDL_Rect            dst_rect;

    int                 width;
    int                 height;
    double              bpp;
    int                 src_linesize;
    int                 dst_linesize;

    codec_t             codec;
    const struct codec_info_t *c_info;
    unsigned            rgb:1;
    unsigned            interlaced:1;
    unsigned            deinterlace:1;
    unsigned            fs:1;
};

void show_help(void);
void cleanup_screen(struct state_sdl *s);
void reconfigure_screen(void *s, unsigned int width, unsigned int height, codec_t codec);

extern int should_exit;

int
display_sdl_handle_events(void *arg)
{
    SDL_Event   sdl_event;
    struct state_sdl *s = arg;
    while (SDL_PollEvent(&sdl_event)) {
        switch (sdl_event.type) {
            case SDL_KEYDOWN:
            case SDL_KEYUP:
                if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
                    should_exit = 1;
                    SDL_SemPost(s->semaphore);
                }
                break;

            default:
                break;
        }
    }

    return 0;

}

static void*
display_thread_sdl(void *arg)
{
    struct state_sdl    *s = (struct state_sdl *) arg;
    struct timeval      tv,tv1;
    int                 i;
    unsigned char       *buff = NULL; 
    unsigned char       *line1=NULL, *line2;
    int                 linesize=0;
    int                 height=0;

    gettimeofday(&s->tv, NULL);

    while (!should_exit) {
        display_sdl_handle_events(s);
        SDL_SemWait(s->semaphore);

        if(s->deinterlace) {
                if(s->rgb) {
                        /*FIXME: this will not work! Should not deinterlace whole screen, just subwindow*/
                        vc_deinterlace(s->rgb_image->pixels, s->dst_linesize, s->height);
                } else {
                        vc_deinterlace(*s->yuv_image->pixels, s->dst_linesize, s->height);
                }
        }

        if(s->rgb) {
                SDL_Flip(s->sdl_screen);
                s->frame.data = s->sdl_screen->pixels +
                    s->sdl_screen->pitch * s->dst_rect.y + s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
        } else {
                SDL_UnlockYUVOverlay(s->yuv_image);
                SDL_DisplayYUVOverlay(s->yuv_image, &(s->dst_rect));
                s->frame.data = (unsigned char*)*s->yuv_image->pixels;
        }

        s->frames++;
        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if(seconds > 5) {
                double fps = s->frames / seconds;
                fprintf(stdout, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->tv = tv;
                s->frames = 0;
        }
    }
    return NULL;
}

void
show_help(void)
{
        printf("SDL options:\n");
        printf("\twidth:height:codec[:fs][:i][:d][:f:filename] | help\n");
        printf("\tfs - fullscreen\n");
        printf("\ti - interlaced (deprecated)\n");
        printf("\td - deinterlace\n");
        printf("\tf filename - read frame content from the filename\n");
        show_codec_help();
}

void
cleanup_screen(struct state_sdl *s)
{
    if(s->rgb == 0) {
        SDL_FreeYUVOverlay(s->yuv_image);
    }
}

void
reconfigure_screen(void *state, unsigned int width, unsigned int height, codec_t color_spec)
{
    struct state_sdl *s = (struct state_sdl *) state;
    int                 itemp;
    unsigned int        utemp;
    Window              wtemp;

    unsigned int        x_res_x;
    unsigned int        x_res_y;

    int                 ret, i;

    cleanup_screen(s);

    fprintf(stdout, "Reconfigure to size %dx%d\n", width, height);

    s->width = width;
    s->height = height;

    ret = XGetGeometry(s->display, DefaultRootWindow(s->display), &wtemp, &itemp, 
                    &itemp, &x_res_x, &x_res_y, &utemp, &utemp);

    fprintf(stdout,"Setting video mode %dx%d.\n", x_res_x, x_res_y);
    if(s->fs)
        s->sdl_screen = SDL_SetVideoMode(x_res_x, x_res_y, 0, SDL_FULLSCREEN | SDL_HWSURFACE | SDL_DOUBLEBUF);
    else {
        x_res_x = s->width;
        x_res_y = s->height;
        s->sdl_screen = SDL_SetVideoMode(x_res_x, x_res_y, 0, SDL_HWSURFACE | SDL_DOUBLEBUF);
    }
    if(s->sdl_screen == NULL){
        fprintf(stderr,"Error setting video mode %dx%d!\n", x_res_x, x_res_y);
        free(s);
        exit(128);
    }
    SDL_WM_SetCaption("Ultragrid - SDL Display", "Ultragrid");

    SDL_ShowCursor(SDL_DISABLE);

    for(i = 0; codec_info[i].name != NULL; i++) {
        if(color_spec == codec_info[i].codec) {
            s->c_info = &codec_info[i];
            s->rgb = codec_info[i].rgb;
            s->bpp = codec_info[i].bpp;
        }
    }

    if(s->rgb == 0) {
            s->yuv_image = SDL_CreateYUVOverlay(s->width, s->height, FOURCC_UYVY, s->sdl_screen);
            if (s->yuv_image == NULL) {
                printf("SDL_overlay initialization failed.\n");
                free(s);
                exit(127);
            }
    } 

    int w = s->width;

    if(s->c_info->h_align) {
            w = ((w+s->c_info->h_align-1)/s->c_info->h_align)*s->c_info->h_align;
    }

    s->src_linesize = w*s->bpp;
    if(s->rgb)
            s->dst_linesize = s->width*4;
    else
            s->dst_linesize = s->width*2;

    s->dst_rect.w = s->width;
    s->dst_rect.h = s->height;

    if ((int)x_res_x > s->width) {
        s->dst_rect.x = (x_res_x - s->width) / 2;
    } else if((int)x_res_x < s->width){
        s->dst_rect.w = x_res_x;
    }
    if ((int)x_res_y > s->height) {
        s->dst_rect.y = (x_res_y - s->height) / 2;
    } else if((int)x_res_y < s->height) {
        s->dst_rect.h = x_res_y;
    }

    s->src_rect.w = s->width;
    s->src_rect.h = s->height;

    fprintf(stdout,"Setting SDL rect %dx%d - %d,%d.\n", s->dst_rect.w, s->dst_rect.h, s->dst_rect.x, 
                    s->dst_rect.y);

    s->frame.rshift = s->sdl_screen->format->Rshift;
    s->frame.gshift = s->sdl_screen->format->Gshift;
    s->frame.bshift = s->sdl_screen->format->Bshift;
    s->frame.color_spec = s->c_info->codec;
    s->frame.width = s->width;
    s->frame.height = s->height;
    s->frame.src_bpp = s->bpp;
    s->frame.dst_linesize = s->dst_linesize;

    if(s->rgb) {
        s->frame.data = s->sdl_screen->pixels + 
                s->sdl_screen->pitch * s->dst_rect.y + s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
        s->frame.data_len = s->sdl_screen->pitch *  x_res_y - 
                s->sdl_screen->pitch * s->dst_rect.y + s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
        s->frame.dst_x_offset = s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
        s->frame.dst_bpp = s->sdl_screen->format->BytesPerPixel;
        s->frame.dst_pitch = s->sdl_screen->pitch;
    } else {
        s->frame.data = (unsigned char*)*s->yuv_image->pixels;
        s->frame.data_len = s->width * s->height * 2;
        s->frame.dst_bpp = 2;
        s->frame.dst_pitch = s->dst_linesize;
    }

    switch(color_spec) {
        case R10k:
                s->frame.decoder = vc_copyliner10k;
                break;
        case v210:
                s->frame.decoder = vc_copylinev210;
                break;
        case DVS10:
                s->frame.decoder = vc_copylineDVS10;
                break;
        case DVS8:
        case UYVY:
        case Vuy2:
                s->frame.decoder = memcpy;
                break;
        case RGBA:
                s->frame.decoder = vc_copylineRGBA;
                break;

    }
}

void *
display_sdl_init(char *fmt)
{
    struct state_sdl    *s;
    int                 ret;

    SDL_Surface         *image;
    SDL_Surface         *temp;
    SDL_Rect            splash_src;
    SDL_Rect            splash_dest;

    unsigned int        i;

    s = (struct state_sdl *) calloc(sizeof(struct state_sdl),1);
    s->magic = MAGIC_SDL;

    if(fmt != NULL) {
            if(strcmp(fmt, "help") == 0) {
                show_help();
                free(s);
                return NULL;
            }
        
            if(strcmp(fmt, "fs") == 0) {
                s->fs = 1;
                fmt = NULL;
            } else {

                char *tmp = strdup(fmt);
                char *tok;
           
                tok = strtok(tmp, ":");
                if(tok == NULL) {
                        show_help();
                        free(s);
                        free(tmp);
                        return NULL;
                }
                s->width = atol(tok);
                tok = strtok(NULL, ":");
                if(tok == NULL) {
                        show_help();
                        free(s);
                        free(tmp);
                        return NULL;
                }
                s->height = atol(tok);
                tok = strtok(NULL, ":");
                if(tok == NULL) {
                        show_help();
                        free(s);
                        free(tmp);
                        return NULL;
                }
                s->codec = 0xffffffff;
                for(i = 0; codec_info[i].name != NULL; i++) {
                        if(strcmp(tok, codec_info[i].name) == 0) {
                            s->codec = codec_info[i].codec;
                            s->c_info = &codec_info[i];
                            s->rgb = codec_info[i].rgb;
                        }
                }
                if(s->codec == 0xffffffff) {
                    fprintf(stderr, "SDL: unknown codec: %s\n", tok);
                    free(s);
                    free(tmp);
                    return NULL;
                }
                tok = strtok(NULL, ":");
                while(tok != NULL) {
                        if(tok[0] == 'f' && tok[1] == 's') {
                                s->fs=1;
                        } else if(tok[0] == 'i') {
                                s->interlaced = 1;
                        } else if(tok[0] == 'd') {
                                s->deinterlace = 1;
                        }
                        tok = strtok(NULL, ":");
                }
                free(tmp);

                if(s->width <= 0 || s->height <= 0) {
                    printf("SDL: failed to parse config options: '%s'\n", fmt);
                    free(s);
                    return NULL;
                }
                s->bpp = s->c_info->bpp;
                printf("SDL setup: %dx%d codec %s\n", s->width, s->height, s->c_info->name);
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

    if(fmt != NULL) {
        /*FIXME: kill hd_size at all, use another approach avoiding globals */
        int w = s->width;

        if(s->c_info->h_align) {
            w = ((w+s->c_info->h_align-1)/s->c_info->h_align)*s->c_info->h_align;
        }

        hd_size_x = w;
        hd_size_y = s->height;
        reconfigure_screen(s, w, s->height, s->c_info->codec);
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
 
            splash_dest.x = (int)((s->width - splash_src.w) / 2);
            splash_dest.y = (int)((s->height - splash_src.h) / 2) + 60;
            splash_dest.w = image->w;
            splash_dest.h = image->h;
 
            SDL_BlitSurface(image, &splash_src, s->sdl_screen, &splash_dest);
            SDL_Flip(s->sdl_screen);
        }
    }

    s->frame.width = 0;
    s->frame.height = 0;
    s->frame.color_spec = 0;
    s->frame.state = s;
    s->frame.reconfigure = reconfigure_screen;

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

    /*FIXME: free all the stuff */
    SDL_ShowCursor(SDL_ENABLE);

    SDL_Quit();
}

struct video_frame *
display_sdl_getf(void *state)
{
    struct state_sdl *s = (struct state_sdl *) state;
    assert(s->magic == MAGIC_SDL);
    return &s->frame;
}

int
display_sdl_putf(void *state, char *frame)
{
    int      tmp;
    struct state_sdl *s = (struct state_sdl *) state;

    assert(s->magic == MAGIC_SDL);
    assert(frame != NULL);

    SDL_SemPost(s->semaphore);
    tmp = SDL_SemValue(s->semaphore);
    if(tmp > 1) 
        printf("%d frame(s) dropped!\n", tmp);
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
        dt->id          = DISPLAY_SDL_ID;
        dt->name        = "sdl";
        dt->description = "SDL with Xvideo extension";
        dt->formats     = dformat;
        dt->num_formats = 4;
    }
    return dt;
}

#endif /* X_DISPLAY_MISSING */

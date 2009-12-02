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
 * $Revision: 1.13 $
 * $Date: 2009/12/02 10:39:45 $
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
#include "v_codec.h"

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
#include "compat/platform_semaphore.h"

#include <math.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>

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
    unsigned char       *buffers[2];
    XShmSegmentInfo     vw_shm_segment[2];
    int                 image_display, image_network;
    XvAdaptorInfo       *ai;
    int                 xv_port;
    /* Thread related information follows... */
    pthread_t           thread_id;
    sem_t               semaphore;
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
    char                *filename;
    unsigned            rgb:1;
    unsigned            interlaced:1;
    unsigned            deinterlace:1;
    unsigned            use_file:1;
    unsigned            fs:1;
};

inline void copyline64(unsigned char *dst, unsigned char *src, int len);
inline void copyline128(unsigned char *dst, unsigned char *src, int len);
inline void copylinev210(unsigned char *dst, unsigned char *src, int len);
inline void copyliner10k(struct state_sdl *s, unsigned char *dst, unsigned char *src, int len);
void copylineRGBA(struct state_sdl *s, unsigned char *dst, unsigned char *src, int len);
void deinterlace(struct state_sdl *s, unsigned char *buffer);
void show_help(void);

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
                    platform_sem_post(&s->semaphore);
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
deinterlace(struct state_sdl *s, unsigned char *buffer)
{
    int i,j;
    long pitch = s->dst_linesize;
    register long pitch2 = pitch*2;
    unsigned char *bline1, *bline2, *bline3;
    register unsigned char *line1, *line2, *line3;

    bline1 = buffer;
    bline2 = buffer + pitch;
    bline3 = buffer + 3*pitch; 
        for(i=0; i < s->dst_linesize; i+=16) {
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
                for(j=0; j < s->height-4; j+=2) {
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

void
copyline64(unsigned char *dst, unsigned char *src, int len)
{
        register uint64_t *d, *s;

        register uint64_t a1,a2,a3,a4;

        d = (uint64_t *)dst;
        s = (uint64_t *)src;

        while(len > 0) {
                a1 = *(s++);
                a2 = *(s++);
                a3 = *(s++);
                a4 = *(s++);

                a1 = (a1 & 0xffffff) | ((a1 >> 8) & 0xffffff000000LL);
                a2 = (a2 & 0xffffff) | ((a2 >> 8) & 0xffffff000000LL);
                a3 = (a3 & 0xffffff) | ((a3 >> 8) & 0xffffff000000LL);
                a4 = (a4 & 0xffffff) | ((a4 >> 8) & 0xffffff000000LL);

                *(d++) = a1 | (a2 << 48);       /* 0xa2|a2|a1|a1|a1|a1|a1|a1 */
                *(d++) = (a2 >> 16)|(a3 << 32); /* 0xa3|a3|a3|a3|a2|a2|a2|a2 */
                *(d++) = (a3 >> 32)|(a4 << 16); /* 0xa4|a4|a4|a4|a4|a4|a3|a3 */

                len -= 12;
    }
}

void
copylinev210(unsigned char *dst, unsigned char *src, int len)
{
    struct {
                 unsigned a:10;
                 unsigned b:10;
                 unsigned c:10;
                 unsigned p1:2;
    } *s;
    register uint32_t *d;
    register uint32_t tmp;

    d = (uint32_t *)dst;
    s = (void*)src;

    while(len > 0) {
        tmp = (s->a >> 2) | (s->b >> 2) << 8 | (((s)->c >> 2) << 16);
        s++;
        *(d++) = tmp | ((s->a >> 2) << 24);
        tmp = (s->b >> 2) | (((s)->c >> 2) << 8);
        s++;
        *(d++) = tmp | ((s->a >> 2) << 16) | ((s->b >> 2)<<24);
        tmp = (s->c >> 2);
        s++;
        *(d++) = tmp | ((s->a >> 2) << 8) | ((s->b >> 2) << 16) | ((s->c >> 2) << 24);
        s++;

        len -= 12;
    }
}

void
copyliner10k(struct state_sdl *state, unsigned char *dst, unsigned char *src, int len)
{
    struct {
        unsigned r:8;

        unsigned gh:6;
        unsigned p1:2;

        unsigned bh:4;
        unsigned p2:2;
        unsigned gl:2;

        unsigned p3:2;
        unsigned p4:2;
        unsigned bl:4;
    } *s;
    register uint32_t *d;
    register uint32_t tmp;
    int rshift = state->sdl_screen->format->Rshift;
    int gshift = state->sdl_screen->format->Gshift;
    int bshift = state->sdl_screen->format->Bshift;
//    int ashift = state->sdl_screen->format->Ashift;

    d = (uint32_t *)dst;
    s = (void*)src;

    while(len > 0) {
        tmp = (s->r << rshift) | (((s->gh << 2) | s->gl) << gshift) | (((s->bh << 4) | s->bl) << bshift);// | (0xff << ashift);
        s++;
        *(d++) = tmp;
        tmp = (s->r << rshift) | (((s->gh << 2) | s->gl) << gshift) | (((s->bh << 4) | s->bl) << bshift);// | (0xff << ashift);
        s++;
        *(d++) = tmp;
        tmp = (s->r << rshift) | (((s->gh << 2) | s->gl) << gshift) | (((s->bh << 4) | s->bl) << bshift);// | (0xff << ashift);
        s++;
        *(d++) = tmp;
        tmp = (s->r << rshift) | (((s->gh << 2) | s->gl) << gshift) | (((s->bh << 4) | s->bl) << bshift);// | (0xff << ashift);
        s++;
        *(d++) = tmp;
        len -= 16;
    }
}

void
copylineRGBA(struct state_sdl *state, unsigned char *dst, unsigned char *src, int len)
{
    register uint32_t *d=(uint32_t*)dst;
    register uint32_t *s=(uint32_t*)src;
    register uint32_t tmp;
    int rshift = state->sdl_screen->format->Rshift;
    int gshift = state->sdl_screen->format->Gshift;
    int bshift = state->sdl_screen->format->Bshift;

    if(rshift == 0 && gshift == 8 && bshift == 16) {
            memcpy(dst, src, len);
    } else {
            while(len > 0) {
                register unsigned int r,g,b;
                tmp = *(s++);
                r = tmp & 0xff;
                g = (tmp >> 8) & 0xff;
                b = (tmp >> 16) & 0xff;
                tmp = (r << rshift) | (g << gshift) | (b << bshift);
                *(d++) = tmp; 
                tmp = *(s++);
                r = tmp & 0xff;
                g = (tmp >> 8) & 0xff;
                b = (tmp >> 16) & 0xff;
                tmp = (r << rshift) | (g << gshift) | (b << bshift);
                *(d++) = tmp; 
                tmp = *(s++);
                r = tmp & 0xff;
                g = (tmp >> 8) & 0xff;
                b = (tmp >> 16) & 0xff;
                tmp = (r << rshift) | (g << gshift) | (b << bshift);
                *(d++) = tmp; 
                tmp = *(s++);
                r = tmp & 0xff;
                g = (tmp >> 8) & 0xff;
                b = (tmp >> 16) & 0xff;
                tmp = (r << rshift) | (g << gshift) | (b << bshift);
                *(d++) = tmp;
                len -= 16; 
            }
    }
}

/* convert 10bits Cb Y Cr A Y Cb Y A to 8bits Cb Y Cr Y Cb Y */

#if !(HAVE_MACOSX || HAVE_32B_LINUX)

void
copyline128(unsigned char *d, unsigned char *s, int len)
{
    register unsigned char *_d=d,*_s=s;

        while(len > 0) {

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
        len -= 24;
    }
}

#endif /* !(HAVE_MACOSX || HAVE_32B_LINUX) */


static void*
display_thread_sdl(void *arg)
{
    struct state_sdl    *s = (struct state_sdl *) arg;
    struct timeval      tv;
    int                 i;
    unsigned char       *buff = NULL; 
    unsigned char *line1=NULL, *line2;
    int linesize=0;
    int height=0;

    if(s->use_file && s->filename) {
            FILE *in;
            buff = (unsigned char*)malloc(s->src_linesize*s->height);
            in = fopen(s->filename, "r");
            if(!in || fread(buff, s->src_linesize*s->height, 1, in) == 0) {
                    printf("SDL: Cannot read image file %s\n", s->filename);
                    exit(126);
            }
            fclose(in);
    }

    gettimeofday(&s->tv, NULL);

    if(s->use_file && buff) 
        line1 = buff;

    if(s->rgb) {
        linesize = s->sdl_screen->pitch;
        if(linesize > s->src_linesize)
            linesize = s->src_linesize;
            height = s->height;
        if(height > s->dst_rect.h)
            height = s->dst_rect.h;
    }

    while (!should_exit) {
        display_sdl_handle_events(s);

        if(!(s->use_file && buff)) {
                platform_sem_wait(&s->semaphore);
                line1 = s->buffers[s->image_display];
        }

        if(s->rgb) {
                SDL_LockSurface(s->sdl_screen);
                line2 = s->sdl_screen->pixels;
                line2 += s->sdl_screen->pitch * s->dst_rect.y + s->dst_rect.x * s->sdl_screen->format->BytesPerPixel;
        } else {
                SDL_LockYUVOverlay(s->yuv_image);
                line2 = *s->yuv_image->pixels;
        }

        switch(s->codec) {
                case R10k:
                        for(i = 0; i < height; i++) {
                                copyliner10k(s, line2, line1, linesize);
                                line1 += s->src_linesize;
                                line2 += s->sdl_screen->pitch;;
                        }
                        break;
                case v210:
                        for(i = 0; i < s->height; i++) {
                                copylinev210(line2, line1, s->dst_linesize);
                                line1 += s->src_linesize;
                                line2 += s->dst_linesize;
                        }
                        break;
                case DVS10:
                        if(s->interlaced == 0) {
                                for(i = 0; i < s->height; i++) {
#if (HAVE_MACOSX || HAVE_32B_LINUX)
                                        copyline64(line2, line1, s->dst_linesize);
#else
                                        copyline128(line2, line1, s->dst_linesize);
#endif
                                        line1 += s->src_linesize;
                                        line2 += s->dst_linesize;
                                }
                        } else {
                                for(i = 0; i < s->height; i+=2) {
#if (HAVE_MACOSX || HAVE_32B_LINUX)
                                        copyline64(line2, line1, s->dst_linesize);
                                        copyline64(line2+s->dst_linesize, 
                                                   line1+s->src_linesize*s->height/2, 
                                                   s->dst_linesize);
#else /* (HAVE_MACOSX || HAVE_32B_LINUX) */
                                        copyline128(line2, line1, s->dst_linesize);
                                        copyline128(line2+s->dst_linesize, 
                                                    line1+s->src_linesize*s->height/2, s->dst_linesize);
#endif /* (HAVE_MACOSX || HAVE_32B_LINUX) */
                                        line1 += s->src_linesize;
                                        line2 += s->dst_linesize*2;
                                }
                        }
                        break;
                case DVS8:
                case UYVY:
                        if(s->interlaced == 0) {
                                for(i = 0; i < s->height; i++) {
                                        memcpy(line2, line1, s->dst_linesize);
                                        line2 += s->dst_linesize;
                                        line1 += s->src_linesize;
                                }
                        } else {
                               for(i = 0; i < s->height; i+=2) {
                                       memcpy(line2, line1, s->dst_linesize);
                                       memcpy(line2+s->dst_linesize, line1+s->height/2*s->src_linesize,
                                                       s->dst_linesize);
                                       line1 += s->src_linesize;
                                       line2 += 2*s->dst_linesize;
                               }
                        }
                        break;
                case RGBA:

                        if(s->interlaced == 0) {
                                for(i = 0; i < height; i++) {
                                        copylineRGBA(s, line2, line1, linesize);
                                        line2 += s->sdl_screen->pitch;
                                        line1 += s->src_linesize;
                                }
                        } else {
                               for(i = 0; i < height; i+=2) {
                                       copylineRGBA(s, line2, line1, linesize);
                                       copylineRGBA(s, line2+s->sdl_screen->pitch, line1+s->height/2*s->src_linesize,
                                                       linesize);
                                       line1 += s->src_linesize;
                                       line2 += 2*s->sdl_screen->pitch;
                               }
                        }
                        break;

        }

 
        if(s->deinterlace) {
                if(s->rgb) {
                        /*FIXME: this will not work! Should not deinterlace whole screen, just subwindow*/
                        deinterlace(s, s->rgb_image->pixels);
                } else {
                        deinterlace(s, *s->yuv_image->pixels);
                }
        }

        if(s->rgb) {
                SDL_UnlockSurface(s->sdl_screen);
                SDL_Flip(s->sdl_screen);
        } else {
                SDL_UnlockYUVOverlay(s->yuv_image);
                SDL_DisplayYUVOverlay(s->yuv_image, &(s->dst_rect));
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

        if(s->use_file && buff)
                usleep(50000);

        // printf("FPS: %f\n", 1000000.0/tv2);

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

void *
display_sdl_init(char *fmt)
{
    struct state_sdl    *s;
    int                 ret;

    SDL_Surface         *image;
    SDL_Surface         *temp;
    SDL_Rect            splash_src;
    SDL_Rect            splash_dest;

    int                 itemp;
    unsigned int        utemp;
    Window              wtemp;

    unsigned int        x_res_x;
    unsigned int        x_res_y;

    const struct codec_info_t *c_info=NULL;

    unsigned int        i;

    s = (struct state_sdl *) calloc(sizeof(struct state_sdl),1);
    s->magic = MAGIC_SDL;

    if(fmt!=NULL) {
            if(strcmp(fmt, "help") == 0) {
                show_help();
                free(s);
                return NULL;
            }
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
                        c_info = &codec_info[i];
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
                    } else if(tok[0] == 'f') {
                            s->use_file =1;
                            tok = strtok(NULL, ":");
                            s->filename = strdup(tok);
                    }
                    tok = strtok(NULL, ":");
            }
            free(tmp);
    }

    if(s->width <= 0 || s->height <= 0) {
            printf("SDL: failed to parse config options: '%s'\n", fmt);
            free(s);
            return NULL;
    }

    s->bpp = c_info->bpp;

    printf("SDL setup: %dx%d codec %s\n", s->width, s->height, c_info->name);

    asm("emms\n");

    platform_sem_init(&s->semaphore, 0, 0);

    debug_msg("Window initialized %p\n", s);

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

    /* Get XWindows resolution */
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

    if(s->rgb == 0) {
            s->yuv_image = SDL_CreateYUVOverlay(s->width, s->height, FOURCC_UYVY, s->sdl_screen);
            if (s->yuv_image == NULL) {
                printf("SDL_overlay initialization failed.\n");
                free(s);
                exit(127);
            }
    } 

    int w = s->width;

    if(c_info->h_align) {
            w = ((w+c_info->h_align-1)/c_info->h_align)*c_info->h_align;
    }

    /*FIXME: kill hd_size at all, use another approach avoiding globals */
    hd_size_x = w;
    hd_size_y = s->height;
    /*FIXME: kill hd_color_bpp at all, use linesize instead! 
     *       however, not sure about computing pos (x,y) in transmit.c
     *       mess with green pixels with DVS and MTU 9000 could be because of pixel size 8/3B not 3B!
     */
    hd_color_bpp = ceil(s->bpp);

    s->buffers[0] = malloc(w*s->height*hd_color_bpp);
    s->buffers[1] = malloc(w*s->height*hd_color_bpp);

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

    /*FIXME: free all the stuff */
    free(s->filename);

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
    int      tmp;
    struct state_sdl *s = (struct state_sdl *) state;

    assert(s->magic == MAGIC_SDL);
    assert(frame != NULL);

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

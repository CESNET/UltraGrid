/*
 * FILE:    video_display/xv.c
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
 * $Revision: 1.3 $
 * $Date: 2009/12/11 15:27:18 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef X_DISPLAY_MISSING /* Don't try to compile if X is not present */

#include "debug.h"
#include "video_display.h"
#include "video_display/xv.h"
#include "tv.h"

/* For X shared memory... */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/Xv.h>
#include <X11/extensions/Xvlib.h>
#include <X11/extensions/XShm.h>
#include <host.h>

#include <video_codec.h>

#ifdef HAVE_XINERAMA
#include <X11/extensions/Xinerama.h>
#endif

#define MAGIC_XV	DISPLAY_XV_ID

struct state_xv {
        struct video_frame       frame;
        struct video_frame      *tiles;
        int                      put_frame:1;
        int                      new_frame:1;

	Display			*display;
        struct {
                int              width;
                int              height;
        } display_attr;
	Window			 window;
	GC			 gc;
	int			 vw_depth;
	Visual			*vw_visual;
        union {
                XvImage		*vw_xvimage[2];
                XImage          *vw_ximage[2];
        };
	XShmSegmentInfo		 vw_shm_segment[2];
	int			 image_display, image_network;
	XvAdaptorInfo		*ai;
	int			 xv_port;
	/* Thread related information follows... */
	pthread_t		 thread_id;
	pthread_mutex_t		 lock;
	pthread_cond_t		 boss_cv;
	pthread_cond_t		 worker_cv;
	int			 work_to_do;
	int			 boss_waiting;
	int			 worker_waiting;

        int                      win_initialized;
        int                      yuv;
        int                      fs;
        int                      deinterlace;
	/* For debugging... */
	uint32_t		 magic;	
};

static void reconfigure_screen_xv(void *s, unsigned int width, unsigned int height,
                        codec_t codec, double fps, int aux, struct tile_info tile_info);
static void cleanup_xv(struct state_xv *s);
static struct video_frame * get_tile_buffer(void *s, struct tile_info tile_info);
static void update_tile_data(struct state_xv *s);
static void update_fullscreen_state(struct state_xv *s);

struct rect
{
        int x, y, w, h;
};
static void getDest(struct rect src, int disp_w, int disp_h, struct rect *res)
{
        double src_aspect = (double) src.w / src.h;
        double disp_aspect = (double) disp_w / disp_h;
        if (src_aspect > disp_aspect)
        {
                res->x = 0;
                res->w = disp_w;
                res->h = (double) src.h * disp_w / src.w;
                res->y = (disp_h - res->h) / 2;
        }
        else
        {
                res->y = 0;
                res->h = disp_h;
                res->w = (double) src.w * disp_h / src.h;
                res->x = (disp_w - res->w) / 2;
        }
}

static void update_fullscreen_state(struct state_xv *s)
{
        XEvent xev;
        XSizeHints *size_hints;

        size_hints = XAllocSizeHints();
        if(s->fs) {
                size_hints->flags =  PMinSize | PMaxSize | PWinGravity | PAspect | PBaseSize;
                size_hints->min_width =
                        size_hints->max_width=
                        size_hints->base_width=
                        size_hints->min_aspect.x=
                        size_hints->max_aspect.x=
                        s->display_attr.width;
                size_hints->min_height =
                        size_hints->max_height=
                        size_hints->base_height=
                        size_hints->min_aspect.y=
                        size_hints->max_aspect.y=
                        s->display_attr.height;
                size_hints->win_gravity=StaticGravity;
        } else {
                size_hints->flags = PBaseSize;
                size_hints->base_width=s->frame.width;
                size_hints->base_height=s->frame.height;
                if (s->yuv) {
                        size_hints->flags |= PAspect;
                        size_hints->min_aspect.x=s->frame.width;
                        size_hints->max_aspect.x=size_hints->min_aspect.x;
                        size_hints->min_aspect.y=s->frame.height;
                        size_hints->max_aspect.y=size_hints->min_aspect.y;
                } else {
                        size_hints->flags |=  PMinSize | PMaxSize;
                        size_hints->min_width=s->frame.width;
                        size_hints->min_height=s->frame.height;
                        size_hints->max_width=size_hints->min_width=s->frame.width;
                        size_hints->max_height=size_hints->min_height;
                }
        }

        memset(&xev, 0, sizeof(xev));
        xev.type = ClientMessage;
        xev.xclient.serial = 0;
        xev.xclient.send_event=True;
        xev.xclient.window = s->window;
        xev.xclient.message_type = XInternAtom(s->display, "_NET_WM_STATE", False);
        xev.xclient.format = 32;
        xev.xclient.data.l[0] = s->fs ? 1 : 0;
        xev.xclient.data.l[1] = XInternAtom(s->display, "_NET_WM_STATE_FULLSCREEN", False);
        xev.xclient.data.l[2] = 0;

        XUnmapWindow(s->display, s->window);
        XSendEvent(s->display, DefaultRootWindow(s->display), False,
                        SubstructureRedirectMask|SubstructureNotifyMask, &xev);
        XSetWMNormalHints(s->display, s->window, size_hints);
        XMoveWindow(s->display, s->window, 0, 0);
        XFree(size_hints);

        /* shouldn't be needed */
        if (s->fs) {
                XMoveResizeWindow(s->display, s->window, 0, 0, s->display_attr.width, s->display_attr.height);
        } else {
                XMoveResizeWindow(s->display, s->window, 0, 0, s->frame.width, s->frame.height);
        }

        XMapRaised(s->display, s->window);
        XRaiseWindow(s->display, s->window);
        XMoveWindow(s->display, s->window, 0, 0);
        XFlush(s->display);
}


static void handle_events_xv(struct state_xv *s)
        {
        XEvent event;
        if (XCheckWindowEvent(s->display, s->window, KeyPress, &event))
        {
                switch (event.xkey.keycode) {
                        case 24: /* 'q' */
                                should_exit = 1;
                                break;
                        case 40: /* 'd' */
                                s->deinterlace = s->deinterlace ? FALSE : TRUE;
                                printf("Deinterlacing: %s\n", s->deinterlace ?
                                                "ON" : "OFF");
                                break;
                        case 41: /* 'f' */
                                if (s->fs)
                                        s->fs = FALSE;
                                else
                                        s->fs = TRUE;
                                update_fullscreen_state(s);
                                break;
                }
        }
}

static void*
display_thread_xv(void *arg)
{
	struct state_xv        *s = (struct state_xv *) arg;
        struct rect res, image;
        image.x = 0;
        image.y = 0;

	while (!should_exit) {
                handle_events_xv(s);

		pthread_mutex_lock(&s->lock);

		while (s->work_to_do == FALSE) {
			s->worker_waiting = TRUE;
			pthread_cond_wait(&s->worker_cv, &s->lock);
			s->worker_waiting = FALSE;
		}

		s->work_to_do     = FALSE;

		if (s->boss_waiting) {
			pthread_cond_signal(&s->boss_cv);
		}
		pthread_mutex_unlock(&s->lock);


                if (s->deinterlace) {
                        if (s->yuv) {
                                vc_deinterlace(s->vw_xvimage[s->image_display]->data,
                                       s->frame.dst_linesize, s->frame.height);
                        } else {
                                vc_deinterlace(s->vw_ximage[s->image_display]->data,
                                        s->frame.dst_linesize, s->frame.height);
                        }
                }

                if (s->fs)
                {
                        if (s->yuv)
                        {
                                image.w = s->vw_xvimage[s->image_display]->width;
                                image.h = s->vw_xvimage[s->image_display]->height;
                                getDest(image, s->display_attr.width, s->display_attr.height, &res);
                                XvShmPutImage(s->display, s->xv_port, s->window, s->gc, s->vw_xvimage[s->image_display], 0, 0, 
                                              s->vw_xvimage[s->image_display]->width, s->vw_xvimage[s->image_display]->height, 
                                              res.x, res.y, res.w, res.h, False);
                        } else {
                                int x = (s->display_attr.width - s->vw_xvimage[s->image_display]->width) / 2;
                                int y = (s->display_attr.height - s->vw_xvimage[s->image_display]->height) / 2;
                                XShmPutImage(s->display, s->window, s->gc, s->vw_ximage[s->image_display], 0, 0, x, y,
                                                s->vw_ximage[s->image_display]->width, s->vw_ximage[s->image_display]->height, True);
                        }
                } else {
                        if (s->yuv)
                        {
                                XWindowAttributes wattr;
                                XGetWindowAttributes(s->display, s->window, &wattr);
                                XvShmPutImage(s->display, s->xv_port, s->window, s->gc, s->vw_xvimage[s->image_display], 0, 0, 
                                              s->vw_xvimage[s->image_display]->width, s->vw_xvimage[s->image_display]->height, 
                                              0, 0, wattr.width, wattr.height, False);
                        } else {
                                XShmPutImage(s->display, s->window, s->gc, s->vw_ximage[s->image_display], 0, 0, 0, 0,
                                                s->vw_ximage[s->image_display]->width, s->vw_ximage[s->image_display]->height, True);
                        }
                }


		XFlush(s->display);
	}
	return NULL;
}


void *
display_xv_init(void)
{
	struct state_xv		*s;
	unsigned int		 p_version, p_release, p_request_base, p_event_base, p_error_base, p; 
	unsigned int          	 p_num_adaptors, i, j;
        XWindowAttributes        xattr;
        XineramaScreenInfo      *screen_info;
        int num;

	s = (struct state_xv *) malloc(sizeof(struct state_xv));
	s->magic   = MAGIC_XV;
	s->xv_port = -1;

	if (!(s->display = XOpenDisplay(NULL))) {
		printf("Unable to open display.\n");
		return NULL;
	}

        s->fs = FALSE;
        s->vw_depth  = DefaultDepth(s->display, DefaultScreen(s->display));
        s->vw_visual = DefaultVisual(s->display, DefaultScreen(s->display));
        XGetWindowAttributes(s->display, DefaultRootWindow(s->display), &xattr);
        s->display_attr.width = xattr.width;
        s->display_attr.height = xattr.height;
#ifdef HAVE_XINERAMA
        if ((screen_info = XineramaQueryScreens(s->display, &num)) != NULL)
        {
                int i;
                for (i = 0; i < num; ++i)
                {
                        if (screen_info[i].x_org + screen_info[i].width > 
                                        s->display_attr.width)
                                s->display_attr.width =
                                        screen_info[i].x_org +
                                        screen_info[i].width;
                        if (screen_info[i].y_org + screen_info[i].height > 
                                        s->display_attr.height)
                                s->display_attr.height =
                                        screen_info[i].y_org +
                                        screen_info[i].height;
                }
        }
        XFree(screen_info);
#endif


        /* For now, we only support 24-bit TrueColor displays... */
        if (s->vw_depth != 24) {
                printf("Unable to open display: not 24 bit colour\n");
                return NULL;
        }
        if (s->vw_visual->class != TrueColor) {
                printf("Unable to open display: not TrueColor visual\n");
                return NULL;
        }

	/* Do we support the Xv extension? */
	if (XvQueryExtension(s->display, &p_version, &p_release, &p_request_base, &p_event_base, &p_error_base) != Success) {
		printf("Cannot activate Xv extension\n");
		abort();
	}
	debug_msg("Xv version       = %u.%u\n", p_version, p_release);
	debug_msg("Xv request base  = %u\n", p_request_base);
	debug_msg("Xv event base    = %u\n", p_event_base);
	debug_msg("Xv error base    = %u\n", p_error_base);
	if (XvQueryAdaptors(s->display, DefaultRootWindow(s->display), &p_num_adaptors, &s->ai) != Success) {
		printf("Cannot query Xv adaptors\n");
		abort();
	}
	s->xv_port = 0;
	debug_msg("Xv adaptor count = %d\n", p_num_adaptors);
	for (i = 0;  i < p_num_adaptors && s->xv_port == 0; i++) {
		debug_msg("Xv adaptor %d name = %s\n", i, s->ai[i].name);
		debug_msg("Xv adaptor %d type = %s%s%s%s%s\n", i, (s->ai[i].type & XvInputMask)   ? "[input] "  : "",
						  		  (s->ai[i].type & XvOutputMask)  ? "[output] " : "",
								  (s->ai[i].type & XvVideoMask)   ? "[video] "  : "",
								  (s->ai[i].type & XvStillMask)   ? "[still] "  : "",
								  (s->ai[i].type & XvImageMask)   ? "[image] "  : "");
		for (p = s->ai[i].base_id; p < s->ai[i].base_id + s->ai[i].num_ports; p++) {
			unsigned int		 encodings;
			int			 formats;
			int			 k;
			XvEncodingInfo		*ei;
			XvImageFormatValues	*fo;

                        if(!XvGrabPort(s->display, p, CurrentTime)) {
                                 debug_msg("Grabed port: %d\n", p);
                                 s->xv_port = p;
                         } else {
                                 debug_msg("Cannot grab port: %d\n", p);
                                 abort();
                         }

			if (XvQueryEncodings(s->display, p, &encodings, &ei) != Success) {
				printf("Cannot query Xv encodings\n");
				abort();
			}
			for (j = 0; j < encodings; j++) {
				debug_msg("Xv adaptor %d port %d coding %d = %s\n", i, p - s->ai[i].base_id, j, ei[j].name);
			}
			XvFreeEncodingInfo(ei);
			fo = XvListImageFormats(s->display, p, &formats);
			for (k = 0; k < formats; k++) {
				debug_msg("Xv adaptor %d port %d format %d = 0x%08lx %s\n", i, p - s->ai[i].base_id, k, fo[k].id, fo[k].guid);
			}
			if (fo != NULL) {
				XFree(fo);
			}
		}
	}
	s->image_network = 0;
	s->image_display = 1;

	pthread_mutex_init(&s->lock, NULL);
	pthread_cond_init(&s->boss_cv, NULL);
	pthread_cond_init(&s->worker_cv, NULL);
	s->work_to_do     = FALSE;
	s->boss_waiting   = FALSE;
	s->worker_waiting = TRUE;

        s->win_initialized = 0;
        s->frame.state = s;
        s->frame.reconfigure = (reconfigure_t) reconfigure_screen_xv;
        s->frame.get_tile_buffer = (get_tile_buffer_t) get_tile_buffer;

        s->frame.rshift = 0;
        s->frame.gshift = 8;
        s->frame.bshift = 16;


        s->frame.aux &= ~AUX_TILED; /* do not expect tiled video by default */
        s->tiles = NULL;

        s->deinterlace = FALSE;

        s->new_frame = FALSE;

	if (pthread_create(&(s->thread_id), NULL, display_thread_xv, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	return (void *) s;
}

static void reconfigure_screen_xv(void *arg, unsigned int width, unsigned int height,
                        codec_t codec, double fps, int aux, struct tile_info tile_info)
{
        struct state_xv *s = (struct state_xv *) arg;
        int i;

        assert(s->magic == MAGIC_XV);

        if (s->win_initialized)
                cleanup_xv(s);

        if (aux & AUX_TILED) {
                s->frame.width = width * tile_info.x_count;
                s->frame.height = height * tile_info.y_count;
                s->put_frame = FALSE; /* do not put until we have last tile */
        } else {
                s->frame.width = width;
                s->frame.height = height;
                s->put_frame = TRUE; /* always put (whole) frame */
        }

        switch (codec) {
                case R10k:
                        s->frame.decoder = (decoder_t)vc_copyliner10k;
                        s->frame.dst_bpp = get_bpp(RGBA);
                        s->yuv = 0;
                        break;
                case RGBA:
                        s->frame.decoder = (decoder_t)memcpy; /* or vc_copylineRGBA?
                                                                 but we have default
                                                                 {r,g,b}shift */
                        
                        s->frame.dst_bpp = get_bpp(RGBA);
                        s->yuv = 0;
                        break;
                case v210:
                        s->frame.decoder = (decoder_t)vc_copylinev210;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        s->yuv = 1;
                        break;
                case DVS10:
                        s->frame.decoder = (decoder_t)vc_copylineDVS10;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        s->yuv = 1;
                        break;
                case Vuy2:
                case DVS8:
                case UYVY:
                        s->frame.decoder = (decoder_t)memcpy;
                        s->frame.dst_bpp = get_bpp(UYVY);
                        s->yuv = 1;
                        break;
        }

        s->frame.fps = fps;
        s->frame.aux = aux;
        s->frame.src_bpp = get_bpp(codec);
        s->frame.color_spec = codec; // src (!)
        s->frame.dst_linesize = s->frame.width * s->frame.dst_bpp;
        s->frame.dst_pitch = s->frame.dst_linesize;
        s->frame.data_len = s->frame.dst_linesize * s->frame.height;
        s->frame.dst_x_offset = 0;


	s->window = XCreateSimpleWindow(s->display, DefaultRootWindow(s->display), 0, 0, s->frame.width,
                       s->frame.height, 0, XWhitePixel(s->display, DefaultScreen(s->display)),
                       XBlackPixel(s->display, DefaultScreen(s->display)));
        XSelectInput(s->display, s->window, KeyPress);

	s->gc = XCreateGC(s->display, s->window, 0, 0);

	XMapWindow(s->display, s->window);

	XStoreName(s->display, s->window, "UltraGrid - XVideo display");

        update_fullscreen_state(s);


                /* Create the image buffer, shared with the X server... */
        for (i = 0; i < 2; i++) {
                if (s->yuv) {
                        s->vw_xvimage[i] = XvShmCreateImage(s->display, s->xv_port, 0x59565955, 0, s->frame.width,
                                       s->frame.height, &s->vw_shm_segment[i]);
                        if (s->vw_xvimage[i] == NULL) {
                                printf("Cannot create XV shared memory image\n");
                                abort();
                        }
                        if (s->vw_xvimage[i]->width != (int) s->frame.width || s->vw_xvimage[i]->height != (int) s->frame.height) {
                                printf("Display does not support %dx%d pixel Xvideo shared memory images\n",
                                                s->frame.width, s->frame.height);
                                abort();
                        }
                } else {
                        s->vw_ximage[i] = XShmCreateImage(s->display, s->vw_visual, s->vw_depth, ZPixmap, NULL, &s->vw_shm_segment[i],
                                        s->frame.width, s->frame.height);
                        if (s->vw_ximage[i] == NULL) {
                                printf("Cannot create shared memory image\n");
                                abort();
                        }
                        if (s->vw_ximage[i]->width != (int) s->frame.width || s->vw_ximage[i]->height != (int) s->frame.height) {
                                printf("Display does not support %dx%d pixel shared memory images\n",
                                                s->frame.width, s->frame.height);
                                abort();
                        }
                }

                if (s->yuv) {
                        s->vw_shm_segment[i].shmid    = shmget(IPC_PRIVATE, s->vw_xvimage[i]->data_size, IPC_CREAT|0777);
                } else {
                        s->vw_shm_segment[i].shmid = shmget(IPC_PRIVATE, s->vw_ximage[i]->bytes_per_line * s->vw_ximage[i]->height, IPC_CREAT|0777);
                }
                s->vw_shm_segment[i].shmaddr  = shmat(s->vw_shm_segment[i].shmid, 0, 0);
                s->vw_shm_segment[i].readOnly = False;
                debug_msg("vw_shm_segment.shmid       = %d\n", s->vw_shm_segment[i].shmid);
                debug_msg("vw_shm_segment.shmaddr     = %d\n", s->vw_shm_segment[i].shmaddr);

                if (s->yuv) {
                        s->vw_xvimage[i]->data = s->vw_shm_segment[i].shmaddr;
                } else {
                        s->vw_ximage[i]->data = s->vw_shm_segment[i].shmaddr;
                }

                if (XShmAttach(s->display, &s->vw_shm_segment[i]) == 0) {
                        printf("Cannot attach shared memory segment\n");
                        abort();
                }
        }

        if (s->yuv)
                s->frame.data = s->vw_xvimage[s->image_network]->data;
        else
                s->frame.data = s->vw_ximage[s->image_network]->data;

        if (s->tiles != NULL) {
                free(s->tiles);
                s->tiles = NULL;
        }
        if (aux & AUX_TILED) {
                int x, y;
                const int x_cnt = tile_info.x_count;

                s->frame.tile_info = tile_info;
                s->tiles = (struct video_frame *)
                        malloc(s->frame.tile_info.x_count *
                                        s->frame.tile_info.y_count *
                                        sizeof(struct video_frame));
                for (y = 0; y < s->frame.tile_info.y_count; ++y)
                        for(x = 0; x < s->frame.tile_info.x_count; ++x) {
                                memcpy(&s->tiles[y*x_cnt + x], &s->frame,
                                                sizeof(struct video_frame));
                                s->tiles[y*x_cnt + x].width = width;
                                s->tiles[y*x_cnt + x].height = height;
                                s->tiles[y*x_cnt + x].tile_info.pos_x = x;
                                s->tiles[y*x_cnt + x].tile_info.pos_y = y;
                                s->tiles[y*x_cnt + x].dst_x_offset +=
                                        x * width * s->frame.dst_bpp;
                                s->tiles[y*x_cnt + x].data +=
                                        y * height *  s->frame.dst_pitch;
                                s->tiles[y*x_cnt + x].src_linesize =
                                        vc_getsrc_linesize(width, codec);
                                s->tiles[y*x_cnt + x].dst_linesize =
                                        vc_getsrc_linesize((x + 1) * width, codec);

                        }
        }

	/* Get our window onto the screen... */
	XFlush(s->display);

        s->win_initialized = 1;

	debug_msg("Window initialized %p\n", s);
}

void
display_xv_done(void *state)
{
	struct state_xv *s = (struct state_xv *) state;

	assert(s->magic == MAGIC_XV);

        cleanup_xv(s);

	XvFreeAdaptorInfo(s->ai);
}

struct video_frame *
display_xv_getf(void *state)
{
	struct state_xv *s = (struct state_xv *) state;
	assert(s->magic == MAGIC_XV);

        if(s->win_initialized) {
                if (s->yuv)
                        s->frame.data = s->vw_xvimage[s->image_network]->data;
                else
                        s->frame.data = s->vw_ximage[s->image_network]->data;
                if (s->new_frame) {
                        if (s->frame.aux & AUX_TILED)
                                update_tile_data(s);
                        s->new_frame = 0;
                }
        } else
                s->frame.data = NULL;
        return &s->frame;
}

int
display_xv_putf(void *state, char *frame)
{
	int		 tmp;
	struct state_xv *s = (struct state_xv *) state;

	assert(s->magic == MAGIC_XV);
	//assert(frame == s->vw_image[s->image_network]->data);

        if(s->put_frame) {
                pthread_mutex_lock(&s->lock);
                /* Wait for the worker to finish... */
                while (s->work_to_do) {
                        s->boss_waiting = TRUE;
                        pthread_cond_wait(&s->boss_cv, &s->lock);
                        s->boss_waiting = FALSE;
                }

                /* ...and give it more to do... */
                tmp = s->image_display;
                s->image_display = s->image_network;
                s->image_network = tmp;
                s->work_to_do    = TRUE;

                /* ...and signal the worker */
                if (s->worker_waiting) {
                        pthread_cond_signal(&s->worker_cv);
                }
                pthread_mutex_unlock(&s->lock);

                s->new_frame = 1;
        }
	return 0;
}

display_type_t *
display_xv_probe(void)
{
        display_type_t          *dt;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
		dt->id	        = DISPLAY_XV_ID;
		dt->name        = "xv";
		dt->description = "X Window System with Xvideo extension";
        }
        return dt;
}

static void cleanup_xv(struct state_xv *s)
{
	int		 i;

	for (i = 0; i < 2; i++) {
		XShmDetach(s->display, &(s->vw_shm_segment[i]));
                if (!s->yuv)
                        XDestroyImage(s->vw_ximage[i]);
		shmdt(s->vw_shm_segment[i].shmaddr);
		shmctl(s->vw_shm_segment[i].shmid, IPC_RMID, 0);
	}
	XUnmapWindow(s->display, s->window);
        XFreeGC(s->display, s->gc);
        XDestroyWindow(s->display, s->window);
}

static struct video_frame * get_tile_buffer(void *state, struct tile_info tile_info) 
{
        struct state_xv *s = (struct state_xv *)state;

        assert(s->tiles != NULL); /* basic sanity test... */

        if(tile_info.pos_x + tile_info.pos_y * tile_info.x_count ==
                        s->frame.tile_info.x_count * 
                        s->frame.tile_info.y_count - 1) 
                s->put_frame = TRUE; /* we have last tile */
        else
                s->put_frame = FALSE; /* we don't have last tile */

        return &s->tiles[tile_info.pos_x + tile_info.pos_y * tile_info.x_count];
}

static void update_tile_data(struct state_xv *s) 
{
        int x, y;
        int x_cnt;

        x_cnt = s->frame.tile_info.x_count;
        for (y = 0; y < s->frame.tile_info.y_count; ++y)
                for(x = 0; x < s->frame.tile_info.x_count; ++x)
                        s->tiles[y*x_cnt + x].data =
                                s->frame.data + 
                                y * s->tiles[y*x_cnt + x].height *
                                s->frame.dst_pitch;
}
#endif /* X_DISPLAY_MISSING */


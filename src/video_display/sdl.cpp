/**
 * @file   video_display/sdl.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2015 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "host.h"

#include "debug.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video_display.h"
#include "tv.h"
#include "audio/audio.h"
#include "audio/utils.h"
#include "utils/ring_buffer.h"
#include "video.h"

#ifdef HAVE_MACOSX
#include "utils/autorelease_pool.h"
extern "C" void NSApplicationLoad();
#elif defined HAVE_LINUX
#include "x11_common.h"
#endif                          /* HAVE_MACOSX */

#include <math.h>

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>

#include <condition_variable>
#include <mutex>
#include <queue>

/* splashscreen (xsedmik) */
#include "video_display/splashscreen.h"

#define MAGIC_SDL   0x155734ae
#define FOURCC_UYVY 0x59565955
#define FOURCC_YUYV 0x32595559

#define MAX_BUFFER_SIZE 1

using namespace std;

struct state_sdl {
        uint32_t                magic;

        struct timeval          tv;
        int                     frames;

        SDL_Overlay  *          yuv_image;
        SDL_Surface  *          sdl_screen;
        SDL_Rect                dst_rect;

        bool                    deinterlace;
        bool                    fs;
        bool                    nodecorate;
        bool                    fixed_size;
        int                     fixed_w, fixed_h;

        int                     screen_w, screen_h;
        
        struct ring_buffer      *audio_buffer;
        struct audio_frame      audio_frame;
        bool                    play_audio;

        queue<struct video_frame *> frame_queue;
        queue<struct video_frame *> free_frame_queue;
        struct video_desc current_desc; /// with those desc new data frames will be generated (getf)
        struct video_desc current_display_desc;
        mutex                   lock;
        condition_variable      frame_ready_cv;
        condition_variable      frame_consumed_cv;
        
#ifdef HAVE_MACOSX
        void                   *autorelease_pool;
#endif
        volatile bool           should_exit;
        uint32_t sdl_flags_win, sdl_flags_fs;

        struct module   mod;

        state_sdl(struct module *parent) : magic(MAGIC_SDL), frames(0), yuv_image(nullptr), sdl_screen(nullptr), dst_rect(),
                      deinterlace(false), fs(false), nodecorate(false), fixed_size(false),
                      fixed_w(0), fixed_h(0),
                      screen_w(0), screen_h(0), audio_buffer(nullptr), audio_frame(), play_audio(false),
                      current_desc(), current_display_desc(),
#ifdef HAVE_MACOSX
                      autorelease_pool(nullptr),
#endif
                      should_exit(false), sdl_flags_win(0), sdl_flags_fs(0)
        {
                gettimeofday(&tv, NULL);
                module_init_default(&mod);
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);
        }

        ~state_sdl() {
                module_done(&mod);
        }
};

static void loadSplashscreen(struct state_sdl *s);
static void show_help(void);
static int display_sdl_putf(void *state, struct video_frame *frame, int nonblock);
static int display_sdl_reconfigure(void *state, struct video_desc desc);
static int display_sdl_reconfigure_real(void *state, struct video_desc desc);

static void cleanup_screen(struct state_sdl *s);
static void configure_audio(struct state_sdl *s);
static int display_sdl_handle_events(struct state_sdl *s);
static void sdl_audio_callback(void *userdata, Uint8 *stream, int len);
static void compute_dst_rect(struct state_sdl *s, int vid_w, int vid_h, int window_w, int window_h, codec_t codec);
static bool update_size(struct state_sdl *s, int win_w, int win_h);
                
/** 
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
static void loadSplashscreen(struct state_sdl *s) {
        struct video_desc desc;

        desc.width = 512;
        desc.height = 512;
        desc.color_spec = RGBA;
        desc.interlacing = PROGRESSIVE;
        desc.fps = 1;
        desc.tile_count = 1;

        display_sdl_reconfigure(s, desc);

        struct video_frame *frame = vf_alloc_desc_data(desc);

        const char *data = splash_data;
        memset(frame->tiles[0].data, 0, frame->tiles[0].data_len);
        for (unsigned int y = 0; y < splash_height; ++y) {
                char *line = frame->tiles[0].data;
                line += vc_get_linesize(frame->tiles[0].width,
                                frame->color_spec) *
                        (((frame->tiles[0].height - splash_height) / 2) + y);
                line += vc_get_linesize(
                                (frame->tiles[0].width - splash_width)/2,
                                frame->color_spec);
                for (unsigned int x = 0; x < splash_width; ++x) {
                        HEADER_PIXEL(data,line);
                        line += 4;
                }
        }

        display_sdl_putf(s, frame, PUTF_BLOCKING);
}

static void compute_dst_rect(struct state_sdl *s, int vid_w, int vid_h, int window_w, int window_h, codec_t codec)
{
        s->dst_rect.x = 0;
        s->dst_rect.y = 0;
        s->dst_rect.w = vid_w;
        s->dst_rect.h = vid_h;

	if (codec_is_a_rgb(codec)) {
		if (window_w > vid_w) {
			s->dst_rect.x = ((int) window_w - vid_w) / 2;
		} else if (window_w < vid_w) {
			s->dst_rect.w = window_w;
		}
		if (window_h > vid_h) {
			s->dst_rect.y = ((int) window_h - vid_h) / 2;
		} else if (window_h < vid_h) {
			s->dst_rect.h = window_h;
		}
	} else if (!codec_is_a_rgb(codec) && (vid_w != window_w || vid_h != window_h)) {
		double frame_aspect = (double) vid_w / vid_h;
		double screen_aspect = (double) window_w / window_h;
		if(screen_aspect > frame_aspect) {
			s->dst_rect.h = window_h;
			s->dst_rect.w = window_h * frame_aspect;
			s->dst_rect.x = ((int) window_w - s->dst_rect.w) / 2;
		} else {
			s->dst_rect.w = window_w;
			s->dst_rect.h = window_w / frame_aspect;
			s->dst_rect.y = ((int) window_h - s->dst_rect.h) / 2;
		}
	}

        fprintf(stdout, "Setting SDL rect %dx%d - %d,%d.\n", s->dst_rect.w,
                s->dst_rect.h, s->dst_rect.x, s->dst_rect.y);
}

static bool update_size(struct state_sdl *s, int win_w, int win_h)
{
        unsigned int x_res_x, x_res_y;

        if (s->fs) {
                x_res_x = s->screen_w;
                x_res_y = s->screen_h;
        } else {
                x_res_x = win_w;
                x_res_y = win_h;
        }
        SDL_Surface *sdl_screen_old = s->sdl_screen;
        s->sdl_screen = SDL_SetVideoMode(x_res_x, x_res_y, 0, s->fs ? s->sdl_flags_fs : s->sdl_flags_win);
        fprintf(stdout, "Setting video mode %dx%d.\n", x_res_x, x_res_y);
        compute_dst_rect(s, s->current_display_desc.width, s->current_display_desc.height, x_res_x, x_res_y, s->current_display_desc.color_spec);
	if (s->sdl_screen == NULL) {
		fprintf(stderr, "Error setting video mode %dx%d!\n", x_res_x,
			x_res_y);
                s->sdl_screen = sdl_screen_old;
                return false;
	} else {
                return true;
        }
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
static int display_sdl_handle_events(struct state_sdl *s)
{
        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                auto msg_univ = reinterpret_cast<struct msg_universal *>(msg);
                struct response *r;
                if (strncasecmp(msg_univ->text, "win-title ", strlen("win_title ")) == 0) {
                        const char *title = msg_univ->text + strlen("win_title");
                        SDL_WM_SetCaption(title, title);
                        r = new_response(RESPONSE_OK, NULL);
                } else {
                        fprintf(stderr, "[SDL] Unknown command received: %s\n", msg_univ->text);
                        r = new_response(RESPONSE_BAD_REQUEST, NULL);
                }
                free_message(msg, r);
        }

        SDL_Event sdl_event;
        while (SDL_PollEvent(&sdl_event)) {
                switch (sdl_event.type) {
                case SDL_VIDEORESIZE:
                        update_size(s, sdl_event.resize.w, sdl_event.resize.h);
                        break;
                case SDL_KEYDOWN:
                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "d")) {
                                s->deinterlace = s->deinterlace ? FALSE : TRUE;
                                printf("Deinterlacing: %s\n", s->deinterlace ? "ON"
                                                : "OFF");
                                return 1;
                        }

                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
                                exit_uv(0);
                        }

                        if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "f")) {
                                s->fs = !s->fs;
                                update_size(s, s->current_display_desc.width, s->current_display_desc.height);
                                return 1;
                        }
                        break;
                case SDL_QUIT:
                        exit_uv(0);
                }
        }

        return 0;

}

static void display_sdl_run(void *arg)
{
        struct state_sdl *s = (struct state_sdl *)arg;
        struct timeval tv;

        while (!s->should_exit) {
                display_sdl_handle_events(s);

                struct video_frame *frame = NULL;

                {
                        unique_lock<mutex> lk(s->lock);
                        if (s->frame_ready_cv.wait_for(lk, std::chrono::milliseconds(100),
                                        [s]{return s->frame_queue.size() > 0;})) {
                                frame = s->frame_queue.front();
                                s->frame_queue.pop();
                                lk.unlock();
                                s->frame_consumed_cv.notify_one();
                        } else {
                                continue;
                        }
                }

                if (s->deinterlace) {
                        vc_deinterlace((unsigned char *) frame->tiles[0].data,
                                        vc_get_linesize(frame->tiles[0].width,
                                                frame->color_spec), frame->tiles[0].height);
                }

                if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
                        if (!display_sdl_reconfigure_real(s, video_desc_from_frame(frame))) {
                                goto free_frame;
                        }
                }

                if (codec_is_a_rgb(frame->color_spec)) {
                        decoder_t decoder = nullptr;
                        if (s->sdl_screen->format->BitsPerPixel != 32 && s->sdl_screen->format->BitsPerPixel != 24) {
                                log_msg(LOG_LEVEL_WARNING, "[SDL] Unsupported bpp %d!\n",
                                                s->sdl_screen->format->BitsPerPixel);
                                goto free_frame;
                        }
                        codec_t dst_codec = s->sdl_screen->format->BitsPerPixel == 32 ? RGBA : RGB;
                        if (frame->color_spec == RGBA) {
                                decoder = dst_codec == RGBA ? vc_copylineRGBA : vc_copylineRGBAtoRGB;
                        } else {
                                decoder = dst_codec == RGBA ? vc_copylineRGBtoRGBA : vc_copylineRGB;
                        }
                        assert(decoder != nullptr);
                        SDL_LockSurface(s->sdl_screen);
                        size_t linesize = vc_get_linesize(frame->tiles[0].width, frame->color_spec);
                        size_t dst_linesize = vc_get_linesize(frame->tiles[0].width, dst_codec);
                        for (size_t i = 0; i < min<size_t>(frame->tiles[0].height, s->sdl_screen->h); ++i) {
                                decoder((unsigned char *) s->sdl_screen->pixels +
                                                s->sdl_screen->pitch * (s->dst_rect.y + i) +
                                                s->dst_rect.x *
                                                s->sdl_screen->format->BytesPerPixel,
                                                (unsigned char *) frame->tiles[0].data + i * linesize,
                                                min<long>(dst_linesize,s->sdl_screen->pitch),
                                                s->sdl_screen->format->Rshift,
                                                s->sdl_screen->format->Gshift,
                                                s->sdl_screen->format->Bshift);
                        }

                        SDL_UnlockSurface(s->sdl_screen);
                        SDL_Flip(s->sdl_screen);
                } else {
			SDL_LockYUVOverlay(s->yuv_image);
                        memcpy(*s->yuv_image->pixels, frame->tiles[0].data, frame->tiles[0].data_len);
                        SDL_UnlockYUVOverlay(s->yuv_image);
                        SDL_DisplayYUVOverlay(s->yuv_image, &s->dst_rect);
		}

free_frame:
                s->lock.lock();
                s->free_frame_queue.push(frame);
                s->lock.unlock();

		s->frames++;
		gettimeofday(&tv, NULL);
		double seconds = tv_diff(tv, s->tv);
		if (seconds > 5) {
			double fps = s->frames / seconds;
			log_msg(LOG_LEVEL_INFO, "[SDL] %d frames in %g seconds = %g FPS\n",
				s->frames, seconds, fps);
			s->tv = tv;
			s->frames = 0;
		}
	}
}

static void show_help(void)
{
        printf("SDL options:\n");
        printf("\t-d sdl[:fs|:d|:nodecorate|:fixed_size[=WxH]]* | help\n");
        printf("\tfs - fullscreen\n");
        printf("\td - deinterlace\n");
        printf("\tnodecorate - disable WM decoration\n");
        printf("\tfixed_size[=WxH] - use fixed sized window\n");
        //printf("\t<f> - read frame content from the filename\n");
}

static void cleanup_screen(struct state_sdl *s)
{
        if (!codec_is_a_rgb(s->current_display_desc.color_spec)) {
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

static int display_sdl_reconfigure(void *state, struct video_desc desc)
{
	struct state_sdl *s = (struct state_sdl *)state;

        s->current_desc = desc;
        return 1;
}

static int display_sdl_reconfigure_real(void *state, struct video_desc desc)
{
	struct state_sdl *s = (struct state_sdl *)state;

	fprintf(stdout, "Reconfigure to size %dx%d\n", desc.width,
			desc.height);

        uint32_t flags_common = SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_RESIZABLE;
        s->sdl_flags_fs = flags_common | SDL_FULLSCREEN;
        s->sdl_flags_win =  flags_common | (s->nodecorate ? SDL_NOFRAME : 0);
        s->current_display_desc = desc;

        if (!s->fixed_size ||
                        !s->sdl_screen /* first run */) {
                if (!update_size(s, desc.width, desc.height)) {
                        memset(&s->current_display_desc, 0, sizeof s->current_display_desc);
                        return FALSE;
                }
        } else {
                SDL_FillRect(s->sdl_screen, NULL, 0x000000);
                SDL_Flip(s->sdl_screen);
                compute_dst_rect(s, s->current_display_desc.width, s->current_display_desc.height, s->sdl_screen->w, s->sdl_screen->h, s->current_display_desc.color_spec);
        }

        if (window_title) {
                SDL_WM_SetCaption(window_title, window_title);
        } else {
                SDL_WM_SetCaption("Ultragrid - SDL Display", "Ultragrid");
        }

	SDL_ShowCursor(SDL_DISABLE);

	if (!codec_is_a_rgb(desc.color_spec)) {
		s->yuv_image =
		    SDL_CreateYUVOverlay(desc.width, desc.height, desc.color_spec == UYVY ?
                                    FOURCC_UYVY : FOURCC_YUYV, s->sdl_screen);
                if (s->yuv_image == NULL) {
                        printf("SDL_overlay initialization failed.\n");
                        memset(&s->current_display_desc, 0, sizeof s->current_display_desc);
                        return FALSE;
                }
        }

        return TRUE;
}

static void *display_sdl_init(struct module *parent, const char *fmt, unsigned int flags)
{
        struct state_sdl *s = new state_sdl(parent);
        int ret;
	const SDL_VideoInfo *video_info;

        if (fmt != NULL) {
                if (strcmp(fmt, "help") == 0) {
                        show_help();
                        delete s;
                        return &display_init_noerr;
                }
                
                char *tmp = strdup(fmt);
                char *ptr = tmp;
                char *tok;
                char *save_ptr = NULL;
                
                while((tok = strtok_r(ptr, ":", &save_ptr)))
                {
                        if (strcmp(tok, "fs") == 0) {
                                s->fs = 1;
                        } else if (strcmp(tok, "d") == 0) {
                                s->deinterlace = 1;
                        } else if (strcmp(tok, "nodecorate") == 0) {
                                s->nodecorate = true;
                        } else if (strncmp(tok, "fixed_size", strlen("fixed_size")) == 0) {
                                s->fixed_size = true;
                                if (strncmp(tok, "fixed_size=", strlen("fixed_size=")) == 0) {
                                        char *size = tok + strlen("fixed_size=");
                                        if (strchr(size, 'x')) {
                                                s->fixed_w = atoi(size);
                                                s->fixed_h = atoi(strchr(size, 'x') + 1);
                                        }
                                }
                        }
                        ptr = NULL;
                }
                
                free (tmp);
        }
        
#ifdef HAVE_MACOSX
        /* Startup function to call when running Cocoa code from a Carbon application. 
         * Whatever the fuck that means. 
         * Avoids uncaught exception (1002)  when creating CGSWindow */
        NSApplicationLoad();
        s->autorelease_pool = autorelease_pool_allocate();
#endif

        ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE);
        
        if (ret < 0) {
                printf("Unable to initialize SDL.\n");
                delete s;
                return NULL;
        }
        
	video_info = SDL_GetVideoInfo();
        s->screen_w = video_info->current_w;
        s->screen_h = video_info->current_h;
        
        SDL_SysWMinfo info;
        memset(&info, 0, sizeof(SDL_SysWMinfo));
        ret = SDL_GetWMInfo(&info);
#ifdef HAVE_LINUX
        if (ret == 1) {
                x11_set_display(info.info.x11.display);
        } else if (ret == 0) {
                fprintf(stderr, "[SDL] Warning: SDL_GetWMInfo unimplemented\n");
        } else if (ret == -1) {
                fprintf(stderr, "[SDL] Warning: SDL_GetWMInfo failure: %s\n", SDL_GetError());
        } else abort();
#endif

        if (s->fixed_size && s->fixed_w && s->fixed_h) {
                struct video_desc desc;
                desc.width = s->fixed_w;
                desc.height = s->fixed_h;
                desc.color_spec = RGBA;
                desc.interlacing = PROGRESSIVE;
                desc.fps = 1;
                desc.tile_count = 1;

                display_sdl_reconfigure_real(s, desc);
        }
        loadSplashscreen(s);	
        
        if(flags & DISPLAY_FLAG_AUDIO_EMBEDDED) {
                s->play_audio = TRUE;
                configure_audio(s);
        } else {
                s->play_audio = FALSE;
        }

        return (void *)s;
}

static void display_sdl_done(void *state)
{
        struct state_sdl *s = (struct state_sdl *)state;

        assert(s->magic == MAGIC_SDL);

	cleanup_screen(s);

        /*FIXME: free all the stuff */
        SDL_ShowCursor(SDL_ENABLE);

        SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE);
        SDL_Quit();
#ifdef HAVE_MACOSX
        autorelease_pool_destroy(s->autorelease_pool);
#endif

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                vf_free(buffer);
        }

        delete s;
}

static struct video_frame *display_sdl_getf(void *state)
{
        struct state_sdl *s = (struct state_sdl *)state;
        assert(s->magic == MAGIC_SDL);

        lock_guard<mutex> lock(s->lock);

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
                        return buffer;
                } else {
                        vf_free(buffer);
                }
        }

        return vf_alloc_desc_data(s->current_desc);
}

static int display_sdl_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_sdl *s = (struct state_sdl *)state;

        assert(s->magic == MAGIC_SDL);

        if (!frame) {
                s->should_exit = true;
                return 0;
        }

        if (nonblock == PUTF_DISCARD) {
                vf_free(frame);
                return 0;
        }

        std::unique_lock<std::mutex> lk(s->lock);
        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK) {
                vf_free(frame);
                printf("1 frame(s) dropped!\n");
                return 1;
        }
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.push(frame);
        lk.unlock();
        s->frame_ready_cv.notify_one();

        return 0;
}

static int display_sdl_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {UYVY, YUYV, RGBA, RGB};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static void sdl_audio_callback(void *userdata, Uint8 *stream, int len) {
        struct state_sdl *s = (struct state_sdl *)userdata;
        if (ring_buffer_read(s->audio_buffer, (char *) stream, len) != len)
        {
                fprintf(stderr, "[SDL] audio buffer underflow!!!\n");
                usleep(500);
        }
}

static void configure_audio(struct state_sdl *s)
{
        s->audio_frame.data = NULL;
        
        SDL_Init(SDL_INIT_AUDIO);
        
        if(SDL_GetAudioStatus() !=  SDL_AUDIO_STOPPED) {
                s->play_audio = FALSE;
                fprintf(stderr, "[SDL] Audio init failed - driver is already used (testcard?)\n");
                return;
        }
        
        s->audio_buffer = ring_buffer_init(1<<20);
}

static int display_sdl_reconfigure_audio(void *state, int quant_samples, int channels,
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
                case 16:
                        sample_type = AUDIO_S16LSB;
                        break;
                /* TO enable in sdl 1.3
                 * case 32:
                        sample_type = AUDIO_S32;
                        break; */
                default:
                        return FALSE;
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

        return TRUE;
error:
        s->play_audio = FALSE;
        s->audio_frame.max_size = 0;
        s->audio_frame.data = NULL;
        return FALSE;
}

static void display_sdl_put_audio_frame(void *state, struct audio_frame *frame) {
        struct state_sdl *s = (struct state_sdl *)state;
        char *tmp;

        if(!s->play_audio)
                return;

        if(frame->bps == 4 || frame->bps == 3) {
                tmp = (char *) malloc(frame->data_len / frame->bps * 2);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len);
                ring_buffer_write(s->audio_buffer, tmp, frame->bps * 2);
                free(tmp);
        } else {
                ring_buffer_write(s->audio_buffer, frame->data, frame->data_len);
        }
}

static const struct video_display_info display_sdl_info = {
        [](struct device_info **available_cards, int *count) {
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "SDL");
                strcpy((*available_cards)[0].name, "SDL SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_sdl_init,
        display_sdl_run,
        display_sdl_done,
        display_sdl_getf,
        display_sdl_putf,
        display_sdl_reconfigure,
        display_sdl_get_property,
        display_sdl_put_audio_frame,
        display_sdl_reconfigure_audio,
};

REGISTER_MODULE(sdl, &display_sdl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);


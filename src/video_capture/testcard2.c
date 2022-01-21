/**
 * @file   video_capture/testcard2.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @todo
 * Passing grabbed frames from thread is terribly broken. It needs to be reworked.
 */
/*
 * Copyright (c) 2011-2021 CESNET z.s.p.o.
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
#include "tv.h"
#include "utils/fs.h"
#include "utils/thread.h"
#include "video.h"
#include "video_capture.h"
#include "testcard_common.h"
#include "compat/platform_semaphore.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#else
#include <SDL/SDL.h>
#endif
#ifdef HAVE_LIBSDL_TTF
#ifdef HAVE_SDL2
#include <SDL2/SDL_ttf.h>
#else
#include <SDL/SDL_ttf.h>
#endif
#endif
#include "audio/types.h"
#include <pthread.h>
#include <time.h>
#include <limits.h>

#define AUDIO_BUFFER_SIZE (AUDIO_SAMPLE_RATE * AUDIO_BPS * \
                s->audio.ch_count * BUFFER_SEC)
#define AUDIO_BPS 2
#define AUDIO_SAMPLE_RATE 48000
#define BUFFER_SEC 1
#define DEFAULT_FORMAT "1920:1080:24:UYVY"
#define MOD_NAME "[testcard2] "

#ifdef _WIN32
#define FONT_DIR "C:\\windows\\fonts"
static const char * const font_candidates[] = { "cour.ttf", };
#elif defined __APPLE__
#define FONT_DIR "/System/Library/Fonts"
static const char * const font_candidates[] = { "Monaco.ttf", "Geneva.ttf", "Keyboard.ttf", };
#else
#define FONT_DIR "/usr/share/fonts"
static const char * const font_candidates[] = { "truetype/freefont/FreeMonoBold.ttf", "truetype/DejaVuSansMono.ttf",
        "TTF/DejaVuSansMono.ttf", "liberation/LiberationMono-Regular.ttf", }; // Arch
#endif

void * vidcap_testcard2_thread(void *args);
void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height);
void toR10k(unsigned char *in, unsigned int width, unsigned int height);

struct testcard_state2 {
        int count;
        int size;
        SDL_Surface *surface;
        struct timeval t0;
        struct video_frame *frame;
        struct tile *tile;
        struct audio_frame audio;
        int aligned_x;
        struct timeval start_time;
        int play_audio_frame;
        
        double audio_remained,
                seconds_tone_played;
        struct timeval last_audio_time;
        char *audio_tone, *audio_silence;
        unsigned int grab_audio:1;
        sem_t semaphore;
        
        pthread_t thread_id;

        volatile bool should_exit;
};

static int configure_audio(struct testcard_state2 *s)
{
        s->audio.bps = AUDIO_BPS;
        s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        
        s->audio_silence = calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        
        s->audio_tone = calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        short int * data = (short int *)(void *) s->audio_tone;
        for(int i=0; i < (int) AUDIO_BUFFER_SIZE/2; i+=2 )
        {
                data[i] = data[i+1] = (float) sin( ((double)i/(double)200) * M_PI * 2. ) * SHRT_MAX;
        }

        printf("[testcard2] playing audio\n");
        
        return 0;
}

static int vidcap_testcard2_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                printf("testcard2 is an alternative implementation of testing signal source.\n");
                printf("It is less maintained than mainline testcard and has less features but has some extra ones, i. a. a timer (if SDL(2)_ttf is found.\n");
                printf("\n");
                printf("testcard2 options:\n");
                printf("\t-t testcard2[:<width>:<height>:<fps>:<codec>]\n");
                show_codec_help("testcard2", (codec_t[]){RGBA, RGB, UYVY, VIDEO_CODEC_NONE},
                                (codec_t[]){R10k, v210, VIDEO_CODEC_NONE}, NULL);

                return VIDCAP_INIT_NOERR;
        }

        struct testcard_state2 *s = calloc(1, sizeof(struct testcard_state2));
        if (!s)
                return VIDCAP_INIT_FAIL;

        char *fmt = strdup(strlen(vidcap_params_get_fmt(params)) != 0 ? vidcap_params_get_fmt(params) : DEFAULT_FORMAT);

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        
        char *tmp = strtok(fmt, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return VIDCAP_INIT_FAIL;
        }
        s->tile->width = atoi(tmp);
        if(s->tile->width % 2 != 0) {
                fprintf(stderr, "Width must be multiple of 2.\n");
                free(s);
                return VIDCAP_INIT_FAIL;
        }
        tmp = strtok(NULL, ":");
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                free(s);
                return VIDCAP_INIT_FAIL;
        }
        s->tile->height = atoi(tmp);
        tmp = strtok(NULL, ":");
        if (!tmp) {
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return VIDCAP_INIT_FAIL;
        }

        s->frame->fps = atof(tmp);

        tmp = strtok(NULL, ":");
        if (!tmp) {
                free(s);
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                return VIDCAP_INIT_FAIL;
        }

        double bpp = 0;

        codec_t codec = get_codec_from_name(tmp);
        if (codec == VIDEO_CODEC_NONE) {
                codec = UYVY;
        }
        bpp = get_bpp(codec);

        s->frame->color_spec = codec;

        if(bpp == 0) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                return VIDCAP_INIT_FAIL;
        }

        s->aligned_x = vc_get_linesize(s->tile->width, codec) / bpp;

        s->frame->interlacing = PROGRESSIVE;
        s->size = s->aligned_x * s->tile->height * bpp;

        {
                unsigned int rect_size = (s->tile->width + COL_NUM - 1) / COL_NUM;
                SDL_Rect r;
                int col_num = 0;
                s->surface =
                    SDL_CreateRGBSurface(SDL_SWSURFACE, s->aligned_x, s->tile->height * 2,
                                         32, 0xff, 0xff00, 0xff0000,
                                         0xff000000);
                for (unsigned j = 0; j < s->tile->height; j += rect_size) {
                        if (j == rect_size * 2) {
                                r.w = s->tile->width;
                                r.h = rect_size / 4;
                                r.x = 0;
                                r.y = j;
                                SDL_FillRect(s->surface, &r, 0xffffffff);
                                r.y = j + rect_size * 3 / 4;
                                SDL_FillRect(s->surface, &r, 0);
                        }
                        for (unsigned i = 0; i < s->tile->width; i += rect_size) {
                                r.w = rect_size;
                                r.h = rect_size;
                                r.x = i;
                                r.y = j;
                                printf("Fill rect at %d,%d\n", r.x, r.y);
                                //if (j != rect_size * 2) {
                                        SDL_FillRect(s->surface, &r,
                                                     rect_colors[col_num]);
                                        col_num = (col_num + 1) % COL_NUM;
                                /*} else {
                                        r.h = rect_size / 2;
                                        r.y += rect_size / 4;
                                        SDL_FillRect(s->surface, &r, grey);
                                        grey += 0x00010101 * (255 / COL_NUM);
                                }*/
                        }
                }
        }

        if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                s->grab_audio = TRUE;
                if(configure_audio(s) != 0) {
                        s->grab_audio = FALSE;
                        fprintf(stderr, "[testcard2] Disabling audio output. "
                                        "\n");
                }
        } else {
                s->grab_audio = FALSE;
        }

        s->count = 0;
        s->audio_remained = 0.0;
        s->seconds_tone_played = 0.0;
        s->play_audio_frame = FALSE;

        platform_sem_init(&s->semaphore, 0, 0);
        printf("Testcard capture set to %dx%d, bpp %f\n", s->tile->width, s->tile->height, bpp);

        s->tile->data_len = s->size;
        s->tile->data = (char *) malloc(s->size);

        if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                s->grab_audio = TRUE;
                if(configure_audio(s) != 0) {
                        s->grab_audio = FALSE;
                        fprintf(stderr, "[testcard] Disabling audio output. "
                                        "SDL-mixer missing, running on Mac or other problem.");
                }
        } else {
                s->grab_audio = FALSE;
        }
        
        if(!s->grab_audio) {
                s->audio_tone = NULL;
                s->audio_silence = NULL;
        }
        
        gettimeofday(&s->start_time, NULL);
        
        pthread_create(&s->thread_id, NULL, vidcap_testcard2_thread, s);

        free(fmt);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_testcard2_done(void *state)
{
        struct testcard_state2 *s = state;

        s->should_exit = true;
        pthread_join(s->thread_id, NULL);

        free(s->tile->data);
        
        free(s->audio_tone);
        free(s->audio_silence);
        free(s);
}

void * vidcap_testcard2_thread(void *arg)
{
        set_thread_name(__func__);

        struct testcard_state2 *s;
        s = (struct testcard_state2 *)arg;
        struct timeval curr_time;
        struct timeval next_frame_time;
        srand(time(NULL));
        int prev_x1 = rand() % (s->tile->width - 300);
        int prev_y1 = rand() % (s->tile->height - 300);
        int down1 = rand() % 2, right1 = rand() % 2;
        int prev_x2 = rand() % (s->tile->width - 100);
        int prev_y2 = rand() % (s->tile->height - 100);
        int down2 = rand() % 2, right2 = rand() % 2;
        
        int stat_count_prev = 0;
        
        gettimeofday(&s->last_audio_time, NULL);
        
#ifdef HAVE_LIBSDL_TTF
        SDL_Surface *text;
        SDL_Color col = { 0, 0, 0, 0 };
        TTF_Font * font = NULL;
        
        if(TTF_Init() == -1)
        {
          log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to initialize SDL_ttf: %s\n",
            TTF_GetError());
          exit(128);
        }

        for (unsigned i = 0; font == NULL && i < sizeof font_candidates / sizeof font_candidates[0]; ++i) {
                char font_path[MAX_PATH_SIZE] = "";
                strncpy(font_path, FONT_DIR, sizeof font_path - 1); // NOLINT (security.insecureAPI.strcpy)
                strncat(font_path, "/", sizeof font_path - strlen(font_path) - 1); // NOLINT (security.insecureAPI.strcpy)
                strncat(font_path, font_candidates[i], sizeof font_path - strlen(font_path) - 1); // NOLINT (security.insecureAPI.strcpy)
                font = TTF_OpenFont(font_path, 108);
        }
        if(!font) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to load any usable font (last font tried: %s)!\n", TTF_GetError());
                exit(128);
        }

#endif
        
        while(!s->should_exit)
        {
                SDL_Rect r;
                SDL_Surface *surf = SDL_ConvertSurface(s->surface, s->surface->format, SDL_SWSURFACE);
                
                r.w = 300;
                r.h = 300;
                r.x = prev_x1 + (right1 ? 1 : -1) * 4;
                r.y = prev_y1 + (down1 ? 1 : -1) * 4;
                if(r.x < 0) right1 = 1;
                if(r.y < 0) down1 = 1;
                if((unsigned int) r.x + r.w > s->tile->width) right1 = 0;
                if((unsigned int) r.y + r.h > s->tile->height) down1 = 0;
                prev_x1 = r.x;
                prev_y1 = r.y;
                
                SDL_FillRect(surf, &r, 0x00000000);
                
                r.w = 100;
                r.h = 100;
                r.x = prev_x2 + (right2 ? 1 : -1) * 12;
                r.y = prev_y2 + (down2 ? 1 : -1) * 9;
                if(r.x < 0) right2 = 1;
                if(r.y < 0) down2 = 1;
                if((unsigned int) r.x + r.w > s->tile->width) right2 = 0;
                if((unsigned int) r.y + r.h > s->tile->height) down2 = 0;
                prev_x2 = r.x;
                prev_y2 = r.y;
                
                SDL_FillRect(surf, &r, 0xffff00aa);
                
                r.w = s->tile->width;
                r.h = 150;
                r.x = 0;
                r.y = s->tile->height - r.h - 30;
                SDL_FillRect(surf, &r, 0xffffffff);
                
#ifdef HAVE_LIBSDL_TTF                
                char frames[64];
                double since_start = tv_diff(next_frame_time, s->start_time);
                snprintf(frames, sizeof frames, "%02d:%02d:%02d %3d", (int) since_start / 3600,
                                (int) since_start / 60 % 60,
                                (int) since_start % 60,
                                 s->count % (int) s->frame->fps);
                text = TTF_RenderText_Solid(font,
                        frames, col);
#endif
#ifdef HAVE_LIBSDL_TTF
                SDL_Rect src_rect;
                src_rect.x=0;
                src_rect.y=0;
                
                r.y += (r.h - text->h) / 2;
                r.x = (s->tile->width - src_rect.w) / 2;
                src_rect.w=text->w;
                src_rect.h=text->h;
                SDL_BlitSurface(text,  &src_rect,  surf, &r);
                SDL_FreeSurface(text);
#endif
                char *data = surf->pixels;
                            
                if (s->frame->color_spec == UYVY || s->frame->color_spec == v210) {
                        char *tmp = (char *) malloc(s->size * 2);
                        vc_copylineRGBAtoUYVY((unsigned char *) tmp, (unsigned char *) surf->pixels,
                                        s->frame->tiles[0].height * vc_get_linesize(s->frame->tiles[0].width, UYVY), 0, 0, 0);
                        data = tmp;
                }

                if (s->frame->color_spec == v210) {
                        surf->pixels =
                            (char *)tov210((unsigned char *) surf->pixels, s->tile->width, s->tile->height);
                }

                if (s->frame->color_spec == R10k) {
                        toR10k((unsigned char *) surf->pixels, s->tile->width, s->tile->height);
                }
                
                memcpy(s->tile->data, data, s->tile->data_len);

                if (data != surf->pixels) {
                        free(data);
                }
                SDL_FreeSurface(surf);
                
next_frame:
                next_frame_time = s->start_time;
                long long since_start_usec = (s->count * 1000000LLU) / s->frame->fps;
                tv_add_usec(&next_frame_time, since_start_usec);
                
                gettimeofday(&curr_time, NULL);
                
                if(tv_gt(next_frame_time, curr_time)) {
                        int sleep_time = tv_diff_usec(next_frame_time, curr_time);
                        usleep(sleep_time);
                } else {
                        if((++s->count) % ((int) s->frame->fps * 5) == 0) {
                                s->play_audio_frame = TRUE;
                        }
                        ++stat_count_prev;
                        goto next_frame;
                }
                
                if((++s->count) % ((int) s->frame->fps * 5) == 0) {
                        s->play_audio_frame = TRUE;
                }
                platform_sem_post(&s->semaphore);
                
                
                double seconds = tv_diff(curr_time, s->t0);
                if (seconds >= 5) {
                        float fps = (s->count - stat_count_prev) / seconds;
                        log_msg(LOG_LEVEL_INFO, "[testcard2] %d frames in %g seconds = %g FPS\n",
                                (s->count - stat_count_prev), seconds, fps);
                        s->t0 = curr_time;
                        stat_count_prev = s->count;
                }
        }

        return NULL;
}

static void grab_audio(struct testcard_state2 *s)
{
        struct timeval curr_time;
        gettimeofday(&curr_time, NULL);
        
        double seconds = tv_diff(curr_time, s->last_audio_time);
        if(s->play_audio_frame) {
                s->seconds_tone_played = 0.0;
                s->play_audio_frame = FALSE;
        }
        
        s->audio.data_len = ((int)((seconds + s->audio_remained) * AUDIO_SAMPLE_RATE));
        if(s->seconds_tone_played < 1.0) {
                s->audio.data = s->audio_tone;
                s->audio.data_len = s->audio.data_len / 400 * 400;
        } else {
                s->audio.data = s->audio_silence;
        }
        
        s->seconds_tone_played += (double) s->audio.data_len / AUDIO_SAMPLE_RATE;
        
        s->audio_remained = (seconds + s->audio_remained) * AUDIO_SAMPLE_RATE - s->audio.data_len;
        s->audio_remained /= AUDIO_SAMPLE_RATE;
        s->audio.data_len *= s->audio.ch_count * AUDIO_BPS;
        
        s->last_audio_time = curr_time;
}

static struct video_frame *vidcap_testcard2_grab(void *arg, struct audio_frame **audio)
{
        struct testcard_state2 *s;

        s = (struct testcard_state2 *)arg;

        platform_sem_wait(&s->semaphore);
        
        *audio = NULL;
        if(s->grab_audio){
                grab_audio(s);
                if(s->audio.data_len)
                        *audio = &s->audio;
                else
                        *audio = NULL;
         }
        
        return s->frame;
}

static struct vidcap_type *vidcap_testcard2_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        *deleter = free;
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "testcard2";
                vt->description = "Video testcard 2";
        }
        return vt;
}

static const struct video_capture_info vidcap_testcard2_info = {
        vidcap_testcard2_probe,
        vidcap_testcard2_init,
        vidcap_testcard2_done,
        vidcap_testcard2_grab,
        false
};

REGISTER_MODULE(testcard2, &vidcap_testcard2_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


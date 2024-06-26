/**
 * @file   video_capture/testcard2.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @todo
 * Merge to mainline testcard.
 */
/*
 * Copyright (c) 2011-2023 CESNET z.s.p.o.
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
#include "utils/macros.h"
#include "utils/thread.h"
#include "video.h"
#include "video_capture.h"
#include "testcard_common.h"
#include "compat/platform_semaphore.h"
#include <stdio.h>
#include <stdlib.h>
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
#define BANNER_HEIGHT 150L
#define BANNER_MARGIN_BOTTOM 75L
#define BUFFER_SEC 1
#define DEFAULT_FORMAT "1920:1080:24:UYVY"
#define EPS_PLUS_1 1.0001
#define FONT_HEIGHT 108
#define MOD_NAME "[testcard2] "

#ifdef HAVE_LIBSDL_TTF
#ifdef _WIN32
#define DEFAULT_FONT_DIR "C:\\windows\\fonts"
static const char * const font_candidates[] = { "cour.ttf", };
#elif defined __APPLE__
#define DEFAULT_FONT_DIR "/System/Library/Fonts"
static const char *const font_candidates[] = {
        "Monaco.ttf",
        "Monaco.dfont",
        "Geneva.ttf",
        "Keyboard.ttf",
};
#else
#define DEFAULT_FONT_DIR "/usr/share/fonts"
static const char *const font_candidates[] = {
        "DejaVuSansMono.ttf", // bundled in AppImage
        "truetype/freefont/FreeMonoBold.ttf", "truetype/dejavu/DejaVuSansMono.ttf", // Ubuntu
        "TTF/DejaVuSansMono.ttf", "liberation/LiberationMono-Regular.ttf", // Arch
        "liberation-mono/LiberationMono-Regular.ttf", // Fedora
};
#endif
#endif // defined HAVE_LIBSDL_TTF

void * vidcap_testcard2_thread(void *args);

struct testcard_state2 {

        int count;
        unsigned char *bg; ///< bars coverted to dest color_spec
        struct timeval t0;
        struct video_desc desc;
        char *data;
        pthread_mutex_t lock;
        pthread_cond_t data_consumed_cv;
        struct audio_frame audio;
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

static bool parse_fmt(struct testcard_state2 *s, char *fmt) {
        char *save_ptr = 0;
        char *tmp      = strtok_r(fmt, ":", &save_ptr);
        if (!tmp) {
                log_msg(LOG_LEVEL_ERROR, "Missing width for testcard\n");
                return false;
        }
        errno = 0;
        s->desc.width = strtol(tmp, NULL, 10);
        if (s->desc.width == 0 || s->desc.width % 2 != 0 || errno != 0) {
                log_msg(LOG_LEVEL_ERROR,
                        "Width must be a positive multiple of 2.\n");
                return false;
        }
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (tmp == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Missing height for testcard\n");
                return false;
        }
        s->desc.height = strtol(tmp, NULL, 10);
        if (s->desc.height == 0 || errno != 0) {
                log_msg(LOG_LEVEL_ERROR, "Wrong height: %s.\n", tmp);
                return false;
        }
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (tmp == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Missing FPS for testcard\n");
                return false;
        }
        if (!parse_fps(tmp, &s->desc)) {
                log_msg(LOG_LEVEL_ERROR, "Wrong FPS for testcard\n");
                return false;
        }

        tmp = strtok_r(NULL, ":", &save_ptr);
        if (tmp == NULL || get_codec_from_name(tmp) == VIDEO_CODEC_NONE) {
                log_msg(LOG_LEVEL_ERROR, "Missing/wrong codec for testcard\n");
                return false;
        }
        s->desc.color_spec = get_codec_from_name(tmp);
        return true;
}

static int vidcap_testcard2_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                printf("testcard2 is an alternative implementation of testing signal source.\n");
                printf("It is less maintained than mainline testcard and has less features but has some extra ones, i. a. a timer (if SDL(2)_ttf is found.\n");
                printf("\n");
                printf("testcard2 options:\n");
                printf("\t-t testcard2[:<width>:<height>:<fps>:<codec>]\n");
                testcard_show_codec_help("testcard2", true);

                return VIDCAP_INIT_NOERR;
        }

        struct testcard_state2 *s = calloc(1, sizeof(struct testcard_state2));
        if (!s)
                return VIDCAP_INIT_FAIL;
        s->desc.tile_count = 1;
        s->desc.interlacing = PROGRESSIVE;

        char *fmt = strdup(strlen(vidcap_params_get_fmt(params)) != 0 ? vidcap_params_get_fmt(params) : DEFAULT_FORMAT);
        bool ret = parse_fmt(s, fmt);
        free(fmt);
        if (!ret) {
                free(s);
                return VIDCAP_INIT_FAIL;
        }

        {
                unsigned int rect_size = (s->desc.width + COL_NUM - 1) / COL_NUM;
                int col_num = 0;
                struct testcard_pixmap surface;
                surface.data = malloc(vc_get_datalen(s->desc.width, s->desc.height, RGBA));
                surface.w = s->desc.width;
                surface.h = s->desc.height;
                for (unsigned i = 0; i < s->desc.width; i += rect_size) {
                        struct testcard_rect r;
                        r.w = MIN(rect_size, s->desc.width - i);
                        r.h = s->desc.height;
                        r.x = i;
                        r.y = 0;
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Fill rect at %d,%d\n", r.x, r.y);
                        testcard_fillRect(&surface, &r,
                                        rect_colors[col_num]);
                        col_num = (col_num + 1) % COL_NUM;
                }
                s->bg = malloc(vc_get_datalen(s->desc.width, s->desc.height, s->desc.color_spec));
                testcard_convert_buffer(RGBA, s->desc.color_spec, s->bg, surface.data, s->desc.width, s->desc.height);
                free(surface.data);
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
        printf("Testcard capture set to %dx%d\n", s->desc.width, s->desc.height);

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
        
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->data_consumed_cv, NULL);

        pthread_create(&s->thread_id, NULL, vidcap_testcard2_thread, s);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_testcard2_done(void *state)
{
        struct testcard_state2 *s = state;

        pthread_mutex_lock(&s->lock);
        free(s->data);
        s->data = NULL;
        s->should_exit = true;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->data_consumed_cv);

        pthread_join(s->thread_id, NULL);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->data_consumed_cv);
        free(s->audio_tone);
        free(s->audio_silence);
        free(s->bg);
        free(s->data);
        free(s);
}

/**
 * Only text banner is rendered in RGBA, other elements (background, squares) are already
 * converted to destination color space. Keep in mind that the regions should be aligned
 * to 6 (v210 block size), won't work for R12L
 */
void * vidcap_testcard2_thread(void *arg)
{
        set_thread_name(__func__);

        struct testcard_state2 *s;
        s = (struct testcard_state2 *)arg;
        struct timeval next_frame_time = { 0 };
        srand(time(NULL));
        int prev_x1 = rand() % ((s->desc.width - 300) / 6) * 6;
        int prev_y1 = rand() % ((s->desc.height - 300) / 6) * 6;
        int down1 = rand() % 2, right1 = rand() % 2;
        int prev_x2 = rand() % ((s->desc.width - 96) / 6) * 6;
        int prev_y2 = rand() % ((s->desc.height - 96) / 6) * 6;
        int down2 = rand() % 2, right2 = rand() % 2;
        
        int stat_count_prev = 0;
        
        gettimeofday(&s->last_audio_time, NULL);
        
#ifdef HAVE_LIBSDL_TTF
#define EXIT_THREAD { free(banner); exit_uv(1); s->should_exit = true; platform_sem_post(&s->semaphore); return NULL; }
        TTF_Font * font = NULL;
        uint32_t *banner = malloc(vc_get_datalen(s->desc.width, BANNER_HEIGHT, RGBA));
        if(TTF_Init() == -1)
        {
          log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to initialize SDL_ttf: %s\n",
            TTF_GetError());
          EXIT_THREAD
        }

        const char *font_dir = IF_NOT_NULL_ELSE(getenv("UG_FONT_DIR"), DEFAULT_FONT_DIR);
        for (unsigned i = 0; font == NULL && i < sizeof font_candidates / sizeof font_candidates[0]; ++i) {
                char font_path[MAX_PATH_SIZE] = "";
                strncpy(font_path, font_dir, sizeof font_path - 1); // NOLINT (security.insecureAPI.strcpy)
                strncat(font_path, "/", sizeof font_path - strlen(font_path) - 1); // NOLINT (security.insecureAPI.strcpy)
                strncat(font_path, font_candidates[i], sizeof font_path - strlen(font_path) - 1); // NOLINT (security.insecureAPI.strcpy)
                font = TTF_OpenFont(font_path, FONT_HEIGHT);
        }
        if(!font) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to load any usable font (last font tried: %s)!\n", TTF_GetError());
                EXIT_THREAD
        }

#endif
        /// @note R12l has pixel block size 8 pixels, so the below won't work for that pixfmt
        unsigned char square_cols[2][48];
        uint32_t src[6 + MAX_PADDING] = { 0 };
        testcard_convert_buffer(RGBA, s->desc.color_spec, square_cols[0], (unsigned char *) src, 6, 1);
        for (int i = 0; i < 6; ++i) src[i] = 0xffff00aa;
        testcard_convert_buffer(RGBA, s->desc.color_spec, square_cols[1], (unsigned char *) src, 6, 1);

        ptrdiff_t block_size = 6 * EPS_PLUS_1 * get_bpp(s->desc.color_spec);

        while(!s->should_exit)
        {
                size_t data_len = vc_get_datalen(s->desc.width, s->desc.height, s->desc.color_spec);
                unsigned char *tmp = malloc(data_len);
                memcpy(tmp, s->bg, data_len);

                struct testcard_rect r;
                r.w = 300;
                r.h = 300;
                r.x = prev_x1 + (right1 ? 1 : -1) * 6;
                r.y = prev_y1 + (down1 ? 1 : -1) * 6;
                if(r.x < 0) { right1 = 1; r.x = 0; }
                if(r.y < 0) { down1 = 1; r.y = 0; }
                if((unsigned int) r.x + r.w > s->desc.width) { right1 = 0; r.w = s->desc.width - r.x; }
                if((unsigned int) r.y + r.h > s->desc.height) { down1 = 0; r.h = s->desc.height - r.y; }
                prev_x1 = r.x;
                prev_y1 = r.y;

                unsigned char *ptr = tmp + r.y * vc_get_linesize(s->desc.width, s->desc.color_spec) + (int) (r.x * EPS_PLUS_1 * get_bpp(s->desc.color_spec));
                for (int y = 0; y < r.h; ++y) {
                        for (int x = 0; x < r.w / 6; x += 1) {
                                memcpy(ptr + block_size * x, square_cols[0], block_size);
                        }
                        ptr += vc_get_linesize(s->desc.width, s->desc.color_spec);
                }

                r.w = 96;
                r.h = 96;
                r.x = prev_x2 + (right2 ? 1 : -1) * 12;
                r.y = prev_y2 + (down2 ? 1 : -1) * 9;
                if(r.x < 0) { right2 = 1; r.x = 0; }
                if(r.y < 0) { down2 = 1; r.y = 0; }
                if((unsigned int) r.x + r.w > s->desc.width)  { right2 = 0; r.w = s->desc.width - r.x; }
                if((unsigned int) r.y + r.h > s->desc.height)  { down2 = 0; r.h = s->desc.height - r.y; }
                prev_x2 = r.x;
                prev_y2 = r.y;

                ptr = tmp + (long) r.y * vc_get_linesize(s->desc.width, s->desc.color_spec) + (long) (r.x * EPS_PLUS_1 * get_bpp(s->desc.color_spec));
                for (int y = 0; y < r.h; ++y) {
                        for (int x = 0; x < r.w / 6; x += 1) {
                                memcpy(ptr + block_size * x, square_cols[1], block_size);
                        }
                        ptr += vc_get_linesize(s->desc.width, s->desc.color_spec);
                }

#ifdef HAVE_LIBSDL_TTF
                memset(banner, 0xFF, 4L * s->desc.width * BANNER_HEIGHT);

                SDL_Color col = { 0, 0, 0, 0 };

                char frames[64];
                double since_start = tv_diff(next_frame_time, s->start_time);
                snprintf(frames, sizeof frames, "%02d:%02d:%02d %3d", (int) since_start / 3600,
                                (int) since_start / 60 % 60,
                                (int) since_start % 60,
                                 s->count % (int) s->desc.fps);
                SDL_Surface *text = TTF_RenderText_Solid(font,
                        frames, col);
                long xoff = ((long) s->desc.width - text->w) / 2;
                long yoff = (BANNER_HEIGHT - text->h) / 2;
                for (int i = 0 ; i < text->h; i++) {
                        uint32_t *d = banner + xoff + (i + yoff) * s->desc.width;
                        for (int j = 0 ; j < MIN(text->w, (int) s->desc.width - xoff); j++) {
                                if (((char *)text->pixels) [i * text->pitch + j]) {
                                        *d = 0x00000000U;
                                }
                                d++;
                        }
                }
                testcard_convert_buffer(RGBA, s->desc.color_spec, tmp + (s->desc.height - BANNER_MARGIN_BOTTOM - BANNER_HEIGHT) * vc_get_linesize(s->desc.width, s->desc.color_spec), (unsigned char *) banner, s->desc.width, BANNER_HEIGHT);
                SDL_FreeSurface(text);
#endif

next_frame:
                next_frame_time = s->start_time;
                long long since_start_usec = (s->count * 1000000LLU) / s->desc.fps;
                tv_add_usec(&next_frame_time, since_start_usec);
                struct timeval curr_time;
                gettimeofday(&curr_time, NULL);
                if(tv_gt(next_frame_time, curr_time)) {
                        int sleep_time = tv_diff_usec(next_frame_time, curr_time);
                        usleep(sleep_time);
                } else {
                        if((++s->count) % ((int) s->desc.fps * 5) == 0) {
                                s->play_audio_frame = TRUE;
                        }
                        ++stat_count_prev;
                        goto next_frame;
                }
                
                if((++s->count) % ((int) s->desc.fps * 5) == 0) {
                        s->play_audio_frame = TRUE;
                }
                pthread_mutex_lock(&s->lock);
                while (s->data != NULL) {
                        pthread_cond_wait(&s->data_consumed_cv, &s->lock);
                }
                s->data = (char *) tmp;
                pthread_mutex_unlock(&s->lock);
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

#ifdef HAVE_LIBSDL_TTF
        free(banner);
#endif

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
        struct testcard_state2 *s = (struct testcard_state2 *)arg;

        if (s->should_exit) {
                return NULL;
        }
        platform_sem_wait(&s->semaphore);
        if (s->should_exit) {
                return NULL;
        }
        assert(s->data != NULL);

        struct video_frame *frame = vf_alloc_desc(s->desc);
        frame->callbacks.data_deleter = vf_data_deleter;
        frame->callbacks.dispose = vf_free;

        pthread_mutex_lock(&s->lock);
        frame->tiles[0].data = s->data;
        s->data = NULL;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->data_consumed_cv);
        
        *audio = NULL;
        if(s->grab_audio){
                grab_audio(s);
                if(s->audio.data_len)
                        *audio = &s->audio;
                else
                        *audio = NULL;
         }
        
        return frame;
}

static void vidcap_testcard2_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = NULL;
        *count = 0;
}

static const struct video_capture_info vidcap_testcard2_info = {
        vidcap_testcard2_probe,
        vidcap_testcard2_init,
        vidcap_testcard2_done,
        vidcap_testcard2_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(testcard2, &vidcap_testcard2_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);


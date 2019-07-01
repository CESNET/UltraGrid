/*
 * FILE:   testcard.c
 * AUTHOR: Colin Perkins <csp@csperkins.org
 *         Alvaro Saurin <saurin@dcs.gla.ac.uk>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2006 University of Glasgow
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/testcard_common.h"
#include "song1.h"
#include "utils/vf_split.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#ifdef HAVE_LIBSDL_MIXER
#include <SDL/SDL.h>
#include <SDL/SDL_mixer.h>
#endif /* HAVE_LIBSDL_MIXER */
#include "audio/audio.h"

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_BPS 2
#define BUFFER_SEC 1
#define AUDIO_BUFFER_SIZE (AUDIO_SAMPLE_RATE * AUDIO_BPS * \
                audio_capture_channels * BUFFER_SEC)
#define DEFAULT_FORMAT "1920:1080:50i:UYVY"

using namespace std;

struct testcard_rect {
        int x, y, w, h;
};
struct testcard_pixmap {
        int w, h;
        void *data;
};

enum class image_pattern : int {
        BARS = 0,
        BLANK,
        NOISE
};

#define DEFAULT_BLANK_COLOR 0xff000000

struct testcard_state {
        std::chrono::steady_clock::time_point last_frame_time;
        int count;
        int size;
        int pan;
        struct testcard_pixmap pixmap;
        char *data;
        std::chrono::steady_clock::time_point t0;
        struct video_frame *frame;
        int frame_linesize;
        struct video_frame *tiled;

        struct audio_frame audio;
        char **tiles_data;
        int tiles_cnt_horizontal;
        int tiles_cnt_vertical;

        char *audio_data;
        volatile int audio_start, audio_end;
        unsigned int grab_audio:1;

        unsigned int still_image;
        enum image_pattern pattern;
        uint32_t blank_color = DEFAULT_BLANK_COLOR;
};

static void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, int color)
{
        int cur_x, cur_y;
        int *data = (int *) s->data;

        for (cur_x = r->x; cur_x < r->x + r->w; ++cur_x)
                for(cur_y = r->y; cur_y < r->y + r->h; ++cur_y)
                        if(cur_x < s->w)
                                *(data + s->w * cur_y + cur_x) = color;
}

#if defined HAVE_LIBSDL_MIXER && ! defined HAVE_MACOSX
static void grab_audio(int chan, void *stream, int len, void *udata)
{
        UNUSED(chan);
        struct testcard_state *s = (struct testcard_state *) udata;

        if(s->audio_end + len <= (int) AUDIO_BUFFER_SIZE) {
                memcpy(s->audio_data + s->audio_end, stream, len);
                s->audio_end += len;
        } else {
                int offset = AUDIO_BUFFER_SIZE - s->audio_end;
                memcpy(s->audio_data + s->audio_end, stream, offset);
                memcpy(s->audio_data, (char *) stream + offset, len - offset);
                s->audio_end = len - offset;
        }
        /* just hack - Mix_Volume doesn't mute correctly the audio */
        memset(stream, 0, len);
}
#endif

static int configure_audio(struct testcard_state *s)
{
        UNUSED(s);

#if defined HAVE_LIBSDL_MIXER && ! defined HAVE_MACOSX
        char filename[1024] = "";
        int fd;
        Mix_Music *music;
        ssize_t bytes_written = 0l;

        SDL_Init(SDL_INIT_AUDIO);

        if( Mix_OpenAudio( AUDIO_SAMPLE_RATE, AUDIO_S16LSB,
                        audio_capture_channels, 4096 ) == -1 ) {
                fprintf(stderr,"[testcard] error initalizing sound\n");
                return -1;
        }
        strncpy(filename, "/tmp/uv.midiXXXXXX", sizeof filename - 1);
        fd = mkstemp(filename);
        if (fd < 0) {
                perror("mkstemp");
                return -1;
        }

        do {
                ssize_t ret;
                ret = write(fd, song1 + bytes_written,
                                sizeof(song1) - bytes_written);
                if(ret < 0) return -1;
                bytes_written += ret;
        } while (bytes_written < (ssize_t) sizeof(song1));
        close(fd);
        music = Mix_LoadMUS(filename);

        s->audio_data = (char *) calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        s->audio_start = 0;
        s->audio_end = 0;
        s->audio.bps = AUDIO_BPS;
        s->audio.ch_count = audio_capture_channels;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;

        // register grab as a postmix processor
        if(!Mix_RegisterEffect(MIX_CHANNEL_POST, grab_audio, NULL, s)) {
                printf("[testcard] Mix_RegisterEffect: %s\n", Mix_GetError());
                return -1;
        }

        if(Mix_PlayMusic(music,-1)==-1){
                fprintf(stderr, "[testcard] error playing midi\n");
                return -1;
        }
        Mix_Volume(-1, 0);

        printf("[testcard] playing audio\n");

        return 0;
#else
        return -2;
#endif
}

static int configure_tiling(struct testcard_state *s, const char *fmt)
{
        char *tmp, *token, *saveptr = NULL;
        int tile_cnt;
        int x;

        int grid_w, grid_h;

        if(fmt[1] != '=') return 1;

        tmp = strdup(&fmt[2]);
        token = strtok_r(tmp, "x", &saveptr);
        grid_w = atoi(token);
        token = strtok_r(NULL, "x", &saveptr);
        grid_h = atoi(token);
        free(tmp);

        s->tiled = vf_alloc(grid_w * grid_h);
        s->tiles_cnt_horizontal = grid_w;
        s->tiles_cnt_vertical = grid_h;
        s->tiled->color_spec = s->frame->color_spec;
        s->tiled->fps = s->frame->fps;
        s->tiled->interlacing = s->frame->interlacing;

        tile_cnt = grid_w *
                                grid_h;
        assert(tile_cnt >= 1);

        s->tiles_data = (char **) malloc(tile_cnt *
                        sizeof(char *));
        /* split only horizontally!!!!!! */
        vf_split(s->tiled, s->frame, grid_w,
                        1, 1 /*prealloc*/);
        /* for each row, make the tile data correct.
         * .data pointers of same row point to same block,
         * but different row */
        for(x = 0; x < grid_w; ++x) {
                int y;

                s->tiles_data[x] = s->tiled->tiles[x].data;

                s->tiled->tiles[x].width = s->frame->tiles[0].width/ grid_w;
                s->tiled->tiles[x].height = s->frame->tiles[0].height / grid_h;
                s->tiled->tiles[x].data_len = s->frame->tiles[0].data_len / (grid_w * grid_h);

                s->tiled->tiles[x].data =
                        s->tiles_data[x] = (char *) realloc(s->tiled->tiles[x].data,
                                s->tiled->tiles[x].data_len * grid_h * 2);


                memcpy(s->tiled->tiles[x].data + s->tiled->tiles[x].data_len  * grid_h,
                                s->tiled->tiles[x].data, s->tiled->tiles[x].data_len * grid_h);
                /* recopy tiles vertically */
                for(y = 1; y < grid_h; ++y) {
                        memcpy(&s->tiled->tiles[y * grid_w + x],
                                        &s->tiled->tiles[x], sizeof(struct tile));
                        /* make the pointers correct */
                        s->tiles_data[y * grid_w + x] =
                                s->tiles_data[x] +
                                y * s->tiled->tiles[x].height *
                                vc_get_linesize(s->tiled->tiles[x].width, s->tiled->color_spec);

                        s->tiled->tiles[y * grid_w + x].data =
                                s->tiles_data[x] +
                                y * s->tiled->tiles[x].height *
                                vc_get_linesize(s->tiled->tiles[x].width, s->tiled->color_spec);
                }
        }

        return 0;
}

static const codec_t codecs_8b[] = {RGBA, RGB, UYVY, YUYV, VIDEO_CODEC_NONE};
static const codec_t codecs_10b[] = {R10k, v210, VIDEO_CODEC_NONE};
static const codec_t codecs_12b[] = {R12L, VIDEO_CODEC_NONE};

static int vidcap_testcard_init(struct vidcap_params *params, void **state)
{
        struct testcard_state *s;
        char *filename;
        const char *strip_fmt = NULL;
        FILE *in = NULL;
        unsigned int i, j;
        unsigned int rect_size = COL_NUM;
        codec_t codec = RGBA;
        int aligned_x;
        char *save_ptr = NULL;

        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                printf("testcard options:\n");
                printf("\t-t testcard:<width>:<height>:<fps>:<codec>[:filename=<filename>][:p][:s=<X>x<Y>][:i|:sf][:still][:pattern=bars|blank|noise|0x<AAGGBBRR>]\n");
                printf("\t<filename> - use file named filename instead of default bars\n");
                printf("\tp - pan with frame\n");
                printf("\ts - split the frames into XxY separate tiles\n");
                printf("\ti|sf - send as interlaced or segmented frame (if none of those is set, progressive is assumed)\n");
                printf("\tstill - send still image\n");
                printf("\tpattern - pattern to use\n");
                show_codec_help("testcard", codecs_8b, codecs_10b, codecs_12b);
                return VIDCAP_INIT_NOERR;
        }

        s = new testcard_state();
        if (!s)
                return VIDCAP_INIT_FAIL;

        s->frame = vf_alloc(1);
        s->frame->interlacing = PROGRESSIVE;

        char *fmt = strdup(vidcap_params_get_fmt(params));
        char *tmp;
        int h_align = 0;
        double bpp = 0;

        if (strlen(fmt) == 0) {
                free(fmt);
                fmt = strdup(DEFAULT_FORMAT);
        }

        tmp = strtok_r(fmt, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                goto error;
        }
        vf_get_tile(s->frame, 0)->width = atoi(tmp);
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                goto error;
        }
        vf_get_tile(s->frame, 0)->height = atoi(tmp);
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                goto error;
        }

        char *endptr;
        s->frame->fps = strtod(tmp, &endptr);
        if (endptr[0] != '\0') { // optional interlacing suffix
                s->frame->interlacing = get_interlacing_from_suffix(endptr);
                if (s->frame->interlacing != PROGRESSIVE &&
                                s->frame->interlacing != SEGMENTED_FRAME &&
                                s->frame->interlacing != INTERLACED_MERGED) { // tff or bff
                        log_msg(LOG_LEVEL_ERROR, "Unsuppored interlacing format!\n");
                        goto error;
                }
                if (s->frame->interlacing == INTERLACED_MERGED) {
                        s->frame->fps /= 2;
                }
        }

        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for testcard '%s'\n", fmt);
                goto error;
        }

        codec = get_codec_from_name(tmp);
        if (codec == VIDEO_CODEC_NONE) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                goto error;
        }
        {
                const codec_t *sets[] = {codecs_8b, codecs_10b, codecs_12b};
                bool supported = false;
                for (int i = 0; i < (int) (sizeof sets / sizeof sets[0]); ++i) {
                        const codec_t *it = sets[i];
                        while (*it != VIDEO_CODEC_NONE) {
                                if (codec == *it++) {
                                        supported = true;
                                }
                        }
                }
                if (!supported) {
                        log_msg(LOG_LEVEL_ERROR, "Unsupported codec '%s'\n", tmp);
                        goto error;
                }
        }
        h_align = get_halign(codec);
        bpp = get_bpp(codec);

        s->frame->color_spec = codec;
        s->still_image = FALSE;

        if(bpp == 0) {
                fprintf(stderr, "Unsupported codec '%s'\n", tmp);
                goto error;
        }

        aligned_x = vf_get_tile(s->frame, 0)->width;
        if (h_align) {
                aligned_x = (aligned_x + h_align - 1) / h_align * h_align;
        }

        rect_size = (vf_get_tile(s->frame, 0)->width + rect_size - 1) / rect_size;

        s->frame_linesize = aligned_x * bpp;
        s->size = aligned_x * vf_get_tile(s->frame, 0)->height * bpp;

        filename = NULL;

        tmp = strtok_r(NULL, ":", &save_ptr);
        while (tmp) {
                if (strcmp(tmp, "p") == 0) {
                        s->pan = 48;
                } else if (strncmp(tmp, "filename=", strlen("filename=")) == 0) {
                        filename = tmp + strlen("filename=");
                        in = fopen(filename, "r");
                        if (!in) {
                                perror("fopen");
                                goto error;
                        }
                        fseek(in, 0L, SEEK_END);
                        long filesize = ftell(in);
                        assert(filesize >= 0);
                        fseek(in, 0L, SEEK_SET);

                        s->data = (char *) malloc(s->size * bpp * 2);

                        if (s->size != filesize) {
                                fprintf(stderr, "Error wrong file size for selected "
                                                "resolution and codec. File size %ld, "
                                                "computed size %d\n", filesize, s->size);
                                goto error;
                        }

                        if (!in || fread(s->data, filesize, 1, in) == 0) {
                                fprintf(stderr, "Cannot read file %s\n", filename);
                                goto error;
                        }

                        fclose(in);
                        in = NULL;

                        memcpy(s->data + s->size, s->data, s->size);
                        vf_get_tile(s->frame, 0)->data = s->data;
                } else if (strncmp(tmp, "s=", 2) == 0) {
                        strip_fmt = tmp;
                } else if (strcmp(tmp, "i") == 0) {
                        s->frame->interlacing = INTERLACED_MERGED;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'i' option. Use format testcard:1920:1080:50i:UYVY instead!\n");
                } else if (strcmp(tmp, "sf") == 0) {
                        s->frame->interlacing = SEGMENTED_FRAME;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'sf' option. Use format testcard:1920:1080:25sf:UYVY instead!\n");
                } else if (strcmp(tmp, "still") == 0) {
                        s->still_image = TRUE;
                } else if (strncmp(tmp, "pattern=", strlen("pattern=")) == 0) {
                        const char *pattern = tmp + strlen("pattern=");
                        if (strcmp(pattern, "bars") == 0) {
                                s->pattern = image_pattern::BARS;
                        } else if (strcmp(pattern, "blank") == 0) {
                                s->pattern = image_pattern::BLANK;
                        } else if (strcmp(pattern, "noise") == 0) {
                                s->pattern = image_pattern::NOISE;
                        } else if (strstr(pattern, "0x") == pattern) {
                                s->pattern = image_pattern::BLANK;
                                if (sscanf(pattern + 2, "%x", &s->blank_color) != 1) {
                                        LOG(LOG_LEVEL_ERROR) << "[testcard] Wrong color!\n";
                                        goto error;
                                }
                        } else {
                                fprintf(stderr, "[testcard] Unknown pattern!\n");;
                                goto error;
                        }
                } else {
                        fprintf(stderr, "[testcard] Unknown option: %s\n", tmp);
                        goto error;
                }
                tmp = strtok_r(NULL, ":", &save_ptr);
        }

        if (!filename) {
                struct testcard_rect r;
                int col_num = 0;
                s->pixmap.w = aligned_x;
                s->pixmap.h = vf_get_tile(s->frame, 0)->height * 2;
                int pixmap_len = s->pixmap.w * s->pixmap.h * 4; // maximal size (RGBA/r10k - has 4 bpp)
                s->pixmap.data = malloc(pixmap_len);

                if (s->pattern == image_pattern::BLANK) {
                        for (int i = 0; i < pixmap_len / 4; ++i) {
                                ((uint32_t *) s->pixmap.data)[i] = s->blank_color;

                        }
                } else if (s->pattern == image_pattern::NOISE) {
                        uint8_t *sample = (uint8_t *) s->pixmap.data;
                        for (int i = 0; i < pixmap_len; ++i) {
                                *sample++ = rand() % 0xff;
                        }
                } else {
                        assert (s->pattern == image_pattern::BARS);
                        for (j = 0; j < vf_get_tile(s->frame, 0)->height; j += rect_size) {
                                int grey = 0xff010101;
                                if (j == rect_size * 2) {
                                        r.w = vf_get_tile(s->frame, 0)->width;
                                        r.h = rect_size / 4;
                                        r.x = 0;
                                        r.y = j;
                                        testcard_fillRect(&s->pixmap, &r, 0xffffffff);
                                        r.h = rect_size - (rect_size * 3 / 4);
                                        r.y = j + rect_size * 3 / 4;
                                        testcard_fillRect(&s->pixmap, &r, 0xff000000);
                                }
                                for (i = 0; i < vf_get_tile(s->frame, 0)->width; i += rect_size) {
                                        r.x = i;
                                        r.y = j;
                                        r.w = rect_size;
                                        r.h = min(rect_size, s->frame->tiles[0].height - r.y);
                                        printf("Fill rect at %d,%d\n", r.x, r.y);
                                        if (j != rect_size * 2) {
                                                testcard_fillRect(&s->pixmap, &r,
                                                                rect_colors[col_num]);
                                                col_num = (col_num + 1) % COL_NUM;
                                        } else {
                                                r.h = rect_size / 2;
                                                r.y += rect_size / 4;
                                                testcard_fillRect(&s->pixmap, &r, grey);
                                                grey += 0x00010101 * (255 / COL_NUM);
                                        }
                                }
                        }
                }
                s->data = (char *) s->pixmap.data;
                if (codec == UYVY || codec == v210 || codec == YUYV) {
                        char *tmp = (char *) malloc(s->size * bpp * 2);
                        vc_copylineRGBAtoUYVY((unsigned char *) tmp, (unsigned char *) s->data,
                                        s->frame->tiles[0].height * vc_get_linesize(s->frame->tiles[0].width, UYVY), 0, 0, 0);
                        free (s->data);
                        s->data = tmp;
                }

                if (codec == v210) {
                        s->data =
                            (char *)tov210((unsigned char *) s->data, aligned_x,
                                           aligned_x, vf_get_tile(s->frame, 0)->height, bpp);
                        free(s->pixmap.data);
                }

                if (codec == R12L) {
                        s->data =
                            (char *)toRGB((unsigned char *) s->data, vf_get_tile(s->frame, 0)->width,
                                           vf_get_tile(s->frame, 0)->height);
                        free(s->pixmap.data);
                        auto tmp = (unsigned char *) malloc(s->size);
                        int dst_linesize = vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec);
                        int src_linesize = vc_get_linesize(s->frame->tiles[0].width, RGB);
                        for (int i = 0; i < (int) vf_get_tile(s->frame, 0)->height; ++i) {
                                vc_copylineRGBtoR12L(tmp + i * dst_linesize,
                                                (unsigned char *) s->data + i * src_linesize, dst_linesize, 0, 0, 0);
                        }
                        free(s->data);
                        s->data = (char *) tmp;
                }

                if (codec == R10k) {
                        toR10k((unsigned char *) s->data, vf_get_tile(s->frame, 0)->width,
                                        vf_get_tile(s->frame, 0)->height);
                }

                if(codec == RGB) {
                        s->data =
                            (char *)toRGB((unsigned char *) s->data, vf_get_tile(s->frame, 0)->width,
                                           vf_get_tile(s->frame, 0)->height);
                        free(s->pixmap.data);
                }

                if(codec == YUYV) {
                        for (int i = 0; i < s->size; i += 2) {
                                swap(s->data[i], s->data[i + 1]);
                        }
                }

                vf_get_tile(s->frame, 0)->data = (char *) malloc(2 * s->size);

                memcpy(vf_get_tile(s->frame, 0)->data, s->data, s->size);
                memcpy(vf_get_tile(s->frame, 0)->data + s->size, vf_get_tile(s->frame, 0)->data, s->size);

                free(s->data);
                s->data = vf_get_tile(s->frame, 0)->data;
        }

        s->count = 0;
        s->last_frame_time = std::chrono::steady_clock::now();

        printf("Testcard capture set to %dx%d, bpp %f\n", vf_get_tile(s->frame, 0)->width,
                        vf_get_tile(s->frame, 0)->height, bpp);

        vf_get_tile(s->frame, 0)->data_len = s->size;

        if(strip_fmt != NULL) {
                if(configure_tiling(s, strip_fmt) != 0) {
                        goto error;
                }
        }

        if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
                s->grab_audio = TRUE;
                if(configure_audio(s) != 0) {
                        s->grab_audio = FALSE;
                        fprintf(stderr, "[testcard] Disabling audio output. "
                                        "SDL-mixer missing, running on Mac or other problem.\n");
                }
        } else {
                s->grab_audio = FALSE;
        }

        free(fmt);

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(fmt);
        free(s->data);
        vf_free(s->frame);
        if (in)
                fclose(in);
        delete s;
        return VIDCAP_INIT_FAIL;
}

static void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = (struct testcard_state *) state;
        free(s->data);
        if (s->tiled) {
                int i;
                for (i = 0; i < s->tiles_cnt_horizontal; ++i) {
                        free(s->tiles_data[i]);
                }
                vf_free(s->tiled);
        }
        vf_free(s->frame);
        if(s->audio_data) {
                free(s->audio_data);
        }
        delete s;
}

static struct video_frame *vidcap_testcard_grab(void *arg, struct audio_frame **audio)
{
        struct testcard_state *state;
        state = (struct testcard_state *)arg;

        std::chrono::steady_clock::time_point curr_time =
                std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::duration<double>>(curr_time - state->last_frame_time).count() <
            1.0 / (double)state->frame->fps) {
                return NULL;
        }

        state->last_frame_time = curr_time;
        state->count++;

        double seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(curr_time - state->t0).count();
        if (seconds >= 5.0) {
                float fps = state->count / seconds;
                log_msg(LOG_LEVEL_INFO, "[testcard] %d frames in %g seconds = %g FPS\n",
                                state->count, seconds, fps);
                state->t0 = curr_time;
                state->count = 0;
        }

        if (state->grab_audio) {
#ifdef HAVE_LIBSDL_MIXER
                state->audio.data = state->audio_data + state->audio_start;
                if(state->audio_start <= state->audio_end) {
                        int tmp = state->audio_end;
                        state->audio.data_len = tmp - state->audio_start;
                        state->audio_start = tmp;
                } else {
                        state->audio.data_len =
                                AUDIO_BUFFER_SIZE -
                                state->audio_start;
                        state->audio_start = 0;
                }
                if(state->audio.data_len > 0)
                        *audio = &state->audio;
                else
                        *audio = NULL;
#endif
        } else {
                *audio = NULL;
        }

        if(!state->still_image) {
                vf_get_tile(state->frame, 0)->data += state->frame_linesize;
        }
        if(vf_get_tile(state->frame, 0)->data > state->data + state->size)
                vf_get_tile(state->frame, 0)->data = state->data;

        /*char line[state->frame.src_linesize * 2 + state->pan];
          unsigned int i;
          memcpy(line, state->frame.data,
          state->frame.src_linesize * 2 + state->pan);
          for (i = 0; i < state->frame.height - 3; i++) {
          memcpy(state->frame.data + i * state->frame.src_linesize,
          state->frame.data + (i + 2) * state->frame.src_linesize +
          state->pan, state->frame.src_linesize);
          }
          memcpy(state->frame.data + i * state->frame.src_linesize,
          state->frame.data + (i + 2) * state->frame.src_linesize +
          state->pan, state->frame.src_linesize - state->pan);
          memcpy(state->frame.data +
          (state->frame.height - 2) * state->frame.src_linesize - state->pan,
          line, state->frame.src_linesize * 2 + state->pan);
#ifdef USE_EPILEPSY
if(!(state->count % 2)) {
unsigned int *p = state->frame.data;
for(i=0; i < state->frame.src_linesize*state->frame.height/4; i++) {
         *p = *p ^ 0x00ffffffL;
         p++;
         }
         }
#endif
*/
        if (state->tiled) {
                /* update tile data instead */
                int i;
                int count = state->tiled->tile_count;

                for (i = 0; i < count; ++i) {
                        /* shift - for semantics of vars refer to configure_tiling*/
                        state->tiled->tiles[i].data += vc_get_linesize(
                                        state->tiled->tiles[i].width, state->tiled->color_spec);
                        /* if out of data, move to beginning
                         * keep in mind that we have two "pictures" for
                         * every tile stored sequentially */
                        if(state->tiled->tiles[i].data >= state->tiles_data[i] +
                                        state->tiled->tiles[i].data_len * state->tiles_cnt_vertical) {
                                state->tiled->tiles[i].data = state->tiles_data[i];
                        }
                }

                return state->tiled;
        }
        return state->frame;
}

static struct vidcap_type *vidcap_testcard_probe(bool verbose)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "testcard";
                vt->description = "Video testcard";

                if (verbose) {
                        vt->card_count = 1;
                        vt->cards = (struct device_info *) calloc(vt->card_count, sizeof(struct device_info));
                        vt->cards[0].id[0] = '\0';
                        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Testing signal");

                        struct {
                                int width;
                                int height;
                        } sizes[] = {
                                {1280, 720},
                                {1920, 1080},
                                {3840, 2160},
                        };
                        int framerates[] = {24, 30, 60};
                        const char * const pix_fmts[] = {"UYVY", "RGB"};

                        snprintf(vt->cards[0].modes[0].name,
                                        sizeof vt->cards[0].name, "Default");
                        snprintf(vt->cards[0].modes[0].id,
                                        sizeof vt->cards[0].id,
                                        "{\"width\":\"\", "
                                        "\"height\":\"\", "
                                        "\"format\":\"\", "
                                        "\"fps\":\"\"}");

                        int i = 1;
                        for(const auto &pix_fmt : pix_fmts){
                                for(const auto &size : sizes){
                                        for(const auto &fps : framerates){
                                                snprintf(vt->cards[0].modes[i].name,
                                                                sizeof vt->cards[0].name,
                                                                "%dx%d@%d %s",
                                                                size.width, size.height,
                                                                fps, pix_fmt);
                                                snprintf(vt->cards[0].modes[i].id,
                                                                sizeof vt->cards[0].id,
                                                                "{\"width\":\"%d\", "
                                                                "\"height\":\"%d\", "
                                                                "\"format\":\"%s\", "
                                                                "\"fps\":\"%d\"}",
                                                                size.width, size.height,
                                                                pix_fmt, fps);
                                                i++;
                                        }
                                }

                        }

                }
        }
        return vt;
}

static const struct video_capture_info vidcap_testcard_info = {
        vidcap_testcard_probe,
        vidcap_testcard_init,
        vidcap_testcard_done,
        vidcap_testcard_grab,
};

REGISTER_MODULE(testcard, &vidcap_testcard_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

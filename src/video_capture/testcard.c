/**
 * @file   video_capture/testcard.c
 * @author Colin Perkins <csp@csperkins.org
 * @author Alvaro Saurin <saurin@dcs.gla.ac.uk>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 */
/*
 * Copyright (c) 2005-2006 University of Glasgow
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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
 */
/**
 * @file
 * @todo
 * * fix broken tiling (perhaps wrapper over pattern generator)
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/pam.h"
#include "utils/string.h"
#include "utils/vf_split.h"
#include "utils/video_pattern_generator.h"
#include "utils/y4m.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/testcard_common.h"

enum {
        AUDIO_SAMPLE_RATE        = 48000,
        BUFFER_SEC               = 1,
        DEFAULT_AUDIO_BPS        = 2,
        DEFAULT_AUIDIO_FREQUENCY = 1000,
};
#define MOD_NAME "[testcard] "
#define AUDIO_BUFFER_SAMPLES (AUDIO_SAMPLE_RATE * BUFFER_SEC)
#define DEFAULT_FORMAT ((struct video_desc) { 1920, 1080, UYVY, 25.0, INTERLACED_MERGED, 1 })
#define DEFAULT_PATTERN "bars"

struct audio_len_pattern {
        int count;
        int samples[5];
        int current_idx;
};
static const int alen_pattern_2997[] = { 1602, 1601, 1602, 1601, 1602 };
static const int alen_pattern_5994[] = { 801, 801, 800, 801, 801 };
static const int alen_pattern_11988[] = { 400, 401, 400, 401, 400 };
_Static_assert(sizeof alen_pattern_2997 <= sizeof ((struct audio_len_pattern *) 0)->samples && sizeof alen_pattern_5994 <= sizeof ((struct audio_len_pattern *) 0)->samples, "insufficient length");

struct testcard_state {
        long long audio_frames;
        long long video_frames;
        time_ns_t last_frame_time;
        int pan;
        video_pattern_generator_t generator;
        struct video_frame *frame;
        struct video_frame *tiled;
        int fps_num;
        int fps_den;
        struct audio_frame audio;
        struct audio_len_pattern apattern;
        int audio_frequency;
        long long capture_frames;

        char **tiles_data;
        int tiles_cnt_horizontal;
        int tiles_cnt_vertical;

        char *audio_data;
        bool grab_audio;
        bool still_image;
        char pattern[128];
};

static void
generate_audio_sine(struct testcard_state *s)
{
        const double scale = 0.1;
        char        *out   = s->audio_data;

        for (int i = 0; i < AUDIO_BUFFER_SAMPLES; i += 1) {
                const int32_t val =
                    round(sin(((double) i / ((double) AUDIO_SAMPLE_RATE /
                                             s->audio_frequency)) *
                              M_PI * 2.) *
                          INT32_MAX * scale);
                for (int j = 0; j < s->audio.ch_count; ++j) {
                        STORE_I32_SAMPLE(out, val, s->audio.bps);
                        out += s->audio.bps;
                }
        }
}

static bool configure_audio(struct testcard_state *s)
{
        if (audio_capture_sample_rate != 0 &&
            audio_capture_sample_rate != AUDIO_SAMPLE_RATE) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Requested %d Hz audio but only %d HZ is "
                                 "currently supported by "
                                 "this module!\n",
                        audio_capture_sample_rate, AUDIO_SAMPLE_RATE);
        }
        s->audio.bps = IF_NOT_NULL_ELSE(audio_capture_bps, DEFAULT_AUDIO_BPS);
        s->audio.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        s->audio.max_size = AUDIO_BUFFER_SAMPLES * s->audio.bps * s->audio.ch_count;
        s->audio.data = s->audio_data = (char *) realloc(s->audio.data, 2 * s->audio.max_size);
        s->audio.flags |= TIMESTAMP_VALID;
        if ((AUDIO_SAMPLE_RATE * s->fps_den) % s->fps_num == 0) {
                s->apattern.count = 1;
                s->apattern.samples[0] = (AUDIO_SAMPLE_RATE * s->fps_den) / s->fps_num;
        } else if (s->fps_den == 1001 && s->fps_num == 30000) {
                s->apattern.count = sizeof alen_pattern_2997 / sizeof alen_pattern_2997[0];
                memcpy(s->apattern.samples, alen_pattern_2997, sizeof alen_pattern_2997);
        } else if (s->fps_den == 1001 && s->fps_num == 60000) {
                s->apattern.count = sizeof alen_pattern_5994 / sizeof alen_pattern_5994[0];
                memcpy(s->apattern.samples, alen_pattern_5994, sizeof alen_pattern_5994);
        } else if (s->fps_den == 1001 && s->fps_num == 120000) {
                s->apattern.count = sizeof alen_pattern_11988 / sizeof alen_pattern_11988[0];
                memcpy(s->apattern.samples, alen_pattern_11988, sizeof alen_pattern_11988);
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Audio not implemented for %f FPS! Please report a bug if it is a common frame rate.\n", s->frame->fps);
                return false;
        }

        generate_audio_sine(s);
        memcpy(s->audio.data + s->audio.max_size, s->audio.data, s->audio.max_size);
        s->grab_audio = true;

        return true;
}

#if 0
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
#endif

static struct video_desc parse_format(char **fmt, char **save_ptr) {
        struct video_desc desc = { 0 };
        desc.tile_count = 1;
        char *tmp = strtok_r(*fmt, ":", save_ptr);
        if (!tmp) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Missing width!\n");
                return (struct video_desc) { 0 };
        }
        desc.width = MAX(strtol(tmp, NULL, 0), 0);

        if ((tmp = strtok_r(NULL, ":", save_ptr)) == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Missing height!\n");
                return (struct video_desc) { 0 };
        }
        desc.height = MAX(strtol(tmp, NULL, 0), 0);

        if (desc.width * desc.height == 0) {
                fprintf(stderr, "Wrong dimensions for testcard.\n");
                return (struct video_desc) { 0 };
        }

        if ((tmp = strtok_r(NULL, ":", save_ptr)) == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Missing FPS!\n");
                return (struct video_desc) { 0 };
        }
        if (!parse_fps(tmp, &desc)) {
                return (struct video_desc) { 0 };
        }

        if ((tmp = strtok_r(NULL, ":", save_ptr)) == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Missing pixel format!\n");
                return (struct video_desc) { 0 };
        }
        desc.color_spec = get_codec_from_name(tmp);
        if (desc.color_spec == VIDEO_CODEC_NONE) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown codec '%s'\n", tmp);
                return (struct video_desc) { 0 };
        }
        if (!testcard_has_conversion(desc.color_spec)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec '%s'\n", tmp);
                return (struct video_desc) { 0 };
        }

        *fmt = NULL;
        return desc;
}

static size_t testcard_load_from_file_pam(const char *filename, struct video_desc *desc, char **in_file_contents) {
        struct pam_metadata info;
        unsigned char *data = NULL;
        if (pam_read(filename, &info, &data, malloc) == 0) {
                return false;
        }
        switch (info.depth) {
                case 3:
                        desc->color_spec = info.maxval == 255 ? RGB : RG48;
                        break;
                case 4:
                        desc->color_spec = RGBA;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "Unsupported PAM/PNM channel count %d!\n", info.depth);
                        return 0;
        }
        desc->width = info.width;
        desc->height = info.height;
        size_t data_len = vc_get_datalen(desc->width, desc->height, desc->color_spec);
        *in_file_contents = (char  *) malloc(data_len);
        if (desc->color_spec == RG48) {
                uint16_t *in = (uint16_t *)(void *) data;
                uint16_t *out = (uint16_t *)(void *) *in_file_contents;
                for (size_t i = 0; i < (size_t) info.width * info.height * 3; ++i) {
                        *out++ = ntohs(*in++) * ((1<<16U) / (info.maxval + 1));
                }
        } else {
                memcpy(*in_file_contents, data, data_len);
        }
        free(data);
        return data_len;
}

static size_t testcard_load_from_file_y4m(const char *filename, struct video_desc *desc, char **in_file_contents) {
        struct y4m_metadata info;
        unsigned char *data = NULL;
        if (y4m_read(filename, &info, &data, malloc) == 0) {
                return 0;
        }
        if (!((info.subsampling == Y4M_SUBS_422 || info.subsampling == Y4M_SUBS_444) && info.bitdepth == 8) || (info.subsampling == Y4M_SUBS_444 && info.bitdepth > 8)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only 8-bit Y4M with subsampling 4:2:2 and 4:4:4 or higher bit depths with subsampling 4:4:4 are supported.\n");
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Provided Y4M picture has subsampling %d and bit depth %d bits.\n", info.subsampling, info.bitdepth);
                return 0;
        }
        desc->width = info.width;
        desc->height = info.height;
        desc->color_spec = info.bitdepth == 8 ? UYVY : Y416;
        size_t data_len = vc_get_datalen(desc->width, desc->height, desc->color_spec);
        *in_file_contents = (char *) malloc(data_len);
        if (info.bitdepth == 8) {
                if (info.subsampling == Y4M_SUBS_422) {
                        i422_8_to_uyvy(desc->width, desc->height, (char *) data, *in_file_contents);
                } else {
                        i444_8_to_uyvy(desc->width, desc->height, (char *) data, *in_file_contents);
                }
        } else {
                i444_16_to_y416(desc->width, desc->height, (char *) data, *in_file_contents, info.bitdepth);
        }
        free(data);
        return data_len;
}

static size_t testcard_load_from_file(const char *filename, struct video_desc *desc, char **in_file_contents, bool deduce_pixfmt) {
        if (ends_with(filename, ".pam") || ends_with(filename, ".pnm") || ends_with(filename, ".ppm")) {
                return testcard_load_from_file_pam(filename, desc, in_file_contents);
        } else if (ends_with(filename, ".y4m")) {
                return testcard_load_from_file_y4m(filename, desc, in_file_contents);
        }

        if (deduce_pixfmt && strchr(filename, '.') != NULL && get_codec_from_file_extension(strrchr(filename, '.') + 1)) {
                desc->color_spec = get_codec_from_file_extension(strrchr(filename, '.') + 1);
        }
        long data_len = vc_get_datalen(desc->width, desc->height, desc->color_spec);
        *in_file_contents = (char *) malloc(data_len);
        FILE *in = fopen(filename, "r");
        if (in == NULL) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "%s fopen: %s\n", filename, ug_strerror(errno));
                return 0;
        }
        fseek(in, 0L, SEEK_END);
        long filesize = ftell(in);
        if (filesize == -1) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "ftell: %s\n", ug_strerror(errno));
                filesize = data_len;
        }
        fseek(in, 0L, SEEK_SET);

        do {
                if (data_len != filesize) {
                        int level = data_len < filesize ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR;
                        log_msg(level, MOD_NAME "Wrong file size for selected resolution"
                                "and codec. File size %ld, computed size %ld\n", filesize, data_len);
                        filesize = data_len;
                        if (level == LOG_LEVEL_ERROR) {
                                data_len = 0; break;
                        }
                }

                if (fread(*in_file_contents, filesize, 1, in) != 1) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot read file %s\n", filename);
                        data_len = 0; break;
                }
        } while (false);

        fclose(in);
        if (data_len == 0) {
                free(*in_file_contents);
                *in_file_contents = NULL;
        }
        return data_len;
}

static void show_help(bool full) {
        printf("testcard options:\n");
        color_printf(TBOLD(TRED("\t-t testcard") "[:size=<width>x<height>][:fps=<fps>][:codec=<codec>]") "[:file=<filename>][:p][:s=<X>x<Y>][:i|:sf][:still][:pattern=<pattern>] " TBOLD("| -t testcard:[full]help\n"));
        color_printf("or\n");
        color_printf(TBOLD(TRED("\t-t testcard") ":<width>:<height>:<fps>:<codec>") "[:other_opts]\n");
        color_printf("where\n");
        color_printf(TBOLD("\t  file ") "      - use file for input data instead of predefined pattern\n"
                               "\t               (raw or PAM/PNM/Y4M)\n");
        color_printf(TBOLD("\t  fps  ") "      - frames per second (with optional 'i' suffix for interlaced)\n");
        color_printf(TBOLD("\t  i|sf ") "      - send as interlaced or segmented frame\n");
        color_printf(TBOLD("\t  mode ") "      - use specified mode (use 'mode=help' for list)\n");
        color_printf(TBOLD("\t   p   ") "      - pan with frame\n");
        color_printf(TBOLD("\tpattern") "      - pattern to use, use \"" TBOLD("pattern=help") "\" for options\n");
        color_printf(TBOLD("\t   s   ") "      - split the frames into XxY separate tiles (currently defunct)\n");
        color_printf(TBOLD("\t still ") "      - send still image\n");
        if (full) {
                color_printf(TBOLD("       afrequency") "    - embedded audio frequency\n");
                color_printf(TBOLD("\t frames") "      - total number of video "
                                                "frames to capture\n");
        }
        color_printf("\n");
        testcard_show_codec_help("testcard", false);
        color_printf("\n");
        color_printf("Examples:\n");
        color_printf(TBOLD("\t%s -t testcard:file=picture.pam\n"), uv_argv[0]);
        color_printf(TBOLD("\t%s -t testcard:mode=VGA\n"), uv_argv[0]);
        color_printf(TBOLD("\t%s -t testcard:s=1920x1080:f=59.94i\n"), uv_argv[0]);
        color_printf("\n");
        color_printf("Default mode: %s\n", video_desc_to_string(DEFAULT_FORMAT));
        color_printf(TBOLD("Note:") " only certain codec and generator combinations produce full-depth samples (not up-sampled 8-bit), use " TBOLD("pattern=help") " for details.\n");
}

static int vidcap_testcard_init(struct vidcap_params *params, void **state)
{
        struct testcard_state *s = NULL;
        char *filename = NULL;
        const char *strip_fmt = NULL;
        char *save_ptr = NULL;
        int ret = VIDCAP_INIT_FAIL;
        char *tmp;
        char *in_file_contents = NULL;
        size_t in_file_contents_size = 0;

        if (strcmp(vidcap_params_get_fmt(params), "help") == 0 || strcmp(vidcap_params_get_fmt(params), "fullhelp") == 0) {
                show_help(strcmp(vidcap_params_get_fmt(params), "fullhelp") == 0);
                return VIDCAP_INIT_NOERR;
        }

        if ((s = calloc(1, sizeof *s)) == NULL) {
                return VIDCAP_INIT_FAIL;
        }
        strncat(s->pattern, DEFAULT_PATTERN, sizeof s->pattern - 1);
        s->audio_frequency = DEFAULT_AUIDIO_FREQUENCY;

        char *fmt = strdup(vidcap_params_get_fmt(params));
        char *ptr = fmt;

        struct video_desc desc = DEFAULT_FORMAT;
        desc.color_spec = VIDEO_CODEC_NONE;
        desc.fps = 0;
        if (strlen(ptr) > 0 && isdigit(ptr[0])) {
                desc = parse_format(&ptr, &save_ptr);
                if (!desc.width) {
                        goto error;
                }
        }

        tmp = strtok_r(ptr, ":", &save_ptr);
        while (tmp) {
                if (strcmp(tmp, "p") == 0) {
                        s->pan = 48;
                } else if (strncmp(tmp, "s=", 2) == 0) {
                        strip_fmt = tmp;
                } else if (strcmp(tmp, "i") == 0) {
                        desc.interlacing = INTERLACED_MERGED;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'i' option. Use format testcard:1920:1080:50i:UYVY instead!\n");
                } else if (strcmp(tmp, "sf") == 0) {
                        desc.interlacing = SEGMENTED_FRAME;
                        log_msg(LOG_LEVEL_WARNING, "[testcard] Deprecated 'sf' option. Use format testcard:1920:1080:25sf:UYVY instead!\n");
                } else if (strcmp(tmp, "still") == 0) {
                        s->still_image = true;
                } else if (IS_KEY_PREFIX(tmp, "pattern")) {
                        const char *pattern = strchr(tmp, '=') + 1;
                        strncpy(s->pattern, pattern, sizeof s->pattern - 1);
                } else if (IS_KEY_PREFIX(tmp, "codec")) {
                        desc.color_spec = get_codec_from_name(strchr(tmp, '=') + 1);
                        if (desc.color_spec == VIDEO_CODEC_NONE) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong color spec: %s\n", strchr(tmp, '=') + 1);
                                goto error;
                        }
                } else if (IS_KEY_PREFIX(tmp, "mode")) {
                        codec_t saved_codec = desc.color_spec;
                        desc = get_video_desc_from_string(strchr(tmp, '=') + 1);
                        desc.color_spec = saved_codec;
                } else if (IS_KEY_PREFIX(tmp, "size")) {
                        tmp = strchr(tmp, '=') + 1;
                        if (isdigit(tmp[0]) && strchr(tmp, 'x') != NULL) {
                                desc.width = atoi(tmp);
                                desc.height = atoi(strchr(tmp, 'x') + 1);
                        } else {
                                struct video_desc size_dsc =
                                    get_video_desc_from_string(tmp);
                                desc.width = size_dsc.width;
                                desc.height = size_dsc.height;
                        }
                } else if (IS_KEY_PREFIX(tmp, "fps")) {
                        if (!parse_fps(strchr(tmp, '=') + 1, &desc)) {
                                goto error;
                        }
                } else if (IS_KEY_PREFIX(tmp, "file")) {
                        filename = strchr(tmp, '=') + 1;
                } else if (strstr(tmp, "afrequency=") == tmp) {
                        s->audio_frequency = atoi(strchr(tmp, '=') + 1);
                } else if (IS_KEY_PREFIX(tmp, "frames")) {
                        s->capture_frames =
                            strtoll(strchr(tmp, '=') + 1, NULL, 0);
                } else {
                        fprintf(stderr, "[testcard] Unknown option: %s\n", tmp);
                        goto error;
                }
                tmp = strtok_r(NULL, ":", &save_ptr);
        }

        if (desc.width <= 0 || desc.height <= 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong video format: %s\n", video_desc_to_string(desc));;
                goto error;
        }

        if (filename) {
                if ((in_file_contents_size = testcard_load_from_file(filename, &desc, &in_file_contents, desc.color_spec == VIDEO_CODEC_NONE)) == 0) {
                        goto error;
                }
        }
        desc.color_spec = IF_NOT_NULL_ELSE(desc.color_spec, DEFAULT_FORMAT.color_spec);
        desc.fps = IF_NOT_NULL_ELSE(desc.fps, DEFAULT_FORMAT.fps);

        if (!s->still_image && codec_is_planar(desc.color_spec)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Planar pixel format '%s', using still picture.\n", get_codec_name(desc.color_spec));
                s->still_image = true;
        }

        s->frame = vf_alloc_desc(desc);
        s->frame->flags |= TIMESTAMP_VALID;

        s->generator = video_pattern_generator_create(s->pattern, s->frame->tiles[0].width, s->frame->tiles[0].height, s->frame->color_spec,
                        s->still_image ? 0 : vc_get_linesize(desc.width, desc.color_spec) + s->pan);
        if (!s->generator) {
                ret = strstr(s->pattern, "help") != NULL ? VIDCAP_INIT_NOERR : VIDCAP_INIT_FAIL;
                goto error;
        }
        if (in_file_contents_size > 0) {
                video_pattern_generator_fill_data(s->generator, in_file_contents);
        }

        s->last_frame_time = get_time_in_ns();

        log_msg(LOG_LEVEL_INFO, MOD_NAME "capture set to %s, bpc %d, pattern: %s, audio %s\n", video_desc_to_string(desc),
                get_bits_per_component(s->frame->color_spec), s->pattern, (s->grab_audio ? "on" : "off"));

        if (strip_fmt != NULL) {
                log_msg(LOG_LEVEL_ERROR, "Multi-tile testcard (stripping) is currently broken, you can use eg. \"-t aggregate -t testcard[args] -t testcard[args]\" instead!\n");
                goto error;
#if 0
                if(configure_tiling(s, strip_fmt) != 0) {
                        goto error;
                }
#endif
        }

        s->fps_num = get_framerate_n(s->frame->fps);
        s->fps_den = get_framerate_d(s->frame->fps);

        if ((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) != 0) {
                if (!configure_audio(s)) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot initialize audio!\n");
                        goto error;
                }
        }

        free(fmt);
        free(in_file_contents);

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(fmt);
        vf_free(s->frame);
        free(in_file_contents);
        free(s);
        return ret;
}

static void vidcap_testcard_done(void *state)
{
        struct testcard_state *s = (struct testcard_state *) state;
        if (s->tiled) {
                int i;
                for (i = 0; i < s->tiles_cnt_horizontal; ++i) {
                        free(s->tiles_data[i]);
                }
                vf_free(s->tiled);
        }
        vf_free(s->frame);
        video_pattern_generator_destroy(s->generator);
        free(s->audio_data);
        free(s);
}

static audio_frame *vidcap_testcard_get_audio(struct testcard_state *s)
{
        if (!s->grab_audio) {
                return NULL;
        }

        s->audio.data += (ptrdiff_t)s->audio.ch_count * s->audio.bps *
                         s->apattern.samples[s->apattern.current_idx];
        s->apattern.current_idx =
            (s->apattern.current_idx + 1) % s->apattern.count;
        s->audio.data_len = s->audio.ch_count * s->audio.bps *
                            s->apattern.samples[s->apattern.current_idx];
        if (s->audio.data >=
            s->audio_data + s->audio.max_size) {
                s->audio.data -= s->audio.max_size;
        }
        s->audio.timestamp =
            ((int64_t)s->audio_frames * 90000 + AUDIO_SAMPLE_RATE - 1) /
            AUDIO_SAMPLE_RATE;
        s->audio_frames +=
            s->audio.data_len / (s->audio.ch_count * s->audio.bps);
        return &s->audio;
}

static struct video_frame *vidcap_testcard_grab(void *arg, struct audio_frame **audio)
{
        struct testcard_state *state;
        state = (struct testcard_state *)arg;

        time_ns_t curr_time = get_time_in_ns();
        if (state->video_frames + 1 == state->capture_frames ||
            (curr_time - state->last_frame_time) / NS_IN_SEC_DBL <
                1.0 / state->frame->fps) {
                return NULL;
        }
        state->last_frame_time = curr_time;
        state->frame->timestamp =
            (state->video_frames * state->fps_den * 90000 + state->fps_num - 1) /
            state->fps_num;

        *audio = vidcap_testcard_get_audio(state);

        vf_get_tile(state->frame, 0)->data = video_pattern_generator_next_frame(state->generator);

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

        state->video_frames += 1;
        return state->frame;
}

static void vidcap_testcard_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;

        *count = 1;
        *available_devices = (struct device_info *) calloc(*count, sizeof(struct device_info));
        struct device_info *card = *available_devices;
        snprintf(card->name, sizeof card->name, "Testing signal");

        struct size {
                int width;
                int height;
        } sizes[] = {
                {1280, 720},
                {1920, 1080},
                {3840, 2160},
        };
        int framerates[] = {24, 30, 60};
        const char * const pix_fmts[] = {"UYVY", "RGB"};

        snprintf(card->modes[0].name,
                        sizeof card->modes[0].name, "Default");
        snprintf(card->modes[0].id,
                        sizeof card->modes[0].id,
                        "{\"width\":\"\", "
                        "\"height\":\"\", "
                        "\"format\":\"\", "
                        "\"fps\":\"\"}");

        int i = 1;
        for (const char * const *pix_fmt = pix_fmts; pix_fmt != pix_fmts + sizeof pix_fmts / sizeof pix_fmts[0]; pix_fmt++) {
                for (const struct size *size = sizes; size != sizes + sizeof sizes / sizeof sizes[0]; size++) {
                        for (const int *fps = framerates; fps != framerates + sizeof framerates / sizeof framerates[0]; fps++) {
                                snprintf(card->modes[i].name,
                                                sizeof card->name,
                                                "%dx%d@%d %s",
                                                size->width, size->height,
                                                *fps, *pix_fmt);
                                snprintf(card->modes[i].id,
                                                sizeof card->modes[0].id,
                                                "{\"width\":\"%d\", "
                                                "\"height\":\"%d\", "
                                                "\"format\":\"%s\", "
                                                "\"fps\":\"%d\"}",
                                                size->width, size->height,
                                                *pix_fmt, *fps);
                                i++;
                        }
                }
        }
        dev_add_option(card, "Still", "Send still image", "still", ":still", true);
        dev_add_option(card, "Pattern", "Pattern to use", "pattern", ":pattern=", false);
}

static const struct video_capture_info vidcap_testcard_info = {
        vidcap_testcard_probe,
        vidcap_testcard_init,
        vidcap_testcard_done,
        vidcap_testcard_grab,
        MOD_NAME,
};

REGISTER_MODULE(testcard, &vidcap_testcard_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

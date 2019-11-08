/**
 * @file   video_capture/banner.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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

#include "debug.h"

#include "lib_common.h"
#include "video_capture.h"
#include "video.h"

#include <chrono>

using namespace std;
using namespace std::chrono;

struct banner_state {
        std::chrono::steady_clock::time_point last_frame_time;
        int count;
        int size;
        int pattern_height;
        char *data;
        std::chrono::steady_clock::time_point t0;
        struct video_frame *frame;
};

static void usage()
{
        printf("Banner capture card scrolls predefined banner. Provided file should be "
                        "uncompressed and in specified color space.\n");
        printf("Banner height is given by file size divided by line length in bytes.\n\n");
        printf("banner options:\n");
        printf("\t-t banner:<width>:<height>:<fps>:<codec>:filename=<filename>[:i|:sf]\n");
        printf("\t<filename> - use file named filename\n");
        printf("\ti|sf - send as interlaced or segmented frame (if none of those is set, progressive is assumed)\n");
}

static int vidcap_banner_init(struct vidcap_params *params, void **state)
{
        struct banner_state *s;
        char *filename;
        FILE *in = NULL;
        char *save_ptr = NULL;

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        if (vidcap_params_get_fmt(params) == NULL || strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                usage();
                return VIDCAP_INIT_NOERR;
        }

        s = new banner_state();
        if (!s)
                return VIDCAP_INIT_FAIL;

        s->frame = vf_alloc(1);

        char *fmt = strdup(vidcap_params_get_fmt(params));
        char *tmp;

        tmp = strtok_r(fmt, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for banner '%s'\n", fmt);
                usage();
                goto error;
        }
        vf_get_tile(s->frame, 0)->width = atoi(tmp);
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for banner '%s'\n", fmt);
                usage();
                goto error;
        }
        vf_get_tile(s->frame, 0)->height = atoi(tmp);
        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for banner '%s'\n", fmt);
                usage();
                goto error;
        }

        s->frame->fps = atof(tmp);

        tmp = strtok_r(NULL, ":", &save_ptr);
        if (!tmp) {
                fprintf(stderr, "Wrong format for banner '%s'\n", fmt);
                usage();
                goto error;
        }

        s->frame->color_spec = get_codec_from_name(tmp);
        if (s->frame->color_spec == VIDEO_CODEC_NONE) {
                fprintf(stderr, "Unknown codec '%s'\n", tmp);
                goto error;
        }

        filename = NULL;

        s->size = vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec)
                * s->frame->tiles[0].height;

        tmp = strtok_r(NULL, ":", &save_ptr);
        while (tmp) {
                if (strncmp(tmp, "filename=", strlen("filename=")) == 0) {
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

                        s->data = (char *) malloc(filesize * 2);

                        if (s->size > filesize || (filesize % vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec)) != 0) {
                                fprintf(stderr, "Error wrong file size for selected "
                                                "resolution and codec. File size %ld, "
                                                "computed size %d\n", filesize, s->size);
                                goto error;
                        }

                        if (!in || fread(s->data, filesize, 1, in) != 1) {
                                fprintf(stderr, "Cannot read file %s\n", filename);
                                goto error;
                        }

                        fclose(in);
                        in = NULL;

                        s->pattern_height = filesize / vc_get_linesize(s->frame->tiles[0].width, s->frame->color_spec);

                        memcpy(s->data + filesize, s->data, filesize);
                        vf_get_tile(s->frame, 0)->data = s->data;
                } else if (strcmp(tmp, "i") == 0) {
                        s->frame->interlacing = INTERLACED_MERGED;
                } else if (strcmp(tmp, "sf") == 0) {
                        s->frame->interlacing = SEGMENTED_FRAME;
                } else {
                        fprintf(stderr, "[banner] Unknown option: %s\n", tmp);
                        usage();
                        goto error;
                }
                tmp = strtok_r(NULL, ":", &save_ptr);
        }

        s->count = 0;
        s->last_frame_time = std::chrono::steady_clock::now();

        printf("banner capture set to %dx%d\n", vf_get_tile(s->frame, 0)->width,
                        vf_get_tile(s->frame, 0)->height);

        vf_get_tile(s->frame, 0)->data_len = s->size;

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

static void vidcap_banner_done(void *state)
{
        struct banner_state *s = (struct banner_state *) state;
        free(s->data);
        vf_free(s->frame);
        delete s;
}

static struct video_frame *vidcap_banner_grab(void *arg, struct audio_frame **audio)
{
        struct banner_state *state;
        state = (struct banner_state *)arg;

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
                log_msg(LOG_LEVEL_INFO, "[banner] %d frames in %g seconds = %g FPS\n",
                                state->count, seconds, fps);
                state->t0 = curr_time;
                state->count = 0;
        }

        *audio = NULL;

        vf_get_tile(state->frame, 0)->data +=
                vc_get_linesize(state->frame->tiles[0].width, state->frame->color_spec);
        if(vf_get_tile(state->frame, 0)->data >= state->data + state->pattern_height *  vc_get_linesize(state->frame->tiles[0].width, state->frame->color_spec))
                vf_get_tile(state->frame, 0)->data = state->data;

        return state->frame;
}

static struct vidcap_type *vidcap_banner_probe(bool /* verbose */, void (**deleter)(void *))
{
        struct vidcap_type *vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "banner";
                vt->description = "Video banner";
        }
        return vt;
}

static const struct video_capture_info vidcap_banner_info = {
        vidcap_banner_probe,
        vidcap_banner_init,
        vidcap_banner_done,
        vidcap_banner_grab,
};

REGISTER_MODULE(banner, &vidcap_banner_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */

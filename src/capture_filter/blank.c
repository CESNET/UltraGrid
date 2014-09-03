/*
 * FILE:    capture_filter/blank.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
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
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of CESNET nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"
#include "messaging.h"
#include "module.h"

#include "video.h"
#include "video_codec.h"

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_blank {
        struct module mod;

        int x, y, width, height;
        double x_relative, y_relative, width_relative, height_relative;

        struct video_desc saved_desc;

        bool in_relative_units;
        bool outline;
};

static bool parse(struct state_blank *s, char *cfg)
{
        int vals[4];
        double vals_relative[4];
        unsigned int counter = 0;
        bool outline = false;

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));

        if (strchr(cfg, '%')) {
                s->in_relative_units = true;
        } else {
                s->in_relative_units = false;
        }

        char *item, *save_ptr = NULL;
        while ((item = strtok_r(cfg, ":", &save_ptr))) {
                if (s->in_relative_units) {
                        vals_relative[counter] = atof(item) / 100.0;
                        if (vals_relative[counter] < 0.0)
                                vals_relative[counter] = 0.0;
                } else {
                        vals[counter] = atoi(item);
                        if (vals[counter] < 0)
                                vals[counter] = 0;
                }

                cfg = NULL;
                counter += 1;
                if (counter == sizeof(vals) / sizeof(int))
                        break;
        }
        while ((item = strtok_r(cfg, ":", &save_ptr))) {
                if (strcmp(item, "outline") == 0) {
                        outline = true;
                } else {
                        fprintf(stderr, "[Blank] Unknown config value: %s\n",
                                        item);
                        return false;
                }
        }

        if(counter != sizeof(vals) / sizeof(int)) {
                fprintf(stderr, "[Blank] Few config values.\n");
                return false;
        }

        if (s->in_relative_units) {
                s->x_relative = vals_relative[0];
                s->y_relative = vals_relative[1];
                s->width_relative = vals_relative[2];
                s->height_relative = vals_relative[3];
        } else {
                s->x = vals[0];
                s->y = vals[1];
                s->width = vals[2];
                s->height = vals[3];
        }
        s->outline = outline;

        return true;
}

static int init(struct module *parent, const char *cfg, void **state)
{
        if (cfg && strcasecmp(cfg, "help") == 0) {
                printf("Blanks specified rectangular area:\n\n");
                printf("blank usage:\n");
                printf("\tblank:x:y:widht:height[:outline]\n");
                printf("\t\tor\n");
                printf("\tblank:x%%:y%%:widht%%:height%%[:outline]\n");
                printf("\t(all values in pixels)\n");
                return 1;
        }

        struct state_blank *s = calloc(1, sizeof(struct state_blank));
        assert(s);

        if (cfg) {
                char *tmp = strdup(cfg);
                bool ret = parse(s, tmp);
                free(tmp);
                if (!ret) {
                        free(s);
                        return -1;
                }
        }

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, parent);

        *state = s;
        return 0;
}

static void done(void *state)
{
        struct state_blank *s = state;
        module_done(&s->mod);
        free(state);
}

static void process_message(struct state_blank *s, struct msg_universal *msg)
{
        parse(s, msg->text);
}

/**
 * @note v210 etc. will be green
 */
static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_blank *s = state;
        codec_t codec = in->color_spec;

        assert(in->tile_count == 1);

        if (s->in_relative_units && !video_desc_eq(s->saved_desc,
                                video_desc_from_frame(in))) {
                s->x = s->x_relative * in->tiles[0].width;
                s->y = s->y_relative * in->tiles[0].height;
                s->width = s->width_relative * in->tiles[0].width;
                s->height = s->height_relative * in->tiles[0].height;
                s->saved_desc = video_desc_from_frame(in);
        }

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                process_message(s, (struct msg_universal *) msg);
                free_message(msg);
        }

        for(int y = s->y; y < s->y + s->height; ++y) {
                if(y >= (int) in->tiles[0].height) {
                        break;
                }
                unsigned char pattern[4];

                memset(pattern, 0, sizeof(pattern));
                if (codec == UYVY) {
                        pattern[0] = 127;
                        pattern[1] = 0;
                }

                int start = s->x * get_bpp(codec);
                int length = s->width * get_bpp(codec);
                int linesize = vc_get_linesize(in->tiles[0].width, codec);
                // following code won't work correctly eg. for v210
                if(start >= linesize) {
                        return in;
                }
                if(start + length > linesize) {
                        length = linesize - start;
                }
                if (codec == UYVY || codec_is_a_rgb(codec)) {
                        // bpp should be integer here, so we can afford this
                        for (int x = start; x < start + length; x += get_bpp(codec)) {
                                memcpy(in->tiles[0].data  + y * linesize + x, pattern,
                                                get_bpp(codec));
                                if (x == start && s->outline &&
                                                y != s->y && y != s->y + s->height - 1) {
                                        x = start + length - 2 * get_bpp(codec);
                                }
                        }
                } else { //fallback
                        if (s->outline &&
                                        y != s->y && y != s->y + s->height - 1) {
                                memset(in->tiles[0].data + y * linesize + start, 0,
                                                get_pf_block_size(codec));
                                memset(in->tiles[0].data + y * linesize + start + length -
                                                get_pf_block_size(codec), 0,
                                                get_pf_block_size(codec));
                        } else {
                                memset(in->tiles[0].data + y * linesize + start, 0, length);
                        }
                }
        }
        return in;
}

struct capture_filter_info capture_filter_blank = {
        .name = "blank",
        .init = init,
        .done = done,
        .filter = filter,
};


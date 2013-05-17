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

#include "video.h"
#include "video_codec.h"

static int init(const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_blank {
        int x,y, width, height;
};

static int init(const char *cfg, void **state)
{
        int vals[4];
        unsigned int counter = 0;
        char *cfg_mutable = NULL, *tmp = NULL;
        char *item, *save_ptr;

        if(cfg) {
                if(strcasecmp(cfg, "help") == 0) {
                        printf("Blanks specified rectangular area:\n\n");
                        printf("blank usage:\n");
                        printf("\tblank:x:y:widht:height\n");
                        printf("\t(all values in pixels)\n");
                        return 1;
                }
                cfg_mutable = tmp = strdup(cfg);
                while((item = strtok_r(cfg_mutable, ":", &save_ptr))) {
                        if(counter > sizeof(vals) / sizeof(int)) {
                                fprintf(stderr, "[Blank] Trailing config values.\n");
                                return -1;
                        }
                        vals[counter] = atoi(item);

                        cfg_mutable = NULL;
                        counter += 1;
                }
        }

        if(counter != sizeof(vals) / sizeof(int)) {
                fprintf(stderr, "[Blank] Few config values.\n");
                return -1;
        }

        struct state_blank *s = calloc(1, sizeof(struct state_blank));
        assert(s);
        s->x = vals[0];
        s->y = vals[1];
        s->width = vals[2];
        s->height = vals[3];

        free(tmp);

        *state = s;
        return 0;
}

static void done(void *state)
{
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_blank *s = state;

        for(int y = s->y; y < s->y + s->height; ++y) {
                if(y < (int) in->tiles[0].height) {
                        int linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
                        double bpp = get_bpp(in->color_spec);
                        int start = s->x * bpp;
                        if(start < linesize) {
                                int length = s->width * bpp;
                                if(start + length <= linesize) {
                                        memset(in->tiles[0].data + y * linesize + start, 0, length);
                                } else {
                                        memset(in->tiles[0].data + y * linesize + start, 0, linesize - start);
                                }
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


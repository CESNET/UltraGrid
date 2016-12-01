/*
 * FILE:    capture_filter/pixelization.cpp
 * AUTHORS: Matej Minarik <396546@mail.muni.cz>
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
#endif /* HAVE_CONFIG_H */


#include "pixelization.h"
#include "video.h"
#include "video_codec.h"

#include <iostream>

using namespace std;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

static bool parse(struct state_pixelization *s, char *cfg)
{
    int vals[4];
    int pixSize;
    double vals_relative[4];
    unsigned int counter = 0;

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
    if ((item = strtok_r(cfg, ":", &save_ptr))) {
        pixSize = atoi(item);
    }else{
        fprintf(stderr, "[Pixel] Few config values.\n");
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
    s->pixelSize = pixSize;

    return true;
}

static int init(struct module *parent, const char *cfg, void **state)
{
    if (cfg && strcasecmp(cfg, "help") == 0) {
        printf("Pixel pixelate rectangular area:\n\n");
        printf("pixel usage:\n");
        printf("\tpixel:x:y:widht:height:pixelSize\n");
        printf("\t(all values are relative)\n");
        printf("\t\tor\n");
        printf("\tpixel:x%%:y%%:widht%%:height%%:pixelSize\n");
        printf("\t(all values in pixels)\n");
        return 1;
    }

    struct state_pixelization *s = (struct state_pixelization *)calloc(1, sizeof(struct state_pixelization));
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
    struct state_pixelization *s = (struct state_pixelization *)state;
    module_done(&s->mod);
    free(state);
}

static void process_message(struct state_pixelization *s, struct msg_universal *msg)
{
    parse(s, msg->text);
}

/**
 * @note v210 etc. will be green
 */
struct video_frame * state_pixelization::filter(struct video_frame *in)
{
    codec_t codec = in->color_spec;

    assert(in->tile_count == 1);

    //change relative to exact sizes
    if (in_relative_units && !video_desc_eq(saved_desc,
                        video_desc_from_frame(in))) {
        x = x_relative * in->tiles[0].width;
        y = y_relative * in->tiles[0].height;
        width = width_relative * in->tiles[0].width;
        height = height_relative * in->tiles[0].height;
        saved_desc = video_desc_from_frame(in);
    }

    //for all pixel lines
    for(int y_ax = y; y_ax < y + height; y_ax += pixelSize) {
        if(y_ax + pixelSize - 1 >= (int) in->tiles[0].height) { //if pixelization would calculate with points out of picture stop loop
            break;
        }

        int start = x;// * get_bpp(codec);
        int length = width;// * get_bpp(codec);
        int linesize = in->tiles[0].width;//vc_get_linesize(in->tiles[0].width, codec);
        // following code won't work correctly eg. for v210
        if(start >= linesize) {
            return in;
        }
        if(start + length > linesize) {
            length = linesize - start;
        }
        if (codec == UYVY) {
            // bpp should be integer here, so we can afford this
            
                //start /= get_bpp(codec);         
            
            for (int x = start; x < start + length; x += pixelSize) {
                if(x + pixelSize - 1 >= (int)in->tiles[0].width) { //if pixelization would calculate with points out of picture stop loop
                    break;
                }
            //calculate average value    
                int sum_part_Y = 0;  
                int sum_part_U = 0;  
                int sum_part_V = 0;
                bool part_U = true;
                for(int pixelY = 0; pixelY < pixelSize; pixelY++ ){
                    for(int pixelX = 0; pixelX < pixelSize * get_bpp(codec); pixelX += get_bpp(codec)){
                        sum_part_Y += (unsigned char)in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec) + 1];
                        if(part_U){
                            sum_part_U += (unsigned char)in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec)];
                        }else{
                            sum_part_V += (unsigned char)in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec)];
                        }
                        part_U = !part_U;   //swap U -> V and V -> U
                    }
                }
                int num_Y = pixelSize * pixelSize;
                int num_U = 0;
                int num_V = 0;
                if(num_Y % 2 == 0){
                    //even
                    num_U = num_Y /2;
                    num_V = num_Y /2;
                }else{
                    //odd
                    num_U = ((pixelSize /2) + 1) * pixelSize;
                    num_V = num_Y - num_U;
                }
                int average_Y = sum_part_Y / num_Y;
                int average_U = sum_part_U / num_U;
                int average_V = sum_part_V / num_V;
                //set calculated value to pixels
                part_U = true;
                for(int pixelY = 0; pixelY < pixelSize; pixelY++ ){
                    for(int pixelX = 0; pixelX < pixelSize * get_bpp(codec); pixelX += get_bpp(codec)){
                        in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec) + 1] = average_Y;
                        if(part_U){
                            in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec)] = average_U;
                        }else{
                            in->tiles[0].data[(y_ax + pixelY) * linesize * (int)get_bpp(codec) + pixelX + x * (int)get_bpp(codec)] = average_V;
                        }
                        part_U = !part_U;   //swap U -> V and V -> U
                    }
                }
            }

        }else if(codec == RGB){ //fallback 
            for (int x = start; x < start + length; x += pixelSize) {
                if(x + pixelSize -1 >= (int)in->tiles[0].width) { //if pixelization would calculate with points out of picture stop loop
                    break;
                }
                //calculate average value
                int sum_R = 0;
                int sum_G = 0;
                int sum_B = 0;
                for(int pixelY = 0; pixelY < pixelSize; pixelY++){
                    for(int pixelX = 0; pixelX < pixelSize * get_bpp(codec); pixelX += get_bpp(codec)){
                        sum_R += (unsigned char)in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX];
                        sum_G += (unsigned char)in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX + 1];
                        sum_B += (unsigned char)in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX + 2];
                    }
                }
                int average_R = sum_R / (pixelSize * pixelSize);
                int average_G = sum_G / (pixelSize * pixelSize);
                int average_B = sum_B / (pixelSize * pixelSize);

                //set calculated value to pixels
                for(int pixelY = 0; pixelY < pixelSize; pixelY++){
                    for(int pixelX = 0; pixelX < pixelSize * get_bpp(codec); pixelX += get_bpp(codec)){
                        in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX] = average_R;
                        in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX + 1] = average_G;
                        in->tiles[0].data[((y_ax + pixelY) * linesize + x) * (int)get_bpp(codec) + pixelX + 2] = average_B;
                    }
                }
            }

        }
    }
    return in;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    return ((state_pixelization *) state)->filter(in);
}

static const struct capture_filter_info capture_filter_pixelization = {
    .init = init,
    .done = done,
    .filter = filter,
};

REGISTER_MODULE(pixelization, &capture_filter_pixelization, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);